"""이 기계에서 실제로 얼마를 먹는지 잰다 — VRAM / RAM / CPU / 시간.

추정하지 않는다. 학습 스텝과 prefill 을 진짜로 돌려서 잰다.
스펙 §44~45 의 규칙을 그대로 따른다: reset_peak_memory_stats -> 워밍업 ->
cuda.synchronize() -> 타이머 -> 실행 -> synchronize -> 타이머 정지.
allocated 와 reserved 를 둘 다 남긴다.

    .conda/python.exe scripts/probe_resources.py
    .conda/python.exe scripts/probe_resources.py --skip-train

    # 로컬 산출물을 잰다 (Q7 의 S0)
    .conda/python.exe scripts/probe_resources.py --skip-infer --only-cpt-config \
        --model artifacts/models/untied_t2b_mean \
        --out reports/tables/resource_probe_untied.md

`--model` 은 hub repo 든 로컬 경로든 받는다. 주지 않으면 1차와 같은 대상
(학습 0.5B, 추론 0.5B/1.5B)을 잰다.

결과는 `--out` 이 가리키는 파일에 저장되고, 기본값은
reports/tables/resource_probe.md 다. **이미 있는 파일은 덮어쓰지 않는다** —
`--force` 를 줘야 덮어쓴다. 이 프로브는 원장에 안 남고 파일 하나가 유일한
기록이라, 다른 대상을 재려다 앞선 측정을 지우면 되돌릴 수 없다.

이건 실험이 아니라 환경 측정이므로 원장(LEDGER.tsv)에는 기록하지 않는다 —
코퍼스도 없고 비교 대상도 없다. 실험 결과는 RunContext 를 통해서만 들어간다.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

# 이 환경에서 transformers 의 attn_implementation="sdpa" 는 기본 디스패처가
# MATH 백엔드로 떨어진다. 8192 토큰에서 9,553MB / 1,733ms 대 1,341MB / 142ms —
# 메모리 7배, 속도 12배 차이다. FLASH 는 이 Windows torch 빌드에 컴파일되어
# 있지 않으므로(=No available kernel) 아래 둘을 명시적으로 강제한다.
# 강제하지 않으면 Level 4 측정 전체가 무의미해진다. docs/RULES.md 9번 참조.
EFFICIENT_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

MB = 1024 * 1024
QWEN05 = "Qwen/Qwen2.5-0.5B"
QWEN15 = "Qwen/Qwen2.5-1.5B"


# ── 계측 ──────────────────────────────────────────────────────────────────
def rss_mb() -> float:
    """이 프로세스의 상주 메모리(RAM)."""
    import psutil

    return psutil.Process().memory_info().rss / MB


def sys_ram() -> tuple:
    import psutil

    vm = psutil.virtual_memory()
    return vm.total / MB, vm.available / MB


def reset() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peaks() -> tuple:
    return (torch.cuda.max_memory_allocated() / MB,
            torch.cuda.max_memory_reserved() / MB)


# ── 학습 ──────────────────────────────────────────────────────────────────
def probe_train(repo: str, seq: int, micro_bs: int, ckpt: bool,
                opt: str, steps: int = 3) -> dict:
    reset()
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    if opt == "adamw_8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

    V = model.config.vocab_size
    ids = torch.randint(0, V, (micro_bs, seq), device="cuda")

    with sdpa_kernel(EFFICIENT_SDPA):
        # 워밍업 1스텝: optimizer state 가 이때 만들어진다 (VRAM 의 큰 몫)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        reset()
        t0 = time.perf_counter()
        for _ in range(steps):
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps

    alloc, resv = peaks()
    ram = rss_mb()
    tok_s = micro_bs * seq / dt
    del model, optimizer, ids, out
    reset()
    return {"alloc": alloc, "resv": resv, "ram": ram, "sec": dt, "tok_s": tok_s}


# ── 추론 ──────────────────────────────────────────────────────────────────
def probe_prefill(repo: str, n_tokens: int, warmup: int = 3, runs: int = 5,
                  last_logit_only: bool = True, efficient: bool = True) -> dict:
    """prefill 한 번의 비용.

    last_logit_only=True 가 실제 생성 경로다. 다음 토큰 하나만 필요하므로
    LM head 를 마지막 위치에만 적용한다.

    False 로 두면 모든 위치의 logits 를 만든다 — 학습·평가(BPB) 경로이며,
    vocab 이 151,936 이라 seq x 151,936 텐서가 KV cache 보다 훨씬 커진다.
    두 경로를 구분하지 않으면 "토크나이저가 메모리를 얼마나 아끼는가"를
    완전히 잘못 재게 된다.
    """
    reset()
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    cfg = model.config
    ids = torch.randint(0, cfg.vocab_size, (1, n_tokens), device="cuda")

    kw = {"logits_to_keep": 1} if last_logit_only else {}
    ctx = sdpa_kernel(EFFICIENT_SDPA) if efficient else torch.autograd.set_grad_enabled(False)
    with torch.inference_mode(), ctx:
        for _ in range(warmup):
            model(input_ids=ids, use_cache=True, **kw)
        torch.cuda.synchronize()
        reset()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(input_ids=ids, use_cache=True, **kw)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs

    alloc, resv = peaks()
    kvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    kv_mb = 2 * cfg.num_hidden_layers * kvh * hd * 2 * n_tokens / MB
    ram = rss_mb()
    del model, ids
    reset()
    return {"alloc": alloc, "resv": resv, "ram": ram, "ms": dt * 1000, "kv_mb": kv_mb}


def label_of(repo: str) -> str:
    """리포트 제목에 쓸 이름. hub repo 든 로컬 경로든 마지막 마디를 쓴다."""
    return Path(repo).name or repo


def main() -> int:
    ap = argparse.ArgumentParser(description="자원 사용량 실측")
    ap.add_argument("--model", action="append", metavar="REPO_OR_PATH",
                    help="잴 대상. 여러 번 줄 수 있다. 안 주면 1차와 같은 대상")
    ap.add_argument("--out", default=None,
                    help="리포트 경로 (기본 reports/tables/resource_probe.md)")
    ap.add_argument("--force", action="store_true",
                    help="--out 이 이미 있어도 덮어쓴다")
    ap.add_argument("--only-cpt-config", action="store_true",
                    help="본 CPT 와 같은 설정 하나만 잰다 "
                         "(seq 2048 / micro_bs 2 / ckpt / adamw_8bit)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-infer", action="store_true")
    args = ap.parse_args()

    # 덮어쓰기는 여기서 막는다. 측정을 다 돌린 뒤에 거부하면 GPU 시간이 날아간다.
    out = Path(args.out) if args.out else ROOT / "reports" / "tables" / "resource_probe.md"
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and not args.force:
        print(f"이미 있다: {out}")
        print("다른 파일에 쓰려면 --out 을, 덮어쓰려면 --force 를 줘라.")
        return 1

    train_models = args.model or [QWEN05]
    infer_models = args.model or [QWEN05, QWEN15]

    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / MB
    lines: list = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# 자원 사용량 실측")
    emit()
    emit("`scripts/probe_resources.py` 로 실제 측정. 추정값이 아니다.")
    emit()
    emit("```")
    emit(f"GPU        {props.name}  {total_vram:.0f} MB")
    ram_total, ram_free = sys_ram()
    emit(f"CPU        {os.cpu_count()} logical cores  ({platform.processor()[:48]})")
    emit(f"RAM        {ram_total / 1024:.0f} GB total / {ram_free / 1024:.0f} GB free")
    emit(f"torch      {torch.__version__}   dtype=bf16")
    emit("attention  SDPA, EFFICIENT+CUDNN 강제 (FLASH 는 이 빌드에 없음)")
    emit("```")
    emit()

    # 본 CPT 설정. --only-cpt-config 가 고르는 것이 이 한 줄이다.
    CPT_CONFIG = (2048, 2, True, "adamw_8bit")
    ALL_CONFIGS = [
        (1024, 1, True,  "adamw"),
        (1024, 1, True,  "adamw_8bit"),
        (1024, 1, False, "adamw_8bit"),
        (1024, 4, True,  "adamw_8bit"),
        (2048, 1, True,  "adamw_8bit"),
        CPT_CONFIG,
        (4096, 1, True,  "adamw_8bit"),
        (8192, 1, True,  "adamw_8bit"),
    ]

    if not args.skip_train:
        configs = [CPT_CONFIG] if args.only_cpt_config else ALL_CONFIGS
        for repo in train_models:
            emit(f"## 학습 ({label_of(repo)} full CPT)")
            emit()
            emit("| seq | micro_bs | grad_ckpt | optimizer | VRAM alloc | VRAM resv | 여유 | RAM | sec/step | tok/s |")
            emit("|---:|---:|:--:|---|---:|---:|---:|---:|---:|---:|")
            for seq, bs, ckpt, opt in configs:
                try:
                    r = probe_train(repo, seq, bs, ckpt, opt)
                    free = total_vram - r["resv"]
                    emit(f"| {seq} | {bs} | {'O' if ckpt else 'X'} | {opt} | "
                         f"{r['alloc']:.0f} MB | {r['resv']:.0f} MB | {free:.0f} MB | "
                         f"{r['ram']:.0f} MB | {r['sec']:.2f} | {r['tok_s']:.0f} |")
                except Exception as exc:
                    emit(f"| {seq} | {bs} | {'O' if ckpt else 'X'} | {opt} | "
                         f"실패: {type(exc).__name__} | | | | | |")
                    reset()
            emit()
            # 경계는 reserved 다. allocated 로 재면 캐싱 할당자가 잡아 둔 몫이
            # 빠져서, Windows WDDM 이 시스템 RAM 으로 흘리며 이미 느려진 지점을
            # "들어간다" 고 읽게 된다 (1차 Q6-E 에서 겪었다).
            emit("여유 = 장치 총량 − **reserved**. allocated 가 아니다.")
            emit()

    if not args.skip_infer:
        for repo in infer_models:
            label = label_of(repo)
            emit(f"## 추론 prefill ({label})")
            emit()
            emit("생성 경로 (`logits_to_keep=1`) — 다음 토큰 하나만 계산한다")
            emit()
            emit("| 입력 토큰 | 한국어 원문(대략) | VRAM alloc | VRAM resv | KV cache | prefill ms | RAM |")
            emit("|---:|---:|---:|---:|---:|---:|---:|")
            for n in (1024, 4096, 8192, 16384, 26000, 32768):
                try:
                    r = probe_prefill(repo, n, last_logit_only=True)
                    emit(f"| {n:,} | ~{int(n / 0.659):,}자 | {r['alloc']:.0f} MB | "
                         f"{r['resv']:.0f} MB | {r['kv_mb']:.0f} MB | "
                         f"{r['ms']:.1f} | {r['ram']:.0f} MB |")
                except Exception as exc:
                    emit(f"| {n:,} | | 실패: {type(exc).__name__} | | | | |")
                    reset()
            emit()
            emit("백엔드를 강제하지 않으면 (기본 디스패처 = MATH 로 폴백)")
            emit()
            emit("| 입력 토큰 | VRAM alloc | prefill ms | 강제 대비 |")
            emit("|---:|---:|---:|---|")
            for n in (4096, 8192):
                try:
                    bad = probe_prefill(repo, n, efficient=False)
                    good = probe_prefill(repo, n, efficient=True)
                    emit(f"| {n:,} | {bad['alloc']:.0f} MB | {bad['ms']:.1f} | "
                         f"메모리 {bad['alloc'] / good['alloc']:.1f}배, "
                         f"시간 {bad['ms'] / good['ms']:.1f}배 |")
                except Exception as exc:
                    emit(f"| {n:,} | 실패: {type(exc).__name__} | | |")
                    reset()
            emit()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
