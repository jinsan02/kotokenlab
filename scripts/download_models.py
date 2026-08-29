"""외부 모델·토크나이저를 내려받는다 (스펙 §3).

역할 분리 (16GB VRAM 제약, 스펙 §2)

    Qwen2.5-0.5B    핵심 실험 모델 — Full CPT / surgery / ablation   → 전체 가중치
    Qwen2.5-1.5B    scale validation — alignment / LoRA·QLoRA        → 전체 가중치
    HCX SEED 0.5B   한국어 특화 external baseline                    → 전체 가중치 (작다)
    A.X 4.0 Light   Qwen 기반 한국어 adaptation 산업 사례            → 토크나이저만
    A.X 4.0         상동 (72B — 학습 대상 아님)                      → 토크나이저만

토크나이저만 받는 쪽은 Level 1 intrinsic 벤치마크(§14~16)에만 쓰인다.
7B/72B 가중치는 필요해지는 시점(4-bit 추론 비교)에 따로 받는다.

주의: 스펙이 지정한 HyperCLOVAX-SEED-Text-Base-0.5B 는 공개되어 있지 않다.
같은 계열의 Instruct 0.5B 로 대체한다. 토크나이저는 동일하고, HCX 는 애초에
인과 실험이 아니라 external reference 다 (스펙 §56, docs/RULES.md 4번).

    .conda/python.exe scripts/download_models.py
    .conda/python.exe scripts/download_models.py --only Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 캐시는 저장소 안에 두되 git 에는 넣지 않는다 (.gitignore).
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from huggingface_hub import snapshot_download  # noqa: E402

from src.utils import ledger  # noqa: E402

TOKENIZER_ONLY = [
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "config.json", "generation_config.json",
    "chat_template.jinja", "*.md",
]

# (repo_id, 역할, 전체 가중치를 받는가)
TARGETS: list = [
    ("Qwen/Qwen2.5-0.5B", "main experimental backbone (full CPT)", True),
    ("Qwen/Qwen2.5-1.5B", "scale validation (alignment + LoRA/QLoRA)", True),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
     "external baseline: Korean-specialized small LLM", True),
    ("skt/A.X-4.0-Light", "external reference: Qwen-based Korean adaptation (7B)", False),
    ("skt/A.X-4.0", "external reference: Korean adaptation (72B, tokenizer only)", False),
]


def download(repo_id: str, full: bool) -> Path:
    kwargs = {} if full else {"allow_patterns": TOKENIZER_ONLY}
    path = snapshot_download(repo_id=repo_id, **kwargs)
    return Path(path)


def describe(repo_id: str, path: Path) -> dict:
    """받은 모델의 구조를 읽어 레지스트리 한 행으로 만든다 (검토 D1).

    revision 은 snapshot 디렉토리 이름이 곧 커밋 해시다. 스펙 §58 이 요구하는
    model.revision 이 여기서 확정된다 — HF 저장소는 갱신되므로 repo_id 만으로는
    나중에 같은 것을 다시 가져올 수 없다.

    embedding_share 를 함께 기록하는 이유는 검토 A2 다. 임베딩이 전체에서
    차지하는 비중이 스케일마다 크게 달라서, 토크나이저 수술 결과의 유효 범위가
    이 값에 달려 있다.
    """
    from transformers import AutoConfig, AutoTokenizer

    row: dict = {"repo_id": repo_id, "revision": path.name,
                 "name": repo_id.split("/")[-1]}
    try:
        cfg = AutoConfig.from_pretrained(str(path))
        tok = AutoTokenizer.from_pretrained(str(path))
        V, H, L = cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers
        heads = cfg.num_attention_heads
        kvh = getattr(cfg, "num_key_value_heads", heads)
        hd = getattr(cfg, "head_dim", H // heads)
        tied = bool(getattr(cfg, "tie_word_embeddings", False))
        emb = V * H
        inter = cfg.intermediate_size
        per_layer = 2 * H * H + 2 * H * (kvh * hd) + 3 * H * inter
        total = emb + L * per_layer + (0 if tied else emb)
        row.update({
            "vocab_size": V, "tokenizer_len": len(tok),
            "hidden_size": H, "n_layers": L, "n_heads": heads,
            "n_kv_heads": kvh, "head_dim": hd,
            "embedding_params": emb, "total_params": total,
            "embedding_share": round(emb / total, 4),
            "tie_word_embeddings": tied,
            # bf16 기준: 2(K,V) x layers x kv_heads x head_dim x 2 bytes
            "kv_bytes_per_token": 2 * L * kvh * hd * 2,
        })
    except Exception as exc:
        row["note"] = f"config 해석 실패: {type(exc).__name__}"
    return row


def register(repo_id: str, path: Path, role: str, full: bool) -> None:
    """이미 같은 (repo_id, revision) 이 있으면 다시 적지 않는다 (append-only 이지만 중복은 막는다)."""
    known = {(r.get("repo_id"), r.get("revision"))
             for r in ledger.read_rows("models")}
    if (repo_id, path.name) in known:
        return
    row = describe(repo_id, path)
    row["role"] = role
    row["scope"] = "full" if full else "tokenizer_only"
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    row["files_mb"] = round(size / 1024 / 1024, 1)
    ledger.append_row("models", row)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="모델·토크나이저 다운로드")
    parser.add_argument("--only", action="append", default=None,
                        help="특정 repo_id 만 받는다 (반복 가능)")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="전부 토크나이저만 받는다")
    args = parser.parse_args(argv)

    print(f"HF_HOME = {os.environ['HF_HOME']}\n")
    failed = []
    for repo_id, role, full in TARGETS:
        if args.only and repo_id not in args.only:
            continue
        if args.tokenizer_only:
            full = False
        kind = "full weights" if full else "tokenizer only"
        print(f"── {repo_id}  [{kind}]\n   {role}")
        try:
            path = download(repo_id, full)
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            register(repo_id, path, role, full)
            print(f"   OK  {size / 1024 / 1024:.0f} MB  rev={path.name[:12]}\n")
        except Exception as exc:
            print(f"   FAIL  {type(exc).__name__}: {exc}\n", file=sys.stderr)
            failed.append(repo_id)

    if failed:
        print(f"실패: {failed}", file=sys.stderr)
        return 1
    print(f"모든 대상 다운로드 완료 -> {ledger.table_path('models')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
