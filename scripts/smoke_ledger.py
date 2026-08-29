"""원장 전체를 임시 디렉토리에서 한 번 돌려본다 (E2E 확인용).

실제 experiments/ 를 건드리지 않는다. 임시 루트에 run 하나를 만들고
메트릭 테이블 다섯 개에 모두 기록한 뒤, tools/validate_ledger.py 로 검사한다.

    .conda/python.exe scripts/smoke_ledger.py

새 컬럼을 추가했거나 스키마를 고쳤을 때 이걸 먼저 돌려본다.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402
from tools.validate_ledger import validate  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="kotokenlab_smoke_") as tmp:
        root = Path(tmp)
        run_id = make_run_id("cpt", "kosub", "mean", "50m", seed=42)
        config = {
            "model": "qwen2.5-0.5b",
            "tokenizer": "kosub_v3",
            "sequence_length": 1024,
            "precision": "bf16",
            "target_tokens": 50_000_000,
        }

        with RunContext(
            run_id, phase="cpt", config=config, seed=42, root=root,
            skip_env_check=True, set_seeds=False,
            model="qwen2.5-0.5b", tokenizer_version="kosub_v3",
            init_method="mean", vocab_size=151_936, target_tokens=50_000_000,
        ) as run:
            run.log("tokenizer_metrics", tokenizer_version="kosub_v3",
                    split="dev", domain="news", n_docs=1000, n_chars=2_400_000,
                    n_bytes=6_900_000, n_tokens=1_050_000,
                    tok_per_char=0.4375, bytes_per_tok=6.571,
                    p50_len=228, p95_len=711, p99_len=1250)
            run.log("train_curve", step=1000, tokens_seen=1_000_000,
                    raw_bytes_seen=3_100_000, train_loss=2.413, dev_bpb=1.207,
                    lr=1e-4, grad_norm=0.83, tok_per_s=8100.0)
            run.log("lm_metrics", checkpoint="checkpoint_50m",
                    tokens_seen=50_000_000, raw_bytes_seen=155_000_000,
                    split="dev", domain="news", n_bytes=6_900_000,
                    total_nll=5_770_000.0, bpb=1.207, bpc=2.981)
            run.log("capability", checkpoint="checkpoint_50m", benchmark="kmmlu",
                    lang="ko", n_items=1400, n_shot=5, metric="accuracy",
                    value=0.483, ci_lo=0.471, ci_hi=0.495)
            run.log("system_bench", model="qwen2.5-0.5b",
                    tokenizer_version="kosub_v3", mode="raw_prompt",
                    raw_chars=10_000, input_tokens=5_000, n_warmup=25, n_runs=100,
                    prefill_ms_mean=182.4, prefill_ms_p95=191.0,
                    ttft_ms_mean=198.7, decode_tok_s_mean=61.2,
                    peak_alloc_mb=3120, peak_reserved_mb=3584)
            run.tokens_seen = 50_000_000
            run.raw_bytes_seen = 155_000_000

        ledger.append_manifest_rows(
            "dev",
            [{"doc_id": "news_00001234", "source": "news", "domain": "news",
              "date": "2025-11-02", "language": "ko", "sha256": "a" * 64,
              "split": "dev", "char_count": 2381, "byte_count": 6137}],
            root=root,
        )

        print(f"run_id = {run_id}\n")
        for table in ("ledger", *ledger.METRIC_TABLES):
            path = ledger.table_path(table, root)
            n = len(ledger.read_rows(table, root))
            print(f"  {path.name:<24} {n} rows")
        print(f"  {'dev.tsv':<24} 1 rows  (manifest)")

        errors = validate(root)
        if errors:
            print("\n검사 실패:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            return 1

        rows = ledger.read_rows("ledger", root)
        print(f"\nLEDGER 상태 전이: {[r['status'] for r in rows]}")
        print(f"config_sha256   : {rows[0]['config_sha256']}")
        print("\n원장 E2E 통과")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
