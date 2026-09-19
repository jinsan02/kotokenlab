# 자원 사용량 실측

`scripts/probe_resources.py` 로 실제 측정. 추정값이 아니다.

```
GPU        NVIDIA GeForce RTX 5070 Ti  16275 MB
CPU        24 logical cores  (AMD64 Family 26 Model 68 Stepping 0, AuthenticAM)
RAM        31 GB total / 12 GB free
torch      2.7.1+cu128   dtype=bf16
attention  SDPA, EFFICIENT+CUDNN 강제 (FLASH 는 이 빌드에 없음)
```

## 학습 (Qwen2.5-1.5B full CPT)

| seq | micro_bs | grad_ckpt | optimizer | VRAM alloc | VRAM resv | 여유 | RAM | sec/step | tok/s |
|---:|---:|:--:|---|---:|---:|---:|---:|---:|---:|
| 2048 | 1 | O | adamw_8bit | 10490 MB | 13458 MB | 2817 MB | 1712 MB | 0.43 | 4761 |

여유 = 장치 총량 − **reserved**. allocated 가 아니다.

