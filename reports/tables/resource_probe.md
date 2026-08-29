# 자원 사용량 실측

`scripts/probe_resources.py` 로 실제 측정. 추정값이 아니다.

```
GPU        NVIDIA GeForce RTX 5070 Ti  16303 MB
CPU        24 logical cores  (AMD64 Family 26 Model 68 Stepping 0, AuthenticAM)
RAM        31 GB total / 15 GB free
torch      2.7.1+cu128   dtype=bf16
attention  SDPA, EFFICIENT+CUDNN 강제 (FLASH 는 이 빌드에 없음)
```

## 학습 (Qwen2.5-0.5B full CPT)

| seq | micro_bs | grad_ckpt | optimizer | VRAM alloc | VRAM resv | 여유 | RAM | sec/step | tok/s |
|---:|---:|:--:|---|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | O | adamw | 4993 MB | 5990 MB | 10313 MB | 1665 MB | 0.16 | 6292 |
| 1024 | 1 | O | adamw_8bit | 4053 MB | 5300 MB | 11003 MB | 1718 MB | 0.16 | 6287 |
| 1024 | 1 | X | adamw_8bit | 5377 MB | 5802 MB | 10501 MB | 1737 MB | 0.14 | 7296 |
| 1024 | 4 | O | adamw_8bit | 10433 MB | 13326 MB | 2977 MB | 1832 MB | 0.48 | 8597 |
| 2048 | 1 | O | adamw_8bit | 6183 MB | 7716 MB | 8587 MB | 1852 MB | 0.27 | 7489 |
| 2048 | 2 | O | adamw_8bit | 10433 MB | 13326 MB | 2977 MB | 1857 MB | 0.45 | 9089 |
| 4096 | 1 | O | adamw_8bit | 10434 MB | 13326 MB | 2977 MB | 1862 MB | 0.47 | 8795 |
| 8192 | 1 | O | adamw_8bit | 18938 MB | 23132 MB | -6829 MB | 11596 MB | 2.22 | 3692 |

## 추론 prefill (Qwen2.5-0.5B)

생성 경로 (`logits_to_keep=1`) — 다음 토큰 하나만 계산한다

| 입력 토큰 | 한국어 원문(대략) | VRAM alloc | VRAM resv | KV cache | prefill ms | RAM |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | ~1,553자 | 1016 MB | 1066 MB | 12 MB | 23.0 | 2077 MB |
| 4,096 | ~6,215자 | 1158 MB | 1240 MB | 48 MB | 80.7 | 2024 MB |
| 8,192 | ~12,430자 | 1349 MB | 1430 MB | 96 MB | 183.4 | 2025 MB |
| 16,384 | ~24,861자 | 1731 MB | 1810 MB | 192 MB | 471.6 | 2026 MB |
| 26,000 | ~39,453자 | 2184 MB | 2384 MB | 305 MB | 918.3 | 2028 MB |
| 32,768 | ~49,723자 | 2495 MB | 2718 MB | 384 MB | 1309.6 | 2029 MB |

백엔드를 강제하지 않으면 (기본 디스패처 = MATH 로 폴백)

| 입력 토큰 | VRAM alloc | prefill ms | 강제 대비 |
|---:|---:|---:|---|
| 4,096 | 3184 MB | 590.1 | 메모리 2.7배, 시간 7.4배 |
| 8,192 | 9561 MB | 1891.6 | 메모리 7.1배, 시간 10.3배 |

## 추론 prefill (Qwen2.5-1.5B)

생성 경로 (`logits_to_keep=1`) — 다음 토큰 하나만 계산한다

| 입력 토큰 | 한국어 원문(대략) | VRAM alloc | VRAM resv | KV cache | prefill ms | RAM |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | ~1,553자 | 3056 MB | 3246 MB | 28 MB | 58.3 | 2575 MB |
| 4,096 | ~6,215자 | 3335 MB | 3534 MB | 112 MB | 219.3 | 2576 MB |
| 8,192 | ~12,430자 | 3706 MB | 3928 MB | 224 MB | 506.1 | 2577 MB |
| 16,384 | ~24,861자 | 4450 MB | 4980 MB | 448 MB | 1182.3 | 2578 MB |
| 26,000 | ~39,453자 | 5324 MB | 6066 MB | 711 MB | 2156.8 | 2582 MB |
| 32,768 | ~49,723자 | 5938 MB | 6808 MB | 896 MB | 2972.0 | 3352 MB |

백엔드를 강제하지 않으면 (기본 디스패처 = MATH 로 폴백)

| 입력 토큰 | VRAM alloc | prefill ms | 강제 대비 |
|---:|---:|---:|---|
| 4,096 | 5021 MB | 868.5 | 메모리 1.5배, 시간 4.0배 |
| 8,192 | 10662 MB | 3179.8 | 메모리 2.9배, 시간 6.5배 |

