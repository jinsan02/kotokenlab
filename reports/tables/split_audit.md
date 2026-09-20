# 분할 감사 — 평가 문서가 학습 풀에 있는가

> **이 표는 코드가 적는다.** `.conda/python.exe tools/split_audit.py`
> `data/manifests/*.tsv` 의 `sha256` 열만 읽는다. **본문을 열지 않는다** —
> `final_test` 도 해시만 본다 (RULES 2번의 "개봉" 이 아니다).

정확 일치만 본다. near-dup 은 파이프라인이 split 전에 MinHash 로 걸렀고,
그 결과를 다시 검증하려면 본문이 필요하므로 여기서는 하지 않는다.

## 한국어

| split | 문서 | 고유 해시 | split 내부 중복 |
|---|---:|---:|---:|
| `train` | 1,068,219 | 1,068,219 | 0 |
| `dev` | 59,992 | 59,992 | 0 |
| `final_test` | 58,681 | 58,681 | 0 |

| 겹침 | 문서 수 |
|---|---:|
| `train` ∩ `dev` | 0 |
| `train` ∩ `final_test` | 0 |
| `dev` ∩ `final_test` | 0 |

## 영어 대조군

| split | 문서 | 고유 해시 | split 내부 중복 |
|---|---:|---:|---:|
| `train_control_english` | 46,942 | 46,942 | 0 |
| `dev_control_english` | 2,491 | 2,491 | 0 |
| `final_test_control_english` | 2,527 | 2,527 | 0 |

| 겹침 | 문서 수 |
|---|---:|
| `train_control_english` ∩ `dev_control_english` | 0 |
| `train_control_english` ∩ `final_test_control_english` | 0 |
| `dev_control_english` ∩ `final_test_control_english` | 0 |

## 코드 대조군

| split | 문서 | 고유 해시 | split 내부 중복 |
|---|---:|---:|---:|
| `train_control_code` | 27,197 | 27,197 | 0 |
| `dev_control_code` | 1,478 | 1,478 | 0 |
| `final_test_control_code` | 1,541 | 1,541 | 0 |

| 겹침 | 문서 수 |
|---|---:|
| `train_control_code` ∩ `dev_control_code` | 0 |
| `train_control_code` ∩ `final_test_control_code` | 0 |
| `dev_control_code` ∩ `final_test_control_code` | 0 |

## 판정

**샌 곳이 없다.** 어느 두 split 에도 같은 문서가 없고, split 안에서도
정확 중복이 없다. 문서 단위 분할이 설계대로 지켜졌다.

이 결과가 말하지 않는 것:

- **near-dup.** 문구만 조금 다른 문서는 해시가 다르므로 여기서 안 잡힌다
- **사전학습 오염.** Qwen 이 이 문서들을 봤는지는 이 도구로 알 수 없다
- **평가 문항 오염.** KMMLU 쪽은 `contamination.md` 가 따로 센다
