# 가상환경

환경은 설치하고 잊는 것이 아니라 **버전이 찍히는 실험 변수**다.
torch 나 transformers 를 올리면 결과가 달라질 수 있고, 그 사실이 원장에 없으면
나중에 두 실험을 비교할 수 없다.

## 하드웨어

```
GPU     NVIDIA GeForce RTX 5070 Ti
VRAM    16 GB  (16303 MiB)
Driver  610.88
Compute sm_120 (Blackwell)
```

sm_120 이라 **CUDA 12.8 이상 빌드의 torch 가 필요하다**. cu121/cu124 휠은 이 GPU 에서
커널을 못 찾는다.

## 위치와 생성

환경은 `C:\llm_tokenizer\.conda` 안에 산다. **git 에는 들어가지 않는다.**

```bash
C:/Miniconda3/Scripts/conda.exe create -p C:/llm_tokenizer/.conda python=3.11 -y
C:/llm_tokenizer/.conda/python.exe -m pip install --upgrade pip
C:/llm_tokenizer/.conda/python.exe -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
C:/llm_tokenizer/.conda/python.exe -m pip install -r env/requirements-lock.txt
```

`conda activate` 를 쓰지 않고 **인터프리터 절대경로**를 쓴다. 셸 상태에 의존하지
않아야 에이전트든 배치 스크립트든 같은 환경을 잡는다.

```
C:\llm_tokenizer\.conda\python.exe
```

## 버전 관리 파일

| 파일 | 역할 | git |
|---|---|---|
| `env/environment.yml` | 선언적 재현 스펙 (사람이 읽는 쪽) | 커밋 |
| `env/requirements-lock.txt` | `pip freeze` 스냅샷 (기계가 읽는 쪽) | 커밋 |
| `env/ENV_SNAPSHOT.tsv` | 환경 변경 이력 원장 (append-only) | 커밋 |
| `.conda/` | 실제 바이너리 | **제외** |

## 강제 규칙

`src/utils/env.py` 가 실행 시점의 버전들을 모아 정규화 JSON 의 sha256 을 만든다
(`env_sha256`). `RunContext` 는 진입할 때 이 해시가 `ENV_SNAPSHOT.tsv` 에 있는지 본다.

**없으면 실험이 시작되지 않는다.**

패키지를 하나라도 바꿨다면:

```bash
.conda/python.exe -m src.utils.env --register "transformers 4.51 -> 4.57, Qwen3 지원"
.conda/python.exe -m pip freeze > env/requirements-lock.txt
git add env/ && git commit    # chore(infra): ...
```

이렇게 하면 모든 실험 행의 `env_sha256` 가 "이 결과는 어느 환경에서 나왔나"에 답한다.

### 조회

```bash
.conda/python.exe -m src.utils.env          # 현재 환경과 등록 여부
.conda/python.exe -m src.utils.env --check  # 미등록이면 종료코드 1
```

## 설치된 것 (최초 등록 스냅샷)

```
env_sha256  6b03dffe5ba1a24143e55927f16d8ee8373a4f6155c6cf520784c79b9a1cae09
python 3.11.15   cuda 12.8   cudnn 90701   driver 610.88
```

| 패키지 | 버전 | 왜 |
|---|---|---|
| `torch` | 2.7.1+cu128 | sm_120 동작이 확인된 조합 |
| `transformers` | 5.16.1 | Qwen2.5 tokenizer surgery |
| `tokenizers` | 0.23.1 | BBPE 학습·병합 규칙 조작 |
| `datasets` | 5.0.1 | corpus 스트리밍 |
| `accelerate` | 1.14.0 | gradient checkpointing / 혼합정밀 |
| `peft` | 0.20.0 | 1.5B LoRA / QLoRA (스펙 §55) |
| `bitsandbytes` | 0.50.2 | 8bit AdamW, NF4 4-bit 추론 (스펙 §27) |
| `sentencepiece` | 0.2.2 | 외부 토크나이저 비교 (HCX / A.X) |
| `kiwipiepy` | 0.23.2 | fertility 의 "정답 분절" 기준 (스펙 §14). **`env_sha256` 에 포함** |
| `psutil` | 7.2.2 | RAM 계측 |
| `numpy` `pandas` `scipy` `matplotlib` `tqdm` `pyyaml` `pytest` | — | 분석·리포트·테스트 |

전체 목록은 `env/requirements-lock.txt`, 변경 이력은 `env/ENV_SNAPSHOT.tsv`.

## attention 백엔드 — 이 환경에서 가장 위험한 함정

측정 결과 이 torch 빌드에서 사용 가능한 백엔드는 이렇다:

```
FLASH          X   "Torch was not compiled with flash attention"
MEM_EFFICIENT  O   1.01 ms /   21 MB
CUDNN          O   0.50 ms /   21 MB
MATH           O  16.85 ms / 2,158 MB      (1x14x4096x64, bf16, causal)
```

문제는 **`attn_implementation="sdpa"` 만 주면 transformers 경로에서 디스패처가
MATH 로 떨어진다**는 것이다:

| 8,192 토큰 prefill | VRAM | 시간 |
|---|---:|---:|
| 기본 | 9,561 MB | 1,892 ms |
| `mem_efficient+cudnn` 강제 | **1,430 MB** | **183 ms** |

메모리 7.1배, 시간 10.3배. MATH 는 `n×n` attention 행렬을 실제로 만들기 때문이다.
이걸 모르고 진행했으면 16k 토큰 이상은 전부 OOM 이고, 측정한 지연은 커널 비효율을
잰 것이지 토크나이저 효과가 아니게 된다.

**그래서 모든 forward 를 감싼다:**

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

EFFICIENT_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="sdpa")
with sdpa_kernel(EFFICIENT_SDPA):
    out = model(input_ids=ids)
```

`attn_backend` 는 `env_sha256` 에 포함된다 (현재 `mem_efficient+cudnn`).
백엔드가 달라지면 다른 환경이고, `RunContext` 가 실행을 막는다.

스펙 §120 의 Flash Attention 역할은 이 조합이 대신한다.
`flash-attn` 은 Windows/sm_120 휠이 없어 설치하지 않는다.

## 시간 검증은 환경 해시와 분리한다

패키지·GPU 환경은 비교적 고정되지만 시계 동기화 상태는 매일 바뀐다. 따라서
`env_sha256` 에 시간을 넣지 않고 `experiments/clock_checks.tsv` 에 별도로 기록한다.
실험 전 아래 명령으로 외부 HTTPS 시각과의 오차를 확인해야 한다.

```bash
C:/llm_tokenizer/.conda/python.exe tools/check_clock.py --record
```

성공한 검사는 24시간 유효하고 이후 `RunContext` 행의 `clock_check_sha256`로 연결된다.
Windows Time 서비스가 Local CMOS Clock을 원본으로 쓰더라도 HTTPS 오차 검사는 가능하지만,
운영체제 자체 재동기화는 관리자 권한으로 별도 수행한다.

## 다른 환경과 섞지 않는다

`C:\aimers\.conda` 는 이전 프로젝트(KBO 예측)의 환경이다. 거기서 검증된
torch 2.7.1+cu128 조합을 **가져오기만** 했고, 그 환경 자체는 건드리지 않는다.
그쪽 transformers 를 올리면 그쪽 재현성이 깨진다.
