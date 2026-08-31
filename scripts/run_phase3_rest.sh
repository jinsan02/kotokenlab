#!/bin/sh
# Phase 3 의 나머지 — 본 CPT 3조건.
#
#   사용법:  sh scripts/run_phase3_rest.sh
#
# 정렬 단계는 뺐다 (원장: align_t2b_mean_probelr1e5t2b_seed42).
#   손해는 C0 에서, 이득은 T2b 에서 쟀는데 둘을 동시에 만족하는 lr 이 없었다.
#   1e-4 는 T2b 를 3MB 만에 -15.24% 고치지만 C0 의 영어를 +3.61% 깎고,
#   1e-5 는 아무도 안 다치게 하지만 T2b 도 17MB 에서 -3.03% 밖에 못 고친다.
#   tie_word_embeddings 라 임베딩 행렬 하나가 모든 언어의 출력 로짓 방향이므로
#   한국어 행만 움직이는 lr 이 존재하지 않는다.
#
#   예열이 필요하다는 근거도 없다 — 노이즈 플로어에서 T2b CPT 를 정렬 없이
#   3번 돌렸고 전부 정상 종료했다. 정렬을 빼면 조건 간 차이가 토크나이저와
#   수술만 남는다. 그게 우리가 재려는 것이다.
#
# CPT 예산 168.5MB 는 C0 가 50M 토큰을 보는 원문량이다 (실측 0.2968 tok/byte).
# 세 조건이 같은 원문을 보고 T2b 는 그것을 약 34.7M 토큰으로 처리한다 —
# 토큰이 적은 것이 손해가 아니라 그게 압축 개선의 실체다 (RULES 12).
#
# POOL 을 50,000 으로 잡은 이유
#   train.jsonl 은 평균 4,029 B/doc 이라 168.5MB 에 42,034 문서가 필요하다.
#   예전 값(48,000 pool - 22,000 skip = 26,000 문서 = 103.8MB)은 정렬이
#   앞 문서를 먹는 전제였고, 그대로 두면 예산 부족으로 죽었다.
#   50,000 문서 = 201MB 로 19% 여유를 둔다.
#
# seed 는 순서만 바꾼다. 세 조건이 같은 pool/skip/seed 이므로 같은 문서를 본다.
set -e
cd /c/llm_tokenizer
P=./.conda/python.exe
BASE=Qwen/Qwen2.5-0.5B
REV=060db6499f32faf8b98477b0a26969ef7d8b9987
POOL=50000
BUDGET=168500000

echo "########## 본 CPT (3조건) ##########"

$P -m src.training.cpt --model $BASE --revision $REV --name c0_qwen \
   --seed 42 --budget-bytes $BUDGET --pool-docs $POOL \
   --eval-bytes 20000000 --eval-budget 2000000 --tag main --save

for M in kot2a_v1_n30000_none:t2a_none kot2b_v2_n30000_mean:t2b_mean; do
  DIR=$(echo "$M" | cut -d: -f1)
  NAME=$(echo "$M" | cut -d: -f2)
  $P -m src.training.cpt --model artifacts/models/$DIR --name $NAME \
     --seed 42 --budget-bytes $BUDGET --pool-docs $POOL \
     --eval-bytes 20000000 --eval-budget 2000000 --tag main --save
done

echo "########## PHASE 3 DONE ##########"
