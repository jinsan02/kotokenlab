#!/bin/sh
# Step 5~6 전체 실행. GPU 작업은 엄격히 순차다 (CLAUDE.md: 학습 두 개 동시 금지).
#
#   0) 스모크        — 코드가 도는지만 2분에 확인. 실패하면 뒤를 태우지 않는다
#   1) 비율 탐침     — 정렬 코퍼스의 언어 혼합비를 데이터로 고른다  (약 45분)
#   2) 정렬 x 3조건  — 고른 비율로 67MB                             (약 2.5시간)
#   3) 본 CPT x 3조건 — 정렬 산출물에서 이어서 168.5MB              (약 5시간)
#
# 1) 이 있는 이유
#   한국어만으로 정렬했더니 C0 의 영어가 +8.8%, 코드가 +13.1% 나빠졌다 (34MB 지점,
#   회복 없이 가속). backbone 이 얼면 모든 적응이 embedding 에서만 일어나는데
#   tie_word_embeddings 라 그 행렬 하나가 모든 언어의 표현이자 출력 로짓 방향이다.
#   영어·코드를 섞으면 그 토큰들이 직접 gradient 신호를 받아 제자리를 지킨다.
#   비율은 근거 없이 고르지 않고 17MB 짜리 탐침 3개로 정한다.
#
# 판정 기준 (docs/PLAN.md 사전 등록 절)
#   영어·코드가 2 sigma(0.000115) 안에 머무는 비율 중 한국어가 가장 좋은 것.
#   어느 비율도 만족하지 못하면 멈추고 보고한다 — 정렬 자체를 재검토해야 한다.
set -e
cd /c/llm_tokenizer
P=./.conda/python.exe
BASE=Qwen/Qwen2.5-0.5B
REV=060db6499f32faf8b98477b0a26969ef7d8b9987
T2A=artifacts/models/kot2a_v1_n30000_none
T2B=artifacts/models/kot2b_v2_n30000_mean
ALIGN_POOL=22000
CPT_POOL=48000
CPT_SKIP=22000
BUDGET=168500000

echo "########## 0. 스모크 ##########"
$P -m src.training.alignment --model $T2B --name t2b_mean \
   --rungs 1500000,3000000 --pool-docs 1200 --eval-budget 300000 \
   --mix ko=0.6,en=0.2,code=0.2 --tag smoke

echo "########## 1. 혼합비 탐침 (C0 에서, 17MB) ##########"
# C0 로 재는 이유: 수술을 안 받아 embedding 이 이미 정렬돼 있으므로, 여기서
# 나빠지는 것은 전부 정렬 절차가 낸 손해다. 손해가 가장 적은 비율이 답이다.
for M in ko=1.0,en=0.0,code=0.0 ko=0.8,en=0.1,code=0.1 ko=0.6,en=0.2,code=0.2; do
  TAG=$(echo "probe_$M" | tr -d '=.,' | tr 'a-z' 'a-z')
  $P -m src.training.alignment --model $BASE --revision $REV --name c0_qwen \
     --rungs 17000000 --pool-docs $ALIGN_POOL --eval-budget 1000000 \
     --mix "$M" --tag "$TAG"
done
echo "!! 탐침 결과를 보고 --mix 를 고른 뒤 아래를 실행한다 (run_phase3_rest.sh)"
