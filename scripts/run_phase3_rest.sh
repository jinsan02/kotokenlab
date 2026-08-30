#!/bin/sh
# Phase 3 의 나머지 — 혼합비를 확정한 뒤에 실행한다.
#
#   사용법:  MIX=ko=0.6,en=0.2,code=0.2 sh scripts/run_phase3_rest.sh
#
#   2) 정렬 x 3조건   67MB     약 2.5시간
#   3) 본 CPT x 3조건 168.5MB  약 5시간
#
# CPT 예산 168.5MB 는 C0 가 50M 토큰을 보는 원문량이다 (실측 0.2968 tok/byte).
# 세 조건이 같은 원문을 보고 T2b 는 그것을 약 34.7M 토큰으로 처리한다 —
# 토큰이 적은 것이 손해가 아니라 그게 압축 개선의 실체다 (RULES 12).
#
# --skip-docs 는 정렬이 이미 본 문서를 CPT 가 다시 학습하지 않게 한다.
set -e
cd /c/llm_tokenizer
P=./.conda/python.exe
MIX=${MIX:-ko=0.6,en=0.2,code=0.2}
BASE=Qwen/Qwen2.5-0.5B
REV=060db6499f32faf8b98477b0a26969ef7d8b9987
ALIGN_POOL=22000
CPT_POOL=48000
CPT_SKIP=22000
BUDGET=168500000

echo "########## 2. 정렬 (3조건)  mix=$MIX ##########"
$P -m src.training.alignment --model $BASE --revision $REV --name c0_qwen \
   --pool-docs $ALIGN_POOL --mix "$MIX" --tag v2 --save-final
$P -m src.training.alignment --model artifacts/models/kot2a_v1_n30000_none \
   --name t2a_none --pool-docs $ALIGN_POOL --mix "$MIX" --tag v2 --save-final
$P -m src.training.alignment --model artifacts/models/kot2b_v2_n30000_mean \
   --name t2b_mean --pool-docs $ALIGN_POOL --mix "$MIX" --tag v2 --save-final

echo "########## 3. 본 CPT (3조건) ##########"
for C in c0_qwen t2a_none t2b_mean; do
  $P -m src.training.cpt --model artifacts/models/align_${C}_v2_seed42 \
     --name $C --seed 42 --budget-bytes $BUDGET \
     --pool-docs $CPT_POOL --skip-docs $CPT_SKIP \
     --eval-bytes 20000000 --eval-budget 2000000 --tag main --save
done
echo "########## PHASE 3 DONE ##########"
