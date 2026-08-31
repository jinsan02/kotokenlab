#!/bin/sh
# 정렬 탐침 — **완료됐고, 결론은 정렬 단계를 뺀다는 것이었다.**
#
# 이 스크립트는 재현용으로 남긴다. 원장의 align_c0_qwen_probe_* 행이 여기서
# 나왔다. 결과와 판단은 reports/tables/alignment_probe.md, 설계 변경 이유는
# docs/DESIGN_DELTA.md 1-5 에 있다. 본 CPT 는 scripts/run_phase3_rest.sh 다.
#
#   0) 스모크     — 코드가 도는지만 2분에 확인. 실패하면 뒤를 태우지 않는다
#   1) 비율 탐침  — 정렬 코퍼스의 언어 혼합비를 데이터로 고른다  (약 45분)
#
# 1) 이 있던 이유
#   한국어만으로 정렬했더니 C0 의 영어가 +8.8%, 코드가 +13.1% 나빠졌다 (34MB
#   지점, 회복 없이 가속). backbone 이 얼면 모든 적응이 embedding 에서만
#   일어나는데 tie_word_embeddings 라 그 행렬 하나가 모든 언어의 표현이자 출력
#   로짓 방향이다. 영어·코드를 섞으면 그 토큰들이 직접 gradient 신호를 받아
#   제자리를 지킬 것으로 봤다.
#
#   **가설은 반증됐다.** ko100 -> ko60 에서 영어 손해가 +3.61% -> +3.18% 로
#   사실상 평평했다. 코퍼스의 40% 를 영어·코드로 채워도 소용이 없었다.
#
# 판정 기준 (docs/PLAN.md "정렬 혼합비" 절에 사전 등록)
#   영어·코드가 2 sigma(0.000115) 안에 머무는 비율 중 한국어가 가장 좋은 것.
#   어느 비율도 만족하지 못하면 멈추고 보고한다.
#
#   세 비율 모두 탈락해서 멈췄다. 다만 그 기준 자체가 나빴다 — 영어 손해는
#   기준의 260배라 애초에 통과 가능한 비율이 없었다. sigma 는 CPT 의 seed
#   노이즈이지 정렬 손해의 척도가 아니다. 자세한 것은 PLAN.md 같은 절.
set -e
cd /c/llm_tokenizer
P=./.conda/python.exe
BASE=Qwen/Qwen2.5-0.5B
REV=060db6499f32faf8b98477b0a26969ef7d8b9987
T2A=artifacts/models/kot2a_v1_n30000_none
T2B=artifacts/models/kot2b_v2_n30000_mean
ALIGN_POOL=22000

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

echo "########## 2. lr 탐침 (혼합비가 답이 아니었으므로) ##########"
# 손해는 C0 에서, 이득은 임베딩이 실제로 망가진 T2b 에서 따로 잰다.
# 1e-5 는 아무도 안 다치게 하지만(en +0.01%) T2b 도 안 고친다(ko -3.03%).
# 두 줄이 함께 있어야 "쓸 만한 작동점이 없다" 는 결론이 성립한다.
$P -m src.training.alignment --model $BASE --revision $REV --name c0_qwen \
   --rungs 17000000 --pool-docs $ALIGN_POOL --eval-budget 1000000 \
   --mix ko=1.0,en=0.0,code=0.0 --lr 1e-5 --tag probelr1e5ko100

$P -m src.training.alignment --model $T2B --name t2b_mean \
   --rungs 17000000 --pool-docs $ALIGN_POOL --eval-budget 1000000 \
   --mix ko=1.0,en=0.0,code=0.0 --lr 1e-5 --tag probelr1e5t2b

echo "########## 결론: 정렬을 뺀다. 본 CPT 는 run_phase3_rest.sh ##########"
