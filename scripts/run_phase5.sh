#!/bin/sh
# Phase 5 — Q6 시스템 벤치마크. 사전 등록은 docs/PLAN.md "Q6 시스템 벤치마크".
#
#   사용법:  sh scripts/run_phase5.sh
#
# 압축 -30.2% 가 품질에서는 값을 하지 못했다 (Q4 부정). 남은 물음은 지연과
# 메모리에서는 값을 하는가다. T2b 가 이길 수 있는 유일한 축이다.
#
# 대상은 CPT 를 마친 체크포인트 3종 — 실제로 배포할 물건이다.
# 각 모델이 raw_prompt 4단 + equal_tokens 4단을 돈다.
#
# 예측 3번(equal_tokens 에서 T2b ≈ C0)은 **널 대조군** 이다. vocab 도
# 아키텍처도 같으니 차이가 나오면 계측기가 고장난 것이다. 결과를 해석하기
# 전에 이것부터 본다.
set -e
cd /c/llm_tokenizer
P=./.conda/python.exe

for M in cpt_c0_qwen_main_seed42:c0_qwen \
         cpt_t2a_none_main_seed42:t2a_none \
         cpt_t2b_mean_main_seed42:t2b_mean; do
  DIR=$(echo "$M" | cut -d: -f1)
  NAME=$(echo "$M" | cut -d: -f2)
  echo "########## $NAME ##########"
  $P scripts/run_system_bench.py --model artifacts/models/$DIR \
     --name $NAME --tag v1
done

echo "########## PHASE 5 DONE ##########"
