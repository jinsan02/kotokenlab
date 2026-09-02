#!/bin/sh
# Phase 6-E — 배치 처리량. 사전 등록은 docs/PLAN.md "Phase 6" E 절.
#
# Q6 본 측정은 배치 1 이었다. "메모리를 30% 아끼므로 요청을 1.4배 받는다" 는
# 그 표로 뒷받침되지 않는다. 여기서 잰다.
#
# T2a 는 생략한다 — raw_prompt 에서 토큰 수가 C0 와 같아 배치 거동도 같다는
# 것이 Q6 에서 이미 확인됐다 (널 대조군 2).
set -e
cd /c/llm_tokenizer
P=./.conda/python.exe

for M in cpt_c0_qwen_main_seed42:c0_qwen cpt_t2b_mean_main_seed42:t2b_mean; do
  DIR=$(echo "$M" | cut -d: -f1); NAME=$(echo "$M" | cut -d: -f2)
  echo "########## $NAME ##########"
  $P scripts/run_batch_bench.py --model artifacts/models/$DIR --name $NAME --tag v1
done
echo "########## PHASE 6E DONE ##########"
