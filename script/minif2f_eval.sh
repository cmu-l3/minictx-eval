#!/bin/bash

TASK="tactic_prediction_context"
MAX_ITERS=100
NUM_SAMPLES=32
TEMPERATURES=0.0
DATASET="minif2f"
DATA="data/minif2f.jsonl"

MODEL="l3lab/ntp-mathlib-context-deepseek-coder-1.3b"
NAME="deepseekCT"

OUTPUT_DIR="output/${NAME}_minif2f"

REPL_PATH="./repl"
LEAN_PATH="./miniF2F-lean4"

python check.py --task ${TASK} --model-name ${MODEL} --dataset-name ${DATASET} --dataset-path ${DATA} --output-dir ${OUTPUT_DIR} --max-iters ${MAX_ITERS} --num-samples ${NUM_SAMPLES} --temperatures ${TEMPERATURES} --repl-path ${REPL_PATH} --lean-env-path ${LEAN_PATH} > "minif2f_context.out"