#!/bin/bash

TASK="tactic_prediction_context"
MAX_ITERS=100
NUM_SAMPLES=32
TEMPERATURES=0.0
DATASET="pfr"
DATA="data/minictx/pfr.jsonl"
PREMISE="data/minictx-declarations/PFR-declarations"

MODEL="l3lab/ntp-mathlib-context-deepseek-coder-1.3b"
NAME="deepseekCT"

OUTPUT_DIR="output/${NAME}_pfr"

REPL_PATH="./repl"
LEAN_PATH="./test-envs/pfr"

python check_dev.py --task ${TASK} --model-name ${MODEL} --dataset-name ${DATASET} --dataset-path ${DATA} --premise-path ${PREMISE} --output-dir ${OUTPUT_DIR} --max-iters ${MAX_ITERS} --num-samples ${NUM_SAMPLES} --temperatures ${TEMPERATURES} --repl-path ${REPL_PATH} --lean-env-path ${LEAN_PATH} > "pfr_premise.out"