#!/bin/bash
set -euo pipefail

cd "${BASH_SOURCE[0]%/*}"
REPO_DIR=$(pwd)/../../

LLM_MODEL=${QWEN_TARGET_MODEL:-${LLM_MODEL:-}}
SSM_MODEL=${QWEN_DRAFT_MODEL:-${SSM_MODEL:-}}
if [ -z "${LLM_MODEL:-}" ] || [ -z "${SSM_MODEL:-}" ]; then
    echo "QWEN_TARGET_MODEL and QWEN_DRAFT_MODEL must be set" >&2
    exit 1
fi

INPUT_DIR=$REPO_DIR/exps/fig8,9
OUTPUT_DIR=$REPO_DIR/results/fig8,9/qwen
TEST_SCRIPT=${TEST_SCRIPT:-$REPO_DIR/exps/test.sh}

DRAFT_TP=${DRAFT_TP:-1}
TARGET_TP=${TARGET_TP:-3}
RPS_MIN=${RPS_MIN:-2.4}
RPS_MAX=${RPS_MAX:-4.2}
RPS_STEP=${RPS_STEP:-0.2}
OUTPUT_LENGTH=${OUTPUT_LENGTH:-256}

ENABLE_VLLM_SPEC=${ENABLE_VLLM_SPEC:-OFF}
ENABLE_PEARL_SPEC=${ENABLE_PEARL_SPEC:-OFF}
ENABLE_ADASERVE=${ENABLE_ADASERVE:-OFF}
ENABLE_SLOPEARL=${ENABLE_SLOPEARL:-OFF}

run_system() {
    local enable_flag="$1"
    local system_name="$2"
    local rps="$3"
    if [ "$enable_flag" != "ON" ]; then
        return 0
    fi
    OUTPUT_FILE="$OUTPUT_DIR/$system_name/rps${rps}_ol${OUTPUT_LENGTH}.txt" \
    DATASETS_FILE="$INPUT_DIR/emission_rps${rps}_ol${OUTPUT_LENGTH}.json" \
    LLM_MODEL="$LLM_MODEL" \
    SSM_MODEL="$SSM_MODEL" \
    DRAFT_TP="$DRAFT_TP" \
    TARGET_TP="$TARGET_TP" \
    ENABLE_VLLM_SPEC=$([ "$system_name" = "vllm-spec" ] && echo ON || echo OFF) \
    ENABLE_PEARL_SPEC=$([ "$system_name" = "pearl-spec" ] && echo ON || echo OFF) \
    ENABLE_ADASERVE=$([ "$system_name" = "adaserve" ] && echo ON || echo OFF) \
    ENABLE_SLOPEARL=$([ "$system_name" = "slopearl" ] && echo ON || echo OFF) \
    bash "$TEST_SCRIPT"
}

for RPS in $(seq "$RPS_MIN" "$RPS_STEP" "$RPS_MAX"); do
    run_system "$ENABLE_VLLM_SPEC" "vllm-spec" "$RPS"
    run_system "$ENABLE_PEARL_SPEC" "pearl-spec" "$RPS"
    run_system "$ENABLE_ADASERVE" "adaserve" "$RPS"
    run_system "$ENABLE_SLOPEARL" "slopearl" "$RPS"
done
