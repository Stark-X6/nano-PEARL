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

INPUT_DIR=$REPO_DIR/exps/fig11
OUTPUT_DIR=$REPO_DIR/results/fig11/qwen
TEST_SCRIPT=${TEST_SCRIPT:-$REPO_DIR/exps/test.sh}

DRAFT_TP=${DRAFT_TP:-1}
TARGET_TP=${TARGET_TP:-3}
OUTPUT_LENGTH=${OUTPUT_LENGTH:-256}
SLO_SCALE_MIN=${SLO_SCALE_MIN:-0.6}
SLO_SCALE_MAX=${SLO_SCALE_MAX:-1.6}
SLO_SCALE_STEP=${SLO_SCALE_STEP:-0.2}

ENABLE_VLLM_SPEC=${ENABLE_VLLM_SPEC:-OFF}
ENABLE_PEARL_SPEC=${ENABLE_PEARL_SPEC:-OFF}
ENABLE_ADASERVE=${ENABLE_ADASERVE:-OFF}
ENABLE_SLOPEARL=${ENABLE_SLOPEARL:-OFF}

run_system() {
    local enable_flag="$1"
    local system_name="$2"
    local scale="$3"
    if [ "$enable_flag" != "ON" ]; then
        return 0
    fi
    OUTPUT_FILE="$OUTPUT_DIR/$system_name/slo${scale}_ol${OUTPUT_LENGTH}.txt" \
    DATASETS_FILE="$INPUT_DIR/emission_slo${scale}_ol${OUTPUT_LENGTH}.json" \
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

for SLO_SCALE in $(seq "$SLO_SCALE_MIN" "$SLO_SCALE_STEP" "$SLO_SCALE_MAX"); do
    run_system "$ENABLE_VLLM_SPEC" "vllm-spec" "$SLO_SCALE"
    run_system "$ENABLE_PEARL_SPEC" "pearl-spec" "$SLO_SCALE"
    run_system "$ENABLE_ADASERVE" "adaserve" "$SLO_SCALE"
    run_system "$ENABLE_SLOPEARL" "slopearl" "$SLO_SCALE"
done
