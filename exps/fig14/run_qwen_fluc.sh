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

INPUT_DIR=$REPO_DIR/exps/fig14
OUTPUT_DIR=$REPO_DIR/results/fig14/qwen
TEST_SCRIPT=${TEST_SCRIPT:-$REPO_DIR/exps/test.sh}

DRAFT_TP=${DRAFT_TP:-1}
TARGET_TP=${TARGET_TP:-3}
OUTPUT_LENGTH=${OUTPUT_LENGTH:-256}

ENABLE_VLLM_SPEC=${ENABLE_VLLM_SPEC:-OFF}
ENABLE_PEARL_SPEC=${ENABLE_PEARL_SPEC:-OFF}
ENABLE_ADASERVE=${ENABLE_ADASERVE:-OFF}
ENABLE_SLOPEARL=${ENABLE_SLOPEARL:-OFF}

run_system() {
    local enable_flag="$1"
    local system_name="$2"
    if [ "$enable_flag" != "ON" ]; then
        return 0
    fi
    OUTPUT_FILE="$OUTPUT_DIR/$system_name/ol${OUTPUT_LENGTH}.txt" \
    DATASETS_FILE="$INPUT_DIR/emission_fluc_6m_peak4.0.json" \
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

run_system "$ENABLE_VLLM_SPEC" "vllm-spec"
run_system "$ENABLE_PEARL_SPEC" "pearl-spec"
run_system "$ENABLE_ADASERVE" "adaserve"
run_system "$ENABLE_SLOPEARL" "slopearl"
