#!/bin/bash
set -euo pipefail

cd "${BASH_SOURCE[0]%/*}/.."

if [ -z "${LLM_MODEL:-}" ]; then
    echo "LLM_MODEL is not set" >&2
    exit 1
fi

if [ -z "${SSM_MODEL:-}" ]; then
    echo "SSM_MODEL is not set" >&2
    exit 1
fi

if [ -z "${DATASETS_FILE:-}" ]; then
    echo "DATASETS_FILE is not set" >&2
    exit 1
fi

DRAFT_TP=${DRAFT_TP:-1}
TARGET_TP=${TARGET_TP:-3}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
TEMPERATURE=${TEMPERATURE:-0.0}
NUM_PEARL_STEPS=${NUM_PEARL_STEPS:-100}
BASELINE_LATENCY_PER_TOKEN_MS=${BASELINE_LATENCY_PER_TOKEN_MS:-28}
MAX_GAMMA=${MAX_GAMMA:-16}
MIN_GAMMA=${MIN_GAMMA:-1}
CORRECTION_FACTOR=${CORRECTION_FACTOR:-1.0}
DRY_RUN=${DRY_RUN:-0}

ENABLE_VLLM_SPEC=${ENABLE_VLLM_SPEC:-OFF}
ENABLE_PEARL_SPEC=${ENABLE_PEARL_SPEC:-OFF}
ENABLE_ADASERVE=${ENABLE_ADASERVE:-OFF}
ENABLE_SLOPEARL=${ENABLE_SLOPEARL:-OFF}

print_or_run() {
    local system_name="$1"
    local output_file="$2"
    shift 2
    local cmd=(
        python benchmark_slo/run_workload.py
        --system "$system_name"
        --input-file "$DATASETS_FILE"
        --draft-model "$SSM_MODEL"
        --target-model "$LLM_MODEL"
        --draft-tp "$DRAFT_TP"
        --target-tp "$TARGET_TP"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$MAX_NUM_SEQS"
        --temperature "$TEMPERATURE"
        --num-pearl-steps "$NUM_PEARL_STEPS"
        --baseline-latency-per-token-ms "$BASELINE_LATENCY_PER_TOKEN_MS"
        --max-gamma "$MAX_GAMMA"
        --min-gamma "$MIN_GAMMA"
        --correction-factor "$CORRECTION_FACTOR"
        --ignore-eos
        --enforce-eager
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf '%s ' "${cmd[@]}"
        printf '> %s\n' "$output_file"
        return 0
    fi

    mkdir -p "$(dirname "$output_file")"
    "${cmd[@]}" > "$output_file"
}

run_if_enabled() {
    local flag_value="$1"
    local system_name="$2"
    local output_file="$3"
    if [ "$flag_value" = "ON" ]; then
        print_or_run "$system_name" "$output_file"
    fi
}

OUTPUT_FILE=${OUTPUT_FILE:-}
if [ -n "$OUTPUT_FILE" ]; then
    run_if_enabled "$ENABLE_VLLM_SPEC" "vllm-spec" "$OUTPUT_FILE"
    run_if_enabled "$ENABLE_PEARL_SPEC" "pearl-spec" "$OUTPUT_FILE"
    run_if_enabled "$ENABLE_ADASERVE" "adaserve" "$OUTPUT_FILE"
    run_if_enabled "$ENABLE_SLOPEARL" "slopearl" "$OUTPUT_FILE"
fi
