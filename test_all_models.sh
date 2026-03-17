#!/bin/bash
# =============================================================
# Test all model presets
#
# Two modes:
#   1. Quick smoke test (default): 1 run, affair only, noise=0
#   2. Full benchmark: 10 runs, all secrets, noise=0,10,20,50,100
#
# Usage:
#   bash test_all_models.sh --gpus 0                        # smoke test all, GPU 0
#   bash test_all_models.sh --gpus 0,1 --full               # full benchmark, GPU 0,1
#   bash test_all_models.sh --gpus 0 gemma3-27b             # smoke test one model
#   bash test_all_models.sh --gpus 0,1 --full gemma3-27b    # full benchmark one model
# =============================================================

set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ---- Default models ----
MODELS=(
    "qwen3-32b"
    "gemma3-27b"
    "phi4-reasoning-plus"
)

# ---- Parse args ----
FULL_MODE=false
FILTER=""
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            FULL_MODE=true
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            FILTER="$1"
            shift
            ;;
    esac
done

if [ -z "$GPUS" ]; then
    echo "Error: --gpus is required. Example: --gpus 0  or  --gpus 0,1"
    exit 1
fi

if [ -n "$FILTER" ]; then
    MODELS=("$FILTER")
fi

# ---- Set test params based on mode ----
if [ "$FULL_MODE" = true ]; then
    SECRET="all"
    N_NOISE="0,10,20,50,100"
    N_RUNS=10
    MODE_LABEL="FULL BENCHMARK"
else
    SECRET="affair"
    N_NOISE="0"
    N_RUNS=1
    MODE_LABEL="SMOKE TEST"
fi

# ---- Run ----
echo "========================================"
echo "  $MODE_LABEL"
echo "  GPUs=$GPUS  secrets=$SECRET  noise=$N_NOISE  runs=$N_RUNS"
echo "========================================"
echo ""

PASSED=()
FAILED=()

for MODEL in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Testing: $MODEL  (GPUs: $GPUS)"
    echo "----------------------------------------"

    if CUDA_VISIBLE_DEVICES="$GPUS" python test_eval_easy.py \
        --model "$MODEL" \
        --secret "$SECRET" \
        --n_noise "$N_NOISE" \
        --n_runs "$N_RUNS" 2>&1 | tee "/tmp/test_${MODEL}.log"; then

        if grep -q "Overall" "/tmp/test_${MODEL}.log"; then
            echo ""
            echo "  ✅ $MODEL PASSED"
            PASSED+=("$MODEL")
        else
            echo ""
            echo "  ❌ $MODEL FAILED (no results found, see /tmp/test_${MODEL}.log)"
            FAILED+=("$MODEL")
        fi
    else
        echo ""
        echo "  ❌ $MODEL FAILED (see /tmp/test_${MODEL}.log)"
        FAILED+=("$MODEL")
    fi
    echo ""
done

# ---- Summary ----
echo ""
echo "========================================"
echo "  Summary ($MODE_LABEL)"
echo "========================================"
echo ""
echo "Passed (${#PASSED[@]}):"
for m in "${PASSED[@]}"; do
    echo "  ✅ $m"
done
echo ""
echo "Failed (${#FAILED[@]}):"
for m in "${FAILED[@]}"; do
    echo "  ❌ $m"
done
echo ""