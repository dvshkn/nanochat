#!/bin/bash
#
# Implementation of G_stack model growth training:
# https://arxiv.org/abs/2405.15319v2
#

# Like miniseries.sh but with layer stacking
# Usage: ./stackseries.sh [series_name]
# Example: ./stackseries.sh jan11
# Default series name is today's date (e.g., jan11)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Setup (skip with SKIP_SETUP=1)
if [ -z "$SKIP_SETUP" ]; then
    # uv
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate

    # Tokenizer, download 1000 shards for pretraining
    # (probably this can be reduced but it's tricky to determine the exact right number, TODO).
    python -m nanochat.dataset -n 1000
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
else
    source .venv/bin/activate
fi

# Series name: from arg, env var, or default to today's date (e.g., jan11)
SERIES_NAME="${1:-${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
# Data ratios to train with fixed depth, unlike regular miniseries
d=12
DATA_RATIOS=(9.5 8.5 7.5 6.5)
# Hardware
# NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NPROC_PER_NODE=1 # For single GPU
# Logging
WANDB_RUN="${WANDB_RUN:-${SERIES_NAME}_stackseries}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_stackseries_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,model_dim,num_params,num_scaling_params,num_iterations,tokens_trained,param_data_ratio,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "${SERIES_NAME} Stackseries Training"
log "=============================================="

for dr in "${DATA_RATIOS[@]}"; do
    # Growth factor `g` from the paper controls how much to stack, doing simple
    # doubling of layers here
    G=2

    # Inventing some simple nomenclature here for DEPTH_DESC even though we will
    # only be stacking once. Example:
    #
    # d8s16s32 -- train at depth 8
    #             stack to depth 16 and train (future phase)
    #             stack to depth 32 and train (future phase, final depth)
    DEPTH_2=${d}
    DEPTH_1=$((d / G))
    DR_2=6.5
    DR_1=${dr}
    DEPTH_DESC_2="sdr${DR_1}bdr${DR_2}d${DEPTH_2}"
    DEPTH_DESC_1="sdr${DR_1}bdr${DR_2}d${DEPTH_1}s${DEPTH_2}"
    TAG_2="${SERIES_NAME}_stackseries_${DEPTH_DESC_2}"
    TAG_1="${SERIES_NAME}_stackseries_${DEPTH_DESC_1}"

    # Reduce --device-batch-size to avoid OOM at larger depths
    if [ $d -ge 28 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=8"
    elif [ $d -ge 20 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=16"
    else
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=32"
    fi

    # Train at smaller size ---------------------------------------------------

    log "Training ${DEPTH_DESC_1}"
    START_TIME=$(date +%s)

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$DEPTH_1 \
        --final-depth=$DEPTH_2 \
        --target-param-data-ratio=$DR_1 \
        --run="${WANDB_RUN}_${DEPTH_DESC_1}" \
        --model-tag="${TAG_1}" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        $DEVICE_BATCH_SIZE_ARG \
        2>&1 | tee "$RESULTS_DIR/${TAG_1}_train.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
    fi

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract stats from log
    LOG_FILE="$RESULTS_DIR/${TAG_1}_train.log"
    NUM_PARAMS=$(grep "Parameter counts:" -A 6 "$LOG_FILE" | grep -oP 'total\s+: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
    NUM_LM_HEAD_PARAMS=$(grep "Parameter counts:" -A 6 "$LOG_FILE" | grep -oP 'lm_head\s+: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
    NUM_TX_MATRIX_PARAMS=$(grep "Parameter counts:" -A 6 "$LOG_FILE" | grep -oP 'transformer_matrices\s+: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
    NUM_SCALING_PARAMS=$((NUM_LM_HEAD_PARAMS + NUM_TX_MATRIX_PARAMS))
    NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
    TOKENS_TRAINED=$((NUM_ITERS * 524288))
    PARAM_DATA_RATIO=$(python -c "print(f'{$TOKENS_TRAINED / $NUM_SCALING_PARAMS:.2f}')")
    MODEL_DIM=$((d * 64))
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')

    if [ -z "$CORE_SCORE" ]; then
        CORE_SCORE="0.0"
    fi

    log "  d=$DEPTH_1: params=$NUM_PARAMS, scaling=$NUM_SCALING_PARAMS, ratio=$PARAM_DATA_RATIO, bpb=$VAL_BPB, CORE=$CORE_SCORE, time=${TRAIN_TIME}s"

    # Append to CSV
    echo "$DEPTH_1,$MODEL_DIM,$NUM_PARAMS,$NUM_SCALING_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$PARAM_DATA_RATIO,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"

    # Stack time --------------------------------------------------------------

    python -m scripts.stack \
        --g=$G \
        --src-model-tag=$TAG_1 \
        --dest-model-tag=$TAG_2 \
        2>&1 | tee "$RESULTS_DIR/${DEPTH_DESC_1}_${DEPTH_DESC_2}_stacking.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
    fi

    # Train at larger size ----------------------------------------------------

    log "Training ${DEPTH_DESC_2}"
    START_TIME=$(date +%s)

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$DEPTH_2 \
        --target-param-data-ratio=$DR_2 \
        --warmdown-ratio=1 \
        --resume-from-step=0 \
        --run="${WANDB_RUN}_${DEPTH_DESC_2}" \
        --model-tag="${TAG_2}" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        $DEVICE_BATCH_SIZE_ARG \
        2>&1 | tee "$RESULTS_DIR/${TAG_2}_train.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
    fi

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract stats from log
    LOG_FILE="$RESULTS_DIR/${TAG_2}_train.log"
    NUM_PARAMS=$(grep "Parameter counts:" -A 6 "$LOG_FILE" | grep -oP 'total\s+: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
    NUM_LM_HEAD_PARAMS=$(grep "Parameter counts:" -A 6 "$LOG_FILE" | grep -oP 'lm_head\s+: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
    NUM_TX_MATRIX_PARAMS=$(grep "Parameter counts:" -A 6 "$LOG_FILE" | grep -oP 'transformer_matrices\s+: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
    NUM_SCALING_PARAMS=$((NUM_LM_HEAD_PARAMS + NUM_TX_MATRIX_PARAMS))
    NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
    TOKENS_TRAINED=$((NUM_ITERS * 524288))
    PARAM_DATA_RATIO=$(python -c "print(f'{$TOKENS_TRAINED / $NUM_SCALING_PARAMS:.2f}')")
    MODEL_DIM=$((d * 64))
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')

    if [ -z "$CORE_SCORE" ]; then
        CORE_SCORE="0.0"
    fi

    log "  d=$DEPTH_2: params=$NUM_PARAMS, scaling=$NUM_SCALING_PARAMS, ratio=$PARAM_DATA_RATIO, bpb=$VAL_BPB, CORE=$CORE_SCORE, time=${TRAIN_TIME}s"

    # Append to CSV
    echo "$DEPTH_2,$MODEL_DIM,$NUM_PARAMS,$NUM_SCALING_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$PARAM_DATA_RATIO,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
done

log "=============================================="
log "${SERIES_NAME} Stackseries Complete!"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
