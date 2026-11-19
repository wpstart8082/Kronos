#!/usr/bin/env bash
# End-to-end A-share finetune pipeline, to be run INSIDE the kronos-finetune
# container (see docker/finetune.Dockerfile for the image and the docker run
# command with GPU + mongo network + volume mounts).
#
# Usage:
#   bash finetune/run_finetune.sh <stage>
# Stages:
#   audit        MongoDB coverage audit (read-only)
#   export       MongoDB -> hfq-adjusted qlib binary dump
#   verify       read back the qlib dump and sanity-check it
#   preprocess   qlib_data_preprocess.py -> train/val/test pickles
#   selfcheck    dataset.py sample-shape self check
#   smoke        1-epoch tiny-budget trainings (FT_SMOKE=1), tokenizer+predictor
#   tokenizer    full tokenizer finetune (torchrun, 1 GPU)
#   predictor    full predictor finetune (torchrun, 1 GPU)
#   backtest     inference + qlib TopkDropout backtest vs SH000300
#   baseline     zero-shot backtest with the original pretrained checkpoints
#   all          audit -> export -> verify -> preprocess -> selfcheck
#
# Every stage logs to /workspace/outputs/logs/<stage>.log and is idempotent
# (safe to re-run; it overwrites that stage's own outputs only).

set -euo pipefail

STAGE="${1:-all}"
LOGDIR="/workspace/outputs/logs"
mkdir -p "$LOGDIR"

NGPU="${NGPU:-1}"

# The train scripts resolve `from model.kronos import ...` via sys.path.append("../"),
# which is CWD-relative -> they MUST be launched from inside finetune/.
cd /workspace/finetune

run_logged() {
    local name="$1"; shift
    echo "[$(date '+%F %T')] === stage: $name ==="
    "$@" 2>&1 | tee "$LOGDIR/${name}.log"
}

case "$STAGE" in
    audit)
        run_logged audit python data_prep/audit_mongo_coverage.py
        ;;
    export)
        run_logged export python data_prep/export_mongo_to_qlib.py \
            --out /workspace/data/qlib_data/cn_data_as
        ;;
    verify)
        run_logged verify python data_prep/export_mongo_to_qlib.py \
            --verify --out /workspace/data/qlib_data/cn_data_as
        ;;
    preprocess)
        run_logged preprocess python qlib_data_preprocess.py
        ;;
    selfcheck)
        run_logged selfcheck python dataset.py
        ;;
    smoke)
        FT_SMOKE=1 run_logged smoke_tokenizer \
            torchrun --standalone --nproc_per_node="$NGPU" train_tokenizer.py
        FT_SMOKE=1 run_logged smoke_predictor \
            torchrun --standalone --nproc_per_node="$NGPU" train_predictor.py
        ;;
    tokenizer)
        run_logged train_tokenizer \
            torchrun --standalone --nproc_per_node="$NGPU" train_tokenizer.py
        ;;
    predictor)
        run_logged train_predictor \
            torchrun --standalone --nproc_per_node="$NGPU" train_predictor.py
        ;;
    backtest)
        run_logged backtest python qlib_test.py --device cuda:0
        ;;
    backtest_only)
        # reuse a saved predictions.pkl (skips the ~5.5h inference)
        run_logged backtest_only python backtest_only.py "${2:?predictions.pkl path required}"
        ;;
    baseline)
        FT_RESULT_NAME=as_baseline_v1 \
        FT_TOKENIZER_PATH=/workspace/models/Kronos-Tokenizer-base \
        FT_MODEL_PATH=/workspace/models/Kronos-base \
        run_logged baseline_backtest python qlib_test.py --device cuda:0
        ;;
    all)
        bash "$0" audit
        bash "$0" export
        bash "$0" verify
        bash "$0" preprocess
        bash "$0" selfcheck
        ;;
    *)
        echo "unknown stage: $STAGE" >&2
        exit 1
        ;;
esac
echo "[$(date '+%F %T')] stage $STAGE finished"
