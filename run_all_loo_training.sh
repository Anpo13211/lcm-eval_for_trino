#!/bin/bash
################################################################################
# Leave-One-Out Cross-Validation - 全モデル訓練スクリプト
#
# 使い方:
#   chmod +x run_all_loo_training.sh
#   ./run_all_loo_training.sh
#
# または個別実行:
#   ./run_all_loo_training.sh flat         # Flat Vector のみ
#   ./run_all_loo_training.sh dace         # DACE のみ
#   ./run_all_loo_training.sh queryformer  # QueryFormer のみ
#   ./run_all_loo_training.sh qppnet       # QPPNet のみ
#   ./run_all_loo_training.sh zeroshot     # Zero-Shot のみ
################################################################################

set -e  # エラーで停止

# ディレクトリ設定
PROJECT_ROOT="/home/anpo13211/lcm-eval-copy"
DATA_DIR="/home/anpo13211/lcm-eval-copy/explain_analyze_results"
STATISTICS_DIR="/home/anpo13211/zero-shot_datasets"
OUTPUT_DIR="$PROJECT_ROOT/model_results"
LOGS_DIR="$PROJECT_ROOT/logs"

# ログディレクトリ作成
mkdir -p "$LOGS_DIR"

# デバイス設定
#   DEVICE_MODE: cpu | cuda | cuda_multi
#   CUDA_DEVICE: 単一GPU指定 (DEVICE_MODE=cuda の場合)
#   NUM_GPUS:    torchrun で使用するGPU数 (DEVICE_MODE=cuda_multi の場合)
DEVICE_MODE="cuda_multi"
CUDA_DEVICE="cuda:0"
NUM_GPUS=7
PATIENCE=20

if [ "$DEVICE_MODE" = "cuda_multi" ]; then
    LAUNCH_CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS --tee 3"
    DEVICE_ARG="--device cuda"
    DEVICE_DESC="CUDA multi-GPU (torchrun, nproc=$NUM_GPUS)"
elif [ "$DEVICE_MODE" = "cuda" ]; then
    LAUNCH_CMD="python"
    DEVICE_ARG="--device $CUDA_DEVICE"
    DEVICE_DESC="CUDA single GPU ($CUDA_DEVICE)"
else
    DEVICE_MODE="cpu"
    LAUNCH_CMD="python"
    DEVICE_ARG="--device cpu"
    DEVICE_DESC="CPU"
fi

# NCCL / torchrun watchdog対策（ハング時の強制停止を防ぐ）
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_WATCHDOG_TIMEOUT=7200000
export TORCH_DISTRIBUTED_TIMEOUT=7200

# 共通パラメータ
EPOCHS=100
BATCH_SIZE=512
HIDDEN_DIM=256
LEARNING_RATE=0.001
DBMS="trino"

# モデル固有のハイパーパラメータ（元論文の実装に合わせる）
DACE_HIDDEN_DIM=128
QUERYFORMER_BATCH_SIZE=512
QPPNET_EPOCHS=50
QPPNET_HIDDEN_DIM=128

echo "================================================================================"
echo "Leave-One-Out Cross-Validation Training - All Models"
echo "================================================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Data Directory: $DATA_DIR"
echo "Statistics Directory: $STATISTICS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Device Mode: $DEVICE_DESC"
echo ""
echo "Common Hyperparameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Learning Rate: $LEARNING_RATE"
echo "Model-specific overrides:"
echo "  DACE hidden dim: $DACE_HIDDEN_DIM"
echo "  QueryFormer batch size: $QUERYFORMER_BATCH_SIZE"
echo "  QPPNet epochs: $QPPNET_EPOCHS"
echo "  QPPNet hidden dim: $QPPNET_HIDDEN_DIM"
echo "================================================================================"
echo "Start Time: $(date)"
echo "================================================================================"
echo

# 関数定義
run_flat() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1/5: Flat Vector Model (LightGBM)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Starting: $(date)"
    echo
    
    cd "$PROJECT_ROOT/src"
    python -m training.scripts.train_unified_flat_loo \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/flat_loo" \
        --dbms "$DBMS" \
        --num_boost_round 1000 \
        --lr 0.05
    
    echo
    echo "✅ Flat Vector completed: $(date)"
    echo
}

run_dace() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "2/5: DACE Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Starting: $(date)"
    echo
    
    cd "$PROJECT_ROOT/src"
    eval "$LAUNCH_CMD -m training.scripts.train_unified_dace_loo \
        --data_dir \"$DATA_DIR\" \
        --output_dir \"$OUTPUT_DIR/dace_loo\" \
        --dbms \"$DBMS\" \
        --epochs \"$EPOCHS\" \
        --batch_size \"$BATCH_SIZE\" \
        --hidden_dim \"$DACE_HIDDEN_DIM\" \
        --node_length 22 \
        --lr \"$LEARNING_RATE\" \
        --patience \"$PATIENCE\" \
        $DEVICE_ARG"
    
    echo
    echo "✅ DACE completed: $(date)"
    echo
}

run_queryformer() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "3/5: QueryFormer Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Starting: $(date)"
    echo
    
    cd "$PROJECT_ROOT/src"
    eval "$LAUNCH_CMD -m training.scripts.train_unified_queryformer_loo \
        --data_dir \"$DATA_DIR\" \
        --output_dir \"$OUTPUT_DIR/queryformer_loo\" \
        --dbms \"$DBMS\" \
        --epochs \"$EPOCHS\" \
        --batch_size \"$QUERYFORMER_BATCH_SIZE\" \
        --hidden_dim \"$HIDDEN_DIM\" \
        --lr \"$LEARNING_RATE\" \
        --statistics_dir \"$STATISTICS_DIR\" \
        --patience \"$PATIENCE\" \
        $DEVICE_ARG"
    
    echo
    echo "✅ QueryFormer completed: $(date)"
    echo
}

run_qppnet() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "4/5: QPPNet Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Starting: $(date)"
    echo
    
    cd "$PROJECT_ROOT/src"
    eval "$LAUNCH_CMD -m training.scripts.train_qppnet_loo \
        --data_dir \"$DATA_DIR\" \
        --output_dir \"$OUTPUT_DIR/qppnet_loo\" \
        --epochs \"$QPPNET_EPOCHS\" \
        --batch_size \"$BATCH_SIZE\" \
        --hidden_dim \"$QPPNET_HIDDEN_DIM\" \
        --lr \"$LEARNING_RATE\" \
        --statistics_dir \"$STATISTICS_DIR\" \
        $DEVICE_ARG"
    
    echo
    echo "✅ QPPNet completed: $(date)"
    echo
}

run_zeroshot() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "5/5: Zero-Shot Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Starting: $(date)"
    echo
    
    cd "$PROJECT_ROOT/src"
    eval "$LAUNCH_CMD -m training.scripts.train_unified_zeroshot_loo \
        --data_dir \"$DATA_DIR\" \
        --output_dir \"$OUTPUT_DIR/zeroshot_loo\" \
        --dbms \"$DBMS\" \
        --epochs \"$EPOCHS\" \
        --batch_size \"$BATCH_SIZE\" \
        --hidden_dim \"$HIDDEN_DIM\" \
        --lr \"$LEARNING_RATE\" \
        --statistics_dir \"$STATISTICS_DIR\" \
        --patience \"$PATIENCE\" \
        $DEVICE_ARG"
    
    echo
    echo "✅ Zero-Shot completed: $(date)"
    echo
}

# メイン処理
run_target() {
    case "$1" in
        flat) run_flat ;;
        dace) run_dace ;;
        queryformer) run_queryformer ;;
        qppnet) run_qppnet ;;
        zeroshot) run_zeroshot ;;
        *)
            echo "Unknown target: $1"
            echo "Usage: $0 [flat|dace|queryformer|qppnet|zeroshot|...]"
            exit 1
            ;;
    esac
}

if [ $# -eq 0 ]; then
    run_flat
    run_dace
    run_queryformer
    run_qppnet
    run_zeroshot
else
    for target in "$@"; do
        run_target "$target"
    done
fi

echo "================================================================================"
echo "All requested models completed!"
echo "End Time: $(date)"
echo "================================================================================"
echo
echo "Results saved to:"
[ -f "$OUTPUT_DIR/flat_loo/trino_leave_one_out_results.json" ] && echo "  ✓ $OUTPUT_DIR/flat_loo/trino_leave_one_out_results.json"
[ -f "$OUTPUT_DIR/dace_loo/trino_leave_one_out_results.json" ] && echo "  ✓ $OUTPUT_DIR/dace_loo/trino_leave_one_out_results.json"
[ -f "$OUTPUT_DIR/queryformer_loo/trino_leave_one_out_results.json" ] && echo "  ✓ $OUTPUT_DIR/queryformer_loo/trino_leave_one_out_results.json"
[ -f "$OUTPUT_DIR/qppnet_loo/summary.json" ] && echo "  ✓ $OUTPUT_DIR/qppnet_loo/summary.json"
[ -f "$OUTPUT_DIR/zeroshot_loo/trino_leave_one_out_results.json" ] && echo "  ✓ $OUTPUT_DIR/zeroshot_loo/trino_leave_one_out_results.json"
echo

echo "View results:"
echo "  cd $OUTPUT_DIR"
echo "  cat flat_loo/trino_leave_one_out_results.json | jq '.average_metrics'"
echo "  cat dace_loo/trino_leave_one_out_results.json | jq '.average_metrics'"
echo "  cat queryformer_loo/trino_leave_one_out_results.json | jq '.average_metrics'"
echo "  cat qppnet_loo/summary.json | jq '.'"
echo "  cat zeroshot_loo/trino_leave_one_out_results.json | jq '.average_metrics'"

