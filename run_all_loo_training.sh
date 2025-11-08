#!/bin/bash
################################################################################
# Leave-One-Out Cross-Validation - 全モデル訓練スクリプト
#
# 使い方:
#   chmod +x run_all_loo_training.sh
#   ./run_all_loo_training.sh
#
# または個別実行:
#   ./run_all_loo_training.sh flat      # Flat Vector のみ
#   ./run_all_loo_training.sh dace      # DACE のみ
#   ./run_all_loo_training.sh queryformer  # QueryFormer のみ
#   ./run_all_loo_training.sh zeroshot  # Zero-Shot のみ
################################################################################

set -e  # エラーで停止

# ディレクトリ設定
PROJECT_ROOT="/home/anpo13211/lcm-eval-copy"
DATA_DIR="/home/anpo13211/lcm-eval-copy/explain_analyze_results"
STATISTICS_DIR="/home/anpo13211/zero-shot_datasets"
OUTPUT_DIR="$PROJECT_ROOT/models"
LOGS_DIR="$PROJECT_ROOT/logs"

# ログディレクトリ作成
mkdir -p "$LOGS_DIR"

# デバイス設定（GPUがある場合は cuda:0、なければ cpu）
DEVICE="cpu"

# 共通パラメータ（全モデルで統一）
EPOCHS=100
BATCH_SIZE=32
HIDDEN_DIM=256
LEARNING_RATE=0.001
DBMS="trino"

echo "================================================================================"
echo "Leave-One-Out Cross-Validation Training - All Models"
echo "================================================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Data Directory: $DATA_DIR"
echo "Statistics Directory: $STATISTICS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""
echo "Unified Hyperparameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Learning Rate: $LEARNING_RATE"
echo "================================================================================"
echo "Start Time: $(date)"
echo "================================================================================"
echo

# 関数定義
run_flat() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1/4: Flat Vector Model (LightGBM)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Estimated time: 20-40 minutes"
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
    echo "2/4: DACE Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Estimated time: 2-4 hours (GPU) / 8-16 hours (CPU)"
    echo "Starting: $(date)"
    echo
    
    cd "$PROJECT_ROOT/src"
    python -m training.scripts.train_unified_dace_loo \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/dace_loo" \
        --dbms "$DBMS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --hidden_dim "$HIDDEN_DIM" \
        --node_length 22 \
        --lr "$LEARNING_RATE" \
        --device "$DEVICE"
    
    echo
    echo "✅ DACE completed: $(date)"
    echo
}

run_queryformer() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "3/4: QueryFormer Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Estimated time: 3-6 hours (GPU) / 10-20 hours (CPU)"
    echo "Starting: $(date)"
    echo
    
    cd "$PROJECT_ROOT/src"
    python -m training.scripts.train_unified_queryformer_loo \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/queryformer_loo" \
        --dbms "$DBMS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --hidden_dim "$HIDDEN_DIM" \
        --lr "$LEARNING_RATE" \
        --device "$DEVICE" \
        --statistics_dir "$STATISTICS_DIR"
    
    echo
    echo "✅ QueryFormer completed: $(date)"
    echo
}

run_zeroshot() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "4/4: Zero-Shot Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Estimated time: 4-8 hours (GPU) / 12-24 hours (CPU)"
    echo "Starting: $(date)"
    echo
    
    cd "$PROJECT_ROOT/src"
    python -m training.scripts.train_unified_zeroshot_loo \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/zeroshot_loo" \
        --dbms "$DBMS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --hidden_dim "$HIDDEN_DIM" \
        --lr "$LEARNING_RATE" \
        --device "$DEVICE" \
        --statistics_dir "$STATISTICS_DIR"
    
    echo
    echo "✅ Zero-Shot completed: $(date)"
    echo
}

# メイン処理
if [ $# -eq 0 ]; then
    # 引数なし: すべてのモデルを実行
    run_flat
    run_dace
    run_queryformer
    run_zeroshot
elif [ "$1" = "flat" ]; then
    run_flat
elif [ "$1" = "dace" ]; then
    run_dace
elif [ "$1" = "queryformer" ]; then
    run_queryformer
elif [ "$1" = "zeroshot" ]; then
    run_zeroshot
else
    echo "Usage: $0 [flat|dace|queryformer|zeroshot]"
    echo "  No argument: Run all models"
    echo "  flat: Run Flat Vector only"
    echo "  dace: Run DACE only"
    echo "  queryformer: Run QueryFormer only"
    echo "  zeroshot: Run Zero-Shot only"
    exit 1
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
[ -f "$OUTPUT_DIR/zeroshot_loo/trino_leave_one_out_results.json" ] && echo "  ✓ $OUTPUT_DIR/zeroshot_loo/trino_leave_one_out_results.json"
echo

echo "View results:"
echo "  cd $OUTPUT_DIR"
echo "  cat flat_loo/trino_leave_one_out_results.json | jq '.average_metrics'"
echo "  cat dace_loo/trino_leave_one_out_results.json | jq '.average_metrics'"
echo "  cat queryformer_loo/trino_leave_one_out_results.json | jq '.average_metrics'"
echo "  cat zeroshot_loo/trino_leave_one_out_results.json | jq '.average_metrics'"

