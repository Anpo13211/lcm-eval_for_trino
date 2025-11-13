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
OUTPUT_DIR="$PROJECT_ROOT/models"
LOGS_DIR="$PROJECT_ROOT/logs"

# ログディレクトリ作成
mkdir -p "$LOGS_DIR"

# デバイス設定（GPUがある場合は cuda:0、なければ cpu）
DEVICE="cpu"

# 共通パラメータ
EPOCHS=100
BATCH_SIZE=32
HIDDEN_DIM=256
LEARNING_RATE=0.001
DBMS="trino"

# モデル固有のハイパーパラメータ（元論文の実装に合わせる）
DACE_HIDDEN_DIM=128
QUERYFORMER_BATCH_SIZE=16
QPPNET_EPOCHS=50
QPPNET_HIDDEN_DIM=128

echo "================================================================================"
echo "Leave-One-Out Cross-Validation Training - All Models"
echo "================================================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Data Directory: $DATA_DIR"
echo "Statistics Directory: $STATISTICS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
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
    python -m training.scripts.train_unified_dace_loo \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/dace_loo" \
        --dbms "$DBMS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --hidden_dim "$DACE_HIDDEN_DIM" \
        --node_length 22 \
        --lr "$LEARNING_RATE" \
        --device "$DEVICE"
    
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
    python -m training.scripts.train_unified_queryformer_loo \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/queryformer_loo" \
        --dbms "$DBMS" \
        --epochs "$EPOCHS" \
        --batch_size "$QUERYFORMER_BATCH_SIZE" \
        --hidden_dim "$HIDDEN_DIM" \
        --lr "$LEARNING_RATE" \
        --device "$DEVICE" \
        --statistics_dir "$STATISTICS_DIR"
    
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
    python -m training.scripts.train_qppnet_loo \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/qppnet_loo" \
        --epochs "$QPPNET_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --hidden_dim "$QPPNET_HIDDEN_DIM" \
        --lr "$LEARNING_RATE" \
        --device "$DEVICE" \
        --statistics_dir "$STATISTICS_DIR"
    
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
    run_qppnet
    run_zeroshot
elif [ "$1" = "flat" ]; then
    run_flat
elif [ "$1" = "dace" ]; then
    run_dace
elif [ "$1" = "queryformer" ]; then
    run_queryformer
elif [ "$1" = "qppnet" ]; then
    run_qppnet
elif [ "$1" = "zeroshot" ]; then
    run_zeroshot
else
    echo "Usage: $0 [flat|dace|queryformer|qppnet|zeroshot]"
    echo "  No argument: Run all models"
    echo "  flat: Run Flat Vector only"
    echo "  dace: Run DACE only"
    echo "  queryformer: Run QueryFormer only"
    echo "  qppnet: Run QPPNet only"
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

