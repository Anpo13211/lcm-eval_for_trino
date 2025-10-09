# Trino用モデル実装

このディレクトリには、Trinoクエリプラン向けに再実装されたモデルが含まれています。

## 📁 ディレクトリ構成

```
src/trino_models/
├── models/                      # モデル実装
│   ├── flat_vector/
│   │   ├── __init__.py
│   │   └── trino_flat_vector.py
│   └── zero_shot/
│       ├── __init__.py
│       ├── trino_plan_batching.py
│       └── trino_zero_shot.py
└── scripts/                     # このディレクトリ
    ├── README.md                # このファイル
    ├── train_flat_vector.py    # Flat-Vectorトレーニング
    ├── predict_flat_vector.py  # Flat-Vector予測
    ├── inspect_flat_vector.py  # Flat-Vectorモデル情報
    ├── train_zeroshot.py       # Zero-Shotトレーニング
    ├── collect_stats.py        # 統計情報収集
    └── analyze_stats.py        # 統計情報分析
```

## 🎯 Flat-Vector Model（Trino向け再実装）

PostgreSQL用に実装されたFlat-VectorモデルをTrino向けに再実装したものです。

### 概要

Flat-Vectorモデルは、クエリプランツリーを平坦化して、各演算子タイプごとに以下の情報を集計します：

1. **演算子の出現回数**: 各演算子タイプがプラン内に何回出現するか
2. **演算子ごとのカーディナリティ合計**: 各演算子タイプの行数の合計

これらの特徴ベクトルを使用して、LightGBM（勾配ブースティング決定木）でクエリの実行時間を予測します。

### 使用方法

#### 1. トレーニング

```bash
PYTHONPATH=src python -m trino_models.scripts.train_flat_vector \
    --train_files accidents_valid_verbose.txt \
    --test_file accidents_valid_verbose.txt \
    --output_dir models/trino_flat_vector \
    --num_boost_round 1000 \
    --early_stopping_rounds 20 \
    --val_ratio 0.15 \
    --use_act_card \
    --seed 42
```

**主要なオプション:**

- `--train_files`: トレーニング用ファイル（カンマ区切りで複数指定可）
- `--test_file`: テスト用ファイル
- `--output_dir`: モデルの出力ディレクトリ
- `--num_boost_round`: ブースティングラウンド数（デフォルト: 1000）
- `--early_stopping_rounds`: 早期停止ラウンド数（デフォルト: 20）
- `--val_ratio`: 検証セットの割合（デフォルト: 0.15）
- `--use_act_card`: 実際のカーディナリティを使用（指定しない場合は推定カーディナリティを使用）
- `--seed`: ランダムシード（デフォルト: 42）

#### 2. 予測

```bash
PYTHONPATH=src python -m trino_models.scripts.predict_flat_vector \
    --model_dir models/trino_flat_vector \
    --input_file new_queries.txt \
    --output_file predictions.json \
    --use_act_card \
    --seed 42
```

**主要なオプション:**

- `--model_dir`: モデルディレクトリ
- `--input_file`: 入力ファイル（クエリプラン）
- `--output_file`: 出力ファイル（予測結果JSON）
- `--use_act_card`: 実際のカーディナリティを使用（トレーニング時と同じ設定を使用）
- `--seed`: ランダムシード（トレーニング時と同じ値を使用）

#### 3. モデル情報の表示

```bash
PYTHONPATH=src python -m trino_models.scripts.inspect_flat_vector \
    --model_dir models/trino_flat_vector \
    --seed 42
```

### 評価メトリクス

既存の`src/training/training/metrics.py`のメトリクスを使用しています：

- **RMSE (Root Mean Squared Error)**: 予測誤差の二乗平均平方根
- **MAPE (Mean Absolute Percentage Error)**: 平均絶対パーセント誤差
- **Q-Error**: カーディナリティ推定誤差の指標
  - Median Q-Error (50th percentile)
  - P95 Q-Error (95th percentile)
  - P99 Q-Error (99th percentile)
  - Max Q-Error (100th percentile)

### 出力ファイル

トレーニング後、以下のファイルが生成されます：

- `flat_vector_model_{seed}.txt`: LightGBMモデル
- `op_idx_dict_{seed}.json`: 演算子タイプのインデックス辞書
- `metrics_{seed}.json`: トレーニング/検証/テストメトリクス

## 🔄 PostgreSQL版との違い

このTrino版の実装は、以下の点で元のPostgreSQL版と異なります：

| 項目 | PostgreSQL版 | Trino版 |
|------|--------------|---------|
| プランパーサー | `parse_plan.py` | `trino/parse_plan.py` |
| カーディナリティフィールド | `est_card`, `act_card` | `est_rows`, `act_output_rows` |
| 演算子タイプ | `Seq Scan`, `Hash Join`, etc. | `TableScan`, `InnerJoin`, etc. |
| 実行時間の単位 | ミリ秒 | ミリ秒 |

## 📊 モデルの特徴

**利点:**
- ✅ シンプルで理解しやすい
- ✅ トレーニングが高速
- ✅ 少ないメモリ使用量
- ✅ 実装が簡単
- ✅ 既存のメトリクスを再利用

**制限:**
- ⚠️ プランの構造情報を考慮しない
- ⚠️ 演算子の順序を考慮しない
- ⚠️ テーブル・カラムの統計情報を直接使用しない

## 🧪 例

完全なワークフロー例：

```bash
# 1. Trinoから統計情報を収集（オプション）
PYTHONPATH=src python -m trino_models.scripts.collect_stats \
    --catalog iceberg \
    --schema imdb \
    --output-dir datasets_statistics

# 2. モデルのトレーニング
PYTHONPATH=src python -m trino_models.scripts.train_flat_vector \
    --train_files accidents_valid_verbose.txt \
    --test_file accidents_valid_verbose.txt \
    --output_dir models/trino_flat_vector \
    --use_act_card \
    --seed 42

# 3. 新しいクエリの予測
PYTHONPATH=src python -m trino_models.scripts.predict_flat_vector \
    --model_dir models/trino_flat_vector \
    --input_file new_queries.txt \
    --output_file predictions.json \
    --use_act_card \
    --seed 42

# 4. モデル情報の表示
PYTHONPATH=src python -m trino_models.scripts.inspect_flat_vector \
    --model_dir models/trino_flat_vector \
    --seed 42
```

## 📚 参考資料

- 元のFlat-Vector実装: `src/models/tabular/train_tabular_baseline.py`
- メトリクス実装: `src/training/training/metrics.py`
- Trinoプランパーサー: `src/cross_db_benchmark/benchmark_tools/trino/parse_plan.py`

