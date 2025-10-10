# Trino用モデル実装

このドキュメントでは、Trinoクエリプラン向けに再実装されたモデルについて説明します。

## 📁 ディレクトリ構成

```
lcm-eval_for_trino/
├── src/
│   ├── trino_lcm/              # 🆕 Trino専用実装（すべてここに集約）
│   │   ├── models/             # モデル実装
│   │   │   ├── flat_vector/    # Flat-Vectorモデル
│   │   │   │   ├── __init__.py
│   │   │   │   └── trino_flat_vector.py
│   │   │   └── zero_shot/      # Zero-Shotモデル
│   │   │       ├── __init__.py
│   │   │       ├── trino_plan_batching.py
│   │   │       └── trino_zero_shot.py
│   │   └── scripts/            # スクリプト
│   │       ├── README.md
│   │       ├── train_flat_vector.py    # Flat-Vectorトレーニング
│   │       ├── predict_flat_vector.py  # Flat-Vector予測
│   │       ├── inspect_flat_vector.py  # Flat-Vectorモデル情報
│   │       ├── train_zeroshot.py       # Zero-Shotトレーニング
│   │       ├── collect_stats.py        # 統計情報収集
│   │       └── analyze_stats.py        # 統計情報分析
│   └── cross_db_benchmark/
│       └── benchmark_tools/
│           └── trino/          # Trinoプランパーサー（変更なし）
│               ├── parse_plan.py
│               ├── plan_operator.py
│               └── ...
└── README_TRINO.md             # このファイル
```

## 🎯 実装されているモデル

### 1. Flat-Vector Model（PostgreSQL版の再実装）

**場所**: `src/trino_lcm/`

クエリプランを平坦化して、演算子タイプごとに出現回数とカーディナリティを集計し、
LightGBMで実行時間を予測するシンプルなモデル。

**使用方法**:
```bash
# トレーニング
PYTHONPATH=src python -m trino_lcm.scripts.train_flat_vector \
    --train_files accidents_valid_verbose.txt \
    --test_file accidents_valid_verbose.txt \
    --output_dir models/trino_flat_vector \
    --use_act_card

# 予測
PYTHONPATH=src python -m trino_lcm.scripts.predict_flat_vector \
    --model_dir models/trino_flat_vector \
    --input_file new_queries.txt \
    --output_file predictions.json \
    --use_act_card

# モデル情報表示
PYTHONPATH=src python -m trino_lcm.scripts.inspect_flat_vector \
    --model_dir models/trino_flat_vector
```

**特徴**:
- ✅ シンプルで理解しやすい
- ✅ トレーニングが高速
- ✅ 既存のメトリクス（Q-Error, RMSE, MAPE）を使用
- ✅ PostgreSQL版と同じアーキテクチャ

**詳細**: `src/trino_lcm/scripts/README.md`を参照

### 2. Zero-Shot Model（Graph Neural Network）

**場所**: `src/trino_lcm/`

グラフニューラルネットワークを使用した高精度なモデル。
プランの詳細な構造とデータベース統計情報を活用。

**使用方法**:
```bash
# トレーニング
PYTHONPATH=src python -m trino_lcm.scripts.train_zeroshot \
    --train_files accidents_valid_verbose.txt \
    --test_file accidents_valid_verbose.txt \
    --output_dir models/trino_zeroshot \
    --statistics_dir datasets_statistics \
    --catalog iceberg \
    --schema imdb
```

**特徴**:
- ✅ 高精度な予測
- ✅ プランの詳細な構造を考慮
- ✅ テーブル・カラムの統計情報を活用
- ⚠️ トレーニング時間が長い
- ⚠️ 複雑な実装

## 🔄 PostgreSQL版との対応

| PostgreSQL | Trino | 説明 |
|------------|-------|------|
| `src/models/tabular/train_tabular_baseline.py` | `scripts/trino/train_flat_vector.py` | Flat-Vectorモデル |
| `parse_plan.py` | `trino/parse_plan.py` | プランパーサー |
| `est_card`, `act_card` | `est_rows`, `act_output_rows` | カーディナリティフィールド |
| `Seq Scan`, `Hash Join` | `TableScan`, `InnerJoin` | 演算子タイプ |

## 📊 評価メトリクス

すべてのモデルで、`src/training/training/metrics.py`で定義された共通のメトリクスを使用：

- **RMSE (Root Mean Squared Error)**: 予測誤差の二乗平均平方根
- **MAPE (Mean Absolute Percentage Error)**: 平均絶対パーセント誤差
- **Q-Error**: カーディナリティ推定誤差の指標
  - Median Q-Error (50th percentile)
  - P95 Q-Error (95th percentile)
  - P99 Q-Error (99th percentile)
  - Max Q-Error (100th percentile)

## 🛠️ データ収集

### 1. 統計情報の収集

```bash
PYTHONPATH=src python -m trino_lcm.scripts.collect_stats \
    --catalog iceberg \
    --schema imdb \
    --output-dir datasets_statistics
```

### 2. クエリプランの取得

Trinoで`EXPLAIN ANALYZE VERBOSE`を実行：

```sql
EXPLAIN ANALYZE VERBOSE
SELECT * FROM your_table WHERE condition;
```

出力をテキストファイルに保存。複数のプランは`-- stmt:`で区切る。

### 3. 統計情報の分析

```bash
PYTHONPATH=src python -m trino_lcm.scripts.analyze_stats \
    --stats_dir datasets_statistics/iceberg_imdb
```

## 🧪 完全なワークフロー例

```bash
# 1. 統計情報の収集
PYTHONPATH=src python -m trino_lcm.scripts.collect_stats \
    --catalog iceberg \
    --schema imdb \
    --output-dir datasets_statistics

# 2. Flat-Vectorモデルのトレーニング
PYTHONPATH=src python -m trino_lcm.scripts.train_flat_vector \
    --train_files train_plans.txt \
    --test_file test_plans.txt \
    --output_dir models/trino_flat_vector \
    --use_act_card

# 3. 予測
PYTHONPATH=src python -m trino_lcm.scripts.predict_flat_vector \
    --model_dir models/trino_flat_vector \
    --input_file new_queries.txt \
    --output_file predictions.json \
    --use_act_card

# 4. Zero-Shotモデルのトレーニング（より高精度）
PYTHONPATH=src python -m trino_lcm.scripts.train_zeroshot \
    --train_files train_plans.txt \
    --test_file test_plans.txt \
    --output_dir models/trino_zeroshot \
    --statistics_dir datasets_statistics \
    --catalog iceberg \
    --schema imdb
```

## 📚 ドキュメント

- **Flat-Vectorモデル詳細**: `src/trino_lcm/scripts/README.md`
- **メインREADME**: `README.md`（元のlcm-evalプロジェクト）

## 💡 実行方法

すべてのスクリプトは、ルートディレクトリから以下の形式で実行します：

```bash
PYTHONPATH=src python -m trino_lcm.scripts.<script_name> [options]
```

**利用可能なスクリプト**:
- `train_flat_vector` - Flat-Vectorモデルのトレーニング
- `predict_flat_vector` - Flat-Vectorモデルの予測
- `inspect_flat_vector` - Flat-Vectorモデルの情報表示
- `train_zeroshot` - Zero-Shotモデルのトレーニング
- `collect_stats` - Trino統計情報の収集
- `analyze_stats` - Trino統計情報の分析

## 🐛 トラブルシューティング

### 未知の演算子タイプ

```
⚠️  警告: 未知の演算子タイプ 'NewOperator' をスキップ
```

**解決策**: トレーニングデータに新しい演算子タイプを含むクエリプランを追加。

### カーディナリティが見つからない

`act_output_rows`や`est_rows`が含まれていない場合、カーディナリティは0として扱われます。

**解決策**: `EXPLAIN ANALYZE VERBOSE`を使用して詳細な実行統計を取得。

### メモリ不足

**解決策**: `--max_plans_per_file`オプションでプラン数を制限：

```bash
python scripts/trino/train_flat_vector.py \
    --train_files large_file.txt \
    --max_plans_per_file 1000 \
    ...
```

## 🎓 比較: Flat-Vector vs Zero-Shot

| 特徴 | Flat-Vector | Zero-Shot |
|------|-------------|-----------|
| モデル | LightGBM | Graph Neural Network |
| 特徴量 | 演算子 + カーディナリティ | 詳細なプラン構造 + 統計情報 |
| トレーニング時間 | 短い（数分） | 長い（数時間） |
| 精度 | 中程度 | 高い |
| 複雑さ | 低 | 高 |
| メモリ使用量 | 少ない | 多い |
| 解釈性 | 高い | 低い |

**推奨事項**:
- **Flat-Vector**: シンプルで高速なベースライン、初期プロトタイピング
- **Zero-Shot**: 高精度が必要な場合、詳細なプラン解析が必要な場合

## 📄 ライセンス

このプロジェクトは元のlcm-evalプロジェクトと同じライセンスの下で配布されます。

