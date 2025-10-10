"""
Trino統計情報分析スクリプト

collect_stats.pyで収集した統計情報を分析して、特徴量を抽出します。

Usage:
    # ルートディレクトリから実行
    python -m trino_lcm.scripts.analyze_stats \
        --stats_dir datasets_statistics/iceberg_imdb
"""

import json
import sys
from typing import Dict, Any


def analyze_column_stats(col_stats: Dict[str, Any]) -> Dict[str, Any]:
    """カラム統計から有用な特徴量を抽出"""
    features = {
        'column_name': col_stats['column_name'],
        
        # 数値特徴
        'data_size': col_stats.get('data_size'),  # バイト単位のデータサイズ
        'distinct_values_count': col_stats.get('distinct_values_count'),  # ユニークな値の数
        'nulls_fraction': col_stats.get('nulls_fraction'),  # NULL値の割合（0~1）
        
        # 範囲特徴（数値カラムの場合のみ）
        'low_value': col_stats.get('low_value'),  # 最小値
        'high_value': col_stats.get('high_value'),  # 最大値
        
        # 派生特徴
        'avg_value_size': None,  # 1値あたりの平均サイズ
        'cardinality_ratio': None,  # distinct / row_count（選択性の指標）
    }
    
    return features


def compute_derived_features(
    table_stats: Dict[str, Any],
    col_features: Dict[str, Any]
) -> Dict[str, Any]:
    """派生特徴量を計算"""
    row_count = table_stats.get('row_count')
    data_size = col_features.get('data_size')
    distinct_count = col_features.get('distinct_values_count')
    
    # 1値あたりの平均サイズ
    if data_size is not None and row_count and row_count > 0:
        col_features['avg_value_size'] = data_size / row_count
    
    # カーディナリティ比率（選択性）
    if distinct_count is not None and row_count and row_count > 0:
        col_features['cardinality_ratio'] = distinct_count / row_count
    
    return col_features


def compare_with_postgres_features():
    """Postgres zero-shotで使われている特徴量との比較"""
    print("\n📊 Postgres zero-shotで使われているカラム特徴量:")
    print("=" * 70)
    
    postgres_features = [
        ('avg_width', '平均カラム幅（バイト）'),
        ('correlation', '物理的順序との相関'),
        ('n_distinct', 'ユニークな値の数（推定）'),
        ('null_frac', 'NULL値の割合'),
        ('min_val', '最小値'),
        ('max_val', '最大値'),
        ('num_rows', 'テーブルの行数'),
    ]
    
    for feat, desc in postgres_features:
        print(f"  - {feat:20s}: {desc}")
    
    print("\n🔄 Trinoの統計情報とのマッピング:")
    print("=" * 70)
    
    mappings = [
        ('avg_width', 'avg_value_size', '✅ data_size / row_count で計算可能'),
        ('correlation', 'N/A', '❌ Trinoでは取得不可'),
        ('n_distinct', 'distinct_values_count', '✅ 直接取得可能'),
        ('null_frac', 'nulls_fraction', '✅ 直接取得可能'),
        ('min_val', 'low_value', '✅ 直接取得可能（数値のみ）'),
        ('max_val', 'high_value', '✅ 直接取得可能（数値のみ）'),
        ('num_rows', 'row_count', '✅ テーブルレベルで取得可能'),
    ]
    
    for pg_feat, trino_feat, status in mappings:
        print(f"  {pg_feat:20s} → {trino_feat:30s} {status}")
    
    print("\n➕ Trino固有の追加特徴量:")
    print("=" * 70)
    
    trino_extras = [
        ('data_size', 'カラム全体のデータサイズ（バイト）'),
        ('cardinality_ratio', 'distinct_count / row_count（選択性指標）'),
    ]
    
    for feat, desc in trino_extras:
        print(f"  + {feat:20s}: {desc}")


def main():
    if len(sys.argv) < 2:
        stats_file = 'trino_statistics.json'
    else:
        stats_file = sys.argv[1]
    
    print(f"📖 統計情報ファイル: {stats_file}")
    
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f"❌ ファイルが見つかりません: {stats_file}")
        sys.exit(1)
    
    catalog = stats.get('catalog')
    schema = stats.get('schema')
    tables = stats.get('tables', {})
    
    print(f"📊 Catalog: {catalog}, Schema: {schema}")
    print(f"📊 テーブル数: {len(tables)}\n")
    
    # 各テーブルの統計情報を分析
    for table_name, table_stats in tables.items():
        print(f"\n{'='*70}")
        print(f"📋 テーブル: {table_name}")
        print(f"{'='*70}")
        
        row_count = table_stats.get('row_count')
        columns = table_stats.get('columns', {})
        
        print(f"  行数: {int(row_count):,}" if row_count else "  行数: N/A")
        print(f"  カラム数: {len(columns)}")
        print()
        
        print(f"{'カラム名':<25} {'Distinct':<12} {'Nulls':<10} "
              f"{'Avg Size':<12} {'Cardinality':<15}")
        print("-" * 80)
        
        for col_name, col_stats in columns.items():
            # 特徴量を抽出・計算
            features = analyze_column_stats(col_stats)
            features = compute_derived_features(table_stats, features)
            
            distinct = features.get('distinct_values_count')
            nulls = features.get('nulls_fraction')
            avg_size = features.get('avg_value_size')
            cardinality = features.get('cardinality_ratio')
            
            distinct_str = f"{int(distinct):,}" if distinct else "N/A"
            nulls_str = f"{nulls:.2%}" if nulls is not None else "N/A"
            avg_size_str = f"{avg_size:.1f}B" if avg_size else "N/A"
            card_str = f"{cardinality:.4f}" if cardinality else "N/A"
            
            print(f"{col_name:<25} {distinct_str:<12} {nulls_str:<10} "
                  f"{avg_size_str:<12} {card_str:<15}")
        
        print()
        
        # 選択性の高いカラム（インデックス候補）を表示
        high_card_cols = []
        for col_name, col_stats in columns.items():
            features = analyze_column_stats(col_stats)
            features = compute_derived_features(table_stats, features)
            cardinality = features.get('cardinality_ratio')
            
            if cardinality and cardinality > 0.9:  # 90%以上がユニーク
                high_card_cols.append((col_name, cardinality))
        
        if high_card_cols:
            print("  🔑 高選択性カラム（インデックス候補）:")
            for col_name, card in sorted(high_card_cols, key=lambda x: -x[1]):
                print(f"    - {col_name}: {card:.4f}")
        
        # NULL値の多いカラムを表示
        high_null_cols = []
        for col_name, col_stats in columns.items():
            nulls = col_stats.get('nulls_fraction')
            if nulls and nulls > 0.5:  # 50%以上がNULL
                high_null_cols.append((col_name, nulls))
        
        if high_null_cols:
            print("  ⚠️  NULL値の多いカラム:")
            for col_name, nulls in sorted(high_null_cols, key=lambda x: -x[1]):
                print(f"    - {col_name}: {nulls:.2%}")
    
    # Postgres特徴量との比較
    compare_with_postgres_features()
    
    print("\n" + "=" * 70)
    print("✅ 分析完了")
    print("=" * 70)
    print("\n💡 次のステップ:")
    print("  1. trino_plan_batching.py の column 特徴量にこれらの統計情報を追加")
    print("  2. create_dummy_feature_statistics() を実統計で置き換え")
    print("  3. RobustScaler で正規化してモデルに入力")


if __name__ == '__main__':
    main()

