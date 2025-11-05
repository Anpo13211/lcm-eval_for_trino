"""
Zero-shotモデル用の統計情報読み込みユーティリティ

zero-shot形式の統計情報をPostgres形式（zero-shotモデルの標準形式）に変換します。
これにより、Postgres版とTrino版の両方で統一された統計情報形式を使用できます。
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional


def load_zero_shot_column_statistics(dataset: str, prefer_zero_shot: bool = True) -> Dict:
    """
    zero-shot形式のcolumn_statistics.jsonを読み込む
    
    Args:
        dataset: データセット名
        prefer_zero_shot: Trueの場合、zero-shot_datasets配下を優先
    
    Returns:
        zero-shot形式の統計情報: {table: {column: {stats}}}
    """
    import os
    import json
    from pathlib import Path
    
    # zero-shot_datasets配下を優先
    if prefer_zero_shot:
        zero_shot_path = Path('/Users/an/query_engine/lakehouse/zero-shot_datasets') / dataset / 'column_statistics.json'
        if zero_shot_path.exists():
            with open(zero_shot_path) as f:
                return json.load(f)
    
    # フォールバック: cross_db_benchmark/datasets配下
    fallback_path = Path('cross_db_benchmark/datasets') / dataset / 'column_statistics.json'
    if fallback_path.exists():
        with open(fallback_path) as f:
            return json.load(f)
    
    return {}


def convert_zero_shot_to_postgres_format(zero_shot_stats: Dict, zero_shot_table_stats: Optional[Dict] = None) -> Dict:
    """
    zero-shot形式の統計情報をPostgres形式（zero-shotモデル標準形式）に変換
    
    Args:
        zero_shot_stats: zero-shot形式のカラム統計 {table: {column: {stats}}}
        zero_shot_table_stats: zero-shot形式のテーブル統計（オプション）
    
    Returns:
        Postgres形式の統計情報: {
            'column_stats': {(table, column): SimpleNamespace},
            'table_stats': {table: SimpleNamespace}
        }
    """
    column_stats = {}
    table_stats = {}
    
    # カラム統計の変換
    # テーブル名を小文字に統一
    for table_name, table_cols in zero_shot_stats.items():
        table_name_lower = table_name.lower()
        for col_name, col_stats in table_cols.items():
            key = (table_name_lower, col_name)
            
            # zero-shot形式からPostgres形式に変換
            # zero-shot形式のフィールドをPostgres形式のフィールドにマッピング
            column_stats[key] = SimpleNamespace(
                tablename=table_name_lower,  # 小文字に統一
                attname=col_name,
                # zero-shot形式の統計情報をPostgres形式にマッピング
                null_frac=col_stats.get('nan_ratio', 0.0),
                avg_width=0,  # zero-shot形式にはない（必要に応じて計算可能）
                n_distinct=col_stats.get('num_unique', -1),
                correlation=0,  # zero-shot形式にはない
                data_type=col_stats.get('datatype', 'misc'),
                # zero-shot形式の追加情報も保持（将来の拡張用）
                min_val=col_stats.get('min'),
                max_val=col_stats.get('max'),
                percentiles=col_stats.get('percentiles'),
                mean_val=col_stats.get('mean'),
                # 元のzero-shot形式のデータも保持（後方互換性）
                _zero_shot_stats=col_stats
            )
    
    # テーブル統計の変換
    # テーブル名を小文字に統一
    if zero_shot_table_stats:
        for table_name, table_stat in zero_shot_table_stats.items():
            table_name_lower = table_name.lower()
            table_stats[table_name_lower] = SimpleNamespace(
                relname=table_name_lower,  # 小文字に統一
                reltuples=table_stat.get('row_count', table_stat.get('reltuples', 0)),
                relpages=table_stat.get('relpages', 0)
            )
    
    return {
        'column_stats': column_stats,
        'table_stats': table_stats
    }


def load_database_statistics_for_zeroshot(
    dataset: str,
    prefer_zero_shot: bool = True,
    stats_dir: str = 'datasets_statistics'
) -> Dict:
    """
    zero-shotモデル用のデータベース統計情報を読み込む
    
    優先順位:
    1. zero-shot形式（zero-shot_datasets配下）→ Postgres形式に変換
    2. datasets_statistics配下（Trino形式）→ Postgres形式に変換
    
    Args:
        dataset: データセット名
        prefer_zero_shot: Trueの場合、zero-shot形式を優先
        stats_dir: datasets_statisticsのルートディレクトリ
    
    Returns:
        Postgres形式の統計情報: {
            'column_stats': {(table, column): SimpleNamespace},
            'table_stats': {table: SimpleNamespace}
        }
    """
    import json
    from pathlib import Path
    
    # 1. zero-shot形式から読み込む（優先）
    if prefer_zero_shot:
        zero_shot_col_stats = load_zero_shot_column_statistics(dataset, prefer_zero_shot=True)
        if zero_shot_col_stats:
            # テーブル統計も読み込む（datasets_statisticsから）
            table_stats_path = Path(stats_dir) / f'iceberg_{dataset}' / 'table_stats.json'
            zero_shot_table_stats = {}
            if table_stats_path.exists():
                with open(table_stats_path) as f:
                    zero_shot_table_stats = json.load(f)
            
            postgres_format = convert_zero_shot_to_postgres_format(zero_shot_col_stats, zero_shot_table_stats)
            print(f"✅ zero-shot形式から読み込み: {len(postgres_format['column_stats'])} カラム")
            return postgres_format
    
    # 2. datasets_statistics配下から読み込む（フォールバック）
    stats_dir_path = Path(stats_dir) / f'iceberg_{dataset}'
    column_stats_path = stats_dir_path / 'column_stats.json'
    table_stats_path = stats_dir_path / 'table_stats.json'
    
    if column_stats_path.exists() and table_stats_path.exists():
        with open(column_stats_path) as f:
            trino_col_stats = json.load(f)
        with open(table_stats_path) as f:
            trino_table_stats = json.load(f)
        
        # Trino形式（{table.column: {stats}}）をPostgres形式に変換
        # テーブル名を小文字に統一
        column_stats = {}
        for col_key, col_stat in trino_col_stats.items():
            # table.column形式をパース
            if '.' in col_key:
                table_name, col_name = col_key.split('.', 1)
            else:
                # フォールバック: column_statsにtableとcolumnフィールドがある場合
                table_name = col_stat.get('table', 'unknown')
                col_name = col_stat.get('column', col_key)
            
            # テーブル名を小文字に統一
            table_name_lower = table_name.lower()
            key = (table_name_lower, col_name)
            column_stats[key] = SimpleNamespace(
                tablename=table_name_lower,  # 小文字に統一
                attname=col_name,
                null_frac=col_stat.get('null_frac', 0.0),
                avg_width=col_stat.get('avg_width', 0),
                n_distinct=col_stat.get('n_distinct', -1),
                correlation=col_stat.get('correlation', 0),
                data_type=col_stat.get('data_type', col_stat.get('datatype', 'misc'))
            )
        
        table_stats = {}
        for table_name, table_stat in trino_table_stats.items():
            # テーブル名を小文字に統一
            table_name_lower = table_name.lower()
            table_stats[table_name_lower] = SimpleNamespace(
                relname=table_name_lower,  # 小文字に統一
                reltuples=table_stat.get('row_count', table_stat.get('reltuples', 0)),
                relpages=table_stat.get('relpages', 0)
            )
        
        print(f"✅ datasets_statisticsから読み込み: {len(column_stats)} カラム")
        return {
            'column_stats': column_stats,
            'table_stats': table_stats
        }
    
    # 統計情報が見つからない場合
    print(f"⚠️  統計情報が見つかりません（dataset={dataset}）")
    return {
        'column_stats': {},
        'table_stats': {}
    }

