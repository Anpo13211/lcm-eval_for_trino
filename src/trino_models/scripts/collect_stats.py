"""
Trinoからテーブルとカラムの統計情報を収集するスクリプト

Usage:
    # ルートディレクトリから実行
    python -m trino_models.scripts.collect_stats \
        --catalog iceberg \
        --schema imdb \
        --tables name,cast_info
    
統計情報は以下の構造で保存されます:
    datasets_statistics/
        <catalog>_<schema>/
            column_stats.json  # カラムレベルの統計情報
            table_stats.json   # テーブルレベルの統計情報
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def run_trino_query(catalog: str, schema: str, query: str) -> List[List[str]]:
    """Trinoクエリを実行して結果を返す"""
    cmd = [
        'docker', 'exec', 'lakehouse-trino-1', 
        'trino',
        '--catalog', catalog,
        '--schema', schema,
        '--execute', query,
        '--output-format', 'CSV_UNQUOTED'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 結果をパース（CSVヘッダーを除く）
        lines = result.stdout.strip().split('\n')
        # WARNINGメッセージをスキップ
        data_lines = [line for line in lines if not line.startswith('WARNING') and line.strip()]
        
        # CSVとしてパース
        rows = []
        for line in data_lines:
            # 簡易的なCSVパース（引用符なしの場合）
            row = [field.strip() for field in line.split(',')]
            rows.append(row)
        
        return rows
    except subprocess.CalledProcessError as e:
        print(f"❌ クエリ実行エラー: {e.stderr}", file=sys.stderr)
        return []


def get_table_list(catalog: str, schema: str) -> List[str]:
    """スキーマ内のテーブル一覧を取得"""
    query = "SHOW TABLES"
    rows = run_trino_query(catalog, schema, query)
    
    # ヘッダー行をスキップ
    if rows and rows[0][0].lower() == 'table':
        rows = rows[1:]
    
    return [row[0] for row in rows if row]


def get_table_stats(catalog: str, schema: str, table: str) -> Dict[str, Any]:
    """テーブルの統計情報を取得"""
    query = f"SHOW STATS FOR {table}"
    rows = run_trino_query(catalog, schema, query)
    
    if not rows:
        return {}
    
    # ヘッダー行（最初の行）
    header = rows[0]
    data_rows = rows[1:]
    
    # 統計情報を構造化
    stats = {
        'table_name': table,
        'row_count': None,
        'columns': {}
    }
    
    for row in data_rows:
        if len(row) < len(header):
            continue
        
        # カラム名がNULLの行は全体統計
        column_name = row[0]
        if not column_name or column_name.upper() == 'NULL':
            # row_countを取得
            if len(row) >= 5 and row[4] and row[4] != 'NULL':
                try:
                    stats['row_count'] = float(row[4])
                except ValueError:
                    pass
            continue
        
        # カラム統計を構造化
        col_stats = {
            'column_name': column_name,
            'data_size': parse_float(row[1]) if len(row) > 1 else None,
            'distinct_values_count': parse_float(row[2]) if len(row) > 2 else None,
            'nulls_fraction': parse_float(row[3]) if len(row) > 3 else None,
            'low_value': row[5] if len(row) > 5 and row[5] and row[5] != 'NULL' else None,
            'high_value': row[6] if len(row) > 6 and row[6] and row[6] != 'NULL' else None,
        }
        
        stats['columns'][column_name] = col_stats
    
    return stats


def parse_float(value: str) -> Optional[float]:
    """文字列をfloatに変換（失敗時はNone）"""
    if not value or value.upper() == 'NULL':
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Trinoからテーブルとカラムの統計情報を収集'
    )
    parser.add_argument(
        '--catalog',
        default='iceberg',
        help='Trinoカタログ名（デフォルト: iceberg）'
    )
    parser.add_argument(
        '--schema',
        default='imdb',
        help='スキーマ名（デフォルト: imdb）'
    )
    parser.add_argument(
        '--tables',
        help='テーブル名のカンマ区切りリスト（省略時は全テーブル）'
    )
    parser.add_argument(
        '--output-dir',
        default='datasets_statistics',
        help='出力ディレクトリ（デフォルト: datasets_statistics）'
    )
    
    args = parser.parse_args()
    
    print(f"📊 Trino統計情報収集開始")
    print(f"  Catalog: {args.catalog}")
    print(f"  Schema: {args.schema}")
    print()
    
    # テーブルリストを取得
    if args.tables:
        tables = [t.strip() for t in args.tables.split(',')]
        print(f"📋 指定されたテーブル: {', '.join(tables)}")
    else:
        print("📋 全テーブルを取得中...")
        tables = get_table_list(args.catalog, args.schema)
        print(f"   見つかったテーブル: {', '.join(tables)}")
    
    print()
    
    # 出力ディレクトリを作成
    schema_dir_name = f"{args.catalog}_{args.schema}"
    output_dir = Path(args.output_dir) / schema_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 出力ディレクトリ: {output_dir}")
    print()
    
    # 各テーブルの統計情報を収集
    table_stats = {}  # テーブルレベルの統計
    column_stats = {}  # カラムレベルの統計
    
    for table in tables:
        print(f"📊 {table} の統計情報を収集中...")
        stats = get_table_stats(args.catalog, args.schema, table)
        
        if stats:
            # テーブル統計を保存
            table_stats[table] = {
                'table_name': table,
                'row_count': stats.get('row_count'),
                'num_columns': len(stats.get('columns', {}))
            }
            
            # カラム統計を保存（table.column形式のキー）
            for col_name, col_stat in stats.get('columns', {}).items():
                full_col_name = f"{table}.{col_name}"
                column_stats[full_col_name] = {
                    'table': table,
                    'column': col_name,
                    'data_size': col_stat.get('data_size'),
                    'distinct_values_count': col_stat.get('distinct_values_count'),
                    'nulls_fraction': col_stat.get('nulls_fraction'),
                    'low_value': col_stat.get('low_value'),
                    'high_value': col_stat.get('high_value'),
                    # 派生特徴
                    'avg_value_size': (
                        col_stat.get('data_size') / stats.get('row_count')
                        if col_stat.get('data_size') is not None 
                           and stats.get('row_count') 
                           and stats.get('row_count') > 0
                        else None
                    ),
                    'cardinality_ratio': (
                        col_stat.get('distinct_values_count') / stats.get('row_count')
                        if col_stat.get('distinct_values_count') is not None
                           and stats.get('row_count')
                           and stats.get('row_count') > 0
                        else None
                    )
                }
            
            # サマリーを表示
            row_count = stats.get('row_count')
            num_columns = len(stats.get('columns', {}))
            print(f"   ✅ 行数: {int(row_count) if row_count else 'N/A'}, "
                  f"カラム数: {num_columns}")
            
            # カラム統計のサマリー
            for col_name, col_stat in stats.get('columns', {}).items():
                distinct = col_stat.get('distinct_values_count')
                nulls = col_stat.get('nulls_fraction')
                
                info_parts = []
                if distinct is not None:
                    info_parts.append(f"distinct={int(distinct)}")
                if nulls is not None:
                    info_parts.append(f"nulls={nulls:.2%}")
                
                info_str = ", ".join(info_parts) if info_parts else "no stats"
                print(f"     - {col_name}: {info_str}")
        else:
            print(f"   ❌ 統計情報の取得に失敗")
        
        print()
    
    # メタデータを追加
    metadata = {
        'catalog': args.catalog,
        'schema': args.schema,
        'num_tables': len(table_stats),
        'num_columns': len(column_stats)
    }
    
    # JSON出力
    table_stats_file = output_dir / 'table_stats.json'
    column_stats_file = output_dir / 'column_stats.json'
    metadata_file = output_dir / 'metadata.json'
    
    with open(table_stats_file, 'w', encoding='utf-8') as f:
        json.dump(table_stats, f, indent=2, ensure_ascii=False)
    
    with open(column_stats_file, 'w', encoding='utf-8') as f:
        json.dump(column_stats, f, indent=2, ensure_ascii=False)
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 統計情報を保存しました:")
    print(f"   📄 {table_stats_file}")
    print(f"   📄 {column_stats_file}")
    print(f"   📄 {metadata_file}")
    print()
    print(f"📈 収集されたテーブル数: {len(table_stats)}")
    print(f"📈 収集されたカラム数: {len(column_stats)}")
    
    # サマリー統計
    total_rows = sum(
        stats.get('row_count', 0) or 0 
        for stats in table_stats.values()
    )
    
    print(f"📊 合計行数: {int(total_rows):,}")
    print()
    print(f"💡 使用方法:")
    print(f"   統計情報ディレクトリ: {output_dir}")
    print(f"   エンコーディング時にこのディレクトリを指定してください")


if __name__ == '__main__':
    main()

