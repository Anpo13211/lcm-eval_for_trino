"""
Trinoからテーブルとカラムの統計情報を収集するスクリプト

Usage:
    # 単一スキーマの統計情報を収集
    python src/trino_lcm/scripts/collect_stats.py \
        --catalog iceberg \
        --schema imdb \
        --tables name,cast_info
    
    # または全テーブル
    python src/trino_lcm/scripts/collect_stats.py \
        --catalog iceberg \
        --schema accidents
    
    # 複数スキーマを指定
    python src/trino_lcm/scripts/collect_stats.py \
        --catalog iceberg \
        --schemas accidents,airline,imdb
    
    # 全データセットの統計情報を一括収集
    python src/trino_lcm/scripts/collect_stats.py \
        --catalog iceberg \
        --all-schemas
    
統計情報は以下の構造で保存されます:
    datasets_statistics/
        <catalog>_<schema>/
            column_stats.json  # カラムレベルの統計情報（PostgreSQL互換）
            table_stats.json   # テーブルレベルの統計情報（PostgreSQL互換）
            metadata.json      # メタデータ
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# 利用可能なデータセット名（全スキーマ処理時に使用）
AVAILABLE_DATASETS = [
    'accidents',
    'airline',
    'baseball',
    'basketball',
    'carcinogenesis',
    'consumer',
    'credit',
    'employee',
    'fhnk',
    'financial',
    'geneea',
    'genome',
    'hepatitis',
    'imdb',
    'movielens',
    'seznam',
    'ssb',
    'tournament',
    'tpc_h',
    'walmart'
]


def run_trino_query(catalog: str, schema: str, query: str) -> List[List[str]]:
    """Trinoクエリを実行して結果を返す"""
    try:
        # Trinoクライアントライブラリを使用
        from trino.dbapi import connect
        
        # 接続設定（環境変数から取得）
        host = os.getenv('TRINO_HOST', 'localhost')
        port = int(os.getenv('TRINO_PORT', '8080'))
        user = os.getenv('TRINO_USER', 'admin')
        
        conn = connect(
            host=host,
            port=port,
            user=user,
            catalog=catalog,
            schema=schema
        )
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        # 結果を取得
        rows = cursor.fetchall()
        
        # 元の型を保持（文字列変換はしない）
        # None値はそのまま保持し、後の処理で適切に処理する
        result = []
        for row in rows:
            result.append(list(row))  # 元の型を保持
        
        cursor.close()
        conn.close()
        
        return result
        
    except ImportError:
        print("❌ trinoライブラリがインストールされていません", file=sys.stderr)
        print("   pip install trino を実行してください", file=sys.stderr)
        return []
    except Exception as e:
        print(f"❌ Trino接続エラー: {e}", file=sys.stderr)
        return []


def quote_identifier(identifier: str) -> str:
    """
    Trinoの識別子（テーブル名、カラム名）を適切にクォートする
    
    Trinoでは以下の場合にダブルクォートが必要：
    - 数字で始まる
    - 特別文字を含む
    - 予約語
    - 大文字と小文字を区別する必要がある
    
    Returns:
        クォートされた識別子
    """
    # 既にクォートされている場合はそのまま返す
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier
    
    # Trinoの主要な予約語リスト
    # 一般的なSQL予約語とTrino固有の予約語を含む
    trino_reserved_words = {
        'order', 'group', 'select', 'from', 'where', 'having', 'limit', 'offset',
        'join', 'inner', 'outer', 'left', 'right', 'full', 'cross', 'on', 'using',
        'union', 'intersect', 'except', 'all', 'distinct',
        'as', 'and', 'or', 'not', 'in', 'exists', 'between', 'is', 'null',
        'case', 'when', 'then', 'else', 'end',
        'cast', 'create', 'drop', 'alter', 'table', 'view', 'index',
        'insert', 'update', 'delete', 'values',
        'grant', 'revoke', 'commit', 'rollback', 'transaction',
        'primary', 'key', 'foreign', 'references', 'constraint', 'default',
        'set', 'show', 'describe', 'explain', 'use', 'database', 'schema',
        'like', 'ilike', 'similar', 'over', 'partition', 'window',
        'lateral', 'array', 'map', 'row', 'timestamp', 'date', 'time', 'interval',
        'asc', 'desc', 'any', 'some', 'by', 'with', 'recursive'
    }
    
    # 小文字に変換して予約語チェック
    identifier_lower = identifier.lower()
    
    # 数字で始まる、または特別文字を含む場合はクォート
    if re.match(r'^\d', identifier) or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        return f'"{identifier}"'
    
    # 予約語の場合はクォート
    if identifier_lower in trino_reserved_words:
        return f'"{identifier}"'
    
    return identifier


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
    quoted_table = quote_identifier(table)
    query = f"SHOW STATS FOR {quoted_table}"
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
        
        # カラム名を取得（文字列に変換）
        column_name = str(row[0]) if row[0] is not None else None
        
        # カラム名がNULLの行は全体統計
        if not column_name or column_name.upper() == 'NULL':
            # row_countを取得
            if len(row) >= 5 and row[4] is not None:
                try:
                    # 数値型の場合はそのまま、文字列の場合は変換
                    if isinstance(row[4], (int, float)):
                        stats['row_count'] = float(row[4])
                    elif isinstance(row[4], str) and row[4].upper() != 'NULL':
                        stats['row_count'] = float(row[4])
                except (ValueError, TypeError):
                    pass
            continue
        
        # カラム統計を構造化
        col_stats = {
            'column_name': column_name,
            'data_size': parse_float(row[1]) if len(row) > 1 else None,
            'distinct_values_count': parse_float(row[2]) if len(row) > 2 else None,
            'nulls_fraction': parse_float(row[3]) if len(row) > 3 else None,
            'low_value': str(row[5]) if len(row) > 5 and row[5] is not None else None,
            'high_value': str(row[6]) if len(row) > 6 and row[6] is not None else None,
        }
        
        stats['columns'][column_name] = col_stats
    
    return stats


def parse_float(value: Any) -> Optional[float]:
    """値をfloatに変換（失敗時はNone）"""
    if value is None:
        return None
    
    # 既に数値型の場合
    if isinstance(value, (int, float)):
        return float(value)
    
    # 文字列の場合
    if isinstance(value, str):
        if not value or value.upper() == 'NULL':
            return None
        try:
            return float(value)
        except ValueError:
            return None
    
    # その他の型
    return None


def estimate_pages(row_count: float, avg_row_size: int = 100) -> int:
    """
    行数からページ数を推定
    PostgreSQLのrelpages互換値を生成
    """
    if row_count == 0:
        return 0
    
    # PostgreSQLのデフォルトページサイズは8KB
    page_size = 8192
    # 推定総サイズ（バイト）
    estimated_size = row_count * avg_row_size
    # ページ数を計算
    pages = max(1, int(estimated_size / page_size))
    
    return pages


def get_column_percentiles(catalog: str, schema: str, table: str, column: str) -> Optional[List[float]]:
    """
    カラムのパーセンタイル値（11個: 0.0, 0.1, ..., 1.0）を取得
    
    注意: Trinoのapprox_percentile()関数が使用できない環境もあるため、
    エラーハンドリングを強化しています。
    
    Returns:
        パーセンタイル値のリスト、失敗時はNone
    """
    try:
        # テーブル名とカラム名をクォート
        quoted_table = quote_identifier(table)
        quoted_column = quote_identifier(column)
        
        # まずapprox_percentile()が使用可能かテスト
        test_query = f"SELECT approx_percentile({quoted_column}, 0.5) FROM {quoted_table} LIMIT 1"
        test_rows = run_trino_query(catalog, schema, test_query)
        
        if not test_rows or not test_rows[0] or test_rows[0][0] is None:
            print(f"      ⚠️ approx_percentile()関数が使用できません（値がNULLまたは結果なし）")
            return None
        
        # 11個のパーセンタイル値を一度に取得
        percentiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        percentile_queries = [
            f"approx_percentile({quoted_column}, {p})" for p in percentiles
        ]
        query = f"SELECT {', '.join(percentile_queries)} FROM {quoted_table} LIMIT 1"
        
        rows = run_trino_query(catalog, schema, query)
        
        if rows and len(rows) > 0 and len(rows[0]) == 11:
            # 結果をfloatのリストに変換
            result = []
            for val in rows[0]:
                try:
                    # None値または'NULL'文字列をチェック
                    if val is None:
                        print(f"      ⚠️ パーセンタイル値の一部がNone")
                        return None
                    
                    # 文字列として'NULL'の場合
                    if isinstance(val, str) and val.upper() == 'NULL':
                        print(f"      ⚠️ パーセンタイル値の一部がNULL")
                        return None
                    
                    # 値をfloatに変換（0.0も有効な値として処理）
                    result.append(float(val))
                except (ValueError, TypeError) as e:
                    print(f"      ⚠️ パーセンタイル値の変換エラー: {e}, 値: {val} (型: {type(val)})")
                    return None
            return result
        else:
            expected_count = len(rows[0]) if rows and rows[0] else 0
            print(f"      ⚠️ パーセンタイル取得結果の形式が不正: 期待11個、実際{expected_count}個")
            return None
    except Exception as e:
        print(f"      ⚠️ approx_percentile()エラー: {e}")
        return None


def get_column_min_max(catalog: str, schema: str, table: str, column: str) -> Tuple[Optional[float], Optional[float]]:
    """
    カラムのMIN/MAX値を取得
    
    Returns:
        (min_value, max_value)
    """
    try:
        # テーブル名とカラム名をクォート
        quoted_table = quote_identifier(table)
        quoted_column = quote_identifier(column)
        query = f"SELECT MIN({quoted_column}), MAX({quoted_column}) FROM {quoted_table}"
        rows = run_trino_query(catalog, schema, query)
        
        if rows and len(rows) > 0 and len(rows[0]) >= 2:
            min_val = rows[0][0]
            max_val = rows[0][1]
            
            try:
                # 値が既に数値型の場合（run_trino_queryが元の型を保持）
                if isinstance(min_val, (int, float)):
                    min_float = float(min_val)
                elif min_val is None:
                    min_float = None
                elif isinstance(min_val, str):
                    # 文字列が'NULL'または空の場合
                    if min_val.upper() == 'NULL' or not min_val.strip():
                        min_float = None
                    else:
                        try:
                            min_float = float(min_val)
                        except ValueError:
                            min_float = None
                else:
                    # その他の型はNoneとして扱う
                    min_float = None
                
                # 同様にmax_valを処理
                if isinstance(max_val, (int, float)):
                    max_float = float(max_val)
                elif max_val is None:
                    max_float = None
                elif isinstance(max_val, str):
                    if max_val.upper() == 'NULL' or not max_val.strip():
                        max_float = None
                    else:
                        try:
                            max_float = float(max_val)
                        except ValueError:
                            max_float = None
                else:
                    max_float = None
                
                return (min_float, max_float)
            except (ValueError, TypeError, AttributeError) as e:
                print(f"      ⚠️ MIN/MAX値の変換エラー: {e}, min_val={min_val}, max_val={max_val}")
                return (None, None)
        return (None, None)
    except Exception as e:
        print(f"      ⚠️ MIN/MAX取得エラー: {e}")
        return (None, None)


def get_column_data_type(catalog: str, schema: str, table: str, column: str) -> Optional[str]:
    """
    カラムのデータ型を取得（Icebergテーブル用）
    
    IcebergのDESCRIBE出力形式: [column_name, data_type, '', '']
    PostgreSQLとは異なるので注意が必要
    
    Returns:
        データ型（bigint, double, varchar等）、失敗時はNone
    """
    try:
        # テーブル名をクォート
        quoted_table = quote_identifier(table)
        query = f"DESCRIBE {quoted_table}"
        rows = run_trino_query(catalog, schema, query)
        
        for row in rows:
            # IcebergのDESCRIBE出力は [column_name, data_type, '', ''] の形式
            # カラム名は最初の要素、データ型は2番目の要素
            if len(row) >= 2 and row[0] == column:
                data_type = row[1].strip() if row[1] else None
                return data_type
        
        return None
    except Exception as e:
        print(f"      ⚠️ DESCRIBEエラー: {e}")
        return None


def trino_to_pg_data_type(trino_type: Optional[str]) -> str:
    """
    Trinoのデータ型をPostgreSQL互換のデータ型に変換
    
    Args:
        trino_type: Trinoのデータ型（bigint, double, varchar等）
    
    Returns:
        PostgreSQL互換のデータ型名
    """
    if not trino_type:
        return 'unknown'
    
    trino_type_lower = trino_type.lower()
    
    # 整数型
    if trino_type_lower in ('bigint', 'integer', 'int', 'smallint', 'tinyint', 'boolean'):
        return 'integer'
    
    # 浮動小数点型
    if trino_type_lower in ('double', 'real', 'float'):
        return 'double precision'
    
    # 数値型（DECIMAL, NUMERIC）
    if trino_type_lower in ('decimal', 'numeric'):
        return 'numeric'
    
    # 日付・時刻型
    if trino_type_lower == 'date':
        return 'date'
    if trino_type_lower in ('timestamp', 'timestamp with time zone'):
        return 'timestamp'
    if trino_type_lower in ('time', 'time with time zone'):
        return 'time'
    
    # 文字列型
    if trino_type_lower in ('varchar', 'char', 'string', 'text'):
        return 'character varying'
    
    # バイナリ型
    if trino_type_lower in ('varbinary', 'binary'):
        return 'bytea'
    
    # JSON/ARRAY等の複合型
    if trino_type_lower in ('json', 'jsonb'):
        return 'json'
    if 'array' in trino_type_lower:
        return 'array'
    if 'map' in trino_type_lower:
        return 'map'
    
    # デフォルト：不明な型は文字列として扱う
    return 'character varying'


def infer_data_type(col_stat: Dict[str, Any]) -> str:
    """
    カラム統計から データ型を推測（フォールバック用）
    PostgreSQL互換のデータ型名を返す
    
    注意: この関数はtrino_data_typeが取得できない場合のフォールバックです
    """
    low = col_stat.get('low_value')
    high = col_stat.get('high_value')
    
    # 値が存在しない場合
    if low is None and high is None:
        return 'unknown'
    
    # 数値型の判定
    if low is not None and high is not None:
        try:
            float(low)
            float(high)
            # 整数かどうか判定
            if '.' not in str(low) and '.' not in str(high):
                return 'integer'
            return 'double precision'
        except (ValueError, TypeError):
            pass
    
    # 日付型の判定
    if low and isinstance(low, str):
        # YYYY-MM-DD形式
        if re.match(r'\d{4}-\d{2}-\d{2}', low):
            return 'date'
        # タイムスタンプ形式
        if re.match(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', low):
            return 'timestamp'
    
    # デフォルトは文字列型
    return 'character varying'


def collect_schema_stats(
    catalog: str,
    schema: str,
    tables: Optional[List[str]],
    output_dir_base: Path
) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    """
    単一スキーマの統計情報を収集
    
    Returns:
        (成功フラグ, テーブル統計, カラム統計)
    """
    # テーブルリストを取得
    if tables:
        table_list = tables
        print(f"📋 指定されたテーブル: {', '.join(table_list)}")
    else:
        print("📋 全テーブルを取得中...")
        table_list = get_table_list(catalog, schema)
        if not table_list:
            print(f"   ⚠️ テーブルが見つかりませんでした")
            return False, {}, {}
        print(f"   見つかったテーブル: {', '.join(table_list)}")
    
    print()
    
    # 出力ディレクトリを作成
    schema_dir_name = f"{catalog}_{schema}"
    schema_output_dir = output_dir_base / schema_dir_name
    schema_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 出力ディレクトリ: {schema_output_dir}")
    print()
    
    # 各テーブルの統計情報を収集
    table_stats = {}
    column_stats = {}
    
    for table in table_list:
        print(f"📊 {table} の統計情報を収集中...")
        stats = get_table_stats(catalog, schema, table)
        
        if stats:
            row_count = stats.get('row_count', 0) or 0
            
            table_stats[table] = {
                'reltuples': row_count,
                'relpages': estimate_pages(row_count),
                'table_name': table,
                'row_count': row_count,
                'num_columns': len(stats.get('columns', {}))
            }
            
            # カラム統計を保存
            for col_name, col_stat in stats.get('columns', {}).items():
                tuple_key = f"('{table}', '{col_name}')"
                
                distinct_count = col_stat.get('distinct_values_count')
                null_frac = col_stat.get('nulls_fraction', 0) or 0
                data_size = col_stat.get('data_size')
                
                # データ型を取得
                data_type_raw = get_column_data_type(catalog, schema, table, col_name)
                
                # PostgreSQL互換の型に変換（優先順位：trino_data_type > 推論）
                if data_type_raw:
                    inferred_type = trino_to_pg_data_type(data_type_raw)
                else:
                    # fallback: 統計情報から推測
                    inferred_type = infer_data_type(col_stat)
                
                # QueryFormer用のデータ型変換
                queryformer_datatype = None
                if data_type_raw:
                    if data_type_raw in ('bigint', 'integer', 'int'):
                        queryformer_datatype = 'int'
                    elif data_type_raw in ('double', 'real', 'float'):
                        queryformer_datatype = 'float'
                    else:
                        queryformer_datatype = 'misc'
                elif inferred_type == 'integer':
                    queryformer_datatype = 'int'
                elif inferred_type == 'double precision':
                    queryformer_datatype = 'float'
                else:
                    queryformer_datatype = 'misc'
                
                # MIN/MAX/パーセンタイルを取得（数値型の場合）
                min_val, max_val = None, None
                percentiles = None
                if queryformer_datatype in ('int', 'float'):
                    # MIN/MAXは必須（QueryFormerで使用）
                    min_val, max_val = get_column_min_max(catalog, schema, table, col_name)
                    
                    # パーセンタイルはTrino環境によっては取得できない可能性がある
                    # MIN/MAXが取得できた場合のみパーセンタイルを取得
                    if min_val is not None and max_val is not None:
                        percentiles = get_column_percentiles(catalog, schema, table, col_name)
                    else:
                        # MIN/MAXが取得できない場合、パーセンタイルも取得しない
                        percentiles = None
                
                column_stats[tuple_key] = {
                    'n_distinct': distinct_count if distinct_count is not None else -1,
                    'null_frac': null_frac,
                    'avg_width': (
                        int(data_size / row_count)
                        if data_size is not None and row_count > 0
                        else 0
                    ),
                    'correlation': 0.0,
                    'data_type': inferred_type,
                    'datatype': queryformer_datatype,
                    'min': min_val,
                    'max': max_val,
                    'percentiles': percentiles,
                    'table': table,
                    'column': col_name,
                    'data_size': data_size,
                    'distinct_values_count': distinct_count,
                    'nulls_fraction': null_frac,
                    'low_value': col_stat.get('low_value'),
                    'high_value': col_stat.get('high_value'),
                    'trino_data_type': data_type_raw,
                }
            
            row_count = stats.get('row_count')
            num_columns = len(stats.get('columns', {}))
            print(f"   ✅ 行数: {int(row_count) if row_count else 'N/A'}, "
                  f"カラム数: {num_columns}")
            
            columns = list(stats.get('columns', {}).items())
            for col_name, col_stat in columns[:5]:
                distinct = col_stat.get('distinct_values_count')
                nulls = col_stat.get('nulls_fraction')
                info_parts = []
                if distinct is not None:
                    info_parts.append(f"distinct={int(distinct)}")
                if nulls is not None:
                    info_parts.append(f"nulls={nulls:.2%}")
                info_str = ", ".join(info_parts) if info_parts else "no stats"
                print(f"     - {col_name}: {info_str}")
            
            if len(columns) > 5:
                print(f"     ... 他 {len(columns) - 5} カラム")
        else:
            print(f"   ❌ 統計情報の取得に失敗")
            return False, {}, {}
        
        print()
    
    # メタデータとJSON出力
    metadata = {
        'catalog': catalog,
        'schema': schema,
        'num_tables': len(table_stats),
        'num_columns': len(column_stats)
    }
    
    table_stats_file = schema_output_dir / 'table_stats.json'
    column_stats_file = schema_output_dir / 'column_stats.json'
    metadata_file = schema_output_dir / 'metadata.json'
    
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
    
    total_rows = sum(stats.get('row_count', 0) or 0 for stats in table_stats.values())
    print(f"📊 合計行数: {int(total_rows):,}")
    print()
    
    return True, table_stats, column_stats


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
        help='スキーマ名（単一スキーマ処理時）'
    )
    parser.add_argument(
        '--schemas',
        help='スキーマ名のカンマ区切りリスト（複数スキーマ処理時）'
    )
    parser.add_argument(
        '--all-schemas',
        action='store_true',
        help='全データセット（スキーマ）の統計情報を収集'
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
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='エラーが発生しても続行する（複数スキーマ処理時）'
    )
    
    args = parser.parse_args()
    
    # スキーマリストを決定
    if args.all_schemas:
        schemas = AVAILABLE_DATASETS
    elif args.schemas:
        schemas = [s.strip() for s in args.schemas.split(',')]
    elif args.schema:
        schemas = [args.schema]
    else:
        # デフォルト: imdb
        schemas = ['imdb']
    
    # 出力ディレクトリ
    output_dir_base = Path(args.output_dir)
    output_dir_base.mkdir(parents=True, exist_ok=True)
    
    # 複数スキーマ処理かどうか
    is_multi_schema = len(schemas) > 1
    
    if is_multi_schema:
        print(f"🚀 Trino統計情報一括収集開始")
        print(f"  Catalog: {args.catalog}")
        print(f"  スキーマ数: {len(schemas)}")
        print(f"  出力ディレクトリ: {output_dir_base}")
        print()
        
        success_count = 0
        failed_schemas = []
        
        for i, schema in enumerate(schemas, 1):
            print(f"\n{'='*80}")
            print(f"[{i}/{len(schemas)}] {schema} の統計情報を収集中...")
            print(f"{'='*80}\n")
            
            tables = None
            if args.tables:
                tables = [t.strip() for t in args.tables.split(',')]
            
            success, _, _ = collect_schema_stats(
                catalog=args.catalog,
                schema=schema,
                tables=tables,
                output_dir_base=output_dir_base
            )
            
            if success:
                success_count += 1
                print(f"✅ {schema} の統計情報収集完了\n")
            else:
                failed_schemas.append(schema)
                print(f"❌ {schema} の統計情報収集失敗\n")
                if not args.continue_on_error:
                    print(f"❌ エラーにより処理を中断します")
                    break
        
        # 最終サマリー
        print("\n" + "=" * 80)
        print("📊 統計情報収集完了")
        print("=" * 80)
        print(f"✅ 成功: {success_count}/{len(schemas)} スキーマ")
        
        if failed_schemas:
            print(f"❌ 失敗: {len(failed_schemas)} スキーマ")
            print(f"   失敗したスキーマ: {', '.join(failed_schemas)}")
        else:
            print("🎉 すべてのスキーマの統計情報収集に成功しました！")
        
        print()
        print(f"📁 統計情報の保存先: {output_dir_base}/")
        
        # 生成されたディレクトリをリスト
        if output_dir_base.exists():
            subdirs = [d.name for d in output_dir_base.iterdir() if d.is_dir()]
            print(f"   生成されたディレクトリ数: {len(subdirs)}")
            for subdir in sorted(subdirs)[:10]:  # 最初の10個のみ表示
                print(f"   - {subdir}")
            if len(subdirs) > 10:
                print(f"   ... 他 {len(subdirs) - 10} ディレクトリ")
        
        sys.exit(0 if success_count == len(schemas) else 1)
    
    else:
        # 単一スキーマ処理
        schema = schemas[0]
        print(f"📊 Trino統計情報収集開始")
        print(f"  Catalog: {args.catalog}")
        print(f"  Schema: {schema}")
        print()
        
        tables = None
        if args.tables:
            tables = [t.strip() for t in args.tables.split(',')]
        
        success, _, _ = collect_schema_stats(
            catalog=args.catalog,
            schema=schema,
            tables=tables,
            output_dir_base=output_dir_base
        )
        
        if success:
            print(f"\n💡 使用方法:")
            schema_dir_name = f"{args.catalog}_{schema}"
            schema_output_dir = output_dir_base / schema_dir_name
            print(f"   統計情報ディレクトリ: {schema_output_dir}")
            print(f"   エンコーディング時にこのディレクトリを指定してください")
            sys.exit(0)
        else:
            print(f"\n❌ 統計情報の収集に失敗しました")
            sys.exit(1)


if __name__ == '__main__':
    main()