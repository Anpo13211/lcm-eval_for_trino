"""
Trinoデータベース接続クラス
"""

import os
import time
from pathlib import Path

import pandas as pd
from trino.dbapi import connect

from cross_db_benchmark.benchmark_tools.database import DatabaseConnection, ExecutionMode
from cross_db_benchmark.benchmark_tools.utils import load_schema_sql, load_schema_json, load_column_statistics


class TrinoDatabaseConnection(DatabaseConnection):
    def __init__(self, db_name=None, database_kwargs=None, **kwargs):
        super().__init__(db_name=db_name, database_kwargs=database_kwargs)
        self.connection = None

    def _get_connection(self):
        """Trino接続を取得"""
        if self.connection is None:
            # Trino接続パラメータ
            trino_kwargs = {
                'host': self.database_kwargs.get('host', 'localhost'),
                'port': self.database_kwargs.get('port', 8080),
                'user': self.database_kwargs.get('user', 'admin'),
                'catalog': self.database_kwargs.get('catalog', 'hive'),
                'schema': self.database_kwargs.get('schema', 'default'),
            }
            self.connection = connect(**trino_kwargs)
        return self.connection

    def load_database(self, dataset, data_dir, force=False):
        """データベースをロード（Trinoでは簡略化）"""
        print(f"Trino database loading for {dataset} is simplified")
        print(f"Data directory: {data_dir}")
        
        # Trinoでは直接的なデータベース作成は不要
        # スキーマファイルの確認のみ（zero-shot_datasets配下を優先）
        try:
            schema = load_schema_json(dataset, prefer_zero_shot=True)
            print(f"Schema found: {len(schema.tables)} tables")
        except FileNotFoundError:
            print("No schema file found")
        
        schema_sql = load_schema_sql(dataset, 'trino.sql')
        if schema_sql:
            print("Schema SQL found, but Trino loading is handled externally")
        else:
            print("No Trino schema file found")

    def replicate_tuples(self, dataset, data_dir, no_prev_replications, vac_analyze=True):
        """タプルの複製（Trinoでは簡略化）"""
        print(f"Trino tuple replication for {dataset} is simplified")
        print(f"Replications: {no_prev_replications}")

    def set_statement_timeout(self, timeout_sec):
        """ステートメントタイムアウトを設定"""
        # Trinoでは接続レベルでタイムアウトを設定
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SET SESSION query_max_run_time = '{timeout_sec}s'")

    def run_query_collect_statistics(self, sql, repetitions, prefix, hint_validation,
                                     include_hint_notices, explain_only):
        """クエリを実行して統計を収集"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        results = []
        
        for i in range(repetitions):
            try:
                start_time = time.perf_counter()
                
                if explain_only:
                    # EXPLAIN ANALYZEを実行
                    explain_sql = f"EXPLAIN ANALYZE VERBOSE {sql}"
                    cursor.execute(explain_sql)
                    plan_result = cursor.fetchall()
                    
                    # プランを文字列として結合
                    plan_text = '\n'.join([row[0] for row in plan_result])
                    
                    execution_time = time.perf_counter() - start_time
                    
                    results.append({
                        'sql': sql,
                        'plan': plan_text,
                        'execution_time': execution_time,
                        'repetition': i
                    })
                else:
                    # 通常のクエリ実行
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    
                    execution_time = time.perf_counter() - start_time
                    
                    results.append({
                        'sql': sql,
                        'rows': len(rows),
                        'execution_time': execution_time,
                        'repetition': i
                    })
                    
            except Exception as e:
                print(f"Query failed: {e}")
                results.append({
                    'sql': sql,
                    'error': str(e),
                    'repetition': i
                })
        
        return results

    def collect_db_statistics(self):
        """データベース統計を収集"""
        # Trinoでは統計情報の収集は簡略化
        return {
            'database': 'trino',
            'catalog': self.database_kwargs.get('catalog', 'hive'),
            'schema': self.database_kwargs.get('schema', 'default')
        }

    def submit_query(self, sql, db_created=True):
        """クエリを実行"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql)
            if cursor.description:
                return cursor.fetchall()
            return []
        except Exception as e:
            print(f"Query failed: {e}")
            raise

    def check_if_database_exists(self):
        """データベースの存在確認（Trinoでは簡略化）"""
        # Trinoではスキーマの存在確認
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SHOW SCHEMAS")
            schemas = [row[0] for row in cursor.fetchall()]
            return self.db_name in schemas
        except:
            return False

    def drop(self):
        """データベースを削除"""
        if self.connection:
            self.connection.close()
            self.connection = None
