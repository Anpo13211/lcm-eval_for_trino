"""
Trinoの外部統計情報管理
"""

import json
import os
from typing import Dict, Optional, Any

class TrinoExternalStats:
    """Trinoの外部統計情報を管理するクラス"""
    
    def __init__(self, stats_file: Optional[str] = None):
        """
        Args:
            stats_file: 統計情報ファイルのパス（JSON形式）
        """
        self.stats_file = stats_file or "trino_table_stats.json"
        self.stats_cache = {}
        self._load_stats()
    
    def _load_stats(self):
        """統計情報ファイルを読み込み"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    self.stats_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load stats file {self.stats_file}: {e}")
                self.stats_cache = {}
        else:
            # デフォルトの統計情報を作成
            self._create_default_stats()
    
    def _create_default_stats(self):
        """デフォルトの統計情報を作成"""
        self.stats_cache = {
            "accidents:public.upravna_enota": {
                "reltuples": 58,
                "relpages": 1,
                "columns": {
                    "id_upravna_enota": {
                        "avg_width": 4,
                        "correlation": 0.9,
                        "n_distinct": 58,
                        "null_frac": 0.0
                    },
                    "povrsina": {
                        "avg_width": 4,
                        "correlation": 0.1,
                        "n_distinct": 58,
                        "null_frac": 0.0
                    }
                }
            },
            "accidents:public.oseba": {
                "reltuples": 954036,
                "relpages": 4770,
                "columns": {
                    "upravna_enota": {
                        "avg_width": 4,
                        "correlation": 0.1,
                        "n_distinct": 1000,
                        "null_frac": 0.0
                    },
                    "varnostni_pas_ali_celada": {
                        "avg_width": 1,
                        "correlation": 0.1,
                        "n_distinct": 2,
                        "null_frac": 0.0
                    },
                    "vozniski_staz_LL": {
                        "avg_width": 4,
                        "correlation": 0.1,
                        "n_distinct": 100,
                        "null_frac": 0.0
                    }
                }
            }
        }
        self._save_stats()
    
    def _save_stats(self):
        """統計情報ファイルを保存"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats_cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save stats file {self.stats_file}: {e}")
    
    def get_table_stats(self, table_name: str) -> Optional[Dict[str, Any]]:
        """テーブルの統計情報を取得"""
        return self.stats_cache.get(table_name)
    
    def get_reltuples(self, table_name: str) -> Optional[int]:
        """テーブルのreltuplesを取得"""
        table_stats = self.get_table_stats(table_name)
        return table_stats.get("reltuples") if table_stats else None
    
    def get_relpages(self, table_name: str) -> Optional[int]:
        """テーブルのrelpagesを取得"""
        table_stats = self.get_table_stats(table_name)
        return table_stats.get("relpages") if table_stats else None
    
    def get_column_stats(self, table_name: str, column_name: str) -> Optional[Dict[str, Any]]:
        """カラムの統計情報を取得"""
        table_stats = self.get_table_stats(table_name)
        if table_stats and "columns" in table_stats:
            return table_stats["columns"].get(column_name)
        return None
    
    def get_column_avg_width(self, table_name: str, column_name: str) -> Optional[float]:
        """カラムの平均幅を取得"""
        column_stats = self.get_column_stats(table_name, column_name)
        return column_stats.get("avg_width") if column_stats else None
    
    def get_column_correlation(self, table_name: str, column_name: str) -> Optional[float]:
        """カラムの相関を取得"""
        column_stats = self.get_column_stats(table_name, column_name)
        return column_stats.get("correlation") if column_stats else None
    
    def get_column_n_distinct(self, table_name: str, column_name: str) -> Optional[int]:
        """カラムの異なる値の数を取得"""
        column_stats = self.get_column_stats(table_name, column_name)
        return column_stats.get("n_distinct") if column_stats else None
    
    def get_column_null_frac(self, table_name: str, column_name: str) -> Optional[float]:
        """カラムのNULL値の割合を取得"""
        column_stats = self.get_column_stats(table_name, column_name)
        return column_stats.get("null_frac") if column_stats else None
    
    def update_table_stats(self, table_name: str, stats: Dict[str, Any]):
        """テーブルの統計情報を更新"""
        self.stats_cache[table_name] = stats
        self._save_stats()
    
    def update_column_stats(self, table_name: str, column_name: str, stats: Dict[str, Any]):
        """カラムの統計情報を更新"""
        if table_name not in self.stats_cache:
            self.stats_cache[table_name] = {"columns": {}}
        if "columns" not in self.stats_cache[table_name]:
            self.stats_cache[table_name]["columns"] = {}
        
        self.stats_cache[table_name]["columns"][column_name] = stats
        self._save_stats()
    
    def list_tables(self) -> list:
        """統計情報があるテーブルのリストを取得"""
        return list(self.stats_cache.keys())
    
    def has_table(self, table_name: str) -> bool:
        """テーブルの統計情報があるかチェック"""
        return table_name in self.stats_cache
    
    def has_column(self, table_name: str, column_name: str) -> bool:
        """カラムの統計情報があるかチェック"""
        table_stats = self.get_table_stats(table_name)
        if table_stats and "columns" in table_stats:
            return column_name in table_stats["columns"]
        return False


# グローバルインスタンス
_global_stats = None

def get_global_stats() -> TrinoExternalStats:
    """グローバルな統計情報インスタンスを取得"""
    global _global_stats
    if _global_stats is None:
        _global_stats = TrinoExternalStats()
    return _global_stats

def set_global_stats(stats: TrinoExternalStats):
    """グローバルな統計情報インスタンスを設定"""
    global _global_stats
    _global_stats = stats
