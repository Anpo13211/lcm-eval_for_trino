# Core Parsers Module

統一されたパーサーインターフェースを提供するモジュール。

## 📋 目的

すべての DBMS パーサーを `src/core/` で一元管理し、統一されたインターフェースを提供します。

## 📁 ファイル構成

```
src/core/parsers/
├── __init__.py       # パッケージエントリポイント
├── base.py           # AbstractPlanParser（抽象基底クラス）
├── adapter.py        # レガシーパーサーのアダプター
└── README.md         # このファイル
```

## 🎯 AbstractPlanParser

すべての DBMS パーサーが実装すべき抽象基底クラス。

### 必須メソッド

```python
from core.parsers import AbstractPlanParser

class MyDBMSParser(AbstractPlanParser):
    def __init__(self):
        super().__init__("mydbms")  # DBMS名を指定
    
    def parse_explain_analyze_file(
        self,
        file_path: str,
        min_runtime: float = 0,
        max_runtime: float = float('inf'),
        **kwargs
    ) -> Tuple[List[Any], List[float]]:
        """
        EXPLAIN ANALYZE ファイルをパース
        
        Returns:
            (parsed_plans, runtimes)
        """
        # 実装
        pass
    
    def parse_raw_plan(
        self,
        plan_text: str,
        analyze: bool = True,
        **kwargs
    ) -> Tuple[Any, float, float]:
        """
        生のプランテキストをパース
        
        Returns:
            (root_operator, execution_time, planning_time)
        """
        # 実装
        pass
```

### オプションメソッド

```python
def parse_multiple_plans(
    self,
    plan_texts: List[str],
    analyze: bool = True,
    **kwargs
) -> PlanParseResult:
    """
    複数プランの一括パース（デフォルト実装あり）
    """
    pass

def get_statistics(
    self,
    parsed_plans: List[Any]
) -> Dict[str, Any]:
    """
    パース済みプランから統計情報を抽出（オプション）
    """
    pass

def validate_plan(self, plan: Any) -> bool:
    """
    プランの妥当性をチェック（オプション）
    """
    pass
```

## 🔄 レガシーパーサーとの互換性

既存のパーサー（`cross_db_benchmark/benchmark_tools/` にあるもの）を
新しいインターフェースでラップできます。

### アダプターの使用

```python
from core.parsers.adapter import wrap_legacy_parser

# 既存のパーサーをラップ
from cross_db_benchmark.benchmark_tools.postgres.parse_plan import PostgresPlanParser

legacy_parser = PostgresPlanParser()
unified_parser = wrap_legacy_parser(legacy_parser, "postgres")

# 新しい統一インターフェースで使用
plans, runtimes = unified_parser.parse_explain_analyze_file("plans.txt")
```

### プラグインでの使用例

```python
from core.plugins.dbms_plugin import DBMSPlugin
from core.parsers.adapter import wrap_legacy_parser

class PostgreSQLPlugin(DBMSPlugin):
    name = "postgres"
    display_name = "PostgreSQL"
    
    def get_parser(self):
        # レガシーパーサーをラップして返す
        from cross_db_benchmark.benchmark_tools.postgres.parse_plan import PostgresPlanParser
        legacy = PostgresPlanParser()
        return wrap_legacy_parser(legacy, self.name)
```

## 📦 PlanParseResult

パース結果を型安全に返すためのデータクラス。

```python
from core.parsers import PlanParseResult

result = PlanParseResult(
    parsed_plans=[plan1, plan2, plan3],
    runtimes=[100.5, 200.3, 150.7],
    planning_times=[5.0, 6.2, 4.8],
    metadata={'source': 'file.txt'}
)

print(f"Parsed {len(result.parsed_plans)} plans")
print(f"Average runtime: {np.mean(result.runtimes):.2f}ms")
```

## 🔌 プラグインシステムとの統合

パーサーはプラグインシステム経由で取得できます。

```python
from core.plugins.registry import DBMSRegistry

# プラグイン経由で取得（推奨）
parser = DBMSRegistry.get_parser("trino")
plans, runtimes = parser.parse_explain_analyze_file("plans.txt")

# または直接取得
from core.parsers import get_parser_for_dbms
parser = get_parser_for_dbms("trino")
```

## 🆕 新しい DBMS のパーサー実装

新しい DBMS のパーサーを追加する手順：

### 1. パーサークラスを作成

```python
# src/plugins/mydbms/parser.py

from core.parsers import AbstractPlanParser
from typing import Any, List, Tuple

class MyDBMSParser(AbstractPlanParser):
    """My DBMS のパーサー"""
    
    def __init__(self):
        super().__init__("mydbms")
    
    def parse_explain_analyze_file(
        self,
        file_path: str,
        min_runtime: float = 0,
        max_runtime: float = float('inf'),
        **kwargs
    ) -> Tuple[List[Any], List[float]]:
        """ファイルからプランをパース"""
        
        plans = []
        runtimes = []
        
        with open(file_path, 'r') as f:
            for line in f:
                # MyDBMS 固有のフォーマットをパース
                plan, runtime = self._parse_line(line)
                
                # フィルタリング
                if min_runtime <= runtime <= max_runtime:
                    plans.append(plan)
                    runtimes.append(runtime)
        
        return plans, runtimes
    
    def parse_raw_plan(
        self,
        plan_text: str,
        analyze: bool = True,
        **kwargs
    ) -> Tuple[Any, float, float]:
        """生プランテキストをパース"""
        
        # MyDBMS 固有のパース処理
        root_operator = self._build_plan_tree(plan_text)
        
        # 実行時間を抽出
        execution_time = self._extract_execution_time(plan_text)
        planning_time = self._extract_planning_time(plan_text)
        
        return root_operator, execution_time, planning_time
    
    def _parse_line(self, line: str):
        # 実装
        pass
    
    def _build_plan_tree(self, text: str):
        # 実装
        pass
    
    def _extract_execution_time(self, text: str):
        # 実装
        pass
    
    def _extract_planning_time(self, text: str):
        # 実装
        pass
```

### 2. プラグインに統合

```python
# src/plugins/mydbms/plugin.py

from core.plugins.dbms_plugin import DBMSPlugin
from .parser import MyDBMSParser

class MyDBMSPlugin(DBMSPlugin):
    name = "mydbms"
    display_name = "My DBMS"
    
    def get_parser(self):
        return MyDBMSParser()
    
    def get_statistics_converter(self):
        # 統計情報変換器を返す
        pass
    
    def get_connection_factory(self):
        # 接続クラスを返す
        pass
    
    def get_operator_normalizer(self):
        # オペレータ正規化器を返す
        pass
```

### 3. プラグインを登録

```python
# src/core/init_plugins.py に追加

from plugins.mydbms.plugin import MyDBMSPlugin

DBMSRegistry.register(MyDBMSPlugin())
```

### 完了！

```python
# 使用例
parser = DBMSRegistry.get_parser("mydbms")
plans, runtimes = parser.parse_explain_analyze_file("plans.txt")
```

## 🎯 設計原則

1. **統一インターフェース**: すべての DBMS で同じメソッド名・シグネチャ
2. **型安全**: 戻り値は明確な型で定義
3. **拡張可能**: オプションメソッドでカスタマイズ可能
4. **後方互換**: アダプターでレガシーパーサーも使用可能
5. **プラグイン統合**: プラグインシステムとシームレスに連携

## 🔄 移行ガイド

### 従来のコード

```python
# 悪い例: DBMS ごとに異なるインポートと使い方
if dbms == "postgres":
    from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_plan
    plans, runtimes = parse_plan(file_path, ...)
elif dbms == "trino":
    from cross_db_benchmark.benchmark_tools.trino.parse_plan import parse_plan
    plans, runtimes = parse_plan(file_path, ...)
```

### 新しいコード

```python
# 良い例: 統一インターフェース
from core.plugins.registry import DBMSRegistry

parser = DBMSRegistry.get_parser(dbms_name)
plans, runtimes = parser.parse_explain_analyze_file(file_path)
```

## 📊 利点

| 項目 | 従来 | 新アーキテクチャ |
|-----|------|--------------|
| **インポート** | DBMS ごとに異なる | 統一 |
| **メソッド名** | 不統一 | 統一 |
| **型安全性** | なし | あり |
| **テスト** | DBMS ごと | 共通インターフェースで一括 |
| **新規追加** | 散在する実装を参考 | 明確な手順 |

このモジュールにより、パーサーが完全に `src/core/` で管理され、
すべての DBMS で統一されたインターフェースを使用できるようになりました！

