# Abstract Base Classes for Database Query Plan Parsing

このディレクトリには、全てのデータベースシステム（PostgreSQL, Trino, MySQL等）で共通のインターフェースを定義する抽象基底クラスが含まれています。

## 📁 ファイル構成

```
abstract/
├── __init__.py          # パッケージ初期化
├── plan_operator.py     # AbstractPlanOperator（抽象プラン演算子）
├── plan_parser.py       # AbstractPlanParser（抽象パーサー）
└── README.md           # このファイル
```

## 🎯 目的

1. **統一インターフェース**: 全DBMSで同じAPIを提供
2. **PostgreSQL互換**: 特徴量名はPostgreSQLに準拠
3. **拡張性**: 新しいDBMS追加時の契約を明確化
4. **型安全性**: 抽象メソッドで実装を強制

## 📐 AbstractPlanOperator

クエリプラン演算子の抽象基底クラス。

### 必須特徴量（全DBMS実装で提供必須）

| 特徴量名 | 型 | 説明 |
|---------|-----|------|
| `op_name` | `str` | 演算子名 |
| `est_card` | `float` | 推定カーディナリティ（行数） |
| `act_card` | `float` | 実測カーディナリティ（行数） |
| `est_width` | `float` | 推定行幅（バイト） |
| `workers_planned` | `int` | 計画された並列ワーカー数 |
| `act_children_card` | `float` | 子ノードの実測カーディナリティ積 |
| `est_children_card` | `float` | 子ノードの推定カーディナリティ積 |

### 任意特徴量（DBMSで取得可能な場合のみ）

| 特徴量名 | 型 | 説明 |
|---------|-----|------|
| `est_cost` | `Optional[float]` | 推定コスト（PostgreSQL形式） |
| `est_startup_cost` | `Optional[float]` | 推定起動コスト |
| `act_time` | `Optional[float]` | 実測時間（ms） |
| `table` | `Optional[str]` | テーブル名 |
| `columns` | `Optional[List[str]]` | カラムリスト |
| `output_columns` | `Optional[List[Dict]]` | 出力カラム情報 |
| `filter_columns` | `Optional[Any]` | フィルター条件 |

### 必須実装メソッド

```python
@abstractmethod
def parse_lines(self, alias_dict: Optional[Dict] = None, **kwargs) -> None:
    """DBMS固有の生プラン文字列から特徴量を抽出"""
    pass

@abstractmethod
def parse_columns_bottom_up(
    self, 
    column_id_mapping: Dict,
    partial_column_name_mapping: Dict,
    table_id_mapping: Dict,
    **kwargs
) -> set:
    """ボトムアップでカラム情報を統計情報と照合"""
    pass
```

### 実装例（Trino）

```python
from cross_db_benchmark.benchmark_tools.abstract import AbstractPlanOperator

class TrinoPlanOperator(AbstractPlanOperator):
    def __init__(self):
        super().__init__()
        self.database_type = "trino"
    
    def parse_lines(self, alias_dict=None, **kwargs):
        # Trino固有のパース処理
        trino_params = self._parse_trino_format()
        
        # ★ PostgreSQL互換名にマッピング
        self.op_name = trino_params['op_name']
        self.est_card = trino_params['est_rows']           # est_rows → est_card
        self.act_card = trino_params['act_output_rows']    # act_output_rows → act_card
        self.est_width = trino_params['est_width']
        
        # Trinoにはest_costがない
        self.est_cost = None
    
    def parse_columns_bottom_up(self, column_id_mapping, ...):
        # 実装
        pass
```

## 📖 AbstractPlanParser

クエリプランパーサーの抽象基底クラス。

### 必須実装メソッド

```python
@abstractmethod
def parse_plans(
    self,
    run_stats: Any,
    min_runtime: float = 100,
    max_runtime: float = 30000,
    **kwargs
) -> Dict[str, Any]:
    """複数のクエリプランを一括パース"""
    pass

@abstractmethod
def parse_single_plan(
    self, 
    plan_text: str,
    **kwargs
) -> Optional[AbstractPlanOperator]:
    """単一のクエリプランをパース"""
    pass
```

### 戻り値の形式

`parse_plans()` は以下の形式の辞書を返す必要があります：

```python
{
    'parsed_plans': List[AbstractPlanOperator],  # パースされたプラン
    'avg_runtimes': List[float],                 # 実行時間（ms）
    'database': str,                              # DBMS名（例: "trino"）
    'stats': {                                    # 統計情報（任意）
        'total_plans': int,
        'avg_runtime': float,
        'min_runtime': float,
        'max_runtime': float,
    }
}
```

### 実装例（Trino）

```python
from cross_db_benchmark.benchmark_tools.abstract import AbstractPlanParser

class TrinoPlanParser(AbstractPlanParser):
    def __init__(self):
        super().__init__(database_type="trino")
    
    def parse_plans(self, run_stats, min_runtime=100, max_runtime=30000, **kwargs):
        # 既存のパース処理を利用
        legacy_plans = parse_trino_legacy(run_stats)
        
        # AbstractPlanOperatorに変換
        abstract_plans = []
        for legacy_plan in legacy_plans:
            if min_runtime <= legacy_plan.runtime <= max_runtime:
                abstract_plan = TrinoPlanOperator(legacy_plan)
                abstract_plans.append(abstract_plan)
        
        # ★ 統一形式で返す
        return {
            'parsed_plans': abstract_plans,
            'avg_runtimes': [p.plan_runtime for p in abstract_plans],
            'database': 'trino',
            'stats': self.get_statistics(abstract_plans)
        }
    
    def parse_single_plan(self, plan_text: str, **kwargs):
        # 実装
        pass
```

## ✅ 新しいDBMS追加手順

### 1. PlanOperatorを実装

```python
# src/cross_db_benchmark/benchmark_tools/mysql/abstract_plan_operator.py
from cross_db_benchmark.benchmark_tools.abstract import AbstractPlanOperator

class MySQLPlanOperator(AbstractPlanOperator):
    def __init__(self):
        super().__init__()
        self.database_type = "mysql"
    
    def parse_lines(self, alias_dict=None, **kwargs):
        # MySQL固有のパース処理
        mysql_params = self._parse_mysql_format()
        
        # PostgreSQL互換名にマッピング
        self.op_name = mysql_params['operator']
        self.est_card = mysql_params['rows']            # rows → est_card
        self.act_card = mysql_params['actual_rows']     # actual_rows → act_card
        # ...
    
    def parse_columns_bottom_up(self, column_id_mapping, ...):
        # 実装
        pass
```

### 2. Parserを実装

```python
# src/cross_db_benchmark/benchmark_tools/mysql/abstract_plan_parser.py
from cross_db_benchmark.benchmark_tools.abstract import AbstractPlanParser

class MySQLPlanParser(AbstractPlanParser):
    def __init__(self):
        super().__init__(database_type="mysql")
    
    def parse_plans(self, run_stats, **kwargs):
        # 実装（統一形式で返す）
        pass
    
    def parse_single_plan(self, plan_text: str, **kwargs):
        # 実装
        pass
```

### 3. 完了！

既存のトレーニングスクリプトやモデルが自動的にMySQLに対応します。

## 🚫 禁止事項

### ❌ DBMS固有の特徴量名を直接使用

```python
# ❌ ダメな例
self.est_rows = 1000        # Trino固有
self.estimated_rows = 1000  # MySQL固有
```

### ✅ PostgreSQL互換名を使用

```python
# ✅ 正しい例
self.est_card = 1000        # 統一名
```

### ❌ 戻り値の形式を変更

```python
# ❌ ダメな例
def parse_plans(self, ...):
    return {
        'plans': [...],           # 'parsed_plans'ではない
        'times': [...],           # 'avg_runtimes'ではない
    }
```

### ✅ 統一形式を使用

```python
# ✅ 正しい例
def parse_plans(self, ...):
    return {
        'parsed_plans': [...],    # 必須キー
        'avg_runtimes': [...],    # 必須キー
        'database': 'trino',      # 必須キー
        'stats': {...}            # 任意キー
    }
```

## 📚 使用例

```python
from cross_db_benchmark.benchmark_tools.postgres.abstract_plan_parser import PostgresPlanParser
from cross_db_benchmark.benchmark_tools.trino.abstract_plan_parser import TrinoPlanParser

# どのDBMSでも同じコード
def process_plans(parser, run_stats):
    result = parser.parse_plans(run_stats)
    
    for plan in result['parsed_plans']:
        # 統一されたAPIで特徴量にアクセス
        print(f"DB: {plan.database_type}")
        print(f"Op: {plan.op_name}")
        print(f"Est Card: {plan.est_card}")
        print(f"Act Card: {plan.act_card}")

# PostgreSQL
pg_parser = PostgresPlanParser()
process_plans(pg_parser, pg_run_stats)

# Trino（同じコード！）
trino_parser = TrinoPlanParser()
process_plans(trino_parser, trino_run_stats)
```

## 🔍 検証

```python
# プランの検証
errors = plan.validate()
if errors:
    print(f"Validation errors: {errors}")

# パーサーの検証
parser = TrinoPlanParser()
result = parser.parse_plans(run_stats)
errors = parser.validate_parsed_plans(result['parsed_plans'])
if errors:
    print(f"Parse errors: {errors}")
```

## 📞 サポート

質問や問題がある場合は、このREADMEと抽象クラスのdocstringを参照してください。
全ての必須事項と例が記載されています。






