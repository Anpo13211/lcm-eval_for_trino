import math
import re
from types import SimpleNamespace

from cross_db_benchmark.benchmark_tools.abstract.plan_operator import AbstractPlanOperator
from cross_db_benchmark.benchmark_tools.generate_workload import Aggregator, ExtendedAggregator, LogicalOperator
from cross_db_benchmark.benchmark_tools.trino.parse_filter import parse_filter, PredicateNode
from cross_db_benchmark.benchmark_tools.trino.utils import child_prod

# Trino特有の正規表現パターン
trino_estimates_regex = re.compile(r'Estimates: \{rows: ([\d,?]+) \(([\d.?]+[KMG]?B?)\)(?:, cpu: ([\d.?]+[KMG]?)?)?(?:, memory: ([\d.?]+[KMG]?B?))?(?:, network: ([\d.?]+[KMG]?B?))?\}')
trino_cpu_regex = re.compile(r'CPU:\s*([\d.]+)\s*([a-z]+)', re.IGNORECASE)
trino_scheduled_regex = re.compile(r'Scheduled:\s*([\d.]+)\s*([a-z]+)', re.IGNORECASE)
trino_blocked_regex = re.compile(r'Blocked:?\s*([\d.]+)\s*([a-z]+)', re.IGNORECASE)
trino_output_regex = re.compile(r'Output:\s*([\d,]+)\s+row(?:s)?\s*\(([\d.]+[KMG]?B)\)', re.IGNORECASE)
trino_input_regex = re.compile(r'Input:\s*([\d,]+)\s+row(?:s)?\s*\(([\d.]+[KMG]?B)\)', re.IGNORECASE)
trino_operator_regex = re.compile(r'^(\s*)([A-Za-z]+(?:\[[^\]]*\])?)')
trino_table_regex = re.compile(r'table = ([^:\s]+):([^\s]+)')
trino_columns_regex = re.compile(r'columns=\[([^\]]+)\]')
trino_layout_regex = re.compile(r'Layout: \[([^\]]+)\]')
trino_filter_regex = re.compile(r'filterPredicate = (\([^)]*(?:\([^)]*\)[^)]*)*\))')
trino_dynamic_filters_regex = re.compile(r'dynamicFilters = \{([^}]+)\}')
trino_constraints_regex = re.compile(r'constraints=\[([^\]]+)\]')
literal_regex = re.compile(r"(\'[^\']+\'::[^\'\)]+)")


class TrinoPlanOperator(AbstractPlanOperator):
    """Trinoプラン演算子を表現するクラス"""
    
    def __init__(self, plain_content=None, children=None, plan_parameters=None, plan_runtime=0):
        super().__init__(plain_content, children, plan_parameters, plan_runtime)
        self.database_type = "trino"
        # plan_parametersがNoneまたはdictの場合、SimpleNamespaceに変換
        if self.plan_parameters is None or isinstance(self.plan_parameters, dict):
            if self.plan_parameters is None:
                self.plan_parameters = SimpleNamespace()
            else:
                # dictからSimpleNamespaceに変換
                params_dict = self.plan_parameters
                self.plan_parameters = SimpleNamespace(**params_dict)
    
    def _get_param(self, key, default=None):
        """plan_parametersから値を取得（SimpleNamespace対応）"""
        if isinstance(self.plan_parameters, SimpleNamespace):
            return getattr(self.plan_parameters, key, default)
        else:
            return self.plan_parameters.get(key, default)
    
    def _has_param(self, key):
        """plan_parametersにキーが存在するかチェック（SimpleNamespace対応）"""
        if isinstance(self.plan_parameters, SimpleNamespace):
            return hasattr(self.plan_parameters, key)
        else:
            return key in self.plan_parameters
    
    def _set_param(self, key, value):
        """plan_parametersに値を設定（SimpleNamespace対応）"""
        if isinstance(self.plan_parameters, SimpleNamespace):
            setattr(self.plan_parameters, key, value)
        else:
            self.plan_parameters[key] = value
    
    def _update_params(self, params_dict):
        """plan_parametersを一括更新（SimpleNamespace対応）"""
        if isinstance(self.plan_parameters, SimpleNamespace):
            for key, value in params_dict.items():
                setattr(self.plan_parameters, key, value)
        else:
            self.plan_parameters.update(params_dict)
    
    def _normalize_trino_table_name(self, full_name):
        """
        Trinoのフルテーブル名から純粋なテーブル名を抽出
        
        入力例: iceberg:accidents.nesreca$data@4293306324159404154
        出力例: nesreca
        
        形式:
        - catalog:schema.table$data@snapshotId
        - catalog:schema.table
        - schema.table
        - table
        """
        # $data@... のようなサフィックスを削除
        if '$' in full_name:
            full_name = full_name.split('$')[0]
        
        # catalog:schema.table から table を抽出
        if ':' in full_name:
            # catalog部分を削除
            full_name = full_name.split(':', 1)[1]
        
        # schema.table から table を抽出
        if '.' in full_name:
            # 最後のドットの後ろがテーブル名
            full_name = full_name.split('.')[-1]
        
        return full_name
    
    def _parse_duration_to_ms(self, value_str, unit_str):
        """時間表現をミリ秒に変換"""
        if value_str is None or unit_str is None:
            return 0.0

        value = float(value_str)
        unit = unit_str.lower()

        multiplier = {
            'ns': 1e-6,
            'us': 1e-3,
            'µs': 1e-3,
            'ms': 1.0,
            's': 1000.0,
            'm': 60000.0,
            'min': 60000.0,
            'h': 3600000.0,
            'hr': 3600000.0
        }.get(unit)

        if multiplier is None:
            # 未知の単位の場合はそのまま返す
            return value

        return value * multiplier
    
    def parse_lines(self, alias_dict=None, parse_baseline=False, parse_join_conds=False):
        """Trinoプラン演算子の行を解析"""
        if not self.plain_content:
            return
        
        op_line = self.plain_content[0]
        
        # 演算子名を抽出（既に設定されている場合は上書きしない）
        if not self._has_param('op_name'):
            op_name_match = trino_operator_regex.search(op_line)
            if op_name_match:
                op_name = op_name_match.group(2).strip()
                # 演算子名をクリーンアップ
                if '[' in op_name:
                    op_name = op_name.split('[')[0]
                # ★ 統一インターフェースに格納
                self._set_param('op_name', op_name)
        
        # テーブル情報を抽出（既に設定されている場合は上書きしない）
        if not self._has_param('table') and 'table = ' in op_line:
            table_start = op_line.find('table = ')
            if table_start != -1:
                table_start += len('table = ')
                # カンマ、スペース、角括弧のいずれかまでを抽出
                table_end = op_line.find(',', table_start)  # カンマも区切り文字として認識
                space_pos = op_line.find(' ', table_start)
                bracket_pos = op_line.find('[', table_start)
                
                # 最も近い区切り文字を使用
                positions = [pos for pos in [table_end, space_pos, bracket_pos] if pos != -1]
                if positions:
                    table_end = min(positions)
                else:
                    table_end = len(op_line)
                
                table_full_name = op_line[table_start:table_end]
                
                # Trinoのテーブル名を正規化
                # 形式: catalog:schema.table$data@snapshotId
                # 例: iceberg:accidents.nesreca$data@4293306324159404154
                # → nesreca を抽出
                table_name = self._normalize_trino_table_name(table_full_name)
                
                self._set_param('table', table_name)
        
        # カラム情報を抽出
        if 'columns=[' in op_line:
            columns_start = op_line.find('columns=[')
            if columns_start != -1:
                columns_start += len('columns=[')
                columns_end = op_line.find(']', columns_start)
                if columns_end != -1:
                    columns_str = op_line[columns_start:columns_end]
                    columns = [col.strip() for col in columns_str.split(',')]
                    self._set_param('columns', columns)
        
        # レイアウト情報を抽出（追加の行から）
        # PostgreSQLと同様に、parse_linesの段階でoutput_columnsも生成する
        for line in self.plain_content[1:]:
            if 'Layout: [' in line:
                layout_start = line.find('Layout: [')
                if layout_start != -1:
                    layout_start += len('Layout: [')
                    layout_end = line.find(']', layout_start)
                    if layout_end != -1:
                        layout_str = line[layout_start:layout_end]
                        layout = [col.strip() for col in layout_str.split(',')]
                        self._set_param('layout', layout)
                        
                        # Layoutからoutput_columnsを生成（PostgreSQLと同様）
                        try:
                            output_columns = self.parse_output_columns(layout_str)
                            self._set_param('output_columns', output_columns)
                        except Exception as e:
                            # デバッグモード: エラーを出力
                            import os
                            if os.getenv('DEBUG_TRINO_PARSE', 'false').lower() == 'true':
                                print(f"⚠️  Warning: Failed to parse output columns from layout: {e}")
                break
        
        # 結合条件を抽出（Join演算子の場合）
        if 'criteria = ' in op_line:
            start = op_line.find('criteria = ')
            if start != -1:
                start += len('criteria = ')
                # 括弧の対応を考慮して抽出
                paren_count = 0
                end = start
                for i, char in enumerate(op_line[start:], start):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            end = i + 1
                            break
                if end > start:
                    criteria = op_line[start:end]
                    self._set_param('criteria', criteria)
        
        # Distribution情報を抽出（Join演算子の場合）
        # distribution = REPLICATED (broadcast) または PARTITIONED (hash distributed)
        if 'distribution = ' in op_line:
            start = op_line.find('distribution = ')
            if start != -1:
                start += len('distribution = ')
                # カンマまたは角括弧まで抽出
                end = start
                for i, char in enumerate(op_line[start:], start):
                    if char in [',', ']', ')']:
                        end = i
                        break
                else:
                    end = len(op_line)
                
                if end > start:
                    distribution = op_line[start:end].strip()
                    self._set_param('distribution', distribution)
        
        # フィルター条件を抽出
        if 'filterPredicate = ' in op_line:
            start = op_line.find('filterPredicate = ')
            if start != -1:
                start += len('filterPredicate = ')
                # 括弧の対応を考慮して抽出
                paren_count = 0
                end = start
                for i, char in enumerate(op_line[start:], start):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            end = i + 1
                            break
                if end > start:
                    filter_condition = op_line[start:end]
                    self._set_param('filter_condition', filter_condition)
                    # フィルター条件の詳細解析（必要に応じて）
                    try:
                        parse_tree = parse_filter(filter_condition, parse_baseline=parse_baseline)
                        self.add_filter(parse_tree, filter_type='predicate')
                    except Exception as e:
                        # デバッグモード: エラーを出力
                        import os
                        if os.getenv('DEBUG_TRINO_PARSE', 'false').lower() == 'true':
                            print(f"⚠️  Warning: Failed to parse filter condition '{filter_condition}': {e}")
                        # フィルター条件の解析に失敗した場合は、生の文字列を保存
                        pass
        
        # 動的フィルターを抽出
        dynamic_filters_match = trino_dynamic_filters_regex.search(op_line)
        if dynamic_filters_match:
            dynamic_filters_str = dynamic_filters_match.group(1)
            self._set_param('dynamic_filters', dynamic_filters_str)
            
            # dynamicFiltersを()形式に変換してフィルター解析に使用
            # {id_upravna_enota = #df_553} -> (id_upravna_enota = #df_553)
            if dynamic_filters_str.startswith('{') and dynamic_filters_str.endswith('}'):
                # 複数の動的フィルターがある場合は最初のもののみを使用
                filters = dynamic_filters_str[1:-1].split(',')
                if filters:
                    first_filter = filters[0].strip()
                    # 等号を適切な演算子に変換
                    if ' = ' in first_filter:
                        # #df_553のような動的フィルターIDを適切な形式に変換
                        filter_condition = f"({first_filter})"
                        try:
                            parse_tree = parse_filter(filter_condition, parse_baseline=parse_baseline)
                            if parse_tree:
                                self.add_filter(parse_tree, filter_type='dynamic')
                        except Exception as e:
                            # デバッグモード: エラーを出力
                            import os
                            if os.getenv('DEBUG_TRINO_PARSE', 'false').lower() == 'true':
                                print(f"⚠️  Warning: Failed to parse dynamic filter '{filter_condition}': {e}")
                            # フィルター条件の解析に失敗した場合は、生の文字列を保存
                            pass
        
        # 制約条件を抽出
        constraints_match = trino_constraints_regex.search(op_line)
        if constraints_match:
            constraints_str = constraints_match.group(1)
            constraints = [constraint.strip() for constraint in constraints_str.split(',')]
            self._set_param('constraints', constraints)
        
        # 追加の行を解析
        for line in self.plain_content[1:]:
            stripped_line = line.strip()

            # Estimates情報を抽出
            estimates_match = trino_estimates_regex.search(stripped_line)
            if estimates_match:
                # Estimatesの生の文字列を保存（reltuples推定用）
                self._set_param('estimates', stripped_line)
                rows_str = estimates_match.group(1)
                size_str = estimates_match.group(2)
                cpu_str = estimates_match.group(3) if estimates_match.group(3) else None
                memory_str = estimates_match.group(4) if estimates_match.group(4) else None
                network_str = estimates_match.group(5) if estimates_match.group(5) else None

                # 行数を数値に変換
                if rows_str == '?':
                    rows = 0
                else:
                    rows = int(rows_str.replace(',', ''))

                # サイズをバイトに変換
                if size_str == '?':
                    size_bytes = 0
                else:
                    size_bytes = self._parse_size(size_str)

                # CPU値を数値に変換
                if cpu_str == '?' or cpu_str is None:
                    est_cpu = 0
                else:
                    est_cpu = self._parse_size(cpu_str)

                # Memory値を数値に変換
                if memory_str == '?' or memory_str is None:
                    est_memory = 0
                else:
                    est_memory = self._parse_size(memory_str)

                # Network値を数値に変換
                if network_str == '?' or network_str is None:
                    est_network = 0
                else:
                    est_network = self._parse_size(network_str)

                # 行幅を計算（サイズ / 行数）
                est_width = size_bytes / rows if rows > 0 else 0

                self._update_params({
                    'est_rows': rows,
                    'est_size': size_bytes,
                    'est_width': est_width,
                    'est_cpu': est_cpu,
                    'est_memory': est_memory,
                    'est_network': est_network
                })

            # 実測CPU時間を抽出（最初に見つかった値を優先）
            if not self._has_param('act_cpu_time'):
                cpu_match = trino_cpu_regex.search(stripped_line)
                if cpu_match:
                    cpu_time = self._parse_duration_to_ms(cpu_match.group(1), cpu_match.group(2))
                    self._set_param('act_cpu_time', cpu_time)

            # Scheduled時間を抽出
            if not self._has_param('act_scheduled_time'):
                scheduled_match = trino_scheduled_regex.search(stripped_line)
                if scheduled_match:
                    scheduled_time = self._parse_duration_to_ms(scheduled_match.group(1), scheduled_match.group(2))
                    self._set_param('act_scheduled_time', scheduled_time)

            # Blocked時間を抽出
            if not self._has_param('act_blocked_time'):
                blocked_match = trino_blocked_regex.search(stripped_line)
                if blocked_match:
                    blocked_time = self._parse_duration_to_ms(blocked_match.group(1), blocked_match.group(2))
                    self._set_param('act_blocked_time', blocked_time)

            # Output情報を抽出
            if not self._has_param('act_output_rows'):
                output_match = trino_output_regex.search(stripped_line)
                if output_match:
                    output_rows = int(output_match.group(1).replace(',', ''))
                    output_size = self._parse_size(output_match.group(2))
                    self._set_param('act_output_rows', output_rows)
                    self._set_param('act_output_size', output_size)

            # Input情報を抽出
            if not self._has_param('act_input_rows'):
                input_match = trino_input_regex.search(stripped_line)
                if input_match:
                    input_rows = int(input_match.group(1).replace(',', ''))
                    input_size = self._parse_size(input_match.group(2))
                    self._set_param('act_input_rows', input_rows)
                    self._set_param('act_input_size', input_size)

            # その他のメトリクス情報
            if not self._has_param('has_metrics') and 'metrics:' in stripped_line:
                # メトリクス情報の解析（必要に応じて詳細化）
                self._set_param('has_metrics', True)

        # op_line自体に含まれるメトリクスも考慮
        if not self._has_param('act_cpu_time'):
            cpu_match = trino_cpu_regex.search(op_line)
            if cpu_match:
                cpu_time = self._parse_duration_to_ms(cpu_match.group(1), cpu_match.group(2))
                self._set_param('act_cpu_time', cpu_time)

        if not self._has_param('act_scheduled_time'):
            scheduled_match = trino_scheduled_regex.search(op_line)
            if scheduled_match:
                scheduled_time = self._parse_duration_to_ms(scheduled_match.group(1), scheduled_match.group(2))
                self._set_param('act_scheduled_time', scheduled_time)

        if not self._has_param('act_blocked_time'):
            blocked_match = trino_blocked_regex.search(op_line)
            if blocked_match:
                blocked_time = self._parse_duration_to_ms(blocked_match.group(1), blocked_match.group(2))
                self._set_param('act_blocked_time', blocked_time)

        if not self._has_param('act_output_rows'):
            output_match = trino_output_regex.search(op_line)
            if output_match:
                output_rows = int(output_match.group(1).replace(',', ''))
                output_size = self._parse_size(output_match.group(2))
                self._set_param('act_output_rows', output_rows)
                self._set_param('act_output_size', output_size)

        if not self._has_param('act_input_rows'):
            input_match = trino_input_regex.search(op_line)
            if input_match:
                input_rows = int(input_match.group(1).replace(',', ''))
                input_size = self._parse_size(input_match.group(2))
                self._set_param('act_input_rows', input_rows)
                self._set_param('act_input_size', input_size)

        if not self._has_param('has_metrics') and 'metrics:' in op_line:
            self._set_param('has_metrics', True)
        
        # Estimates情報がない場合のデフォルト値設定
        if not self._has_param('est_rows'):
            self._update_params({
                'est_rows': 0,
                'est_size': 0,
                'est_width': 0,
                'est_cpu': 0,
                'est_memory': 0,
                'est_network': 0
            })
        
        # テーブル演算子の場合、統計情報を推定（フォールバック戦略付き）
        if self._has_param('table'):
            # reltuplesの推定
            reltuples = self._get_reltuples_with_fallback()
            if reltuples is not None:
                self._set_param('reltuples', reltuples)
            
            # relpagesの推定
            relpages = self._get_relpages_with_fallback()
            if relpages is not None:
                self._set_param('relpages', relpages)
            
            # カラム統計の推定
            self._estimate_column_stats()
        
        # ★ Trino固有の特徴量を統一インターフェースにマッピング
        self._map_to_unified_interface()
    
    def _map_to_unified_interface(self):
        """Trino固有の特徴量を統一インターフェースにマッピング"""
        # est_rows → est_card
        if self._has_param('est_rows'):
            est_rows = self._get_param('est_rows')
            self._set_param('est_card', est_rows)
        
        # act_output_rows → act_card
        if self._has_param('act_output_rows'):
            act_output_rows = self._get_param('act_output_rows')
            self._set_param('act_card', act_output_rows)
        
        # est_width は既に設定済み
        if not self._has_param('est_width'):
            self._set_param('est_width', 0.0)
        
        # workers_planned は Trino では通常 0
        if not self._has_param('workers_planned'):
            self._set_param('workers_planned', 0)
        
        # est_cost: Trinoにはest_costがないが、est_cpu（Estimatesのcpu値）をest_costの代わりとして使用
        # est_cpuは推定値なので、PostgreSQLのest_costと意味的に近い
        # フォールバック: est_cpuがない場合はact_cpu_timeを使用、それもなければact_scheduled_time、それもなければ0.0
        if self._has_param('est_cpu'):
            est_cpu = self._get_param('est_cpu')
            self._set_param('est_cost', est_cpu)
        elif self._has_param('act_cpu_time'):
            act_cpu_time = self._get_param('act_cpu_time')
            self._set_param('est_cost', act_cpu_time)
        elif self._has_param('act_scheduled_time'):
            act_scheduled_time = self._get_param('act_scheduled_time')
            self._set_param('est_cost', act_scheduled_time)
        else:
            self._set_param('est_cost', 0.0)
        
        # act_time は実際のCPU時間から設定
        if self._has_param('act_cpu_time'):
            act_cpu_time = self._get_param('act_cpu_time')
            self._set_param('act_time', act_cpu_time)
        elif self._has_param('act_scheduled_time'):
            act_scheduled_time = self._get_param('act_scheduled_time')
            self._set_param('act_time', act_scheduled_time)
        else:
            self._set_param('act_time', 0)
        
        # est_startup_cost は Trino にはない
        self._set_param('est_startup_cost', None)
        
        # 子ノードのカーディナリティは後で計算される
        self._set_param('act_children_card', 1.0)
        self._set_param('est_children_card', 1.0)
    
    def _get_reltuples_with_fallback(self):
        """フォールバック戦略付きでreltuplesを取得"""
        # 戦略1: Estimatesから抽出を試行
        reltuples = self._extract_reltuples_from_estimates()
        if reltuples is not None:
            return reltuples
        
        # 戦略2: 外部統計情報から取得を試行
        reltuples = self._get_external_table_stats()
        if reltuples is not None:
            return reltuples
        
        # 戦略3: データ型から推定を試行
        reltuples = self._estimate_reltuples_from_datatype()
        if reltuples is not None:
            return reltuples
        
        # 戦略4: デフォルト値を使用
        return self._get_default_reltuples()
    
    def _extract_reltuples_from_estimates(self):
        """Estimatesからreltuplesを抽出"""
        estimates = self._get_param('estimates', '')
        if not estimates:
            return None
        
        # 最初の段階のrowsを抽出（フィルター前の推定行数）
        first_stage_match = re.search(r'rows: ([\d,]+)', estimates)
        if first_stage_match:
            reltuples = int(first_stage_match.group(1).replace(',', ''))
            return reltuples
        
        return None
    
    def _get_external_table_stats(self):
        """datasets_statisticsから統計情報を読み込んでreltuplesを取得"""
        table_name = self._get_param('table', '')
        if not table_name:
            return None
        
        # Trinoのテーブル名を正規化（例: iceberg:accidents.nesreca$data@... → nesreca）
        normalized_table_name = self._normalize_trino_table_name(table_name)
        
        # サポートされているデータセット名のリスト（datasets_statisticsディレクトリと一致）
        supported_datasets = [
            'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
            'consumer', 'credit', 'employee', 'fhnk', 'financial', 'geneea',
            'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb',
            'tournament', 'tpc_h', 'walmart'
        ]
        
        # テーブル名からスキーマ名を推測（全てのデータセット名をチェック）
        schema_name = None
        table_name_lower = table_name.lower()
        normalized_table_name_lower = normalized_table_name.lower()
        
        for dataset in supported_datasets:
            if dataset in table_name_lower or dataset in normalized_table_name_lower:
                schema_name = dataset
                break
        
        if not schema_name:
            return None
        
        # datasets_statisticsから読み込み（環境変数でパス指定可能）
        try:
            import json
            import os
            from pathlib import Path
            
            # DATASETS_STATISTICS_DIR環境変数でパスを指定可能（デフォルト: datasets_statistics）
            stats_base = os.getenv('DATASETS_STATISTICS_DIR', 'datasets_statistics')
            stats_dir = Path(stats_base) / f'iceberg_{schema_name}'
            table_stats_file = stats_dir / 'table_stats.json'
            
            if table_stats_file.exists():
                with open(table_stats_file, 'r') as f:
                    table_stats_dict = json.load(f)
                    
                # 正規化されたテーブル名で検索
                if normalized_table_name in table_stats_dict:
                    stats = table_stats_dict[normalized_table_name]
                    # reltuplesまたはrow_countを返す
                    return int(stats.get('reltuples', stats.get('row_count', 0)))
        except Exception:
            # 読み込みに失敗した場合はNoneを返す（フォールバックに進む）
            pass
        
        return None
    
    def _estimate_reltuples_from_datatype(self):
        """データ型からreltuplesを推定"""
        # カラム情報から推定
        columns = self._get_param('columns', [])
        if not columns:
            return None
        
        # データ型に基づく推定
        estimated_rows = 0
        for column in columns:
            if 'char(' in column:
                # 文字列型の場合、長さから推定
                char_match = re.search(r'char\((\d+)\)', column)
                if char_match:
                    char_length = int(char_match.group(1))
                    estimated_rows += char_length * 1000  # 仮の推定
            elif 'integer' in column or 'int4' in column:
                # 整数型の場合
                estimated_rows += 10000  # 仮の推定
            elif 'bigint' in column or 'int8' in column:
                # 長整数型の場合
                estimated_rows += 5000  # 仮の推定
        
        return estimated_rows if estimated_rows > 0 else None
    
    def _get_default_reltuples(self):
        """デフォルトのreltuples値を取得"""
        # テーブル名に基づくデフォルト値
        table_name = self._get_param('table', '').lower() if self._get_param('table') else ''
        
        # 一般的なテーブルサイズの推定
        if 'user' in table_name or 'customer' in table_name:
            return 100000  # ユーザー/顧客テーブル
        elif 'order' in table_name or 'transaction' in table_name:
            return 1000000  # 注文/取引テーブル
        elif 'log' in table_name or 'event' in table_name:
            return 10000000  # ログ/イベントテーブル
        elif 'accidents' in table_name:
            return 954036  # 実際のaccidentsテーブルのサイズ
        else:
            return 10000  # デフォルト値
    
    def _get_relpages_with_fallback(self):
        """フォールバック戦略付きでrelpagesを取得"""
        # relpagesは常に0を返す（Trinoではページ概念がないため）
        return 0
    
    
    def _estimate_column_stats(self):
        """カラム統計を推定"""
        columns = self._get_param('columns', [])
        if not columns:
            return
        
        column_stats = {}
        for column in columns:
            # カラム名を抽出
            column_name = column.split(':')[0] if ':' in column else column
            
            # avg_widthの推定
            avg_width = self._estimate_avg_width(column)
            if avg_width is not None:
                column_stats[f'{column_name}_avg_width'] = avg_width
            
            # correlationの推定
            correlation = self._estimate_correlation(column)
            if correlation is not None:
                column_stats[f'{column_name}_correlation'] = correlation
            
            # n_distinctの推定
            n_distinct = self._estimate_n_distinct(column)
            if n_distinct is not None:
                column_stats[f'{column_name}_n_distinct'] = n_distinct
            
            # null_fracの推定
            null_frac = self._estimate_null_frac(column)
            if null_frac is not None:
                column_stats[f'{column_name}_null_frac'] = null_frac
        
        if column_stats:
            self._update_params(column_stats)
    
    def _get_external_column_stats(self, table_name, column_name):
        """column_statistics.jsonから統計情報を読み込む"""
        if not table_name or not column_name:
            return None
        
        # Trinoのテーブル名を正規化（例: iceberg:accidents.nesreca$data@... → nesreca）
        normalized_table_name = self._normalize_trino_table_name(table_name)
        
        # サポートされているデータセット名のリスト
        supported_datasets = [
            'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
            'consumer', 'credit', 'employee', 'fhnk', 'financial', 'geneea',
            'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb',
            'tournament', 'tpc_h', 'walmart'
        ]
        
        # テーブル名からスキーマ名を推測
        schema_name = None
        table_name_lower = table_name.lower()
        normalized_table_name_lower = normalized_table_name.lower()
        
        for dataset in supported_datasets:
            if dataset in table_name_lower or dataset in normalized_table_name_lower:
                schema_name = dataset
                break
        
        if not schema_name:
            return None
        
        # column_statistics.jsonから読み込み
        try:
            import json
            import os
            from pathlib import Path
            
            # ZERO_SHOT_DATASETS_DIR環境変数でパスを指定（必須）
            stats_base = os.getenv('ZERO_SHOT_DATASETS_DIR')
            if not stats_base:
                return None
            
            stats_dir = Path(stats_base) / schema_name
            column_stats_file = stats_dir / 'column_statistics.json'
            
            if column_stats_file.exists():
                with open(column_stats_file, 'r') as f:
                    column_stats_dict = json.load(f)
                    
                # 正規化されたテーブル名で検索
                if normalized_table_name in column_stats_dict:
                    table_stats = column_stats_dict[normalized_table_name]
                    # カラム名で検索（大文字小文字を区別しない）
                    column_name_lower = column_name.lower()
                    for col_name, col_stats in table_stats.items():
                        if col_name.lower() == column_name_lower:
                            return col_stats
        except Exception:
            # 読み込みに失敗した場合はNoneを返す（フォールバックに進む）
            pass
        
        return None
    
    def _estimate_avg_width(self, column):
        """カラムの平均幅を推定"""
        # column_statistics.jsonから取得を試みる
        table_name = self._get_param('table', '')
        column_name = column.split(':')[0] if ':' in column else column
        
        col_stats = self._get_external_column_stats(table_name, column_name)
        if col_stats:
            # datatypeから平均幅を推定
            datatype = col_stats.get('datatype', '')
            if datatype in ['int', 'integer', 'bigint', 'smallint', 'tinyint']:
                return 8  # 整数型は8バイト
            elif datatype in ['float', 'double', 'real']:
                return 8  # 浮動小数点型は8バイト
            elif datatype == 'categorical':
                # 一意の値のリストから平均幅を推定
                unique_vals = col_stats.get('unique_vals', [])
                if unique_vals:
                    avg_len = sum(len(str(v)) for v in unique_vals[:100]) / min(len(unique_vals), 100)
                    return int(avg_len)
                return 0
            elif 'varchar' in datatype or 'char' in datatype:
                # 文字列型の場合、一意の値のリストから平均幅を推定
                unique_vals = col_stats.get('unique_vals', [])
                if unique_vals:
                    avg_len = sum(len(str(v)) for v in unique_vals[:100]) / min(len(unique_vals), 100)
                    return int(avg_len)
                return 0
        
        # フォールバック: 0を返す
        return 0
    
    def _estimate_correlation(self, column):
        """カラムの相関を推定"""
        # 相関は複数のカラム間の関係なので、単一カラムの統計からは取得できない
        # 将来的に複数カラムの相関情報が利用可能になった場合に実装
        return 0
    
    def _estimate_n_distinct(self, column):
        """カラムの異なる値の数を推定"""
        # column_statistics.jsonから取得を試みる
        table_name = self._get_param('table', '')
        column_name = column.split(':')[0] if ':' in column else column
        
        col_stats = self._get_external_column_stats(table_name, column_name)
        if col_stats:
            num_unique = col_stats.get('num_unique')
            if num_unique is not None:
                return int(num_unique)
        
        # フォールバック: 0を返す
        return 0
    
    def _estimate_null_frac(self, column):
        """カラムのNULL値の割合を推定"""
        # column_statistics.jsonから取得を試みる
        table_name = self._get_param('table', '')
        column_name = column.split(':')[0] if ':' in column else column
        
        col_stats = self._get_external_column_stats(table_name, column_name)
        if col_stats:
            nan_ratio = col_stats.get('nan_ratio')
            if nan_ratio is not None:
                return float(nan_ratio)
        
        # フォールバック: 0を返す
        return 0
    
    def _parse_size(self, size_str):
        """サイズ文字列をバイト数に変換"""
        if not size_str or size_str == '?':
            return 0
        
        size_str = size_str.upper()
        if size_str.endswith('B'):
            size_str = size_str[:-1]
        
        if size_str.endswith('K'):
            return float(size_str[:-1]) * 1024
        elif size_str.endswith('M'):
            return float(size_str[:-1]) * 1024 * 1024
        elif size_str.endswith('G'):
            return float(size_str[:-1]) * 1024 * 1024 * 1024
        else:
            return float(size_str)
    
    def parse_output_columns(self, layout_str):
        """出力カラムを解析"""
        output_columns = []
        for col in layout_str.split(','):
            col = col.strip()
            
            # 集約関数の検出（TrinoのLayout形式に対応）
            agg = None
            col_name = col.split(':')[0].strip() if ':' in col else col.strip()
            
            # 関数形式の検出（例：count(...)）
            if 'count(' in col.lower():
                agg = Aggregator.COUNT
            elif 'sum(' in col.lower():
                agg = Aggregator.SUM
            elif 'avg(' in col.lower():
                agg = Aggregator.AVG
            elif 'min(' in col.lower():
                agg = ExtendedAggregator.MIN
            elif 'max(' in col.lower():
                agg = ExtendedAggregator.MAX
            # TrinoのLayout形式の検出（例：count:bigint, avg:double）
            elif col_name.lower() == 'count':
                agg = Aggregator.COUNT
            elif col_name.lower() == 'sum':
                agg = Aggregator.SUM
            elif col_name.lower() == 'avg':
                agg = Aggregator.AVG
            elif col_name.lower() == 'min':
                agg = ExtendedAggregator.MIN
            elif col_name.lower() == 'max':
                agg = ExtendedAggregator.MAX
            
            # カラム名を抽出
            columns = []
            if ':' in col:
                col_name = col.split(':')[0].strip()
                if '.' in col_name:
                    columns.append(tuple(col_name.split('.')))
                else:
                    columns.append((col_name,))
            
            output_columns.append(dict(aggregation=str(agg) if agg else None, columns=columns))
        
        return output_columns
    
    def add_filter(self, parse_tree, filter_type='predicate'):
        """フィルター条件を追加（種類を区別）"""
        if parse_tree is not None:
            # フィルターの種類を記録
            parse_tree.filter_type = filter_type
            
            existing_filter = self._get_param('filter_columns')
            if existing_filter is not None:
                parse_tree = PredicateNode(None, [existing_filter, parse_tree])
                parse_tree.operator = LogicalOperator.AND
            
            self._set_param('filter_columns', parse_tree)
    
    def parse_columns_bottom_up(self, column_id_mapping, partial_column_name_mapping, table_id_mapping, alias_dict,
                                table_samples=None, col_stats=None):
        """カラム情報を統計情報と照合"""
        if alias_dict is None:
            alias_dict = dict()
        
        # 現在のノードで考慮されるテーブルを追跡
        node_tables = set()
        table = self._get_param('table')
        if table is not None:
            node_tables.add(table)
        
        # 子ノードからテーブル情報を収集
        for c in self.children:
            node_tables.update(
                c.parse_columns_bottom_up(column_id_mapping, partial_column_name_mapping, table_id_mapping, alias_dict,
                                         table_samples=table_samples, col_stats=col_stats))
        
        # 子ノードのカーディナリティを計算
        self._set_param('act_children_card', child_prod(self, 'act_output_rows'))
        self._set_param('est_children_card', child_prod(self, 'est_rows'))
        
        # workers_plannedのデフォルト値を設定（Trinoでは並列度の概念が異なる）
        if not self._has_param('workers_planned'):
            self._set_param('workers_planned', 1)
        
        # 出力カラムの処理: parse_linesで既に生成されたoutput_columnsのカラムIDへの変換のみを行う
        # PostgreSQLと同じロジック
        output_columns = self._get_param('output_columns')
        if output_columns is not None:
            for output_column in output_columns:
                col_ids = []
                for c in output_column['columns']:
                    try:
                        c_id = self.lookup_column_id(c, column_id_mapping, node_tables, partial_column_name_mapping, alias_dict)
                        col_ids.append(c_id)
                    except Exception as e:
                        # subgbは特殊なケースなのでスキップ
                        if len(c) > 0 and c[0] != 'subgb':
                            # デバッグモード: エラーを出力
                            import os
                            if os.getenv('DEBUG_TRINO_PARSE', 'false').lower() == 'true':
                                print(f"⚠️  Warning: Could not find unique table for column {c}: {e}")
                            # エラーを記録するが、処理は続行（PostgreSQLと異なり、例外を投げない）
                
                output_column['columns'] = col_ids
        
        # フィルターカラムの処理
        filter_columns = self._get_param('filter_columns')
        if filter_columns is not None:
            # 既に辞書形式の場合でも、カラムIDに変換する必要がある
            if isinstance(filter_columns, dict):
                # 辞書形式のfilter_columns内のcolumnをカラムIDに変換
                def convert_dict_filter_columns(fc_dict):
                    if not isinstance(fc_dict, dict):
                        return
                    
                    column = fc_dict.get('column')
                    if column is not None and isinstance(column, (tuple, list)):
                        if len(column) == 1:
                            col_name = str(column[0]).strip('"')
                            # Trinoエイリアス（_数字）を削除
                            import re
                            col_name_without_suffix = re.sub(r'_\d+$', '', col_name)
                            
                            # テーブル名を推測してカラムIDを取得
                            for table_name in partial_column_name_mapping.get(col_name, set()):
                                if (table_name, col_name) in column_id_mapping:
                                    fc_dict['column'] = column_id_mapping[(table_name, col_name)]
                                    break
                            else:
                                # サフィックス削除版で再試行
                                for table_name in partial_column_name_mapping.get(col_name_without_suffix, set()):
                                    if (table_name, col_name_without_suffix) in column_id_mapping:
                                        fc_dict['column'] = column_id_mapping[(table_name, col_name_without_suffix)]
                                        break
                        elif len(column) == 2:
                            table_name, col_name = column
                            table_name = str(table_name).strip('"')
                            col_name = str(col_name).strip('"')
                            if (table_name, col_name) in column_id_mapping:
                                fc_dict['column'] = column_id_mapping[(table_name, col_name)]
                    
                    # 子ノードを再帰的に変換
                    for child in fc_dict.get('children', []):
                        convert_dict_filter_columns(child)
                
                convert_dict_filter_columns(filter_columns)
            else:
                # PredicateNode形式の場合のみ処理
                # lookup_columnsを呼び出してカラムIDを設定（失敗しても続行）
                try:
                    filter_columns.lookup_columns(self, column_id_mapping=column_id_mapping, node_tables=node_tables,
                                                partial_column_name_mapping=partial_column_name_mapping,
                                                alias_dict=alias_dict)
                except Exception:
                    # lookup_columnsが失敗しても続行（columnがtupleのままでもsample_vec生成は可能）
                    pass
                
                # sample_vecを生成（to_dict()の前に実行）
                # lookup_columnsの後、filter_columns.columnはカラムID（整数）またはtupleのまま
                if table_samples is not None and col_stats is not None:
                    try:
                        from models.workload_driven.preprocessing.sample_vectors_trino import construct_filter_sample
                        
                        # filter_columnsがPredicateNode形式の場合のみsample_vecを生成
                        if hasattr(filter_columns, 'operator') and hasattr(filter_columns, 'column'):
                            sample_vec_series = construct_filter_sample(table_samples, col_stats, filter_columns)
                            # バイナリベクトルに変換
                            sample_vec = [1 if s_i else 0 for s_i in sample_vec_series]
                            
                            # 1000次元に調整
                            if len(sample_vec) < 1000:
                                sample_vec = sample_vec + [0] * (1000 - len(sample_vec))
                            elif len(sample_vec) > 1000:
                                sample_vec = sample_vec[:1000]
                            
                            self._set_param('sample_vec', sample_vec)
                        else:
                            # PredicateNode形式でない場合は空のsample_vecを設定
                            self._set_param('sample_vec', [0] * 1000)
                    except Exception as e:
                        # sample_vec生成に失敗した場合は空のsample_vecを設定
                        # デバッグ用: エラーを出力（最初の数回のみ）
                        if not hasattr(self, '_sample_vec_error_count'):
                            self._sample_vec_error_count = 0
                        if self._sample_vec_error_count < 3:
                            import traceback
                            op_name = self._get_param('op_name', 'Unknown')
                            print(f"Warning: Failed to generate sample_vec for {op_name}: {e}")
                            traceback.print_exc()
                            self._sample_vec_error_count += 1
                        self._set_param('sample_vec', [0] * 1000)
                else:
                    # テーブルサンプルがない場合は空のsample_vecを設定
                    self._set_param('sample_vec', [0] * 1000)
                
                # to_dict()を呼び出して辞書形式に変換
                self._set_param('filter_columns', filter_columns.to_dict())
        
        # テーブルIDに変換
        table = self._get_param('table')
        if table is not None:
            # QueryFormer互換性のため、'tablename'フィールドも設定
            if not self._has_param('tablename'):
                self._set_param('tablename', table)
            
            # table_id_mappingが空でない場合のみID変換を行う
            if table_id_mapping:
                if table in table_id_mapping:
                    self._set_param('table', table_id_mapping[table])
                else:
                    # マッピングにない場合は削除（QueryFormer互換性のため）
                    if isinstance(self.plan_parameters, SimpleNamespace):
                        delattr(self.plan_parameters, 'table')
                    else:
                        del self.plan_parameters['table']
            # table_id_mappingが空の場合は、文字列テーブル名をそのまま保持
        
        return node_tables
    
    def lookup_column_id(self, c, column_id_mapping, node_tables, partial_column_name_mapping, alias_dict):
        """カラムIDを検索"""
        assert isinstance(c, tuple)
        
        # 完全修飾名の場合
        if len(c) == 2:
            table = c[0].strip('"')
            column = c[1].strip('"')
            
            if table in alias_dict:
                table = alias_dict[table]
                if table is None:
                    return self.lookup_column_id((c[1],), column_id_mapping, node_tables, partial_column_name_mapping, alias_dict)
        
        # カラム名のみの場合
        elif len(c) == 1:
            column = c[0].strip('"')
            potential_tables = partial_column_name_mapping[column].intersection(node_tables)
            assert len(potential_tables) == 1, f"Did not find unique table for column {column} (node_tables: {node_tables})"
            table = list(potential_tables)[0]
        else:
            raise NotImplementedError
        
        col_id = column_id_mapping[(table, column)]
        return col_id
    
    def merge_recursively(self, node):
        """別のノードと再帰的にマージ"""
        self_op_name = self._get_param('op_name')
        node_op_name = node._get_param('op_name') if hasattr(node, '_get_param') else getattr(node.plan_parameters, 'op_name', None) if isinstance(node.plan_parameters, SimpleNamespace) else node.plan_parameters.get('op_name')
        assert self_op_name == node_op_name
        assert len(self.children) == len(node.children)
        
        # node.plan_parametersから全ての属性をコピー
        if isinstance(node.plan_parameters, SimpleNamespace):
            for key in vars(node.plan_parameters):
                self._set_param(key, getattr(node.plan_parameters, key))
        else:
            self._update_params(node.plan_parameters)
        
        for self_c, c in zip(self.children, node.children):
            self_c.merge_recursively(c)
    
    def parse_lines_recursively(self, alias_dict=None, parse_baseline=False, parse_join_conds=False):
        """再帰的に行を解析"""
        self.parse_lines(alias_dict=alias_dict, parse_baseline=parse_baseline, parse_join_conds=parse_join_conds)
        for c in self.children:
            c.parse_lines_recursively(alias_dict=alias_dict, parse_baseline=parse_baseline, parse_join_conds=parse_join_conds)
    
    def min_card(self):
        """最小カーディナリティを取得"""
        # SimpleNamespace 統一後は getattr を使用
        act_card = getattr(self.plan_parameters, 'act_output_rows', None)
        if act_card is None:
            act_card = math.inf
        
        for c in self.children:
            child_min_card = c.min_card()
            if child_min_card < act_card:
                act_card = child_min_card
        
        return act_card
    
    def recursive_str(self, pre):
        """再帰的に文字列表現を生成"""
        pre_whitespaces = ''.join(['\t' for _ in range(pre)])
        current_string = pre_whitespaces + str(self.plan_parameters)
        node_strings = [current_string]
        
        for c in self.children:
            node_strings += c.recursive_str(pre + 1)
        
        return node_strings
    
    def __str__(self):
        """文字列表現"""
        rec_str = self.recursive_str(0)
        return '\n'.join(rec_str)
