import collections
import re
from typing import List, Any

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # tqdmが利用できない場合は通常のrangeを使用
    def tqdm(iterable, *args, **kwargs):
        return iterable

from cross_db_benchmark.benchmark_tools.abstract.plan_parser import AbstractPlanParser
from cross_db_benchmark.benchmark_tools.generate_workload import LogicalOperator
from cross_db_benchmark.benchmark_tools.trino.plan_operator import TrinoPlanOperator
from cross_db_benchmark.benchmark_tools.trino.utils import plan_statistics

# Trino特有の正規表現パターン
# Queuedの単位はus/μs/msのいずれか、Executionの単位はms/s/m/us/μsのいずれか
trino_timing_regex = re.compile(r'Queued: ([\d.]+)(?:us|μs|ms)?, Analysis: ([\d.]+)ms, Planning: ([\d.]+)ms, Execution: ([\d.]+)(ms|s|m|us|μs)?')
trino_fragment_regex = re.compile(r'Fragment (\d+) \[(\w+)\]')
trino_cpu_regex = re.compile(r'CPU: ([\d.]+)ms')
trino_scheduled_regex = re.compile(r'Scheduled: ([\d.]+)ms')
trino_blocked_regex = re.compile(r'Blocked ([\d.]+)s \(Input: ([\d.]+)s, Output: ([\d.]+)ns\)')
trino_input_regex = re.compile(r'Input: ([\d,]+) rows \(([\d.]+[KMG]?B)\)')
trino_output_regex = re.compile(r'Output: ([\d,]+) rows \(([\d.]+[KMG]?B)\)')
trino_estimates_regex = re.compile(r'Estimates: \{rows: ([\d,?]+) \(([\d.]+[KMG]?B)\)(?:, cpu: ([\d.]+[KMG]?)?)?(?:, memory: ([\d.]+[KMG]?B))?(?:, network: ([\d.]+[KMG]?B))?\}')


class TrinoPlanParser(AbstractPlanParser):
    """Trinoプランパーサー"""
    
    def __init__(self):
        super().__init__(database_type="trino")
    
    def parse_raw_plan(self, plan_text, analyze=True, parse=True, **kwargs):
        """
        Parse raw Trino EXPLAIN ANALYZE text output.
        
        This is the unified interface implementation for Trino.
        Wraps parse_trino_raw_plan_v2 to provide the standard interface.
        
        Args:
            plan_text: Raw Trino EXPLAIN ANALYZE output as string
            analyze: Whether to extract execution statistics (default: True)
            parse: Whether to parse the plan structure (default: True)
            **kwargs: Additional Trino-specific parameters
        
        Returns:
            tuple: (root_operator, execution_time, planning_time)
        """
        return parse_trino_raw_plan_v2(plan_text, analyze=analyze, parse=parse)
    
    def parse_plans(self, run_stats, min_runtime=100, max_runtime=30000, parse_baseline=False, cap_queries=None,
                   parse_join_conds=False, include_zero_card=False, explain_only=False, **kwargs):
        """Trinoプランを一括解析"""
        
        # 統計情報の初期化
        parsed_plans = []
        avg_runtimes = []
        no_tables = []
        no_filters = []
        op_perc = collections.defaultdict(int)
        database_stats = {}
        
        # クエリ数の制限
        query_list = run_stats.query_list
        if cap_queries:
            query_list = query_list[:cap_queries]
        
        for q in tqdm(query_list):
            # タイムアウトクエリをスキップ
            if hasattr(q, 'timeout') and q.timeout:
                continue
            
            # プランが存在しない場合はスキップ
            if not hasattr(q, 'verbose_plan') or not q.verbose_plan:
                continue
            
            # 実行時間の確認
            if hasattr(q, 'execution_time') and q.execution_time < min_runtime:
                continue
            
            if hasattr(q, 'execution_time') and q.execution_time > max_runtime:
                continue
            
            try:
                # プランテキストを結合
                plan_lines = []
                for l in q.verbose_plan:
                    if isinstance(l, (list, tuple)) and len(l) > 0:
                        plan_lines.append(l[0])
                    elif isinstance(l, str):
                        plan_lines.append(l)
                    else:
                        plan_lines.append('')
                
                plan_text = '\n'.join(plan_lines)
                
                # 統一インターフェースを使用
                root_operator, execution_time, planning_time = self.parse_raw_plan(
                    plan_text, analyze=True, parse=True
                )
                
                if root_operator is None:
                    continue
                
                # プラン統計を計算
                stats = plan_statistics(root_operator)
                
                # プラン情報を構築
                plan_info = {
                    'root_operator': root_operator,
                    'execution_time': execution_time,
                    'planning_time': planning_time,
                    'stats': stats,
                    'query_id': getattr(q, 'query_id', len(parsed_plans)),
                    'sql': getattr(q, 'sql', ''),
                }
                
                # TrinoPlanOperatorオブジェクトにplan_runtimeを設定
                root_operator.plan_runtime = execution_time
                root_operator.database_id = getattr(q, 'database_id', 'unknown')
                
                parsed_plans.append(root_operator)
                avg_runtimes.append(execution_time)
                
                # 統計情報を更新
                if stats['no_tables'] == 0:
                    no_tables.append(len(parsed_plans) - 1)
                
                if stats['no_filters'] == 0:
                    no_filters.append(len(parsed_plans) - 1)
                
                # 演算子統計を更新
                for op_name in stats['operators']:
                    op_perc[op_name] += 1
                    
            except Exception as e:
                print(f"Error parsing query {getattr(q, 'query_id', 'unknown')}: {e}")
                continue
        
        # 結果を構築（統一フォーマット）
        parsed_runs = {
            'parsed_plans': parsed_plans,
            'database_stats': getattr(run_stats, 'database_stats', {}),
            'run_kwargs': getattr(run_stats, 'run_kwargs', {})
        }
        
        # 統計情報を構築
        stats = {
            'runtimes': str(avg_runtimes),
            'no_tables': str(no_tables),
            'no_filters': str(no_filters),
            'total_plans': len(parsed_plans),
            'avg_runtime': np.mean(avg_runtimes) if avg_runtimes else 0,
            'min_runtime': np.min(avg_runtimes) if avg_runtimes else 0,
            'max_runtime': np.max(avg_runtimes) if avg_runtimes else 0,
            'no_tables_count': len(no_tables),
            'no_filters_count': len(no_filters),
            'operator_percentages': dict(op_perc)
        }
        
        return parsed_runs, stats
    
    def parse_single_plan(self, plan_text, analyze=True, parse=True, **kwargs):
        """単一のTrinoプランを解析"""
        try:
            root_operator, execution_time, planning_time = self.parse_raw_plan(
                plan_text, analyze=analyze, parse=parse
            )
            return root_operator
        except Exception as e:
            print(f"Error parsing single plan: {e}")
            return None
    
    def parse_explain_analyze_file(
        self,
        file_path: str,
        min_runtime: float = 100,
        max_runtime: float = 30000,
        **kwargs
    ) -> tuple[List[Any], List[float]]:
        """
        Parse EXPLAIN ANALYZE results from a text file (Trino-specific implementation).
        
        This method overrides the base implementation to ensure join_conds are set.
        """
        parsed_plans, runtimes = super().parse_explain_analyze_file(
            file_path, min_runtime=min_runtime, max_runtime=max_runtime, **kwargs
        )
        
        # 各プランにjoin_condsとplan_runtimeを設定
        for i, plan in enumerate(parsed_plans):
            if not hasattr(plan, 'join_conds') or plan.join_conds is None:
                join_conds = extract_join_conditions_trino(plan)
                plan.join_conds = join_conds
            
            # plan_runtimeを設定（runtimesリストから）
            if i < len(runtimes):
                plan.plan_runtime = runtimes[i]
        
        # テーブルサンプルとカラム統計が提供されている場合は、sample_vecを生成
        table_samples = kwargs.get('table_samples')
        col_stats = kwargs.get('col_stats')
        
        # カラムIDマッピングも提供されている場合は、カラムIDに変換
        column_id_mapping = kwargs.get('column_id_mapping')
        partial_column_name_mapping = kwargs.get('partial_column_name_mapping')
        table_id_mapping = kwargs.get('table_id_mapping')
        
        if table_samples is not None and col_stats is not None:
            # augment_sample関数を直接使用してsample_vecを生成
            from models.workload_driven.preprocessing.sample_vectors_trino import augment_sample
            
            for plan in parsed_plans:
                try:
                    # プランツリーを再帰的に走査してsample_vecを生成
                    augment_sample(table_samples, col_stats, plan)
                except Exception as e:
                    # エラーが発生しても続行（sample_vecは空のまま）
                    # デバッグ用: 最初の数回のみエラーを出力
                    if not hasattr(TrinoPlanParser, '_sample_vec_error_count'):
                        TrinoPlanParser._sample_vec_error_count = 0
                    if TrinoPlanParser._sample_vec_error_count < 3:
                        print(f"Warning: Failed to generate sample_vec in parse_explain_analyze_file: {e}")
                        import traceback
                        traceback.print_exc()
                        TrinoPlanParser._sample_vec_error_count += 1
        
        # カラムIDマッピングが提供されている場合は、カラムIDに変換
        if column_id_mapping is not None and partial_column_name_mapping is not None:
            for plan in parsed_plans:
                try:
                    # parse_columns_bottom_upを呼び出してカラムIDに変換
                    plan.parse_columns_bottom_up(
                        column_id_mapping=column_id_mapping,
                        partial_column_name_mapping=partial_column_name_mapping,
                        table_id_mapping=table_id_mapping if table_id_mapping else {},
                        alias_dict={},
                        table_samples=None,  # sample_vecは既に生成済み
                        col_stats=None
                    )
                except Exception:
                    # エラーが発生しても続行
                    pass
        
        return parsed_plans, runtimes


def parse_trino_plan_simple(plan_text):
    """Trinoプランを簡易的に解析"""
    lines = plan_text.strip().split('\n')
    
    # タイミング情報を抽出
    queued_time = 0
    analysis_time = 0
    planning_time = 0
    execution_time = 0
    
    for line in lines:
        timing_match = trino_timing_regex.search(line)
        if timing_match:
            queued_value = float(timing_match.group(1))
            # Queuedの単位を確認（正規表現ではキャプチャしていないので、msかus/μsかを判定）
            queued_str = timing_match.group(0)
            if 'ms' in queued_str:
                queued_time = queued_value  # ms
            else:
                queued_time = queued_value / 1000  # us/μs to ms
            
            analysis_time = float(timing_match.group(2))
            planning_time = float(timing_match.group(3))
            execution_time = float(timing_match.group(4))
            execution_unit = timing_match.group(5) if timing_match.group(5) else 'ms'
            
            # 実行時間の単位をミリ秒に統一
            if execution_unit == 's':
                execution_time = execution_time * 1000  # s to ms
            elif execution_unit == 'm':
                execution_time = execution_time * 60000  # m to ms
            elif execution_unit in ('us', 'μs'):
                execution_time = execution_time / 1000  # us/μs to ms
            # execution_unit == 'ms' の場合はそのまま
            
            break
    
    # Fragmentを検出
    fragments = []
    current_fragment = None
    
    for line in lines:
        fragment_match = trino_fragment_regex.search(line)
        if fragment_match:
            if current_fragment:
                fragments.append(current_fragment)
            current_fragment = {
                'id': fragment_match.group(1),  # 文字列として保持
                'type': fragment_match.group(2),
                'lines': [line]
            }
        elif current_fragment:
            current_fragment['lines'].append(line)
    
    if current_fragment:
        fragments.append(current_fragment)
    
    # すべてのFragmentから演算子を抽出
    all_operators = []
    for fragment in fragments:
        fragment_operators = extract_operators_from_fragment(fragment)
        # Fragment情報を演算子に追加
        for operator in fragment_operators:
            operator['fragment_id'] = fragment['id']
            operator['fragment_type'] = fragment['type']
            
            # TrinoPlanOperatorを作成して詳細解析
            from cross_db_benchmark.benchmark_tools.trino.plan_operator import TrinoPlanOperator
            trino_operator = TrinoPlanOperator(operator['lines'])
            trino_operator.plan_parameters['op_name'] = operator['name']
            trino_operator.plan_parameters['fragment_id'] = fragment['id']
            trino_operator.plan_parameters['fragment_type'] = fragment['type']
            trino_operator.parse_lines_recursively()
            
            # 解析された情報を演算子に追加
            operator.update(trino_operator.plan_parameters)
            
        all_operators.extend(fragment_operators)
    
    return all_operators, execution_time, planning_time


def extract_operators_from_fragment(fragment):
    """Fragmentから演算子を抽出"""
    operators = []
    current_operator = None
    current_lines = []
    
    for line in fragment['lines']:
        stripped = line.strip()
        
        # 演算子の開始を検出（より正確な条件）
        if is_operator_line(stripped):
            # 前の演算子を保存
            if current_operator:
                current_operator['lines'] = current_lines
                operators.append(current_operator)
            
            # 新しい演算子を開始
            depth = count_indent_depth(line)
            current_operator = {
                'name': extract_operator_name(stripped),
                'depth': depth,
                'lines': []
            }
            current_lines = [line]
        else:
            # 演算子の詳細情報
            if current_operator:
                current_lines.append(line)
    
    # 最後の演算子を保存
    if current_operator:
        current_operator['lines'] = current_lines
        operators.append(current_operator)
    return operators


def is_operator_line(line):
    """行が演算子の開始行かどうかを判定"""
    if not line:
        return False
    
    # 明らかに演算子でない行を除外
    exclude_patterns = [
        'Fragment', 'CPU:', 'Scheduled:', 'Blocked', 'Peak Memory:',
        'Output buffer', 'Task output', 'Task input', 'Output layout:',
        'Output partitioning:', 'metrics:', 'Input avg.:', 'Input std.dev.:',
        'Distribution:', 'dynamicFilterAssignments', 'Dynamic filters:',
        'Left (probe)', 'Right (build)', 'Reorder joins cost'
    ]
    
    for pattern in exclude_patterns:
        if line.startswith(pattern):
            return False
    
    # 演算子のパターンを検出
    operator_patterns = [
        r'^\w+\[',  # Aggregate[type = FINAL]
        r'^\w+\s*$',  # TableScan
        r'^└─\s*\w+',  # └─ LocalExchange
        r'^├─\s*\w+',  # ├─ ScanFilter
        r'^\w+\[.*\]',  # ScanFilterProject[table = ...]
    ]
    
    for pattern in operator_patterns:
        if re.match(pattern, line):
            return True
    
    return False


def extract_operator_name(line):
    """行から演算子名を抽出"""
    # 行をクリーンアップ
    line = line.strip()
    
    # 演算子名のパターンを検出
    patterns = [
        r'^└─\s*(\w+)',  # └─ LocalExchange
        r'^├─\s*(\w+)',  # ├─ ScanFilter
        r'^(\w+)\[',     # Aggregate[type = FINAL]
        r'^(\w+)\s*$',   # TableScan
        r'^(\w+)\[.*\]', # ScanFilterProject[table = ...]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    
    return 'Unknown'


def count_indent_depth(line):
    """行のインデント深度を計算"""
    # スペースとタブのインデントを計算
    space_indent = len(line) - len(line.lstrip())
    
    # Trinoの階層記号（└─, ├─）を考慮
    if '└─' in line:
        # └─ は子ノードを示す（親より1レベル深い）
        return space_indent + 1
    elif '├─' in line:
        # ├─ は兄弟ノードを示す（同じレベル）
        return space_indent
    else:
        # 通常のインデント
        return space_indent


def create_trino_plan_operator(operator_info):
    """演算子情報からTrinoPlanOperatorを作成"""
    operator = TrinoPlanOperator(operator_info['lines'])
    operator.plan_parameters['op_name'] = operator_info['name']
    operator.plan_parameters['depth'] = operator_info['depth']
    # Fragment情報を設定
    if 'fragment_id' in operator_info:
        operator.plan_parameters['fragment_id'] = operator_info['fragment_id']
    if 'fragment_type' in operator_info:
        operator.plan_parameters['fragment_type'] = operator_info['fragment_type']
    return operator


def extract_join_conditions_trino(root_operator):
    """Trinoプランから結合条件を抽出"""
    join_conds = []
    
    def traverse_plan(node):
        """プランを再帰的に走査して結合条件を収集"""
        # 現在のノードがJoin演算子かチェック
        op_name = node.plan_parameters.get('op_name', '').lower()
        
        # Join演算子の検出（InnerJoin, LeftJoin, RightJoin, FullJoin, CrossJoinなど）
        if 'join' in op_name:
            # criteriaパラメータから結合条件を抽出
            # InnerJoin[criteria = (id_upravna_enota = upravna_enota_4), ...]のような形式
            criteria = node.plan_parameters.get('criteria')
            if criteria:
                # 括弧を除去して結合条件を正規化
                join_cond = criteria.strip('()')
                if join_cond:
                    join_conds.append(join_cond)
            else:
                # filterPredicateから結合条件を推測（結合条件らしいフィルタを探す）
                filter_condition = node.plan_parameters.get('filter_condition')
                if filter_condition:
                    # 等号を含むフィルタを結合条件として扱う
                    if '=' in filter_condition and '.' in filter_condition:
                        # テーブル名.カラム名の形式を検出
                        join_cond = filter_condition.strip('()')
                        if join_cond:
                            join_conds.append(join_cond)
        
        # 子ノードを再帰的に走査
        for child in node.children:
            traverse_plan(child)
    
    traverse_plan(root_operator)
    return join_conds


def parse_trino_raw_plan_v2(plan_text, analyze=True, parse=True):
    """Trinoの生プランテキストを解析（改良版）"""
    if not parse:
        # タイミング情報のみ抽出
        lines = plan_text.strip().split('\n')
        execution_time = 0
        planning_time = 0
        
        for line in lines:
            timing_match = trino_timing_regex.search(line)
            if timing_match:
                execution_time = float(timing_match.group(4))
                planning_time = float(timing_match.group(3))
                break
        
        return None, execution_time, planning_time
    
    # プランを解析
    all_operators, execution_time, planning_time = parse_trino_plan_simple(plan_text)
    
    if not all_operators:
        return None, execution_time, planning_time
    
    # Fragment 1の演算子をルートとして、全Fragmentの演算子を統合
    root_operators = [op for op in all_operators if op.get('fragment_id') == '1']
    
    if not root_operators:
        return None, execution_time, planning_time
    
    # 全Fragmentの演算子を統合（テーブル情報を含む）
    all_fragment_operators = all_operators
    
    # 演算子を階層構造に変換（全Fragmentの演算子を含む）
    root_operator = build_hierarchy(root_operators, all_fragment_operators)
    
    # 結合条件を抽出してroot_operatorに設定
    join_conds = extract_join_conditions_trino(root_operator)
    root_operator.join_conds = join_conds
    
    return root_operator, execution_time, planning_time


def parse_trino_plans(run_stats, min_runtime=100, max_runtime=30000, parse_baseline=False, cap_queries=None,
                      parse_join_conds=False, include_zero_card=False, explain_only=False):
    """Trinoプランを一括解析"""
    
    # 統計情報の初期化
    parsed_plans = []
    avg_runtimes = []
    no_tables = []
    no_filters = []
    op_perc = collections.defaultdict(int)
    database_stats = {}
    
    # クエリ数の制限
    query_list = run_stats.query_list
    if cap_queries:
        query_list = query_list[:cap_queries]
    
    for q in tqdm(query_list):
        # タイムアウトクエリをスキップ
        if hasattr(q, 'timeout') and q.timeout:
            continue
        
        # プランが存在しない場合はスキップ
        if not hasattr(q, 'verbose_plan') or not q.verbose_plan:
            continue
        
        # 実行時間の確認
        if hasattr(q, 'execution_time') and q.execution_time < min_runtime:
            continue
        
        if hasattr(q, 'execution_time') and q.execution_time > max_runtime:
            continue
        
        try:
            # プランテキストを結合
            plan_lines = []
            for l in q.verbose_plan:
                if isinstance(l, (list, tuple)) and len(l) > 0:
                    plan_lines.append(l[0])
                elif isinstance(l, str):
                    plan_lines.append(l)
                else:
                    plan_lines.append('')
            
            plan_text = '\n'.join(plan_lines)
            
            # プランを解析
            root_operator, execution_time, planning_time = parse_trino_raw_plan_v2(
                plan_text, analyze=True, parse=True
            )
            
            if root_operator is None:
                continue
            
            # プラン統計を計算
            stats = plan_statistics(root_operator)
            
            # プラン情報を構築
            plan_info = {
                'root_operator': root_operator,
                'execution_time': execution_time,
                'planning_time': planning_time,
                'stats': stats,
                'query_id': getattr(q, 'query_id', len(parsed_plans)),
                'sql': getattr(q, 'sql', ''),
            }
            
            # TrinoPlanOperatorオブジェクトにplan_runtimeを設定
            root_operator.plan_runtime = execution_time
            root_operator.database_id = getattr(q, 'database_id', 'unknown')
            
            parsed_plans.append(root_operator)
            avg_runtimes.append(execution_time)
            
            # 統計情報を更新
            if stats['no_tables'] == 0:
                no_tables.append(len(parsed_plans) - 1)
            
            if stats['no_filters'] == 0:
                no_filters.append(len(parsed_plans) - 1)
            
            # 演算子統計を更新
            for op_name in stats['operators']:
                op_perc[op_name] += 1
                
        except Exception as e:
            print(f"Error parsing query {getattr(q, 'query_id', 'unknown')}: {e}")
            continue
    
    # 結果を構築
    parsed_runs = {
        'parsed_plans': parsed_plans,
        'avg_runtimes': avg_runtimes,
        'no_tables': no_tables,
        'no_filters': no_filters,
        'op_perc': dict(op_perc),
        'database': 'trino'
    }
    
    # 統計情報を構築
    stats = {
        'total_plans': len(parsed_plans),
        'avg_runtime': np.mean(avg_runtimes) if avg_runtimes else 0,
        'min_runtime': np.min(avg_runtimes) if avg_runtimes else 0,
        'max_runtime': np.max(avg_runtimes) if avg_runtimes else 0,
        'no_tables_count': len(no_tables),
        'no_filters_count': len(no_filters),
        'operator_percentages': dict(op_perc)
    }
    
    return parsed_runs, stats


def build_hierarchy(operators, all_fragment_operators=None):
    """演算子のリストを階層構造に変換（全Fragmentの演算子を含む）"""
    if not operators:
        return None
    
    # ルート演算子を作成（最初の演算子）
    root_operator = create_trino_plan_operator(operators[0])
    
    # 子ノードを再帰的に構築
    build_children(root_operator, operators, 1, operators[0]['depth'])
    
    # 全Fragmentの演算子を統合（テーブル情報を含む）
    if all_fragment_operators:
        integrate_all_fragments(root_operator, all_fragment_operators)
    
    # 演算子の詳細解析
    root_operator.parse_lines_recursively()
    
    # 子ノードのカーディナリティを計算とoutput_columnsの生成（Trino用に簡略化）
    try:
        root_operator.parse_columns_bottom_up({}, {}, {}, alias_dict={}, table_samples=None, col_stats=None)
    except (KeyError, ValueError) as e:
        # Trinoの複雑なカラム名に対応するため、エラーを無視してoutput_columnsのみ生成
        print(f"⚠️  parse_columns_bottom_upでエラー（無視）: {e}")
        # 手動でoutput_columnsを生成
        generate_output_columns_manually(root_operator)
    
    return root_operator


def generate_output_columns_manually(node, is_root=True):
    """手動でoutput_columnsを生成（Trino用）- ルートノードのみ"""
    # ルートノードのみでoutput_columnsを生成（中間演算子の複雑なカラム名を避ける）
    if is_root:
        if 'layout' in node.plan_parameters:
            layout = node.plan_parameters['layout']
            if layout:
                try:
                    output_columns = node.parse_output_columns(','.join(layout))
                    node.plan_parameters['output_columns'] = output_columns
                except Exception as e:
                    print(f"⚠️  output_columns生成でエラー（無視）: {e}")
    
    # 子ノードも再帰的に処理（is_root=False）
    for child in node.children:
        generate_output_columns_manually(child, is_root=False)


def integrate_all_fragments(root_operator, all_fragment_operators):
    """全Fragmentの演算子を統合（テーブル情報を含む）"""
    # 存在するFragmentを動的に検出
    fragment_ids = set()
    for operator in all_fragment_operators:
        fragment_id = operator.get('fragment_id', '')
        if fragment_id:
            fragment_ids.add(fragment_id)
    
    print(f"🔍 検出されたFragment: {sorted(fragment_ids)}")
    
    # Fragment 1以外のすべてのFragmentの演算子を追加
    for operator in all_fragment_operators:
        fragment_id = operator.get('fragment_id', '')
        if fragment_id != '1':  # Fragment 1以外のすべてのFragment
            # テーブル演算子を子ノードとして追加
            child_operator = create_trino_plan_operator(operator)
            root_operator.children.append(child_operator)


def build_children(parent, operators, start_idx, parent_depth):
    """子ノードを再帰的に構築"""
    i = start_idx
    while i < len(operators):
        operator = operators[i]
        
        # 親より深い演算子は子ノード
        if operator['depth'] > parent_depth:
            child = create_trino_plan_operator(operator)
            parent.children.append(child)
            
            # 子ノードの子を再帰的に構築
            i = build_children(child, operators, i + 1, operator['depth'])
        else:
            # 同じ深度または浅い深度の演算子は兄弟ノード
            break
    
    return i


def parse_trino_plans_v2(run_stats, min_runtime=100, max_runtime=30000, parse_baseline=False, cap_queries=None,
                        parse_join_conds=False, include_zero_card=False, explain_only=False):
    """Trinoプランを一括解析（改良版）"""
    # カラム統計情報のマッピング作成
    column_id_mapping = dict()
    table_id_mapping = dict()
    partial_column_name_mapping = collections.defaultdict(set)
    
    database_stats = run_stats.database_stats
    
    # テーブルサイズ情報をカラム統計に追加
    table_sizes = dict()
    for table_stat in database_stats.table_stats:
        table_sizes[table_stat.relname] = table_stat.reltuples
    
    for i, column_stat in enumerate(database_stats.column_stats):
        table = column_stat.tablename
        column = column_stat.attname
        column_stat.table_size = table_sizes[table]
        column_id_mapping[(table, column)] = i
        partial_column_name_mapping[column].add(table)
    
    # テーブル統計のインデックス作成
    for i, table_stat in enumerate(database_stats.table_stats):
        table = table_stat.relname
        table_id_mapping[table] = i
    
    # 個別クエリの解析
    parsed_plans = []
    avg_runtimes = []
    no_tables = []
    no_filters = []
    op_perc = collections.defaultdict(int)
    
    for q in tqdm(run_stats.query_list):
        # タイムアウトクエリをスキップ
        if hasattr(q, 'timeout') and q.timeout:
            continue
        
        alias_dict = dict()
        
        if not explain_only:
            if q.analyze_plans is None:
                continue
            
            if len(q.analyze_plans) == 0:
                continue
            
            # 平均実行時間を計算
            ex_times = []
            planning_times = []
            for analyze_plan in q.analyze_plans:
                _, ex_time, planning_time = parse_trino_raw_plan_v2(analyze_plan, analyze=True, parse=False)
                ex_times.append(ex_time)
                planning_times.append(planning_time)
            avg_runtime = sum(ex_times) / len(ex_times)
            
            # プランをツリー構造に解析
            analyze_plan, _, _ = parse_trino_raw_plan_v2(q.analyze_plans[0], analyze=True, parse=True)
            
            if not analyze_plan:
                continue
                
        else:
            avg_runtime = 0
        
        # EXPLAINのみのプラン解析
        if hasattr(q, 'verbose_plan') and q.verbose_plan:
            verbose_plan, _, _ = parse_trino_raw_plan_v2(q.verbose_plan, analyze=False, parse=True)
        else:
            verbose_plan = None
        
        if not explain_only and analyze_plan:
            analyze_plan = analyze_plan
        else:
            analyze_plan = verbose_plan
        
        if not analyze_plan:
            continue
        
        # 結合条件を抽出して設定（まだ設定されていない場合）
        if not hasattr(analyze_plan, 'join_conds') or analyze_plan.join_conds is None:
            join_conds = extract_join_conditions_trino(analyze_plan)
            analyze_plan.join_conds = join_conds
        
        # プラン統計情報を収集
        stats_result = plan_statistics(analyze_plan)
        tables = stats_result['tables']
        filter_columns = stats_result['filter_columns']
        operators = stats_result['operators']
        
        # カラム情報を統計情報と照合
        try:
            # テーブルサンプルとカラム統計を取得（オプショナル）
            table_samples = getattr(run_stats, 'table_samples', None)
            col_stats = database_stats.column_stats if hasattr(database_stats, 'column_stats') else None
            
            analyze_plan.parse_columns_bottom_up(column_id_mapping, partial_column_name_mapping, table_id_mapping,
                                               alias_dict=alias_dict,
                                               table_samples=table_samples,
                                               col_stats=col_stats)
        except Exception as e:
            print(f"Warning: Column parsing failed: {e}")
        
        analyze_plan.plan_runtime = avg_runtime
        
        if not explain_only:
            # 結果のチェック
            if hasattr(analyze_plan, 'min_card') and analyze_plan.min_card() == 0 and not include_zero_card:
                continue
            
            if min_runtime is not None and avg_runtime < min_runtime:
                continue
            
            if avg_runtime > max_runtime:
                continue
        
        # 統計情報を収集
        avg_runtimes.append(avg_runtime)
        no_tables.append(len(tables))
        for _, op in filter_columns:
            op_perc[op] += 1
        # フィルター数（AND, ORを除く）
        no_filters.append(len([fc for fc in filter_columns if fc[0] is not None]))
        
        # SQL文字列をプランに追加（存在する場合のみ）
        if hasattr(q, 'sql'):
            analyze_plan.sql = q.sql
        else:
            analyze_plan.sql = 'SELECT * FROM unknown'  # デフォルト
        
        parsed_plans.append(analyze_plan)
        
        if cap_queries is not None and len(parsed_plans) >= cap_queries:
            print(f"Parsed {cap_queries} queries. Stopping parsing.")
            break
    
    # 統計情報の出力
    if not no_tables == []:
        print(f"Table statistics: "
              f"\n\tmean: {np.mean(no_tables):.1f}"
              f"\n\tmedian: {np.median(no_tables)}"
              f"\n\tmax: {np.max(no_tables)}")
        print("Operators statistics (appear in x% of queries)")
        for op, op_count in op_perc.items():
            print(f"\t{str(op)}: {op_count / len(avg_runtimes) * 100:.0f}%")
        print(f"Runtime statistics: "
              f"\n\tmedian: {np.median(avg_runtimes) / 1000:.2f}s"
              f"\n\tmax: {np.max(avg_runtimes) / 1000:.2f}s"
              f"\n\tmean: {np.mean(avg_runtimes) / 1000:.2f}s")
        print(f"Parsed {len(parsed_plans)} plans ({len(run_stats.query_list) - len(parsed_plans)} had zero-cardinalities "
              f"or were too fast).")
    
    parsed_runs = dict(parsed_plans=parsed_plans, database_stats=database_stats,
                       run_kwargs=run_stats.run_kwargs)
    
    stats = dict(
        runtimes=str(avg_runtimes),
        no_tables=str(no_tables),
        no_filters=str(no_filters)
    )
    
    return parsed_runs, stats
