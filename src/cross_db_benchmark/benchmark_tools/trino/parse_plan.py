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
# Queuedの単位はus/μs/msのいずれか、Analysis/Planning/Executionの単位はms/s/m/us/μsのいずれか
trino_timing_regex = re.compile(r'Queued: ([\d.]+)(us|μs|ms)?, Analysis: ([\d.]+)(ms|s|m|us|μs)?, Planning: ([\d.]+)(ms|s|m|us|μs)?, Execution: ([\d.]+)(ms|s|m|us|μs)?')
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
            # Group 1: Queued値, Group 2: Queued単位
            queued_value = float(timing_match.group(1))
            queued_unit = timing_match.group(2) or 'ms'
            if queued_unit in ('us', 'μs'):
                queued_time = queued_value / 1000  # us/μs to ms
            else:
                queued_time = queued_value  # ms
            
            # Group 3: Analysis値, Group 4: Analysis単位
            analysis_value = float(timing_match.group(3))
            analysis_unit = timing_match.group(4) or 'ms'
            if analysis_unit == 's':
                analysis_time = analysis_value * 1000  # s to ms
            elif analysis_unit == 'm':
                analysis_time = analysis_value * 60000  # m to ms
            elif analysis_unit in ('us', 'μs'):
                analysis_time = analysis_value / 1000  # us/μs to ms
            else:
                analysis_time = analysis_value  # ms
            
            # Group 5: Planning値, Group 6: Planning単位
            planning_value = float(timing_match.group(5))
            planning_unit = timing_match.group(6) or 'ms'
            if planning_unit == 's':
                planning_time = planning_value * 1000  # s to ms
            elif planning_unit == 'm':
                planning_time = planning_value * 60000  # m to ms
            elif planning_unit in ('us', 'μs'):
                planning_time = planning_value / 1000  # us/μs to ms
            else:
                planning_time = planning_value  # ms
            
            # Group 7: Execution値, Group 8: Execution単位
            execution_value = float(timing_match.group(7))
            execution_unit = timing_match.group(8) or 'ms'
            if execution_unit == 's':
                execution_time = execution_value * 1000  # s to ms
            elif execution_unit == 'm':
                execution_time = execution_value * 60000  # m to ms
            elif execution_unit in ('us', 'μs'):
                execution_time = execution_value / 1000  # us/μs to ms
            else:
                execution_time = execution_value  # ms
            
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
            from types import SimpleNamespace
            trino_operator = TrinoPlanOperator(operator['lines'])
            # SimpleNamespace対応: ヘルパーメソッドまたは直接setattrを使用
            if isinstance(trino_operator.plan_parameters, SimpleNamespace):
                setattr(trino_operator.plan_parameters, 'op_name', operator['name'])
                setattr(trino_operator.plan_parameters, 'fragment_id', fragment['id'])
                setattr(trino_operator.plan_parameters, 'fragment_type', fragment['type'])
            else:
                trino_operator.plan_parameters['op_name'] = operator['name']
                trino_operator.plan_parameters['fragment_id'] = fragment['id']
                trino_operator.plan_parameters['fragment_type'] = fragment['type']
            trino_operator.parse_lines_recursively()
            
            # 解析された情報を演算子に追加（SimpleNamespaceから辞書に変換）
            if isinstance(trino_operator.plan_parameters, SimpleNamespace):
                operator.update(vars(trino_operator.plan_parameters))
            else:
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
    
    # 行から先頭の空白と│（パイプ）文字を除去してから判定
    stripped = line.strip()
    # │で始まる場合は除去
    if stripped.startswith('│'):
        stripped = stripped[1:].strip()
    
    # 演算子のパターンを検出
    operator_patterns = [
        r'^\w+\[',  # Aggregate[type = FINAL]
        r'^\w+\s*$',  # TableScan
        r'^└─\s*\w+',  # └─ LocalExchange
        r'^├─\s*\w+',  # ├─ ScanFilter
        r'^\w+\[.*\]',  # ScanFilterProject[table = ...]
    ]
    
    for pattern in operator_patterns:
        if re.match(pattern, stripped):
            return True
    
    return False


def extract_operator_name(line):
    """行から演算子名を抽出"""
    # 行をクリーンアップ
    stripped = line.strip()
    # │で始まる場合は除去
    if stripped.startswith('│'):
        stripped = stripped[1:].strip()
    
    # 演算子名のパターンを検出
    patterns = [
        r'^└─\s*(\w+)',  # └─ LocalExchange
        r'^├─\s*(\w+)',  # ├─ ScanFilter
        r'^(\w+)\[',     # Aggregate[type = FINAL]
        r'^(\w+)\s*$',   # TableScan
        r'^(\w+)\[.*\]', # ScanFilterProject[table = ...]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1)
    
    return 'Unknown'


def count_indent_depth(line):
    """行のインデント深度を計算"""
    # スペースとタブのインデントを計算
    # │（パイプ）文字はインデントとして扱わない（視覚的な接続線のため）
    line_without_pipe = line.replace('│', ' ')
    space_indent = len(line_without_pipe) - len(line_without_pipe.lstrip())
    
    # Trinoの階層記号（└─, ├─）は同じレベルを示す
    # 実際のdepthはインデント（スペース数）で決まる
    # └─ や ├─ の前のスペース数を基準にする
    stripped = line_without_pipe.lstrip()
    if stripped.startswith('└─') or stripped.startswith('├─'):
        # 記号の前のスペース数がdepth
        return space_indent
    else:
        # 通常のインデント（記号なし）
        return space_indent


def create_trino_plan_operator(operator_info):
    """演算子情報からTrinoPlanOperatorを作成"""
    from types import SimpleNamespace
    operator = TrinoPlanOperator(operator_info['lines'])
    
    # operator_infoに含まれるすべての情報をplan_parametersにコピー
    # これにより、parse_trino_plan_simpleで抽出されたテーブル情報などが保持される
    for key, value in operator_info.items():
        if key not in ['lines', 'children']:  # linesとchildrenは除外
            if isinstance(operator.plan_parameters, SimpleNamespace):
                setattr(operator.plan_parameters, key, value)
            else:
                operator.plan_parameters[key] = value
    
    # 注: plan_parametersのSimpleNamespace変換は、parse_lines()の後に行う
    # （parse_lines()が辞書を前提としているため）
    
    return operator


def extract_join_conditions_trino(root_operator):
    """Trinoプランから結合条件を抽出"""
    join_conds = []
    
    def traverse_plan(node):
        """プランを再帰的に走査して結合条件を収集"""
        # 現在のノードがJoin演算子かチェック
        op_name = getattr(node.plan_parameters, 'op_name', '').lower()
        
        # Join演算子の検出（InnerJoin, LeftJoin, RightJoin, FullJoin, CrossJoinなど）
        if 'join' in op_name:
            # criteriaパラメータから結合条件を抽出
            # InnerJoin[criteria = (id_upravna_enota = upravna_enota_4), ...]のような形式
            criteria = getattr(node.plan_parameters, 'criteria', None)
            if criteria:
                # 括弧を除去して結合条件を正規化
                join_cond = criteria.strip('()')
                if join_cond:
                    join_conds.append(join_cond)
            else:
                # filterPredicateから結合条件を推測（結合条件らしいフィルタを探す）
                filter_condition = getattr(node.plan_parameters, 'filter_condition', None)
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
                # 正規表現のグループ: 1=Queued値, 2=Queued単位, 3=Analysis値, 4=Analysis単位,
                # 5=Planning値, 6=Planning単位, 7=Execution値, 8=Execution単位
                execution_time = float(timing_match.group(7))  # Execution値
                execution_unit = timing_match.group(8)  # Execution単位
                if execution_unit and execution_unit == 's':
                    execution_time *= 1000
                elif execution_unit and execution_unit in ('us', 'μs'):
                    execution_time /= 1000
                elif execution_unit and execution_unit == 'm':
                    execution_time *= 60000
                
                planning_time = float(timing_match.group(5))  # Planning値
                planning_unit = timing_match.group(6)  # Planning単位
                if planning_unit and planning_unit == 's':
                    planning_time *= 1000
                elif planning_unit and planning_unit in ('us', 'μs'):
                    planning_time /= 1000
                elif planning_unit and planning_unit == 'm':
                    planning_time *= 60000
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


def convert_plan_parameters_to_namespace(node):
    """
    全ノードのplan_parametersを辞書からSimpleNamespaceに変換
    
    これにより、PostgreSQLと統一されたアクセス方法が可能になる:
    - 辞書: node.plan_parameters['op_name']
    - SimpleNamespace: node.plan_parameters.op_name (← 統一後)
    
    Args:
        node: ルートノード（再帰的に全子ノードも変換）
    """
    from types import SimpleNamespace
    
    # plan_parametersが辞書の場合のみSimpleNamespaceに変換
    if isinstance(node.plan_parameters, dict):
        node.plan_parameters = SimpleNamespace(**node.plan_parameters)
    
    # 子ノードも再帰的に変換
    if hasattr(node, 'children') and node.children:
        for child in node.children:
            convert_plan_parameters_to_namespace(child)


def build_hierarchy(operators, all_fragment_operators=None):
    """演算子のリストを階層構造に変換（全Fragmentの演算子を含む）"""
    if not operators:
        return None
    
    # ルート演算子を作成（最初の演算子）
    root_operator = create_trino_plan_operator(operators[0])
    
    # 子ノードを再帰的に構築
    build_children(root_operator, operators, 1, operators[0]['depth'])
    
    # 演算子の詳細解析（先にパースしてからFragmentを統合）
    root_operator.parse_lines_recursively()
    
    # 全Fragmentの演算子を統合（テーブル情報を含む）
    # parse_lines_recursively()の後に実行することで、RemoteSourceが正しく検出される
    if all_fragment_operators:
        integrate_all_fragments(root_operator, all_fragment_operators)
    
    # 子ノードのカーディナリティを計算とoutput_columnsの生成（Trino用に簡略化）
    # 注: この処理は plan_parameters が辞書形式のままで実行される
    parse_columns_success = False
    try:
        root_operator.parse_columns_bottom_up({}, {}, {}, alias_dict={}, table_samples=None, col_stats=None)
        parse_columns_success = True
    except (KeyError, ValueError) as e:
        # Trinoの複雑なカラム名に対応するため、エラーを無視してoutput_columnsのみ生成
        print(f"⚠️  parse_columns_bottom_upでエラー（無視）: {e}")
    
    # parse_columns_bottom_upが成功してもoutput_columnsが空の場合があるため、常に手動生成を試みる
    # 手動でoutput_columnsを生成（この時点ではまだ辞書形式）
    generate_output_columns_manually(root_operator)
    
    # 全ての処理が完了した後、plan_parametersをSimpleNamespaceに変換
    # これにより、PostgreSQLと統一されたアクセス方法が可能になる
    convert_plan_parameters_to_namespace(root_operator)
    
    return root_operator


def generate_output_columns_manually(node, is_root=True):
    """
    手動でoutput_columnsを生成（Trino用）- 全ノードで実行
    
    Note: Trinoでは各ノードのLayoutが異なる意味を持つ：
    1. ルートノードのLayout: クエリの最終SELECT句
    2. 中間ノードのLayout: その演算子が出力するカラム（中間結果）
    
    機械学習モデルは両方の情報を使うため、全ノードでoutput_columnsを生成します。
    ただし、is_final_output フラグで区別できるようにします。
    
    Args:
        node: Plan operator node
        is_root: True if this is the root node (query's final output)
    """
    # plan_parametersがdictかSimpleNamespaceか判定
    params = node.plan_parameters
    
    # layoutを取得（dictとSimpleNamespace両対応）
    layout = None
    if isinstance(params, dict):
        layout = params.get('layout')
    elif hasattr(params, 'layout'):
        layout = params.layout
    
    if layout and len(layout) > 0 and layout != ['']:
        try:
            # layoutからoutput_columnsを生成
            output_columns = node.parse_output_columns(','.join(layout))
            
            # plan_parametersに設定（dictとSimpleNamespace両対応）
            if isinstance(params, dict):
                params['output_columns'] = output_columns
                # フラグを追加: これが最終出力かどうか
                params['is_final_output'] = is_root
            else:
                params.output_columns = output_columns
                params.is_final_output = is_root
            
            # Debug出力（簡略化）
            # if is_root or len(output_columns) > 0:
            #     op_name = params.get('op_name', 'Unknown') if isinstance(params, dict) else getattr(params, 'op_name', 'Unknown')
            #     print(f"✓ Generated {len(output_columns)} output_columns for {op_name} {'(FINAL OUTPUT)' if is_root else ''}")
        except Exception as e:
            print(f"⚠️  output_columns生成でエラー（無視）: {e}")
    
    # 子ノードも再帰的に処理（is_root=False）
    for child in node.children:
        generate_output_columns_manually(child, is_root=False)


def integrate_all_fragments(root_operator, all_fragment_operators):
    """
    全Fragmentの演算子を統合（RemoteSourceを通じた正しい階層構造）
    
    RemoteSource[sourceFragmentIds = [X]] は Fragment X を子として参照する。
    この関数は、各 RemoteSource の下に参照先 Fragment の演算子を正しく配置する。
    """
    import re
    
    # 存在するFragmentを動的に検出
    fragment_ids = set()
    for operator in all_fragment_operators:
        fragment_id = operator.get('fragment_id', '')
        if fragment_id:
            fragment_ids.add(fragment_id)
    
    # Fragment ごとに演算子をグループ化
    fragment_operators = {}
    for operator in all_fragment_operators:
        fragment_id = operator.get('fragment_id', '')
        if fragment_id:
            if fragment_id not in fragment_operators:
                fragment_operators[fragment_id] = []
            fragment_operators[fragment_id].append(operator)
    
    # RemoteSource ノードを見つけて、参照先 Fragment を子として接続
    def attach_fragments_to_remote_sources(node):
        """
        再帰的にRemoteSourceを探し、sourceFragmentIdsで参照されている
        Fragmentの演算子を子として接続する
        """
        # op_nameを複数の方法で取得
        op_name = None
        
        # 1. plan_parametersから取得
        if hasattr(node, 'plan_parameters'):
            if hasattr(node.plan_parameters, 'op_name'):
                op_name = node.plan_parameters.op_name
            elif isinstance(node.plan_parameters, dict):
                op_name = node.plan_parameters.get('op_name', '')
        
        # 2. plain_contentから取得（op_nameが設定されていない場合）
        if not op_name and hasattr(node, 'plain_content') and node.plain_content:
            first_line = node.plain_content[0] if node.plain_content else ''
            # RemoteSourceの行を検出
            if 'RemoteSource' in first_line:
                op_name = 'RemoteSource'
        
        if op_name == 'RemoteSource':
            # 既に子ノードが存在する場合はスキップ（重複を避ける）
            if node.children:
                # 既存の子ノードも再帰的に処理
                for child in list(node.children):
                    attach_fragments_to_remote_sources(child)
                return
            
            # sourceFragmentIds を抽出
            source_fragment_ids = []
            # TrinoPlanOperatorでは lines は plain_content として保存される
            # 複数の場所からsourceFragmentIdsを抽出を試みる
            content_to_search = []
            
            # 1. plain_contentから抽出
            if hasattr(node, 'plain_content') and node.plain_content:
                content_to_search.extend(node.plain_content)
            
            # 2. plan_parametersから抽出（既にパースされている場合）
            if hasattr(node, 'plan_parameters'):
                if hasattr(node.plan_parameters, 'sourceFragmentIds'):
                    source_fragment_ids = node.plan_parameters.sourceFragmentIds
                    if isinstance(source_fragment_ids, list):
                        source_fragment_ids = [str(fid) for fid in source_fragment_ids]
                    else:
                        source_fragment_ids = [str(source_fragment_ids)]
                elif isinstance(node.plan_parameters, dict):
                    if 'sourceFragmentIds' in node.plan_parameters:
                        source_fragment_ids = node.plan_parameters['sourceFragmentIds']
                        if isinstance(source_fragment_ids, list):
                            source_fragment_ids = [str(fid) for fid in source_fragment_ids]
                        else:
                            source_fragment_ids = [str(source_fragment_ids)]
            
            # 3. plain_contentから正規表現で抽出（上記で見つからなかった場合）
            if not source_fragment_ids and content_to_search:
                for line in content_to_search:
                    match = re.search(r'sourceFragmentIds\s*=\s*\[([^\]]+)\]', line)
                    if match:
                        # "2" or "3, 4" のような形式
                        ids_str = match.group(1)
                        source_fragment_ids = [fid.strip() for fid in ids_str.split(',')]
                        break
            
            # 参照先 Fragment の演算子を子として追加
            for frag_id in source_fragment_ids:
                if frag_id in fragment_operators:
                    # Fragment の演算子を階層構造に変換
                    frag_ops = fragment_operators[frag_id]
                    if frag_ops:
                        # Fragment のルート演算子を作成（最初の演算子）
                        # operator_infoにはfragment_idが含まれているはず
                        frag_root = create_trino_plan_operator(frag_ops[0])
                        
                        # fragment_idが正しく設定されているか確認して、設定されていない場合は手動で設定
                        if hasattr(frag_root, 'plan_parameters'):
                            from types import SimpleNamespace
                            if isinstance(frag_root.plan_parameters, dict):
                                if 'fragment_id' not in frag_root.plan_parameters:
                                    frag_root.plan_parameters['fragment_id'] = frag_id
                            elif isinstance(frag_root.plan_parameters, SimpleNamespace):
                                if not hasattr(frag_root.plan_parameters, 'fragment_id'):
                                    setattr(frag_root.plan_parameters, 'fragment_id', frag_id)
                        
                        # Fragment 内の子ノードを再帰的に構築
                        if len(frag_ops) > 1:
                            build_children(frag_root, frag_ops, 1, frag_ops[0]['depth'])
                        
                        # Fragment の演算子を詳細解析（plain_contentを設定）
                        frag_root.parse_lines_recursively()
                        
                        # Fragment の演算子を処理済みとしてマーク（重複を避ける）
                        # この Fragment に含まれる RemoteSource も再帰的に処理（追加前に処理）
                        attach_fragments_to_remote_sources(frag_root)
                        
                        # RemoteSource の子として追加（処理後に追加）
                        node.children.append(frag_root)
        
        # 既存の子ノードも再帰的に処理
        for child in list(node.children):  # list() でコピーして、追加中の変更を避ける
            attach_fragments_to_remote_sources(child)
    
    # ルートから RemoteSource を探して Fragment を接続
    attach_fragments_to_remote_sources(root_operator)


def build_children(parent, operators, start_idx, parent_depth):
    """
    子ノードを再帰的に構築
    
    Trinoのインデント構造に基づいて階層を決定します：
    - depth > parent_depth: 確実に子ノード
    - depth == parent_depth: 状況による（同じ深度の兄弟演算子）
    - depth < parent_depth: 親レベルに戻る
    """
    i = start_idx
    while i < len(operators):
        operator = operators[i]
        
        if operator['depth'] > parent_depth:
            # 深い深度：確実に子ノード
            child = create_trino_plan_operator(operator)
            parent.children.append(child)
            
            # 子ノードの子を再帰的に構築
            i = build_children(child, operators, i + 1, operator['depth'])
            
        elif operator['depth'] == parent_depth:
            # 同じ深度：Trinoでは `└─` 表記で同じインデントでも親子関係を示す
            # 例: Aggregate (depth=4) と ScanFilterProject (depth=4) は親子関係
            
            # 親の後に続く同じ深度の演算子を子ノードとして扱う
            child = create_trino_plan_operator(operator)
            parent.children.append(child)
            
            # この子ノードの子を再帰的に構築
            i = build_children(child, operators, i + 1, operator['depth'])
            
        else:
            # 浅い深度：親レベルに戻る
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
    # テーブル名は既に小文字に統一されている前提
    table_sizes = dict()
    for table_stat in database_stats.table_stats:
        table_sizes[table_stat.relname] = table_stat.reltuples
    
    for i, column_stat in enumerate(database_stats.column_stats):
        table = column_stat.tablename
        column = column_stat.attname
        # テーブル名は既に小文字に統一されている前提
        column_stat.table_size = table_sizes.get(table, 0)
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
