"""
Trino to QPPNet Adapter

TrinoのプラントリーをQPPNetが期待するPostgreSQL形式に変換するアダプター層。
実装コストO(N)で、既存のQPPNetコードをそのまま活用可能にする。
"""

from types import SimpleNamespace
from typing import Dict, Any, List


# Trino演算子をPostgreSQL演算子名にマッピング
TRINO_TO_POSTGRES_OP_MAPPING = {
    'TableScan': 'Seq Scan',
    'ScanFilter': 'Seq Scan',
    'ScanFilterProject': 'Seq Scan',
    'Filter': 'Result',
    'FilterProject': 'Result',
    'Project': 'Result',
    'InnerJoin': 'Hash Join',
    'LeftJoin': 'Hash Join',
    'RightJoin': 'Hash Join',
    'CrossJoin': 'Nested Loop',
    'Aggregate': 'Aggregate',
    'PartialAggregate': 'Aggregate',
    'FinalAggregate': 'Aggregate',
    'Sort': 'Sort',
    'TopN': 'Sort',
    'TopNRanking': 'Sort',
    'HashBuild': 'Hash',
    'Exchange': 'Gather',
    'LocalExchange': 'Gather',
    'RemoteSource': 'Gather',
    'Output': 'Result',
    'Values': 'Result',
    'Limit': 'Limit',
    'Window': 'WindowAgg',
    'Union': 'Append',
    'SemiJoin': 'Hash Join',
}

DATA_MOVEMENT_OPERATORS = {'Exchange', 'LocalExchange', 'RemoteSource'}
LEAF_NODE_TYPES = {
    'Seq Scan',
    'Index Scan',
    'Index Only Scan',
    'Bitmap Heap Scan',
    'Bitmap Index Scan'
}

def _get_trino_op_name(op) -> str:
    """Trino演算子名を取得"""
    params = op.plan_parameters
    if isinstance(params, SimpleNamespace):
        return getattr(params, 'op_name', 'Unknown')
    elif isinstance(params, dict):
        return params.get('op_name', 'Unknown')
    else:
        return 'Unknown'


def adapt_trino_plan_to_qppnet(trino_plan_operator) -> Dict[str, Any]:
    """
    TrinoのプランをQPPNetが期待するPostgreSQL形式に変換
    
    Args:
        trino_plan_operator: TrinoPlanOperatorオブジェクト
    
    Returns:
        PostgreSQL EXPLAIN ANALYZE JSON形式の辞書
    """
    # プラン全体の情報
    converted_nodes = _convert_and_collect(trino_plan_operator)
    plan_dict = {
        'Plan': converted_nodes[0] if converted_nodes else {},
        'Execution Time': getattr(trino_plan_operator, 'plan_runtime', 0) * 1000,  # ms単位に変換
        'Planning Time': 0  # Trinoは通常planning timeを分離しない
    }
    return plan_dict


def _convert_and_collect(op, skip_data_movement=True) -> List[Dict[str, Any]]:
    """
    演算子を変換し、リスト形式で返す（データ移動ノードは子ノードに委譲）
    """
    params = op.plan_parameters
    
    # 演算子名を取得
    trino_op_name = _get_trino_op_name(op)
    
    # データ移動ノードは透明化して子を直接返す
    if skip_data_movement and trino_op_name in DATA_MOVEMENT_OPERATORS:
        converted_children = []
        for child in op.children:
            converted_children.extend(_convert_and_collect(child, skip_data_movement=True))
        return converted_children

    # QPPNetが期待する基本特徴量を構築
    def get_param(key, default=None):
        if isinstance(params, SimpleNamespace):
            return getattr(params, key, default)
        else:
            return params.get(key, default)
    
    node_type = TRINO_TO_POSTGRES_OP_MAPPING.get(trino_op_name, trino_op_name)
    node_dict = {
        'Node Type': node_type,
        'Plan Width': int(get_param('est_width', 0)),
        'Plan Rows': int(get_param('est_rows', 0)),
        'Total Cost': float(get_param('est_cost', 0)),
        'Actual Rows': int(get_param('act_output_rows', 0)),
        # Actual Total Timeは不要（グローバルlossのため）
    }
    
    # 演算子固有の特徴量を追加
    _add_scan_features(node_dict, node_type, params, get_param)
    _add_join_features(node_dict, node_type, params, get_param)
    _add_hash_features(node_dict, node_type, params, get_param)
    _add_sort_features(node_dict, node_type, params, get_param)
    _add_aggregate_features(node_dict, node_type, params, get_param)
    
    # 子ノードを再帰的に変換
    node_dict['Plans'] = []
    extra_nodes: List[Dict[str, Any]] = []
    for child in op.children:
        child_nodes = _convert_and_collect(child, skip_data_movement=True)
        if node_type in LEAF_NODE_TYPES:
            extra_nodes.extend(child_nodes)
        else:
            node_dict['Plans'].extend(child_nodes)
    
    if node_type in LEAF_NODE_TYPES:
        # スキャンノードは子を持たせず、追加ノードは親に委譲
        return [node_dict] + extra_nodes
    else:
        return [node_dict]


def _add_scan_features(node_dict: Dict, node_type: str, params, get_param):
    """Scan演算子の特徴量を追加"""
    if 'Scan' in node_type:
        node_dict['Relation Name'] = get_param('table', '')
        
        # Min/Max/Meanはカラム統計から取得される想定
        # ここではデフォルト値を設定（実際の値はcolumn_statisticsから取得される）
        node_dict['Min'] = 0
        node_dict['Max'] = 0
        node_dict['Mean'] = 0


def _add_join_features(node_dict: Dict, node_type: str, params, get_param):
    """Join演算子の特徴量を追加"""
    if 'Join' in node_type or 'Nested Loop' in node_type:
        # Trinoの結合タイプを推測
        trino_op_name = get_param('op_name', '')
        if 'Inner' in trino_op_name:
            join_type = 'Inner'
        elif 'Left' in trino_op_name:
            join_type = 'Left'
        elif 'Right' in trino_op_name:
            join_type = 'Right'
        elif 'Cross' in trino_op_name:
            join_type = 'Cross'
        elif 'Semi' in trino_op_name:
            join_type = 'Semi'
        else:
            join_type = 'Hash'  # デフォルト
        
        node_dict['Join Type'] = join_type
        
        # Parent Relationshipの推測
        # Trinoには直接対応する情報はないが、Join Typeから推測
        if 'Left' in join_type or 'Right' in join_type:
            parent_relationship = 'Outer'
        elif 'Semi' in join_type:
            parent_relationship = 'SubPlan'
        else:
            parent_relationship = 'Inner'  # デフォルト
        
        node_dict['Parent Relationship'] = parent_relationship
        
        # ★ Distribution情報をHash Algorithmとして使用
        # REPLICATED = broadcast join, PARTITIONED = hash distributed join
        distribution = get_param('distribution', '')
        if distribution == 'REPLICATED':
            hash_algorithm = 'Broadcast'
        elif distribution == 'PARTITIONED':
            hash_algorithm = 'Hash'
        elif 'Cross' in trino_op_name:
            hash_algorithm = 'Nested Loop'
        else:
            hash_algorithm = 'Hash'  # デフォルト
        
        node_dict['Hash Algorithm'] = hash_algorithm


def _add_hash_features(node_dict: Dict, node_type: str, params, get_param):
    """Hash演算子の特徴量を追加"""
    if 'Hash' in node_type:
        est_memory = get_param('est_memory', 0)
        # Hash Bucketsはメモリから推定（1KB per bucket と仮定）
        node_dict['Hash Buckets'] = max(1, int(est_memory / 1024))
        node_dict['Peak Memory Usage'] = int(est_memory)


def _add_sort_features(node_dict: Dict, node_type: str, params, get_param):
    """Sort演算子の特徴量を追加"""
    if 'Sort' in node_type:
        # Trinoには詳細なソート情報がないため、デフォルト値を使用
        node_dict['Sort Key'] = ['unknown']
        node_dict['Sort Method'] = 'quicksort'


def _add_aggregate_features(node_dict: Dict, node_type: str, params, get_param):
    """Aggregate演算子の特徴量を追加"""
    if node_type == 'Aggregate':
        trino_op_name = get_param('op_name', '')
        
        # Strategyの推定
        if 'Partial' in trino_op_name:
            strategy = 'Hashed'
            partial_mode = 'Partial'
        elif 'Final' in trino_op_name:
            strategy = 'Hashed'
            partial_mode = 'Finalize'
        else:
            strategy = 'Plain'
            partial_mode = 'Simple'
        
        node_dict['Strategy'] = strategy
        node_dict['Partial Mode'] = partial_mode


def validate_converted_plan(plan_dict: Dict) -> bool:
    """
    変換されたプランが有効かどうかを検証
    
    Args:
        plan_dict: 変換されたプラン辞書
    
    Returns:
        有効な場合True
    """
    required_keys = ['Plan', 'Execution Time', 'Planning Time']
    if not all(key in plan_dict for key in required_keys):
        return False
    
    # プランノードの検証
    plan = plan_dict['Plan']
    required_node_keys = ['Node Type', 'Plan Width', 'Plan Rows', 'Total Cost']
    if not all(key in plan for key in required_node_keys):
        return False
    
    return True


def print_conversion_summary(trino_op, postgres_dict):
    """変換のサマリーを出力（デバッグ用）"""
    params = trino_op.plan_parameters
    
    def get_param(key, default=None):
        if isinstance(params, SimpleNamespace):
            return getattr(params, key, default)
        else:
            return params.get(key, default)
    
    print(f"Converted: {get_param('op_name')} -> {postgres_dict['Plan']['Node Type']}")
    print(f"  Est Rows: {get_param('est_rows')} -> {postgres_dict['Plan']['Plan Rows']}")
    print(f"  Act Rows: {get_param('act_output_rows')} -> {postgres_dict['Plan']['Actual Rows']}")
    print(f"  Runtime: {getattr(trino_op, 'plan_runtime', 0)*1000:.2f}ms")

