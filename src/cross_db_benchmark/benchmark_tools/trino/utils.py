import math


def plan_statistics(plan_op, tables=None, filter_columns=None, operators=None, skip_columns=False, conv_to_dict=False):
    """Trinoプランの統計情報を収集"""
    if tables is None:
        tables = set()
    if operators is None:
        operators = set()
    if filter_columns is None:
        filter_columns = set()
    
    params = plan_op.plan_parameters
    
    # SimpleNamespace 統一後は hasattr を使用
    if hasattr(params, 'table'):
        tables.add(getattr(params, 'table'))
    if hasattr(params, 'op_name'):
        operators.add(getattr(params, 'op_name'))
    if hasattr(params, 'filter_columns') and not skip_columns:
        list_columns(getattr(params, 'filter_columns'), filter_columns)
    
    for c in plan_op.children:
        plan_statistics(c, tables=tables, filter_columns=filter_columns, operators=operators, skip_columns=skip_columns,
                        conv_to_dict=conv_to_dict)
    
    # 辞書形式で統計情報を返す
    return {
        'tables': list(tables),
        'filter_columns': list(filter_columns),
        'operators': list(operators),
        'no_tables': len(tables),
        'no_filters': len(filter_columns)
    }


def child_prod(p, feature_name, default=1):
    """子ノードの特徴量の積を計算"""
    child_feat = [getattr(c.plan_parameters, feature_name, None) for c in p.children
                  if getattr(c.plan_parameters, feature_name, None) is not None]
    if len(child_feat) == 0:
        return default
    return math.prod(child_feat)


def list_columns(n, columns):
    """カラム情報をリストに追加"""
    from cross_db_benchmark.benchmark_tools.trino.parse_filter import PredicateNode
    
    if isinstance(n, PredicateNode):
        columns.add((n.column, n.operator))
        for c in n.children:
            list_columns(c, columns)
    elif isinstance(n, dict):
        # filter_columnsがdict形式の場合
        if 'column' in n and 'operator' in n:
            columns.add((n['column'], n['operator']))
        if 'children' in n:
            for c in n['children']:
                list_columns(c, columns)
    else:
        # その他の型の場合はスキップ
        pass
