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
    
    if conv_to_dict:
        params = vars(params)
    
    if 'table' in params:
        tables.add(params['table'])
    if 'op_name' in params:
        operators.add(params['op_name'])
    if 'filter_columns' in params and not skip_columns:
        list_columns(params['filter_columns'], filter_columns)
    
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
    child_feat = [c.plan_parameters.get(feature_name) for c in p.children
                  if c.plan_parameters.get(feature_name) is not None]
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
