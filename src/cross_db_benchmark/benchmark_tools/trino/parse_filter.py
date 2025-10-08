import re

from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator


class PredicateNode:
    """Trinoの述語ノードを表現するクラス"""
    
    def __init__(self, text, children):
        self.text = text
        self.children = children
        self.column = None
        self.operator = None
        self.literal = None
        self.filter_feature = None
    
    def __str__(self):
        return self.to_tree_rep(depth=0)
    
    def to_dict(self):
        return dict(
            column=self.column,
            operator=str(self.operator),
            literal=self.literal,
            literal_feature=self.filter_feature,
            children=[c.to_dict() for c in self.children]
        )
    
    def lookup_columns(self, plan, **kwargs):
        if self.column is not None:
            self.column = plan.lookup_column_id(self.column, **kwargs)
        for c in self.children:
            c.lookup_columns(plan, **kwargs)
    
    def parse_lines_recursively(self, parse_baseline=False):
        self.parse_lines(parse_baseline=parse_baseline)
        for c in self.children:
            c.parse_lines_recursively(parse_baseline=parse_baseline)
        
        # ベースライン解析の場合、統計的に予測困難な条件を除外
        if parse_baseline:
            self.children = [c for c in self.children if
                           c.operator in {LogicalOperator.AND, LogicalOperator.OR,
                                        Operator.IS_NOT_NULL, Operator.IS_NULL}
                           or c.literal is not None]
    
    def parse_lines(self, parse_baseline=False):
        """Trinoの述語を解析"""
        if not self.text:
            return
        
        keywords = [w.strip() for w in self.text.split(' ') if len(w.strip()) > 0]
        
        # 論理演算子の検出
        if all([k == 'AND' for k in keywords]):
            self.operator = LogicalOperator.AND
        elif all([k == 'OR' for k in keywords]):
            self.operator = LogicalOperator.OR
        else:
            # 比較演算子の検出
            repr_op = [
                ('= ANY', Operator.IN),
                ('=', Operator.EQ),
                ('>=', Operator.GEQ),
                ('>', Operator.GEQ),
                ('<=', Operator.LEQ),
                ('<', Operator.LEQ),
                ('<>', Operator.NEQ),
                ('~~', Operator.LIKE),
                ('!~~', Operator.NOT_LIKE),
                ('IS NOT NULL', Operator.IS_NOT_NULL),
                ('IS NULL', Operator.IS_NULL),
                # Trino特有の演算子
                ('IN', Operator.IN),
                ('NOT IN', Operator.NOT_IN),
                ('BETWEEN', Operator.BETWEEN),
                ('NOT BETWEEN', Operator.NOT_BETWEEN),
                ('LIKE', Operator.LIKE),
                ('NOT LIKE', Operator.NOT_LIKE),
                ('ILIKE', Operator.LIKE),  # Trinoの大文字小文字を区別しないLIKE
                ('NOT ILIKE', Operator.NOT_LIKE),
            ]
            
            node_op = None
            literal = None
            column = None
            filter_feature = 0
            
            for op_rep, op in repr_op:
                split_str = f' {op_rep} '
                self.text = self.text + ' '
                
                if split_str in self.text:
                    assert node_op is None
                    node_op = op
                    parts = self.text.split(split_str)
                    if len(parts) >= 2:
                        column = parts[0].strip()
                        literal = parts[1].strip()
                    
                    # カラム名の正規化
                    if column and '.' in column:
                        column = tuple(column.split('.'))
                    elif column:
                        column = (column,)
                    
                    # リテラル値の処理
                    if node_op == Operator.IN:
                        # IN句の処理
                        if literal.startswith('(') and literal.endswith(')'):
                            literal = literal[1:-1]
                        filter_feature = literal.count(',') + 1
                    elif node_op == Operator.LIKE or node_op == Operator.NOT_LIKE:
                        # LIKE句の処理
                        filter_feature = literal.count('%')
                    
                    break
            
            if parse_baseline:
                # ベースライン解析の場合のリテラル値処理
                if node_op in {Operator.IS_NULL, Operator.IS_NOT_NULL}:
                    literal = None
                elif node_op == Operator.IN:
                    # IN句の値をリストに変換
                    if isinstance(literal, str):
                        literal = literal.strip("'").strip("{}")
                        literal = [c.strip('"') for c in literal.split('",')]
                else:
                    # 型キャストの処理
                    if '::' in literal:
                        literal = literal.split('::')[0].strip("'")
                    
                    # 数値の変換
                    try:
                        if literal.replace('.', '').replace('-', '').isdigit():
                            literal = float(literal)
                    except ValueError:
                        pass
            
            assert node_op is not None, f"Could not parse: {self.text}"
            
            self.column = column
            self.operator = node_op
            self.literal = literal
            self.filter_feature = filter_feature
    
    def to_tree_rep(self, depth=0):
        """ツリー表現を生成"""
        rep_text = '\n' + ''.join(['\t'] * depth)
        rep_text += self.text
        
        for c in self.children:
            rep_text += c.to_tree_rep(depth=depth + 1)
        
        return rep_text


def parse_recursively(filter_cond, offset, _class=PredicateNode):
    """Trinoのフィルター条件を再帰的に解析"""
    escaped = False
    node_text = ''
    children = []
    
    while True:
        if offset >= len(filter_cond):
            return _class(node_text, children), offset
        
        if filter_cond[offset] == '(' and not escaped:
            child_node, offset = parse_recursively(filter_cond, offset + 1, _class=_class)
            children.append(child_node)
        elif filter_cond[offset] == ')' and not escaped:
            return _class(node_text, children), offset
        elif filter_cond[offset] == "'":
            escaped = not escaped
            node_text += "'"
        else:
            node_text += filter_cond[offset]
        offset += 1


def parse_filter(filter_cond, parse_baseline=False):
    """Trinoのフィルター条件を解析"""
    if not filter_cond:
        return None
    
    parse_tree, _ = parse_recursively(filter_cond, offset=0)
    assert len(parse_tree.children) == 1
    parse_tree = parse_tree.children[0]
    parse_tree.parse_lines_recursively(parse_baseline=parse_baseline)
    
    if parse_tree.operator not in {LogicalOperator.AND, LogicalOperator.OR, Operator.IS_NOT_NULL, Operator.IS_NULL} \
            and parse_tree.literal is None:
        return None
    if parse_tree.operator in {LogicalOperator.AND, LogicalOperator.OR} and len(parse_tree.children) == 0:
        return None
    
    return parse_tree
