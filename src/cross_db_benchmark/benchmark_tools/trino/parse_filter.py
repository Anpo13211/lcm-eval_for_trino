import re
from typing import List

from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
from cross_db_benchmark.benchmark_tools.abstract.filter_parser import AbstractPredicateNode, AbstractFilterParser


class TrinoPredicateNode(AbstractPredicateNode):
    """Trino-specific predicate node implementation"""
    
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
            # 比較演算子の検出（長いパターンを先に配置）
            repr_op = [
                ('NOT BETWEEN', Operator.NOT_BETWEEN),
                ('NOT ILIKE', Operator.NOT_LIKE),
                ('NOT LIKE', Operator.NOT_LIKE),
                ('NOT IN', Operator.NOT_IN),
                ('NOT NULL', Operator.IS_NOT_NULL),
                ('IS NOT NULL', Operator.IS_NOT_NULL),
                ('IS NULL', Operator.IS_NULL),
                ('= ANY', Operator.IN),
                ('BETWEEN', Operator.BETWEEN),
                ('ILIKE', Operator.LIKE),  # Trinoの大文字小文字を区別しないLIKE
                ('LIKE', Operator.LIKE),
                ('IN', Operator.IN),
                ('>=', Operator.GEQ),
                ('<=', Operator.LEQ),
                ('<>', Operator.NEQ),
                ('!~~', Operator.NOT_LIKE),
                ('~~', Operator.LIKE),
                ('>', Operator.GEQ),
                ('<', Operator.LEQ),
                ('=', Operator.EQ),
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
            
            # 特殊な構文のパース（TrinoのLikePattern、IN句など）
            if node_op is None:
                # column, LikePattern 'pattern'のパース（既に$like()が削除されている形式）
                like_pattern_match = re.search(r'^([^,]+),\s*LikePattern\s*\'([^\']+)\'', self.text)
                if like_pattern_match:
                    column = like_pattern_match.group(1).strip()
                    pattern = like_pattern_match.group(2)
                    # Trinoのパターンは[]で囲まれていることが多い
                    if pattern.startswith('[') and pattern.endswith(']'):
                        pattern = pattern[1:-1]
                    # %をそのまま使用してSQL LIKE形式に変換
                    # %P%N% -> %P%N%のようなパターン
                    if not pattern.startswith('%'):
                        pattern = f'%{pattern}%'
                    node_op = Operator.LIKE
                    literal = pattern
                    column = (column,)
                
                # IN (varchar 'val1', varchar 'val2', ...)のパース
                elif ' IN ' in self.text or self.text.strip().startswith('varchar '):
                    # varchar 'val1', varchar 'val2', ... の形式（IN句から抽出された値のリスト）
                    values = []
                    for val_match in re.finditer(r"(?:varchar|bigint|integer|double)\s*'([^']*)'", self.text):
                        values.append(val_match.group(1))
                    
                    if values:
                        # カラム名は前のコンテキストから推測する必要がある
                        # ただし、単独のリストとして扱う場合は、ノードにカラム情報がない
                        node_op = Operator.IN
                        literal = values
                        column = None  # カラム名は別途設定される必要がある
            
            if node_op is None:
                print(f"Warning: Could not parse: '{self.text}'")
                # Set default operator for unparseable text
                node_op = Operator.EQ
                column = None
                literal = self.text
            
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


class TrinoFilterParser(AbstractFilterParser):
    """Trino-specific filter parser implementation"""
    
    def __init__(self):
        super().__init__(database_type="trino")
    
    def create_predicate_node(self, text: str, children: List[AbstractPredicateNode]) -> TrinoPredicateNode:
        """Create Trino-specific predicate node"""
        return TrinoPredicateNode(text, children)


def parse_recursively(filter_cond, offset, _class=TrinoPredicateNode):
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
    """Legacy function for backward compatibility"""
    parser = TrinoFilterParser()
    return parser.parse_filter(filter_cond, parse_baseline)


# Backward compatibility aliases
PredicateNode = TrinoPredicateNode
