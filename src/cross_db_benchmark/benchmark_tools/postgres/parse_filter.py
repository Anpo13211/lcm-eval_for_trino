import re
from typing import List

from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
from cross_db_benchmark.benchmark_tools.abstract.filter_parser import AbstractPredicateNode, AbstractFilterParser


class PostgresPredicateNode(AbstractPredicateNode):
    """PostgreSQL-specific predicate node implementation"""
    
    def parse_lines(self, parse_baseline=False):
        """
        self.text = "  I   love   Python  " -> keywords = ["I", "love", "Python"]
        """
        keywords = [w.strip() for w in self.text.split(' ') if len(w.strip()) > 0]
        if all([k == 'AND' for k in keywords]):
            self.operator = LogicalOperator.AND
        elif all([k == 'OR' for k in keywords]):
            self.operator = LogicalOperator.OR
        else:
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
                ('IS NULL', Operator.IS_NULL)
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
                    literal = self.text.split(split_str)[1]
                    column = self.text.split(split_str)[0]

                    # dirty hack to cope with substring calls in
                    is_substring = self.text.startswith('"substring')
                    if is_substring:
                        self.children[0] = self.children[0].children[0]

                    # current restriction: filters on aggregations (i.e., having clauses) are not encoded using
                    # individual columns
                    agg_ops = {'sum', 'min', 'max', 'avg', 'count'}
                    is_having = column.lower() in agg_ops or (len(self.children) == 1
                                                              and self.children[0].text in agg_ops)
                    if is_having:
                        column = None
                        self.children = []
                    else:

                        def recursive_inner(n):
                            # column names can be arbitrarily deep, hence find recursively
                            if len(n.children) == 0:
                                return n.text
                            return recursive_inner(n.children[0])

                        # sometimes column is in parantheses
                        if node_op == Operator.IN:
                            literal = self.children[-1].text
                            if len(self.children) == 2:
                                column = self.children[0].text
                            self.children = []
                        elif len(self.children) == 2:
                            literal = self.children[-1].text
                            column = recursive_inner(self)
                            self.children = []
                        elif len(self.children) == 1:
                            column = recursive_inner(self)
                            self.children = []
                        elif len(self.children) == 0:
                            pass
                        else:
                            raise NotImplementedError

                        # column and literal are sometimes swapped
                        type_suffixes = ['::bpchar']
                        if any([column.endswith(ts) for ts in type_suffixes]):
                            tmp = literal
                            literal = column
                            column = tmp.strip()

                        # additional features for special operators
                        # number of values for in operator
                        if node_op == Operator.IN:
                            filter_feature = literal.count(',')
                        # number of wildcards for LIKE
                        elif node_op == Operator.LIKE or node_op == Operator.NOT_LIKE:
                            filter_feature = literal.count('%')

                        break

            if parse_baseline:
                if node_op in {Operator.IS_NULL, Operator.IS_NOT_NULL}:
                    literal = None
                elif node_op == Operator.IN:
                    literal = literal.split('::')[0].strip("'").strip("{}")
                    literal = [c.strip('"') for c in literal.split('",')]
                else:
                    if '::text' in literal:
                        literal = literal.split("'::text")[0].strip("'")
                    elif '::bpchar' in literal:
                        literal = literal.split("'::bpchar")[0].strip("'")
                    elif '::date' in literal:
                        literal = literal.split("'::date")[0].strip("'")
                    elif '::time without time zone' in literal:
                        literal = literal.split("'::time")[0].strip("'")
                    elif '::double precision' in literal:
                        literal = float(literal.split("'::double precision")[0].strip("'"))
                    elif '::numeric' in literal:
                        literal = float(literal.split("'::numeric")[0].strip("'"))
                    elif '::integer' in literal:
                        literal = float(literal.split("'::integer")[0].strip("'"))
                    # column comparison. ignored.
                    elif re.match(r"\D\w*\.\D\w*", literal.replace('"', '').replace('\'', '').strip()):
                        literal = None
                    else:
                        try:
                            literal = float(literal.strip())
                        except ValueError:
                            print(
                                f"Could not parse literal {literal} (maybe a join condition? if so, this can be ignored)")
                            literal = None

            assert node_op is not None, f"Could not parse: {self.text}"

            self.column = column
            if column is not None:
                self.column = tuple(column.split('.'))
            self.operator = node_op
            self.literal = literal
            self.filter_feature = filter_feature

    def to_tree_rep(self, depth=0):
        rep_text = '\n' + ''.join(['\t'] * depth)
        rep_text += self.text

        for c in self.children:
            rep_text += c.to_tree_rep(depth=depth + 1)

        return rep_text


class PostgresFilterParser(AbstractFilterParser):
    """PostgreSQL-specific filter parser implementation"""
    
    def __init__(self):
        super().__init__(database_type="postgres")
    
    def create_predicate_node(self, text: str, children: List[AbstractPredicateNode]) -> PostgresPredicateNode:
        """Create PostgreSQL-specific predicate node"""
        return PostgresPredicateNode(text, children)


def parse_recursively(filter_cond, offset, _class=PostgresPredicateNode):
    """
    フィルター条件をパースするための再帰関数。
    ネストされた括弧や引用符を処理し、ノードのテキストと子ノードを生成します。

    params:
        filter_cond: フィルター条件の文字列
        offset: 現在解析している文字列内の位置
        _class: ノードのクラス
    returns:
        PredicateNode: パースされたノード
        int: 次の解析位置

    入力: "name = 'John' AND (age > 25)" の場合
    最終的なツリー構造：
        PredicateNode(
            text="name = 'John' AND ",
            children=[
                PredicateNode(
                    text="age > 25",
                    children=[]
                )
            ]
        )
    """
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
    parser = PostgresFilterParser()
    return parser.parse_filter(filter_cond, parse_baseline)


# Backward compatibility aliases
PredicateNode = PostgresPredicateNode
