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
        # $構文は関数呼び出しなので、AND/OR演算子の検出時には除外する
        keywords_for_logic = [k for k in keywords if not k.startswith('$')]
        
        # AND/ORの検出ロジック
        # 1. すべてがAND/ORの場合（例: ' AND  AND $like AND $like' → ['AND', 'AND', 'AND']）
        # 2. AND/ORのみで構成されている場合（$構文を除く）
        #    （例: ' AND $like' → ['AND'] → ANDとして認識）
        # 3. AND/ORと他のキーワードが混在する場合:
        #    - 'A AND $not' → ['A', 'AND'] → ANDが含まれているが、'A'も含まれているため、
        #      これは比較演算子として処理されるべき（ただし、実際にはこのようなケースは
        #      parse_recursivelyの設計上、親ノードは' AND $not'になることが多い）
        
        # すべてがAND/ORの場合
        if all([k == 'AND' for k in keywords_for_logic]) and len(keywords_for_logic) > 0:
            self.operator = LogicalOperator.AND
        elif all([k == 'OR' for k in keywords_for_logic]) and len(keywords_for_logic) > 0:
            self.operator = LogicalOperator.OR
        # AND/ORのみで構成されている場合（$構文を除く）
        # 例: ' AND $like' → ['AND']
        elif len(keywords_for_logic) > 0 and all([k in ['AND', 'OR'] for k in keywords_for_logic]):
            # ANDが含まれている場合はAND、ORが含まれている場合はOR
            # ただし、両方が含まれる場合は最初に見つかった方を使用（通常はANDが先）
            if 'AND' in keywords_for_logic:
                self.operator = LogicalOperator.AND
            elif 'OR' in keywords_for_logic:
                self.operator = LogicalOperator.OR
        # AND/ORと他のキーワードが混在する場合でも、AND/ORが含まれていれば論理演算子として扱う
        # 例: 'A AND $not' → ['A', 'AND'] → ANDが含まれているのでANDとして認識
        # （ただし、実際のTrinoフィルターではこのようなケースは稀）
        elif 'AND' in keywords_for_logic and len(keywords_for_logic) > 1:
            # ANDが含まれている場合、ANDとして扱う
            # 子ノードに'A'が含まれている場合、それは比較演算子の左辺として処理される
            self.operator = LogicalOperator.AND
        elif 'OR' in keywords_for_logic and len(keywords_for_logic) > 1:
            # ORが含まれている場合、ORとして扱う
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
                    # CAST(...) AS char(3)形式の処理
                    if column and 'CAST' in column.upper():
                        cast_match = re.search(r'CAST\s*\(([^)]+)\s+AS\s+[^)]+\)', column, re.IGNORECASE)
                        if cast_match:
                            column = cast_match.group(1).strip()
                    
                    if column and '.' in column:
                        column = tuple(column.split('.'))
                    elif column:
                        column = (column,)
                    
                    # リテラル値の処理
                    if node_op == Operator.IN:
                        # IN句の処理
                        if literal.startswith('(') and literal.endswith(')'):
                            literal = literal[1:-1]
                        
                        # char(3) 'val'形式の値を抽出
                        values = []
                        for val_match in re.finditer(r"(?:varchar|bigint|integer|double|char\([^)]+\))\s*'([^']*)'", literal):
                            values.append(val_match.group(1))
                        
                        if values:
                            literal = values
                            filter_feature = len(values)
                        else:
                            # フォールバック: カンマ区切りのカウント
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
                # または IN (char(3) 'val1', char(3) 'val2', ...)のパース
                elif ' IN ' in self.text or self.text.strip().startswith('varchar ') or self.text.strip().startswith('char('):
                    # varchar 'val1', varchar 'val2', char(3) 'val1', ... の形式（IN句から抽出された値のリスト）
                    values = []
                    # char(3)形式も含む、より包括的な正規表現
                    for val_match in re.finditer(r"(?:varchar|bigint|integer|double|char\([^)]+\))\s*'([^']*)'", self.text):
                        values.append(val_match.group(1))
                    
                    if values:
                        # カラム名は前のコンテキストから推測する必要がある
                        # ただし、単独のリストとして扱う場合は、ノードにカラム情報がない
                        node_op = Operator.IN
                        literal = values
                        column = None  # カラム名は別途設定される必要がある
                
                # CAST(...) AS char(3) 形式の処理（IN句の左側）
                elif 'CAST' in self.text.upper() and ' AS ' in self.text.upper():
                    # CAST式は通常、親ノード（IN句全体）で処理される
                    # ここではカラム名を抽出する試み
                    cast_match = re.search(r'CAST\s*\(([^)]+)\s+AS\s+[^)]+\)', self.text, re.IGNORECASE)
                    if cast_match:
                        column_str = cast_match.group(1).strip()
                        # カラム名の正規化
                        if '.' in column_str:
                            column = tuple(column_str.split('.'))
                        else:
                            column = (column_str,)
                        # CAST式自体はIN句の一部として扱われるため、ここでは演算子を設定しない
                        # 親ノードがIN句として処理されることを期待
            
            if node_op is None:
                # 警告を出す前に、意味のない子ノードかどうかをチェック
                # 子ノード（単独では意味を持たない部分）の警告は抑制
                text_stripped = self.text.strip()
                
                # より包括的な意味のない子ノードのチェック
                # これらは親ノードで処理されるため、警告は抑制
                looks_like_alias_cast = re.match(
                    r'^[A-Za-z0-9_."]+\s+AS\s+[A-Za-z0-9_()]+$', text_stripped, re.IGNORECASE
                )

                # AND演算子と$構文の組み合わせをチェック（例: ' AND  AND $like AND $like'）
                # AND、空白、$構文のみを含む場合は警告を抑制
                cleaned_text = text_stripped.replace('$like', '').replace('$not', '').replace('like', '').replace('not', '')
                has_only_and_and_dollar = (
                    cleaned_text.replace(' ', '').replace('AND', '').replace('$', '') == '' or
                    re.match(r'^[\sAND\$]+$', cleaned_text, re.IGNORECASE)
                )
                
                is_meaningless_child = (
                    # 単純な数値のみ（IN句の値の一部など）
                    text_stripped.isdigit() or
                    # 型キャストの型指定部分のみ（例: "char(3)"）
                    re.match(r'^[a-z]+\([^)]+\)$', text_stripped, re.IGNORECASE) or
                    # 列名＋AS＋型名（CAST/別名の一部）
                    looks_like_alias_cast or
                    # Trino特有の構文（$like, $notなどで始まる、または含む）
                    text_stripped.startswith('$') or
                    '$' in text_stripped or
                    # AND演算子のみ、またはAND演算子と$構文の組み合わせ
                    has_only_and_and_dollar or
                    # AND演算子のみの形式（空白とANDのみ）
                    re.match(r'^(\s*AND\s*)+$', text_stripped, re.IGNORECASE) or
                    # ' AND $' で始まる形式（$not, $likeの前のAND部分）
                    re.match(r'^\s*AND\s+\$', text_stripped, re.IGNORECASE) or
                    # ' AND AND $...' のような形式（複数のANDと$構文）
                    re.match(r'^(\s*AND\s*)+.*\$', text_stripped, re.IGNORECASE) or
                    # 不完全な文字列リテラルのリスト（IN句の一部）
                    (text_stripped.startswith("char") and "'" in text_stripped and '=' not in text_stripped) or
                    # カラム名のみ（AS句の一部など）
                    (text_stripped.endswith(' AS char') and '=' not in text_stripped) or
                    # 関数呼び出しの引数（カンマが含まれているが、比較演算子が含まれていない）
                    # 例: 'ime_upravna_enota, LikePattern "[%Bis%trica%]"'
                    (',' in text_stripped and not any(op in text_stripped for op in ['=', '>', '<', '!', '>=', '<=', '<>'])) or
                    # LikePatternを含む（関数呼び出しの引数の一部）
                    ('LikePattern' in text_stripped) or
                    # プレースホルダーや省略記号（テストケースなど）
                    (text_stripped == '...' or text_stripped.strip() == '...') or
                    # 空白のみ、または空白と演算子だけ
                    (not text_stripped or re.match(r'^[\s\&\$\|]+$', text_stripped))
                )
                
                if not is_meaningless_child:
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
