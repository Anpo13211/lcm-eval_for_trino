from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
from cross_db_benchmark.benchmark_tools.postgres.parse_filter import parse_filter
from cross_db_benchmark.benchmark_tools.postgres.utils import list_columns


def test_nested_condition():
    filter_cond = "((company_type_id >= 2) AND ((note)::text ~~ '%(2009)%'::text) AND (company_id >= 420) AND (company_id <= 1665) AND ((movie_id <= 1034200) OR ((movie_id <= 1793763) AND (movie_id >= 1728789) AND (movie_id <= 1786561))))"
    parse_tree = parse_filter(filter_cond)
    columns = set()
    list_columns(parse_tree, columns)
    assert columns == {(('company_type_id',), Operator.GEQ), (('movie_id',), Operator.LEQ), (('note',), Operator.LIKE),
                       (('company_id',), Operator.LEQ), (('movie_id',), Operator.GEQ), (('company_id',), Operator.GEQ),
                       (None, LogicalOperator.AND), (None, LogicalOperator.OR)}


def test_in_conditions():
    filter_cond = '(((name)::text ~~ \'%Michael%\'::text) AND ((name_pcode_cf)::text ~~ \'%A5362%\'::text) AND (((imdb_index)::text = ANY (\'{IV,II,III,I}\'::text[])) OR ((surname_pcode)::text = \'R5\'::text)))'
    parse_tree = parse_filter(filter_cond)
    columns = set()
    list_columns(parse_tree, columns)
    assert columns == {(('name_pcode_cf',), Operator.LIKE), (('surname_pcode',), Operator.EQ),
                       (('imdb_index',), Operator.IN), (('name',), Operator.LIKE),
                       (None, LogicalOperator.AND), (None, LogicalOperator.OR)}
