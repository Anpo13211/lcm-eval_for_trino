import itertools
import json
import math
import os
from pathlib import Path
import random
from typing import List, Optional, Tuple, Set

import sqlparse
import tqdm
from sqlparse.sql import Statement, Function, Where, Comparison, TokenList, Token

from cross_db_benchmark.benchmark_tools.generate_workload import generate_workload
from classes.paths import LocalPaths


class Operator:
    eq = "="
    geq = ">="


def create_scan_query(metadata: dict, workload: dict) -> Optional[List[str]]:
    """Creating simple sequential scan queries"""
    statements = []

    if not (metadata["datatype"] == "categorical" and workload["operator"] in [Operator.geq]):
        if workload["aggregation"]:
            agg_expression = f'{workload["aggregation"]}(*)'
        else:
            agg_expression = '*'
        for literal in metadata["literals"]:
            tab_name = metadata['tab_name'].replace('"', '')
            sql_statement = (f"/*+SeqScan({tab_name})*/ "
                             f"SELECT {agg_expression} "
                             f"FROM {metadata['tab_name']} "
                             f"WHERE {metadata['tab_name']}.{metadata['col_name']}{workload['operator']}'{literal}';")
            statements.append(sql_statement)
    return statements


def get_metadata(column_vals: dict, tab_name: str, col_name: str) -> Optional[dict]:
    """
    Get relevant metadata out of column statistics that we need to filter out unimportant columns
    """
    dtype = column_vals["datatype"]
    result = dict(tab_name=tab_name, col_name=col_name, datatype=dtype, nan_ratio=column_vals["nan_ratio"])
    if dtype in ["float", "int", "double"]:
        min, max = column_vals["min"], column_vals["max"]
        if not math.isnan(min) and not math.isnan(max):
            # Cast min and max value
            if dtype == "int":
                min, max = int(min), int(max)

            elif dtype == "float":
                # assert (min * 10) % 10 == 0, f'Value cant be cast to int properly {min}'
                # assert (max * 10) % 10 == 0, f'Value cant be cast to int properly {max}'
                min, max = int(min), int(max)
            else:
                raise ValueError("Unexpected datatype")

            # Compute step size for literal iteration
            if (max - min) > 100:
                step = math.ceil((max - min) / 100)
            else:
                step = 1
            literals = list(range(min, max, step))
            assert len(literals) <= 100, f'Length of literals for {col_name} is {len(literals)}'
            result.update(dict(max=max, min=min, step=step, literals=literals))
            return result

    elif dtype == "categorical":
        num_unique = column_vals["num_unique"]
        if num_unique <= 100:
            literals = list(column_vals["unique_vals"])
        else:
            literals = list(column_vals["unique_vals"])[0:100]

        result.update(dict(literals=literals, num_unique=column_vals["num_unique"]))
        return result


def filter_metadata(mdata: dict) -> Optional[dict]:
    """Filtering out columns of dataset that are not of interest"""

    if "_id" not in mdata["col_name"] and mdata["col_name"] != "id" and float(mdata['nan_ratio'] < 0.9) and len(
            mdata["literals"]) > 5:
        if mdata["datatype"] == "categorical":
            print(f'Table: {mdata["tab_name"]: <50}'
                  f'Column: {mdata["col_name"]: <20}'
                  f'Num_unique: {mdata["num_unique"] : <41}',
                  f'Num_statements: {len(mdata["literals"]): <6}'
                  f'Nan_ratio: {round(mdata["nan_ratio"], 2): >3}')
        elif mdata["datatype"] in ["float", "int", "double"]:
            print(f'Table: {mdata["tab_name"]: <50}'
                  f'Column: {mdata["col_name"]: <20}'
                  f'Min: {mdata["min"]: <12}'
                  f'Max: {mdata["max"]: <12} '
                  f'Step: {mdata["step"]: <12} '
                  f'Num_statements: {len(mdata["literals"]): <6}'
                  f'Nan_ratio: {round(mdata["nan_ratio"], 2): >3}')
        return mdata


def extract_unique_join_workloads(file_path: Path,
                                  workload_name: str,
                                  min_join_nr: int,
                                  max_join_nr: int,
                                  len_groups: int,
                                  min_preds_nr: int = 0,
                                  check_unique: bool = True) -> list[[int, Tuple[Statement]]]:
    """At first read all queries and extract subselection of these by
    optionally looking at unique sets of join tables"""

    distinct_tables, resulting_queries = list(), dict()
    with open(file_path, 'r') as file:
        for line_nr, line in tqdm.tqdm(enumerate(file)):
            if line_nr < 10000:

                # Extract query first
                parsed_query = sqlparse.parse(line)
                assert len(parsed_query) == 1, "Multiple queries found per line"
                parsed_query = parsed_query[0]

                # Get distinct join tables from query
                join_tables = get_query_tables(parsed_query)
                predicates = get_preds_from_query(parsed_query)

                if join_tables and "+" not in parsed_query.normalized:

                    if len(join_tables) >= 1 and len(predicates) >= min_preds_nr:
                        if min_join_nr <= len(join_tables) <= max_join_nr:
                            # look if group (=len of tables) is actually required
                            group = str(len(join_tables))
                            if group not in resulting_queries:
                                resulting_queries[group] = []

                            if len(resulting_queries[group]) < len_groups:
                                # For join order experiments we are only interested in unique sets of tables
                                # For others we do not care
                                if (check_unique and join_tables not in distinct_tables) or (not check_unique):
                                    distinct_tables.append(join_tables)
                                    query_id = workload_name + "_" + str(line_nr)
                                    resulting_queries[group].append((query_id, parsed_query))

        print(f'Extracted {len(resulting_queries)} queries from {file_path}')
    return [q for v in resulting_queries.values() for q in v]


def get_query_tables(query: Tuple[Statement], query_tables: Set[TokenList] = None, depth: int = 0) -> set[Token]:
    """Given a query, this method will extract the unique set of join tables recursively"""
    if query_tables is None:
        query_tables = set()
    for token in query:
        if not isinstance(token, (Function, Where, Comparison)) and ' as ' not in token.value:
            if str(token.ttype) in ["Token.Name", "Token.Literal.String.Symbol"]:
                query_tables.add(token)
            if hasattr(token, "tokens"):
                get_query_tables(token.tokens, query_tables, depth=depth + 1)
    if depth == 0:
        return list(sorted(query_tables, key=lambda tables: tables.normalized))
    else:
        return query_tables


def get_preds_from_query(query: Tuple[Statement]):
    comparisons = []
    for token in query:
        if isinstance(token, Where):
            for subtoken in token.tokens:
                if type(subtoken) == Comparison:
                    comparisons.append(subtoken)
    return comparisons


def write_wl_to_file(workload: List[str], file_path: Path) -> None:
    file_path = str(file_path).replace('"', '')
    with open(file_path, 'w') as file:
        for string in workload:
            file.write(string + '\n')
    print("Written workload to file {}".format(file_path))


def add_join_order_brackets(tables):
    if len(tables) == 1:
        return [tables[0]]
    elif len(tables) == 2:
        return [f"({tables[0]} {tables[1]})"]
    else:
        results = []
        for i in range(1, len(tables)):
            left_results = add_join_order_brackets(tables[:i])
            right_results = add_join_order_brackets(tables[i:])
            for left in left_results:
                for right in right_results:
                    results.append(f"({left} {right})")
        return results


def catalan_number(n: int) -> int:
    return math.factorial(2 * n) // math.factorial(n)


def permutate_join_orders(query: Tuple[Statement], max_permutations: int = None, exhaustive: bool = False) -> List[str]:
    """
    Generates permutations of a given SQL query with different leading hints. This function takes a SQL query and
    generates permutations of the same query by rearranging the order of tables in the join clause. The leading hints
    in the SQL query are adjusted according to the new order of tables in each permutation. The number of
    permutations can be limited by setting the `max_permutations` parameter.

    Parameters: query (Tuple[Statement]): The original SQL query as a tuple of sqlparse.sql.Statement objects.
    max_permutations (int, optional): The maximum number of permutations to generate. If not provided, all possible
    permutations will be generated.
    exhaustive (bool, optional): If set to True, full permutations are generated using all
    tables in the query. If set to False, permutations are generated using subsets of the tables. Default is False.
    """
    tables = get_query_tables(query)
    permutations = list(itertools.permutations(tables))
    random.Random(0).shuffle(permutations)

    if max_permutations is None:
        max_permutations = len(permutations)

    resulting_queries = []
    for permutation in permutations[0:max_permutations]:
        table_names = [table.value.replace('"', "") for table in permutation]
        join_orders = add_join_order_brackets(table_names) if exhaustive else [table_names]

        for join_order in join_orders:
            hints = f'Leading({"".join(join_order)})'
            resulting_queries.append(f'/*+{hints}*/ {query}')

    assert len(resulting_queries) == len((set(resulting_queries))), "Queries contain duplicates"

    if exhaustive:
        assert len(resulting_queries) == catalan_number(len(tables) - 1), \
            (f"Number of permutations incorrect, "
             f"found: {len(set(resulting_queries))}, "
             f"expected {(catalan_number(len(tables) - 1))})")
    else:
        assert len(resulting_queries) == min(max_permutations, math.factorial(len(tables))), \
            (f"Number of permutations incorrect, "
             f"found: {len(set(resulting_queries))}, "
             f"expected {min(max_permutations, math.factorial(len(tables)))}")
    return resulting_queries


def permutate_physical_implementation(initial_query: Tuple[Statement]) -> List[str]:
    """
    Create and append hints for different join implementations
    """
    tables = get_query_tables(initial_query)
    resulting_queries = []
    for join_impl in ["HashJoin", "MergeJoin", "NestLoop"]:
        hints = []

        # Add join hints iteratively according to tables.
        for i in range(2, len(tables) + 1):
            subset_tables = tables[0:i]
            sorted_subset_tables = sorted(subset_tables, key=lambda x: x.value)
            table_names = [table.value.replace('"', "") for table in sorted_subset_tables]
            hints.append(f'{join_impl}({" ".join(table_names)})')

        hints = " ".join(hints)
        resulting_queries.append(f'/*+{hints}*/ {initial_query}')

    return resulting_queries


def generate_join_order_wls(workload_path: Path, target_folder: Path, workload_name: str,
                            num_queries_per_num_tables: int = 3, min_join_nr: int = 3, max_join_nr: int = 5,
                            exhaustive: bool = False, check_unique: bool = True) -> None:
    """Generate join order permutations for a given workload"""

    print(f'Generating join order permutations for workload {workload_name}')
    unique_queries = extract_unique_join_workloads(workload_name=workload_name,
                                                   file_path=workload_path,
                                                   min_join_nr=min_join_nr,
                                                   max_join_nr=max_join_nr,
                                                   len_groups=num_queries_per_num_tables,
                                                   check_unique=check_unique)

    for (query_id, query) in unique_queries:
        sql_statements = permutate_join_orders(query=query, max_permutations=None, exhaustive=exhaustive)
        os.makedirs(target_folder, exist_ok=True)
        file_path = target_folder / str(query_id + ".sql")
        write_wl_to_file(sql_statements, file_path)
    print(f'Written {len(unique_queries)} workloads to {target_folder}')

def generate_physical_plan_wls(workload_path: Path, target_folder: Path, workload_name: str, nr_of_tables: int = 2,
                               num_queries: int = 100):
    """
    Generate join implementation permutations for a given workload
    """
    print(f'Generating join order permutations for workload {workload_name}')

    unique_queries = extract_unique_join_workloads(workload_name=workload_name,
                                                   file_path=workload_path,
                                                   min_join_nr=nr_of_tables,
                                                   max_join_nr=nr_of_tables,
                                                   min_preds_nr=2,  # Otherwise, MSCN has issues
                                                   len_groups=num_queries,
                                                   check_unique=False)
    for (query_id, query) in unique_queries:
        sql_statements = permutate_physical_implementation(query)
        os.makedirs(target_folder, exist_ok=True)
        file_path = target_folder / str(query_id + ".sql")
        write_wl_to_file(sql_statements, file_path)


def generate_filter_workloads(schema_path: Path, target_folder: Path, workload_name: str) -> None:
    """Generate simple sequential scan queries along a filter literal"""

    print(f'Generating cardinality plans for workload {workload_name}')
    filter_workloads = [
        dict(name="seq_point_filter", aggregation=None, operator=Operator.eq),
        dict(name="range_point_filter", aggregation=None, operator=Operator.geq)
        # dict(name="agg_point_filter", aggregation=Agg.count, operator=Operator.eq),
        # dict(name="range_agg_point_filter", aggregation=Agg.count, operator=Operator.geq)
    ]

    # Collect all tables and columns into a common list that is later randomized
    all_schemas = dict()
    with open(schema_path) as f:
        schema = json.load(f)
        for table_name, table_values in schema.items():
            if workload_name == "baseball":
                table_name = '"' + table_name + '"'
            for column_name, column_values in table_values.items():
                if workload_name == "baseball":
                    column_name = '"' + column_name + '"'
                if table_name not in all_schemas:
                    all_schemas[table_name] = dict()
                all_schemas[table_name][column_name] = column_values

    tables = list(all_schemas.keys())
    random.shuffle(tables)
    selected_tables = tables[0:4] # assuming 4 way join first


    counter = 0
    for table_name, table_values, column_name, column_values in all_schemas:
        metadata = get_metadata(col_name=column_name, tab_name=table_name, column_vals=column_values)
        if metadata and counter < 20:
            filtered_metadata = filter_metadata(metadata)
            if filtered_metadata:
                counter += 1
                for workload in filter_workloads:
                    sql_statements = create_scan_query(metadata=filtered_metadata, workload=workload)
                    if sql_statements:
                        os.makedirs(target_folder / workload["name"], exist_ok=True)
                        file_path = target_folder / workload["name"] / str(table_name + "." + column_name + ".sql")
                        write_wl_to_file(sql_statements, file_path)


def is_sortable(obj) -> bool:
    cls = obj.__class__
    return cls.__lt__ != object.__lt__ or \
        cls.__gt__ != object.__gt__


def split_list_equally(a: list, n: int):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == '__main__':
    # Some queries will have timeouts, so generating more and delete again

    join_order_selected = False       # Having 6 workloads, longer runtime
    join_order_full = False          # ALl enumerations of JOB-LIGHT

    physical_plan = False
    cardinality_plan = False
    num_predicates = True

    if join_order_full:
        # Generate full plan enumerations to test left-deep, right-deep and bushy plans
        generate_join_order_wls(
            workload_path=LocalPaths().code / "experiments" / "evaluation_workloads" / "imdb" / "job-light.sql",
            target_folder=LocalPaths().workloads / 'evaluation' / 'imdb' / 'join_order_full',
            num_queries_per_num_tables=100,
            workload_name="job_light",
            min_join_nr=1,
            check_unique=False,
            exhaustive=True),

    if join_order_selected:
        # Generating workloads - 1. join order
        generate_join_order_wls(
            workload_path=LocalPaths().code / "experiments" / "evaluation_workloads" / "imdb" / "job-light.sql",
            target_folder=LocalPaths().workloads / 'evaluation' / 'imdb' / 'join_order_selected',
            num_queries_per_num_tables=5,
            workload_name="job_light",
            exhaustive=True)

        generate_join_order_wls(
            workload_path=LocalPaths().workloads / "training" / "tpc_h" / "workload_100_s1.sql",
            target_folder=LocalPaths().workloads / 'evaluation' / 'tpc_h' / 'join_order_selected',
            workload_name="tpc_h",
            num_queries_per_num_tables=6,
            exhaustive=True)

        generate_join_order_wls(
            workload_path=LocalPaths().workloads / "training" / "baseball" / "workload_100k_s1.sql",
            target_folder=LocalPaths().workloads / 'evaluation' / 'baseball' / 'join_order_selected',
            workload_name="baseball",
            num_queries_per_num_tables=4,
            exhaustive=True)

    if physical_plan:
        # Generating workloads - 2. physical plan selection
        generate_physical_plan_wls(
            workload_path=Path("../../experiments/evaluation_workloads/imdb/scale.sql"),
            target_folder=LocalPaths().workloads / 'evaluation' / 'imdb' / 'physical_plan',
            workload_name="scale")

        generate_physical_plan_wls(
            workload_path=Path("../../data/workloads/tpc_h/workload_100k_s1.sql"),
            target_folder=LocalPaths().workloads / 'evaluation' / 'tpc_h' / 'physical_plan',
            workload_name="tpc_h")

        generate_physical_plan_wls(
            workload_path=Path("../../data/workloads/baseball/workload_100k_s1.sql"),
            target_folder=LocalPaths().workloads / 'evaluation' / 'baseball' / 'physical_plan',
            workload_name="baseball")

    if cardinality_plan:
        # Generating workloads - 3. cardinality plan
        generate_filter_workloads(
            schema_path=Path("./cross_db_benchmark/datasets/imdb/column_statistics.json"),
            target_folder=LocalPaths().workloads / 'evaluation' / 'imdb' / 'cardinality_plan',
            workload_name="imdb")

        generate_filter_workloads(
            schema_path=Path("./cross_db_benchmark/datasets/tpc_h/column_statistics.json"),
            target_folder=LocalPaths().workloads / 'evaluation' / 'tpc_h' / 'cardinality_plan',
            workload_name="tpc_h")

        generate_filter_workloads(
            schema_path=Path("./cross_db_benchmark/datasets/baseball/column_statistics.json"),
            target_folder=LocalPaths().workloads / 'evaluation' / 'baseball' / 'cardinality_plan',
            workload_name="baseball")

    if num_predicates:
        for workload in ["imdb", "tpc_h", "baseball"]:
            for num_predicates in range(0, 10):
                generate_workload(dataset=workload,
                                  target_path=LocalPaths().workloads / 'evaluation' / workload / 'num_predicates' / f'{num_predicates}_attributes.sql',
                                  num_queries=100,
                                  min_no_predicates=num_predicates,
                                  max_no_predicates=num_predicates,
                                  max_no_aggregates=1,
                                  max_no_group_by=0,
                                  max_no_joins=3,
                                  max_cols_per_agg=2,
                                  seed=1,
                                  force=True)
