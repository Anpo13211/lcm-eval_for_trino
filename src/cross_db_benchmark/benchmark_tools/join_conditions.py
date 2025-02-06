from cross_db_benchmark.benchmark_tools.load_database import create_db_conn
from cross_db_benchmark.benchmark_tools.utils import load_schema_json


def check_schema_graph_recursively(table, visited_tables, visited_relationships, schema):
    if table in visited_tables:
        raise NotImplementedError("Schema is cyclic")
    visited_tables.add(table)

    for r_id, r in enumerate(schema.relationships):
        if r_id in visited_relationships:
            continue

        table_left, _, table_right, _ = r
        if table_left == table or table_right == table:
            visited_relationships.add(r_id)
            if table_left == table:
                check_schema_graph_recursively(table_right, visited_tables, visited_relationships, schema)
            elif table_right == table:
                check_schema_graph_recursively(table_left, visited_tables, visited_relationships, schema)


def check_join_conditions(dataset, database, db_name, database_conn_args, database_kwarg_dict):
    db_conn = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)

    # check if tables are a connected acyclic graph
    schema = load_schema_json(dataset)
    visited_tables = set()
    visited_relationships = set()
    check_schema_graph_recursively(schema.tables[0], visited_tables, visited_relationships, schema)
    assert len(visited_tables) == len(schema.tables), "Schema graph is not connected"
    print("Schema graph is acyclic and connected")

    print("Checking join conditions...")
    db_conn.test_join_conditions(dataset)
