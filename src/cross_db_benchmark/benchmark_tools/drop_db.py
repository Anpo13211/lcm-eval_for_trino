from cross_db_benchmark.benchmark_tools.load_database import create_db_conn


def drop_db(database, db_name, database_conn_args, database_kwarg_dict):
    db_conn = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)
    db_conn.drop()
