import argparse
import json

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.load_database import create_db_conn

def get_table_rows(db_name):
    db_conn = create_db_conn(database=DatabaseSystem.POSTGRES,
                             db_name=db_name,
                             database_conn_args=dict(user="postgres", password="bM2YGRAX*bG_QAilUid§2iD", host="localhost"),
                             database_kwarg_dict=dict())

    sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    result = db_conn.get_result(sql, db_created=True)

    table_rows = {}
    for table in result:
        table_name = table[0]
        row_count = db_conn.get_result(f"SELECT COUNT(*) FROM {table_name}",  db_created=True)
        if row_count != 0:
            table_rows[table_name] = row_count[0][0]
    with open(f'{db_name}.json', "w") as outfile:
        json.dump(table_rows, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", default=None)
    args = parser.parse_args()
    # Update these values with your PostgreSQL connection information
    get_table_rows(args.db_name)

    """
    try:
        table_rows = (db_conn)
        for table, rows in table_rows.items():
            print(f"Table: {table}, Rows: {rows}")
    except psycopg2.Error as e:
        print("Error connecting to PostgreSQL:", e)
    finally:
        if connection:
            connection.close()
    """
if __name__ == "__main__":
    main()