import json
import random
from typing import List

from classes.paths import LocalPaths


def get_longest_tables(database: str, num_tables: int = 5) -> List[str]:
    with open(LocalPaths().dataset_path / database / 'table_lengths.json', 'r') as f:
        table_lengths = json.load(f)
        print(table_lengths)

        # Get the longest n tables
        longest_tables = sorted(table_lengths.keys(), key=lambda x: x[1], reverse=False)[:num_tables]
        return longest_tables


def get_columns(table: str, database: str) -> dict:
    path = LocalPaths().dataset_path / database / 'column_statistics.json'
    assert path.exists(), f"Path {path} does not exist"
    with open(LocalPaths().dataset_path / database / 'column_statistics.json', 'r') as f:
        table_columns = json.load(f)
        return dict(table_columns.get(table, ))


if __name__ == '__main__':
    #database, num_cols, num_tables = "baseball", 10, 5
    # database, num_cols, num_tables = "imdb", 10, 9
    database, num_cols, num_tables = "tpc_h_pk", 5, 5

    target_path = LocalPaths().evaluation_workloads / database
    longest_tables = get_longest_tables(num_tables=num_tables, database=database)
    index_statements = []
    for table in longest_tables:
        print("Table ", table)
        columns = get_columns(table, database)
        if columns:
            # iterate through first n columns
            # shuffle columns with fixed seed
            column_keys = list(columns.keys())
            # shuffle(column_keys)
            random.Random(0).shuffle(column_keys)
            columns_keys = column_keys[:num_cols]

            for column in columns_keys:
                column_stats = columns[column]
                if column_stats['datatype'] != 'misc': # and column_stats['nan_ratio'] < 0.5:
                    if column_stats['datatype'] in ['int', 'float']:
                        print(column)
                        index_name = f"idx_{table}_{column}"
                        percentiles = column_stats['percentiles']

                        index_statements.append(f"CREATE INDEX {index_name} ON {table} ({column});")
                        # Only allow if we have unique percentiles to avoid heavily skewed data
                        seq_scan_statements = []
                        index_scan_statements = []
                        if len(percentiles) == len(set(percentiles))  and "id" not in column:
                            if database == "baseball":
                                # Add " to the column name
                                col = f'"{column}"'
                            else:
                                col = column
                            for percentile in percentiles:
                                if column_stats['datatype'] == 'int':
                                    percentile = int(percentile)
                                    print(column, column_stats['datatype'], percentile)

                                seq_scan_statements.append(f"/*+SeqScan({table})*/ SELECT * FROM {table} WHERE {col} >= {percentile};")
                                index_scan_statements.append(f"/*+IndexScan({table} {index_name.lower()})*/ SELECT * FROM {table} WHERE {col} >= {percentile};")

                            with open(target_path / f'scan_costs_percentiles/seq.{table}.{column}.sql', 'w') as f:
                                f.write('\n'.join(seq_scan_statements))
                            with open(target_path / f'scan_costs_percentiles/index.{table}.{column}.sql', 'w') as f:
                                f.write('\n'.join(index_scan_statements))

    with open(target_path / 'index_creation_statements.sql', 'w') as f:
        f.write('\n'.join(index_statements))
