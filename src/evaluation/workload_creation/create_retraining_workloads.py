import json
from cross_db_benchmark.benchmark_tools.generate_workload import generate_workload
from classes.paths import LocalPaths

if __name__ == '__main__':
    # Read out column statistics to later filter out non-numerical columns.
    imdb_path = LocalPaths().dataset_path / "imdb" / "column_statistics.json"
    with open(imdb_path, 'r') as f:
        imdb_schema = json.load(f)
    print(imdb_schema)
    for workload in ["imdb"]:
        # Some queries will have timeouts, so generating more and delete again
        index_target_path = LocalPaths().workloads / 'retraining' / workload / 'index_retraining.sql'
        seq_target_path = LocalPaths().workloads / 'retraining' / workload / 'seq_retraining.sql'
        index_queries = []
        seq_queries = []
        cap = 1000

        queries = generate_workload(dataset=workload,
                                    target_path=index_target_path,
                                    num_queries=2000,
                                    min_no_predicates=1,
                                    max_no_predicates=2,
                                    max_no_aggregates=0,
                                    max_no_group_by=0,
                                    max_no_joins=0,
                                    max_cols_per_agg=1,
                                    seed=1,
                                    force=True)

        failing_queries = []
        for query in queries:
            try:
                table = query.split('FROM')[1].split(' ')[1].replace('"', '')
                filter_column = query.split('WHERE')[1].split(' ')[1].replace('"', '')
                filter_column = filter_column.split('.')[1]

                datatype = imdb_schema[table][filter_column]['datatype']
                if datatype in ['int', 'float']:
                    index_name = f"idx_{table}_{filter_column}"
                    seq_queries.append(f"/*+SeqScan({table})*/ " + query)
                    index_queries.append(f"/*+IndexScan({table} {index_name})*/ " + query)

            except IndexError as e:
                print("Erroneous query: ", {query})

        with open(index_target_path, "w") as text_file:
            text_file.write('\n'.join(index_queries[0:cap]))

        with open(seq_target_path, "w") as text_file:
            text_file.write('\n'.join(seq_queries[0:cap]))

        print(f'Generated {len(seq_queries[0:cap]) + len(index_queries[0:cap])} queries for {workload}')
