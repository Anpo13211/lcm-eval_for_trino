import os

from cross_db_benchmark.benchmark_tools.utils import load_column_statistics, load_schema_json
from cross_db_benchmark.datasets.datasets import database_list
from cross_db_benchmark.meta_tools.scale_dataset import get_dataset_size
from training.training.checkpoint import save_csv


def generate_dataset_statistics(target, data_dir):
    dataset_stats = []
    for db in database_list:
        dataset = db.db_name

        column_stats = load_column_statistics(dataset, namespace=False)
        schema = load_schema_json(dataset)

        size_gb = get_dataset_size(os.path.join(data_dir, db.source_dataset), schema)
        size_gb *= db.scale

        curr_stats = dict(
            dataset_name=db.db_name,
            no_tables=len(schema.tables),
            no_relationships=len(schema.relationships),
            no_columns=sum([len(column_stats[t]) for t in column_stats.keys()]),
            size_gb=size_gb
        )
        dataset_stats.append(curr_stats)

    save_csv(dataset_stats, target)
