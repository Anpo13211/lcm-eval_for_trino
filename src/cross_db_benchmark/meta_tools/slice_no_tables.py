import json
import math
import os

import numpy as np

from cross_db_benchmark.benchmark_tools.parse_run import dumper
from cross_db_benchmark.benchmark_tools.postgres.utils import plan_statistics
from cross_db_benchmark.benchmark_tools.utils import load_json
from training.training.checkpoint import save_csv


def no_tables(p):
    tables, _, _ = plan_statistics(p, skip_columns=True, conv_to_dict=True)
    return len(tables)


def slice_by_table_no(source_path, target_path, min_no_tables, max_no_tables, workload_slice_stats):
    assert os.path.exists(source_path)
    run_stats = load_json(source_path)

    no_table_stats = [no_tables(p) for p in run_stats.parsed_plans]
    no_tabs, counts = np.unique(no_table_stats, return_counts=True)
    for no_tab, count in zip(no_tabs, counts):
        print(f'No {no_tab} tables: {count}')

    prev_len = len(run_stats.parsed_plans)
    if min_no_tables is None:
        min_no_tables = 0
    if max_no_tables is None:
        max_no_tables = math.inf
    run_stats.parsed_plans = [p for p, no_tab in zip(run_stats.parsed_plans, no_table_stats)
                              if min_no_tables <= no_tab <= max_no_tables]
    slice_idx = [i for i, no_tab in enumerate(no_table_stats) if min_no_tables <= no_tab <= max_no_tables]
    print(f'Reduced no of queries from {prev_len} to {len(run_stats.parsed_plans)}')

    if workload_slice_stats is not None:
        save_csv([dict(slice_idx=str(slice_idx))], workload_slice_stats)

    with open(target_path, 'w') as outfile:
        json.dump(run_stats, outfile, default=dumper)
