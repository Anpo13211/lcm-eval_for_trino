import argparse
import multiprocessing
import multiprocessing as mp
import os
import time
from os.path import relpath

import pandas as pd

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.parse_run import parse_run


def compute(input):
    source, target, d, wl, parse_baseline, cap_queries, max_query_ms, include_zero_card, explain_only = input
    if parse_baseline:
        parse_join_conds = False
    else:
        parse_join_conds = True

    no_plans, stats = parse_run(source, target, DatabaseSystem.POSTGRES, min_query_ms=0, cap_queries=cap_queries,
                                parse_baseline=parse_baseline, parse_join_conds=parse_join_conds, max_query_ms=max_query_ms,
                                include_zero_card=include_zero_card, explain_only=explain_only)
    return dict(dataset=d, workload=wl, no_plans=no_plans, **stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--parsed_plan_dir', default=None)
    parser.add_argument('--parsed_plan_dir_baseline', default=None)
    parser.add_argument('--combine', default=None)
    parser.add_argument('--target_stats_path', default=None)
    parser.add_argument('--workload_prefix', default='')
    parser.add_argument('--workloads', nargs='+', default=None)
    parser.add_argument('--min_query_ms', default=100, type=int)
    parser.add_argument('--max_query_ms', default=30000, type=int)  # 30s
    parser.add_argument('--cap_queries', default=5000, type=int)
    parser.add_argument('--include_zero_card', action='store_true')
    parser.add_argument('--explain_only', action='store_true')
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))
    args = parser.parse_args()

    cap_queries = args.cap_queries
    if cap_queries == 'None':
        cap_queries = None

    explain_only = False
    if args.explain_only:
        explain_only = True
    setups = []
    for path, subdirs, files in os.walk(args.raw_dir):
        for workload_name in files:
            wl_path = relpath(path, args.raw_dir)
            source = os.path.join(args.raw_dir, wl_path, workload_name) #, path, workload_name)
            target = os.path.join(args.parsed_plan_dir, wl_path, workload_name) #, path, workload_name)

            setups.append((source, target, "postgres", workload_name, False, cap_queries, args.max_query_ms, args.include_zero_card, explain_only))

            target = os.path.join(args.parsed_plan_dir_baseline, wl_path, workload_name)
            setups.append((source, target, "postgres", workload_name, True, cap_queries, args.max_query_ms, args.include_zero_card, explain_only))


    start_t = time.perf_counter()
    proc = multiprocessing.cpu_count() - 2
    p = mp.Pool(initargs=('arg',), processes=proc)
    eval = p.map(compute, setups)

    eval = pd.DataFrame(eval)
    if args.target_stats_path is not None:
        eval.to_csv(args.target_stats_path, index=False)

    print()
    print(eval[['dataset', 'workload', 'no_plans']].to_string(index=False))

    print()
    print(eval[['workload', 'no_plans']].groupby('workload').sum().to_string())

    print()
    print(f"Total plans parsed in {time.perf_counter() - start_t:.2f} secs: {eval.no_plans.sum()}")
