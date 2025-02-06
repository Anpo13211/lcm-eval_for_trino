import argparse
import multiprocessing
import multiprocessing as mp
import os
import time

import pandas as pd

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.parse_run import parse_run
from cross_db_benchmark.datasets.datasets import ext_database_list


def compute(input):
    source, target, d, wl, parse_baseline, cap_queries, max_query_ms, include_zero_card = input
    no_plans, stats = parse_run(source, target, args.database, min_query_ms=args.min_query_ms, cap_queries=cap_queries,
                                parse_baseline=parse_baseline, parse_join_conds=True, max_query_ms=max_query_ms,
                                include_zero_card=include_zero_card)
    return dict(dataset=d, workload=wl, no_plans=no_plans, **stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--parsed_plan_dir', default=None)
    parser.add_argument('--combine', default=None)
    parser.add_argument('--target_stats_path', default=None)
    parser.add_argument('--workload_prefix', default='')
    parser.add_argument('--workloads', nargs='+', default=None)
    parser.add_argument('--min_query_ms', default=100, type=int)
    parser.add_argument('--max_query_ms', default=30000, type=int)  # 30s
    parser.add_argument('--cap_queries', default=5000, type=int)
    parser.add_argument('--include_zero_card', action='store_true')
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))
    args = parser.parse_args()

    cap_queries = args.cap_queries
    if cap_queries == 'None':
        cap_queries = None

    setups = []
    for db in ext_database_list:
        d = db.db_name

        # list available workload traces for this db
        curr_setups = []
        for wl in args.workloads:
            source = os.path.join(args.raw_dir, d, wl)
            parse_baseline = not 'complex' in wl
            if not os.path.exists(source):
                print(f"Missing: {d}: {wl}")
                continue
            target = os.path.join(args.parsed_plan_dir, d, args.workload_prefix + wl)
            curr_setups.append(
                (source, target, d, wl, parse_baseline, cap_queries, args.max_query_ms, args.include_zero_card))

        # either parse workload for workload or combine them into a single file
        if args.combine is None:
            setups += curr_setups
        else:
            if len(curr_setups) == 0:
                continue

            # should not combine complex and non-complex workloads
            assert all(s[4] == curr_setups[0][4] for s in curr_setups)
            sources = [s[0] for s in curr_setups]
            combined_target = os.path.join(args.parsed_plan_dir, d, args.combine)
            setups.append(
                tuple([sources, combined_target] + [curr_setups[0][i] for i in range(2, len(curr_setups[0]))]))

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
