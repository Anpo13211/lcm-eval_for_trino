import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_raw_plan


def compare_plans(run_stats, alt_run_stats, min_runtime=100):
    # parse individual queries
    sql_q_id = dict()
    for i, q in enumerate(run_stats.query_list):
        sql_q_id[q.sql.strip()] = i

    q_errors = []
    for q2 in tqdm(alt_run_stats.query_list):
        q_id = sql_q_id.get(q2.sql.strip())
        if q_id is None:
            continue

        q = run_stats.query_list[q_id]

        if q.analyze_plans is None or q2.analyze_plans is None:
            continue

        if len(q.analyze_plans) == 0 or len(q2.analyze_plans) == 0:
            continue

        assert q.sql == q2.sql

        # parse the plan as a tree
        analyze_plan, ex_time, _ = parse_raw_plan(q.analyze_plans[0], analyze=True, parse=True)
        analyze_plan2, ex_time2, _ = parse_raw_plan(q2.analyze_plans[0], analyze=True, parse=True)
        analyze_plan.parse_lines_recursively()
        analyze_plan2.parse_lines_recursively()

        if analyze_plan.min_card() == 0:
            continue

        if ex_time < min_runtime:
            continue

        q_error = max(ex_time2 / ex_time, ex_time / ex_time2)
        q_errors.append(q_error)

    # statistics in seconds
    q_errors = np.array(q_errors)
    print(f"Q-Error/Deviation of both runs: "
          f"\n\tmedian: {np.median(q_errors):.2f}"
          f"\n\tmax: {np.max(q_errors):.2f}"
          f"\n\tmean: {np.mean(q_errors):.2f}")
    print(f"Parsed {len(q_errors)} plans ")
