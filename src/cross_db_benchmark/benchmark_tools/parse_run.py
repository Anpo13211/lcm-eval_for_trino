import json
import os

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.combine_plans import combine_traces
from cross_db_benchmark.benchmark_tools.utils import load_json
from core.plugins.registry import DBMSRegistry


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


def parse_run(source_paths, target_path, database, min_query_ms=100, max_query_ms=30000,
              parse_baseline=False, cap_queries=None, parse_join_conds=False, include_zero_card=False,
              explain_only=False):
    """
    Parse query execution plans from raw database output.
    
    Args:
        database: DatabaseSystem enum OR string (e.g., 'postgres', 'trino')
                 Supports both for backward compatibility
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Convert DatabaseSystem enum to string if needed (backward compatibility)
    if isinstance(database, DatabaseSystem):
        dbms_name = database.value
    else:
        dbms_name = database
    
    # Get parser from registry (O(1) lookup, no if-elif chain!)
    try:
        parser = DBMSRegistry.get_parser(dbms_name)
    except KeyError:
        raise NotImplementedError(
            f"Database '{dbms_name}' not registered. "
            f"Available: {DBMSRegistry.list_plugins()}"
        )
    
    # Use combine_traces for all DBMS (common functionality)
    comb_func = combine_traces

    if not isinstance(source_paths, list):
        source_paths = [source_paths]

    assert all([os.path.exists(p) for p in source_paths])
    run_stats = [load_json(p) for p in source_paths]
    run_stats = comb_func(run_stats)

    # Call parser's parse_plans method
    parsed_runs, stats = parser.parse_plans(run_stats, min_runtime=min_query_ms, max_runtime=max_query_ms,
                                           parse_baseline=parse_baseline, cap_queries=cap_queries,
                                           parse_join_conds=parse_join_conds,
                                           include_zero_card=include_zero_card, explain_only=explain_only)

    with open(target_path, 'w') as outfile:
        json.dump(parsed_runs, outfile, default=dumper)
    return len(parsed_runs['parsed_plans']), stats
