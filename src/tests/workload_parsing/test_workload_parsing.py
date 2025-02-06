import numpy as np

from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_raw_plan
from cross_db_benchmark.benchmark_tools.postgres.utils import plan_statistics


def test_runtime_parsing():
    plan_steps = ['->  Aggregate  (cost=36.15..36.16 rows=1 width=8) (actual time=16.957..16.958 rows=1 loops=1)',
                  '  ->  Seq Scan on "L_UNIQUE_CARRIERS"  (cost=0.00..32.12 rows=1609 width=0) (actual time=0.371..16.803 rows=1609 loops=1)',
                  '        Filter: (("Code")::text <> \'OD\'::text)',
                  '        Rows Removed by Filter: 1',
                  'Planning time: 0.419 ms',
                  'Execution time: 17.002 ms']
    root_operator, ex_time, planning_time = parse_raw_plan(plan_steps, analyze=True, parse=False)
    assert np.isclose(ex_time, 17.002)
    assert np.isclose(planning_time, 0.419)


def test_simple_parsing():
    plan_steps = [
        'Finalize Aggregate  (cost=598680.22..598680.23 rows=1 width=32) (actual time=1830.469..1835.470 rows=1 loops=1)',
        '  ->  Gather  (cost=598680.00..598680.21 rows=2 width=32) (actual time=1830.300..1835.435 rows=3 loops=1)',
        '        Workers Planned: 2',
        '        Workers Launched: 2',
        '        ->  Partial Aggregate  (cost=597680.00..597680.01 rows=1 width=32) (actual time=1826.951..1826.954 rows=1 loops=3)',
        '              ->  Hash Join  (cost=214.80..597586.92 rows=12410 width=8) (actual time=398.965..1825.010 rows=4970 loops=3)',
        '                    Hash Cond: ("On_Time_On_Time_Performance_2016_1"."DestAirportID" = "L_AIRPORT_ID"."Code")',
        '                    ->  Hash Join  (cost=1.16..597330.79 rows=16176 width=12) (actual time=396.801..1820.171 rows=13492 loops=3)',
        '                          Hash Cond: ("On_Time_On_Time_Performance_2016_1"."Month" = "L_MONTHS"."Code")',
        '                          ->  Parallel Seq Scan on "On_Time_On_Time_Performance_2016_1"  (cost=0.00..596640.13 rows=194111 width=16) (actual time=0.127..1794.208 rows=149817 loops=3)',
        '                                Filter: ("DestAirportID" >= 14981)',
        '                                Rows Removed by Filter: 3641738',
        '                          ->  Hash  (cost=1.15..1.15 rows=1 width=4) (actual time=0.026..0.026 rows=1 loops=3)',
        '                                Buckets: 1024  Batches: 1  Memory Usage: 9kB',
        '                                ->  Seq Scan on "L_MONTHS"  (cost=0.00..1.15 rows=1 width=4) (actual time=0.019..0.021 rows=1 loops=3)',
        '                                      Filter: (("Description")::text = \'July\'::text)',
        '                                      Rows Removed by Filter: 11',
        '                    ->  Hash  (cost=152.55..152.55 rows=4887 width=4) (actual time=2.086..2.087 rows=4887 loops=3)',
        '                          Buckets: 8192  Batches: 1  Memory Usage: 236kB',
        '                          ->  Seq Scan on "L_AIRPORT_ID"  (cost=0.00..152.55 rows=4887 width=4) (actual time=0.035..1.216 rows=4887 loops=3)',
        '                                Filter: ((("Description")::text <> \'Oakland, MD: Garrett County\'::text) AND ("Code" <= 15116))',
        '                                Rows Removed by Filter: 1483',
        'Planning time: 1.106 ms',
        'Execution time: 1835.611 ms'
    ]
    analyze_plan, ex_time, planning_time = parse_raw_plan(plan_steps, analyze=True, parse=True)

    analyze_plan.parse_lines_recursively()

    assert analyze_plan.plan_parameters['act_card'] == 1
    assert analyze_plan.plan_parameters['act_time'] == 1835.47

    # check that all tables, filter columns and operators were found
    tables, filter_columns, operators = plan_statistics(analyze_plan)
    assert tables == {'L_MONTHS', 'L_AIRPORT_ID', 'On_Time_On_Time_Performance_2016_1'}
    assert filter_columns == {(('"Description"',), Operator.NEQ), (('"DestAirportID"',), Operator.GEQ),
                              (('"Description"',), Operator.EQ), (('"Code"',), Operator.LEQ),
                              (None, LogicalOperator.AND)}
    assert operators == {'Hash Join', 'Parallel Seq Scan', 'Hash', 'Seq Scan', 'Finalize Aggregate',
                         'Partial Aggregate', 'Gather'}
