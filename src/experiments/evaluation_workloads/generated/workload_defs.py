import os

from cross_db_benchmark.datasets.datasets import ext_database_list


def toy_workload_def():
    return {
        # a small workload for testing
        'workload_50_s0': dict(num_queries=50,
                               max_no_predicates=5,
                               max_no_aggregates=3,
                               max_no_group_by=0,
                               max_cols_per_agg=2,
                               seed=0),
        'complex_workload_50_s0': dict(num_queries=50,
                                       max_no_predicates=5,
                                       max_no_aggregates=3,
                                       max_no_group_by=2,
                                       max_cols_per_agg=2,
                                       complex_predicates=True,
                                       seed=0),
        # 'workload_10k_s0': dict(num_queries=10000,
        #                         max_no_predicates=5,
        #                         max_no_aggregates=3,
        #                         max_no_group_by=0,
        #                         max_cols_per_agg=2,
        #                         seed=0),
    }


def query_opt_workload_def():
    return {
        # for query optimization (300s timeouts, not complex)
        'workload_100k_s7': dict(num_queries=100000,
                                 max_no_predicates=5,
                                 max_no_aggregates=3,
                                 max_no_group_by=0,
                                 max_cols_per_agg=2,
                                 seed=7),
        # for query optimization (300s timeouts, not complex)
        'workload_100k_s8': dict(num_queries=100000,
                                 max_no_predicates=5,
                                 max_no_aggregates=3,
                                 max_no_group_by=0,
                                 max_cols_per_agg=2,
                                 seed=8),
        # for query optimization (300s timeouts, not complex)
        'workload_100k_s9': dict(num_queries=100000,
                                 max_no_predicates=5,
                                 max_no_aggregates=3,
                                 max_no_group_by=0,
                                 max_cols_per_agg=2,
                                 seed=9),
        # for complex predicates + query optimization (300s timeouts)
        'complex_workload_200k_s2': dict(num_queries=200000,
                                         max_no_predicates=5,
                                         max_no_aggregates=3,
                                         max_no_group_by=0,
                                         max_cols_per_agg=2,
                                         complex_predicates=True,
                                         seed=2),
        # for complex predicates + query optimization (300s timeouts)
        'complex_workload_200k_s3': dict(num_queries=200000,
                                         max_no_predicates=5,
                                         max_no_aggregates=3,
                                         max_no_group_by=0,
                                         max_cols_per_agg=2,
                                         complex_predicates=True,
                                         seed=3),
        # for complex predicates + query optimization (300s timeouts)
        'complex_workload_200k_s4': dict(num_queries=200000,
                                         max_no_predicates=5,
                                         max_no_aggregates=3,
                                         max_no_group_by=0,
                                         max_cols_per_agg=2,
                                         complex_predicates=True,
                                         seed=4),
    }

def db_gen_workload_def():
    return {
        # # this will be capped at 10k, no group bys created currently
        # 'workload_100k_s1': dict(num_queries=100000,
        #                          max_no_predicates=5,
        #                          max_no_aggregates=3,
        #                          max_no_group_by=0,
        #                          max_cols_per_agg=2,
        #                          seed=1),
        # # for complex predicates, this will be capped at 5k
        # 'complex_workload_200k_s1': dict(num_queries=200000,
        #                                  max_no_predicates=5,
        #                                  max_no_aggregates=3,
        #                                  max_no_group_by=0,
        #                                  max_cols_per_agg=2,
        #                                  complex_predicates=True,
        #                                  seed=1),
        # # we completely ignore group by columns (appear in no benchmark)
        # # for index workloads, will also be capped at 5k
        # 'workload_100k_s2': dict(num_queries=100000,
        #                          max_no_predicates=5,
        #                          max_no_aggregates=3,
        #                          max_no_group_by=0,
        #                          max_cols_per_agg=2,
        #                          seed=2),
        # # another version of index (statically created beforehand)
        # 'workload_100k_s10': dict(num_queries=100000,
        #                           max_no_predicates=5,
        #                           max_no_aggregates=3,
        #                           max_no_group_by=2,
        #                           max_cols_per_agg=2,
        #                           seed=10),
        # revision train queries. will also be capped at 5k queries (statically created beforehand)
        'complex_tpchworkload_200k_s10': dict(num_queries=200000,
                                              max_no_predicates=5,
                                              max_no_aggregates=3,
                                              max_no_group_by=3,
                                              max_cols_per_agg=2,
                                              complex_predicates=True,
                                              seed=10,
                                              left_outer_join_ratio=0.01,
                                              groupby_limit_prob=0.2,
                                              groupby_having_prob=0.2,
                                              exists_predicate_prob=0.3,
                                              max_no_exists=2,
                                              outer_groupby_prob=0.2
                                              )

    }


def wl_driven_train_def():
    return {
        # only for imdb (will all be capped at 25k)
        'workload_400k_s2': dict(num_queries=400000,
                                 max_no_predicates=5,
                                 max_no_aggregates=1,
                                 max_no_group_by=0,
                                 max_cols_per_agg=1,
                                 seed=2),
        'workload_400k_s3': dict(num_queries=400000,
                                 max_no_predicates=5,
                                 max_no_aggregates=1,
                                 max_no_group_by=0,
                                 max_cols_per_agg=1,
                                 seed=3),
        'workload_400k_s4': dict(num_queries=400000,
                                 max_no_predicates=5,
                                 max_no_aggregates=1,
                                 max_no_group_by=0,
                                 max_cols_per_agg=1,
                                 seed=4),
        'workload_400k_s5': dict(num_queries=400000,
                                 max_no_predicates=5,
                                 max_no_aggregates=1,
                                 max_no_group_by=0,
                                 max_cols_per_agg=1,
                                 seed=5),
        # only for imdb (will all be capped at 25k)
        'complex_workload_400k_s2': dict(num_queries=400000,
                                         max_no_predicates=5,
                                         max_no_aggregates=1,
                                         max_no_group_by=0,
                                         max_cols_per_agg=1,
                                         complex_predicates=True,
                                         seed=2),
        'complex_workload_400k_s3': dict(num_queries=400000,
                                         max_no_predicates=5,
                                         max_no_aggregates=1,
                                         max_no_group_by=0,
                                         max_cols_per_agg=1,
                                         complex_predicates=True,
                                         seed=3),
        'complex_workload_400k_s7': dict(num_queries=400000,
                                         max_no_predicates=5,
                                         max_no_aggregates=1,
                                         max_no_group_by=0,
                                         max_cols_per_agg=1,
                                         complex_predicates=True,
                                         seed=7),
        'complex_workload_400k_s8': dict(num_queries=400000,
                                         max_no_predicates=5,
                                         max_no_aggregates=1,
                                         max_no_group_by=0,
                                         max_cols_per_agg=1,
                                         complex_predicates=True,
                                         seed=8),
        # new for JOB imdb (will be capped later)
        'complex_workload_400k_s4': dict(num_queries=400000,
                                         max_no_predicates=5,
                                         max_no_aggregates=1,
                                         max_no_group_by=0,
                                         max_cols_per_agg=1,
                                         complex_predicates=True,
                                         seed=4),
        'complex_workload_400k_s5': dict(num_queries=400000,
                                         max_no_predicates=5,
                                         max_no_aggregates=1,
                                         max_no_group_by=0,
                                         max_cols_per_agg=1,
                                         complex_predicates=True,
                                         seed=5),
        'complex_workload_400k_s6': dict(num_queries=400000,
                                         max_no_predicates=5,
                                         max_no_aggregates=1,
                                         max_no_group_by=0,
                                         max_cols_per_agg=1,
                                         complex_predicates=True,
                                         seed=6),
    }


def generate_workload_defs(workload_dir):
    workload_gen_setups = []

    # workloads that are defined for every dataset
    for dataset in ext_database_list:
        wl_dict = dict()
        wl_dict.update(db_gen_workload_def())

        for workload_name, workload_args in wl_dict.items():
            workload_path = os.path.join(workload_dir, dataset.db_name, f'{workload_name}.sql')
            workload_gen_setups.append((dataset.source_dataset, workload_path, dataset.max_no_joins, workload_args))
    return workload_gen_setups


def generalization_workload_defs(dataset=None, workload_dir=None, return_setups=False):
    base_wl_kwargs = dict(num_queries=100000,
                          max_no_predicates=5,
                          max_no_aggregates=1,
                          max_no_group_by=0,
                          max_cols_per_agg=1,
                          seed=0,
                          max_no_joins_static=False,
                          max_no_aggregates_static=False,
                          max_no_predicates_static=False,
                          max_no_group_by_static=False
                          )
    # joins
    wls = []
    workload_gen_setups = []
    # for no_joins in range(5, 15):
    for no_joins in [1, 5, 7, 10]:
        workload_args = base_wl_kwargs.copy()
        workload_args.update(max_no_joins_static=True)
        wl_name = f'generalization_no_joins_{no_joins}'
        wls.append(wl_name)
        if return_setups:
            workload_path = os.path.join(workload_dir, dataset.db_name, f'{wl_name}.sql')
            workload_gen_setups.append((dataset.source_dataset, workload_path, no_joins, workload_args))
    # predicates
    # for no_preds in range(5, 21):
    for no_preds in [1, 5, 10, 15]:
        workload_args = base_wl_kwargs.copy()
        workload_args.update(dict(max_no_predicates_static=True, max_no_predicates=no_preds))
        wl_name = f'generalization_no_preds_{no_preds}'
        wls.append(wl_name)
        if return_setups:
            workload_path = os.path.join(workload_dir, dataset.db_name, f'{wl_name}.sql')
            workload_gen_setups.append((dataset.source_dataset, workload_path, dataset.max_no_joins, workload_args))
    # max_no_aggregates_static
    # for no_aggs in range(2, 21):
    for no_aggs in [1, 5, 10, 15, 20]:
        workload_args = base_wl_kwargs.copy()
        workload_args.update(dict(max_no_aggregates_static=True, max_no_aggregates=no_aggs))
        wl_name = f'generalization_no_aggs_{no_aggs}'
        wls.append(wl_name)
        if return_setups:
            workload_path = os.path.join(workload_dir, dataset.db_name, f'{wl_name}.sql')
            workload_gen_setups.append((dataset.source_dataset, workload_path, dataset.max_no_joins, workload_args))
    # max_no_aggregates_static
    # for no_groupby in range(1, 6):
    for no_groupby in [1, 3, 5]:
        workload_args = base_wl_kwargs.copy()
        workload_args.update(dict(max_no_group_by_static=True, max_no_group_by=no_groupby))
        wl_name = f'generalization_no_groupby_{no_groupby}'
        wls.append(wl_name)
        if return_setups:
            workload_path = os.path.join(workload_dir, dataset.db_name, f'{wl_name}.sql')
            workload_gen_setups.append((dataset.source_dataset, workload_path, dataset.max_no_joins, workload_args))

    if return_setups:
        return workload_gen_setups

    return wls
