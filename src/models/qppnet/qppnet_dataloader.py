import functools
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from cross_db_benchmark.benchmark_tools.postgres.json_plan import operator_tree_from_json, OperatorTree
from cross_db_benchmark.benchmark_tools.utils import load_json
from classes.classes import DataLoaderOptions, QPPNetModelConfig, ModelConfig
from classes.workload_runs import WorkloadRuns
from training.dataset.dataset_creation import read_workload_runs, derive_label_normalizer
from training.dataset.plan_dataset import PlanDataset


def create_qppnet_dataloader(workload_runs: WorkloadRuns,
                             statistics_file: Path,
                             model_config: QPPNetModelConfig,
                             column_statistics: Path,
                             data_loader_options: DataLoaderOptions,
                             use_trino: bool = False) \
        -> (Optional[Pipeline], dict, Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]):

    feature_statistics = load_json(statistics_file, namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"

    train_loader, val_loader, test_loaders = Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]

    plan_collator = qppnet_collator

    dataloader_args = dict(batch_size=model_config.batch_size,
                           shuffle=data_loader_options.shuffle,
                           num_workers=model_config.num_workers,
                           pin_memory=data_loader_options.pin_memory)

    column_statistics = load_json(column_statistics, namespace=False)

    if workload_runs.train_workload_runs:
        assert workload_runs.test_workload_runs == [], ("Unseen Test workload runs are not allowed when training "
                                                        "workload driven models")
        print("Create dataloader for training, validation and test data")

        label_norm, train_dataset, val_dataset, database_statistics \
            = create_datasets(workload_run_paths=workload_runs.train_workload_runs,
                              model_config=model_config,
                              val_ratio=data_loader_options.val_ratio)

        test_dataset, val_dataset = val_dataset.split(0.5)
        print(f"Created datasets of size: "
              f"train {len(train_dataset)}, "
              f"validation: {len(val_dataset)}, t"
              f"est: {len(test_dataset)}")

        train_collate_fn = functools.partial(plan_collator,
                                                 db_statistics=database_statistics,
                                                 feature_statistics=feature_statistics,
                                                 column_statistics=column_statistics,
                                                 plan_featurization=model_config.featurization,
                                                 use_trino=use_trino)

        dataloader_args.update(collate_fn=train_collate_fn)
        train_loader = DataLoader(train_dataset, **dataloader_args)
        val_loader = DataLoader(val_dataset, **dataloader_args)
        test_loaders = [DataLoader(test_dataset, **dataloader_args)]

    if workload_runs.test_workload_runs:
        test_loaders = []
        for test_workload in workload_runs.test_workload_runs:
            print(f"Create dataloader for test data: {test_workload}")
            _, _, test_dataset, database_statistics = \
                create_datasets(workload_run_paths=[test_workload],
                                model_config=model_config,
                                val_ratio=1.0)

            train_collate_fn = functools.partial(plan_collator,
                                                 db_statistics=database_statistics,
                                                 feature_statistics=feature_statistics,
                                                 column_statistics=column_statistics,
                                                 plan_featurization=model_config.featurization,
                                                 use_trino=use_trino)

            dataloader_args.update(collate_fn=train_collate_fn)
            test_loaders.append(DataLoader(test_dataset, **dataloader_args))

    return None, feature_statistics, train_loader, val_loader, test_loaders


def create_datasets(workload_run_paths,
                    model_config: ModelConfig,
                    val_ratio=0.15,
                    shuffle_before_split=True) -> (Pipeline, PlanDataset, PlanDataset, PlanDataset, dict):

    plans, database_statistics = read_workload_runs(workload_run_paths=workload_run_paths,
                                                    limit_queries=model_config.limit_queries,
                                                    limit_queries_affected_wl=model_config.limit_queries_affected_wl,
                                                    execution_mode=model_config.execution_mode)

    no_plans = len(plans)
    plan_indexes = list(range(no_plans))
    if shuffle_before_split:
        np.random.shuffle(plan_indexes)

    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_indexes = plan_indexes[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if model_config.cap_training_samples is not None:
        print(f'Limiting dataset to {model_config.cap_training_samples}')
        prev_train_length = len(train_indexes)
        train_indexes = train_indexes[:model_config.cap_training_samples]
        replicate_factor = max(prev_train_length // len(train_indexes), 1)
        train_indexes = train_indexes * replicate_factor

    train_dataset = PlanDataset([plans[i] for i in train_indexes], train_indexes)

    val_dataset = None
    if val_ratio > 0:
        val_indexes = plan_indexes[split_train:]
        val_dataset = PlanDataset([plans[i] for i in val_indexes], val_indexes)

    # derive label normalization
    runtimes = np.array([getattr(p, "Execution Time") / 1000 for p in plans])
    label_norm = derive_label_normalizer(model_config.loss_class_name, runtimes)

    return label_norm, train_dataset, val_dataset, database_statistics


def qppnet_collator(plans, feature_statistics: dict = None, db_statistics: dict = None, column_statistics: dict = None, plan_featurization=None, use_trino=False, debug_print=False):
    labels = []
    query_plans = []

    # iterate over plans and create lists of edges and features per node
    sample_idxs = []
    errors = []
    for sample_idx, p in plans:
        try:
            # Trinoの場合はアダプターで変換
            if use_trino:
                from models.qppnet.trino_adapter import adapt_trino_plan_to_qppnet
                # pがTrinoPlanOperatorの場合、そのまま変換
                plan_dict = adapt_trino_plan_to_qppnet(p)
                query_plan: OperatorTree = operator_tree_from_json(plan_dict)
            else:
                # PostgreSQLの場合は従来通り
                query_plan: OperatorTree = operator_tree_from_json(vars(p))
            
            # Debug: Print tree structure for first plan
            if sample_idx == 0 and use_trino and debug_print:
                print(f"\n[DEBUG] Sample plan tree structure:")
                def print_tree(node, depth=0):
                    print(f"{'  ' * depth}{node.node_type} (children: {len(node.children)})")
                    for child in node.children:
                        print_tree(child, depth + 1)
                print_tree(query_plan)
            
            #if query_plan.min_cardinality() != 0:
            query_plan.encode_recursively(column_statistics, feature_statistics, plan_featurization)
            sample_idxs.append(sample_idx)
            labels.append(query_plan.runtime)
            query_plans.append(query_plan)
        except (ValueError, KeyError, AttributeError) as e:
            errors.append((sample_idx, str(e)))

    if errors and debug_print:
        print(f"Encoding errors: {len(errors)}/{len(plans)}")
        # 最初の数個だけ詳細を表示
        for i, (idx, err_msg) in enumerate(errors[:3]):
            print(f"  Error {i+1} (sample {idx}): {err_msg[:100]}")
    
    return query_plans, labels, sample_idxs
