import functools
import math
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.linear_model import LinearRegression
from wandb.apis.public import Run

from classes.classes import FlatModelConfig, ModelConfig, ScaledPostgresModelConfig, \
    AnalyticalEstCardModelConfig, AnalyticalActCardModelConfig
from classes.workload_runs import WorkloadRuns
from cross_db_benchmark.benchmark_tools.utils import load_json
from training.dataset.dataset_creation import read_workload_runs
from training.training.checkpoint import save_csv
from training.training.metrics import QError, RMSE, MAPE
from training.training.train import to_rows


def train_tabular_model(workload_runs: WorkloadRuns,
                        config: ModelConfig,
                        statistics_file: Path,
                        model_dir: Path,
                        cap_training_samples: int = None,
                        early_stopping_rounds: int = 20,
                        num_boost_round: int = 1000):
    # seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    os.makedirs(model_dir, exist_ok=True)

    param_dict = {}
    # create tabular representation using the corresponding method
    if isinstance(config, FlatModelConfig):
        gen_dataset_func = gen_flattened_dataset
    elif isinstance(config, ScaledPostgresModelConfig):
        gen_dataset_func = gen_optimizer_cost_dataset
    elif isinstance(config, AnalyticalEstCardModelConfig):
        gen_dataset_func = functools.partial(gen_analytical_flattened_dataset, act_card=False)
    elif isinstance(config, AnalyticalActCardModelConfig):
        gen_dataset_func = functools.partial(gen_analytical_flattened_dataset, act_card=True)
    else:
        raise RuntimeError("Not supported")

    if config.type == 'wl_driven':
        x_train, x_test, y_train, y_test = gen_dataset_func(workload_runs=workload_runs.train_workload_runs,
                                                            statistics_file=statistics_file,
                                                            cap_training_samples=cap_training_samples,
                                                            val_ratio=0.20,
                                                            featurization=config.featurization,
                                                            shuffle=True)
        raise NotImplementedError("This needs to be solved for workload driven model due to the correct split")
    elif config.type == 'wl_agnostic':
        x_train, x_test, y_train, y_test = gen_dataset_func(workload_runs=workload_runs.train_workload_runs,
                                                            statistics_file=statistics_file,
                                                            cap_training_samples=cap_training_samples,
                                                            featurization=config.featurization,
                                                            val_ratio=0.20,
                                                            shuffle=True)
    else:
        raise RuntimeError("Not supported")

    print(f"Training model {config.name.NAME} "
          f"with seed {config.seed} "
          f"and {len(x_train)} training points "
          f"and {len(x_test)} validation points")

    start_t = time.perf_counter()
    if isinstance(config, FlatModelConfig):
        train_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_test, label=y_test, reference=train_data)
        param = dict(metric='mse',
                     objective='regression',
                     random_state=config.seed,
                     seed=config.seed,
                     bagging_seed=config.seed,
                     feature_fraction_seed=config.seed)
        bst = lgb.train(param, train_data, num_boost_round=num_boost_round, valid_sets=[val_data])
        param_dict.update(num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds)
        bst.save_model(f'{model_dir}_{config.seed}.txt')
        print(f'Model saved to {model_dir}_{config.seed}.txt')

    elif isinstance(config, (ScaledPostgresModelConfig, AnalyticalEstCardModelConfig, AnalyticalActCardModelConfig)):
        m = LinearRegression()
        m.fit(x_train, y_train)
        joblib.dump(m, f'{model_dir}_{config.seed}.pkl')
        print(f'Model saved to {model_dir}_{config.seed}.pkl')

    else:
        raise NotImplementedError('Not supported tabular model')

    param_dict.update(training_time=time.perf_counter() - start_t,
                      no_train_points=len(x_train),
                      no_val_points=len(x_test))

    save_csv([param_dict], str(model_dir) + f'_{config.seed}_params.csv')


def predict_tabular_model(config: ModelConfig,
                          workload_runs: WorkloadRuns,
                          statistics_file: Path,
                          model_dir: Path,
                          target_path: Path,
                          cap_training_samples: bool = None,
                          run: Optional[Run] = None):
    # seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if isinstance(config, FlatModelConfig):
        print(f"Loading model from {model_dir}_{config.seed}.txt")
        bst = lgb.Booster(model_file=f'{model_dir}_{config.seed}.txt')

        # Define a function to apply lower bound
        def predict_with_lower_bound(data, lower_bound, bst=bst):
            predictions = bst.predict(data, num_iteration=bst.best_iteration)
            predictions = [max(prediction, lower_bound) for prediction in predictions]
            return predictions

        # Partially apply the predict_with_lower_bound function
        pred_func = functools.partial(predict_with_lower_bound, lower_bound=0.01)
        # pred_func = functools.partial(bst.predict, num_iteration=bst.best_iteration)

        gen_dataset_func = gen_flattened_dataset

    elif isinstance(config, ScaledPostgresModelConfig):
        print(f"Loading model from {model_dir}_{config.seed}.pkl")
        m = joblib.load(filename=f'{model_dir}_{config.seed}.pkl')
        pred_func = m.predict
        gen_dataset_func = gen_optimizer_cost_dataset

    else:
        raise NotImplementedError('Not supported tabular model')

    test_datasets = []
    for test_wl in workload_runs.test_workload_runs:
        x_test, _, y_test, _ = gen_dataset_func(workload_runs=[test_wl],
                                                statistics_file=statistics_file,
                                                cap_training_samples=cap_training_samples,
                                                featurization=config.featurization,
                                                val_ratio=0,
                                                shuffle=False)
        test_datasets.append((x_test, y_test))

    metrics = [RMSE(),
               MAPE(),
               QError(percentile=50, early_stopping_metric=True),
               QError(percentile=95),
               QError(percentile=100)]

    for (x_test, y_test), test_path in zip(test_datasets, workload_runs.target_test_csv_paths):
        print(f"Starting validation for {test_path}")
        test_stats = dict()
        start_t = time.perf_counter()
        pred_runtimes = pred_func(x_test)
        test_stats.update(test_time=time.perf_counter() - start_t,
                          no_test_points=len(y_test),
                          test_labels=y_test,
                          test_preds=pred_runtimes)

        for metric in metrics:
            metric.evaluate(metrics_dict=test_stats,
                            model=None,
                            labels=y_test,
                            preds=pred_runtimes,
                            probs=None)
        save_csv([test_stats], target_csv_path=str(test_path.with_suffix('')) + f"_test_stats.csv")

        # Logging all queries
        rows = to_rows(indexes=[int(i) for i in range(0, len(y_test))],
                       labels=[float(f) for f in y_test],
                       predictions=[float(f) for f in pred_runtimes])

        save_csv(rows, target_csv_path=str(test_path) + "_test_pred.csv")

        if run:
            test_stats.pop('test_labels')
            test_stats.pop('test_preds')
            wandb.log({f"{config.name.NAME}/{model_dir.stem}/{config.seed}/test_scores":
                wandb.Table(columns=["metric", "value"], data=list(test_stats.items()))})
            wandb.log({f"{config.name.NAME}/{model_dir.stem}/{config.seed}/test_preds":
                           wandb.Table(dataframe=pd.DataFrame(rows))})
    return


def gen_optimizer_cost_dataset(workload_runs, statistics_file, cap_training_samples, featurization, shuffle, val_ratio=0.15):
    plans, database_statistics = read_workload_runs(workload_runs)
    feature_vecs = []
    labels = []
    for plan in plans:
        # append to all feature vectors and runtime labels
        feature_vecs.append([plan.plan_parameters.est_cost])
        labels.append(plan.plan_runtime / 1000)

    X_train, X_val, Y_train, Y_val = slice_train_val(cap_training_samples, feature_vecs, labels, plans, val_ratio,
                                                     shuffle)
    return X_train, X_val, Y_train, Y_val


def gen_flattened_dataset(workload_runs, statistics_file, cap_training_samples, featurization, val_ratio=0.15, shuffle=True, ):
    feature_statistics = load_json(statistics_file, namespace=False)
    op_idx_dict = feature_statistics['op_name']['value_dict']
    no_ops = len(op_idx_dict)
    plans, database_statistics = read_workload_runs(workload_runs)
    feature_vecs = []
    labels = []
    for plan in plans:
        # number an operator occurs
        feature_num_vec = np.zeros(no_ops)
        # number of rows per op (estimated by postgres)
        feature_row_vec = np.zeros(no_ops)

        def extract_features(p, featurization):
            op_name = p.plan_parameters.op_name
            op_idx = op_idx_dict[op_name]
            feature_num_vec[op_idx] += 1
            if featurization.PLAN_FEATURE == 'est_card':
                if 'est_card' in vars(p.plan_parameters):
                    feature_row_vec[op_idx] += p.plan_parameters.est_card
                else:
                    feature_row_vec[op_idx] += p.plan_parameters.est_rows
            elif featurization.PLAN_FEATURE == 'act_card':
                if 'act_card' in vars(p.plan_parameters):
                    feature_row_vec[op_idx] += p.plan_parameters.act_card
                elif 'act_rows' in vars(p.plan_parameters):
                    feature_row_vec[op_idx] += p.plan_parameters.act_rows
                else:
                    feature_row_vec[op_idx] += p.plan_parameters.est_card


            for c in p.children:
                extract_features(c, featurization)

        extract_features(plan, featurization)

        # append to all feature vectors and runtime labels
        feature_vecs.append(np.concatenate((feature_num_vec, feature_row_vec)))
        labels.append(plan.plan_runtime / 1000)
    X_train, X_val, Y_train, Y_val = slice_train_val(cap_training_samples, feature_vecs, labels, plans, val_ratio,
                                                     shuffle=shuffle)
    return X_train, X_val, Y_train, Y_val


def extract_card(p, act_card):
    if not act_card:
        if 'est_card' in vars(p.plan_parameters):
            card = p.plan_parameters.est_card
        else:
            card = p.plan_parameters.est_rows
    else:
        if 'est_card' in vars(p.plan_parameters):
            if 'act_card' in vars(p.plan_parameters):
                card = p.plan_parameters.act_card
            # fallback: for some plan operators, pg does not annotate act cards
            else:
                card = p.plan_parameters.est_card
        else:
            card = p.plan_parameters.act_rows
    return card


def gen_analytical_flattened_dataset(workload_runs, statistics_file, cap_training_samples, val_ratio=0.15,
                                     act_card=False):
    feature_statistics = load_json(statistics_file, namespace=False)
    op_idx_dict = feature_statistics['op_name']['value_dict']
    no_ops = len(op_idx_dict)
    plans, database_statistics = read_workload_runs(workload_runs)
    feature_vecs = []
    labels = []
    for plan in plans:
        # number an operator occurs
        feature_num_vec = np.zeros(no_ops)
        # number of rows per op (estimated by postgres)
        feature_row_vec = np.zeros(no_ops)
        feature_row_vec_left_input = np.zeros(no_ops)
        feature_row_vec_right_input = np.zeros(no_ops)
        feature_row_vec_prod_input = np.zeros(no_ops)

        def extract_features(p):
            op_name = p.plan_parameters.op_name
            op_idx = op_idx_dict[op_name]
            feature_num_vec[op_idx] += 1

            # output
            feature_row_vec[op_idx] += extract_card(p, act_card)

            child_card = [extract_card(c, act_card) for c in p.children]
            if len(child_card) >= 1:
                feature_row_vec_left_input[op_idx] += child_card[0]
                feature_row_vec_prod_input[op_idx] += math.prod(child_card)
            if len(child_card) >= 2:
                feature_row_vec_right_input[op_idx] += child_card[1]

            for c in p.children:
                extract_features(c)

        extract_features(plan)

        # append to all feature vectors and runtime labels
        feature_vecs.append(np.concatenate((feature_num_vec, feature_row_vec, feature_row_vec_left_input,
                                            feature_row_vec_right_input, feature_row_vec_prod_input)))
        labels.append(plan.plan_runtime / 1000)
    X_train, X_val, Y_train, Y_val = slice_train_val(cap_training_samples, feature_vecs, labels, plans, val_ratio)
    return X_train, X_val, Y_train, Y_val


def slice_train_val(cap_training_samples, feature_vecs, labels, plans, val_ratio, shuffle=True):
    feature_vecs = np.array(feature_vecs)
    labels = np.array(labels)
    no_plans = len(plans)
    plan_idxs = list(range(no_plans))
    if shuffle:
        np.random.shuffle(plan_idxs)
    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_idxs = plan_idxs[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if cap_training_samples is not None:
        train_idxs = train_idxs[:cap_training_samples]
    X_val = None
    Y_val = None
    if val_ratio > 0:
        val_idxs = plan_idxs[split_train:]
        X_val = feature_vecs[val_idxs]
        Y_val = labels[val_idxs]
    X_train = feature_vecs[train_idxs]
    Y_train = labels[train_idxs]
    return X_train, X_val, Y_train, Y_val
