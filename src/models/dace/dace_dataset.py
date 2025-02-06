import functools
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Optional, List

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.nn.functional import pad
from torch.utils.data import DataLoader

from cross_db_benchmark.benchmark_tools.utils import load_json
from classes.classes import DACEModelConfig, DataLoaderOptions
from classes.workload_runs import WorkloadRuns
from training.dataset.plan_dataset import PlanDataset
from training.preprocessing.feature_statistics import FeatureType


def create_dace_dataloader(statistics_file: Path,
                           model_config: DACEModelConfig,
                           workload_runs: WorkloadRuns,
                           dataloader_options: DataLoaderOptions) -> Tuple[dict, DataLoader, DataLoader, list[DataLoader]]:

    feature_statistics = load_json(statistics_file, namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"

    train_loader, val_loader, test_loaders = Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]

    dataloader_args = dict(batch_size=model_config.batch_size,
                           shuffle=dataloader_options.shuffle,
                           num_workers=model_config.num_workers,
                           pin_memory=dataloader_options.pin_memory,
                           collate_fn=functools.partial(dace_collator,
                                                        feature_statistics=feature_statistics,
                                                        config=model_config))

    if workload_runs.train_workload_runs:
        train_dataset, val_dataset = create_dace_datasets(workload_run_paths=workload_runs.train_workload_runs,
                                                          model_config=model_config,
                                                          val_ratio=dataloader_options.val_ratio)
        train_loader: DataLoader = DataLoader(train_dataset, **dataloader_args)
        val_loader: DataLoader = DataLoader(val_dataset, **dataloader_args)

    test_loaders = []
    if workload_runs.test_workload_runs:
        dataloader_args.update(shuffle=False)
        # For each test workload run create a distinct test loader
        print("Creating dataloader for test data")
        test_loaders = []
        for test_path in workload_runs.test_workload_runs:
            test_dataset, _ = create_dace_datasets([test_path],
                                                    model_config=model_config,
                                                    shuffle_before_split=False,
                                                    val_ratio=0.0)
            test_loader = DataLoader(test_dataset, **dataloader_args)
            test_loaders.append(test_loader)

    return feature_statistics, train_loader, val_loader, test_loaders


def create_dace_datasets(workload_run_paths,
                         model_config: DACEModelConfig,
                         val_ratio=0.15,
                         shuffle_before_split=True) -> (PlanDataset, PlanDataset):
    plans = []
    for workload_run in workload_run_paths:
        plans += read_workload_run(workload_run)

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

    return train_dataset, val_dataset


def dace_collator(batch: Tuple, feature_statistics: dict, config: DACEModelConfig):
    # Get plan encodings
    add_numerical_scalers(feature_statistics)

    # Get op_name to one-hot, using feature_statistics
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)

    labels, sample_idxs, seq_encodings, attention_masks, loss_masks, all_runtimes = [], [], [], [], [], []

    for sample_idx, p in batch:
        sample_idxs.append(sample_idx)
        seq_encoding, attention_mask, loss_mask, run_times = get_plan_encoding(query_plan=p,
                                                                               model_config=config,
                                                                               op_name_to_one_hot=op_name_to_one_hot,
                                                                               plan_parameters=config.featurization.PLAN_FEATURES,
                                                                               feature_statistics=feature_statistics)
        #labels.append(run_times[0] * config.max_runtime)
        labels.append(torch.tensor(p.plan_runtime) / 1000)
        all_runtimes.append(run_times)
        seq_encodings.append(seq_encoding)
        attention_masks.append(attention_mask)
        loss_masks.append(loss_mask)

    labels = torch.stack(labels)
    all_runtimes = torch.stack(all_runtimes)
    seq_encodings = torch.stack(seq_encodings)
    attention_masks = torch.stack(attention_masks)
    loss_masks = torch.stack(loss_masks)

    return seq_encodings, attention_masks, loss_masks, all_runtimes, labels, sample_idxs


def get_op_name_to_one_hot(feature_statistics: dict) -> dict:
    op_name_to_one_hot = {}
    op_names = feature_statistics["op_name"]["value_dict"]
    op_names_no = len(op_names)
    for i, name in enumerate(op_names.keys()):
        op_name_to_one_hot[name] = np.zeros((1, op_names_no), dtype=np.int32)
        op_name_to_one_hot[name][0][i] = 1
    return op_name_to_one_hot


def add_numerical_scalers(feature_statistics: dict) -> None:
    for k, v in feature_statistics.items():
        if v["type"] == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v["center"]
            scaler.scale_ = v["scale"]
            feature_statistics[k]["scaler"] = scaler


def pad_sequence(seq_encoding: np.ndarray, padding_value: int = 0,
                 node_length: int = 18, max_length: int = 20) -> Tuple[torch.Tensor, int]:
    """
    This pads seqs to the same length, and transform seqs to a tensor
    seqs:           list of seqs (seq shape: (1, feature_no))
    padding_value:  padding value
    returns:        padded seqs, seqs_length
    """
    seq_length = seq_encoding.shape[1]
    seq_padded = pad(torch.from_numpy(seq_encoding),
                     pad=(0, max_length * node_length - seq_encoding.shape[1]),
                     value=padding_value)
    seq_padded = seq_padded.to(dtype=torch.float32)
    return seq_padded, seq_length


def get_loss_mask(seq_length: int,
                  pad_length: int,
                  node_length: int,
                  heights: list,
                  loss_weight: float = 0.5) -> torch.Tensor:
    seq_length = int(seq_length / node_length)
    loss_mask = np.zeros(pad_length)
    loss_mask[:seq_length] = np.power(loss_weight, np.array(heights))
    loss_mask = torch.from_numpy(loss_mask).float()
    return loss_mask


def read_workload_run(workload_run: Path) -> list[SimpleNamespace]:
    plans: list[SimpleNamespace] = []
    try:
        workload_path = os.path.join(workload_run)
        run = load_json(workload_path)
    except json.JSONDecodeError:
        raise ValueError(f"Error reading {workload_run}")

    db_name = workload_run.parent.name  # This is basically the database name
    #assert db_name in [db.db_name for db in database_list], f"Database {db_name} not found in the list of databases"

    db_count = 0
    for plan_id, plan in enumerate(run.parsed_plans):
        plan.database_id = db_name
        plan.plan_id = plan_id
        plans.append(plan)
        db_count += 1

    print("Database {:s} has {:d} plans.".format(str(db_name), db_count))
    return plans


def scale_feature(feature_statistics: dict, feature: str, node: SimpleNamespace) -> np.ndarray:
    """Scaling a feature according to feature statistics"""
    if feature_statistics[feature]["type"] == str(FeatureType.numeric):
        scaler = feature_statistics[feature]["scaler"]
        if hasattr(node, feature):
            attribute = getattr(node, feature)
        else:
            assert feature == "act_card"
            feature = "est_card" # In some rare cases, postgres has no act_card, use est card instead
            attribute = getattr(node, feature)

        return scaler.transform(np.array([attribute]).reshape(-1, 1))
    else:
        return feature_statistics[feature]["value_dict"][node["op_type"]]


def generate_seqs_encoding(seq: list, op_name_to_one_hot: dict, plan_parameters: list,
                           feature_statistics: dict) -> np.ndarray:
    seq_encoding = []
    for node in seq:
        # add op_name encoding
        op_encoding = op_name_to_one_hot[node.op_name]
        seq_encoding.append(op_encoding)
        # add other features, and scale them
        for feature in plan_parameters[1:]:
            feature_encoding = scale_feature(feature_statistics, feature, node)
            seq_encoding.append(feature_encoding)
    seq_encoding = np.concatenate(seq_encoding, axis=1)
    return seq_encoding


def get_attention_mask(adj: list, seq_length: int, pad_length: int, node_length, heights: list) -> torch.Tensor:
    """
    # adjs:         List, each element is a tuple of (parent, child)
    # seqs_length:  List, each element is the length of a seq
    # pad_length:   int, the length of the padded seqs
    # return:       attention mask
    """

    seq_length = int(seq_length / node_length)
    attention_mask_seq = np.ones((pad_length, pad_length))
    for a in adj:
        attention_mask_seq[a[0], a[1]] = 0

    # based on the reachability of the graph, set the attention mask
    for i in range(seq_length):
        for j in range(seq_length):
            if attention_mask_seq[i, j] == 0:
                for k in range(seq_length):
                    if attention_mask_seq[j, k] == 0:
                        attention_mask_seq[i, k] = 0

    # node can reach itself
    for i in range(pad_length):
        attention_mask_seq[i, i] = 0

    # Convert to tensor
    attention_mask_seq = torch.tensor(attention_mask_seq, dtype=torch.bool)
    return attention_mask_seq


def get_plan_sequence(plan: SimpleNamespace, pad_length: int = 20) -> Tuple[list, list, list, list]:
    """
    plan:           a plan read from json file
    pad_length:     int, the length of the padded seqs (the number of nodes in the plan)
    return:         seq, run_times, adjs, heights, database_id
    seq:            List, each element is a node's plan_parameters
    run_times:      List, each element is a node's runtime
    adjs:           List, each element is a tuple of (parent, child)
    heights:        List, each element is a node's height
    database_id: int, the id of the database
    """
    seq = []
    run_times = []
    adjs = []  # [(parent, child)]
    heights = []  # the height of each node, root node's height is 0
    depth_first_search(plan, seq, adjs, -1, run_times, heights, 0)

    # padding run_times to the same length
    if len(run_times) < pad_length:
        run_times = run_times + [1] * (pad_length - len(run_times))
    return seq, run_times, adjs, heights


def depth_first_search(plan: SimpleNamespace, seq: list, adjs: list,
                       parent_node_id: int, run_times: list, heights: list, cur_height: int) -> None:
    cur_node_id = len(seq)
    seq.append(plan.plan_parameters)
    heights.append(cur_height)
    if hasattr(plan.plan_parameters, "act_time"):
        act_time = plan.plan_parameters.act_time
    else:
        act_time = 0.01
    run_times.append(act_time)

    if parent_node_id != -1:  # not root node
        adjs.append((parent_node_id, cur_node_id))
    if hasattr(plan, "children"):
        for child in plan.children:
            depth_first_search(child, seq, adjs, cur_node_id, run_times, heights, cur_height + 1)


def get_plan_encoding(query_plan: SimpleNamespace,
                      model_config: DACEModelConfig,
                      op_name_to_one_hot: dict,
                      plan_parameters: list,
                      feature_statistics: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Obtaining a plan encoding in the form of plan_meta: (seq_encoding, run_times, attention_mask, loss_mask, database_id)
    plan:       a plan read from json file
    pad_length: int, the length of the padded seqs (the number of nodes in the plan)
    """

    seq, run_times, adjacency_matrix, heights = get_plan_sequence(query_plan, model_config.pad_length)
    assert len(seq) == len(heights)

    # Normalize runtimes
    run_times = np.array(run_times).astype(np.float32) / model_config.max_runtime + 1e-7
    run_times = torch.from_numpy(run_times)

    # Encode seq
    seq_encoding = generate_seqs_encoding(seq, op_name_to_one_hot, plan_parameters, feature_statistics)

    # Pad seq_encoding
    seq_encoding, seq_length = pad_sequence(seq_encoding=seq_encoding,
                                            padding_value=0,
                                            node_length=model_config.node_length,
                                            max_length=model_config.pad_length)

    # get attention mask
    attention_mask = get_attention_mask(adjacency_matrix,
                                        seq_length,
                                        model_config.pad_length,
                                        model_config.node_length,
                                        heights)

    # get loss mask
    loss_mask = get_loss_mask(seq_length,
                              model_config.pad_length,
                              model_config.node_length,
                              heights,
                              model_config.loss_weight)

    return seq_encoding, attention_mask, loss_mask, run_times
