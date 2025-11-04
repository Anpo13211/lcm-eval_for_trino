"""
Trino向けのDACEデータセットモジュール

主な変更点：
1. Trinoのプラン構造に対応（plan_parametersはSimpleNamespace形式、PostgreSQLと統一）
2. サブプランの実行時間が取れないため、ルートノードの実行時間のみを訓練ラベルとして使用
3. loss_maskをルートノード（index 0）のみ1.0に設定
4. Trinoの時間メトリクス（act_cpu_time, act_scheduled_time）に対応
5. Trinoの特徴量マッピング（est_rows→est_card, act_output_rows→act_card）
"""

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
                           dataloader_options: DataLoaderOptions,
                           preloaded_plans=None) -> Tuple[dict, DataLoader, DataLoader, list[DataLoader]]:
    """
    Create DACE dataloaders.
    
    Args:
        preloaded_plans: Optional dict mapping file paths to already loaded plans
    """
    feature_statistics = load_json(statistics_file, namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"

    train_loader, val_loader, test_loaders = Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]

    dataloader_args = dict(batch_size=model_config.batch_size,
                           shuffle=dataloader_options.shuffle,
                           num_workers=model_config.num_workers,
                           pin_memory=dataloader_options.pin_memory,
                           collate_fn=functools.partial(dace_collator_trino,
                                                        feature_statistics=feature_statistics,
                                                        config=model_config))

    if workload_runs.train_workload_runs:
        train_dataset, val_dataset = create_dace_datasets(workload_run_paths=workload_runs.train_workload_runs,
                                                          model_config=model_config,
                                                          val_ratio=dataloader_options.val_ratio,
                                                          preloaded_plans=preloaded_plans)
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
                                                    val_ratio=0.0,
                                                    preloaded_plans=preloaded_plans)
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


def dace_collator_trino(batch: Tuple, feature_statistics: dict, config: DACEModelConfig):
    """
    Trino向けのcollator関数
    
    主な変更点：
    - Trinoのプラン構造に対応
    - ルートノードの実行時間のみを訓練ラベルとして使用
    """
    # Get plan encodings
    add_numerical_scalers(feature_statistics)

    # Get op_name to one-hot, using feature_statistics
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)

    labels, sample_idxs, seq_encodings, attention_masks, loss_masks, all_runtimes = [], [], [], [], [], []

    for sample_idx, p in batch:
        sample_idxs.append(sample_idx)
        seq_encoding, attention_mask, loss_mask, run_times = get_plan_encoding_trino(
            query_plan=p,
            model_config=config,
            op_name_to_one_hot=op_name_to_one_hot,
            plan_parameters=config.featurization.PLAN_FEATURES,
            feature_statistics=feature_statistics
        )
        
        # Trinoでは全体の実行時間のみを使用
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


def get_loss_mask_trino(seq_length: int,
                        pad_length: int,
                        node_length: int,
                        heights: list,
                        loss_weight: float = 0.5) -> torch.Tensor:
    """
    Trino向けのloss mask生成関数
    
    Trinoではサブプランの実行時間が取れないため、
    ルートノード（index 0）のみに重み1.0を設定し、
    他のノードは0にする。これにより、全体の実行時間のみで訓練される。
    
    Args:
        seq_length: シーケンス長（ノード特徴量の合計長）
        pad_length: パディング後の長さ（ノード数）
        node_length: 1ノードあたりの特徴量長
        heights: 各ノードの高さ（ルートノードは0）- Trinoモードでは使用しない
        loss_weight: 重み（Trinoモードでは使用しない）
    
    Returns:
        loss_mask: ルートノードのみ1.0、他は0.0のマスク
    """
    seq_length = int(seq_length / node_length)
    loss_mask = np.zeros(pad_length)
    
    # Trinoモード: ルートノード（index 0）のみ重み1.0を設定
    if seq_length > 0:
        loss_mask[0] = 1.0
    
    loss_mask = torch.from_numpy(loss_mask).float()
    return loss_mask


def read_workload_run(workload_run: Path) -> list[SimpleNamespace]:
    """
    Workload runを読み込む（JSONまたはTXTファイル対応）
    
    JSONファイルの場合: 従来通りJSONから読み込む
    TXTファイルの場合: TrinoのEXPLAIN ANALYZE形式から直接読み込む
    """
    workload_path = Path(workload_run)
    
    # ファイル拡張子で判断
    if workload_path.suffix.lower() == '.txt':
        # Trinoの.txtファイルから読み込む
        return read_workload_run_trino(workload_path)
    else:
        # JSONファイルから読み込む（従来の方法）
        plans: list[SimpleNamespace] = []
        try:
            run = load_json(str(workload_path))
        except json.JSONDecodeError:
            raise ValueError(f"Error reading {workload_run}")

        db_name = workload_path.parent.name  # This is basically the database name

        db_count = 0
        for plan_id, plan in enumerate(run.parsed_plans):
            plan.database_id = db_name
            plan.plan_id = plan_id
            plans.append(plan)
            db_count += 1

        print("Database {:s} has {:d} plans.".format(str(db_name), db_count))
        return plans


def read_workload_run_trino(workload_run: Path) -> list[SimpleNamespace]:
    """
    TrinoのEXPLAIN ANALYZE .txtファイルからプランを読み込む
    """
    from trino_lcm.scripts.train_zeroshot import load_plans_from_files
    from types import SimpleNamespace
    
    # TrinoPlanOperatorのリストを取得
    trino_plans = load_plans_from_files([str(workload_run)])
    
    # TrinoPlanOperatorをSimpleNamespaceに変換
    plans: list[SimpleNamespace] = []
    db_name = workload_run.parent.name
    
    for plan_id, trino_plan in enumerate(trino_plans):
        # TrinoPlanOperatorをSimpleNamespaceに変換
        plan_ns = convert_trino_plan_to_namespace(trino_plan)
        plan_ns.database_id = db_name
        plan_ns.plan_id = plan_id
        plans.append(plan_ns)
    
    print("Database {:s} has {:d} plans (from Trino .txt file).".format(str(db_name), len(plans)))
    return plans


def convert_trino_plan_to_namespace(trino_plan) -> SimpleNamespace:
    """
    TrinoPlanOperatorをSimpleNamespaceに変換
    
    DACEの関数が期待する形式に変換:
    - plan.plan_parameters: SimpleNamespace（既にSimpleNamespace）
    - plan.plan_runtime: 実行時間（ミリ秒）
    - plan.children: 子ノードのリスト
    """
    from types import SimpleNamespace
    
    # plan_parametersは既にSimpleNamespaceなのでそのまま使用
    plan_params = trino_plan.plan_parameters if hasattr(trino_plan, 'plan_parameters') else SimpleNamespace()
    
    # plan_runtimeを取得（ミリ秒）
    plan_runtime = getattr(trino_plan, 'plan_runtime', 0)
    
    # 子ノードを再帰的に変換
    children = []
    if hasattr(trino_plan, 'children') and trino_plan.children:
        for child in trino_plan.children:
            children.append(convert_trino_plan_to_namespace(child))
    
    # SimpleNamespaceを作成
    plan_ns = SimpleNamespace()
    plan_ns.plan_parameters = plan_params
    plan_ns.plan_runtime = plan_runtime
    plan_ns.children = children
    
    return plan_ns


def scale_feature_trino(feature_statistics: dict, feature: str, node_params: SimpleNamespace) -> np.ndarray:
    """
    Trino向けの特徴量スケーリング関数
    
    Args:
        feature_statistics: 特徴量統計情報
        feature: スケーリングする特徴量名
        node_params: ノードのパラメータ（SimpleNamespace形式）
    
    Returns:
        スケーリング済みの特徴量
    """
    # 統計ファイルに存在しない特徴量の場合はデフォルト値0を返す
    if feature not in feature_statistics:
        # Trinoにはない特徴量（例: est_cost）の場合は0を返す
        return np.array([[0.0]])
    
    if feature_statistics[feature]["type"] == str(FeatureType.numeric):
        scaler = feature_statistics[feature]["scaler"]
        
        # Trinoの特徴量マッピング
        if feature == "est_card":
            attribute = getattr(node_params, "est_rows", 0)
        elif feature == "act_card":
            # Trinoでは act_output_rows を使用
            attribute = getattr(node_params, "act_output_rows", None)
            if attribute is None:
                # フォールバック: est_rows を使用
                attribute = getattr(node_params, "est_rows", 0)
        elif feature == "est_cost":
            # Trinoでは est_cpu を優先（Estimatesのcpu値、推定値なのでより適切）
            # フォールバック: est_cpuがない場合はact_cpu_timeを使用、それもなければact_scheduled_time、それもなければ0.0
            attribute = getattr(node_params, "est_cpu", None)
            if attribute is None:
                attribute = getattr(node_params, "act_cpu_time", None)
            if attribute is None:
                attribute = getattr(node_params, "act_scheduled_time", None)
            if attribute is None:
                attribute = 0.0
        else:
            attribute = getattr(node_params, feature, 0)
        
        return scaler.transform(np.array([attribute]).reshape(-1, 1))
    else:
        # カテゴリカル特徴量の場合
        op_type = getattr(node_params, "op_type", "Unknown")
        return feature_statistics[feature]["value_dict"].get(op_type, 0)


def generate_seqs_encoding_trino(seq: list, op_name_to_one_hot: dict, plan_parameters: list,
                                  feature_statistics: dict) -> np.ndarray:
    """
    Trino向けのシーケンスエンコーディング生成関数
    
    Args:
        seq: ノードのplan_parameters（SimpleNamespace形式）のリスト
        op_name_to_one_hot: オペレータ名からone-hotベクトルへのマッピング
        plan_parameters: エンコードする特徴量のリスト（例: ["op_name", "est_cost", "est_card"]）
        feature_statistics: 特徴量統計情報
    
    Returns:
        エンコード済みのシーケンス
    """
    seq_encoding = []
    for node_params in seq:
        # op_name エンコーディングを追加
        op_name = getattr(node_params, "op_name", "Unknown")
        
        if op_name not in op_name_to_one_hot:
            # 未知のop_nameの場合は、ゼロベクトルを使用
            op_names_no = len(op_name_to_one_hot)
            op_encoding = np.zeros((1, op_names_no), dtype=np.int32)
        else:
            op_encoding = op_name_to_one_hot[op_name]
        seq_encoding.append(op_encoding)
        
        # 他の特徴量を追加してスケーリング
        for feature in plan_parameters[1:]:  # op_name以外
            # 統計ファイルに存在する特徴量のみを処理
            if feature in feature_statistics:
                feature_encoding = scale_feature_trino(feature_statistics, feature, node_params)
                seq_encoding.append(feature_encoding)
            else:
                # 存在しない特徴量（例: est_cost）の場合は0を返す
                seq_encoding.append(np.array([[0.0]]))
    
    seq_encoding = np.concatenate(seq_encoding, axis=1)
    return seq_encoding


def get_attention_mask(adj: list, seq_length: int, pad_length: int, node_length, heights: list) -> torch.Tensor:
    """
    Attention maskを生成
    
    Args:
        adj: 隣接リスト [(parent, child), ...]
        seq_length: シーケンス長
        pad_length: パディング後の長さ
        node_length: ノード長
        heights: ノードの高さリスト
    
    Returns:
        attention_mask: 0はattention可能、1は不可能
    """
    seq_length = int(seq_length / node_length)
    attention_mask_seq = np.ones((pad_length, pad_length))
    
    # 隣接行列から到達可能性を設定
    for a in adj:
        attention_mask_seq[a[0], a[1]] = 0

    # グラフの到達可能性に基づいてattention maskを設定
    for i in range(seq_length):
        for j in range(seq_length):
            if attention_mask_seq[i, j] == 0:
                for k in range(seq_length):
                    if attention_mask_seq[j, k] == 0:
                        attention_mask_seq[i, k] = 0

    # ノードは自分自身に到達可能
    for i in range(pad_length):
        attention_mask_seq[i, i] = 0

    # テンソルに変換
    attention_mask_seq = torch.tensor(attention_mask_seq, dtype=torch.bool)
    return attention_mask_seq


def get_plan_sequence_trino(plan: SimpleNamespace, pad_length: int = 20) -> Tuple[list, list, list, list]:
    """
    Trino向けのプランシーケンス生成関数
    
    Trinoのプラン構造を走査し、ノードのシーケンス、実行時間、隣接行列、高さを取得
    
    Args:
        plan: Trinoプラン（SimpleNamespace形式）
        pad_length: パディング長（ノード数）
    
    Returns:
        seq: ノードのplan_parameters（辞書）のリスト
        run_times: 各ノードの実行時間リスト（Trinoではデフォルト値）
        adjs: 隣接リスト [(parent, child), ...]
        heights: 各ノードの高さリスト
    """
    seq = []
    run_times = []
    adjs = []  # [(parent, child)]
    heights = []  # the height of each node, root node's height is 0
    depth_first_search_trino(plan, seq, adjs, -1, run_times, heights, 0)

    # padding run_times to the same length
    if len(run_times) < pad_length:
        run_times = run_times + [1] * (pad_length - len(run_times))
    return seq, run_times, adjs, heights


def depth_first_search_trino(plan: SimpleNamespace, seq: list, adjs: list,
                              parent_node_id: int, run_times: list, heights: list, cur_height: int) -> None:
    """
    Trino向けのDFS関数
    
    Trinoのプラン構造を深さ優先探索で走査
    
    主な変更点：
    1. plan_parametersがSimpleNamespace形式（PostgreSQLと統一）
    2. act_timeの代わりにact_cpu_timeやact_scheduled_timeを使用
    3. Trinoではサブプランの実行時間が正確に取れないため、デフォルト値を使用
    
    Args:
        plan: 現在のプランノード
        seq: ノードのplan_parametersを格納するリスト
        adjs: 隣接リストを格納するリスト
        parent_node_id: 親ノードのID（-1はルート）
        run_times: 各ノードの実行時間を格納するリスト
        heights: 各ノードの高さを格納するリスト
        cur_height: 現在の高さ
    """
    cur_node_id = len(seq)
    
    # plan_parametersを追加（SimpleNamespace形式）
    if hasattr(plan, 'plan_parameters'):
        seq.append(plan.plan_parameters)
    else:
        # plan_parametersがない場合はデフォルト値
        from types import SimpleNamespace
        seq.append(SimpleNamespace(op_name="Unknown", est_rows=0, est_cost=0))
    
    heights.append(cur_height)
    
    # Trinoの実行時間を取得
    # ルートノードの場合はplan_runtimeを使用、それ以外は0
    if cur_height == 0 and hasattr(plan, 'plan_runtime'):
        # ルートノードの実行時間（ミリ秒）
        act_time = plan.plan_runtime
    else:
        # サブノードの実行時間はTrinoでは取得できないので、0に設定
        # 損失計算では使用されない（loss_maskが0のため）
        act_time = 0.0

        # cpu_time を使う場合は以下を使う
        # if hasattr(plan, 'plan_parameters'):
        #     # act_cpu_time または act_scheduled_time があれば使用
        #     act_time = getattr(plan.plan_parameters, 'act_cpu_time', None)
        #     if act_time is None:
        #         act_time = getattr(plan.plan_parameters, 'act_scheduled_time', 0.01)
    
    run_times.append(act_time)

    # 親ノードとの関係を追加
    if parent_node_id != -1:  # not root node
        adjs.append((parent_node_id, cur_node_id))
    
    # 子ノードを再帰的に処理
    if hasattr(plan, "children") and plan.children:
        for child in plan.children:
            depth_first_search_trino(child, seq, adjs, cur_node_id, run_times, heights, cur_height + 1)


def get_plan_encoding_trino(query_plan: SimpleNamespace,
                            model_config: DACEModelConfig,
                            op_name_to_one_hot: dict,
                            plan_parameters: list,
                            feature_statistics: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Trino向けのプランエンコーディング関数
    
    プランからエンコーディング、attention mask、loss maskを生成
    
    Args:
        query_plan: Trinoクエリプラン
        model_config: DACEモデル設定
        op_name_to_one_hot: オペレータ名からone-hotへのマッピング
        plan_parameters: エンコードする特徴量リスト
        feature_statistics: 特徴量統計情報
    
    Returns:
        seq_encoding: エンコード済みシーケンス
        attention_mask: Attention mask
        loss_mask: Loss mask（Trinoではルートノードのみ1.0）
        run_times: 実行時間（正規化済み）
    """
    # プランシーケンスを取得
    seq, run_times, adjacency_matrix, heights = get_plan_sequence_trino(query_plan, model_config.pad_length)
    assert len(seq) == len(heights)

    # 実行時間を正規化
    run_times = np.array(run_times).astype(np.float32) / model_config.max_runtime + 1e-7
    run_times = torch.from_numpy(run_times)

    # シーケンスをエンコード
    seq_encoding = generate_seqs_encoding_trino(seq, op_name_to_one_hot, plan_parameters, feature_statistics)

    # シーケンスをパディング
    seq_encoding, seq_length = pad_sequence(seq_encoding=seq_encoding,
                                            padding_value=0,
                                            node_length=model_config.node_length,
                                            max_length=model_config.pad_length)

    # attention maskを取得
    attention_mask = get_attention_mask(adjacency_matrix,
                                        seq_length,
                                        model_config.pad_length,
                                        model_config.node_length,
                                        heights)

    # loss maskを取得（Trinoモード: ルートノードのみ1.0）
    loss_mask = get_loss_mask_trino(seq_length,
                                    model_config.pad_length,
                                    model_config.node_length,
                                    heights,
                                    model_config.loss_weight)

    return seq_encoding, attention_mask, loss_mask, run_times

