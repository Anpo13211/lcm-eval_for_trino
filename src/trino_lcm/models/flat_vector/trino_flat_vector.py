"""
Trino Flat-Vector Model Implementation

Trinoクエリプラン用のFlat-Vectorモデル。
クエリプランを平坦化し、演算子タイプとカーディナリティから特徴量を抽出して、
LightGBMで実行時間を予測する。

主な機能:
- load_trino_plans_from_files: Trinoプランファイルからプランを読み込む
- collect_operator_types: 演算子タイプを収集してインデックスに変換
- extract_flat_features: プランから特徴ベクトルを抽出
- create_flat_vector_dataset: データセットを作成
- train_flat_vector_model: モデルをトレーニング
- predict_flat_vector_model: モデルで予測
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.trino.parse_plan import parse_trino_plans, trino_timing_regex


class MockQuery:
    """モックのQueryクラス（Trinoプランのパース用）"""
    
    def __init__(self, plan_text):
        self.plan_text = plan_text
        self.timeout = False
        self.analyze_plans = [plan_text]
        self.verbose_plan = plan_text.split('\n')
        
        # 実行時間を抽出
        execution_time = None
        timing_match = trino_timing_regex.search(plan_text)
        if timing_match:
            execution_time = float(timing_match.group(4))
            execution_unit = timing_match.group(5)
            if execution_unit == 's':
                execution_time *= 1000
        
        if execution_time is None:
            execution_time_match = re.search(r'Execution(?: Time)?: ([\d.]+)(ms|s)', plan_text)
            if execution_time_match:
                execution_time = float(execution_time_match.group(1))
                if execution_time_match.group(2) == 's':
                    execution_time *= 1000
        
        self.execution_time = execution_time if execution_time is not None else 1000.0


class MockRunStats:
    """モックのRunStatsクラス（Trinoプランのパース用）"""
    
    def __init__(self, plans_text):
        self.plans_text = plans_text
        self.query_list = [MockQuery(plan_text) for plan_text in plans_text]
        self.database_stats = {}
    
    def __iter__(self):
        for plan_text in self.plans_text:
            yield plan_text


def load_trino_plans_from_files(file_paths: List[Path], max_plans_per_file=None, verbose=True):
    """
    複数のファイルからTrinoプランを読み込み
    
    Args:
        file_paths: プランファイルのパスのリスト
        max_plans_per_file: 各ファイルから読み込む最大プラン数
        verbose: 詳細ログを出力するか
    
    Returns:
        全プランのリスト
    """
    all_plans = []
    
    if verbose:
        print(f"📂 {len(file_paths)}個のファイルからプランを読み込み中...")
    
    iterator = tqdm(file_paths, desc="ファイル読み込み") if verbose else file_paths
    
    for file_idx, file_path in enumerate(iterator):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # クエリプランを分割
        plans_text = []
        current_plan = []
        
        for line in content.split('\n'):
            if line.startswith('-- ') and 'stmt' in line and current_plan:
                plans_text.append('\n'.join(current_plan))
                current_plan = [line]
            else:
                current_plan.append(line)
        
        if current_plan:
            plans_text.append('\n'.join(current_plan))
        
        # 最大プラン数を制限
        if max_plans_per_file:
            plans_text = plans_text[:max_plans_per_file]
        
        if verbose:
            print(f"  - {file_path}: {len(plans_text)}個のプランを検出")
        
        # MockRunStatsを作成してパース
        mock_stats = MockRunStats(plans_text)
        
        parsed_runs, _ = parse_trino_plans(
            mock_stats,
            min_runtime=0,
            max_runtime=1000000,
            parse_baseline=False,
            include_zero_card=True
        )
        
        if verbose:
            print(f"  - パース結果: {len(parsed_runs['parsed_plans'])}個のプラン")
        
        # データベースIDを設定
        for plan in parsed_runs['parsed_plans']:
            plan.database_id = file_idx
        
        all_plans.extend(parsed_runs['parsed_plans'])
    
    if verbose:
        print(f"  - 総プラン数: {len(all_plans)}")
    
    return all_plans


def collect_operator_types(plans) -> Dict[str, int]:
    """
    プランから演算子タイプを収集してインデックスに変換
    
    Args:
        plans: TrinoPlanOperatorオブジェクトのリスト
    
    Returns:
        演算子名 -> インデックスの辞書
    """
    op_types = set()
    
    def extract_op_types(plan):
        op_name = getattr(plan.plan_parameters, 'op_name', None)
        if op_name:
            op_types.add(op_name)
        for child in plan.children:
            extract_op_types(child)
    
    for plan in plans:
        extract_op_types(plan)
    
    # ソートして再現性を確保
    sorted_ops = sorted(list(op_types))
    op_idx_dict = {op: idx for idx, op in enumerate(sorted_ops)}
    
    print(f"📊 見つかった演算子タイプ: {len(op_idx_dict)} 種類")
    for op_name, idx in sorted(op_idx_dict.items(), key=lambda x: x[1]):
        print(f"  - {idx:3d}: {op_name}")
    
    return op_idx_dict


def extract_flat_features(plan, op_idx_dict: Dict[str, int], use_act_card=True):
    """
    Flat-Vector特徴量を抽出
    
    Args:
        plan: TrinoPlanOperatorオブジェクト
        op_idx_dict: 演算子名 -> インデックスの辞書
        use_act_card: True=実際のカーディナリティ、False=推定カーディナリティ
    
    Returns:
        特徴ベクトル（numpy配列）
    """
    no_ops = len(op_idx_dict)
    
    # 演算子の出現回数
    feature_num_vec = np.zeros(no_ops)
    # 演算子ごとの行数
    feature_row_vec = np.zeros(no_ops)
    
    def extract_features_recursive(p):
        op_name = getattr(p.plan_parameters, 'op_name', None)
        if not op_name:
            return
        
        if op_name not in op_idx_dict:
            # 未知の演算子タイプ（学習時に見なかった演算子）
            print(f"⚠️  警告: 未知の演算子タイプ '{op_name}' をスキップ")
            return
        
        op_idx = op_idx_dict[op_name]
        feature_num_vec[op_idx] += 1
        
        # カーディナリティの抽出
        if use_act_card:
            # 実際のカーディナリティを使用（SimpleNamespace対応）
            card = getattr(p.plan_parameters, 'act_output_rows', None)
            if card is None:
                card = getattr(p.plan_parameters, 'act_card', None)
            if card is None:
                # フォールバック: 実際のカーディナリティがない場合は推定値を使用
                card = getattr(p.plan_parameters, 'est_rows', 0)
        else:
            # 推定カーディナリティを使用（SimpleNamespace対応）
            card = getattr(p.plan_parameters, 'est_rows', None)
            if card is None:
                card = getattr(p.plan_parameters, 'est_card', 0)
        
        feature_row_vec[op_idx] += card
        
        # 子ノードを再帰的に処理
        for child in p.children:
            extract_features_recursive(child)
    
    extract_features_recursive(plan)
    
    # 特徴ベクトルを連結
    feature_vec = np.concatenate((feature_num_vec, feature_row_vec))
    
    return feature_vec


def create_flat_vector_dataset(plans, op_idx_dict: Dict[str, int], use_act_card=True, verbose=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    データセットを作成
    
    Args:
        plans: TrinoPlanOperatorオブジェクトのリスト
        op_idx_dict: 演算子名 -> インデックスの辞書
        use_act_card: True=実際のカーディナリティ、False=推定カーディナリティ
        verbose: 詳細ログを出力するか
    
    Returns:
        (特徴行列, ラベル配列)
    """
    feature_vecs = []
    labels = []
    
    iterator = tqdm(plans, desc="特徴量抽出") if verbose else plans
    
    for plan in iterator:
        feature_vec = extract_flat_features(plan, op_idx_dict, use_act_card)
        feature_vecs.append(feature_vec)
        
        # ラベルは実行時間（秒単位）
        runtime_ms = plan.plan_runtime
        runtime_s = runtime_ms / 1000.0
        labels.append(runtime_s)
    
    X = np.array(feature_vecs)
    y = np.array(labels)
    
    return X, y


def train_flat_vector_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 20,
    seed: int = 42,
    verbose: bool = True
) -> lgb.Booster:
    """
    Flat-Vectorモデルをトレーニング
    
    Args:
        X_train: トレーニング特徴量
        y_train: トレーニングラベル
        X_val: 検証特徴量
        y_val: 検証ラベル
        num_boost_round: ブースティングラウンド数
        early_stopping_rounds: 早期停止ラウンド数
        seed: ランダムシード
        verbose: 詳細ログを出力するか
    
    Returns:
        トレーニング済みのLightGBMモデル
    """
    if verbose:
        print("🚀 LightGBMモデルのトレーニング開始")
    
    # LightGBMデータセットの作成
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # ハイパーパラメータ
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1 if not verbose else 0,
        'random_state': seed,
        'seed': seed,
        'bagging_seed': seed,
        'feature_fraction_seed': seed
    }
    
    # コールバック
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose)
    ]
    if verbose:
        callbacks.append(lgb.log_evaluation(period=50))
    
    # トレーニング
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    return bst


def predict_flat_vector_model(bst: lgb.Booster, X: np.ndarray, lower_bound: float = 0.01) -> np.ndarray:
    """
    モデルで予測
    
    Args:
        bst: LightGBMモデル
        X: 特徴量
        lower_bound: 予測値の下限
    
    Returns:
        予測値（秒単位）
    """
    predictions = bst.predict(X, num_iteration=bst.best_iteration)
    
    # 下限値を適用
    predictions = np.maximum(predictions, lower_bound)
    
    return predictions

