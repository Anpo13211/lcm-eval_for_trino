"""
Trino Zero-Shot Model Training Script

Trinoクエリプラン向けのZero-Shotモデル（Graph Neural Network）のトレーニング。

Usage:
    # ルートディレクトリから実行
    python -m trino_lcm.scripts.train_zeroshot \
        --train_files accidents_valid_verbose.txt \
        --test_file accidents_valid_verbose.txt \
        --output_dir models/trino_zeroshot \
        --statistics_dir datasets_statistics \
        --catalog iceberg \
        --schema imdb
"""

import sys
import os
import warnings

# Suppress torchdata deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchdata')

# 環境変数の設定（必須 - import前に実行）
for i in range(11):
    env_key = f'NODE{i:02d}'
    env_value = os.environ.get(env_key)
    # `.env` で "None" や空文字が設定されていると ast.literal_eval が失敗するため明示的に初期化する
    if env_value in (None, '', 'None'):
        os.environ[env_key] = '[]'

# ZERO_SHOT_DATASETS_DIR環境変数の設定（column_statistics.jsonから統計情報を取得するため）
# デフォルトパスを設定（環境変数が既に設定されている場合は上書きしない）
if 'ZERO_SHOT_DATASETS_DIR' not in os.environ:
    default_zero_shot_dir = '/Users/an/query_engine/lakehouse/zero-shot_datasets'
    if os.path.exists(default_zero_shot_dir):
        os.environ['ZERO_SHOT_DATASETS_DIR'] = default_zero_shot_dir
        print(f"ℹ️  ZERO_SHOT_DATASETS_DIR を設定しました: {default_zero_shot_dir}")

# スクリプトがsrc/trino_lcm/scripts/にある場合、src/を親パスに追加
from pathlib import Path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import json
import re
import functools
from pathlib import Path
from typing import Optional, Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.trino.parse_plan import parse_trino_plans, trino_timing_regex
from trino_lcm.models.zero_shot import trino_plan_collator, load_database_statistics
from models.zeroshot.zero_shot_model import ZeroShotModel
from training.featurizations import TrinoTrueCardDetail
from classes.classes import ZeroShotModelConfig
from training.preprocessing.feature_statistics import gather_feature_statistics, FeatureType
from training.training.metrics import QError, RMSE


class TrinoPlanDataset(Dataset):
    """Trinoクエリプランのデータセット"""
    
    def __init__(self, plans):
        """
        Args:
            plans: TrinoPlanOperatorオブジェクトのリスト
        """
        self.plans = plans
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return idx, self.plans[idx]


class MockQuery:
    """モックのQueryクラス"""
    
    def __init__(self, plan_text):
        self.plan_text = plan_text
        self.timeout = False
        self.analyze_plans = [plan_text]  # parse_trino_plansが期待する形式
        
        # verbose_planは文字列形式で提供（parse_trino_raw_plan_v2が期待する形式）
        self.verbose_plan = plan_text
        
        # SQL文を抽出（オプション、プランテキストから抽出を試みる）
        # 最初の'-- stmt'で始まる行を探す
        sql_lines = []
        for line in plan_text.split('\n'):
            if line.strip().startswith('-- stmt'):
                # SQL文の行を抽出（次の行から）
                sql_lines = []
            elif sql_lines is not None and line.strip() and not line.strip().startswith('--'):
                sql_lines.append(line.strip())
                if sql_lines and line.strip().endswith(';'):
                    break
        self.sql = ' '.join(sql_lines) if sql_lines else 'SELECT * FROM unknown'  # デフォルトのSQL
        
        # 実行時間を抽出
        execution_time = None
        
        timing_match = trino_timing_regex.search(plan_text)
        if timing_match:
            # 正規表現のグループ: 1=Queued値, 2=Queued単位, 3=Analysis値, 4=Analysis単位,
            # 5=Planning値, 6=Planning単位, 7=Execution値, 8=Execution単位
            execution_time = float(timing_match.group(7))  # Execution値
            execution_unit = timing_match.group(8)  # Execution単位
            if execution_unit and execution_unit == 's':
                execution_time *= 1000
            elif execution_unit and execution_unit in ('us', 'μs'):
                execution_time /= 1000
            elif execution_unit and execution_unit == 'm':
                execution_time *= 60000
        
        if execution_time is None:
            # 古い書式 ("Execution Time: <value><unit>") にも対応
            execution_time_match = re.search(r'Execution(?: Time)?: ([\d.]+)(ms|s)', plan_text)
            if execution_time_match:
                execution_time = float(execution_time_match.group(1))
                if execution_time_match.group(2) == 's':
                    execution_time *= 1000
        
        self.execution_time = execution_time if execution_time is not None else 1000.0  # デフォルトの実行時間（ミリ秒）


class MockRunStats:
    """モックのRunStatsクラス（テキストファイル用）"""
    
    def __init__(self, plans_text):
        self.plans_text = plans_text
        # parse_trino_plansで必要な属性を追加
        self.query_list = [MockQuery(plan_text) for plan_text in plans_text]
        
        # database_statsをSimpleNamespace形式で初期化（parse_trino_plans_v2が期待する形式）
        from types import SimpleNamespace
        self.database_stats = SimpleNamespace(
            table_stats=[],  # リスト形式
            column_stats=[]  # リスト形式
        )
        
        # run_kwargsを追加（parse_trino_plans_v2が期待する形式）
        self.run_kwargs = {}
    
    def __iter__(self):
        for plan_text in self.plans_text:
            yield plan_text


def split_query_plans(file_path):
    """クエリプランファイルを個別のプランに分割"""
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
    
    return plans_text


def load_plans_from_files(file_paths, max_plans_per_file=None):
    """
    複数のファイルからプランを読み込み
    
    Args:
        file_paths: プランファイルのパスのリスト
        max_plans_per_file: 各ファイルから読み込む最大プラン数
    
    Returns:
        全プランのリスト
    """
    all_plans = []
    
    print(f"📂 {len(file_paths)}個のファイルからプランを読み込み中...")
    
    for file_idx, file_path in enumerate(tqdm(file_paths, desc="ファイル読み込み")):
        # JSONファイルの場合
        if str(file_path).endswith('.json'):
            with open(file_path, 'r') as f:
                run_data = json.load(f)
            
            # parse_trino_plansを使用してプランを解析
            # TODO: JSONからrun_statsオブジェクトを再構築
            pass
        
        # テキストファイルの場合（EXPLAIN ANALYZE出力）
        else:
            # ファイルを読み込んで個別のクエリプランに分割
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
            
            print(f"  - {file_path}: {len(plans_text)}個のプランを検出")
            
            # MockRunStatsを作成してパース
            mock_stats = MockRunStats(plans_text)
            
            # parse_trino_plans_v2を使用（統計情報に対応）
            from cross_db_benchmark.benchmark_tools.trino.parse_plan import parse_trino_plans_v2
            
            # 統計情報を読み込んでdatabase_statsに設定（オプション）
            # 注意: 統計情報がない場合は空のリストで動作する
            # 統計情報の読み込みではプラン数に制限をかけない（Noneを渡す）
            try:
                from training.dataset.dataset_creation import read_explain_analyze_txt
                _, db_stats_from_txt = read_explain_analyze_txt(
                    file_path,
                    path_index=file_idx,
                    limit_per_ds=None  # 統計情報は全クエリから取得（プラン読み込みとは独立）
                )
                # database_statsを更新（リスト形式に変換）
                from types import SimpleNamespace
                mock_stats.database_stats = SimpleNamespace(
                    table_stats=list(db_stats_from_txt.table_stats.values()) if isinstance(db_stats_from_txt.table_stats, dict) else [],
                    column_stats=list(db_stats_from_txt.column_stats.values()) if isinstance(db_stats_from_txt.column_stats, dict) else []
                )
            except Exception as e:
                print(f"  ⚠️  統計情報の読み込みに失敗（統計情報なしで続行）: {e}")
            
            parsed_runs, _ = parse_trino_plans_v2(
                mock_stats,
                min_runtime=0,
                max_runtime=1000000,
                parse_baseline=False,
                include_zero_card=True
            )
            
            print(f"  - パース結果: {len(parsed_runs['parsed_plans'])}個のプラン")
            
            # データベースIDを設定
            for plan in parsed_runs['parsed_plans']:
                plan.database_id = file_idx
            
            all_plans.extend(parsed_runs['parsed_plans'])
    
    print(f"  - 総プラン数: {len(all_plans)}")
    return all_plans


def create_feature_statistics_from_plans(plans, plan_featurization, output_path=None):
    """
    プランから特徴量統計を動的に収集
    
    Args:
        plans: TrinoPlanOperatorオブジェクトのリスト
        plan_featurization: 特徴量化設定
        output_path: 統計情報の出力パス（オプション）
    
    Returns:
        feature_statistics辞書
    """
    print("📊 プランから特徴量統計を収集中...")
    
    # 実際に使用されている演算子を収集（op_name用）
    actual_op_names = set()
    # フィルター演算子も収集（operator用）
    filter_operators = set()
    
    def collect_operators(node):
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters if isinstance(node.plan_parameters, dict) else vars(node.plan_parameters)
            op_name = params.get('op_name')
            if op_name:
                actual_op_names.add(op_name)
            
            # フィルター演算子を収集
            filter_col = params.get('filter_columns')
            if filter_col:
                def collect_filter_ops(filter_node):
                    # PredicateNodeオブジェクトの場合
                    if hasattr(filter_node, 'operator'):
                        op = filter_node.operator
                        if op is not None:
                            filter_operators.add(str(op))
                    # 辞書形式の場合
                    elif isinstance(filter_node, dict) and 'operator' in filter_node:
                        op = filter_node['operator']
                        if op is not None:
                            filter_operators.add(str(op))
                    
                    # 子ノードを再帰的に処理（両形式に対応）
                    children = None
                    if hasattr(filter_node, 'children'):
                        children = filter_node.children
                    elif isinstance(filter_node, dict) and 'children' in filter_node:
                        children = filter_node['children']
                    
                    if children:
                        for child in children:
                            collect_filter_ops(child)
                
                collect_filter_ops(filter_col)
        
        if hasattr(node, 'children'):
            for child in node.children:
                collect_operators(child)
    
    for plan in plans:
        collect_operators(plan)
    
    print(f"  - 検出されたプラン演算子 (op_name): {sorted(actual_op_names)}")
    print(f"  - 検出されたフィルター演算子 (operator): {sorted(filter_operators)}")
    
    # ダミーの統計情報から開始（実際の演算子を含むように更新）
    feature_statistics = create_dummy_feature_statistics(
        plan_featurization, 
        actual_op_names=actual_op_names if actual_op_names else None,
        filter_operators=filter_operators if filter_operators else None
    )
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(feature_statistics, f, indent=2)
        print(f"  - 特徴量統計を {output_path} に保存")
    
    return feature_statistics


def create_dummy_feature_statistics(plan_featurization, actual_op_names=None, filter_operators=None):
    """
    ダミーの特徴量統計情報を作成
    
    Args:
        plan_featurization: 特徴量化設定
        actual_op_names: 実際に使用されているプラン演算子のセット（オプション）
        filter_operators: 実際に使用されているフィルター演算子のセット（オプション）
    """
    feature_statistics = {}
    
    # すべての特徴量を定義
    all_features = set()
    for features in plan_featurization.VARIABLES.values():
        all_features.update(features)
    
    for feat_name in all_features:
        if feat_name == 'op_name':
            if actual_op_names:
                # 実際のプランから収集した演算子を使用し、連続したIDを割り当てる
                sorted_ops = sorted(actual_op_names)
                operator_dict = {op: idx for idx, op in enumerate(sorted_ops)}
                print(f"  - op_name: {len(sorted_ops)}個の演算子を連続IDで割り当て")
            else:
                # フォールバック: 元のハードコードされたマッピング（後方互換性のため）
                operator_dict = {
                    'Aggregate': 0, 'LocalExchange': 1, 'RemoteSource': 2,
                    'ScanFilter': 3, 'ScanFilterProject': 4, 'Project': 5,
                    'InnerJoin': 6, 'HashJoin': 7, 'NestedLoopJoin': 8,
                    'Sort': 87, 'Limit': 95, 'TopN': 11,
                    'TableScan': 13, 'FilterProject': 14, 'Exchange': 15,
                    'LeftJoin': 32, 'ScanProject': 60, 'Filter': 61,
                    'CrossJoin': 96,  # 実際に使用されているがマッピングにない演算子を追加
                }
                max_operator_id = max(operator_dict.values()) + 1
            
            if actual_op_names:
                # 連続IDの場合、no_valsは演算子数に余裕を持たせる
                # 実際の演算子数に対して十分な余裕を持たせる（将来の拡張にも対応）
                no_vals = max(200, len(operator_dict) * 2)  # 2倍の余裕を持たせる
            else:
                # ハードコードされたIDの場合、最大ID+1を使用
                max_operator_id = max(operator_dict.values()) + 1
                no_vals = max(200, max_operator_id * 2)  # 2倍の余裕を持たせる
            
            feature_statistics[feat_name] = {
                'type': str(FeatureType.categorical),
                'value_dict': operator_dict,
                'no_vals': no_vals
            }
        elif feat_name == 'operator':
            if filter_operators:
                # 実際のプランから収集したフィルター演算子を使用し、連続したIDを割り当てる
                sorted_ops = sorted(filter_operators)
                operator_dict = {op: idx for idx, op in enumerate(sorted_ops)}
                print(f"  - operator: {len(sorted_ops)}個のフィルター演算子を連続IDで割り当て")
            else:
                # フォールバック: 元のハードコードされたマッピング
                operator_dict = {
                    'Aggregate': 0, 'LocalExchange': 1, 'RemoteSource': 2,
                    'ScanFilter': 3, 'ScanFilterProject': 4, 'Project': 5,
                    'InnerJoin': 6, 'HashJoin': 7, 'NestedLoopJoin': 8,
                    'Sort': 87, 'Limit': 95, 'TopN': 11,
                    'TableScan': 13, 'FilterProject': 14, 'Exchange': 15,
                    'LeftJoin': 32, 'ScanProject': 60, 'Filter': 61,
                    'CrossJoin': 96,
                }
            
            if filter_operators:
                # 連続IDの場合、余裕を持たせる
                max_operator_id = len(operator_dict)
                no_vals = max(200, max_operator_id * 2)  # 2倍の余裕を持たせる
            else:
                # ハードコードされたIDの場合
                max_operator_id = max(operator_dict.values()) + 1
                no_vals = max(200, max_operator_id * 2)  # 2倍の余裕を持たせる
            
            feature_statistics[feat_name] = {
                'type': str(FeatureType.categorical),
                'value_dict': operator_dict,
                'no_vals': no_vals
            }
        elif feat_name == 'aggregation':
            # 集約関数の特徴量統計
            aggregation_dict = {
                'Aggregator.COUNT': 0,
                'Aggregator.SUM': 1,
                'Aggregator.AVG': 2,
                'Aggregator.MIN': 3,
                'Aggregator.MAX': 4,
                None: 5  # 集約なし
            }
            feature_statistics[feat_name] = {
                'type': str(FeatureType.categorical),
                'value_dict': aggregation_dict,
                'no_vals': len(aggregation_dict)
            }
        elif feat_name in ['table_name', 'column_name']:
            feature_statistics[feat_name] = {
                'type': str(FeatureType.categorical),
                'value_dict': {},
                'no_vals': 1000
            }
        elif feat_name in ['rows', 'size', 'cpu', 'memory', 'network']:
            feature_statistics[feat_name] = {
                'type': str(FeatureType.numeric),
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 1000000.0,
                'center': 0.0,
                'scale': 1.0
            }
        else:
            # その他の特徴量は数値として扱う
            feature_statistics[feat_name] = {
                'type': str(FeatureType.numeric),
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 100.0,
                'center': 0.0,
                'scale': 1.0
            }
    
    return feature_statistics


def collect_feature_statistics(workload_run_paths, output_path):
    """
    Trinoワークロードから特徴量統計を収集
    
    Args:
        workload_run_paths: ワークロード実行結果のパスのリスト
        output_path: 統計情報の出力パス
    """
    print(f"📊 特徴量統計情報を収集中...")
    
    # gather_feature_statistics関数を使用
    gather_feature_statistics(workload_run_paths, output_path)
    
    print(f"✅ 特徴量統計情報を {output_path} に保存しました")


def train_epoch(model, train_loader, optimizer, device):
    """1エポックのトレーニング"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for graph, features, labels, sample_idxs in tqdm(train_loader, desc="Training"):
        # データをデバイスに転送
        graph = graph.to(device)
        features = {k: v.to(device) for k, v in features.items()}
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)
        
        optimizer.zero_grad()
        
        # フォワードパス
        predictions = model((graph, features))
        
        # 損失計算
        loss = model.loss_fxn(predictions, labels_tensor)
        
        # バックプロパゲーション
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def validate(model, val_loader, device):
    """検証"""
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions_all = []
    labels_all = []
    
    with torch.no_grad():
        for graph, features, labels, sample_idxs in tqdm(val_loader, desc="Validation"):
            # データをデバイスに転送
            graph = graph.to(device)
            features = {k: v.to(device) for k, v in features.items()}
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)
            
            # フォワードパス
            predictions = model((graph, features))
            
            # 損失計算
            loss = model.loss_fxn(predictions, labels_tensor)
            total_loss += loss.item()
            num_batches += 1
            
            predictions_all.append(predictions.cpu().numpy())
            labels_all.append(labels_tensor.cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # メトリクス計算
    if len(predictions_all) > 0:
        predictions_all = np.concatenate(predictions_all).flatten()
        labels_all = np.concatenate(labels_all).flatten()
        
        # ゼロ除算を防ぐために小さな値でクリッピング
        epsilon = 1e-6
        safe_predictions = np.clip(predictions_all, epsilon, None)
        safe_labels = np.clip(labels_all, epsilon, None)
        
        # Q-Error (metrics.pyの実装を使用)
        median_q_error = QError(percentile=50).evaluate_metric(labels=safe_labels, preds=safe_predictions)
        q_errors = np.maximum(safe_predictions / safe_labels, safe_labels / safe_predictions)
        mean_q_error = float(np.mean(q_errors))
        
        # RMSE (metrics.pyの実装を使用)
        rmse = RMSE().evaluate_metric(labels=labels_all, preds=predictions_all)
        
        return avg_loss, median_q_error, mean_q_error, rmse
    
    return avg_loss, None, None, None


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Zero-Shot training."""
    parser = argparse.ArgumentParser(description='Train Trino Zero-Shot Model (統合版)')
    
    # モード選択
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'train_multi_all'],
        default='train',
        help='Training mode: train (single dataset) or train_multi_all (leave-one-out across all datasets)'
    )
    
    # データ関連の引数
    parser.add_argument('--train_files', type=str, required=False,
                        help='トレーニング用ファイルパス（カンマ区切り、trainモードで必須）')
    parser.add_argument('--test_file', type=str, required=False,
                        help='テスト用ファイルパス（trainモードで必須）')
    parser.add_argument('--statistics_file', type=str, default=None,
                        help='特徴量統計情報ファイルのパス（オプション）')
    parser.add_argument('--statistics_dir', type=str, default=None,
                        help='データベース統計情報のルートディレクトリ（指定時のみ統計情報を使用）')
    parser.add_argument('--catalog', type=str, default=None,
                        help='Trinoカタログ名（統計情報使用時に必要）')
    parser.add_argument('--schema', type=str, default=None,
                        help='スキーマ名（統計情報使用時に必要）')
    
    # モデル関連の引数
    parser.add_argument('--output_dir', type=str, default='models/trino_zeroshot',
                        help='モデル出力ディレクトリ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='エポック数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学習率')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隠れ層の次元数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='デバイス (cuda/cpu)')
    
    # データ処理関連の引数
    parser.add_argument('--max_plans_per_file', type=int, default=None,
                        help='各ファイルから読み込む最大プラン数')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='検証セットの割合')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoaderのワーカー数')
    parser.add_argument(
        '--plans_dir',
        type=str,
        default='/Users/an/query_engine/explain_analyze_results/',
        help='Directory containing .txt plan files for multiple datasets (required for train_multi_all mode)'
    )
    
    return parser


def run(args) -> int:
    """Run Zero-Shot training with parsed arguments."""
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Trino Zero-Shot Model Training (統合版)")
    print(f"Mode: {args.mode}")
    if 'ZERO_SHOT_DATASETS_DIR' in os.environ:
        print(f"ZERO_SHOT_DATASETS_DIR: {os.environ['ZERO_SHOT_DATASETS_DIR']}")
    print("=" * 80)
    
    # train_multi_allモードの処理
    if args.mode == 'train_multi_all':
        return run_train_multi_all(args, output_dir)
    
    # 従来のtrainモード
    if not args.train_files or not args.test_file:
        raise ValueError("--train_files and --test_file are required for train mode")
    
    print(f"Train files: {args.train_files}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # データベース統計情報の準備（プランから抽出を優先）
    db_statistics = {}
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print()
    
    # 1. プランの読み込み
    print("📂 ステップ1: プランの読み込み")
    train_file_paths = [Path(p.strip()) for p in args.train_files.split(',')]
    test_file_path = Path(args.test_file)
    
    # トレーニングプランの読み込み
    train_plans = load_plans_from_files(train_file_paths, args.max_plans_per_file)
    
    # テストプランの読み込み
    test_plans = load_plans_from_files([test_file_path], args.max_plans_per_file)
    
    print()
    
    # 1.5. データベース統計情報の準備（オプション - フォールバック用）
    # 注意: 統計情報は既にplan_parametersに含まれているため、通常は外部統計ファイルは不要
    # ただし、互換性のためフォールバックとして残す
    db_statistics = {}
    if args.catalog and args.schema and args.statistics_dir:
        stats_dir_path = Path(args.statistics_dir) / f"{args.catalog}_{args.schema}"
        if stats_dir_path.exists():
            try:
                loaded_stats = load_database_statistics(
                    catalog=args.catalog,
                    schema=args.schema,
                    stats_dir=args.statistics_dir,
                    prefer_zero_shot=True
                )
                
                from types import SimpleNamespace
                for file_idx, file_path in enumerate([Path(p.strip()) for p in args.train_files.split(',')] + [Path(args.test_file)]):
                    db_stats = SimpleNamespace(
                        table_stats=loaded_stats.get('table_stats', {}),
                        column_stats=loaded_stats.get('column_stats', {})
                    )
                    db_statistics[file_idx] = db_stats
                
                has_stats = (
                    loaded_stats.get('table_stats') or 
                    loaded_stats.get('column_stats')
                )
                
                if has_stats:
                    print(f"ℹ️  外部統計情報を読み込みました（フォールバック用）")
                    print(f"   - テーブル統計: {len(loaded_stats.get('table_stats', {}))} テーブル")
                    print(f"   - カラム統計: {len(loaded_stats.get('column_stats', {}))} カラム")
                    print(f"   注意: 統計情報は既にplan_parametersに含まれているため、外部統計は補完用途です")
            except Exception as e:
                print(f"⚠️  外部統計情報の読み込みに失敗（plan_parametersから統計情報を使用）: {e}")
    
    if not db_statistics:
        print(f"ℹ️  統計情報はplan_parametersから自動的に取得されます")
    print()
    
    # 2. トレーニング/検証セットの分割
    print("📊 ステップ2: トレーニング/検証セットの分割")
    val_size = int(len(train_plans) * args.val_ratio)
    # 検証セットが空にならないように、少なくとも1個は確保（ただし、train_plansが1個の場合は除く）
    if val_size == 0 and len(train_plans) > 1:
        val_size = 1
    train_size = len(train_plans) - val_size
    
    train_plans_split, val_plans_split = random_split(
        train_plans, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  - トレーニングプラン: {len(train_plans_split)}")
    print(f"  - 検証プラン: {len(val_plans_split)}")
    print(f"  - テストプラン: {len(test_plans)}")
    print()
    
    # 3. 特徴量統計の準備
    print("🔧 ステップ3: 特徴量統計の準備")
    plan_featurization = TrinoTrueCardDetail()
    
    # 特徴量統計情報の読み込みまたは作成
    if args.statistics_file and Path(args.statistics_file).exists():
        with open(args.statistics_file, 'r') as f:
            feature_statistics = json.load(f)
        print(f"  - 既存の統計情報を読み込み: {len(feature_statistics)} features")
    else:
        # 【重要】全プラン（train + val + test）から特徴量統計を収集
        # embeddingテーブルはモデル初期化時に固定されるため、事前にすべての演算子を収集する必要がある
        all_plans_for_stats = train_plans + test_plans
        print(f"  - 統計収集対象: {len(all_plans_for_stats)}個のプラン（train + test）")
        feature_statistics = create_feature_statistics_from_plans(
            all_plans_for_stats,
            plan_featurization,
            args.statistics_file
        )
    
    # db_statisticsがNoneの場合は空の辞書を使用（後方互換性）
    if db_statistics is None:
        db_statistics = {}
    
    print()
    
    # 4. データセットとDataLoaderの作成
    print("📦 ステップ4: データセットとDataLoaderの作成")
    
    # collate_fnを作成（バッチをグラフに変換）
    collate_fn = functools.partial(
        trino_plan_collator,
        feature_statistics=feature_statistics,
        db_statistics=db_statistics,
        plan_featurization=plan_featurization
    )
    
    # DataLoaderの作成
    train_loader = DataLoader(
        TrinoPlanDataset([train_plans[i] for i in train_plans_split.indices]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        TrinoPlanDataset([train_plans[i] for i in val_plans_split.indices]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        TrinoPlanDataset(test_plans),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"  - トレーニングバッチ数: {len(train_loader)}")
    print(f"  - 検証バッチ数: {len(val_loader)}")
    print(f"  - テストバッチ数: {len(test_loader)}")
    print()
    
    # 5. モデル作成
    print("🤖 ステップ5: モデル作成")
    model_config = ZeroShotModelConfig(
        hidden_dim=args.hidden_dim,
        hidden_dim_plan=args.hidden_dim,
        hidden_dim_pred=args.hidden_dim,
        p_dropout=0.1,
        featurization=plan_featurization,
        output_dim=1,
        batch_size=args.batch_size
    )
    
    # Trino固有の設定
    # encoders: 各ノードタイプの特徴量をエンコード
    encoders = [
        ('column', plan_featurization.COLUMN_FEATURES),
        ('table', plan_featurization.TABLE_FEATURES),
        ('output_column', plan_featurization.OUTPUT_COLUMN_FEATURES),
        ('filter_column', plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES),
        ('plan', plan_featurization.PLAN_FEATURES),
        ('logical_pred', plan_featurization.FILTER_FEATURES),
    ]
    
    # prepasses: Trino固有のメッセージパッシング（columnからoutput_columnへ）
    # allow_emptyはmessage_passing内でallow_empty_edgesから自動的に設定されるため、ここでは指定しない
    prepasses = [dict(model_name='column_output_column', e_name='col_output_col')]
    tree_model_types = ['column_output_column']
    
    # ZeroShotModelを直接使用（allow_empty_edges=TrueでTrino対応）
    model = ZeroShotModel(
        model_config=model_config,
        device=args.device,
        feature_statistics=feature_statistics,
        plan_featurization=plan_featurization,
        prepasses=prepasses,
        add_tree_model_types=tree_model_types,
        encoders=encoders,
        allow_empty_edges=True  # Trinoではエッジが存在しない場合があるため
    )
    
    model = model.to(args.device)
    
    print(f"  - モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 6. オプティマイザとスケジューラ
    print("⚙️  ステップ6: オプティマイザ設定")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    print(f"  - Optimizer: Adam (lr={args.lr})")
    print(f"  - Scheduler: ReduceLROnPlateau")
    print()
    
    # 7. トレーニングループ
    print("🚀 ステップ7: トレーニング開始")
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # トレーニング
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        
        # 検証
        val_result = validate(model, val_loader, args.device)
        val_loss, val_median_q_error, val_mean_q_error, val_rmse = val_result
        
        # 学習率スケジューラ
        scheduler.step(val_loss)
        
        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # モデルの保存
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        
        # ログ出力
        if (epoch + 1) % 5 == 0 or epoch == 0:
            median_q = f"{val_median_q_error:.4f}" if val_median_q_error is not None else "N/A"
            mean_q = f"{val_mean_q_error:.4f}" if val_mean_q_error is not None else "N/A"
            rmse_val = f"{val_rmse:.4f}" if val_rmse is not None else "N/A"
            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Median Q-Error: {median_q}, "
                f"Val Mean Q-Error: {mean_q}, "
                f"Val RMSE: {rmse_val}, "
                f"Best: {best_val_loss:.4f} (Epoch {best_epoch})"
            )
    
    print()
    print("✅ トレーニング完了!")
    print()
    
    # 8. テストセットでの評価
    print("📊 ステップ8: テストセットでの最終評価")
    
    # ベストモデルを読み込み
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_result = validate(model, test_loader, args.device)
    test_loss, test_median_q_error, test_mean_q_error, test_rmse = test_result
    
    print(f"【テストセット評価結果】")
    print(f"  - サンプル数: {len(test_plans)}")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - RMSE: {test_rmse:.4f}秒 ({test_rmse*1000:.2f}ms)")
    print(f"  - Median Q-Error: {test_median_q_error:.4f}")
    print(f"  - Mean Q-Error: {test_mean_q_error:.4f}")
    print()
    
    print("=" * 80)
    print("トレーニング完了！")
    print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Test Median Q-Error: {test_median_q_error:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print("=" * 80)
    
    return 0


def load_all_datasets_once(plans_dir: Path, available_datasets: list, max_plans_per_file=None):
    """
    Parse all datasets' .txt plans under plans_dir once.
    This is more efficient than parsing for each leave-one-out iteration.
    
    Returns: dict {dataset_name: [list of plans]}
    """
    def infer_dataset_name(p: Path, ALL_DATASETS: list) -> str:
        stem = p.stem
        parts = stem.split('_')
        matched_dataset = None
        for i in range(len(parts), 0, -1):
            candidate = '_'.join(parts[:i])
            if candidate in ALL_DATASETS:
                matched_dataset = candidate
                break
        if matched_dataset:
            return matched_dataset
        return stem.split('_')[0]
    
    ALL_DATASETS = [
        'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
        'consumer', 'credit', 'employee', 'fhnk', 'financial', 'geneea',
        'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb',
        'tournament', 'tpc_h', 'walmart'
    ]
    
    txt_files = sorted([p for p in plans_dir.glob('*.txt')])
    dataset_to_files = {}
    for p in txt_files:
        ds = infer_dataset_name(p, ALL_DATASETS)
        if ds in available_datasets:
            dataset_to_files.setdefault(ds, []).append(p)
    
    all_plans_by_dataset = {}
    print("=" * 80)
    print("ステップ0: 全データセットのプランを読み込み中...")
    print("=" * 80)
    print()
    
    for ds in available_datasets:
        if ds in dataset_to_files:
            files = dataset_to_files[ds]
            print(f"  読み込み中: {ds} ({len(files)} ファイル)...")
            plans = load_plans_from_files(files, max_plans_per_file)
            all_plans_by_dataset[ds] = plans
            print(f"    ✅ {ds}: {len(plans)} プラン")
    
    print(f"\n✅ 全データセットの読み込み完了")
    print(f"  - 読み込んだデータセット: {len(all_plans_by_dataset)}")
    for ds, plans in all_plans_by_dataset.items():
        print(f"    - {ds}: {len(plans)} プラン")
    print()
    
    return all_plans_by_dataset


def run_train_multi_all(args, output_dir: Path) -> int:
    """20個すべてのデータセットについてleave-one-out validationを実行"""
    # サポートされている20個のデータセット（アルファベット順）
    ALL_DATASETS = [
        'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
        'consumer', 'credit', 'employee', 'fhnk', 'financial', 'geneea',
        'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb',
        'tournament', 'tpc_h', 'walmart'
    ]
    
    plans_dir = Path(args.plans_dir)
    
    # 利用可能なデータセットを確認
    txt_files = sorted([p for p in plans_dir.glob('*.txt')])
    available_datasets = set()
    for p in txt_files:
        stem = p.stem  # .txtを除いたファイル名
        parts = stem.split('_')
        # 最長マッチ: ALL_DATASETSから最長の一致を探す（tpc_hなどアンダースコアを含むデータセット名に対応）
        matched_dataset = None
        for i in range(len(parts), 0, -1):
            candidate = '_'.join(parts[:i])
            if candidate in ALL_DATASETS:
                matched_dataset = candidate
                break
        if matched_dataset:
            available_datasets.add(matched_dataset)
    
    available_datasets = sorted(list(available_datasets))
    print(f"\n{'='*80}")
    print(f"Leave-One-Out Validation for All Datasets (Zero-Shot)")
    print(f"{'='*80}")
    print(f"利用可能なデータセット: {len(available_datasets)} / {len(ALL_DATASETS)}")
    print(f"データセット: {', '.join(available_datasets)}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"{'='*80}\n")
    
    # 最初に1回だけ全データセットのプランを読み込む
    all_plans_by_dataset = load_all_datasets_once(
        plans_dir=plans_dir,
        available_datasets=available_datasets,
        max_plans_per_file=args.max_plans_per_file
    )
    
    # 各データセットについて訓練・テストを実行
    results_summary = []
    plan_featurization = TrinoTrueCardDetail()
    
    for idx, test_dataset in enumerate(available_datasets, 1):
        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(available_datasets)}] Testing dataset: {test_dataset}")
        print(f"{'#'*80}\n")
        
        try:
            # 既に読み込んだプランからtrain/testを分割
            if test_dataset not in all_plans_by_dataset:
                print(f"⚠️  {test_dataset}: プランが見つかりません。スキップします。")
                results_summary.append({
                    'test_dataset': test_dataset,
                    'status': 'skipped',
                    'reason': 'missing plans'
                })
                continue
            
            train_plans = []
            test_plans = all_plans_by_dataset[test_dataset]
            
            for ds, plans in all_plans_by_dataset.items():
                if ds != test_dataset:
                    train_plans.extend(plans)
            
            if not train_plans or not test_plans:
                print(f"⚠️  {test_dataset}: 訓練プランまたはテストプランが見つかりません。スキップします。")
                results_summary.append({
                    'test_dataset': test_dataset,
                    'status': 'skipped',
                    'reason': 'missing plans'
                })
                continue
            
            # モデルディレクトリ
            model_dir = output_dir / f'models_{test_dataset}'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📊 Leave-One-Out Validation [{idx}/{len(available_datasets)}]:")
            print(f"  - Training datasets: {len(all_plans_by_dataset) - 1} datasets")
            print(f"  - Training plans: {len(train_plans)}")
            print(f"  - Test dataset: {test_dataset}")
            print(f"  - Test plans: {len(test_plans)}")
            print()
            
            # トレーニング/検証セットの分割（19個のデータセットをtrain/valに分割）
            val_size = int(len(train_plans) * args.val_ratio)
            if val_size == 0 and len(train_plans) > 1:
                val_size = 1
            train_size = len(train_plans) - val_size
            
            train_plans_split, val_plans_split = random_split(
                train_plans,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            print(f"✅ 19個のデータセットから作成:")
            print(f"  - Train plans: {len(train_plans_split)}")
            print(f"  - Val plans (from 19 datasets): {len(val_plans_split)}")
            print()
            
            # 特徴量統計の準備（全プランから）
            all_plans_for_stats = train_plans + test_plans
            statistics_file = model_dir / 'feature_statistics.json' if args.statistics_file is None else Path(args.statistics_file)
            feature_statistics = create_feature_statistics_from_plans(
                all_plans_for_stats,
                plan_featurization,
                str(statistics_file) if statistics_file != model_dir / 'feature_statistics.json' else None
            )
            
            # データベース統計情報（オプション - フォールバック用）
            # 注意: 統計情報は既にplan_parametersに含まれているため、通常は外部統計ファイルは不要
            db_statistics = {}
            if args.statistics_dir:
                try:
                    loaded_stats = load_database_statistics(
                        catalog='iceberg',
                        schema=test_dataset,
                        stats_dir=args.statistics_dir,
                        prefer_zero_shot=True
                    )
                    from types import SimpleNamespace
                    db_stats = SimpleNamespace(
                        table_stats=loaded_stats.get('table_stats', {}),
                        column_stats=loaded_stats.get('column_stats', {})
                    )
                    db_statistics[0] = db_stats
                    print(f"ℹ️  外部統計情報を読み込みました（フォールバック用）")
                except Exception as e:
                    print(f"⚠️  外部統計情報の読み込みに失敗（plan_parametersから統計情報を使用）: {e}")
            
            # データセットとDataLoaderの作成
            collate_fn = functools.partial(
                trino_plan_collator,
                feature_statistics=feature_statistics,
                db_statistics=db_statistics,
                plan_featurization=plan_featurization
            )
            
            train_loader = DataLoader(
                TrinoPlanDataset([train_plans[i] for i in train_plans_split.indices]),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                TrinoPlanDataset([train_plans[i] for i in val_plans_split.indices]),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn
            )
            
            test_loader = DataLoader(
                TrinoPlanDataset(test_plans),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn
            )
            
            # モデル作成
            model_config = ZeroShotModelConfig(
                hidden_dim=args.hidden_dim,
                hidden_dim_plan=args.hidden_dim,
                hidden_dim_pred=args.hidden_dim,
                p_dropout=0.1,
                featurization=plan_featurization,
                output_dim=1,
                batch_size=args.batch_size
            )
            
            encoders = [
                ('column', plan_featurization.COLUMN_FEATURES),
                ('table', plan_featurization.TABLE_FEATURES),
                ('output_column', plan_featurization.OUTPUT_COLUMN_FEATURES),
                ('filter_column', plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES),
                ('plan', plan_featurization.PLAN_FEATURES),
                ('logical_pred', plan_featurization.FILTER_FEATURES),
            ]
            prepasses = [dict(model_name='column_output_column', e_name='col_output_col')]
            tree_model_types = ['column_output_column']
            
            model = ZeroShotModel(
                model_config=model_config,
                device=args.device,
                feature_statistics=feature_statistics,
                plan_featurization=plan_featurization,
                prepasses=prepasses,
                add_tree_model_types=tree_model_types,
                encoders=encoders,
                allow_empty_edges=True
            )
            model = model.to(args.device)
            
            # オプティマイザとスケジューラ
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=False
            )
            
            # トレーニングループ
            best_val_loss = float('inf')
            best_epoch = 0
            
            for epoch in range(args.epochs):
                train_loss = train_epoch(model, train_loader, optimizer, args.device)
                val_result = validate(model, val_loader, args.device)
                val_loss, val_median_q_error, val_mean_q_error, val_rmse = val_result
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), model_dir / 'best_model.pt')
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    median_q = f"{val_median_q_error:.4f}" if val_median_q_error is not None else "N/A"
                    mean_q = f"{val_mean_q_error:.4f}" if val_mean_q_error is not None else "N/A"
                    print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Median Q-Error: {median_q}, Val Mean Q-Error: {mean_q}")
            
            # テストセットでの評価
            model.load_state_dict(torch.load(model_dir / 'best_model.pt'))
            test_result = validate(model, test_loader, args.device)
            test_loss, test_median_q_error, test_mean_q_error, test_rmse = test_result
            
            # テスト結果を保存
            test_results = {
                'test_loss': float(test_loss),
                'test_median_q_error': float(test_median_q_error) if test_median_q_error is not None else None,
                'test_mean_q_error': float(test_mean_q_error) if test_mean_q_error is not None else None,
                'test_rmse': float(test_rmse) if test_rmse is not None else None,
                'test_samples': len(test_plans)
            }
            
            results_file = model_dir / 'test_results.json'
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            results_summary.append({
                'test_dataset': test_dataset,
                'model_dir': str(model_dir),
                'best_val_loss': float(best_val_loss),
                'best_epoch': int(best_epoch),
                **test_results,
                'status': 'completed'
            })
            
            print(f"✅ [{idx}/{len(available_datasets)}] {test_dataset} の訓練・テスト完了")
            print(f"   モデル保存先: {model_dir}")
            print()
            
        except Exception as e:
            print(f"❌ [{idx}/{len(available_datasets)}] {test_dataset} でエラーが発生:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'test_dataset': test_dataset,
                'status': 'failed',
                'error': str(e)
            })
            continue
    
    # 全体のサマリーを保存
    summary_file = output_dir / 'leave_one_out_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'total_datasets': len(available_datasets),
            'completed': len([r for r in results_summary if r['status'] == 'completed']),
            'failed': len([r for r in results_summary if r['status'] == 'failed']),
            'skipped': len([r for r in results_summary if r.get('status') == 'skipped']),
            'results': results_summary
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("🎉 全データセットでのLeave-One-Out Validation完了！")
    print("=" * 80)
    print(f"完了: {len([r for r in results_summary if r['status'] == 'completed'])}/{len(available_datasets)}")
    print(f"失敗: {len([r for r in results_summary if r['status'] == 'failed'])}/{len(available_datasets)}")
    print(f"サマリーファイル: {summary_file}")
    print()
    
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for Zero-Shot training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    import sys
    from typing import Optional, Sequence
    sys.exit(main())

