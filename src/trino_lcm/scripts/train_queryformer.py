#!/usr/bin/env python3
"""
Trino QueryFormer Training Script

Usage:
    python src/train_entry.py query-former \\
        --mode train \\
        --model_type query_former \\
        --dataset accidents \\
        --txt_file ../explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt \\
        --output_dir data/runs/trino/accidents \\
        --device cuda \\
        --epochs 100 \\
        --batch_size 32
"""

import warnings

# Suppress torchdata deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchdata')

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.optim as opt
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

# 環境変数を設定（classes.pyのエラーを回避）
for i in range(20):
    os.environ.setdefault(f'NODE{i:02d}', '{}')

from cross_db_benchmark.benchmark_tools.trino.parse_plan import TrinoPlanParser
from cross_db_benchmark.benchmark_tools.utils import load_schema_json, load_json
from models.workload_driven.preprocessing.sample_vectors_trino import get_table_samples_from_csv, augment_sample
from models.query_former.model import QueryFormer
from models.workload_driven.dataset.dataset_creation import PlanModelInputDims
from training.preprocessing.feature_statistics import gather_feature_statistics
from training.training.metrics import RMSE, MAPE, QError, Metric
from training.training.checkpoint import save_checkpoint, load_checkpoint
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# 共通ユーティリティ関数
# ============================================================================

def plan_to_dict(node):
    """プランノードを辞書形式に変換（SimpleNamespaceも完全に辞書に変換）"""
    from types import SimpleNamespace
    
    def convert_to_dict(obj):
        """SimpleNamespaceやその他のオブジェクトを辞書に変換"""
        if isinstance(obj, SimpleNamespace):
            result = {}
            for k, v in vars(obj).items():
                result[k] = convert_to_dict(v)
            return result
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_dict(item) for item in obj]
        else:
            return obj
    
    result = {}
    if hasattr(node, 'plan_parameters'):
        if hasattr(node.plan_parameters, '__dict__'):
            result = convert_to_dict(node.plan_parameters)
        elif isinstance(node.plan_parameters, dict):
            result = convert_to_dict(node.plan_parameters)
    
    # 'table'を'tablename'としても追加
    if 'table' in result and 'tablename' not in result:
        result['tablename'] = result['table']
    
    if hasattr(node, 'children') and node.children:
        result['children'] = [plan_to_dict(c) for c in node.children]
    
    return result


def dict_to_namespace_recursive(d):
    """辞書を再帰的にSimpleNamespaceに変換"""
    if isinstance(d, dict):
        ns = SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(ns, k, dict_to_namespace_recursive(v))
            elif isinstance(v, list):
                setattr(ns, k, [dict_to_namespace_recursive(item) if isinstance(item, dict) else item for item in v])
            else:
                setattr(ns, k, v)
        return ns
    return d


def convert_filter_node(filter_node, column_id_mapping, partial_column_name_mapping):
    """filter_columns（SimpleNamespace形式）のcolumnとliteralを変換"""
    if not hasattr(filter_node, 'column'):
        return
    
    # columnの変換（Noneは維持）
    column = filter_node.column
    if column is not None and isinstance(column, (tuple, list)):
        if len(column) == 1:
            col_name = str(column[0]).strip('"')
            # Trinoエイリアス（_数字）を削除
            import re
            col_name_without_suffix = re.sub(r'_\d+$', '', col_name)
            
            # テーブル名を推測してカラムIDを取得
            mapped = None
            for table_name in partial_column_name_mapping.get(col_name, set()):
                if (table_name, col_name) in column_id_mapping:
                    mapped = column_id_mapping[(table_name, col_name)]
                    break
            
            if mapped is None:
                # サフィックス削除版で再試行
                for table_name in partial_column_name_mapping.get(col_name_without_suffix, set()):
                    if (table_name, col_name_without_suffix) in column_id_mapping:
                        mapped = column_id_mapping[(table_name, col_name_without_suffix)]
                        break
            
            filter_node.column = mapped
        elif len(column) == 2:
            table_name, col_name = column
            table_name = str(table_name).strip('"')
            col_name = str(col_name).strip('"')
            filter_node.column = column_id_mapping.get((table_name, col_name))
    
    # literalの変換（リスト形式とシンプル形式の両方に対応）
    literal = filter_node.literal
    if literal is not None:
        if isinstance(literal, list):
            # IN句のリスト形式
            converted_list = []
            for item in literal:
                if isinstance(item, str) and "'" in item:
                    parts = item.split("'")
                    if len(parts) >= 2:
                        item_value = parts[1]
                        try:
                            if '.' in item_value:
                                converted_list.append(float(item_value))
                            else:
                                converted_list.append(int(item_value))
                        except (ValueError, TypeError):
                            converted_list.append(item_value)
                    else:
                        converted_list.append(item)
                else:
                    converted_list.append(item)
            filter_node.literal = converted_list
        elif isinstance(literal, str) and "'" in literal:
            # varchar 'B' -> 'B', bigint '343' -> 343
            parts = literal.split("'")
            if len(parts) >= 2:
                literal_value = parts[1]
                try:
                    if '.' in literal_value:
                        filter_node.literal = float(literal_value)
                    else:
                        filter_node.literal = int(literal_value)
                except (ValueError, TypeError):
                    filter_node.literal = literal_value
    
    # 子ノードを再帰的に変換
    if hasattr(filter_node, 'children') and filter_node.children:
        for child in filter_node.children:
            convert_filter_node(child, column_id_mapping, partial_column_name_mapping)


def convert_to_namespace(node, column_id_mapping, partial_column_name_mapping):
    """辞書形式のplan_parametersとfilter_columnsをSimpleNamespaceに変換"""
    if hasattr(node, 'plan_parameters'):
        if isinstance(node.plan_parameters, dict):
            # 辞書をSimpleNamespaceに変換
            node.plan_parameters = dict_to_namespace_recursive(node.plan_parameters)
        
        # filter_columnsのcolumnとliteralを変換
        if hasattr(node.plan_parameters, 'filter_columns') and node.plan_parameters.filter_columns:
            # filter_columnsが辞書の場合は先にSimpleNamespaceに変換
            if isinstance(node.plan_parameters.filter_columns, dict):
                node.plan_parameters.filter_columns = dict_to_namespace_recursive(node.plan_parameters.filter_columns)
            # その後、columnとliteralを変換
            convert_filter_node(node.plan_parameters.filter_columns, column_id_mapping, partial_column_name_mapping)
    
    # 子ノードを再帰的に変換
    if hasattr(node, 'children') and node.children:
        for child in node.children:
            convert_to_namespace(child, column_id_mapping, partial_column_name_mapping)


def build_column_id_mapping(database_statistics):
    """カラムIDマッピングを作成"""
    column_id_mapping = {}
    partial_column_name_mapping = {}
    
    for i, col_stat in enumerate(database_statistics.column_stats):
        key = (col_stat.tablename, col_stat.attname)
        column_id_mapping[key] = i
        
        if col_stat.attname not in partial_column_name_mapping:
            partial_column_name_mapping[col_stat.attname] = set()
        partial_column_name_mapping[col_stat.attname].add(col_stat.tablename)
    
    return column_id_mapping, partial_column_name_mapping


def build_feature_statistics(train_plans_dict, train_plans):
    """訓練プランからfeature_statisticsを生成"""
    from training.preprocessing.feature_statistics import gather_values_recursively
    from sklearn.preprocessing import RobustScaler
    
    values = gather_values_recursively(train_plans_dict)
    feature_statistics = {}
    
    for k, vals in values.items():
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        
        if all([isinstance(v, (int, float)) or v is None for v in vals]):
            scaler = RobustScaler()
            np_vals = np.array(vals, dtype=np.float64).reshape(-1, 1)
            np_vals = np_vals[np.isfinite(np_vals)].reshape(-1, 1)
            if np_vals.size == 0:
                feature_statistics[k] = dict(max=0.0, scale=1.0, center=0.0, type='numeric')
            else:
                scaler.fit(np_vals)
                feature_statistics[k] = dict(
                    max=float(np_vals.max()),
                    scale=float(scaler.scale_.item()),
                    center=float(scaler.center_.item()),
                    type='numeric'
                )
        else:
            unique_values = set(vals)
            feature_statistics[k] = dict(
                value_dict={v: id for id, v in enumerate(unique_values)},
                no_vals=len(unique_values),
                type='categorical'
            )
    
    # feature_statisticsの互換性チェック
    if 'column' in feature_statistics and 'max' not in feature_statistics['column']:
        if 'no_vals' in feature_statistics['column']:
            feature_statistics['column']['max'] = feature_statistics['column']['no_vals'] - 1
        else:
            feature_statistics['column']['max'] = 0
    if 'columns' not in feature_statistics:
        feature_statistics['columns'] = feature_statistics.get('column', {'max': 0})
    # columnsにもmaxキーを確実に設定
    if 'columns' in feature_statistics and 'max' not in feature_statistics['columns']:
        if 'column' in feature_statistics and 'max' in feature_statistics['column']:
            feature_statistics['columns']['max'] = feature_statistics['column']['max']
        elif 'columns' in feature_statistics and 'no_vals' in feature_statistics['columns']:
            feature_statistics['columns']['max'] = feature_statistics['columns']['no_vals'] - 1
        else:
            feature_statistics['columns']['max'] = 0
    if 'tablename' not in feature_statistics:
        feature_statistics['tablename'] = {'value_dict': {}, 'no_vals': 0, 'type': 'categorical'}
    
    # 'table'キーを確実に正しい形式で作成
    if 'tablename' in feature_statistics and 'no_vals' in feature_statistics['tablename']:
        feature_statistics['table'] = {'max': max(0, feature_statistics['tablename']['no_vals'] - 1)}
    elif 'table' not in feature_statistics:
        feature_statistics['table'] = {'max': 0}
    elif 'table' in feature_statistics and feature_statistics['table'].get('type') == 'numeric':
        if 'tablename' in feature_statistics and 'no_vals' in feature_statistics['tablename']:
            feature_statistics['table'] = {'max': max(0, feature_statistics['tablename']['no_vals'] - 1)}
        else:
            feature_statistics['table'] = {'max': 0}
    
    # join conds from training
    join_conds_set = set()
    for p in train_plans:
        if hasattr(p, 'join_conds') and p.join_conds:
            join_conds_set.update(p.join_conds)
    feature_statistics['join_conds'] = {
        'value_dict': {jc: idx for idx, jc in enumerate(join_conds_set)} if join_conds_set else {},
        'no_vals': len(join_conds_set),
        'type': 'categorical'
    }
    
    return feature_statistics


def create_test_loader(
    test_plans,
    feature_statistics,
    column_statistics,
    database_statistics,
    label_norm,
    batch_size,
    max_num_filters,
    histogram_bin_size
):
    """テストローダーを作成"""
    from models.query_former.dataloader import encode_query_plan
    from torch.utils.data import TensorDataset
    
    all_features = []
    all_join_ids = []
    all_attention_bias = []
    all_rel_pos = []
    all_node_heights = []
    all_labels = []
    
    # カラムIDマッピングを作成
    column_id_mapping, partial_column_name_mapping = build_column_id_mapping(database_statistics)
    
    # plan_parametersをSimpleNamespaceに変換
    for tp in test_plans:
        convert_to_namespace(tp, column_id_mapping, partial_column_name_mapping)
    
    # エンコード
    for i, plan in enumerate(test_plans):
        try:
            if not hasattr(plan, 'join_conds'):
                plan.join_conds = []
            if not hasattr(plan, 'database_id'):
                plan.database_id = 0
            
            features, join_ids, attention_bias, rel_pos, node_heights = encode_query_plan(
                query_index=i,
                query_plan=plan,
                feature_statistics=feature_statistics,
                column_statistics=column_statistics,
                database_statistics=database_statistics,
                word_embeddings=None,
                dim_word_embedding=100,
                dim_word_hash=100,
                dim_bitmaps=1000,
                max_filter_number=max_num_filters,
                histogram_bin_size=histogram_bin_size
            )
            all_features.append(features)
            all_join_ids.append(join_ids)
            all_attention_bias.append(attention_bias)
            all_rel_pos.append(rel_pos)
            all_node_heights.append(node_heights)
            label = plan.plan_runtime / 1000 if hasattr(plan, 'plan_runtime') else 1.0
            all_labels.append(label)
        except Exception as e:
            print(f"  警告: テストプラン{i+1}のエンコードに失敗: {e}")
            continue
    
    if not all_features:
        raise ValueError("No test features encoded. Aborting.")
    
    features_tensor = torch.cat(all_features)
    join_ids_tensor = torch.cat(all_join_ids)
    attention_bias_tensor = torch.cat(all_attention_bias)
    rel_pos_tensor = torch.cat(all_rel_pos)
    node_heights_tensor = torch.cat(all_node_heights)
    
    # Test labelsを正規化
    test_labels_array = np.array(all_labels).reshape(-1, 1)
    if label_norm is not None:
        normalized_test_labels = torch.tensor(
            label_norm.transform(np.log1p(test_labels_array)).flatten(),
            dtype=torch.float32
        )
    else:
        normalized_test_labels = torch.tensor(all_labels, dtype=torch.float32)
    
    test_dataset_tensor = TensorDataset(
        features_tensor,
        join_ids_tensor,
        attention_bias_tensor,
        rel_pos_tensor,
        node_heights_tensor,
        normalized_test_labels.view(-1, 1)
    )
    test_loader = DataLoader(test_dataset_tensor, batch_size=batch_size, shuffle=False)
    
    return test_loader


# ============================================================================
# メイン関数
# ============================================================================

def parse_and_prepare_data(
    txt_file: Path,
    dataset: str,
    output_dir: Path,
    no_samples: int = 1000
) -> tuple:
    """
    .txtファイルをパースし、sample_vecを追加し、feature_statisticsを生成
    
    Returns:
        (parsed_plans, feature_statistics, column_statistics, database_statistics)
    """
    print("=" * 80)
    print("ステップ1: データの準備")
    print("=" * 80)
    print()
    
    # ワーキングディレクトリをsrcに設定（load_schema_jsonが相対パスを使用するため）
    original_cwd = os.getcwd()
    src_dir = Path(__file__).parent
    os.chdir(src_dir)
    
    parser = TrinoPlanParser()
    
    # 統計情報を先に読み込む（カラムIDマッピングを作成するため）
    print(f"統計情報を読み込み中（{dataset}）...")
    
    # column_statisticsを読み込む（Trino統計を優先）
    from cross_db_benchmark.benchmark_tools.utils import load_column_statistics
    prefer_trino = os.getenv('PREFER_TRINO_STATS', 'true').lower() in ('true', '1', 'yes')
    print(f"  PREFER_TRINO_STATS: {prefer_trino}")
    
    zero_shot_column_stats = load_column_statistics(dataset, namespace=False, prefer_trino=prefer_trino)
    
    # テーブルサンプルとカラム統計を準備
    print(f"テーブルサンプルを取得中（{dataset}）...")
    table_samples = get_table_samples_from_csv(dataset, data_dir=None, no_samples=no_samples)
    
    schema = load_schema_json(dataset)
    # col_statsはsample_vec生成用（SimpleNamespace形式）
    # zero-shotのcolumn_statisticsから作成
    col_stats = []
    for table_name, table_cols in zero_shot_column_stats.items():
        for col_name in table_cols.keys():
            col_stats.append(SimpleNamespace(
                tablename=table_name,
                attname=col_name,
                attnum=len(col_stats)
            ))
    
    print(f"✅ テーブルサンプル: {len(table_samples)} テーブル")
    print(f"✅ カラム統計: {len(col_stats)} カラム")
    print()
    
    # table_stats.jsonはdatasets_statisticsから読み込む
    # Docker環境とローカル環境の両方に対応
    stats_base = os.getenv('DATASETS_STATISTICS_DIR', Path(__file__).parent.parent.parent.parent / 'datasets_statistics')
    stats_dir = Path(stats_base) / f'iceberg_{dataset}'
    if not stats_dir.exists():
        raise FileNotFoundError(f"統計情報が見つかりません: {stats_dir}")
    
    with open(stats_dir / 'table_stats.json') as f:
        table_stats_dict = json.load(f)
    
    # zero-shot形式のcolumn_statisticsをPostgres形式に変換し、カラムIDマッピングを作成
    column_id_mapping = {}
    partial_column_name_mapping = {}
    table_id_mapping = {}
    
    column_stats_list = []
    # zero-shot形式: {table_name: {column_name: {stats...}}}
    # これを順序付きリストに変換し、IDマッピングを作成
    for table_name, table_cols in zero_shot_column_stats.items():
        for col_name, col_stats in table_cols.items():
            idx = len(column_stats_list)
            column_id_mapping[(table_name, col_name)] = idx
            
            if col_name not in partial_column_name_mapping:
                partial_column_name_mapping[col_name] = set()
            partial_column_name_mapping[col_name].add(table_name)
            
            # SimpleNamespace形式のcolumn_statsを作成
            column_stats_list.append(SimpleNamespace(
                tablename=table_name,
                attname=col_name,
                attnum=idx,
                null_frac=col_stats.get('nan_ratio', 0.0),
                avg_width=0,  # zero-shot形式にはない
                n_distinct=col_stats.get('num_unique', -1),
                correlation=0  # zero-shot形式にはない
            ))
    
    for i, (table_name, stats) in enumerate(table_stats_dict.items()):
        table_id_mapping[table_name] = i
    
    print(f"✅ カラムIDマッピング: {len(column_id_mapping)} カラム")
    print(f"✅ テーブルIDマッピング: {len(table_id_mapping)} テーブル")
    print()
    
    # プランをパース（sample_vec付き、カラムID変換付き）
    print(f"プランをパース中: {txt_file.name}")
    parsed_plans, runtimes = parser.parse_explain_analyze_file(
        str(txt_file),
        min_runtime=0,
        max_runtime=float('inf'),
        table_samples=table_samples,
        col_stats=col_stats,
        column_id_mapping=column_id_mapping,
        partial_column_name_mapping=partial_column_name_mapping,
        table_id_mapping=table_id_mapping
    )
    
    print(f"✅ {len(parsed_plans)} プランをパース完了（カラムID変換済み）")
    
    # sample_vec統計を確認
    total_sample_vec_ones = 0
    total_nodes_with_sample_vec = 0
    
    def count_sample_vec(node):
        nonlocal total_sample_vec_ones, total_nodes_with_sample_vec
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters if isinstance(node.plan_parameters, dict) else vars(node.plan_parameters)
            sample_vec = params.get('sample_vec')
            if sample_vec:
                total_nodes_with_sample_vec += 1
                total_sample_vec_ones += sum(sample_vec) if isinstance(sample_vec, (list, tuple)) else 0
        
        if hasattr(node, 'children'):
            for child in node.children:
                count_sample_vec(child)
    
    for plan in parsed_plans:
        count_sample_vec(plan)
    
    print(f"✅ sample_vec: {total_nodes_with_sample_vec} ノード, 1の総数={total_sample_vec_ones}")
    print()
    
    # feature_statisticsを生成
    print("feature_statisticsを生成中...")
    
    plans_dict = [plan_to_dict(p) for p in parsed_plans]
    
    feature_statistics = build_feature_statistics(plans_dict, parsed_plans)
    
    print(f"✅ feature_statistics: {len(feature_statistics)} 特徴量")
    
    # feature_statisticsを保存
    feature_stats_file = output_dir / 'feature_statistics.json'
    feature_stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_stats_file, 'w') as f:
        json.dump(feature_statistics, f, indent=2)
    
    print(f"✅ feature_statisticsを保存: {feature_stats_file}")
    print()
    
    # zero-shot形式のcolumn_statisticsをPostgres形式に変換
    column_statistics = {}
    for table_name, table_cols in zero_shot_column_stats.items():
        column_statistics[table_name] = {}
        for col_name, col_stats in table_cols.items():
            # zero-shot形式からPostgres形式に変換
            converted_stats = {
                'datatype': col_stats.get('datatype', 'misc'),
                'min': col_stats.get('min'),
                'max': col_stats.get('max'),
                'percentiles': col_stats.get('percentiles'),
                'num_unique': col_stats.get('num_unique', -1),
                'null_frac': col_stats.get('nan_ratio', 0.0)
            }
            column_statistics[table_name][col_name] = converted_stats
    
    print(f"✅ column_statistics: {len(column_statistics)} テーブル")
    
    # column_statisticsを保存
    column_stats_file = output_dir / 'column_statistics.json'
    with open(column_stats_file, 'w') as f:
        json.dump(column_statistics, f, indent=2)
    
    print(f"✅ column_statisticsを保存: {column_stats_file}")
    print()
    
    # database_statisticsを作成（既に読み込み済みのtable_stats_dictを使用）
    table_stats_list = []
    for table_name, stats in table_stats_dict.items():
        table_stats_list.append(SimpleNamespace(
            relname=table_name,
            reltuples=stats.get('reltuples', stats.get('row_count', 0)),
            relpages=stats.get('relpages', 0)
        ))
    
    # column_stats_listは既に作成済み（上記の変換処理で）
    # ここでは確認のみ
    
    database_statistics = SimpleNamespace(
        table_stats=table_stats_list,
        column_stats=column_stats_list,
        database_type='trino'
    )
    
    print(f"✅ database_statistics作成完了")
    print()
    
    # ワーキングディレクトリを元に戻す
    os.chdir(original_cwd)
    
    return parsed_plans, feature_statistics, column_statistics, database_statistics


def parse_all_datasets_once(
    plans_dir: Path,
    available_datasets: list,
    no_samples: int = 1000
):
    """
    Parse all datasets' .txt plans under plans_dir once.
    This is more efficient than parsing for each leave-one-out iteration.

    Returns: (all_plans_by_dataset, column_statistics, database_statistics)
      - all_plans_by_dataset: dict {dataset_name: [list of plans]}
      - column_statistics: combined column statistics dict
      - database_statistics: SimpleNamespace with table_stats and column_stats
    """
    from cross_db_benchmark.benchmark_tools.trino.parse_plan import TrinoPlanParser
    from models.workload_driven.preprocessing.sample_vectors_trino import get_table_samples_from_csv
    
    assert plans_dir.exists(), f"plans_dir not found: {plans_dir}"
    txt_files = sorted([p for p in plans_dir.glob('*.txt')])
    assert txt_files, f"No txt files found in {plans_dir}"

    def infer_dataset_name(p: Path) -> str:
        ALL_DATASETS = [
            'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
            'consumer', 'credit', 'employee', 'fhnk', 'financial', 'geneea',
            'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb',
            'tournament', 'tpc_h', 'walmart'
        ]
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

    dataset_to_files = {}
    for p in txt_files:
        ds = infer_dataset_name(p)
        if ds in available_datasets:
            dataset_to_files.setdefault(ds, []).append(p)

    parser = TrinoPlanParser()

    # Global column stats and mappings
    global_column_stats_list = []
    global_column_id = {}
    combined_column_statistics = {}
    combined_table_stats = []
    
    def load_stats_and_prepare_ids(dataset: str):
        from cross_db_benchmark.benchmark_tools.utils import load_column_statistics
        prefer_trino = os.getenv('PREFER_TRINO_STATS', 'true').lower() in ('true', '1', 'yes')
        
        try:
            zero_shot_column_stats = load_column_statistics(dataset, namespace=False, prefer_trino=prefer_trino)
            
            # table_stats.jsonはdatasets_statisticsから読み込む
            stats_dir = Path('src').parent / 'datasets_statistics' / f'iceberg_{dataset}'
            if stats_dir.exists():
                with open(stats_dir / 'table_stats.json') as f:
                    table_stats_dict = json.load(f)
            else:
                table_stats_dict = {}
        except FileNotFoundError:
            zero_shot_column_stats = {}
            table_stats_dict = {}
        
        for table_name, tstats in table_stats_dict.items():
            combined_table_stats.append(SimpleNamespace(
                relname=f"{dataset}.{table_name}",
                reltuples=tstats.get('reltuples', tstats.get('row_count', 0)),
                relpages=tstats.get('relpages', 0)
            ))
        
        column_id_mapping_ds = {}
        partial_column_name_mapping_ds = {}
        for table_name, table_cols in zero_shot_column_stats.items():
            for col_name, col_stats in table_cols.items():
                key = (dataset, table_name, col_name)
                if key not in global_column_id:
                    gid = len(global_column_stats_list)
                    global_column_id[key] = gid
                    global_column_stats_list.append(SimpleNamespace(
                        tablename=f"{dataset}.{table_name}",
                        attname=col_name,
                        attnum=gid,
                        null_frac=col_stats.get('nan_ratio', 0.0),
                        avg_width=0,
                        n_distinct=col_stats.get('num_unique', -1),
                        correlation=0
                    ))
                column_id_mapping_ds[(table_name, col_name)] = global_column_id[key]
                partial_column_name_mapping_ds.setdefault(col_name, set()).add(table_name)
        
        for table_name, table_cols in zero_shot_column_stats.items():
            pref_t = f"{dataset}.{table_name}"
            for col_name, col_stats in table_cols.items():
                combined_column_statistics.setdefault(pref_t, {})[col_name] = {
                    'datatype': col_stats.get('datatype', 'misc'),
                    'min': col_stats.get('min'),
                    'max': col_stats.get('max'),
                    'percentiles': col_stats.get('percentiles'),
                    'num_unique': col_stats.get('num_unique', -1),
                    'null_frac': col_stats.get('nan_ratio', 0.0)
                }
        
        return {}, table_stats_dict, column_id_mapping_ds, partial_column_name_mapping_ds
    
    def parse_plans_for_dataset(dataset: str, files: list):
        column_stats_dict, table_stats_dict, column_id_mapping_ds, partial_col_map_ds = load_stats_and_prepare_ids(dataset)
        
        from cross_db_benchmark.benchmark_tools.utils import load_column_statistics
        prefer_trino = os.getenv('PREFER_TRINO_STATS', 'true').lower() in ('true', '1', 'yes')
        try:
            zero_shot_column_stats = load_column_statistics(dataset, namespace=False, prefer_trino=prefer_trino)
        except FileNotFoundError:
            zero_shot_column_stats = {}
        
        table_samples = None
        col_stats_ns = []
        try:
            table_samples = get_table_samples_from_csv(dataset, data_dir=None, no_samples=no_samples)
            for table_name, table_cols in zero_shot_column_stats.items():
                for col_name in table_cols.keys():
                    col_stats_ns.append(SimpleNamespace(
                        tablename=table_name,
                        attname=col_name,
                        attnum=len(col_stats_ns)
                    ))
        except Exception:
            table_samples = None
            col_stats_ns = []
        
        parsed = []
        for f in files:
            p_plans, runtimes = parser.parse_explain_analyze_file(
                str(f),
                min_runtime=0,
                max_runtime=float('inf'),
                table_samples=table_samples,
                col_stats=col_stats_ns,
                column_id_mapping=column_id_mapping_ds,
                partial_column_name_mapping=partial_col_map_ds,
                table_id_mapping={}
            )
            for p in p_plans:
                setattr(p, 'database_id', 0)
                setattr(p, 'dataset', dataset)
            parsed.extend(p_plans)
        return parsed
    
    all_plans_by_dataset = {}
    for ds in available_datasets:
        if ds in dataset_to_files:
            files = dataset_to_files[ds]
            print(f"  読み込み中: {ds} ({len(files)} ファイル)...")
            all_plans_by_dataset[ds] = parse_plans_for_dataset(ds, files)
            print(f"    ✅ {ds}: {len(all_plans_by_dataset[ds])} プラン")
    
    database_statistics = SimpleNamespace(
        table_stats=combined_table_stats,
        column_stats=global_column_stats_list,
        database_type='trino'
    )
    
    return all_plans_by_dataset, combined_column_statistics, database_statistics


def parse_and_prepare_leave_one_out(
    plans_dir: Path,
    test_dataset: str,
    no_samples: int = 1000
):
    """
    Parse multiple datasets' .txt plans under plans_dir.
    Use 19 datasets for training and 1 held-out dataset for testing.

    Returns: (train_plans, test_plans, feature_statistics, column_statistics, database_statistics)
    """
    from cross_db_benchmark.benchmark_tools.trino.parse_plan import TrinoPlanParser
    from cross_db_benchmark.benchmark_tools.utils import load_schema_json
    from models.workload_driven.preprocessing.sample_vectors_trino import get_table_samples_from_csv
    
    assert plans_dir.exists(), f"plans_dir not found: {plans_dir}"
    txt_files = sorted([p for p in plans_dir.glob('*.txt')])
    assert txt_files, f"No txt files found in {plans_dir}"

    def infer_dataset_name(p: Path) -> str:
        # 最長マッチ: ALL_DATASETSから最長の一致を探す（tpc_hなどアンダースコアを含むデータセット名に対応）
        ALL_DATASETS = [
            'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
            'consumer', 'credit', 'employee', 'fhnk', 'financial', 'geneea',
            'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb',
            'tournament', 'tpc_h', 'walmart'
        ]
        stem = p.stem  # .txtを除いたファイル名
        parts = stem.split('_')
        matched_dataset = None
        for i in range(len(parts), 0, -1):
            candidate = '_'.join(parts[:i])
            if candidate in ALL_DATASETS:
                matched_dataset = candidate
                break
        if matched_dataset:
            return matched_dataset
        # フォールバック: 最初の部分を返す
        return stem.split('_')[0]

    dataset_to_files = {}
    for p in txt_files:
        ds = infer_dataset_name(p)
        dataset_to_files.setdefault(ds, []).append(p)

    assert test_dataset in dataset_to_files, f"test_dataset '{test_dataset}' not found among: {list(dataset_to_files.keys())}"

    train_datasets = [ds for ds in dataset_to_files.keys() if ds != test_dataset]
    test_files = dataset_to_files[test_dataset]

    parser = TrinoPlanParser()

    # Global column stats and mappings
    global_column_stats_list = []  # SimpleNamespace(tablename=ds.table, attname=col, attnum=global_id,...)
    global_column_id = {}  # (dataset, table, column) -> id
    
    combined_column_statistics = {}  # {ds.table: {col: stats}}
    combined_table_stats = []
    
    # Helper to load stats and build global column ids for a dataset
    def load_stats_and_prepare_ids(dataset: str):
        # column_statisticsを読み込む（Trino統計を優先）
        from cross_db_benchmark.benchmark_tools.utils import load_column_statistics
        prefer_trino = os.getenv('PREFER_TRINO_STATS', 'true').lower() in ('true', '1', 'yes')
        
        try:
            zero_shot_column_stats = load_column_statistics(dataset, namespace=False, prefer_trino=prefer_trino)
            
            # table_stats.jsonはdatasets_statisticsから読み込む
            stats_dir = Path('src').parent / 'datasets_statistics' / f'iceberg_{dataset}'
            if stats_dir.exists():
                with open(stats_dir / 'table_stats.json') as f:
                    table_stats_dict = json.load(f)
            else:
                table_stats_dict = {}
        except FileNotFoundError:
            zero_shot_column_stats = {}
            table_stats_dict = {}
        
        # Build table stats entries (with dataset prefix)
        for table_name, tstats in table_stats_dict.items():
            combined_table_stats.append(SimpleNamespace(
                relname=f"{dataset}.{table_name}",
                reltuples=tstats.get('reltuples', tstats.get('row_count', 0)),
                relpages=tstats.get('relpages', 0)
            ))
        
        # Per-dataset column_id_mapping to global ids
        # zero-shot形式: {table_name: {column_name: {stats...}}}
        column_id_mapping_ds = {}
        partial_column_name_mapping_ds = {}
        for table_name, table_cols in zero_shot_column_stats.items():
            for col_name, col_stats in table_cols.items():
                key = (dataset, table_name, col_name)
                if key not in global_column_id:
                    gid = len(global_column_stats_list)
                    global_column_id[key] = gid
                    global_column_stats_list.append(SimpleNamespace(
                        tablename=f"{dataset}.{table_name}",
                        attname=col_name,
                        attnum=gid,
                        null_frac=col_stats.get('nan_ratio', 0.0),
                        avg_width=0,  # zero-shot形式にはない
                        n_distinct=col_stats.get('num_unique', -1),
                        correlation=0  # zero-shot形式にはない
                    ))
                # For parsing within this dataset
                column_id_mapping_ds[(table_name, col_name)] = global_column_id[key]
                partial_column_name_mapping_ds.setdefault(col_name, set()).add(table_name)
        
        # Build combined column_statistics with prefixed table (zero-shot形式から変換)
        for table_name, table_cols in zero_shot_column_stats.items():
            pref_t = f"{dataset}.{table_name}"
            for col_name, col_stats in table_cols.items():
                combined_column_statistics.setdefault(pref_t, {})[col_name] = {
                    'datatype': col_stats.get('datatype', 'misc'),
                    'min': col_stats.get('min'),
                    'max': col_stats.get('max'),
                    'percentiles': col_stats.get('percentiles'),
                    'num_unique': col_stats.get('num_unique', -1),
                    'null_frac': col_stats.get('nan_ratio', 0.0)
                }
        
        # 戻り値は互換性のためダミーデータを返す（実際には使用されない）
        return {}, table_stats_dict, column_id_mapping_ds, partial_column_name_mapping_ds
    
    # Parse plans for a dataset list
    def parse_plans_for_dataset(dataset: str, files: list):
        column_stats_dict, table_stats_dict, column_id_mapping_ds, partial_col_map_ds = load_stats_and_prepare_ids(dataset)
        
        # column_statisticsを読み込んでcol_stats_nsを作成
        from cross_db_benchmark.benchmark_tools.utils import load_column_statistics
        prefer_trino = os.getenv('PREFER_TRINO_STATS', 'true').lower() in ('true', '1', 'yes')
        try:
            zero_shot_column_stats = load_column_statistics(dataset, namespace=False, prefer_trino=prefer_trino)
        except FileNotFoundError:
            zero_shot_column_stats = {}
        
        # Prepare table samples and simple col_stats namespaces for sample_vec generation (optional)
        table_samples = None
        col_stats_ns = []
        try:
            table_samples = get_table_samples_from_csv(dataset, data_dir=None, no_samples=no_samples)
            # zero-shotのcolumn_statisticsからcol_stats_nsを作成
            for table_name, table_cols in zero_shot_column_stats.items():
                for col_name in table_cols.keys():
                    col_stats_ns.append(SimpleNamespace(
                        tablename=table_name,
                        attname=col_name,
                        attnum=len(col_stats_ns)
                    ))
        except Exception:
            # Fallback: no samples/col_stats; sample_vec will be zeros during encoding
            table_samples = None
            col_stats_ns = []
        
        parsed = []
        for f in files:
            p_plans, runtimes = parser.parse_explain_analyze_file(
                str(f),
                min_runtime=0,
                max_runtime=float('inf'),
                table_samples=table_samples,
                col_stats=col_stats_ns,
                column_id_mapping=column_id_mapping_ds,
                partial_column_name_mapping=partial_col_map_ds,
                table_id_mapping={}  # not required here
            )
            # annotate database_id and dataset tag
            for p in p_plans:
                setattr(p, 'database_id', 0)  # not used by simple pipeline
                setattr(p, 'dataset', dataset)
            parsed.extend(p_plans)
        return parsed
    
    # Collect train and test plans
    train_plans = []
    for ds in train_datasets:
        files = dataset_to_files[ds]
        train_plans.extend(parse_plans_for_dataset(ds, files))
    test_plans = []
    test_plans.extend(parse_plans_for_dataset(test_dataset, test_files))
    
    # Build feature_statistics from training plans only
    train_plans_dict = [plan_to_dict(p) for p in train_plans]
    feature_statistics = build_feature_statistics(train_plans_dict, train_plans)
    
    # Build combined database_statistics
    database_statistics = SimpleNamespace(
        table_stats=combined_table_stats,
        column_stats=global_column_stats_list,
        database_type='trino'
    )
    
    return train_plans, test_plans, feature_statistics, combined_column_statistics, database_statistics


def create_simple_dataloader(
    plans: List,
    feature_statistics: dict,
    column_statistics: dict,
    database_statistics: SimpleNamespace,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    shuffle: bool = True,
    dim_bitmaps: int = 1000,
    histogram_bin_size: int = 10,
    max_num_filters: int = 5
):
    """
    QueryFormer用の簡易データローダーを作成
    """
    from models.query_former.dataloader import encode_query_plan
    from torch.utils.data import TensorDataset
    import torch
    
    print("=" * 80)
    print("ステップ2: データローダーの作成")
    print("=" * 80)
    print()
    
    # 各プランをエンコード
    all_features = []
    all_join_ids = []
    all_attention_bias = []
    all_rel_pos = []
    all_node_heights = []
    all_labels = []
    
    print(f"{len(plans)} プランをエンコード中...")
    
    # カラムIDマッピングを作成
    print("カラムIDマッピングを作成中...")
    column_id_mapping, partial_column_name_mapping = build_column_id_mapping(database_statistics)
    
    print(f"✅ カラムIDマッピング: {len(column_id_mapping)} カラム")
    
    # plan_parametersをSimpleNamespaceに変換
    print("plan_parametersをSimpleNamespaceに変換中...")
    for plan in plans:
        convert_to_namespace(plan, column_id_mapping, partial_column_name_mapping)
    
    print(f"✅ plan_parametersとfilter_columnsを変換完了")
    print()
    
    for i, plan in enumerate(plans):
        try:
            # join_condsが存在しない場合は空リストを設定
            if not hasattr(plan, 'join_conds'):
                plan.join_conds = []
            
            # database_idが存在しない場合は0を設定
            if not hasattr(plan, 'database_id'):
                plan.database_id = 0
            
            features, join_ids, attention_bias, rel_pos, node_heights = encode_query_plan(
                query_index=i,
                query_plan=plan,
                feature_statistics=feature_statistics,
                column_statistics=column_statistics,
                database_statistics=database_statistics,
                word_embeddings=None,  # QueryFormerでは使用されない
                dim_word_embedding=100,
                dim_word_hash=100,
                dim_bitmaps=dim_bitmaps,
                max_filter_number=max_num_filters,
                histogram_bin_size=histogram_bin_size
            )
            
            all_features.append(features)
            all_join_ids.append(join_ids)
            all_attention_bias.append(attention_bias)
            all_rel_pos.append(rel_pos)
            all_node_heights.append(node_heights)
            
            label = plan.plan_runtime / 1000 if hasattr(plan, 'plan_runtime') else 1.0
            all_labels.append(label)
            
            if (i + 1) % 10 == 0 or i == len(plans) - 1:
                print(f"  進捗: {i+1}/{len(plans)}")
                
        except Exception as e:
            if i < 2:  # 最初の2つのみ詳細を表示
                print(f"\n  ❌ プラン{i+1}のエンコードに失敗: {e}")
                import traceback
                traceback.print_exc()
                print()
            else:
                print(f"  警告: プラン{i+1}のエンコードに失敗: {e}")
            continue
    
    print(f"✅ {len(all_features)} プランをエンコード完了")
    print()
    
    # エンコードされたプランがない場合はエラー
    if len(all_features) == 0:
        raise ValueError("すべてのプランのエンコードに失敗しました。データの確認が必要です。")
    
    # データセット分割
    n_samples = len(all_features)
    indices = list(range(n_samples))
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Tensorに変換
    features_tensor = torch.cat(all_features)
    join_ids_tensor = torch.cat(all_join_ids)
    attention_bias_tensor = torch.cat(all_attention_bias)
    rel_pos_tensor = torch.cat(all_rel_pos)
    node_heights_tensor = torch.cat(all_node_heights)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    
    # label normalizationを作成
    runtimes_array = np.array(all_labels).reshape(-1, 1)
    label_norm = MinMaxScaler()
    label_norm.fit(np.log1p(runtimes_array))
    
    # 正規化されたラベル
    normalized_labels = torch.tensor(
        label_norm.transform(np.log1p(runtimes_array)).flatten(),
        dtype=torch.float32
    )
    
    # データセットを作成
    def create_dataset(indices):
        return TensorDataset(
            features_tensor[indices],
            join_ids_tensor[indices],
            attention_bias_tensor[indices],
            rel_pos_tensor[indices],
            node_heights_tensor[indices],
            normalized_labels[indices]
        )
    
    train_dataset = create_dataset(train_indices)
    val_dataset = create_dataset(val_indices)
    
    # データローダーを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ データローダー作成完了")
    print(f"  - Train: {len(train_dataset)} サンプル, {len(train_loader)} バッチ")
    print(f"  - Val: {len(val_dataset)} サンプル, {len(val_loader)} バッチ")
    print()
    
    return train_loader, val_loader, label_norm, feature_statistics


def train_queryformer_trino(
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_statistics: dict,
    label_norm,
    config: dict,
    model_dir: Path,
    device: str = 'cpu',
    test_loader: Optional[DataLoader] = None,
    save_test_results: bool = False,
    test_dataset_name: Optional[str] = None
):
    """
    QueryFormerモデルをトレーニング
    """
    from classes.classes import QueryFormerModelConfig
    
    print("=" * 80)
    print("ステップ3: モデルの初期化とトレーニング")
    print("=" * 80)
    print()
    
    # feature_statisticsの互換性チェック（PlanModelInputDims/extract_dimensions用）
    if 'table' not in feature_statistics:
        # tablenameからtableを推測
        if 'tablename' in feature_statistics and 'no_vals' in feature_statistics['tablename']:
            feature_statistics['table'] = {'max': max(0, feature_statistics['tablename']['no_vals'] - 1)}
        else:
            feature_statistics['table'] = {'max': 0}
    elif 'max' not in feature_statistics['table']:
        # tableキーは存在するがmaxキーがない場合
        if 'tablename' in feature_statistics and 'no_vals' in feature_statistics['tablename']:
            feature_statistics['table']['max'] = max(0, feature_statistics['tablename']['no_vals'] - 1)
        else:
            feature_statistics['table']['max'] = 0
    
    # columnsエイリアスの確認
    if 'columns' not in feature_statistics:
        feature_statistics['columns'] = feature_statistics.get('column', {'max': 0})
    # columnsにもmaxキーを確実に設定
    if 'columns' in feature_statistics and 'max' not in feature_statistics['columns']:
        if 'column' in feature_statistics and 'max' in feature_statistics['column']:
            feature_statistics['columns']['max'] = feature_statistics['column']['max']
        elif 'no_vals' in feature_statistics['columns']:
            feature_statistics['columns']['max'] = feature_statistics['columns']['no_vals'] - 1
        else:
            feature_statistics['columns']['max'] = 0
    # columnにもmaxキーを確実に設定（columnsからコピー）
    if 'column' in feature_statistics and 'max' not in feature_statistics['column']:
        if 'columns' in feature_statistics and 'max' in feature_statistics['columns']:
            feature_statistics['column']['max'] = feature_statistics['columns']['max']
        elif 'no_vals' in feature_statistics['column']:
            feature_statistics['column']['max'] = feature_statistics['column']['no_vals'] - 1
        else:
            feature_statistics['column']['max'] = 0
    
    # モデル設定
    # Only pass allowed fields to the model config
    allowed_model_keys = {
        'embedding_size', 'ffn_dim', 'head_size', 'dropout', 'attention_dropout_rate',
        'n_layers', 'use_sample', 'use_histogram', 'histogram_bin_number',
        'hidden_dim_prediction', 'max_num_filters'
    }
    model_kwargs = {k: v for k, v in config.items() if k in allowed_model_keys}
    model_config = QueryFormerModelConfig(
        device=device,
        **model_kwargs
    )
    
    # input_dimsを作成
    input_dims = PlanModelInputDims(
        feature_statistics,
        dim_word_embedding=100,
        dim_word_hash=100,
        dim_bitmaps=1000
    )
    
    # モデルを初期化
    model = QueryFormer(
        config=model_config,
        input_dims=input_dims,
        feature_statistics=feature_statistics,
        label_norm=label_norm
    )
    
    model = model.to(device)
    
    print(f"✅ QueryFormerモデル初期化完了")
    print(f"  - パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {device}")
    print()
    
    # オプティマイザー
    # Optimizer learning rate: not part of QueryFormerModelConfig
    optimizer = opt.AdamW(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # メトリクス
    metrics = [RMSE(), MAPE(), QError()]
    
    # チェックポイントを読み込み（存在する場合）
    csv_stats = []
    epochs_wo_improvement = 0
    epoch = 0
    
    if model_dir.exists():
        csv_stats, epochs_wo_improvement, epoch, model, optimizer, metrics, finished = \
            load_checkpoint(
                model=model,
                target_path=model_dir,
                config=model_config,
                optimizer=optimizer,
                metrics=metrics,
                filetype='.pt'
            )
        
        if finished:
            print(f"モデルは既にトレーニング済みです")
            return model, metrics
    
    # トレーニングループ
    epochs = config.get('epochs', 100)
    patience = config.get('patience', 20)
    
    print(f"トレーニング開始: {epochs} エポック")
    print("-" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(epoch, epochs):
        # トレーニング
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch in train_loader:
            features, join_ids, attention_bias, rel_pos, node_heights, labels = batch
            
            features = features.to(device)
            join_ids = join_ids.to(device)
            attention_bias = attention_bias.to(device)
            rel_pos = rel_pos.to(device)
            node_heights = node_heights.to(device)
            labels = labels.to(device)
            
            # フォワードパス
            predictions = model((features, join_ids, attention_bias, rel_pos, node_heights))
            # Ensure labels shape matches original QLoss expectation (N,1)
            labels = labels.view(-1, 1)
            if train_samples == 0:
                print(f"predictions shape: {predictions.shape}, labels shape: {labels.shape}")
            
            loss = model.loss_fxn(predictions, labels)
            
            # バックワードパス
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            train_samples += len(labels)
        
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
        
        # バリデーション
        model.eval()
        val_loss = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features, join_ids, attention_bias, rel_pos, node_heights, labels = batch
                
                features = features.to(device)
                join_ids = join_ids.to(device)
                attention_bias = attention_bias.to(device)
                rel_pos = rel_pos.to(device)
                node_heights = node_heights.to(device)
                labels = labels.to(device)
                
                predictions = model((features, join_ids, attention_bias, rel_pos, node_heights))
                labels = labels.view(-1, 1)
                loss = model.loss_fxn(predictions, labels)
                
                val_loss += loss.item() * len(labels)
                val_samples += len(labels)
        
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
        
        print(f"エポック {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_wo_improvement = 0
            
            # ベストモデルを保存
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, model_dir / 'best_model.pt')
        else:
            epochs_wo_improvement += 1
            
            if epochs_wo_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print()
    print(f"✅ トレーニング完了")
    print(f"  - Best val_loss: {best_val_loss:.4f}")
    print()
    
    # テストセットでの最終評価（提供されている場合）
    if test_loader is not None:
        print("=" * 80)
        print("ステップ4: テストセットでの最終評価")
        print("=" * 80)
        print()
        
        model.eval()
        test_loss = 0
        test_samples = 0
        all_test_preds = []
        all_test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                features, join_ids, attention_bias, rel_pos, node_heights, labels = batch
                
                features = features.to(device)
                join_ids = join_ids.to(device)
                attention_bias = attention_bias.to(device)
                rel_pos = rel_pos.to(device)
                node_heights = node_heights.to(device)
                labels = labels.to(device)
                
                predictions = model((features, join_ids, attention_bias, rel_pos, node_heights))
                labels = labels.view(-1, 1)
                loss = model.loss_fxn(predictions, labels)
                
                test_loss += loss.item() * len(labels)
                test_samples += len(labels)
                
                # 逆変換して保存
                if label_norm is not None:
                    preds_denorm = label_norm.inverse_transform(predictions.cpu().numpy())
                    labels_denorm = label_norm.inverse_transform(labels.cpu().numpy())
                else:
                    preds_denorm = predictions.cpu().numpy()
                    labels_denorm = labels.cpu().numpy()
                
                all_test_preds.append(preds_denorm.flatten())
                all_test_labels.append(labels_denorm.flatten())
        
        avg_test_loss = test_loss / test_samples if test_samples > 0 else 0
        all_test_preds = np.concatenate(all_test_preds)
        all_test_labels = np.concatenate(all_test_labels)
        
        # メトリクス計算（RMSE, QErrorは既にグローバルimport済み）
        # Q-Error計算（0以下の値をクリップ）
        min_val = 0.1
        all_test_preds_clipped = np.clip(all_test_preds, min_val, np.inf)
        all_test_labels_clipped = np.clip(all_test_labels, min_val, np.inf)
        
        q_error_metric = QError(min_val=min_val)
        rmse_metric = RMSE()
        
        q_error_value = q_error_metric.evaluate_metric(labels=all_test_labels_clipped, preds=all_test_preds_clipped)
        rmse_value = rmse_metric.evaluate_metric(labels=all_test_labels, preds=all_test_preds)
        
        print(f"📊 テストセット評価結果:")
        print(f"  - Test Loss: {avg_test_loss:.4f}")
        print(f"  - Test RMSE: {rmse_value:.4f}秒" if rmse_value else "  - Test RMSE: N/A")
        median_q_error = np.median(np.maximum(all_test_preds_clipped / all_test_labels_clipped, all_test_labels_clipped / all_test_preds_clipped)) if len(all_test_preds_clipped) > 0 else None
        print(f"  - Test Median Q-Error: {median_q_error:.4f}" if median_q_error is not None else "  - Test Median Q-Error: N/A")
        print(f"  - Test Mean Q-Error: {q_error_value:.4f}" if q_error_value else "  - Test Mean Q-Error: N/A")
        print(f"  - サンプル数: {test_samples}")
        print()
        
        # テスト結果を保存（save_test_resultsがTrueの場合）
        if save_test_results:
            test_results = {
                'test_dataset': test_dataset_name or 'unknown',
                'test_loss': float(avg_test_loss),
                'test_rmse': float(rmse_value) if rmse_value else None,
                'test_median_q_error': float(median_q_error) if median_q_error is not None else None,
                'test_mean_q_error': float(q_error_value) if q_error_value else None,
                'test_samples': int(test_samples),
                'predictions': all_test_preds.tolist(),
                'labels': all_test_labels.tolist()
            }
            
            results_file = model_dir / 'test_results.json'
            results_file.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"✅ テスト結果を保存: {results_file}")
            print()
    
    return model, metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for QueryFormer training."""
    parser = argparse.ArgumentParser(description='Trino QueryFormer Training')
    parser.add_argument('--mode', choices=['train', 'train_multi', 'train_multi_all', 'predict'], default='train')
    parser.add_argument('--model_type', default='query_former')
    parser.add_argument('--dataset', required=False, default=None, help='Dataset name (e.g., accidents, imdb)')
    parser.add_argument('--txt_file', required=False, default=None, help='Path to EXPLAIN ANALYZE .txt file')
    parser.add_argument('--output_dir', required=True, help='Output directory for models and stats')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--histogram_bin_number', type=int, default=10)
    parser.add_argument('--max_num_filters', type=int, default=30)
    parser.add_argument('--no_samples', type=int, default=1000, help='Number of table samples')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plans_dir', type=str, default='/Users/an/query_engine/explain_analyze_results/', help='Dir of .txt plans for multiple datasets')
    parser.add_argument('--test_dataset', type=str, default=None, help='Dataset name to hold out (prefix in filename)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio for splitting 19 training datasets (default: 0.15)')
    return parser


def run(args) -> int:
    """Run QueryFormer training with parsed arguments."""
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    txt_file = Path(args.txt_file).resolve() if args.txt_file else None  # 絶対パスに変換
    output_dir = Path(args.output_dir).resolve()
    model_dir = output_dir / 'models'
    
    print(f"\n{'='*80}")
    print(f"Trino QueryFormer {args.mode.upper()}")
    print(f"{'='*80}")
    # train_multi_allモードではデータセット情報は後で表示されるため、ここでは表示しない
    if args.mode != 'train_multi_all':
        print(f"データセット: {args.dataset}")
        print(f"入力ファイル: {txt_file}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"デバイス: {args.device}")
    print(f"{'='*80}\n")
    
    # データを準備
    if args.mode == 'train':
        assert args.dataset and args.txt_file, "--dataset and --txt_file are required in train mode"
        parsed_plans, feature_statistics, column_statistics, database_statistics = parse_and_prepare_data(
            txt_file=txt_file,
            dataset=args.dataset,
            output_dir=output_dir,
            no_samples=args.no_samples
        )
        
        # データローダーを作成
        train_loader, val_loader, label_norm, feature_statistics = create_simple_dataloader(
            plans=parsed_plans,
            feature_statistics=feature_statistics,
            column_statistics=column_statistics,
            database_statistics=database_statistics,
            batch_size=args.batch_size,
            val_ratio=0.15,
            shuffle=True,
            histogram_bin_size=args.histogram_bin_number,
            max_num_filters=args.max_num_filters
        )
        
        # モデルをトレーニング
        model_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            # learning rate is used by the optimizer, not the model config
            'learning_rate': args.learning_rate,
            'patience': args.patience,
            'embedding_size': args.embedding_size,
            'histogram_bin_number': args.histogram_bin_number,
            'max_num_filters': args.max_num_filters
        }
        
        # モデルはmodel_dirに保存されるため、戻り値は使用しない
        _ = train_queryformer_trino(
            train_loader=train_loader,
            val_loader=val_loader,
            feature_statistics=feature_statistics,
            label_norm=label_norm,
            config=model_config,
            model_dir=model_dir,
            device=args.device
        )
        
        print("=" * 80)
        print("🎉 トレーニング完了！")
        print("=" * 80)
        print(f"モデル保存先: {model_dir}")
        print()
    
    elif args.mode == 'train_multi':
        assert args.test_dataset is not None, "--test_dataset is required for train_multi mode"
        plans_dir = Path(args.plans_dir)
        train_plans, test_plans, feature_statistics, column_statistics, database_statistics = \
            parse_and_prepare_leave_one_out(
                plans_dir=plans_dir,
                test_dataset=args.test_dataset,
                no_samples=args.no_samples
            )
        
        # データセット数の計算
        train_datasets_set = set()
        for p in train_plans:
            ds = getattr(p, 'dataset', None)
            if ds:
                train_datasets_set.add(ds)
        
        print(f"📊 Leave-One-Out Validation:")
        print(f"  - Training datasets: {len(train_datasets_set)} datasets")
        print(f"  - Training plans: {len(train_plans)} plans")
        print(f"  - Test dataset: {args.test_dataset}")
        print(f"  - Test plans: {len(test_plans)} plans")
        print()
        
        # Create loaders: 19 datasetsをtraining (約85%)とvalidation (約15%)に分割
        train_loader, val_loader_train, label_norm, feature_statistics = create_simple_dataloader(
            plans=train_plans,
            feature_statistics=feature_statistics,
            column_statistics=column_statistics,
            database_statistics=database_statistics,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,  # 19個のデータセットをtrain/valに分割（デフォルト: 0.15）
            shuffle=True,
            histogram_bin_size=args.histogram_bin_number,
            max_num_filters=args.max_num_filters
        )
        
        print(f"✅ 19個のデータセットから作成:")
        print(f"  - Train loader: {len(train_loader.dataset)} サンプル")
        print(f"  - Val loader (from 19 datasets): {len(val_loader_train.dataset)} サンプル")
        print()
        
        # Build test loader separately (use same feature_statistics)
        test_loader = create_test_loader(
            test_plans=test_plans,
            feature_statistics=feature_statistics,
            column_statistics=column_statistics,
            database_statistics=database_statistics,
            label_norm=label_norm,
            batch_size=args.batch_size,
            max_num_filters=args.max_num_filters,
            histogram_bin_size=args.histogram_bin_number
        )
        
        print(f"✅ テストデータセット（{args.test_dataset}）から作成:")
        print(f"  - Test loader: {len(test_loader.dataset)} サンプル")
        print()
        
        # Train using validation loader from 19 datasets for early stopping
        # After training, evaluate on test loader
        # モデルとテスト結果はmodel_dirに保存されるため、戻り値は使用しない
        model_dir = output_dir / f'models_{args.test_dataset}'
        _ = train_queryformer_trino(
            train_loader=train_loader,
            val_loader=val_loader_train,  # 19個のデータセットから分割したvalidation setを使用
            feature_statistics=feature_statistics,
            label_norm=label_norm,  # 訓練データから作成したlabel_normを使用
            config={
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'patience': args.patience,
                'embedding_size': args.embedding_size,
                'histogram_bin_number': args.histogram_bin_number,
                'max_num_filters': args.max_num_filters
            },
            model_dir=model_dir,
            device=args.device,
            test_loader=test_loader  # テスト用に追加
        )
        print("=" * 80)
        print("🎉 マルチデータセット学習完了！")
        print("=" * 80)
        print(f"モデル保存先: {model_dir}")
        print()
        return 0
    
    elif args.mode == 'train_multi_all':
        # 20個すべてのデータセットについてleave-one-out validationを実行
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
        print(f"Leave-One-Out Validation for All Datasets")
        print(f"{'='*80}")
        print(f"利用可能なデータセット: {len(available_datasets)} / {len(ALL_DATASETS)}")
        print(f"データセット: {', '.join(available_datasets)}")
        print(f"出力ディレクトリ: {output_dir}")
        print(f"{'='*80}\n")
        
        # 最初に1回だけ全データセットのプランを読み込む
        print("=" * 80)
        print("ステップ0: 全データセットのプランを読み込み中...")
        print("=" * 80)
        print()
        
        all_plans_by_dataset, all_column_statistics, all_database_statistics = parse_all_datasets_once(
            plans_dir=plans_dir,
            available_datasets=available_datasets,
            no_samples=args.no_samples
        )
        
        print(f"✅ 全データセットの読み込み完了")
        print(f"  - 読み込んだデータセット: {len(all_plans_by_dataset)}")
        for ds, plans in all_plans_by_dataset.items():
            print(f"    - {ds}: {len(plans)} プラン")
        print()
        
        # 各データセットについて訓練・テストを実行
        results_summary = []
        
        for idx, test_dataset in enumerate(available_datasets, 1):
            print(f"\n{'#'*80}")
            print(f"# [{idx}/{len(available_datasets)}] Testing dataset: {test_dataset}")
            print(f"{'#'*80}\n")
            
            try:
                # 既に読み込んだプランからtrain/testを分割
                train_plans = []
                test_plans = all_plans_by_dataset[test_dataset]
                
                for ds, plans in all_plans_by_dataset.items():
                    if ds != test_dataset:
                        train_plans.extend(plans)
                
                # feature_statisticsを全データセット（訓練+テスト）から生成（未知の演算子タイプを避けるため）
                all_plans_for_stats = train_plans + test_plans
                all_plans_dict = [plan_to_dict(p) for p in all_plans_for_stats]
                feature_statistics = build_feature_statistics(all_plans_dict, all_plans_for_stats)
                
                # column_statisticsとdatabase_statisticsは既に読み込まれている
                column_statistics = all_column_statistics
                database_statistics = all_database_statistics
                
                # データセット数の計算
                train_datasets_set = set()
                for p in train_plans:
                    ds = getattr(p, 'dataset', None)
                    if ds:
                        train_datasets_set.add(ds)
                
                print(f"📊 Leave-One-Out Validation [{idx}/{len(available_datasets)}]:")
                print(f"  - Training datasets: {len(train_datasets_set)} datasets")
                print(f"  - Training plans: {len(train_plans)} plans")
                print(f"  - Test dataset: {test_dataset}")
                print(f"  - Test plans: {len(test_plans)} plans")
                print()
                
                # データローダーを作成（19個のデータセットをtrain/valに分割）
                train_loader, val_loader_train, label_norm, feature_statistics = create_simple_dataloader(
                    plans=train_plans,
                    feature_statistics=feature_statistics,
                    column_statistics=column_statistics,
                    database_statistics=database_statistics,
                    batch_size=args.batch_size,
                    val_ratio=args.val_ratio,
                    shuffle=True,
                    histogram_bin_size=args.histogram_bin_number,
                    max_num_filters=args.max_num_filters
                )
                
                print(f"✅ 19個のデータセットから作成:")
                print(f"  - Train loader: {len(train_loader.dataset)} サンプル")
                print(f"  - Val loader (from 19 datasets): {len(val_loader_train.dataset)} サンプル")
                print()
                
                # テストローダーを作成（train_multiと同じロジック）
                test_loader = create_test_loader(
                    test_plans=test_plans,
                    feature_statistics=feature_statistics,
                    column_statistics=column_statistics,
                    database_statistics=database_statistics,
                    label_norm=label_norm,
                    batch_size=args.batch_size,
                    max_num_filters=args.max_num_filters,
                    histogram_bin_size=args.histogram_bin_number
                )
                
                print(f"✅ テストデータセット（{test_dataset}）から作成:")
                print(f"  - Test loader: {len(test_loader.dataset)} サンプル")
                print()
                
                # モデルディレクトリ（各データセットごとに分ける）
                model_dir = output_dir / f'models_{test_dataset}'
                
                # 訓練実行（モデルとテスト結果はmodel_dirに保存されるため、戻り値は使用しない）
                _ = train_queryformer_trino(
                    train_loader=train_loader,
                    val_loader=val_loader_train,
                    feature_statistics=feature_statistics,
                    label_norm=label_norm,
                    config={
                        'epochs': args.epochs,
                        'batch_size': args.batch_size,
                        'learning_rate': args.learning_rate,
                        'patience': args.patience,
                        'embedding_size': args.embedding_size,
                        'histogram_bin_number': args.histogram_bin_number,
                        'max_num_filters': args.max_num_filters
                    },
                    model_dir=model_dir,
                    device=args.device,
                    test_loader=test_loader,
                    save_test_results=True,
                    test_dataset_name=test_dataset
                )
                
                # 結果を保存（最後のテスト結果を取得）
                # train_queryformer_trino内でテスト結果が保存されるため、ここではサマリーを記録
                results_summary.append({
                    'test_dataset': test_dataset,
                    'model_dir': str(model_dir),
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
        import json
        with open(summary_file, 'w') as f:
            json.dump({
                'total_datasets': len(available_datasets),
                'completed': len([r for r in results_summary if r['status'] == 'completed']),
                'failed': len([r for r in results_summary if r['status'] == 'failed']),
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
    
    elif args.mode == 'predict':
        # 予測モード（TODO: 実装）
        print("予測モードは未実装です")
        return 1
    
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for QueryFormer training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())

