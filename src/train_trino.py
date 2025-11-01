#!/usr/bin/env python3
"""
Trino QueryFormer Training Script

Usage:
    python src/train_trino.py \\
        --mode train \\
        --model_type query_former \\
        --dataset accidents \\
        --txt_file ../explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt \\
        --output_dir data/runs/trino/accidents \\
        --device cuda \\
        --epochs 100 \\
        --batch_size 32
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

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
from gensim.models import KeyedVectors


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
    
    # テーブルサンプルとカラム統計を準備
    print(f"テーブルサンプルを取得中（{dataset}）...")
    table_samples = get_table_samples_from_csv(dataset, data_dir=None, no_samples=no_samples)
    
    schema = load_schema_json(dataset)
    col_stats = []
    for table_name in schema.tables:
        if table_name in table_samples:
            df = table_samples[table_name]
            for col_name in df.columns:
                col_stats.append(SimpleNamespace(
                    tablename=table_name,
                    attname=col_name,
                    attnum=len(col_stats)
                ))
    
    print(f"✅ テーブルサンプル: {len(table_samples)} テーブル")
    print(f"✅ カラム統計: {len(col_stats)} カラム")
    print()
    
    # 統計情報を先に読み込む（カラムIDマッピングを作成するため）
    print(f"統計情報を読み込み中（{dataset}）...")
    stats_dir = Path('..') / 'datasets_statistics' / f'iceberg_{dataset}'
    
    if not stats_dir.exists():
        raise FileNotFoundError(f"統計情報が見つかりません: {stats_dir}")
    
    with open(stats_dir / 'column_stats.json') as f:
        column_stats_dict = json.load(f)
    
    with open(stats_dir / 'table_stats.json') as f:
        table_stats_dict = json.load(f)
    
    # カラムIDマッピングを作成
    column_id_mapping = {}
    partial_column_name_mapping = {}
    table_id_mapping = {}
    
    column_stats_list = []
    for col_key, stats in column_stats_dict.items():
        idx = len(column_stats_list)
        table_name = stats.get('table')
        col_name = stats.get('column')
        
        if table_name and col_name:
            column_id_mapping[(table_name, col_name)] = idx
            
            if col_name not in partial_column_name_mapping:
                partial_column_name_mapping[col_name] = set()
            partial_column_name_mapping[col_name].add(table_name)
        
        column_stats_list.append(stats)
    
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
    
    def plan_to_dict(node):
        result = {}
        if hasattr(node, 'plan_parameters'):
            if hasattr(node.plan_parameters, '__dict__'):
                result = vars(node.plan_parameters).copy()
            elif isinstance(node.plan_parameters, dict):
                result = node.plan_parameters.copy()
        
        # 'table'を'tablename'としても追加
        if 'table' in result and 'tablename' not in result:
            result['tablename'] = result['table']
        
        if hasattr(node, 'children') and node.children:
            result['children'] = [plan_to_dict(c) for c in node.children]
        
        return result
    
    plans_dict = [plan_to_dict(p) for p in parsed_plans]
    
    from training.preprocessing.feature_statistics import gather_values_recursively
    from sklearn.preprocessing import RobustScaler
    
    value_dict = gather_values_recursively(plans_dict)
    
    statistics_dict = dict()
    for k, values in value_dict.items():
        values = [v for v in values if v is not None]
        if len(values) == 0:
            continue
        
        if all([isinstance(v, (int, float)) or v is None for v in values]):
            scaler = RobustScaler()
            np_values = np.array(values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(np_values)
            
            statistics_dict[k] = dict(
                max=float(np_values.max()),
                scale=scaler.scale_.item(),
                center=scaler.center_.item(),
                type='numeric'
            )
        else:
            unique_values = set(values)
            statistics_dict[k] = dict(
                value_dict={v: id for id, v in enumerate(unique_values)},
                no_vals=len(unique_values),
                type='categorical'
            )
    
    feature_statistics = statistics_dict
    
    # 'tablename'がない場合は追加
    if 'tablename' not in feature_statistics:
        table_names = set()
        for plan_dict in plans_dict:
            if 'tablename' in plan_dict:
                table_names.add(plan_dict['tablename'])
            if 'table' in plan_dict:
                table_names.add(plan_dict['table'])
        
        if table_names:
            feature_statistics['tablename'] = {
                'value_dict': {v: idx for idx, v in enumerate(table_names)},
                'no_vals': len(table_names),
                'type': 'categorical'
            }
    
    # 'join_conds'を追加
    join_conds_set = set()
    for plan in parsed_plans:
        if hasattr(plan, 'join_conds') and plan.join_conds:
            join_conds_set.update(plan.join_conds)
    
    feature_statistics['join_conds'] = {
        'value_dict': {jc: idx for idx, jc in enumerate(join_conds_set)} if join_conds_set else {},
        'no_vals': len(join_conds_set),
        'type': 'categorical'
    }
    
    # 'column'に'max'を追加
    if 'column' in feature_statistics and 'max' not in feature_statistics['column']:
        feature_statistics['column']['max'] = feature_statistics['column']['no_vals'] - 1
    
    # 'columns'エイリアスを追加
    if 'columns' not in feature_statistics:
        feature_statistics['columns'] = feature_statistics.get('column', {'max': 0})
    
    # 'table'エイリアスを追加（PlanModelInputDims/extract_dimensions互換）
    if 'table' not in feature_statistics:
        if 'tablename' in feature_statistics and 'no_vals' in feature_statistics['tablename']:
            feature_statistics['table'] = {'max': max(0, feature_statistics['tablename']['no_vals'] - 1)}
        else:
            feature_statistics['table'] = {'max': 0}
    
    print(f"✅ feature_statistics: {len(feature_statistics)} 特徴量")
    
    # feature_statisticsを保存
    feature_stats_file = output_dir / 'feature_statistics.json'
    feature_stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_stats_file, 'w') as f:
        json.dump(feature_statistics, f, indent=2)
    
    print(f"✅ feature_statisticsを保存: {feature_stats_file}")
    print()
    
    column_statistics = {}
    for col_key, stats in column_stats_dict.items():
        table_name = stats.get('table')
        col_name = stats.get('column')
        
        if table_name and col_name:
            if table_name not in column_statistics:
                column_statistics[table_name] = {}
            
            column_statistics[table_name][col_name] = {
                'datatype': stats.get('datatype', 'misc'),
                'min': stats.get('min'),
                'max': stats.get('max'),
                'percentiles': stats.get('percentiles'),
                'num_unique': stats.get('distinct_values_count', -1),
                'null_frac': stats.get('null_frac', 0)
            }
    
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
    
    column_stats_list = []
    for col_key, stats in column_stats_dict.items():
        column_stats_list.append(SimpleNamespace(
            tablename=stats.get('table'),
            attname=stats.get('column'),
            attnum=len(column_stats_list),
            null_frac=stats.get('null_frac', 0),
            avg_width=stats.get('avg_width', 0),
            n_distinct=stats.get('n_distinct', -1),
            correlation=stats.get('correlation', 0)
        ))
    
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
        # Assume dataset name is prefix before first underscore
        stem = p.name
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
        stats_dir = Path('src').parent / 'datasets_statistics' / f'iceberg_{dataset}'
        if not stats_dir.exists():
            # No stats available; return empty dicts and mappings
            column_stats_dict = {}
            table_stats_dict = {}
        else:
            with open(stats_dir / 'column_stats.json') as f:
                column_stats_dict = json.load(f)
            with open(stats_dir / 'table_stats.json') as f:
                table_stats_dict = json.load(f)
        
        # Build table stats entries (with dataset prefix)
        for table_name, tstats in table_stats_dict.items():
            combined_table_stats.append(SimpleNamespace(
                relname=f"{dataset}.{table_name}",
                reltuples=tstats.get('reltuples', tstats.get('row_count', 0)),
                relpages=tstats.get('relpages', 0)
            ))
        
        # Per-dataset column_id_mapping to global ids
        column_id_mapping_ds = {}
        partial_column_name_mapping_ds = {}
        for _, cstats in column_stats_dict.items():
            table_name = cstats.get('table')
            col_name = cstats.get('column')
            if table_name is None or col_name is None:
                continue
            key = (dataset, table_name, col_name)
            if key not in global_column_id:
                gid = len(global_column_stats_list)
                global_column_id[key] = gid
                global_column_stats_list.append(SimpleNamespace(
                    tablename=f"{dataset}.{table_name}",
                    attname=col_name,
                    attnum=gid,
                    null_frac=cstats.get('null_frac', 0),
                    avg_width=cstats.get('avg_width', 0),
                    n_distinct=cstats.get('n_distinct', -1),
                    correlation=cstats.get('correlation', 0)
                ))
            # For parsing within this dataset
            column_id_mapping_ds[(table_name, col_name)] = global_column_id[key]
            partial_column_name_mapping_ds.setdefault(col_name, set()).add(table_name)
        
        # Build combined column_statistics with prefixed table
        for _, cstats in column_stats_dict.items():
            tname = cstats.get('table')
            cname = cstats.get('column')
            if tname and cname:
                pref_t = f"{dataset}.{tname}"
                combined_column_statistics.setdefault(pref_t, {})[cname] = {
                    'datatype': cstats.get('datatype', 'misc'),
                    'min': cstats.get('min'),
                    'max': cstats.get('max'),
                    'percentiles': cstats.get('percentiles'),
                    'num_unique': cstats.get('distinct_values_count', -1),
                    'null_frac': cstats.get('null_frac', 0)
                }
        
        return column_stats_dict, table_stats_dict, column_id_mapping_ds, partial_column_name_mapping_ds
    
    # Parse plans for a dataset list
    def parse_plans_for_dataset(dataset: str, files: list):
        column_stats_dict, table_stats_dict, column_id_mapping_ds, partial_col_map_ds = load_stats_and_prepare_ids(dataset)
        
        # Prepare table samples and simple col_stats namespaces for sample_vec generation (optional)
        table_samples = None
        col_stats_ns = []
        try:
            table_samples = get_table_samples_from_csv(dataset, data_dir=None, no_samples=no_samples)
            schema = load_schema_json(dataset)
            for table_name in schema.tables:
                if table_name in table_samples:
                    df = table_samples[table_name]
                    for col_name in df.columns:
                        col_stats_ns.append(SimpleNamespace(tablename=table_name, attname=col_name, attnum=len(col_stats_ns)))
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
    def plan_to_dict(node):
        result = {}
        if hasattr(node, 'plan_parameters'):
            if hasattr(node.plan_parameters, '__dict__'):
                result = vars(node.plan_parameters).copy()
            elif isinstance(node.plan_parameters, dict):
                result = node.plan_parameters.copy()
        # add tablename alias
        if 'table' in result and 'tablename' not in result:
            result['tablename'] = result['table']
        if hasattr(node, 'children') and node.children:
            result['children'] = [plan_to_dict(c) for c in node.children]
        return result
    
    train_plans_dict = [plan_to_dict(p) for p in train_plans]
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
    if 'column' in feature_statistics and 'max' not in feature_statistics['column']:
        feature_statistics['column']['max'] = feature_statistics['column']['no_vals'] - 1
    if 'columns' not in feature_statistics:
        feature_statistics['columns'] = feature_statistics.get('column', {'max': 0})
    if 'tablename' not in feature_statistics:
        feature_statistics['tablename'] = {'value_dict': {}, 'no_vals': 0, 'type': 'categorical'}
    if 'table' not in feature_statistics:
        feature_statistics['table'] = {'max': max(0, feature_statistics.get('tablename', {}).get('no_vals', 1) - 1)}
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
    column_id_mapping = {}
    partial_column_name_mapping = {}
    
    for i, col_stat in enumerate(database_statistics.column_stats):
        key = (col_stat.tablename, col_stat.attname)
        column_id_mapping[key] = i
        
        if col_stat.attname not in partial_column_name_mapping:
            partial_column_name_mapping[col_stat.attname] = set()
        partial_column_name_mapping[col_stat.attname].add(col_stat.tablename)
    
    print(f"✅ カラムIDマッピング: {len(column_id_mapping)} カラム")
    
    # filter_columnsのcolumnをカラムIDに変換し、literalを実値に変換
    def convert_filter_node(filter_node):
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
                for table_name in partial_column_name_mapping.get(col_name, set()):
                    if (table_name, col_name) in column_id_mapping:
                        filter_node.column = column_id_mapping[(table_name, col_name)]
                        break
                else:
                    # サフィックス削除版で再試行
                    for table_name in partial_column_name_mapping.get(col_name_without_suffix, set()):
                        if (table_name, col_name_without_suffix) in column_id_mapping:
                            filter_node.column = column_id_mapping[(table_name, col_name_without_suffix)]
                            break
                    else:
                        # 変換できない場合はNoneに設定
                        filter_node.column = None
            elif len(column) == 2:
                table_name, col_name = column
                table_name = str(table_name).strip('"')
                col_name = str(col_name).strip('"')
                if (table_name, col_name) in column_id_mapping:
                    filter_node.column = column_id_mapping[(table_name, col_name)]
                else:
                    filter_node.column = None
        
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
                convert_filter_node(child)
    
    # plan_parametersをSimpleNamespaceに変換し、filter_columnsも変換
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
    
    def convert_to_namespace(node):
        """辞書形式のplan_parametersとfilter_columnsをSimpleNamespaceに変換"""
        if hasattr(node, 'plan_parameters'):
            if isinstance(node.plan_parameters, dict):
                # 辞書をSimpleNamespaceに変換
                node.plan_parameters = dict_to_namespace_recursive(node.plan_parameters)
            
            # filter_columnsのcolumnとliteralを変換
            if hasattr(node.plan_parameters, 'filter_columns') and node.plan_parameters.filter_columns:
                convert_filter_node(node.plan_parameters.filter_columns)
        
        # 子ノードを再帰的に変換
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                convert_to_namespace(child)
    
    print("plan_parametersをSimpleNamespaceに変換中...")
    for plan in plans:
        convert_to_namespace(plan)
    
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
    device: str = 'cpu'
):
    """
    QueryFormerモデルをトレーニング
    """
    from classes.classes import QueryFormerModelConfig
    
    print("=" * 80)
    print("ステップ3: モデルの初期化とトレーニング")
    print("=" * 80)
    print()
    
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
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Trino QueryFormer Training')
    parser.add_argument('--mode', choices=['train', 'train_multi', 'predict'], default='train')
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
    
    args = parser.parse_args()
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    txt_file = Path(args.txt_file).resolve() if args.txt_file else None  # 絶対パスに変換
    output_dir = Path(args.output_dir).resolve()
    model_dir = output_dir / 'models'
    
    print(f"\n{'='*80}")
    print(f"Trino QueryFormer {args.mode.upper()}")
    print(f"{'='*80}")
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
        
        model, metrics = train_queryformer_trino(
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
        
        # Create loaders: train on 19 datasets, validate on held-out dataset
        train_loader, _, label_norm, feature_statistics = create_simple_dataloader(
            plans=train_plans,
            feature_statistics=feature_statistics,
            column_statistics=column_statistics,
            database_statistics=database_statistics,
            batch_size=args.batch_size,
            val_ratio=0.0,
            shuffle=True,
            histogram_bin_size=args.histogram_bin_number,
            max_num_filters=args.max_num_filters
        )
        
        # Build test loader separately (use same feature_statistics)
        # Reuse function by treating test plans as entire dataset and no split
        from models.query_former.dataloader import encode_query_plan
        all_features = []
        all_join_ids = []
        all_attention_bias = []
        all_rel_pos = []
        all_node_heights = []
        all_labels = []
        # Reuse conversion utilities from single-dataset path
        column_id_mapping = {}
        partial_column_name_mapping = {}
        for i, col_stat in enumerate(database_statistics.column_stats):
            key = (col_stat.tablename, col_stat.attname)
            column_id_mapping[key] = i
            cname = col_stat.attname
            tname = col_stat.tablename
            partial_column_name_mapping.setdefault(cname, set()).add(tname)
        def dict_to_namespace_recursive(d):
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
        def convert_filter_node(filter_node):
            if not hasattr(filter_node, 'column'):
                return
            column = filter_node.column
            if column is not None and isinstance(column, (tuple, list)):
                if len(column) == 1:
                    col_name = str(column[0]).strip('"')
                    import re
                    col_name_without_suffix = re.sub(r'_\d+$', '', col_name)
                    mapped = None
                    for tname in partial_column_name_mapping.get(col_name, set()):
                        key = (tname, col_name)
                        if key in column_id_mapping:
                            mapped = column_id_mapping[key]
                            break
                    if mapped is None:
                        for tname in partial_column_name_mapping.get(col_name_without_suffix, set()):
                            key = (tname, col_name_without_suffix)
                            if key in column_id_mapping:
                                mapped = column_id_mapping[key]
                                break
                    filter_node.column = mapped
                elif len(column) == 2:
                    table_name, col_name = column
                    table_name = str(table_name).strip('"')
                    col_name = str(col_name).strip('"')
                    key = (table_name, col_name)
                    filter_node.column = column_id_mapping.get(key)
            literal = filter_node.literal
            if literal is not None:
                if isinstance(literal, list):
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
            if hasattr(filter_node, 'children') and filter_node.children:
                for child in filter_node.children:
                    convert_filter_node(child)
        def convert_to_namespace(node):
            if hasattr(node, 'plan_parameters'):
                if isinstance(node.plan_parameters, dict):
                    node.plan_parameters = dict_to_namespace_recursive(node.plan_parameters)
                if hasattr(node.plan_parameters, 'filter_columns') and node.plan_parameters.filter_columns:
                    convert_filter_node(node.plan_parameters.filter_columns)
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    convert_to_namespace(child)
        for tp in test_plans:
            convert_to_namespace(tp)
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
                    max_filter_number=args.max_num_filters,
                    histogram_bin_size=args.histogram_bin_number
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
        assert all_features, "No test features encoded. Aborting."
        features_tensor = torch.cat(all_features)
        join_ids_tensor = torch.cat(all_join_ids)
        attention_bias_tensor = torch.cat(all_attention_bias)
        rel_pos_tensor = torch.cat(all_rel_pos)
        node_heights_tensor = torch.cat(all_node_heights)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32).view(-1, 1)
        from torch.utils.data import TensorDataset
        test_dataset_tensor = TensorDataset(features_tensor, join_ids_tensor, attention_bias_tensor, rel_pos_tensor, node_heights_tensor, labels_tensor)
        val_loader = DataLoader(test_dataset_tensor, batch_size=args.batch_size, shuffle=False)
        
        # Train using test loader as validation
        model_dir = output_dir / f'models_{args.test_dataset}'
        model, metrics = train_queryformer_trino(
            train_loader=train_loader,
            val_loader=val_loader,
            feature_statistics=feature_statistics,
            label_norm=None,
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
            device=args.device
        )
        print("=" * 80)
        print("🎉 マルチデータセット学習完了！")
        print("=" * 80)
        print(f"モデル保存先: {model_dir}")
        print()
        return 0
    
    elif args.mode == 'predict':
        # 予測モード（TODO: 実装）
        print("予測モードは未実装です")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

