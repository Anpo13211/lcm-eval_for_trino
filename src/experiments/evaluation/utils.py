import glob
import os
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import seaborn as sns
from pandas import CategoricalDtype

palette = sns.color_palette("deep", 10)


@dataclass()
class FilenameAttribute:
    name: str
    pos: int = None
    end_pos: int = None
    default: any = None
    integer: bool = False


@dataclass()
class ZeroShotConfiguration:
    name: str
    label: str
    folder_name: str


def get_zero_shot_configs(include_deepdb=True, include_est=False, include_exact=True):
    configs = []
    if include_exact:
        configs += [ZeroShotConfiguration('zero_shot_exact', 'Zero-Shot (Exact)', 'tune')]
    if include_deepdb:
        configs += [ZeroShotConfiguration('zero_shot_deepdb_est', 'Zero-Shot (DeepDB Est.)', 'tune_deepdb')]
    if include_est:
        configs += [ZeroShotConfiguration('zero_shot_est', 'Zero-Shot (Estimated)', 'tune_est')]
    return configs


def get_few_shot_configs(include_deepdb=True, include_est=False, include_exact=True):
    configs = []
    if include_exact:
        configs += [ZeroShotConfiguration('few_shot_exact', 'Few-Shot (Exact)', 'tune')]
    if include_deepdb:
        configs += [ZeroShotConfiguration('few_shot_deepdb_est', 'Few-Shot (DeepDB Est.)', 'tune_deepdb')]
    if include_est:
        configs += [ZeroShotConfiguration('few_shot_est', 'Few-Shot (Estimated)', 'tune_est')]
    return configs


# baselines & zero shot configurations
def read_csvs(csv_dir, filename_attributes, test_only=True, complex_only=False, skip_updates=True):
    assert os.path.exists(csv_dir)

    csv_dir += '/*.csv'

    all_dfs = []
    for f in glob.glob(csv_dir):
        filename = os.path.basename(f)

        test = filename.startswith('test_')

        if test_only and not filename.startswith('test_'):
            continue
        elif filename.startswith('test_'):
            filename = filename[len('test_'):]
        filename = filename.replace('.csv', '').replace('tpc_h', 'tpc-h')

        if not 'complex_workload' in filename and complex_only:
            continue

        if '_repl_' in filename and skip_updates:
            continue

        # read and augment dataset
        try:
            curr_df = pd.read_csv(f)
        except:
            continue
        for fa in filename_attributes:
            try:
                if fa.pos is not None:
                    value = filename.split('_')
                    if fa.end_pos is None:
                        value = value[fa.pos]
                    else:
                        value = '_'.join(value[fa.pos:fa.end_pos])
                else:
                    value = fa.default
                if fa.integer:
                    value = int(value)
                curr_df[fa.name] = value
            except:
                continue
        curr_df['test'] = test

        all_dfs.append(curr_df)

    return pd.concat(all_dfs)


def read_zero_shot_generalization():
    dfs_zero_shot = []
    for complex in [False, True]:
        for c in get_zero_shot_configs(include_deepdb=True, include_est=True):
            if complex:
                dir = f'../data/db_complex_generalization_{c.folder_name}'
            else:
                dir = f'../data/db_generalization_{c.folder_name}'
            if not os.path.exists(dir):
                continue

            exp_cards = read_csvs(dir, [FilenameAttribute('dataset', pos=0),
                                        FilenameAttribute('workload', pos=2),
                                        FilenameAttribute('approach', default=c.name)],
                                  complex_only=complex)
            dfs_zero_shot.append(exp_cards)
    return pd.concat(dfs_zero_shot)


@dataclass()
class Approach:
    name: str
    label: str
    color: any
    few_shot: bool = False


class Approaches(Enum):
    AnalyticalEstCard = Approach('AnalyticalEstCard', 'Learned Analytical Model \n(Est. Cardinalities)', palette[8])
    AnalyticalActCard = Approach('AnalyticalActCard', 'Learned Analytical Model \n(Act. Cardinalities)', palette[7])
    LightGBM = Approach('LightGBM', 'Flattened Plans', palette[6])
    Optimizer = Approach('ScaledOptimizer', 'Scaled Optimizer Costs (Postgres)', palette[5])
    CloudDWOptimizer = Approach('ScaledOptimizer', 'Scaled Optimizer Costs (Cloud DW)', palette[5])
    MSCN = Approach('MSCN', 'MSCN (Workload-Driven)', palette[4])
    E2E = Approach('TPool', 'E2E (Workload-Driven)', palette[3])
    ZeroShotExact = Approach('zero_shot_exact', 'Zero-Shot\n(Exact Cardinalities)', palette[0])
    ZeroShotEst = Approach('zero_shot_est', 'Zero-Shot\n(Est. Cardinalities)', palette[1])
    ZeroShotDeepDB = Approach('zero_shot_deepdb_est', 'Zero-Shot\n(DeepDB Est. Cardinalities)', palette[2])
    FewShotExact = Approach('few_shot_exact', 'Few-Shot\n(Exact Cardinalities)', palette[0], few_shot=True)
    FewShotEst = Approach('few_shot_est', 'Few-Shot\n(Est. Cardinalities)', palette[1], few_shot=True)
    FewShotDeepDB = Approach('few_shot_deepdb_est', 'Few-Shot\n(DeepDB Est. Cardinalities)', palette[2], few_shot=True)


def get_plotting_info(approaches, fs=False):
    model_names = [a.value.name for a in approaches]
    labels = [a.value.label for a in approaches]
    palette = [a.value.color for a in approaches]

    if not fs:
        return model_names, labels, palette

    few_shot = [a.value.few_shot for a in approaches]
    return model_names, labels, palette, few_shot


def filter_fs(l, few_shot, inv=True):
    return [le for le, fs in zip(l, few_shot) if fs != inv]


@dataclass()
class Workload:
    name: str
    label: str


class ImdbWorkloads(Enum):
    Scale = Workload('scale', 'Scale')
    Synthetic = Workload('synthetic', 'Synthetic')
    JOBLight = Workload('job-light', 'JOB-light')


def prepare_df(df_curr, model_names, labels, workloads=None):
    df_curr = df_curr[df_curr.approach.isin(model_names)].copy()
    df_curr.approach = df_curr['approach'].astype(CategoricalDtype(
        model_names,
        ordered=True
    ))

    sort_cols = ['approach']
    if workloads is not None:
        sort_cols.append('workload')
        df_curr = df_curr[df_curr.database.isin({wl.value.name for wl in workloads})]
        df_curr.database = df_curr['workload'].astype(CategoricalDtype(
            [wl.value.name for wl in workloads],
            ordered=True
        ))

    if 'dataset' in df_curr.columns:
        sort_cols.append('dataset')

    df_curr = df_curr.sort_values(sort_cols)

    df_curr.approach = df_curr['approach'].astype(str)
    df_curr = df_curr.replace({'approach': {n: l for n, l in zip(model_names, labels)}})

    if workloads is not None:
        df_curr.database = df_curr['workload'].astype(str)
        df_curr = df_curr.replace({'workload': {wl.value.name: wl.value.label for wl in workloads}})

    return df_curr
