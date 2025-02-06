import collections
import json
import os

import pandas as pd
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.parse_run import dumper
from cross_db_benchmark.benchmark_tools.utils import load_schema_json, load_string_statistics, load_column_statistics, \
    load_json


def create_sentences(dataset, plan_paths, data_dir, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)

    # per column find regex queries
    runs = [load_json(plan_path) for plan_path in plan_paths]
    col_stats = runs[0].database_stats.column_stats
    regex_patterns = collections.defaultdict(set)
    for run in runs:
        for p in run.parsed_plans:
            augment_regex_patterns(p, regex_patterns, col_stats)

    # load statistics
    schema = load_schema_json(dataset)
    column_statistics = load_column_statistics(dataset, namespace=False)
    string_statistics = load_string_statistics(dataset, namespace=False)

    # read individual table csvs and derive sentences
    sentences = []
    for t in schema.tables:
        table_dir = os.path.join(data_dir, f'{t}.csv')
        assert os.path.exists(data_dir), f"Could not find table csv {table_dir}"
        print(f"Generating sentences for word embeddings for {t}")

        # find string columns
        sentence_columns, id_columns = find_sentence_columns(column_statistics, schema, string_statistics, t)

        # df_sample = pd.read_csv(table_dir, **vars(schema.csv_kwargs), nrows=100000)
        df_sample = pd.read_csv(table_dir, **vars(schema.csv_kwargs))
        if len(df_sample) > 100000:
            df_sample = df_sample.sample(random_state=0, n=100000)

        for _, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0]):
            sentence = []
            sentence_reg = []

            # base sentences
            for c in df_sample.columns:
                if c not in sentence_columns:
                    continue
                val = str(row[c])
                sentence.append(f'{c}_{val}')
                if c in id_columns:
                    sentence_reg.append(f'{c}_{val}')
                if ' ' in val:
                    vals = val.split(' ')
                    sentence += [f'{c}_{v}' for v in vals]
            if len(sentence) > 0:
                sentences.append(sentence)

            # regex query sentence (sentence_reg)
            for c in df_sample.columns:
                val = str(row[c])
                for regex_p in regex_patterns[(t, c)]:
                    # check whether it applies
                    if all([str_part in val for str_part in regex_p]):
                        sentence_reg += [f'{c}_{str_part}' for str_part in regex_p]
            if len(sentence_reg) > 0:
                sentences.append(sentence_reg)

    with open(target, 'w') as outfile:
        json.dump(sentences, outfile, default=dumper)


def augment_regex_pattern_per_filter(filter, regex_patterns, col_stats):
    if filter.operator in {'LIKE', 'NOT LIKE'}:
        col_name = col_stats[filter.column].attname
        table_name = col_stats[filter.column].tablename
        reg_parts = [str(fp.strip()) for fp in filter.literal.split('%') if len(fp.strip()) > 0]
        if len(reg_parts) > 0:
            regex_patterns[(table_name, col_name)].add(tuple(reg_parts))

    for c in filter.children:
        augment_regex_pattern_per_filter(c, regex_patterns, col_stats)


def augment_regex_patterns(plan, regex_patterns, col_stats):
    filter_column = vars(plan.plan_parameters).get('filter_columns')
    if filter_column is not None:
        augment_regex_pattern_per_filter(filter_column, regex_patterns, col_stats)
    for c in plan.children:
        augment_regex_patterns(c, regex_patterns, col_stats)


def find_sentence_columns(column_statistics, schema, string_statistics, t):
    string_columns = {c for c, v in column_statistics[t].items() if v['datatype'] == 'categorical'}
    string_columns.update([c for c, v in string_statistics[t].items()])
    id_columns = set()
    for t_l, c_l, t_r, c_r in schema.relationships:
        if not (isinstance(c_l, list) or isinstance(c_l, tuple)):
            c_l = [c_l]
            c_r = [c_r]
        if t_l == t:
            id_columns.update(c_l)
        if t_r == t:
            id_columns.update(c_r)
    sentence_columns = string_columns.union(id_columns)
    return sentence_columns, id_columns
