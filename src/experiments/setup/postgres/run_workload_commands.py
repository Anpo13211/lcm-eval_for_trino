import os

from cross_db_benchmark.datasets.datasets import database_list, ext_database_list
from experiments.setup.utils import strip_commands


def gen_run_workload_commands(workload_path="../zero-shot-data/workloads", workload_name=None, database_conn='user=postgres,password=postgres,host=localhost',
                              database='postgres', cap_workload=10000, query_timeout=30, with_indexes=False, repetitions=1,
                              datasets=None, hints=None, db_list=ext_database_list):
    assert workload_name is not None

    cap_workload_cmd = ''
    if cap_workload is not None:
        cap_workload_cmd = f'--cap_workload {cap_workload}'

    index_prefix = ''
    index_cmd = ''
    if with_indexes:
        index_prefix = 'index_'
        index_cmd = '--with_indexes'

    hint_cmd = ''
    if hints is not None:
        hint_cmd = f'--hints {hints}'

    exp_commands = []
    for dataset in db_list:
        if datasets is not None and dataset.db_name not in datasets:
            continue
        exp_commands.append(f"""python3 run_benchmark.py 
          --run_workload
          --query_timeout {query_timeout}
          --source {workload_path}/{dataset.db_name}/{workload_name}.sql
          --target ../zero-shot-data/runs/raw/{dataset.db_name}/{index_prefix}{workload_name}.json
          --database {database}
          --db_name {dataset.db_name}
          --database_conn {database_conn}
          --repetitions_per_query {repetitions}
          {cap_workload_cmd}
          {index_cmd}
          {hint_cmd}
          """)
    # --run_kwargs hardware=[hw_placeholder]
    exp_commands = strip_commands(exp_commands)
    return exp_commands


def gen_evaluation_workload_commands(database_conn='user=postgres,password=postgres,host=localhost',
                                     database='postgres', query_timeout=30, path_prefix='',
                                     explain_workload_variants=False, datasets=None):
    exp_commands = []
    for dataset in ext_database_list:
        if datasets is not None and dataset.db_name not in datasets:
            continue

        eval_path = f'experiments/evaluation_workloads/{dataset.db_name}'

        if not os.path.exists(eval_path):
            continue

        for f in os.listdir(eval_path):
            filename = f.replace('.sql', '')

            db_name = dataset.db_name

            run_cmd = '--run_workload'
            if explain_workload_variants:
                run_cmd = '--explain_workload_variants'

            exp_commands.append(f"""python3 run_benchmark.py 
              {run_cmd}
              --query_timeout {query_timeout}
              --source {eval_path}/{filename}.sql
              --target ../zero-shot-data/runs/raw/{path_prefix}{db_name}/{filename}_[hw_placeholder].json
              --database {database}
              --db_name {db_name}
              --database_conn {database_conn}
              --run_kwargs hardware=[hw_placeholder]
              """)

    exp_commands = strip_commands(exp_commands)
    return exp_commands


def gen_update_eval_workload_commands(workloads=None, database_conn='user=postgres,password=postgres,host=localhost',
                                      database='postgres', cap_workload=10000, query_timeout=30, datasets=None,
                                      wl_base_path=None,
                                      max_rep=0, min_rep=0):
    cap_workload_cmd = ''
    if cap_workload is not None:
        cap_workload_cmd = f'--cap_workload {cap_workload}'

    if wl_base_path is None:
        wl_base_path = 'experiments/evaluation_workloads'

    print("Note: exactly one command can be run per machine with this setup (since the state of the database is changed)")
    exp_commands = []
    for dataset in database_list:
        if datasets is not None and dataset.db_name not in datasets:
            continue

        for current_rep in range(min_rep, max_rep):
            curr_command = update_cmd(cap_workload_cmd, database, database_conn, dataset, current_rep, query_timeout,
                                      workloads, wl_base_path)
            exp_commands.append(curr_command)

    # exp_commands = strip_commands(exp_commands)
    return exp_commands


def update_cmd(cap_workload_cmd, database, database_conn, dataset, max_rep, query_timeout, workloads, wl_base_path):
    curr_commands = []
    for repl in range(max_rep):
        curr_commands.append(
            f"""
                python3 run_benchmark.py
                  --scale_in_db
                  --data_dir ../zero-shot-data/datasets/{dataset.db_name}
                  --dataset {dataset.db_name}
                  --database {database}
                  --db_name {dataset.db_name}
                  --database_conn {database_conn}
                  --no_prev_replications {repl}
                """
        )

    for workload_name in workloads:
        curr_commands.append(f"""python3 run_benchmark.py 
              --run_workload
              --query_timeout {query_timeout}
              --source {wl_base_path}/{dataset.db_name}/{workload_name}.sql
              --target ../zero-shot-data/runs/raw/{dataset.db_name}/{workload_name}_repl_{max_rep}_[hw_placeholder].json
              --database {database}
              --db_name {dataset.db_name}
              --database_conn {database_conn}
              {cap_workload_cmd}
              --run_kwargs hardware=[hw_placeholder]
              """)
    curr_commands = strip_commands(curr_commands)
    curr_command = ' && '.join(curr_commands)
    return curr_command
