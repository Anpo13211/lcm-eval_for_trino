from random import random, shuffle
from typing import List, Optional

from cross_db_benchmark.benchmark_tools.database import ExecutionMode
from classes.paths import LocalPaths, CloudlabPaths
from classes.workloads import EvaluationWorkload, EvalWorkloads
from octopus.script_preparation import strip_commands
from octopus.step import Rsync, KillAllScreens, Remove
from scripts.exp_runner.exp_runner import ExpRunner


def generate_evaluation_workloads(cap_workload: Optional[str],
                                  query_timeout: int,
                                  database_conn: str,
                                  workloads: list[EvaluationWorkload],
                                  repetitions=3,
                                  use_pg_10_hints: bool = False,
                                  mode: ExecutionMode = ExecutionMode.RAW_OUTPUT) -> List[str]:
    exp_commands = []

    cap_workload_cmd = ''
    if cap_workload is not None:
        cap_workload_cmd = f'--cap_workload {cap_workload}'

    for workload in workloads:
        if use_pg_10_hints:
            workload.folder = workload.folder + "_pg_10_hinted"
        sql_path = workload.get_sql_path(CloudlabPaths().evaluation_workloads)
        if mode == ExecutionMode.RAW_OUTPUT:
            target_path = workload.get_workload_path(CloudlabPaths().raw)
        elif mode == ExecutionMode.JSON_OUTPUT:
            target_path = workload.get_workload_path(CloudlabPaths().json)
        else:
            raise ValueError(f'Unknown mode {mode}')
        exp_commands.append(
            f"""python3 run_benchmark.py 
          --run_workload
          --query_timeout {query_timeout}
          --source {sql_path}
          --target {target_path}
          --database postgres
          --db_name {workload.database.db_name}
          --database_conn {database_conn}
          --repetitions_per_query {repetitions} {cap_workload_cmd} 
          --mode {mode}
          --run_explain_only """
        )
    exp_commands = strip_commands(exp_commands)
    return exp_commands


def generate_augment_vector_commands(workloads: list[EvaluationWorkload]) -> List[str]:
    exp_commands = []
    for workload in workloads:
        source_path = workload.get_workload_path(CloudlabPaths().parsed_plans_baseline)
        target_path = workload.get_workload_path(CloudlabPaths().augmented_plans_baseline)
        exp_commands.append(
            f"""python3 baseline.py
          --augment_sample_vectors
          --dataset {workload.database.db_name}
          --data_dir {CloudlabPaths().data}/datasets/{workload.database.db_name}
          --source {source_path}
          --target {target_path} 
          """)

    exp_commands = strip_commands(exp_commands)
    return exp_commands


def shuffle_and_split(lst, n):
    shuffle(lst)
    avg_size = len(lst) / n
    parts = []
    for i in range(n):
        start_index = int(i * avg_size)
        end_index = int((i + 1) * avg_size)
        parts.append(lst[start_index:end_index])
    return parts


if __name__ == '__main__':
    # Read node names from file
    with open(LocalPaths().node_list, 'r') as f:
        node_names = f.read().splitlines()

    runner = ExpRunner(replicate=False, node_names=node_names)

    setup_steps = [Rsync(src=[str(LocalPaths().code)], dest=[CloudlabPaths().root], update=True, put=True),
                   Rsync(src=[str(LocalPaths().workloads)], dest=[CloudlabPaths().data], update=True, put=True),
                   ]

    purge_steps = [KillAllScreens(),
                   #Remove([str(CloudlabPaths().runs)], directory=True)]
                   ]
    use_pg_10_hints = False
    pickup_steps = [Rsync(src=[CloudlabPaths().runs], dest=[LocalPaths().data], put=False, update=True)]

    # Splitting workload into chunks of equal size to be distributed on the workers
    workload_splits = shuffle_and_split(EvalWorkloads.FullJoinOrder.imdb, len(node_names))

    # workloads_to_nodes = [(node_names[0], EvalWorkloads.JoinOrderSelected.imdb)]
    # for node_name, workloads in workloads_to_nodes:

    for node_name, workloads in zip(node_names, workload_splits):
        commands = []
        for mode in [ExecutionMode.JSON_OUTPUT, ExecutionMode.RAW_OUTPUT]:

            commands += generate_evaluation_workloads(cap_workload=None,
                                                      query_timeout=30,
                                                      database_conn=runner.database_conn,
                                                      workloads=workloads,
                                                      repetitions=1,
                                                      use_pg_10_hints=use_pg_10_hints,
                                                      mode=mode)

            if mode == ExecutionMode.RAW_OUTPUT:
                commands += [f'python3 parse_all.py '
                             f'--raw_dir {CloudlabPaths().raw} '
                             f'--parsed_plan_dir {CloudlabPaths().parsed_plans} '
                             f'--parsed_plan_dir_baseline {CloudlabPaths().parsed_plans_baseline} '
                             f'--target_stats_path {CloudlabPaths().code}/experiments/postgres_workload_stats.csv '
                             f'--min_query_ms 0 '
                             f'--max_query_ms 60000000000 '
                             f'--include_zero_card '
                             f'--explain_only']

        runner.run_exp(node_names=[node_name],
                       commands=commands,
                       set_up_steps=setup_steps,
                       purge_steps=purge_steps,
                       pickup_steps=pickup_steps)
