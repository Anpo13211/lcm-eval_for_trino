from pathlib import Path
from typing import List, Optional

from octopus.script_preparation import strip_commands
from octopus.step import Rsync, KillAllScreens

from cross_db_benchmark.benchmark_tools.database import ExecutionMode
from cross_db_benchmark.datasets.datasets import Database
from classes.paths import LocalPaths, CloudlabPaths
from scripts.exp_runner.exp_runner import ExpRunner


def generate_training_workload_command(cap_workload: Optional[int],
                                       query_timeout: int,
                                       database_conn: str,
                                       db_list: List[Database],
                                       workload: str = "workload_100k_s1",
                                       mode: ExecutionMode = ExecutionMode.JSON_OUTPUT) -> List[str]:
    exp_commands = []

    cap_workload_cmd = ''
    if cap_workload is not None:
        cap_workload_cmd = f'--cap_workload {cap_workload}'

    for database in db_list:
        sql_path = CloudlabPaths().training_workloads / database.db_name / Path(workload + ".sql")
        if mode == ExecutionMode.JSON_OUTPUT:
            target_path = CloudlabPaths().json / database.db_name / workload / Path(workload + ".json")
        else:
            target_path = CloudlabPaths().raw / database.db_name / workload / Path(workload + ".json")

        exp_commands.append(
            f"""python3 run_benchmark.py 
          --run_workload
          --query_timeout {query_timeout}
          --source {sql_path}
          --target {target_path}
          --database postgres
          --db_name {database.db_name}
          --database_conn {database_conn}
          --mode {mode}
          --min_query_ms 100
          --repetitions_per_query 1 {cap_workload_cmd} """)
    exp_commands = strip_commands(exp_commands)
    return exp_commands


if __name__ == '__main__':

    with open(LocalPaths().node_list, 'r') as f:
        node_names = f.read().splitlines()

    node_names_to_db = [(node_names[4], Database("tpc_h_pk"))]

    runner = ExpRunner(replicate=False, node_names=node_names)

    setup_steps = [Rsync(src=[str(LocalPaths().code)], dest=[CloudlabPaths().root], update=True, put=True),
                   Rsync(src=[str(LocalPaths().training_workloads)], dest=[CloudlabPaths().workloads], update=True,
                         put=True)]

    purge_steps = [KillAllScreens()]
                   #Remove([str(CloudlabPaths.runs)], directory=True)]

    pickup_steps = [Rsync(src=[CloudlabPaths().runs], dest=[LocalPaths().data], put=False, update=True)]

    cap_workload = 10000 # regularly 5000, but not for tpc_h_pk - here there were 10k used formerly.
    query_timeout = 30
    wl_name = "workload_100k_s1"

    commands = []
    for node_name, database in node_names_to_db:
        commands += generate_training_workload_command(workload=wl_name,
                                                       cap_workload=cap_workload,
                                                       query_timeout=query_timeout,
                                                       database_conn=runner.database_conn,
                                                       db_list=[database],
                                                       mode=ExecutionMode.JSON_OUTPUT)


        commands += generate_training_workload_command(workload=wl_name,
                                                       cap_workload=cap_workload,
                                                       query_timeout=query_timeout,
                                                       database_conn=runner.database_conn,
                                                       db_list=[database],
                                                       mode=ExecutionMode.RAW_OUTPUT)


        commands += [f'python3 parse_all.py '
                             f'--raw_dir {CloudlabPaths().raw} '
                             f'--parsed_plan_dir {CloudlabPaths().parsed_plans} '
                             f'--parsed_plan_dir_baseline {CloudlabPaths().parsed_plans_baseline} '
                             f'--target_stats_path {CloudlabPaths().code}/experiments/postgres_workload_stats.csv '
                             f'--min_query_ms 0 '
                             f'--max_query_ms 60000000000 '
                             f'--workloads {wl_name} '
                             f'--include_zero_card '
                             f'--cap_queries {cap_workload}']

        commands += [f'python3 baseline.py '
                     f'--augment_sample_vectors '
                     f'--dataset {database.db_name} '
                     f'--data_dir {CloudlabPaths().data}/datasets/{database.db_name} '
                     f'--source {CloudlabPaths().parsed_plans_baseline / database.db_name / wl_name /  f"{wl_name}.json"} '
                     f'--target {CloudlabPaths().augmented_plans_baseline / database.db_name / f"{wl_name}.json"} ']

        runner.run_exp(node_names=[node_name],
                       commands=commands,
                       set_up_steps=setup_steps,
                       purge_steps=purge_steps,
                       pickup_steps=pickup_steps)
