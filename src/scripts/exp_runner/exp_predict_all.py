from pathlib import Path
from typing import List

from octopus.step import Rsync, KillAllScreens

from cross_db_benchmark.datasets.datasets import Database
from classes.classes import ScaledPostgresModelConfig, E2EModelConfig, MSCNModelConfig, QPPNetModelConfig, \
    ZeroShotModelConfig, DACEModelConfig, FlatModelConfig, ModelConfig, DACEModelNoCostsConfig, QPPModelNoCostsConfig, \
    QPPModelActCardsConfig, FlatModelActCardModelConfig, ZeroShotModelActCardConfig, DACEModelActCardConfig, \
    TrainingServers, QueryFormerModelConfig
from classes.paths import LocalPaths, ClusterPaths
from classes.workloads import EvalWorkloads
from scripts.exp_runner.exp_runner import ExpRunner
from collections import defaultdict


def generate_commands(database: Database,
                      model: ModelConfig,
                      workload_paths: List[Path],
                      seeds=None):
    if seeds is None:
        seeds = [0]

    # Initialize a dictionary to store the workload paths grouped by folder
    workloads_by_folder = defaultdict(list)

    # Iterate over the workload paths
    for path in workload_paths:
        # Get the folder name from the path
        folder = path.parent.name
        # Add the path to the list of workloads in the corresponding folder
        workloads_by_folder[folder].append(str(path))

    commands = []

    for seed in seeds:
        for folder, workload_paths in workloads_by_folder.items():
            workloads = " ".join([str(path) for path in workload_paths])
            command = (f"python3 main.py "
                       f"--mode predict "
                       f"--model_type {model.name.NAME} "
                       f"--test_workload_runs {workloads} "
                       f"--statistics_file {model.get_statistics(ClusterPaths(), database)} "
                       f"--model_dir {model.get_model_dir(ClusterPaths(), database)} "
                       f"--target_dir {model.get_eval_dir(ClusterPaths(), database) / folder} "
                       f"--seed {seed} ")

            if model.column_statistics:
                command += f" --column_statistics {model.get_column_stats(database)} "

            if model.word_embeddings:
                command += f" --word_embeddings {model.get_word_embeddings(source_path=ClusterPaths(), database=database)} "

            if model.hyperparameter:
                command += f" --hyperparameter_path {model.hyperparameter} "

            commands.append(command)
    return commands


if __name__ == '__main__':
    setup_steps = [Rsync(src=[str(LocalPaths().code)], dest=[ClusterPaths().root], update=True, put=True),
                   Rsync(src=[str(LocalPaths().workloads)], dest=[ClusterPaths().data], update=True, put=True),
                   Rsync(src=[str(LocalPaths().runs)], dest=[ClusterPaths().data], update=True, put=True),
                   ]
    purge_steps = [KillAllScreens()]
    pickup_steps = [Rsync(src=[ClusterPaths().evaluation], dest=[LocalPaths().data], put=False, update=True)]

    model_configs = [
        #FlatModelConfig(),
        #ScaledPostgresModelConfig(),
        MSCNModelConfig(),
        #E2EModelConfig(),
        #QPPNetModelConfig(),
        #ZeroShotModelConfig(),
        #DACEModelConfig(),
        #QueryFormerModelConfig()
    ]

    no_costs_model_configs = [
        DACEModelNoCostsConfig(),
        QPPModelNoCostsConfig()
    ]

    act_card_model_configs = [
        FlatModelActCardModelConfig(),
        QPPModelActCardsConfig(),
        ZeroShotModelActCardConfig(),
        DACEModelActCardConfig()
    ]

    node = TrainingServers.NODE04
    seeds = [0, 1, 2]

    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       root_path=ClusterPaths().root,
                       python_version=node["python"])
    commands = []
    wls_to_dbs = [
        (Database("imdb"), EvalWorkloads.ScanCostsPercentile.imdb),
        (Database("baseball"), EvalWorkloads.ScanCostsPercentile.baseball),
    ]

    for database, target_workloads in wls_to_dbs:
        for model in model_configs:
            workload_paths = [wl.get_workload_path(model.get_data_base_dir()) for wl in target_workloads]
            commands += generate_commands(database, model, workload_paths, seeds)

    runner.run_exp(node_names=[node["hostname"]],
                   commands=commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=pickup_steps,
                   screens_per_node=len(wls_to_dbs),
                   offset=0)
