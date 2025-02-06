import ast
import os
from typing import List

from octopus.step import Rsync, KillAllScreens

from cross_db_benchmark.datasets.datasets import Database
from classes.classes import ModelConfig, ModelType, \
    DACEModelConfig, DACEModelNoCostsConfig, QPPModelNoCostsConfig, FlatModelActCardModelConfig, QPPModelActCardsConfig, \
    DACEModelActCardConfig, QPPNetModelConfig, ZeroShotModelActCardConfig, QueryFormerModelConfig, TrainingServers, \
    ScaledPostgresModelConfig, ZeroShotModelConfig, FlatModelConfig, E2EModelConfig, MSCNModelConfig
from classes.paths import LocalPaths, ClusterPaths
from scripts.exp_runner.exp_runner import ExpRunner


def generate_commands(databases: List[Database], seeds: list[int], python_version: str,
                      model_list: list[ModelConfig] = ModelConfig) -> List[str]:

    commands = []
    for database in databases:
        for model in model_list:
            retraining_workload_runs = [str(model.get_data_base_dir() / "imdb" / "index_retraining" / "index_retraining.json"),
                                        str(model.get_data_base_dir() / "imdb" / "seq_retraining" / "seq_retraining.json")]
            retraining_workload_runs = " ".join(retraining_workload_runs)
            test_workload_runs = model.get_test_workloads(database)
            statistics_path = model.get_statistics(source_path=ClusterPaths(), database=database)
            if model == QPPNetModelConfig():
                statistics_path = ClusterPaths().json / "imdb/feature_stats_retraining.json"
            for seed in seeds:
                command = (f"python{python_version} main.py "
                           f"--mode retrain "
                           f"--wandb_project cost-eval-retrain "
                           f"--model_type {model.name.NAME} "
                           f"--device {model.device} "
                           f"--model_dir {model.get_model_dir(ClusterPaths(), database, True)} "
                           f"--target_dir {model.get_eval_dir(ClusterPaths(), database, True)} "
                           f"--workload_runs {retraining_workload_runs} "
                           f"--statistics_file {statistics_path} "
                           f"--seed {seed} "
                           f"--wandb_name {model.name.NAME}/{database.db_name}/{seed}")

                if model.type == ModelType.WL_AGNOSTIC:
                    command += f" --test_workload_runs {test_workload_runs} "

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
                   Rsync(src=[str(LocalPaths().runs)], dest=[ClusterPaths().data], update=True, put=True)]

    purge_steps = [KillAllScreens()]
    pickup_steps = [Rsync(src=[ClusterPaths().runs], dest=[LocalPaths().data], put=False, update=True)]

    databases = [Database('imdb')]
    seeds = [0, 1, 2]

    node = TrainingServers.NODE04
    # model = ScaledPostgresModelConfig()
    # model = FlatModelConfig()
    # model = E2EModelConfig()
    # model = DACEModelConfig()
    # model = QueryFormerModelConfig()
    model = ZeroShotModelConfig()

    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       root_path=ClusterPaths().root,
                       python_version=node["python"])

    eval_commands = []
    for database in databases:
        eval_commands += generate_commands(databases=[database],
                                           model_list=[model],
                                           python_version=node["python"],
                                           seeds=seeds)

    runner.run_exp(node_names=[node["hostname"]],
                   commands=eval_commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=pickup_steps,
                   screens_per_node=len(databases),
                   offset=0)
