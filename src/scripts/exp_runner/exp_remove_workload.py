import shutil
from typing import List

from octopus.step import Step, Remove

from cross_db_benchmark.datasets.datasets import Database
from classes.classes import MODEL_CONFIGS, TrainingServers
from classes.paths import LocalPaths, ClusterPaths
from scripts.exp_runner.exp_runner import ExpRunner

if __name__ == '__main__':
    """ This script removes a certain workload from localhost and cloudlab machines"""

    node = TrainingServers.NODE00

    runner = ExpRunner(replicate=True,
                       node_names=[node['hostname']],
                       python_version=node['python'],
                       root_path=ClusterPaths().root)

    databases = [Database("tpc-h"), Database("imdb"), Database("baseball")]
    study = "agg_range_filter"

    for paths in [LocalPaths(), ClusterPaths()]:
        paths_to_remove = []
        for database in databases:
            paths_to_remove += [
                paths.augmented_plans_baseline / database.db_name / study,
                paths.parsed_plans / database.db_name / study,
                paths.parsed_plans_baseline / database.db_name / study,
                paths.json / database.db_name / study,
                paths.raw / database.db_name / study]

            for model in MODEL_CONFIGS:
                paths_to_remove.append(paths.evaluation / model.name.NAME / database.db_name / study)
        print(paths_to_remove)

        if isinstance(paths, LocalPaths):
            print("Removing from localhost")
            for path in paths_to_remove:
                shutil.rmtree(path, ignore_errors=True)

        if isinstance(paths, ClusterPaths):
            print("Removing from cluster")
            setup_steps: List[Step] = []
            for path in paths_to_remove:
                setup_steps.append(Remove(files=str(path), directory=True))
                #setup_steps: List[Step] = [Remove(files=" ".join([str(p) for p in paths_to_remove]), directory=True)]
            runner.run_exp(node_names=[node['hostname']],
                           commands=[],
                           set_up_steps=setup_steps,
                           purge_steps=[],
                           pickup_steps=[])
