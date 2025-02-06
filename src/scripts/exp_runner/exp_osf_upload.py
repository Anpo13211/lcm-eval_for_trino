import os

from octopus.step import Rsync, KillAllScreens
from classes.classes import TrainingServers
from classes.paths import LocalPaths, ClusterPaths

from scripts.exp_runner.exp_runner import ExpRunner


# This script is used to upload the submission data to the OSF project

if __name__ == '__main__':
    setup_steps = [Rsync(src=[str(LocalPaths().workloads)], dest=[ClusterPaths().data], update=True, put=True),
                   Rsync(src=[str(LocalPaths().runs)], dest=[ClusterPaths().data], update=True, put=True)]

    purge_steps = [KillAllScreens()]
    pickup_steps = [Rsync(src=[ClusterPaths().evaluation], dest=[LocalPaths().data], put=False, update=True)]

    node = TrainingServers().NODE03
    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       root_path=ClusterPaths().root,
                       python_version=node["python"])

    username = os.getenv('OSF_USERNAME')
    project = os.getenv('OSF_SUBMISSION_PROJECT')
    password = os.getenv('OSF_PASSWORD')

    source_paths = [ClusterPaths().evaluation, ClusterPaths().models, ClusterPaths().runs]
    commands = []
    for source in source_paths:
        target = str(source).replace(str(ClusterPaths().data), "")
        commands.append(f"OSF_PASSWORD={password} osf -p {project} -u {username} upload -r {source}/ {target} ")

    runner.run_exp(node_names=[node["hostname"]],
                   commands=commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=pickup_steps,
                   screens_per_node=1,
                   offset=0)
