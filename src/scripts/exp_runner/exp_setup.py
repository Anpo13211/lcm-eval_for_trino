from typing import List

from octopus.step import Cmd, KillAllScreens, SetupVenv, LocalCmd, \
    Step, Rsync

from classes.paths import LocalPaths, CloudlabPaths
from scripts.exp_runner.exp_runner import ExpRunner

if __name__ == '__main__':
    """ 
    This script sets-up various machines at cloudlab.
    Currently this needs to be executed twice as the target machines needs to be rebooted in between.
    """
    # Read nodenames from file
    with open(LocalPaths().node_list, 'r') as f:
        node_names = f.read().splitlines()

    runner = ExpRunner(replicate=True, node_names=node_names, python_version="3.9")
    postgres_version = "16"

    commands = [
        f'python3.9 setup.py '
        f'--data_dir ~/cost-eval/data/datasets/ '
        f'--osf_username {runner.osf_username} '
        f'--osf_password {runner.osf_password} '
        f'--osf_project {runner.osf_project} ' 
        f'--database_conn {runner.database_conn}'
    ]

    setup_steps: List[Step] = [
        # Download server keys
        LocalCmd(cmd=' && '.join([f'ssh-keyscan -H {node} ' f'>> {LocalPaths().known_hosts}' for node in node_names])),
        # Rsync the current repository
        Rsync(src=[str(LocalPaths().code)], dest=[CloudlabPaths().root], update=True, put=True),
        Rsync(src=[str(LocalPaths().requirements)], dest=[CloudlabPaths().root], update=True, put=True),
        # Resize the disk if it is a cloudlab instance
        Cmd(cmd=f'{CloudlabPaths().code}/scripts/postgres_installation/resize_partition.sh'),
        # Continue resize the disk if it is a cloudlab instance
        Cmd(cmd=f'{CloudlabPaths().code}/scripts/postgres_installation/resize_partition_cont.sh'),
        # Install tools
        Cmd(cmd=f'{CloudlabPaths().code}/scripts/postgres_installation/install_tools.sh'),
        # Install postgres
        Cmd(cmd=f'{CloudlabPaths().code}/scripts/postgres_installation/install_postgres_{postgres_version}.sh'),
        # Setup virtual environment
        SetupVenv(use_requirements_txt=True,
                  requirements_txt_filename=f'{CloudlabPaths().root}/requirements/requirements_cloudlab.txt',
                  force=False,
                  python_cmd=f'python{runner.python_version}',
                  python_version=runner.python_version)
    ]

    purge_steps: List[Step] = [KillAllScreens(),
                               Cmd(cmd='sudo service postgresql restart')]

    runner.run_exp(node_names=node_names,
                   commands=commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=[])
