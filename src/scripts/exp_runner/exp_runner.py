from pathlib import Path

import argparse
import functools
import os
from random import randint
from typing import List

from dotenv import load_dotenv

from classes.paths import LocalPaths, CloudlabPaths, ClusterPaths
from octopus.experiment import Experiment
from octopus.experiment_task import ExperimentTask
from octopus.node import Node
from octopus.script_preparation import distribute_exp_scripts
from octopus.step import CheckActiveScreens, CheckOutput, Step


class ExpRunner:
    database_conn = f'user=postgres,password=bM2YGRAX*bG_QAilUid§2iD,host=localhost'

    def __init__(self,
                 replicate: bool,
                 node_names: List[str],
                 root_path: Path = CloudlabPaths().root,
                 python_version: str = "3.9"):

        # Read environments variable and arguments
        # Read credentials from environment variables
        load_dotenv()
        self.osf_username = os.getenv('OSF_USERNAME')
        self.osf_password = os.getenv('OSF_PASSWORD')
        self.osf_project = os.getenv('OSF_PROJECT')

        if "cloudlab" in node_names[0]:
            self.ssh_username = os.getenv('CLOUDLAB_SSH_USERNAME')
            self.ssh_passphrase = os.getenv('CLOUDLAB_SSH_PASSPHRASE')
            self.private_key_path = os.getenv('CLOUDLAB_SSH_KEY_PATH')
            self.root_path = CloudlabPaths().root
        else:
            self.ssh_username = os.getenv('CLUSTER_SSH_USERNAME')
            self.ssh_passphrase = os.getenv('CLUSTER_SSH_PASSPHRASE')
            self.private_key_path = os.getenv('CLUSTER_SSH_KEY_PATH')
            self.root_path = ClusterPaths().root
        self.python_version = python_version

        assert self.osf_project is not None, "Please set environment variable OSF_PROJECT"
        assert self.osf_password is not None, "Please set environment variable OSF_PASSWORD"
        assert self.osf_username is not None, "Please set environment variable OSF_USERNAME"
        assert self.ssh_username is not None, "Please set environment variable SSH_USERNAME "
        assert self.ssh_passphrase is not None, "Please set environment variable SSH_PASSPHRASE"
        assert self.private_key_path is not None, "Please set environment variable PRIVATE_KEY_PATH"

        args = self.parse_args()

        self.tasks = args.task
        # Set shell options
        # -e: exit immediately if a command exits with a non-zero status
        # -x: print commands and their arguments as they are executed
        if args.stop_on_error:
            self.prefix = f"set -e; set -x; cd {self.root_path}/src \n"
        else:
            self.prefix = f"set -x; cd {self.root_path}/src \n"

        # Create nodes
        self.nodes = self.create_nodes(node_names, self.ssh_username, self.private_key_path, self.ssh_passphrase)

        # Define script distribution
        self.exp_script_gen = functools.partial(distribute_exp_scripts, shuffle=False, replicate=replicate)
        self.shuffle_seed = randint(0, 8192)

    def run_exp(self,
                node_names: List,
                commands: List[str],
                set_up_steps: List[Step],
                purge_steps: List[Step],
                pickup_steps: List[Step],
                offset: int = 0,
                screens_per_node: int = 1):

        target_nodes = [n for n in self.nodes if n.ssh_hostname in node_names]

        e = Experiment(
            experiment_commands=commands,
            experiment_prefix=self.prefix,
            local_cmd_prefix='',
            nodes=target_nodes,
            setup_steps=set_up_steps,
            purge_steps=purge_steps,
            monitor_steps=[CheckActiveScreens()],
            check_error_steps=[CheckOutput([f'grep -B 5 Error output_screen*'])],
            pickup_steps=pickup_steps,
            exp_folder_path=str(self.root_path),
            exp_script_gen=self.exp_script_gen,
            slurm_num_cpus_per_task=20,
            slurm_num_gpus_per_job=1,
            run_exp_message="Cost Model Evaluation",
            run_exp_expected_runtime="05:00",
            use_run_exp_numa_socket="0",
            #use_run_exp_pin=True,
            python_version=self.python_version,
            screen_or_job_id_offset=offset)

        print(f'Execute Tasks: {self.tasks}')
        for task in self.tasks:
            e.run_task(task)

    def create_nodes(self, names: List[str], username: str, key_path: str, passphrase: str) -> List[Node]:
        # Create nodes
        nodes = []
        nodes += [Node(ssh_hostname=ssh_hostname,
                       ssh_username=username,
                       ssh_private_key=key_path,
                       rsync_private_key=key_path,
                       ssh_passphrase=passphrase,
                       hardware_desc='c8220',
                       ssh_hostkeys=str(LocalPaths().known_hosts),
                       use_slurm="dgx" in ssh_hostname,
                       use_run_exp= self.root_path == ClusterPaths().root and "dgx" not in ssh_hostname,
                       test_connection=False) for ssh_hostname in names]
        return nodes

    @staticmethod
    def parse_args() -> argparse.Namespace:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(f'--task',
                            type=ExperimentTask,
                            choices=list(ExperimentTask),
                            default=[ExperimentTask.PRINT],
                            nargs='+', help='type of tasks which should be executed', required=True)
        parser.add_argument('--stop_on_error',
                            action='store_true',
                            help='stop when error detected',
                            default=True)
        return parser.parse_args()
