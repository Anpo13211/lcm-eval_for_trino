import numpy as np
import os
import paramiko
import random
import shutil
import string
from colorama import Fore, Style
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from random import seed
from typing import List

from octopus.experiment_task import ExperimentTask
from octopus.node import Node
from octopus.script_preparation import generate_exp_scripts_alternating_cuda_device
from octopus.step import Step, LocalStep, Rsync


class ShareManager(BaseManager):
    pass


def multi_proc_step_executor(steps: List[Step], node: Node):
    """
    Execute all steps on node (function to be executed by spawned processes)
    :param steps: list of steps to execute
    :param node: target node
    """
    for s in steps:
        print('\N{hammer} ' + str(s))
        s.run(node)


class Experiment:
    def __init__(self, setup_steps: List[Step], nodes: List[Node],
                 experiment_prefix: str,
                 experiment_commands,
                 purge_steps: List[Step], monitor_steps: List[Step],
                 check_error_steps: List[Step], pickup_steps: List[Step],
                 exp_folder_path: str,
                 run_exp_message='Experiment', shuffle_seed: int = 0, run_exp_white_list_users=None,
                 use_run_exp_numa_socket: str = None, use_run_exp_exclusive: bool = False,
                 use_run_exp_pin: bool = False,
                 run_exp_expected_runtime: string = None,
                 slurm_num_parallel_tasks_per_job: int = 1, slurm_num_gpus_per_job: int = 0,
                 slurm_num_cpus_per_task: int = 1,
                 local_cmd_prefix='',
                 exp_script_gen=generate_exp_scripts_alternating_cuda_device,
                 python_version: str = '3', screen_or_job_id_offset: int = 0,
                 ssh_path = 'ssh'):
        """
        Initialize experiment
        :param screens_or_jobs_per_node: number of screen or slurm jobs per experiment node
        :param setup_steps: list of setup steps to perform before starting experiments
        :param nodes: list of experiment nodes
        :param experiment_prefix: commands executed before experiment
        :param experiment_commands:
        :param exp_folder_path: path to experiment folder
        :param purge_steps: steps executed after experiment finished for cleaning everything up
        :param monitor_steps: steps related to monitoring
        :param check_error_steps: steps checking for errors
        :param pickup_steps: steps picking up results
        :param run_exp_message: public description message of run_exp job [run_exp]
        :param shuffle_seed: seed used for shuffling
        :param run_exp_white_list_users: list of users where shared experiment execution is possible [run_exp]
        :param use_run_exp_numa_socket: numa socket to use [run_exp]
        :param use_run_exp_exclusive: request exclusive experiment access (if not set: shared) [run_exp]
        :param use_run_exp_pin: pin executable to specified numa nodes (does invoke numactl) [run_exp]
        :param run_exp_expected_runtime: expected runtime of the experiment ([hours:minutes])[run_exp]
        :param slurm_num_parallel_tasks_per_job: number of parallel tasks to be executed in one slurm job [slurm]
        :param slurm_num_gpus_per_job: number of gpus assigned to each slurm job [slurm]
        :param slurm_num_cpus_per_task: number of cpus assigned to each task running in parallel in a slurm job [slurm]
        :param local_cmd_prefix: local command prefix e.g. for executing command on windows in wsl instead of powershell (will be applied to all commands executed locally)
        :param exp_script_gen: function used to batch commands to hardware
        :param python_version: python version used for venv environment
        :param screen_or_job_id_offset: offset to apply when assigning screen/job id
        """
        self.screen_or_job_id_offset = screen_or_job_id_offset
        self.setup_steps = setup_steps
        self.purge_steps = purge_steps
        self.monitor_steps = monitor_steps
        self.check_error_steps = check_error_steps
        self.pickup_steps = pickup_steps
        self.nodes: List[Node] = nodes

        assert os.path.isabs(exp_folder_path), f'exp_folder_path must be an absolute path: {exp_folder_path}'

        for s in [*setup_steps, *purge_steps, *monitor_steps, *check_error_steps, *pickup_steps]:
            s.exp_folder_name = exp_folder_path
            s.local_cmd_prefix = local_cmd_prefix
        self.experiment_prefix = experiment_prefix
        self.experiment_commands = experiment_commands
        assert self.experiment_commands is not None
        self.experiment_scripts = []
        self.use_run_exp_numa_socket = use_run_exp_numa_socket
        self.use_run_exp_exclusive = use_run_exp_exclusive
        self.use_run_exp_pin = use_run_exp_pin
        self.run_exp_expected_runtime = run_exp_expected_runtime
        self.run_exp_message = run_exp_message
        self.exp_folder_name = exp_folder_path
        self.slurm_num_parallel_tasks_per_job = slurm_num_parallel_tasks_per_job
        self.slurm_num_gpus_per_job = slurm_num_gpus_per_job
        self.slurm_num_cpus_per_task = slurm_num_cpus_per_task
        self.shuffle_seed = shuffle_seed
        self.run_exp_white_list_users = run_exp_white_list_users
        if self.run_exp_white_list_users is None:
            self.run_exp_white_list_users = []
        self.exp_script_gen = exp_script_gen
        self.python_version = python_version
        self.ssh_path = ssh_path

    def run_task(self, task: ExperimentTask):
        """
        Run steps of task stage.
        :param task: task stage
        """
        if len(self.nodes) == 0:
            print('Skipping, no free nodes available')
            return
        if task == ExperimentTask.SELECT_RUNEXP_NODES:
            self.select_nodes_on_runexp_cluster()
        elif task == ExperimentTask.SETUP:
            self.run_steps(self.setup_steps)
        elif task == ExperimentTask.DELIVER_SCRIPTS:
            self.deliver_exp_scripts()
        elif task == ExperimentTask.START:
            self.start()
        elif task == ExperimentTask.MONITOR:
            self.run_steps(self.monitor_steps)
        elif task == ExperimentTask.PURGE:
            self.run_steps(self.purge_steps)
        elif task == ExperimentTask.CHECK_ERRORS:
            self.run_steps(self.check_error_steps)
        elif task == ExperimentTask.PICKUP:
            self.run_steps(self.pickup_steps, concurrent=True)
        elif task == ExperimentTask.PRINT:
            self.print()

    def select_nodes_on_runexp_cluster(self):
        """
        Check for available nodes on cluster using run_exp. Nodes which are exclusively used by other users will be discarded.
        Discarded nodes will be removed from self.nodes.
        :return: list of available nodes
        """
        own_experiment_running = False
        available_nodes = []

        # iterate over nodes
        for n in self.nodes:
            try:
                exps = n.run_cmd([f'run_exp'], return_response=True)
                # check for exclusive usage of node
                if any(['EXCLUSIVE' in l for l in exps]):
                    continue

                # check if experiments allowing shared usage are issued by white_list users. If not, discard node
                if any(['Start:' in l and not any([wl in l for wl in self.run_exp_white_list_users]) for l in exps]):
                    continue

                available_nodes.append(n)

                if n.ssh_username in exps:
                    own_experiment_running = True
            except paramiko.ssh_exception.NoValidConnectionsError:
                print(f'Skipping {n.ssh_hostname}')

        print(f"Available Nodes: {', '.join([h.ssh_hostname for h in available_nodes])}")
        self.nodes = available_nodes

        if not own_experiment_running:
            self.nodes = available_nodes
        else:
            print('Own experiment already running')
            self.nodes = []

    def run_steps(self, steps: List[Step], concurrent: bool = True):
        """
        Execute steps on all nodes (steps will be replicated)
        :param steps: list of steps
        :param concurrent: execute steps on multiple nodes at the same time, or step after step / node after node
        """
        if concurrent:
            # make sure to always run all remote steps before local step runs
            remote_steps = []
            for s in steps:
                if isinstance(s, Step):
                    remote_steps.append(s)
                elif isinstance(s, LocalStep):
                    self.run_remote_parallel(remote_steps)
                    remote_steps = []
                    print('\N{hammer} ' + str(s))
                    s.run()
            self.run_remote_parallel(remote_steps)
        else:
            for s in steps:
                if isinstance(s, Step):
                    for n in self.nodes:
                        s.run(n)
                elif isinstance(s, LocalStep):
                    s.run()

    def run_remote_parallel(self, steps: List[Step]):
        """
        Execute list of steps on each node in parallel (steps will be replicated)
        :param steps: list of steps
        """
        assert all([isinstance(s, Step) for s in steps])

        ShareManager.register('sharedData', Node)
        manager = ShareManager()
        manager.start()

        processes = []
        for i, node in enumerate(self.nodes):
            # remove unpickleable ssh_client from object
            node.ssh_client = None

            # spawn scheduler process for each node
            processes.append(Process(target=multi_proc_step_executor, args=(steps, node)))
            processes[-1].start()
        for p in processes:
            p.join()

    def print(self):
        """
        Print experiment commands
        """
        if len(self.experiment_commands) == 0:
            print(Fore.RED + f'No experiment command set' + Style.RESET_ALL)

        for exp_cmd in self.experiment_commands:
            print(exp_cmd)

    def run_exp_args(self, socket: int = 0):
        """
        Assemble arguments for run_exp command
        """

        if self.use_run_exp_numa_socket is None:
            s = socket
        else:
            s = self.use_run_exp_numa_socket

        tmp = f'-m "{self.run_exp_message} (Socket {s})" ' \
              f'--numa {s}'
        if self.use_run_exp_exclusive:
            tmp = tmp + ' --exclusive'
        if self.use_run_exp_pin:
            tmp = tmp + ' --pin'
        if self.run_exp_expected_runtime is not None and self.run_exp_expected_runtime != '':
            tmp = tmp + f' --time {self.run_exp_expected_runtime}'
        return tmp

    def deliver_exp_scripts(self):
        """
        Generate experiment script files and transfer them to experiment nodes.
        """
        print(f'Seeding for delivery with shuffle seed {self.shuffle_seed}')
        np.random.seed(self.shuffle_seed)
        seed(self.shuffle_seed)

        # generate experiment scripts
        self.experiment_scripts = self.exp_script_gen(prefix=self.experiment_prefix,
                                                      experiment_commands=self.experiment_commands,
                                                      nodes=self.nodes,
                                                      slurm_num_gpus_per_job=self.slurm_num_gpus_per_job,
                                                      slurm_cpus_per_task=self.slurm_num_cpus_per_task,
                                                      slurm_num_parallel_tasks_per_job=self.slurm_num_parallel_tasks_per_job)

        # write scripts for each node and screen/job to file and transfer to experiment server
        script_ctr = 0
        for node_no, n in enumerate(self.nodes):
            random_string = ''.join(random.choices(string.ascii_lowercase, k=10))
            os.makedirs(f'run_exp_script_{random_string}', exist_ok=True)
            for i in range(n.num_parallel_screens_or_jobs):
                script = self.experiment_scripts[script_ctr]
                script_ctr += 1

                # replace job id in scripts (used for log file naming)
                script = script.replace(f'[job_id]', f'{i + self.screen_or_job_id_offset}')

                if '[socket]' in script:
                    socket = n.available_sockets[i % len(n.available_sockets)]
                    script = script.replace('[socket]', str(socket))

                with open(f'run_exp_script_{random_string}/run_exp_s{i + self.screen_or_job_id_offset}.sh', "w", newline='\n') as f:
                    f.write(script)

                # make script executable
                if os.name == 'posix':
                    os.system(f"chmod +x run_exp_script_{random_string}/run_exp_s{i + self.screen_or_job_id_offset}.sh")
                elif os.name == 'nt':
                    os.system(f"attrib +x run_exp_script_{random_string}/run_exp_s{i + self.screen_or_job_id_offset}.sh")

            Rsync(src=f'run_exp_script_{random_string}/*.sh', dest='~/.', put=True, ssh_path=self.ssh_path).run(n)
            shutil.rmtree(f'run_exp_script_{random_string}')

    def start(self):
        """
        Start experiment
        """
        # execute experiment scripts for each node and screen
        for n in self.nodes:
            # rye can only install its packages in this folder structure https://github.com/astral-sh/rye/pull/1222
            if "rye" in self.python_version:
                env_cmd = f'source {self.exp_folder_name}/.venv/bin/activate'
            else :
                if n.python_version is None:
                    python_version = self.python_version
                else:
                    python_version = n.python_version

                env_cmd = f'source {self.exp_folder_name}/venv{python_version}/bin/activate'

            for i in range(n.num_parallel_screens_or_jobs):
                offset_screen_id = i + self.screen_or_job_id_offset
                socket = n.available_sockets[i % len(n.available_sockets)]

                if n.use_run_exp:
                    # include run_exp args up front
                    screen_cmd = f'run_exp {self.run_exp_args(socket=socket)} "./run_exp_s{offset_screen_id}.sh > output_screen{offset_screen_id} 2>&1"'

                    n.run_cmd([
                        f'{env_cmd} && screen -d -m -S screen{offset_screen_id} {screen_cmd}'])
                elif n.use_slurm:
                    cmd = f'sbatch ./run_exp_s{offset_screen_id}.sh'
                    n.run_cmd([
                        f'{env_cmd} && '
                        f'rm -f output_job{offset_screen_id}_* && {cmd}'
                    ])
                else:
                    # directly log via screen
                    screen_cmd = f"./run_exp_s{offset_screen_id}.sh 2>&1"
                    # cleanup previous output files and dead screens
                    n.run_cmd([f'{env_cmd} && rm -f output_screen{offset_screen_id} && '
                               f'screen -d -L -Logfile output_screen{offset_screen_id} -m -S screen{offset_screen_id} {screen_cmd}'])
