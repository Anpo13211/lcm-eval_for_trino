import collections
import random
import re
from typing import List

from octopus.node import Node


def chunks(lst: List, n_chunks: int) -> List[List]:
    """
    Split list into n chunks
    :param lst: input list
    :param n_chunks: number of chunks
    :return: chunks as list of lists
    """
    chunks = []
    # https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks/29679492
    d, r = divmod(len(lst), n_chunks)
    for i in range(n_chunks):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        chunks.append(lst[si:si + (d + 1 if i < r else d)])

    # sanity check (all chunks must contain same elements as original lst)
    assigned_items = set()
    for chunk in chunks:
        assigned_items.update(chunk)
    assert set(lst) == set(assigned_items)

    return chunks


def create_scripts_for_commands(cmd_lists: List[List[str]], nodes: List[Node],
                                prefix: str, slurm_cpus_per_task: int, slurm_gpus_per_job: int,
                                slurm_num_parallel_tasks_per_job: int):
    cmd_list_offset = 0

    scripts = []

    for n in nodes:
        for job_id in range(n.num_parallel_screens_or_jobs):
            cmds = cmd_lists[cmd_list_offset]

            if n.use_slurm:
                # assemble slurm config
                slurm_config = f'#!/bin/bash\n#SBATCH --ntasks {min(len(cmds), slurm_num_parallel_tasks_per_job)}\n#SBATCH --cpus-per-task {slurm_cpus_per_task}'
                if slurm_gpus_per_job is not None and slurm_gpus_per_job > 0:
                    slurm_config = f'{slurm_config}\n#SBATCH --gpus {slurm_gpus_per_job}'
                slurm_config = f'{slurm_config}\n#SBATCH -e job_[job_id].err\n#SBATCH -o job_[job_id].out'

                if prefix is not None:
                    slurm_config = f'{slurm_config}\n{prefix}'

                # add script commands to config
                offset = 0
                parallel_offset = 0
                while offset < len(cmds):
                    # remove line breaks
                    cmd = cmds[offset].replace('\n', '')
                    slurm_config = f'{slurm_config}\n{cmd} > ~/output_job[job_id]_{offset} 2>&1 &'

                    parallel_offset += 1
                    offset += 1

                    if parallel_offset == slurm_num_parallel_tasks_per_job:
                        slurm_config = f'{slurm_config}\nwait'
                        parallel_offset = 0
                if not slurm_config.endswith('wait'):
                    slurm_config = f'{slurm_config}\nwait'
                scripts.append(slurm_config)
            else:
                # assemble all commands of chunk into one file (will be executed on after another)
                scripts.append(prefix + '\n'.join([cmd.strip() for cmd in cmds]))

            cmd_list_offset += 1

    return scripts


def generate_exp_scripts_alternating_cuda_device(prefix: str, experiment_commands: List[str], nodes: List[Node],
                                                 cuda_devices: List[str] = None,
                                                 replicate: bool = False, slurm_cpus_per_task: int = 1,
                                                 slurm_num_parallel_tasks_per_job: int = 1,
                                                 slurm_num_gpus_per_job: int = 0):
    """
    Generate experiments scripts alternating cuda devices (shuffle automatically)
    :param prefix: script prefix
    :param experiment_commands: list of experiment commands
    :param nodes: list of nodes used in this experiment
    :param screens_per_node: number of parallel screen per node
    :param cuda_devices: list of cuda devices
    :param replicate: flag indicating replication of all commands to all nodes (true) / distributing them on different nodes (false)
    :return: combined experiment commands
    """
    raise Exception(f'implementation not finished')
    random.shuffle(experiment_commands)
    print(f"Having generated {len(experiment_commands)} experiment_commands")

    assert all(['[device_placeholder]' in cmd for cmd in experiment_commands])

    # calculate total number of workers
    n_worker = len(nodes) * screens_per_node

    # create a chunk of commands for each screen of each node
    if replicate:
        cmd_lists = [experiment_commands for _ in range(n_worker)]
    else:
        cmd_lists = chunks(experiment_commands, n_worker)

    # if no cuda devices given, assume cuda0 and cuda1 are existing
    if cuda_devices is None:
        cuda_devices = ['cuda:0', 'cuda:1']

    # combine commands
    return [prefix + '\n'.join(
        [(cmd.replace('[device_placeholder]', f' --device {cuda_devices[i % len(cuda_devices)]}')).strip() for cmd in
         cmds]) for
            i, cmds in enumerate(cmd_lists)]


def generate_cpu_exp_scripts(prefix: str, experiment_commands: List[str], nodes: List[Node],
                             shuffle: bool = True,
                             replicate: bool = False,
                             slurm_cpus_per_task: int = 1,
                             slurm_num_parallel_tasks_per_job: int = 1,
                             slurm_num_gpus_per_job: int = 0
                             ):
    """
    Generate experiment scripts (w/o GPU)
    :param prefix: script prefix
    :param experiment_commands: list of experiment commands
    :param nodes: list of nodes
    :param screens_per_node: parallel screens on each node
    :param shuffle: randomize distribution & execution order of the commands
    :param replicate: flag indicating replication of all commands to all nodes (true) / distributing them on different nodes (false)
    :return: combined experiment commands
    """

    raise Exception(f'implementation not finished')
    # compute number of workers
    n_worker = len(nodes) * screens_per_node
    if shuffle:
        random.shuffle(experiment_commands)
    print(f"Having generated {len(experiment_commands)} experiment_commands")

    assert all(['[device_placeholder]' in cmd for cmd in experiment_commands])

    # create a chunk of commands for each screen of each node
    if replicate:
        cmd_lists = [experiment_commands for _ in range(n_worker)]
    else:
        cmd_lists = chunks(experiment_commands, n_worker)

    # combine commands and remove gpu device placeholder
    return [prefix + '\n'.join(
        [(cmd.replace('[device_placeholder]', '')).strip() for cmd in
         cmds]) for
            i, cmds in enumerate(cmd_lists)]


def distribute_exp_scripts(prefix: str, experiment_commands: List[str], nodes: List[Node],
                           shuffle: bool = True,
                           replicate: bool = False,
                           slurm_cpus_per_task: int = 1,
                           slurm_num_parallel_tasks_per_job: int = 1,
                           slurm_num_gpus_per_job: int = 0
                           ):
    """
    Create experiment scripts distributing commands over all nodes (without adjustments)
    :param prefix: script prefix
    :param experiment_commands: list of experiment commands
    :param nodes: list of nodes
    :param shuffle: randomize distribution & execution order of the commands
    :param replicate: flag indicating replication of all commands to all nodes (true) / distributing them on different nodes (false)
    :param slurm_cpus_per_task: number of cpus assigned to each parallel task
    :param slurm_num_parallel_tasks_per_job: number of tasks executed in parallel in a job
    :param slurm_num_gpus_per_job: number of gpus for each job
    :return: experiment script
    """
    # compute number of workers
    n_worker = sum([n.num_parallel_screens_or_jobs for n in nodes])
    if shuffle:
        random.shuffle(experiment_commands)
    print(f"Having generated {len(experiment_commands)} experiment_commands")

    # create a chunk of commands for each screen of each node
    if replicate:
        cmd_lists = [experiment_commands for _ in range(n_worker)]
    else:
        cmd_lists = chunks(experiment_commands, n_worker)

    return create_scripts_for_commands(cmd_lists=cmd_lists, nodes=nodes,
                                       prefix=prefix,
                                       slurm_cpus_per_task=slurm_cpus_per_task,
                                       slurm_num_parallel_tasks_per_job=slurm_num_parallel_tasks_per_job,
                                       slurm_gpus_per_job=slurm_num_gpus_per_job)


def replicate_commands_per_hardware(prefix: str, experiment_commands: List[str], nodes: List[Node],
                                    shuffle: bool = False) -> List[str]:
    """
    Replicate commands on different hardware types (e.g. cloudlab c220 vs. c820)
    :param prefix: experiment script prefix
    :param experiment_commands: list of experiment commands
    :param nodes: list of nodes
    :param shuffle: randomize distribution & execution order of the commands
    :return: list containing the experiment script for each node
    """


    if shuffle:
        random.shuffle(experiment_commands)
    print(f"Having generated {len(experiment_commands)} experiment_commands")

    # check if all commands can be placed on arbitrary nodes (hw_placeholder ist set in each command)
    assert all(['[hw_placeholder]' in cmd for cmd in experiment_commands])

    # group nodes by hardware description
    hw_nodes = collections.defaultdict(list)
    for node_id, n in enumerate(nodes):
        assert n.num_parallel_screens_or_jobs == 1, "We assume an isolated environment for now."
        hw_nodes[n.hardware_desc].append(node_id)

    # create experiments stubs for each node
    exp_scripts = ['' for _ in nodes]

    # iterate over hardware types and distribute commands on nodes of same hardware type
    for hw, node_ids in hw_nodes.items():
        curr_cmds = [cmd.replace('[hw_placeholder]', hw) for cmd in experiment_commands]

        # create chunk of commands for all nodes of same hardware type
        batched_cmds = chunks(curr_cmds, len(node_ids))

        for node_id, cmd_batch in zip(node_ids, batched_cmds):
            curr_script = prefix + '\n'.join(cmd_batch)
            exp_scripts[node_id] = curr_script

    assert all(cmds != '' for cmds in exp_scripts)

    return exp_scripts


def strip_single_command(cmd: str):
    """
    ???
    :param cmd: command
    :return:
    """
    cmd = cmd.replace('\n', ' ')
    regex = re.compile(r"\s+", re.IGNORECASE)
    cmd = regex.sub(" ", cmd)
    return cmd


def strip_commands(exp_commands: List[str]):
    """
    ???
    :param exp_commands: list of commands
    :return:
    """
    exp_commands = [strip_single_command(cmd) for cmd in exp_commands]
    return exp_commands
