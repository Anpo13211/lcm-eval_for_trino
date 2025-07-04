import argparse
import functools
import os
from random import randint

from octopus.experiment import Experiment
from octopus.experiment_task import ExperimentTask
from octopus.node import Node
from octopus.script_preparation import distribute_exp_scripts
from octopus.step import RsyncCurrentRepo, Cmd, SetupVenv, CreateScript, Rsync, KillAllScreens, \
    CheckActiveScreens, CheckOutput, ScreenCmd, LocalMethodCall

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--task', type=ExperimentTask, choices=list(ExperimentTask), default=[ExperimentTask.PRINT],
                        nargs='+', help='type of tasks which should be executed')
    parser.add_argument('--stop_on_error', action='store_true', help='stop when error detected')

    args = parser.parse_args()

    # define script distribution
    exp_script_gen = functools.partial(distribute_exp_scripts, shuffle=True)
    shuffle_seed = randint(0, 8192)

    # define commands
    experiment_commands = [
        'python3 main.py --device cuda:[socket]'
    ]

    # define nodes
    nodes = []

    # read credentials from environment variables
    ssh_username = os.getenv('SSH_USERNAME')
    ssh_passphrase = os.getenv('SSH_PASSPHRASE')

    assert ssh_username is not None, "Please set environment variable SSH_USERNAME " \
                                     "(alternatively you can also load phrsase in your " \
                                     "ssh-agent and deactivate this check)"
    assert ssh_passphrase is not None, "Please set environment variable SSH_PASSPHRASE"

    # cloudlab nodes setup
    print("Test connection to nodes...")
    nodes += [Node(
        ssh_hostname=ssh_hostname,
        ssh_username=ssh_username,
        ssh_private_key='<path/to/key>',
        ssh_passphrase=ssh_passphrase,
        hardware_desc='<hardware_desc>',
        ssh_hostkeys='<path/to/key>',
        use_run_exp=True,
        use_slurm=False,
        python_version='3.10',
        available_sockets=[0, 1],
        tags={'<hw_tag>'}) for ssh_hostname in [
        '<host.name>'
    ]]
    root_folder = '~/my-project-name'

    # set shell options
    # -e: exit immediately if a command exits with a non-zero status
    # -x: print commands and their arguments as they are executed
    if args.stop_on_error:
        prefix = f"set -e; set -x; cd {root_folder} \n"
    else:
        prefix = f"set -x; cd {root_folder} \n"

    e = Experiment(
        screen_or_job_id_offset=0,
        experiment_commands=experiment_commands,
        experiment_prefix=prefix,
        nodes=nodes,
        setup_steps=[
            RsyncCurrentRepo(),
            # execute a .sh script from the current repo
            Cmd(cmd='path/to/script.sh>', tags={'<hw_tag>'}),
            SetupVenv(use_requirements_txt=True, force=False, requirements_txt_filename='other_requirements.txt'),
            # create script file from list of commands
            CreateScript(scriptname='setup.sh',
                         script_cmd=f'cmd1\ncmd2\ncmd3\n'),
            # transfer script to destination
            Rsync(src=[f'setup.sh'], dest=[f'.'], put=True, update=True),
            LocalMethodCall(os.remove, {'path': 'setup.sh'}),
            ScreenCmd(cmd=['./setup.sh'], logfile='setup.log', cleanup_logfile=True),
        ],
        purge_steps=[
            KillAllScreens(),
        ],
        monitor_steps=[
            CheckActiveScreens(),
        ],
        check_error_steps=[
            CheckOutput([f'grep -B 5 Error output_screen*'])
        ],
        pickup_steps=[
            Rsync(src=[f'~/{root_folder}/artifacts'], dest=[f'<local_path>/{root_folder}/artifacts/'],
                  put=False, update=True),
        ],
        exp_folder_path=root_folder,
        run_exp_message='this is a message',
        run_exp_expected_runtime='10:00', # hh:mm
        use_run_exp_pin=True,
        exp_script_gen=exp_script_gen,
    )
    print(f'Execute Tasks: {args.task}')
    for task in args.task:
        e.run_task(task)
