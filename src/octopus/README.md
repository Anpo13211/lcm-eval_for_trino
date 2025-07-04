# octopus

<i>An easy and robust single node experiment runner</i>

## How to use

To use octopus for execution of experiments, you first need to define the server/nodes available for the experiments.
Define object for each available server with information on how to connect to the server.

```
node=Node(
    ssh_hostname=<ssh_hostname>,
    ssh_username=<ssh_username>,
    ssh_private_key=<path_to_private_key>,
    hardware_desc=<name_of_hardware>, #e.g. 'c8220' or 'c220g2'
    ssh_hostkeys=<path_to_known_hosts>,
    tags={<hardware_tags>}, #e.g. 'cloudlab' or 'dmlab'
    use_run_exp=True,
    use_slurm=False,
    python_version='3.10',
    num_parallel_screens_or_jobs=2, # if None - number of available sockets will be used
    available_sockets=[0,1],
    ...
)
```

Afterwards define your experiment:

```
e=Experiment(
    screens_per_node=<int>,
    experiment_commands=[<command>,<command>],
    experiment_prefix=<str>,
    ... (see below)
)
```

The experiment commands are the commands that should be executed on the server. 
The experiment prefix is a command that should be executed before the experiment commands.
The experiment command can contain the following placeholders:

- `[job_id]`: the job id of the experiment
- `[socket]`: the socket number used for the cmd e.g. `device=cuda:[socket]`

Finally, execute the tasks of your experiment in any order you want:

```
task=ExperimentTask.SETUP
e.run_task(task)
```

Originally the lifecycle is intended to be:

1. SETUP (execute setup steps) - <b><i>custom steps</i></b>
2. DELIVER_SCRIPTS (create experiment scripts from defined exp steps)
3. START (start experiments)
4. MONITOR (monitor execution) - <b><i>custom steps</i></b>
5. PURGE (kill screens and restart postgres)
6. CHECK_ERRORS (check for errors during execution) - <b><i>custom steps</i></b>
7. PICKUP (download results) - <b><i>custom steps</i></b>

Further Lifecycle stage: PRINT (prints experiment commands)

A sample experiment configuration can be found [here (example_exp.py)](example_exp.py).

## Experiment Arguments

| Argument                         | Description                                                                          | Sample Vale                                                                                       |
|----------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| screen_or_job_id_offset          | Offset for screen or job id                                                          | `0`                                                                                               |
| experiment_commands              | Experiment commands which shuold be executed                                         | `["python3 run_benchmark.py"]`                                                                    |
| experiment_prefix                | Commands executed before starting experiment (i.e. navigate to folder)               | `cd zero-shot-learned-db\n`                                                                       |
| nodes                            | List of available nodes                                                              | `[node1,node2]`                                                                                   |
| setup_steps                      | List of steps used to transfer repo, data and prepare infrastructure for experiments | `[RsyncCurrentRepo(),SetupVenv(),...]`                                                            |
| purge_steps                      | List of steps cleaning up after experiment execution                                 | `[KillAllScreens(),Cmd(cmd='sudo service postgresql restart')]`                                   |
| monitor_steps                    | List of steps monitoring experiment execution                                        | `[CheckActiveScreens()]`                                                                          |
| check_error_steps                | steps checking for errors during experiment execution                                | `[CheckOutput([f'grep -B 5 Error output_screen*'])]`                                              |
| pickup_steps                     | steps picking up results                                                             | `[Rsync(src=[f'zero-shot-data/runs/'],dest=[f'../zero-shot-data/runs/'],put=False, update=True)]` |
| exp_folder_name                  | name of the experiment folder (repo name)                                            | `'zero-shot-learned-db'`                                                                          |
| use_run_exp                      | use run_exp job scheduling for execution                                             | `False`                                                                                           |
| run_exp_message                  | message to display in run_exp jobs                                                   | `Zero-Shot experiments`                                                                           |
| use_run_exp_numa_socket          | numa socket to use in run_exp script                                                 | `1`                                                                                               |
| run_exp_white_list_users         | whitelist of user where shared execution is allowed                                  | `['cbinnig']`                                                                                     |
| use_run_exp_exclusive            | Request exclusive node access                                                        | `True`                                                                                            |
| use_run_exp_pin                  | Pin to specified num nodes (does invoke numactl)                                     | `True`                                                                                            |
| run_exp_expected_time            | Expeted experiment runtime (run_exp) [hours:minutes]                                 | `30:13`                                                                                           |
| slurm_num_parallel_tasks_per_job | Number of tasks to be executed at the same time in one slurm job                     | `2`                                                                                               |
| slurm_num_gpus_per_job           | Number of GPUs to be requested by each slurm job                                     | `0`                                                                                               |
| slurm_num_cpus_per_job           | Number of CPUs assigned to each slurm job                                            | `4`                                                                                               |
| exp_script_gen                   | function used to batch commands to hardware                                          | `functools.partial(replicate_commands_per_hardware, shuffle=True)`                                |
| shuffle_seed                     | seed used for shuffling                                                              | `1`                                                                                               |

## Available Steps

### Remote Steps

| Steps                  | Description                                                                                                        | Arguments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CloneRepo`            | Step cloning a repository                                                                                          | `repo_url:str`(URL to repository)<br/>`recurse_submodules:bool` (load git submodules)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `SetupVenv`            | Step setting up venv python environment                                                                            | `requirements:List[str]=None`(list of python librarires to install)<br/>`force:bool=False`(force recreation of venv env.)<br/>`use_requirements_txt:bool`(install dependencies described in requirements.txt)<br/>`python_cmd:str` (python command to use e.g. 'python3' or 'python')                                                                                                                                                                                                                                                                                                      |
| `PythonCmd`            | Step executing a python command                                                                                    | `cmd:str`(python command)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `Cmd`                  | Step executing a single command or list of <br>commands on remote server                                           | `cmd`(single command or list of commands)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `ScreenCmd`            | Step executing command in screen                                                                                   | `cmd`(single command or list of commands)<br/>`logfile:str`(path to logfile)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `SCP`                  | Step transfering file via scp                                                                                      | `src:str`(source file)<br/>`dest:str`(destination file)<br/>`put:bool`(upload or download file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `Remove`               | Step removing files                                                                                                | `files:List[str]`(list of files/folders to remove)<br/>`directory:bool`(remove also folders)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `RemoveRecursively`    | Step deleting all files with given name<br> in directory recursively                                               | `root_dir:str`(directory in which to search for occurence of filename)<br/>`name:str`(name of the file)<br/>`search_for_directory:bool`(search directory with this name)                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `KillAllScreens`       | Step killing all screens on nodes                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `KillAllSlurmJobs`     | Step killing all slurm jobs                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `DownloadGDriveFolder` | Step downloading files from gdrive                                                                                 | `gfile_id:str`(id of gdrive file)<br/>`target_dir:str`(path to target directory)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `RsyncCurrentRepo`     | Step transferring local experiment <br>repository to node via rsync                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `Rsync`                | Step transferring folder via rsync                                                                                 | `src`(source path or list of source paths)<br/>`dest`(target path or list of target paths)<br/>`put:bool`(upload or donwload files)<br/>`delete:bool`(delete extraneous files from dest dirs)<br/>`update:bool`(skip files that are newer on the receiver)<br/>`ignore_existing:bool`(skip updating files that exist on receiver)<br/>`skip:bool`(skip this step)<br/>`ìnclude`(single file/folder or list of files which should not be exceluded)                                                                                                                                         |
| `CheckActiveScreens`   | Step checking the number of active screens on node                                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `CheckOutput`          | Step executing list of commands and print the output                                                               | `commands:List[str]`(list of commands)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `CountGeneratedFiles`  | Step printing n files in folder                                                                                    | `path`(file path or list of paths)<br/>`list_n_files`(number of files in path to print)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `SetupRYEVenv`         | Step for setting up a Python virtual environment (venv). To use this venv, set the global Python version to `rye`. | `requirements: List[str]` (List of required Python libraries to be installed)<br/>`force: bool` (Force recreation of the virtual environment)<br/>`no_lock: bool` (Do not update the requirements.lock file)<br/>`no_dev: bool` (Install only the non-development packages)<br/>`rye_add: List[str]` (List of additional packages to be installed on the server using the command `rye add {item}`. If you do not add `--no-sync` to your package installation string, it will automatically sync your repo, which can cause problems.)<br/>`rye_version` (Version of Rye to be used)<br/> |                                                                                                                                                                                                                                                                                                                                                           |

### Local Steps

| Steps             | Description                                                               | Arguments                                                                                                   |
|-------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `LocalCopy`       | Step copying files locally                                                | `src:List[str]`(source paths)<br/>`dest:List[str]`(dest paths)<br/>`recursive:bool=False`(copy recursively) |
| `CreateScript`    | Step writing command to script file                                       | `scriptname:str`(name of the scriptfile)<br/>`script_cmd:str`(command)                                      |
| `LocalCmd`        | Step executing command locally                                            | `cmd:str`(command)                                                                                          |
| `LocalMethodCall` | Step invoking python function locally                                     | `func`<br/>`args`                                                                                           |
| `CombineCSVFiles` | Step combining results of multiple csv <br>files in single file (locally) | `source_files`(single source file or list of files)<br/>`target_file`(single target file or list of files)  |







