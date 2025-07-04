import glob
import os
import re
from typing import List

import pandas as pd

from octopus.node import Node


class Step:
    """
    Abstract class representing a step in an experiment (i.e. clone repo, exec command, ...)
    """

    def __init__(self, tags: List[str] = None, local_cmd_prefix: str = ""):
        """
        hardware tags on which this step should be executed (e.g. only execute resize_partition on cloudlab)
        :param tags: list of hardware tags (str)
        :param local_cmd_prefix: local command prefix e.g. for executing command on windows in wsl instead of powershell
        """
        self.tags = tags
        self.local_cmd_prefix = local_cmd_prefix

    def run(self, node):
        # if node does not have required tags, skip
        execute = True
        if self.tags is not None:
            if len(node.tags.intersection(self.tags)) == 0:
                execute = False
                print("Skipping since tags do not match")

        if execute:
            self._run(node)

    def _run(self, node):
        raise NotImplementedError


class CloneRepo(Step):
    """
    Step cloning a repository
    """

    def __init__(self, repo_url: str, recurse_submodules: bool, **kwargs):
        """
        Setup CloneRepo step
        :param repo_url: url from repository
        :param recurse_submodules: boolean flag indicating if git submodules will be recursively loaded or not (see git clone --recurse-submodules documentation)
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.repo_url = repo_url
        self.recurse_submodules = recurse_submodules
        self.exp_folder_name = None

    def _run(self, node: Node):
        """
        Execute step on node
        :param node: experiment node
        """
        # check if repository is already existing on server
        exists = node.file_exists(self.exp_folder_name.exp_folder_name)

        if exists:
            print(f"Skipping CloneRepo {self.exp_folder_name.exp_folder_name}")
            return

        # clone repo on server
        print(f"Cloning repository {self.repo_url} to {node.ssh_hostname}")
        node.run_cmd([f"git clone {self.repo_url} {'--recurse-submodules' if self.recurse_submodules else ''}"])

    def __str__(self):
        return f'CloneRepo({self.repo_url})'


class SetupVenv(Step):
    """
    Step setting up venv python environment
    """

    def __init__(self, requirements: List[str] = None, force: bool = False, use_requirements_txt: bool = False,
                 requirements_txt_filename: str = 'requirements.txt',
                 python_cmd: str = None, python_version: str = None, **kwargs):
        """
        Initialize SetupVenv step
        :param requirements: list of required python libraries which should be installed
        :param force: force recreation of venv environment
        :param use_requirements_txt: install requirements specified in requirements.txt
        :param python_cmd: python command to use (e.g. 'python3' or 'python')
        :param python_version:python version number (e.g. '3' or '3.6') - used to create separate venv files. If python version is none, take defalt python version from node definition
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.python_cmd = python_cmd
        self.exp_folder_name = None
        self.requirements = requirements
        self.force = force
        self.use_requirements_txt = use_requirements_txt
        self.requirements_txt_filename = requirements_txt_filename
        self.python_version = python_version

    def _run(self, node: Node):
        """
        Execute step on node
        :param node: experiment node
        """

        if self.python_version is None:
            assert node.python_version is not None, "Please set python version in node or in step"
            python_version = node.python_version
        else:
            python_version = self.python_version

        if self.python_cmd is None:
            python_cmd = f'python{python_version}'
        else:
            python_cmd = self.python_cmd

        if self.force:
            # remove existing venv folder
            node.run_cmd([f'rm -rf {self.exp_folder_name}/venv{python_version}'])

        # check if venv folder already exists on server
        exists = node.file_exists(self.exp_folder_name + f'/venv{python_version}')

        # navigate to experiment folder
        cmds = [f'cd {self.exp_folder_name}']
        action = 'Updating'
        if not exists:
            action = 'Creating'

            # add create venv command to command queue
            cmds.append(f'{python_cmd} -m venv venv{python_version}')

        print(f"{action} venv {self.exp_folder_name} at {node.ssh_hostname}")

        # activate environment and update pip
        cmds += [f'source venv{python_version}/bin/activate', 'pip3 install --upgrade setuptools',
                 'pip3 install --upgrade pip']

        # install requirements defined in requirements.txt
        if self.use_requirements_txt:
            cmds.append(f'pip3 install -r {self.requirements_txt_filename}')

        # install manually added requirements
        if self.requirements is not None:
            for r in self.requirements:
                cmds.append(f'pip3 install {r}')
        # execute commands
        node.run_cmd(cmds)

    def __str__(self):
        return f'SetupVenv({self.requirements})'


class PythonCmd(Step):
    """
    Step executing a python command
    """

    def __init__(self, cmd: str, python_version: str = '3', **kwargs):
        """
        Initialize PythonCmd step
        :param cmd: python command
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None
        self.cmd = cmd
        self.python_version = python_version

    def _run(self, node: Node):
        """
        Execute python command step
        :param node: experiment node
        """
        print(f"Runnning python command {self.exp_folder_name} at {node.ssh_hostname}")

        # navigate to experiment folder, create venv environment, activate environment and execute python command
        cmd = f"cd {self.exp_folder_name} && python3 -m venv venv{self.python_version} && source venv{self.python_version}/bin/activate && {self.cmd}"

        # execute commands
        node.run_cmd([cmd])

    def __str__(self):
        return f'PythonCmd({self.cmd})'


class Cmd(Step):
    """
    Step executing a single command or list of commands on remote server
    """

    def __init__(self, cmd: str, **kwargs):
        """
        Initialize cmd step
        :param cmd: single command or list of commands
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None
        if not isinstance(cmd, list):
            cmd = [cmd]
        self.cmd = cmd

    def _run(self, node: Node):
        """
        Run cmd step
        :param node: experiment node
        """
        node.run_cmd(self.cmd)

    def __str__(self):
        return f'Cmd({self.cmd})'


class ScreenCmd(Step):
    """
    Step executing command in screen
    """

    def __init__(self, cmd, logfile: str = None, cleanup_logfile: bool = False, **kwargs):
        """
        Initialize ScreenCmd step
        :param cmd: single command or list of commands
        :param logfile: path to logfile
        :param cleanup_logfile: cleanup the logfile before writing to it - this will remove all previously collected data
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None
        self.logfile = logfile
        self.cleanup_logfile = cleanup_logfile
        if not isinstance(cmd, list):
            cmd = [cmd]
        self.cmd = cmd

    def _run(self, node: Node):
        """
        Run ScreenCmd step
        :param node: experiment node
        """
        node.run_screen_cmd(self.cmd, logfile=self.logfile, recreate_logfile=self.cleanup_logfile)

    def __str__(self):
        return f'ScreenCmd({self.cmd})'


class SCP(Step):
    """
    Step transfering file via scp
    """

    def __init__(self, src: str, dest: str, put: bool, **kwargs):
        """
        Initialize SCP step
        :param src: source file
        :param dest: destination file
        :param put: upload (true) or download (false) file
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.src = os.path.expanduser(src)
        self.dest = dest
        self.put = put
        self.exp_folder_name = None

    def _run(self, node: Node):
        """
        Run SCP step on experiment node
        :param node: experiment node
        """
        if self.put:
            # check if file is already on target server
            exists = node.file_exists(self.dest)

            if exists:
                print(f"Skipping SCP {self.dest}")
                return

            # create folders on remote server
            node.run_cmd([f"mkdir -p {self.dest}"])
        else:
            # create folders on local machine
            os.makedirs(self.dest, exist_ok=True)

        # transfer files via scp
        node.scp(self.src, self.dest, put=self.put)

    def __str__(self):
        return f'SCP({self.src}->{self.dest})'


class Remove(Step):
    """
    Remove files
    """

    def __init__(self, files: List[str], directory: bool = False, **kwargs):
        """
        Initialize remove step
        :param files: list of files/folders to remove
        :param directory: remove also folders recursively
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.files = files
        self.exp_folder_name = None
        self.directory = directory

    def _run(self, node: Node):
        """
        Run Remove step
        :param node: experiment node
        """
        rec = ''
        if self.directory:
            # also remove directories
            rec = '-r'

        # remove files
        node.run_cmd([f"rm {rec} {self.files}"])

    def __str__(self):
        return f'RemoveFiles({self.files})'


class RemoveRecursively(Step):
    """
    Step searching for file in directory and removing all instances of files with this name.
    """

    def __init__(self, root_dir: str, name: str, search_for_directory: bool, **kwargs):
        """
        Initialize remove recursively step
        :param root_dir: directory in which to search for occurence of filename
        :param name: name of the file
        :param search_for_directory: search directory with this name
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.root_dir = root_dir
        self.name = name
        self.search_for_directory = search_for_directory

    def _run(self, node: Node):
        """
        Run remove recursively step
        :param node: experiment node
        """
        cmd = f"cd {self.root_dir} && "
        if not self.search_for_directory:
            # search for files with given name and remove them
            cmd += f"find . -name \"{self.name}\" -type f -delete"
        else:
            # search for folders with given name and remove them
            cmd += f"find . -type d -name '{self.name}' -exec rm -r {{}} +"
        node.run_cmd([cmd])

    def __str__(self):
        return f'RemoveRecursively({self.root_dir}/{self.name})'


class KillAllScreens(Step):
    """
    Step killing all screens on nodes
    """

    def __init__(self, **kwargs):
        """
        Intialize kill all screens step
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None

    def _run(self, node: Node):
        """
        Run kill all screens step. All screen will be terminated on node.
        :param node: experiment node
        """
        node.run_cmd([f"killall screen -u {node.ssh_username}"])
        node.run_cmd([f"killall /bin/sh -u {node.ssh_username}"])
        node.run_cmd([f"pkill -f python -u {node.ssh_username}"])
        node.run_cmd([f"killall python3 -u {node.ssh_username}"])

    def __str__(self):
        return f'KillAllScreens'


class KillAllSlurmJobs(Step):
    """
    Step killing all slurm jobs
    """

    def __init__(self, **kwargs):
        """
        Initialize KillAllSlurmJobs step
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None

    def _run(self, node: Node):
        """
        Execute cancel slurm jobs command
        :param node: experiment node
        """
        node.run_cmd(["squeue -u $USER | awk '{print $1}' | tail -n+2 | xargs scancel"])

    def __str__(self):
        return f'KillAllScreens'


class DownloadGDriveFolder(Step):
    """
    Step downloading files from gdrive
    """

    def __init__(self, gfile_id: str, target_dir: str, **kwargs):
        """
        Initialize DownloadGDrive folder step by passing gDrive file_id and the target directory on experiment node where the file should be downloaded to.
        :param gfile_id: id of gDrive file
        :param target_dir: path to target directory
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.target_dir = target_dir
        self.gfile_id = gfile_id
        self.exp_folder_name = None

    def _run(self, node: Node):
        """
        Download files from gDrive
        :param node: experiment node
        """
        # check if file exists on node
        exists = node.file_exists(self.target_dir)

        if exists:
            print(f"Skipping DownloadGDrive {self.target_dir}")
            return

        node.run_cmd(
            [f"curl -c /tmp/cookies 'https://drive.google.com/uc?export=download&id={self.gfile_id}' > /tmp/gd.html"],
            return_response=True)
        content = ''.join(node.run_cmd([f'cat /tmp/gd.html'], return_response=True))
        # Content was moved case
        if 'The document has moved <A HREF="' in content:
            download_link = content.split('The document has moved <A HREF="')[1].split('"')[0]
        # File too large case
        elif 'uc-download-link' in content:
            download_link = 'https://drive.google.com' + \
                            content.split('uc-download-link')[1].split('href="')[1].split('"')[0]
            download_link = download_link.replace('&amp;', '&')
        else:
            raise NotImplementedError(f'Unmatched case: {content}')

        cmds = [f"curl -L -b /tmp/cookies '{download_link}' > gdrive_download.tar.gz",
                f"mkdir -p {self.target_dir}",
                f"mv gdrive_download.tar.gz {self.target_dir}",
                f"cd {self.target_dir}",
                f"tar -xzvf gdrive_download.tar.gz",
                ]
        node.run_cmd(cmds)

    def __str__(self):
        return f'DownloadGDrive({self.gfile_id})'


class RsyncCurrentRepo(Step):
    """
    Transfer local experiment repository to node via rsync
    """

    def __init__(self, gitignore_path: str = '../.gitignore',local_folder_path = '.', ssh_path:str='ssh', **kwargs):
        """
        Initialize RsyncCurrentRepo step
        :param ssh_path: path to ssh
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None
        self.local_folder_name = str(local_folder_path)+'/*'
        self.gitignore_path = gitignore_path
        self.ssh_path = ssh_path

    def _run(self, node: Node):
        """
        Rsync local repo
        :param node: experiment node
        """
        gitignore = open(self.gitignore_path, 'r').readlines()
        gitignore.append('\n')
        gitignore.append('.git\n')
        gitignore.append('.idea\n')

        rsync_ignore_name = f'tmp_rsync_ignore{node.ssh_hostname}'
        gitignore.append(f'{rsync_ignore_name}\n')

        with open(rsync_ignore_name, 'w') as file:
            file.writelines(gitignore)

        try:
            # create directory either remotely (in case of upload) or locally (in case of download)
            print(f"Creating directory {self.exp_folder_name} on {node.ssh_hostname}")
            node.run_cmd([f'mkdir -p {self.exp_folder_name}'])
        except:
            pass

        cmd = f"""{self.local_cmd_prefix} rsync -rzvht --update -e "{self.ssh_path} -i {node.rsync_private_key}" --exclude-from=tmp_rsync_ignore{node.ssh_hostname} --progress {self.local_folder_name} {node.ssh_username}@{node.ssh_hostname}:{self.exp_folder_name}/ """
        os.system(cmd)
        os.remove(f'tmp_rsync_ignore{node.ssh_hostname}')

    def __str__(self):
        return f'RsyncCurrentRepo'


class Rsync(Step):
    """
    Step transferring folder via rsync
    """

    def __init__(self, src, dest, put: bool = False, delete: bool = False, update: bool = False,
                 ignore_existing: bool = False, skip: bool = False,
                 include=None, ssh_path:str='ssh', **kwargs):
        """
        Initialize rsync step
        :param src: source path or list of source paths
        :param dest: target path or list of target paths
        :param put: upload (true) or download (false) files
        :param delete: delete extraneous files from dest dirs
        :param update: skip files that are newer on the receiver
        :param ignore_existing: skip updating files that exist on receiver
        :param skip: skip this step
        :param include: single file/folder or list of files/folders which should not be exclude
        :param ssh_path: path to ssh
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.src = src
        if not isinstance(src, list):
            self.src = [src]

        self.dest = dest
        if not isinstance(dest, list):
            self.dest = [dest]

        # create absolute paths
        # if put:
        #    self.src = [os.path.abspath(d) for d in self.src]
        # else:
        #    self.dest = [os.path.abspath(d) for d in self.dest]

        self.put = put
        self.delete = delete
        self.update = update
        self.ignore_existing = ignore_existing
        self.exp_folder_name = None
        self.skip = skip
        self.include = include
        if isinstance(self.include, str):
            self.include = [self.include]

        self.ssh_path = ssh_path

    def _run(self, node: Node):
        """
        Rsync files between local machine and experiment node
        :param node: experiment node
        """
        if self.skip:
            print(f'Skipping Rsync {self.src} {self.dest}')
            return

        try:
            # create directory either remotely (in case of upload) or locally (in case of download)
            if self.put:
                mkdir_cmds = []
                for dest in self.dest:
                    # check if directory is default directory
                    if dest not in ['~', '~/', '~/.']:
                        mkdir_cmds.append(f'mkdir -p {dest}')

                if len(mkdir_cmds)>0:
                    node.run_cmd(mkdir_cms)
            else:
                for dest in self.dest:
                    os.makedirs(dest, exist_ok=True)
        except:
            pass

        # delete extraneous files
        delete_cmd = ''
        if self.delete:
            delete_cmd = '--delete'

        # skip updating files that exist on receiver
        ignore_ex = ''
        if self.ignore_existing:
            ignore_ex = '--ignore-existing'

        # skip files newer on the receiver
        update = ''
        if self.update:
            update = '--update'

        # don't exclude these files/folders
        include = ''
        if self.include is not None:
            includes = ' '.join([f"--include='{i}'" for i in self.include])
            include = f"-am {includes} --include='*/' --exclude='*'"

        options = f"{include} {delete_cmd} {ignore_ex} {update} --progress"

        # execute rsync commands
        for src, dest in zip(self.src, self.dest):
            if self.put:
                cmd = f"""
                {self.local_cmd_prefix} rsync -rzvht -e "{self.ssh_path} -i {node.rsync_private_key}" {options} {src} {node.ssh_username}@{node.ssh_hostname}:{dest}
                """
            else:
                cmd = f"""
                {self.local_cmd_prefix} rsync -rzvht -e "{self.ssh_path} -i {node.rsync_private_key}" {options} {node.ssh_username}@{node.ssh_hostname}:{src} {dest}/
                """
            os.system(cmd)

    def __str__(self):
        return f'CountGeneratedFiles({self.src}->{self.dest})'


class CheckActiveScreens(Step):
    """
    Step checking the number of active screens on node
    """

    def __init__(self, **kwargs):
        """
        Initialize check active screens step
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None

    def _run(self, node: Node):
        """
        Execute check active screens
        :param node: experiment node
        """
        response = ''.join(node.run_cmd(['screen -list'], return_response=True))
        print(response)
        active_screens = re.findall('screen\d+', response)
        if len(active_screens) == 0:
            active_screens = re.findall('\d+..node-0', response)
        print(f'\N{personal computer} Node ({node.ssh_hostname})')
        print(f'- {len(active_screens)} Active Screens')
        for sc in active_screens:
            print('\t\N{ok hand sign} ' + sc)

    def __str__(self):
        return f'CheckActiveScreens'


class CheckOutput(Step):
    """
    Step executing list of commands and print the output
    """

    def __init__(self, commands: List[str], **kwargs):
        """
        Initialize check output step
        :param commands: list of commands
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None
        self.commands = commands

    def _run(self, node: Node):
        """
        Execute commands and print results
        :param node: experiment node
        """
        lines = node.run_cmd(self.commands, return_response=True)
        response = '\n'.join([l.strip() for l in lines])
        print(f'{node.ssh_hostname}: {response}')

    def __str__(self):
        return f'CheckOutput'


class CountGeneratedFiles(Step):
    """
    Step printing n files in folder
    """

    def __init__(self, path, list_n_files: int, **kwargs):
        """
        Initialize count generated files step
        :param path: file path or list of paths
        :param list_n_files: number of files in path to print
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.path = path
        if not isinstance(path, list):
            self.path = [path]
        self.exp_folder_name = None
        self.list_n_files = list_n_files

    def _run(self, node: Node):
        """
        Iterate over all paths and print n generated files
        :param node:  experiment node
        """
        for path in self.path:
            response = node.run_cmd([f'ls -lt {path}'], return_response=True)
            print(f'\N{personal computer} Node ({node.ssh_hostname})')
            print(f'- {len(response)} Created Files')
            if self.list_n_files > 0:
                for f in response[:self.list_n_files]:
                    print('\t\N{ledger} ' + f.strip())

    def __str__(self):
        return f'CountGeneratedFiles({self.path})'


class LocalStep:
    """
    Class of steps which are executed only locally
    """

    def __init__(self, local_cmd_prefix: str = ''):
        """
        Init local step
        :param local_cmd_prefix: local command prefix e.g. for executing command on windows in wsl instead of powershell
        """
        self.local_cmd_prefix = local_cmd_prefix

    def run(self):
        raise NotImplementedError


class LocalCopy(Step):
    """
    Copy files locally
    """

    def __init__(self, src: List[str], dest: List[str], recursive: bool = False, **kwargs):
        """
        Initialize localcopy step
        :param src: source paths
        :param dest: destination paths
        :param recursive: copy recursively
        :param kwargs: further arguments
        """
        Step.__init__(self, **kwargs)
        self.src = src
        self.dest = dest
        self.rec_cmd = ''
        if recursive:
            self.rec_cmd = '-r'
        self.exp_folder_name = None

    def _run(self, node: Node):
        """
        Copy files
        :param node: experiment node - not used
        """
        for src, dest in zip(self.src, self.dest):
            os.makedirs(dest, exist_ok=True)
            os.system(f'{self.local_cmd_prefix} cp {self.rec_cmd} {src} {dest}')

    def __str__(self):
        return f'Copy({self.src}->{self.dest})'


class CreateScript(LocalStep):
    """
    Step writing command to script file
    """

    def __init__(self, scriptname: str = '', script_cmd: str = '', **kwargs):
        """
        Initialize create script step
        :param scriptname: name of the script file
        :param script_cmd: command
        :param kwargs: further arguments - not used
        """
        LocalStep.__init__(self, **kwargs)
        self.scriptname = scriptname
        self.script_cmd = script_cmd

    def run(self):
        """
        Write command to script file
        """
        with open(self.scriptname, "w", newline='\n') as f:
            f.write(self.script_cmd)
        os.system(f"{self.local_cmd_prefix} chmod +x {self.scriptname}")

    def __str__(self):
        return f'CreateScript({self.scriptname})'


class LocalCmd(LocalStep):
    """
    Step executing command locally (really locally - ignoring local_cmd_prefix)
    """

    def __init__(self, cmd: str = '', **kwargs):
        """
        Initialize local cmd step
        :param cmd: command
        :param kwargs: further arguments - not used
        """
        LocalStep.__init__(self, **kwargs)
        self.cmd = cmd.replace('\n', '')

    def run(self):
        """
        Execute command
        """
        os.system(f"{self.local_cmd_prefix} {self.cmd}")

    def __str__(self):
        return f'LocalCmd({self.cmd})'


class LocalMethodCall(LocalStep):
    """
    Invoke python function locally
    """

    def __init__(self, func=None, args=None, **kwargs):
        """
        Initialize local method call step
        :param func:
        :param args:
        :param kwargs: further arguments - not used
        """
        LocalStep.__init__(self, **kwargs)
        self.func = func
        self.args = args

    def run(self):
        """
        Execute function and pass arguments to function
        :return:
        """
        self.func(**self.args)

    def __str__(self):
        return f'LocalMethodCall({self.func})'


class CombineCSVFiles(LocalStep):
    """
    Step combining results of multiple csv files in single file
    """

    def __init__(self, source_files, target_file, remove: bool):
        """
        Initialize
        :param source_files: single source file or list of files
        :param target_file: single target file or list of files
        :param remove: remove source files
        """
        LocalStep.__init__(self)

        self.source_files = source_files
        if not isinstance(source_files, list):
            self.source_files = [source_files]

        self.target_file = target_file
        if not isinstance(target_file, list):
            self.target_file = [target_file]

        self.remove = remove

    def run(self):
        """
        Combine files
        """
        for source_files, target_file in zip(self.source_files, self.target_file):
            all_filenames = [i for i in glob.glob(source_files)]
            if len(all_filenames) == 0:
                print(f'No files to combine in {source_files}')
                continue

            # concatenate data
            combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
            # create target dir
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            # write to target file
            combined_csv.to_csv(target_file, index=False, header=True)

            print(f'Combining {len(all_filenames)} files in {source_files}')

            # remove source files
            if self.remove:
                os.system(f'rm {source_files}')

    def __str__(self):
        return f'CombineCSVFiles({self.source_files}->{self.target_file})'


class SetupRYEVenv(Step):
    """
    Step setting up venv python environment
    """

    def __init__(self, force: bool = False, no_lock: bool = True, no_dev: bool = True, rye_add: List[str] = None,
                 rye_version=None, python_version:str='3.12.7', **kwargs):
        """
        Initialize the RyeSetupVenv step.

        :param requirements: List of required Python libraries to be installed.
        :param force: Force recreation of the virtual environment.
        :param no_lock: Do not update the requirements.lock file.
        :param no_dev: Install only the non-development packages.
        :param rye_add: List of additional packages to be installed on the server using the command `rye add {item}`.
                       If you do not add `--no-sync` to your package installation string, it will automatically sync your repo, which can cause problems.
        :param rye_version: Version of Rye to be used.
        :param kwargs: Additional arguments.
        """
        Step.__init__(self, **kwargs)
        self.exp_folder_name = None
        self.force = force
        self.no_dev = no_dev
        self.no_lock = no_lock
        self.rye_add = rye_add
        self.rye_version = rye_version
        self.python_version = python_version

    def _run(self, node: Node):
        """
        Execute step on node
        :param node: experiment node
        """

        # navigate to experiment folder
        cmds = [f'cd {self.exp_folder_name}']

        # check if rye is installed
        if not node.file_exists("$HOME/.rye/env"):
            print("Try to install rye!")
            rye_version_str = f'RYE_VERSION="{self.rye_version}"' if self.rye_version else ''
            rye_installation_cmd = f'curl -sSf https://rye.astral.sh/get | {rye_version_str} RYE_INSTALL_OPTION="--yes" bash'
            cmds.append(rye_installation_cmd)

        rye_run_setup = f'source "$HOME/.rye/env"'
        cmds.append(rye_run_setup)

        # added new package: best way to do it is to add --no-sync flag
        if self.rye_add:
            rye_add = "rye add "
            for cmd in self.rye_add:
                cmds.append(rye_add + cmd)

        rye_sync = "rye sync"

        # add no lock flag -> do not update requirement.lock
        if self.no_lock:
            rye_sync += " --no-lock"
        # forces reinstallation of packages
        if self.force:
            rye_sync += " --force"
        # install only the no-dev packages
        if self.no_dev:
            rye_sync += " --no-dev"
        cmds.append(f"rye pin {self.python_version}")
        cmds.append(rye_sync)

        # execute commands
        node.run_cmd(cmds)

    def __str__(self):
        return f'SetupRYEVenv'
