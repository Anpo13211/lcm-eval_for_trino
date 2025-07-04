import paramiko
import sys
from scp import SCPClient
from typing import List, Tuple, Set


class Node:
    """
    Object representing execution node.
    """

    def __init__(self, ssh_hostname: str = None, ssh_username: str = None, ssh_private_key: str = None,
                 rsync_private_key: str = None, ssh_passphrase: str = None, ssh_hostkeys: str = None,
                 hardware_desc: str = None, tags: Set[str] = None, test_connection: bool = True,
                 use_slurm: bool = False, use_run_exp: bool = False, python_version: str = None,
                 available_sockets: List[int] = [0], num_parallel_screens_or_jobs: int = None):
        """
        Create node with given data.
        :param ssh_hostname: ssh hostname of experiment node
        :param ssh_username: ssh username of experiment node
        :param ssh_private_key: path to private ssh key
        :param rsync_private_key: path to private ssh key used for rsync
        :param ssh_passphrase: passphrase for ssh key (or load key in ssh agent)
        :param ssh_hostkeys: path to authorized ssh host keys
        :param hardware_desc: name of the node hardware
        :param tags: set of hardware tags (e.g. cloudlab, dmnode, ...)
        :param test_connection: test connection to server after creating object
        :param use_slurm: this node required job execution via slurm
        :param use_run_exp: this node needs job execution via run_exp
        :param python_version: python version to use
        :param available_sockets: list of available sockets
        :param num_parallel_screens_or_jobs: number of parallel screens or jobs - if none, use number of available sockets
        """
        self.ssh_client = None

        self.ssh_hostname = ssh_hostname.strip()
        self.ssh_username = ssh_username
        self.ssh_private_key = ssh_private_key
        self.rsync_private_key = rsync_private_key
        if self.rsync_private_key is None:
            self.rsync_private_key = self.ssh_private_key
        self.ssh_passphrase = ssh_passphrase
        self.ssh_hostkeys = ssh_hostkeys
        self.hardware_desc = hardware_desc
        if tags is None:
            tags = set()
        self.tags = tags
        self.use_slurm = use_slurm
        self.use_run_exp = use_run_exp

        self.python_version = python_version

        self.available_sockets = available_sockets
        if num_parallel_screens_or_jobs is None:
            self.num_parallel_screens_or_jobs = len(self.available_sockets)
        else:
            self.num_parallel_screens_or_jobs = num_parallel_screens_or_jobs

        # test server connection
        if test_connection:
            self.connect_ssh_client()

    def connect_ssh_client(self):
        """
        Connect the ssh client
        """
        if self.ssh_client is None:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.load_host_keys(self.ssh_hostkeys)
        try:
            self.ssh_client.connect(self.ssh_hostname, username=self.ssh_username, key_filename=self.ssh_private_key,
                                    passphrase=self.ssh_passphrase)
        except paramiko.ssh_exception.SSHException as e:
            print('Authentication with server failed. Is your username and keyfile correct?')
            raise e

    def run_cmd(self, commands: List[str], return_response: bool = False):
        """
        Execute command and return results.
        :param commands: list of commands
        :param return_response: boolean value determining if results should be returned or not
        :return: result
        """
        # connect to server
        self.connect_ssh_client()

        cmd = ' && '.join(commands)

        if not return_response:
            print(f"{self.ssh_hostname} > {cmd}")

        # execute commands on server
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)

        errs = stderr.read()
        if errs:
            print('\N{large red circle} ' + str(errs))

        stdout.channel.recv_exit_status()
        response = stdout.readlines()
        if not return_response:
            for line in response:
                print(f"{self.ssh_hostname} > {line.strip()}")

        # close ssh connection
        self.ssh_client.close()

        if return_response:
            return response

        # interactive
        # self.ssh_client.connect(self.ssh_hostname, username=self.ssh_username, key_filename=self.ssh_private_key)
        #
        # cmd = ' && '.join(commands)
        #
        # if not return_response:
        #     print(f"{self.ssh_hostname} > {cmd}")
        # stdin, stdout, stderr = self.ssh_client.exec_command(cmd, get_pty=True)
        #
        # response = []
        # while True:
        #     line = stdout.readline()
        #     line = line.rstrip().strip()
        #     if not return_response:
        #         print(f"{self.ssh_hostname} > {line}")
        #     response.append(line)
        #     if stdout.channel.exit_status_ready():
        #         break
        #
        # errs = stderr.read()
        # if errs:
        #     print('\N{large red circle} ' + str(errs))
        #
        # self.ssh_client.close()
        #
        # if return_response:
        #     return response

    def run_screen_cmd(self, commands, logfile: str = None, recreate_logfile: bool = False):
        """
        Create screen on node and execute commands in screen
        :param commands: commands to execute
        :param logfile: path to logfile
        :param recreate_logfile: recreate the logfile before writing to it - this will remove all previously collected logs
        """

        screen_options = '-dm bash -c'

        if recreate_logfile:
            target_commands = [f'rm {logfile}']
        else:
            target_commands = []

        # for every command append both stdout and stderr to file and output in console
        target_commands.extend([f'{cmd} |& tee -a {logfile}' for cmd in commands])

        # establish ssh connection and create screen + execute commands in screen
        self.connect_ssh_client()
        screen_cmd = f"screen {screen_options} '{';'.join(target_commands)}'"
        self.ssh_client.exec_command(screen_cmd)

    def scp(self, src, dest, put=True):
        """
        Transfer file
        :param src: source path
        :param dest: destination path
        :param put: upload (true) / download (false)
        """

        # open ssh connection
        self.connect_ssh_client()

        def progress4(filename: bytes, size: int, sent: int, peername: Tuple[str, int]):
            """
            Print progress
            :param filename: name of the transferred file
            :param size: total size of the file
            :param sent: bytes sent
            :param peername:  ???
            """
            sys.stdout.write("(%s:%s) %s's progress: %.2f%%   \r" % (
                peername[0], peername[1], filename, float(sent) / float(size) * 100))

        # create scp client and configure printing of transfer progress
        scp = SCPClient(self.ssh_client.get_transport(), progress4=progress4)

        if put:
            # upload
            scp.put(src, dest)
        else:
            # download
            scp.get(src, dest)

        # close connections
        scp.close()
        self.ssh_client.close()

    def file_exists(self, file_to_check: str) -> bool:
        """
        Check if file is exisitng on remote server
        :param file_to_check: path to remote file
        :return: boolean value indicating file exists (true) / does not exist (false)
        """

        # open ssh connection
        self.connect_ssh_client()

        # execute test -e command
        stdin, stdout, stderr = self.ssh_client.exec_command('test -e {0} && echo exists'.format(file_to_check))
        errs = stderr.read()
        if errs:
            raise Exception('Failed to check existence of {0}: {1}'.format(file_to_check, errs))

        stdout.channel.recv_exit_status()
        response = stdout.readlines()
        file_exists = False
        if len(response) >= 1:
            file_exists = response[0].strip() == 'exists'
        self.ssh_client.close()
        return file_exists
