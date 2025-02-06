import re


def strip_single_command(cmd):
    cmd = cmd.replace('\n', ' ')
    regex = re.compile(r"\s+", re.IGNORECASE)
    cmd = regex.sub(" ", cmd)
    return cmd


def strip_commands(exp_commands):
    exp_commands = [strip_single_command(cmd) for cmd in exp_commands]
    return exp_commands
