from enum import Enum


class ExperimentTask(Enum):
    """
    Available experiments tasks
    - select runexp nodes: select free nodes on runexp cluster
    - setup: setup node infrastructure
    - deliver scripts: upload run scripts
    - start: start experiments
    - monitor: monitor experiment status
    - pickup: pickup experiment results
    - purge: clean up everything
    - check_errors: check for errors during experiments
    - print: print experiment commands
    """
    SELECT_RUNEXP_NODES = 'select_runexp_nodes'
    SETUP = 'setup'
    DELIVER_SCRIPTS = 'deliver'
    START = 'start'
    MONITOR = 'monitor'
    PICKUP = 'pickup'
    PURGE = 'purge'
    CHECK_ERRORS = 'check_errors'
    PRINT = 'print'

    def __str__(self):
        return self.value
