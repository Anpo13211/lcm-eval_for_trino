import os
from pathlib import Path

from dotenv import load_dotenv


class Paths:
    root: Path
    data: Path
    code: Path
    runs: Path
    raw: Path
    json: Path
    parsed_plans: Path
    parsed_plans_baseline: Path
    augmented_plans_baseline: Path
    workloads: Path
    evaluation_workloads: Path
    training_workloads: Path

    evaluation: Path
    retraining_evaluation: Path

    models: Path
    retraining_models: Path
    sentences: Path
    known_hosts: Path

    def __init__(self, root_path: Path, data_path: Path = None):
        self.root = root_path
        if data_path:
            self.data = data_path
        else:
            self.data = root_path / 'data'
        self.code = root_path / 'src'
        self.runs = self.data / 'runs'
        self.raw = self.runs / 'raw'
        self.json = self.runs / 'json'
        self.parsed_plans = self.runs / 'parsed_plans'
        self.parsed_plans_baseline = self.runs / 'parsed_plans_baseline'
        self.augmented_plans_baseline = self.runs / 'augmented_plans_baseline'
        self.workloads = self.data / 'workloads'
        self.evaluation_workloads = self.workloads / 'evaluation'
        self.training_workloads = self.workloads / 'training'

        self.evaluation = self.data / 'evaluation'
        self.retraining_evaluation = self.data / 'retraining_evaluation'

        self.models = self.data / 'models'
        self.retraining_models = self.data / 'retraining_models'

        self.sentences = self.runs / 'sentences'


class LocalPaths(Paths):
    def __init__(self):
        load_dotenv()
        super().__init__(root_path=Path(os.getenv('LOCAL_ROOT_PATH')))
        self.node_list = self.code / 'scripts/misc/hostnames'
        self.requirements = self.root / 'requirements'
        self.plotting_path = self.data / 'plots'
        self.dataset_path = self.code / 'cross_db_benchmark' / 'datasets'
        self.known_hosts = Path(os.getenv('LOCAL_KNOWN_HOSTS_PATH'))


class CloudlabPaths(Paths):

    def __init__(self):
        load_dotenv()
        super().__init__(root_path=Path(os.getenv('CLOUDLAB_ROOT_PATH')))


class ClusterPaths(Paths):

    def __init__(self):
        load_dotenv()
        super().__init__(root_path=Path(os.getenv('CLUSTER_ROOT_PATH')),
                         data_path=Path(os.getenv('CLUSTER_STORAGE_PATH')))
