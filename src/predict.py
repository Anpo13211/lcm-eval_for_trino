from pathlib import Path
from typing import List

from dgl.dataloading import DataLoader
from torch import nn
from wandb.apis.public import Run

from  classes.classes import ModelConfig
from classes.workload_runs import WorkloadRuns
from training.training.checkpoint import save_csv
from training.training.metrics import Metric
from training.training.train import validate_model
from training.training.utils import find_early_stopping_metric


def predict_model(mode: str,
                  config: ModelConfig,
                  test_loaders: List[DataLoader],
                  workload_runs: WorkloadRuns,
                  model_dir: Path,
                  metrics: List[Metric],
                  model: nn.Module,
                  epoch: int,
                  run: Run):

    if test_loaders is not None:
        if not (model_dir is None or config.name.NAME is None):
            for test_path, test_loader in zip(workload_runs.target_test_csv_paths, test_loaders):
                print(f"Starting validation for {test_path}")
                test_stats = dict()

                # In case of retraining, do not load the totally best model but the latest one
                if mode != "retrain":
                    early_stop_m = find_early_stopping_metric(metrics)
                    model.load_state_dict(early_stop_m.best_model)

                validate_model(config=config,
                               val_loader=test_loader,
                               model=model,
                               epoch=epoch,
                               epoch_stats=test_stats,
                               metrics=metrics,
                               custom_batch_to=config.batch_to_func,
                               log_all_queries=True,
                               run=run,
                               model_dir=model_dir,
                               target_path=test_path)

                save_csv([test_stats], str(test_path) + "_test_stats.csv")
        else:
            print("Skipping saving the test stats")
