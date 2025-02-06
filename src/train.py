import time
from pathlib import Path
from typing import List

import torch
from dgl.dataloading import DataLoader
from torch import nn
from torch.optim import Optimizer
from wandb.apis.public import Run

import wandb
from  classes.classes import ModelConfig
from classes.paths import LocalPaths

from training.training.checkpoint import save_checkpoint
from training.training.metrics import Metric
from training.training.train import validate_model, train_epoch


def train_model(csv_stats: List,
                model_dir: Path,
                config: ModelConfig,
                train_loader: DataLoader,
                epochs_wo_improvement: int,
                epoch: int,
                val_loader: DataLoader,
                model: nn.Module,
                optimizer: Optimizer,
                metrics: List[Metric],
                run: Run,
                pt_profiler: bool = False):

    prof = None
    if pt_profiler:
        log_path = f'{LocalPaths.evaluation}/{config.name.NAME}/profile_runs/'
        print(f'Profile with pytorch profiler to logdir {log_path}')
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1,
                                             warmup=1,
                                             active=1,
                                             repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
            with_stack=True, record_shapes=True)
        prof.start()

    while epoch < config.epochs:
        print(f"Epoch {epoch}")
        epoch_stats = dict()
        epoch_stats.update(epoch=epoch)
        epoch_start_time: float = time.perf_counter()

        train_epoch(epoch_stats=epoch_stats,
                    train_loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    max_epoch_tuples=config.max_epoch_tuples,
                    custom_batch_to=config.batch_to_func,
                    profiler=prof)

        any_best_metric = validate_model(config=config,
                                         val_loader=val_loader,
                                         model=model,
                                         epoch=epoch,
                                         epoch_stats=epoch_stats,
                                         metrics=metrics,
                                         max_epoch_tuples=config.max_epoch_tuples,
                                         custom_batch_to=config.batch_to_func,
                                         log_all_queries=False,
                                         target_path=None,
                                         run=None)

        epoch_stats.update(epoch=epoch, epoch_time=time.perf_counter() - epoch_start_time)

        if run:
            wandb.log(epoch_stats)

        # See if we can already stop the training
        stop_early = False
        if not any_best_metric:
            epochs_wo_improvement += 1
            if config.early_stopping_patience is not None and epochs_wo_improvement > config.early_stopping_patience:
                stop_early = True
        else:
            epochs_wo_improvement = 0

        # also set finished to true if this is the last epoch
        if epoch == config.epochs - 1:
            stop_early = True

        epoch_stats.update(stop_early=stop_early)
        print(f"--> Epochs without improvement: {epochs_wo_improvement}/{config.early_stopping_patience}")

        # save stats to file
        csv_stats.append(epoch_stats)

        # save current state of training allowing us to resume if this is stopped
        save_checkpoint(epochs_wo_improvement=epochs_wo_improvement,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        target_path=model_dir,
                        config=config,
                        metrics=metrics,
                        csv_stats=csv_stats,
                        finished=stop_early)
        epoch += 1
        if stop_early:
            print(f"Early stopping kicked in due to no improvement in {config.early_stopping_patience} epochs")
            if prof:
                prof.stop()
            break
