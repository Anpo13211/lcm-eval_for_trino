"""
Unified dataset creation using plugin architecture

This replaces DBMS-specific dataset creation with a unified implementation
that works for PostgreSQL, Trino, and future DBMS.
"""

import functools
from pathlib import Path
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline

from core.graph.unified_collator import unified_plan_collator
from core.plugins.registry import DBMSRegistry
from core.statistics.schema import StandardizedStatistics
from classes.classes import ZeroShotModelConfig, DataLoaderOptions, ModelConfig
from classes.workload_runs import WorkloadRuns
from training.dataset.dataset_creation import create_datasets, derive_label_normalizer
from cross_db_benchmark.benchmark_tools.utils import load_json


def create_unified_zeroshot_dataloader(
    workload_runs: WorkloadRuns,
    statistics_file: Path,
    model_config: ZeroShotModelConfig,
    data_loader_options: DataLoaderOptions,
    dbms_name: str = "postgres"
) -> Tuple[Optional[Pipeline], dict, Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]]:
    """
    Create ZeroShot dataloaders using unified plugin architecture.
    
    This replaces create_zeroshot_dataloader() with DBMS-agnostic version.
    
    Args:
        workload_runs: Training/test workload paths
        statistics_file: Feature statistics file
        model_config: Model configuration
        data_loader_options: Dataloader options
        dbms_name: DBMS name (e.g., "postgres", "trino")
    
    Returns:
        tuple: (label_norm, feature_statistics, train_loader, val_loader, test_loaders)
    
    Example:
        # PostgreSQL
        loaders = create_unified_zeroshot_dataloader(
            workload_runs=runs,
            statistics_file=stats_file,
            model_config=config,
            data_loader_options=options,
            dbms_name="postgres"
        )
        
        # Trino (same code, different dbms_name)
        loaders = create_unified_zeroshot_dataloader(
            ...,
            dbms_name="trino"
        )
    """
    label_norm = None
    train_loader, val_loader, test_loaders = None, None, []
    
    # Load feature statistics
    feature_statistics = load_json(statistics_file, namespace=False) if statistics_file.exists() else {}
    
    if not feature_statistics:
        print("⚠️  Feature statistics file is empty or not found!")
    
    # Create unified collator
    # This single collator works for all DBMS!
    plan_collator = functools.partial(
        unified_plan_collator,
        dbms_name=dbms_name,
        plan_featurization=model_config.featurization
    )
    
    dataloader_args = dict(
        batch_size=model_config.batch_size,
        shuffle=data_loader_options.shuffle,
        num_workers=model_config.num_workers,
        pin_memory=data_loader_options.pin_memory
    )
    
    # Create training/validation dataloaders
    if workload_runs.train_workload_runs:
        print("Creating dataloader for training and validation data")
        
        label_norm, train_dataset, val_dataset, database_statistics = create_datasets(
            workload_run_paths=workload_runs.train_workload_runs,
            model_config=model_config,
            val_ratio=data_loader_options.val_ratio
        )
        
        # Convert database_statistics to StandardizedStatistics if needed
        standardized_stats = _ensure_standardized_statistics(
            database_statistics,
            dbms_name
        )
        
        # Create collate function with statistics
        train_collate_fn = functools.partial(
            plan_collator,
            db_statistics=standardized_stats,
            feature_statistics=feature_statistics
        )
        
        dataloader_args.update(collate_fn=train_collate_fn)
        train_loader = DataLoader(train_dataset, **dataloader_args)
        val_loader = DataLoader(val_dataset, **dataloader_args)
    
    # Create test dataloaders
    if workload_runs.test_workload_runs:
        print("Creating dataloader for test data")
        test_loaders = []
        
        for test_path in workload_runs.test_workload_runs:
            _, test_dataset, _, test_database_statistics = create_datasets(
                [test_path],
                model_config=model_config,
                shuffle_before_split=False,
                val_ratio=0.0
            )
            
            # Convert to StandardizedStatistics
            standardized_stats = _ensure_standardized_statistics(
                test_database_statistics,
                dbms_name
            )
            
            test_collate_fn = functools.partial(
                plan_collator,
                db_statistics=standardized_stats,
                feature_statistics=feature_statistics
            )
            
            dataloader_args.update(collate_fn=test_collate_fn)
            test_loader = DataLoader(test_dataset, **dataloader_args)
            test_loaders.append(test_loader)
    
    return label_norm, feature_statistics, train_loader, val_loader, test_loaders


def _ensure_standardized_statistics(db_statistics: dict, dbms_name: str) -> dict:
    """
    Ensure database statistics are in StandardizedStatistics format.
    
    Args:
        db_statistics: Raw database statistics (dict or StandardizedStatistics)
        dbms_name: DBMS name
    
    Returns:
        Dictionary mapping database_id to StandardizedStatistics
    """
    result = {}
    
    for db_id, stats in db_statistics.items():
        if isinstance(stats, StandardizedStatistics):
            # Already standardized
            result[db_id] = stats
        else:
            # Convert from DBMS-specific format
            converter = DBMSRegistry.get_statistics_converter(dbms_name)
            result[db_id] = converter.convert(stats)
    
    return result


def create_unified_dataloader_for_model(
    model_config: ModelConfig,
    workload_runs: WorkloadRuns,
    statistics_file: Path,
    data_loader_options: DataLoaderOptions,
    dbms_name: str,
    **kwargs
):
    """
    Generic dataloader creation for any model type.
    
    Routes to appropriate dataloader based on model type.
    
    Args:
        model_config: Model configuration
        workload_runs: Workload runs
        statistics_file: Statistics file
        data_loader_options: Dataloader options
        dbms_name: DBMS name
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model-specific dataloader tuple
    """
    from classes.classes import ModelName
    
    # Check model type
    if model_config.name == ModelName.ZEROSHOT or model_config.name == ModelName.ZEROSHOT_ACT_CARD:
        return create_unified_zeroshot_dataloader(
            workload_runs=workload_runs,
            statistics_file=statistics_file,
            model_config=model_config,
            data_loader_options=data_loader_options,
            dbms_name=dbms_name
        )
    
    # Add other model types as needed
    elif model_config.name == ModelName.DACE or model_config.name == ModelName.DACE_ACT_CARDS:
        # DACE has its own dataloader
        from models.dace.dace_dataset import create_dace_dataloader
        return create_dace_dataloader(
            statistics_file=statistics_file,
            model_config=model_config,
            workload_runs=workload_runs,
            dataloader_options=data_loader_options
        )
    
    else:
        raise NotImplementedError(
            f"Unified dataloader not yet implemented for model: {model_config.name}"
        )

