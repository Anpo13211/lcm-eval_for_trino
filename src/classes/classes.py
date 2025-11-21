import ast
import os
from pathlib import Path
from typing import List, Callable, Optional

import attrs as attrs
import seaborn as sns
from attr import field
from dotenv import load_dotenv

from models.query_former.dataloader import query_former_batch_to
from models.workload_driven.dataset.mscn_batching import mscn_plan_collator
from models.workload_driven.dataset.plan_tree_batching import plan_batch_to, baseline_plan_collator
from cross_db_benchmark.benchmark_tools.database import DatabaseSystem, ExecutionMode
from cross_db_benchmark.datasets.datasets import Database, database_list
from classes.paths import Paths, ClusterPaths
from training.batch_to_funcs import dace_batch_to, simple_batch_to, batch_to
from training.featurizations import Featurization, DACEFeaturization, QPPNetFeaturization, \
    PostgresEstSystemCardDetail, DACEFeaturizationNoCosts, QPPNetNoCostsFeaturization, FlatModelFeaturization, \
    FlatModelActCardFeaturization, QPPNetActCardsFeaturization, PostgresTrueCardDetail, DACEActCardFeaturization
from training.unified_featurizations import UnifiedEstSystemCardDetail, UnifiedTrueCardDetail
from core.graph.unified_collator import unified_plan_collator


class TrainingServers:
    load_dotenv()
    # 環境変数がNoneの場合は空の辞書をデフォルト値として使用
    NODE00 = ast.literal_eval(os.getenv("NODE00", "{}"))
    NODE01 = ast.literal_eval(os.getenv("NODE01", "{}"))
    NODE02 = ast.literal_eval(os.getenv("NODE02", "{}"))
    NODE03 = ast.literal_eval(os.getenv("NODE03", "{}"))
    NODE04 = ast.literal_eval(os.getenv("NODE04", "{}"))
    NODE05 = ast.literal_eval(os.getenv("NODE05", "{}"))


class InputFormats:
    parsed = "parsed"
    baseline_parsed = 'baseline'
    json = 'json'


@attrs.define(frozen=True, slots=False)
class Name:
    NAME: str = field(default=None)
    DISPLAY_NAME: str = field(default=None)


class ModelName:
    POSTGRES: Name = Name("postgres", "Sc. PG10")
    POSTGRES_V16: Name = Name("postgres_16", "Sc. PG16")
    FLAT: Name = Name("flat", "Flat Vector")
    FLAT_ACT_CARDS = Name("flat_act_cards", "Flat Vector (act. card.)")
    MSCN = Name("mscn", "MSCN")
    E2E = Name("e2e", "E2E")
    ZEROSHOT = Name("zeroshot", "Zero-Shot")
    ZEROSHOT_ACT_CARD = Name("zeroshot_act_card", "Zero-Shot (act. card)")
    QPP_NET = Name("qppnet", "QPP-Net")
    QPP_NET_NO_COSTS = Name("qppnet_no_costs", "QPP-Net\n(w/o PG costs)")
    QPP_NET_ACT_CARDS = Name("qppnet_act_cards", "QPP-Net (act. card.)")
    DACE = Name("dace", "DACE")
    DACE_ACT_CARDS = Name("dace_act_cards", "DACE (act. card.)")
    DACE_NO_COSTS = Name("dace_no_costs", "DACE\n(w/o PG costs)")
    QUERY_FORMER = Name("query_former", "QueryFormer")


class ModelType:
    WL_DRIVEN = "wl_driven"
    WL_AGNOSTIC = "wl_agnostic"


class ColorManager:
    DARK = sns.color_palette("muted", 9)
    ALTERNATIVE = sns.color_palette("pastel", 9)
    COLOR_MAP = {
        ModelName.POSTGRES: DARK[7],
        ModelName.POSTGRES_V16: DARK[8],
        ModelName.FLAT: DARK[5],
        ModelName.MSCN: DARK[4],
        ModelName.E2E: DARK[3],
        ModelName.ZEROSHOT: DARK[6],
        ModelName.QPP_NET: DARK[2],
        ModelName.QUERY_FORMER: DARK[0],
        ModelName.DACE: DARK[1],
        ModelName.FLAT_ACT_CARDS: ALTERNATIVE[5],
        ModelName.ZEROSHOT_ACT_CARD: ALTERNATIVE[6],
        ModelName.DACE_ACT_CARDS: ALTERNATIVE[1],
        ModelName.DACE_NO_COSTS: ALTERNATIVE[1],
        ModelName.QPP_NET_NO_COSTS: ALTERNATIVE[2],
        ModelName.QPP_NET_ACT_CARDS: ALTERNATIVE[2],
    }
    COLOR_PALETTE = {name.DISPLAY_NAME: color for (name, color) in COLOR_MAP.items()}

    @staticmethod
    def get_color(model_name: Name) -> str:
        return ColorManager.COLOR_MAP.get(model_name)


@attrs.define(frozen=True, slots=False)
class ModelConfig:
    name: Name = field(default=None)
    type: ModelType = field(default=None)
    device: str = field(default=None)
    featurization: Featurization = field(default=None)
    batch_to_func: staticmethod = field(default=None)
    collator_func: staticmethod = field(default=None)
    optimizer_class_name: str = "AdamW"
    optimizer_kwargs: dict = dict(lr=1e-3)
    batch_size: int = 64
    seed: int = 0
    epochs: int = field(default=200)
    early_stopping_patience: int = 20
    num_workers: int = 20
    max_epoch_tuples: int = field(default=None)
    cap_training_samples: int = None
    hidden_dim_plan: int = 256
    hidden_dim_pred: int = field(default=None)
    loss_class_name: str = "QLoss"
    loss_class_kwargs: dict = dict()
    fc_out_kwargs: dict = field(default=None)
    final_mlp_kwargs: dict = field(default=None)
    node_type_kwargs: dict = field(default=None)
    tree_layer_kwargs: dict = field(default=None)
    tree_layer_name: dict = field(default=None)
    limit_queries: bool = None
    limit_queries_affected_wl: bool = None
    execution_mode: ExecutionMode = ExecutionMode.RAW_OUTPUT
    input_format: InputFormats = field(default=None)
    hyperparameter: Path = None
    column_statistics: bool = False
    word_embeddings: bool = False

    def color(self) -> str:
        return ColorManager.get_color(self.name)

    def get_model_dir(self, source_path: Paths, database: Database, retrain: bool = False) -> Path:
        if retrain:
            return source_path.retraining_models / self.name.NAME / database.db_name
        else:
            return source_path.models / self.name.NAME / database.db_name

    def get_eval_dir(self, source_path: Paths, database: Database, retrain: bool = False) -> Path:
        if retrain:
            return source_path.retraining_evaluation / self.name.NAME / database.db_name
        else:
            return source_path.evaluation / self.name.NAME / database.db_name

    def get_statistics(self, source_path: Paths, database: Database) -> Path:
        if self.name in [ModelName.MSCN, ModelName.QUERY_FORMER]:
            if database == Database('imdb'):
                return source_path.augmented_plans_baseline / database.db_name / 'statistics_combined.json'
            else:
                return source_path.augmented_plans_baseline / database.db_name / 'statistics.json'
        if isinstance(self, QPPNetModelConfig):
            return source_path.json / database.db_name / 'feature_stats_training_data_scans_joins_physical.json'
        else:
            return source_path.parsed_plans / 'statistics_complex_workload_combined.json'

    def get_column_stats(self, database: Database) -> Optional[Path]:
        if self.column_statistics:
            return Path(f'./cross_db_benchmark/datasets/{database.db_name}/column_statistics.json')
        else:
            return None

    def get_word_embeddings(self, source_path: Paths, database: Database) -> Optional[Path]:
        if self.word_embeddings:
            return source_path.sentences / database.db_name / 'word2vec.m'
        else:
            return None

    def get_training_workloads(self, target_db: Database, wl_name: str = "workload_100k_s1_c8220") -> List[str]:
        if self.type == ModelType.WL_AGNOSTIC:
            training_workloads = []
            for db in database_list:
                if db.db_name != target_db.db_name:
                    path = self.get_data_base_dir() / Path(db.db_name) / f"{wl_name}.json"
                    training_workloads.append(str(path))

            assert len(training_workloads) == 19, f"Found {len(training_workloads)}, but it should be 19"
            return training_workloads

        elif self.type == ModelType.WL_DRIVEN:
            path = self.get_data_base_dir() / target_db.db_name / f"{wl_name}.json"
            return [str(path)]

    def get_test_workloads(self, target_db: Database) -> str:
        return str(self.get_data_base_dir() / target_db.db_name / "workload_100k_s1_c8220.json")

    def get_data_base_dir(self) -> Path:
        if self.input_format == InputFormats.baseline_parsed:
            return ClusterPaths().augmented_plans_baseline
        elif self.input_format == InputFormats.parsed:
            return ClusterPaths().parsed_plans
        elif self.input_format == InputFormats.json:
            return ClusterPaths().json


@attrs.define(frozen=True, slots=False)
class TabularModelConfig(ModelConfig):
    gen_dataset_func: staticmethod = field(default=None)


@attrs.define(frozen=True, slots=False)
class FlatModelConfig(TabularModelConfig):
    name: ModelName = ModelName.FLAT
    type: ModelType = ModelType.WL_AGNOSTIC
    device: str = "cpu"
    input_format: str = InputFormats.parsed
    featurization: Featurization = FlatModelFeaturization


@attrs.define(frozen=True, slots=False)
class FlatModelActCardModelConfig(FlatModelConfig):
    name: Name = ModelName.FLAT_ACT_CARDS
    input_format: str = InputFormats.parsed
    featurization: Featurization = FlatModelActCardFeaturization


@attrs.define(frozen=True, slots=False)
class ScaledPostgresModelConfig(TabularModelConfig):
    name: Name = ModelName.POSTGRES
    type: ModelType = ModelType.WL_AGNOSTIC
    device: str = "cpu"
    input_format: str = InputFormats.parsed

@attrs.define(frozen=True, slots=False)
class ScaledPostgresModelV16Config(ScaledPostgresModelConfig):
    name: Name = ModelName.POSTGRES_V16


@attrs.define(frozen=True, slots=False)
class AnalyticalEstCardModelConfig(TabularModelConfig):
    model_name: Name = Name("analytical_est_card", "Analytical (est. card)")


@attrs.define(frozen=True, slots=False)
class AnalyticalActCardModelConfig(TabularModelConfig):
    model_name: Name = Name("analytical_est_card", "Analytical (act. card)")


@attrs.define(frozen=True, slots=False)
class QPPNetModelConfig(ModelConfig):
    name: Name = ModelName.QPP_NET
    type: ModelType = ModelType.WL_DRIVEN
    execution_mode: ExecutionMode = ExecutionMode.JSON_OUTPUT
    optimizer_kwargs: dict = dict(lr=1e-3)
    featurization: Featurization = QPPNetFeaturization
    loss_class_name: str = "QPPLoss"
    loss_class_kwargs: dict = dict()
    hidden_dim_plan: int = 128
    batch_to_func: Callable = simple_batch_to
    device: str = "cpu"
    input_format: str = InputFormats.json
    column_statistics: bool = True


@attrs.define(frozen=True, slots=False)
class QPPModelNoCostsConfig(QPPNetModelConfig):
    name: Name = ModelName.QPP_NET_NO_COSTS
    featurization: Featurization = QPPNetNoCostsFeaturization


@attrs.define(frozen=True, slots=False)
class QPPModelActCardsConfig(QPPNetModelConfig):
    name: Name = ModelName.QPP_NET_ACT_CARDS
    featurization: Featurization = QPPNetActCardsFeaturization


@attrs.define(frozen=True, slots=False)
class E2EModelConfig(ModelConfig):
    name: Name = ModelName.E2E
    type: ModelType = ModelType.WL_DRIVEN
    device: str = "cuda:0"
    hidden_dim_plan: int = 256
    hidden_dim_pred: int = 256
    batch_to_func: Callable = plan_batch_to
    collator_func: Callable = baseline_plan_collator
    execution_mode: ExecutionMode = ExecutionMode.RAW_OUTPUT
    input_format: str = InputFormats.baseline_parsed
    column_statistics: bool = True
    word_embeddings: bool = True


@attrs.define(frozen=True, slots=False)
class TPoolModelConfig(E2EModelConfig):
    model_name: Name = ModelName.E2E


@attrs.define(frozen=True, slots=False)
class TlstmModelConfig(E2EModelConfig):
    model_name: Name = ModelName.E2E


@attrs.define(frozen=True, slots=False)
class TestSumModelConfig(E2EModelConfig):
    model_name: Name = Name("testsum", "TestSum")


@attrs.define(frozen=True, slots=False)
class ZeroShotModelConfig(ModelConfig):
    name: Name = ModelName.ZEROSHOT
    type: ModelType = ModelType.WL_AGNOSTIC
    featurization: Featurization = UnifiedEstSystemCardDetail
    hidden_dim_plan: int = 256
    hidden_dim_pred: int = 256
    skip_train: bool = False
    p_dropout: float = 0.1
    optimizer_kwargs: dict = dict()
    fc_out_kwargs: dict = dict(p_dropout=0.1,
                               activation_class_name='LeakyReLU',
                               activation_class_kwargs={},
                               norm_class_name='Identity',
                               norm_class_kwargs={},
                               residual=False,
                               dropout=True,
                               activation=True,
                               inplace=True)
    final_mlp_kwargs: dict = dict(width_factor=1,
                                  n_layers=2,
                                  loss_class_name="QLoss",  # MSELoss
                                  loss_class_kwargs=dict())
    tree_layer_kwargs: dict = dict(width_factor=1,
                                   n_layers=2)
    node_type_kwargs: dict = dict(width_factor=1,
                                  n_layers=2,
                                  one_hot_embeddings=True,
                                  max_emb_dim=32,
                                  drop_whole_embeddings=False)

    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)
    hidden_dim: int = 128
    output_dim: int = 1
    batch_size: int = 64
    tree_layer_name: str = 'MscnConv'  # GATConv MscnConv
    # database: Can be DatabaseSystem enum OR string (e.g., 'postgres', 'trino')
    # For new code, prefer string to leverage plugin registry
    database: str = "postgres"
    batch_to_func: Callable = batch_to
    # collator_func: Now using unified_plan_collator (works for all DBMS)
    collator_func: Callable = unified_plan_collator
    execution_mode: ExecutionMode = ExecutionMode.RAW_OUTPUT
    device: str = "cuda:0"
    input_format: str = InputFormats.parsed
    hyperparameter: Path = Path(f'conf/tuned_hyperparameters/tune_est_best_config.json')


@attrs.define(frozen=True, slots=False)
class ZeroShotModelActCardConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZEROSHOT_ACT_CARD
    featurization: Featurization = UnifiedTrueCardDetail
    hyperparameter: Path = Path(f'conf/tuned_hyperparameters/tune_best_config.json')


@attrs.define(frozen=True, slots=False)
class MSCNModelConfig(ModelConfig):
    name: Name = ModelName.MSCN
    type: ModelType = ModelType.WL_DRIVEN
    device: str = "cuda:0"
    mscn_enc_layers: int = 1
    mscn_est_layers: int = 1
    batch_to_func: Callable = batch_to
    collator_func: Callable = mscn_plan_collator
    execution_mode: ExecutionMode = ExecutionMode.RAW_OUTPUT
    input_format: str = InputFormats.baseline_parsed
    column_statistics: bool = True
    word_embeddings: bool = True


@attrs.define(frozen=True, slots=False)
class DACEModelConfig(ModelConfig):
    name: Name = ModelName.DACE
    type: ModelType = ModelType.WL_AGNOSTIC
    execution_mode: ExecutionMode = ExecutionMode.RAW_OUTPUT
    node_length: int = 22  # This needs to be equals to the number of operator types plus the remaining node feats.
    hidden_dim: int = 128
    output_dim: int = 1
    mlp_activation: str = "ReLU"
    transformer_activation: str = "gelu"
    mlp_dropout: float = 0.3
    transformer_dropout: float = 0.2
    max_runtime: int = 30000
    pad_length: int = 22
    loss_weight: float = 0.5
    batch_to_func: Callable = dace_batch_to
    loss_class_name: str = "DaceLoss"
    loss_class_kwargs: dict = dict()
    featurization: Featurization = DACEFeaturization
    device: str = "cuda:0"
    input_format: str = InputFormats.parsed
    optimizer_class_name: str = "Adam"
    optimizer_kwargs: dict = dict(lr=1e-3)


@attrs.define(frozen=True, slots=False)
class DACEModelNoCostsConfig(DACEModelConfig):
    name: Name = ModelName.DACE_NO_COSTS
    node_length: int = 21  # This needs to be equals to the number of operator types plus the remaining node feats.
    featurization: Featurization = DACEFeaturizationNoCosts


@attrs.define(frozen=True, slots=False)
class DACEModelActCardConfig(DACEModelConfig):
    name: Name = ModelName.DACE_ACT_CARDS
    node_length: int = 22
    featurization: Featurization = DACEActCardFeaturization


@attrs.define(frozen=True, slots=False)
class QueryFormerModelConfig(ModelConfig):
    name: Name = ModelName.QUERY_FORMER
    type: ModelType = ModelType.WL_DRIVEN
    device: str = "cuda:0"
    hidden_dim_plan: int = 256
    execution_mode: ExecutionMode = ExecutionMode.RAW_OUTPUT
    input_format: str = InputFormats.baseline_parsed
    column_statistics: bool = True
    word_embeddings: bool = True
    embedding_size: int = 64
    ffn_dim: int = 32
    head_size: int = 8
    dropout: float = 0.1
    attention_dropout_rate: float = 0.1
    n_layers: int = 8
    use_sample: bool = True
    use_histogram: bool = True
    histogram_bin_number: int = 10
    hidden_dim_prediction: int = 128
    max_num_filters: int = 6
    batch_to_func: Callable = query_former_batch_to
    loss_class_name: str = "QLoss"  # Originally MSELoss
    optimizer_class_name: str = "AdamW"  # Originally Adam


@attrs.define(frozen=True, slots=False)
class DataLoaderOptions:
    val_ratio: float = 0.20
    shuffle: bool = True
    pin_memory: bool = False
    dim_bitmaps: int = 1000


MODEL_CONFIGS = [
    ScaledPostgresModelConfig(),
    ScaledPostgresModelV16Config(),
    FlatModelConfig(),
    MSCNModelConfig(),
    E2EModelConfig(),
    QPPNetModelConfig(),
    QueryFormerModelConfig(),
    ZeroShotModelConfig(),
    DACEModelConfig(),
]

NO_COSTS_MODEL_CONFIGS = MODEL_CONFIGS + [QPPModelNoCostsConfig(), DACEModelNoCostsConfig()]

ACT_CARD_MODEL_CONFIGS = [
    ScaledPostgresModelConfig(),
    FlatModelConfig(),
    FlatModelActCardModelConfig(),
    QPPNetModelConfig(),
    QPPModelActCardsConfig(),
    ZeroShotModelConfig(),
    ZeroShotModelActCardConfig(),
    DACEModelConfig(),
    DACEModelActCardConfig(),
]

ACT_CARD_ALL_MODEL_CONFIGS = [
    ScaledPostgresModelConfig(),
    ScaledPostgresModelV16Config(),
    FlatModelConfig(),
    MSCNModelConfig(),
    E2EModelConfig(),
    QPPNetModelConfig(),
    QueryFormerModelConfig(),
    ZeroShotModelConfig(),
    DACEModelConfig(),
    FlatModelActCardModelConfig(),
    QPPModelActCardsConfig(),
    ZeroShotModelActCardConfig(),
    DACEModelActCardConfig()
]