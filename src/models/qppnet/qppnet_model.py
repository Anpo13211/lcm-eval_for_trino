from typing import Tuple
import os

import torch
import torch.nn as nn

from cross_db_benchmark.benchmark_tools.postgres.json_plan import OperatorTree
from classes.classes import QPPNetModelConfig
from classes.workload_runs import WorkloadRuns
from training import losses
from models.zeroshot.postgres_plan_batching import add_numerical_scalers


QPPNET_VERBOSE = os.environ.get("QPPNET_VERBOSE", "").lower() in {"1", "true", "yes"}


class NeuralUnit(nn.Module):
    """ Neural Unit that covers all operators """
    def __init__(self,
                 node_type: str,
                 features: list,
                 num_layers: int = 5,
                 hidden_size: int = 128,
                 output_dim: int = 32,
                 feature_statistics: dict= None):

        super().__init__()

        # Compute input sizes. As One-Hot-encoding is applied, this depends on the feature_statistics.
        input_dim = 0
        for qpp_feature in features:
            if qpp_feature in feature_statistics.keys():
                if feature_statistics[qpp_feature]["type"] == "numeric":
                    input_dim += 1
                elif feature_statistics[qpp_feature]["type"] == "categorical":
                    input_dim += feature_statistics[qpp_feature]["no_vals"]
            elif qpp_feature in ["Min", "Max", "Mean"]:
                input_dim += 1
            else:
                raise ValueError(f"Feature {qpp_feature} not in feature statistics")

        # Compute output sizes.
        if all(nt not in node_type for nt in ["Scan", "Result"]):
            input_dim = input_dim + output_dim
            if "Join" in node_type or "Nested Loop" in node_type:
                input_dim = input_dim + output_dim

        if node_type == "Bitmap Heap Scan":
            input_dim = input_dim + output_dim

        if QPPNET_VERBOSE:
            print(f"Initializing neural unit for {node_type} "
                  f"with input_dim {input_dim} "
                  f"and output_dim: {output_dim}")

        self.node_type = node_type
        self.dense_block = self.build_block(num_layers=num_layers,
                                            hidden_size=hidden_size,
                                            output_size=output_dim,
                                            input_dim=input_dim)

    @staticmethod
    def build_block(num_layers: int, hidden_size: int, output_size: int, input_dim: int) -> nn.Sequential:
        """Construct a block consisting of linear Dense layers.
        Parameters:
            num_layers  (int)
            hidden_size (int)           -- the number of channels in the conv layer.
            output_size (int)           -- size of the output layer
            input_dim   (int)           -- input size, depends on each node_type
            norm_layer                  -- normalization layer
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))

        """
        assert num_layers >= 2, "Num of layers need to be greater than 1"
        dense_block = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            dense_block += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        dense_block += [nn.Linear(hidden_size, output_size), nn.ReLU()]

        for layer in dense_block:
            try:
                nn.init.xavier_uniform_(layer.weight)
            except:
                pass
        return nn.Sequential(*dense_block)

    def forward(self, x):
        """ Forward function """
        out = self.dense_block(x)
        return out


class QPPNet(nn.Module):
    """ QPPNet Architecture"""

    def __init__(self, model_config: QPPNetModelConfig, workload_runs: WorkloadRuns, feature_statistics: dict, label_norm=None, use_global_loss=False):
        super().__init__()
        self.device = model_config.device
        self.hidden_dim = model_config.hidden_dim_plan
        self.batch_size = model_config.batch_size
        self.feature_statistics = feature_statistics
        self.operator_types = model_config.featurization.QPP_NET_OPERATOR_TYPES
        self.use_global_loss = use_global_loss  # Trino等でクエリ全体のlossを使う場合True

        # If the dataset does not contain any indexes, remove index nodes from the operator types to prevent errors
        # and to make the model faster. This is the case for TPC_H
        """
        if (workload_runs.train_workload_runs and "tpc_h" in str(workload_runs.train_workload_runs[0])
                or workload_runs.test_workload_runs and "tpc_h" in str(workload_runs.test_workload_runs[0])):

                remove_ops = []
                for operator_type in self.operator_types:
                    if "Index" in operator_type:
                        remove_ops.append(operator_type)

                for op in remove_ops:
                    self.operator_types.pop(op)
        """
        self.loss_fxn = losses.__dict__[model_config.loss_class_name](self, **model_config.loss_class_kwargs)

        # Initialize neural units
        self.neural_units = nn.ModuleDict({
            operator_type: NeuralUnit(node_type=operator_type,
                                      features=features,
                                      feature_statistics=feature_statistics,
                                      hidden_size=self.hidden_dim)
            for operator_type, features in self.operator_types.items()})

        # Initialize optimizers
        self.optimizers = {}
        for operator_type, _ in self.operator_types.items():
            self.optimizers[operator_type] = torch.optim.Adam(self.neural_units[operator_type].parameters(),
                                                              **model_config.optimizer_kwargs)

        self.label_norm = None  # This could be in future implemented but would need to be done on operator level

        # Prepare robust encoder for the numerical fields
        add_numerical_scalers(self.feature_statistics)

    def check_for_nan(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f'Parameter {name} has NaN value!')

    def forward_single_query(self, query_plan: OperatorTree) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Iteratively do a forward pass on a single query in form of an OperatorTree.
        Returns the full predicted vector and the predicted runtime, which is the first element of the vector"""
        # self.check_for_nan()

        # Read out most important information from query
        node_properties = query_plan.properties
        node_type = query_plan.node_type
        
        # グローバルlossモードの場合、演算子レベルのActual Total Timeは不要
        if not self.use_global_loss:
            label = torch.tensor(node_properties.pop("Actual Total Time")).to(self.device)
        else:
            label = None
        
        assert node_type in self.operator_types.keys(), f"Unseen operator of type {node_type} faced in query"
        features = query_plan.encoded_features

        # Iterative forward pass
        for children in query_plan.children:
            child_pred_vector, _ = self.forward_single_query(children)
            features = torch.cat((features, child_pred_vector), axis=0)

        # 入力次元を調整（Trinoでは子ノードの数が想定より少ないことがある）
        expected_input_dim = self.neural_units[node_type].dense_block[0].in_features
        if features.shape[0] < expected_input_dim:
            pad_size = expected_input_dim - features.shape[0]
            pad = torch.zeros(pad_size, device=features.device)
            features = torch.cat((features, pad), axis=0)
        elif features.shape[0] > expected_input_dim:
            features = features[:expected_input_dim]

        if torch.isnan(features).any():
            raise ValueError(f"Features {features} for {node_type} have Nan values.")

        pred_vector = self.neural_units[node_type](features)

        if torch.isnan(pred_vector).any():
            raise ValueError(f"Prediction {pred_vector} for {features} and {node_type} has Nan values.")

        # Predicted time is assumed to be the first column
        predicted_operator_time = pred_vector[0].clone().reshape(1)

        # We apply a lower bound of predictions to avoid nans, as otherwise runtime can be 0 and q-error infinite.
        predicted_operator_time = torch.max(predicted_operator_time, torch.tensor(0.1).to(self.device))

        # 演算子レベルのloss計算（グローバルlossモードでは実行しない）
        if not self.use_global_loss and label is not None:
            self.loss_fxn.update_operator_loss(node_type=node_type, predicted_operator_time=predicted_operator_time, label=label)

        return pred_vector, predicted_operator_time / 1000

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear previous gradients of the neural units"""
        for operator in self.optimizers:
            self.optimizers[operator].zero_grad()

    def forward(self, input):
        # Initializing variables for the forward pass.
        query_plans: list[OperatorTree] = input
        predictions: list[torch.Tensor] = []

        # グローバルlossモード以外では演算子レベルのlossをリセット
        if not self.use_global_loss:
            self.loss_fxn.reset_accumulated_losses()

        # Iterate over all query plans and forward them
        for query_plan in query_plans:
            _, pred = self.forward_single_query(query_plan)
            predictions.append(pred)
            # グローバルlossモード以外では演算子レベルのlossを累積
            if not self.use_global_loss:
                self.loss_fxn.update_total_losses()

        if predictions:
            return torch.stack(predictions).squeeze(-1)
        else:
            return torch.tensor([])

    def backward(self):
        # Do step on all operator-level optimizers
        for operator in self.optimizers:
            self.optimizers[operator].step()
