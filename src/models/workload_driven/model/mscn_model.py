import torch
from torch import nn
from torch.nn import ReLU

from models.workload_driven.dataset.dataset_creation import MSCNInputDims
from classes.classes import MSCNModelConfig
from training import losses
from models.zeroshot.message_aggregators.pooling import PoolingType, PoolingConv
from models.zeroshot.zero_shot_model import PassDirection


class MSCNModel(nn.Module):

    def __init__(self,
                 model_config: MSCNModelConfig,
                 input_dims: MSCNInputDims,
                 device='cpu'):

        super().__init__()

        self.hidden_dim = model_config.hidden_dim_plan
        self.input_dim_table = input_dims.input_dim_table
        self.input_dim_pred = input_dims.input_dim_pred
        self.input_dim_join = input_dims.input_dim_join
        self.input_dim_agg = input_dims.input_dim_agg
        self.device = device
        self.label_norm = None

        self.pool_model = PoolingConv(PoolingType.MEAN)
        self.pool_node_types = ['table', 'pred', 'agg', 'join']

        input_dims = {
            'table': input_dims.input_dim_table,
            'pred': input_dims.input_dim_pred,
            'agg': input_dims.input_dim_agg, ''
            'join': input_dims.input_dim_join
        }

        self.loss_fxn = losses.__dict__[model_config.loss_class_name](self, **model_config.loss_class_kwargs)

        feature_encoders = dict()
        for node_t in self.pool_node_types:
            layers = [nn.Linear(input_dims[node_t], self.hidden_dim),
                      ReLU()]
            for _ in range(model_config.mscn_enc_layers - 1):
                layers += [nn.Linear(input_dims[node_t], self.hidden_dim),
                           ReLU()]
            feature_encoders[node_t] = nn.Sequential(*layers)
        self.feature_encoders = nn.ModuleDict(feature_encoders)

        est_ops = [nn.BatchNorm1d(len(self.pool_node_types) * self.hidden_dim),
                   nn.Linear(len(self.pool_node_types) * self.hidden_dim, self.hidden_dim),
                   ReLU()]

        for _ in range(model_config.mscn_est_layers):
            est_ops += [nn.Linear(self.hidden_dim, self.hidden_dim),
                        ReLU()]
        est_ops += [nn.Linear(self.hidden_dim, 1)]

        self.est_layer = nn.Sequential(*est_ops)

    def encode_features(self, features):
        enc_features = {
            node_t: self.feature_encoders[node_t](features[node_t])
            for node_t in features.keys()
        }

        return enc_features

    def forward(self, input):
        """
        Returns logits for output classes
        """
        graph, features = input

        # encode features using simple MLPs
        enc_plan_feats = self.encode_features(features)

        # average sets and concatenate
        out = self.pool_set_features(graph, enc_plan_feats)
        # final estimation layer to predict runtime
        out = self.est_layer(out)
        return out

    def pool_set_features(self, graph, plan_feats):
        node_types = list(plan_feats.keys())
        plan_feats['plan'] = torch.zeros(graph.num_nodes('plan'), self.hidden_dim, device=self.device)

        pooled_feats = []
        for node_type in node_types:
            pass_directions = []
            pd = PassDirection(model_name='pool',
                               g=graph,
                               e_name=f'{node_type}_plan',
                               n_dest='plan',
                               allow_empty=True)
            pass_directions.append(pd)

            if len(pd.etypes) == 0:
                out = torch.zeros(graph.num_nodes('plan'), self.hidden_dim, device=self.device)
            else:
                out_dict = self.pool_model(graph, etypes=pd.etypes,
                                           in_node_types=pd.in_types,
                                           out_node_types=pd.out_types,
                                           feat_dict=plan_feats)

                out = out_dict['plan']
            pooled_feats.append(out)
        # Workaround if no join features are given: Append zero vector
        if len(pooled_feats) < 4:
            pooled_feats.append(torch.zeros(graph.num_nodes('plan'), self.hidden_dim, device=self.device))
        return torch.cat(pooled_feats, dim=1)
