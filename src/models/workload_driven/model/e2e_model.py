import torch
from torch import nn
from torch.nn import ReLU

from models.workload_driven.dataset.dataset_creation import InputDims
from models.workload_driven.model.tree_lstm import LstmConv
from classes.classes import E2EModelConfig, TPoolModelConfig, TlstmModelConfig, TestSumModelConfig
from training import losses
from models.zeroshot.message_aggregators.pooling import PoolingType, PoolingConv
from models.zeroshot.zero_shot_model import PassDirection


class E2EModel(nn.Module):

    def __init__(self,
                 model_config: E2EModelConfig,
                 input_dims: InputDims,
                 device='cpu'):

        super().__init__()

        self.hidden_dim = model_config.hidden_dim_pred + model_config.hidden_dim_plan
        self.hidden_dim_pred = model_config.hidden_dim_pred
        self.hidden_dim_plan = model_config.hidden_dim_plan
        self.input_dim_pred = input_dims.input_dim_pred
        self.input_dim_plan = input_dims.input_dim_plan
        self.device = device
        self.label_norm = None

        # original architecture by Sun et al.
        if isinstance(model_config, TPoolModelConfig):
            self.tree_models = nn.ModuleDict({
                'and': PoolingConv(PoolingType.MIN),
                'or': PoolingConv(PoolingType.MAX),
                'intra_plan': LstmConv(hidden_dim=self.hidden_dim)
            })

        # uses LSTMs for predicate encoding
        elif isinstance(model_config, TlstmModelConfig):
            self.tree_models = nn.ModuleDict({
                'and': LstmConv(hidden_dim=model_config.hidden_dim_pred),
                'or': LstmConv(hidden_dim=model_config.hidden_dim_pred),
                'intra_plan': LstmConv(hidden_dim=self.hidden_dim)
            })

        # just for testing
        elif isinstance(model_config, TestSumModelConfig):
            self.tree_models = nn.ModuleDict({
                'and': PoolingConv(PoolingType.MIN),
                'or': PoolingConv(PoolingType.MAX),
                'intra_plan': PoolingConv(PoolingType.SUM)
            })

        else:
            raise NotImplementedError

        self.loss_fxn = losses.__dict__[model_config.loss_class_name](self, **model_config.loss_class_kwargs)

        self.pred_encoding = nn.Sequential(
            *[nn.BatchNorm1d(input_dims.input_dim_pred),
              nn.Linear(input_dims.input_dim_pred, model_config.hidden_dim_pred),
              ReLU()])

        self.plan_encoding = nn.Sequential(
            *[nn.Linear(input_dims.input_dim_plan, model_config.hidden_dim_plan),
              ReLU()])

        self.est_layer = nn.Sequential(
            *[nn.Linear(self.hidden_dim, self.hidden_dim),
              ReLU(),
              nn.Linear(self.hidden_dim, self.hidden_dim),
              ReLU(),
              nn.Linear(self.hidden_dim, 1)])

    def forward(self, input):
        """
        Returns logits for output classes
        """
        plan_graph, features, pred_graph, plan_pred_mapping = input

        # encode predicates (including message passing)
        plan_pred_hidden = self.encode_predicates(features, pred_graph, plan_pred_mapping)

        # encode plan nodes and append predicates to feature vectors of plan nodes
        enc_plan_feats = dict()
        for node_type, plan_feat in features.items():
            if node_type not in plan_pred_hidden:
                continue
            enc_plan_feat = self.plan_encoding(plan_feat)
            pred_feat = plan_pred_hidden[node_type]
            enc_plan_feat = torch.cat((enc_plan_feat, pred_feat), dim=1)
            enc_plan_feats[node_type] = enc_plan_feat

        # bottom-up pass on plan nodes
        out = self.plan_message_passing(plan_graph, enc_plan_feats)

        # final estimation layer to predict runtime
        out = self.est_layer(out)
        return out

    def encode_predicates(self, features, pred_graph, plan_pred_mapping):
        # encode base predicates
        base_pred_hidden = self.pred_encoding(features['pred'])
        # message passing along logical and/or
        pred_hidden_dict = self.predicate_message_passing(pred_graph, base_pred_hidden)
        # predicate feature vector for plan nodes
        plan_pred_hidden = {
            node_t: torch.zeros((feats.size(0), base_pred_hidden.size(1)), device=self.device)
            for node_t, feats in features.items() if node_t.startswith('plan')
        }
        pred_types = {
            'and': 1,
            'or': -1,
            'pred': 0
        }
        max_depth = plan_pred_mapping[:, 0].max()
        # [d_u, u_node_id, pred_id, pred_op]
        for node_type, pred_op in pred_types.items():
            for d in range(max_depth + 1):
                rel_idx = (plan_pred_mapping[:, 3] == pred_op) & (plan_pred_mapping[:, 0] == d)
                pred_idxs = plan_pred_mapping[rel_idx, 2]
                plan_idxs = plan_pred_mapping[rel_idx, 1]
                plan_pred_hidden[f'plan{d}'][plan_idxs] = pred_hidden_dict[node_type][pred_idxs]

        return plan_pred_hidden

    def plan_message_passing(self, plan_graph, plan_feats):
        # one pass
        pass_directions = []
        if plan_graph.max_depth is not None:
            # intra_pred from deepest node to top node
            for d in reversed(range(plan_graph.max_depth + 1)):
                pd = PassDirection(model_name='intra_plan',
                                   g=plan_graph,
                                   e_name='intra_plan',
                                   n_dest=f'plan{d}',
                                   allow_empty=True)
                pd.out_types = [f'plan{d}']
                pass_directions.append(pd)

        # make sure all edge types are considered in the message passing
        plan_feats = self.execute_pass(plan_feats, pass_directions, plan_graph)

        out = plan_feats['plan0']

        return out

    def predicate_message_passing(self, pred_graph, pred_feats):
        feat_dict = {
            'pred': pred_feats,
        }
        for log_type in ['and', 'or']:
            if log_type not in pred_graph.ntypes:
                continue

            no_nodes = pred_graph.number_of_nodes(log_type)
            init_feats = torch.zeros((no_nodes, pred_feats.shape[1]), device=self.device)
            feat_dict[log_type] = init_feats

        # one pass:
        pass_directions = []
        if pred_graph.max_depth is not None:
            # intra_pred from deepest node to top node
            for d in reversed(range(pred_graph.max_depth)):
                pass_directions += [PassDirection(model_name=v_node_t,
                                                  g=pred_graph,
                                                  e_name=f'{v_node_t}_d{d}',
                                                  allow_empty=True) for v_node_t in ['and', 'or']]

        # make sure all edge types are considered in the message passing
        feat_dict = self.execute_pass(feat_dict, pass_directions, pred_graph)

        return feat_dict

    def execute_pass(self, feat_dict, pass_directions, pred_graph):
        combined_e_types = set()
        for pd in pass_directions:
            combined_e_types.update(pd.etypes)
        # In case of single plan nodes or predicate nodes, dummy edges were introduced (see batching).
        # So this assertion does not hold true anymore.
        # assert combined_e_types == set(pred_graph.canonical_etypes)
        for pd in pass_directions:
            # can happen if one depth does not exist for both AND & OR
            if len(pd.etypes) == 0:
                continue
            out_dict = self.tree_models[pd.model_name](pred_graph, etypes=pd.etypes,
                                                       in_node_types=pd.in_types,
                                                       out_node_types=pd.out_types,
                                                       feat_dict=feat_dict)
            for out_type, hidden_out in out_dict.items():
                feat_dict[out_type] = hidden_out
        return feat_dict
