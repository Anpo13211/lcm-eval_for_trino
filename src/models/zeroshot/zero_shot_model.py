from torch import nn

from classes.classes import ZeroShotModelConfig
from models.zeroshot.message_aggregators import message_aggregators
from models.zeroshot.utils.fc_out_model import FcOutModel
from models.zeroshot.utils.node_type_encoder import NodeTypeEncoder


class ZeroShotModel(FcOutModel):
    """
    A zero-shot cost model that predicts query runtimes on unseen databases out-of-the-box without retraining.
    """

    def __init__(self, model_config: ZeroShotModelConfig, device='cpu', feature_statistics=None,
                 add_tree_model_types=None, prepasses=None, plan_featurization=None, encoders=None, label_norm=None,
                 allow_empty_edges=False):

        super().__init__(output_dim=model_config.output_dim,
                         input_dim=model_config.hidden_dim,
                         final_out_layer=True,
                         **model_config.final_mlp_kwargs)

        self.label_norm = label_norm
        self.test = False
        self.skip_message_passing = False
        self.device = device
        self.hidden_dim = model_config.hidden_dim
        self.allow_empty_edges = allow_empty_edges

        # use different models per edge type
        # ノードタイプだけでなく、エッジタイプにも異なるモデルを適用する
        if add_tree_model_types is None:
            add_tree_model_types = []
        tree_model_types = add_tree_model_types + ['to_plan', 'intra_plan', 'intra_pred']
        self.tree_models = nn.ModuleDict({
            node_type: message_aggregators.__dict__[model_config.tree_layer_name](hidden_dim=self.hidden_dim, **model_config.tree_layer_kwargs)
            for node_type in tree_model_types
        })

        # these message passing steps are performed in the beginning (dependent on the concrete database system at hand)
        self.prepasses = prepasses if prepasses is not None else []

        if plan_featurization is not None:
            self.plan_featurization = plan_featurization
            # different models to encode plans, tables, columns, filter_columns and output_columns
            model_config.node_type_kwargs.update(output_dim=model_config.hidden_dim)
            self.node_type_encoders = nn.ModuleDict({
                enc_name: NodeTypeEncoder(features, feature_statistics, **model_config.node_type_kwargs)
                for enc_name, features in encoders
            })

    def encode_node_types(self, g, features):
        """
        Initializes the hidden states based on the node type specific models.
        """
        # initialize hidden state per node type
        hidden_dict = dict()
        for node_type, input_features in features.items():
            # encode all plans with same model
            if node_type not in self.node_type_encoders.keys():
                if node_type.startswith('logical_pred'):
                    node_type_m = self.node_type_encoders['logical_pred']
                elif node_type.startswith('plan'):
                    node_type_m = self.node_type_encoders['plan']
                else:
                    # その他のノードタイプ（column, table, filter_columnなど）はplanモデルを使用
                    node_type_m = self.node_type_encoders['plan']
            else:
                node_type_m = self.node_type_encoders[node_type]
            hidden_dict[node_type] = node_type_m(input_features)

        return hidden_dict

    def forward(self, input):
        """
        Returns logits for output classes
        """
        graph, features = input
        features = self.encode_node_types(graph, features)
        out = self.message_passing(graph, features)

        return out

    def message_passing(self, g, feat_dict, allow_empty_edges=None):
        """
        Bottom-up message passing on the graph encoding of the queries in the batch. Returns the hidden states of the
        root nodes.
        
        Args:
            g: DGL graph
            feat_dict: Feature dictionary
            allow_empty_edges: If True, allows message passing steps with no edges (for Trino compatibility).
                              If None, uses self.allow_empty_edges
        """
        # 引数が指定されていない場合はインスタンス変数を使用
        if allow_empty_edges is None:
            allow_empty_edges = self.allow_empty_edges

        # also allow skipping this for testing
        if not self.skip_message_passing:
            # all passes before predicates, to plan and intra_plan passes
            pass_directions = [
                PassDirection(g=g, allow_empty=allow_empty_edges, **prepass_kwargs)
                for prepass_kwargs in self.prepasses
            ]

            if g.max_pred_depth is not None and g.max_pred_depth > 0:
                # intra_pred from deepest node to top node
                for d in reversed(range(g.max_pred_depth)):
                    pd = PassDirection(model_name='intra_pred',
                                       g=g,
                                       e_name='intra_predicate',
                                       n_dest=f'logical_pred_{d}',
                                       allow_empty=allow_empty_edges)
                    pass_directions.append(pd)

            # filter_columns & output_columns to plan
            pass_directions.append(PassDirection(model_name='to_plan', g=g, e_name='to_plan', allow_empty=allow_empty_edges))

            # intra_plan from deepest node to top node
            for d in reversed(range(g.max_depth)):
                pd = PassDirection(model_name='intra_plan',
                                   g=g,
                                   e_name='intra_plan',
                                   n_dest=f'plan{d}',
                                   allow_empty=allow_empty_edges)
                pass_directions.append(pd)

            # make sure all edge types are considered in the message passing
            # Note: Trino版ではエッジが存在しない場合があるため、allow_empty_edges=Trueの場合はスキップ
            if not allow_empty_edges:
                combined_e_types = set()
                for pd in pass_directions:
                    combined_e_types.update(pd.etypes)
                assert combined_e_types == set(g.canonical_etypes)

            for pd in pass_directions:
                if len(pd.etypes) > 0:
                    out_dict = self.tree_models[pd.model_name](g, etypes=pd.etypes,
                                                               in_node_types=pd.in_types,
                                                               out_node_types=pd.out_types,
                                                               feat_dict=feat_dict)
                    for out_type, hidden_out in out_dict.items():
                        feat_dict[out_type] = hidden_out

        # compute top nodes of dags
        out = feat_dict['plan0']

        # feed them into final feed forward network
        if not self.test:
            out = self.fcout(out)

        return out


class PassDirection:
    """
    Defines a message passing step on the encoded query graphs.
    """
    def __init__(self, model_name, g, e_name=None, n_dest=None, allow_empty=False):
        """
        Initializes a message passing step.
        :param model_name: which edge model should be used to combine the messages
        :param g: the graph on which the message passing should be performed
        :param e_name: edges are defined by triplets: (src_node_type, edge_type, dest_node_type). Only incorporate edges
            in the message passing step where edge_type=e_name
        :param n_dest: further restrict the edges that are incorporated in the message passing by the condition
            dest_node_type=n_dest
        :param allow_empty: allow that no edges in the graph qualify for this message passing step.
            Otherwise, this will raise an error.
        """
        self.etypes = set()
        self.in_types = set()
        self.out_types = set()
        self.model_name = model_name

        for curr_n_src, curr_e_name, curr_n_dest in g.canonical_etypes:
            if e_name is not None and curr_e_name != e_name:
                continue

            if n_dest is not None and curr_n_dest != n_dest:
                continue

            self.etypes.add((curr_n_src, curr_e_name, curr_n_dest))
            self.in_types.add(curr_n_src)
            self.out_types.add(curr_n_dest)

        self.etypes = list(self.etypes)
        self.in_types = list(self.in_types)
        self.out_types = list(self.out_types)
        if not allow_empty:
            assert len(self.etypes) > 0, f"No nodes in the graph qualify for e_name={e_name}, n_dest={n_dest}"
