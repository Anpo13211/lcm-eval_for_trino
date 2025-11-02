"""
Trino Zero-Shot Model
Trinoクエリプラン用のZero-Shotモデル
"""

from torch import nn

from classes.classes import ZeroShotModelConfig
from models.zeroshot.message_aggregators import message_aggregators
from models.zeroshot.utils.fc_out_model import FcOutModel
from models.zeroshot.utils.node_type_encoder import NodeTypeEncoder
from models.zeroshot.zero_shot_model import PassDirection


class TrinoZeroShotModel(FcOutModel):
    """
    Trino用のZero-Shotコストモデル
    未学習のデータベースでもクエリ実行時間を予測できる
    """

    def __init__(self, model_config: ZeroShotModelConfig, device='cpu', feature_statistics=None,
                 add_tree_model_types=None, prepasses=None, plan_featurization=None, encoders=None, label_norm=None):

        super().__init__(output_dim=model_config.output_dim,
                         input_dim=model_config.hidden_dim,
                         final_out_layer=True,
                         **model_config.final_mlp_kwargs)

        self.label_norm = label_norm
        self.test = False
        self.skip_message_passing = False
        self.device = device
        self.hidden_dim = model_config.hidden_dim

        # エッジタイプごとに異なるモデルを使用
        if add_tree_model_types is None:
            add_tree_model_types = []
        tree_model_types = add_tree_model_types + ['to_plan', 'intra_plan', 'intra_pred']
        self.tree_models = nn.ModuleDict({
            node_type: message_aggregators.__dict__[model_config.tree_layer_name](
                hidden_dim=self.hidden_dim, **model_config.tree_layer_kwargs)
            for node_type in tree_model_types
        })

        # データベースシステム固有のメッセージパッシングステップ
        self.prepasses = prepasses if prepasses is not None else []

        if plan_featurization is not None:
            self.plan_featurization = plan_featurization
            # プラン、テーブル、カラム、フィルターカラム、出力カラムのエンコーダー
            model_config.node_type_kwargs.update(output_dim=model_config.hidden_dim)
            self.node_type_encoders = nn.ModuleDict({
                enc_name: NodeTypeEncoder(features, feature_statistics, **model_config.node_type_kwargs)
                for enc_name, features in encoders
            })

    def encode_node_types(self, g, features):
        """
        ノードタイプ別の特徴量エンコーディング
        """
        hidden_dict = dict()
        for node_type, input_features in features.items():
            # ノードタイプに応じて適切なエンコーダーを選択
            if node_type not in self.node_type_encoders.keys():
                if node_type.startswith('logical_pred'):
                    # 論理述語ノード（AND, OR等）
                    node_type_m = self.node_type_encoders['logical_pred']
                elif node_type.startswith('plan'):
                    # プランノード（深度別）
                    node_type_m = self.node_type_encoders['plan']
                else:
                    # その他のノードタイプ（column, table, filter_column等）
                    # デフォルトでplanエンコーダーを使用
                    if 'column' in self.node_type_encoders and 'column' in node_type:
                        node_type_m = self.node_type_encoders['column']
                    elif 'table' in self.node_type_encoders and node_type == 'table':
                        node_type_m = self.node_type_encoders['table']
                    elif 'filter_column' in self.node_type_encoders and node_type == 'filter_column':
                        node_type_m = self.node_type_encoders['filter_column']
                    else:
                        node_type_m = self.node_type_encoders['plan']
            else:
                node_type_m = self.node_type_encoders[node_type]
            
            hidden_dict[node_type] = node_type_m(input_features)

        return hidden_dict

    def forward(self, input):
        """
        フォワードパス
        
        Args:
            input: (graph, features)のタプル
        
        Returns:
            予測値（ログスケール）
        """
        graph, features = input
        features = self.encode_node_types(graph, features)
        out = self.message_passing(graph, features)

        return out

    def message_passing(self, g, feat_dict):
        """
        グラフエンコーディングのボトムアップメッセージパッシング
        ルートノードの隠れ状態を返す
        """

        if not self.skip_message_passing:
            # データベース固有のプリパス
            pass_directions = [
                PassDirection(g=g, **prepass_kwargs)
                for prepass_kwargs in self.prepasses
            ]

            # 述語の深い方から浅い方へのメッセージパッシング
            # Trinoでは (filter_column, intra_predicate, filter_column) などのエッジも存在するため、
            # n_destを指定せずにすべてのintra_predicateエッジを処理する
            if g.max_pred_depth is not None and g.max_pred_depth > 0:
                # まず、すべてのintra_predicateエッジを深度に関係なく処理
                # これにより filter_column -> filter_column のエッジも処理される
                pd_all_predicates = PassDirection(model_name='intra_pred',
                                                  g=g,
                                                  e_name='intra_predicate',
                                                  allow_empty=True)  # エッジが存在しない場合もエラーにしない
                if len(pd_all_predicates.etypes) > 0:
                    pass_directions.append(pd_all_predicates)

            # フィルターカラムと出力カラムからプランへ
            pass_directions.append(PassDirection(model_name='to_plan', g=g, e_name='to_plan'))

            # プランの深い方から浅い方へのメッセージパッシング
            for d in reversed(range(g.max_depth)):
                pd = PassDirection(model_name='intra_plan',
                                   g=g,
                                   e_name='intra_plan',
                                   n_dest=f'plan{d}')
                pass_directions.append(pd)

            # すべてのエッジタイプが考慮されているか確認
            combined_e_types = set()
            for pd in pass_directions:
                combined_e_types.update(pd.etypes)
            
            # エッジタイプの検証（デバッグ時のみ有効化）
            if hasattr(g, 'canonical_etypes') and len(g.canonical_etypes) > 0:
                missing_etypes = set(g.canonical_etypes) - combined_e_types
                if missing_etypes:
                    print(f"⚠️  未処理のエッジタイプ: {missing_etypes}")

            # メッセージパッシングの実行
            for pd in pass_directions:
                if len(pd.etypes) > 0:
                    out_dict = self.tree_models[pd.model_name](g, etypes=pd.etypes,
                                                               in_node_types=pd.in_types,
                                                               out_node_types=pd.out_types,
                                                               feat_dict=feat_dict)
                    for out_type, hidden_out in out_dict.items():
                        feat_dict[out_type] = hidden_out

        # DAGのトップノード（ルートプランノード）を取得
        out = feat_dict['plan0']

        # 最終的なフィードフォワードネットワーク
        if not self.test:
            out = self.fcout(out)

        return out

