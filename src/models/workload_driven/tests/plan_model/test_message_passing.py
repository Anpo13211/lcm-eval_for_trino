import collections

import dgl
import numpy as np
import torch

from models.workload_driven.dataset.plan_tree_batching import tensorize_feats
from models.workload_driven.model.plan_model import PlanModel


def test_pred_message_passing():
    # first predicate
    #                   and0 [d0] = 1
    #       or0 [d1] = 1             and1 [d1] = 2
    # p0[0]       p1[1]         p2[2]            p3[3]
    #
    # simple predicate
    # p4 [4]

    pred_feats = [[0, i] for i in range(5)]
    pred_feats = torch.from_numpy(np.array(pred_feats, dtype=np.float32))

    # define edges
    data_dict = collections.defaultdict(list)
    data_dict[('pred', 'or_d1', 'or')] = [(0, 0), (1, 0)]
    data_dict[('pred', 'and_d1', 'and')] = [(2, 1), (3, 1)]
    data_dict[('and', 'and_d0', 'and')] = [(1, 0)]
    data_dict[('or', 'and_d0', 'and')] = [(0, 0)]

    num_nodes_dict = {
        'pred': 5,
        'and': 2,
        'or': 1,
    }
    m = create_test_model()

    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    graph.max_depth = 2

    feat_dict = m.predicate_message_passing(graph, pred_feats)

    exp_feat_dict = {
        'pred': torch.from_numpy(np.array(pred_feats)),
        'and': torch.from_numpy(np.array([[0, 1.], [0, 2.]], dtype=np.float32)),
        'or': torch.from_numpy(np.array([[0, 1.]], dtype=np.float32)),
    }
    for k, v in exp_feat_dict.items():
        assert torch.allclose(v, feat_dict[k])


def test_plan_message_passing():
    #                         plan0_0
    #       plan1_0                         plan1_1
    # plan2_0       plan2_1         plan2_2            plan2_3

    num_nodes_dict = {
        'plan2': 4,
        'plan1': 2,
        'plan0': 1,
    }
    plan_feats = dict()
    max_depth = 2
    for d in range(max_depth + 1):
        feats = [[1 if i == d else 0 for i in range(max_depth + 1)] for _ in range(num_nodes_dict[f'plan{d}'])]
        assert len(feats) == num_nodes_dict[f'plan{d}']

        plan_feats[f'plan{d}'] = feats
    plan_feats = tensorize_feats(plan_feats)

    # define edges
    data_dict = collections.defaultdict(list)
    data_dict[('plan2', 'intra_plan', 'plan1')] = [(0, 0), (1, 0), (2, 1), (3, 1)]
    data_dict[('plan1', 'intra_plan', 'plan0')] = [(0, 0), (1, 0)]

    m = create_test_model()

    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    graph.max_depth = 2

    out = m.plan_message_passing(graph, plan_feats)

    exp_out = [num_nodes_dict[f'plan{d}'] for d in range(max_depth + 1)]
    assert np.allclose(exp_out, out.numpy())


def test_plan_message_passing_lstm():
    #                         plan0_0
    #       plan1_0                         plan1_1
    # plan2_0       plan2_1         plan2_2            plan2_3

    num_nodes_dict = {
        'plan2': 4,
        'plan1': 2,
        'plan0': 1,
    }
    plan_feats = dict()
    max_depth = 2
    for d in range(max_depth + 1):
        feats = [[0, 0, 1, 1] for _ in range(num_nodes_dict[f'plan{d}'])]
        assert len(feats) == num_nodes_dict[f'plan{d}']

        plan_feats[f'plan{d}'] = feats
    plan_feats = tensorize_feats(plan_feats)

    # define edges
    data_dict = collections.defaultdict(list)
    data_dict[('plan2', 'intra_plan', 'plan1')] = [(0, 0), (1, 0)]
    data_dict[('plan1', 'intra_plan', 'plan0')] = [(0, 0), (1, 0)]

    m = create_test_model(model_name='TPool')

    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    graph.max_depth = 2

    out = m.plan_message_passing(graph, plan_feats)
    assert out.size(0) == 1 and out.size(1) == 4


def create_test_model(model_name='TestSum'):
    m = PlanModel(input_dim_pred=4, hidden_dim_pred=0, input_dim_plan=4, hidden_dim_plan=4, device='cpu',
                  model_name=model_name, loss_class_name='QLoss', loss_class_kwargs=dict())
    return m
