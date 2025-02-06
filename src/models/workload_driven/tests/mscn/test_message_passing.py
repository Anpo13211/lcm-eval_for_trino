import dgl
import numpy as np

from models.workload_driven.dataset.plan_tree_batching import tensorize_feats
from models.workload_driven.model.mscn_model import MSCNModel


def test_plan_message():
    # averaging of nodes
    m = MSCNModel(hidden_dim=2, input_dim_table=2, input_dim_pred=2, input_dim_join=2,
                  input_dim_agg=2, loss_class_name='QLoss', loss_class_kwargs=dict(), device='cpu')

    num_nodes_dict = {
        'plan': 2,
        'table': 4,
        'pred': 4,
        'agg': 4,
        'join': 4,
    }
    features = dict()
    for i, k in enumerate(m.pool_node_types):
        if k == 'plan':
            continue
        features[k] = [[1 + i, i], [i, 1 + i], [1 + i, i], [i, 1 + i]]

    edge_dict = {
        etype: [(0, 0), (1, 0), (2, 1), (3, 1)]
        for etype in [('table', 'table_plan', 'plan'), ('pred', 'pred_plan', 'plan'), ('agg', 'agg_plan', 'plan'),
                      ('join', 'join_plan', 'plan')]
    }

    graph = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
    features = tensorize_feats(features)

    out = m.pool_set_features(graph, features)
    exp_out = np.array([[0.5000, 0.5000, 1.5000, 1.5000, 2.5000, 2.5000, 3.5000, 3.5000],
                        [0.5000, 0.5000, 1.5000, 1.5000, 2.5000, 2.5000, 3.5000, 3.5000]])
    assert np.allclose(exp_out, out.numpy())
