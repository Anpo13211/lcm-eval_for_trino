import dgl
import numpy as np

from tests.utils import message_passing


def test_model_message_passing():
    """
    Test the message passing in the zero shot models. Each node type is one hot encoded. The forward passes are modified
    to simply add up the feature vectors of the child nodes. We should thus obtain a count of all node types in the
    graph.
    :return:
    """
    # define the edges among different node types in the graph
    data_dict = dict()
    # col0
    # col1 -> out_col0 -> plan0_0
    # col2 -> out_col1 -> plan1_1
    data_dict[('column', 'col_output_col', 'output_column')] = [(0, 0), (1, 0), (2, 1)]
    data_dict[('output_column', 'to_plan', 'plan0')] = [(0, 0)]
    data_dict[('output_column', 'to_plan', 'plan1')] = [(1, 0)]
    # filter_column -> plan1_0
    data_dict[('filter_column', 'to_plan', 'plan1')] = [(0, 0)]
    # table -> plan0_0
    data_dict[('table', 'to_plan', 'plan0')] = [(0, 0)]
    data_dict[('plan1', 'intra_plan', 'plan0')] = [(0, 0)]
    data_dict[('plan2', 'intra_plan', 'plan1')] = [(0, 0), (1, 0)]

    # derive a heterograph
    g = dgl.heterograph(data_dict)
    g.max_depth = 2
    g.max_pred_depth = None

    no_nodes_per_type, model_out = message_passing(g)
    # output should be the number of nodes per node type
    assert np.allclose(model_out, no_nodes_per_type)


def test_model_message_passing_with_nested_predicates():
    """
    Similar to previous test but with nested predicates
    :return:
    """
    # define the edges among different node types in the graph
    data_dict = dict()
    # col0
    # col1 -> out_col0 -> plan0_0
    # col2 -> out_col1 -> plan1_1
    data_dict[('column', 'col_output_col', 'output_column')] = [(0, 0), (1, 0), (2, 1)]
    data_dict[('output_column', 'to_plan', 'plan0')] = [(0, 0)]
    data_dict[('output_column', 'to_plan', 'plan1')] = [(1, 0)]

    # filter_column_0, filter_column_1 -> pred_depth_1 -> pred_depth_0
    data_dict[('filter_column', 'intra_predicate', 'logical_pred_1')] = [(0, 0), (1, 0)]
    data_dict[('logical_pred_1', 'intra_predicate', 'logical_pred_0')] = [(0, 0)]
    # filter_column_2 -> pred_depth_0
    data_dict[('filter_column', 'intra_predicate', 'logical_pred_0')] = [(2, 0)]
    data_dict[('logical_pred_0', 'to_plan', 'plan1')] = [(0, 0)]
    # filter_column_3 -> plan0
    data_dict[('filter_column', 'to_plan', 'plan0')] = [(3, 0)]

    # table -> plan0_0
    data_dict[('table', 'to_plan', 'plan0')] = [(0, 0)]
    data_dict[('plan1', 'intra_plan', 'plan0')] = [(0, 0)]
    data_dict[('plan2', 'intra_plan', 'plan1')] = [(0, 0), (1, 0)]

    # derive a heterograph
    g = dgl.heterograph(data_dict)
    g.max_depth = 2
    g.max_pred_depth = 2

    no_nodes_per_type, model_out = message_passing(g)
    # output should be the number of nodes per node type
    assert np.allclose(model_out, no_nodes_per_type)


def test_model_message_passing_no_filters():
    """
    Similar to previous tests but this time without any filter column.
    :return:
    """
    # define the edges among different node types in the graph
    data_dict = dict()
    # col0
    # col1 -> out_col0 -> plan0_0
    # col2 -> out_col1 -> plan1_1
    data_dict[('column', 'col_output_col', 'output_column')] = [(0, 0), (1, 0), (2, 1)]
    data_dict[('output_column', 'to_plan', 'plan0')] = [(0, 0)]
    data_dict[('output_column', 'to_plan', 'plan1')] = [(1, 0)]
    # table -> plan0_0
    data_dict[('table', 'to_plan', 'plan0')] = [(0, 0)]
    data_dict[('plan1', 'intra_plan', 'plan0')] = [(0, 0)]
    data_dict[('plan2', 'intra_plan', 'plan1')] = [(0, 0), (1, 0)]

    # derive a heterograph
    g = dgl.heterograph(data_dict)
    g.max_depth = 2
    g.max_pred_depth = 0

    no_nodes_per_type, model_out = message_passing(g)
    # output should be the number of nodes per node type
    assert np.allclose(model_out, no_nodes_per_type)
