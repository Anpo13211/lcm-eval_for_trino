from typing import Tuple, List, Optional
import numpy as np
import torch
from torch import Tensor


def floyd_warshall_transform(adjacency_matrix: np.ndarray) -> np.ndarray:
    """This transforms the adjacency matrix into a distance matrix using the Floyd-Warshall algorithm."""
    num_rows, num_cols = adjacency_matrix.shape
    assert num_rows == num_cols
    transformed_matrix = adjacency_matrix.copy().astype('long')
    for i in range(num_rows):
        for j in range(num_cols):
            if i == j:
                transformed_matrix[i][j] = 0
            elif transformed_matrix[i][j] == 0:
                transformed_matrix[i][j] = 60

    for k in range(num_rows):
        for i in range(num_rows):
            for j in range(num_rows):
                transformed_matrix[i][j] = min(transformed_matrix[i][j],
                                               transformed_matrix[i][k] + transformed_matrix[k][j])
    return transformed_matrix


class QueryFormerBatch:
    node_distances: Tensor
    attention_bias: Tensor
    node_features: Tensor
    node_heights: Tensor

    def __init__(self,
                 attention_bias: Tensor,
                 node_distances: Tensor,
                 node_heights: Tensor,
                 node_features: Tensor,
                 labels: object = None) -> None:
        super(QueryFormerBatch, self).__init__()
        self.node_heights = node_heights
        self.node_features = node_features
        self.labels = labels
        self.attention_bias = attention_bias
        self.node_distances = node_distances

    def __len__(self):
        return self.in_degree.size(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def collator(batch: List[Tuple[dict]]) -> Tuple[QueryFormerBatch, Tuple]:
    # Get labels
    labels = batch[1]

    # Get features / model inputs
    node_features = torch.cat([s['features'] for s in batch[0]])
    attention_bias = torch.cat([s['attention_bias'] for s in batch[0]])
    relative_positions = torch.cat([s['rel_pos'] for s in batch[0]])
    node_heights = torch.cat([s['node_heights'] for s in batch[0]])
    return QueryFormerBatch(attention_bias, relative_positions, node_heights, node_features), labels


class TreeNode:
    def __init__(self, operator_type_id: int,
                 filter_info: np.ndarray,
                 filter_mask: np.ndarray,
                 sample_bitmap_vec: np.ndarray,
                 histogram_info: np.ndarray,
                 table_name: int):
        self.operator_type_id = operator_type_id
        self.encoded_filter_info = filter_info
        self.filter_mask = filter_mask
        self.sample_bitmap_vec = sample_bitmap_vec
        self.histogram_info = histogram_info
        self.table_name = table_name

        self.children: List[TreeNode] = []
        self.parent: Optional[TreeNode] = None
        self.feature_vector = None

    def add_child(self, tree_node):
        self.children.append(tree_node)

    def __str__(self):
        return '{} with {}, {}, children'.format(self.operator_type_id,
                                                 self.encoded_filter_info,
                                                 len(self.children))

    def __repr__(self):
        return self.__str__()

    def featurize(self) -> np.ndarray:
        output = np.concatenate([np.array(self.operator_type_id).reshape(1),
                                 self.encoded_filter_info.flatten(),
                                 self.filter_mask,
                                 self.histogram_info.flatten(),
                                 np.array(self.table_name).reshape(1),
                                 self.sample_bitmap_vec])
        return output

    @staticmethod
    def print_nested(node, indent=0):
        print('--' * indent + '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str,
                                                                    len(node.children)))
        for k in node.children:
            TreeNode.print_nested(k, indent + 1)
