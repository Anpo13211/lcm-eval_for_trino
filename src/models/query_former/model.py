from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from classes.classes import QueryFormerModelConfig
from models.workload_driven.dataset.dataset_creation import InputDims
from training import losses


class FinalPredictionLayer(nn.Module):
    def __init__(self,
                 in_feature: int = 69,
                 hid_units: int = 256,
                 contract: int = 1,
                 mid_layers: bool = True,
                 res_con: bool = True):

        super(FinalPredictionLayer, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        self.out_mlp_1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp_1 = nn.Linear(hid_units, hid_units // contract)
        self.mid_mlp_2 = nn.Linear(hid_units // contract, hid_units)
        self.out_mlp_2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        hid = self.out_mlp_1(features)
        hid = F.relu(hid)

        if self.mid_layers:
            mid = F.relu(self.mid_mlp_1(hid))
            mid = F.relu(self.mid_mlp_2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = self.out_mlp_2(hid)
        return out


class FeatureEmbedding(nn.Module):
    def __init__(self,
                 feature_statistics: dict,
                 embedding_size: int = 32,
                 histogram_bin_number: int = 50,
                 max_filter_number: int = 3):
        # Initialize model
        super(FeatureEmbedding, self).__init__()
        self.embed_size = embedding_size
        self.bin_number = histogram_bin_number
        self.max_filter_number = max_filter_number
        small_embed_size = embedding_size // 8 + 1

        # Define embedding layers
        num_operators = feature_statistics['op_name']['no_vals']
        self.operator_type_embedding = nn.Embedding(num_operators + 2, embedding_size)

        num_tables = feature_statistics['tablename']['no_vals']
        self.table_embedding = nn.Embedding(num_tables + 2, embedding_size)

        num_columns = int(feature_statistics['columns']['max'])
        self.column_embedding = nn.Embedding(num_columns + 2, embedding_size)

        num_filter_operators = feature_statistics['operator']['no_vals']
        self.filter_operator_embedding = nn.Embedding(num_filter_operators + 2, embedding_size // 8)

        # Define linear layers
        self.linear_filter = nn.Linear(embedding_size + small_embed_size, embedding_size + small_embed_size)
        self.linear_filter_2 = nn.Linear(embedding_size + small_embed_size, embedding_size + small_embed_size)
        self.linear_type = nn.Linear(embedding_size, embedding_size)
        self.linear_sample = nn.Linear(1000, embedding_size)
        self.linear_hist = nn.Linear(histogram_bin_number, embedding_size)
        self.linear_final = nn.Linear(embedding_size * 4 + small_embed_size, embedding_size * 4 + small_embed_size)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        splits = [1,
                  self.max_filter_number * 3,
                  self.max_filter_number,
                  self.bin_number * self.max_filter_number,
                  1,
                  1000]
        operator_type_id, filter_ids, filter_masks, hists, table, sample = torch.split(input_feature, splits, dim=-1)

        type_embedding = self.get_operator_type_embedding(operator_type_id)
        filter_embedding = self.get_filter_embedding(filter_ids, filter_masks)
        histogram_embedding = self.get_histogram_embedding(hists, filter_masks)
        table_embedding = self.get_table_embedding(table, sample)

        out = torch.cat((
            type_embedding,
            filter_embedding,
            table_embedding,
            histogram_embedding),
            dim=1)

        out = F.leaky_relu(self.linear_final(out))
        return out

    def get_operator_type_embedding(self, type_id):
        emb = self.operator_type_embedding(type_id.long())
        return emb.squeeze(1)

    def get_table_embedding(self, table, sample):
        emb = self.table_embedding(table.long()).squeeze(1)
        emb += self.linear_sample(sample)
        return emb

    def get_histogram_embedding(self, hists, filters_mask):
        # batch * 50 * 3
        histExpand = hists.view(-1, self.bin_number, self.max_filter_number).transpose(1, 2)

        emb = self.linear_hist(histExpand)
        emb[~filters_mask.bool()] = 0.  # mask out space holder

        # avg by # of filters
        num_filters = torch.sum(filters_mask, dim=1)
        total = torch.sum(emb, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg

    def get_filter_embedding(self, filters_id, filters_mask):
        # get filters, then apply mask
        filter_expand = filters_id.view(-1, self.max_filter_number, 3)
        column_ids = filter_expand[:, :, 0].long()
        operator_ids = filter_expand[:, :, 1].long()
        literals = filter_expand[:, :, 2].unsqueeze(-1)  # b by 3 by 1

        embedded_columns = self.column_embedding(column_ids)
        embedded_operators = self.filter_operator_embedding(operator_ids)

        concat = torch.cat((embedded_columns, embedded_operators, literals), dim=-1)
        concat = F.leaky_relu(self.linear_filter(concat))
        concat = F.leaky_relu(self.linear_filter_2(concat))

        # apply mask
        concat[~filters_mask.bool()] = 0.

        # avg by # of filters
        num_filters = torch.sum(filters_mask, dim=1)
        total = torch.sum(concat, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg


class QueryFormer(nn.Module):
    def __init__(self,
                 config: QueryFormerModelConfig,
                 input_dims: InputDims,
                 feature_statistics: dict,
                 label_norm):

        super(QueryFormer, self).__init__()

        # Initialize Model
        hidden_dim = config.embedding_size * 4 + config.embedding_size // 8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = config.head_size
        self.device = config.device
        self.label_norm = None
        self.loss_fxn = losses.__dict__[config.loss_class_name](self, **config.loss_class_kwargs)

        # Define encoders for features
        self.feature_embedding_layer = FeatureEmbedding(embedding_size=config.embedding_size,
                                                        histogram_bin_number=config.histogram_bin_number,
                                                        feature_statistics=feature_statistics,
                                                        max_filter_number=config.max_num_filters)

        # Define encoders for further inputs
        self.relative_position_encoder = nn.Embedding(64, config.head_size, padding_idx=0)
        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.join_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        # Define intermediate transformer layers
        self.input_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, config.ffn_dim, config.dropout, config.attention_dropout_rate, config.head_size)
             for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, config.head_size)

        # Define prediction layer
        self.prediction_layer = FinalPredictionLayer(in_feature=hidden_dim, hid_units=config.hidden_dim_prediction)

    def forward(self, input_batch: Tuple) -> torch.Tensor:
        # Read batch
        node_features, joins, attention_bias, node_distances, node_heights = input_batch
        batch_size, number_nodes = node_features.size()[:2]

        # Shape the attention bias
        tree_attn_bias = attention_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)

        # Relative position encoding and rearranging dimensions
        # [n_batch, n_node, n_node, n_head] -> [# n_batch, n_head, n_node, n_node]
        node_distances_bias = self.relative_position_encoder(node_distances).permute(0, 3, 1, 2)

        # Adding relative position encoding to attention bias but excluding the first position in the last two dimensions
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + node_distances_bias

        # Reset relative position encoding
        # Reshape the weights of super token virtual distance to the shape of (1, headsize, 1).
        # This distance is an embedding layer that represents the distance to the super token.
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)

        # Adding reshaped weigths to batches and heads ot of tree_attn_bias,
        # but only for the first position in the last dimension
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        # Embed features
        # features_view = node_features.view(-1, 1165)
        features_view = node_features.flatten(start_dim=0, end_dim=1)

        embedded_features = self.feature_embedding_layer(features_view).view(batch_size, -1, self.hidden_dim)

        # Add height encoding to the embedded features
        encoded_heights = self.height_encoder(node_heights)
        embedded_features = embedded_features + encoded_heights

        # Add join encodings to the embedded features
        joins = joins.repeat(1, 6)
        embedded_joins = self.join_encoder(joins.long())
        embedded_features = embedded_features + embedded_joins

        # Add super token to the embedded features
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        super_node_feature = torch.cat([super_token_feature, embedded_features], dim=1)

        # Pass through transformer layers
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)

        # Final layer normalization
        out = self.final_ln(output)
        out = self.prediction_layer(out[:, 0, :])
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_dropout_rate: float, head_size: int):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None) -> torch.Tensor:
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout_rate: float,
                 attention_dropout_rate: float, head_size: int):
        super(EncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
