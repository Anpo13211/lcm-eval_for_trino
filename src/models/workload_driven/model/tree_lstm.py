import dgl.function as fn
import torch
from torch import nn


class LstmConv(nn.Module):

    def __init__(self, hidden_dim=4):
        super().__init__()
        self.message_dim = hidden_dim // 2
        self.lstm_input_dim = hidden_dim
        # input size, hidden size
        self.lstm = nn.LSTMCell(self.lstm_input_dim, self.message_dim)

    def forward(self, graph=None, etypes=None, in_node_types=None, out_node_types=None, feat_dict=None):
        with graph.local_scope():
            graph.ndata['h'] = feat_dict

            x_t = graph.ndata['h']
            if len(etypes) > 0:
                # G_t and R_t are left and right halfs of hidden feature vectors
                graph.multi_update_all({etype: (fn.copy_u('h', 'm'), fn.mean('m', 'ft')) for etype in etypes},
                                       cross_reducer='mean')
                # nodes without children have initialization zero which is correct
                rst = graph.ndata['ft']

            else:
                rst = {n_type: torch.zeros_like(feat_dict[n_type]) for n_type in out_node_types}

            assert len(out_node_types) == 1
            out_type = list(out_node_types)[0]
            assert rst[out_type].shape[1] == self.message_dim * 2
            # slice in half to obtain G_t and R_t
            G_t = rst[out_type][:, :self.message_dim]
            R_t = rst[out_type][:, self.message_dim:]
            x_t = x_t[out_type]

            assert G_t.size(1) == R_t.size(1) == self.message_dim
            assert x_t.size(1) == self.lstm_input_dim
            G_t_2, R_t_2 = self.lstm(x_t, (G_t, R_t))
            new_hidden = torch.cat((G_t_2, R_t_2), dim=1)

            out_dict = {out_type: new_hidden}

            return out_dict
