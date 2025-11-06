import loralib as lora
import torch
import torch.nn as nn

from classes.classes import DACEModelConfig
from training import losses


class DACELora(nn.Module):
    """# create DACE model with lora"""
    def __init__(self, config: DACEModelConfig):
        super(DACELora, self).__init__()
        self.label_norm = None
        self.device = config.device
        self.config = config
        self.loss_fxn = losses.__dict__[config.loss_class_name](self, **config.loss_class_kwargs)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.node_length,
                dim_feedforward=config.hidden_dim,
                nhead=1,
                batch_first=True,
                activation=config.transformer_activation,
                dropout=config.transformer_dropout),
            num_layers=1)

        self.node_length = config.node_length
        if config.mlp_activation == "ReLU":
            self.mlp_activation = nn.ReLU()
        elif config.mlp_activation == "GELU":
            self.mlp_activation = nn.GELU()
        elif config.mlp_activation == "LeakyReLU":
            self.mlp_activation = nn.LeakyReLU()
        self.mlp_hidden_dims = [128, 64, 1]

        self.mlp = nn.Sequential(
            *[lora.Linear(self.node_length, self.mlp_hidden_dims[0], r=16),
              nn.Dropout(config.mlp_dropout),
              self.mlp_activation,
              lora.Linear(self.mlp_hidden_dims[0], self.mlp_hidden_dims[1], r=8),
              nn.Dropout(config.mlp_dropout),
              self.mlp_activation,
              lora.Linear(self.mlp_hidden_dims[1], config.output_dim, r=4)])

        self.sigmoid = nn.Sigmoid()

    def forward_batch(self, x, attn_mask=None) -> torch.Tensor:
        # change x shape to (batch, seq_len, input_size) from (batch, len)
        # one node is 18 bits
        x = x.view(x.shape[0], -1, self.node_length)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                outputs = []
                for seq, mask in zip(x, attn_mask):
                    mask = mask.to(device=seq.device)
                    outputs.append(self.transformer_encoder(seq.unsqueeze(0), mask=mask))
                out = torch.cat(outputs, dim=0)
            else:
                attn_mask = attn_mask.to(device=x.device)
                out = self.transformer_encoder(x, mask=attn_mask)
        else:
            out = self.transformer_encoder(x)
        out = self.mlp(out)
        out = self.sigmoid(out).squeeze(dim=2)
        return out

    def forward(self, x, attn_mask=None):
        seq_encodings, attention_masks, loss_masks, real_run_times = x
        self.loss_fxn.loss_masks = loss_masks
        self.loss_fxn.real_run_times = real_run_times
        preds = self.forward_batch(seq_encodings, attention_masks)
        self.loss_fxn.preds = preds # we append the full prediction to the loss function
        predicted_runtimes = preds[:, 0]
        predicted_runtimes = predicted_runtimes * self.config.max_runtime / 1000
        return predicted_runtimes
