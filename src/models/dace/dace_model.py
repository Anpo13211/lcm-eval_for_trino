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
        # マルチタスク学習: output_dimを4に拡張（runtime, cpu, blocked, queued）
        self.mlp_hidden_dims = [128, 64, config.output_dim]
        # output_dimが1の場合は4に拡張（マルチタスク対応）
        if config.output_dim == 1:
            self.output_dim = 4  # runtime, cpu, blocked, queued
        else:
            self.output_dim = config.output_dim

        self.mlp = nn.Sequential(
            *[lora.Linear(self.node_length, self.mlp_hidden_dims[0], r=16),
              nn.Dropout(config.mlp_dropout),
              self.mlp_activation,
              lora.Linear(self.mlp_hidden_dims[0], self.mlp_hidden_dims[1], r=8),
              nn.Dropout(config.mlp_dropout),
              self.mlp_activation,
              lora.Linear(self.mlp_hidden_dims[1], self.output_dim, r=4)])

        self.sigmoid = nn.Sigmoid()

    def forward_batch(self, x, attn_mask=None) -> torch.Tensor:
        # change x shape to (batch, seq_len, input_size) from (batch, len)
        # one node is 18 bits
        x = x.view(x.shape[0], -1, self.node_length)
        out = self.transformer_encoder(x, mask=attn_mask)
        out = self.mlp(out)  # Shape: (batch, seq_len, output_dim) where output_dim=4
        out = self.sigmoid(out)  # Shape: (batch, seq_len, 4)
        # squeezeはoutput_dim=4の場合は適用されない（dim=2に要素が4つあるため）
        return out  # Shape: (batch, seq_len, 4)

    def forward(self, x, attn_mask=None):
        # マルチタスク対応: 要素数で判定
        if len(x) == 4:
            # 従来の形式（後方互換性: wall timeのみ）
            seq_encodings, attention_masks, loss_masks, real_run_times = x
            self.loss_fxn.loss_masks = loss_masks
            self.loss_fxn.loss_masks_multitask = None  # マルチタスク用のloss maskなし
            self.loss_fxn.real_run_times = real_run_times
            self.loss_fxn.real_cpu_times = None
            self.loss_fxn.real_blocked_times = None
            self.loss_fxn.real_queued_times = None
        elif len(x) == 8:
            # マルチタスク形式（新形式: loss_mask_multitaskあり）
            seq_encodings, attention_masks, loss_masks, loss_masks_multitask, real_run_times, real_cpu_times, real_blocked_times, real_queued_times = x
            self.loss_fxn.loss_masks = loss_masks  # wall time用（ルートノードのみ）
            self.loss_fxn.loss_masks_multitask = loss_masks_multitask  # マルチタスク用（高さに応じて減衰）
            self.loss_fxn.real_run_times = real_run_times
            self.loss_fxn.real_cpu_times = real_cpu_times
            self.loss_fxn.real_blocked_times = real_blocked_times
            self.loss_fxn.real_queued_times = real_queued_times
        else:
            # マルチタスク形式（旧形式: loss_mask_multitaskなし、9要素）
            # この形式は後方互換性のために残しているが、実際には使用されない
            seq_encodings, attention_masks, loss_masks, real_run_times, real_cpu_times, real_blocked_times, real_queued_times = x[:7]
            self.loss_fxn.loss_masks = loss_masks
            self.loss_fxn.loss_masks_multitask = loss_masks  # 後方互換性: loss_masksと同じを使用
            self.loss_fxn.real_run_times = real_run_times
            self.loss_fxn.real_cpu_times = real_cpu_times
            self.loss_fxn.real_blocked_times = real_blocked_times
            self.loss_fxn.real_queued_times = real_queued_times
        
        preds = self.forward_batch(seq_encodings, attention_masks)
        self.loss_fxn.preds = preds # we append the full prediction to the loss function
        
        # メインの予測（runtime）のみを返す（後方互換性のため）
        # predsのShape: (batch, seq_len, 4)
        # ルートノード（index 0）のruntime予測を取得
        predicted_runtimes = preds[:, 0, 0]  # (batch,) - 各バッチのルートノードのruntime予測
        predicted_runtimes = predicted_runtimes * self.config.max_runtime / 1000
        return predicted_runtimes
