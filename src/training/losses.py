import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, model, weight=None, **kwargs):
        super().__init__()

    def forward(self, input, target):
        return F.mse_loss(input.view(-1), target.view(-1), reduction='mean')


class QLoss(nn.Module):
    """
    Regression loss that minimizes the q-error for each prediction
    """

    def __init__(self, model, weight=None, min_val=1e-3, penalty_negative=1e5, **kwargs):
        self.min_val = min_val
        self.penalty_negative = penalty_negative
        super().__init__()

    def forward(self, input, target):
        input_zero_mask = input == 0
        target_zero_mask = target == 0

        if input_zero_mask.any():
            input = input + input_zero_mask * torch.full(input.shape, 0.0000001, device=input.device)
        if target_zero_mask.any():
            target = target + target_zero_mask * torch.full(input.shape, 0.0000001, device=target.device)

        q_error = torch.zeros((len(target), 1), device=target.device)

        # create mask for entries which should be penalized for negative/too small estimates
        penalty_mask = input < self.min_val
        inverse_penalty_mask = input >= self.min_val
        q_error_penalties = torch.mul(1 - input, penalty_mask) * self.penalty_negative

        # influence on loss for a negative estimate is >= penalty_negative constant
        q_error = torch.add(q_error, q_error_penalties)

        # calculate normal q error for other instances
        input_masked = torch.mul(input, inverse_penalty_mask)
        target_masked = torch.mul(target.reshape((-1, 1)), inverse_penalty_mask)

        q_error = torch.add(q_error, torch.max(torch.div(input_masked, target.reshape((-1, 1))),
                                               torch.div(target_masked, input)))

        loss = torch.mean(q_error)
        return loss


class DaceLoss(nn.Module):
    def __init__(self,  model, **loss_kwargs):
        super().__init__()
        self.loss_masks = None  # wall time用（ルートノードのみ）
        self.loss_masks_multitask = None  # マルチタスク用（CPU/blocked/queued timeが存在する全ノード）
        self.preds = None
        self.real_run_times = None
        self.real_cpu_times = None
        self.real_blocked_times = None
        self.real_queued_times = None
        # マルチタスク損失の重み（デフォルト: 0.1）
        self.multitask_weight = loss_kwargs.get('multitask_weight', 0.1)
        self.to(model.device)

    def forward(self, input, target):
        input = self.preds  # Shape: (batch, seq_len, 4) - [runtime, cpu, blocked, queued]
        target = self.real_run_times  # Shape: (batch, seq_len)
        
        # メインの損失（runtime予測）
        # input[:, :, 0] は Shape: (batch, seq_len) - 各ノードのruntime予測
        loss = torch.max(input[:, :, 0] / (target + 1e-7), (target + 1e-7) / input[:, :, 0])
        loss = loss * self.loss_masks
        loss = torch.log(torch.where(loss > 1, loss, 1))
        loss_main = torch.sum(loss, dim=1)
        loss_main = torch.mean(loss_main)
        
        # マルチタスク損失（CPU, blocked, queued）
        # マルチタスク用のloss maskを使用（CPU/blocked/queued timeが存在する全ノード）
        loss_mask_for_multitask = self.loss_masks_multitask if self.loss_masks_multitask is not None else self.loss_masks
        loss_multitask = torch.tensor(0.0, device=self.real_run_times.device)
        
        if self.real_cpu_times is not None and input.shape[2] >= 2:
            # CPU time損失（サブプランも使用）
            # input[:, :, 1] は Shape: (batch, seq_len) - 各ノードのCPU time予測
            loss_cpu = torch.max(input[:, :, 1] / (self.real_cpu_times + 1e-7), 
                                (self.real_cpu_times + 1e-7) / input[:, :, 1])
            loss_cpu = loss_cpu * loss_mask_for_multitask
            loss_cpu = torch.log(torch.where(loss_cpu > 1, loss_cpu, 1))
            loss_cpu = torch.sum(loss_cpu, dim=1)
            loss_cpu = torch.mean(loss_cpu)
            loss_multitask = loss_multitask + loss_cpu
        
        if self.real_blocked_times is not None and input.shape[2] >= 3:
            # Blocked time損失（サブプランも使用）
            # input[:, :, 2] は Shape: (batch, seq_len) - 各ノードのblocked time予測
            loss_blocked = torch.max(input[:, :, 2] / (self.real_blocked_times + 1e-7), 
                                    (self.real_blocked_times + 1e-7) / input[:, :, 2])
            loss_blocked = loss_blocked * loss_mask_for_multitask
            loss_blocked = torch.log(torch.where(loss_blocked > 1, loss_blocked, 1))
            loss_blocked = torch.sum(loss_blocked, dim=1)
            loss_blocked = torch.mean(loss_blocked)
            loss_multitask = loss_multitask + loss_blocked
        
        if self.real_queued_times is not None and input.shape[2] >= 4:
            # Queued time損失（サブプランも使用）
            # input[:, :, 3] は Shape: (batch, seq_len) - 各ノードのqueued time予測
            loss_queued = torch.max(input[:, :, 3] / (self.real_queued_times + 1e-7), 
                                   (self.real_queued_times + 1e-7) / input[:, :, 3])
            loss_queued = loss_queued * loss_mask_for_multitask
            loss_queued = torch.log(torch.where(loss_queued > 1, loss_queued, 1))
            loss_queued = torch.sum(loss_queued, dim=1)
            loss_queued = torch.mean(loss_queued)
            loss_multitask = loss_multitask + loss_queued
        
        # 総損失: loss_main + λ * loss_multitask
        total_loss = loss_main + self.multitask_weight * loss_multitask
        
        return total_loss


class QPPLoss(torch.nn.Module):
    def __init__(self, model, **loss_kwargs):
        super().__init__()
        self.operator_types = model.operator_types
        self.device = model.device
        self.zeros = torch.zeros(1)
        self.curr_losses = {operator: 0 for operator, _ in self.operator_types.items()}
        self.accumulated_loss = {operator: [] for operator, _ in self.operator_types.items()}
        self.total_loss = None
        self.total_losses = None

    def update_operator_loss(self, node_type: str, predicted_operator_time: torch.Tensor, label: torch.Tensor) -> None:
        loss = torch.sqrt(((predicted_operator_time - label) ** 2) + 1e-6)
        # This works also with Q-error, but original paper uses MSE
        # We apply a lower bound of loss and labels to avoid nans.
        # This additionally helps to preventing loss explosions.
        # predicted_operator_time = torch.max(predicted_operator_time, torch.tensor(0.001))
        # label = torch.max(label, torch.tensor(0.001))
        # loss = torch.max(predicted_operator_time / label, label / predicted_operator_time)
        # ---------------------------------------------------------------------------------------------

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"Loss was {loss} "
                             f"for node type {node_type}, "
                             f"prediction: {predicted_operator_time}, "
                             f"label: {label}, "
                             f"q-Error: {loss}")

        # Gather loss in accumulated loss
        self.accumulated_loss[node_type].append(loss)

    def reset_accumulated_losses(self) -> None:
        self.accumulated_loss = {operator: [] for operator, _ in self.operator_types.items()}
        self.total_loss = torch.zeros(1, dtype=torch.double).to(self.device)

    def update_total_losses(self) -> None:
        for operator_type, loss in self.accumulated_loss.items():
            if loss and loss != []:
                operator_loss = torch.cat(loss).to(self.device)
                self.total_loss += torch.sum(operator_loss)

    def forward(self, input, target) -> torch.Tensor:
        out = torch.mean(self.total_loss)
        self.reset_accumulated_losses()
        return out
