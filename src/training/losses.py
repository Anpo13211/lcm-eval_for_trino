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
        self.loss_masks = None
        self.preds = None
        self.real_run_times = None
        self.to(model.device)

    def forward(self, input, target):
        # NOTE: The loss is computed externally using the stored tensors from the
        #       model forward pass.  The incoming ``input``/``target`` tensors are
        #       ignored to keep parity with the previous behaviour.
        del input, target

        # ``self.loss_masks`` indicates which nodes should contribute to the loss.
        # In the current DACE implementation this is usually only the root node.
        mask = self.loss_masks > 0

        # Guard against zeros which would otherwise yield divisions by zero and
        # subsequently NaNs.  For masked-out entries we simply plug in ``1`` so
        # that the final logarithm evaluates to ``log(1) = 0`` and has no effect.
        eps = 1e-6
        safe_preds = torch.where(
            mask,
            torch.clamp(self.preds, min=eps),
            torch.ones_like(self.preds),
        )
        safe_targets = torch.where(
            mask,
            torch.clamp(self.real_run_times, min=eps),
            torch.ones_like(self.real_run_times),
        )

        q_error = torch.max(safe_preds / safe_targets, safe_targets / safe_preds)
        # Replace masked values with 1 (log(1) -> 0) before weighting.
        q_error = torch.where(mask, q_error, torch.ones_like(q_error))

        loss = torch.log(torch.where(q_error > 1, q_error, torch.ones_like(q_error)))
        loss = loss * self.loss_masks
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        return loss


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
