import numpy as np
import torch

from training.training.utils import recursive_to


def dace_batch_to(batch, device, label_norm):
    # マルチタスク対応: 要素数で判定（後方互換性あり）
    if len(batch) == 6:
        # 従来の形式（後方互換性: wall timeのみ）
        seq_encodings, attention_masks, loss_masks, run_times, labels, sample_idxs = batch
        recursive_to(seq_encodings, device)
        recursive_to(attention_masks, device)
        recursive_to(run_times, device)
        recursive_to(loss_masks, device)
        recursive_to(labels, device)
        return (seq_encodings, attention_masks, loss_masks, run_times), labels, sample_idxs
    elif len(batch) == 9:
        # マルチタスク形式（旧形式: loss_mask_multitaskなし）
        seq_encodings, attention_masks, loss_masks, run_times, cpu_times, blocked_times, queued_times, labels, sample_idxs = batch
        recursive_to(seq_encodings, device)
        recursive_to(attention_masks, device)
        recursive_to(run_times, device)
        recursive_to(cpu_times, device)
        recursive_to(blocked_times, device)
        recursive_to(queued_times, device)
        recursive_to(loss_masks, device)
        recursive_to(labels, device)
        # loss_mask_multitaskはloss_masksと同じとする（後方互換性）
        return (seq_encodings, attention_masks, loss_masks, loss_masks, run_times, cpu_times, blocked_times, queued_times), labels, sample_idxs
    else:
        # マルチタスク形式（新形式: loss_mask_multitaskあり）
        seq_encodings, attention_masks, loss_masks, loss_masks_multitask, run_times, cpu_times, blocked_times, queued_times, labels, sample_idxs = batch
        recursive_to(seq_encodings, device)
        recursive_to(attention_masks, device)
        recursive_to(run_times, device)
        recursive_to(cpu_times, device)
        recursive_to(blocked_times, device)
        recursive_to(queued_times, device)
        recursive_to(loss_masks, device)
        recursive_to(loss_masks_multitask, device)
        recursive_to(labels, device)
        return (seq_encodings, attention_masks, loss_masks, loss_masks_multitask, run_times, cpu_times, blocked_times, queued_times), labels, sample_idxs


def simple_batch_to(batch, device, label_norm):
    query_plans, labels, sample_idxs = batch
    if label_norm is not None:
        labels = label_norm.transform(np.asarray(labels).reshape(-1, 1))
        labels = labels.reshape(-1)
    labels = torch.as_tensor(labels, device=device, dtype=torch.float)
    recursive_to(query_plans, device)
    recursive_to(labels, device)
    return query_plans, labels, sample_idxs


def batch_to(batch, device, label_norm):
    graph, features, label, sample_idxs = batch

    # normalize the labels for training
    if label_norm is not None:
        label = label_norm.transform(np.asarray(label).reshape(-1, 1))
        label = label.reshape(-1)

    label = torch.as_tensor(label, device=device, dtype=torch.float)
    recursive_to(features, device)
    recursive_to(label, device)
    # recursive_to(graph, device)
    graph = graph.to(device, non_blocking=True)
    return (graph, features), label, sample_idxs
