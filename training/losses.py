"""Loss functions."""

import torch
import torch.nn.functional as F


def MSELoss(y, preds, weights=None, reduction='mean'):
    if weights is not None:
        return weightedMSELoss(y, preds, weights, reduction)
    if reduction == 'mean':
        return ((y - preds) ** 2).mean()
    elif reduction == 'sum':
        return ((y - preds) ** 2).sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def MAELoss(y, preds, weights=None, reduction='mean'):
    if weights is not None:
        return weightedMAELoss(y, preds, weights, reduction)
    if reduction == 'mean':
        return torch.abs(y - preds).mean()
    elif reduction == 'sum':
        return torch.abs(y - preds).sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def weightedMSELoss(y, preds, weights, reduction):
    if reduction == 'mean':
        return (weights * (y - preds) ** 2).mean()
    elif reduction == 'sum':
        return (weights * (y - preds) ** 2).sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def weightedMAELoss(y, preds, weights, reduction):
    if reduction == 'mean':
        return (weights * torch.abs(y - preds)).mean()
    elif reduction == 'sum':
        return (weights * torch.abs(y - preds)).sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def pairwise_rank_loss(
        preds,
        targets,
        ranking_groups,
        ranking_items=None,
        min_target_gap=0.1,
):
    """Logistic ordering loss for comparable samples in the same group.

    ``ranking_groups`` identifies the domain within which target values should
    be ordered (a host material for native defects). ``ranking_items`` can be
    supplied to exclude pairs representing different configurations of the
    same physical item (a native-defect group). Target differences are measured
    in the original target units while prediction differences may be scaled.
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1).to(device=preds.device)
    ranking_groups = ranking_groups.reshape(-1).to(device=preds.device)
    if not (len(preds) == len(targets) == len(ranking_groups)):
        raise ValueError("preds, targets, and ranking_groups must have equal length")
    if min_target_gap < 0:
        raise ValueError("min_target_gap must be non-negative")

    target_diff = targets[:, None] - targets[None, :]
    pred_diff = preds[:, None] - preds[None, :]
    valid = ranking_groups[:, None].eq(ranking_groups[None, :])
    valid &= torch.triu(
        torch.ones_like(target_diff, dtype=torch.bool),
        diagonal=1,
    )
    valid &= target_diff.abs().ge(float(min_target_gap))

    if ranking_items is not None:
        ranking_items = ranking_items.reshape(-1).to(device=preds.device)
        if len(ranking_items) != len(preds):
            raise ValueError("ranking_items must have the same length as preds")
        valid &= ~ranking_items[:, None].eq(ranking_items[None, :])

    if not torch.any(valid):
        # Keep the zero connected to ``preds`` so backward() remains valid.
        return preds.sum() * 0.0

    direction = target_diff[valid].sign()
    return F.softplus(-direction * pred_diff[valid]).mean()
