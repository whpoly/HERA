"""Loss functions."""

import torch


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
