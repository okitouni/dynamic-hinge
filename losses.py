import torch
import torch.nn as nn
from typing import Union


def kr_loss(y_pred, y_true, reduction='mean'):
    # assumes y_true is in {0, 1}^n
    sign = 2 * y_true - 1
    # this is the same as torch.mean(y_pred[y_true == 1]) - torch.mean(y_pred[y_true == 0])
    # perhaps negative labels should be weighted less?
    return reduce_loss(y_pred * sign, reduction=reduction)


def hinge_loss(y_pred, y_true, margin: Union[float, torch.Tensor]=1.0, reduction='mean', scale=False):
    # assumes y_true is in {0, 1}^n
    if scale:
      y_true = 2 * y_true - 1
    hinge = torch.relu(margin - y_true * y_pred).sum(dim=-1)
    return reduce_loss(hinge, reduction=reduction)


def label_dist(x, y, p=2):
    """Compute the distances between each point x and its nearest neighbor from each class.
    Assumes that the labels are in {0, 1}^c.

    Args:
        x (torch.Tensor): Input tensor. Has shape (n, d).
        y (torch.Tensor): Labels. Has shape (n, c).

    Returns:
        torch.Tensor: Distances. Has shape (n, c).
    """
    dist = torch.cdist(x, x, p=p)
    class_distances = []
    # could be made more efficient but this is fine for now
    for row in range(y.shape[0]):
        d = []
        for i in range(y.shape[1]):
            d.append(torch.amin(dist[row][y[:, i] == 1]).item())
        class_distances.append(d)
    class_distances = torch.tensor(class_distances).to(x.device)
    return class_distances

def get_class_sep(x, y, p=2):
  y = y.squeeze()
  d = torch.empty_like(y).float()
  for yi in y.unique():
    d[y == yi] = torch.cdist(x[y == yi], x[y != yi], p=p).amin(axis=1)
  return d.view(-1, 1)


class KRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return kr_loss(y_pred, y_true)


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0, scale=False):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, y_pred, y_true):
        return hinge_loss(y_pred, y_true, margin=self.margin, scale=self.scale)


class HKRLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5, scale=False):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.scale = scale

    def forward(self, y_pred, y_true):
        return self.alpha * kr_loss(y_pred, y_true) + (1 - self.alpha) * hinge_loss(y_pred, y_true, margin=self.margin, scale=self.scale)


class DynamicHingeLoss(nn.Module):
    def __init__(self, margin: Union[torch.Tensor, float]=1.0, p=2, x=None, y_true=None, reduction='mean', scale=False):
        super().__init__()
        self.scale = scale
        margin = margin
        self.p = p
        self.reduction = reduction
        if x is not None and y_true is not None:
            margin += get_class_sep(x, y_true, p=self.p)/2
        self.register_buffer("margin", margin)

    def forward(self, y_pred, y_true):
        return hinge_loss(y_pred, y_true, margin=self.margin, reduction=self.reduction, scale=self.scale)


def reduce_loss(loss, reduction='mean'):
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss
