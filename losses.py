import torch
import torch.nn as nn


def kr_loss(y_true, y_pred, reduction='mean'):
    # assumes y_true is in {0, 1}^n
    sign = 2 * y_true - 1
    # this is the same as torch.mean(y_pred[y_true == 1]) - torch.mean(y_pred[y_true == 0])
    # perhaps negative labels should be weighted less?
    return reduce_loss(y_pred * sign, reduction=reduction)


def hinge_loss(y_true, y_pred, margin=1.0, reduction='mean'):
    # assumes y_true is in {0, 1}^n
    sign = 2 * y_true - 1
    # suppose target = [0, 0, 1]
    # then sign = [-1, -1, 1]
    # suppose distances to classes are as follows:
    # [0.5, 0.6, 0.1] -> margin
    # [0.5, 0.5, 1] -> dist of x to each class -> margins preds should be -0.5 0.5 , -1 but can also get the other one 0.5 -0.5 -1
    # thus one wants to actually only maximize the margin of the correct class an everything is 0.
    # 
    # perfect loss happens when
    # y_pred = [<-0.5, <-0.6, >0.1]
    # relu(margin - (x[y] - x[i])) 
    # margin = dyi 
    # Now I actually want margin to be dist to closest class or 0 if wrong label sign * yPred
    hinge = torch.relu(margin - sign * y_pred).sum(dim=-1)
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
    class_distances = torch.tensor(class_distances)
    return class_distances


class KRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return kr_loss(y_true, y_pred)


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, y_true, y_pred):
        return hinge_loss(y_true, y_pred, margin=self.margin)


class HKRLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5):
        super().__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        return self.alpha * kr_loss(y_true, y_pred) + (1 - self.alpha) * hinge_loss(y_true, y_pred, margin=self.margin)


class DynamicHingeLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, x=None, y_true=None, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
        if x is not None and y_true is not None:
            self.margin += label_dist(x, y_true, p=self.p)/2

    def forward(self, y_true, y_pred):
        return hinge_loss(y_true, y_pred, margin=self.margin, reduction=self.reduction)


def reduce_loss(loss, reduction='mean'):
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss
