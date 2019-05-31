import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse

# from https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


# from https://github.com/bermanmaxim/LovaszSoftmax
def _mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs.requires_grad_()
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad.requires_grad_())
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
    logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
    labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    per_image: compute the loss per image instead of per batch
    ignore: void class id
    """
    if per_image:
        loss = _mean(
            lovasz_hinge_flat(
                *_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss
