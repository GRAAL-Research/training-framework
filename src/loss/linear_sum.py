import torch
from torch import Tensor, nn


class LinearSum(nn.Module):
    @staticmethod
    def forward(preds, targets) -> Tensor:
        linear_preds = preds.clone() * 2 - 1
        one_hot_targets = nn.functional.one_hot(targets, preds.size(dim=1))
        linear_targets = one_hot_targets.clone() * 2 - 1
        return torch.sum((1 - (linear_preds * linear_targets)) / 2)
