# pylint: disable=invalid-name

import torch
from torch import nn


class StraightThroughEstimator(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        out = torch.sign(x) + x - x.detach()
        out[out == 0] = 1
        out[torch.abs(x) > 1] = out[torch.abs(x) > 1].detach()
        return out
