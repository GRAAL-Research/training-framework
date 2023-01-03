# pylint: disable=too-few-public-methods

import torch


class DivideBy:
    def __init__(self, divider: int) -> None:
        self.divider = divider

    def __call__(self, tensor) -> torch.Tensor:
        return torch.div(tensor, self.divider)
