# pylint: disable=attribute-defined-outside-init

from copy import deepcopy

import torch
import torchvision
from omegaconf.dictconfig import DictConfig

from .mnist_full import MnistFull


class Mnist2Digits(MnistFull):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        valid_pct: float,
        low_digit: int,
        high_digit: int,
        cfg: DictConfig,
    ):
        if low_digit >= high_digit:
            raise ValueError("Error : low_digit must be lower than high_digit.")
        super().__init__(data_path, batch_size, valid_pct, cfg)
        self.low_digit = low_digit
        self.high_digit = high_digit

    def modify_mnist_full(
        self, mnist_full: torchvision.datasets.mnist.MNIST
    ) -> torchvision.datasets.mnist.MNIST:
        low_high_indices = torch.where(
            (mnist_full.targets == self.low_digit)
            + (mnist_full.targets == self.high_digit)
        )
        mnist_2digits = deepcopy(mnist_full)
        mnist_2digits.data = mnist_full.data[low_high_indices]
        mnist_2digits.targets = mnist_full.targets[low_high_indices]
        mnist_2digits.targets[mnist_2digits.targets == self.low_digit] = 0
        mnist_2digits.targets[mnist_2digits.targets == self.high_digit] = 1
        return mnist_2digits

    @staticmethod
    def get_n_class():
        return 2
