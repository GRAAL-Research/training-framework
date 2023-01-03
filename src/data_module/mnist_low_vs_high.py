# pylint: disable=attribute-defined-outside-init

from copy import deepcopy

import torchvision
from omegaconf.dictconfig import DictConfig

from .mnist_full import MnistFull


class MnistLowVsHigh(MnistFull):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        valid_pct: float,
        treshold_low_digit: int,
        cfg: DictConfig,
    ):
        if treshold_low_digit < 0 or 9 <= treshold_low_digit:
            raise ValueError("Error : threshold_digit must be in [0, 8]")
        super().__init__(data_path, batch_size, valid_pct, cfg)
        self.treshold_low_digit = treshold_low_digit

    def modify_mnist_full(
        self, mnist_full: torchvision.datasets.mnist.MNIST
    ) -> torchvision.datasets.mnist.MNIST:
        mnist_low_high = deepcopy(mnist_full)
        mnist_low_high.targets[
            mnist_full.targets <= self.treshold_low_digit
        ] = 0
        mnist_low_high.targets[mnist_full.targets > self.treshold_low_digit] = 1
        return mnist_low_high

    @staticmethod
    def get_n_class():
        return 2
