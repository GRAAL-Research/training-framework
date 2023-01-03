# pylint: disable=attribute-defined-outside-init

from copy import deepcopy

import torchvision

from .mnist_full import MnistFull


class MnistEvenVsOdd(MnistFull):
    def modify_mnist_full(
        self,
        mnist_full: torchvision.datasets.mnist.MNIST,
    ) -> torchvision.datasets.mnist.MNIST:
        mnist_even_odd = deepcopy(mnist_full)
        mnist_even_odd.targets[mnist_full.targets % 2 == 0] = 0
        mnist_even_odd.targets[mnist_full.targets % 2 == 1] = 1

        return mnist_even_odd

    @staticmethod
    def get_n_class():
        return 2
