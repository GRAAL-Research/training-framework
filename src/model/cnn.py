# pylint: disable=too-many-ancestors

from omegaconf.dictconfig import DictConfig
from torch import nn

from utils.get_info_from_class import get_in_channel, get_n_class

from .model import Model


class CNN(Model):
    def __init__(self, n_conv, cfg: DictConfig):
        super().__init__(cfg)
        self.model = nn.Sequential(
            nn.Conv2d(
                get_in_channel(cfg), 8, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        linear_n_input_neuron = 14 * 14 * 8

        if n_conv >= 2:
            second_convolution = nn.Sequential(
                nn.Conv2d(8, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.model.add_module("conv2", second_convolution)
            linear_n_input_neuron = 7 * 7 * 64

            if n_conv == 3:
                third_convolution = nn.Sequential(
                    nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                linear_n_input_neuron = 3 * 3 * 256
                self.model.add_module("conv3", third_convolution)

        output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_n_input_neuron, get_n_class(cfg)),
            nn.Softmax(dim=1),
        )
        self.model.add_module("output", output)
