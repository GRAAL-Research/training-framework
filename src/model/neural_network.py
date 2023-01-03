# pylint: disable=no-name-in-module, import-error, too-many-ancestors

import torch
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from torch import nn

from utils.get_info_from_class import get_n_class

from .model import Model


class NeuralNetwork(Model):
    def __init__(
        self,
        n_input_neuron: int,
        n_hidden_neuron: int,
        n_hidden_layer: int,
        hidden_layer_activation: nn.Module,
        is_batch_norm: bool,
        cfg: DictConfig,
    ):
        super().__init__(cfg)
        self.model = nn.Sequential()
        if n_hidden_layer == 0:
            self.model.append(nn.Linear(n_input_neuron, get_n_class(self.cfg)))
        else:
            for i in range(n_hidden_layer):
                if i == 0:
                    self.model.append(
                        nn.Linear(n_input_neuron, n_hidden_neuron)
                    )
                else:
                    self.model.append(
                        nn.Linear(n_hidden_neuron, n_hidden_neuron)
                    )
                if is_batch_norm:
                    self.model.append(nn.BatchNorm1d(n_hidden_neuron))
                self.model.append(instantiate(hidden_layer_activation))
            self.model.append(nn.Linear(n_hidden_neuron, get_n_class(self.cfg)))
        self.model.append(nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, int(torch.mul(x.shape[-2], x.shape[-1])))
        for layer in self.model:
            x = layer(x)
        return x
