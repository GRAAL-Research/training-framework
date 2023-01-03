from typing import List, Optional, Tuple, Union

import torch
from pytorch_lightning import Callback
from torch import nn


class BinaryWeight(Callback):
    def __init__(
        self,
        is_stochastic: bool,
        is_on_linear_layer: bool,
        is_on_conv_layer: bool,
    ):
        super().__init__()
        self.is_stochastic = is_stochastic

        self.layer_idx = 0
        self.real_valued_weights: List[torch.Tensor] = []

        self.layer_type_binarised: Union[Tuple, Tuple[Optional[nn.Module]]] = ()
        if is_on_linear_layer:
            self.layer_type_binarised += (nn.Linear,)
        if is_on_conv_layer:
            self.layer_type_binarised += (nn.Conv2d,)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, unused=0
    ):
        self.real_valued_weights = []
        pl_module.model.apply(self.real_to_binary_weights)

    def on_after_backward(self, trainer, pl_module) -> None:
        self.layer_idx = 0
        pl_module.model.apply(self.binary_to_real_weights)

    def real_to_binary_weights(self, layer):
        if isinstance(layer, self.layer_type_binarised):
            self.real_valued_weights.append(torch.clone(layer.weight.data))
            if self.is_stochastic:
                hard_sigmoid = torch.clip((layer.weight.data + 1) / 2, 0, 1)
                layer.weight.data = torch.sign(
                    torch.bernoulli(hard_sigmoid) - 0.5
                )
            else:
                layer.weight.data = torch.sign(layer.weight.data)

    def binary_to_real_weights(self, layer):
        if isinstance(layer, self.layer_type_binarised):
            layer.weight.data = torch.clone(
                self.real_valued_weights[self.layer_idx]
            )
            self.layer_idx += 1
