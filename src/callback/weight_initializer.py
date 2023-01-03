from hydra.utils import call
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import Callback
from torch import nn


class WeightInitializer(Callback):
    def __init__(
        self,
        weight_initializer: DictConfig,
    ):
        super().__init__()
        self.weight_initializer = weight_initializer
        self.layer_type_binarised = (nn.Linear, nn.Conv2d)

    def on_fit_start(self, trainer, pl_module) -> None:
        pl_module.model.apply(self.init_weights)

    def init_weights(self, layer: nn.Module) -> None:
        if isinstance(layer, self.layer_type_binarised):
            call(
                self.weight_initializer,
                tensor=layer.weight.data,
                _partial_=False,
            )
