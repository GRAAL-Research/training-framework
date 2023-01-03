# pylint: disable=no-name-in-module, import-error, too-many-ancestors
# pylint: disable=arguments-differ

from typing import Any, Dict, Union

import torch
import torchmetrics
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningModule
from torch import nn


class Model(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.loss_function = instantiate(cfg.loss)

        self.train_acc_metric = torchmetrics.Accuracy()
        self.val_acc_metric = torchmetrics.Accuracy()
        self.test_acc_metric = torchmetrics.Accuracy()
        self.model = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        inputs, targets = batch
        preds = self.forward(inputs)
        self.train_acc_metric(preds, targets)
        return {"loss": self.loss_function(preds, targets)}

    def validation_step(
        self, batch: Any, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any], None]:
        inputs, targets = batch
        preds = self.forward(inputs)
        self.val_acc_metric(preds, targets)
        return {"batch_val_loss": self.loss_function(preds, targets)}

    def test_step(
        self, batch: Any, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any], None]:
        inputs, targets = batch
        preds = self.forward(inputs)
        self.test_acc_metric(preds, targets)
        return {"batch_test_loss": self.loss_function(preds, targets)}

    def configure_optimizers(self):
        optimizer = instantiate(
            self.cfg.optimizer,
            params=self.model.parameters(),
            _partial_=False,
        )
        return optimizer
