import torch
from pytorch_lightning import Callback


class ValLossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.batch_outputs = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.batch_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_losses = torch.Tensor(
            [output["batch_val_loss"] for output in self.batch_outputs]
        )
        pl_module.log("val_loss", torch.mean(val_losses), sync_dist=True)
        self.batch_outputs = []
