import torch
from pytorch_lightning import Callback


class TrainLossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.batch_outputs = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        self.batch_outputs.append(outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        train_losses = torch.Tensor(
            [output["loss"] for output in self.batch_outputs]
        )
        pl_module.log("train_loss", torch.mean(train_losses), sync_dist=True)
        self.batch_outputs = []
