import torch
from pytorch_lightning import Callback


class TestLossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.batch_outputs = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.batch_outputs.append(outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        test_losses = torch.Tensor(
            [output["batch_test_loss"] for output in self.batch_outputs]
        )
        pl_module.log("test_loss", torch.mean(test_losses), sync_dist=True)
        self.batch_outputs = []
