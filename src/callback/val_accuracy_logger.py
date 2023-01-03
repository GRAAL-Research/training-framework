from pytorch_lightning.callbacks import Callback


class ValAccuracyLogger(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        val_accuracy = pl_module.val_acc_metric.compute()
        pl_module.log("val_accuracy", val_accuracy, sync_dist=True)
        pl_module.val_acc_metric.reset()
