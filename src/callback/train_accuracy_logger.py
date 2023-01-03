from pytorch_lightning.callbacks import Callback


class TrainAccuracyLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        train_accuracy = pl_module.train_acc_metric.compute()
        pl_module.log("train_accuracy", train_accuracy, sync_dist=True)
        pl_module.train_acc_metric.reset()
