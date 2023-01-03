from pytorch_lightning.callbacks import Callback


class TestAccuracyLogger(Callback):
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        test_accuracy = pl_module.test_acc_metric.compute()
        pl_module.log("test_accuracy", test_accuracy, sync_dist=True)
        pl_module.test_acc_metric.reset()
