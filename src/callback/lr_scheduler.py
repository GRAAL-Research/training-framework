from hydra.utils import instantiate

from .val_loss_logger import ValLossLogger


class LRScheduler(ValLossLogger):
    def __init__(self):
        super().__init__()
        self.scheduler = None

    def on_train_start(self, trainer, pl_module):
        self.scheduler = instantiate(
            pl_module.cfg.callback.scheduler,
            optimizer=trainer.optimizers[0],
            _partial_=False,
        )

    def on_validation_end(self, trainer, pl_module):
        if self.scheduler is not None:
            self.scheduler.step(trainer.callback_metrics["val_loss"])
