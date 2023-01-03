from .train_accuracy_logger import TrainAccuracyLogger
from .val_accuracy_logger import ValAccuracyLogger


class EarlyStopping(ValAccuracyLogger, TrainAccuracyLogger):
    def __init__(
        self,
        val_acc_min_delta: float,
        val_acc_patience: int,
        train_acc_stopping_threshold: int,
    ):
        super().__init__()
        self.val_acc_min_delta = val_acc_min_delta
        self.val_acc_patience = val_acc_patience
        self.train_acc_stopping_threshold = train_acc_stopping_threshold

        self.best_val_acc = 0
        self.wait_count = 0

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        val_accuracy = trainer.callback_metrics["val_accuracy"]
        if val_accuracy > self.best_val_acc + self.val_acc_min_delta:
            self.best_val_acc = val_accuracy
            self.wait_count = 0
        else:
            self.wait_count += 1
        if (
            self.val_acc_patience <= self.wait_count
            and trainer.min_epochs <= trainer.current_epoch
        ):
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if (
            trainer.callback_metrics["train_accuracy"]
            > self.train_acc_stopping_threshold
            and trainer.min_epochs <= trainer.current_epoch
        ):
            trainer.should_stop = True
