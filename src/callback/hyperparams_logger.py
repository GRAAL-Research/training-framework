from typing import Union
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import Callback


class HyperparamsLogger(Callback):
    def __init__(self, mlflow_logger):
        super().__init__()
        self.run_id = mlflow_logger.run_id
        self.cfg = None
        self.logger = None

    def on_train_start(self, trainer, pl_module):
        self.cfg = pl_module.cfg
        self.logger = pl_module.logger.experiment
        self.log_hyperparams_from_omegaconf_dict()

    def log_hyperparams_from_omegaconf_dict(self):
        for key, value in self.cfg.items():
            self.explore_recursive(key, value)

    def explore_recursive(self, parent_key, parent_value):
        if isinstance(parent_value, ListConfig):
            for child_key, child_value in enumerate(parent_value):
                self.track_hyperparam(f"{parent_key}.{child_key}", child_value)

        elif isinstance(parent_value, DictConfig):
            for child_key, child_value in parent_value.items():
                if isinstance(child_value, (DictConfig, ListConfig)):
                    self.explore_recursive(
                        f"{parent_key}.{child_key}", child_value
                    )
                else:
                    self.track_hyperparam(
                        f"{parent_key}.{child_key}", child_value
                    )

    def track_hyperparam(self, name: str, value: Union[str, int, float]):
        self.logger.log_param(
            self.run_id,
            name,
            value,
        )
