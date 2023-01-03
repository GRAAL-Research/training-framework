from pathlib import Path

import yaml
from pytorch_lightning.callbacks import Callback


class ModelLogger(Callback):
    def __init__(self, mlflow_logger):
        super().__init__()
        self.run_id = mlflow_logger.run_id

    def on_fit_end(self, trainer, pl_module):
        if self.run_id:
            logger_data_path = pl_module.cfg.logger.tracking_uri.split("/")[1]
            run_path = Path(
                logger_data_path, pl_module.logger.experiment_id, self.run_id
            )

            self.save_checkpoint(pl_module, trainer, run_path)
            self.update_meta_data_file(pl_module, run_path)

    @staticmethod
    def save_checkpoint(pl_module, trainer, run_path: Path):
        model_name = (
            f"epoch={pl_module.current_epoch - 1}"
            f"-step={pl_module.global_step}.ckpt"
        )
        artifact_path = Path(run_path, "artifacts", model_name)
        trainer.save_checkpoint(artifact_path)

    def update_meta_data_file(self, pl_module, run_path: Path):
        logger_data_path = pl_module.cfg.logger.tracking_uri.split("/")[1]
        run_meta_data_path = Path(run_path, "meta.yaml")
        run_meta_data = self.get_yaml(run_meta_data_path)

        artifact_path = Path(
            logger_data_path,
            pl_module.logger.experiment_id,
            self.run_id,
            "artifacts",
        )
        self.replace_artifact_uri_in_yaml(
            run_meta_data_path, run_meta_data, str(artifact_path)
        )

    @staticmethod
    def get_yaml(yaml_path: Path) -> dict:
        with open(yaml_path, encoding="utf-8") as file:
            return yaml.safe_load(file)

    @staticmethod
    def replace_artifact_uri_in_yaml(
        run_meta_data_path: Path, yaml_loaded: dict, new_value: str
    ):
        yaml_loaded["artifact_uri"] = new_value
        with open(run_meta_data_path, "w", encoding="utf-8") as file:
            yaml.dump(yaml_loaded, file)
