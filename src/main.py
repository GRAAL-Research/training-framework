# pylint: disable=no-name-in-module, no-value-for-parameter, import-error
# pylint: disable=unexpected-keyword-arg

import warnings

import hydra
from hydra.utils import instantiate
from lightning_lite.utilities.seed import seed_everything

from utils.callbacks import get_callbacks
from utils.config_file_used import (
    get_name_of_config_file_used,
    get_parent_path_of_config_file_used,
)

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*ddp_spawn and num_workers=0 may.*")


@hydra.main(
    version_base="1.2",
    config_path=get_parent_path_of_config_file_used(),
    config_name=get_name_of_config_file_used(),
)
def main(cfg):
    seed_everything(cfg.random_seed)
    logger = instantiate(cfg.logger)
    trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=get_callbacks(cfg, logger),
        _partial_=False,
    )
    model = instantiate(cfg.model, cfg=cfg, _partial_=False)
    datamodule = instantiate(cfg.data_module, cfg=cfg, _partial_=False)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()
