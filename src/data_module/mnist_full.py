# pylint: disable=attribute-defined-outside-init, protected-access

from typing import Optional

import pytorch_lightning as pl
import torchvision
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from .transform import DivideBy


class MnistFull(pl.LightningDataModule):
    def __init__(
        self, data_path: str, batch_size: int, valid_pct: float, cfg: DictConfig
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.valid_pct = valid_pct
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        transform_list = [transforms.ToTensor(), DivideBy(255)]
        if "AlexNet" in self.cfg.model._target_:
            transform_list.append(transforms.Resize(224))
        transform = transforms.Compose(transform_list)

        mnist_full = MNIST(
            root=self.data_path, train=True, transform=transform, download=True
        )
        mnist_low_high = self.modify_mnist_full(mnist_full)

        datasets_split_size = [
            round(
                len(mnist_low_high.data)
                - len(mnist_low_high.data) * self.valid_pct
            ),
            round(len(mnist_low_high.data) * self.valid_pct),
        ]
        self.mnist_train, self.mnist_val = random_split(
            mnist_low_high, datasets_split_size
        )
        mnist_test_full = MNIST(
            root=self.data_path,
            train=False,
            transform=transform,
            download=True,
        )
        self.mnist_test = self.modify_mnist_full(mnist_test_full)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=0
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=0
        )

    def modify_mnist_full(
        self, mnist_full: torchvision.datasets.mnist.MNIST
    ) -> torchvision.datasets.mnist.MNIST:
        return mnist_full

    @staticmethod
    def get_n_class():
        return 10

    @staticmethod
    def get_in_channel():
        return 1
