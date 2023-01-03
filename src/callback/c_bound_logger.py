# pylint: disable=too-many-ancestors, too-many-instance-attributes

from copy import deepcopy
from typing import List, Tuple

import torch
from omegaconf import ListConfig
from pytorch_lightning.callbacks import Callback
from torch import nn

from loss import LinearSum


class CBoundLogger(Callback):
    def __init__(self, n_estimators: ListConfig, noise_intensities: ListConfig):
        super().__init__()
        if not self.is_unique(n_estimators):
            raise ValueError("Error: n_estimators contains identical values.")
        if not self.is_unique(noise_intensities):
            raise ValueError(
                "Error: noise_intensities contains identical values."
            )
        self.n_estimators: List = sorted(list(n_estimators))
        self.noise_intensities: List = sorted(list(noise_intensities))

        self.n_estimator: int = 0
        self.noise_intensity: float = 0

        self.loss_function = LinearSum()
        self.batches: List[List[torch.Tensor]] = []
        self.data_len: int = 0
        self.noised_models: List[nn.Module] = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch: List, batch_idx: int, unused=0
    ):
        if batch_idx == 0:
            self.batches = []
            self.data_len = 0
        self.data_len += len(batch[0])
        self.batches.append(batch)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.batches:
            return
        for noise_intensity in self.noise_intensities:
            self.noise_intensity = noise_intensity
            self.noised_models = []
            for n_estimator in self.n_estimators:
                self.n_estimator = n_estimator
                (
                    gibbs_risk,
                    disagreement,
                ) = self.get_gibbs_risk_and_disagreement(pl_module)
                self.log_c_bound_info(pl_module, gibbs_risk, disagreement)

    def get_gibbs_risk_and_disagreement(self, pl_module) -> Tuple[float, float]:
        saved_model = deepcopy(pl_module.model)
        self.noised_models += self.generate_noised_models(pl_module.model)
        gibbs_risk = 0.0
        squared_pred_sum_scaled_for_c_bound = 0.0
        for batch in self.batches:
            with torch.no_grad():
                inputs, targets = batch
                preds_mean = self.get_preds_mean(pl_module, inputs)
                gibbs_risk += (
                    self.loss_function(preds_mean, targets) / self.data_len
                )
                squared_pred_sum_scaled_for_c_bound += float(
                    (2 * preds_mean[:, 0] - 1).pow(2).sum()
                )

        disagreement = (
            1 / 2
            - 1 / (2 * self.data_len) * squared_pred_sum_scaled_for_c_bound
        )
        pl_module.model = saved_model
        return gibbs_risk, disagreement

    @staticmethod
    def get_c_bound(gibbs_risk: float, disagreement: float) -> float:
        return 1 - (((1 - 2 * gibbs_risk) ** 2) / (1 - 2 * disagreement))

    def log_c_bound_info(
        self, pl_module, gibbs_risk: float, disagreement: float
    ):
        gibbs_risk_tag = "_gibbs_risk"
        disagreement_tag = "_disagreement"
        c_bound_tag = "_c_bound"

        if len(self.n_estimators) == 1:
            if len(self.noise_intensities) == 1:
                end_tag = ""
            else:
                end_tag = f"_i{self.noise_intensity}"
        else:
            if len(self.noise_intensities) == 1:
                end_tag = f"_e{self.n_estimator}"
            else:
                end_tag = f"_e{self.n_estimator}_i{self.noise_intensity}"

        c_bound = self.get_c_bound(gibbs_risk, disagreement)
        pl_module.log(gibbs_risk_tag + end_tag, gibbs_risk)
        pl_module.log(disagreement_tag + end_tag, disagreement)
        pl_module.log(c_bound_tag + end_tag, c_bound)

    def generate_noised_models(self, model: nn.Module) -> List:
        noised_models: List[nn.Module] = []
        n_estimator_idx = self.n_estimators.index(self.n_estimator)

        if n_estimator_idx == 0:
            nb_estimators_to_create = self.n_estimator
        else:
            last_n_estimator = self.n_estimators[n_estimator_idx - 1]
            nb_estimators_to_create = self.n_estimator - last_n_estimator

        for _ in range(nb_estimators_to_create):
            noised_model = deepcopy(model)
            noised_model.apply(self.noised_weights)
            noised_models.append(noised_model)
        return noised_models

    def noised_weights(self, layer) -> None:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.weight.data += (
                self.noise_intensity
                * torch.randn(
                    layer.weight.data.size(), device=layer.weight.data.device
                )
                * torch.mean(layer.weight.data)
            )

    def get_preds_mean(self, pl_module, inputs: torch.Tensor) -> torch.Tensor:
        models_preds = []
        for noised_model in self.noised_models:
            pl_module.model = noised_model
            preds = pl_module.forward(inputs)
            models_preds.append(preds)
        return torch.mean(torch.stack(models_preds), 0)

    @staticmethod
    def is_unique(a_list_config: ListConfig):
        return len(a_list_config) == len(set(a_list_config))
