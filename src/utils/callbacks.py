# pylint: disable=import-error
from typing import List

from hydra.utils import instantiate
from pytorch_lightning import Callback

from callback import (
    HyperparamsLogger,
    LRScheduler,
    ModelLogger,
    TestAccuracyLogger,
    TestLossLogger,
    TrainAccuracyLogger,
    TrainLossLogger,
    ValAccuracyLogger,
    ValLossLogger,
    WeightInitializer,
)
from utils.get_info_from_class import get_n_class


def get_callbacks(cfg, logger) -> List:
    callbacks: List[Callback] = []

    callbacks = add_logger_callbacks(cfg, callbacks, logger)

    if cfg.callback.is_weight_initializer_present:
        callbacks.append(WeightInitializer(cfg.callback.weight_initializer))

    if cfg.callback.is_binarizing_weights:
        callbacks.append(instantiate(cfg.callback.binary_weight))

    callbacks = add_c_bound_callback(cfg, callbacks)

    if cfg.callback.is_scheduler_present:
        callbacks.append(LRScheduler())

    if cfg.callback.is_early_stopping_present:
        callbacks.append(instantiate(cfg.callback.early_stopping))

    for callback in callbacks:
        callbacks = filter_callbacks_of_super_classes(
            callbacks, callback.__class__
        )

    return callbacks


def add_logger_callbacks(cfg, callbacks: list, logger) -> List:
    if cfg.callback.is_model_logged:
        callbacks.append(ModelLogger(logger))

    if cfg.callback.is_hyperparams_logged:
        callbacks.append(HyperparamsLogger(logger))

    if cfg.callback.is_loss_logged:
        loss_loggers = [
            TrainLossLogger(),
            ValLossLogger(),
            TestLossLogger(),
        ]
        callbacks += loss_loggers
    if cfg.callback.is_accuracy_logged:
        acc_loggers = [
            TrainAccuracyLogger(),
            ValAccuracyLogger(),
            TestAccuracyLogger(),
        ]
        callbacks += acc_loggers
    return callbacks


def add_c_bound_callback(cfg, callbacks) -> List:
    if cfg.callback.is_c_bound_logged:
        n_class = get_n_class(cfg)
        if n_class != 2:
            raise ValueError(
                "C-Bound is only implemented for binary classification."
                f"Your data module needs {n_class} classes."
            )
        callbacks.append(instantiate(cfg.callback.c_bound_logger))
    return callbacks


def filter_callbacks_of_super_classes(callbacks, child_class):
    super_classes = [
        super_class.__name__ for super_class in child_class.__mro__[1:]
    ]
    return [
        callback
        for callback in callbacks
        if type(callback).__name__ not in super_classes
    ]
