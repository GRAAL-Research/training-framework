# pylint: disable=protected-access
import importlib
import inspect

from omegaconf import DictConfig, ListConfig


def get_n_class(cfg: DictConfig):
    return get_info_from_data_module(cfg, "n_class")


def get_in_channel(cfg: DictConfig):
    return get_info_from_data_module(cfg, "in_channel")


def get_info_from_data_module(cfg: DictConfig, info: str):
    return get_info_from_a_class(cfg.data_module, info, "data_module")


def get_info_from_a_class(
    cfg_object: ListConfig, info: str, class_directory: str
) -> int:
    info_per_class = {}
    for class_name, a_class in inspect.getmembers(
        importlib.import_module(".", class_directory), inspect.isclass
    ):
        key = f"{class_directory}.{class_name}"
        if info == "n_class":
            info_per_class[key] = a_class.get_n_class()
        elif info == "in_channel":
            info_per_class[key] = a_class.get_in_channel()
        else:
            raise ValueError(f"Error: getting {info} is not implemented yet.")
    actual_class_name = cfg_object._target_
    if actual_class_name not in info_per_class:
        raise ValueError(
            f"Error: {actual_class_name} is not in "
            f"{list(info_per_class.keys())}"
        )
    return info_per_class[actual_class_name]
