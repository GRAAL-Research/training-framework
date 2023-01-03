from pathlib import Path

from omegaconf import OmegaConf

CONF_PATH = Path("conf")


def get_parent_path_of_config_file_used() -> str:
    return str(Path("..", CONF_PATH))


def get_name_of_config_file_used() -> str:
    file_extension = "yaml"
    config_file_used = Path(CONF_PATH, f"_config_file_used.{file_extension}")
    return (
        f"{OmegaConf.load(config_file_used).config_file_used}.{file_extension}"
    )
