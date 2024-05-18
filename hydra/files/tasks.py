from collections.abc import Iterable

import invoke
import yaml
from invoke.config import DataProxy


def config_to_dict(config: DataProxy, keys: Iterable[str]) -> dict:
    """invokeのconfigをdictに変換する"""
    return {
        key: config_to_dict(val, val.keys())
        if isinstance((val := config.get(key)), DataProxy)
        else val
        for key in keys
    }


@invoke.task
def print_config(c: invoke.Context) -> None:
    print(
        yaml.dump(
            config_to_dict(
                c.config,
                ("version", "env", "user", "gcp", "train", "pipelines"),
            )
        )
    )
