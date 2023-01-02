import os
from typing import List, Union
from zlib import crc32

import hydra
import numpy as np
from omegaconf import DictConfig


def hash_split(identifier: Union[str, int], ratio: list[float]) -> int:
    """Return the set id of identifier

    Given a list of set ratios, for example [0.7, 0.2, 0.1], assign identifier to one the sets.
    The ratio of a set is the probability of identifier to get assigned to it.
    The return index corresponds to the index of the ratio.

    Args:
      identifier: id to hash and assign a set to
      ratio: list of set ratios for each set

    Returns:
      index of containing set from ratio list
    """

    if sum(ratio) - 1 > 0.0000001:
        raise ValueError("sum of ratios != 1")
    if isinstance(identifier, str):
        identifier = bytes(identifier, "utf-8")
    elif isinstance(identifier, int):
        identifier = np.int64(identifier)
    else:
        raise ValueError
    h = crc32(identifier) & 0xFFFFFFFF
    cs = np.array(ratio).cumsum() * 2**32
    i = np.searchsorted(cs, h)
    return i


def instantiate_entities(entities_cfg: DictConfig) -> List:
    """Instantiates callable entities"""
    entities = []

    if not isinstance(entities_cfg, DictConfig):
        raise TypeError("Entities config must be a DictConfig!")

    for _, cb_conf in entities_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            entities.append(hydra.utils.instantiate(cb_conf))

    return entities


def configure_clearml(project_name: str, experiment_name: str, config_file: str):
    os.environ["CLEARML_PROJECT"] = project_name
    os.environ["CLEARML_TASK"] = experiment_name
    os.environ["CLEARML_UPLOAD_MODEL_ON_SAVE"] = "FALSE"
