"""Normalize Python options without silently accepting foreign encoder flags."""

from collections.abc import Mapping
import os

from ._arrays import integer


def threads(value) -> int:
    value = integer(value, "threads", 0, 256)
    return min(256, os.cpu_count() or 1) if value == 0 else value


def lossless_only(lossless, distance) -> None:
    if lossless is not None and lossless is not True:
        raise ValueError("rs_jxl only provides lossless encoding")
    if distance is not None and distance != 0:
        raise ValueError("distance must be zero; lossy encoding is not implemented")


def encoder(effort, count, source_search, encoder_options, overrides) -> dict:
    result = {"effort": integer(effort, "effort", 1, 11), "threads": threads(count),
              "source_search": source_search}
    if encoder_options is not None:
        if not isinstance(encoder_options, Mapping):
            raise TypeError("encoder_options must be a mapping of Rust option names")
        if encoder_options.keys() & (result.keys() | {"strip_metadata"}):
            raise ValueError("effort, threads, source_search and strip_metadata must use their explicit arguments")
        result.update(encoder_options)
    aliases = {"modular_group_size": "group_size_shift", "modular_colorspace": "rct",
               "modular_predictor": "predictor", "modular_palette_colors": "palette_colors"}
    for name, value in overrides.items():
        if value is None:
            continue
        key = aliases.get(name, name)
        if name in ("iterations", "modular_ma_tree_learning_percent"):
            key, value = "learning_fraction", float(value) / 100.0
        if key in result:
            raise ValueError(f"duplicate encoder option: {key}")
        result[key] = value
    return result


def limits(value=None, **explicit) -> dict:
    result = dict(value) if value is not None else {}
    for key, item in explicit.items():
        if item is not None:
            if key in result:
                raise ValueError(f"duplicate resource limit: {key}")
            result[key] = item
    return {name: integer(item, name, 0) for name, item in result.items()}
