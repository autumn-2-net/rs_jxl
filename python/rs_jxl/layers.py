"""Explicit native JPEG XL layers; Python only transports samples and options."""

from collections.abc import Mapping

from . import _options
from ._arrays import integer
from ._ffi import execute
from ._io import write_or_return
from .multiframe import _images


def _mapping(value, name, allowed):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = {}
    for key, item in value.items():
        if key in result:
            raise ValueError(f"duplicate {name} field: {key}")
        if key not in allowed:
            raise TypeError(f"unknown {name} field: {key}")
        result[key] = item
    return result


def _pair(value, name, low, high):
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a pair of integers")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    return [integer(item, f"{name}[{index}]", low, high)
            for index, item in enumerate(value)]


def _blend(value, name):
    result = _mapping(value, name, {"mode", "source", "alpha_channel", "clamp"})
    if "mode" in result:
        mode = result["mode"]
        if not isinstance(mode, str):
            raise TypeError(f"{name}.mode must be a string")
        if mode not in ("replace", "add", "blend", "alpha_weighted_add", "multiply"):
            raise ValueError(f"unknown {name}.mode: {mode}")
    for key in ("source", "alpha_channel"):
        if key in result:
            result[key] = integer(result[key], f"{name}.{key}", 0, 0xff)
    if "clamp" in result and not isinstance(result["clamp"], bool):
        raise TypeError(f"{name}.clamp must be a boolean")
    return result


def _frames(value):
    if not isinstance(value, (list, tuple)):
        raise TypeError("layer_options must be a list of frame mappings")
    frames = []
    for index, frame in enumerate(value):
        name = f"layer_options[{index}]"
        frame = _mapping(frame, name, {"origin", "duration", "timecode", "name",
                                      "blend", "extra_blend", "save_as_reference"})
        if "origin" in frame:
            frame["origin"] = _pair(frame["origin"], f"{name}.origin", -(1 << 31), (1 << 31) - 1)
        for key in ("duration", "timecode", "save_as_reference"):
            if key in frame:
                high = 0xff if key == "save_as_reference" else 0xffffffff
                frame[key] = integer(frame[key], f"{name}.{key}", 0, high)
        if "name" in frame and not isinstance(frame["name"], str):
            raise TypeError(f"{name}.name must be a string")
        if "blend" in frame:
            frame["blend"] = _blend(frame["blend"], f"{name}.blend")
        if "extra_blend" in frame:
            if not isinstance(frame["extra_blend"], (list, tuple)):
                raise TypeError(f"{name}.extra_blend must be a list of blend mappings")
            frame["extra_blend"] = [_blend(blend, f"{name}.extra_blend[{extra}]")
                                    for extra, blend in enumerate(frame["extra_blend"])]
        frames.append(frame)
    return frames


def encode_layers(images, output=None, *, canvas=None, layer_options=None,
                  tps=None, loops=0, layout="auto", effort=7, threads=1,
                  lossless=True, distance=0, extra_channels=None,
                  bits_per_sample=0, color_encoding=None, icc_profile=None,
                  allow_device_copy=False, strip_metadata=False, return_info=False,
                  encoder_options=None, borrow_inputs=False, _context=None,
                  **tool_options):
    """Encode exact raw layers, allowing different sizes with matching metadata.

    canvas is (width, height); native code infers nonnegative extents when it
    is omitted. Negative origins require an explicit canvas. layer_options is
    one mapping per image: origin, duration, timecode, name, blend, extra_blend
    and save_as_reference. Defaults are zero-duration replacement layers using
    blank reference slot 1. Explicit source 0 reuses a preceding zero-duration
    composite; timed references need explicit save slots.

    decode_layer preserves input pixels, including samples outside the canvas;
    normal decode composites and clips to the canvas. This is pixel encoding,
    not JPEG byte transcoding. Canvas, blending and references belong to Rust.
    """
    _options.lossless_only(lossless, distance)
    layers = {}
    if canvas is not None:
        layers["canvas"] = _pair(canvas, "canvas", 1, 0xffffffff)
    if layer_options is not None:
        layers["frames"] = _frames(layer_options)
    loops = integer(loops, "loops", 0, 0xffffffff)
    timebase = [1000, 1]
    if tps is not None:
        timebase = _pair(tps, "tps", 1, 0xffffffff)
    timecodes = any(frame.get("timecode", 0) != 0 for frame in layers.get("frames", []))
    if tps is not None or loops != 0 or timecodes:
        layers["animation"] = {"ticks_per_second_numerator": timebase[0],
                               "ticks_per_second_denominator": timebase[1],
                               "loops": loops, "timecodes": timecodes}
    source_search = tool_options.pop("source_search", False)
    if not isinstance(source_search, bool):
        raise TypeError("source_search must be a boolean")
    options = _options.encoder(effort, threads, source_search, encoder_options, tool_options)
    options["strip_metadata"] = strip_metadata
    payload, specs = _images(images, layout=layout, extra_channels=extra_channels,
                             borrow_inputs=borrow_inputs, bits_per_sample=bits_per_sample,
                             color_encoding=color_encoding, icc_profile=icc_profile,
                             allow_device_copy=allow_device_copy)
    if "frames" in layers and len(layers["frames"]) != len(specs):
        raise ValueError("one layer_options mapping per image is required")
    data, meta = execute({"operation": "encode_layers", "images": specs,
                         "options": options, "layer_options": layers}, payload.data,
                        context=_context)
    result = write_or_return(data, output)
    return (result, meta) if return_info else result
