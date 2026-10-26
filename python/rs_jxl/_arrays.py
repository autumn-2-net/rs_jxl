"""Typed ndarray/tensor transport with explicit layouts and no sample narrowing."""

from collections.abc import Mapping
import operator

import numpy as np

from ._io import is_source, read_bytes


DTYPES = {"u8": np.dtype("u1"), "u16": np.dtype("<u2"), "u32": np.dtype("<u4"),
          "f16": np.dtype("<f2"), "f32": np.dtype("<f4")}
_NAMES = {("u", 1): "u8", ("u", 2): "u16", ("u", 4): "u32",
          ("f", 2): "f16", ("f", 4): "f32"}


def integer(value, name: str, low: int = 0, high: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < low or (high is not None and result > high):
        raise ValueError(f"{name} is outside [{low}, {high if high is not None else 'unbounded'}]")
    return int(result)


def as_array(value, *, layout="auto", allow_device_copy=False, plane=False):
    tensor = type(value).__module__.split(".")[0] == "torch"
    if tensor:
        if value.device.type != "cpu" and not allow_device_copy:
            raise ValueError("GPU tensor input requires allow_device_copy=True (codec runs on CPU)")
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        raise TypeError("pixel input must be a NumPy ndarray or Torch tensor")
    name = _NAMES.get((value.dtype.kind, value.dtype.itemsize))
    if name is None:
        raise TypeError("supported dtypes: uint8, uint16, uint32 (1..31 bits), float16, float32")
    mode = layout.lower()
    if mode not in ("auto", "hw", "hwc", "chw"):
        raise ValueError("layout must be auto, hw, hwc or chw")
    if value.ndim == 2:
        value = value[:, :, None]
    elif value.ndim == 3 and mode != "hw":
        if mode == "chw" or (mode == "auto" and tensor):
            value = np.moveaxis(value, 0, -1)
    else:
        raise ValueError("expected HxW or HxWxC/CHW pixel data")
    if min(value.shape) <= 0 or value.shape[2] not in ((1,) if plane else (1, 2, 3, 4)):
        raise ValueError("invalid dimensions/channel count; additional planes use extra_channels")
    return np.ascontiguousarray(value, dtype=DTYPES[name]), name


class Payload:
    def __init__(self):
        self.data = bytearray()

    def add(self, data) -> dict:
        view = memoryview(data).cast("B")
        offset = len(self.data)
        self.data.extend(view)
        return {"offset": offset, "length": view.nbytes}

    def image(self, src, *, layout="auto", extra_channels=None, bits_per_sample=0,
              color_encoding=None, icc_profile=None, orientation=1, allow_device_copy=False):
        extras = list(extra_channels) if extra_channels is not None else []
        if is_source(src):
            if extras or bits_per_sample or color_encoding is not None or icc_profile is not None or orientation != 1:
                raise ValueError("metadata overrides/extra_channels require array or tensor input")
            return {"format": "encoded", **self.add(read_bytes(src))}
        array, dtype = as_array(src, layout=layout, allow_device_copy=allow_device_copy)
        h, w, channels = array.shape
        spec = {"format": "pixels", "width": w, "height": h, "channels": channels,
                "dtype": dtype, **self.add(array), "extra_channels": [],
                "orientation": integer(orientation, "orientation", 1, 8)}
        if bits_per_sample:
            spec["bits_per_sample"] = integer(bits_per_sample, "bits_per_sample", 1, 32)
        if color_encoding is not None:
            if not isinstance(color_encoding, str):
                raise TypeError("color_encoding must name a supported color space")
            spec["color_encoding"] = color_encoding
        if icc_profile is not None:
            spec["icc"] = list(read_bytes(icc_profile))
        for item in extras:
            item = dict(item) if isinstance(item, Mapping) else {"data": item}
            if "data" not in item:
                raise ValueError("each extra channel requires data")
            values, kind = as_array(item.pop("data"), layout=layout,
                                    allow_device_copy=allow_device_copy, plane=True)
            shift = integer(item.get("dim_shift", 0), "dim_shift", 0, 30)
            scale = 1 << shift
            if values.shape[:2] != ((h + scale - 1) // scale, (w + scale - 1) // scale):
                raise ValueError("extra channel dimensions do not match its dim_shift")
            allowed = {"type", "name", "bits_per_sample", "exponent_bits_per_sample",
                       "dim_shift", "alpha_premultiplied", "spot_color", "cfa_channel"}
            unknown = item.keys() - allowed
            if unknown:
                raise TypeError(f"unknown extra-channel fields: {sorted(unknown)}")
            spec["extra_channels"].append({"type": "optional", **item,
                                            "dtype": kind, **self.add(values)})
        return spec


def unpack_plane(data: bytes, spec: dict, *, out="numpy", device=None):
    try:
        dtype = DTYPES[spec["dtype"]]
        h, w = integer(spec["height"], "height", 1), integer(spec["width"], "width", 1)
        channels = integer(spec.get("channels", 1), "channels", 1, 4)
        offset, length = integer(spec["offset"], "offset"), integer(spec["length"], "length")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("invalid native pixel descriptor") from error
    if length != h * w * channels * dtype.itemsize or offset > len(data) - length:
        raise RuntimeError("native pixel buffer size does not match descriptor")
    if out == "bytes":
        if device is not None:
            raise ValueError("device is only supported for Torch output")
        return data[offset:offset + length]
    array = np.frombuffer(data, dtype=dtype, count=h * w * channels, offset=offset)
    array = array.reshape(h, w, channels).astype(dtype.newbyteorder("="), copy=True)
    if out == "numpy":
        if device is not None:
            raise ValueError("device is only supported for Torch output")
        return array
    if out == "torch":
        import torch
        tensor = torch.from_numpy(array)
        return tensor.to(device) if device is not None else tensor
    raise ValueError("out must be numpy, torch or bytes")
