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
    """Buffers for one synchronous native call, snapshotting mutable inputs.

    ``borrow_inputs=True`` avoids copying eligible contiguous inputs. Callers
    must not mutate borrowed inputs or their aliases while the GIL-released
    native call uses them, just as for NumPy native operations. Read-only views
    do not prevent mutation through another alias. Immutable bytes may always
    be borrowed; read-only views of mutable storage are snapshotted by default.

    The first buffer is retained directly (after any required snapshot or
    normalization). A second append consolidates the buffers into a bytearray,
    snapshotting their current contents. Borrowed exporters remain retained
    until the payload is discarded. Keep the payload alive until execute returns
    and do not append to it during execution. Borrowing is best-effort: layout,
    byte-order conversion and consolidation can still require copies.
    """

    def __init__(self, *, borrow_inputs=False):
        self.data = bytearray()
        self._borrow_inputs = borrow_inputs
        self._buffers = []
        self._count = 0

    def add(self, data) -> dict:
        view = memoryview(data)
        if not view.c_contiguous:
            view = memoryview(view.tobytes())
        view = view.cast("B")
        offset = len(self.data)
        borrow = self._borrow_inputs or isinstance(view.obj, bytes)
        if not self._count:
            self.data = view if borrow else bytearray(view)
        elif isinstance(self.data, bytearray):
            self.data.extend(view)
        else:
            combined = bytearray(self.data)
            combined.extend(view)
            self.data = combined
        if borrow:
            self._buffers.append(view)
        self._count += 1
        return {"offset": offset, "length": view.nbytes}

    def image(self, src, *, layout="auto", extra_channels=None, bits_per_sample=0,
              color_encoding=None, icc_profile=None, orientation=1, allow_device_copy=False):
        extras = list(extra_channels) if extra_channels is not None else []
        if is_source(src):
            if extras or bits_per_sample or color_encoding is not None or icc_profile is not None or orientation != 1:
                raise ValueError("metadata overrides/extra_channels require array or tensor input")
            data = src if isinstance(src, (bytes, bytearray, memoryview)) else read_bytes(src)
            return {"format": "encoded", **self.add(data)}
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


def copy_into(data, spec: dict, destination: np.ndarray) -> np.ndarray:
    """Copy a decoded plane directly from its buffer into an existing ndarray.

    The destination must have the exact native-endian dtype and HWC shape,
    including the trailing gray channel, and be writable and C-contiguous.
    No sample conversion or narrowing is allowed. All descriptor, buffer and
    destination validation completes before any destination bytes are changed.
    The source buffer owner must remain alive throughout this synchronous copy;
    callers must not concurrently mutate either buffer or their aliases.
    """
    try:
        dtype = DTYPES[spec["dtype"]]
        h, w = integer(spec["height"], "height", 1), integer(spec["width"], "width", 1)
        channels = integer(spec.get("channels", 1), "channels", 1, 4)
        offset, length = integer(spec["offset"], "offset"), integer(spec["length"], "length")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("invalid native pixel descriptor") from error
    try:
        view = memoryview(data).cast("B")
    except (TypeError, ValueError) as error:
        raise RuntimeError("native pixel buffer must be C-contiguous") from error
    if length != h * w * channels * dtype.itemsize or offset > view.nbytes - length:
        raise RuntimeError("native pixel buffer size does not match descriptor")
    if not isinstance(destination, np.ndarray):
        raise TypeError("destination must be a NumPy ndarray")
    if destination.dtype != dtype.newbyteorder("="):
        raise TypeError("destination must have the exact native-endian pixel dtype")
    if destination.shape != (h, w, channels):
        raise ValueError("destination shape must match the native HWC pixel shape")
    if not destination.flags.writeable or not destination.flags.c_contiguous:
        raise ValueError("destination must be writable and C-contiguous")
    source = np.frombuffer(view, dtype=dtype, count=h * w * channels, offset=offset)
    source = source.reshape(h, w, channels)
    np.copyto(destination, source, casting="equiv")
    return destination


def unpack_plane(data: bytes | bytearray | memoryview, spec: dict, *, out="numpy", device=None):
    """Materialize owned output before the source native buffer is released."""
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
        return bytes(data[offset:offset + length])
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
