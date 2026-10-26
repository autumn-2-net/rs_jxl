"""Image/layer decoding and JPEG reconstruction, independent of Python image libraries."""

from . import _options
from ._arrays import integer, unpack_plane
from ._ffi import execute
from ._io import is_jxl, read_bytes, write_or_return


def _frame(data, descriptor, out, device, extras, include_alpha):
    value = unpack_plane(data, descriptor, out=out, device=device)
    meta = dict(descriptor)
    if extras:
        meta["extra_channels"] = [
            {**spec, "data": unpack_plane(data, spec, out=out, device=device)}
            for spec in descriptor.get("extra_channels", [])
            if include_alpha or spec["type"] != "alpha"
        ]
    return value, meta


def decode(src, *, frame=0, out="numpy", coalesced=True, return_info=False,
           return_extra_channels=False, include_alpha_extra=False, threads=1,
           format="native", dither="jxl-rs", adjust_orientation=True, device=None,
           limits=None, max_pixels=None, max_output_bytes=None, strip_metadata=False):
    """Decode one display frame, all frames (frame=None), or raw layers.

    Arrays and tensors are owned HWC data, including a trailing gray dimension.
    Raw layer pixels are decoded layer samples, not arithmetic residuals.
    """
    if out not in ("numpy", "torch", "bytes") or (device is not None and out != "torch"):
        raise ValueError("out must be numpy/torch/bytes; device is Torch-only")
    frame = None if frame is None else integer(frame, "frame", 0)
    data = read_bytes(src)
    if is_jxl(data):
        options = {"threads": _options.threads(threads), "coalescing": coalesced,
                   "frame_index": frame, "format": format, "dither": dither,
                   "include_extra_channels": return_extra_channels,
                   "adjust_orientation": adjust_orientation,
                   "limits": _options.limits(limits, max_pixels=max_pixels,
                                              max_output_bytes=max_output_bytes)}
        raw, meta = execute({"operation": "decode", "options": options}, data)
    else:
        if frame not in (0, None):
            raise IndexError("PNG/JPEG pixel input contains only frame 0")
        if limits is not None or max_pixels is not None or max_output_bytes is not None:
            raise ValueError("resource limits on PNG/JPEG pixel decoding are not exposed yet")
        if format != "native" or not adjust_orientation or not coalesced or dither != "jxl-rs":
            raise ValueError("JXL-specific decode options cannot be applied to PNG/JPEG input")
        raw, meta = execute({"operation": "read_image", "options": {"strip_metadata": strip_metadata}}, data)
    decoded = [_frame(raw, item, out, device, return_extra_channels, include_alpha_extra)
               for item in meta["frames"]]
    if frame is None:
        values = [value for value, _ in decoded]
        meta = {**meta, "frames": [entry for _, entry in decoded]}
        return (values, meta) if return_info or return_extra_channels else values
    if len(decoded) != 1:
        raise RuntimeError("native decoder returned an unexpected selected frame count")
    value, details = decoded[0]
    meta = {**meta, **details}
    return (value, meta) if return_info or return_extra_channels else value


def decode_layer(src, *, layer=0, return_info=True, **options):
    if "frame" in options or "coalesced" in options:
        raise ValueError("use layer= for raw layer decoding")
    return decode(src, frame=layer, coalesced=False, return_info=return_info, **options)


def decode_extra_channel(src, channel=0, *, frame=0, out="numpy", **options):
    channel = integer(channel, "channel", 0)
    frame = integer(frame, "frame", 0)
    if "return_extra_channels" in options or "include_alpha_extra" in options:
        raise ValueError("decode_extra_channel always includes all native extra channels")
    _, metadata = decode(src, frame=frame, out=out, return_extra_channels=True,
                         include_alpha_extra=True, **options)
    extras = metadata["extra_channels"]
    if channel >= len(extras):
        raise IndexError("extra channel index out of range")
    return extras[channel]["data"]


def info(src, *, threads=1, coalesced=True, limits=None):
    """Inspect JXL headers, frame/layer metadata and extra-channel descriptors."""
    data = read_bytes(src)
    if not is_jxl(data):
        raise ValueError("info expects a JXL file; use decode(..., return_info=True) for PNG/JPEG")
    _, meta = execute({"operation": "info", "options": {
        "threads": _options.threads(threads), "coalescing": coalesced,
        "limits": _options.limits(limits),
    }}, data)
    return meta


def reconstruct_jpeg(src, output=None, *, threads=1, limits=None):
    """Recover original JPEG bytes; pixel-encoded JXL cannot be reconstructed."""
    data, _ = execute({"operation": "reconstruct_jpeg", "options": {
        "threads": _options.threads(threads), "limits": _options.limits(limits),
    }}, read_bytes(src))
    return write_or_return(data, output)


def decode_to_png(src, output=None, *, frame=0, threads=1, coalesced=True, limits=None):
    """Export an exact native 8/16-bit JXL frame as PNG, without Pillow narrowing."""
    data, _ = execute({"operation": "decode_to_png", "options": {
        "threads": _options.threads(threads), "frame_index": integer(frame, "frame", 0),
        "coalescing": coalesced, "limits": _options.limits(limits),
    }}, read_bytes(src))
    return write_or_return(data, output)


decode_to_jpeg = reconstruct_jpeg
