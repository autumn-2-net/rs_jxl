"""Lossless still-image and byte-reversible JPEG entry points."""

from ._arrays import Payload
from ._ffi import execute
from ._io import is_source, read_bytes, write_or_return
from . import _options


def transcode_jpeg(src, output=None, *, effort=9, threads=1, source_search=False,
                   brotli_quality=None, no_cfl=False, order="source", limits=None,
                   return_info=False, _context=None):
    """Preserve original JPEG bytes; unsupported JPEG forms fail without pixel fallback."""
    options = _options.encoder(effort, threads, source_search, None, {})
    options.update(no_cfl=no_cfl, order=order, limits=_options.limits(limits))
    if brotli_quality is not None:
        options["brotli_quality"] = brotli_quality
    data, meta = execute({"operation": "transcode_jpeg", "options": options}, read_bytes(src), context=_context)
    result = write_or_return(data, output)
    return (result, meta) if return_info else result


def encode(src, output=None, *, layout="auto", effort=7, threads=1,
           lossless=True, distance=0, lossless_jpeg=True, source_search=False,
           strip_metadata=False, extra_channels=None, bits_per_sample=0,
           color_encoding=None, icc_profile=None, orientation=1,
           allow_device_copy=False, encoder_options=None, return_info=False,
           borrow_inputs=False, _context=None, **tool_options):
    """Encode arrays/tensors/PNG bytes; JPEG bytes default to reversible transcoding.

    NumPy auto layout is HWC, Torch auto layout is CHW; output is always lossless.
    GPU inputs require allow_device_copy=True. Unknown codec options are errors.
    """
    _options.lossless_only(lossless, distance)
    options = _options.encoder(effort, threads, source_search, encoder_options, tool_options)
    if is_source(src):
        src = read_bytes(src, borrow=borrow_inputs)
        if src[:2] == b"\xff\xd8" and lossless_jpeg:
            if (extra_channels is not None or bits_per_sample or color_encoding is not None
                    or icc_profile is not None or orientation != 1 or strip_metadata):
                raise ValueError("JPEG byte transcoding cannot apply pixel/metadata overrides")
            data, meta = execute({"operation": "transcode_jpeg", "options": options}, src, context=_context)
            result = write_or_return(data, output)
            return (result, meta) if return_info else result
    payload = Payload(borrow_inputs=borrow_inputs)
    image = payload.image(src, layout=layout, extra_channels=extra_channels,
                          bits_per_sample=bits_per_sample, color_encoding=color_encoding,
                          icc_profile=icc_profile, orientation=orientation,
                          allow_device_copy=allow_device_copy)
    options["strip_metadata"] = strip_metadata
    data, meta = execute({"operation": "encode", "options": options, "images": [image]}, payload.data, context=_context)
    result = write_or_return(data, output)
    return (result, meta) if return_info else result


def convert(src, output=None, **options):
    """Alias for encode; this does not silently re-encode existing JXL files."""
    return encode(src, output, **options)
