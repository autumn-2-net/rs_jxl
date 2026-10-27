"""Exact full/crop and byte-selected automatic multiframe strategies."""

from collections.abc import Mapping

from . import _options
from ._arrays import Payload, integer
from ._ffi import execute
from ._io import write_or_return


def _images(frames, *, extra_channels=None, borrow_inputs=False, **options):
    frames = list(frames)
    if not frames:
        raise ValueError("at least one frame is required")
    extras = list(extra_channels) if extra_channels is not None else []
    payload, specs = Payload(borrow_inputs=borrow_inputs), []
    for index, image in enumerate(frames):
        channels = []
        for extra in extras:
            extra = dict(extra) if isinstance(extra, Mapping) else {"data": extra}
            values = extra.get("data")
            if isinstance(values, (list, tuple)):
                if len(values) != len(frames):
                    raise ValueError("extra channel frame list must match the frame count")
                extra["data"] = values[index]
            channels.append(extra)
        specs.append(payload.image(image, extra_channels=channels, **options))
    return payload, specs


def encode_multiframe(frames, output=None, *, layout="auto", durations=1,
                      reference="auto", strategy="crop", tps=(1000, 1), loops=0,
                      effort=7, threads=1, lossless=True, distance=0,
                      extra_channels=None, bits_per_sample=0, color_encoding=None,
                      icc_profile=None, allow_device_copy=False, strip_metadata=False,
                      return_info=False, encoder_options=None, borrow_inputs=False,
                      _context=None, **tool_options):
    """Preserve every full input state; duration=0 remains a genuine layer.

    reference='auto' uses the smaller first/previous crop. strategy='auto'
    additionally compares eligible full/crop/mask/multiple-rectangle encodes.
    """
    _options.lossless_only(lossless, distance)
    references = {"auto": "smallest_crop", "first": "first", "previous": "previous",
                  "full": "full", "none": "full"}
    if reference not in references or strategy not in ("crop", "auto"):
        raise ValueError("reference must be auto/first/previous/full; strategy must be crop/auto")
    payload, specs = _images(frames, layout=layout, extra_channels=extra_channels,
                             borrow_inputs=borrow_inputs,
                             bits_per_sample=bits_per_sample, color_encoding=color_encoding,
                             icc_profile=icc_profile, allow_device_copy=allow_device_copy)
    if durations is None:
        durations = 1
    try:
        ticks = [integer(durations, "duration", 0, 0xffffffff)] * len(specs)
    except TypeError:
        ticks = [integer(value, "duration", 0, 0xffffffff) for value in durations]
    if len(ticks) != len(specs):
        raise ValueError("one duration per frame is required")
    if len(tps) != 2:
        raise ValueError("tps must be (numerator, denominator)")
    timebase = [integer(tps[0], "tps numerator", 1, 1 << 30),
                integer(tps[1], "tps denominator", 1, 1024)]
    options = _options.encoder(effort, threads, False, encoder_options, tool_options)
    options["strip_metadata"] = strip_metadata
    data, meta = execute({"operation": "encode_multiframe", "images": specs,
                         "options": options, "durations": ticks, "timebase": timebase,
                         "loops": integer(loops, "loops", 0, 0xffffffff),
                         "reference": references[reference], "auto": strategy == "auto"}, payload.data, context=_context)
    result = write_or_return(data, output)
    return (result, meta) if return_info else result


def analyze_multiframe(frames, *, layout="auto", extra_channels=None,
                       allow_device_copy=False, strip_metadata=False):
    """Return exact change bounds against previous/first, including invisible RGB."""
    payload, specs = _images(frames, layout=layout, extra_channels=extra_channels,
                             allow_device_copy=allow_device_copy)
    _, meta = execute({"operation": "analyze_multiframe", "images": specs,
                      "options": {"strip_metadata": strip_metadata}}, payload.data)
    return meta
