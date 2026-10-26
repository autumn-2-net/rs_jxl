"""Pure-Rust lossless JPEG XL codec through a versioned CFFI ABI."""

from ._ffi import CodecError
from .decode import (decode, decode_extra_channel, decode_layer, decode_to_jpeg,
                     decode_to_png, info, reconstruct_jpeg)
from .encode import convert, encode, transcode_jpeg
from .multiframe import analyze_multiframe, encode_multiframe

__version__ = "0.1.0"
__all__ = ["CodecError", "__version__", "analyze_multiframe", "convert", "decode",
           "decode_extra_channel", "decode_layer", "decode_to_jpeg", "decode_to_png",
           "encode", "encode_multiframe", "info", "reconstruct_jpeg", "transcode_jpeg"]
