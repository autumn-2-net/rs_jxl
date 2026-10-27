"""Reusable codec workers. Each call retains independent image/reference state."""

from ._context import NativeContext
from ._options import threads as normalize_threads
from .encode import encode, transcode_jpeg
from .decode import decode, decode_into, decode_layer, decode_extra_channel, info, reconstruct_jpeg, decode_to_png
from .multiframe import analyze_multiframe, encode_multiframe
from .layers import encode_layers


class _Context:
    _kind = None

    def __init__(self, *, threads=1):
        self._native = NativeContext(self._kind, threads)

    @property
    def threads(self):
        return self._native.threads

    @property
    def closed(self):
        return self._native.closed

    def close(self):
        self._native.close()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("codec context is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def _call(self, operation, *args, **kwargs):
        if self.closed:
            raise RuntimeError("codec context is closed")
        if "_context" in kwargs:
            raise TypeError("cannot override a bound codec context")
        kwargs.setdefault("threads", self.threads)
        if normalize_threads(kwargs["threads"]) != self.threads:
            raise ValueError("threads must match the codec context")
        return operation(*args, _context=self._native, **kwargs)


class Encoder(_Context):
    """Reuse workers for lossless encoding/analysis; native calls are serialized."""
    _kind = "encoder"

    def encode(self, src, output=None, **options):
        return self._call(encode, src, output, **options)

    def transcode_jpeg(self, src, output=None, **options):
        return self._call(transcode_jpeg, src, output, **options)

    def encode_multiframe(self, frames, output=None, **options):
        return self._call(encode_multiframe, frames, output, **options)

    def encode_layers(self, images, output=None, **options):
        return self._call(encode_layers, images, output, **options)

    def analyze_multiframe(self, frames, **options):
        return self._call(analyze_multiframe, frames, **options)


class Decoder(_Context):
    """Reuse workers, not parser or reference state, across independent inputs."""
    _kind = "decoder"

    def decode(self, src, **options):
        return self._call(decode, src, **options)

    def decode_into(self, src, destination, **options):
        return self._call(decode_into, src, destination, **options)

    def decode_layer(self, src, **options):
        return self._call(decode_layer, src, **options)

    def decode_extra_channel(self, src, channel=0, **options):
        return self._call(decode_extra_channel, src, channel, **options)

    def info(self, src, **options):
        return self._call(info, src, **options)

    def reconstruct_jpeg(self, src, output=None, **options):
        return self._call(reconstruct_jpeg, src, output, **options)

    def decode_to_png(self, src, output=None, **options):
        return self._call(decode_to_png, src, output, **options)
