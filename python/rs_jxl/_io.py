"""Path and binary stream adapters; native code never opens paths."""

import os
from pathlib import Path


def is_source(value) -> bool:
    return isinstance(value, (str, os.PathLike, bytes, bytearray, memoryview)) or hasattr(value, "read")


def read_bytes(value) -> bytes:
    if isinstance(value, (str, os.PathLike)):
        return Path(value).read_bytes()
    if hasattr(value, "read"):
        value = value.read()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError("expected a path, encoded bytes or a binary readable stream")


def write_or_return(data: bytes, output=None):
    if output is None:
        return data
    if isinstance(output, (str, os.PathLike)):
        path = Path(output)
        path.write_bytes(data)
        return path
    if not hasattr(output, "write"):
        raise TypeError("output must be a path or binary writable stream")
    view = memoryview(data)
    while view:
        written = output.write(view)
        if not isinstance(written, int) or not 0 < written <= len(view):
            raise OSError("binary output stream did not accept the requested bytes")
        view = view[written:]
    return None


def is_jxl(data: bytes) -> bool:
    return data.startswith((b"\xff\x0a", b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a"))
