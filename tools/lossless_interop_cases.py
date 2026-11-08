"""Small, reproducible lossless fixtures and exact independent-oracle checks."""

from dataclasses import dataclass
import io
import struct
import zlib

import numpy as np


@dataclass
class PixelCase:
    name: str
    pixels: np.ndarray


def pixel_cases():
    cases = []
    specs = [
        ("gray8_single", np.uint8, 1, 1, 1),
        ("gray16_single", np.uint16, 1, 1, 1),
        ("ga8_edges", np.uint8, 17, 33, 2),
        ("ga16_edges", np.uint16, 17, 33, 2),
        ("rgb8_row", np.uint8, 1, 257, 3),
        ("rgb16_column", np.uint16, 257, 1, 3),
        ("rgba8_edges", np.uint8, 33, 17, 4),
        ("rgba16_edges", np.uint16, 17, 33, 4),
        ("rgb8_tiles", np.uint8, 129, 131, 3),
        ("rgba16_groups", np.uint16, 129, 131, 4),
    ]
    for name, dtype, height, width, channels in specs:
        maximum = np.iinfo(dtype).max
        values = np.arange(height * width * channels, dtype=np.uint64)
        values = ((values * 1237) ^ (values // 7 * 2017) ^ 251) & maximum
        pixels = values.astype(dtype).reshape(height, width, channels)
        edge = np.array([0, 1, 2, maximum // 2, maximum // 2 + 1,
                         maximum - 1, maximum], dtype=dtype)
        pixels.reshape(-1)[:min(pixels.size, len(edge))] = edge[:min(pixels.size, len(edge))]
        if channels in (2, 4):
            pixels[..., -1] = np.resize(edge, height * width).reshape(height, width)
            # Nonzero color at zero alpha must survive even when it is invisible.
            pixels[0, 0, :-1] = maximum - 17
            pixels[0, 0, -1] = 0
        if name.endswith("tiles"):
            tile = pixels[:8, :8].copy()
            pixels = np.tile(tile, (17, 17, 1))[:height, :width].copy()
            pixels[64, 64] ^= np.array([1, 3, 7], dtype=dtype)
        cases.append(PixelCase(name, pixels))
    for name, channels in [("rgb32_gradient", 3), ("gray32_edges", 1)]:
        words = np.array([0, 0x80000000, 0x40000000, 0xC0000000, 0x2EDBE6FF,
                          0, 0, 0x40000000, 0, 0x3F7FFFFF, 0xBF800001], dtype=np.uint32)
        values = np.resize(words.view(np.float32), 17 * 33 * channels)
        cases.append(PixelCase(name, values.reshape(17, 33, channels)))
    unit = np.resize(np.array([0.5, 0.625, 0.75, 0.875, 1.0], dtype=np.float32), 17 * 33 * 3)
    cases.append(PixelCase("rgb32_unit", unit.reshape(17, 33, 3)))
    return cases


def png_bytes(pixels):
    """Serialize unfiltered 8/16-bit fixtures, including GA16 without narrowing."""
    if pixels.dtype not in (np.dtype("uint8"), np.dtype("uint16")) or pixels.ndim != 3:
        raise ValueError("expected HWC uint8/uint16")
    height, width, channels = pixels.shape
    color = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    depth = pixels.dtype.itemsize * 8

    def chunk(kind, data):
        return (struct.pack(">I4s", len(data), kind) + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    packed = pixels.astype(">u2" if depth == 16 else "u1").tobytes()
    row_bytes = width * channels * pixels.dtype.itemsize
    rows = b"".join(b"\0" + packed[y * row_bytes:(y + 1) * row_bytes] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def pfm_bytes(pixels):
    if pixels.dtype != np.float32 or pixels.ndim != 3 or pixels.shape[2] not in (1, 3):
        raise ValueError("expected gray/RGB float32")
    height, width, channels = pixels.shape
    header = f"{'Pf' if channels == 1 else 'PF'}\n{width} {height}\n-1.0\n".encode("ascii")
    return header + pixels[::-1].astype("<f4").tobytes()


def jpeg_cases():
    from PIL import Image

    y, x = np.indices((33, 35), dtype=np.uint32)
    rgb = np.stack(((x * 31 + y * 17) % 256, (x * 7 + y * 47) % 256,
                    (x * 53 ^ y * 13) % 256), axis=-1).astype(np.uint8)
    for name, pixels, progressive, subsampling in [
        ("jpeg_rgb444", rgb, False, 0),
        ("jpeg_progressive420", rgb, True, 2),
        ("jpeg_gray", rgb[..., 0], False, 0),
    ]:
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="JPEG", quality=87,
                                     progressive=progressive, subsampling=subsampling)
        # Archival verification must retain uninterpreted APP/COM and trailing data.
        data = stream.getvalue()
        comment = b"rs-jxl lossless interop fixture"
        data = data[:2] + b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment + data[2:]
        yield name, data + b"\0trailing archival bytes\xff"


def exact_pixels(actual, expected, *, normalized=False):
    """Integer NPY oracle conversion is exact at <=16 bits, never for U31."""
    if normalized:
        wanted_shape = (1, *expected.shape)
        if actual.shape != wanted_shape or actual.dtype != np.dtype("float32"):
            raise AssertionError(f"official shape/dtype {actual.shape}/{actual.dtype}, expected {wanted_shape}/float32")
        actual = actual[0]
        if expected.dtype.kind == "u":
            if expected.dtype.itemsize > 2:
                raise ValueError("float32 NPY cannot certify wide integer output")
            if not np.all(np.isfinite(actual)) or np.any(actual < 0) or np.any(actual > 1):
                raise AssertionError("invalid normalized integer samples")
            actual = np.rint(actual.astype(np.float64) * np.iinfo(expected.dtype).max).astype(expected.dtype)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"shape/dtype {actual.shape}/{actual.dtype}, expected {expected.shape}/{expected.dtype}")
    if actual.dtype == np.float32:
        mismatch = actual.view(np.uint32) != expected.view(np.uint32)
    else:
        mismatch = actual != expected
    count = int(np.count_nonzero(mismatch))
    if count:
        index = tuple(int(value) for value in np.argwhere(mismatch)[0])
        raise AssertionError(f"{count} different samples; first {index}: actual={actual[index]!r}, expected={expected[index]!r}")
    return {"exact": True, "shape": list(expected.shape), "dtype": str(expected.dtype),
            "samples": expected.size}
