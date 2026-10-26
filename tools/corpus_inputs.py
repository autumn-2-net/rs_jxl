"""Read-only corpus inputs and independently checked, temporary snapshots.

No encoder invocation or corpus discovery. PNG expectations are canonical RGBA
samples at the snapshot depth; JPEG expectations are the original file bytes.
"""

from io import BytesIO
from pathlib import Path
import struct
import tempfile
import zlib

import compare_cpp_port as port


MAX_INPUT_BYTES = port.MAX_INPUT_BYTES
MAX_PIXELS = port.MAX_PIXELS
MAX_DIMENSION = 16384


class UnsupportedInput(ValueError):
    """Valid input outside the single-image integer/byte-preserving contract."""


def _dimensions(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid zero or negative image dimensions")
    if max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise UnsupportedInput("Dimensions exceed CLI limits: 16384 per axis / 64M pixels")


def png_framing(data):
    """Validate PNG framing, including packed depths, before any decoder runs."""
    if not data.startswith(port.PNG_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    offset, seen, kept, removed = 8, set(), [], []
    info, palette_entries, ended_idat, animated = None, 0, False, False
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError("Truncated PNG chunk")
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + length + 12
        if length > 0x7fffffff or end > len(data):
            raise ValueError("Invalid or truncated PNG chunk payload")
        body = data[offset + 8:end - 4]
        if zlib.crc32(body, zlib.crc32(kind)) != struct.unpack_from(">I", data, end - 4)[0]:
            raise ValueError(f"PNG CRC mismatch: {kind!r}")
        if any(not (65 <= c <= 90 or 97 <= c <= 122) for c in kind) or kind[2] & 32:
            raise ValueError("Invalid PNG chunk type")
        if info is None and kind != b"IHDR":
            raise ValueError("PNG must start with IHDR")
        if kind == b"IHDR":
            if info is not None or length != 13:
                raise ValueError("Invalid or repeated PNG IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
            legal = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
            if depth not in legal.get(color, ()) or compression or filtering or interlace not in (0, 1):
                raise ValueError("Invalid PNG depth/color/compression/filter/interlace")
            if not 0 < width <= 0x7fffffff or not 0 < height <= 0x7fffffff:
                raise ValueError("Invalid PNG dimensions")
            info = dict(width=width, height=height, bit_depth=depth, color_type=color)
        elif kind == b"PLTE":
            if (kind in seen or b"IDAT" in seen or b"tRNS" in seen or color in (0, 4)
                    or not length or length % 3 or length > 768):
                raise ValueError("Invalid PNG palette or palette ordering")
            palette_entries = length // 3
            if color == 3 and palette_entries > 1 << depth:
                raise ValueError("PNG palette exceeds index depth")
        elif kind == b"tRNS":
            if kind in seen or b"IDAT" in seen:
                raise ValueError("Invalid PNG transparency ordering")
            if color == 3:
                if not palette_entries or not 0 < length <= palette_entries:
                    raise ValueError("Invalid PNG palette transparency")
            elif color in (0, 2) and length == (2 if color == 0 else 6):
                key = struct.unpack(">" + "H" * (length // 2), body)
                if max(key) >= 1 << depth:
                    raise ValueError("PNG transparency key exceeds sample depth")
                info["transparency_key"] = key
            else:
                raise ValueError("Invalid PNG transparency key")
        elif kind == b"IDAT":
            if ended_idat or (color == 3 and not palette_entries):
                raise ValueError("Invalid PNG IDAT/palette ordering")
        elif kind == b"IEND":
            if length or end != len(data) or b"IDAT" not in seen:
                raise ValueError("Invalid PNG IEND or trailing data")
        elif not kind[0] & 32:
            raise ValueError(f"Unknown critical PNG chunk: {kind!r}")
        if kind in (b"acTL", b"fcTL", b"fdAT"):
            animated = True
        if kind != b"IDAT" and b"IDAT" in seen:
            ended_idat = True
        if kind in (b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"):
            kept.append(data[offset:end])
        else:
            removed.append(kind.decode("ascii"))
        seen.add(kind)
        offset = end
    if b"IEND" not in seen:
        raise ValueError("Missing PNG IEND")
    if animated:
        raise UnsupportedInput("APNG is outside single-frame input acceptance")
    _dimensions(info["width"], info["height"])
    info["removed_metadata_chunks"] = removed
    return port.PNG_SIGNATURE + b"".join(kept), info


def _pixels(samples, depth):
    """Canonicalize integer gray/GA/RGB/RGBA without premultiplication."""
    _, np = port.pixel_libraries()
    if samples.ndim == 2:
        samples = samples[:, :, None]
    height, width, channels = samples.shape
    if channels not in (1, 2, 3, 4):
        raise ValueError("Unexpected sample channel count")
    rgba = np.empty((height, width, 4), dtype=samples.dtype)
    rgba[:, :, :3] = samples[:, :, :1] if channels <= 2 else samples[:, :, :3]
    rgba[:, :, 3] = samples[:, :, -1] if channels in (2, 4) else (1 << depth) - 1
    return port.Pixels(width, height, depth, rgba)


def _packed_png(original, payload, info):
    from PIL import Image

    cv2, np = port.pixel_libraries()
    try:
        samples = cv2.imdecode(np.frombuffer(original, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise ValueError(f"Packed PNG oracle failed: {error}") from error
    if (samples is None or samples.dtype != np.uint8
            or samples.shape[:2] != (info["height"], info["width"])):
        raise ValueError("Packed PNG oracle failed or changed dimensions/depth")
    if samples.ndim == 3:
        samples = samples[:, :, [2, 1, 0, 3] if samples.shape[2] == 4 else [2, 1, 0]]
    expected = _pixels(samples, 8)
    key = info.get("transparency_key")
    if key is not None:
        scaled_key = key[0] * 255 // ((1 << info["bit_depth"]) - 1)
        expected.rgba[:, :, 3] = np.where(expected.rgba[:, :, 0] == scaled_key, 0, 255)
    try:
        with Image.open(BytesIO(payload)) as image:
            if info["color_type"] == 0:
                image = image.convert("L")
                # Pillow exposes the original packed tRNS key, not its 8-bit value.
                if key is not None:
                    image.info["transparency"] = scaled_key
            elif np.asarray(image).max() >= len(image.getpalette()) // 3:
                raise ValueError("PNG pixel references a missing palette entry")
            expanded = image.convert("RGBA")
            expanded.info.clear()
            output = BytesIO()
            expanded.save(output, format="PNG")
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Invalid packed PNG: {error}") from error
    return output.getvalue(), expected


def _jpeg_dimensions(data):
    from PIL import Image

    # Validate marker lengths and entropy framing, never entropy-decode pixels.
    offset, in_scan, scanned, ended = 2, False, False, False
    while offset < len(data):
        if in_scan:
            offset = data.find(b"\xff", offset)
            if offset < 0:
                break
        if data[offset] != 255:
            raise ValueError("Invalid JPEG marker framing")
        while offset < len(data) and data[offset] == 255:
            offset += 1
        if offset == len(data):
            break
        marker = data[offset]
        offset += 1
        if in_scan and (marker == 0 or 0xd0 <= marker <= 0xd7):
            continue
        if marker == 0xd9:
            ended = True
            break
        if marker in (0, 0xd8) or 0xd0 <= marker <= 0xd7:
            raise ValueError("Invalid JPEG marker ordering")
        if marker == 1:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset:offset + 2], "big")
        if length < 2 or offset + length > len(data):
            raise ValueError("Invalid or truncated JPEG segment")
        offset += length
        in_scan = marker == 0xda or (in_scan and marker == 0xdc)
        scanned |= marker == 0xda
    if not ended or not scanned:
        raise ValueError("Truncated JPEG or missing scan/EOI")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "JPEG":
                raise ValueError("Invalid JPEG header")
            width, height = image.size
    except Image.DecompressionBombError as error:
        raise UnsupportedInput("JPEG dimensions exceed CLI limits: 64M pixels") from error
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Invalid JPEG header: {error}") from error
    _dimensions(width, height)
    return width, height


def _integer(token):
    if not token or not token.isdigit():
        raise ValueError("Invalid Netpbm integer header")
    return int(token)


def _netpbm(data):
    _, np = port.pixel_libraries()
    stream = BytesIO(data)
    magic = stream.read(2)
    if magic == b"P7":
        if stream.readline().strip():
            raise ValueError("Invalid PAM magic line")
        fields = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Truncated PAM header")
            line = line.split(b"#", 1)[0].strip()
            if line == b"ENDHDR":
                break
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or parts[0] in fields:
                raise ValueError("Invalid or duplicate PAM header field")
            fields[parts[0]] = parts[1]
        if not {b"WIDTH", b"HEIGHT", b"DEPTH", b"MAXVAL"} <= fields.keys():
            raise ValueError("Missing PAM header field")
        width, height, channels, maxval = [_integer(fields[key]) for key in
                                           (b"WIDTH", b"HEIGHT", b"DEPTH", b"MAXVAL")]
        tuple_type = fields.get(b"TUPLTYPE")
        if tuple_type is not None:
            tuples = {b"GRAYSCALE": 1, b"GRAYSCALE_ALPHA": 2, b"RGB": 3, b"RGB_ALPHA": 4}
            if tuple_type not in tuples:
                raise UnsupportedInput(f"Unsupported PAM tuple type: {tuple_type!r}")
            if tuples[tuple_type] != channels:
                raise ValueError("PAM tuple type and depth disagree")
    else:
        def token():
            value = bytearray()
            while True:
                c = stream.read(1)
                if not c:
                    raise ValueError("Truncated Netpbm header")
                if c == b"#" and not value:
                    stream.readline()
                elif c.isspace():
                    if value:
                        # A CRLF is a single header delimiter, not raster data.
                        if c == b"\r" and data[stream.tell():stream.tell() + 1] == b"\n":
                            stream.read(1)
                        return _integer(bytes(value))
                else:
                    value.extend(c)

        if stream.read(1) not in (b" ", b"\t", b"\r", b"\n", b"\v", b"\f"):
            raise ValueError("Invalid Netpbm magic delimiter")
        width, height, maxval = token(), token(), token()
        channels = 1 if magic == b"P5" else 3
    _dimensions(width, height)
    if channels <= 0 or not 1 <= maxval <= 65535:
        raise ValueError("Invalid Netpbm depth/MAXVAL")
    if channels > 4:
        raise UnsupportedInput("Netpbm supports only 1..4 channels")
    if maxval not in (255, 65535):
        raise UnsupportedInput(f"Non-full Netpbm MAXVAL {maxval}; only 255/65535 are supported")
    depth = 8 if maxval == 255 else 16
    count = width * height * channels
    if len(data) - stream.tell() != count * (depth // 8):
        raise ValueError("Truncated Netpbm raster or trailing data")
    samples = np.frombuffer(data, dtype=np.uint8 if depth == 8 else ">u2",
                            count=count, offset=stream.tell())
    samples = samples.astype(np.uint8 if depth == 8 else np.uint16, copy=False)
    return samples.reshape(height, width, channels), depth


def _netpbm_png(data):
    cv2, _ = port.pixel_libraries()
    samples, depth = _netpbm(data)
    expected = _pixels(samples, depth)
    channels = samples.shape[2]
    if channels == 1:
        encoded_samples = samples[:, :, 0]
    elif channels == 3:
        encoded_samples = samples[:, :, ::-1]
    else:
        encoded_samples = expected.rgba[:, :, [2, 1, 0, 3]]
    try:
        ok, encoded = cv2.imencode(".png", encoded_samples)
    except cv2.error as error:
        raise ValueError(f"Netpbm PNG conversion failed: {error}") from error
    if not ok:
        raise ValueError("Netpbm PNG conversion failed")
    conversion = "netpbm-to-png" + ("-grayalpha-to-rgba" if channels == 2 else "")
    return encoded.tobytes(), expected, conversion


def prepare(path, directory):
    """Return (info, unique snapshot Path, Pixels/original JPEG bytes).

    The caller owns directory (normally a TemporaryDirectory) and its cleanup.
    Only snapshots in that directory are written; source files are never opened
    for writing. UnsupportedInput distinguishes exclusions from malformed data.
    """
    path, directory = Path(path), Path(directory)
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise UnsupportedInput("Input exceeds 512 MiB CLI limit")
    with path.open("rb") as stream:
        original = stream.read(MAX_INPUT_BYTES + 1)
    if len(original) > MAX_INPUT_BYTES:
        raise UnsupportedInput("Input exceeds 512 MiB CLI limit")
    removed = []
    if original.startswith(port.PNG_SIGNATURE):
        payload, framing = png_framing(original)
        removed = framing["removed_metadata_chunks"]
        if framing["bit_depth"] < 8:
            payload, expected = _packed_png(original, payload, framing)
            conversion = f"png-packed{framing['bit_depth']}-to-rgba8"
        else:
            payload, _ = port.png_payload(original)
            expected = port.png_samples(original)
            conversion = "metadata-strip" if removed else "none"
        kind = "png"
    elif original.startswith(b"\xff\xd8"):
        width, height = _jpeg_dimensions(original)
        kind, payload, expected, conversion = "jpeg", original, original, "none"
    elif original[:2] in (b"PF", b"Pf"):
        raise UnsupportedInput("PFM floating-point inputs are not supported")
    elif original[:2] in (b"P5", b"P6", b"P7"):
        payload, expected, conversion = _netpbm_png(original)
        kind = "png"
    elif (not original or port.PNG_SIGNATURE.startswith(original) or original == b"\xff"
          or path.suffix.lower() in (".png", ".jpg", ".jpeg", ".jpe")):
        raise ValueError("Invalid or truncated image signature")
    else:
        raise UnsupportedInput("Unsupported input format; expected PNG, JPEG, or integer P5/P6/P7")
    if len(payload) > MAX_INPUT_BYTES:
        raise UnsupportedInput("Snapshot exceeds 512 MiB CLI limit")
    if kind == "png":
        payload, _ = port.png_payload(payload)
        port.compare_pixels(expected, port.png_samples(payload))
        width, height, depth = expected.width, expected.height, expected.depth
    else:
        depth = None
    input_hash = port.sha256(original)
    if port.file_sha256(path) != input_hash:
        raise ValueError("Source changed during preparation")
    with tempfile.NamedTemporaryFile(prefix="input-", suffix="." + kind,
                                     dir=directory, delete=False) as stream:
        source = Path(stream.name)
        stream.write(payload)
    info = dict(kind=kind, width=width, height=height, pixels=width * height,
                depth=depth, input_sha256=input_hash, payload_sha256=port.file_sha256(source),
                input_bytes=len(original), conversion=conversion, removed_metadata_chunks=removed)
    return info, source, expected
