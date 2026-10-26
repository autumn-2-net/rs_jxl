# Python API and Packaging

Distribution `rs-jxl`, import `rs_jxl`, Python 3.10+. The C++ `jxlpy` package
can coexist. There is no C++ linkage, CLI subprocess codec or Pillow dependency.
Only supported lossless Rust controls are accepted; this is not full libjxl API
parity. Unknown and inapplicable settings fail instead of being ignored.

## Installation

```powershell
cargo fetch --locked
cargo build --locked --offline --release -p rs-jxl-ffi
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

Fetch needs network only for uncached locked dependencies. Packaging defaults
to a locked/offline release build. Install the resulting platform wheel with
pip: users need cffi/numpy, not Rust. Build requirements are in pyproject.toml.
Install Torch separately using the desired CPU/CUDA build.

Set `RS_JXL_NATIVE_LIB` to a prebuilt `rs_jxl_ffi.dll` (Windows),
`librs_jxl_ffi.so` (Linux), or `librs_jxl_ffi.dylib` (macOS) to skip compilation.
The packager checks native format/architecture, not by executing the library.
Wheels are py3-none-platform, never pure-any. Foreign cross-builds are rejected;
Linux manylinux repair remains release work. macOS runtime is out of scope.
Development/editable imports use PYTHONPATH=python and target/release, honoring
CARGO_TARGET_DIR/CARGO_BUILD_TARGET. Wheels use their bundled library; explicit
RS_JXL_NATIVE_LIB overrides either lookup and fails on an invalid path/ABI.

## Data Contract

- Inputs: paths, encoded bytes/bytearray/memoryview, binary readable streams,
  NumPy arrays or Torch tensors. Streams are read at their current position.
- Auto layout: HWC for NumPy, CHW for Torch; HW gray accepted. Explicit hwc/chw
  overrides this. Noncontiguous input is copied without narrowing.
- Storage: uint8/uint16/uint32 (declared 1..31 bits), float16/float32. Signed
  integers/float64 fail. Endianness is normalized without changing samples.
- Outputs: owned writable HWC arrays/tensors, including C=1 gray. out="bytes"
  is packed little-endian samples. out="torch" returns CPU unless device="cuda".
- GPU input requires allow_device_copy=True. Codec execution is CPU; tensors
  detach from autograd. No output points into freed Rust memory.
- output=None returns bytes; a path writes and returns Path; a binary writable
  stream receives all bytes and returns None. Encoding finishes before writing;
  I/O failure can leave partial output. Encoded inputs are fully buffered.

## Still Images

```python
import numpy as np
import rs_jxl

rgba = np.zeros((64, 96, 4), dtype=np.uint16)
rgba[..., 0] = 12345  # retained under zero alpha
data = rs_jxl.encode(rgba, effort=9, threads=8)
assert np.array_equal(rs_jxl.decode(data), rgba)
archived = rs_jxl.encode("input.jpg", effort=9)
original = rs_jxl.reconstruct_jpeg(archived)
pixel_jxl = rs_jxl.encode("input.jpg", lossless_jpeg=False, strip_metadata=True)
```

encode defaults: effort=7, threads=1, lossless=True, distance=0,
lossless_jpeg=True, source_search=False. JPEG bytes default to coefficient
transcoding; unsupported structures fail without pixel fallback.
transcode_jpeg is explicit, defaults effort=9, and exposes source_search,
brotli_quality, no_cfl, order=source/default/adaptive, limits. decode_to_jpeg
aliases reconstruct_jpeg: original bytes only, not lossy JPEG encoding.

PNG/JPEG pixel input shares the CLI implementation. PNG retains 8/16-bit
gray/GA/RGB/RGBA, including invisible RGB. JPEG pixel losslessness is relative
to its decoder's samples. Unsupported metadata requires strip_metadata=True;
supported color metadata remains. No EXIF orientation transform is applied.
Arrays accept bits_per_sample, color_encoding=srgb/linear_srgb, icc_profile
(bytes/path), orientation=1..8; explicit color and ICC are mutually exclusive.
Encoded input cannot apply array metadata overrides. Lossy requests fail.

## Encoder Parameters

Effort 1..11 uses the Rust source backend; source_search=True enables the still
scheduler (e11 tournament). Same effort does not promise C++ size/speed parity.
threads=0 explicitly selects CPU count capped at 256; threads=1 is serial.

encoder_options={...} or keyword arguments accept group_size_shift, rct,
predictor, palette_colors, patches, learning_fraction and ma_max_leaves.
Aliases modular_group_size/modular_colorspace/modular_predictor/
modular_palette_colors map to the first four respectively. iterations and
modular_ma_tree_learning_percent map to learning_fraction=percent/100.

```python
data = rs_jxl.encode(
    "content.png", effort=9, modular_group_size=3, iterations=100,
    modular_predictor=0, modular_palette_colors=10000,
    patches=False, modular_colorspace=0,
)
```

Duplicate controls fail. source_search rejects explicit tool overrides because
its candidates cannot guarantee them. ma_max_leaves is an explicit resource
ceiling, not an implicit quality reduction. Raw cjxl flags and numeric libjxl
frame-setting IDs are not forwarded to another encoder.

## Extra Channels

```python
depth = np.zeros((64, 96), dtype=np.uint16)
data = rs_jxl.encode(rgba, effort=3, extra_channels=[
    {"data": depth, "type": "depth", "name": "distance"},
])
pixels, details = rs_jxl.decode(data, return_extra_channels=True)
depth_back = details["extra_channels"][0]["data"]  # HxWx1
```

Explicit planes append after interleaved alpha. Specs expose type/name/depth,
dim_shift, alpha_premultiplied, spot_color, cfa_channel where applicable.
Input subsampled size is ceil(width/2**dim_shift) x ceil(height/2**dim_shift).
Rendered decoded planes are full resolution; metadata keeps original dim_shift.
return_extra_channels=True returns (pixels, metadata); alpha is excluded unless
include_alpha_extra=True. decode_extra_channel(..., channel=N) uses the complete
native index including alpha. Mixed plane dtypes remain distinct.

## Frames and Layers

```python
changed = rgba.copy()
changed[12:20, 30:40, 0] = 7
data = rs_jxl.encode_multiframe(
    [rgba, changed], durations=[10, 10], reference="previous", effort=3,
)
second = rs_jxl.decode(data, frame=1)
all_frames = rs_jxl.decode(data, frame=None)
bounds = rs_jxl.analyze_multiframe([rgba, changed])
```

reference=previous/first/full/auto (none aliases full). Default strategy="crop"
uses geometry; reference=auto chooses the smaller first/previous bbox, not
measured bytes. strategy="auto" additionally encodes eligible full/crop/mask/
multiple-rectangle candidates and retains byte winners, at greater cost.
return_info=True returns (encoded_output, strategy_report).

All channels including hidden RGB participate in differences. durations is a
tick count or list, default 1; tps=(1000,1), loops=0. Zero is a true layer, not
a fabricated display duration; visible-frame count can differ from layer count.
decode_layer(data, layer=N) returns (raw_layer_pixels, metadata), with crop,
blend and references under header. This is not signed arithmetic residuals.
Multiframe extra data accepts a shared plane or list/tuple of one plane per
input. analyze_multiframe returns exact previous/first bounds, not byte estimates.

## Decoding and Safety

decode defaults to frame=0, coalesced native samples, orientation adjustment,
dither="jxl-rs". frame=None retains all frames; targeted selection uses native
target-frame decoding. dither="libjxl" affects floating-to-U8 only, not exact
integers/JPEG bytes. format=u8/u16/f32 explicitly requests sample conversion.
info reads JXL headers/metadata. decode_to_png exports representable native
8/16-bit pixels; unsupported float/wide-integer/extra-plane output is rejected.

Decode/info/reconstruction limits={...}: max_input_bytes/max_output_bytes/
max_metadata_bytes/max_pixels/max_samples/max_frames. decode also accepts
max_pixels/max_output_bytes directly. Inclusive budgets are not RSS bounds;
omission uses core defaults. PNG/JPEG pixel read_image does not expose budgets
yet. JPEG transcode limits are in the native protocol; unsupported limits fail.

Input pointers live only through the call. Rust owns result handles; Python
copies buffers and frees results in finally. CFFI checks ABI version/size and
releases the GIL during native calls. Independent calls have independent pools
and results; avoid outer-worker times inner-thread oversubscription. Invalid C
pointers cannot be made safe by an ABI check; see the header's caller contract.

[Protocol](../crates/rs-jxl-ffi/protocol.md),
[C header](../crates/rs-jxl-ffi/include/rs_jxl.h).
Not implemented: Python incremental streaming, zero-copy outputs, reusable
decoder contexts, arbitrary manual blend headers, full C++ parameter parity.

## Tests

Install pytest separately for development. The full command below includes
the pytest-parameterized packaging tests; unittest alone only runs the adapter
and native binding tests.

```powershell
$env:PYTHONPATH = "python"
python -B -m pytest python/tests -q
cargo test --locked --offline --workspace
```

RS_JXL_DJXL optionally supplies official decoder paths for independent tests.
Set RS_JXL_TEST_CUDA=1 to require the GPU transfer checks on a CUDA host.
Acceptance compares original samples including hidden RGB, or original JPEG
bytes, not compressed JXL identity. Cross-platform runtime claims need actual
tests on those systems, not just a matching wheel platform tag.

### Initial Binding Validation

On Windows x86-64 with Python 3.10 and the release DLL:

- Python suite: 116 passed, including packaging, native bindings, official
  djxl interoperability and CUDA transfers. Two environment/tooling warnings
  (pkg_resources deprecation and Py_DEBUG configuration) remained non-fatal.
- The real platform wheel was installed outside the source package path with
  RS_JXL_NATIVE_LIB unset; all 32 native integration tests passed using its
  bundled DLL. Wheel and sdist checks retained native sources, runtime assets
  and third-party license/provenance notices.
- Rust workspace: 2,352 passed, 107 ignored, zero failures. Ignored external
  corpus/oracle tests are not counted as passed. Strict workspace Clippy and
  rustfmt checks passed; existing Python tooling tests also passed (181).

This is a functional and packaging checkpoint, not a performance comparison
or Linux/macOS runtime certification. No full dataset benchmark was rerun.
