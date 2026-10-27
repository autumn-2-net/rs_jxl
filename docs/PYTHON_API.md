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

### Input Copies and Borrowing

`encode` and `encode_multiframe` default to `borrow_inputs=False`: mutable pixel
buffers, including CPU tensors and read-only views of mutable storage, are
snapshotted before the native call. Files/streams are read completely into memory;
this is not a file mapping or incremental input API. Immutable bytes may be
retained directly. Mutable encoded input passed through `read_bytes` is copied.

`borrow_inputs=True` permits borrowing eligible contiguous encoding buffers for
one synchronous call. The caller must not mutate the source or any alias until
the call returns, including from other threads while CFFI releases the GIL.
Marking one view read-only does not prevent mutation through another alias.
Exporters are retained through the call. This is best-effort transport borrowing,
not end-to-end zero-copy encoding: layout/byte-order normalization, GPU-to-CPU
transfer, multiple frames/extra planes consolidated into one payload, and Rust
image preparation can still allocate and copy. Borrowing does not bypass dtype,
layout, metadata or resource checks.

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

## Reports

`encode(..., return_info=True)` and `transcode_jpeg(..., return_info=True)`
return `(encoded_output, meta)`. The first element keeps the ordinary bytes,
Path or None behavior of `output`; codec statistics are under `meta["report"]`.
This also applies when `encode` selects JPEG byte transcoding.

```python
data, meta = rs_jxl.encode(rgba, effort=9, source_search=True, return_info=True)
report = meta["report"]
print(report["route"], report["bytes"], report["codec_seconds"])
```

Still-image reports expose `route`, `implementation`, `pixels`, `bytes`, `bpp`,
`codec_seconds`, `mpix_per_second`, `full_encodes`, `probe_encodes`,
`winner_index`, `effective_parameters` and `preparation_reuse`. Optional values
can be None; a direct candidate has no search winner index. The Python projection
does not include the complete Rust search transcript. `implementation` is the
stable lowercase string `modular` or `fast_lossless`. When effective parameters
are present, their `patches` value is `auto`, `on` or `off`, not a Rust Debug name.
`effective_parameters` is a scheduler `CandidateParams` projection, not a full
backend recipe or exact tool list; FJXL uses its own default effort.
JPEG reports instead provide `route` (`jpeg_transcode` or `jpeg_search`), `codec_seconds`,
`backend_effort`, `winner_index`, `full_encodes` and `trials` with each trial's
stage, backend effort and bytes. Output size also remains in top-level
`meta["bytes"]`; do not assume identical report fields for both operations.

`codec_seconds` excludes worker-pool creation, external pixel-input decoding,
Python/FFI transport and external I/O. For JPEG transcoding it includes JPEG
parsing and coefficient preparation. It covers codec work for the call, including
search when requested, not an exact analysis/backend breakdown. Still-image bpp is
final output bits per original pixel; MPix/s uses the original pixel count and
codec time, not the sum of candidate/probe pixels. These are diagnostics, not
performance-parity claims. Multiframe `return_info` retains its separate
strategy metadata described below; no universal timing report is implied.

## Reusable Contexts

```python
with rs_jxl.Encoder(threads=8) as encoder, rs_jxl.Decoder(threads=8) as decoder:
    data, meta = encoder.encode(rgba, effort=9, return_info=True)
    pixels = decoder.decode(data)
    archived = encoder.transcode_jpeg("input.jpg", effort=7)
    original_jpeg = decoder.reconstruct_jpeg(archived)
```

Constructors accept `threads=1`; `threads=0` resolves the CPU count capped at
256, as in the one-shot API. `Encoder` supplies `encode`, `transcode_jpeg` and
`encode_multiframe`. `Decoder` supplies `decode`, `decode_into`, `decode_layer`,
`decode_extra_channel`, `info`, `reconstruct_jpeg` and `decode_to_png`. Options
and return values match the corresponding module-level functions.

Thread counts are fixed for a context. Methods inherit that count, and an
explicit `threads` value must normalize to the same count. All other options
are per-call; effort, limits, frame selection and output format from one call do
not become defaults for the next. Contexts reuse workers, not input buffers,
parser state, decoded frames, reference slots or candidate caches. One thread
does not create a pool. `info` and PNG/JPEG pixel reading are stateless and do
not reuse JXL decoder workers. Context-bound `decode`/`decode_into` still send
PNG/JPEG `read_image` requests through the context's closed/lifecycle guard.

`close()` is idempotent; `.threads` and `.closed` expose the context's settings
and lifetime. Context-manager exit closes it, and operations after close raise
RuntimeError. The context lock serializes the closed check and native execution
against close; it is released before Python result conversion or copying.
Conversion callbacks can close or reenter the context without holding that lock,
and the current result retains independent ownership. This is not whole-method
serialization or protection against concurrent mutation of caller buffers.
Independent contexts can run concurrently; avoid multiplying outer workers by large inner
thread counts. Ordinary request/codec failures permit reuse. A native panic
poisons the handle; close it rather than attempting more work. Context creation
requires the optional native context symbols and raises ImportError when an
older ABI-v1 library lacks them; one-shot ABI compatibility is unchanged.

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
The strategy report is the metadata dictionary itself, with `frames` candidate
details for strategy="auto"; it is not the still/JPEG `meta["report"]` schema.
`borrow_inputs` has the same snapshot/borrowing contract as still-image encode,
though consolidation of several buffers can require copying.

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
Selected-frame decoding does not validate later data. Use frame=None when
whole-input decoding is required; a selected frame's `complete` flag is not
whole-file validation.

### Preallocated Output

```python
destination = np.empty(rgba.shape, dtype=rgba.dtype)
returned, metadata = rs_jxl.decode_into(data, destination, return_info=True)
assert returned is destination
```

`decode_into(src, destination, ...)` decodes one nonnegative frame index (not
frame=None). The destination must be a writable, C-contiguous NumPy ndarray
with the exact returned HWC shape, including C=1 for gray, and exact
native-endian dtype. Torch destinations, reshaping, narrowing and implicit
destination conversions are not supported. Explicit `format` conversion, when
requested, occurs in the decoder and the destination must match that result.
This entry point does not fill separate extra-channel destinations.

Decoding and descriptor/buffer/destination checks complete before any copy;
decode or validation failures leave the destination untouched. Do not mutate
the destination or its aliases concurrently with the call. The copy reads the
live Rust result directly, avoiding intermediate Python bytes and a newly
allocated returned array. It still uses Rust output storage and codec scratch,
and is not direct decoding into caller memory or a zero-copy output view.
Without `return_info`, the destination itself is returned; with it the result
is `(destination, metadata)`. `Decoder.decode_into` has the same contract.

### Limits and Errors

Decode/info/reconstruction limits={...}: max_input_bytes/max_output_bytes/
max_metadata_bytes/max_pixels/max_samples/max_frames. decode also accepts
max_pixels/max_output_bytes directly. Inclusive budgets are not RSS bounds;
omission uses core defaults. PNG/JPEG pixel read_image does not expose budgets
yet. JPEG transcode limits are in the native protocol; unsupported limits fail.

Input pointers live only through the call. Rust owns result handles; Python
copies into owned output or the validated destination while the result is live,
without returning a borrowed Rust view. For nonempty data results, the internal
`_ffi._result_buffer` attaches an `ffi.gc` owner to the data pointer and
`ffi.buffer` retains that owner. Temporary derived views retained by an exception
traceback can therefore safely keep the native result alive beyond helper scope
exit. Cleanup releases the helper view; the owner's finalizer frees the native
result once no retaining references remain. Native error responses and empty-data
results still free directly in `finally`. This lifetime protection does not
change the public owned-output contract or expose a zero-copy output API.

CFFI checks ABI version/size and releases the GIL during native calls. One-shot
calls create their own resources; explicit contexts reuse workers, with
independently owned results. Invalid C pointers cannot be made safe by an ABI
check; see the header's caller contract.

Native failures raise `CodecError` with the original message and a `.kind`
category: `invalid_input`, `resource_limit`, `encoding`, `decoding`, `io`,
`internal` or `unsupported`. The ABI preserves its `error` string and adds
`error_kind` and optional `error_stage`. Python currently exposes `.kind`, not
a separate stage attribute; older libraries without a category yield `unknown`.
Classification follows known typed Rust errors; unclassified/string-only errors
can still be `invalid_input`. Python argument validation and path/stream failures
can raise ordinary TypeError, ValueError, IndexError or OSError before/after the
native call. Do not classify failures by matching message substrings.

[Protocol](../crates/rs-jxl-ffi/protocol.md),
[C header](../crates/rs-jxl-ffi/include/rs_jxl.h).
Not implemented: Python incremental streaming, public zero-copy output views,
direct native decode into user buffers, arbitrary manual blend headers or full
C++ parameter parity. Reusable contexts and `decode_into` do not close those
gaps. Native Rust decoder metadata types still include maintained-upstream
reexports; this is not a complete backend-independent public type layer.

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

### Current Verification

Validated on Windows x86-64 with Python 3.10 on 2026-10-27:

- Full Rust workspace regression: 2,377 passed, zero failed, 107 ignored before
  the final diagnostic-retention review fix. Ignored tests are not passes.
- After that fix, the complete `codec_context` target passed all 8 tests,
  including two new artifact/receipt-preservation tests. This focused rerun is
  separate from the earlier full-workspace count.
- Final strict workspace Clippy (`--all-targets -- -D warnings`), formatting
  and whitespace checks passed. Release CLI and FFI builds completed.
- Full `pytest python/tests` against the final release DLL: 165 passed with
  official `djxl` interoperability and CUDA transfers enabled. Two preexisting
  tooling warnings (`pkg_resources` and `Py_DEBUG`) remained non-fatal.
- The final wheel was installed into `out/python-architecture-final-installed`.
  From the `out` working directory, with `RS_JXL_NATIVE_LIB` unset and the imported
  package path asserted to be installed, all 92 binding, context, adapter and
  buffer tests passed using the bundled DLL. This includes 45 native binding
  and context tests; packaging tests are included in the 165-test source run.

The pure derived-view test in [test_adapters.py](../python/tests/test_adapters.py)
uses a mocked result to check owner retention and exactly-once release after a
derived NumPy view is dropped. Native tests additionally cover closing a context
from an output-conversion callback, closed PNG/JPEG input callbacks, independent
concurrent handles, error recovery, multiframe state and byte-exact JPEG recovery.

These are functional and installed-wheel checks, not a performance comparison,
a full-corpus rerun, a release-profile Rust test run or cross-platform runtime
certification. The historical counts below describe a different checkpoint.

### Initial Binding Validation (Historical)

The earlier binding checkpoint recorded the following on Windows x86-64 with
Python 3.10 and the then-current release DLL, before this context/buffer refactor:

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
