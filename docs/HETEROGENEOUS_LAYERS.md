# Heterogeneous-Size Layers

`rs_jxl::layers` encodes differently sized pixel images as regular JPEG XL
layers on a shared canvas. It does not resize, pad or convert input samples.
The native Rust module owns geometry, timing, validation and frame coding;
Python and the C boundary only transport images and options.

## Python

```python
import rs_jxl

data, report = rs_jxl.encode_layers(
    ["large.png", "small.png"], effort=9, threads=4, return_info=True,
)
original_small, metadata = rs_jxl.decode_layer(data, layer=1, adjust_orientation=False)
```

Inputs use the existing array/tensor/path/encoded-bytes/stream adapters. Output
may be returned as bytes, written to a path, or written to a binary stream.
`Encoder.encode_layers` reuses a native worker pool. Arrays can use explicit
input borrowing under the same lifetime and mutation contract as other APIs.

The default canvas is `(max_width, max_height)` for origin-zero images. Each
image retains its dimensions and samples, including RGB under zero alpha.
Layout, bit depth, color encoding, orientation and extra-channel metadata must
match; images are not silently converted to a common format. JPEG input is
decoded as pixels, not stored using byte-reversible JPEG transcoding.

Each default layer has duration zero, REPLACE blending and background source 1.
Slot 1 is initially zero and these defaults never overwrite it. Thus normal
coalesced decoding produces the last layer on an otherwise zero canvas, not
stale edges from earlier images. For RGB, zero means black; RGBA has zero alpha.
`decode_layer(data, layer=i)` instead returns `(pixels, metadata)` for the raw
rectangle; `return_info=False` returns only pixels. Decoding normally applies
orientation metadata. Use `adjust_orientation=False` for the stored coordinate
and sample order when orientation is not identity.
Zero-duration intermediate layers are not separate displayed animation frames.

## Placement and Animation

```python
data = rs_jxl.encode_layers(
    [first, second], canvas=(1920, 1080),
    layer_options=[
        {"origin": [40, 20], "duration": 100, "name": "first"},
        {"origin": [-10, 50], "duration": 200, "name": "second"},
    ],
    tps=(1000, 1), loops=0, effort=9,
)
display_frames = rs_jxl.decode(data, frame=None)
second_original, metadata = rs_jxl.decode_layer(data, layer=1, adjust_orientation=False)
```

Canvas order is `(width, height)`; origins are `(x, y)` in canvas coordinates.
Without an explicit canvas, nonnegative origins contribute to the inferred
maximum right/bottom extent. Negative origins require an explicit canvas.
A layer may extend beyond an explicit canvas: only composited output is clipped;
raw decoding still returns every original sample, including out-of-canvas data.

Provide exactly one `layer_options` mapping per image when specifying settings.
Allowed fields are `origin`, `duration`, `timecode`, `name`, `blend`,
`extra_blend` and `save_as_reference`. Blends accept `mode`, `source`,
`alpha_channel` and `clamp`. Unknown or malformed fields are rejected.
The full Python contract is in [PYTHON_API.md](PYTHON_API.md#explicit-layers).

Nonzero durations or timecodes infer animation metadata when not explicit.
Python's default inferred timebase is 1000 ticks/second; `tps` overrides it.
Native explicit animation metadata with timecodes disabled rejects nonzero
frame timecodes. `loops=0` means unlimited animation repetitions, not no frames.

## Explicit Reuse

For a timed background and a later crop that should inherit it:

```python
data = rs_jxl.encode_layers(
    [background, crop],
    layer_options=[
        {"duration": 100, "save_as_reference": 2},
        {"origin": [100, 200], "duration": 100, "blend": {"source": 2}},
    ],
    effort=9,
)
```

This REPLACE crop overwrites only its rectangle. The caller supplies the crop;
this API does not infer differences or search for repeated blocks. Source 0 can
reuse the immediately preceding zero-duration composite under default saves.
Timed frames need explicit saved references. A saved source is no longer zero;
explicitly saving slot 1 changes subsequent source-1 backgrounds intentionally.

Reference IDs are 0..3, not arbitrary previous-frame indices. The encoder's
existing patch tools reserve slot 3. Post-transform references are used for
frame blending; patch dictionaries require saved pre-transform content.
Missing references are valid zero backgrounds for blending, but are not valid
patch sources. These are existing JXL semantics, not private metadata.

Exact archival reconstruction uses raw-layer decoding or tested REPLACE
compositing. Other blend modes are exposed but do not promise that displayed
pixels equal raw input samples, especially with fractional alpha or floats.

## Rust and C

```rust,ignore
use rs_jxl::{EncodeOptions, layers::{self, LayerOptions}};

let options = EncodeOptions { threads: 4, ..Default::default() };
let result = layers::encode(&images, &options, &LayerOptions::default())?;
```

`LayerOptions.canvas` is optional; `frames` may be empty for independent defaults
or contain one `FrameOptions` per input. Use `layers::default_frame_options()`
when overriding individual settings: bare `FrameOptions::default()` uses source
0, the low-level compositing convention. `LayerOptions.animation` overrides
per-input animation metadata for the sequence. `layers::plan` returns borrowed
frame inputs and a metadata-only `ImageDescriptor`, without a dummy canvas.

`LayerEncodeResult` contains bytes, canvas dimensions, per-layer geometry and
frame settings, and `codec_seconds` (native planning/coding, excluding pool
creation, transport and external PNG/JPEG decoding). `Encoder::encode_layers`
uses the context pool. Reference-only frames remain available via low-level
`encode_frames`; this API promises one regular readable layer per input.

The C ABI uses `operation: "encode_layers"` with `images`, `options` and
`layer_options: {canvas, frames, animation}`. It is additive to ABI version 1.
See [protocol](../crates/rs-jxl-ffi/protocol.md). Explicit tool options are
supported; source-search and legacy automatic-frame keys are not silently used.

## Scope and Compatibility

This is separate from `encode_multiframe(auto=True)` and `analyze_multiframe`,
which operate on equal-size complete states. No automatic heterogeneous-image
matching or temporal displaced-patch search is claimed. Possible improvements
without a new bitstream are described in [MULTIFRAME_REUSE_ROADMAP.md](MULTIFRAME_REUSE_ROADMAP.md).

Original dimensions, exact samples, zero-background behavior, offsets, references,
extras, output limits and reusable contexts have dedicated native tests in
`crates/rs-jxl/tests/layers.rs`; Python/FFI tests exercise transport and validation.
Independent decoder checks use official `djxl` in addition to the maintained
Rust decoder. These are focused acceptance cases, not a full format-conformance
or cross-platform certification.

## Verification Checkpoint (2026-10-27)

Windows x86-64, release `rs_jxl_ffi.dll`, Python 3.10:

- Native layer tests: 10 passed, including actual generated patch dictionaries
  whose reference-only atlases must not shift regular layer indices.
- Focused core regression tests: 66 additional passes covering reusable pools,
  encode validation, raw-layer decoding, automatic multiframe/analysis and
  streaming sequence validation. Four separate opt-in stream CLI-oracle cases
  were not run in that Rust command.
- Full C ABI suite: 44 passed. Full Python suite: 148 passed, including CUDA
  adapter coverage and official `djxl` 0.12/0.13 cross-decoding fixtures.
- Independent official-decoder layer cases compare original integer samples
  against expected canvases: partial initial layers on zero background,
  negative/positive offset clipping with saved references, and zero-duration
  sequences displaying only the final independent layer. Both 8- and 16-bit
  hidden RGB are checked.

The original three `t3` JPEG images (702x1000, 638x900, 638x900) were encoded
with effort 9 and four threads as pixel-lossless layers. Their raw decoded
rectangles matched the same native JPEG pixel loader exactly. This is not
byte-reversible JPEG transcoding. Independent comparison explicitly used
`lossless_jpeg=False` and the same file inputs, retaining input metadata.

| Route | Bytes | bpp per original input pixel | Native codec seconds |
| --- | ---: | ---: | ---: |
| Explicit heterogeneous layers | 587,165 | 2.53854 | 1.682 |
| Independent pixel JXL, sum | 588,409 | 2.54392 | 1.160 |

Layer wall time was 1.732 seconds; native throughput was about 1.10 MPix/s.
These are single smoke-run observations, not controlled repeated performance
measurements. Shared settings/header overhead differ, and no automatic temporal
matching is performed: the 0.21% size difference is not evidence for a new
inter-frame compression algorithm. A separate array-based independent run gave
586,696 bytes; it is not used in the table because file metadata was not passed
through the same input route.
