# Exact Native Integer Decoding

## Contract

The maintained `vendor/jxl-rs` decoder has an opt-in integer capture path for
original-color Modular JPEG XL. It retains the decoded integer samples after
the reversible Modular transforms, before normalization or float rendering.
This is needed for 17..31-bit data: an `f32` cannot represent every integer
above 2^24, and converting float output back to `u32` is not an exact decoder.

This mode does not change the existing rendered decoder. It is a native-sample
interface, not another output quantization setting. Compressed bytes need not
match any encoder baseline; tests compare original samples.

## Integration API

The project API selects this path automatically for `SampleFormat::Native`
when the source declares a wide integer color or extra plane:

```rust,ignore
use rs_jxl::decode::{decode, DecodeOptions};
use rs_jxl::PixelData;

let image = decode(&jxl_bytes, &DecodeOptions::default())?;
let PixelData::U32(samples) = &image.frames[0].pixels else {
    // Narrow integer images still return U8/U16, floating input returns F32.
    return Err("expected wide integer color".into());
};
```

Extra planes retain their own U8/U16/U32 storage. The interleaved alpha uses
the color range, so its separately returned plane is authoritative for unequal
depths. Disabling `include_extra_channels` explicitly discards that authoritative
plane; the remaining interleaved alpha can be rounded and is not a lossless
archive of unequal-depth alpha. Explicit F32 remains a normalized rendering
request, not exact U31.
Native wide output with a requested color/alpha conversion fails explicitly.
CLI `decode input.jxl output.npy` preserves native integers; NPY widens mixed
integer planes to their widest storage type without scaling sample values.

The lower-level maintained decoder API is:

```rust,ignore
use jxl::api::{Event, JxlDecoder, JxlDecoderOptions};

let mut options = JxlDecoderOptions::default();
options.render_spot_colors = false;
let mut decoder = JxlDecoder::new(options);
let mut capture = None;
loop {
    match decoder.process(&mut input, None, None)? {
        Event::BasicInfo => {
            // Inspect metadata here. Enable only if native integer output
            // is requested; explicit rendered float output uses the old path.
            capture = Some(decoder.enable_native_integer_capture()?);
        }
        Event::FrameComplete { .. } => {
            let frame = capture.as_ref().unwrap().take().unwrap();
            // frame.width, frame.height, frame.num_color_channels
            // frame.planes[c].bits_per_sample
            // frame.planes[c].samples: Vec<u32>, row-major original code values
            consume(frame);
        }
        Event::Complete => break,
        Event::NeedMoreInput { .. } => supply_more_input(),
        Event::FrameHeader => {}
    }
}
```

`enable_native_integer_capture()` can also be called before the first input.
It must not be enabled after a frame header has been constructed. The handle
survives the decoder releasing the final frame. Calling `take()` consumes the
output once; starting the next frame discards unread previous output. Decode
errors invalidate capture and its references. `reset()` disables capture.

Color planes are first: one gray plane or three RGB planes. All extra planes
follow in codestream order, including alpha and black. Extra planes retain
their own bit depths, independently of color depth. No alpha plane is silently
discarded or interleaved. Ordinary `JxlOutputBuffer`s are not populated in this
mode; pass `None`. CMS output profiles and ordinary output pixel formats do
not transform captured samples.

`coalescing=true` publishes visible full-canvas frames. `coalescing=false`
publishes crop-sized regular/skip-progressive layers, including zero-duration
layers. Internal reference-only frames are decoded but not published. The
existing frame-header API supplies durations, names, crop positions and blend
metadata. `adjust_orientation` transforms the output planes exactly; stored
references remain in codestream coordinates.

## Supported Integer Semantics

- Integer color and extra planes with declared depths 1..31, full resolution.
- Modular prediction, entropy coding, reversible RCT, palette and squeeze use
  the same existing integer inverse-transform implementations as rendering.
- All groups/passes must finish before any native frame is published.
  `flush_pixels()` does not publish an approximate native preview.
- Four integer reference slots, including pre-color-transform patch sources
  and post-color-transform composited sources.
- Full replacement, crop replacement, negative/off-canvas crop positions,
  zero-duration reference chains, and independent per-extra-channel sources.
- Patch `None`, `Replace`, and `Add`; signed patch residuals remain signed until
  reconstruction. Addition is checked for overflow, not silently wrapped.
- Binary-alpha `BlendAbove`/`BlendBelow` for patches and frame composition.
  Opacity is interpreted at the alpha plane's own bit depth. Associated and
  unassociated alpha use distinct rules. Unassociated transparent-over-
  transparent blending yields zero color according to the blend operation;
  this is not a hidden-RGB-preserving substitute for crop/REPLACE.
- Original invisible RGB is retained by full/crop replacement.

Reference images and intermediate patch values use signed integer storage.
Published samples must be in the declared unsigned range. Out-of-range output
is rejected, never clamped into a purportedly exact image.

## Explicit Boundaries

These requests/streams return `Error::NativeIntegerUnsupported`; callers must
not silently fall back to rendered float output while promising native lossless
samples:

- VarDCT, XYB, YCbCr conversion, floating-point color or extra planes.
- Color/extra upsampling, subsampled extra planes, or LF frames.
- Noise synthesis, splines, Gaborish and EPF restoration filters.
- Fractional alpha blending, multiply and alpha-weighted-add operations.
- Blending controlled by a non-alpha extra plane.
- Requested spot-color rendering when spot channels are present; disable it
  to retrieve the original planes.
- Requested alpha premultiplication conversion or tone mapping, scan-only
  capture, JPEG coefficient capture at the same time, or seeking while capture
  is enabled. Sequential dependency decoding is supported.

`Error::NativeIntegerInvalid` denotes an invalid/incomplete integer capture or
arithmetic overflow. Ordinary parser, allocation, sample-limit and entropy
validation errors are propagated. A successful frame does not certify trailing
container boxes; callers needing the whole container validated must await
`Event::Complete` too.

This is not a claim that every mathematically integral case of every blend
operation is supported. It intentionally excludes fractional rendering instead
of manufacturing an integer result by rounding.

The usual sample/pixel limits and fallible allocations apply, but
`max_output_bytes` bounds returned pixels, not total working memory. Integer
capture uses i64 intermediate planes, reference slots and temporary composition
buffers before publishing U32 output. No process-wide RSS bound is promised;
use conservative `max_pixels`/`max_samples` for untrusted input.

## Implementation And Validation

Main files:

- `vendor/jxl-rs/jxl/src/api/native_integer.rs`: public frame/plane/capture types.
- `vendor/jxl-rs/jxl/src/frame/native_integer.rs`: integer group sink, references,
  patch composition, crop composition and oriented publication.
- `vendor/jxl-rs/jxl/src/frame/native_integer_tests.rs`: helper-level exactness,
  boundaries, mixed depths, clipping and alpha semantics.

Integration hooks are in API decoder/frame parsing and frame decode/render
setup. The patch parser accepts reference dimensions/transform-space metadata
without requiring a floating-point reference image. Non-capture decoding keeps
the original pixel and reference pipeline.

The integer group scheduler mirrors `frame/render.rs`: mark final sections,
prepare Modular transform dependencies, decode groups and drain ready inverse
transforms. It deliberately uses no floating-point conversion or floating-point
reference buffer. Composition follows libjxl `dec_modular.cc`,
`dec_patch_dictionary.cc` and `render_pipeline/stage_blending.cc` semantics;
the existing vendored inverse transforms retain their upstream attribution.

Tests should cover every depth 17..31 around powers of two and maximum values,
random samples, mixed-depth extras, gray/RGB/RGBA, all orientations, multiple
groups, forced RCT/palette/squeeze/predictor routes, patches, raw layers,
reference chains, binary alpha, truncated input and unsupported rendering.
For wide samples, compare directly with the original integers. An official
decoder's normalized float output is not a reliable U31 equality oracle.

The Windows integration run passed seven wide-integer tests, including all 15
depths with gray/RGB/RGBA and threads 1/4, multigroup decoding, unequal alpha
depths, references and reversible tool routes. Eleven vendor integer helper
tests passed in the isolated fixture copy. See the dated checkpoint in
`DECODER_STATUS.md`; this is not a full conformance certification.
The initial implementation prioritizes correctness over peak-memory or
composition throughput optimization.
