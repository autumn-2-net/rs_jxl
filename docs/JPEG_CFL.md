# JPEG AC Chroma From Luma

This is a bounded pure-Rust JPEG coefficient prerequisite, not a JPEG-to-JXL
encoder. No pixel decoding, requantization, C++ dependency, coefficient-image
clone, or frame serialization is introduced.

## Frozen Sources

Port reference: `b866b3defec13494bfbaf705db93581cda118e37`, read-only libjxl.

- `lib/jxl/enc_frame.cc:779`: `FindAvgIndexOfSumMaximum`.
- `lib/jxl/enc_frame.cc:815,896-998`: zero-base JPEG correlation maps, AC fitting,
  force-option and three-component/4:4:4 gates, tile-row dispatch.
- `lib/jxl/enc_frame.cc:1044-1085`: separate original DC and integer AC residuals.
- `lib/jxl/chroma_from_luma.h`: 64-pixel / 8-block tiles, color factor 84,
  `RatioJPEG`, 11-bit fixed-point arithmetic, ratio limit 524287.
- `lib/jxl/chroma_from_luma.cc:46`: both base correlations zero for `XYB=false`.
- `lib/jxl/quantizer.h:36`: all three default zero biases are `0.5f`.
- `lib/jxl/enc_params.h:98`: source `force_cfl_jpeg_recompression` default is true.
- `lib/jxl/dec_group.cc:374-439`: inverse integer correlation, transpose, and
  replacement of DC from the independently reconstructed DC image.

BSD attribution is retained in `jpeg/cfl.rs` and `docs/PORTING_KERNELS.md`.

## Parent Integration

Parent registers `pub mod cfl;` in `jpeg/mod.rs`; this slice does not edit it.

```rust,ignore
use rs_jxl::jpeg::cfl::{CflMode, JpegCfl};

let mode = if force_cfl_jpeg_recompression {
    CflMode::Enabled
} else {
    CflMode::Disabled
};
let cfl = JpegCfl::compute(&layout, mode);
let tile_dimensions = cfl.dimensions_in_tiles();
let ytox = cfl.ytox_map();
let ytob = cfl.ytob_map();
cfl.forward_block(jxl_channel, plane_block_x, plane_block_y, &mut block)?;
```

`compute` borrows the validated `TranscodeLayout`. Maps remain tied to that
layout, so subsequent block application cannot accidentally use another JPEG's
quantization tables. It owns only two row-major `Vec<i8>` maps. Coordinates for
`forward_block`/`inverse_block` are **plane-local block coordinates**; `factor`
takes full-frame tile coordinates and a JXL channel index. Luma is channel 1;
`ytox` predicts channel 0, `ytob` predicts channel 2. `factor(1, ..)` returns zero.

Enabled means requested AND three components AND relative 4:4:4. There is no
RGB-only gate: YCbCr and RGB both qualify, using the layout's `JpegOrder`.
Disabled, grayscale, and subsampled layouts produce zero maps. Block application
then only transposes the original coefficients, including synthetic grayscale
chroma zeros. No default option or caller is changed by this module.

Tile dimensions use the layout's source-compatible padded frame-block geometry;
partial tiles are clipped, and stored MCU padding participates. Scalar fitting
uses existing `execution::map_with_state` independently per tile row, with channels
0 then 2. It creates no pool. A parent serial scope remains caller-local; an
installed caller pool is reused. Scratch is one 257-entry difference array per
active worker, reused across tiles. Row results are merged in deterministic order.

## Arithmetic Contracts

- Fitting visits block rows, block columns, then natural-order coefficients
  1 through 63. DC never votes. All source float intermediates stay `f32`, with
  the same multiplication/addition/division order and no explicit FMA or `f64`.
- The threshold is `(84.0f32 * 0.5f32) * 0.9999f32`. The zero base correlation
  leaves the source offset term `127.0f32 * scaled_m`.
- The interval updates use `ceil(from)` and `floor(to + 1)` with a sentinel at
  256. Only the lower endpoint's lower bound and upper endpoint's upper bound
  are clamped. Outside ranges cast no votes.
- Maximum selection averages the FIRST and LAST maximum indices, rounding up.
  Tied disjoint plateaus may yield a midpoint with a lower score; this is source
  behavior, not a claim of globally optimal selected factors. A nonzero factor
  requires more than one additional zero vote relative to index 127.
- Source assignment `best - 127` to `int8_t` is explicitly modeled with a Rust
  signed-byte cast: candidate 255 becomes -128 on the source's two's-complement
  targets. It is not clamped to 127 or changed to a different tie policy.
- `ratio_jpeg` computes `factor * 2048 / 84`, with signed division truncating
  toward zero. The two subsequent integer products each add 1024 and use an
  arithmetic right shift by 11. Negative half rounding is therefore asymmetric.
- Low-level `forward_ac_transposed` takes natural-order signed-i16 chroma and
  luma blocks plus natural-order fixed-point quantization ratios, and writes a
  transposed signed-i32 block. `inverse_ac_transposed` reverses this using an
  original/restored natural-order luma block. No full coefficient buffers are
  cloned: only caller output and a single local block scratch are used.
- Both low-level helpers copy DC unchanged. The source encoder's temporary AC
  block may contain a correlated DC, but it stores original DC separately and
  the decoder overwrites temporary DC from that image. This API keeps original
  DC explicitly; it neither drops DC nor applies the RGB DC offset itself.
- Signed-i16 inputs, including -32768, are not unsigned-reinterpreted, clamped,
  or restricted wholesale to the source comment's nominal 1024 magnitude.
  Each signed-i32 intermediate is checked. Larger coefficient/ratio/factor
  combinations that would overflow the scalar C++ expression return
  `ArithmeticOverflow`, rather than relying on C++ signed-overflow undefined
  behavior or Rust debug/release differences. The parent must propagate this
  error, not silently change the requested CfL mode.
- Inverse reconstruction checks the final signed-i16 range rather than using the
  source decoder's SIMD demotion/clamp. Unsupported values are explicit errors.
  Every helper leaves the caller output unchanged on error. Error coefficient
  indices are natural-order indices. DC does not consume a quantization ratio.

`compute` only fits maps; it does not preflight every subsequent integer block
application. Successful fitting alone is not an overflow-checked frame result.
`inverse_block` uses the layout's borrowed luma for encoder-side verification;
a future decoder uses the low-level inverse with its separately restored luma.

## Focused Tests

The module includes 11 tests covering signed-ratio goldens, negative-half
rounding, empty/tied CDFs, the strict improvement gate, positive and negative
correlation maps, the signed-byte endpoint, DC preservation, transposition,
partial tiles, quantization ratios from zero to near the source limit,
independent brute-force candidate-bin scoring, an independent wide-integer
reconstruction oracle, full signed-i16 input extremes, overflow/range errors,
grayscale/subsampling/RGB gates, bounds, and serial versus caller-pool results.

No Cargo builds/tests are run by this slice. Parent owns registration and central
compilation, test execution, linting, release checks, and corpus/oracle validation.

## Remaining Boundaries

- Default JPEG transcode integration: parent still connects map serialization,
  original DC handling, transposed AC group/pass storage, adaptive order inputs,
  AC entropy coding, and complete VarDCT/JXL frame assembly. This file does not
  establish working JPEG-to-JXL output or official decoder compatibility.
- Explicit unsupported input combinations: signed-i32 prediction overflow is
  reported at block application; no wrapping extension or silent fallback is
  provided. Quantization redefinition/global-table incompatibility remains the
  validated layout's responsibility.
- Optional: final-plan replay/exchange diagnostics and SIMD fitting are not
  ported. The frozen JPEG fitter itself is scalar with a SIMD TODO. General XYB
  color correlation, adjustable base ratios/color factors and DC correlation
  are not part of JPEG-compatible CfL and are outside this slice.
- The inverse helper is a bounded coefficient primitive, not a JPEG/JXL decoder
  or replacement for the source decoder's separate syntax/range validation.
