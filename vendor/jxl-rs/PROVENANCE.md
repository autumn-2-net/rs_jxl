# Decoder Source Provenance

Upstream: https://github.com/libjxl/jxl-rs
Revision: `5ccf8694e3779e2931ca487d46251e3050bf849b`
Package version: 0.8.0
Imported: 2026-10-23

This is maintained decoder source, not an alternative encoder or C++ binding.
The independent encoder remains a direct port of the optimized C++ source.
Original copyright notices, AUTHORS, LICENSE and PATENTS are retained.

Imported crates: jxl, jxl_macros, jxl_simd and jxl_transforms, including their
source and supporting manifests/licenses. Upstream test-image corpora, fuzz
workspaces, Git metadata and build products are excluded. The runtime dither
lookup table is source data, not a generated codec binary. Fixture-dependent
upstream tests require the original corpus from the pinned revision; ordinary
root-workspace tests use independently generated encoder fixtures instead.

The decoder keeps its own nested Cargo workspace and upstream lint settings.
The parent depends on local paths; its lockfile records utility dependencies.
Default parent features enable scalar/x86 SIMD and Brotli box decoding, not
new ARM work. Existing upstream non-x86 source is retained unchanged.

## Local Patch Ledger

- U8 floating-point output quantization: retain upstream jxl-rs DZgas by default
  and expose a per-decoder `Libjxl` option using the C++ libjxl v0.12.0
  `render_pipeline/stage_write.cc` blue-noise constants. Preserve source
  attribution and repeat columns for Rust SIMD widths. Native integer pixels,
  Modular smooth-squeeze, F32 output and JPEG byte
  reconstruction are unchanged. Small upstream floating-point reconstruction
  differences can still cross a quantization threshold; this does not promise
  universal bit-exact agreement with the C++ pixel renderer.
- Workspace import: retain the four decoder crates and exclude unrelated
  upstream CLI/CMS/integration/gain-map packages. No algorithm change.
- Extra-channel API: expose source depth, exponent, name, dimension shift,
  spot color and CFA index already present in decoded headers.
- Header strings: decode UTF-8 byte sequences as UTF-8 instead of converting
  each byte to a character. Invalid UTF-8 produces an explicit error.
- Legacy profile compatibility: an explicit default-off option permits Main10
  complexity in unmarked streams, matching older djxl; budget checks remain.
- F16 conversion/output: retain signaling-NaN sign and payload bits. Finite
  values keep the existing SIMD path; tests cover all 65,536 F16 patterns.
- Spline boundaries: apply the support-radius color floor after multiplying
  intensity, and clip signed horizontal bounds before conversion to usize.
  This matches the C++ reference for zero/low-intensity and off-canvas segments.
- Spline resampling: use libjxl's residual arc-length interpolation instead
  of accumulating a normalized fixed step, reducing long-curve position drift.
- Opt-in JPEG coefficient capture: preserve quantized DC/AC coefficients and
  raw quantization tables before dequantization. Normal pixel output does not
  enable the capture; reconstruction metadata parsing remains in `rs-jxl`.
- Layer/preview API: expose crop, blend/reference, timecode and preview headers;
  allow uncoalesced regular layers, including zero-duration layers. Preview
  rendering uses preview dimensions rather than the main canvas. Header-only
  scanning skips invisible reference/LF sections as well as visible pixels.
- Mixed-resolution patches: allow independent color patches with differently
  upsampled extras; reject cross-resolution extra/alpha blending where libjxl
  does too. Preserve reference storage separately from raw-layer output.
- Progressive quality: select eligible VarDCT HF passes at full canvas size,
  retain full reference frames, and expose actual output progression rather
  than equating the requested ratio with a resized image.
- JPEG capture extensions: decode quantized groups without a pixel/IDCT pipeline,
  accumulate progressive passes with overflow checks, recover exact extra-DC
  precision, and honor compatible non-unit quantizers. Fractional/clamped or
  otherwise unreconstructable data is rejected, not silently rounded.
- JPEG coefficient capture fast paths: compute integer quantization ratios once
  per frame, cache each chroma channel's last-factor CfL scales per group, and
  bypass prediction arithmetic for zero factors. Cache group geometry and move
  group-buffer allocation outside the capture-state mutex. Keep signed division,
  both fixed-point roundings, each checked intermediate, coefficient bounds and
  coefficient-order failures; a cached scale overflow is reported only when that
  AC coefficient is visited. Differential tests cover all i8 factors, extreme
  quantization/residual values, factor changes, transposition and error ordering.
  Completion still validates every DC, but skips the second strided write pass
  when the RGB-offset correction is zero (including all YCbCr captures).
- JPEG capture SIMD: borrow checked i32 scratch blocks instead of copying and
  clearing 192 temporary values per block. An AVX2 kernel in `jxl_simd` transposes
  8x8 integer blocks, restores CfL and narrows only after validating the complete
  output range. Guarded envelopes prove i32 multiplication/addition cannot
  overflow on the committed path; all rejected fast attempts leave output
  untouched and replay the ordered scalar checks. Ignored DC remains unchanged.
  Native Rust intrinsics are isolated in the existing SIMD crate; `jxl` retains
  its unsafe-code prohibition. No new ARM path. No-AVX builds keep scalar capture.
- Optional AVX-512 JPEG capture: a four-register 8x8 transpose uses two-source
  permutations, unsigned interval checks collect compact masks, and 16 checked
  coefficients narrow per store. This is a separate native Rust kernel, not an
  automatic AVX2 width substitution. It retains the same conservative arithmetic
  envelope, signed rounding, DC preservation and no-commit-on-failure contract.
  The parent `decoder-avx512` feature enables runtime dispatch with AVX2/scalar
  fallback; it is not a global CPU-specific compiler flag or an encoder change.
- AVX2 capture range checks translate the accepted signed interval to the
  lowest signed values, replacing two comparisons plus OR with wrapping add
  plus one comparison. Packing retains the original 128-bit stores: the paired
  256-bit-store experiment was slower and was removed.
- JPEG coefficient entropy dispatch: select ANS, Huffman or LZ77 once per
  block rather than per symbol. Cache nonzero contexts and specialize unshifted
  accumulation using checked i32 addition; shifted passes retain checked wide
  arithmetic. Token-zero bypass has the same HybridUint value and consumes no
  extra bits. Differential loop tests preserve contexts, coefficients, partial
  state and overflow/exhaustion failures.
- Noise rendering: use base color correlation rather than LF/DC offsets,
  following libjxl `stage_noise.cc`; retain every encoded 10-bit strength in
  SIMD LUT lookup instead of truncating entries through a BF16 table.
- Rendering arithmetic: align 2x/4x/8x upsampling's three accumulators and final
  sum, Gaborish's shared reciprocal/pairwise sums, and scalar adaptive LF
  smoothing's grouping with the C++ reference. This is source arithmetic
  compatibility, not a relaxed floating-output tolerance or a performance claim.
- XYB Gamma/DCI output: use libjxl's near-black cutoff at linear values `<=1e-5`
  instead of mirroring negative inputs. Native original-color samples bypass
  this stage. This does not resolve the separately documented sRGB residuals.

Update this ledger and docs/DECODER_STATUS.md for behavioral patches. Do not
replace this directory from upstream without reviewing and replaying patches.

## ANS Differential Regression Coverage

`entropy_coding/ans.rs` has additional test-only comparisons against its original
symbol/state update, including short inputs, refill boundaries and continued
reading. The parent `decoder_hybrid_uint` target compiles this actual module and
the actual bit reader, and runs the upstream histogram fuzz test using test-only
`arbtest`. The runtime ANS implementation is unchanged by this test addition.
See the parent `docs/DECODER_ENTROPY_20261025.md` for the rejected experiments.

## Exact Weighted-Predictor Lookup

`frame/modular/predict.rs` uses the private `weighted_weights.rs` helper for
the four error-to-weight calculations. For errors below 256 and header weights
below 16, a shared compile-time 16 KiB table returns the exact original integer
formula, including reciprocal truncation before multiplication. Other inputs
retain that formula across the full u32/u8 domain. Error accumulation still
wraps as before. This adds no unsafe code and removes four unchecked reciprocal
table reads from the predictor. There are no bitstream or rendering changes.

The 2026-10-25 follow-up expands exact-formula regression coverage through
error 1023, the 1024 boundary, and fallback-domain inputs. The experimental
1024-entry table was rejected; the runtime table remains 256 entries/16 KiB.
See the parent `docs/DECODER_THREADS_20261025.md` for measurements. Direct typed
output ownership is implemented in the parent adapter, not a vendored patch.

The parent repository's `decoder_weight_lookup` integration target compiles the
actual helper in isolation because upstream unit-test fixtures are not bundled;
ordinary decoder tests and real-image comparisons exercise it in production.
See `docs/DECODER_SINGLE_20261024.md`, Refinement Masks and Exact Weight Lookup,
for measurements and the explicit upstream test-fixture limitation.

## Optional Decoder Diagnostics

The non-default `decoder-profile` feature adds inclusive channel-route, Modular
substream, palette/RCT and VarDCT group scopes. Periodic narrow-function probes
are diagnostic only, include timer overhead, and do not provide precise CPU
attribution for inlined arithmetic. The parent exposes the same opt-in feature
and appends these counters to `RS_JXL_PROFILE=1` output. Ordinary builds compile
out the module and all hooks. No codec decisions or arithmetic change and no
unsafe code is added. See `docs/DECODER_SINGLE_20261024.md`, Stage Hotspot Audit.

## Native Integer Capture

The public `JxlNativeIntegerCapture` extension intercepts reconstructed Modular
planes before float conversion. It retains integer references, composes exact
Replace/Add/binary-alpha operations, and applies integer-compatible patches.
The ordinary rendered-output path is unchanged when capture is disabled.
Fractional/resampled/VarDCT operations are explicitly rejected by this capture
mode rather than approximated. See `docs/NATIVE_INTEGER_DECODE.md` in the parent
repository for API scope and verification evidence.

## Input Validation and Sequential Containers

- Patch dictionaries now validate the entropy reader's final state, reject
  overflowing coordinates/extents and out-of-range u32 blend symbols before
  conversion, and check reference-slot indexes before access.
- Version-1 out-of-order `jxlp` input can decode sequentially even where a
  physical-file seek checkpoint cannot be represented. New seek entries stop
  at that boundary; version-0 ordering restrictions remain unchanged.

## Per-Decoder Output Dithering

`JxlDecoderOptions::dither` selects `DitherMode::JxlRs` (the unchanged upstream
default) or `DitherMode::Libjxl` for floating-point samples saved as U8. Decoder
state passes the mode to normal/extra-channel save stages and LF previews;
simple and low-memory pipelines share an immutable table reference selected at
stage construction. SIMD loops do not branch on the mode. The original
`dither_32x32.bin` ranks and global `DITHER_TABLE` remain upstream-compatible,
including for Modular smooth-squeeze reconstruction. The separate reference
array is in `util/dither_djxl.rs`; there is no global mode or environment switch.
Floating-point output, native integer output and JPEG-byte capture do not use
this output choice. This is table selection, not a promise of complete pixel
identity with libjxl or a change to rounding arithmetic.

The parent `decoder_dither` target compiles the actual `util/dither.rs` with
minimal `util::box_array` and `api::DitherMode` adapters. `decode_dither` exercises
the production decoder's per-instance state, output orientations, extra planes,
native/float output and JPEG reconstruction. These changes require central
build/test validation; no local verification result is implied by this ledger.
