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

Update this ledger and docs/DECODER_STATUS.md for behavioral patches. Do not
replace this directory from upstream without reviewing and replaying patches.
