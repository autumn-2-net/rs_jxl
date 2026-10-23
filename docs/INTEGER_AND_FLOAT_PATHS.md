# Integer And Floating-Point Paths

Pixel lossless means recovering every original declared sample, including RGB
under alpha zero. It does not mean identical compressed bytes or identical
normalized intermediate floating-point values.

## Integer Input

`Image::modular_sample` keeps U8/U16/U32 input as integer samples. Modular
transforms, predictors, residual packing and entropy symbols use integer data.
Integer inputs are not routed through a blanket floating-point conversion.

Float costs in MA learning or entropy-model selection choose a reversible plan;
they do not quantize the original pixels. Approximate cost estimates can affect
size or speed without making the output lossy.

The source patch detector is different. Official libjxl uses Image3F and f32
metrics, and its Add compositor uses floating-point arithmetic. The port keeps
those source detection rules. Integer patch atlas/residual samples are signed
integers, with a separate decoder-arithmetic replay guard on covered pixels.
That guard checks recovered original integer values, not an epsilon or f32
bit equality. Exact integer addition alone cannot prove correctness after many
overlapping patches because the official decoder adds normalized floats.

## Float Input

Actual F16/F32/custom-float samples have a separate typed path. Their declared
bit words, including signed zeros and NaN payloads, remain distinct from signed
integer residuals. A float Add candidate must have representable atlas/residual
samples and recover the declared samples exactly. A failed check is not accepted
as visually lossless and is not silently rounded.

Thus neither "float implies lossy" nor "the source uses float, so it is safe"
is a valid correctness test. Each operation needs its own recovery argument
and independent decoding checks. The guarded float path is not a reason to
convert ordinary integer images to floats.

## SIMD Opportunities

Native std::arch SIMD already exists for integer RCT and stateless prediction,
with scalar fallback. The current prediction kernel uses i64 lanes to cover the
full i32 working domain after transforms. An i32 specialization could process
more lanes, but it needs bounds on transformed samples and intermediate
expressions, not just the original PNG bit depth. Narrow unsigned subtraction
or saturating instructions cannot represent signed residuals correctly.

Other candidates are fusing deinterleave/conversion passes, reducing temporary
buffers, and vectorizing patch comparisons or residual generation. None is a
measured speedup merely because it uses integers. Floating-point SIMD is also
vectorized, and overall throughput can be limited by memory or search.

The source RCT gradient-cost scan now has a fused i32 SSE2/AVX2 path. Unlike
generic prediction, this cost estimator explicitly uses source wrapping-i32
residuals. Context spread, signed packing and HybridUint classification share
the pass. The source classification trick uses exactly representable small
integer-to-f32 conversion to inspect exponent/mantissa bits; it does not
normalize or quantize image samples. Entropy scoring remains floating point.

An opt-in release microbenchmark compares the old i64-SIMD row plus scalar
second scan with this fused path, using the same source histogram backend.
On the Windows x86_64 development host, 1025x513 samples, two warmup pairs and
seven alternating repetitions gave these medians:

| Fixture | Old ms | Fused ms | Speedup | Fused MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Smooth | 4.290 | 0.662 | 6.479x | 794.178 |
| Repeated | 3.907 | 0.853 | 4.581x | 616.442 |
| U16 noise | 4.270 | 0.693 | 6.163x | 758.985 |

This is a cost-kernel benchmark, NOT complete image encoding throughput.
Signed-extreme inputs and all per-context costs are checked before timing.
Run `cargo test --locked --release -p rs-jxl --lib old_runtime_simd_vs_fused_cost -- --ignored --nocapture --test-threads=1`.

The warmed two-image CLI comparison of the integrated checkpoint measured
1.004x--1.063x throughput, with unchanged sizes and exact official-decoder
sample recovery. It includes entropy-writer changes as well as this kernel;
small differences may be noise. See [PORT_VERIFICATION.md](PORT_VERIFICATION.md)
for modes, repetitions, seconds and bpp. Do not apply the microbenchmark's
4.6x--6.5x factor to whole encoding.

Current correctness evidence and remaining port work are tracked in
[PIXEL_LOSSLESS_PORT_STATUS.md](PIXEL_LOSSLESS_PORT_STATUS.md). Integer/SIMD
optimization must retain official-decoder compatibility and exact sample
recovery; no compressed-output hash target is involved.
