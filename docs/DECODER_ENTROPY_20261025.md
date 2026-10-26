# Decoder Entropy Experiments, 2026-10-25

## Outcome

**No runtime optimization from this round is retained.** Four candidates slowed
down the measured aggregate. The remaining candidate improved three JPEG pixel
cases by only 1.38%, without establishing a general benefit on PNG controls.
ANS, bit buffering, coefficient accumulation and quantization-bias arithmetic
are restored to their implementations at the start of this round. Previous
optimizations, including the optional AVX512 JPEG AC-mask kernel, remain intact.

Retained changes are the actual-source ANS regression tests, their test-only
dependencies, and this experiment record. No encoder, bitstream, public API,
default SIMD feature, unsafe implementation, CI, commit or push changes.

## Method

The earlier diagnostic stages identified coefficient entropy decoding as the
largest measured VarDCT group cost. Those inclusive scopes do not establish that
the ANS primitive alone is responsible for all coefficient-loop time. This
round tested five separate, narrowly scoped hypotheses against real images.

All builds and measurements were serialized. Each candidate used the same
default AVX2 feature set, three rotating measured rounds, one warmup, ten
in-process decodes per run, and one decoder thread. Reported speed is process
wall time divided by repetitions; startup, input IO and allocations are
amortized, not subtracted. This is a bounded development-host experiment, not
a statistical confidence interval or a full-corpus result.

Pixel timing uses F32 buffers: official `djxl --disable_output` forces F32.
U8 output, F32 output and original JPEG bytes were independently checked.
For all five candidates, all tested Rust U8 and F32 samples were identical to
the preceding Rust checkpoint, and all original JPEGs reconstructed exactly.
The existing difference from official rendering remained at most 1 U8 level
with the default `jxl-rs` dither. This is not exact pixel identity with djxl.

The common inputs were:

| Original JPEG | MPix | Input JXL bpp |
| --- | ---: | ---: |
| wallhaven-yqxzqx.jpg | 14.7456 | 4.3686 |
| wallhaven-9oo8k1.jpg | 9.4372 | 0.9593 |
| wallhaven-vpyekp.jpg | 2.4576 | 0.3020 |

bpp is unchanged: no candidate re-encoded the inputs. Compressed JXL byte
equality is not an optimization target. File hashes only protect benchmark
input/binary identity; correctness checks decoded samples or original JPEGs.

## Results

Each row has its own interleaved baseline run. Aggregate MPix/s is total pixels
divided by the sum of per-image median times, not a mean of per-image speeds.

| Candidate | Before MPix/s | Candidate MPix/s | Throughput change | Decision |
| --- | ---: | ---: | ---: | --- |
| Conditional ANS renormalization | 45.438 | 44.108 | -2.93% | Rejected |
| Variable 0/16-bit ANS shift/read | 47.284 | 45.378 | -4.03% | Rejected |
| Unconditional zero-coefficient accumulation | 47.319 | 46.828 | -1.04% | Rejected |
| Early ANS bit-buffer refill | 47.069 | 47.720 | +1.38% | Not retained |
| Skip SIMD division for all-small coefficients | 47.124 | 46.316 | -1.71% | Rejected |

- Conditional renormalization avoided unused 16-bit lookahead but introduced a
  state-dependent branch. The large JPEG slowed from 388.445 to 403.984 ms.
- Variable shifting consumed either zero or sixteen bits without that explicit
  branch. It still lost, especially on the smallest case, 34.810 to 38.440 ms.
- Zero accumulation removed the token-zero continue, retained token-based
  nonzero accounting and wrapping integer accumulation, and did additional work
  for zeros. One case improved, but the aggregate regressed.
- Early refill followed the shared-lookahead organization in libjxl's
  `lib/jxl/dec_ans.h` and `dec_bit_reader.h`. It retained checked accesses and
  optimistic EOF accounting. The measured improvement was small; it was not
  sufficient to justify changing the shared reader path in this round.
- The quantization-bias candidate skipped division only when every SIMD lane
  selected the existing small-coefficient formula. It used no approximate
  reciprocal and changed no floating-point result in the tested images. The
  added mask/branch did not improve whole-image performance.

### Early Refill Controls

These are pixel-lossless JXL inputs; all original PNG integer samples were
verified, including hidden RGB. Times below are F32 decoding, not PNG writing.

| Original PNG | Before ms | Candidate ms | Throughput change |
| --- | ---: | ---: | ---: |
| kodim19.png | 69.031 | 69.192 | -0.23% |
| news.ycombinator.com.png | 115.860 | 116.468 | -0.52% |
| clover.png | 186.728 | 188.821 | -1.11% |
| 9bde6aaf390109f97622d35ff1e900ee.png | 255.275 | 254.418 | +0.34% |

These small differences do not demonstrate a broad speedup or a broad
regression. Do not promote early refill based only on the +1.38% JPEG result.

## Regression Coverage

`crates/rs-jxl/tests/decoder_hybrid_uint.rs` now also compiles the maintained
ANS source. Its differential test retains the original update formula and
compares symbols, ANS states, logical bit positions, available bits, EOF errors
and continued reading. It covers alphabet exponents 5..8, every low 12-bit
state index at selected high-state boundaries, single-symbol, two-symbol and
mixed histograms, byte/bit alignments, short inputs and repeated EOF reads.
The upstream histogram fuzz test runs through the test-only `arbtest` dependency.
Neither this dependency nor its `arbitrary` dependency enters runtime decoding.

Final focused test targets passed: `decode` (31), `decode_dither` (4),
`decode_hostile` (4), `decode_jpeg` (8), `decode_streaming` (17),
`decode_wide_integer` (7), `decoder_hybrid_uint` (15): **86 passed, 2 ignored**.
The ignored tests are a hostile-input subprocess entry point and an explicit
external-JPEG corpus test. Real JPEG checks above ran separately. This is not
full upstream conformance; the parent harness avoids unavailable upstream test
fixtures while compiling the actual ANS and bit-reader source.

Workspace/all-target Clippy with `decoder-avx512,decoder-profile` and warnings
denied also passed. `git diff --check` reported no whitespace errors.
The default Release binary was rebuilt after restoring runtime code. The final
`restored-smoke` check again verified original JPEG bytes and exact previous
Rust U8/F32 samples on `wallhaven-vpyekp.jpg`; its one-shot timings are not a
performance result. `target/release/rs-jxl.exe` is not an experimental candidate.

## Artifacts

Under `out/entropy-20261025/`, each of `conditional-pixels`, `variable-pixels`,
`zero-pixels`, `prefill-pixels`, `sparse-bias-pixels`, and `prefill-controls`
contains `report.json` and `REPORT.md`, including commands, raw trials, binary
identities and correctness results. Frozen `rs-jxl-*.exe` candidates in that
directory are experiments, not the retained release implementation.

Use `tools/benchmark_jpeg_pixel_decode.py` for JPEG-transcoded JXL to pixels and
`tools/compare_decoder_checkpoints.py` for lossless PNG controls or original
JPEG byte reconstruction. These are different timed workloads.

## Next Investigation

The coarse entropy scope includes context calculation, histogram/configuration
access, symbol decoding, extra-bit decoding and coefficient stores. The next
useful investigation is to separate these costs, ideally with captured real
symbol/context traces, before another rewrite of the state update. SIMD width
or fewer source-level operations alone has not predicted whole-image gains.
No trace-driven speedup is claimed or implemented by this document.
