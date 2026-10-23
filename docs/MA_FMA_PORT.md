# MA AVX2+FMA Port

## Scope And Status

This bounded default-path port changes only `tools/ma_simd.rs` and this document.
The pure-Rust `DefaultScorer` now selects AVX2+FMA when runtime detection confirms
both features, followed by AVX2 without FMA, SSE2, and scalar fallbacks. Its
constructor, estimate/move signatures, and `lanes()` API are unchanged: both
AVX2 profiles report eight lanes, SSE2 four, scalar one. Existing trainer calls
therefore activate the new default without changes to MA training or scheduling.

No C++ bindings, native codec dependency, process-environment selector, f64
precision upgrade, new logarithm approximation, or public tuning option was
introduced. The optional `LogTable` path remains non-FMA, including its log-count
cache and FastLog2f fallback. `CountEntropyTable` and its f64 kernels are unchanged.
FMA changes heuristic rounding and may change trees or compressed bytes; neither
whole-JXL identity nor identical decisions across CPU targets is a goal.

## Frozen Source

Read-only libjxl baseline: `b866b3defec13494bfbaf705db93581cda118e37`, supplied at
`../jxl_python/libjxl`. The translated code retains the JPEG XL Authors'
BSD-3-Clause attribution; the notice is in `docs/PORTING_KERNELS.md`.

- `lib/jxl/base/fast_math-inl.h::FastLog2f`: signed-bit exponent extraction,
  mantissa reduction, coefficients, division result plus exponent.
- `lib/jxl/base/rational_polynomial-inl.h::EvalRationalPolynomial`:
  two Horner `MulAdd` operations per degree-two polynomial; `FastDivision<float>`
  takes the active `JXL_TRUE` real-division branch, not reciprocal refinement.
- `third_party/highway/hwy/ops/x86_256-inl.h::MulAdd`: `_mm256_fmadd_ps` unless
  `HWY_DISABLE_BMI2_FMA` is defined. `set_macros-inl.h` marks this AVX2 profile
  as native FMA. The Rust gate needs AVX2 and FMA, not unrelated BMI operations.
- `third_party/highway/hwy/ops/generic_ops-inl.h::{ReduceAcrossBlocks,
  ReduceWithinBlocks, SumOfLanes}`: AVX2 adds opposite 128-bit halves, then
  reverses four lanes and pairs two lanes to reduce.
- `lib/jxl/modular/encoding/enc_ma.cc::{EstimateBits, SmallHistogramEntropy,
  EntropyContribution, MoveCountsAndEstimateBitsImpl}`: integer equality mask,
  probability floor, per-side contribution tables, weighted movement and lane
  accumulators. Movement fused into scoring is distinct from arithmetic FMA.

## Exact Arithmetic Profile

For positive normal probability `p`, the exponent is
`(bits(p) - 0x3f2aaaab) >> 23`; removing that exponent produces the mantissa.
Set `x = mantissa - 1`. The frozen coefficients, rounded once to f32, are:

| Coefficient | f32 Bits |
| --- | --- |
| P0 | `b5f85ab0` |
| P1 | `3fb6e02b` |
| P2 | `3f3e11c7` |
| Q0 | `3f7d8625` |
| Q1 | `3f813ced` |
| Q2 | `3e32458c` |

The FMA backend evaluates `fma(fma(P2, x, P1), x, P0)` and the analogous Q
polynomial, then performs a separate f32 division and exponent addition. The
native implementation uses `std::arch::x86_64::_mm256_fmadd_ps`. Scalar tails and
table construction use `f32::mul_add` for the same four operations. Fallbacks
retain separate multiplies/adds; no fast-math reassociation is requested.

Scoring still computes `count_f32 * inverse_total` separately, floors it at
`1/4096`, and masks the logarithm to zero on **integer** `count == total`.
Converted-float equality would be incorrect near 2^24. The contribution product
and accumulator subtraction remain separate operations, as written in the
source `Sub(bits, Mul(count, log))`. They are not replaced by `NegMulAdd`.

Each lane accumulates contributions in input order. Tail bins enter lanes
0..tail_length before the reduction, matching source zero padding without
reading beyond slices. For eight lane sums `s`, let `a[i] = s[i] + s[i+4]` for
i=0..3; return `(a[0] + a[3]) + (a[1] + a[2])`. This is not a serial sum.

The <=256 triangular contribution table is keyed by numerical profile. Each
used profile initializes its own 132612-byte `OnceLock` cache; normal dispatch
uses only one. Reusing the non-FMA table on an FMA backend would break exact
estimate/move equivalence. Above/below lookup decisions remain independent at
totals 256/257. Larger populations use that backend's polynomial directly.

The caller-owned validated total limit remains i32::MAX. Exact weighted count
movement, immutable default-path deltas, length checks before mutation, bounded
AVX2 gathers, zero/single-symbol handling, and per-side gates are unchanged.
No trainer penalty, acceptance threshold, tie rule, or compactor is edited.

## Focused Verification

New tests in `ma_simd.rs` provide:

- An independent scalar `f32::mul_add` oracle with fixed coefficient bit
  patterns, checking range-reduction boundaries and a probability grid. On
  capable hosts, direct native AVX2+FMA lanes must match it bit-for-bit;
  AVX2 non-FMA lanes must still match their non-FMA scalar expression.
- Runtime dispatch and separate/shared contribution-cache checks per profile.
- Deterministic histograms that distinguish separate contribution accumulation
  from an incorrect fused accumulator, and the Highway reduction from a serial
  sum. Every available backend is checked against its own lane oracle.

Existing default tests now include the FMA backend when available: exhaustive
logical small-table entries, weighted moves with immutable deltas, unaligned
slices and all tails, independent cache/fallback combinations, total 256/257,
floor boundaries at 4096, integer masks near 2^24 and i32::MAX, empty/single-bin
histograms, guard sentinels, and pre-mutation length failures. Move costs must
equal fresh estimates bit-for-bit within each profile. Cross-profile checks
retain their bounded numerical tolerance, not bit equality. Existing optional
LogTable and f64 cache tests remain in place.

All new compile, Clippy, unit, integration, and official-decoder results are
**pending parent central execution**. This worker runs no Cargo builds/tests.
Targeted standalone `rustfmt --check --edition 2024 --config skip_children=true`
passed for `ma_simd.rs`; this checks syntax/format, not type correctness.
The parent's earlier approximately 943 core tests and 56 release official checks
predate this FMA change and are not evidence that this checkpoint passes.

Suggested bounded central filter: `cargo test -p rs-jxl tools::ma_simd`;
parent-owned trainer and full pixel checks remain necessary. Official decoder
acceptance compares original samples, including hidden RGB, not JXL hashes.

## Limits

This models the frozen explicit Highway AVX2+FMA operations under the normal
floating-point environment. It does not claim equivalence to arbitrary C++
compiler fast-math/contraction options, AVX-512 widths, modified floating-point
rounding modes, or non-x86 SIMD targets. Unsupported CPUs retain the existing
fallbacks. Sparse-block scheduling, configurable small-table limits, source
training policy, and other MA gaps are outside this file ownership and are not
declared complete here. No source mutation, commit, or performance claim.
