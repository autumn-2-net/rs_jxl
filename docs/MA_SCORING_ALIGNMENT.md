# MA Default Scoring Alignment Audit

Status: **default f32 scoring integrated and centrally tested**. The original
audit below explains the corrected implementation difference; this does not
certify complete source-target parity or pixel-port acceptance.
Frozen source: libjxl `b866b3defec13494bfbaf705db93581cda118e37`.

The corrected frame-area route exposes MA learning as the remaining primary
hotspot. See [measurements](PORT_VERIFICATION.md). On the 4K natural-image
profile, split sweeps accumulated 58.54 worker seconds and sample preparation
25.39; the whole CLI took 20.98 seconds. Nested/parallel times overlap.

## Original Difference

- C++ `modular/encoding/enc_ma.cc::EstimateBits` uses the default f32
  FastLog2f/Highway path. `JXL_MA_CLIPPED_ENTROPY` and
  `JXL_MA_COUNT_LOG_TABLE` select optional alternatives.
- Rust `modular/ma_train.rs::Entropy` uses incremental `ClippedEntropy` for
  node weights up to `1 << 20`, and direct f64 log2 rescans above that.
  Candidate costs and comparisons are also f64. Thus not every node is
  incremental, but neither branch reproduces the default source arithmetic.
- C++ `MoveCountsAndEstimateBits<false>` fuses histogram movement and both
  costs. Rust `evaluate_property_cached` moves each symbol, then asks both
  entropy objects for costs.
- C++ `SmallHistogramEntropy` caches contributions when total <=256, with
  each side choosing independently. That default optimization is not active
  in Rust training.

## Existing Reusable Rust Tools

`tools/ma_simd.rs::LogTable::new(0)` already provides FastLog2f, SSE2/AVX2 and
a fused movement kernel. `new(65536)` is a different log-count strategy and
must not be substituted for source default policy.

The existing public movement function validates/recomputes totals and clears
the delta buffer on every call. Training needs a bounded internal entry point
for immutable cached deltas and already validated weighted totals. It must
not mutate the reusable histogram cache or silently trust invalid public input.

## Implemented Integration

1. Port the shared small-histogram contribution table (approximately 130 KiB).
2. Integrate the source-default parent cost and fused split sweep together.
3. Preserve f32 rounding through penalties, four candidate classes and final
   threshold comparisons; changing only the entropy function is insufficient.
4. Compute weighted bin populations once per property and reuse across
   predictors. Current `used` counts distinct records, not sample multiplicity.
5. Preserve sample fraction, all properties/predictors, stable candidate order,
   histogram reuse, parallel commit order and complete-search resource guards.
   Do not make this a weaker effort preset.

## Numerical and Safety Checks

- Keep the 1/4096 probability floor and integer count==total zero-entropy gate.
- Preserve strict `<` versus fast-decode `<=` and source tie handling.
- Preserve `(rbits + total_extra as f32) - extra_below as f32`; integer
  subtraction before conversion changes rounding.
- Audit FMA, vector reduction and padded-lane behavior against the selected
  C++ target. Current Rust kernels use nonfused operations, sequential lane
  reductions and scalar tails; API reuse alone is not parity proof.
- The current SIMD API bounds total weight at i32::MAX; the trainer accepts
  u32. Reject or explicitly handle larger weights, never silently narrow.
- Test total=256 boundaries, the 4096 floor, counts near 2^24, weighted moves,
  immutable cache reuse, near-threshold candidates and serial/parallel results.
- Validate full output with official decoders and compare bytes/time/bpp on
  both a natural image and a screenshot. Compressed-byte equality is not an
  acceptance target; decoded samples are.

## Verification and Remaining Differences

`ma_train` now uses `DefaultScorer` for both parent cost and fused split scoring.
The decisions, predictor penalties, extra-bit operations and thresholds remain
f32 end to end. Public cost fields retain f64 types for API compatibility and
are validated for finite f32 conversion. Weighted totals are bounded at
i32::MAX before histogram construction. Cached bin weights are updated with
their distinct-record counts, which remain separate for partitioning.

`ma_simd` dispatches native SSE2/AVX2, with an independently tested scalar
fallback and a shared 132612-byte triangular contribution cache. The new path
uses frozen Highway's reduction order and logically zero-padded tails. It is
the non-FMA source expression: FMA-enabled C++ targets can still select a
different valid tree. The older optional entropy helpers remain available,
but are no longer silently selected as the default trainer.

Central debug and release workspace tests pass: 578 core, 10 API, 34 CLI,
helper tests and doctest.
Strict all-target Clippy passes. This includes weighted cache reuse, immutable
deltas, histogram limits, small-cache boundaries, floor/rounding edges, and
serial/parallel decision equivalence. Two old equality fixtures were corrected:
FastLog2f at probability 1/2 is approximate, so equality must be tested at the
actual estimator value, not a mathematical log2 value of exactly 100 bits.

Same-session single e9 candidate, eight threads, 33,554,432 sample ceiling,
metadata stripped; startup/input decoding/stdout capture included:

| Image | Before bytes | After bytes | Before seconds | After seconds | After bpp | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `9bde6aaf390109f97622d35ff1e900ee.png` | 412813 | 412541 | 3.2758 | 2.7954 | 2.266510 | 0.521 |
| `wallhaven-mlzoy1.png` | 14179361 | 14179167 | 16.8496 | 12.3221 | 13.675894 | 0.673 |

Both official 0.12 and 0.13 decoders reproduced every sample, including hidden
RGB. These are single-run measurements, not a statistically established speedup
or a completed performance port. The source sampling fraction and candidate
search were not reduced. The saved pre-change binary includes the previous
local learner-ID correction, unlike some earlier documented measurements.

The full native screenshot scheduler also passed both decoders: before
34.2735 seconds, after 27.5230 seconds; both outputs were 146660 bytes,
0.805753 bpp. This same-session comparison shows a reduction in whole-search
time without a measured size penalty, but remains a single pair of runs.
