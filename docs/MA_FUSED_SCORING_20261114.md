# Fused MA Description Scoring

## Scope

Optimize native pixel-lossless encoding after `8313d64`, whose source-e9
defaults combine Symmetric learning, Half description cost and single-pass
Distribution Offset. This change does not alter candidate budgets, effort
defaults, JPEG transcoding or decoder syntax. It does not add another encode
or a compressed-size comparison. Exact decoded samples, including invisible
RGB, remain the correctness requirement; equal compressed bytes are not a goal.

## Measured Hotspots

Eight existing manifest images cover a game screenshot, CLIC/DIV2K photos,
manga, a 16-bit synthetic image, Urban100, a texture and an icon. Diagnostics
use a frozen pre-change Release binary and are separate from timed trials.
Single-thread inclusive sums:

| Stage | Seconds |
| --- | ---: |
| Complete frame encoding | 23.3023 |
| MA learning | 15.0239 |
| MA split sweep | 11.4446 |
| MA gather | 2.6535 |
| LZ77 local parsing | 2.1584 |
| Tokenization | 1.3347 |
| Entropy model construction | 0.9929 |
| Single-pass Offset fitting | 0.3791 |

Detailed diagnostic scopes identify histogram scoring (4.9144 s), collection
(2.2247 s), and paired histogram filling (1.9414 s) inside property evaluation.
Middle-region scoring is only 0.0503 s on this subset. Detailed instrumentation
adds overhead: those measurements must not be combined with the ordinary
timing pass. Nested scopes overlap; parallel worker sums are not wall-time
fractions.

## Changes

1. **Fuse active-bin counting with entropy scoring.** Description-only scoring
   previously bypassed the existing whole-property SIMD sweep, dispatched each
   threshold separately and scanned both histograms again to count active bins.
   SSE2/AVX2 now accumulate integer nonzero masks during the existing histogram
   update/scoring pass. Scalar fallback is fused too. Tail bounds, zero padding,
   sparse layout, extra-bit ordering and scoring-profile arithmetic are retained.
   The existing no-description specialization compiles out the additional work.
2. **Give scalar description logarithms an FMA code-generation context.** The
   portable Windows baseline calls CRT `fmaf` four times per logarithm, up to
   eight calls per threshold. The CRT can use hardware FMA, but those calls have
   avoidable dispatch/prologue overhead. A small runtime-guarded native helper
   uses the same polynomial, fused operations and actual division. Non-FMA
   profiles retain their existing arithmetic.
3. **Remove small redundant Offset costs.** Narrowed residual samples need no
   second sort when all values fit `i32`; wide wrapping still sorts. Selecting
   the four most frequent modes no longer copies/sorts every group. Literal
   validation and residual collection share a pass; invalid late tokens discard
   only local scratch. Stream association, wide arithmetic and reuse guards
   remain intact. This stage was only 1.6% of measured single-thread time, so it
   is not expected to explain most of any aggregate speedup.

The optional AVX512 profile retains its existing scoring/counting route. It is
not silently substituted for AVX2 or enabled by this change.

## Existing ISA Comparison

Same pre-change binary, eight images, three repeats per image, rotating order.
Times sum per-image medians; all 96 outputs passed independent official djxl
sample comparison. Both profiles produced 7,970,719 bytes, mean 4.971681 bpp.

| Existing profile | Threads | Seconds | MPix/s |
| --- | ---: | ---: | ---: |
| AVX2/FMA default | 1 | 23.3812 | 0.3586 |
| Opt-in AVX512 | 1 | 23.1243 | 0.3626 |
| AVX2/FMA default | 16 | 3.8789 | 2.1618 |
| Opt-in AVX512 | 16 | 3.9324 | 2.1324 |

The small mixed result does not justify changing default ISA selection. This
is the existing kernel comparison, not a claim that AVX512 cannot be improved.

## Paired E9 Results

Windows x64, Ryzen 9 7945HX, portable workspace Release builds. The same eight
images were encoded three times per version and thread count, alternating
version order. No compilation or second benchmark ran alongside these trials.

| Threads | Before seconds | After seconds | Time reduction | Before MPix/s | After MPix/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23.6343 | 20.9160 | 11.50% | 0.3548 | 0.4009 |
| 4 | 7.1906 | 6.5977 | 8.24% | 1.1662 | 1.2709 |
| 16 | 3.8863 | 3.4483 | 11.27% | 2.1577 | 2.4317 |

All versions/thread counts produced 7,970,719 bytes, mean 4.971681 bpp and
weighted 7.604396 bpp. All 144 outputs passed official djxl sample verification.
These are paired measurements, not comparisons against historical timing on
another build or workload. The 16-bit grid is slightly slower at 16 threads
(0.07384 to 0.07469 s median); other medians improve. The aggregate gain is not
an every-image guarantee. Results are from `final-paired`; the earlier build's
separate `paired` results are retained, not mixed into these medians.

A separate 32-image pass contains 24 stratified originals across six datasets,
four 16-bit precision cases and four previously adverse cases. At 16 threads,
native time changes from **11.4633 to 10.1566 s** (11.40% less), throughput from
**2.0288 to 2.2898 MPix/s**. Total size remains **25,972,953 bytes**, mean bpp
**5.757505**, weighted bpp **8.934526**. All 64 outputs passed independent sample
verification. One DIV2K image is slower by 0.0073 s and four small synthetic
images by 0.0008..0.0139 s in this single-repeat pass. These broad timings are a
regression check, not a substitute for the repeated thread comparison.
Final artifacts are in `final-broad32`; `broad32` is the earlier build's pass.

## Lower-Effort Control

The first implementation shared a callback that conditionally added description
cost even in the disabled path. An initial repeated e8 control measured +1.89%
single-thread time and -2.35% at 16 threads. The final implementation gives
disabled scoring its original simple callback, without description state.

Final e8 results on the same eight images, three repeats:

| Threads | Before seconds | After seconds | Time change |
| ---: | ---: | ---: | ---: |
| 1 | 13.1150 | 13.2868 | +1.31% |
| 16 | 2.2481 | 2.3404 | +4.11% |

All 96 outputs pass official verification; size remains 8,079,858 bytes and mean
bpp 5.030636. This is **not an e8 speedup or proof of performance neutrality**.
The mixed multithread direction between runs and visible individual outliers
show measurement sensitivity, but do not justify dismissing the measured
regressions. The remaining lower-effort cost needs a more controlled follow-up
before claiming an all-effort improvement. This report's supported speedup is
the current source-e9 description-scoring route. The workstation was not
reserved exclusively or CPU-affinity pinned. Raw trials are retained in
`e8-control`, `e8-repeat` and `final-e8`.

## Diagnostic Follow-Up

Same eight images, single thread, detailed instrumentation on both versions:

| Inclusive stage | Before seconds | After seconds |
| --- | ---: | ---: |
| Frame encoding | 24.8639 | 21.9447 |
| MA split sweep | 12.9844 | 10.1124 |
| Histogram scoring | 4.9144 | 2.3067 |
| Histogram collection | 2.2247 | 2.2388 |
| Paired histogram fill | 1.9414 | 1.9469 |
| MA gather | 2.6709 | 2.7013 |
| Tokenization | 1.3262 | 1.3220 |
| LZ77 local parsing | 2.1751 | 2.1716 |
| Single-pass Offset fitting | 0.3858 | 0.3912 |

Scoring time decreases 53.06%, while gathering, collection, tokenization and
LZ77 are approximately unchanged. This supports the scoring-path explanation;
it does not isolate fusion from scalar FMA dispatch. Offset cleanup has no
demonstrated stage-time gain on this subset, despite removing allocations and
redundant sorting/validation work. No larger Offset claim is made.

The next measured opportunities are histogram collection and sample gathering,
not blindly widening all vectors. Indexed histogram additions can collide;
naive gather/add/scatter would lose counts. Existing paired/sharded collection
should be evaluated before adding another scatter kernel. Contiguous scoring
is already vectorized. AVX512 requires its own fused implementation/measurement,
not merely flipping the existing flag. Detailed scopes overlap and add overhead;
these numbers are diagnostic, not the timing results above.

## Validation And Limits

- 45 SIMD tests pass, including dense/sparse/tail boundaries, independent active
  counts, scalar/SSE2/AVX2/FMA/reciprocal/AVX512 profiles, and description-log
  rounding-boundary oracles. These are primitive equivalence checks, not a
  compressed-byte equality acceptance condition.
- 47 Offset tests pass; one existing opt-in microbenchmark remains ignored.
  Added cases exercise wide narrowing, frequency ties and late invalid-token
  fallback without input mutation.
- Two scoring-dispatch tests and three Offset/Symmetric integration tests pass.
- The final Release native library passes all three
  `python/tests/test_source_ma_options.py` tests, including 8/16-bit invisible
  RGB, still images, animations and zero-duration layers.
- Two benchmark-harness tests, formatting checks and `git diff --check` pass.
- Final e9/e8 timed comparisons verify 304 outputs; final diagnostic follow-up
  verifies another eight. Of these 312 outputs, 160 use the final encoder and
  152 the frozen baseline. There are 32 distinct manifest images, not 312 images.

No full corpus, new official-encoder baseline, Linux runtime, sanitizer run or
full-workspace regression suite was performed. The decoder and public options
are unchanged. Speed conclusions are limited to this CPU and these samples;
lower-effort regressions and small per-image slowdowns are not hidden.

## Reproduction

Use matching workspace features for the old and new Release builds:

```text
cargo build --release --workspace --lib --bins --example portfolio_lab
python tools/benchmark_ma_kernels.py --manifest MANIFEST.json --variant before=OLD.exe --variant after=target/release/examples/portfolio_lab.exe --decoder DJXL.exe --out OUT/paired --threads 1 4 16 --repeats 3
```

`--profile stages` or `--profile detail` starts a separate diagnostic run;
diagnostic results are excluded from timing summaries. `--effort 8` checks the
shared no-description kernel. `--avx512 NAME` opts only that named variant into
the existing AVX512 experiment.

The runner requires an explicit manifest, freezes executables, records input
provenance and native receipts, rotates variant order, and checks every output
with official djxl. Temporary outputs are removed after sample verification.
It never discovers or starts the whole corpus. Mean bpp is the arithmetic mean
of per-image bpp; weighted bpp is total bits divided by total pixels. Repeats
use per-image medians before summation. Native timing excludes CLI startup,
input/output and independent decoding. Timing differences include both SIMD
and Offset changes; they do not isolate each patch's contribution.

Artifacts reside in the sibling `jxl_python/out/ma-kernels-20261114` directory.
`profile-before-v2`, `detail-before`, and `before-isa` hold completed baseline
diagnostics and ISA comparisons; `detail-after` contains the final follow-up.
The earlier `profile-before` attempt stopped
on a diagnostic-parser error and is not an acceptance/timing result.
