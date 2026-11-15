# MA Diagnostic Gating and Serial Reduction

## Scope

Two default-path implementation changes on top of `9cba690`, the conditional
entropy-bound optimization. No effort defaults, scoring formulas, thresholds,
candidate budgets, clustering policy or decoder syntax change.

1. Snapshot profiling once per learner. Property evaluation/win/predictor-win
   and gap counters, property minimum-cost collection, and diagnostic formatting
   are enabled only by the existing `RS_JXL_PROFILE` switch. Resource accounting,
   scheduling counters and experiment receipts remain unchanged.
2. Reduce serial property results immediately, including `property_dedup=true`.
   Previously this ordinary path collected all six-way candidate arrays before
   reduction. Keep the ordered duplicate-property proofs and strict tie rules.
   Parallel properties still use indexed collection and ordered reduction.

An error drains the remaining serial evaluations and returns the first
property-ordered error. Neither the partial cache nor the updated proof mask is
published. The new tests cover duplicate/constant/inherited properties,
lookahead shortlists, profiling on/off, cache contents, retry and stable ties.

Profiling retains its property notes. Serial no-dedup evaluations now also
contribute to the evaluation/gap diagnostics. In detailed profiles, immediate
serial reduction is inside `ma.sweep_dispatch`; the separate
`ma.sweep_reduce` scope covers parallel-result reduction. Neither diagnostic
change is an encoding-policy change.

## Method

Windows x64, Rust 1.97.1, workspace Release build, existing AVX2/FMA path
(`RS_JXL_MA_AVX512=0` for both versions). Before is the frozen executable in
`out/cluster-bound200-20261115/bin/portfolio_lab.exe`; after is the local change.
Both execute one ordinary native encode, without outer candidate search.

Times sum per-image medians. Native time includes pool setup, but excludes
image I/O, process startup and independent decoding. Versions alternate order;
no compilation or second benchmark was launched alongside timed trials.
Profiling is off for timings and checked in a separate invocation.

Three full-resolution images use 1/4/8 threads and five repetitions per
version/effort: DIV2K `0446.png` (2040x1356), RGBA document screenshot
`9bde6aaf390109f97622d35ff1e900ee.png` (1264x1152), and 16-bit
`rough_linen_Diffuse_1k.png` (1024x1026). Five repetitions combine the initial
two and a subsequent three-repeat pass, without discarding observations.

This is a targeted five-image experiment, not a new 200-image corpus result.
Small timing changes below are point estimates, not established global gains.

## Thread Results

| Effort | Threads | Before seconds | After seconds | Time change | Before MPix/s | After MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e8 | 1 | 9.909822 | 9.934519 | +0.249% | 0.5321 | 0.5308 |
| e8 | 4 | 2.880561 | 2.878456 | -0.073% | 1.8305 | 1.8319 |
| e8 | 8 | 1.876402 | 1.776352 | -5.332% | 2.8102 | 2.9684 |
| e9 | 1 | 16.954581 | 16.779137 | -1.035% | 0.3110 | 0.3143 |
| e9 | 4 | 4.839914 | 4.807721 | -0.665% | 1.0895 | 1.0968 |
| e9 | 8 | 3.246568 | 3.299203 | +1.621% | 1.6242 | 1.5983 |

All compared output sizes are unchanged. The three-image totals are
9,155,124 bytes at e8 and 8,866,664 bytes at e9. Arithmetic mean per-image bpp
is respectively 16.742353 and 16.164490; the subset includes a 16-bit image.

The document screenshot's e9/8-thread median initially rose by 4.08%. A
separate seven-pair follow-up measured 1.405677 -> 1.395584 seconds (-0.718%).
The slowdown did not reproduce consistently; this does not establish a
screenshot speedup either. The original five-repeat aggregate remains in the
table rather than being replaced by the more favorable follow-up.

## MA Heavy Images

Two additional original images, eight threads, two repetitions per version.
These results remain separate from the five-repeat thread matrix.

| Image | Effort | Before seconds | After seconds | Time change | After MPix/s | Unchanged bpp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-mlzoy1, 3840x2160 | e8 | 2.23246 | 2.26749 | +1.57% | 3.658 | 13.65781 |
| wallhaven-mlzoy1, 3840x2160 | e9 | 3.22508 | 3.28598 | +1.89% | 2.524 | 13.36216 |
| 1515.hdr, 3024x4032 | e8 | 9.81383 | 9.34299 | -4.80% | 1.305 | 20.27812 |
| 1515.hdr, 3024x4032 | e9 | 15.20466 | 14.69618 | -3.34% | 0.830 | 17.14086 |

Keep the changes as limited implementation cleanup with measured benefits on
some MA-heavy workloads. Do not claim universal acceleration, or attribute
the combined result to either change individually. Ordinary e9 images and
single-thread encoding did not show a large general improvement.

## Verification and Artifacts

- `cargo test -p rs-jxl --lib modular::ma_train:: -- --quiet`: 281 passed.
- `cargo build -p rs-jxl-cli --example portfolio_lab --release`: passed.
- 211 official djxl original-sample comparisons passed: 180 thread-matrix
  outputs, 16 heavy-image outputs, 14 screenshot follow-up outputs, and one
  separate profiling check. This includes RGBA hidden RGB and 16-bit samples.
- All measured per-image output sizes are unchanged. Equal JXL bytes are not
  a correctness criterion and were not required.
- The separate profiling check emitted nonzero property diagnostic notes.

Local artifacts are under `out/ma-reduction-20261115/`: `comparison.json`,
`threads-e8`, `threads-e9`, `repeat-e8`, `repeat-e9`, `heavy-e8`, `heavy-e9`,
`screenshot-repeat7`, `profile`, input manifests, frozen binaries and
`source.patch`. The existing `tools/benchmark_ma_kernels.py` produced the
receipts and pixel checks; `summarize.py` in the artifact directory aggregates
the two thread-matrix passes.
