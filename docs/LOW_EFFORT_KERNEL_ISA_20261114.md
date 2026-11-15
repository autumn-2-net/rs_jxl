# Low-Effort Kernel and ISA Experiments

Baseline: `dc61f47`, following `LOW_EFFORT_PERFORMANCE_20261114.md`.
Scope: lossless pixel encoding, chiefly e5. This is a targeted six-original
cohort, not a replacement for the existing 200-image charts. Acceptance is
original decoded sample equality, including hidden RGB and 16-bit samples,
not identical encoded JXL bytes.

## Initial Experiments

The first follow-up peels channel-row boundaries, reuses the interior WP
kernel during sampling, caches validated small-error weight-table rows, and
rejects spatial LZ candidates with unequal first values before materializing
them. It also experimented with fusing mixed-tree Gradient prediction into
the property walk. The latter was subsequently removed in favor of the
existing batched Gradient SIMD kernel.

Six originals at e2-e5, one/eight threads, two repeats, interleaved with the
pushed baseline and official cjxl 0.12: 288 official pixel checks passed.
The first checkpoint reduced e3 time by 11.3%/9.8% and e4 by 6.1%/8.1%, but
e5 improved only 0.4%/0.3%, too little to establish a meaningful gain.

Default WP coefficient factorization was then tested separately. Across
108 verified e5 encodes (three binaries, two thread counts, three repeats),
the factorized version took 6.8473/1.6508 seconds versus the preceding
checkpoint's 6.8248/1.6735 seconds. There was no stable improvement, so this
special-case branch was removed.

An explicit SSE2 four-lane wide-error update also passed independent i128
reference tests, all integer-boundary tests, and 72 official pixel checks.
It was not retained: total e5 time was 6.7468/1.5373 seconds versus the pushed
baseline's 6.5896/1.5500 seconds. Size and mean bpp were unchanged.

Inspection of an optimized, isolated arithmetic wrapper explains why this
particular rewrite was unpromising: LLVM already vectorizes the original
four-i64 expression into two SSE2 groups. The explicit version needed extra
shuffle/pack instructions. This wrapper is not proof of every full-encoder
call site's assembly, but it disproves the assumption that scalar-looking
Rust necessarily executes this operation one lane at a time.

## Whole-Channel ISA Experiment

The next experiment compiles the complete generic channel walk and MA
sampling walk for three targets, keeping shared safe loop bodies:

- Portable baseline, which can still contain compiler-generated SSE2.
- AVX2, with dispatch once per channel or training stripe.
- AVX512F/BW/DQ/VL, opt-in for comparison rather than automatically preferred.

Pixel dependencies, full-width prediction arithmetic, warm-up, RNG order,
sample budgets, reference properties and error propagation are unchanged.
No per-pixel indirect call or runtime feature check is introduced. This does
not enable the separate narrow-i32 Weighted experiment or change MA/LZ policy.

`RS_JXL_MODULAR_ISA=portable|avx2|avx512` is a process-start diagnostic override,
cached on first use. Unsupported or unknown explicit selections safely use
the portable path. An omitted override or `auto` also keeps the portable
path until a reliable benefit warrants a different default. It affects only
these two loops, not every SIMD operation
in the codec. `RS_JXL_PROFILE=1` reports the actually selected loop ISA.

The test machine reports AVX2 and all four required AVX512 features, including
OS vector-state support. Wider ISA support alone is not a speedup claim.

### First Same-Binary Comparison

Six originals cover a small photograph, a document screenshot, a large
wallpaper, manga, transparent artwork, and a 16-bit RGB pattern. At e5,
two repeats per image/ISA/thread count produced 120 encodes, each checked
against the original pixels using official djxl. The elapsed totals below
sum each image's median native encode time, excluding input/output and
verification. Variants were interleaved, not run concurrently.

| Variant | One thread (s) | Eight threads (s) | Mean image bpp |
| --- | ---: | ---: | ---: |
| Pushed baseline | 6.80445 | 1.63113 | 6.875543 |
| Current portable | 6.85301 | 1.65576 | 6.875543 |
| Current AVX2 | 6.81604 | 1.64425 | 6.875543 |
| Current AVX512 | 6.77151 | 1.63770 | 6.875543 |
| Official cjxl 0.12 | 5.80476 | 1.30136 | 6.988006 |

The Rust variants all total 16,542,788 bytes; official cjxl totals 16,588,693.
AVX512 is only 1.19%/1.09% faster than the same build's portable selection
here, and is not consistently faster than the pushed baseline. Two repeats
on this small cohort do not establish a reliable ISA improvement. Do not
promote AVX512 or claim that it closes the e5 gap on these numbers.

Normal-build executable SHA256:
`dd4748d664cfb63da2b42d205c9c1cfe873ab808e88dbe816ff3a5b4f89bc61b`.
This hash identifies the tested executable, not a required encoding result.
Raw results: `out/low-effort-hotspots-20261114/isa6/`.

Release disassembly subsequently found that this first build specialized
the entry wrappers but not the hot row/pixel helpers. AVX2 and AVX512 encode
entries both called the same baseline row body, which called baseline
prediction/update helpers. The inspected update used legacy SSE2 `psubq`,
`psrad`, `paddq`, and `psrlq`. The same problem affected gathering. Thus this
first result does **not** measure a fully vector-specialized Weighted loop.
The follow-up explicitly inlines the row, pixel closure, prediction, and
update chain into the feature-specific entry points, without changing the
predictor arithmetic or selecting a narrower integer range.

### Inlined Encode Comparison

The second normal build ran the same six originals, with three repeats:
180/180 official pixel checks passed. All Rust variants retained the same
aggregate size and bpp as above.

| Variant | One thread (s) | Eight threads (s) |
| --- | ---: | ---: |
| Pushed baseline, freshly measured | 6.81830 | 1.55584 |
| Current portable | 6.62228 | 1.59957 |
| Current AVX2 | 6.78034 | 1.53785 |
| Current AVX512 | 6.62497 | 1.55294 |
| Official cjxl 0.12, freshly measured | 5.69446 | 1.31740 |

AVX512 versus the same-build portable path: +0.04% time on one thread,
-2.92% on eight. AVX2: +2.39%/-3.86%. Neither ISA has established a consistent
end-to-end winner. The inlined portable path improves one-thread e5 relative
to the pushed baseline, but its eight-thread total is worse in this run.
Do not call this a solved e5 performance gap.

The matched PDB/EXE (PDB age 27) confirms encode WP arithmetic inside the
AVX2/AVX512 bodies: AVX2 uses YMM `vpsubq`/`vpaddq`/`vpsrlq`; AVX512 uses
`vpabsq`/`vpmovqd`. Gathering still reaches shared baseline pixel bodies,
so this is not yet a whole-pipeline ISA comparison. The subsequent change
removes the row helper's by-value `FnMut` forwarding layer.

Binary SHA256:
`b4ccff6323f39770419a7d4097b4bd60bdca3e46b5f48ce4e12041851a371b64`.
Results: `out/low-effort-hotspots-20261114/isa-inlined6/`.

An e3/e4 guard run on this same build used the six originals and two repeats,
with 96/96 official pixel checks passing. For the portable selection:

| Effort | Previous t1/t8 (s) | Current t1/t8 (s) | Time change t1/t8 |
| --- | ---: | ---: | ---: |
| e3 | 1.51708 / 0.38687 | 1.40474 / 0.37690 | -7.40% / -2.58% |
| e4 | 3.34104 / 0.78159 | 3.24660 / 0.72195 | -2.83% / -7.63% |

Both effort totals retained their original byte counts. Individual images
and runs vary; these are small-cohort observations, not a corpus guarantee.
Results: `out/low-effort-hotspots-20261114/controls6/`.

### Final Callback Experiment

After replacing the training row helper's by-value callback with a borrowed
callback, the final same-binary run used six originals, one/eight threads,
two repeats, and all three ISA selections. All 72 official pixel checks
passed, with 16,542,788 bytes and mean image bpp 6.875543 for every selection.

| Selection | One thread (s) | Eight threads (s) | MPix/s t1/t8 |
| --- | ---: | ---: | ---: |
| Portable | 6.88781 | 1.59660 | 1.719 / 7.416 |
| AVX2 | 6.60450 | 1.54098 | 1.793 / 7.684 |
| AVX512 | 6.53942 | 1.58742 | 1.811 / 7.459 |

AVX512 is 5.06% faster than portable on one thread here, but only 0.57%
faster on eight. AVX2 is 4.12%/3.48% faster than portable. Variation across
the two inlined experiments is material: this is evidence for retaining
diagnostic choices, not an automatic AVX512 preference or a general speed
guarantee. A single pixel has four dependent WP sub-predictors; selecting
AVX512 does not automatically create eight independent pixels of work.

Normal default remains portable. Neither instruction choice changes the
MA tree policy, LZ search budget, sample fraction, or encoding format.
Results: `out/low-effort-hotspots-20261114/isa-final6/`.
Binary SHA256:
`5db3f4a7f62cef5944931f830c13c285be3b4e24b4a7aa5cf57549be3234b8bd`.

Final disassembly with the matching PDB (age 28) confirms that the gather
pixel-closure barrier is gone in the inspected variants. The AVX2 gather
function at RVA `1a7540..1af01b` contains YMM `vpsubq`, `vpaddq`, and
`vpsrlq` at `1a837f`, `1a8391`, and `1a8399`. The AVX512 gather function
at `1af240..1b6bd4` contains `vpabsq` and `vpmovqd` at `1b001c` and
`1b0031`. This verifies instruction propagation, not full vectorization:
the scalar weight-table helper remains at RVA `4c6f90`.

## Structural e5 Follow-up

Source inspection, distinct from measured improvements:

1. Ordinary nonresponsive e4 disables LZ77; e5 uses `Lz77b1w3t`, a fast
   single-entry matcher, not Optc/DP. e4 fixes RCT to 6; e5 compares four
   RCTs (`0,6,5,10`). Sampling fraction and property/bin counts also increase.
   Thus the e4/e5 timing jump is not simply a slower version of one kernel.
2. Independent groups run training and then tokenization on the same
   prepared planes and WP header (`frame.rs::independent`). Both can walk
   Weighted history. A group-scoped cache of rounded prediction and WP
   property could replace the second history calculation when domains match.
   Two i32 values cost eight bytes per channel sample: a 256x256 RGB group
   is 1.5 MiB, not a full-image cache. Channel/reference identity, transformed
   layout, header and group boundaries must agree. Stripe warm-up is not
   automatically the same history as full-group tokenization. This is a
   candidate optimization, not an implemented cache or claimed speedup.
   Start with whole-channel gather (`stripes=1`) inside one independent
   group invocation. Cache only the rounded prediction and signed error
   property 15; training itself consumes compact samples and does not
   replay WP. Training stream ID 0 can differ from the wire stream ID, so
   full property records are not reusable. Writing and reading the cache
   adds about 16 bytes/sample of memory traffic; benchmark that trade-off
   at both one and eight threads before adopting it.
3. LZ processing materializes raw literals, a value-only copy, commands,
   and final entropy tokens. Rejected searches currently construct a full
   literal command vector before `commands_to_tokens` builds another vector.
   A borrowed-literal result/internal output sink could remove this work
   without reducing match coverage. The public parser contract still needs
   to produce owned commands for its existing callers.
4. The existing `SourceScout` is intentionally conservative: `Literal`
   requires complete token coverage and no repeated prefix. Sparse negative
   evidence keeps `Full`, so just enabling it does not cheaply reject large
   unprofitable groups. At e5 its `Fast` recommendation usually still means
   the already-selected fast matcher. Any useful early rejection must be a
   separately measured size/time trade-off, not described as exact pruning.
5. Repeated model counters do not prove repeated image encodes. The image,
   MA tree, and context-map streams each need entropy coding; source coding
   chooses one model type rather than running the `Coding::Auto` two-model
   competition. Do not remove these calls based only on aggregate counts.

Priority: validate reusable WP work and remove intermediate materialization
first, then measure a genuine low-cost LZ acceptance policy. Do not reduce
e5 to e4 or silently drop its tools to manufacture a speedup.

## Verification and Collateral Fix

- Final Modular unit suite with detailed profiling compiled: 808 passed,
  three explicitly ignored microbenchmarks.
- Patch experiment tool: eight tests passed.
- Final Release Python DLL bindings: 33 passed, one skipped.
- Benchmark-tool unit tests: three passed.
- This work did not rerun or overwrite the 200-image plots or full corpus.
- No Linux or other-CPU runtime performance claim is made from this Windows
  test machine.

The broad unit run exposed an existing block-context experiment conflict:
it added a fixed block plan after e9 projection had installed Symmetric,
Description Half and Distribution Offset. The experiment now declares its
intent before projection, separately for residual/reference targets. Normal
e9 defaults and explicit-conflict validation are unchanged. This was an
encoding-configuration rejection, not an observed decoded-pixel mismatch.

## e5 Stage Diagnosis

The first detailed build ran four originals, three ISA selections and
one/eight threads: 24 official pixel checks passed. Selected one-thread
portable inclusive stage times, in seconds:

| Original | Total | Gather | Tokenize | Tree training | LZ parse | Fast seed within LZ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Document screenshot | 1.126 | 0.247 | 0.255 | 0.050 | 0.159 | 0.079 |
| 4K wallpaper | 5.206 | 1.291 | 1.325 | 0.328 | 1.232 | 0.801 |
| Manga | 0.662 | 0.139 | 0.181 | 0.045 | 0.147 | 0.100 |
| 16-bit RGB pattern | 0.112 | 0.022 | 0.023 | 0.002 | n/a | 0.006 |

These are diagnostic-build timings, not a comparison to the uninstrumented
table above. Parent stages overlap with their children; do not sum all
columns. At e5, the repeated pixel walks are a larger opportunity than
tree splitting alone. LZ matching remains an independent substantial cost.

Pixel samples support investigating Weighted and visitor/sample-collection
work, but individual measurements are only tens of nanoseconds, comparable
to clock and bookkeeping overhead. They do not justify exact per-operation
wall-time percentages. Keep the coarse stages as the main evidence.
Raw diagnostics: `out/low-effort-hotspots-20261114/profile4/`.

## Detailed Diagnostics

Build the optional feature only for investigation:

```powershell
cargo build --release -p rs-jxl-cli --example portfolio_lab --features rs-jxl/encoder-kernel-profile
$env:RS_JXL_PROFILE = '1'
$env:RS_JXL_MODULAR_ISA = 'avx2'
```

The additional pixel-stage counters are compiled out of ordinary builds.
One rotating-phase sample is taken per 128 visited pixels; the initial phase
depends on channel/stream/stripe so short tasks do not all time pixel zero.

- `gather`: RNG + Weighted; property materialization; sample accounting,
  reference properties and predictor residuals; compact sample collection.
- `channel_visit`: Weighted; properties; caller visitor (normally tree
  lookup and token emission, but also used by legacy sample collection).
- Existing Fast LZ diagnostics: query setup, spatial filtering, spatial
  match extension, ring matching, and sampled candidate/length counts.

Use each stage's own call count: gathering's residual/collector stages run
only for selected samples, while Weighted history advances through all
pixels, including stripe warm-up. The timestamps and local bookkeeping
perturb the sampled work. The empty-clock measurement is an overhead hint,
not a complete correction; do not multiply sampled times into claimed wall
totals. Inclusive parent stages and parallel sums overlap.

Headline performance must come from the normal build with profiling off.
`tools/benchmark_low_effort.py --isa NAME=avx2` (or `portable`, `avx512`)
sets an override only for that named native candidate after cleaning the
inherited experiment environment. Official cjxl receives no ISA override.

Local evidence, excluded from Git:

- `out/low-effort-hotspots-20261114/compare6/`
- `out/low-effort-hotspots-20261114/coefficients6/`
- `out/low-effort-hotspots-20261114/simd6/`
- `out/low-effort-hotspots-20261114/wp_asm.s`

## Structural Follow-up Implementation

The two materialization proposals above were subsequently implemented:

- Invocation-local Weighted history records the rounded prediction and signed
  property 15 during whole-channel sampling. Tokenization can reuse those two
  values on the same immutable transformed planes and WP header. It still
  calculates wire stream/channel/reference properties normally. Incomplete
  captures are never published. Unsupported plans and stripes use the existing
  walk. An 8 MiB cache ceiling is a fallback threshold, not an input or tree
  limit. `RS_JXL_WP_REUSE=0` disables reuse for diagnostic comparisons.
- The internal source-LZ result can borrow unchanged literals instead of
  constructing a replacement command vector. Fast matching allocates commands
  only after accepting its first match. The public owned-command API remains
  unchanged, including its enabled-with-no-match state.

The first normal-build comparison, before connecting history reuse to the
remaining single-stream writer, used the same six originals at e5, one/eight
threads and three interleaved repetitions. All 144 official djxl pixel checks
passed, including transparent and 16-bit inputs.

| Variant | One thread (s) | Eight threads (s) | MPix/s t1/t8 |
| --- | ---: | ---: | ---: |
| Previous ISA-final build | 6.02625 | 1.44989 | 1.965 / 8.167 |
| Structural, reuse disabled | 6.18158 | 1.39420 | 1.915 / 8.493 |
| Structural, reuse enabled | 5.75898 | 1.31581 | 2.056 / 8.999 |
| Official cjxl 0.12 | 5.15985 | 1.16029 | 2.295 / 10.205 |

All Rust variants total 16,542,788 bytes and arithmetic mean image bpp
6.875543. The enabled checkpoint reduces total time by 4.44%/9.25% against
the previous build on this small cohort, not on every image. This does not
establish an independent speedup from the LZ change: its one-thread result
with reuse disabled was worse. Nor does it close the official e5 speed gap.
Evidence: `out/low-effort-hotspots-20261114/structural6/summary.json`.

## Debug Stack Overflow Investigation

A parallel Modular Debug test run intermittently overflowed a worker stack.
Serial execution and a later parallel rerun passed; those passes alone were
not accepted as a fix. Windows CDB reproduced `STATUS_STACK_OVERFLOW` in
`block_context::plan::spatial`, reached by
`parallel_domains_and_spatial_plans_keep_ordered_histogram_decisions`.

The old Debug function prologue requests `0x17810` (96,272) bytes via
`__chkstk` on every recursive spatial call, excluding its `fork/join`
callers. `Candidate` contains four inline 128-bin histograms, and unoptimized
by-value candidate/result temporaries multiply that footprint. The recursion
halves the rectangle/budget and is finite; it is not evidence of an infinite
MA learner loop. Rayon nesting can retain additional parent calls, explaining
why different scheduling can change whether the available stack suffices.

This trace identifies an optional block-context planner, not ordinary e5's
new Weighted history cache. The corrective change moves candidate histograms
off the recursive stack while retaining partitioning, rate decisions and
parallel work. Production thread-stack sizes and planning limits must not be
increased or reduced to hide the failure.
Local debugger evidence: `out/low-effort-hotspots-20261114/stack-debug.log`.

After the fix, the matching Debug disassembly requests only `0x730` (1,840)
bytes in `spatial`, down 98.09%. Histogram initialization is isolated outside
the recursive frames; parent merges reuse the lower child's allocation.
Three new tests check candidate size/allocation reuse, a 256 KiB serial
stack, and one/four workers with 256/512 KiB stacks. Those smaller stacks
are test configurations only. The 20 block-context tests and full Modular
suite (816 passed, three existing ignored microbenchmarks) passed. The full
suite also covers the final single-stream Weighted-cache integration.

## Compact Token Follow-up

The next optimization targets repeated work after sampling, not reduced MA
training strength. After static channel/group filtering, the common low-effort
tree uses Gradient/Weighted leaves and properties 9, 10 and 15. When complete
Weighted history is available, a specialized token walk computes only these
three properties and emits residuals directly. It avoids the generic property
record and predictor-row scratch. Original entropy context IDs, integer
wrapping, Gradient LUT routing and edge-neighbor rules are retained. Trees
with other properties, predictors, nonzero offsets or nonunit multipliers
fall back to the existing path. This is not a new bitstream or search policy.

Fast LZ77 also accumulates its output wire-token count when accepting a match:
replace `length` literals with two tokens. The savings gate no longer needs a
second scan over all seed commands. Indexed, careful and experimental quick
parsers retain their existing counting path. Serial and parallel floating-point
savings reduction order is unchanged.

An independent read-only review checked wide-integer prediction, static tree
filtering and history identity. Targeted tests compare the compact path against
the generic visitor on edge widths, five WP presets and full-range i32 samples;
they also cover Gradient LUT routing and production token dispatch. LZ tests
compare accumulated counts with a command scan, including rejected matches,
empty streams, enabled-with-no-match results and real two-worker execution.

### Compact Walk Results

Six original images, e5, three interleaved repetitions, native encode time
excluding image I/O and decoder validation. Sum of per-image median times:

| Variant | One thread (s) | Eight threads (s) | MPix/s t1/t8 |
| --- | ---: | ---: | ---: |
| Previous ISA-final | 6.00732 | 1.45659 | 1.971 / 8.129 |
| Complete local-history reuse | 5.76794 | 1.39824 | 2.053 / 8.468 |
| Compact pair + incremental seed count | 5.11557 | 1.32475 | 2.315 / 8.938 |
| Official cjxl 0.12 | 5.24535 | 1.22141 | 2.257 / 9.694 |

The compact checkpoint saves 14.84%/9.05% against ISA-final and
11.31%/5.26% against the immediately preceding cache checkpoint. All Rust
variants total 16,542,788 bytes, mean per-image bpp 6.875543. Official totals
16,588,693 bytes, mean bpp 6.988006. Every one of the 144 independent djxl
pixel validations passed. This small cohort does not establish a general win
over official: eight-thread time is still 8.46% worse, and some small images
do not improve. Do not interpret the 2.47% one-thread lead as statistically
established superiority.

The three-image coarse profile confirms the new route is active: 4K wallpaper
tokenization falls from 0.650553 to 0.305072 seconds; the document screenshot
falls from 0.146283 to 0.064138 seconds. These are single instrumented runs,
not the timing table above. Fast-LZ timing does not show an isolated win, so
the incremental count change must not be credited with the whole improvement.
Monarch never enters the compact path: shared-tree learning did not yet retain
per-group Weighted history. That is the next concrete reuse opportunity.

Additional controls: 64 e1-e4 trials on two originals and eight e9 trials on
RGB/16-bit originals all passed official pixel comparison with unchanged sizes.
Their short-run timings are mixed; they are compatibility controls, not a
claim of universal low-effort acceleration. The current Release Python binding
suite passes 33 tests, with one existing optional test skipped. Modular tests
pass 818 plus the subsequently added Gradient-LUT test; LZ tests pass 339.

Local evidence: `compact6/`, `compact-stages3/`, `compact-controls/` and
`compact-e9-smoke/` beneath `out/low-effort-hotspots-20261114/`.

## Shared-Domain Follow-up

The remaining small-image gap is not only Rayon overhead. The ordinary e5
shared-tree route left `source_gather=None`, selecting a one-worker gather;
its default Fast LZ77 configuration also left `parallelism=None`. Scout mode
already supplied parallel execution options, but ordinary e5 did not. The
Monarch eight-thread diagnostic recorded 0.052345 seconds in gather and
0.067219 seconds in shared LZ parsing, both largely serial.

The follow-up makes two scoped changes:

- Shared learning retains independent Weighted histories for prepared streams.
  Learning can sort streams without confusing cache identity: each cache binds
  immutable planes and its own WP header before task dispatch. The entire
  shared domain keeps the existing 8 MiB cache budget; uncached groups or
  channels use the original walk. This avoids repeating Weighted prediction
  and permits the compact token path on cached shared groups.
- Ordinary e5 with multiple requested threads and a shared full-frame route
  supplies channel-parallel gather and Fast LZ stream dispatch by default.
  Existing minimum-work and scratch-budget checks still decide whether tasks
  run concurrently. Single-thread execution, streaming frame areas, responsive
  independent frames and other efforts retain their dispatch policy. Explicit
  caller gather/LZ configurations remain authoritative. No sampling fraction,
  search budget, predictor set, or row-striping policy is weakened.

Tests add unsorted shared streams with identical pixel data in different
allocations, different WP headers, complete/partial cache captures and one/four
workers. Projection tests cover shared, forced full-frame, streaming and
responsive-independent routes, including explicit serial LZ configurations.
Modular: 821 passed, three existing ignored microbenchmarks; adapter: 30 passed.

### Shared-Domain Results

The six original images were rerun at e5 with one/four/eight threads and three
interleaved repetitions per variant. Both Rust executables were frozen before
testing. No compilation ran concurrently. Times below are sums of per-image
medians, exclude file I/O and decoder validation, and include encoder pool setup.

| Variant | t1 seconds / MPix/s | t4 seconds / MPix/s | t8 seconds / MPix/s |
| --- | ---: | ---: | ---: |
| Compact checkpoint | 5.47090 / 2.164 | 1.95004 / 6.072 | 1.38892 / 8.525 |
| Shared history + e5 dispatch | 5.45158 / 2.172 | 1.74782 / 6.774 | 1.29780 / 9.124 |
| Official cjxl 0.12 | 5.52017 / 2.145 | 1.77700 / 6.663 | 1.26967 / 9.326 |

Against the compact checkpoint, aggregate time changes by -0.35%, -10.37% and
-6.56% at one/four/eight threads. This closes most of this small cohort's
eight-thread gap to official, but the new build is still 2.22% slower there.
The small one/four-thread lead over official is not a statistical superiority
claim. All Rust results total 16,542,788 bytes with arithmetic mean image bpp
6.875543; official totals 16,588,693 bytes and mean bpp 6.988006. Every one of
the 162 official djxl pixel comparisons passed, including invisible RGB and
16-bit samples. Matching output size is an observation, not a requirement to
preserve the old codestream.

Per-image median native seconds (compact -> shared):

| Original | t1 | t4 | t8 |
| --- | ---: | ---: | ---: |
| Monarch | 0.28912 -> 0.24647 | 0.20638 -> 0.13008 | 0.18378 -> 0.12065 |
| Paper 9bde | 0.74121 -> 0.76028 | 0.23474 -> 0.25142 | 0.16950 -> 0.16268 |
| 4K mlzoy1 | 3.60888 -> 3.61276 | 1.15548 -> 1.03128 | 0.77014 -> 0.78193 |
| YasasiiAkuma | 0.42640 -> 0.45728 | 0.15031 -> 0.14450 | 0.10591 -> 0.10183 |
| Transparent jam | 0.31128 -> 0.30163 | 0.12765 -> 0.13499 | 0.08898 -> 0.08013 |
| 16-bit circles | 0.09401 -> 0.07316 | 0.07548 -> 0.05556 | 0.07061 -> 0.05058 |

The gains are not universal. The streaming paths did not receive the new inner
parallel policy; their mixed timings, including slower screenshot/manga cases,
must remain visible. Three repetitions do not distinguish all code-generation,
load and scheduling effects. Do not use this cohort to claim no corpus-wide
regressions or compare its absolute timings to earlier non-simultaneous runs.

A separate instrumented two-image check passed eight more pixel comparisons.
On Monarch, the cache retained five of eight prepared streams (7,864,320 bytes)
under the shared 8 MiB budget. Gather changed from one to eight chunks, with
the same 18 channel tasks. At eight threads, inclusive gather wall time fell
from 0.067370 to 0.023419 seconds and shared LZ parse wall time from 0.069856
to 0.025098 seconds. At one thread, tokenization fell from 0.077330 to
0.037254 seconds because retained predictions enabled compact tokenization.
These stage figures are single instrumented observations, not the medians above.

The 4K wallpaper still uses outer frame-area parallelism. Its one-thread profile
records about 1.155 seconds in MA gather and 0.823 seconds in Fast LZ seed work,
against 0.308 seconds in tokenization and 0.307 seconds in tree training. These
inclusive scopes overlap other scopes and cannot be added indiscriminately.
Its eight-thread median remains 0.78193 seconds versus official's 0.70675:
more work remains in gathering/matching and task balance, not just pool startup.

The rebuilt Release DLL passes 33 binding tests with one optional skip. Eight
additional e3/e9 checks on RGB and 16-bit originals, at one/eight threads, pass
official pixel comparison. No full corpus run was started.

Local evidence under `out/low-effort-hotspots-20261114/`:
`shared6/summary.json`, `shared-stages2/summary.json`, and
`shared-controls/summary.json`. The frozen new benchmark executable is
`bin/shared-e5.exe`; the CLI and FFI Release targets use the same source revision.

## Completed 200-Image Follow-up (2026-11-15)

The full effort refresh completed: 1,800 JXL encodes passed independent djxl
pixel comparisons and all 200 recompressed PNGs passed sample comparisons.
The cohort contains 619,577,505 pixels. JXL uses eight threads, one timed trial
per image/effort. Native timing excludes I/O, process startup and verification,
but includes encoder pool setup. Official 0.12 and previous Rust timings are
cached, non-simultaneous measurements, not controlled paired repetitions.

| Effort | Native seconds | MPix/s | Mean image bpp | Total bytes | Cached official seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.547 | 243.249 | 8.680542 | 647,286,556 | 2.364 |
| 2 | 11.281 | 54.921 | 7.886610 | 577,376,725 | 15.458 |
| 3 | 15.300 | 40.496 | 7.583748 | 580,127,256 | 22.034 |
| 4 | 24.370 | 25.424 | 7.318401 | 547,802,335 | 41.813 |
| 5 | 47.460 | 13.055 | 6.900750 | 473,845,779 | 50.357 |
| 6 | 59.570 | 10.401 | 6.810007 | 466,296,945 | 64.026 |
| 7 | 89.425 | 6.928 | 6.701164 | 454,940,039 | 98.096 |
| 8 | 171.237 | 3.618 | 6.654696 | 451,718,709 | 267.529 |
| 9 | 305.103 | 2.031 | 6.439719 | 424,240,942 | 470.475 |

Mean image bpp gives each image equal weight; total-byte changes need not follow
it. e2-e9 retain the previous full-run sizes. e1 now uses real FastLossless,
whereas the historical Rust e1 followed the e2-like route; its much faster time
must not be presented as an equal-compression speedup. Against cached official
0.12, e9 takes 35.15% less time and 5.42% fewer bytes on this cohort. e1 remains
7.76% slower. These aggregates do not imply every image improves.

PNG level 9 totals 718,252,404 bytes, mean image bpp 9.858680, and 433.437 seconds
with one thread. It uses Pillow optimized PNG for 182 images and OpenCV/libpng
for 18 16-bit images. This is not an optimal-PNG search or a same-thread speed
comparison. Original PNGs total 708,591,234 bytes. e9 exceeds recompressed PNG
on three images by only 1,135 bytes combined; nevertheless paper9bde remains
354,791 bytes versus its original PNG's 323,776 bytes (+9.58%). Recompressed PNG
alone would hide that remaining target.

### Targeted Profiles and Scaling

Six additional e8/e9, eight-thread profiles on original 1515.hdr, 1244.hdr and
phoboslab.org passed official pixel comparison. Stage times below are inclusive
scope sums across tasks: they overlap, can exceed wall time, and must not be
added together or treated as exclusive CPU percentages.

| Image / effort | Native wall seconds | Cluster refine scope sum | MA train scope sum | LZ parse scope sum |
| --- | ---: | ---: | ---: | ---: |
| 1515.hdr e8 | 12.245 | 38.984 | 38.234 | 2.846 |
| 1515.hdr e9 | 22.249 | 88.273 | 57.575 | 1.371 |
| 1244.hdr e8 | 9.505 | 21.548 | 34.280 | 2.677 |
| 1244.hdr e9 | 20.243 | 72.090 | 61.745 | secondary |
| phoboslab.org e8 | 4.732 | 3.410 | 5.580 | 2.777 |
| phoboslab.org e9 | 18.050 | 4.385 | 29.608 | 62.631 |

The HDR cases are not primarily LZ bottlenecks. 1515 e8 evaluates 2,290,296
initial histogram pairs; its lower-bound counters skip only 7,051 pairs across
initial and update phases combined. e9 evaluates 4,639,365 initial pairs.
Do not divide the combined skip count by initial pairs to report a prune rate.
`entropy/cluster.rs::refine_with_workers` evaluates full population costs;
`ans_histogram.rs::fast_population_cost` tries a flat table and three rebalancing
shifts. This is a concrete target for cheaper candidate bounds/ranking before
normalization, not another generic LZ or wider-SIMD change.

A separate uninstrumented 1515 e8 check, two repetitions at each thread count,
passed four more pixel comparisons:

| Threads | Runs (seconds) | Mean seconds | MPix/s from mean |
| --- | --- | ---: | ---: |
| 4 | 22.40819, 21.03890 | 21.72355 | 0.561 |
| 8 | 12.46471, 11.38805 | 11.92638 | 1.022 |

Eight threads reduce time by 45.10% (1.82x speedup). This disproves a wholly
serial route for this case, not all possible task-granularity problems. No
one-thread or 16/32-thread sweep was run here.

### Next Experiments, Not Proven Improvements

1. **HDR clustering work.** Measure normalization and merge-update work
   separately; try stronger cheap bounds or a shortlist followed by exact
   population costs. Check allocation/dispatch costs per merge. The current
   code already has sketch bounds, versioned heap entries and parallel work;
   do not describe these as missing. Approximate candidate selection may alter
   compression and needs final bytes/time plus pixel validation, not an old
   codestream hash requirement.
2. **MA small/middle nodes.** 1515 e8 makes 577,520 node searches across 192
   learners. In `ma_train/split_search.rs`, the allocation-free serial reduction
   excludes `property_dedup`, which ordinary projection enables. Preserve dedup
   while reducing candidates directly. Separately test histogram-cache admission
   for the two-predictor middle-sized subtrees: existing earlier admission only
   covers the 14-predictor case. Do not blindly increase memory budgets.
3. **e3 HDR model selection.** Six HDR images account for 26,300,567 extra bytes
   at e3 versus e2. e2 uses fixed Gradient prediction/property 9; e3 replaces it
   with fixed Weighted prediction/property 15 and WP preset 0, without retaining
   the former model. RCT, LZ and entropy policy do not explain this transition.
   Separate prediction from context routing in ablations, then evaluate a cheap
   representative-tile selector. Threshold saturation and WP quality remain
   hypotheses, not established causes. Similar official sizes point to an
   inherited policy weakness, not a newly invalid bitstream.
4. **e8 marginal work.** e7 to e8 adds 91.48% time for 0.708% fewer total bytes.
   1515 adds 9.38 seconds for 0.048% fewer bytes, and 1546 adds 6.36 seconds for
   0.022%. e8 adds properties, finer bins, sampling and lower split admission,
   plus Best/Precise entropy policy. Newly added MA properties do win many splits;
   removing them is a compression-policy experiment, not proven waste removal.
5. **Long-page LZ.** phobos e9 has a different bottleneck: optimal parsing and
   chain batch filling. Its corpus result takes 15.99 seconds versus official
   13.90, but saves 15.76% bytes, so it is not simply a bad route. Investigate
   useful-match density and DP work separately from HDR clustering.

Evidence: `out/effort-png200-20261114/{points.csv,summary.json,bpp-time.png,`
`size-time.png}`, `out/post200-profile-20261115/heavy/summary.json`, and
`out/post200-profile-20261115/hdr-threads/summary.json`. This follow-up changes
no encoder policy or implementation; the above is a prioritized investigation
plan based on measured stages and source inspection.

## Conditional-Entropy Pruning Experiment

The follow-up changes only Best histogram refinement in `entropy/cluster.rs`.
It does not replace clustering with K-means, add encoder trials, reduce MA
sampling, or change the decoder. The existing logarithmic 16-bin sketch loses
all information about symbol distribution within a bin. Retaining a per-center
conditional entropy makes its merge lower bound considerably tighter.

For histogram `h`, total `N`, sketch `S(h)`, and bucket total `B[b]`, define:

```text
E(h) = sum_s h[s] * log2(N / h[s])
C(h) = sum_b sum_{s in b} h[s] * log2(B[b] / h[s])
E(a+b) >= E(S(a)+S(b)) + C(a) + C(b)
bound = max(0, E(S(a)+S(b)) + C(a) + C(b) - 1 - 0.0001*(Na+Nb))
```

Zero terms are omitted. Homogeneity and concavity of conditional entropy give
the inequality. The last two terms retain the existing source's margin for
fixed-point/f32 ANS population-cost rounding. Shannon statistics are computed
independently in f64, not from the `Histogram.entropy` field, which refinement
overwrites with population cost. Rebuild only the surviving center's statistic
after an accepted merge; other centers retain their values. Surviving candidate
pairs still use the existing exact Fast population-cost calculation and heap.
The separate neighborhood experiment retains its original coarse-bound API.

No new unsafe code, model tables, syntax, public options or global caches are
introduced. The added statistic occupies one f64 per center. This is a
calculation-pruning change, not an approximate nearest-neighbor shortlist.

### Paired Original-Image Results

Evidence root: `out/cluster-bound-20261115/`. The baseline is the frozen
`out/effort-png200-20261114/bin/portfolio_lab.exe`; the candidate is frozen as
`bin/conditional.exe` under the evidence root. The paired report records both
executable hashes and original input hashes. Builds/tests did not overlap timed
encodes. No official encoding baseline was rerun.

Fourteen originals cover two large 16-bit HDR images, a long webpage, manga,
alpha icons, photographs, textures, editor/document screenshots and palette
artwork. At e8/e9 and eight threads, two interleaved repetitions per image and
binary produce 112 independent official djxl pixel checks, all passing. Timings
are sums of per-image medians, excluding I/O and verification. Each image has
equal weight in mean bpp. None of these encodes changed output size.

| Effort / build | Native seconds | MPix/s | Mean image bpp | Total bytes |
| --- | ---: | ---: | ---: | ---: |
| e8 baseline | 31.97589 | 2.19067 | 7.109451 | 85,194,173 |
| e8 conditional bound | 27.33085 | 2.56299 | 7.109451 | 85,194,173 |
| e9 baseline | 69.35344 | 1.01002 | 6.585603 | 76,378,401 |
| e9 conditional bound | 59.35988 | 1.18007 | 6.585603 | 76,378,401 |

Aggregate time falls 14.53% at e8 and 14.41% at e9. The cohort deliberately
contains hotspot images; these percentages are not predictions for the full
200-image corpus. In particular, gains are not universal:

| Original / effort | Baseline seconds | New seconds | Time change |
| --- | ---: | ---: | ---: |
| 1515.hdr e8 | 12.68961 | 10.44196 | -17.71% |
| 1515.hdr e9 | 22.56118 | 17.32019 | -23.23% |
| 1244.hdr e8 | 9.55704 | 7.52971 | -21.21% |
| 1244.hdr e9 | 20.39727 | 15.63578 | -23.34% |
| phoboslab.org e8 | 4.47048 | 4.06911 | -8.98% |
| phoboslab.org e9 | 18.15129 | 18.38911 | +1.31% |
| 0446 e9 | 1.26590 | 1.21294 | -4.18% |
| wallhaven-mlzoy1 e8 | 2.22523 | 2.34647 | +5.45% |
| wallhaven-mlzoy1 e9 | 3.30811 | 3.21032 | -2.96% |

The e9 aggregate excluding both HDR originals is essentially unchanged. The
phobos e9 run ranges overlap and its main bottleneck remains LZ, not clustering.
Do not suppress the small slower-image results or infer universal acceleration
from the total.

### Why the HDR Improvement Is Credible

Six additional profiled encodes pass official pixel validation. These scope
times are inclusive sums across parallel work, not exclusive CPU time or wall
time. The baseline profile is cached from the preceding investigation.

| Original / effort | Old refine scope seconds | New refine scope seconds | Old skipped pairs | New skipped pairs |
| --- | ---: | ---: | ---: | ---: |
| 1515.hdr e8 | 38.984 | 23.596 | 7,051 | 1,662,959 |
| 1515.hdr e9 | 88.273 | 54.473 | 292,229 | 3,902,984 |
| 1244.hdr e8 | 21.548 | 8.676 | 33,102 | 1,227,910 |
| 1244.hdr e9 | 72.090 | 42.423 | 246,353 | 3,499,865 |
| phoboslab.org e8 | 3.410 | 1.787 | 89,917 | 226,695 |
| phoboslab.org e9 | 4.385 | 2.163 | 126,721 | 280,230 |

Skip counts include initial and post-merge candidate evaluations. They are not
percentages of initial pairs. Both the algorithm counters and paired whole-image
time support a real improvement on the HDR cases.

Four wallpaper diagnostics at one/eight threads also pass pixel validation.
Single-thread refine time falls from 0.1894 to 0.0888 seconds, while whole-image
instrumented time is almost unchanged (14.768 to 14.733 seconds). Its MA training
takes about 7.6 seconds. The corresponding eight-thread diagnostic reverses the
first paired timing result, so the initial +5.45% is not by itself evidence of
a repeatable clustering regression. These diagnostics are not headline timings.

Validation: 180 affected entropy tests pass, including empty/singleton and wide
prefix cases, serial/parallel refinement, dense/sparse distributions inside the
same sketch bucket, bucket boundaries, and weights near i32::MAX. Numerical tests
compare the adjusted bound against Shannon entropy and the actual population
cost; primitive refinement tests compare with unpruned merge decisions. Pixel
acceptance remains comparison with the original samples, not old JXL bytes.

The conservative-bound change is retained. Lazy exact-cost evaluation remains a
separate possible follow-up; it was not implemented or credited with these gains.
No full-corpus rerun or commit/push was performed for this experiment.

### Thread Controls and Wallpaper Retest

A separate three-original cohort (16-bit RGB rough-linen texture, DIV2K 0446,
and paper9bde) uses one/eight threads and two interleaved repetitions. All 48
official pixel comparisons pass, with unchanged sizes. These are controls, not
a substitute for single-thread measurements on the large HDR originals.

| Effort / threads | Baseline seconds | New seconds | Time change | Baseline / new MPix/s |
| --- | ---: | ---: | ---: | ---: |
| e8 / 1 | 11.51270 | 11.05686 | -3.96% | 0.45802 / 0.47690 |
| e8 / 8 | 2.02852 | 1.93831 | -4.45% | 2.59943 / 2.72040 |
| e9 / 1 | 19.26271 | 18.53676 | -3.77% | 0.27374 / 0.28446 |
| e9 / 8 | 3.27017 | 3.26210 | -0.25% | 1.61245 / 1.61644 |

For both binaries and thread counts, this cohort totals 9,155,124 bytes at e8
and 8,866,664 bytes at e9; mean image bpp is 16.742353 and 16.164490 respectively.
High bpp here reflects the small cohort including a 16-bit RGB texture. The
eight-thread e9 difference is too small to claim a meaningful speedup.

Five more uninstrumented interleaved wallpaper e8 repetitions give medians of
2.413074 seconds baseline and 2.392030 seconds new (-0.87%). Ranges are
2.28282-2.44521 and 2.30270-2.50071 respectively. The initially observed +5.45%
does not reproduce consistently. Treat this case as effectively unchanged, not
as a demonstrated improvement or proof that no future image can regress.

Across `profile3/`, `paired14/`, `wallpaper-profile/`, `threads3/`, and
`wallpaper-repeat5/`, all 180 original-pixel checks pass on 15 distinct originals.
Every matched baseline/candidate image-effort-thread key has the same output
size. Reports contain raw trial times, commands, receipts and hashes; no encoded
byte identity is required by the experiment.
