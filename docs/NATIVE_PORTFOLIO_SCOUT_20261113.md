# Native Experiment Portfolio Scout

Status: opt-in implementation, not a promotion of experimental tools into the
ordinary e9 default. The authorized full PNG corpus run completed on 2026-11-14;
see the launch record and results below. It improves aggregate size over the
current ordinary preset, but its time cost and historical per-image regressions
do not support default promotion.

## Scope

The native API is `rs_jxl::scheduler::portfolio::{plan, encode, Options}`.
`plan` inspects eligibility and source statistics without encoding. `encode`
returns a complete standard JXL plus an invocation-owned routing report. The
Python script only inventories files, invokes the native test executable and
checks results; it does not choose codec parameters.

This policy supports unsigned 1..16-bit main/extra sample storage accepted by
the existing source adapter, and efforts 8..10. It does not change the ordinary
encoder's wider sample support. Source image metadata and hidden RGB remain
intact. Subsampled extras cannot be cropped by the existing probe API: such an
input retains its complete baseline rather than manufacturing a crop score.

JPEG byte reconstruction is a separate objective and is not mixed with this
PNG pixel-lossless portfolio. The old full-corpus manifest contains JPEGs as
well as PNGs; the runner explicitly inventories JPEGs as excluded, not as
passed pixel cases. No RCT x previous-channel exhaustive sweep is enabled.

## Candidate Registry

Every recipe has a stable native enum identity, name and family. Each starts
from a fresh ordinary source preset; changing one recipe cannot contaminate
the next. Native option mapping lives in `portfolio_catalog.rs`, independently
of statistics (`portfolio_features.rs`) and execution (`portfolio.rs`).

The catalog includes:

- Distribution leaf offsets; prev1/prev2, alone and with Distribution.
- Full14 shallow contexts with static bias, previous-channel or Distribution
  variants; Activity81 replacement and Activity25/81-refined full trees.
- Cost/Symmetric MA learning, description-cost scoring and width-2 lookahead.
- Context-map-aware refinement.
- Run-frequency, hot128 and graph-based palette orders.
- All six exact-delta/zero palette modes.
- RG, YCo, YCg, CoCg and YA/CoCg partial palettes.
- Source Add, its parallel-gather variant, previous-channel/Distribution
  combinations and reference-instance selection; exact tile16/off-grid Replace
  patches. EZeng insertion-only and EZeng plus four MinLA rounds are distinct.

Use `portfolio_lab --list-candidates` for the authoritative names. A recipe
being registered does not mean that its transform was accepted by the internal
kernel, nor that it wins. Structural exclusions are logged separately from
heuristic rejection and exhausted experimental budgets. Known weak experiments
remain measurable but are not automatically tried on every original.

These are candidate compositions, not isolated one-variable ablations. In
particular, partial palettes require their compatible full-frame/cap4096
configuration. The previous fixed-state Palette admission experiment remains
separate: this router does **not** fix every local `attempt()` cost estimate or
claim that one whole-mode winner identifies the cause of every local decision.

## Scout And Final Decision

1. Reuse the native full-coverage scan to find spatially diverse activity
   regions. Copy at most three original-pixel crops without resizing or an
   artificial concatenation seam.
2. Measure bounded, deterministically jittered source samples: tuple diversity,
   flatness, clamped-Gradient zeros/bias, conditional residual-bucket entropy,
   Gradient/Left/Top residual diversity/zero fractions and sampled exact nonconstant tile
   repeats. Interleaved alpha participates in tuples/equality, including
   invisible RGB; separate extra planes affect coverage but not these bounded
   tuple statistics.
   Delta/Zero candidates use their own predictor-direction statistics rather
   than assigning the same priority to all six modes.
3. These features prioritize a family-balanced first round, not hard rejection.
   Source RGB dependencies do not assert dependencies after a particular RCT;
   real candidate probes apply their own RCT, MA, LZ77 and ANS.
4. Encode the ordinary baseline on each crop. Encode a first crop for candidate
   recipes, retaining work for confirmation on other regions. Rank confirmed
   candidates by their complete crop byte counts, not MA residual error alone.
5. Permit at most one uncertainty witness, including exact crop ties. Distant patch repetitions can disappear
   in crops, so repeat/flat evidence can justify a full-image patch witness even
   on tied crop scores. Rejections and untested candidates remain distinguishable.
   A repeated-tile Patch witness now has a reserved slot in both confirmation
   and final selection. Tiny crop gains cannot consume its slot. Flatness alone
   is weaker evidence: its witness cannot displace clearly better crop scores
   outside the uncertainty band. Family diversity
   is a near-tie preference within the uncertainty band, not a hard exclusion
   of a stronger second candidate from the same family. Identical complete
   crop-score vectors within a selected family are correlated evidence: prefer
   an eligible candidate with different evidence before spending another slot
   on that twin. Twins may still fill otherwise unused slots. Identical crop
   byte counts do NOT prove identical output, transforms or final size.
   These are routing
   heuristics, not an Oracle recall guarantee.
6. Fully encode only the admitted finalists. Retain the complete ordinary
   baseline and return the smallest complete output, with baseline winning ties.

The current default experimental budgets are 48 crop encodes, two complete
alternatives, and 128-pixel crop sides. These are configurable **search work**
budgets, not arbitrary MA-tree or decoder limits. The same pool is reused.
Crop roots run concurrently, each with a null inner runner. Roots are capped
by the requested threads and the existing 32 MiB probe scratch estimate under
a 1 GiB dispatch budget (not an RSS guarantee). Complete finalists remain
sequential and use the full pool internally. `--serial-probes` restores
sequential crop roots with internal data parallelism for comparison.
Tiny images use bounded direct trials instead of repeatedly encoding a crop
equal to the whole image. A zero finalist budget returns only the baseline.

Preparation sharing uses image-scoped immutable caches, never shared learned
trees, residual tokens or compressed results. The default retained budget is
128 MiB, half for the original and half divided among crops. Keys include
transforms and preparation options; incompatible roots/recipes bypass reuse.
`--preparation-cache-mib 0` disables it. This is a retained-data budget, not a
limit on all encoder allocations. Concurrent cold misses may build twice.

Losing full candidates may stop when finalized disjoint output sections already
cannot beat the incumbent. This typed `full-bounded` outcome carries a byte
lower bound, not a fabricated output size or an error. It is not based on crop
scores or a timeout. `--no-finalist-bound` disables this optimization. The
baseline is always encoded completely; ties retain the incumbent.

Feature weights, first-round allocation and the 1% crop uncertainty band are
provisional heuristics, not a fitted/validated classifier. They may miss gains.
All candidates have not been promised a probe when the budget is smaller than
the catalog. No hard wall-clock deadline interrupts the native encoder.

## What Is Guaranteed

The final byte count cannot exceed the **same invocation's** ordinary baseline,
because that complete output is retained. This is not a consequence of the
scout's accuracy. It costs a full baseline encode and up to two alternative
encodes; routing can be slower than ordinary e9. We do not hide that cost by
reporting only the winner's execution time.

Losslessness still requires actual decode validation. A compact crop score does
not prove global size or correctness. The candidate registry does not introduce
new decoding syntax. Correctness compares original samples, not JXL hashes.
Source hashes in benchmark manifests identify input files; they are not output
compression objectives.

## Report

The report includes source statistics, crop coordinates, every candidate's
eligibility/decision/reason, optional per-crop bytes, full bytes/time, exceptions,
the winner, encoded probe pixels and counts. `null` means unmeasured, not zero.

- `seconds`: entire native call, including scan, pool setup, ordinary baseline,
  probes and all discarded finalists. CLI startup, PNG read and output I/O are
  outside this measurement.
- `scout_seconds`: planning plus crop-stage work, already included in `seconds`.
- `baseline.seconds`: ordinary complete baseline, also included in `seconds`.
- `candidates[].full_seconds`: each full trial including the baseline entry.
- `candidates[].probe_seconds`: summed elapsed crop work. Concurrent work
  overlaps, so this sum is not scout wall time or CPU time.
- `candidates[].full_lower_bound`: only set for `full-bounded`, with `full_bytes`
  absent. Bounded attempts still count toward `full_encodes` and have timing.
- `probe_workers`: maximum independent roots dispatched in a batch, not an
  estimate of physical CPU utilization.
- `preparation_reuse`: hit/miss/duplicate-build counters and summed per-scope
  retained snapshots/peaks. Crop scopes are released before finalists; these
  totals are not simultaneous process memory measurements.
- `bpp`: final bytes times eight divided by original width times height.

The corpus summary reports both arithmetic mean per-image bpp and the separate
pixel-weighted `8 * sum(bytes) / sum(pixels)`, total native time and throughput.
Failed encodes/decodes and unavailable measurements are not silently dropped
from the population; successful-only aggregates state their denominator.

## Build And Small Checks

```powershell
cargo test -p rs-jxl --lib scheduler::portfolio --locked --offline -j4
cargo test -p rs-jxl --test portfolio --locked --offline -j4
cargo test -p rs-jxl-cli --example portfolio_lab --locked --offline -j4
cargo build --release -p rs-jxl-cli --example portfolio_lab --locked --offline -j4
python -B -m unittest discover -s tools -p test_benchmark_portfolio.py
```

```powershell
.\target\release\examples\portfolio_lab.exe --list-candidates
.\target\release\examples\portfolio_lab.exe image.png --plan-only
.\target\release\examples\portfolio_lab.exe image.png --output trial.jxl --threads 8
```

The example is a development entry point. Installed CLI/Python defaults are
unchanged; this new portfolio is not yet added to the frozen C ABI.

Initial implementation validation: 17 native unit tests, three native integration
tests and the CLI parser test passed. Integration tests also decode every
supported catalog recipe directly, not merely the selected winner. The runner's
18 tests passed with its explicit official-djxl smoke enabled: two 80x72 RGBA
fixtures at 8 and 16 bits, including nonzero RGB under zero alpha. This verifies
wiring and small cases, not corpus compression performance or complete coverage
of all transform activation paths. Tests used the development profile; a
Release binary must be built before timing the deferred corpus run.

## Targeted Parallelism And Selection Check

This is a five-image diagnostic set, NOT the deferred full corpus or a fresh
holdout. The originals total 5,515,650 pixels: Kodak `kodim06`, a Hacker News
screenshot, a transparent spreadsheet icon, a 16-bit RGB circles fixture, and
DIV2K `0533`. No originals were resized. Release effort 9 uses the same default
48-probe/two-finalist budgets. The reference is the committed Portfolio at
`9f71dbf407120204d009667976f920a6cbcc5163`, not ordinary e9 or official cjxl.

Native time includes the baseline, all probes and all finalist attempts;
CLI startup, file I/O and verification are excluded. One run per image was
measured at 1/4/16 threads; 8-thread numbers sum per-image medians of two runs.
The initial old/new run order was interleaved, but the final selection fix was
remeasured later against that frozen old reference. There was no CPU affinity
isolation. Small timing differences are not statistically established.

| Threads | Old total s | New total s | Time reduction | Old MPix/s | New MPix/s | Old scout s | New scout s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50.790 | 47.118 | 7.23% | 0.109 | 0.117 | 10.107 | 9.437 |
| 4 | 22.879 | 17.404 | 23.93% | 0.241 | 0.317 | 7.989 | 3.030 |
| 8 | 18.304 | 13.423 | 26.66% | 0.301 | 0.411 | 7.805 | 2.163 |
| 16 | 17.894 | 11.830 | 33.89% | 0.308 | 0.466 | 8.031 | 1.818 |

These compare the combined parallelism/cache/bound/selection revision. They do
not isolate the contribution of each change. Full-finalist time does not
uniformly improve: at eight threads it changed from 7.276 to 7.770 seconds,
partly because selection changes which candidates execute. Scout time is the
clear aggregate improvement; more threads do not eliminate full encoding work.

| Size metric | Old Portfolio | New Portfolio |
| --- | ---: | ---: |
| Total bytes | 2,044,860 | 1,999,484 |
| Arithmetic mean image bpp | 3.121824 | 3.084036 |
| Pixel-weighted bpp | 2.965902 | 2.900088 |

The 45,376-byte reduction (2.219%) comes entirely from the screenshot:
110,876 to 65,500 bytes, a 40.925% reduction. The other four final sizes are
unchanged. All tested thread counts select outputs with these sizes. This is
not a claim that other inputs cannot regress relative to the old Portfolio.

An intermediate revision regressed the 16-bit fixture by 329 bytes: weak
flatness evidence first displaced its useful candidate; after that was fixed,
two shallow variants with identical crop-score vectors consumed both slots.
The final weak/strong witness and correlated-evidence rules restore 40,234
bytes. Tests retain both cases. These rules were developed on this diagnostic
set, so their generalization still needs the separately authorized corpus run.
Recall is still incomplete: a separately measured `source-prev2` candidate on
the screenshot is 62,646 bytes, smaller than the selected 65,500. No full-catalog
Oracle recall rate is claimed from this check.

Final validation:

- Workspace/all-target offline compilation passed.
- 32 Portfolio and 45 Palette-focused library tests passed.
- Four Portfolio and two Palette-admission integration tests passed.
- Two Portfolio CLI and three Palette-admission CLI tests passed.
- All 27 Python runner tests passed, including actual native/official-djxl
  smoke checks for RGBA8 and RGBA16 with invisible RGB.
- All 25 final Portfolio outputs passed official-djxl original-sample
  comparisons, including 16-bit samples and alpha. The separate admission
  pilot's 15 outputs also passed, but its ranking result was negative; see
  `PALETTE_ADMISSION_20261113.md`.

Reproduction artifacts are under
`../jxl_python/out/portfolio-opt-20261113/`: the root `summary.json` retains
the old reference measurements; `accepted/summary.json`, `accepted/rows.jsonl`
and `accepted/identity.json` identify the final revision. `accepted/run_comparison.py`
is the targeted driver. The earlier root/new and `final/` results are superseded
by `accepted/`; they must not be presented as the final performance result.
All are ignored experiment artifacts, not runtime or packaging dependencies.

Ordinary e9 defaults, frozen C/Python interfaces and the deferred-corpus policy
are unchanged. No claim is made that the new Palette pilot corrects admission.

## Deferred Corpus Run

`tools/benchmark_portfolio.py` defaults to **prepare only**. It never builds an
encoder or starts compression without `--run`. There is no implicit sample size
limit; all eligible original PNGs in the supplied manifest are inventoried,
including large images. Source-file duplicates are logged with their aliases.
Malformed PNG/APNG inputs and missing files retain explicit failure records.

Example preparation after the executable has been built:

```powershell
python -B tools/benchmark_portfolio.py --manifest out/encoder-corpus-20261025-v23/manifest.json --out out/portfolio-full --djxl D:/cpp_pj/jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe
```

Only after the user authorizes the full run, repeat the same command with
`--resume --run`. Verification uses official djxl and exact original sample
comparison including uint16/alpha. Temporary encoded/decoded files are removed.
Input, executable, script and config identities are frozen; resume refuses a
different build or changed input rather than mixing benchmark results.

For unattended runs, create `STOP` in the result directory to finish the
current image and stop before the next. Remove that request before resuming.
`progress.json` includes the runner PID and active original path. Report-file
sharing failures are retried; persistent failures warn without aborting the
measurement journal. Manifest, identity and journal IO failures remain fatal.
Progress/summary reports can be regenerated by resuming a completed run.

The runner's explicit per-process timeout is a failure guard, not a speed result
or successful fallback. Interrupted runs retain journals; a hard-kill stale lock
must be checked before removal. Ctrl-C is not an instruction to silently skip a
slow original. Old official results are not recomputed by this runner.

Before claiming better routing, inspect budget-deferred candidates and perform
independent direct-candidate audits on preselected samples. The retained-baseline
size guarantee alone cannot reveal missed opportunities. Do not tune heuristics
on the eventual full-corpus result and call the same set a fresh holdout.

## Authorized Background Run

Started 2026-11-13 at 23:04 Asia/Shanghai as a hidden independent Windows
process, initial PID 9992. Results are in
`out/portfolio-full-20261113-be4fcd9/results/`; the parent directory contains
frozen executables/scripts, `session.json`, `cached_baselines.json`, and logs.
The encoder is the tested Release build for `be4fcd9`, SHA256
`34cc852563e86a0195d6c92fd418818fab176a2d96169929ad55ad87315a984e`.
Only reporting retry/STOP handling changed after that commit, with 33 runner
tests passing. No encoder changes or old-encoder runs accompany this sweep.

The original 8885-entry catalog contains 5738 PNG entries. Exact-file
deduplication leaves 5512 attempts plus 226 logged aliases. Another 3147 entries
(JPEG and Netpbm) are outside this Portfolio PNG experiment, not correctness
passes. The inventory retains 27 malformed/unsupported PNG records as explicit
failures rather than replacing them. No size-based subsampling or resizing is
used. Run options: e9, 16 threads, serial original files, 48 crop probes, two
full finalists, 128-pixel crops and 128 MiB retained preparation budget.

Every completed new output is checked against original samples through the
frozen official djxl. Generated images are temporary. The existing 1800-second
process guard records failures, never successful speedups. Old results are
identified by hashes in `cached_baselines.json`; comparisons must match source
and encoder-payload identities. Other user workloads may be active and old
timing configurations differ, so historical times are not controlled speedups.

Inspect `results/progress.json`, `results/summary.json` and `results/trials.jsonl`
for actual completion, failures, bpp and whole-policy timing. `session.json`
retains its starting state until the driver finishes; live progress is in
`results/progress.json`. To stop gracefully, create `results/STOP`. After it
stops, remove that file and launch the frozen `run_background.py --resume`
with the same Python environment. Do not launch a second writer while running.

## Full-Corpus Results, 2026-11-14

The run finished at 02:58:59 Asia/Shanghai on November 14, after 3h54m52s wall
time. All 5512 planned attempts have durable journal entries. Of these, 5485
original PNGs passed official djxl exact-sample checks, including hidden RGB
and 16-bit samples. The remaining 27 never entered encoding: 24 APNG inputs,
two invalid IEND/trailing-data files and one invalid iCCP CRC. They remain
explicit failures in the journal, not successful encodes or discarded cases.
There were no encoding, candidate, timeout or pixel-comparison failures among
the accepted static PNGs. ICC warnings in stderr do not imply pixel mismatches.

The audit rechecked every journal checksum/index, frozen input inventory,
native receipt, encoder stdout, script/binary identities and historical journal
hashes. It regenerated the summary and matched the saved summary exactly.
No encoder was rerun. This verifies the recorded experiment, not all possible
format features or metadata preservation (the pixel harness strips metadata).

### Same-Invocation Ordinary E9 Comparison

5485 unique originals contain 5,154,388,544 pixels. Both columns use the same
prepared inputs and 16-thread native implementation. Portfolio time includes
its retained ordinary baseline, probes and losing alternatives.

| Metric | Current ordinary e9 | New Portfolio e9 |
| --- | ---: | ---: |
| Total bytes | 4,066,158,336 | 3,984,599,293 |
| Arithmetic mean per-image bpp | 5.966576 | 5.851736 |
| Pixel-weighted bpp | 6.310985 | 6.184399 |
| Native seconds | 2,530.169 | 10,409.965 |
| Aggregate MPix/s | 2.037 | 0.495 |

Portfolio saves 81,559,043 bytes (77.781 MiB, 2.006%) at 4.114 times the native
time. 4560 outputs are smaller, 925 tie, and none is larger than the retained
same-invocation baseline. That last guarantee follows from retaining the full
baseline; it is not evidence of optimal selection or no historical regressions.

| Dataset | Unique passed | Bytes saved vs current ordinary | Time multiplier |
| --- | ---: | ---: | ---: |
| jxl-testdata | 2917 | 2.831% | 4.75x |
| diverse-downloads | 1027 | 4.311% | 4.29x |
| DIV2K | 800 | 0.965% | 3.27x |
| local-train | 402 | 6.817% | 5.43x |
| manga109 | 109 | 1.724% | 4.38x |
| urban100 | 100 | 2.089% | 4.13x |
| BSDS100 | 100 | 1.203% | 4.41x |

Aliases are assigned to their first original dataset in the frozen inventory;
the counts are not a claim of independent samples or balanced content classes.
Smaller datasets remain in the detailed CSV, not silently excluded from totals.

### Where Time Went

- Ordinary retained baseline: 2530.169 s (24.31% of native total).
- Planning/crop scouting: 1652.543 s (15.87%).
- Additional full candidates, including bounded attempts: 6225.199 s (59.80%).
- Other orchestration: approximately 2.055 s.

There were 213,897 crop encodes and 16,283 full-image attempts, including 2128
safe bounded exits. This is 2.969 full-image attempts per accepted original:
the two-finalist budget is almost always consumed. Losing extra candidates
account for 2445.071 s, but the remaining extra time also includes genuinely
better, expensive winners. Crop scouting is no longer the dominant total cost.
Summed overlapping crop task times must not be added to these wall-clock totals.

2486 images save at most 1%, yet collectively spend an additional 2666.558 s
over ordinary e9. Median native time is 1.178 s, p95 is 4.702 s and p99 is
13.180 s. 76 images exceed 10 s, 13 exceed 30 s and three exceed 60 s.

| Slow original | MPix | Portfolio s | Ordinary s | Winner s | Saving | Winner |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| phoboslab.org.png | 27.136 | 102.693 | 11.437 | 77.851 | 28.35% | source-prev2 |
| Graveyard_27x39_ENGLISH.png | 94.835 | 79.088 | 25.105 | 28.397 | 3.61% | prev2 |
| floral_frame_PNG270.png | 18.062 | 61.139 | 5.772 | 48.473 | 14.20% | source-prev2 |

These three scout stages take only 1.598 s combined. Stronger scouting alone
cannot remove their winner execution cost. The Source+prev2 backend deserves
profiling independently of candidate selection. The 94.8 MP Graveyard original
was encoded without resizing and passed the pixel oracle.

### Historical Comparisons

Each row below uses its own strictly paired cohort: original-file hash AND
actual prepared encoder-payload hash must match, and both versions must have
passed their recorded verification. Duplicate paths do not multiply the current
population. Last attempts are retained, never the smallest historical result.
Do not compare totals across rows with different cohorts.

| Historical reference | Strict pairs | New total-size change | New smaller / equal / larger |
| --- | ---: | ---: | ---: |
| Official libjxl 0.12 e9, old full run | 1111 | -1.083% | 726 / 4 / 381 |
| Optimized C++, old full run | 1111 | +2.158% | 324 / 1 / 786 |
| Optimized C++, later 500-image run | 66 | +4.095% | 20 / 0 / 46 |
| Rust v23, old full run | 5403 | -0.686% | 2857 / 238 / 2308 |
| Rust later 500-image run | 313 | -1.012% | 158 / 15 / 140 |

For the C++ full run, 4367 otherwise verified sources have different prepared
payloads, mostly reflecting input preparation/metadata differences, and seven
lack successful prior verification. They are not included in the strict table.
The 1111-pair cohort's arithmetic mean bpp is 4.013237 (official), 3.876439
(optimized C++) and 3.977528 (new Portfolio).

On the 5403 strict Rust-v23 pairs, weighted bpp improves 6.187622 -> 6.145187,
but arithmetic mean per-image bpp worsens 5.742581 -> 5.781899. Total bytes and
unweighted image means answer different questions; neither should hide the
other. Recorded native time is 5140.469 -> 9799.973 s (1.906x). This is historical
context, not a controlled performance measurement: policies and runtime load
changed. C++ recorded time includes process/I/O and uses different concurrency,
so it is especially unsuitable for a direct native-throughput speedup claim.

There are substantial individual size regressions against old Rust, despite
the current-baseline guarantee. Examples: `pk02_wall_big02a_N.png` grows from
459,594 to 851,149 bytes (+85.20%); `pk02_wall_big02b_N.png` grows from 584,490
to 885,382 (+51.48%). This experiment does not isolate which representation or
encoder change caused those regressions. Inspect old winning routes before
attributing them to the latest optimization alone.

### Candidate Coverage And Next Work

The largest selected savings come from `prev1-distribution` (874 wins,
25,939,138 bytes), `source-prev2` (439 wins, 18,427,829 bytes), and `prev2`
(941 wins, 15,357,414 bytes). These are selected-output contributions, not
marginal gains from disabling each recipe.

The full run also exposes budget starvation. `prev2-distribution`,
`activity81-refine`, `palette-ezeng-minla`, `palette-cocg`, `palette-ya-cocg`
and `source-min-entropy` received ZERO probes/full attempts. `source-medoid`
received only five first probes and no confirmation. Their zero wins cannot be
interpreted as evidence of no compression value. The fixed family rounds and
stable within-family tie order can leave later variants outside the first-probe
budget on every image. This is an all-input routing evaluation, not an exhaustive
all-recipe experiment; full-catalog Oracle recall remains unknown.

Next priorities are to audit starved candidates on a held-out diagnostic subset,
fix the first-round allocation, avoid spending both full slots on negligible
expected gains, and profile costly winning Source+prev2 execution. Simply raising
all search budgets would increase the already dominant full-encoding cost.
Keep this Portfolio opt-in; do not promote the unsuccessful Palette admission
pilot or change ordinary e9 based solely on these aggregate savings.

Detailed read-only analysis is in
`../jxl_python/out/portfolio-analysis-20261114/analysis.json`, with `cases.csv`,
`datasets.csv`, `candidates.csv`, `failures.csv`, and strict `paired_*.csv`
comparisons. `analyze.py` reproduces the audit without executing an encoder.
No codec code or defaults were changed while reviewing this run.

## Cheap Recipe Follow-Up (2026-11-14)

The corpus candidate timings are NOT sufficient to authorize defaults. They are
conditional on crop selection, frequently reuse a warm preparation cache, and
exclude complete output sizes for bounded losing candidates. For example, the
selected `learning-symmetric` timings look cheaper, whereas the independent
measurements below do not.

`portfolio_lab --direct-candidate NAME` now measures one existing catalog recipe
on an original image. It runs exactly one native encode, without scout, fallback,
output bound, or shared preparation cache. The native timer includes recipe
projection and pool setup, but not image input/output. Incompatible scout flags
are rejected, rather than silently ignored. `RS_JXL_PROFILE=1` optionally writes
the normal stage report to stderr. This is a development example, not a new
Python API or default-policy change.

```powershell
target/release/examples/portfolio_lab.exe input.png --output direct.jxl `
    --effort 9 --threads 16 --direct-candidate description-half
```

The first cold audit has 32 original-resolution images: four independently
seeded selections from each of six datasets (24 images), four additional
16-bit precision controls, and four known adverse controls. Candidate order is
deterministically shuffled per image; each execution uses a fresh process.
Inputs use the same metadata-free pixel payload as the corpus, with source and
payload hashes checked. All 192 outputs passed official `djxl` original-sample
comparison, including transparent RGB and 16-bit samples. This tests pixel
encoding, not JPEG transcode or metadata preservation.

Outcome-independent 24-image stratum, e9, 16 threads:

| Direct recipe | Total size change | Native time / baseline | Smaller / equal / larger | Mean image bpp |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0 | 1.000 | 0 / 24 / 0 | 7.287581 |
| Description half | -0.742% | 0.828 | 14 / 3 / 7 | 7.226415 |
| Symmetric learning | -0.400% | 1.069 | 19 / 3 / 2 | 7.255695 |
| Previous 2 channels | -0.885% | 1.220 | 18 / 6 / 0 | 7.215338 |
| MinLA palette | -0.019% | 1.006 | 6 / 17 / 1 | 7.248708 |
| Distribution offsets | -0.531% | 1.145 | 19 / 5 / 0 | 7.247594 |

Baseline totals are 26,034,737 bytes and 12.231979 native seconds. Size change
uses summed bytes; mean bpp is the arithmetic mean over images, not weighted by
pixel count. These are small-sample single measurements, not confidence bounds
or estimates of the whole corpus. The balanced dataset strata intentionally do
not reproduce the full corpus's large icon population.

Adverse controls must remain separate from the representative aggregate.
MinLA grows `status-weather-clear-night.png` from 20,072 to 23,741 bytes
(+18.28%). A 16-bit precision control grows from 217 to 250 bytes (+15.21%).
Description half has ten larger outputs across all 32 cases, despite improving
the aggregate. Hence none of these numbers justify unconditional replacement of
ordinary e9. Distribution still pays for trial serialization; its observed zero
regressions must not be interpreted as free refinement.

The immediate default-policy research priority is description-aware MA splitting:
it can avoid work within one encode, unlike choosing the best of two complete
encodes. Its small-file and per-image regressions still need an admission/model
solution and independent validation. MinLA is better treated as a conditional
palette tool, not a universal image route. Keep Portfolio and all these optional
recipe settings opt-in for now.

The follow-up artifacts are under
`../jxl_python/out/portfolio-analysis-20261114/`: `cheap_routes.json` records the
selection-biased historical audit, while `direct-t16/manifest.json`,
`direct-t16/trials.jsonl` and `direct-t16/summary.json` record independent runs.
`direct_routes.py` reproduces the small audit with a supplied Release example.

### Thread Check

Use the first original of at least 100,000 pixels in each of the six strata,
plus the description/symmetric adverse controls, without selection by measured
speed: eight identical images, 8.385380 MP total. Fresh 1/4-thread runs use the
same frozen Release binary as the 16-thread audit; the 16-thread rows below are
the matching existing measurements, not reruns. Builds were kept separate from
timing. All 48 additional outputs passed official pixel comparison.

| Threads | Baseline seconds | Description-half seconds | Time saved | Baseline MPix/s | Description-half MPix/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.458537 | 22.023280 | 13.49% | 0.329 | 0.381 |
| 4 | 7.973753 | 7.006014 | 12.14% | 1.052 | 1.197 |
| 16 | 4.380111 | 3.905483 | 10.84% | 1.914 | 2.147 |

Baseline is 8,048,552 bytes; description-half is 8,015,254 bytes (-0.414%) at
each thread count. This is evidence that the benefit is not solely warm-cache
or parallel scheduling behavior. It is not a guarantee for every image:
`pkw_tiles01b.png` is +0.722% larger and takes 0.326 vs 0.199 seconds at one
thread. Times are individual measurements, not statistical confidence bounds.
Artifacts: `direct-t1/` and `direct-t4/` beside `direct-t16/`.

### Palette Implementation Follow-Up

The optional MinLA path already builds an exact symmetric adjacency matrix.
It now extracts weighted edges from that matrix for its adjacency seed, instead
of also collecting up to 524,288 duplicate pairs, sorting them, and recounting
the same edges. This removes up to 4 MiB of redundant pair storage per call.
The weighted-edge ordering, endpoint ties, sampled graph, and refinement rules
are unchanged; ordinary adjacency callers share the weighted ordering helper.
No new automatic palette policy or sampling restriction was added.

Focused tests cover sparse/dense graphs, frequency/edge ties, disconnected
colors, empty through 256-color inputs, implicit indices, signed and hidden RGB
samples. The primitive seed comparison is a computation-equivalence check, not
a requirement that compression research preserve old JXL bytes. A separate
ignored `matrix_seed_timing` microbenchmark is available; do not confuse its
seed-only timings with whole-image encoding speed.

Validation: the normal Release example built successfully; 26 focused palette
tests and three CLI tests passed (the manual microbenchmark remains ignored).
All 32 optimized MinLA outputs passed official original-pixel checks and had the
same per-image sizes as the old MinLA recipe. A further five-image old/new
interleaved run, three repetitions each, passed all 30 official pixel checks:

| Image | Old median seconds | Matrix median seconds | Change |
| --- | ---: | ---: | ---: |
| `pkw_wall03f.png` | 0.209324 | 0.203892 | -2.60% |
| `{pk_chleaves1b.png` | 0.085591 | 0.081733 | -4.51% |
| 16-bit progressive circles | 0.170148 | 0.172819 | +1.57% |
| `status-weather-clear-night.png` | 0.313441 | 0.309422 | -1.28% |
| DIV2K `0449.png` | 1.012411 | 1.033450 | +2.08% |

These small mixed whole-image differences do not establish a broad encoder
speedup. The clear implementation benefit is avoiding redundant collection,
sorting, and pair storage; this alone is not a reason to enable MinLA by default.
The two binaries and verification journals are frozen in `minla-matrix-t16/`
and `matrix-interleaved/` alongside the earlier audits. Across cold recipes,
thread checks and this implementation check, 302 official pixel comparisons
passed. No full-corpus rerun or default-policy promotion was performed.
