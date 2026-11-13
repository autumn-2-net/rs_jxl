# Native Experiment Portfolio Scout

Status: opt-in implementation, not a promotion of experimental tools into the
ordinary e9 default. Large-corpus execution is explicitly deferred until the
user authorizes it. There is no new full-corpus performance claim in this note.

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

The runner's explicit per-process timeout is a failure guard, not a speed result
or successful fallback. Interrupted runs retain journals; a hard-kill stale lock
must be checked before removal. Ctrl-C is not an instruction to silently skip a
slow original. Old official results are not recomputed by this runner.

Before claiming better routing, inspect budget-deferred candidates and perform
independent direct-candidate audits on preselected samples. The retained-baseline
size guarantee alone cannot reveal missed opportunities. Do not tune heuristics
on the eventual full-corpus result and call the same set a fresh holdout.
