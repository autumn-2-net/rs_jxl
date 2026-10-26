# MA tree execution experiments, 2026-10-26

## Scope and acceptance

This continues `MA_SWEEP_KERNELS_20261026.md`. The baseline already includes
the preceding small-population SIMD changes; it is not pristine upstream C++
or Git HEAD. All new MA execution switches are default-off. No learning sample,
tree leaf, or predictor ceiling was reduced to manufacture a speed improvement.

Acceptance is exact original PNG samples after decoding with official libjxl
0.12 `djxl`, including invisible RGB and 16-bit samples. Identical JXL bytes
are not an acceptance condition. Size and bpp are measured from actual output.

Machine: Ryzen 9 7945HX, 16 encoder workers. Cases run serially. Other programs
were active on this shared workstation; small timing differences are not a
reliable speed claim. Encoder-call timing includes the complete requested
pilot/search and excludes CLI startup, PNG input decoding, and output I/O.
Profiler runs are diagnostic only, not the uninstrumented timing benchmark.

## Variants

Base arguments:

```text
--search --content-pilot --pilot-parallel-initial --pilot-overlap-automatic
--pilot-near-tie-consensus --subtrees --ma-workers 16
--concat-batches --fast-ranges
```

| Variant | Additional arguments | What changes |
| --- | --- | --- |
| baseline | none | Existing detached-subtree execution |
| dynamic | `--dynamic-subtrees` | Live fork/join after a real split |
| handoff | `--subtree-histograms` | Transfer exact cache/parent statistics into child jobs |
| dynamic_handoff | Both preceding flags | Live branches with cache inheritance |
| refine | `--refine-quantiles` | Existing coarse-to-fine threshold selection |
| hist64 | `--ma-histogram-bins 64` | Existing node-local coarsened histograms |

`handoff` also permits parent/cache reuse for balanced child splits. It is not
just a queue ownership change. Deferred sibling subtraction is completed before
rebasing a child sample view. Invalid/redundant cache entries are not reused.
The inherited cache retains its original shared allocation lease. A failed or
panicking normalization drops the partial cache immediately. Explicit bounded
learning retains its serial leaf-budget commit order.

The approximate variants may change learned trees and file sizes, but still
must reconstruct the original samples exactly. Coarse histograms disable the
ordinary histogram cache, so fewer thresholds do not automatically mean less
total work.

## First screening, r0

Frozen binary: `out/ma-tree-experiments-20261026/experiment-r0-lab.exe`.
SHA256: `f9eb58c528f44b9eef64ed4d854fcfadc47e7104280f32bb416d37eafca94d2a`.
Raw report: `out/ma-tree-experiments-20261026/screen-r0/report.json`.
One measured round, no warmup: 15 images x 6 variants = 90 pixel checks passed.
This is screening, not a repeated performance acceptance result.

| Image | Variant | Seconds | MPix/s | Bytes | bpp |
| --- | --- | ---: | ---: | ---: | ---: |
| floral, 4278x4222 | baseline | 22.780 | 0.793 | 5100273 | 2.259042 |
| floral | dynamic | 29.019 | 0.622 | 5100273 | 2.259042 |
| floral | handoff | 25.923 | 0.697 | 5100273 | 2.259042 |
| floral | dynamic_handoff | 26.440 | 0.683 | 5100273 | 2.259042 |
| floral | refine | 24.855 | 0.727 | 5100320 | 2.259063 |
| floral | hist64 | 24.497 | 0.737 | 5100228 | 2.259023 |
| pizza, 3600x3562 | baseline | 42.289 | 0.303 | 17818198 | 11.116226 |
| pizza | dynamic | 49.036 | 0.262 | 17818198 | 11.116226 |
| pizza | handoff | 35.880 | 0.357 | 17818198 | 11.116226 |
| pizza | dynamic_handoff | 46.339 | 0.277 | 17818198 | 11.116226 |
| pizza | refine | 40.609 | 0.316 | 17817936 | 11.116062 |
| pizza | hist64 | 40.487 | 0.317 | 17813890 | 11.113538 |
| wallhaven-mlzoy1, 3840x2160 | baseline | 30.994 | 0.268 | 14178604 | 13.675351 |
| wallhaven-mlzoy1 | dynamic | 46.895 | 0.177 | 14178604 | 13.675351 |
| wallhaven-mlzoy1 | handoff | 30.996 | 0.268 | 14178604 | 13.675351 |
| wallhaven-mlzoy1 | dynamic_handoff | 40.544 | 0.205 | 14178604 | 13.675351 |
| wallhaven-mlzoy1 | refine | 29.360 | 0.283 | 14178544 | 13.675293 |
| wallhaven-mlzoy1 | hist64 | 29.890 | 0.278 | 14175555 | 13.672410 |

The 12 other images cover icons, textures, Urban100, manga, a natural DIV2K
image, screenshot, scanned document, transparency, and 16-bit input. Their
combined 23.555076 MPix results (sum of per-image measured times, not an average
of rates):

| Variant | Seconds | MPix/s | Total bytes | Aggregate bpp |
| --- | ---: | ---: | ---: | ---: |
| baseline | 15.074987 | 1.562527 | 12590955 | 4.276260 |
| dynamic | 16.236744 | 1.450727 | 12590955 | 4.276260 |
| handoff | 14.636949 | 1.609289 | 12590955 | 4.276260 |
| dynamic_handoff | 15.717104 | 1.498691 | 12590955 | 4.276260 |
| refine | 14.805386 | 1.590980 | 12591002 | 4.276276 |
| hist64 | 15.396016 | 1.529946 | 12592029 | 4.276625 |

## Why floral is not evidence of a speed floor

Diagnostic report:
`out/ma-tree-experiments-20261026/profile-floral-r0/report.json`.
All six profiled outputs also passed independent pixel checks.

All six variants performed 27 probes and four full candidates, with the same
winning cross-channel RCT17 family. The slowdown was not a changed finalist
selection. `ma.find_best_split` call counts were identical for the four exact
execution variants: 569248.

| r0 variant | Property-cache collection calls | Parent-histogram inclusive seconds | Total encode seconds, profiling on |
| --- | ---: | ---: | ---: |
| baseline | 17640 | 2.545 | 26.571 |
| dynamic | 32600 | 7.298 | 31.959 |
| handoff | 13392 | 1.831 | 24.642 |
| dynamic_handoff | 13080 | 1.691 | 30.366 |

The dynamic-only experiment discards reusable state when splitting earlier,
nearly doubling cache collection calls. Restoring that state removes this
extra work, but does not by itself remove scheduling/waiting overhead. This is
an execution-strategy issue, not proof that MA cannot run faster.

The profiled handoff run was faster on floral, unlike the first uninstrumented
screening. Therefore the initial floral regression must not be presented as a
stable intrinsic property of the image. Repeated uninstrumented measurements
are required. The full winning encode was about 6.5-7.0 seconds (2.6-2.8 MPix/s)
inside these instrumented searches; this is not the speed of the complete
search, which also pays for its other candidates.

Profiler scopes are inclusive. In particular, recursive `ma.train_nodes` and
`ma.dynamic_fork` totals count overlapping parent/child and parallel intervals.
They are neither CPU time nor elapsed time and must not be summed or used as
an elapsed-time percentage. Call counts and per-full-candidate elapsed times
provide separate evidence. No automatic hotspot clustering selected these
experimental switches.

## r1 execution correction

The follow-up version returns a live fork's extra slot when the first child
finishes, instead of retaining it until both children finish. When several
independent branches are already active, their inner property sweeps execute
locally instead of submitting another nested property batch. Histogram handoff
has separate diagnostic scopes. The subtree diagnostic now distinguishes a
shared cache budget from per-task budgets instead of reporting both as a
per-task sum.

These changes remain opt-in and are not promoted based on r0 measurements.

### Uninstrumented confirmation

Frozen r1 binary SHA256:
`e3039be513f5d8c52fd413641618f415b45f0f5034aa325c93bdd2521441b60a`.
Report: `out/ma-tree-experiments-20261026/confirm-r1/report.json`.
Two measured rounds in forward/reverse variant order, no warmups. All 12
outputs passed official pixel validation.

| Image | Variant | Median seconds | MPix/s | Observed seconds range |
| --- | --- | ---: | ---: | --- |
| floral | baseline | 24.438928 | 0.739055 | 24.174923-24.702932 |
| floral | handoff | 23.994853 | 0.752733 | 23.937599-24.052107 |
| floral | dynamic_handoff | 25.220111 | 0.716163 | 24.701142-25.739079 |
| pizza | baseline | 42.829720 | 0.299400 | 40.413696-45.245743 |
| pizza | handoff | 41.354153 | 0.310083 | 40.530728-42.177578 |
| pizza | dynamic_handoff | 47.415039 | 0.270446 | 46.527317-48.302761 |

File sizes/bpp remained 5100273/2.259042 for floral and
17818198/11.116226 for pizza. Handoff's median elapsed reductions were only
1.82% and 3.45%; with two observations and this variability, neither establishes
a robust general speedup. In particular, the earlier one-pass pizza result
(42.3 to 35.9 seconds) did not reproduce. Dynamic handoff still regressed.
Do not enable either new switch by default or present the screening win as a
stable 15% improvement. The r1 scheduling correction improved the floral
dynamic experiment relative to r0, but did not beat the existing execution path.

## Validation

- MA learner module: 212 tests passed, including live branch, handoff, bounded
  budget, serial runner, underflow, unwind, packed/wide samples, and shared leases.
- Benchmark harness: 13 mocked tests passed using the configured Python.
- Encoder-lab CLI and input tests: 47 passed.
- Independent official decoding: 112 outputs passed across screening, diagnostic,
  confirmation, the additional wallhaven e9 run, and the three e10 runs.
- Harness now accepts `--effort 1..10` (default 9), records it, and passes it
  to every encode; per-variant effort overrides remain rejected.

## e10 interpretation

In this pilot scheduler, e10 retains the e9 candidates. It also relaxes pilot
eligibility/consensus rejection, admitting rivals that e9 can skip, then tries
deeper versions of the winner and automatic candidate when not already present.
The e9/e10 source MA sampling fraction, property count, and 256-bin quantizer
are the same; the base split threshold changes from 89 to 75, permitting more
splits when the measured gain justifies them.
The source-compatible backend also changes automatic frame routing: tier zero
(effort 10) uses shared full-frame coding instead of the eligible e9 streaming
frame-area route. This can increase learning cost substantially. It is not
simply the same tree search with a little more CPU time.

### Measured e9 versus e10

Reports: `out/ma-tree-experiments-20261026/e10-r1/report.json` and
`out/ma-tree-experiments-20261026/wallhaven-e9-r1/report.json`, plus the e9
baseline rows of `confirm-r1/report.json`. Both efforts use the same frozen r1
binary, 16 workers, and existing baseline execution: the new dynamic/handoff
switches are OFF. e10 and the wallhaven e9 observation were each measured once;
the other e9 values are the two-round medians above. These are not statistical
confidence estimates. All outputs reconstructed exact original pixels.

| Image | Effort | Bytes | bpp | Seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| floral | 9 | 5100273 | 2.259042 | 24.438928 | 0.739055 |
| floral | 10 | 4623627 | 2.047924 | 123.788741 | 0.145908 |
| pizza | 9 | 17818198 | 11.116226 | 42.829720 | 0.299400 |
| pizza | 10 | 17107634 | 10.672927 | 146.083591 | 0.087780 |
| wallhaven-mlzoy1 | 9 | 14178604 | 13.675351 | 31.698346 | 0.261667 |
| wallhaven-mlzoy1 | 10 | 13676431 | 13.191002 | 61.996009 | 0.133789 |

e10 saved 9.35%, 3.99%, and 3.54% respectively, for observed elapsed-time
multipliers of 5.07x, 3.41x, and 1.96x. It has real compression benefits, but
these examples do not justify describing it as cheap. These measurements cover
pixel-lossless PNG input, not JPEG byte-reconstruction or multiframe workloads.

Floral went from four to eleven full-frame trial invocations; pizza went from
four to nine. These include trials terminated by a proven byte lower bound,
not just successfully completed outputs. Their probe counts remained 27 and
24. The e10 overhead therefore includes broader finalist exploration as well
as slower backend decisions. Floral's e10 winner was the RCT17 cross-channel
family; pizza and wallhaven selected automatic tier-zero candidates. Do not
attribute the entire search duration to the final winning tree.

## Decision

Keep the new execution strategies experimental. Cache reuse reduces specific
recomputation, but this round has not established a repeatable broad speed win.
Dynamic branch scheduling still loses to the existing path on these cases;
coarse-to-fine and coarse histograms also failed to produce a clear broad win.
The next useful investigation is work/waiting attribution and cache-preserving
task organization, not more parallelism without accounting for duplicated
statistics. No default policy, image-specific exception, or learning limit was
changed based on this small sample.

Source references: `scheduler/scout_pilot.rs`, `modular/source_options.rs`,
`modular/ma_frontier.rs`, `modular/ma_parallel.rs`, and `modular/ma_train.rs`.
