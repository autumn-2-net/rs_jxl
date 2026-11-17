# Multiframe Rate Feedback, 2026-11-17

Baseline: `53f67154873f5fe4888456f4ab32df314dcd2bcc`, committed and pushed to
`origin/main` before this work. The earlier decoder candidate and its mixed
performance results are preserved in that commit. This round changes only
automatic multiframe encoding decisions; it introduces no JXL syntax or decoder
change.

Follow-up API change: `adaptive_rate_feedback=true` enables this search and remains
the automatic encoder's default. Setting it to false restores the original
first-frame rate policy; see [automatic frames](MULTIFRAME_AUTO.md). The
measurements below describe the original candidate at `0ce4cd7`, before the
explicit policy selector and subsequent speed work.

## Change

The adaptive rectangle partitioner previously used the first full frame's
measured bytes per pixel for the entire sequence. A very simple opening frame
can make large rectangles look cheap even after a switch to complex imagery.

The encoder now also remembers the most recent **winning complete Full** frame's
rate. If the rates differ by at least 2x, each live reference may propose one
additional base-grid partition. The original anchor and extra-grid proposals
remain first in canonical order. Identical source/rectangle plans are deduplicated;
one-rectangle proposals are already covered by crop/full candidates. Only actual
complete encoded bytes select the winner. Estimates never justify pruning.

Only the winning Full updates the rate, for the next input. Learning from every
completed Full would make future candidates depend on whether a losing trial
was pruned. That initial approach was corrected before final testing/builds.
Existing trial/pixel budgets, payload reuse, reference-slot behavior and patch
tools apply without changes. `adaptive_rectangles=false` disables the extra
proposal with the rest of adaptive partitioning.

For identical live reference states all old choices remain available. Retained
reference policies can choose different future states, so this is not a claim
of a global sequence optimum or universal improvement. Static geometry analysis
does not perform the encoding needed to know subsequent winning Full rates;
it retains its caller-specified-rate semantics.

## Fixed Development Inputs

`tools/prepare_multiframe_feedback.py` freezes nine small sequences before the
candidate is benchmarked. No sample is selected by its measured improvement.

- Four existing 256x192 deterministic recurrence/edit controls from
  `scripts/benchmark_multiframe_reuse.py`, seed 20261027.
- Three 128x128 sequences, seed 20261117: flat-to-textured, textured-to-flat,
  and RGBA with hidden RGB. There are 64 separated 3x3 edits, base tile 16 and
  maximum 16 rectangles. These are synthetic development/control cases.
- The existing two-image 638x900 t3 pair, `146810648_p1.jpg` and `p2.jpg`, as a
  real control. Source JPEG pixels are decoded once with the frozen baseline.
- The first photograph with a synthetic flat intro and separated 3x3 edits
  (tile 64, maximum 16 rectangles). This is explicitly a **derived development**
  sequence, not a real animation. Original ICC profiles are preserved in both
  photographic cases; no resizing or color-space conversion is performed.

These are development inputs, not a holdout corpus. JPEG-derived inputs test
pixel losslessness, not byte-reversible JPEG reconstruction. Input files,
expected arrays, profiles and DLLs are hashed. Timings exclude fixture generation,
source IO/decoding and correctness checks.

## Measurement

Both DLLs use `cargo build --offline --release -p rs-jxl-cli -p rs-jxl-ffi`,
rustc 1.97.1, Windows x86_64 / Ryzen 9 7945HX. The frozen before binary is the
preceding decoder V3 build, whose production source matches the baseline commit.

`tools/benchmark_multiframe_feedback.py` uses real public Python Encoder/Decoder
contexts and their actual DLLs. It performs one warmup and three measured calls
per build/case, alternating before/after order. Reported times are medians.
All calls verify full original arrays (including hidden RGB), shapes, types,
frame counts, durations and original ICC bytes outside the timers. Encode time
includes Python packing, FFI, output bytes and search reports; decode time is
reported separately. MPix/s counts all logical full-canvas pixels, not candidate
work or channels. No compilation/tests run concurrently with final timing.

| Sequence | Before bytes | After bytes | Reduction | Encode MPix/s before -> after | Decode MPix/s before -> after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic: revisit first | 450,868 | 450,868 | -0.000% | 0.355 -> 0.369 | 10.121 -> 11.204 |
| Synthetic: revisit middle | 600,822 | 600,822 | -0.000% | 0.308 -> 0.238 | 6.564 -> 7.083 |
| Synthetic: repeated crop | 376,207 | 376,207 | -0.000% | 0.230 -> 0.231 | 11.709 -> 13.565 |
| Synthetic: subtile changes | 164,206 | 164,206 | -0.000% | 0.239 -> 0.245 | 23.765 -> 23.716 |
| Synthetic: flat to textured | 91,283 | 69,601 | 23.753% | 0.311 -> 0.238 | 12.530 -> 8.624 |
| Synthetic: textured to flat | 50,594 | 50,594 | -0.000% | 0.506 -> 0.514 | 15.032 -> 15.113 |
| Synthetic: hidden RGB transition | 91,280 | 69,562 | 23.793% | 0.259 -> 0.206 | 10.509 -> 7.591 |
| Real control: t3 pair | 371,056 | 371,056 | -0.000% | 0.551 -> 0.564 | 37.389 -> 37.432 |
| Derived: photo intro + edits | 313,748 | 240,984 | 23.192% | 0.905 -> 0.717 | 54.441 -> 16.218 |

All nine complete streams total **2,510,064 -> 2,393,900 bytes (-4.628%)**.
Aggregate throughput (total logical pixels divided by summed per-case median
times) is **0.521 -> 0.476 MPix/s encoding** and **25.593 -> 17.263 MPix/s
decoding**. This mixed-set aggregate is not a representative corpus estimate.

The two targeted synthetic transitions total 182,563 -> 139,163 bytes (-23.773%).
The derived photographic case saves 72,764 bytes (-23.192%). All six controls
have identical before/after sizes, including the real t3 pair; there is no claim
of a gain on that original real sequence. Small timing differences on unchanged
controls are observations from three trials, not established speed changes.

The gains come with a substantial decode cost. Each affected final logical
frame changes from one CropPrevious layer to 16 MergedPrevious layers. For the
photo, that frame shrinks from 150,776 to 78,012 bytes and its submitted area
falls from 483,465 to 258,192 pixels. The full sequence is smaller, but each layer
still incurs decoding/composition work: throughput drops from 54.441 to 16.218
MPix/s (elapsed time grows about 3.36x). Encoding throughput drops from 0.905 to
0.717 MPix/s (about 1.26x elapsed time). This is a size improvement, not a speed
optimization. The RGBA fixture retains every hidden RGB sample.

The feature remains part of the existing explicit byte-selected `strategy="auto"`
with adaptive rectangles. Geometry-only `plan`/`strategy="crop"` is unchanged.
For workloads sensitive to layer cost, the existing `max_rectangles`,
`allow_multi_rect` and `adaptive_rectangles` controls remain available; their
size/speed tradeoffs must be measured separately. No separate temporal-copy or
residual coding tool was introduced in this round.

## Verification

- 38 multiframe unit tests and 60 integration tests passed. These cover
  automatic selection, analysis, pruning, multiple grids, historical references,
  lookahead and generated patch-atlas coexistence.
- The new focused integration test independently encodes every original static
  geometry candidate as a complete sequence and proves the feedback output is
  smaller. Pruning on/off keeps the same output. Other new checks preserve
  hidden RGBA, F32 signed zeros/NaN payloads/infinities, Gray31 with Depth16 extras,
  durations and explicit trial/pixel limits.
- All 72 final benchmark encode/decode calls passed exact native output and ICC
  checks. Encoded sizes were stable across warmups/repetitions, and source/DLL
  hashes were unchanged. The prior baseline/baseline harness self-check also
  passed all nine cases; it is not performance evidence.
- Official **djxl 0.12.0 a1ebdf2** independently decoded all 18 saved before/after
  streams to NPY: **60 original RGB8/RGBA8 frames matched exactly**, including
  hidden RGB. Integer words were recovered from its normalized f32 output; this
  does not extend the independent claim to arbitrary wide integers or F32.
  Native checks separately validate original ICC and frame durations.
- Rust 1.89.0 check, release CLI/FFI build, changed-file Rustfmt, Python syntax,
  and `git diff --check` passed. Artifacts include `unit-tests.log`,
  `integration-tests.log`, `msrv-check.log`, `paired.json`, `summary.json`,
  `official-oracle.json`, and the local NPY check script.

## Reproduction

Local inputs/reports are under `out/multiframe-rate-20261117/`; these binaries,
arrays, profiles and encoded streams are not committed. Use a new output folder
for preparation and new report/stream paths for every benchmark invocation.

```powershell
python tools/prepare_multiframe_feedback.py --before out/multiframe-rate-20261117/before.dll --real-dir D:/cpp_pj/jxl_python/test_img/mt_lay/t3 --out out/multiframe-rate-20261117/inputs
python tools/benchmark_multiframe_feedback.py --before out/multiframe-rate-20261117/before.dll --after out/multiframe-rate-20261117/after.dll --manifest out/multiframe-rate-20261117/inputs/manifest.json --threads 4 --effort 9 --out out/multiframe-rate-20261117/paired.json --streams-dir out/multiframe-rate-20261117/streams
```

The manifest and JSON report retain exact options, identities, timings, complete
sizes and native candidate reports. The first full encoded stream of each
build/case is saved for independent decoder checks.
