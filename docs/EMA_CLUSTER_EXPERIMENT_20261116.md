# E-means-inspired histogram clustering experiment

## Scope

Base: `8336aee965641567f0c7a29729633edc786517ee`. Ordinary effort presets
are unchanged. This is an explicit native Rust experiment, not a Faiss
dependency, not a neural model, and not a new JXL decoding rule.

For the later 32-pass, alternate-distance and native-cost relocation tests,
see [the distance follow-up](EMA_DISTANCE_20261116.md). Results below describe
the original cross-entropy experiment without those optional controls.

The starting reference is [e-means](https://arxiv.org/abs/2611.09501),
published November 6, 2026. Its mini-batch EMA optimizer targets Euclidean
k-means. Our existing histogram clusterer is farthest-first seeding, one
sequential assignment pass, and greedy ANS population-cost merging. Therefore
the paper's speedups against multi-epoch Lloyd are not predictions for JXL.

## Adaptation

`entropy/cluster_emeans.rs` implements a categorical, population-weighted
variant. Assignment minimizes histogram cross-entropy against smoothed
centroid probabilities. EMA accumulates symbol populations, not unweighted
normalized vectors. Cosine steps, last-pass labels, and periodic donor
splitting borrow the optimization structure; original integer counts are
regrouped exactly and passed to the existing ANS merge/refinement code.

The [official implementation's recipe](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)
provides the batch and step scaling coefficients used as initial heuristics.
Those empirical laws were fitted to vector datasets, not our weighted
categorical objective. They need local validation. Initialization is either
deterministic shuffled input selection or the existing farthest-point seeder.
The latter may choose fewer centers because of its existing distinctness
threshold. Thus these are whole-pipeline comparisons, not a controlled
initialization-only ablation with fixed K.

Final labels are from the last online pass, as in the reference's last-epoch
mean finalization. There is no extra nearest-final-center sweep. Smoothing,
EMA masses, and split perturbations are never serialized or used as pixels.

Actual Prefix models, preexisting/frozen coding tables, alphabets above 256,
and fewer than 32 nonempty contexts retain Best clustering. A single-context
model keeps its preexisting direct-table shortcut and never calls clustering.
These are experiment eligibility conditions, not new input-format limits.

Batch assignments run against frozen centroids using the caller's pool and
existing nested-work limits. Ordered integer accumulation and center updates
follow each batch. No new global worker pool, GPU runtime, or unsafe kernel.

## Controls

Rust: set `EntropyOptions.clustering` to
`ClusterStrategy::Emeans(emeans::Options { passes: 3, initialization:
emeans::Initialization::Random, ..Default::default() })`. Zero passes are rejected. The default
remains `ClusterStrategy::Best`; no preset selects Emeans.

Development example:

```text
cargo build --release -p rs-jxl-cli --example portfolio_lab
portfolio_lab INPUT.png --output OUTPUT.jxl --effort 9 --threads 8 --direct-candidate baseline --ema-passes 3
```

Add `--ema-farthest` for existing farthest-point seeds. Omit both flags for
the control. `ema_work` in the JSON receipt reports executed proposals,
fallbacks, context counts, initial/occupied center totals, and parallel batches.
These counters are process-cumulative; the example performs one encode.
Single-context ANS bypasses are excluded from these counters; the Prefix
gate records a fallback before its direct-table shortcut.

The paired harness accepts `--ema NAME=PASSES:random|farthest` for named
variants. It rotates execution order, records requested settings, checks the
native receipt, and verifies every result with independent official djxl.
Timing excludes image I/O, process startup and verification. Use separate
profiling runs: inclusive nested/parallel stage sums are not wall-time shares.

```text
python -B tools/benchmark_ma_kernels.py --manifest CASES.json --variant base=portfolio_lab --variant ema3=portfolio_lab --ema ema3=3:random --decoder djxl --effort 9 --threads 1 8 --repeats 2 --profile off --out RUN
```

## Acceptance

The invariant is exact decoded samples, including hidden RGB and 16-bit data,
not identical JXL bytes. Compare complete output bytes, arithmetic per-image
bpp, pixel-weighted bpp, native seconds and MPix/s. A changed partition can
improve its proxy while making the full stream larger; no per-image baseline
encode is hidden inside the experiment. Default adoption requires measured
compression/time benefits, not lower vector MSE.

## Mixed-image results

Sixteen full-resolution images, e9, eight threads, one interleaved pass per
variant. The set covers photographs, manga, screenshots, UI, transparent
assets, palette/pixel art, displacement textures, and two large 16-bit HDR
PNG inputs. Timings are exploratory point estimates, not confidence intervals.

| Method | Total bytes | Mean image bpp | Weighted bpp | Native seconds | MPix/s | Size change | Time change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary Best | 81,634,536 | 8.004119 | 8.093807 | 58.1845 | 1.3868 | baseline | baseline |
| EMA 1, random | 82,567,881 | 8.056854 | 8.186346 | 52.1968 | 1.5458 | +1.1433% | -10.2908% |
| EMA 3, random | 81,778,595 | 8.014762 | 8.108090 | 52.8584 | 1.5265 | +0.1765% | -9.1538% |
| EMA 3, farthest | 81,963,675 | 8.030435 | 8.126440 | 52.7606 | 1.5293 | +0.4032% | -9.3219% |

EMA ran on 14 images; the palette fixture and displacement image did not
enter the experimental path. Their timing differences are noise/control
variation, not EMA gains. Random EMA 3 made one file smaller by 25 bytes,
13 files larger, and two unchanged. The speed benefit is concentrated in the
two HDR inputs. Several UI/document images grow by about 1.5-3.7%; aggregate
bytes conceal these relative regressions.

| EMA 3 case | Size change | Native baseline | Native EMA |
|---|---:|---:|---:|
| 1515 HDR, 3024 x 4032, 16-bit | +0.1270% | 17.1403 s | 12.6632 s |
| 1244 HDR, 3000 x 4000, 16-bit | +0.1559% | 13.8545 s | 12.3030 s |
| Phobos long webpage | +1.5153% | 16.8810 s | 16.9097 s |
| Paper screenshot | +2.1858% | 1.2276 s | 1.3783 s |
| Sublime screenshot | +2.0661% | 2.3859 s | 2.7481 s |

## Iteration and thread check

A separate four-image set uses 1515 HDR, DIV2K 0446, the editor screenshot,
and the paper screenshot. It tests one and eight threads and extends the
iteration count to 12. These are the same full images, not reduced crops.

| Threads | Method | Total bytes | Mean image bpp | Native seconds | MPix/s | Time vs same-thread baseline |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Best | 30,465,219 | 7.679473 | 134.1864 | 0.1383 | baseline |
| 1 | EMA 3 | 30,509,363 | 7.698506 | 108.7538 | 0.1706 | -18.9532% |
| 1 | EMA 12 | 30,490,129 | 7.688959 | 136.7907 | 0.1356 | +1.9408% |
| 8 | Best | 30,465,219 | 7.679473 | 20.1733 | 0.9197 | baseline |
| 8 | EMA 3 | 30,509,363 | 7.698506 | 18.0948 | 1.0253 | -10.3034% |
| 8 | EMA 12 | 30,490,129 | 7.688959 | 20.6069 | 0.9003 | +2.1496% |

EMA 3 grows total bytes by 0.1449%; EMA 12 grows them by 0.0818%.
On HDR 1515, 12 passes save only 13,429 bytes against three passes while
single-thread encode time rises from 90.9362 to 117.6617 seconds. Ordinary
Best is 117.8539 seconds and still 19,750 bytes smaller than EMA 12.
Increasing iterations therefore does not resolve the overall tradeoff.

The repeated eight-thread HDR comparison also shows timing variability:
Best/EMA 3 were 17.1403/12.6632 seconds in the mixed run and 17.0254/14.4560
seconds in the thread check. Do not interpret sub-percent timing changes as
proven effects. Both runs support a benefit on this target, not a universal
speedup.

## Stage diagnostics

A separate eight-thread run profiles 1515 HDR and the paper screenshot.
These numbers are sums of elapsed time across stage invocations; parallel
invocations overlap, and parent/child stages are inclusive. They are not CPU
seconds or percentages of end-to-end wall time. Do not add the columns.

| Input / method | Encode frames elapsed | Cluster accumulated | Seed accumulated | Assignment accumulated | EMA accumulated | ANS merge refinement accumulated |
|---|---:|---:|---:|---:|---:|---:|
| HDR / Best | 17.3921 s | 50.0938 s | 1.0294 s | 0.8348 s | n/a | 48.1462 s |
| HDR / EMA 3 | 14.2350 s | 31.3321 s | 0.0099 s | 0.0003 s | 4.2745 s | 26.9438 s |
| Paper / Best | 1.3398 s | 0.3191 s | 0.0436 s | 0.0477 s | n/a | 0.2219 s |
| Paper / EMA 3 | 1.5106 s | 1.1259 s | 0.0012 s | 0.0000 s | 0.8957 s | 0.2194 s |

The original seed/assignment columns in EMA runs cover fallback streams;
EMA includes its own seed and assignment work. HDR acceleration is mainly
associated with cheaper downstream merge refinement, not replacing an
expensive iterative Lloyd loop. The ordinary implementation has only one
assignment sweep. On paper, original seed/assignment are already cheap and
EMA adds work without reducing refinement meaningfully. This is why simply
porting the paper's headline speedup is not a reliable prediction.

## Verification

- `cargo build --release -p rs-jxl-cli --example portfolio_lab`: passed.
- `cargo test -p rs-jxl --lib entropy:: -- --test-threads=4`: 185 passed.
- `python -B -m unittest discover -s tools -p test_benchmark_ema.py -v`: five passed.
- All 92 full-image outputs passed independent official djxl sample comparison.

Unit coverage includes exact integer regrouping, exercised parallel batch
assignment, serial-scope suppression, empty/sparse/extreme populations,
invalid settings, and actual Prefix/incremental model fallback. The test
suite does not require new experimental streams to match baseline JXL bytes.
Byte comparisons are used only where an explicitly unchanged fallback path
or an equivalent execution order is the behavior being tested.

## Decision and artifacts

Keep this adaptation opt-in. It does not improve overall compression and
does not justify replacing default Best clustering. The useful observation
is narrower: some histogram-heavy HDR streams can trade a small size increase
for less clustering work. That is not evidence that vector k-means losses
predict complete JXL size, nor that e-means itself fails on its original task.

Artifacts are under the sibling workspace's ignored
`out/emeans-20261116/`: input manifests, frozen executable hashes, per-trial
JSONL, analysis JSON, and separate stage diagnostics. `paired16-t8` contains
64 official-decoder pixel checks; `depth4-t1-t8` contains another 24 and
`profile2-t8` another four, for 92 successful checks in total. No corpus-wide
run or default-policy change was made.
