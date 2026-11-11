# Lossless Kernel Follow-Up

## Scope

Follow-up to [Chain Batch scheduling](CHAIN_BATCH_FEEDBACK_20261111.md).
These experiments target the remaining native encode kernels, not search-budget
reductions, decoder changes or a new codestream format. Ordinary Chain remains
the default; the existing opt-in Batch experiment is kept separate.

Three exact implementation changes are retained in this working tree:

- Weighted predictor scoring: replace the per-sample binary search over 33
  fixed context cutoffs with a 1,002-byte compile-time lookup. Clamping to
  `[-500, 501]` preserves both unbounded buckets and all signed-i32 inputs.
- MA cached histogram subtraction: when all 14 predictors are independently
  active in identity order, bind their histogram/column references once.
  Keep record-major order, checked subtraction, error order and partial state.
  Aliased or reordered predictors retain the generic implementation.
- Mutable LZ77 Chain: store the masked 15-bit hash tags in `u16` rather than
  `u32`. `u16::MAX` is outside the hash domain and remains the invalid tag.
  This saves `2 * window` bytes, or 2 MiB at the full 2^20-entry window.
  Match lengths, link positions, distance rules and visit budgets stay unchanged.

## Validation Contract

The frozen baseline already includes the previous Batch scheduling/backoff work:

`982c1f3ec5d92598fe3ab5545168141cb9b185acd72136575fc449ba6ff22050`

Development host: Windows x86_64, Rust 1.97.1, release build with thin LTO.
Use alternating baseline/candidate runs, one verified warmup per variant,
and medians of three measured repetitions. Separate profiled calls are excluded
from timings. Native encode timings exclude CLI startup and PNG/file I/O.
Inclusive profile scopes can overlap across threads and must not be summed as
wall-clock times.

Every produced JXL is decoded by the caller-provided official libjxl 0.12
`djxl`, and compared with original PNG samples including depth and hidden RGB.
PNG metadata is removed for this pixel-only kernel benchmark, with sample
equality checked before encoding. No encoded JXL hash equality is required.
The decoder SHA256 is:

`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`

Local reports, the frozen baseline and the paired-run driver are in the ignored
`out/hotspot-followup-20261111/` directory. These are targeted fixed-family
measurements, not a full-corpus or full outer-scheduler claim. A process guard
fails a benchmark on timeout; it never changes the encoder's search.

Candidate encoder SHA256:

`a053ade8f627e8952c490167841c9d4e170e7e95bca698fb8148be52d28dc1c3`

## Results

Important limitation: an unrelated eight-worker compression task was running
on this host. It was not stopped. Timing ranges overlap, and these observations
are not an idle-host speed guarantee. Both initial results and repeated checks
are kept below, including slowdowns. All measurements use e9. `automatic` means
the normal fixed effort policy, not the outer candidate tournament. The T8
cross6 row explicitly uses Batch; the T1 and literal rows use ordinary Chain.

| Input / route | Threads | Repeats | Before s | After s | Time change | Before MPix/s | After MPix/s | JXL bytes, both | bpp, both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1eb / cross6 + Batch | 8 | 3 | 3.982346 | 3.921380 | -1.53% | 0.9045 | 0.9185 | 351302 | 0.780255 |
| 1eb / cross6 | 1 | 3 | 10.157904 | 9.696352 | -4.54% | 0.3546 | 0.3715 | 351302 | 0.780255 |
| 1eb / literal, initial | 8 | 3 | 3.459809 | 3.610715 | +4.36% | 1.0411 | 0.9976 | 417020 | 0.926217 |
| 1eb / literal, repeat | 8 | 7 | 3.473853 | 3.458612 | -0.44% | 1.0369 | 1.0414 | 417020 | 0.926217 |
| 9bde / automatic | 8 | 3 | 1.497199 | 1.368244 | -8.61% | 0.9726 | 1.0642 | 412539 | 2.266499 |
| wallhaven-n6qzmx / automatic | 8 | 3 | 2.710507 | 2.570681 | -5.16% | 0.7650 | 0.8066 | 459153 | 1.771424 |
| kodim01 / automatic, initial | 8 | 3 | 0.761454 | 0.781410 | +2.62% | 0.5164 | 0.5032 | 454216 | 9.241048 |
| kodim01 / automatic, repeat | 8 | 7 | 0.789084 | 0.801864 | +1.62% | 0.4983 | 0.4904 | 454216 | 9.241048 |
| 2026.6test1 / automatic, initial | 8 | 3 | 1.580668 | 1.667212 | +5.48% | 1.3119 | 1.2438 | 2098189 | 8.094865 |
| 2026.6test1 / automatic, repeat | 8 | 7 | 1.736022 | 1.686934 | -2.83% | 1.1945 | 1.2292 | 2098189 | 8.094865 |
| DJI-FC6310-16bit_709_v4_krita / automatic | 8 | 3 | 0.039036 | 0.038688 | -0.89% | 0.1049 | 0.1059 | 16825 | 32.861328 |

`1eb` and `9bde` refer to the full original filenames in each JSON report.
The 16-bit case is a 64x64 correctness probe, not a representative throughput
sample. Byte counts differ slightly from earlier reports that retained PNG
metadata; compare versions within this experiment, not across differently
prepared inputs.

The separate cross6 diagnostic runs show where the changes plausibly help:

| Inclusive diagnostic scope | T8 before s | T8 after s | T1 before s | T1 after s |
|---|---:|---:|---:|---:|
| Weighted header scoring | 1.049248 | 0.842318 | 0.976356 | 0.794352 |
| MA cached subtraction | 0.418128 | 0.319806 | 0.360306 | 0.280254 |
| MA learning | 1.499285 | 1.399130 | 3.143201 | 3.008263 |
| LZ77 shared parse | 1.615395 | 1.564609 | 4.320110 | 3.830671 |

These are one diagnostic call per variant, not repeated kernel benchmarks or
an ablation separating every change. Weighted scoring drops about 19-20% and
MA subtraction about 22-24% in these observations, but this is not a claim of
20% complete-encode acceleration. Ordinary literal encoding has no convincing
speed gain. The narrowed tags have a proven allocation reduction, not a proven
universal speed gain. The Kodak slowdown remains unresolved under this load;
do not claim no performance regressions or universal improvement.

## Checks And Reproduction

- 26 cost tests passed; one opt-in historical microbenchmark was not run.
- 5 new MA subtraction tests passed, including packed/wide property columns,
  borrowed sample views, arbitrary weights, histogram tails, active aliases,
  and every predictor's checked token/extra-bit failure and partial state.
- 43 matcher tests passed, including an old-u32-tag/scalar-extension oracle,
  forced hash collisions, circular-window reuse, cache on/off and visit order.
- Release encoder build, `cargo check --workspace`, format and diff checks pass.
- 134 official decoder/sample comparisons passed across **six distinct inputs**.
  This count includes diagnostic and warmup encodes, not 134 distinct images.
- All six JSON reports passed; all temporary encoded/decoded files were cleaned.
  Observed output sizes were unchanged. No full corpus, Linux runtime or JPEG
  byte-transcode campaign was performed in this narrowly scoped follow-up.

Local paired-run driver (preserved beside the reports):

```powershell
python -B out/hotspot-followup-20261111/compare.py `
  --before out/hotspot-followup-20261111/encoder-lab-before.exe `
  --after target/release/examples/encoder_lab.exe `
  --decoder <official-djxl.exe> --case <original.png> `
  --family cross6 --threads 8 --batch --repeats 3 --out <new-report.json>
```

Omit `--batch` for the ordinary Chain route; use `--family automatic` for the
normal fixed policy. The script writes only a new report plus temporary files,
records binary/input provenance, verifies all outputs and fails on errors.

The retained changes do not introduce new search knobs, reduce MA samples or
change candidate budgets. Keep the existing Batch default unchanged. An idle
host rerun is needed before drawing small whole-image speed conclusions or
promoting this as a broad performance release.
