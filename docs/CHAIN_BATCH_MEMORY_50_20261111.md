# Chain Batch Memory: 50-Image Validation

The retained optimization is committed as `dc59497`, compared with `b989bf3`.
Both already enable parallel-only Batch; this is **not** a Batch-on/off comparison.
See [the implementation report](CHAIN_BATCH_MEMORY_20261111.md).

## Scope and Selection

- Fifty distinct original-file hashes from the existing 500-image corpus.
  Excluding the six previously measured images leaves 496 eligible images;
  four of those six occurred in that corpus.
- Proportional stratified sampling, seed `20261111`, selected before timing.
  Ten categories, 108,573,353 pixels, maximum 4096 x 4096. Source PNG depths:
  47 eight-bit, two sixteen-bit and one one-bit palette image.
- Same fixed `automatic` candidate, e9, eight threads, and explicit
  `--chain-batch-parallel` in both frozen `encoder_lab` binaries. **This is not
  an outer-scheduler benchmark**, a JPEG transcode benchmark, or an effort sweep.
- Three alternating timed calls per version, plus one separate diagnostic
  profile per version. Profiles precede timing and are excluded from medians;
  there are no additional warmups. This is not a cold-cache measurement.
- Native encoder time excludes process startup, input decoding, disk IO and
  independent correctness verification. Aggregate seconds below sum per-image
  medians; they are not benchmark wall time. The initial scan took 767.8 seconds.
- Official libjxl 0.12 `djxl` decodes every successful output; the checker compares
  original samples, including hidden RGB. Encoded-byte identity is not required.
- Serial benchmark execution, no concurrent builds. This shared Windows host is
  not an isolated performance machine. Small timing changes need replication.

## Input Exceptions

The untouched first pass completed **48/50** cases and **384** verified outputs.
Its two failures remain recorded; no image was replaced:

1. `Job_Simulator_screenshot_-_Kitchen_02.png`: both frozen encoders reject
   contradictory sRGB/gAMA/cHRM metadata before encoding. A supplementary paired
   pixel benchmark strips metadata chunks without recompressing IDAT, verifies
   the snapshot against original samples, and feeds that identical snapshot to
   both encoders. This does not claim successful metadata preservation or repair
   the original file's conflicting declarations.
2. `s04i3p01.png`: the shared Python reference reader rejects one-bit palette
   PNG. The supplementary reader independently checks expanded RGBA samples with
   Pillow and OpenCV. Both encoders still receive the unmodified original PNG;
   its palette entries are eight-bit color samples, not one-bit RGB values.

Including these explicitly qualified supplements, all **50 pixel cases / 400
outputs** pass official decoding and original-sample comparison. Every recorded
profile and timed output has unchanged file size relative to its counterpart.
The audit also checks executable identities, source identities, the deterministic
sample selection, configuration, repeat counts and temporary-file cleanup.

## First-Pass Results

Paired supplements are included below. All bytes and bpp are weighted totals,
not averages of per-image ratios. Negative time change means faster.

| Group | Images | Before seconds | After seconds | Time change | Before MPix/s | After MPix/s | Before/after bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete pixel sample | 50 | 65.5651 | 64.7972 | -1.17% | 1.6560 | 1.6756 | 4.399508 |
| Observed active Batch | 5 | 8.2494 | 7.9759 | -3.32% | 2.3142 | 2.3936 | 1.255071 |
| Batch bypass | 45 | 57.3157 | 56.8213 | -0.86% | 1.5612 | 1.5748 | 5.070381 |

Output totals are **59,708,665 bytes** on both sides. Without either supplement,
the original 48 successful cases give 64.7370 -> 64.0255 seconds (-1.10%).
No image changes its observed Batch activation between checkpoints. Activation
comes from separate profiles, not assertions that each unprofiled timed execution
had exactly the same thread schedule.

| Category | Images | Time change | After MPix/s | Unchanged bpp |
| --- | ---: | ---: | ---: | ---: |
| Alpha/icon | 5 | -12.07% | 1.1236 | 1.196022 |
| Document/web | 6 | +1.27% | 2.6301 | 1.569687 |
| Format-mixed | 5 | -12.06% | 1.3592 | 3.016316 |
| Game/UI | 4 | -6.94% | 1.3569 | 2.937371 |
| Manga | 4 | +0.07% | 0.8632 | 8.752071 |
| Mixed local | 4 | -4.37% | 2.4126 | 2.074049 |
| High-resolution photos | 8 | +0.92% | 1.1657 | 9.566970 |
| Other photos | 6 | +1.37% | 1.0913 | 8.807011 |
| Textures | 4 | -1.48% | 1.2149 | 9.283792 |
| Urban | 4 | -4.44% | 0.9423 | 9.079408 |

Most images bypass this optimization. Large movements on bypass cases are **not
evidence of Batch speedups**. These categories are descriptive, not statistical
confidence claims or proof of general gains on a category.

## Focused Follow-Up

All five active cases and all four initial material regressions are selected for
five alternating timed calls per checkpoint, with separate profiles and official
pixel checks. A material regression was defined before the scan as at least 5%
slower **and** at least 20 ms slower; it is a triage rule, not a significance test.
Original first-pass medians remain unchanged and are not replaced by retests.

| Index | Image | Active Batch | Before seconds | After seconds | Retest time change |
| ---: | --- | --- | ---: | ---: | ---: |
| 5 | `0505.png` | No | 2.432922 | 2.513762 | +3.32% |
| 11 | `5206.png` | No | 2.082120 | 2.028319 | -2.58% |
| 16 | `012.png` | No | 1.417961 | 1.413864 | -0.29% |
| 23 | `8359.png` | Yes | 0.244266 | 0.236210 | -3.30% |
| 27 | `5011.png` | Yes | 3.121187 | 3.105867 | -0.49% |
| 30 | `UchuKigekiM774.png` | No | 1.042087 | 1.075018 | +3.16% |
| 37 | `imdb.com.png` | Yes | 2.516370 | 2.430242 | -3.42% |
| 40 | `SampleB.png` | Yes | 0.120629 | 0.115921 | -3.90% |
| 43 | `gas_stove_PNG10.png` | Yes | 0.877022 | 0.788901 | -10.05% |

The original four flagged regressions were respectively +6.36%, +8.59%, +17.92%
and +11.68%. None repeats above the prespecified 5%/20ms threshold. Two remain
about 3% slower, so this does not prove zero regression on every input.

All five active cases have lower median time in the follow-up. Their aggregate
is **6.879473 -> 6.677141 seconds (-2.94%)**, or **2.7751 -> 2.8592 MPix/s**.
Their combined bpp remains **1.255071**. Absolute times drift from the first pass
on both checkpoints; only paired within-pass comparisons are meaningful here.

The follow-up contributes **108** verified outputs, for **508** successful pixel
checks in total. All nine follow-up cases retain their output sizes across every
profile and timed call, as well as their first-pass sizes.

## Interpretation

- This is a modest improvement on the broader sample, **not** a general 10%
  encoder speedup: only five of fifty images use the affected Batch path.
- Observed active-case gains repeat at about 3% in aggregate, while bypass-case
  timing moves in both directions. Retaining the localized optimization is
  reasonable; these results do not justify widening Batch admission heuristics.
- The five first-pass active profiles report 294,125,568 -> 52,362,432 logical
  initialized-frontier bytes, about 82.2% less. This sums initialized storage
  across streams, not peak RSS, measured bus traffic or hardware cache counters.
  It does not establish that disk IO or memory bandwidth caused earlier slowdowns.
- Compression and original-pixel correctness are unchanged on this sample.
  No encoder changes, recompilation, new search limits or altered effort policy
  were introduced during this validation.
- The corpus is stratified but still finite and curated. File-hash deduplication
  does not eliminate visual near-duplicates. These Windows T8 tests do not cover
  other hosts, thread counts, platforms or full-scheduler interactions.

## Evidence

Local artifacts are under `out/chain-memory50-20261111/`:

- `eligible.json`, `sample.json`: population, exclusions, seed and fixed sample.
- `case-001.json` through `case-050.json`, corresponding logs, `summary.json`:
  untouched initial results, including both failures.
- `supplement-012-metadata.json`, `supplement-012.json`, `supplement-019.json`:
  qualified supplementary evidence.
- `audit.json`: complete 50-case summaries and independently checked provenance.
- `recheck-NNN.json`, `rechecks.json`: separate follow-up results.
- `run.py`, `finish.py`: local orchestration of the committed checkpoint harness.
  They are experiment scripts, not production APIs or crash-safe resumable jobs.

Frozen binaries are `out/chain-batch-feedback-20261111/encoder-lab-before.exe`
and `encoder-lab-memory.exe`; the official oracle is the existing libjxl 0.12
build. Their full SHA256 identities are recorded in every successful report.
Input images, local scripts, JSON receipts and binaries are not committed.
