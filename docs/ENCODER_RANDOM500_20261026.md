# Archived Random-500 Replay

## Scope

This is the 500-input sample requested by the user, NOT a substitute 51-image
regression set. It replays the sibling C++ repository's
`out/lossless_random500_20261022/manifest.json`: seed 20261022, originally sampled
without replacement from 8847 validated inputs. It retains every original input
and its order: 314 PNGs, 182 JPEGs and four Netpbm files. No new sampling or
selection based on current encoder performance occurs.

The checkpoint committed before this work is `8220444` on Rust `main` (not
pushed). The measured encoder additionally contains the cheap-front shortlist
described in [the scheduler documentation](PROGRESSIVE_GROUP_SCHEDULER.md).
The shortlist and its ablation switch are not part of that checkpoint commit.

## Method

- Windows native Release binaries, effort 9, 16 threads, one pass, serial files.
- Pixel route: complete content-pilot search, progressive groups with default
  admission; no lowered effort and no silent timeout fallback.
- JPEG route: byte-reversible `transcode-jpeg --source-search`, not decoded pixels
  and not a lossy conversion. This iteration does not change JPEG algorithms.
- Every output is checked with official djxl. Pixel comparison includes all
  original samples and hidden RGB; JPEG comparison includes every source byte.
- JXL bitstream equality is not a target. Size may improve or regress and is
  reported independently of lossless correctness.
- Pixel call time excludes startup, source parsing and I/O but includes all
  analysis, probes and encoding. JPEG exposes only CLI wall time in this runner.
  These two timing scopes are reported separately.
- Full-source C++ and random-500 C++ results are read from existing journals;
  no old encoder is executed. Strict size pairs require verified source AND
  encoder-payload identity. Historical time is not a controlled speedup.
- Previous Rust v23 results are paired from the existing full-corpus journal.
  Previously failed/unsupported inputs are reported, not silently dropped.
- Current binaries are copied to immutable per-run snapshots. No build or
  other encoding benchmark runs concurrently. Temporary encoded/restored files
  are removed after checking.

The sampler extension to `tools/benchmark_encoder_corpus.py` accepts
`--sample-manifest`, refuses missing/duplicate/relabelled paths, preserves the
original order, and records the source manifest hash. It cannot be combined
with `--limit`. `--pilot-progressive-groups` applies only to pixel input.
Default full-corpus invocation and resume provenance remain unchanged.

## Artifacts

Run directory: `out/encoder-random500-20261026/`.

- `manifest.json`: exact 500 inputs, original sample provenance, binary/script
  hashes, invocation settings and cached baseline origins.
- `progress.json`, `trials.jsonl`: live status and flushed per-image records.
- `summary.json`, `REPORT.md`, `failures.json`: complete statuses and per-dataset
  bytes, bpp and throughput. Unsupported entries are never counted as passes.
- `analysis/comparison.json`, `analysis/REPORT.md`: completed-run comparison
  with cached Rust and C++ results, largest regressions and slowest cases.

The initial 51-image checkpoint check is separate and recorded in
[GROUP_RACE_SAMPLE_20261026.md](GROUP_RACE_SAMPLE_20261026.md).

## Completed Results

All 500 entries were attempted in 653.2 seconds including preparation and
independent verification. **496 passed, four unsupported, zero encoding failures,
zero pixel/JPEG reconstruction mismatches, zero timeouts.**

The unsupported files are the same four Netpbm inputs rejected by the old Rust
adapter: `flower_small.rgb.depth14.ppm`, `flower_small.rgb.depth3.ppm`,
`flower_small.rgba.depth12.pam`, and `flower_small.rgba.depth2.pam`.
Their MAXVAL values are 16383, 7, 4095 and 3; the current test adapter accepts
only full-range 8/16-bit Netpbm. This is not proof that the Rust codec cannot
encode those bit depths. They remain unsupported attempts, not successful tests.

| Current scope | Passed | JXL bytes | bpp | Encode seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| All, CLI wall | 496 | 270,964,038 | 4.690367 | 502.049125 | 0.920553 |
| PNG, complete encoder call | 314 | 237,432,642 | 6.454430 | 394.398995 | 0.746168 |
| JPEG, CLI wall | 182 | 33,531,396 | 1.597925 | 92.110925 | 1.822527 |

The previous Rust corpus had four additional failures in this exact sample:
DIV2K 0145, 0222, 0710 and `2026-06-26_14.43.15.png`, all MA leaf-budget errors.
All four now pass official original-pixel checking. That recovery includes
earlier fixes, not just this turn's admission change.

### Historical Rust Comparison

There are 492 strict successful source/payload/dimension pairs: 75 smaller,
400 equal-size and 17 larger. Aggregate bytes fall from 260,871,442 to
260,593,088 (0.107% smaller). JPEG sizes in this comparison are unchanged.

For the **same 310 PNG pairs**, bytes fall from 227,340,046 to 227,061,692
(0.122% smaller), but recorded call time rises from **282.506239 s to
362.243482 s**, and throughput falls from **1.007471 to 0.785706 MPix/s**.
These are historical, not simultaneous controlled measurements. Nevertheless,
the magnitude and changed full-trial counts mean this cannot be presented as a
successful general speed optimization.

The 500-case sample exercised **zero real-group races**. It tests the
content/depth/work fallback and cheap-decisive paths, but does not validate the
new ambiguous-front shortlist's pruning. Separate paired slow cases are needed
for that change; the 500-case results must not be attributed to it.

### Concrete Problems Found

| Input | Historical call s | Current call s | Historical/current bytes | Observed route difference |
| --- | ---: | ---: | --- | --- |
| `1230.hdr.png` | 11.994 | 28.311 | 20,172,530 / 20,172,530 | Full attempts 2 to 4; two costly representation witnesses before the same neighborhood winner |
| `2026-06-24_15.36.43.png` | 3.486 | 13.884 | 986,916 / 968,009 | Full attempts 1 to 4; extended RCT10/17/26 search, RCT17 wins |
| `football_player_PNG54.png` | 0.962 | 0.299 | 149,269 / 164,626 | Full attempts 3 to 1; old literal winner no longer tried, automatic output is 10.29% larger |

The last result is not an acceptable universal speed/size trade: this input was
already fast. Conversely, expensive fallback witnesses and extended RCT
confirmation spend much more time on other inputs. These require admission and
confirmation-budget improvements in the **ordinary pilot**, not indiscriminate
expansion of group racing or a global effort reduction.

The slowest new PNG calls are 28.31 s (HDR), 22.57 s (the recovered screenshot),
13.88 s and 12.64 s (two screenshots), 11.67 s (`imac_g3`), then 10.13 s.
The complete regression and slow-case lists are in `analysis/REPORT.md`.

### Cached C++ Comparison

Only 248 cases have both identical original and actual encoder-payload hashes
under the older C++ adapters. On this strict subset, current bytes are 55,706,058:

- Official 0.12 cached baseline: 56,626,900 bytes; current is 1.626% smaller.
- Prior optimized C++ full run: 55,375,825 bytes; current is 0.596% larger.
- Prior optimized C++ random-500 run: 55,232,958 bytes; current is 0.857% larger.

Do not extrapolate those subset size differences to all 500 inputs, or compare
CPP process-wall time directly with Rust pixel-call time. No old codec was rerun.

## Reproduction

From the Rust repository, using a Python environment with the existing corpus
adapter dependencies installed:

```powershell
python -B tools/benchmark_encoder_corpus.py `
  --catalog ../jxl_python/configs/lossless_corpus.local.json `
  --sample-manifest ../jxl_python/out/lossless_random500_20261022/manifest.json `
  --encoder out/group-race-sample-20261026/admission-encoder.exe `
  --jpeg-encoder out/group-race-sample-20261026/admission-cli.exe `
  --djxl ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe `
  --cached-run ../jxl_python/out/lossless_all_sources_e9_20261020 `
  --cached-run ../jxl_python/out/lossless_random500_20261022 `
  --threads 16 --pilot-progressive-groups `
  --out out/encoder-random500-20261026
```

Use a new output directory for a new run. To resume the existing run, use its
original arguments plus `--resume`, with unchanged inputs/binaries/scripts.
Completed failed/unsupported attempts remain in the journal and are not retried
as if they never happened.
