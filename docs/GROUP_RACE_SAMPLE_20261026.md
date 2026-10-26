# Group Race Checkpoint and Sample (2026-10-26)

## Checkpoint

User-requested checkpoint: `8220444`,
`Add resumable group racing and lossless scheduler experiments`.
Committed on this Rust repository's `main` using the maintainer's GitHub noreply
identity. This checkpoint was not pushed. It contains the preceding scheduler
implementation and separate, default-off preparation/MA/probe experiments.

The following admission optimization is a working-tree change AFTER that
checkpoint, not a claimed result of the checkpoint binary.

## Initial Small Regression Sample

The user clarified that the requested benchmark is the previously drawn 500
inputs, NOT the 51-case set below. The 51 cases are only a small regression
check. The requested run uses the complete original sample manifest at
`../jxl_python/out/lossless_random500_20261022/manifest.json` (seed 20261022):
314 PNG, 182 JPEG and four Netpbm inputs, in their original order.
Results are in `out/encoder-random500-20261026/`. No input is replaced, and
the sample is not filtered down to PNGs. JPEG uses byte-reversible source search.

The proposed second 51-case pass was superseded by this 500-input run.

Reused the existing sample rather than selecting new favorable cases. In order,
take the union of the cases from:

1. `out/encoder-kernels-20261026/holdout-v2/report.json` (36 cases).
2. `out/progressive-groups-20261026/slow-final/report.json` (3 cases).
3. `out/progressive-groups-20261026/mixed-final/report.json` (12 cases).

There are 51 distinct PNGs, 193,951,843 total pixels, including 20 RGBA images
and eight 16-bit images. Content includes icons, textures, documents, screenshots,
manga, transparent assets, BSDS/DIV2K natural images and known slow inputs.
This is a regression sample, not a newly randomized or independent holdout.
It does not exercise JPEG byte reconstruction or multiframe encoding.

Every produced JXL is decoded by official djxl and compared with the original
decoded samples, including invisible RGB. Metadata sanitization for test input
does not change those samples. Output JXL byte identity is NOT an acceptance
condition. Output sizes are measured, not constrained to match a previous file.
Temporary JXL and decoded files are removed after verification.

Windows Release build, effort 9, 16 encoder threads, serial images, one measured
round per case, no warmup. Encoder-call timing includes the complete requested
search, but excludes process startup, source decoding and file I/O. No competing
build or benchmark ran during measurement.

## Checkpoint Results

`out/group-race-sample-20261026/checkpoint/report.json`: 51/51 passed.
The frozen encoder is `out/group-race-sample-20261026/checkpoint-8220444.exe`.

| Measurement | Sum bytes | Sum call seconds | Aggregate MPix/s | Aggregate bpp |
| --- | ---: | ---: | ---: | ---: |
| Cached historical pilot | 122,163,404 | 259.452493 | 0.747543 | 5.038917 |
| Checkpoint group race | 122,163,404 | 235.672381 | 0.822972 | 5.038917 |

Each individual output size matched its selected cached result. The historical
9.17% lower summed time is NOT a controlled speedup: baseline measurements come
from different runs and binaries. Throughput is total pixels divided by total
call time, not the mean of per-image throughput.

Baseline precedence is fixed before comparison; no best-size or best-time choice:

1. `out/progressive-groups-20261026/cheap-front/report.json`, `old-pilot` (6).
2. `out/progressive-groups-20261026/mixed-final/report.json`, `old-pilot` (8).
3. `out/progressive-groups-20261026/slow-final/report.json`, `old-pilot` (1).
4. `out/ma-pruning-20261026/holdout/report.json`, `pilot` (12).
5. `out/encoder-kernels-20261026/holdout-v2/report.json`, `pilot` (24).

`tools/summarize_encoder_sample.py` validates passed measurements, source/payload
hashes, dimensions, effort and threads. It rejects missing or conflicting pairs;
it does not silently omit failures. Its full per-case JSON and Markdown are in
`out/group-race-sample-20261026/checkpoint-summary/`.

## Optimization

Ambiguous cheap fronts previously discarded their individual rejection decisions
and tested every planned RCT again on original groups. The new shortlist carries
forward existing admission rules while always protecting natural and Zero
representations. It does not change backend effort, MA/LZ tools or wire format.
See [the scheduler design](PROGRESSIVE_GROUP_SCHEDULER.md#ambiguous-front-admission).

The diagnostic state `FrontScreened` records omitted real-group work explicitly;
it must not be described as a rigorous encoded-byte bound. Both the cheap scores
and actual group lengths are retained separately in the report.

## Paired Admission Results

After the requested 500-input replay completed, the two cases known to enter
real-group racing were measured with the same frozen binary, two rounds each,
reversing variant order. No other benchmark/build ran concurrently. Report:
`out/group-race-sample-20261026/admission-paired/report.json`.

| Case | All candidates s | Shortlist s | Old/new MPix/s | Bytes | bpp |
| --- | ---: | ---: | --- | ---: | ---: |
| pizza_PNG44037 | 27.582553 | 15.935763 | 0.465 / 0.805 | 17,818,198 | 11.116226 |
| wallhaven-mlzoy1 | 13.080953 | 12.819602 | 0.634 / 0.647 | 14,178,604 | 13.675351 |

Every output passed official original-pixel comparison; all measured sizes
matched. Pizza's call time dropped **42.2%**. RCT10/17/26 now show
`FrontScreened`, each with zero real groups encoded, instead of roughly four
seconds of group work apiece. Candidate group work falls from 31,015,584 to
23,218,848 pixels; retained encoded-payload checkpoints fall from 44,698,105 to
33,426,878 bytes. These are payload counters, not total process RSS.

Wallhaven's candidate set and group work are unchanged. Its roughly 2% timing
difference is not evidence of an algorithmic gain. The shortlist preserves
automatic even though cheap raw-cross-channel scores are best; automatic remains
the actual full-image winner. This checks the known crop-ranking inversion.

These results justify the localized shortlist change, NOT general scheduler
rollout. The [requested 500-input replay](ENCODER_RANDOM500_20261026.md) exposes
ordinary-pilot regressions and exercises zero real-group races. The new group
scheduler remains opt-in.

Current verification: 330 scheduler tests, 88 CLI tests, 37 lab/parser tests,
19 report-tool tests and 15 corpus-runner tests passed. Two CLI acceptance tests
that require explicit fixture environment variables remain ignored by the unit
command; real-image acceptance used the independent djxl benchmark harness.

## Reproduction

The common lab arguments are:

```text
--search --content-pilot --pilot-parallel-initial --pilot-overlap-automatic
--pilot-near-tie-consensus --subtrees --ma-workers 16 --concat-batches
--fast-ranges --pilot-progressive-groups
```

For a same-binary control add `--pilot-progressive-all-candidates`; for the
shortlist omit it. Keep every other argument unchanged. Use
`tools/benchmark_encoder_tools.py --threads 16 --repeats 2` with both variants
in a JSON `--matrix` to alternate their order. The checkpoint, paired and sample
matrix files are retained under `out/group-race-sample-20261026/`.

To compare development-tool reports without rerunning a baseline (example
syntax; the second 51-case pass was not run):

```powershell
python -B tools/summarize_encoder_sample.py `
  --current out/group-race-sample-20261026/admission/report.json `
  --variant shortlist `
  --baseline out/group-race-sample-20261026/checkpoint/report.json checkpoint-8220444 `
  --out out/group-race-sample-20261026/admission-summary
```

Use a new output directory for each run. The summary tool never overwrites an
existing report and always labels cached timing as historical.
