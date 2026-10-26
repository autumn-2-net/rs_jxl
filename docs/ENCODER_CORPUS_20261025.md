# Current Rust Full-Corpus Run

This run executes only the frozen v23 Rust encoder, source effort 9. Existing
official/C++ results are reused; those encoders are not executed. The independent
official `djxl` still runs as the correctness oracle for every new output.

## Inputs

The source catalog is the existing read-only sibling project's
`configs/lossless_corpus.local.json`. It includes jxl-testdata, diverse downloads,
DIV2K, manga109, urban100, BSDS100, Set14, Set5, CC12M JPEG, local test/test2/train
images and historical screenshots. The previous full inventory had 8861 static
inputs. The new manifest records current inventory, not a random subset.

Already-JXL, animation/video and other unsupported file formats are listed in
`manifest.json` exclusions. Recognized static PNG/JPEG/Netpbm files remain in the
attempt list even when malformed or unsupported; errors are not replaced by
easier images. Packed PNG is expanded losslessly for the pixel oracle. Integer
8/16-bit Netpbm is converted to PNG with exact-sample validation before encoding;
PFM and unsupported integer layouts are reported separately, never narrowed.

## Method

- One encoding attempt per input, no warmups or repeated old-codec comparisons.
- Serial files, 16 Rust workers; no concurrent large images or 32-worker queues.
- PNG uses the same content-pilot arguments as the v23 acceptance matrix.
- Complete PNG encoder-call time includes analysis, probes, search and encoding;
  excludes process startup, image input decode and file I/O. CLI wall is separate.
- JPEG uses coefficient transcode with reconstruction data. Its CLI exposes no
  uninstrumented call timer, so only CLI wall time is reported; never relabeled
  as core encode time. There is no pixel/lossy fallback after a JPEG failure.
- All PNG samples, including hidden RGB and 16-bit precision, are checked after
  official decoding. JPEG reconstruction must equal every original byte.
- bpp is `8 * encoded_bytes / (width * height)`. Group throughput is total pixels
  divided by total time. JPEG wall time is not mixed into PNG call time.
- A 1800-second per-encoder and 300-second per-decoder guard prevents an abnormal
  child from blocking the entire sweep. Hitting it is a recorded failure, not an
  accepted optimization. Continue to the next image; never lower effort to pass.
- Encoded and restored files are temporary. Only reports, logs and input/binary
  identities persist. Original datasets and old reports are read-only.

## Cached Results

The old full run `out/lossless_all_sources_e9_20261020` and newer 500-image run
`out/lossless_random500_20261022` in the C++ repository supply cached records.
Both success and failure records are retained, with last-attempt semantics.
Sizes enter strict paired aggregates only when old verification passed and
original AND actual encoder-payload hashes match. Source-only matches remain in
the per-image journal but are labeled non-comparable because metadata removal or
input normalization can change output size. Missing baselines are not recomputed.

Old thread counts, concurrent jobs and timing methods differ. Cached times remain
historical context; no controlled speedup claim is inferred from them.

## Operation

Runner: `tools/benchmark_encoder_corpus.py`.
Results: `out/encoder-corpus-20261025-v23/`.

- `manifest.json`: frozen current inputs, exclusions, options, binaries and scripts.
- `progress.json`: process ID, active input/phase, completed/total, UTC timestamp.
- `trials.jsonl`: per-image checkpoint, size, bpp, timing, exact commands and checks.
- `summary.json` / `REPORT.md`: overall and per-dataset results.
- `failures.json`: failed and unsupported attempts.
- `stdout.log` / `stderr.log`: detached background process logs.
- `sessions.jsonl`: appended process-start receipts, runner identities and resume checkpoints.

The process is launched hidden with redirected logs. It is independent of the
interactive terminal. Windows must remain running; sleep/suspend pauses useful
progress. No power settings or scheduled wake tasks are changed.

To request a graceful stop, create a file named `STOP` in the result directory;
the runner finishes the current image first. A resume requires the original
arguments plus `--resume`, unchanged inputs/binaries/scripts and removal of the
STOP request. Completed failures are not automatically retried. `run.lock`
prevents simultaneous writers; after a hard kill, verify that its recorded PID
has exited before manually removing a stale lock. An incomplete journal line
causes an explicit error rather than silently discarding results.

This is an unattended measurement run, not a background optimization agent.
Analysis and encoder changes are deferred until the results are reviewed.

## Launch Record

Launched hidden on 2026-10-25 at approximately 21:27 Asia/Shanghai, PID `76372`.
The frozen manifest contains **8885 attempts across 13 datasets**, plus nine
non-still/already-JXL exclusions. Unlike the previous 8861-entry inventory, PNG
files previously excluded during inspection remain explicit attempts here;
animated/unsupported PNGs receive an `unsupported` record during preparation.

Preparation/runner tests: 16 input-adapter tests and five cache/journal tests
passed. The end-to-end smoke test verified packed PNG, 16-bit PNG, 16-bit RGBA
PAM and byte-reversible JPEG, and continued after an explicitly unsupported PFM.
No old encoder was invoked by these checks.

The live run was checked after tool detachment: 87 records had completed and the
next input was being verified. `Graveyard_27x39_ENGLISH.png` exceeded the existing
CLI limit (16384 per axis / 64 million pixels), so it was recorded unsupported,
not encoded and not counted as a correctness pass. This benchmark does not change
encoder limits or silently crop/resize oversized inputs. Consult live progress
and failure files rather than treating this launch checkpoint as the final count.

## Reporting Failure and Recovery

At 2026-10-25 21:48:20 Asia/Shanghai the initial process exited after 988 complete
records (976 passed, 11 failed, one unsupported). The traceback in `stderr.log`
identifies `PermissionError: [WinError 5]` while replacing `progress.json.tmp`
with `progress.json` before preparing index 988. It was an unhandled reporting
IO error, not an encoder timeout or evidence that the process was externally
killed. The log does not identify what held or denied access to the file.

The runner now retries access/sharing errors for a bounded 140 ms in total.
If a derived progress, summary, failure list or Markdown report still cannot be
written, it warns once per affected path until recovery and continues. These
files can be regenerated from `trials.jsonl`. Manifest, session receipt and trial
journal errors remain fatal: no lost measurement is reported as a success.
Final report failure no longer prevents lock cleanup. Twelve runner tests pass,
including simulated persistent report locks and a fatal journal IO failure.

The original manifest stays immutable. An explicit reporting-only maintenance
resume accepts ONLY the changed runner identity, after code review, with:

```text
--resume --resume-reporting-update 317e21ec0c3212c840b9637806a886002fb7ec69028a0d869633bfa810c008d9
```

This option requires the original runner hash; all input inventory, encoder and
decoder binaries, input adapters, options, environment and cached baseline
identities must still match. It is not permission to change measurement or
verification logic. The new runner identity and pre-resume journal hash are
appended to `sessions.jsonl`, not substituted in the original manifest.

The 988 records were checked to have unique consecutive indices 0..987. The
pre-resume journal SHA256 was
`1dd5718b6b14836bc9b59b19e3aa8e73529dc4b06cd506e0786d290a626fe197`.
Existing failure records are retained, not reclassified or retried. Next input:
`jxl-testdata/images/screenshot_game/Fantastic_Contraption_VR_screenshot_06.png`.
Recovery logs use separate `stdout-resume-*` / `stderr-resume-*` filenames to
preserve the original traceback. No encoder rebuild or baseline rerun is needed.

Resumed hidden at 21:57:10 Asia/Shanghai, PID `71388`, with log suffix
`resume-20261025-215710.log`. After detachment, 997 records were complete and the
next encoder was active. The first 988 JSONL records and original manifest were
verified byte-for-byte unchanged via their recorded file hashes. These hashes
protect benchmark provenance only; they are not compressed-output acceptance
criteria. The resumed process passed all unchanged-provenance checks, and its
new error log was empty at this checkpoint.

## Completed Sweep and Repair Triage (2026-10-26)

The resumed run completed all 8885 attempts: 8705 passed (5628 PNG and 3077
JPEG), 97 failed and 83 were unsupported. There were no encoder timeouts and no
pixel/byte verification mismatches among successful encodes. These categories
are not interchangeable: unsupported and failed inputs are not correctness passes.

PNG aggregate encoder-call throughput was 0.98168 MPix/s, including the entire
pilot/search. Images of at least one million pixels reached 1.461 MPix/s;
1806 images below 100000 pixels consumed about 1013 seconds. PNG p95 was 2.583
seconds and the maximum was 22.133 seconds. JPEG wall timings remain separate.

Repair priorities and measured causes:

- 89 PNG failures report `merged MA channel tree exceeds leaf budget`, across
  seven datasets. Both cached C++ variants succeeded. The merged-tree ceiling
  must be distinguished from the per-learner leaf ceiling without removing
  explicit caller resource bounds. All 89 require independent-decoder retries.
- `sublime.png` (330795 bytes) and `terminal.png` (44440 bytes) miss useful global
  representations. Automatic group streaming changes the palette/patch search
  space relative to the cached C++ `--buffering=0` runs. Unconditionally forcing
  full-frame processing is NOT a fix: another screenshot went from about 14 to
  24.5 seconds with unchanged size.
- Three 16-bit synthetic inputs miss literal-palette or raw-Zero representations
  because small-crop ranking reverses at full size. Real full-image witnesses,
  not extrapolated probe sizes, are needed for those distinct representations.
- The C++ Trainyard winners use cross-channel RCT10/17; the Rust pilot front
  previously contained only RCT0/6. The transform kernels already exist.
- The 80x80 `hlog.png` uses 9 full trials and 90 overlapping crop encodes. A
  compact full-output tournament can avoid more screening work than encoding.
- Variable-predictor candidates did not receive the existing indexed LZ77 tools.
  Those tools must be tested through the entire pilot, not just a direct trial.

Experiments and official pixel checks are under `out/corpus-repair-20261026/`.
`families`, `palette`, `fullframe`, `cross-index` contain explicit candidate
experiments; they are not automatic-policy acceptance. `profile` has intrusive
tracing and must not be used as a timing baseline. `pilot-repair-v1` is an
intermediate automatic-policy experiment, not the final accepted build.

The frozen v23 corpus and binaries remain unchanged. No old C++ encoder is
rerun. Repairs are checked against original pixels (including invisible RGB
and 16-bit values), not against the bytes of an earlier compressed JXL.

### MA Budget Repair Contract

The legacy 16384-leaf ceiling was also used after detached-subtree merging and
as the entropy raw-context ceiling. The first failed-image smoke test exposed
the latter only after tree merging succeeded. Both must be corrected together.

- Direct training without an explicit merged limit retains its legacy ceiling.
- Source scheduling with `NativeResourcePolicy.max_tree_leaves=None` uses the
  global decoder bound: at most `min(2^22, 1024 + width*height*channels/16)`
  nodes, converted to a binary-tree leaf ceiling with checked/wide arithmetic.
- `Some(n)`, including `Some(16384)`, remains an explicit whole-tree cap.
- Local stream trees additionally obey their transformed-sample decoder bound;
  patch-reference atlases recalculate the bound from their own dimensions.
- Constants and unfinished roots reserve leaves in the same merged budget.
  Parallel descendants cannot reset that budget to a larger root allowance.
- Raw entropy contexts allow the legal MA leaf range plus the LZ77 distance
  context. This does not increase the 256 clustered histogram-ID ceiling.

No automatic effort reduction, tree truncation or resource-error suppression
is introduced. Reaching a genuine explicit limit can still fail under the
complete-search contract. Supporting a deliberately degraded but lossless
fallback would be a separate policy change, not this bug fix.

### Failed-Input Retry

`out/corpus-repair-20261026/ma-retry-v2/report.json` completed **89/89** former
MA-failed PNG entries (80 distinct original file hashes). Every output was
decoded by official libjxl 0.12 `djxl` and compared against the original integer
samples, including invisible RGB and 16-bit values. All matched; no failed
entry was reclassified as skipped or accepted through a different encoder.

Aggregate encoder-call time was 527.494 seconds, throughput 0.4404 MPix/s and
weighted output 9.1370 bpp for this failure-only subset. This restores encoding
correctness, not performance acceptance. `pizza_PNG44037.png` still took
34.859 seconds (11.1162 bpp); `wallhaven-mlzoy1.png` took 24.3--24.7 seconds
(13.6754 bpp). The frozen full-corpus throughput excludes these formerly failed
inputs; it must not be presented as the throughput of a fully repaired sweep.

At the user's subsequent request, an explicit strict/bounded learning policy
was added separately: see [MA Tree Budget](MA_TREE_BUDGET.md). The default
remains unchanged; bounded stopping is not the explanation for the 89 passes.

### Default-Policy Repair Results

`pilot-repair-v3/report.json` contains the rebuilt encoder's 13 priority cases
plus three formerly failed images, all independently pixel-verified. No MA
budget override or effort reduction was used. Before/after source and prepared
payload hashes match. These hashes identify the input, not an expected JXL file.

The pilot now includes missing representation witnesses and competitive RCT
alternatives rather than trusting only the first crop winner. Variable-predictor
routes use the existing indexed LZ77 tools. Very small full-frame tournaments
dispatch independent candidates within worker/scratch limits on the same pool,
without nested cooperative work or redundant crop encoding. Only completed
outputs can win; estimates are not encoded-byte bounds.

Times below include the complete search, with 16 encoder threads, excluding CLI
startup and input/output IO. The old column reuses the frozen v23 sweep; the new
column is one measured repetition, not an interleaved timing significance test.

| Input | Old bytes | New bytes | Size change | Old s | New s | New MPix/s | New bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trainyard 2/3 | 54652 | 51277 | -6.18% | 2.423 | 0.935 | 0.657 | 0.6677 |
| Trainyard 3/3 | 54391 | 50362 | -7.41% | 2.060 | 1.942 | 0.316 | 0.6558 |
| 16-bit circles center | 58585 | 32358 | -44.77% | 0.408 | 1.326 | 0.130 | 1.4981 |
| 16-bit corner rays | 115369 | 68318 | -40.78% | 0.368 | 1.132 | 0.153 | 3.1629 |
| 16-bit center rays | 184679 | 101833 | -44.86% | 0.302 | 1.069 | 0.162 | 4.7145 |
| hlog | 10544 | 6039 | -42.73% | 1.183 | 0.240 | 0.027 | 7.5488 |
| sublime | 330795 | 140300 | -57.59% | 1.973 | 3.939 | 2.434 | 0.1170 |
| terminal | 44440 | 30730 | -30.85% | 2.116 | 4.769 | 0.367 | 0.1406 |
| 2026-07-08_22.28.55 | 421537 | 404003 | -4.16% | 14.218 | 4.345 | 0.896 | 0.8301 |
| Flinthook tutorial | 286020 | 274880 | -3.89% | 10.703 | 6.754 | 0.120 | 2.7149 |
| 2026-07-03_13.14.38 | 1257744 | 1221190 | -2.91% | 14.927 | 6.519 | 0.597 | 2.5090 |
| creativecommons.org | 2691432 | 2691432 | 0.00% | 13.364 | 11.956 | 0.565 | 3.1884 |
| floral_frame_PNG270 | 5235209 | 5100273 | -2.58% | 22.133 | 21.160 | 0.854 | 2.2590 |

There are explicit runtime tradeoffs: the three small 16-bit cases and both
editor/terminal cases get materially smaller but take longer than v23. The
terminal repair reduced the intermediate v2 runtime from 8.17 to 4.77 seconds,
not below v23's 2.12 seconds. These are not universal speed improvements.
The default final build still takes 35.74 seconds on pizza and 24.72 seconds on
wallhaven-mlzoy1; their bpp remains 11.1162 and 13.6754 respectively.

The 36-image independent holdout in `holdout-v3/report.json` excludes all tuned
inputs and previous MA failures. Seed 20261026 selects eight entries in each
pixel-count stratum [4096,12289), [12289,100000), [100000,1000000), and
[1000000,16000000), plus four additional 16-bit inputs, deduplicated by source
hash. It spans BSDS100, DIV2K, diverse-downloads, jxl-testdata, local-train,
manga109 and urban100. All 36 match original pixels with official djxl:

- 13 outputs are smaller; 23 have equal size; none is larger.
- Total output: 24172130 -> 24159501 bytes.
- Total encoder-call time: 30.613 -> 30.996 seconds (about +1.3%).
- Aggregate throughput: 1.188 -> 1.173 MPix/s; weighted output: 5.3178 -> 5.3151 bpp.
- This limited holdout is not proof of no regressions across the full corpus.

Coarse-to-fine MA split search was also measured in `refine-quantiles-v2/`.
Its roughly 0--16% observed runtime benefit on three slow cases does not justify
changing the default; single-run variance is material. It remains opt-in.

Verification groups passed: 289 scheduler tests, 110 budget tests (one external
oracle test not enabled), 88 normal CLI tests (two environment-driven acceptance
tests not enabled), and 34 encoder-lab tests. The budget group includes the ten
new mode tests. External pixel checks above are separate from those unit tests.

`bounded64-v3`, `bounded-large-v3` and `cli-smoke-v3` separately exercise the
explicit new cap/mode flags. See [the budget experiment table](MA_TREE_BUDGET.md)
for both improvements and counterexamples. They are not mixed into the default
policy table or holdout statistics. The installed CLI smoke also verifies exact
JPEG reconstruction. No old C++ encoder or full 8885-input sweep was rerun.
