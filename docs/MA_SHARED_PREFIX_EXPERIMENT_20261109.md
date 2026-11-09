# Shared MA Prefix Experiment, 2026-11-09

## Decision and Scope

**Rejected as a maintained encoder path.** The prototype was implemented,
compiled and measured, but did not deliver a convincing size/speed tradeoff
across the selected images. Its native field, learner and CLI switch have been
removed. Source snapshots, tests, executable and reports remain under
`out/ma-prefix-20261109/`; the benchmark harness retains support for that frozen
experimental executable. Earlier model-cost options and diagnostics remain.

This experiment replaces the upper part of MA training with layer-wise shared
conditions, then invokes the existing unrestricted lower-tree learner. The
starting revision is `24a625944925c15a81c26b34023f99c95aa0a8f3`, with the earlier
[model-cost experiment and diagnostics](MA_STRATEGY_EXPERIMENTS_20261109.md)
already in the working tree. This is pixel-lossless encoder research, not a
decoder, lossy encoding, or JPEG byte-transcode change.

The prototype default was zero shared levels. Neither the installed CLI nor the
Python library was replaced by the development example. Compressed JXL byte
identity is not a target: acceptance compares the original integer pixels,
including invisible RGB, after independent official decoding.

## Algorithm

1. Build the existing required multiplier skeleton first.
2. At each requested shared level, prepare histograms and predictor plans for
   all live roots. For every property, evaluate its complete threshold curve
   at every root using the existing histogram/SIMD scorer.
3. Select one property/threshold by the sum of positive per-root net gains:

   ```text
   gain(node, condition) = max(0,
       parent_cost - child_below_cost - child_above_cost - split_threshold)
   shared_condition = ordered_argmax(sum_over_nodes(gain))
   ```

4. Split only roots with positive gain at that shared condition. Roots with no
   gain remain available at the next level and to the free-form suffix.
5. After the requested number of levels, run the original complete learner
   independently below the resulting roots, retaining predictors/multipliers.
   The prefix count is cleared in suffix options so detached work does not
   apply it again.

This is a **pruned shared prefix**, not a complete oblivious tree. The number
of prefix levels is not a maximum depth or a limit on the completed MA tree.
All decisions serialize as ordinary JXL MA property/threshold nodes. No new
bitstream feature or decoder algorithm is needed.

The experiment changes upper-level split preference: it maximizes summed net
gain rather than preserving the original property-class priority. The suffix
keeps the original policy. Predictor penalties select a child predictor but
are not added to its wire-bit estimate. The existing optional description-cost
regularizer applies consistently to parents and children.

## Prototype Boundaries

- Native control: `SearchOptions.symmetric_prefix_levels`, default zero.
- Development example control: `encoder_lab --ma-prefix-levels N`, `0..255`.
- Implementation: `modular/ma_train/symmetric_prefix.rs`; routing into the
  unchanged suffix lives in `modular/ma_train/learner.rs`.
- Properties may execute in parallel with private scratch. Reduction is ordered
  for deterministic tie-breaking. The winning property is recomputed once per
  root, avoiding a retained nodes-times-properties-times-bins cost tensor.
- Empty global quantizer bins inherit the previous live-bin costs while retaining
  the exact shared threshold. Partition sizes use distinct records, not weights.
- Required multiplier boundaries and explicit caller budgets remain in force.
  If a complete shared layer does not fit the remaining budget, the original
  suffix handles strict failure or bounded completion. No new silent sample,
  leaf, depth, or work ceiling is introduced.
- Partial threshold sweeps, statistical racing and coarsened collection are
  rejected in combination with a nonzero prefix: their cost rows do not describe
  the complete curves required by this implementation.

This first implementation still evaluates per-node histograms. It is **not** a
fused whole-layer histogram kernel or a new SIMD sample-routing implementation.
It reuses existing native SIMD scoring and parallel execution. It also does not
carry the existing parent's histogram-subtraction cache across shared levels;
any performance result must account for that cost rather than assuming that a
shared tree automatically makes training cheap.

Review also found that the separate optional channel-tree entry bypassed this
prefix hook. The measured route has channel trees disabled, so the reported
experiment is unaffected. The archived prototype is not a supported combination
with channel-tree partitioning. A future implementation must integrate at both
entry points or reject the combination, not silently ignore the prefix request.
The prefix flag also conservatively disables the existing exact predictor
dominance check for populations of at most 256; the suffix clears that flag.
Neither limitation is retained in the maintained encoder after removal.

## Measurement

Windows x86_64, Ryzen 9 7945HX, Release, effort 9, eight threads. Baseline and
experimental settings run in the same frozen executable, with interleaved
variant order. Native encoder seconds exclude process startup, PNG loading,
output IO and validation; they include encoder/context setup. MPix counts
spatial pixels, not channel samples. bpp includes the complete emitted file.

Every trial is decoded by official libjxl 0.12 `djxl` and compared with the
original PNG integer samples. Input/executable hashes identify the experiment;
encoded JXL hashes are never an acceptance criterion. Diagnostic runs are
separate from speed measurements because instrumentation adds overhead.

## Results

The initial screen used four images, four levels (0/2/4/6), and two runs per
setting without warmup. Seven additional holdouts used one run per setting;
their timings are not evidence of a small speedup. Confirmation on the four
larger cases used one warmup and three measured runs per setting (0/2/6).
The following table contains the confirmation medians, not the initial screen.

| Image | Prefix levels | Bytes | bpp | Encode seconds | MPix/s | Size versus 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tecnick 023, 1200x1200 | 0 | 1,530,900 | 8.5050 | 1.1909 | 1.209 | baseline |
| Tecnick 023 | 2 | 1,531,433 | 8.5080 | 0.9226 | 1.561 | +0.035% |
| Tecnick 023 | 6 | 1,532,946 | 8.5164 | 1.0623 | 1.356 | +0.134% |
| Clover, 1024x1024 | 0 | 1,673,489 | 12.7677 | 1.3550 | 0.774 | baseline |
| Clover | 2 | 1,673,184 | 12.7654 | 1.2423 | 0.844 | -0.018% |
| Clover | 6 | 1,674,419 | 12.7748 | 1.2082 | 0.868 | +0.056% |
| PDF 9bde, 1264x1152 | 0 | 412,541 | 2.2665 | 1.6781 | 0.868 | baseline |
| PDF 9bde | 2 | 413,886 | 2.2739 | 1.6912 | 0.861 | +0.326% |
| PDF 9bde | 6 | 416,675 | 2.2892 | 1.8806 | 0.774 | +1.002% |
| wallhaven-mlzoy1, 3840x2160 | 0 | 14,179,178 | 13.6759 | 5.7166 | 1.451 | baseline |
| wallhaven-mlzoy1 | 2 | 14,181,074 | 13.6777 | 5.7530 | 1.442 | +0.013% |
| wallhaven-mlzoy1 | 6 | 14,193,371 | 13.6896 | 6.3659 | 1.303 | +0.100% |

Two levels show a local improvement on Tecnick. Its confirmation ranges overlap
(baseline 1.027..1.208 s; two levels 0.909..1.130 s), and the initial screen had
only a 5.56% median reduction, versus 22.53% in confirmation. Clover's ranges
also overlap substantially. These are not robust corpus-wide speed claims.
The sum of four per-image medians is 9.9406 / 9.6090 / 10.5171 seconds for
0 / 2 / 6 levels: -3.34% for two levels, +5.80% for six levels.

The six-level PDF and 4K regressions reproduce in both rounds. Four levels in
the initial screen also made PDF 0.999% larger and 4K 0.059% larger, with higher
median time on both. There is no clear large-case win to offset those results.

The seven holdouts include Kodak, game UI, webpage, transparent icon, grayscale,
16-bit PNG and another wallpaper. All sizes were deterministic across repeated
trials. Six levels reduce Kodak by 0.402%, but increase the game UI by 1.004%,
webpage by 1.032%, icon by 0.508%, grayscale by 0.177%, 16-bit image by 0.167%
and wallpaper by 0.272%. Four levels save 65 bytes on the tiny 16-bit case;
this does not justify a new production selection policy.

## What the Profiles Show

The instrumented runs confirm real prefix execution, not an ignored switch.
Training populations are unchanged: 3,798,670 distinct records across 25 PDF
groups and 16,219,035 across 135 4K groups. Counts below are summed over those
independent groups; properties are evaluated per node, not once per image.

| Image | Levels | Prefix splits | Prefix node/property evaluations | Suffix split-search calls | Final leaves |
| --- | ---: | ---: | ---: | ---: | ---: |
| PDF | 0 | 0 | 0 | 19,701 | 9,863 |
| PDF | 2 | 75 | 1,275 | 19,712 | 9,906 |
| PDF | 6 | 851 | 19,227 | 19,680 | 10,278 |
| 4K | 0 | 0 | 0 | 148,263 | 74,199 |
| 4K | 2 | 398 | 6,885 | 148,493 | 74,513 |
| 4K | 6 | 4,049 | 111,639 | 150,936 | 77,560 |

The extra prefix work does not substantially replace suffix search. On 4K,
six levels even increase suffix calls and final leaves. Histogram-cache reuse
also falls: 2,047 baseline reuses versus 708 at six levels; corresponding cached
histogram subtractions fall from 32,752 to 11,328. Prefix histograms are not
handed to the suffix cache in this prototype.

Emitted 4K tree-plus-entropy-header bits increase from 1,651,936 to 1,695,006.
For PDF those bits actually decrease slightly (360,591 to 360,084), yet the
complete file grows by 4,134 bytes. That is another example of why model size
alone is not the final compression objective.

Inclusive diagnostic timings overlap across threads and are not wall-time
speed measurements. The evidence supports rejecting this **implementation**,
not rejecting all symmetric trees or all alternative context partitioning.
Simply adding SIMD to threshold comparisons would not remove the measured
extra learning work. A different future experiment would need to share actual
histogram construction/routing across the frontier, preserve useful cached
statistics, or replace substantial suffix work, then remeasure final bytes.
Those improvements have not been implemented or demonstrated here.

## Validation and Artifacts

- Prototype: 223 focused MA tests, including eight new prefix-specific tests;
  23 development-example tests; nine harness tests passed.
- After removal: 215 focused MA tests, 21 development-example tests and nine
  retained harness tests passed. Formatting and whitespace checks passed.
- The maintained Release example was rebuilt after removal, then passed six
  additional official-decoder comparisons on RGB, RGBA and 16-bit inputs with
  the retained `off`/`full` description-cost settings (`final-smoke.json`). The
  native CLI/DLL production artifacts were not replaced. There are 120 successful
  pixel checks including this final cleanup verification.
- Independent oracle: 114 successful original-pixel checks across 11 images
  (32 screen, 28 holdout, 48 confirmation including warmups, six diagnostic).
  RGB, RGBA with invisible samples, grayscale and 16-bit inputs are covered.
- This is targeted Windows testing, not new Linux/full-corpus, float,
  17..31-bit or multiframe acceptance. No such completeness is inferred.
- `screen.json`, `holdout.json`, `recheck.json`, `profile.json` retain all
  commands, timings, settings, size receipts and validation results.
- `encoder-lab.exe` is the frozen prototype, not the production CLI.
- `symmetric_prefix.rs`, `symmetric_prefix_tests.rs` and
  `shared-prefix-experiment.patch` retain the rejected implementation. The
  integration diff is relative to the starting revision and includes earlier
  model-cost/diagnostic edits; it is not a patch to apply blindly to current
  sources. The two new source files are archived separately from that diff.

## Reproduction

Use the frozen prototype for reproduction. Rebuilding current sources does
**not** enable the removed `--ma-prefix-levels` option.

```powershell
out/ma-prefix-20261109/encoder-lab.exe INPUT.png --effort 9 --threads 8 --ma-prefix-levels 4 --description-cost off --model-costs --output OUTPUT.jxl
python -B tools/benchmark_ma_models.py --encoder out/ma-prefix-20261109/encoder-lab.exe --decoder PATH/TO/djxl.exe --case INPUT.png --modes off --prefix-levels 0 2 4 6 --effort 9 --threads 8 --repeats 3 --warmups 1 --out out/ma-prefix-local.json
cargo test --offline -p rs-jxl --lib modular::ma_ -- --test-threads=8 --quiet
cargo test --offline -p rs-jxl-cli --example encoder_lab
python -B -m unittest discover -s tools -p test_benchmark_ma_models.py
```

`--profile` enables diagnostic scopes and prefix summaries. Their parallel
inclusive durations overlap and must not be added together as wall time.
