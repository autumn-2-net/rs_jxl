# PR5007 MA Tree Learning Experiment

## Scope

Native Rust port of [libjxl PR #5007](https://github.com/libjxl/libjxl/pull/5007),
"Improve tree learning", pinned at
`f714d07bb20973ca216709f25bbb190ca53472f3`.
The PR was open when inspected. Source:
[`lib/jxl/modular/encoding/enc_ma.cc`](https://github.com/libjxl/libjxl/blob/f714d07bb20973ca216709f25bbb190ca53472f3/lib/jxl/modular/encoding/enc_ma.cc).
Pre-experiment Rust checkpoint: `eb0b343`.

This changes encoder decisions only. Symmetric intervals become two ordinary
strict-greater binary MA nodes, with three ordinary predictor leaves. No new
predictor, decoder, entropy format, C++ linkage, or external runtime is added.
The earlier LZ77 and post-training refinement experiments remain disabled.

The default remains `TreeLearning::Legacy`, independently of effort. These
controls are experimental native/development APIs, not installed CLI or Python
options. JPEG byte-transcoding and lossy policies are not changed.

## Modes

- `Legacy`: existing learner and split threshold.
- `Cost`: multiply the already sampling-adjusted threshold by 0.72. Charge each
  split `threshold * (1 + 0.052 * log2(1 + PackSigned(split_value)))`, plus
  `0.10 * (global_max_symbols - 1)`.
- `Symmetric`: the cost model above, plus the PR's near-zero interval proposals.
  Only properties 10..14 and 19, 23, ... qualify. High boundaries are 0 or 1;
  the low boundary is the nearest quantized value to `-high-1`, within one.
  A proposal must have at least `max(64, 2 * global_max_symbols)` distinct
  prepared records on each of its three sides. Its two split costs also incur
  an alphabet correction of `10 * (global_max_symbols - 1)`. When an ordinary
  split also wins, the interval must improve estimated gain by at least 1.4x.

`global_max_symbols` means the largest residual token plus one across the
complete learner population and all configured predictors, not the number of
nonzero histogram bins. It is computed once for an enabled experiment and
preserved through disjoint subtree loans. Distinct records control candidate
admission; weighted counts and extra bits control entropy and density estimates.

This patch does not replace the sampler. The existing
`node_threshold * (0.9 * selected_samples / source_pixels + 0.1)` adjustment
remains in place, followed by the PR's 0.72 factor exactly once.

## Integration

The existing native SIMD histogram scoring and parallel property dispatch are
retained. The middle interval reuses collected histogram rows rather than
collecting pixel samples for each proposal. Two-sided expansions preserve
required multiplier partitions and copy the parent's multiplier to every leaf.
Their two additional leaves are reserved atomically; strict/bounded budgets keep
their documented meanings.

The largest child can reuse parent histograms by subtracting the other two.
An outer child can also inherit the existing deferred histogram cache because
its complement is contiguous. A middle child's two disjoint complement ranges
are not falsely represented as a single interval; its property cache is dropped.

Property-partition equivalence no longer proves identical syntax cost, so that
optional pruning is disabled for the experiment. Binary predictor-dominance and
event-sweep shortcuts cannot prove a middle-interval winner. Since 2026-11-14,
they remain available when interval preparation proves there is no eligible
middle interval in this node/property. Existing predictor-equivalence proofs
remain valid.
Description cost can now compose with both learners; all three child regions
and the parent include the selected description weight. Other alternative
entropy/search experiments and seeded/fixed trees remain rejected. Efforts
whose source route is a fixed tree are outside this learner experiment.

See [the combined-profile experiment](SYMMETRIC_MA_DEFAULT_20261114.md) for the
comparison with Half plus single-pass Distribution Offset. The results below
are the original experiment, not measurements against that newer default.

## Running

```text
cargo build --release -p rs-jxl-cli --example encoder_lab
target/release/examples/encoder_lab input.png --effort 9 --threads 8 --tree-learning cost
target/release/examples/encoder_lab input.png --effort 9 --threads 8 --tree-learning symmetric
python -B tools/benchmark_tree_learning.py --encoder PATH_TO_ENCODER_LAB --decoder PATH_TO_OFFICIAL_DJXL --case input.png --synthetic --effort 9 --threads 8 --repeats 3 --out out/tree-learning/results.json
```

Native selection is `ModularOptions.tree_learning`; direct learner selection is
`TrainOptions.tree_learning`. `TreeLearningReceipt` optionally reports executed
learners, prepared records, ordinary cost-model splits, and two-sided splits.
Read it after encoding completes. Counters describe work, including any retried
training domains, not unique nodes in the final emitted tree.
Use a fresh receipt per invocation; cloned options intentionally share it.

The benchmark records actual bytes, bpp, native-call seconds, MPix/s, emitted
model bits, requested mode and work counters. Trials are serial and rotate
variant order; summaries use per-image medians and ratios of corpus totals.
Input decode, process startup, disk writes and oracle decoding are outside the
reported native time. Every output is independently decoded with official djxl
and compared with original samples, including hidden RGB and 16-bit channels.
Encoded-byte equality is not an optimization objective. Binary hashes pin tool
provenance only. A process timeout is a harness guard, not encoder scheduling.

Temporary JXL/PNG outputs are removed. Reports are exclusive-created and size
regressions are retained as results rather than hidden as test failures.

## Results

**Kept opt-in, not promoted.** Twelve real PNGs, 21,178,537 pixels, effort 9,
eight threads, three order-rotated repetitions. The table sums per-image median
encode times. These are complete fixed native candidate calls, not outer
scheduler throughput. Timings include optional work/model counters.

| Mode | Total bytes | Change | Seconds | MPix/s | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy | 23,263,912 | baseline | 13.4158 | 1.5786 | 8.78773 |
| Cost | 23,225,725 | -0.16415% | 13.6062 | 1.5565 | 8.77331 |
| Symmetric | 23,211,990 | -0.22319% | 13.6245 | 1.5544 | 8.76812 |
| Frozen pre-PR Rust control | 23,263,912 | 0% | 13.4985 | 1.5690 | 8.78773 |

Both experiments reduced size on all twelve real samples. Cost saved 38,187
bytes; Symmetric saved 51,922 bytes. Measured time increased 1.42% and 1.56%
respectively against current Legacy. Differences this small are not reliable
evidence of a speed change: individual runs fluctuate, and the frozen control
differs by 0.62% in aggregate. No significant acceleration is claimed.

| Sample | Legacy bytes | Cost bytes | Symmetric bytes | Legacy s | Symmetric s | Legacy -> symmetric MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper screenshot `9bde...` | 412,541 | 411,346 | 409,300 | 1.5608 | 1.7126 | 0.933 -> 0.850 |
| Editor screenshot `3dd...` | 126,449 | 125,864 | 125,913 | 0.4687 | 0.4620 | 4.560 -> 4.626 |
| `wallhaven-49yekn` | 707,442 | 705,646 | 705,276 | 0.9282 | 0.9510 | 2.234 -> 2.180 |
| `wallhaven-mlzoy1` | 14,179,178 | 14,166,611 | 14,166,611 | 5.3067 | 5.2483 | 1.563 -> 1.580 |
| 16-bit linen displacement | 1,841,476 | 1,829,554 | 1,829,554 | 0.3199 | 0.2671 | 3.285 -> 3.933 |
| RGBA `cursor_PNG65` | 14,678 | 14,515 | 14,515 | 0.1480 | 0.1522 | 1.771 -> 1.722 |
| Kenney `Preview` | 57,077 | 56,857 | 56,857 | 0.1659 | 0.1689 | 2.850 -> 2.799 |
| Manga `AisazuNihaIrarenai` | 1,372,611 | 1,368,897 | 1,367,679 | 0.9470 | 0.8380 | 1.021 -> 1.153 |
| DIV2K `0089` | 3,152,231 | 3,148,036 | 3,138,652 | 1.8470 | 1.9746 | 1.697 -> 1.587 |
| Set14 `monarch` | 407,506 | 407,226 | 407,213 | 0.6003 | 0.6422 | 0.655 -> 0.612 |
| Urban100 `img_061` | 958,068 | 956,608 | 955,855 | 0.6654 | 0.7810 | 1.034 -> 0.881 |
| Web screenshot `8279` | 34,655 | 34,565 | 34,565 | 0.4579 | 0.4263 | 0.546 -> 0.587 |

Actual work receipts recorded 226 symmetric decisions per pass across eight of
the twelve cases, including 54 for the paper and 98 for DIV2K. Four cases used
only the revised ordinary split model. The editor screenshot is **49 bytes
larger with Symmetric than Cost**, so adding this candidate is not globally
monotonic even when both beat Legacy.

## Correctness And Limits

The central validation checkpoint passed:

- 606 Modular unit tests, with one existing ignored test, including 23 new
  scoring/learner regressions. They exercise actual symmetric decisions,
  signed/gapped thresholds, weighted versus distinct populations, global
  alphabets, strict/bounded budgets, mandatory multipliers and unsampled ranges,
  constant suffixes, serial/parallel property search, physical/index-only
  partitioning, parent histogram conservation and cache lease release.
- 26 development CLI parser tests and 13 Python benchmark-harness tests.
- After the final equivalent bucket-iteration lint cleanup, 245 focused MA
  tests passed again. Timing reports retain the exact measured binary identity.
- Workspace/all-target compilation with `rs-jxl/experimental-lz77` enabled.
- 198 official djxl 0.12 pixel comparisons: 144 paired real-image trials,
  24 e4 single-thread streaming synthetic checks, 24 e10 eight-thread shared
  synthetic checks, and six e9 full-frame shared checks on the editor and
  16-bit linen images. Temporary encoded/decoded outputs were cleaned.

Synthetic fixtures cover 8/16-bit gray, gray-alpha, RGB and RGBA, including
nonzero invisible colors and partial alpha. **They disprove a no-regression
claim:** the e4 RGB fixture grows from 9,230 to 9,245 bytes (+0.163%), and the
e10 RGB fixture grows from 4,753 to 4,805 bytes (+1.094%). The e10 gray fixture
shrinks from 1,918 to 1,855 bytes (-3.285%). All still reconstruct exactly.

Shared e9 results are a separate layout: editor 104,195 -> 104,175 bytes and
linen 1,811,376 -> 1,807,196 bytes. Do not compare those against the default
frame-area table to attribute layout effects to this learner.

Reports are in ignored `out/pr5007-20261109/`: `paired-e9.json`,
`synthetic-e4-streaming-single.json`, `synthetic-e10-shared.json`, and
`shared-e9.json`. The manifest pins the twelve original files before measurement;
reports include their hashes and tool hashes. PNG parsing reports an existing
incorrect-sRGB-profile warning on an input; sample comparisons still pass.

The first two-sided commit queues low/middle/high pending children rather than
using the binary ready-sibling/live-frontier fork immediately. Later subtree
splits still use existing parallel dispatch. This is a potential optimization,
not a proven performance gain. Linux execution and a larger corpus were not run
in this experiment. This sample is not evidence to change default effort tiers.

Strict Clippy still encounters the four pre-existing `needless_borrow` warnings
in `fast_context.rs` (500, 503, 506, 514). They are not suppressed in project
configuration or silently reported as a clean strict pass.
The diagnostic run with only that lint allowed on the command line passed
`cargo clippy -p rs-jxl -p rs-jxl-cli --all-targets -- -D warnings -A clippy::needless_borrow`.
