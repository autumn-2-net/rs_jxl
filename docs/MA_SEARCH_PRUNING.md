# Exact MA Search Pruning

This experiment removes provably redundant search work, not prediction tools,
training samples, thresholds with uncertain scores, or leaf capacity. It is
opt-in while performance is evaluated. It is not SAT conflict learning.
The proofs concern the current sampled MA scoring objective, not a global
minimum for the final JXL size. Real output size is measured independently.

## Controls

Both the normal CLI and the development `encoder_lab` accept `--ma-prune`.
For example:

```powershell
.\target\release\rs-jxl.exe encode input.png output.jxl --source-effort 9 --source-search --source-pilot --threads 16 --ma-prune
```

The Rust API controls are `ModularOptions::source_property_dedup` and
`ModularOptions::source_ma_dominance`. The corresponding direct learner controls
are `TrainOptions::property_dedup` and `TrainOptions::split_dominance`. The CLI
flag enables both; these controls do not enable predictor deduplication,
statistical races, lower sampling, or a bounded leaf-budget mode.

## Reusable Partition Proofs

Constant properties and exact equal-bin properties already had optional
inherited masks. The new proof also accepts strictly monotone relabeling:
bins `[1, 1, 3, 7]` and `[20, 20, 50, 90]` generate identical ordered partitions.

Ordered nonzero occupancy counts provide a cheap fingerprint. Every candidate
match is then verified against every live record; hash equality alone is never
accepted. Mapping must be single-valued and strictly increasing, so equal
marginals, reversed order, merged bins, or different row assignments do not pass.
Static/nonstatic and WP-property classes must match before comparison.

This proof survives arbitrary subsets and is inherited by both children and
detached subtrees. It belongs to a training tree, not a global image cache.
Different encodes/candidates rebuild their proofs. Relabeling is disabled for
nondefault search policies which can select thresholds by original bin index.

Cross-property discovery retains the existing serial-node, at-least-64-record
gate. Parallel sweeps can consume inherited proofs and discover constants;
they do not add a serial all-property barrier to discover aliases.

## Predictor Dominance

After collecting a predictor's histogram, its extra-bit prefixes give lower
bounds for both child costs even if entropy were zero. If this predictor cannot
strictly improve either incumbent at ANY live threshold, its entire entropy
sweep is skipped. The source f32 addition/subtraction and predictor-penalty order
are retained; no incumbent or class-selection rule is changed.

The initial route requires default scoring and default search, and total
WEIGHT <=256, not merely at most 256 distinct records. Both child entropies then
use the small contribution tables with nonnegative costs. Larger populations,
incremental entropy/refinement and the separate event-sweep path retain their
existing algorithms. This deliberately avoids assuming a general approximate
entropy bound without validating its numeric domain.

Numerical dominance is recomputed per node, never inherited. A predictor losing
at a parent does not prove it loses in a child; similarly, a no-gain split at a
parent is not a valid prohibition on descendant splits. This implementation
does not maintain a global blacklist of unsuccessful parameters or properties.

## Validation and Measurement

Tests compare partition proofs with enumerated threshold partitions, and
dominance decisions with unpruned scoring, including weighted samples, strict
ties, penalties, floating-point evaluation order and inherited parallel masks.
These are local search-equivalence tests, not a compressed-byte acceptance rule.

Real-image acceptance uses official `djxl` original-pixel equality, including
invisible RGB and 16-bit samples. JPEG transcode uses exact reconstructed JPEG
bytes. Encoder-call time includes the complete pilot search but excludes CLI
startup and input/output I/O. Reports also include MPix/s, bpp and output bytes.

`RS_JXL_PROFILE=1` reports `duplicate_properties`, `inherited_properties`,
`constant_properties` and `dominated_predictors` in `ma.search_counts`. Counters
are shared within a learner and appear in nested logs; do not sum every printed
snapshot as if they were disjoint events.

The implementation checks passed on 2026-10-26:

- Modular/MA unit tests: 500 passed, one opt-in microbenchmark ignored.
- Normal CLI and encoder-lab budget/pruning parser tests: four passed.
- The existing encoder-lab unused `Tools::apply` warning is unchanged.

The experiment matrix is `out/ma-pruning-20261026/matrix.json`. Both variants
run the same e9 content pilot with 16 workers; `prune` adds only `--ma-prune`.
Output pixel checks, size, bpp and complete-call timing must all be reported;
micro-level proof counts alone do not establish an end-to-end speedup.

### First Slow-Case Comparison

`out/ma-pruning-20261026/slow/report.json`: one uninstrumented run per variant,
same release executable, complete e9 pilot, 16 workers. All six outputs passed
official original-pixel verification. No sampling/leaf-budget reductions.

| Image | Off s | On s | Time change | On MPix/s | bpp | Bytes (both) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floral_frame_PNG270.png | 19.990131 | 20.625988 | +3.18% | 0.875678 | 2.259042 | 5100273 |
| pizza_PNG44037.png | 33.630939 | 31.125198 | -7.45% | 0.411988 | 11.116226 | 17818198 |
| wallhaven-mlzoy1.png | 23.430872 | 22.996106 | -1.86% | 0.360687 | 13.675351 | 14178604 |

Pruning is **not promoted to a default**: floral is slower in this comparison,
and the small differences need repeated measurements. The useful pizza result
does not establish a universal speedup. Whole-predictor dominance still builds
the predictor histogram before proving its entropy sweep unnecessary; this
initial version does not eliminate all histogram collection or MA search.

### Diverse Holdout and Route Check

`out/ma-pruning-20261026/holdout/report.json` checks 12 existing holdout images:
icons, textures, natural photos, a screenshot, manga, transparent PNG and 16-bit
patterns. All 24 outputs pass original-sample verification. One run per variant:

| Variant | Total call s | Aggregate MPix/s | Total bytes | Weighted bpp |
| --- | ---: | ---: | ---: | ---: |
| Off | 14.675338 | 1.683675 | 14802530 | 4.792693 |
| On | 14.336619 | 1.723454 | 14802530 | 4.792693 |

Aggregate time decreases 2.31%. Some individual cases are still slower, including
0109.png (1.030607 -> 1.079248 s) and the 16-bit triangle pattern
(0.357029 -> 0.369595 s). This supports keeping the route experimental rather
than claiming uniform improvement.

`profile/report.json` checks one texture with `RS_JXL_PROFILE=1`. The pruned run
has nonzero proof hits: maximum printed per-learner snapshots include 9,043
dominated predictor sweeps, 29,946 inherited property skips, 191 duplicate
properties and 1,045 constant properties. These maxima come from different
snapshots and MUST NOT be added as a whole-image total. The disabled run has
zero pruning counters.

The node-level `ma.find_best_split` and `ma.split_sweep` scope counts remain
30,659 and 29,620 respectively. This removes work INSIDE node search, not repeated
execution of identical nodes. Inclusive/concurrent split-sweep stage time is
3.392831 -> 3.065122 s; it is not serial wall time. Instrumented complete calls
are 1.151426 -> 1.121466 s and are excluded from the uninstrumented speed tables.

`cli-smoke/report.json` additionally passes normal-CLI `--ma-prune` 16-bit
decoding and unchanged JPEG-transcode byte-exact reconstruction through official
djxl. The new CLI flag is for pixel encoding; JPEG-transcode options are unchanged.
