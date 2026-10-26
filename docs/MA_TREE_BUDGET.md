# MA Tree Budget

`--ma-budget-mode strict|bounded` is available on normal `encode` (manual,
source candidate/search/scout/pilot) and `encoder_lab` (including content pilot).

- `strict`: return a limit error if a beneficial adaptive split remains when
  the learned tree reaches its leaf ceiling.
- `bounded`: stop adaptive splitting at the ceiling and keep valid lossless
  predictor leaves. Compression size may increase; pixel values do not change.
- Omitted: preserve existing policy, strict for source sampling and bounded
  for legacy/direct training. This mode does not change quality presets or
  sampling fractions. Private resource ceilings have separately been removed;
  see the sample-limit repair below.

Required multiplier partitions, enabled channel/row partitions, and explicitly
requested constant leaves must fit in either mode. Bounded mode does not prune
finished trees, remove constraints, reduce sampling, or suppress other errors.
With no explicit mode, legacy optional constant bypass retains its old behavior.

`--ma-max-leaves N` adds a learned-tree cap, `1..=2097152`, including constant
leaves and channel/subtree grafts. It is intersected with existing resource and
decoder geometry limits, so cannot raise them. Unlike manual `--tree-leaves`,
it survives source scheduler projection and applies to probes, finalists and
patch references. Fixed/custom trees are not learned and are not changed.

```text
rs-jxl encode in.png out.jxl --source-effort 9 --source-search --source-pilot --ma-budget-mode bounded --ma-max-leaves 4096
encoder_lab in.png --search --content-pilot --ma-budget-mode bounded --ma-max-leaves 4096
```

Library controls:

```rust
use rs_jxl::modular::ma_train::MaBudgetMode;
let mut options = rs_jxl::EncodeOptions::default();
options.modular.ma_budget_mode = Some(MaBudgetMode::Bounded);
options.modular.ma_max_leaves = Some(4096);
```

Direct trainers use `TrainOptions::budget_mode: Option<MaBudgetMode>` with the
existing `max_leaves` / `max_merged_leaves` caps. `None` respects the existing
`require_complete_search` boolean; an explicit mode overrides it.

Bounded channel roots and detached subtrees consume remaining leaf capacity in
stable order, reserving every other seed leaf before learning and checking every
graft. Property/ready-child searches may still run in parallel. Detached search
can allocate capacity differently from serial DFS, so tree bytes need not match.
Strict independent-root parallelism is unchanged. Optional stream-tree grouping
still has its own hard merged-tree limit; bounded mode does not relax that check.

Source scheduler clones retain these controls. Scheduler resource reports
describe their projected ceilings, not this additional cap; the effective
learner ceiling is their minimum.

Dedicated tests cover defaults/overrides, tiny serial/parallel caps, mandatory
partitions, detached seed retention, source/reference propagation, CLI parsing,
and local/shared pixel roundtrips (including invisible RGB). Run centrally with
the `ma_budget_mode` library filter and `ma_budget` CLI/example filters.

## Independent Decoder Check

On 2026-10-26, `bounded64-v3/report.json` under
`out/corpus-repair-20261026/` verified three real inputs at a deliberately small
64-leaf cap with official `djxl`: `hlog.png`, the 16-bit circles fixture, and
the formerly failing Wikipedia `004.png`. All decoded samples matched.
The photo took 0.558 seconds instead of the default policy's 1.699 seconds,
but grew from 1723875 to 1731350 bytes (+0.43%). The circles output became
slightly smaller instead. Fewer leaves can reduce model overhead; bounded mode
does not promise either a fixed size penalty or a speedup on every image.
These explicit-cap trials are separate from default-policy performance results.

`bounded-large-v3/report.json` compares explicit caps on three slow images;
all six outputs also pass official pixel verification. These are one-repetition
measurements of complete encoder calls at effort 9 and 16 threads:

| Image | Default bytes / s | 4096 leaves bytes / s | 16384 leaves bytes / s | 4096 leaves bpp |
| --- | ---: | ---: | ---: | ---: |
| pizza_PNG44037 | 17818198 / 35.743 | 17431467 / 30.401 | 17818198 / 35.854 | 10.8750 |
| wallhaven-mlzoy1 | 14178604 / 24.719 | 13989841 / 16.155 | 14178604 / 27.769 | 13.4933 |
| floral_frame_PNG270 | 5100273 / 21.160 | 5230887 / 21.348 | 5100273 / 30.828 | 2.3169 |

The 4096 cap reduces wallhaven time to about 65.4% while also shrinking its
output by 1.33%. But floral becomes 2.56% larger without a speed benefit.
The 16384 bounded setting can be slower even with equal output size: bounded
independent roots consume the global leaf budget in stable order instead of
using the strict mode's concurrent root allowances. These results do not
justify a universal cap or replacing the default strict source policy.

The installed CLI was checked in `cli-smoke-v3/`: PNG with the new bounded
flags passed original-sample verification; JPEG transcode passed original-byte
reconstruction through official djxl. A separate one-leaf strict invocation
returned exit 1 with `MA leaf budget exhausted with beneficial splits remaining`,
as intended. Unit tests additionally check that failing operations do not turn
into partially successful outputs.

## Source-Effort 10 Sample-Limit Repair

The leaf-budget repair above does not fix `grouped MA selected-sample budget
exceeded`. That error is a separate hard ceiling on gathered training samples,
before adaptive tree learning. `--ma-budget-mode bounded` only limits splitting;
it does not cap or reduce sampling and cannot resolve this sample-limit error.

The CLI and `ModularOptions` previously defaulted to 1048576 selected samples.
Both source effort 9 and 10 use a raw learning fraction of 0.5 scaled by 1.3
(approximately 65%). A selected sample is a transformed channel value, not one
whole RGB pixel. Palette transforms and constant channels affect the population.

With automatic buffering, nonresponsive effort 10 selects a shared full-frame
tree, whereas effort 9 can stream frame areas when there are more than eight
groups and forced patches do not prohibit streaming. Therefore an image can
fit effort 9's per-learning-domain limit but exceed the same ceiling in effort
10's full-frame learning. The error text also serves the gather's local atomic
budget; it does not by itself imply erroneous double charging across workers.

On 2026-10-26 the pre-repair release CLI reproduced this on DIV2K `0716.png`
(2040x2040, 8-bit), using 16 threads, `--strip-metadata`, no source search, and
no budget override:

- `--source-effort 9`: success, 4187169 bytes; official libjxl `djxl` reproduced
  every original pixel sample.
- `--source-effort 10`: exit 1, `grouped MA selected-sample budget exceeded`;
  no output file was created.

This ceiling was introduced by the Rust port, not by the JXL format. Local
official `v0.12.0` source (`enc_encoding.cc::GatherTreeData`,
`enc_ma.cc::PrepareForSamples`/`AddSample`) samples according to the requested
fraction and grows/deduplicates storage without that fixed selected-count cap.

The repaired library uses `ModularOptions::sample_limit: Option<usize>`, default
`None`. The normal CLI and `encoder_lab` now agree: no hidden default override,
no automatic cap and no sampling reduction. An explicit
`--source-sample-limit N` / `Some(N)` still errors when exceeded, including an
explicit 1048576. Checked arithmetic and allocation errors remain enforced.

The former 16384 maximum configurable learner leaves and 131072 grouped-tree
ceiling are also removed. Their allowed maximum is now the decoder tree bound;
source integration still applies the tighter geometry-dependent global/local
tree rules. Explicit caller leaf budgets and legacy quality presets remain.

See [the resource-limit audit](RESOURCE_LIMIT_AUDIT.md) for the broader repair
and its verification results. No claim is made that the original 67-image
report corpus has been rerun unless listed there.
