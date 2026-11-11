# Ordinary e9 Cross-Channel MA

This is distinct from the shallow `fast_context` cross-channel experiment.
The existing full MA learner can already use previous compatible transformed
channels. The direct native source candidate defaults `max_properties` to zero.
This experiment changes only that budget, leaving ordinary e9, RCT selection,
MA sampling, palette policy, LZ77 and entropy coding intact.

## Decision Space

At e9 (source speed tier 1), each previous compatible channel supplies four
standard properties: absolute sample, signed sample, absolute clamped-gradient
residual, and signed residual. Budgets 1 and 2 append properties 16..19 and
16..23 respectively to the ordinary search. These are legal JXL context
properties, not new predictors or changes to decoded sample semantics.

The budget is a ceiling, not a guarantee that the trained tree will use every
property. Source projection limits it to available channels; stream geometry
and transformed channel layout determine which references exist. Greedy MA
training with more candidate properties can change the partitioning and is
not guaranteed to produce a smaller final file.

The lab accepts `--previous-channels 0..60` for an explicit native candidate.
JSON reports `max_previous_channels` (requested) and `ma_previous_channels`
(projected). It is not mixed with `--context-cross-channel` or outer search.
No default is changed. No Python ABI extension is part of this experiment.

## Reproduction

```powershell
python -B tools/benchmark_ma_cross.py --encoder out/channel-experiments-20261112/encoder-lab.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --manifest out/ma-leaf-offset-distribution-20261111/sample30.json --effort 9 --threads 8 --repeats 1 --out out/channel-experiments-20261112/ma-cross30.json
```

The three variants are ordinary baseline, `ma-prev1`, and `ma-prev2`. The
harness runs them serially in rotated order, verifies original 8/16-bit samples
with official djxl, includes invisible RGB, checks requested/actual options,
and records complete bytes, weighted bpp and native-call time. Startup, input
PNG loading and output IO are excluded from native time. Synthetic gray/RGB/
RGBA fixtures can be added with `--synthetic`. Single observations are useful
for a size screen, not proof of precise speed differences.

## Results

Frozen binary SHA256:
`8f5c4d554caa8e6c638dfbdced6bd02928c5aa2afcc20908025c465eae4c1e99`.
Report: `out/channel-experiments-20261112/ma-cross30-focused.json`.
The run additionally includes the PDF/VSCode/4K focused images and eight
synthetic images: 41 images x 3 variants = 123 official pixel checks, all passed.
The table below includes **only the original 30-image corpus**, 68,188,326 pixels.
It measures our native direct e9, not official libjxl or the outer scheduler.

| MA previous channels | Bytes | Weighted bpp | Native seconds | MPix/s | Smaller / equal / larger |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 (ordinary baseline) | 35,390,526 | 4.152092 | 37.277 | 1.829 | - |
| 1 | 35,063,937 | 4.113776 | 40.024 | 1.704 | 22 / 7 / 1 |
| 2 | 34,952,156 | 4.100662 | 43.682 | 1.561 | 22 / 7 / 1 |

One prior channel saves 0.923% at 7.37% more observed time; two save 1.239%
at 17.18% more time. `baby.png` is the corpus regression: 221,383 bytes grows
by 32 or 7 bytes. Extra properties do not monotonically improve the final
serialized output of greedy tree/entropy decisions.

| Focused image / previous channels | Bytes | bpp | Native seconds |
| --- | ---: | ---: | ---: |
| PDF `9bde...`, 0 | 412,539 | 2.266499 | 1.874 |
| PDF `9bde...`, 1 | 381,662 | 2.096860 | 1.861 |
| PDF `9bde...`, 2 | 379,961 | 2.087514 | 2.540 |
| VSCode `3dd...`, 0 | 126,447 | 0.473243 | 0.514 |
| VSCode `3dd...`, 1 | 126,447 | 0.473243 | 0.543 |
| VSCode `3dd...`, 2 | 126,447 | 0.473243 | 0.607 |
| 4K `wallhaven-mlzoy1.png`, 0 | 14,178,604 | 13.675351 | 6.221 |
| 4K `wallhaven-mlzoy1.png`, 1 | 14,182,905 | 13.679499 | 7.522 |
| 4K `wallhaven-mlzoy1.png`, 2 | 14,180,893 | 13.677559 | 8.476 |

The PDF saves 7.48% with one previous channel without a material observed time
change in this pass. The 4K photo is both slightly larger and slower. Among
the corpus documents, `5020.png` saves 2.46% with one prior channel
(6.573 -> 6.875s), and `7060.png` saves 5.47% (0.968 -> 1.183s).
These are reasons to retain an explicit candidate, not enable it indiscriminately.

## Interpretation

Full e9 already has the implementation; the experiment expands its decision
space, unlike the new property-18-only shallow axis. It improves real bpp but
still costs more overall. Prefer a one-channel trial before two when investigating
specific correlated content. A future automatic selector must measure admission
cost and misses rather than infer savings from RGB correlation alone: RCT and
palettes change the channels actually presented to MA. No auto route is promoted
from these 30 samples.
