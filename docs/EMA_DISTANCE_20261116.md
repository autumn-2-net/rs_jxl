# EMA clustering: iteration and distance follow-up

This follows `EMA_CLUSTER_EXPERIMENT_20261116.md`. All controls are optional;
ordinary effort presets stay unchanged. No decoder syntax changes or new
dependencies are involved. Final acceptance compares decoded integer samples,
not byte-identical JXL streams.

## Questions and controls

1. Does extending the original cross-entropy EMA to 12 or 32 passes recover
   its file-size deficit?
2. Does one assignment against frozen final centers repair stale online labels?
3. Can another distance represent useful histogram distinctions better?
4. Does a local native ANS-cost refinement improve the resulting partition?

The original recipe changes batch size and learning rate as a function of
pass count, so 3/12/32 are complete operating points, not a pure epoch-only
ablation with a fixed schedule. Distance comparisons use the same 12-pass
recipe, random seeds and center count. Every candidate performs one complete
encode, not an internal best-of tournament.

## Distances

- `cross-entropy`: original smoothed forward-KL equivalent. Missing support
  costs roughly 40 bits per observation under its probability floor.
- `hellinger`: squared Hellinger assignment, maximizing normalized square-root
  affinity. Its centroid stores token-weighted square-root sufficient
  statistics and normalizes their squared values to a probability vector.
  Population weights are tracked separately. Revival splits both root
  statistics and population in half; it does not apply incompatible per-bin
  perturbations. This preserves the root/population invariant but identical
  donor children can remain tied, a limitation of this experimental recipe.
  This is not arithmetic count
  averaging with a different metric bolted on.
- `merge-entropy`: native entropy increase from combining the input histogram
  with a globally rescaled EMA center. Center populations are projected to
  integer counts for reuse of the existing SIMD scoring kernel. Projections
  conserve the input population across centers. Per-bin floor remainder goes
  to the largest occupied bin; small-support rounding remains an approximation.
  These counts only score proposals and are never serialized as observations.

EMA's per-batch distributions and population estimates remain noisy; the
alternative objectives do not guarantee monotonic full-file improvement.
More iterations alone cannot change the mismatch between a proxy objective
and the complete JXL representation.

## Final assignment and relocation

`--ema-final-assignment` freezes the final centers, reassigns each nonempty
input once, and then rebuilds exact original integer counts. This is optional
because the previous experiment deliberately used last-online-pass labels.

`--ema-polish N` performs up to N ordered coordinate passes before ordinary
ANS merge refinement. Candidate destinations are the four best coding-price
proposals plus an available empty center. Each accepted move must reduce
`F(source - input) + F(target + input)`, where F is the existing native Fast
ANS population cost including table description. An emptied table costs zero
because it will be removed. Current costs are updated after every committed
move; no parallel commits use stale deltas. Passes stop when no move improves.

This is an actual native table-cost comparison, but not an exact full-file
objective: context-map signaling and other normalization policies are outside
it. Final bytes and timing must still be measured. The shortlist is heuristic;
it does not promise a globally optimal partition.

## Reproduction

```text
cargo build --release -p rs-jxl-cli --example portfolio_lab
portfolio_lab INPUT.png --output OUTPUT.jxl --effort 9 --threads 8 --direct-candidate baseline --ema-passes 12 --ema-distance hellinger --ema-final-assignment
```

The paired harness retains `--ema NAME=PASSES:random|farthest` and adds
`--ema-distance NAME=MODE`, `--ema-final-assignment NAME`, and
`--ema-polish NAME=PASSES`. Each optional control requires an EMA-enabled
variant; native JSON receipts are checked against requested settings.

Experimental artifacts live in the sibling workspace's ignored
`out/emeans-distance-20261116/`. Native timing excludes input decoding,
file I/O, process startup and independent verification. Profiling is separate
from timing runs; parallel/inclusive stage sums are not wall-time fractions.

## Eight-image screen

Full-resolution 1515 HDR, DIV2K 0446, editor/paper screenshots, an icon,
pixel-art example, transparent particle and natural texture. All e9, eight
threads, one interleaved observation per configuration. All 64 outputs pass
official djxl pixel verification. Timings are exploratory point estimates.

| Method | Total bytes | Change vs Best | Mean bpp | Native seconds | MPix/s |
|---|---:|---:|---:|---:|---:|
| Best | 35,170,216 | baseline | 8.463107 | 20.4834 | 1.0057 |
| Original CE, 3 passes | 35,215,678 | +0.12926% | 8.476917 | 17.2981 | 1.1909 |
| Original CE, 12 passes | 35,195,681 | +0.07241% | 8.469839 | 19.8691 | 1.0368 |
| Original CE, 32 passes | 35,193,239 | +0.06546% | 8.469384 | 22.1051 | 0.9319 |
| CE 12, frozen final assignment | 35,195,684 | +0.07241% | 8.469838 | 20.0072 | 1.0296 |
| Hellinger 12, frozen final assignment | 35,180,845 | +0.03022% | 8.466596 | 19.8489 | 1.0379 |
| Merge entropy 12, frozen final assignment | 35,177,018 | +0.01934% | 8.467126 | 21.9895 | 0.9368 |
| CE 12, one native-cost relocation pass | 35,178,629 | +0.02392% | 8.464680 | 19.1834 | 1.0739 |

The 12-to-32 extension saves only 2,442 bytes while taking about 11.25% more
native time. Frozen final assignment changes only three aggregate bytes,
so stale last-pass labels alone do not explain the deficit. Changing the
objective helps more: merge entropy removes about 73% of the CE-12 excess
bytes, and native relocation about 67%. None beats Best in total bytes.

Per-image regressions remain relevant. On the paper screenshot Best is
354,791 B; CE 3 is 362,546 B, CE 12 is 358,019 B, CE 32 is 358,017 B,
Hellinger 12 is 355,424 B, and native relocation is 355,730 B. More CE
passes almost stop helping; a different objective closes substantially more
of the gap. Native relocation beats Best on three inputs, but their savings
are only 11, 10 and 4 bytes, not large compression gains.

## Three-pass mixed-image check

Repeat the previous experiment's 16 original images, including two HDR
inputs, a long webpage, manga, photographs, screenshots, palette/pixel art,
alpha and 16-bit textures. One interleaved observation, e9, eight threads.
Every output passes official djxl verification (80 additional checks).

| Method | Total bytes | Change vs Best | Mean bpp | Weighted bpp | Native seconds | MPix/s |
|---|---:|---:|---:|---:|---:|---:|
| Best | 81,634,536 | baseline | 8.004119 | 8.093807 | 60.0654 | 1.3433 |
| Original CE 3 | 81,778,595 | +0.17647% | 8.014762 | 8.108090 | 59.4631 | 1.3570 |
| Hellinger 3, frozen final assignment | 81,703,634 | +0.08464% | 8.008210 | 8.100658 | 56.9858 | 1.4159 |
| Merge entropy 3, frozen final assignment | 81,760,559 | +0.15437% | 8.012878 | 8.106302 | 55.7290 | 1.4479 |
| CE 3, one native-cost relocation pass | 81,648,691 | +0.01734% | 8.005551 | 8.095211 | 54.5154 | 1.4801 |

EMA runs on 14/16 inputs; two keep the original fallback. Their time changes
are control noise, not acceleration. The strongest measured tradeoff is
CE 3 plus one native-cost relocation pass: 129,904 bytes smaller than the
original CE 3 result, removing about 90.17% of its excess over Best. It is
still 14,155 bytes larger than Best overall. Four files are smaller, ten
larger, two unchanged; this is not a universally better partition.

| Input | Best bytes | Original CE 3 | CE 3 + relocation |
|---|---:|---:|---:|
| HDR 1515 | 26,124,323 | 26,157,502 | 26,131,701 |
| HDR 1244 | 25,645,233 | 25,685,209 | 25,651,345 |
| Long webpage | 2,917,250 | 2,961,456 | 2,914,556 |
| Editor screenshot | 117,483 | 120,391 | 117,945 |
| Paper screenshot | 354,791 | 362,546 | 356,516 |
| Sublime screenshot | 612,884 | 625,547 | 613,668 |

Timing requires caution. Original CE 3 took 52.8584 seconds in the preceding
16-image experiment but 59.4631 seconds here, with the same output sizes;
the corresponding Best times were 58.1845 and 60.0654 seconds. The present
relocation candidate is 9.24% faster than its interleaved Best control, but
these are not confidence intervals or a proven general speedup. Smaller
reported differences should not guide defaults without repeated paired runs.

## Interpretation

Changing the objective matters more here than repeatedly optimizing the
original one. The useful follow-up is native coding-cost correction, not an
assertion that EMA has become a better compressor than Best. Square-root
distance improves the original EMA size deficit too, but less than relocation.
The final frozen-center sweep alone has essentially no benefit on the screen.
Keep all variants opt-in; no ordinary effort preset is changed.

## Verification and remaining limits

- Release example build passed after review fixes.
- `cargo test -p rs-jxl --lib entropy:: -- --test-threads=4`: 191 passed.
- `python -B -m unittest discover -s tools -p test_benchmark_ema.py -v`: 14 passed.
- 64 screen encodes plus 80 mixed-set encodes: 144 official-decoder pixel
  comparisons passed, including hidden RGB, alpha and 16-bit samples.
- Tests exercise alternate-distance serial/parallel agreement, integer-count
  conservation, Hellinger barycenter/revival invariants, projection population,
  missing-support behavior, native-cost relocation and trimmed alphabets.

Measured full-image timings in this follow-up use eight threads only. Worker
equivalence tests are correctness coverage, not evidence of single-thread
performance. No full-corpus run or default-policy change was made. The new
CLI parser unit test is present, but was not separately run in this follow-up;
all benchmark variants exercised the built release CLI and validated receipts.
