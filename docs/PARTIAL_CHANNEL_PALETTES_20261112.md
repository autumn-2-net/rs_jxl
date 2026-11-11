# Partial-Channel Palette Experiment

Native Rust, lossless only, opt-in. This experiment changes which channels
share a palette, not the ordering heuristic of a whole-image palette. It uses
existing JXL RCT and Palette syntax; no decoder extension or new predictor.

## Candidates

| Lab mode | Forward representation |
| --- | --- |
| `rg` | Palette(R,G), independent B (and alpha) |
| `yco` | RCT6, Palette(Y,Co), independent Cg (and alpha) |
| `ycg` | RCT6, RCT21 permutation, Palette(Y,Cg), independent Co (and alpha) |
| `cocg` | RCT6, independent Y, Palette(Co,Cg) (and alpha) |
| `ya-cocg` | RCT6, RCT14 on channels 1..3, Palette(Co,Cg), Palette(Y,A) |

The last script first permutes `[Y,Co,Cg,A]` to `[Y,A,Co,Cg]`. Its two palette
transforms use the updated metadata prefix after each table insertion. Inverse
transforms restore the original channel order, including invisible RGB.

## Admission And Cost

Before cloning or transforming the planes, scan original integer pixels and
compute exact joint cardinality of the requested pairs using virtual RCT6.
Abort as soon as either pair exceeds the cap. This bounds the tuple set, not
the input size or encoder's legal capabilities. All pixels participate, even
under zero alpha; chroma remains signed. If admitted, construct one candidate
transactionally. A transform decline leaves the ordinary encoding route intact.

Joint cardinality is only an eligibility check, **not a prediction of savings**.
An admitted candidate may be larger or slower. Compare complete JXL byte sizes
against the ordinary baseline before considering automatic routing. No ordinary
production default has changed, and the experiment does not claim an automatic
best-of policy or a size guarantee.

Accepted scripts disable subsequent ordinary palette/RCT selection on their
index channels. MA, weighted-predictor and entropy coding stay native. Declined
scripts keep the normal source policy. Admission itself is included in timing.

## Interfaces

Rust: `modular.palette_policy = Some(PaletteOptions { grouping: Some(Grouping),
.. })`. `modular.palette_group_receipt` optionally records attempts, accepted
scripts and table counts. A receipt bypasses prepared-layout caching so counters
are not silently lost; the grouping is part of the ordinary policy cache key.

Lab:

```powershell
cargo build --release -p rs-jxl-cli --example encoder_lab -j 4
target\release\examples\encoder_lab.exe input.png --effort 9 --threads 8 --buffering 0 --palette-grouping ya-cocg --palette-group-colors 4096 --output candidate.jxl
```

`--palette-group-colors` also works without `--palette-grouping`, for a paired
ordinary baseline using the same cap. No files or Python preprocessing are
required by the native transform API. The lab accepts image paths for testing.

Initial eligibility: integer RGB/RGBA, equal depth and full-resolution planes;
`ya-cocg` additionally requires an actual Alpha extra channel. Float, gray,
subsampled or unequal-depth extras decline to the ordinary route. Existing
palette-kernel depth/headroom constraints still apply. Groupwise/responsive,
delta palettes, other palette-order experiments and the contiguous pre-RCT
representation receipt are explicitly rejected for this experimental policy,
not silently ignored. Standard metadata and alpha association are not rewritten.

## Validation

Tests cover transform inverse order, metadata-prefix shifts, cap rejection
without mutation, signed chroma, 8/16-bit samples, hidden RGB, and complete
native encode/decode across multiple groups. Sixteen focused native tests pass,
including suppression of pre-existing global/local RCT options. Performance
trials additionally use official djxl as an independent pixel oracle.

## Measurements

Frozen binary `out/channel-experiments-20261112/encoder-lab.exe`, SHA256
`8f5c4d554caa8e6c638dfbdced6bd02928c5aa2afcc20908025c465eae4c1e99`.
Report `out/channel-experiments-20261112/palette-groups-focused.json`:
5 real images + 26 synthetic fixtures, 6 variants each, **186 official pixel
checks passed**. Serial rotated trials, e9/T8, one observation per variant.

All variants use cap 4096 and full-frame buffering, including the baseline.
These controlled settings differ from the ordinary e9 cross-channel report;
do not attribute their baseline differences to the new grouping algorithm.

The smallest explicit group candidate is shown below, even when it loses.
For equal-size declined candidates, the displayed mode is arbitrary.

| Image | Baseline bytes / bpp | Group | Candidate bytes / bpp | Seconds baseline -> candidate | MPix/s baseline -> candidate |
| --- | --- | --- | --- | --- | --- |
| PDF `9bde...` | 225,195 / 1.237226 | rg | 300,291 / 1.649806 | 2.026 -> 3.069 | 0.719 -> 0.474 |
| VSCode `3dd...` | 104,193 / 0.389954 | yco | 120,512 / 0.451030 | 1.719 -> 3.021 | 1.244 -> 0.707 |
| `wallhaven-mlzoy1.png` | 13,829,699 / 13.338830 | cocg (declined) | same | 19.044 -> 19.554 | 0.436 -> 0.424 |
| Transparent `muzzle_05.png` | 21,641 / 0.660431 | ya-cocg | 36,645 / 1.118317 | 0.158 -> 0.254 | 1.658 -> 1.032 |
| `TMW_Screenshot_15.png` | 541,831 / 9.030517 | cocg (declined) | same | 2.806 -> 2.775 | 0.171 -> 0.173 |

No real image wins. The PDF and VSCode have accepted scripts yet grow; the
transparent material grows too. All grouping modes decline on the 4K photo
and game screenshot, leaving the same ordinary result. The 4K baseline itself
takes 19 seconds with this cap; that is not grouped-palette execution time
and is not a reason to change its ordinary e9 default.

Nine of the 26 synthetic fixtures have at least one smaller group candidate.
For example the 16-bit synthetic RGB goes from 3,509 to 1,925 bytes with yco;
RGBA goes from 3,530 to 1,979. Other fixtures grow substantially. These cases
demonstrate representability and potential, not real-corpus generalization.
Counting the best of five after encoding all five is also an oracle, not a
free preflight decision.

The source patch policy is left unchanged. Some real inputs contain a patch
reference plus a residual frame; diagnostics count script applications per
preparation, not logical images, so `applied=2` can be valid. All serialized
frames are included in final byte counts and timing.

Decision: retain an explicit research candidate only, **do not add automatic
grouping or exhaustive grouping search**. Low joint cardinality alone is not
enough: it ignores index residuals, extra table/tree costs and correlations
already handled by ordinary MA. Current real-image evidence favors the cheaper
ordinary-e9 cross-channel experiment over this grouping route.

Reproduction (add `--case` or `--manifest` for real images):

```powershell
python -B tools/benchmark_palette_groups.py --encoder out/channel-experiments-20261112/encoder-lab.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --synthetic --threads 8 --repeats 1 --out out/channel-experiments-20261112/palette-groups-synthetic-new.json
```
