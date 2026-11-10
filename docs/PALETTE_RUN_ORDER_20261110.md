# Palette Run-Order Experiment

## Scope and Provenance

This opt-in native Rust experiment borrows the run-start statistic from
[QLIC qlic.c](https://github.com/OzzyMitchell/QLIC/blob/f507ac0343b77c8cbaee1bc1d99920c694c25e4f/codec/src/qlic.c),
`sort_palette_map_by_hot_runs` and `make_exact_cpal_payload`. Inspected commit:
`f507ac0343b77c8cbaee1bc1d99920c694c25e4f`; file blob:
`4059e3bb984f59be51b7f84d4969482e4df144a3`. QLIC identifies its license as
[Apache-2.0](https://github.com/OzzyMitchell/QLIC/blob/f507ac0343b77c8cbaee1bc1d99920c694c25e4f/LICENSE).

QLIC's hot set promotes 128 colors selected by raster run-start counts and
retains lexical ordering within the two partitions. Its CPAL run IDs use a
variable-length integer representation with a size boundary at 128. This is
not a JXL boundary: JXL predicts and entropy-codes indices. The implementation
here is an independent Rust adaptation of the statistic, not a C translation
or import of QLIC's payload, predictor, entropy format or runtime dependency.
Existing libjxl-derived code retains its attribution.

Rust base: `31cc87e`. No decoder, JPEG transcode, lossy encoder, production
default scheduler, numeric source selector or Python ABI changes. The native
ordinary palette acceptance/cost policy remains in force; selecting an order
does not force a palette that would otherwise be declined. There is no dual
full-encode size guard or automatic tournament in the new ordering helper.

## Native API

```rust
use rs_jxl::modular::palette::{PaletteOptions, PaletteOrder, PaletteTools};

let policy = PaletteOptions {
    tools: PaletteTools {
        order: PaletteOrder::HotRuns { hot_colors: 128 },
        ..Default::default()
    },
    ..Default::default()
};
// Install in EncodeOptions.modular.palette_policy.
```

- `RunFrequency`: descending count of raster runs started by a full color tuple,
  then signed tuple lexical order, then original candidate position.
- `HotRuns { hot_colors: k }`: rank by the same score, select the first `k`
  explicit entries, then lexical order within the hot and cold partitions.
  `k=0` or `k>=explicit_count` is plain lexical order and avoids the run scan.
  There is no fixed or automatically selected 128-entry cutoff.
- The first pixel starts a run. Equal tuples across a row boundary remain in
  the same run. Every participating channel is compared, including alpha and
  RGB under zero alpha. Non-promoted implicit colors break runs but are not
  themselves explicit candidates.
- The existing >=3-channel ordered absolute-palette domain is retained.
  Single-channel palettes stay sorted; two-channel palettes do not reorder.
  `ordered=false` bypasses ordering. Existing geometry takes precedence for
  3..4 channels; >4 channels retain full signed tuple comparisons and tree-map
  fallback. The old `from_source_mode(0..9)` contract is unchanged.
- Aliased implicit promotions retain duplicate entries and last-assignment-wins
  mapping. They share a color's run score; tie-breaking is deterministic.
  Promotion still uses the original pixel-count threshold, not run counts.
- Successful exact-delta palettes keep their original residual-frequency
  ordering. Only their absolute fallback uses the selected new order. Exact-zero
  palettes add their delta prefix after absolute ordering and shift indices by
  the existing rule. No implicit or delta wire indices are arbitrarily permuted.

Implementation is isolated in `modular/palette_runs.rs`, dispatched from
`modular/palette_order.rs`. It adds no fields to the ordinary palette hash/map
entries and performs no run scan for existing modes. With an active new order,
the additional cost is one raster scan, O(palette entries) counters and sorting.
It borrows the existing map's scratch indices, then the original final mapping
stage assigns all explicit indices. No sample values are changed or quantized.

## Reproduce

```powershell
cargo build --release -p rs-jxl-cli --example palette_lab
target/release/examples/palette_lab input.png --output trial.jxl --effort 9 --threads 8 --mode hot --hot-colors 128
python tools/benchmark_palette_runs.py --encoder target/release/examples/palette_lab.exe --decoder PATH_TO_OFFICIAL_DJXL --manifest cases.json --modes default frequency runs hot0 hot32 hot128 hot512 --effort 9 --threads 8 --repeats 2 --out out/palette-results.json
```

The lab encodes one native candidate per process, not the outer scheduler search.
Non-palette settings are held constant and all patch detectors are disabled for
every variant. Optional `--palette-colors` and `--full-frame` overrides apply to
all compared modes and must be reported separately from projected-route results.
They never imply that every image accepts a palette.

Native timing includes encoding, thread-pool setup and active palette sorting;
excludes process startup, input decoding, file I/O and independent verification.
Receipts report actual global tuple palettes and the full-image/groupwise route;
they do not cover local palettes or single-channel compaction. A configured
order is not proof that a particular transform was emitted. The harness checks
original PNG samples with official djxl, including hidden RGB, and removes
temporary JXL/PNG outputs. It never requires identical encoded JXL bytes.

## Measurements

Windows x86_64, rustc 1.97.1, release profile, e9, 8 native threads. Independent
oracle: `djxl v0.12.0 a1ebdf2` (Clang 17.0.6). The lab executable SHA256 was
`06a3c67976ab5d77b663582a05c2d3861bda99d392a962e91975cfad8b9f7aff`;
the oracle SHA256 was
`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.
Hashes identify measured binaries, not desired encoded outputs.

Reports under ignored `out/palette-runs-20261110/`:

- `cases.json`: 16 real PNGs chosen before measuring any order. Includes game
  assets, a palette chart, paper/editor/web screenshots, a transparent cursor,
  illustration, manga, a photograph and 16-bit displacement control. Inputs
  were not resized or quantized. Total: 26,336,633 pixels.
- `real-e9.json`: 16 images x 5 orders x 2 rotated repeats = 160 pixel checks.
- `synthetic-e9.json`: 6 format/run-skew fixtures x 7 orders = 42 pixel checks.
- `wallhaven-timing-recheck.json`: 1 image x 2 orders x 5 repeats = 10 checks.
- `full-frame-diagnostic-e9.json`: 4 images x 7 orders = 28 pixel checks.

### Normal Projected Route

Two repeats per order; reported medians are the midpoint of two measurements,
not confidence intervals. `Default` is the same fixed projected candidate in
every comparison, not the outer production scheduler's best-of-search output.
No full-frame/cap override is used in this phase.

| Order | Total bytes | Change | Sum of median seconds | bpp | MPix/s | Smaller / equal / larger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 6,614,347 | baseline | 12.807 | 2.0092 | 2.056 | - |
| FrequencyLuma | 6,650,247 | +0.543% | 12.671 | 2.0201 | 2.078 | 3 / 6 / 7 |
| RunFrequency | 6,632,430 | +0.273% | 13.116 | 2.0147 | 2.008 | 5 / 6 / 5 |
| Hot0 / lexical | 6,612,140 | -0.033% | 12.629 | 2.0085 | 2.085 | 6 / 6 / 4 |
| Hot128 | 6,612,568 | -0.027% | 15.811* | 2.0086 | 1.666* | 6 / 6 / 4 |

`*` The raw Hot128 aggregate includes an unreproduced 7.109s wallhaven trial.
It must not be interpreted as stable algorithmic slowdown; see the recheck below.
Other small time differences are also noisy. Size differences are repeatable in
these trials; this is not a promise of equal encoded bytes for future changes.

| Image | Default bytes | Pixel frequency | Run frequency | Lexical | Hot128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 358colors | 1,294 | 1,294 | 1,294 | 1,294 | 1,294 |
| Kenney Preview | 57,072 | 61,396 | 60,262 | 54,700 | 53,809 |
| Kenney tilemap | 5,136 | 5,134 | 5,054 | 5,051 | 5,051 |
| Cursor GA | 14,678 | 14,678 | 14,678 | 14,678 | 14,678 |
| Sublime screenshot | 661,313 | 684,211 | 673,582 | 660,296 | 655,328 |
| Hacker News screenshot | 110,876 | 118,445 | 117,410 | 111,784 | 115,123 |
| Web8279 | 36,014 | 36,577 | 36,225 | 35,999 | 36,019 |
| Editor3dd | 126,447 | 122,774 | 120,731 | 128,492 | 125,961 |
| Screenshot7dba | 13,910 | 13,093 | 13,031 | 14,304 | 14,304 |
| Paper1eb | 452,765 | 452,828 | 449,914 | 446,531 | 448,229 |
| Paper9bde | 412,539 | 412,539 | 412,539 | 412,539 | 412,539 |
| image.png | 393,268 | 393,324 | 392,441 | 391,506 | 391,897 |
| wallhaven-49yekn | 707,442 | 712,361 | 713,676 | 713,373 | 716,743 |
| Manga Aisazu | 1,372,611 | 1,372,611 | 1,372,611 | 1,372,611 | 1,372,611 |
| Set14 monarch | 407,506 | 407,506 | 407,506 | 407,506 | 407,506 |
| Linen16 | 1,841,476 | 1,841,476 | 1,841,476 | 1,841,476 | 1,841,476 |

There are real local wins: run ordering reduces Editor3dd by 4.52% and
Screenshot7dba by 6.32%; Hot128 reduces Preview by 5.72%. But RunFrequency
increases Preview by 5.59%, and Hot128 increases Hacker News by 3.83% and
wallhaven-49yekn by 1.31%. No new order is a universal default replacement.

Most of these inputs use the existing groupwise route. Its palette orders are
applied locally; global palette receipts are intentionally empty. A zero global
tuple count is not proof that no local palette was tried or accepted. In this
phase only tilemap reports an accepted global tuple palette. Gray/GA controls
and accepted/declined palettes must not be counted as evidence of an active
3+-channel run reorder solely from the requested mode.

### Timing Outlier Recheck

The first wallhaven-49yekn Hot128 trial took 7.109s and the second 1.006s.
Five further interleaved trials measured:

- Default: 0.857..0.975s, median 0.959s, 707,442 bytes, 2.7293 bpp,
  2.161 MPix/s.
- Hot128: 0.904..1.026s, median 0.909s, 716,743 bytes, 2.7652 bpp,
  2.281 MPix/s.

The 7-second value remains in the original report, not silently removed. Its
cause was not established; the follow-up does not support the initial suspicion
of a stable backend slowdown. The 1.31% size regression remains real. Neither
the outlier nor this small recheck proves a general speed benefit/penalty.

### Full-Frame Diagnostic

This separate post-hoc experiment uses the same `--full-frame --palette-colors
10000` settings for all seven orders on four real images. It is one trial per
order, not a new independent validation set. All four report one accepted global
absolute tuple palette: 3,614 / 218 / 2,128 / 256 entries respectively, zero delta
entries. Neither input nor palette samples were quantized.

| Image | Default | Frequency | Runs | Lexical | Hot32 | Hot128 | Hot512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper9bde | 269,794 | 257,033 | 253,096 | 265,000 | 283,633 | 257,627 | 254,051 |
| Editor3dd | 110,581 | 95,944 | 94,654 | 111,830 | 98,131 | 106,877 | 111,830 |
| Web8279 | 32,130 | 32,698 | 31,948 | 30,808 | 32,212 | 31,010 | 31,349 |
| Preview | 53,669 | 59,088 | 56,961 | 51,378 | 55,002 | 51,460 | 51,378 |

RunFrequency reduces Paper9bde by 6.19% and Editor3dd by 14.40% relative to
the **same full-frame baseline**. Pixel-frequency ordering already captures
part of those gains: Runs is another 1.53% / 1.34% smaller than FrequencyLuma.
Relative to the same full-frame baseline, Preview instead grows by 6.13%.

| Four-image aggregate | Bytes | Native seconds | bpp | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Default | 466,174 | 5.146 | 0.8640 | 0.839 |
| FrequencyLuma | 444,763 | 5.024 | 0.8243 | 0.859 |
| RunFrequency | 436,659 | 5.184 | 0.8093 | 0.833 |
| Lexical | 459,016 | 5.326 | 0.8507 | 0.810 |
| Hot32 | 468,978 | 5.539 | 0.8692 | 0.779 |
| Hot128 | 446,974 | 5.381 | 0.8284 | 0.802 |
| Hot512 | 448,608 | 5.221 | 0.8314 | 0.827 |

The run-order aggregate is 6.33% smaller for similar time in this diagnostic.
Changing from the normal projected route to full-frame/10,000 colors independently
changes size and time; that route-change benefit must not be credited to this
sorter. Per-image single-trial speed differences are not stable performance claims.

### Synthetic Controls

Four 48-color fixtures exercise RGB/RGBA at 8/16 bits with different run-count
and pixel-count rankings. They preserve arbitrary hidden RGB. A 300-tuple RGBA16
fixture really crosses the 128-entry cutoff; a grayscale fixture is a no-reorder
control. All five color fixtures publish an accepted global palette with the
expected 48/300 explicit entries.

| Fixture | Default bytes | Runs | Lexical | Hot32 | Hot128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB8 / 48 | 611 | 460 | 322 | 483 | 322 |
| RGBA8 / 48 | 817 | 499 | 381 | 532 | 381 |
| RGB16 / 48 | 760 | 584 | 475 | 600 | 475 |
| RGBA16 / 48 | 1,060 | 675 | 560 | 683 | 560 |
| RGBA16 / 300 | 4,269 | 1,808 | 1,659 | 1,638 | 2,613 |
| Gray control | 493 | 493 | 493 | 493 | 493 |

Hot128 on 48 colors is exactly a lexical-order experiment, not evidence for the
run statistic. On 300 colors, Hot32 is smaller than lexical while Hot128 is
substantially worse. These constructed cases verify mechanism and cutoff
sensitivity; their large gains and millisecond timings are not corpus forecasts.

## Validation and Decision

- Nine focused palette-order unit tests passed, including old source selectors,
  aliases, implicit index isolation and trivial partition scan bypass.
- Nine new integration tests passed, including signed full tuples, raster row
  boundaries, >4 channels, tree/flat maps, low-depth aliases, implicit promotion,
  ordered/geometry bypass and delta-prefix/fallback contracts. This includes
  16 complete native-decoder RGB8/RGBA16 roundtrips with hidden RGB preservation.
- Seven palette harness unit tests and one lab argument test passed.
  Workspace/all-target compilation and formatting checks also passed.
- Clippy passed with the pre-existing unrelated `needless_borrow` lint exempted,
  as in the previous experiment.
- All 240 official djxl comparisons passed exact original samples. Temporary
  encoded outputs were removed. Windows execution is not a Linux runtime claim.

Keep both named orders as opt-in research tools, not a default effort/scheduler
change. Real gains exist, but the mixed corpus does not justify universal use.
The existing cheap palette-versus-raw cost check is not a comparison against
every alternative order and does not guarantee a size/time win. An eventual
automatic selector would need separate evidence for inexpensive prediction of
index residual and palette-table costs; this experiment does not silently add
multiple complete encodes or a fitted per-file exception table.

Follow-up: [sampled order screening](PALETTE_SCREEN_20261110.md) adds a separate
opt-in `ScreenedRuns` selector. It does not change the raw orders or the default
used in this report; its estimates are not an encoded-size guarantee.
