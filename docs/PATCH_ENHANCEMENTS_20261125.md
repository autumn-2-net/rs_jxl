# Lossless patch experiments, 2026-11-25

The reference encoder commit was `f48e919`. Two standard JPEG XL tools were tested on still images at effort 9. The experiments affect encoder decisions about reference frames, dictionaries and residuals. Both are opt-in through the development example `patch_lab`.

Across 20 real images the eight existing controls yielded a per-image minimum totaling 10,930,580 B. Adding five experimental configurations brought that to 10,926,245 B, saving 4,335 B (0.040%). Two known images account for the entire improvement. All eight holdout minima stayed equal. The experiments remain opt-in. The research threshold for moving to shift/scale search or integrating a production selector was 0.5–1% on patch-oriented content. Even the 12 enriched images saved only 0.186% with offline selection.

## What changed

- **Source Add, same-size near matching.** Before discarding a detector fingerprint with too few occurrences, consider extending an existing reference or forming a group with other weak fingerprints. All positions must have identical width and height. A binary shape overlap and sparse signed-color comparison shortlist pairs. The next stage evaluates every original color sample against the candidate template, including one pixel of right and bottom context where available. It estimates zero, left, top and gradient prediction costs and charges for template samples, dictionary entries and positions. The selected plan uses normal signed subtraction, exact Add replay and existing frame writing. Extra channels pass through. Existing exact groups retain their original templates. There is one backend encode per requested mode.
- **Exact Replace, rectangle expansion.** Start with the existing grid or off-grid detector's nonoverlapping, sample-verified tiles. Join adjacent groups horizontally, then vertically, only when their *entire* ordered occurrence lists have the same translation. A merged rectangle covers exactly the original verified tiles. Repack the reference in height-ordered shelves with the original atlas budget. The old square layout remains available when merging or packing fails. This experiment can change atlas entropy as well as dictionary size.

The Add search considers at most 512 under-threshold fingerprint groups, 8 occurrences per such group, 16 peers per seed and 4,096 occupied rectangles. The cap on fully scored interior samples is four million. Template materialization has separate budgets of two million samples each for existing and new groups. The predictors are proxies, so a positive local score can still grow the complete file. For Replace, comparing forced tiles before and after merging isolates the layout effect from the existing estimate-based tile admission policy.

## Measurement

The inputs were frozen by SHA-256 before running the new encoder. The 12 known images were deliberately enriched for patch work in earlier experiments. Eight images from eight other source families were selected from the earlier 60-image manifest. They cover a sprite, website, game, manga, two photographs, a texture and an urban scene. The six deterministic fixtures cover exact repetition, antialias variations that cross all three fingerprint divisors, hidden RGB in RGBA8/RGBA16, and repeated 48×32 regions placed on and off a 16-pixel grid. Real and synthetic results are reported separately.

Size is the complete JXL file in bytes. Encoder throughput is total input pixels divided by summed native encode time. The Windows x86-64 release encoder used eight threads. The runner strips ancillary PNG metadata and verifies original pixel values with both the project's Rust decoder and the official djxl, one decoder thread each. A 16-bit RGBA sample comparison uses full integer words, including colors at alpha zero. Timings exclude PNG I/O, decoding and sample verification. Small timing differences from one pass need repetitions.

The eight existing controls are native E9 with ordinary patch policy, patch off, forced Source Add at fingerprint divisors 1, 2 and 4, component Replace, forced grid Tile16 and forced off-grid Tile16. “Existing best” is the sum of the smallest complete control file *per image*. That choice requires encoding multiple candidates offline.

## Real images

The real cohort contains 44,082,809 pixels. For each experiment, compare against its own existing route at the same effort, thread count, fingerprint divisor and tile origin policy. Smaller/equal/larger counts refer to full-file bytes. The lab also evaluated the ordinary E9 route at 11,274,616 B and 3.511 MPix/s.

| Route, 20 images | Before B | After B | Change B | Smaller / equal / larger | Worst growth B | Encode MPix/s before → after |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Source d1 + near matching | 10,975,100 | 10,990,512 | +15,412 | 3 / 10 / 7 | 10,137 | 1.069 → 1.078 |
| Source d2 + near matching | 10,979,734 | 10,992,544 | +12,810 | 4 / 10 / 6 | 10,488 | 1.082 → 1.047 |
| Source d4 + near matching | 10,985,556 | 10,993,186 | +7,630 | 1 / 11 / 8 | 6,142 | 1.075 → 1.063 |
| Grid Tile16 + rectangles | 11,181,789 | 11,181,324 | −465 | 5 / 13 / 2 | 23 | 0.794 → 0.803 |
| Off-grid Tile16 + rectangles | 11,163,132 | 11,162,744 | −388 | 5 / 13 / 2 | 333 | 0.762 → 0.762 |

At d1 the known cohort grew 5,275 B. The eight holdout files grew 10,137 B, all on the Creative Commons webpage. More dictionary positions alone predict little about final size. On the first known image the detector extended 1,107 positions to 1,355 and saved 1,980 B, although the existing d2 control still won by 1,635 B. On the fourth known image it extended 444 positions to 526 and grew the file by 4,482 B. On this image off-grid square tiles produced the smallest existing file.

The follow-up crossed near matching with d2 and d4 on all 26 images. For `9bde6aaf390109f97622d35ff1e900ee.png`, near+d4 reached 231,243 B from the existing best of 235,549 B. For `8277.png`, near+d2 reached 30,826 B from 30,855 B. Neither rectangle route beat the best existing control on any real image. Evaluating all 13 routes on the 20 real images consumed 553.8 native encoder seconds in total. A normal single E9 encode of each file took 12.6 seconds in total. No production chooser was benchmarked.

The initial matrix used one pass per route, with other local work during early rows. An isolated three-repeat check alternated d1 and near-d1 on the first and fourth known images and the synthetic RGB8 fringe fixture. Summing the two real per-image median times gave 0.524 → 0.509 MPix/s. The individual real images gave 0.509 → 0.478 and 0.538 → 0.542 MPix/s. These timings support no general speedup.

## Mechanism controls

| Synthetic fixture | Smallest existing B | Best new B | Change |
| --- | ---: | ---: | ---: |
| 16 exact RGB8 repeats | 1,117 | 1,117 | 0 |
| 16 antialiased RGB8 variants | 1,851 | 1,162 | −37.22% |
| 16 antialiased RGBA8 variants with hidden color | 3,450 | 1,292 | −62.55% |
| 16 antialiased RGBA16 variants with hidden color | 4,333 | 2,124 | −50.98% |
| 16 exact RGBA16 repeats with hidden color | 2,064 | 2,064 | 0 |
| Repeated exact 48×32 regions | 11,193 | 11,165 | −0.25% |

On the antialiased RGB8 fixture the existing divisors all produced 1,851 B. Near matching emitted a reference and 16 dictionary positions. On the exact rectangle fixture the off-grid merger removed five groups, saving 28 B. Independent frame-header inspection confirmed the visible patch flag and earlier reference-only frame in both outputs. The d1 RGB8 control had neither. Each hidden-color fixture includes 280,071 RGB samples at alpha zero per decode.

## Compatibility and scope

All 338 matrix files (26 images × 13 configurations) and 18 additional timing files passed both decoders. The 712 decoded images had zero differing pixels. Repeated native encodes of each timing fixture produced the same bytes. The raw initial matrix, divisor cross-check and repeat records are under `out/patch-near-20261125/{matrix,combined,timing}.json`. Their release lab SHA-256 was `c8f4d50f09db5f540b1822981a70871c1c4b63d8404d95862fc66cad84b7596b`.

A clean release lab from `f48e919` produced identical SHA-256 output to the experimental binary with the new switches off in 12 comparisons. The checks covered off, d1, grid and off-grid modes on one real image and two mechanism fixtures. The same three inputs were byte-identical between the ordinary lab route and the public CLI with default effort and eight threads. Source and tile tests also covered signed templates, anti-correlated colors, 16-bit integer reuse, position conflicts, full-cost rejection and a rectangle with one mismatched invisible sample.

The optional existing official-decoder test for combined patch, block-context and gather settings needed a fixture correction. It now requests the fixed block context before resolving E9's automatic tree policy for the reference and residual independently. It verifies all original combinations and output samples.

## Reproduction

From the repository root on Windows, with the existing Rust decoder and official djxl available to the runner:

```powershell
cargo +stable build --release --locked -p rs-jxl-cli --example patch_lab
python tools/patch_enhancement/freeze.py --work-dir out/patch-repro
python tools/patch_enhancement/run.py --work-dir out/patch-repro --lab target/release/examples/patch_lab.exe
```

For a public-data-independent mechanism check, add `--synthetic-only` to the freeze command. The full run reads the historical `patch12.json` and `holdout60.json`. Override their paths with `--known12` and `--holdout60`. Override the decoder executables with `--rust` and `--official`. The report records input and executable hashes, native timings, per-image file sizes, detector receipts and both sample comparisons.
