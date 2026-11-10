# MA Leaf Offsets: 100-Image Bpp Comparison (2026-11-10)

## Result

Across 100 images / **209,474,281 pixels**, the Entropy
offset mode reduced complete-file bpp from **4.473571** to
**4.464337**, a **0.009234 bpp** reduction, or
**0.2064%** fewer bytes. Its summed native encoding time
increased by **26.59%**.

All **400/400** official-decoder pixel checks passed. No candidate mode produced
a larger file than its matching baseline. Entropy improved 76
images and left 24 unchanged, but only 8
images saved at least 1%; its median per-image saving was just
**0.0432%**.

Keep the experiment **disabled by default**. The larger sample shows less
benefit than the earlier eight-real-image screen (0.561% for Entropy). A cheap
rejection gate for marginal domains is more promising than applying this
post-training pass universally. No encoder implementation or production
default was changed during this corpus test.

## Scope and Reproduction

- Existing fixed stratified sample of 100 PNGs from the previous 500-image
  corpus, seed `20261110`; no new selection based on offset results.
- Strata: photographs, urban images, documents/web pages, game UI, manga,
  icons/alpha, textures, mixed local inputs and format tests. This is coverage,
  not a claim of matching a real-world workload distribution.
- Windows x64, release binary, effort 9, 8 encoder threads, default buffering.
  This tests a fixed native backend candidate, **not the outer scheduler** and
  **not official cjxl encoding performance**.
- Same binary and effort for all four modes. Mean, Median and Entropy each use
  the default 1024-sample per-leaf fitting reservoir. Default JXL LZ77 behavior;
  unrelated LZ77, tree-learning and patch experiments were not enabled.
- Exactly one measured encode per image/mode: 100 distinct images, 400 encodes.
  Variant order rotates between images. No builds or concurrent encoder
  benchmarks were launched for this run. Timing is a single-observation screen, not a precision
  microbenchmark; do not interpret small per-image time differences as reliable
  speedups.
- Native timers exclude process startup, input/output and independent
  verification; include fitting, optional retokenization and exact arbitration.
- 89 original 8-bit inputs, seven 16-bit inputs, four packed-depth inputs.
  No resizing or cropping. Packed PNGs remain packed when passed to the encoder;
  only the independent oracle expands their samples to 8-bit. Hidden RGB and
  alpha are included in comparison.
- Every output was decoded by official libjxl 0.12 djxl and compared with the
  original samples. Encoded JXL identity was neither required nor optimized.
  Binary/source digests only identify test inputs.
- The test harness now reuses the existing packed-PNG adapter, verifies the
  manifest and real source identities, and checkpoints each completed image.
  Encoded files and decode images were temporary; only the report persists.
- A 1200-second subprocess guard was configured; **no timeouts occurred**.
  This guard is not an encoder search policy.

Artifacts:

- Raw report: `out/ma-leaf-offset-20261110/corpus100-e9.json`.
- Fixed sample: `out/lz-length-20261110/sample100.json`.
- Sample SHA256: `c03114a71f638c45d82c032e7200c7c6aefccd46d98255913cdfd87a1be765fc`.
- Encoder SHA256: `6c7c47d7cd7549b6e61b38e05c705e707b2f23cfb0745c670157ce9c22f39156`.
- Official decoder SHA256: `c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.

```powershell
python -B tools/benchmark_refinement.py --encoder target/release/examples/encoder_lab.exe --decoder <official-djxl-path> --manifest out/lz-length-20261110/sample100.json --variants baseline offset-mean offset-median offset-entropy --effort 9 --threads 8 --repeats 1 --warmups 0 --timeout 1200 --out out/ma-leaf-offset-100-new.json
```

The output report path must be new. Algorithm, controls and focused correctness
checks are documented in [the original experiment](MA_LEAF_OFFSET_20261110.md).

## Aggregate Bpp

`bpp = 8 * sum(complete JXL bytes) / sum(width * height)`.
This is pixel-weighted across images, not the arithmetic mean of per-image bpp.
All channels, alpha and JXL framing are included in the file sizes.

| Mode | Total bytes | bpp | Delta bpp vs baseline | Byte saving |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 117,137,268 | 4.473571 | 0.000000 | 0.0000% |
| Mean | 116,952,981 | 4.466533 | -0.007038 | 0.1573% |
| Median | 116,941,306 | 4.466087 | -0.007484 | 0.1673% |
| Entropy | 116,895,470 | 4.464337 | -0.009234 | 0.2064% |

Entropy saves **241,798 bytes** across the complete corpus.

## Encoding Time

Aggregate throughput uses the same total pixel count divided by the sum of
native encoding times. These are not elapsed times for the complete harness,
which also performs I/O and 400 official decodes.

| Mode | Native seconds | Time increase | MPix/s |
| --- | ---: | ---: | ---: |
| Baseline | 104.954 | 0.00% | 1.996 |
| Mean | 130.940 | 24.76% | 1.600 |
| Median | 132.474 | 26.22% | 1.581 |
| Entropy | 132.861 | 26.59% | 1.577 |

## Per-Image Distribution

Savings are computed independently relative to each image's baseline.

| Mode | Smaller | Equal | Larger | Median saving | Saving >=0.1% | Saving >=1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean | 81 | 19 | 0 | 0.0302% | 26 | 7 |
| Median | 81 | 19 | 0 | 0.0356% | 31 | 9 |
| Entropy | 76 | 24 | 0 | 0.0432% | 33 | 8 |

An after-the-fact oracle that chooses the smallest of all three modes per image
would yield 116,892,736 bytes / 4.464232 bpp / 0.2088% saving.
That saves only **2,734 additional bytes** over Entropy alone.
It is an upper-bound comparison using already measured outputs, not an
implemented automatic selector and not achievable at one-mode runtime by
simply running all modes.

## Category Breakdown

Each row uses its own total pixels as the bpp denominator. Time increases are
summed within that category.

| Category | Images | Baseline bpp | Mean bpp | Median bpp | Entropy bpp | Entropy saving | Entropy time increase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alpha_icon | 10 | 3.235941 | 3.233850 | 3.233849 | 3.233538 | 0.0743% | 24.47% |
| document_web | 12 | 1.061159 | 1.056138 | 1.055456 | 1.054613 | 0.6168% | 24.23% |
| format_mixed | 10 | 5.325713 | 5.320981 | 5.320974 | 5.319366 | 0.1192% | 27.86% |
| game_ui | 8 | 4.959600 | 4.953115 | 4.950750 | 4.950074 | 0.1921% | 49.76% |
| manga | 8 | 6.796498 | 6.792818 | 6.792686 | 6.790795 | 0.0839% | 34.85% |
| mixed_local | 8 | 1.502923 | 1.494943 | 1.493530 | 1.491287 | 0.7742% | 14.36% |
| photo_highres | 16 | 9.121331 | 9.112091 | 9.112168 | 9.110574 | 0.1179% | 26.16% |
| photo_other | 12 | 7.866778 | 7.861407 | 7.861805 | 7.859898 | 0.0875% | 27.32% |
| texture | 8 | 11.773916 | 11.771561 | 11.771532 | 11.770114 | 0.0323% | 38.35% |
| urban | 8 | 11.353794 | 11.318526 | 11.318987 | 11.304710 | 0.4323% | 23.67% |

Document/web and mixed-local categories show higher weighted gains than
textures and the general photo categories, but gains remain image-dependent.
The best individual image is not evidence that all images in its category
benefit similarly.

## Largest Entropy Gains

IDs refer to the complete table below.

| ID / image | Category | Baseline bpp | Entropy bpp | Saving | Native seconds: base / entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 7060.png | document_web | 4.848198 | 4.705956 | 2.9339% | 1.098 / 0.988 |
| 37 / 5206.png | document_web | 0.454596 | 0.441662 | 2.8452% | 1.746 / 2.101 |
| 86 / 9057.png | format_mixed | 0.976888 | 0.960042 | 1.7244% | 0.412 / 0.467 |
| 100 / cursor_PNG65.png | alpha_icon | 0.447937 | 0.440613 | 1.6351% | 0.149 / 0.196 |
| 11 / 0562.png | photo_highres | 9.858702 | 9.706267 | 1.5462% | 1.817 / 2.237 |
| 21 / img_100.png | urban | 13.640434 | 13.436935 | 1.4919% | 1.255 / 1.581 |
| 50 / 5203.png | document_web | 0.602078 | 0.593452 | 1.4327% | 2.783 / 3.194 |
| 62 / wallhaven-vq815p.png | mixed_local | 2.637309 | 2.602163 | 1.3327% | 1.931 / 2.453 |
| 94 / duckduckgo.com.png | document_web | 0.363712 | 0.360540 | 0.8720% | 0.969 / 1.420 |
| 85 / 9024.png | format_mixed | 2.716064 | 2.692474 | 0.8685% | 0.647 / 0.796 |

## Marginal Gains With Added Cost

Examples with less than 0.1% saving and more than 0.1 seconds added in this
single-observation screen, ordered by added native time. These warrant future
rejection-gate work; they are not precise repeated timing claims.

| ID / image | Saving | Native seconds: base / entropy | Added seconds |
| --- | ---: | ---: | ---: |
| 32 / 6613.png | 0.0472% | 3.981 / 5.046 | 1.064 |
| 82 / blue_floor_tiles_01_Diffuse_1k.png | 0.0000% | 1.826 / 2.705 | 0.879 |
| 27 / phone_PNG48891.png | 0.0690% | 3.107 / 3.886 | 0.779 |
| 79 / 6805.png | 0.0927% | 4.535 / 5.311 | 0.776 |
| 60 / 0255.png | 0.0109% | 1.690 / 2.417 | 0.727 |
| 31 / 0325.png | 0.0041% | 1.889 / 2.592 | 0.703 |

## All 100 Images

Bpp values are complete-file bits divided by that image's pixel count.
Each row has four successful official pixel checks. Full paths, source hashes,
sizes, native/decoder timers and offset receipts are retained in the JSON report.

| ID / source | Baseline bpp | Mean bpp | Median bpp | Entropy bpp | Entropy saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / `diverse-downloads/imazen26/7000-lilith-plots/7060.png` | 4.848198 | 4.729401 | 4.724991 | 4.705956 | 2.9339% |
| 2 / `manga109/BakuretsuKungFuGirl.png` | 8.457035 | 8.450769 | 8.450189 | 8.446158 | 0.1286% |
| 3 / `jxl-testdata/images/photo_tecnick/RGB_OR_1200x1200_024.png` | 9.429400 | 9.427422 | 9.427061 | 9.425900 | 0.0371% |
| 4 / `jxl-testdata/images/icon_64/status-weather-clear-night.png` | 3.792969 | 3.792969 | 3.792969 | 3.792969 | 0.0000% |
| 5 / `manga109/YouchienBoueigumi.png` | 4.252015 | 4.243737 | 4.243579 | 4.240955 | 0.2601% |
| 6 / `manga109/TaiyouNiSmash.png` | 6.908971 | 6.902539 | 6.903003 | 6.899733 | 0.1337% |
| 7 / `diverse-downloads/imazen26/8100-lilith-web-screenshots/8277.png` | 1.070985 | 1.069513 | 1.067434 | 1.069257 | 0.1613% |
| 8 / `diverse-downloads/kenney_particle_pack/PNG (Transparent)/muzzle_05.png` | 0.660431 | 0.660431 | 0.660431 | 0.660431 | 0.0000% |
| 9 / `diverse-downloads/imazen26/8100-lilith-web-screenshots/8359.png` | 0.511841 | 0.510732 | 0.509603 | 0.510956 | 0.1729% |
| 10 / `Set14/face.png` | 3.756249 | 3.756249 | 3.756249 | 3.756249 | 0.0000% |
| 11 / `DIV2K/0562.png` | 9.858702 | 9.728570 | 9.729273 | 9.706267 | 1.5462% |
| 12 / `diverse-downloads/kenney_particle_pack/PNG (Black background)/Rotated/muzzle_05_rotated.png` | 0.701904 | 0.701904 | 0.701050 | 0.701630 | 0.0391% |
| 13 / `diverse-downloads/kenney_pixel_platformer/Tiles/tile_0029.png` | 3.209877 | 3.209877 | 3.209877 | 3.209877 | 0.0000% |
| 14 / `DIV2K/0049.png` | 8.196807 | 8.196136 | 8.196035 | 8.195928 | 0.0107% |
| 15 / `DIV2K/0187.png` | 8.956119 | 8.954439 | 8.954193 | 8.954457 | 0.0186% |
| 16 / `local-train/img_480x360_3x16bit_RGB_grid_rectangle_0020x0015_0.png` | 0.010833 | 0.010833 | 0.010833 | 0.010833 | 0.0000% |
| 17 / `diverse-downloads/clic2025/clic2025/final-test/1b4ad095795ac552b38a21d51be7bfaee8e7d0a70619d84767814321df4ed062.png` | 7.679013 | 7.677930 | 7.677789 | 7.677468 | 0.0201% |
| 18 / `BSDS100/299086.png` | 6.374323 | 6.374323 | 6.374323 | 6.374323 | 0.0000% |
| 19 / `urban100/img_003.png` | 14.783456 | 14.775816 | 14.776034 | 14.773380 | 0.0682% |
| 20 / `jxl-testdata/images/textures_plants/plant_14.png` | 2.210694 | 2.206585 | 2.206055 | 2.203926 | 0.3062% |
| 21 / `urban100/img_100.png` | 13.640434 | 13.486290 | 13.488815 | 13.436935 | 1.4919% |
| 22 / `jxl-testdata/images/screenshot_game/Minetest_high-res_texture_pack.png` | 8.052175 | 8.049085 | 8.048572 | 8.048598 | 0.0444% |
| 23 / `DIV2K/0082.png` | 8.367773 | 8.366207 | 8.366178 | 8.365622 | 0.0257% |
| 24 / `diverse-downloads/kenney_pixel_platformer/Preview.png` | 0.965747 | 0.965510 | 0.964004 | 0.965662 | 0.0088% |
| 25 / `diverse-downloads/imazen26/5200-epa-climate-impact-2021-report/5222.png` | 0.148369 | 0.147319 | 0.147099 | 0.147165 | 0.8112% |
| 26 / `diverse-downloads/kenney_pixel_platformer/Tiles/tile_0093.png` | 2.987654 | 2.987654 | 2.987654 | 2.987654 | 0.0000% |
| 27 / `jxl-testdata/images/pngimg/phone_PNG48891.png` | 4.056652 | 4.054105 | 4.054212 | 4.053853 | 0.0690% |
| 28 / `BSDS100/147091.png` | 8.740625 | 8.740260 | 8.740521 | 8.740625 | 0.0000% |
| 29 / `diverse-downloads/imazen26/8100-lilith-web-screenshots/8380.png` | 0.737040 | 0.736735 | 0.736613 | 0.736562 | 0.0649% |
| 30 / `jxl-testdata/images/photo_tecnick/RGB_OR_1200x1200_055.png` | 8.039328 | 8.031678 | 8.031944 | 8.026706 | 0.1570% |
| 31 / `DIV2K/0325.png` | 9.894829 | 9.893464 | 9.893484 | 9.894421 | 0.0041% |
| 32 / `diverse-downloads/imazen26/6600-ia-scans-manuscript-illustrations/6613.png` | 5.999415 | 5.997606 | 5.997756 | 5.996582 | 0.0472% |
| 33 / `urban100/img_070.png` | 8.997894 | 8.997437 | 8.996847 | 8.995412 | 0.0276% |
| 34 / `diverse-downloads/cid22/CID22/CID22-512/training/pexels-photo-4210863.png` | 13.793060 | 13.783783 | 13.778320 | 13.777740 | 0.1111% |
| 35 / `DIV2K/0382.png` | 11.259856 | 11.259356 | 11.259431 | 11.259306 | 0.0049% |
| 36 / `local-test/png/wallhaven-49yekn.png` | 2.729329 | 2.728862 | 2.728665 | 2.728642 | 0.0252% |
| 37 / `diverse-downloads/imazen26/5200-epa-climate-impact-2021-report/5206.png` | 0.454596 | 0.445139 | 0.443158 | 0.441662 | 2.8452% |
| 38 / `diverse-downloads/imazen26/9226-lilith-ai-products/9284.png` | 2.735774 | 2.732468 | 2.732315 | 2.731216 | 0.1666% |
| 39 / `urban100/img_076.png` | 11.760390 | 11.752323 | 11.752392 | 11.748255 | 0.1032% |
| 40 / `DIV2K/0107.png` | 13.169165 | 13.167844 | 13.168544 | 13.167121 | 0.0155% |
| 41 / `DIV2K/0183.png` | 10.622387 | 10.621684 | 10.621840 | 10.622289 | 0.0009% |
| 42 / `urban100/img_019.png` | 14.430836 | 14.400643 | 14.400356 | 14.381939 | 0.3388% |
| 43 / `local-train/img_480x360_3x16bit_RGB_bars_vertical_square_0128_1xhalfPI.png` | 0.006667 | 0.006667 | 0.006667 | 0.006667 | 0.0000% |
| 44 / `diverse-downloads/kenney_particle_pack/PNG (Transparent)/muzzle_01.png` | 1.222595 | 1.222443 | 1.222443 | 1.222595 | 0.0000% |
| 45 / `jxl-testdata/images/screenshot_game/Job_Simulator_screenshot_-_Kitchen_02.png` | 4.683062 | 4.663960 | 4.663062 | 4.654953 | 0.6002% |
| 46 / `DIV2K/0027.png` | 6.481360 | 6.480585 | 6.480585 | 6.480081 | 0.0197% |
| 47 / `local-train/img_480x360_3x16bit_RGB_grid_lines_quarters_1.png` | 0.010926 | 0.010926 | 0.010926 | 0.010926 | 0.0000% |
| 48 / `diverse-downloads/kenney_particle_pack/PNG (Black background)/star_05.png` | 0.512604 | 0.512604 | 0.512604 | 0.512604 | 0.0000% |
| 49 / `DIV2K/0731.png` | 8.707606 | 8.706611 | 8.706620 | 8.707143 | 0.0053% |
| 50 / `diverse-downloads/imazen26/5200-epa-climate-impact-2021-report/5203.png` | 0.602078 | 0.595607 | 0.594430 | 0.593452 | 1.4327% |
| 51 / `diverse-downloads/qoi_web/qoi-benchmark/screenshot_web/microsoft.com.png` | 1.512457 | 1.511003 | 1.510298 | 1.510545 | 0.1264% |
| 52 / `jxl-testdata/images/screenshot_game/Chasing_Aurora_screenshot_10_(Chase).png` | 2.434667 | 2.434074 | 2.434104 | 2.433896 | 0.0316% |
| 53 / `urban100/img_007.png` | 9.082341 | 9.077344 | 9.078378 | 9.076126 | 0.0684% |
| 54 / `jxl-testdata/images/screenshot_game/TMW_Screenshot_15.png` | 9.847483 | 9.841683 | 9.829533 | 9.837400 | 0.1024% |
| 55 / `jxl-testdata/images/photo_wikipedia/035.png` | 8.485386 | 8.479387 | 8.479630 | 8.477287 | 0.0954% |
| 56 / `diverse-downloads/kenney_pixel_platformer/Tilemap/tilemap-backgrounds.png` | 0.294988 | 0.294988 | 0.294988 | 0.294988 | 0.0000% |
| 57 / `manga109/MariaSamaNihaNaisyo.png` | 6.396606 | 6.395157 | 6.394073 | 6.393800 | 0.0439% |
| 58 / `manga109/YoumaKourin.png` | 9.263285 | 9.261729 | 9.261671 | 9.261199 | 0.0225% |
| 59 / `urban100/img_001.png` | 8.840111 | 8.829059 | 8.828938 | 8.822229 | 0.2023% |
| 60 / `DIV2K/0255.png` | 7.081598 | 7.080892 | 7.080881 | 7.080826 | 0.0109% |
| 61 / `manga109/PrismHeart.png` | 4.502663 | 4.500942 | 4.501670 | 4.500304 | 0.0524% |
| 62 / `local-test/png/wallhaven-vq815p.png` | 2.637309 | 2.613455 | 2.608295 | 2.602163 | 1.3327% |
| 63 / `DIV2K/0380.png` | 11.357886 | 11.357023 | 11.356823 | 11.356702 | 0.0104% |
| 64 / `urban100/img_014.png` | 9.334061 | 9.316559 | 9.316643 | 9.305931 | 0.3014% |
| 65 / `jxl-testdata/testdata-main/palette/358colors.png` | 1.996914 | 1.992284 | 1.986111 | 1.996914 | 0.0000% |
| 66 / `diverse-downloads/imazen26/6000-lilith-scans-public-patents/6104.png` | 0.074098 | 0.074098 | 0.074098 | 0.074098 | 0.0000% |
| 67 / `local-test/mt_lay/t1/146845733_p0.png` | 0.005400 | 0.005400 | 0.005400 | 0.005400 | 0.0000% |
| 68 / `diverse-downloads/pngsuite/pngsuite/basi2c08.png` | 1.070313 | 1.070313 | 1.070313 | 1.070313 | 0.0000% |
| 69 / `local-train/0067.png` | 7.166188 | 7.144115 | 7.142858 | 7.135009 | 0.4351% |
| 70 / `jxl-testdata/images/photo_tecnick/RGB_OR_1200x1200_020.png` | 10.284128 | 10.277917 | 10.278450 | 10.278467 | 0.0550% |
| 71 / `diverse-downloads/clic2025/clic2025/final-test/2397c73f8feeb8f630fe055da5551c671794afae80528e8612dc62fc0cf61294.png` | 5.776433 | 5.773188 | 5.773980 | 5.772564 | 0.0670% |
| 72 / `manga109/GOOD_KISS_Ver2.png` | 6.204085 | 6.201892 | 6.202065 | 6.200443 | 0.0587% |
| 73 / `diverse-downloads/cid22/CID22/CID22-512/training/7394483.png` | 5.496979 | 5.496185 | 5.496979 | 5.495758 | 0.0222% |
| 74 / `diverse-downloads/imazen26/9226-lilith-ai-products/9276.png` | 4.473780 | 4.471018 | 4.471074 | 4.469945 | 0.0857% |
| 75 / `Set5/baby.png` | 6.756073 | 6.754456 | 6.754822 | 6.756073 | 0.0000% |
| 76 / `diverse-downloads/polyhaven/white_oak_veneer/white_oak_veneer_Rough_1k.png` | 12.070755 | 12.067970 | 12.067390 | 12.062439 | 0.0689% |
| 77 / `diverse-downloads/imazen26/5000-national-park-service-brochures/5020.png` | 2.011006 | 2.008026 | 2.007830 | 2.006937 | 0.2023% |
| 78 / `jxl-testdata/images/textures_pk01/pk01_pan_wall01_local.png` | 2.083771 | 2.082825 | 2.082062 | 2.081421 | 0.1128% |
| 79 / `diverse-downloads/imazen26/6800-ia-scans-manuscript-text/6805.png` | 2.341877 | 2.340668 | 2.340731 | 2.339707 | 0.0927% |
| 80 / `DIV2K/0434.png` | 9.325843 | 9.322328 | 9.322254 | 9.322296 | 0.0380% |
| 81 / `diverse-downloads/kenney_particle_pack/PNG (Transparent)/trace_03.png` | 0.592804 | 0.592804 | 0.592804 | 0.592804 | 0.0000% |
| 82 / `diverse-downloads/polyhaven/blue_floor_tiles_01/blue_floor_tiles_01_Diffuse_1k.png` | 34.127197 | 34.127197 | 34.127197 | 34.127197 | 0.0000% |
| 83 / `manga109/JijiBabaFight.png` | 8.387326 | 8.385778 | 8.385240 | 8.383767 | 0.0424% |
| 84 / `jxl-testdata/images/textures_plants/plant_46.png` | 8.870041 | 8.856628 | 8.859716 | 8.858365 | 0.1316% |
| 85 / `diverse-downloads/imazen26/9000-lilith-ai-clipart/9024.png` | 2.716064 | 2.697296 | 2.697594 | 2.692474 | 0.8685% |
| 86 / `diverse-downloads/imazen26/9000-lilith-ai-clipart/9057.png` | 0.976888 | 0.963572 | 0.963287 | 0.960042 | 1.7244% |
| 87 / `DIV2K/0244.png` | 8.585008 | 8.584242 | 8.584296 | 8.584002 | 0.0117% |
| 88 / `DIV2K/0633.png` | 6.464547 | 6.461513 | 6.461753 | 6.459847 | 0.0727% |
| 89 / `jxl-testdata/images/textures_pk02/pk02_wall05a_C.png` | 11.587555 | 11.584045 | 11.583771 | 11.587555 | 0.0000% |
| 90 / `diverse-downloads/kenney_particle_pack/PNG (Transparent)/star_09.png` | 0.582245 | 0.582245 | 0.582245 | 0.582245 | 0.0000% |
| 91 / `diverse-downloads/polyhaven/rock_wall_16/rock_wall_16_Displacement_1k.png` | 9.704369 | 9.703003 | 9.703354 | 9.701225 | 0.0324% |
| 92 / `local-train/img_480x360_3x16bit_RGB_bars_horizontal_triangle_0032_2xhalfPI.png` | 0.008472 | 0.008472 | 0.008472 | 0.008472 | 0.0000% |
| 93 / `Set14/barbara.png` | 8.890413 | 8.837963 | 8.842458 | 8.827083 | 0.7123% |
| 94 / `jxl-testdata/images/screenshot_web/duckduckgo.com.png` | 0.363712 | 0.360792 | 0.359838 | 0.360540 | 0.8720% |
| 95 / `Set5/woman.png` | 6.489188 | 6.486842 | 6.486434 | 6.486128 | 0.0472% |
| 96 / `DIV2K/0341.png` | 8.192761 | 8.192525 | 8.192435 | 8.192726 | 0.0004% |
| 97 / `diverse-downloads/imazen26/9094-lilith-ai-illustrations/9119.png` | 10.865321 | 10.864105 | 10.864456 | 10.863902 | 0.0131% |
| 98 / `jxl-testdata/images/icon_64/actions-address-book-new.png` | 7.972656 | 7.962891 | 7.945313 | 7.929688 | 0.5390% |
| 99 / `jxl-testdata/images/textures_photo/IMGP5487_seamless.png` | 4.667809 | 4.666985 | 4.667145 | 4.667809 | 0.0000% |
| 100 / `jxl-testdata/images/pngimg/cursor_PNG65.png` | 0.447937 | 0.443451 | 0.442230 | 0.440613 | 1.6351% |

## Validation

The benchmark harness tests passed (9 tests), as did the reused PNG/packed-input
adapter suite (10 tests). The completed report was independently checked for
100 unique manifest entries, matching source SHA256 and dimensions, exactly
400 verified outputs, and zero file-size regressions. The source encoder and
binary were not modified or rebuilt during this run.
