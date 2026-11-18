# JPEG coefficient compression at high effort

Baseline commit `135a1b9`. The comparison uses the direct JPEG-to-JXL coefficient path with effort 9, decoding speed 0 and four threads on Windows x86_64. All sizes count complete JXL containers.

## Encoder choices

The AC entropy encoder marks every context used by the final group token streams, including LZ lengths and distances. It tries carrying the preceding histogram ID across unused entries, extending the first used ID to the leading gap. Every table retains at least one reference. A candidate must beat the original map in serialized bits, including its nested entropy model. The mapping of every emitted token stays fixed. This applies to high-effort JPEG AC models.

For direct transcodes at efforts 9 through 11 with decoding speed zero, learned auxiliary Modular trees now use the existing symmetric MA learner with half description cost. The search scheduler continues to use its original auxiliary-tree policy. The JPEG samples, coefficient values, and reconstruction metadata use the existing standard syntax.

## Complete-file measurements

| Set | Original JPEG bytes | Previous JXL bytes | Map change alone | Map + symmetric MA | Reduction | Smaller / equal / larger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed diagnostics and photos (45) | 33,406,801 | 27,397,679 | 27,395,321 | 27,372,064 | 0.093% | 44 / 1 / 0 |
| Additional CC12m photos (12) | 2,004,343 | 1,346,627 | 1,346,144 | 1,343,752 | 0.213% | 11 / 0 / 1 |
| Total (57) | 35,411,144 | 28,744,306 | 28,741,465 | 28,715,816 | 0.099% | 55 / 1 / 1 |

Across both sets the context map accounts for 2,841 saved bytes. Switching the auxiliary trees saves a further 25,649 bytes with the map change active. The isolated map change produced 52 smaller containers.

| Set | Input MPix | Encode MPix/s before / after | Reconstruct MPix/s before / after |
| --- | ---: | ---: | ---: |
| Fixed diagnostics and photos (45) | 106.200 | 1.99 / 3.03 | 93.92 / 96.00 |
| Additional CC12m photos (12) | 7.191 | 2.56 / 3.48 | 108.98 / 112.32 |
| Total (57) | 113.391 | 2.02 / 3.05 | 94.75 / 96.89 |

One additional CC12m photo, `000003759.jpg`, grew by 16 bytes.

## Method and verification

The 45-file manifest is `../jxl_python/out/jpeg-opt-20261116/cases.json`. It includes 16 format fixtures, nine project JPEGs and 20 CC12m JPEGs. Some format fixtures share image content. The additional 12 JPEGs were sampled from the same local CC12m directory after excluding those paths, using Python 3.11, sorted filenames and seed 20261118. All 57 original file hashes are distinct. The additional CC12m IDs are 000003759, 000001155, 000001991, 000000432, 000000264, 000000106, 000001292, 000002209, 000001849, 000002848, 000004015, 000002447.

Paired release FFI builds used the same experimental-lz77-hotspots feature. For each input, the public Python Encoder and Decoder ran one warmup and three timed repetitions per build in alternating order. Throughput is total input pixels divided by the sum of per-image median times. Measurements include FFI and returned bytes. File I/O, image dimension discovery, worker creation and byte comparisons are outside the timer.

Every timed encode reproduced its warmup stream hash. Native reconstruction returned the original JPEG on every call. Official djxl 0.12 reconstructed each of the 57 old, 57 new and 57 map-only outputs byte for byte. The map-only ablation used the current `jpeg_entropy_lab` example at effort 9 and four threads with its original auxiliary-tree policy. A separate baseline-420 fixture check also covered direct efforts 3, 8, 10 and 11 and effort 9 source search.

An earlier trial compared general LZ matching on raw and move-to-front AC context maps against the existing map writer. All eight diagnostic files tied the baseline in complete bytes. This trial was dropped.

Local receipts and drivers are in `out/jpeg-map-20261118`. The file `full.json` records original and output hashes, sizes and timings, `fill.json` records the ablation including section costs, and `screen.json` records the discarded trial.
