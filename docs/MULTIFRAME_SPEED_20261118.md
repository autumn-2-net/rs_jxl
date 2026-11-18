# Encoding throughput after 080edb9

Windows x86_64 on the same Ryzen 9 7945HX. The baseline is the archived release DLL from commit 080edb9. Both builds enable the CLI codec feature experimental-lz77-hotspots. Rates include Python, FFI, and output copies.

## Automatic multiframe encoding

Four threads, effort 9, one warmup and three measured calls per build, alternating order. Each call checks original samples, shape, durations and supplied ICC. Each stream hash below matches the corresponding official-decoder-checked baseline from the preceding compression experiment. Aggregate rates divide the total logical pixels by the sum of the case median durations.

| Sequence | Bytes | Encode MPix/s before / after | Decode MPix/s before / after |
| --- | ---: | ---: | ---: |
| revisit-first | 450,868 | 0.372 / 0.382 | 11.153 / 11.095 |
| revisit-middle | 600,822 | 0.337 / 0.337 | 8.222 / 8.385 |
| repeated-crop | 376,207 | 0.330 / 0.325 | 15.214 / 14.386 |
| subtile-changes | 164,206 | 0.321 / 0.333 | 27.851 / 26.942 |
| flat-to-textured | 58,915 | 0.339 / 0.347 | 11.565 / 12.884 |
| textured-to-flat | 50,594 | 0.811 / 0.806 | 18.837 / 18.908 |
| hidden-rgb-transition | 59,118 | 0.282 / 0.279 | 10.231 / 8.947 |
| t3-matching-pair | 371,056 | 0.446 / 0.448 | 45.680 / 47.842 |
| photo-with-flat-intro-and-edits | 165,848 | 0.961 / 0.991 | 68.727 / 69.571 |
| **Development and controls** | **2,297,634** | **0.538 / 0.546** | **30.836 / 30.983** |
| holdout-t1-p0-p1 | 10,943 | 5.311 / 5.611 | 94.984 / 95.186 |
| holdout-t2-p0-p1 | 2,409,055 | 0.566 / 0.632 | 31.311 / 32.446 |
| **Real pairs** | **2,419,998** | **1.945 / 2.137** | **66.893 / 68.024** |

## Pixel input, crop planning and single-frame encoding

Four-thread public API calls with a 2048 x 1024 RGB gradient. Each comparison uses one warmup, seven alternating timed calls and the median. Analysis compares a frame and its bitwise inverse. Crop encodes two identical frames using effort 3 and automatic reference choice. Still encoding uses the first frame at effort 3. Analysis geometry matches across builds. Encoded bytes match and decode to the input pixels.

| Operation | Input | MPix/s before / after |
| --- | --- | ---: |
| analysis | uint8 | 101.51 / 639.78 |
| crop | uint8 | 21.28 / 60.37 |
| still | uint8 | 24.01 / 31.32 |
| analysis | uint16 | 71.69 / 341.34 |
| crop | uint16 | 19.31 / 55.20 |
| still | uint16 | 23.14 / 31.68 |

The crop planner compares complete integer rows and scans only the portions that can enlarge the crop. On the first transition, the previous and first references refer to the same input. The planner now scans it once. The automatic encoder assigns reusable rectangle payloads to individual workers in unbounded trials and restores their caches in canonical order. Masks copy changed runs from the bitmap, and a filled difference box can immediately supply the cost-guided partition.

Full-width U8/U16 input skips redundant sample range scans. Reduced depths and U32 retain their range checks. The FFI reads aligned little-endian word buffers in bulk and retains chunk reads for unaligned input.

Raw samples and the focused pixel benchmark are in the local directory `out/multiframe-speed2/` as `dev.json`, `real.json`, `other.json` and `other.py`. To rerun the multiframe measurements, use `tools/benchmark_multiframe_feedback.py` with the preceding compression report's manifests, the two DLLs, four threads, effort 9, one warmup and three repetitions.

## Layer throughput after b12fb74

Windows x86_64, release build, effort 9, four workers. The explicit layer
tests alternate the two libraries in the same process, discard one warmup,
and measure nine calls per library. The four pairs contain tiled colors,
a gradient, random invisible RGB and crops from two photographs. Each call
checks the original layer samples and the encoded stream hash. Aggregate
MPix/s divides the total layer pixels by the sum of the per-case median times.

| Explicit layer order | Encoded bytes in either build | Encode MPix/s before / after |
| --- | ---: | ---: |
| Small layer first, two group sizes tried | 211,263 | 0.318 / 0.476 |
| Same pairs in reverse order | 211,261 | 0.620 / 0.618 |

Both eligible group-size trials can run in the existing encoder worker pool.
They may use more temporary memory than sequential trials. Wire-size ties
keep the original grouping. Single-thread encoding retains its byte cutoff.

The raw decode benchmark reads all eight small RGBA layers, including
partially clipped layers, and checks their original samples. Throughput counts
the 60,140 layer pixels returned per call. It excludes the unreturned canvas.

| Canvas | Background | Median ms before / after | Raw MPix/s before / after |
| --- | --- | ---: | ---: |
| 192 × 192 | Empty | 8.89 / 8.83 | 6.76 / 6.81 |
| 1536 × 1536 | Empty | 55.31 / 10.13 | 1.09 / 5.94 |
| 1536 × 1536 | Previous reference | 64.86 / 63.07 | 0.93 / 0.95 |

Local scripts and individual measurements are under
`out/multiframe-speed-20261118-b12fb74/` in `layers_bench.py`,
`layers.json`, `raw_bench.py` and `raw.json`.

### Automatic animation encoding on the same builds

When a multipart Replace trial has an incumbent byte cutoff, two workers can
try adjacent rectangles using the remaining byte budget at the start of the
batch. The encoder counts every dispatched pixel and consumes results in
bitstream order. An explicit pixel-work or output-byte limit keeps the
sequential cutoff path. The benchmark uses one warmup and three alternating
measurements per build, recorded in `final.json`.

| Sequences | Bytes in either build | Encode MPix/s before / after | Coalesced decode MPix/s before / after |
| --- | ---: | ---: | ---: |
| All 16, 119 frames | 5,082,445 | 1.105 / 1.135 | 63.59 / 61.38 |
| Two real image pairs | 2,237,072 | 2.261 / 2.372 | 81.19 / 77.79 |
| Nine development sequences and controls | 2,102,904 | 0.499 / 0.542 | 29.35 / 30.03 |
| Newton's cradle and streamz demo | 638,829 | 0.197 / 0.194 | 29.78 / 27.84 |
| Two additional GIFs and synthetic crop/tint | 103,640 | 0.832 / 0.862 | 27.90 / 27.75 |

All 16 encoded streams matched the baseline byte for byte. Absolute timings
varied between rounds. Newton's cradle encoded at 0.103 / 0.123 MPix/s,
and streamz demo at 0.320 / 0.256 MPix/s in this run. A separate five-round
repeat of streamz demo measured 0.254 / 0.275 MPix/s, with the same stream
hash. Its encoding speed change had opposite signs across the two runs.
Official djxl decoded all 119 frames from both builds to their original
samples. The individual comparisons are recorded in `final-png.json`.

## Selecting a single candidate before encoding

Five local PNG sequences (50 frames, 13.724 million input pixels) were run
through Windows x86_64 `rs-jxl encode-frames` with default encoding settings.
The CLI runs include process startup, PNG decoding and file output. Each
comparison alternated two binaries or modes, with one warmup and three timed
calls. The table sums median times per sequence. The two official release
executables were obtained from GitHub and verified against the release asset
hashes. Release and working-tree comparisons were timed in separate runs.

| Build and search | Candidate trials | Output bytes | Seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| 1.2.1 default | 197 | 5,633,172 | 30.16 | 0.455 |
| 1.2.3 default | 410 | 5,175,472 | 49.10 | 0.280 |
| Working tree, `--no-candidate-screen` | 414 | 4,943,184 | 39.37 | 0.349 |
| Working tree, default | 50 | 4,984,137 | 8.59 | 1.597 |

The fixed inputs were the t2 and t3 image pairs, Newton's cradle,
repeated-crop and streamz demo. One trial means one representation per
logical frame. A selected representation can contain several coded rectangles.
Official djxl reproduced the original pixel samples of all 50 frames.

On the separate 16-sequence Python effort-9 run (four threads, 119 frames),
the complete-search baseline totaled 5,082,445 bytes and single-choice
encoding produced 5,112,155 bytes (+0.58%). Every frame matched its input.
For the eight explicit-layer cases above, all streams matched the two-trial
baseline; eligible small-first pairs encoded at 0.386 / 0.490 MPix/s.
The local command-line runner, reports and Python comparisons are in
`out/multiframe-screen-20261118/`.
