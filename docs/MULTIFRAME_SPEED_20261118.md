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
