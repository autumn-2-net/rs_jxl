# Sparse mask trials with source Add patches

Baseline commit `94accb2`. For auto multiframe searches using source Add patch detection, the encoder now offers a plain Multiply/Add pair when at most one eighth of the difference bounding box changes. The density limit controls extra search time. Every other candidate keeps its original pixel and patch options. A local byte comparison selects the winner. The existing mask switches apply. Component and tile patch configurations keep their previous eligibility rules.

## Multiframe measurements

Windows x86_64, effort 9, four threads. The paired public Python/FFI runs alternate build order and use one warmup and three measured repetitions. Throughput uses total logical input pixels divided by summed case median times. Each call checks original samples including hidden color, frame timing and supplied ICC.

| Case | Complete bytes before / after | Encode MPix/s before / after | Decode MPix/s before / after |
| --- | ---: | ---: | ---: |
| revisit-first | 450,868 / 450,868 | 0.378 / 0.392 | 11.251 / 11.202 |
| revisit-middle | 600,822 / 600,822 | 0.346 / 0.338 | 8.013 / 8.074 |
| repeated-crop | 376,207 / 376,207 | 0.346 / 0.338 | 15.104 / 15.022 |
| subtile-changes | 164,206 / 151,080 | 0.347 / 0.268 | 30.431 / 15.910 |
| flat-to-textured | 58,915 / 53,168 | 0.353 / 0.294 | 12.829 / 13.079 |
| textured-to-flat | 50,594 / 50,594 | 0.887 / 0.627 | 20.108 / 19.779 |
| hidden-rgb-transition | 59,118 / 53,209 | 0.301 / 0.249 | 10.884 / 10.903 |
| t3-matching-pair | 371,056 / 371,056 | 0.482 / 0.490 | 49.916 / 50.059 |
| photo-with-flat-intro-and-edits | 165,848 / 165,848 | 1.033 / 1.031 | 71.454 / 69.235 |
| **Development and controls** | **2,297,634 / 2,272,852** | **0.574 / 0.555** | **31.869 / 30.217** |
| holdout-t1-p0-p1 | 10,943 / 10,943 | 6.256 / 6.581 | 105.784 / 99.700 |
| holdout-t2-p0-p1 | 2,409,055 / 2,409,055 | 0.633 / 0.641 | 31.537 / 32.210 |
| **Real pairs** | **2,419,998 / 2,419,998** | **2.207 / 2.260** | **71.180 / 69.590** |

Three sequences with sparse edits get smaller. On subtile changes, decoding the mask layers across a large bounding box costs more than decoding the old small Replace rectangles. Both real pairs produce the same stream hashes as the baseline. Speed variation on these two cases should be interpreted in that context.

Official djxl 0.12 sample comparisons passed on all 18 before/after development streams. A native regression exercises an active source Add atlas on a 16-bit gray/alpha background followed by three winning Multiply/Add updates and checks every displayed sample and duration.

## JPEG reconstruction experiment

Eight existing diagnostics, including the metadata fixture, at effort 9 and four threads. Only the Brotli quality for JPEG reconstruction data varied. Each input/setting uses one warmup and three timed encodes and reconstructions. Every native call recovers the exact input JPEG. Official djxl independently recovers all 24 distinct input/setting pairs. Sizes below cover complete JXL files.

| Brotli quality | Total bytes | Encode MPix/s | Reconstruct MPix/s |
| ---: | ---: | ---: | ---: |
| 9 | 3,489,847 | 3.106 | 126.477 |
| 10 | 3,489,847 | 3.060 | 122.311 |
| 11 | 3,489,847 | 3.101 | 123.574 |

Effort 9 currently selects reconstruction quality 10 by default. The local raw results and per-image hashes are in `out/multiframe-compression2/{dev,real,jpeg}.json`, with the official multiframe check in `oracle/receipt.json`. Reproduction uses `tools/benchmark_multiframe_feedback.py` with the existing nine-case manifest and the two real pairs from the preceding speed report.
