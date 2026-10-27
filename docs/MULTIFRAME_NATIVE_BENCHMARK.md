# Native Multiframe Update: 2026-10-27

This update moves no difference/search algorithm into Python. Exact difference
maps, legacy tiles, adaptive rectangle proposals, trial ordering and proven-size
pruning run in Rust. Python and CLI use the same native implementation.

## Method

- Windows x86-64, release Rust DLL, efforts 9 and 5, four encoder threads,
  `reference="auto"`, duration 1 for every input state.
- Before: retained native DLL from the architecture checkpoint at `f2beb1a`.
  After: the native multiframe update based on that commit.
- Files are read and decoded once through the native API before measurement.
  JPEG inputs to pixel modes are decoded pixels, never JPEG coefficient transcodes.
  All compared modes receive the same arrays and supported color/sample metadata.
- Timings cover the public encode/decode API, including Python packing, FFI,
  codec setup and returned arrays. Initial source decoding, filesystem reads
  and sample verification are excluded. This is not pure kernel throughput.
- Old/new codec benchmarks run sequentially, without concurrent codec benchmarks
  or compiler jobs. The e5 principal table is a separate timing rerun. Each table
  uses one measured call per mode, not a median or confidence interval; unrelated
  host activity was not controlled.
- Every generated stream was decoded and checked against all original native
  samples, including alpha and invisible RGB. Compressed JXL identity is not
  an acceptance criterion. Output streams stay in memory; reports go to disk.
- bpp = total JXL bytes * 8 / (width * height * logical input frame count).
  MPix/s uses that same total logical-frame pixel count, not submitted trial work.

Fixtures:

| Set | Native-decoded dimensions | Frames | Inputs |
| --- | --- | ---: | --- |
| t1 | 4096 x 4096 RGBA | 4 | `146845733_p0.png` through `p3.png` |
| t2 | 2000 x 2183 RGB | 4 | `145702319_p0.jpg` through `p3.jpg` |
| t3-matched | 638 x 900 RGB | 2 | `146810648_p1.jpg`, `146810648_p2.jpg` |

The omitted t3 `p0` has different dimensions. No image was resized or padded to
make it fit. The two matching images were copied to an ignored benchmark folder.

## Effort 9: Primary Comparison

The initial e5 check was not sufficient to assess the usual e9 workflow. The
same inputs were therefore tested at e9 before committing the update. All pixel
modes passed exact native-sample recovery, and both JPEG-transcode cases passed
original-byte reconstruction. The benchmark script now defaults to effort 9.

`pruned` disables the new adaptive proposal but keeps shared analysis and safe
pruning. `adaptive` enables both controls. `New legacy` disables both controls
while retaining the shared native difference-map implementation.

| Set | Mode | Bytes | bpp | Encode s | Encode MPix/s | Decode s | Decode MPix/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t1 | Before auto | 13,831 | 0.001649 | 35.910 | 1.869 | 1.161 | 57.778 |
| t1 | New legacy | 13,831 | 0.001649 | 37.554 | 1.787 | 1.149 | 58.426 |
| t1 | After pruned | 13,831 | 0.001649 | 15.587 | 4.305 | 1.126 | 59.621 |
| t1 | After adaptive | 12,478 | 0.001487 | 15.489 | 4.333 | 2.547 | 26.351 |
| t2 | Before auto | 4,019,080 | 1.841081 | 18.559 | 0.941 | 0.498 | 35.068 |
| t2 | New legacy | 4,019,080 | 1.841081 | 23.763 | 0.735 | 0.576 | 30.305 |
| t2 | After pruned | 4,019,080 | 1.841081 | 20.614 | 0.847 | 0.572 | 30.513 |
| t2 | After adaptive | 3,767,749 | 1.725950 | 39.318 | 0.444 | 3.755 | 4.650 |
| t3-matched | Before auto | 380,878 | 2.653278 | 0.655 | 1.752 | 0.032 | 35.540 |
| t3-matched | New legacy | 380,878 | 2.653278 | 0.664 | 1.730 | 0.030 | 37.706 |
| t3-matched | After pruned | 380,878 | 2.653278 | 0.651 | 1.763 | 0.027 | 43.246 |
| t3-matched | After adaptive | 380,878 | 2.653278 | 1.630 | 0.705 | 0.028 | 41.044 |

### Independent Compression at e9

Each independent pixel still is encoded at the same e9/four-thread settings,
using the same already-decoded source samples and color metadata. This is the
sum of complete still JXL files, not an uncompressed byte count, original PNG
size or JPEG byte-transcode size. All timings below sum the sequential calls.

| Set | Independent bytes | Independent bpp | Encode s | Encode MPix/s | Decode s | Decode MPix/s | Adaptive bytes | Saving vs independent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t1 | 45,714 | 0.005450 | 20.105 | 3.338 | 1.228 | 54.666 | 12,478 | 72.70% |
| t2 | 4,098,261 | 1.877353 | 11.067 | 1.578 | 0.460 | 37.942 | 3,767,749 | 8.06% |
| t3-matched | 381,504 | 2.657638 | 0.670 | 1.714 | 0.026 | 43.453 | 380,878 | 0.16% |

Other e9 comparisons, with byte-reversible JPEG deliberately separate:

| Set | Crop sequence bytes | Crop encode s | Independent JPEG-transcode bytes | JPEG bpp | JPEG encode s | JPEG reconstruct s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t1 | 17,900 | 10.438 | n/a | n/a | n/a | n/a |
| t2 | 4,019,080 | 7.798 | 3,864,602 | 1.770317 | 3.455 | 0.110 |
| t3-matched | 380,878 | 0.670 | 639,786 | 4.456886 | 0.408 | 0.014 |

### Interpretation at e9

- t1 adaptive saves 9.78% versus old auto, with encode time falling from 35.91 s
  to 15.49 s in this run. The final state uses ten layers, and decode time rises
  from 1.16 s to 2.55 s. Pruning alone retains the old size at 15.59 s.
- t2 adaptive saves 6.25% versus old auto but increases encode time from 18.56 s
  to 39.32 s and decode time from 0.50 s to 3.76 s. Changed states use 32, 31 and
  32 layers. This is a size gain, not a general speed gain.
- t2 pruning is faster than new-DLL legacy in this run, but slower than the old
  DLL. Single-call measurements cannot isolate runtime variance from a true
  performance regression; the old-DLL speedup observed at e5 does not establish
  one at e9.
- t3-matched adaptive finds no smaller file, while encode time rises from 0.66 s
  to 1.63 s. Its 0.16% saving against independent stills is small and does not
  demonstrate useful inter-frame prediction.
- Adaptive native planning took 0.0504 s, 0.0170 s and 0.0005 s for t1/t2/t3.
  Completed/pruned candidates were 9/11, 5/9 and 3/0. The remaining costs are
  trial coding and decoding many layers, not a slow Python analysis step.

All legacy candidate sizes match the old DLL; these three fixture groups show
no output-size regression. They do show latency regressions and do not establish
a universal performance improvement. Disable `adaptive_rectangles` when its
extra search/layers are not worth the measured size gain.

## Effort 5: Initial Comparison

`pruned` disables only the new adaptive proposal, retaining the legacy candidate
set with shared analysis and safe pruning. `adaptive` enables both new controls,
with the existing default of at most 32 rectangles per proposal.

| Set | Mode | Bytes | bpp | Encode s | Encode MPix/s | Decode s | Decode MPix/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t1 | Before auto | 14,124 | 0.001684 | 24.111 | 2.783 | 1.350 | 49.718 |
| t1 | After pruned | 14,124 | 0.001684 | 8.297 | 8.088 | 1.357 | 49.451 |
| t1 | After adaptive | 12,990 | 0.001549 | 10.177 | 6.594 | 2.912 | 23.049 |
| t2 | Before auto | 4,394,690 | 2.013142 | 6.392 | 2.732 | 0.532 | 32.804 |
| t2 | After pruned | 4,394,690 | 2.013142 | 5.695 | 3.066 | 0.530 | 32.944 |
| t2 | After adaptive | 4,119,799 | 1.887219 | 9.691 | 1.802 | 3.624 | 4.820 |
| t3-matched | Before auto | 416,849 | 2.903859 | 0.220 | 5.230 | 0.027 | 41.920 |
| t3-matched | After pruned | 416,849 | 2.903859 | 0.205 | 5.594 | 0.030 | 37.686 |
| t3-matched | After adaptive | 416,849 | 2.903859 | 0.464 | 2.475 | 0.033 | 35.121 |

The first broader pass also measured new-DLL legacy ordering with pruning and
adaptive proposals disabled. Its t1/t2/t3 sizes were identical to the old DLL.
Those timings overlapped validation work, so they are not the principal speed
comparison above. All raw measurements are retained in the ignored report root.

## Effort 5: Native Work Reports

| Set | Mode | Completed | Pruned | Submitted trial pixels | Analysis s |
| --- | --- | ---: | ---: | ---: | ---: |
| t1 | Pruned | 7 | 10 | 101,597,184 | 0.0522 |
| t1 | Adaptive | 9 | 11 | 104,525,824 | 0.0869 |
| t2 | Pruned | 5 | 4 | 32,829,504 | 0.0145 |
| t2 | Adaptive | 5 | 9 | 41,347,412 | 0.0154 |
| t3-matched | Pruned | 2 | 0 | 1,148,400 | 0.0005 |
| t3-matched | Adaptive | 3 | 0 | 1,683,176 | 0.0006 |

Submitted pixels include an entire layer when it is submitted, even if some
internal sections are subsequently skipped. They are not a count of completed
pixel-processing operations. Pruned size bounds are not completed encoded sizes.

## Effort 5: Tradeoffs

- t1: pruning preserves 14,124 bytes and cuts measured encode time by 65.59%.
  Adaptive merging saves another 8.03% in size. Its final state uses ten layers,
  and decode time rises from roughly 1.35 s to 2.91 s.
- t2: pruning preserves size and reduces measured encode time by 10.90%.
  Adaptive merging saves 6.26%, but encode time rises by 51.61% versus old auto.
  Its three changed states use 32, 31 and 32 layers, respectively. Decoding is
  substantially slower; this mode is not a general latency win.
- t3-matched: adaptive search finds no smaller output and performs one extra
  candidate encode. Its absolute overhead is about 0.25 s on this pair.
- Analysis is cheap on these inputs, but actual candidate coding and coalescing
  remain significant. This update does not claim to solve those backend costs.

For the old candidate set with safe pruning:

```python
data, report = rs_jxl.encode_multiframe(
    frames, strategy="auto", effort=5, threads=4, return_info=True,
    auto_options={"prune": True, "adaptive_rectangles": False},
)
```

To explore fewer layers, set `max_rectangles` explicitly. That changes the
candidate set and is a caller-controlled size/read-latency tradeoff, not an
implicit downgrade. Defaults still select the smallest actual candidate bytes;
they do not optimize encode/decode time.

## Effort 5: Independent Stills and JPEG Reconstruction

These sizes came from the broader pass at the same effort/thread settings:

| Set | Crop sequence | Independent pixel-JXL sum | Adaptive sequence | Independent JPEG-transcode sum |
| --- | ---: | ---: | ---: | ---: |
| t1 | 18,598 | 47,191 | 12,990 | n/a |
| t2 | 4,394,690 | 4,483,534 | 4,119,799 | 3,893,232 |
| t3-matched | 416,849 | 417,475 | 416,849 | 642,732 |

The JPEG-transcode column is a separate byte-reconstruction contract; every
original JPEG was recovered byte-for-byte. It is not a pixel-mode candidate and
was never silently substituted for one. Independent sums include separate still
containers; multiframe sizes include one sequence container.

## Reproduce

From the Rust repository, with a native library already built:

```powershell
python scripts/benchmark_multiframe_native.py `
  --native target/release/rs_jxl_ffi.dll `
  --folders ../jxl_python/test_img/mt_lay/t1 ../jxl_python/test_img/mt_lay/t2 `
  --effort 9 --threads 4 --out out/multiframe-check
```

For an older DLL, invoke a separate process with `--native <old-library>` and
`--legacy --modes auto`; no new `auto_options` are sent. The script writes both
JSON and Markdown, exits nonzero on failures, and refuses mixed geometry or
unsupported common metadata instead of silently normalizing it.

Use `--effort 5` to repeat the initial lower-effort comparison.

Local records are under `out/multiframe-native-20261027`: `before-e9`, `after-e9`,
`before-e5`, `after-e5`, `isolated-before-e5`, and `isolated-after-e5`.
Full JSON retains per-frame native
candidate reports, including winners and proven lower bounds. These three small
fixture groups do not establish performance across a large corpus, other effort
levels, operating systems, or CPUs. See [MULTIFRAME_AUTO.md](MULTIFRAME_AUTO.md)
for correctness tests and official-decoder compatibility acceptance.
