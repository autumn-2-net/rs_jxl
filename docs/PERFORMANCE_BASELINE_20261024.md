# Lossless Performance Baseline: 2026-10-24

## Scope

Measurement only. No encoder/decoder algorithms were changed for this run.
Special JPEG formats, wide-integer gaps, lossy encoding, WSL and macOS were
outside scope. This is a small diagnostic subset, not a production corpus
acceptance or a population estimate.

- Windows x86_64, AMD Ryzen 9 7945HX; native processes, no WSL.
- Official libjxl 0.12.0, existing local clang-cl 17 Release binaries.
- Optimized C++ frozen source `b866b3defec13494bfbaf705db93581cda118e37`.
  See [build provenance](CPP_REFERENCE_BUILD.md). Its `djxl` was built from the
  existing isolated build directory for this run; no C++ source was changed.
- Current Rust worktree including the pending decoder integration. Rust 1.97.1,
  normal Cargo Release, thin LTO, one codegen unit. This is not a debug binary
  or the reduced-LTO development test profile.
- e9; eight workers requested; files and codec processes run serially.
  Official defaults versus the C++ full scheduler and Rust `--source-search`.
  Equal effort labels do not imply equal work or identical search policies.
- One excluded warmup and three measured rounds, rotating execution order;
  per-image median wall times. No GPU, tuning overrides, or profiling enabled.

## Sample Selection

Seed `20261024`. Eight PNG strata: Kodak, Tecnick, game UI, web screenshot,
photo texture, alpha icon, grayscale, and 16-bit color. Random selection was
made from sorted eligible files without consulting performance or size results.
PNG strata use 65,536..2,500,000 pixels, except the available 16-bit fixtures,
which are only 64x64. No image was resized or cropped.

Two explicit historical regressions are added: the PDF screenshot and the
1080p wallhaven image. These are **not random**. The web and grayscale strata
each have only one eligible file. Three JPEG files are sampled from `test_img`
without a resolution cap; these include a 5120x2880 image.

Ten PNGs total 9.326413 MPix; three JPEGs total 26.640384 MPix.

PNG inputs are identical metadata-stripped snapshots, preserving IDAT and all
original samples, including invisible RGB. This measures pixel coding, not
metadata retention. JPEGs are passed unchanged to the coefficient transcoders;
there is no JPEG-to-PNG conversion or lossy/pixel fallback.

## Timing And Correctness

Encoding includes CLI startup, source read, encoding and JXL output write;
verification is outside timing. Decode benchmarks use `--disable_output`, with
ten in-process repetitions per invocation and three measured invocations.
Wall time divided by ten includes amortized process startup and input reading,
as well as allocations and decoder setup. No PNG serialization is timed.

The C++ no-output path emits float32 buffers, so Rust explicitly uses `--bits
32` for comparable PNG decode work. C++ retains its runner across repetitions;
Rust currently creates one per decode API call. That overhead remains included.
These are application throughput numbers, **not isolated kernel throughput**.
JPEG decoding below means original-byte reconstruction, not IDCT pixel rendering.

Every one of 156 encoded outputs was verified by all three decoders:

- 360 exact PNG sample comparisons, including invisible RGB: passed.
- 108 exact original JPEG byte comparisons: passed.
- Decode speed also measured every encoder/decoder combination: 468 process
  batches including warmups, ten decodes per batch, all successful.
- No compressed JXL byte identity requirement. No timeouts or skipped failures.

Aggregate bpp is `8 * sum(JXL bytes) / sum(pixels)`. Aggregate MPix/s is
`sum(pixels) / 1e6 / sum(per-image median seconds)`, not a mean of rates.

## PNG Results

The decode column uses the **same official-produced streams** for each decoder.

| Codec | Total JXL bytes | bpp | Encode seconds | Encode MPix/s | Decode MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official 0.12 | 5,258,917 | 4.5110 | 9.040 | 1.032 | 37.017 |
| Optimized C++ | 4,781,088 | 4.1011 | 54.388 | 0.171 | 36.601 |
| Rust | 4,778,743 | 4.0991 | 68.455 | 0.136 | 31.835 |

Rust takes 25.9% more encoding time than optimized C++, for only 0.049% fewer
bytes. Against official 0.12 it saves 9.13% bytes but takes 7.57x the time.
The C++ scheduler itself takes 6.02x official time for a similar size gain;
the entire gap is not a Rust-language or port-overhead issue.

Excluding the two explicit regressions, the eight stratified samples give:

| Codec | bpp | Encode seconds | Encode MPix/s |
| --- | ---: | ---: | ---: |
| Official 0.12 | 5.6828 | 7.068 | 0.820 |
| Optimized C++ | 5.4313 | 36.463 | 0.159 |
| Rust | 5.4283 | 42.807 | 0.135 |

Thus Rust's additional time versus C++ is 17.4% without those regressions.
This stratified subset is still not an unbiased estimate of arbitrary images.

## JPEG Archival Results

The reconstruction column again uses identical official-produced JXL streams.

| Codec | Total JXL bytes | bpp | Transcode seconds | Transcode MPix/s | Reconstruction MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official 0.12 | 9,276,600 | 2.7857 | 14.071 | 1.893 | 87.069 |
| Optimized C++ | 9,276,600 | 2.7857 | 10.839 | 2.458 | 86.575 |
| Rust | 9,268,541 | 2.7833 | 17.318 | 1.538 | 31.705 |

Rust takes 59.8% more transcode time than optimized C++, for 0.087% fewer bytes.
Original-byte reconstruction is the larger deficit: Rust takes 2.73x C++ time
on identical streams. This is separate from PNG pixel decoding and has not yet
been attributed to a specific kernel by profiling.

## Per-Image Encoding

Each cell is `bpp / seconds / MPix/s`.

| Input | Dimensions | Official 0.12 | Optimized C++ | Rust |
| --- | --- | --- | --- | --- |
| Kodak kodim19 | 512x768 | 8.9082 / 2.300 / 0.171 | 8.9082 / 8.254 / 0.048 | 8.8897 / 8.646 / 0.045 |
| Tecnick RGB_OR_1200x1200_023 | 1200x1200 | 8.5081 / 1.201 / 1.199 | 8.5081 / 4.927 / 0.292 | 8.5050 / 6.155 / 0.234 |
| Blobby_volley_2_alpha_6 | 797x599 | 3.5546 / 0.302 / 1.583 | 3.1773 / 6.685 / 0.071 | 3.1774 / 8.676 / 0.055 |
| news.ycombinator.com | 1325x1450 | 0.5072 / 0.430 / 4.468 | 0.2442 / 3.875 / 0.496 | 0.2443 / 4.868 / 0.395 |
| clover | 1024x1024 | 12.7831 / 1.212 / 0.865 | 12.1115 / 7.544 / 0.139 | 12.1045 / 9.264 / 0.113 |
| actions-mail-send-receive | 512x512 | 1.0387 / 1.192 / 0.220 | 0.8452 / 3.961 / 0.066 | 0.8466 / 3.806 / 0.069 |
| cvo9xd_keong_macan_grayscale | 500x500 | 2.8849 / 0.330 / 0.757 | 2.8849 / 1.068 / 0.234 | 2.8872 / 1.217 / 0.205 |
| vgqcws_vin_709_g1, RGB16 | 64x64 | 28.8574 / 0.101 / 0.041 | 24.5469 / 0.148 / 0.028 | 24.5508 / 0.174 / 0.023 |
| PDF 9bde6aaf... | 1264x1152 | 2.3784 / 0.962 / 1.513 | 0.8054 / 14.318 / 0.102 | 0.8057 / 20.880 / 0.070 |
| wallhaven-49yekn | 1920x1080 | 2.7327 / 1.010 / 2.052 | 2.6970 / 3.607 / 0.575 | 2.6962 / 4.768 / 0.435 |
| JPEG wallhaven-yqxzqx | 5120x2880 | 4.3686 / 10.771 / 1.369 | 4.3686 / 8.099 / 1.821 | 4.3647 / 13.451 / 1.096 |
| JPEG wallhaven-9oo8k1 | 4096x2304 | 0.9593 / 2.970 / 3.178 | 0.9593 / 2.424 / 3.894 | 0.9586 / 3.460 / 2.728 |
| JPEG wallhaven-vpyekp | 1920x1280 | 0.3020 / 0.330 / 7.445 | 0.3020 / 0.317 / 7.764 | 0.3019 / 0.407 / 6.034 |

## Bitstream Complexity Matters

Decoding each codec's **own** PNG outputs gives 37.017 MPix/s for official,
18.432 for C++, and 18.702 for Rust. That does not mean the C++ decoder became
twice as slow: decoding identical official streams gives 37.017 versus 36.601.
The optimized streams themselves are more expensive to decode in these cases.

For example, the game UI shrinks from 212,122 to 189,608 bytes in C++, but the
official decoder falls from 46.26 to 11.90 MPix/s. On that same smaller C++
stream, Rust achieves 12.47 MPix/s. The full report has the 3x3 decode matrix
for every input, preventing encoder layout costs from being mistaken for decoder
implementation costs.

## Next Profiling Targets

1. Rust JPEG reconstruction: the clearest same-input decode performance gap.
2. Complete PNG search: identify repeated work that yields negligible size gains
   on natural images; do not silently replace e9 with a lower effort.
3. Rust-specific encode hotspots versus C++ at similar output sizes, especially
   the PDF case (20.880 vs 14.318 seconds).
4. Track decode cost alongside size and encoding time when comparing candidate
   layouts. Smaller output alone can conceal a large decoder regression.

These are priorities inferred from measurements, not completed optimizations or
proof of individual hotspots. Broad corpus coverage and large 16-bit performance
remain unmeasured here.

## Artifacts And Reproduction

- Driver: [benchmark_codec_subset.py](../tools/benchmark_codec_subset.py).
- Four focused driver contract tests: `tools/test_benchmark_codec_subset.py`.
- Generated full report and 3x3 decode tables:
  `out/codec-subset-20261024/REPORT.md`.
- Raw metrics, selected paths, binary IDs and exact subprocess commands:
  `out/codec-subset-20261024/report.json` and `trials.jsonl`.
- Retained Rust baseline: `out/codec-subset-20261024/rs-jxl-baseline.exe`.
  SHA256 `c1925893631227f2fae1cbeb702540383f5bbd3b555822420c051f3fd6325638`.
- Optimized decoder SHA256:
  `4305a87f4f04f7e63b2ac10801e9317f70d8776b291a24c672fb0bd0cbde0b74`.

Binary hashes identify executables only; compressed byte equality is not a goal.
All generated files and executable snapshots remain in ignored `out/`.
Temporary encoded/decoded images were removed. No commits or pushes were made.

Run from this repository using a Python environment with Pillow, NumPy and
OpenCV; choose a new output directory on every run:

```powershell
python -B tools/benchmark_codec_subset.py `
  --dataset-root ../jxl_python `
  --official ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/cjxl.exe `
  --official-decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe `
  --cpp out/cpp-b866b3d/libjxl/tools/cjxl.exe `
  --cpp-decoder out/cpp-b866b3d/libjxl/tools/djxl.exe `
  --rust target/release/rs-jxl.exe `
  --out out/codec-subset-new `
  --threads 8 --warmups 1 --repeats 3 --decode-reps 10
```
