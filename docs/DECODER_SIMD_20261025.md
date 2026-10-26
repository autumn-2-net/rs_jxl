# Decoder SIMD Experiments, 2026-10-25

## Scope And Method

Pure Rust, one decoder thread, unchanged pre-encoded JXL inputs. No encoder,
bitstream, ARM, CI, or default feature changes. Builds, correctness tests, and
timed runs are serialized. Timings are rotating process-wall medians divided
by in-process repetitions, including amortized startup, allocation, and input
IO. Small differences are not proof of a speedup on a busy development host.

The pixel benchmark uses F32 because official `djxl --disable_output` selects
F32 regardless of the requested bit depth. U8 files, F32 files, and reconstructed
original JPEG bytes are independently verified. The default `jxl-rs` dither
table is unchanged; its existing U8 differences from libjxl are not SIMD errors.
Here all tested Rust variants produced identical U8 and F32 samples to the
preceding Rust checkpoint. Official rendering retained the known maximum U8
difference of 1. These bounded checks are not a full conformance claim.

## Existing AVX512 Versus AVX2

`decoder-avx512` was already optional. Enabling the existing wider kernels
alone did not establish a useful whole-image win on this host:

| JPEG-transcoded input | MPix | bpp | AVX2 ms | Existing AVX512 ms |
| --- | ---: | ---: | ---: | ---: |
| wallhaven-yqxzqx.jpg | 14.7456 | 4.3686 | 387.680 | 381.999 |
| wallhaven-9oo8k1.jpg | 9.4372 | 0.9593 | 148.349 | 153.628 |
| wallhaven-vpyekp.jpg | 2.4576 | 0.3020 | 33.723 | 35.747 |

Aggregate throughput: 46.758 -> 46.625 MPix/s. Three measured rounds,
one warmup, ten decodes per process. Four pixel-lossless PNG controls also
passed exact original-integer comparison, including a web and PDF screenshot;
their timing changes were approximately -1.3% to +0.8%. Do not enable all
AVX512 paths by default based on these results.

## Rejected Dual-Chain Color Loop

The YCbCr-to-RGB candidate interleaved two independent SIMD groups while
preserving constants, FMA nesting, plane order, and padded-tail behavior.
It added no unsafe production code. This creates independent work for the
CPU scheduler; it does not guarantee hardware dual issue or twice the speed.

| JPEG-transcoded input | Original AVX2 ms | Paired AVX2 ms | Throughput change |
| --- | ---: | ---: | ---: |
| wallhaven-yqxzqx.jpg | 385.890 | 383.162 | +0.71% |
| wallhaven-9oo8k1.jpg | 142.195 | 142.514 | -0.22% |
| wallhaven-vpyekp.jpg | 33.613 | 33.649 | -0.11% |

Aggregate: 47.428 -> 47.630 MPix/s, only +0.42%. This does not justify keeping
the extra loop complexity. Runtime integration and helper files were removed;
the existing color stage is restored. Four boundary-test targets passed for
scalar, SSE4.2, AVX2, and AVX512, including differing row lengths, all relevant
tail widths, unaligned slices, guards, signed zeros, and finite extreme values.

The row microbenchmark also compared single/paired AVX512. Its final version
puts the entire timed loop in an explicit `target_feature` gateway. An initial
generic-closure gateway measured roughly 6-11 ms per 2.1 MPix; explicit gateways
reduced that to roughly 0.8-1.1 ms, exposing significant harness overhead in
the first measurements. **Do not use the initial microbenchmark as evidence
about native kernel speed.** Corrected row timings were noisy and did not
establish an advantage worth overriding the whole-image result.

## Retained Optional AVX512 AC Mask

`crates/rs-jxl/src/jpeg/sequential.rs` adds a native Rust AVX512F/BW kernel
behind the existing `decoder-avx512` feature. Runtime detection still falls
back to AVX2 or scalar. The default build is unchanged. The kernel loads two
32-coefficient halves, excludes DC, checks every AC value against -2047..2047,
permutes with two-source 16-bit vectors, and directly extracts nonzero masks.
This removes the AVX2 implementation's scalar zigzag staging array. It does
not change Huffman symbols, output-budget checks, errors, or progressive scans.
The necessary unsafe loads are confined to complete 64-element arrays; the
unaligned variants require no caller alignment beyond i16.

The explicit-feature-gateway microbenchmark includes the actual maintained
modules via `bench_ac.rs`, not copied kernel formulas. Eight rotating measured
rounds follow a warmup, using 1024 blocks with varied nonzero density. Median
kernel time is **7.4017 ns/block AVX2 versus 1.6262 ns/block AVX512**, about
**4.55x** locally. This excludes the per-call runtime dispatcher, entropy
decoding, Huffman output, allocations, and all other decoder work.

The more important whole-path result is much smaller. Both checkpoints use
the same AVX512 feature set, and the rejected color change is absent in both:

| Original JPEG | bpp | Previous ms | New mask ms | New MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-yqxzqx.jpg | 4.3686 | 432.254 | 433.519 | 34.014 | 36.717 |
| wallhaven-9oo8k1.jpg (progressive control) | 0.9593 | 153.150 | 151.703 | 62.208 | 59.290 |
| wallhaven-vpyekp.jpg | 0.3020 | 18.952 | 18.781 | 130.857 | 151.249 |
| wallhaven-yq5e9x.jpg | 1.7473 | 41.238 | 40.283 | 61.060 | 71.352 |
| wallhaven-jexkwm.jpg | 1.4847 | 143.129 | 141.695 | 58.537 | 63.724 |
| wallhaven-3q3z9d.jpg | 2.3864 | 150.921 | 147.717 | 42.503 | 44.657 |

This table times **original JPEG byte reconstruction**, not pixels. Every
original JPEG was recovered byte-for-byte by both Rust versions and official
djxl. Five rotating measured rounds, one warmup, ten decodes per process.
Aggregate throughput is 46.478 -> 46.774 MPix/s (+0.64%); official is
49.502 MPix/s. The unchanged progressive path also moves by about 1%, so this
is not evidence for a reliable whole-decoder gain of 0.64%. Excluding that
control gives about +0.58%. Retain the small, demonstrably faster optional
kernel, but do not claim a broad speedup or switch the default SIMD policy.

Focused checks cover all signed i16 values at selected zigzag positions,
every DC/AC position, random density masks, invalid coefficient values, and
32 input alignments. The standalone Release checks passed all seven tests
(four existing JPEG-data tests and three mask tests). The ignored timing test
was explicitly run separately. This does not replace complete decoder tests.

Final retained-source validation also passed 331 JPEG-related unit tests
(10 ignored fixture/manual tests), 12 public JPEG/dither integration tests
(one ignored external-fixture test), and workspace/all-target Clippy with
`decoder-avx512,decoder-profile` and warnings denied. The six real-image
original-JPEG comparisons above were run independently of ignored discovery.
No commit or push was made. The last Release build in `target/release` includes
the optional AVX512 feature; an ordinary build without that feature still
selects the previous AVX2/scalar AC-mask paths.

Build the optional path with:

```text
cargo build --locked --offline --release -p rs-jxl-cli --features decoder-avx512
cargo test --locked --offline -p rs-jxl --features decoder-avx512 --lib jpeg::sequential
```

## Artifacts

All paths below are relative to `out/simd-pixels-20261025/`:

- `stock512/report.json`: wider existing pixel paths versus preceding AVX2.
- `stock512-controls/report.json`: four exact PNG controls and timings.
- `pair256/report.json`: paired color loop versus preceding AVX2.
- `acmask512/report.json`: retained mask versus previous AVX512, six JPEGs.
- `acmask-micro.txt`, `bench_ac.rs`: direct-gateway integer kernel experiment.
- `ycbcr.rs`, `ycbcr_tests.rs`: rejected candidate and corrected microbenchmark.
- `ycbcr-micro-direct.txt`: explicit-gateway row timings.
- `ycbcr-micro.txt`: superseded indirect-gateway results, NOT a native-kernel comparison.

JSON reports include binary identities, actual commands, input hashes, raw
trials, bpp, MPix/s, and pixel differences. Compressed byte equality is not an
optimization target; bpp is unchanged because these are decoder experiments.
