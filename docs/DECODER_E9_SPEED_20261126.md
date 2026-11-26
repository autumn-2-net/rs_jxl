# E9 pixel-lossless decoding on 200 images

The corpus is the 200-PNG effort benchmark from 2026-11-25 at `eb868e7`.
Each regenerated E9 stream matched its archived SHA-256. The 182 eight-bit
and 18 sixteen-bit images contain 619,577,505 spatial pixels.

## Measurement

On Windows x86-64 (Ryzen 9 7945HX), the baseline and candidates were built
with Rust 1.97.1 using `cargo +stable build --release --locked -p rs-jxl-ffi`.
The timer covers a public Python `Decoder.decode(bytes)` call through the C ABI
and materialization of an owned native-format NumPy array. Each thread setting
reuses a decoder. Input I/O, PNG decoding and verification take place outside
the timer. Every result is compared with the original samples, including alpha
and invisible RGB, and checked for dimensions, channel layout and sample type.

Each library gets one warmup per image and thread setting. The final comparison
alternates library order over two measured calls. Throughput divides total
spatial megapixels by the sum of per-image median times. Changes are compared
within the same run because absolute throughput varies between runs.

## Output buffer

The C ABI takes ownership of the first decoded eight-bit pixel buffer as its
result storage. Further frames and extra planes append at the usual byte
offsets. Wider samples still use the little-endian packer. Python materializes
its independently owned NumPy output as before. This saves one full pixel copy
for a single eight-bit frame.

## Results

| Threads | Baseline MPix/s | Output buffer change MPix/s | Throughput change |
| ---: | ---: | ---: | ---: |
| 1 | 9.443 | 9.436 | −0.07% |
| 8 | 55.048 | 56.631 | +2.88% |
| 16 | 88.337 | 92.946 | +5.22% |

At 16 threads the eight-bit subset rose from 93.62 to 99.49 MPix/s (+6.27%).
The sixteen-bit subset measured 64.08 and 64.51 MPix/s. A separate eight-thread
run with three measured calls per image gave 61.18 and 62.25 MPix/s (+1.75%)
on the full corpus.

Two Modular decoder experiments were dropped. Compacting terminal MA-tree
splits added 0.75% over the output buffer change in the 200-image eight-thread
trial, but reduced throughput on the 12-image development set at one and eight
threads. Specializing previous-channel property calculation added 0.53% in
the final eight-thread comparison and lost 0.38% in a separate 200-image
comparison at that thread count.

## Other decoder paths

The existing 29-case decoder manifest ran at eight threads, with one warmup
and three timed calls per library. JPEG reconstruction was byte-exact. Pixel
and multiframe outputs matched original samples, including floating-point
words. The six VarDCT renders matched the previously recorded render outputs.

| Family | Cases | Baseline MPix/s | Output buffer change MPix/s |
| --- | ---: | ---: | ---: |
| Pixel output | 11 | 53.49 | 56.07 |
| JPEG reconstruction | 11 | 155.47 | 156.27 |
| VarDCT rendering | 6 | 117.38 | 119.03 |
| Multiframe | 1 | 41.82 | 46.28 |

The JPEG byte reconstruction path bypasses pixel output packing. Its results
serve as a control. The single animation measures only that fixture.

The FFI unit tests, filtered core decoder unit tests and five decoder integration
targets passed (145 tests in total). Two opt-in benchmarks were skipped.

Local data under `out/decode-e9-20261126/` include `prepare.py`, `bench.py`,
`final.json` (per-image samples), `other-ffi.json` (fixtures and hashes),
`comparison.csv` and `decode-e9.png`.
