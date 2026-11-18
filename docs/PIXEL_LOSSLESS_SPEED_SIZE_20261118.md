# Single-frame pixel-lossless encoding and decoding

The baseline is `a6d98dd`. All Windows x86_64 release builds used
`rs-jxl/experimental-lz77-hotspots`. The public Python Encoder and Decoder
used eight threads and integer HWC arrays, with `source_search=False`.

## Implementation

- The bit writer appends aligned bytes by copying a slice and unaligned bytes
  by shifting whole words with a carry. It initializes spare storage in bounded
  batches for subsequent writes. The writer checks the complete bit budget
  before appending and preserves partially written bytes.
- Ordinary efforts 7 through 9 use the existing sampled palette-order selector
  and flat palette lookup. Explicit tool overrides take precedence. For nested
  context maps of 32 through 256 entries, the encoder compares actual prefix and
  ANS bit lengths. The ANS calculation counts extra bits, state and each 16-bit
  renormalization using the same state transitions as emission.
- The decoder can expand explicit zero-predictor palettes from typed I16 or I32
  rows after checking the shared indices. Groups that use the implicit palette
  keep their general lookup path.

## First checkpoint on the fixed 200-image corpus

The existing ten-stratum corpus contains 619,577,505 pixels. The manifest is
`out/static-context-corpus200-20261112/sample200.json`, SHA-256
`b87a4b7a023ace9df0d59818c4ff1a28feb4e02a211932ba4b65577a84fdc9f5`.
The baseline DLL hash starts with `7076d1dd901368e8` and the first checkpoint hash
with `44bcebfd8225d566`.

At fast effort 1, all 200 files kept the same individual sizes, totaling
647,286,556 bytes. Encoding improved **131.164 → 149.833 MPix/s** (+14.23%).
Fixed-stream decoding measured **175.935 → 175.819 MPix/s** (−0.07%).

At high-compression effort 9, files totaled **424,240,942 → 424,231,729 bytes**
(−0.00217%). Of the 200 files, 147 shrank and 53 kept their size. Encoding
measured **2.327 → 2.338 MPix/s** (+0.47%) and decoding
**53.624 → 54.250 MPix/s** (+1.17%). Screened ordering was already active
at effort 9 in the baseline.

At default effort 7, complete files totaled **454,940,039 → 453,449,019 bytes**
(−0.328%), or 5.87420 → 5.85494 bits per pixel. **152** images shrank, **46**
kept the same size, and **2** grew by 3 and 10 bytes. Encoding measured
**6.480 → 6.498 MPix/s** (+0.27%) and decoding **48.004 → 48.236 MPix/s**
(+0.48%). These aggregate time differences are small.

| Effort 7 stratum | Images | Bytes saved | Size change |
| --- | ---: | ---: | ---: |
| Document and web | 24 | 781,546 | −1.09% |
| Mixed formats | 20 | 338,197 | −0.21% |
| High-resolution photos | 32 | 231,812 | −0.23% |
| Other photos | 24 | 81,838 | −0.36% |
| Game and UI | 16 | 28,135 | −1.86% |
| Mixed local | 16 | 19,497 | −0.07% |
| Urban | 16 | 5,471 | −0.04% |
| Alpha icons | 20 | 3,641 | −0.04% |
| Textures | 16 | 444 | −0.002% |
| Manga | 16 | 439 | −0.003% |

Game/UI encoding measured 7.081 → 6.874 MPix/s and manga 4.906 → 4.639 MPix/s.
The largest individual saving was 645,230 bytes on the Graveyard poster.

## Development set and verification

Development used five local originals (Kodak, game screenshot, photographic
texture, icon and web screenshot) plus two synthetic 37-color RGBA patterns
with nonzero colors under zero alpha, in 8 and 16 bits. All seven input hashes
are distinct from the 200-image manifest.

| Effort on seven development images | Bytes before → after | Encode MPix/s before → after | Decode MPix/s before → after |
| --- | ---: | ---: | ---: |
| 1 | 6,196,452 → 6,196,452 | 179.107 → 201.812 | 184.631 → 183.311 |
| 7 | 4,825,010 → 4,821,755 | 5.104 → 5.059 | 42.238 → 42.438 |
| 9 | 4,643,512 → 4,643,229 | 1.651 → 1.656 | 36.708 → 36.174 |

Times cover in-memory calls including array transport and output copies.
Codec instances are reused. Builds alternate order. At efforts 7 and 9 each corpus
image has one measured encode per build. Effort 1 corpus timings and all
development timings use the median of three encodes after one warmup.
Decoder timings use the median of four
calls on the same frozen baseline stream for both builds. Each table divides
total spatial pixels by the sum of the corresponding per-image times.

For every input, both libraries cross-decoded both outputs and official djxl
also decoded the new stream. Original integer samples were checked including
RGB under zero alpha. Packed PNG inputs were expanded to 8-bit samples and
16-bit inputs retained their precision. PNG metadata and file I/O are outside
this pixel-API measurement. Local per-image data, hashes and the benchmark
driver are under `out/pixel-standard-20261118/`.

## Effort 9 throughput follow-up

On Kodak image 02, a one-thread instrumented encode took 2.34 s. MA learning
accounted for 1.85 s, including 0.95 s in split sweeps and 0.40 s gathering
samples. These inclusive times guided the changes. Performance measurements
use release binaries with profiling disabled.

- The native MA scorer skips the logarithm calculation for an all-zero SIMD
  block while preserving signed zero and reduction order. The exact lookup
  table now covers populations through 512. The active numerical profile uses
  527,364 bytes for this table, up from 132,612. A direct comparison against
  the 256 cutoff on eight development images gave a 1.55% aggregate encoding
  gain, with mixed individual results.
- Leaf offset signaling caches repeated integer population costs across
  sweeps. ANS normalization reuses the rounded targets and balancing-bin
  selection across precision trials. A table maps every rounded target to the
  same representable count as the original bit-mask calculation.

The final effort 9 corpus run rotates all six orders of three binaries. Each
image has one whole encode per binary and six timed decodes of the same
baseline stream per binary. The harness checks complete-file byte identity
between the two optimization checkpoints, all nine native encode/decode
combinations, and official djxl decoding of the final file. The final binary
hash starts with `f58446b16815078c`; measurements are in `e9-final-200.jsonl`.

| Effort 9, same 200 images and run | Complete bytes | Encode MPix/s | Fixed-stream decode MPix/s |
| --- | ---: | ---: | ---: |
| Pushed baseline `a6d98dd` | 424,240,942 | 2.0931 | 49.8493 |
| First checkpoint | 424,231,729 | 2.0968 | 49.7669 |
| Final e9 optimization | 424,231,729 | 2.1522 | 50.0358 |

Final encoding throughput improved 2.82% over the pushed baseline and 2.64%
over the first checkpoint. The first checkpoint's 9,213-byte reduction was
preserved exactly on every file. Fixed-stream decoding measured 0.37% higher
than the pushed baseline. Encoding gains varied by stratum, from +7.30% for
mixed formats and +6.53% for game/UI to −3.05% for manga. The paired
three-build measurements above are the comparison for the final encoder.
