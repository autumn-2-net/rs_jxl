# Signed temporal differences for automatic multiframe encoding

The encoder can store a changed rectangle as one standard JPEG XL Add layer.
Each channel carries the signed difference from the selected reference. When a
textured region changes by a small set of values, this can be considerably
cheaper than encoding its new pixels. The automatic search measures the complete
layer and can still choose Full, crop, masks or rectangle partitions.

## Measurement

Windows x86_64, effort 9, four threads, Python reusable Encoder and Decoder.
The baseline includes the multiframe implementation at `135a1b9` and the same
local JPEG transcoder changes as the new build. Each pair alternated build
order, with one warmup and three measured repetitions per build. Throughput
includes input/output copies and uses total logical input pixels divided by
the sum of per-sequence median times. Sizes cover complete JXL files.

| Sequence | Bytes before → after | Encode MPix/s before → after | Decode MPix/s before → after |
| --- | ---: | ---: | ---: |
| revisit-first | 450,868 → 450,868 | 0.387 → 0.392 | 11.647 → 11.599 |
| revisit-middle | 600,822 → 600,822 | 0.334 → 0.276 | 8.479 → 6.320 |
| repeated-crop | 376,207 → 207,720 | 0.238 → 0.195 | 13.271 → 17.624 |
| subtile-changes | 151,080 → 150,655 | 0.184 → 0.155 | 11.694 → 15.425 |
| flat-to-textured | 53,168 → 53,044 | 0.192 → 0.197 | 9.166 → 9.881 |
| textured-to-flat | 50,594 → 50,594 | 0.430 → 0.390 | 11.987 → 14.020 |
| hidden-rgb-transition | 53,209 → 53,109 | 0.167 → 0.159 | 7.694 → 8.258 |
| t3-matching-pair | 371,056 → 371,056 | 0.334 → 0.342 | 36.432 → 36.890 |
| photo-with-flat-intro-and-edits | 165,848 → 165,088 | 0.734 → 0.752 | 54.916 → 64.854 |
| **Development sequences and controls** | **2,272,852 → 2,102,956** | **0.403 → 0.387** | **25.292 → 26.407** |
| newtonscradle, all 36 frames | 439,388 → 393,531 | 0.115 → 0.080 | 11.198 → 12.323 |
| streamz_demo, first 6 of 18 frames | 256,585 → 256,585 | 0.190 → 0.180 | 41.861 → 40.444 |
| **Previously unused GIF animations** | **695,973 → 650,116** | **0.159 → 0.132** | **23.116 → 24.135** |
| holdout-t1-p0-p1 | 10,943 → 10,943 | 4.747 → 4.514 | 72.634 → 73.209 |
| holdout-t2-p0-p1 | 2,409,055 → 2,409,055 | 0.469 → 0.462 | 24.997 → 23.893 |
| **Existing real-image pairs** | **2,419,998 → 2,419,998** | **1.647 → 1.605** | **52.122 → 51.331** |
| **All 13 sequences** | **5,388,823 → 5,173,070** | **0.856 → 0.779** | **44.429 → 44.449** |

Five of the nine development/control files shrank. Three signed Add frames in
repeated-crop saved 168,487 bytes of the 169,896-byte total reduction.
Disabling `allow_add_delta` on that sequence reproduced the baseline file
byte for byte.
In the previously unused Newton's cradle sequence, signed Add won 25 frames.
The encoder tried 248 candidates across that animation, up from 186. The
streamz_demo excerpt retained its original stream hash and used four crop
layers and one Multiply/Add mask after its initial frame. Timings vary even on
identical output, especially for short decodes. Both existing real-image pairs
produced byte-identical streams. Across these 76 input frames the files shrank
4.00%, encoding throughput dropped 9.0%, and aggregate decode throughput was
within 0.1% of baseline.

## Verification and reproduction

All 104 measured and warmup encode/decode pairs checked decoded integer samples, shapes,
durations and supplied ICC bytes. Repeated encodes also had to match the
first file hash for that build. For independent verification, official djxl
0.12.0 (`a1ebdf2`) exported 8-bit animated PNGs for frame-by-frame sample
comparison, including RGB under zero alpha. An additional 36-frame grayscale
test exercises 32 consecutive winning Add layers, a fresh Full reference, and
changes spanning −255 to +255. Both native decoding and djxl reproduced the
original samples and durations. On repeated-crop, djxl's intermediate float
export reaches 1.0000001 at white. Its actual 8-bit output is exact.

`tools/benchmark_multiframe_feedback.py` used
`out/multiframe-rate-20261117/inputs/manifest.json` for development,
`out/multiframe-compression-20261117/holdout-inputs/manifest.json` for the
two real pairs, and `out/multiframe-add/gif-inputs/manifest.json` for the
animations. The GIF inputs are the displayed RGB frames from imageio's
`newtonscradle.gif` (200 × 150) and hvPlot's `streamz_demo.gif`
(1009 × 424), with the original millisecond durations. To recover the test
inputs, seek each source GIF frame in order with Pillow, convert its coalesced
canvas to RGB and save the contiguous HWC array. Pass each frame's duration
in milliseconds with `tps=(1000, 1)`. The input GIF SHA-256 prefixes are
`a663c4e076b5c48e` and `7f36720fc820866c`, respectively. Local JSON reports
under `out/multiframe-add/` record inputs, build hashes and individual trials.

## Palette indexing throughput

This comparison starts from the AddDelta build measured above. Multichannel
palette discovery counts consecutive equal sample tuples in one lookup per
run. Screened palette ordering reuses the raster indices for run counts, tile
sampling and the final index plane. Color comparison includes RGB under zero
alpha and continues across row boundaries. Both builds use the same candidate
set and scoring policy.

The same 13 sequences and 76 frames were measured again with effort 9, four
threads, one warmup and three timed repetitions per build, alternating build
order. Every complete output matched the SHA-256 of the corresponding AddDelta
file checked previously with djxl. All 104 encode/decode pairs passed the
sample, frame duration and supplied ICC checks.

| Input group | Complete bytes in each build | Encode MPix/s before → after | Encode gain | Decode MPix/s before → after |
| --- | ---: | ---: | ---: | ---: |
| Nine development sequences and controls | 2,102,956 | 0.388 → 0.430 | +10.9% | 27.677 → 26.039 |
| Two GIF animations | 650,116 | 0.132 → 0.159 | +20.5% | 25.855 → 25.677 |
| Two existing real-image pairs | 2,419,998 | 1.648 → 1.769 | +7.3% | 49.222 → 48.921 |
| **All 13 sequences** | **5,173,070** | **0.788 → 0.892** | **+13.3%** | **43.721 → 43.156** |

Newton's cradle improved from 0.081 to 0.093 encode MPix/s and the
streamz_demo excerpt from 0.179 to 0.226. On revisit-middle, the three-run
medians were 0.312 and 0.255 MPix/s. A separate run with two warmups and
eleven measured repetitions yielded 0.247 and 0.244 MPix/s on that sequence.
The table uses the original three repetitions for every sequence. Decoder
code was unchanged. Local measurement records are in
`out/multiframe-speed3/{dev,gifs,real,revisit}.json`.
