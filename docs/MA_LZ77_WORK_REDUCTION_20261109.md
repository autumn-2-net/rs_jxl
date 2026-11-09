# MA And LZ77 Work-Reduction Experiments

## Decision

Neither experiment demonstrated a useful whole-image encoding speedup. Both
were removed, including the LZ77 frontier mutation change. Production encoder
code remains at baseline `24a625944925c15a81c26b34023f99c95aa0a8f3`.
This is a negative experiment report, not an enabled optimization or a claim
that MA/LZ77 cannot be accelerated further.

No effort, sample count, predictor, split threshold, candidate, search limit,
format syntax, decoder or public API was changed. Compressed JXL byte identity
was not an acceptance criterion: original integer samples and encoded size were
checked separately.

## Candidates

Both versions included an exact LZ77 frontier update: replace the first
dominated entry in place and drain only additional dominated entries, instead
of draining the entire replaced range and inserting again. Comparisons and
candidate order were unchanged. A differential test checked every intermediate
frontier against the old operation, including ties and more than 16 entries.
Its independent performance contribution was not isolated; the measurements
below are combined candidates, not proof that the frontier change alone is
slower or faster.

### A: Omit The Last Occupied MA Bin

The threshold sweep never moves the last occupied bin to the left. Its
right-side contribution is already in the parent histogram. For uncached,
unpacked, non-coarsened, non-sharded scratch, omit this bin's collection and
truncate storage before it. Cached histograms remain complete for descendants.
This adds a per-record branch in affected collectors.

Four images, alternating builds, four measured calls plus one warmup each:

| Sample | Before seconds | Candidate A seconds | Speedup |
| --- | ---: | ---: | ---: |
| Tecnick RGB_OR_1200x1200_023 | 1.1048 | 1.1520 | 0.959x |
| news.ycombinator.com | 0.5452 | 0.5791 | 0.941x |
| PDF 9bde6aaf | 1.9323 | 1.9392 | 0.996x |
| wallhaven-mlzoy1 | 6.0250 | 5.8552 | 1.029x |

Sum of per-image medians: 9.6072 -> 9.5254 seconds, about 0.85% less time.
This is not convincing: two cases became slower and the large-image candidate
samples spanned 5.455-10.215 seconds. All 128 original-sample checks passed.

### B: Bound Scratch Without Skipping Records

Keep every sample and all collector loops unchanged. Only bound the four
temporary count/extra-bit buffers at `last_occupied_bin + 1`. Avoid clearing
unreachable trailing rows, without a new per-record branch. Cached, packed,
coarsened and sharded layouts retain their original storage sizing.

Eleven images, the same repetition settings:

| Sample | bpp, both builds | Before seconds | Candidate B seconds | Candidate B MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Tecnick RGB_OR_1200x1200_023 | 8.5050 | 1.1836 | 1.2138 | 1.186 |
| news.ycombinator.com | 0.4617 | 0.5540 | 0.5428 | 3.540 |
| PDF 9bde6aaf | 2.2665 | 1.7645 | 1.8080 | 0.805 |
| kodim19 | 8.8897 | 1.1618 | 1.1524 | 0.341 |
| Blobby_volley_2_alpha_6 | 3.5529 | 0.2371 | 0.2454 | 1.945 |
| clover | 12.7677 | 1.2200 | 1.2161 | 0.862 |
| actions-mail-send-receive | 1.0395 | 0.4922 | 0.4855 | 0.540 |
| cvo9xd_keong_macan_grayscale | 2.8872 | 0.1848 | 0.1833 | 1.364 |
| vgqcws_vin_709_g1, 16-bit | 28.8613 | 0.0400 | 0.0403 | 0.102 |
| wallhaven-49yekn | 2.7293 | 1.1737 | 1.1470 | 1.808 |
| wallhaven-mlzoy1 | 13.6754 | 6.1596 | 6.1735 | 1.344 |

The 11 images contain 17.621 million spatial pixels. Sum of per-image median
API times: 14.1711 -> 14.2081 seconds (+0.26%); corresponding throughput is
1.243 -> 1.240 MPix/s. Native codec times, which exclude Python transport and
output copies, total 13.9565 -> 13.9210 seconds (-0.25%). Both differences are
too small relative to observed variation to establish a speedup.

All encoded byte counts and bpp were unchanged. All 352 original-sample checks
passed. Tiny-image throughput includes fixed API/codec overhead and should not
be interpreted as sustained large-image throughput.

## Measurement And Validation

- Windows, Ryzen 9 7945HX, Release thin LTO, one codegen unit.
- Both encoders use e9, eight threads, integer NumPy input and
  `source_search=False`: the ordinary source route with its internal tool
  selection, not the optional outer full-candidate scheduler.
- Reusable CFFI contexts, alternating before/after call order. Timers exclude
  input IO, context construction, hashing and verification. The table uses the
  whole `Encoder.encode(array)` call; native codec time is reported separately.
- Both native libraries decode every encoded candidate, including warmups,
  against original integer samples with hidden RGB included. Timed decoder
  checks use a fixed baseline stream. No decoder performance gain is claimed.
- MA tests compare every candidate score, penalty and predictor against full
  cached histograms, including compacted tokens, paired collectors, weighted
  samples, sparse/wide bins, scratch reuse, refinement and optional layouts.
- All 213 focused MA tests passed on candidate B; the unchanged LZ77 candidate
  had already passed all 248 LZ77 tests, including the new frontier regression.
- No independent official-decoder, full-corpus or cross-platform run was added
  for these rejected candidates. They are not being promoted to production.

## Evidence

Ignored local artifacts under `out/ma-tail-20261109/`:

- `before.dll` / `before.exe`: frozen baseline.
- `after.dll` / `after.exe`, `primary.json`: candidate A and four-image results.
- `bounded.dll` / `bounded.exe`, `bounded-all.json`: candidate B and 11-image results.
- `bounded-experiment.patch`: candidate B source/tests against the baseline,
  retained only for reproducing the rejected experiment.

The reusable harness is `tools/benchmark_pixel_calls.py`. Repetition samples,
min/max timing, byte counts, bpp and sample-check counts are retained in JSON;
binary hashes identify the measured builds, not desired compressed output.

The result rejects these particular redundant-memory-work changes as useful
whole-encoder speed optimizations on this subset. It does not establish an
algorithmic performance ceiling, nor justify reducing search effort silently.
