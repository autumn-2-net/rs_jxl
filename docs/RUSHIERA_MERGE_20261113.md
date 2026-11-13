# Rushiera Fork Integration

## Provenance

Imported source: <https://github.com/Rushiera/rs_jxl>, branch `main`, commit
`3b7442c4b866309c363e7233e67aeb9cdc3d841a`.
Local baseline: `21bb8f9`; common ancestor: `16e1d82`.
The changes are imported as one local commit, without incorporating the fork's
commit history. Source attribution and existing BSD notices are retained. No
license files, CI workflow or Python ABI definition are replaced.

## Included Changes

- Weighted-predictor cost walking: fuse prediction/update, specialize interior
  rows and reuse prepared gathering inputs without changing predictor semantics.
- MA training: fused histogram collection, cache subtraction improvements and
  detailed split-search instrumentation.
- Decoder weighted prediction: four-lane error-history additions using x86-64
  baseline SSE2, plus a uniform-prediction update shortcut. Wide predictions and
  unsigned wrapping history semantics remain unchanged. Other architectures use
  portable arithmetic; no new ARM kernels are added.
- Optional four-lane WP cost kernel and standalone `wp_microbench` example.
- Profiling parent-child edges, MA property counters and LZ77 matcher demand.
- Source CLI controls for grouping, refinement, cross-channel properties,
  buffering and existing LZ77 choices.

The incoming scheduler enables exact property-partition deduplication, small-node
dominance pruning and the SIMD mandatory-prefix rejection filter by default.
These are search-work optimizations, not changes to prediction/decoder syntax.
Distribution offset refinement remains opt-in; the preceding representative and
Delta Palette experiments are preserved.

## Integration Corrections

- LZ77 demand counters originally updated even with profiling disabled. Dispatch
  once per query to a const-specialized counted/uncounted matcher; the ordinary
  inner loop has no counter updates. Differential tests compare match lists and
  check that disabled counters stay zero.
- Scope tracking originally used only names and popped only the top element.
  Give active scopes unique identities, remove the correct live entry on drop
  and bind guards to their originating thread. Test nested, repeated-name,
  out-of-order and disabled scopes without changing process environment variables.
- Original bit depth is insufficient proof for narrow arithmetic after image
  transforms or for a public cost API accepting arbitrary integer planes. Keep
  the experiment opt-in and fall back to wide arithmetic when actual residual
  intermediates do not fit. Test adversarial wide inputs as well as ordinary
  low-bit-depth inputs.
- Preserve an observed residual-overflow flag when WP cost evaluation exits by
  pruning; early termination must not hide that a candidate was unhealthy.
- Preparing capacity for every serial gather task at once can reserve storage
  for all repeated input rows before deduplication. Prepare bounded task batches
  instead, with per-task preparation for heavy batches. This bounds speculative
  reservation, not the samples or learning domain; actual retained rows still
  grow as needed. Tests compare weighted populations, properties, sample budgets
  and capacities, including distinct streams and estimate overshoot.

## Source CLI Options

These options require `encode --source-effort N`; they do not enable search or
cross-channel properties implicitly. With `--source-search`, candidate parameters
describe the starting request; the search can test alternatives and select other
settings. Refinement switches remain disabled unless requested.

| Option | Meaning |
| --- | --- |
| `--source-group-size-shift 0..3` | Request 128/256/512/1024 pixel groups |
| `--source-buffering auto\|full\|above2048\|streaming` | Select the existing source buffering policy, still subject to eligibility gates |
| `--source-max-properties 0..16` | Request a maximum previous-channel count; effective count follows available transformed channels |
| `--source-prev-props 1\|2\|4` | At source e9/e10, choose signed residual only, signed sample plus residual, or the default four properties per previous channel |
| `--source-responsive` | Request the existing responsive/squeeze policy |
| `--source-precise-entropy` | Request the existing Tortoise entropy tier |
| `--source-refine-entropy` | Compare serialized entropy alternatives |
| `--source-refine-leaves` | Enable post-training leaf refitting |
| `--source-refine-offsets` | Enable the existing default offset-refinement configuration, not automatically Distribution mode |
| `--source-refine-context-map` | Enable serialized context-map merge proposals |
| `--source-lz77-careful-first-pass` | Request the careful first LZ77 pass |
| `--source-bucket-matcher` | Request the existing Optc256 bucket matcher |

Example, one candidate with two previous channels and two properties per channel:

```text
rs-jxl encode input.png output.jxl --source-effort 9 --threads 8 --source-max-properties 2 --source-prev-props 2
```

`RS_JXL_PROFILE=1` enables timings and parent edges; `trace` also emits scope entry
and exit. Edge attribution is thread-local, not a reconstruction of causal parentage
across Rayon tasks. Inclusive times overlap and must not be summed as wall time.
`RS_JXL_WP_NARROW=1` enables the optional WP cost experiment; it is not a default
performance guarantee or a change to the encoded predictors.

## Validation

Windows x86-64 checks:

- `cargo check --workspace --all-targets --locked --offline -j4`: passed.
- Library and CLI unit tests: 2,444 + 94 passed; 14 + 2 explicitly ignored.
- After the gather reservation correction, all 41 training tests passed.
- Focused integration suites: 18 tests passed (`weighted_simd_reference`,
  `decoder_weighted_flat`, `decode_wide_integer`, `decode_lossless_edges`,
  `decoder_parallel_pixels`). The flat-WP test covers 180 combinations of
  8/16/31-bit RGBA, five WP presets, three trees, two widths and ANS/prefix coding.
- Both revisions built with the same release command:
  `cargo build --release --locked --offline -p rs-jxl-cli -p rs-jxl-ffi -j4`.
  This uses the workspace CLI/FFI feature union for each revision.

- Python bindings, buffers, reusable contexts and lossless-combination tests:
  84 passed, 1 skipped, using the merged release DLL and an explicit official
  `djxl` oracle. Packaging and CI deployment were not rerun.
- Official libjxl 0.12 `djxl` decoded both revisions at e3/e9 on all ten originals,
  plus both complete e9 scheduler outputs on three originals: 46 pixel-verified
  outputs, no mismatch. The full scheduler cases were Kodak 06, the web screenshot
  and the historical PDF screenshot. Output sizes did not change in these cases.

Local raw evidence and frozen before/after release binaries are stored in
`out/rushiera-merge-20261113/`, outside tracked source. Pixel correctness compares
original decoded integers including invisible RGB, not identical encoded JXL bytes.
No Linux runtime, macOS, WSL or full-corpus acceptance is claimed for this merge.

## Performance Results

Ten original PNGs, 9,338,889 pixels total. Eight cases are stratified selections
from `tools/benchmark_codec_subset.py::select_subset` with seed `20261113`;
the PDF and wallhaven cases are explicit historical regressions. No resizing or
selection by performance result. Included: photographs, game/web screenshots,
texture, transparent icon, grayscale, and a small 16-bit image.

`tools/benchmark_pixel_calls.py` loads frozen before/after DLLs, alternates their
order, warms each route once, then takes per-image medians (3 calls at e9, 5 at
e3). The table sums those medians; throughput is total pixels divided by that sum,
not an arithmetic average of per-image throughputs. Encoding is native
`codec_seconds` without CLI/IO; decoding is the complete reusable Python decoder
call returning NumPy, including its output copy but excluding validation. Both
decoders verify every encoded candidate against original integer samples.

| Effort | Threads | Operation | Before seconds | After seconds | Before MPix/s | After MPix/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| e9 | 1 | Encode | 26.9892 | 25.5254 | 0.346 | 0.366 |
| e9 | 8 | Encode | 6.1528 | 5.9141 | 1.518 | 1.579 |
| e9 | 1 | Decode | 0.9782 | 0.9398 | 9.547 | 9.937 |
| e9 | 8 | Decode | 0.1926 | 0.1828 | 48.480 | 51.084 |
| e3 | 1 | Encode | 2.1632 | 2.3045 | 4.317 | 4.053 |
| e3 | 8 | Encode | 0.5961 | 0.6155 | 15.667 | 15.174 |
| e3 | 1 | Decode | 0.5379 | 0.4869 | 17.361 | 19.179 |
| e3 | 8 | Decode | 0.1182 | 0.1102 | 79.041 | 84.772 |

This is not a universal speedup: e3 native encoding took **6.53% longer at 1T**
and **3.25% longer at 8T** in this run. Several single-thread cases show consistent
regressions across repetitions; do not dismiss the aggregate as startup noise.
No specific cause is established by this A/B test, and no effort reduction or
compression fallback was used to conceal the tradeoff. At e9 the PDF case also
took approximately 3.5% longer at 8T in the reusable API measurement, despite
the aggregate improvement. Tiny changes on sub-millisecond work are not treated
as robust speed claims.

All ten e9 output sizes stayed unchanged: 4,939,353 bytes total; pixel-weighted
4.231213 bpp and arithmetic mean per-image 6.981253 bpp, before and after. The
mean is sensitive to the tiny 16-bit case. The e3 outputs also retained their
individual sizes. Size identity is an observation, not a merge acceptance rule.

Complete e9 scheduling, single CLI measurements including startup/IO:

| Original | Output bytes, both | Before seconds | After seconds |
| --- | ---: | ---: | ---: |
| Kodak 06 | 418729 | 6.457 | 5.552 |
| news.ycombinator.com.png | 58672 | 3.758 | 3.804 |
| Historical PDF 9bde | 146658 | 10.669 | 10.586 |

These three observations validate scheduling and do not establish a general
full-search speedup. Raw reports: `pixel-calls.json`, `pixel-calls-e3.json`,
`official-e9.json`, `official-e3.json`, and `official-scheduler.json` in the
local evidence directory above. Profiling and `RS_JXL_WP_NARROW` were disabled;
no performance benefit is claimed for the optional narrow kernel.
