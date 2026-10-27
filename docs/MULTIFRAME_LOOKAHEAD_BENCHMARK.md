# Multiframe Lookahead And Payload Reuse

## Changes

This update follows the opt-in retained-reference work based on `1a4e85a`.
All encoding decisions and reuse are native Rust. Python/C/CLI only forward
controls and expose native reports. No decoder or bitstream extension is needed.

1. Optional one-input lookahead prevents some premature reference evictions.
   It compares exact next-input difference geometry, does not encode the future,
   and reuses surviving difference maps at the next iteration. It remains a
   heuristic with only two retained canvases, not a sequence-wide optimum.
2. Identical unmasked crop payloads can be shared between candidates within one
   logical input. Frame headers/TOCs are regenerated for the actual source/save
   slots and timing; sample-dependent Modular coding is not repeated on a hit.
3. Configured patch detection does not by itself prevent caching. A no-plan
   result permits ordinary payload reuse. Actual generated patch atlases retain
   the original path and are prepared once, not twice. Mask contents are not
   substituted with ordinary crops. Pruned tails release unused cache entries.

Use lookahead explicitly; payload reuse is automatic:

```python
data, report = rs_jxl.encode_multiframe(
    frames, strategy="auto", reference="auto", effort=9, threads=4,
    auto_options={"retained_reference": True, "reference_lookahead": True},
    return_info=True,
)
```

The CLI flags are `--retained-reference --reference-lookahead`. The complete
contracts and counters are in [MULTIFRAME_AUTO.md](MULTIFRAME_AUTO.md).

## Method

Measured on Windows x86-64, 2026-10-27, release native DLL, e9 and four threads.
The previous DLL was preserved before code changes; the before measurements do
not use a different effort or a reconstructed source revision. Both versions use
identical input arrays and settings. Builds/tests were not running during timing.
Each row is a single pass, not a statistical median or universal speed claim.

Public API times include packing, FFI and codec setup but exclude initial input
IO/decoding and subsequent correctness checks. bpp is total bytes times eight
divided by total logical canvas pixels. Compressed bytes remain in memory only.
Every benchmark row verifies original decoded samples and timing. JPEG sources
in these pixel sequences are not byte-reversible JPEG transcodes.

## Synthetic Cases

The three deterministic fixtures in `scripts/benchmark_multiframe_reuse.py`
have four 256 x 192 RGB frames each. High-entropy random images deliberately
make missed reuse costly; these are capability tests, not natural-image
throughput representatives. Repeated-crop changes are contained in the same
192 x 128 rectangle; first/previous candidates encode identical crop samples.

| Case | Version / Policy | Bytes | bpp | Encode s | Encode MPix/s | Cache Hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Revisit first, X A B X2 | Previous / retained | 645,606 | 26.269775 | 4.219 | 0.046599 | n/a |
| Revisit first | Current / retained | 645,606 | 26.269775 | 3.415 | 0.057569 | 12 |
| Revisit first | Current / lookahead | 484,425 | 19.711304 | 3.403 | 0.057778 | 6 |
| Revisit middle, X A B A2 | Previous / retained | 484,436 | 19.711751 | 3.743 | 0.052521 | n/a |
| Revisit middle | Current / retained | 484,436 | 19.711751 | 3.592 | 0.054738 | 6 |
| Revisit middle | Current / lookahead | 484,436 | 19.711751 | 3.514 | 0.055946 | 6 |
| Repeated crop | Previous / retained | 403,989 | 16.438354 | 5.504 | 0.035719 | n/a |
| Repeated crop | Current / retained | 403,989 | 16.438354 | 3.099 | 0.063438 | 10 |
| Repeated crop | Current / lookahead | 403,989 | 16.438354 | 3.305 | 0.059488 | 10 |

Lookahead removes a **24.97% size regression relative to the previous retained
policy** in the first-revisit fixture. Static first/previous already achieves
484,425 bytes there, so this is not a new gain over that policy's best result.
The middle-revisit fixture retains the existing retained policy's benefit,
where static first/previous produces 645,549 bytes. Lookahead handles both
fixtures without the caller selecting a different reference policy for each.

Repeated-crop output length is unchanged, with an observed **43.70% encoding
time reduction** for the same retained policy. The ten cache hits avoid coding
98,304 pixel areas again; they do not mean ten unique source images or ten
additional references. `trial_pixels` deliberately still counts all submitted
rectangles, including hits, so explicit work-budget semantics stay stable.

Current independent-still totals are 645,769 / 645,712 / 645,941 bytes for first
revisit / middle revisit / repeated crop. Full generated reports include decode
seconds, decode MPix/s, candidate timings and reference maps. In the current
lookahead runs, decode times are respectively 0.0377 / 0.0441 / 0.0339 seconds.
These small timings are noisy, and this change does not optimize the decoder.

## Real Sequences

| Sequence | Version / Policy | Bytes | bpp | Encode s | Encode MPix/s | Decode s | Decode MPix/s | Cache Hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t1, 4 x 4096 x 4096 RGBA | Previous / retained | 12,478 | 0.001487 | 20.217 | 3.319 | 2.975 | 22.555 | n/a |
| t1 | Current / retained | 12,478 | 0.001487 | 21.315 | 3.148 | 3.418 | 19.631 | 0 |
| t1 | Current / lookahead | 12,478 | 0.001487 | 22.137 | 3.032 | 3.334 | 20.129 | 0 |
| t2, 4 x 2000 x 2183 RGB | Previous / retained | 3,767,749 | 1.725950 | 36.671 | 0.476 | 4.428 | 3.944 | n/a |
| t2 | Current / retained | 3,767,749 | 1.725950 | 39.509 | 0.442 | 4.431 | 3.941 | 30 |
| t2 | Current / lookahead | 3,623,903 | 1.660056 | 41.305 | 0.423 | 4.875 | 3.582 | 0 |

t1 has no size benefit or cache hits. Do not claim a real-image speedup from
this update: neither real sequence was faster in these single passes. t2's
same-policy cache avoids recoding 1,614,377 submitted pixel areas, but that did
not yield a lower measured wall time; the synthetic cache timing is not a
substitute for this observation.

Lookahead makes t2 **3.82% smaller**, at a measured 1.80 s extra encode time
relative to the current retained policy, or 4.63 s relative to the previous
DLL. It retains logical input 1 while encoding input 2: the latter still uses
`MergedFirst`, but overwrites its own source slot rather than discarding input
1. Input 3 then selects `MergedRetained` against input 1, reducing that layer
candidate from 1,006,071 to 862,225 bytes. Native reference reports confirm the
actual history; this is not inferred just from a smaller output file.

Lookahead analysis totals 0.0958 s for t1 and 0.0203 s for t2, with six future
map comparisons per sequence. The larger remaining wall time is candidate
encoding and pixel transport, not tens of seconds of lookahead scanning.
Current t2 lookahead output is 11.57% below the previously measured 4,098,261-byte
independent pixel-lossless sum; that still-image sum was not rerun in this update.

## Verification

- 169 Python tests passed against the final release DLL, including independent
  official djxl 0.12/0.13 checks and CUDA array-transfer tests.
- 10 native lookahead tests and the actual generated-patch coexistence test
  passed after the patch-aware cache adjustment. These check exact hidden RGB,
  8/16-bit samples, float words, extra-only changes, zero durations, same-slot
  read/save, multi-rectangle composites, ties and fresh contexts.
- 38 encoding tests passed after the adjustment, including six payload-cache
  tests, budget/error precedence and actual patch bypass. One existing optional
  budget-specific official-decoder test was not run; the Python official-decoder
  comparisons above are separate tests.
- The pruning cache-release unit test passed. Earlier in this update, 48 native
  multiframe/context tests, all 46 FFI tests and 18 focused CLI tests also passed.
- Strict all-target Clippy and workspace formatting passed; 12 benchmark-harness
  tests passed. This is not a full-corpus or cross-platform runtime acceptance.

Correctness compares decoded samples with the input, not compressed-byte hashes.
Lookahead remains opt-in: geometry is not encoded size, and farther-future useful
states can still be evicted. No general no-regression or optimality claim is made.

## Reproduction

```powershell
python -B scripts/benchmark_multiframe_reuse.py --native path/to/previous/rs_jxl_ffi.dll --modes auto-adaptive auto-retained independent --out out/reuse-before.json
python -B scripts/benchmark_multiframe_reuse.py --native target/release/rs_jxl_ffi.dll --out out/reuse-after.json
python -B scripts/benchmark_multiframe_native.py path/to/t1 path/to/t2 --native target/release/rs_jxl_ffi.dll --effort 9 --threads 4 --modes auto-retained auto-lookahead --out out/reuse-real.json
```

The synthetic script defaults to e9/four threads. Current local artifacts live
under `out/multiframe-lookahead-e9/` as `before-synthetic`, `after-synthetic`,
`before-real` and `after-real`, each with JSON/Markdown siblings. These generated
reports and the retained baseline DLL are not committed source files.
