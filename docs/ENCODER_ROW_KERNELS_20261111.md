# MA Collection And Weighted Row Experiments

## Scope

Follow-up to [the exact kernel work](LOSSLESS_KERNEL_FOLLOWUP_20261111.md),
starting from `016e765` on `main`. No encoder parameter, sample count, predictor,
MA split candidate, pruning checkpoint, LZ77 search budget or bitstream syntax
changes in this experiment.

The baseline encoder SHA256 is:

`a053ade8f627e8952c490167841c9d4e170e7e95bca698fb8148be52d28dc1c3`

## Implementation

### Rejected: Weighted Rows

`weighted::State::predict_row_and_update` is an internal row visitor used by
`cost::estimate_wp_cost`. It validates geometry and the starting cursor once,
hoists first-row/previous-row decisions, shares the two history-row offsets,
and advances the public cursor once after a successful row. Predictor formulas,
signed-property tie order, rounding and wrapping remain unchanged. Each pixel
still updates all weighted history needed by the following pixel.

The existing public predict/update methods retain their validation. A returned
visitor error advances history and cursor through the failing pixel, matching
the former fused-pixel call. Invalid geometry/cursor leaves the state unchanged.
The row helper is restricted to local states discarded if a callback panics.
No row scratch allocations or unsafe operations were introduced. This helper,
its cost integration and its four new tests were subsequently removed; neither
`weighted.rs` nor `cost.rs` differs from the starting commit in the final
working tree. The rejected diff is preserved with the local experiment files.

Cost accumulation and its 64-row pruning checks stay in their original order.
This change does not reduce the number of weighted presets considered.

### Deferred: MA Collection

For fourteen independent active predictors in identity order, the histogram
collector binds histogram/column references before scanning samples. Collection
remains one record-major pass with the same integer operations. Alias handling,
nonidentity active order and other predictor counts use the existing paths.

This is not the older `RS_JXL_MA_CACHE_PAIR14` experiment, which scans the sample
indices seven times and remains opt-in. Histogram capacities, allocations,
cache admission and subtraction are unchanged in the tested variant.

The isolated MA experiment also failed to establish a complete-call speedup.
Its runtime change was removed after the final measurements below. Only the
four collector regression tests and their test-only module registration remain
in the source tree. Default encoder behavior is unchanged from `016e765`.

## Rejected Variant Measurements

V1 contained both changes. V2 added an explicit inline hint to the row visitor.
Neither showed a useful decrease in the targeted Weighted scoring scope:

| Diagnostic | Before s | Candidate s |
|---|---:|---:|
| V1 cross6 T8, weighted header | 1.036875 | 1.184894 |
| V1 cross6 T1, weighted header | 0.920651 | 1.004709 |
| V2 cross6 T8, weighted header | 1.029278 | 1.174703 |
| V1 cross6 T8, MA collection | 0.675396 | 0.547498 |
| V1 cross6 T1, MA collection | 0.587591 | 0.421888 |
| V2 cross6 T8, MA collection | 0.741040 | 0.488895 |

These are separate, single diagnostic calls. Inclusive parallel scopes overlap.
They are not summed to derive complete-call time. An unrelated compression job
was still running on the host, so small timing changes are not idle-host proof.

Whole-encode medians (one verified warmup; three measured repetitions on cross6,
five on automatic):

| Variant / input / route | Before s | Candidate s | Time change |
|---|---:|---:|---:|
| V1 / 1eb / cross6 T8 Batch | 4.448095 | 4.667025 | +4.92% |
| V1 / 1eb / cross6 T1 ordinary | 10.593839 | 13.187661 | +24.48% |
| V2 / 1eb / cross6 T8 Batch | 4.586351 | 4.286562 | -6.54% |
| V2 / 9bde / automatic T8 | 1.728229 | 1.780876 | +3.05% |
| V2 / wallhaven-n6qzmx / automatic T8 | 3.300636 | 2.865992 | -13.17% |
| V2 / kodim01 / automatic T8 | 0.766622 | 0.794576 | +3.65% |
| V2 / 2026.6test1 / automatic T8 | 1.785105 | 1.630916 | -8.64% |
| V2 / DJI 16-bit 64x64 / automatic T8 | 0.042787 | 0.041659 | -2.63% |

All sizes and decoded samples matched the baseline/original in these runs.
This does not attribute the entire +24.48% to the row helper: unchanged scopes
also varied under load. The narrower conclusion is that the additional row API
did not improve its intended scoring kernel, whereas MA collection repeatedly
did. MA collection was then tested separately, rather than retaining both changes
based on the combined V2 results.

V1 SHA256: `9f1853d2634a68954e52866525d6ee1db9ee9541e497ddfb6d05fe8579d18968`.

V2 SHA256: `d0bf48d43945df4e654d4899048700bc8a8bdadee432036e80f349b25b3f8c1a`.

## Isolated MA Results

MA-only SHA256:
`e30e120ceb00c17351db8f4e0b8457ce7b54bde2fabb12739a62cdb1dae60d2c`.

All rows use effort 9, one verified warmup and three measured repetitions per
binary, alternating baseline/candidate order. Native encode seconds exclude CLI
startup, PNG I/O and independent decoder checks. `automatic` is the ordinary
candidate family, not an outer scheduler tournament; `cross6` fixes RCT 6.
T8 cross6 also enables the existing Chain Batch experiment; T1 does not.

| Input / route | Before s | MA-only s | Time change | Before MPix/s | MA-only MPix/s | Both bytes | Both bpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1eb / cross6 T8 Batch | 4.040382 | 4.117265 | +1.90% | 0.8915 | 0.8748 | 351302 | 0.780255 |
| 1eb / cross6 T1 | 9.933265 | 10.094038 | +1.62% | 0.3626 | 0.3568 | 351302 | 0.780255 |
| 9bde / automatic T8 | 1.428748 | 1.646032 | +15.21% | 1.0192 | 0.8846 | 412539 | 2.266499 |
| wallhaven-n6qzmx / automatic T8 | 2.664403 | 2.546144 | -4.44% | 0.7783 | 0.8144 | 459153 | 1.771424 |
| kodim01 / automatic T8 | 0.774075 | 0.814930 | +5.28% | 0.5080 | 0.4825 | 454216 | 9.241048 |
| 2026.6test1 / automatic T8 | 1.647853 | 1.796057 | +8.99% | 1.2584 | 1.1545 | 2098189 | 8.094865 |
| DJI 16-bit 64x64 / automatic T8 | 0.039379 | 0.039346 | -0.08% | 0.1040 | 0.1041 | 16825 | 32.861328 |

Separate diagnostic calls still showed a smaller `ma.cache_collect` scope on
1eb: T8 went from 0.711089 to 0.482982 s, and T1 from 0.522994 to 0.402018 s.
These inclusive timings include collection/allocation and are not isolated
microbenchmarks. In the other inputs that scope changed much less, for example
kodim01 went from 0.312156 to 0.312964 s. Local improvement therefore does not
establish an end-to-end win across the tested routes.

The host continued running a separate user compression task during measurements.
Variation is substantial: on the final T1 run, baseline trials ranged from
9.594 to 11.551 s and candidate trials from 9.945 to 13.905 s. This prevents a
clean attribution of every slowdown to this patch, but also prevents claiming
that the patch is faster. The negative rows above are not discarded as noise.
No statistically demonstrated whole-encoder gain is claimed; the runtime patch
is deferred rather than enabled by default.

## Validation And Artifacts

The isolated MA variant passed 70 official-decoder pixel checks across six
distinct inputs, counting diagnostics, warmups and measured calls. All emitted
sizes/bpp were unchanged from the baseline in these runs. PNG metadata was
removed consistently for the pixel benchmark only, after checking that this
preserved the original samples. This is not a metadata roundtrip test.

The collector tests exercise packed/wide properties, weighted/repeated sample
indices, differing histogram strides, borrowed nonzero-base ranges, aliased or
reordered predictors, non-14-predictor fallbacks, and constant populations. The
nine cache tests (four new collector tests, five existing subtraction tests)
passed on both the experimental implementation and the restored default
implementation. `cargo fmt --all -- --check` and `git diff --check` passed.
The new tests remain applicable to future collector experiments.

Compressed JXL identity is not an acceptance condition; original sample
equality, including 16-bit samples and alpha-zero RGB, is required. No new
runtime parameter, unsafe operation or format extension remains in the tree.
No search effort/sample count was reduced to obtain these timings.

Local ignored artifacts under `out/encoder-row-kernels-20261111/`:

- `cross6-t8-final.json`, `cross6-t1-final.json`, `automatic-t8-final.json`:
  isolated MA-only results, commands, source/binary hashes and profiler output.
- `cross6-t8.json`, `cross6-t1.json`, `cross6-t8-v2.json`,
  `automatic-t8-v2.json`: earlier combined-variant results, including failures
  to improve time.
- `encoder-lab-before.exe`, `encoder-lab-v1.exe`, `encoder-lab-v2.exe`,
  `encoder-lab-ma-only.exe`: frozen benchmark executables.
- `rejected-weighted-row.patch`, `deferred-ma-collection.patch`: experimental
  source diffs. The MA diff includes test registration that is already retained;
  do not blindly apply that hunk a second time.

The local driver is `out/hotspot-followup-20261111/compare.py`; each JSON records
its arguments. Temporary encoded/decoded images were removed after validation.
After archiving the experimental executable, the known baseline executable was
restored to `target/release/examples/encoder_lab.exe`, so that path no longer
points at an implementation removed from the working tree.
