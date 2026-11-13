# Source Patch Detector Computation

## Outcome

Now retained as the normal detector implementation after the follow-up stage
analysis and user approval. This does not force Patch admission or enable an
expensive encoding policy. The initial focused experiment did not establish
a whole-encode speed benefit. Across eight real images, native encoding
time changed by -0.11%, while the separately profiled Patch analysis stage
increased by 5.01%. Every per-image compressed size was unchanged.

The initial experiment restored the baseline; the subsequent adoption restores
the integer-seed and deferred-Delta implementation with its nine regression tests.
The historical replayable implementation is retained in
[`experiments/source_patch_detector_20261113.patch`](experiments/source_patch_detector_20261113.patch).
The benchmark harness remains available. No default encoding policy or public
API has changed.

Follow-up: [50-image full-pipeline stage analysis](SOURCE_PATCH_STAGE_PROFILE_20261113.md)
adds opt-in diagnostics and shows that both targeted inner phases do get faster.
Their small contribution and other costs hide that gain in whole-encode time;
the detector optimization is retained, without claiming a whole-encode speedup.

## Scope

Baseline: `1029f1ad9a6fc7e879a20978c74fba66afd28b89`. Two native Rust
computation changes in `tools/patch_source.rs`, without new JXL syntax, a new
predictor, altered MA/LZ77 search, or changes to JPEG recompression. The existing
Source Add Patch admission and encoding policies are retained.

1. Flat 4x4 seeds and neighboring seed origins use raw integer comparisons for
   validated unsigned 1..16-bit color. The allowed absolute differences are 0
   through 13 bits, then 1, 3 and 6 for 14/15/16 bits. These are checked against
   the original reciprocal-multiply f32 normalization and f64 threshold test
   for every valid sample value and both sides of each threshold. Gray needs
   one comparison; RGB still checks every color channel. Floating color keeps
   the old predicate, and extra channels never participate in this color-only
   discovery step. Integer/float, gray/RGB and exact/tolerant dispatch occurs
   once outside the seed scan, using const-specialized loops. Seed origin
   samples are loaded once per cell, not once per neighboring comparison.
2. Quantized fingerprints are still computed for every eligible component, but
   exact Delta arrays are generated only after grouping, occurrence filtering
   and area admission. Each retained group produces one array using its final
   lexicographic `(x,y)` representative and that occurrence's own background
   origin. Replacement representatives update metadata, not the position Vec's
   front. Positions are sorted once at the end. Single-channel reads avoid
   constructing an RGB array during each channel's fingerprint scan.

The second change is stronger than delaying Delta until map lookup: singleton
groups rejected by `min_occurrences` also never allocate Delta arrays. The input
is immutable during discovery, so delayed extraction retains the same operands.
Integer deltas remain signed exact differences; floating deltas retain the
original normalized subtraction and bit representation.

Weighted `similar()` arithmetic is deliberately unchanged. No approximate
fixed-point threshold or reassociation is introduced. The BTreeMap and final
fingerprint/area ordering also remain unchanged. Occurrence-specific differences
still belong in the residual; quantized fingerprint equality is not exact pixel
equality. Alpha and hidden RGB retain the existing lossless replay checks.

## Validation

- 35 focused native Source Patch tests passed, including nine new helper and
  frozen-detector comparisons. Test-only legacy code is not linked into release
  encoders. A work counter checks one Delta buffer per retained group, including
  zero buffers when all groups are rejected.
- Ten benchmark-harness tests passed. Profiling runs are separate from normal
  timing, input/binary identities are recorded, failures invalidate the report,
  and encoded temporary files are removed after official pixel verification.
- The explicit signed-atlas/residual integration test passed against official
  djxl 0.12.0 (`a1ebdf2`) and 0.13.0 (`1e57a79`): 96 main-image and 96
  extra-channel checks, with actual patch activation required by the fixtures.
- Acceptance is official decoding to original samples, including invisible
  RGB. Comparing detector keys and representatives is a local computation
  regression test, not a requirement that compressed JXL bytes match a hash.

## Reproduction

Build and preserve `patch_lab` at the baseline and at the refined candidate.
Use the same Release settings for every executable:

```powershell
cargo build --release -p rs-jxl-cli --example patch_lab -j 4
```

The archived patch contains the final, hoisted-dispatch candidate, not the first
per-comparison-dispatch prototype. Apply it in a disposable worktree based on
the baseline commit (use the archive's absolute path from that worktree):

```powershell
git apply --check path/to/source_patch_detector_20261113.patch
git apply path/to/source_patch_detector_20261113.patch
cargo test -p rs-jxl --lib tools::patches::source -j 4
cargo build --release -p rs-jxl-cli --example patch_lab -j 4
```

Both reverse applicability before restoring production and forward
applicability after restoration were checked. Preserve the executables under
different paths before running the comparison.

The development harness accepts multiple prebuilt binaries and a manifest whose
`cases` array contains `original` PNG paths. It never builds an encoder:

```powershell
python -B tools/benchmark_source_patch_detector.py --encoder baseline=out/source-patch-detector-20261113/bin/baseline.exe --encoder combined-hoisted=out/source-patch-detector-20261113/bin/combined-hoisted.exe --decoder path/to/official/djxl.exe --manifest cases.json --effort 9 --threads 8 --repeats 5 --profile --profile-repeats 3 --out out/source-patch-detector-20261113/results.json
```

All versions use `--mode source`, with no block learner or parallel-gather policy
changes. Versions are interleaved across repetitions. Timing is the native
encode call, excluding startup, PNG I/O and independent decoding. Corpus time is
the sum of per-image medians; both mean image bpp and pixel-weighted bpp are
reported. `--profile` adds independent diagnostic runs per case/version and
does not contaminate those timing summaries. `--profile-repeats` controls their
count (default 1). Diagnostic inclusive stage totals overlap and must not be
added together.

## Initial Comparison

The first integer implementation dispatched through `DetectionView` on every
comparison. On eight real inputs (11,056,470 pixels), e9, eight threads and
three interleaved repetitions, it did not establish a consistent speed win:

| Initial version | Total bytes | Mean bpp | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 3,943,468 | 3.173975 | 13.199862 | 0.837620 |
| Integer comparison only | 3,943,468 | 3.173975 | 12.996339 | 0.850737 |
| Integer comparison + deferred Delta | 3,943,468 | 3.173975 | 13.242505 | 0.834923 |

All 96 timed/diagnostic outputs passed official pixel checks. Single diagnostic
runs actually showed a larger aggregate Patch analysis time for the candidates;
the small whole-encode difference is not evidence of a reliable speedup. This
motivated hoisting type/channel/threshold dispatch out of the seed loops rather
than retaining per-pixel iterator overhead. Initial receipts remain in
`out/source-patch-detector-20261113/focused8.json`; initial binaries are retained
separately from the refined candidate.

## Refined Comparison

The second implementation specializes the seed scan once per image. It retains
the deferred Delta construction and direct single-channel fingerprint reads.
The same eight inputs were tested at e9 with eight threads: five interleaved
timing runs and three independent profiling runs per version and image.

| Version | Total bytes | Mean image bpp | Native seconds | MPix/s | Patch analysis seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3,943,468 | 3.173975 | 12.747871 | 0.867319 | 0.728922 |
| Refined combined | 3,943,468 | 3.173975 | 12.733427 | 0.868303 | 0.765469 |

Time columns sum per-image medians. Patch analysis is one inclusive stage from
the independent profiling runs, not an additional cost to add to native time.
Pixel-weighted bpp is 2.853329 for both versions. The ordinary mean bpp includes
the 16-bit grayscale texture without converting it to an 8-bit equivalent.

| Image | Format | Bytes | Baseline encode s | Candidate encode s | Baseline analysis s | Candidate analysis s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Paper9bde | RGBA8 | 283,434 | 3.614200 | 3.780507 | 0.271967 | 0.284549 |
| Text7dba | RGBA8 | 9,519 | 0.312908 | 0.296297 | 0.057588 | 0.055712 |
| Editor3dd | RGBA8 | 106,886 | 1.535887 | 1.301300 | 0.112798 | 0.110032 |
| image.png | RGBA8 | 347,113 | 3.658548 | 3.578871 | 0.115010 | 0.108591 |
| wallhaven-49yekn | RGBA8 | 707,442 | 0.987713 | 1.036028 | 0.080094 | 0.094533 |
| wallhaven-n6qzmx | RGBA8 | 459,153 | 2.131013 | 2.237756 | 0.080370 | 0.102057 |
| circles_corner_tr_0016 | RGB16 | 40,563 | 0.219311 | 0.220354 | 0.006622 | 0.006001 |
| rough_linen_Rough_1k | Gray16 | 1,989,358 | 0.288291 | 0.282312 | 0.004473 | 0.003994 |

All 128 real-image timed/diagnostic outputs passed original-sample checks with
official djxl 0.12. An additional 15 synthetic fixtures, each encoded by both
versions, passed 30 checks with official djxl 0.13. These include 8/16-bit
gray, gray-alpha, RGB and RGBA, hidden RGB, noise, shifted repeated tiles and
multigroup images. None failed or timed out. The native and explicit
extra-channel integration tests listed above were run on this refined version.

Individual whole-encode timing ranges overlap substantially. For example,
Editor3dd ranged from 1.301 to 1.554 s in the baseline and 1.267 to 1.547 s in
the candidate; its Patch analysis changed by only about 2.5%. The larger
whole-encode difference cannot confidently be attributed to these changes.
Likewise, the analysis slowdown is an observed result, not proof of a particular
compiler or memory-system cause. Removing work does not establish that it was
on the critical path.

Conclusion: these two computation changes are correct on the tested cases, but
are not justified as a default speed optimization. No container replacement or
approximate `similar()` filter was added to chase this result. A future attempt
should first separate seed scan, background propagation, fingerprint work and
Delta extraction costs before adding more specializations.

## Local Receipts

Ignored development artifacts are under `out/source-patch-detector-20261113/`;
they are not packaged or committed:

- `cases.json`: the eight original input paths; each report records their hashes.
- `focused8.json`: first three-version comparison.
- `synthetic.json`: first candidate synthetic checks.
- `refined8.json`: final five timing / three profiling comparison.
- `refined-synthetic.json`: final synthetic original-pixel checks.
- `bin/`: separately preserved baseline and candidate executables.

Frozen executable SHA256 identifiers are provenance, not acceptance targets:

| Binary | SHA256 |
| --- | --- |
| Baseline | `cc3f8a82c88fa793e2c8d2601f0a66de92347dd7473f7456151de7a2ba11f60c` |
| Refined combined | `97c24aedc0713eef01a6e39da0b3166d6faa00b6da85925ae0e8d6560726f420` |
