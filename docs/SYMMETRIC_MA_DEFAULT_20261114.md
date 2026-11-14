# Symmetric MA With Half and Single-Pass Offsets

## Objective

Evaluate the existing near-zero interval learner against the current source-e9
default, not the older unpenalized MA baseline. The acceptance objective is
aggregate bytes and native encode time, with exact decoded samples. Individual
files may grow; encoded-byte identity is not an acceptance requirement.

Ordinary source-policy e9 now enables Symmetric together with Half and
single-pass Distribution Offset. Other efforts and JPEG transcoding are
unchanged. `source_ma_defaults=false` opts out of automatic settings while
preserving explicitly supplied tools. Explicit alternative learners/tree plans
are not replaced; nondefault entropy estimators keep their compatible learner.

Symmetric is not an oblivious/global balanced tree. It evaluates two near-zero
cuts together, then emits two ordinary JXL binary nodes with three child
regions. It includes the PR5007 split-price model. A separate Cost profile
isolates that model from the interval search.

## Changes

- Allow description-only scoring with Cost and Symmetric. Include description
  costs in the middle region and the parent density gate, as already done for
  the outer children and unsplit leaf. Alternative entropy estimators remain
  rejected rather than silently ignored.
- Reject structurally impossible intervals before preparing histogram scratch;
  retain the original distinct-record requirements and threshold semantics.
- Use the existing event and dominance paths for properties with no eligible
  interval. They remain disabled where a middle-region winner could be missed.
- Borrow a single occupied middle histogram row directly instead of clearing
  and accumulating a separate histogram.
- Score identical middle populations once even if two syntax boundaries refer
  to them. Keep their distinct split-signaling costs.

No decoder, bitstream syntax, JPEG coefficient path, or second full encode is
introduced. Offset fitting remains single-pass with no actual-size fallback.

## Reproduction

Build the Release example:

```text
cargo build --release -p rs-jxl-cli --example portfolio_lab
```

Direct combined encoding (Windows executable suffix shown):

```text
target/release/examples/portfolio_lab.exe input.png --output output.jxl --effort 9 --threads 16 --direct-candidate baseline --direct-ma-profile half-single-pass --direct-tree-learning symmetric
```

The same combination is available through native `ModularOptions.tree_learning`,
`source_ma_scoring.description` and `refinement.offsets`. `encoder_lab` accepts
`--tree-learning symmetric --description-cost half --leaf-offset distribution-single-pass`.

`tools/benchmark_symmetric_default.py` requires an explicit image manifest,
encoder and independent decoder. It freezes the binary, verifies each decoded
image including invisible RGB, and retains every trial in JSONL. Repeated
timings and sizes use per-image medians; bytes, pixels and images are counted
once. Differing compressed sizes are not correctness failures: original-sample
comparison is the correctness gate. All measured repeats here had equal sizes.
Mean bpp is the arithmetic mean across images; weighted bpp is total bits
divided by total pixels. Time excludes CLI startup, input/output and decoding.

```text
python tools/benchmark_symmetric_default.py --manifest MANIFEST.json --encoder target/release/examples/portfolio_lab.exe --decoder DJXL.exe --out OUT/t16 --threads 16 --no-thread-subset
python tools/benchmark_symmetric_default.py --manifest MANIFEST.json --encoder target/release/examples/portfolio_lab.exe --decoder DJXL.exe --out OUT/threads --threads 1 4 16 --repeats 3 --profiles baseline symmetric-half-single-pass cost-half-single-pass
```

The runner uses the explicit preceding profile (`baseline`), the current
automatic default, Symmetric with Offset only, Symmetric with Half and Offset,
and Cost with Half and Offset. In the original frozen measurements below,
`default` in JSONL means the preceding default, before the policy switch.
The thread subset is the same first >=100-kpixel stratified case per dataset
plus the existing Half and Symmetric adverse controls.

## Paired 32-Image Results

Release, Windows x64, 16 threads, effort 9, 32 manifest images. These include
24 stratified originals across six datasets, four 16-bit precision cases and
four previously adverse cases. One timing per image/profile in this pass;
use the repeated subset below for timing sensitivity.

| Profile | Total bytes | Mean bpp | Weighted bpp | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous default: Half + single-pass Offset | 26,010,580 | 5.765125 | 8.947469 | 11.5413 | 2.0150 |
| Cost + Half + single-pass Offset | 26,004,755 | 5.764676 | 8.945466 | 11.3268 | 2.0532 |
| Symmetric + Half + single-pass Offset | 25,972,953 | 5.757505 | 8.934526 | 11.6190 | 2.0016 |
| Symmetric + single-pass Offset, no Half | 26,102,018 | 5.789508 | 8.978924 | 14.3270 | 1.6232 |

The combined profile saves 37,627 bytes (0.14466%) versus the previous default,
with a 0.6736% single-run time increase. There are 21 smaller, seven equal and
four larger files. The increases are 11, 52, 18 and 3 bytes, 84 bytes total.
The largest relative increase is 2.965% on a 371-byte image, not a large-image
regression. The independent 24-image stratum saves 0.14159%, with time +0.7469%.

Symmetric selected 514 interval splits in 17 images, alongside 52,071 binary
Cost splits. All 388 refinement domains built exactly one model. Compared
with Cost alone, the intervals account for another 31,802 bytes saved.

All 128 outputs passed independent official djxl sample verification. Source
file/pixel-payload hashes are provenance checks, not compressed-output goals.
Artifacts are under `out/symmetric-default-20261114/t16` in the sibling
`jxl_python` workspace, with manifest, frozen binary, JSONL and summary.

## Thread Sensitivity

The same eight cases were encoded three times per profile and thread count,
rotating profile order. Times below sum per-image medians. Bytes are identical
across thread counts for each profile: 7,985,311 for the preceding default and
7,970,719 for Symmetric plus Half and Offset (0.18274% smaller). Seven images
shrank, one tied. Mean bpp changes from 4.986352 to 4.971681; weighted bpp from
7.618318 to 7.604396.

| Threads | Previous default seconds | Combined seconds | Time change | Combined MPix/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.6412 | 24.0916 | +1.90% | 0.3481 |
| 4 | 7.4333 | 7.4623 | +0.39% | 1.1237 |
| 16 | 3.8118 | 3.8958 | +2.20% | 2.1524 |

All 216 repeated outputs passed official djxl verification. Artifacts are under
`out/symmetric-default-20261114/threads` in the sibling workspace.

This is a modest compression improvement with small measured overhead, not
a large speedup or a proven whole-corpus result. The measurements isolate the
combined profile against the preceding default; they do not isolate the speed
gain of each internal scratch/fast-path optimization. No full corpus or new
official encoder baseline was run. Symmetric without Half is not promoted.

## Validation

Targeted tests cover description pricing for packed/unpacked, single/multiple
middle rows, repeated equivalent populations, predictor ties and impossible
intervals. Serial/parallel training, default projection, explicit overrides and
FFI option projection pass. Integer roundtrips cover Gray, GrayAlpha, RGB and
RGBA at 8/16 bits, full-frame and streaming groups, and invisible RGB.

The rebuilt Release native library also passes all three
`python/tests/test_source_ma_options.py` tests, including native still,
multiframe and zero-duration-layer roundtrips with automatic defaults on/off.
The old Offset benchmark no longer asserts equal compressed sizes for current
defaults and explicit Half/Offset: these are now distinct learner profiles.
Parameter receipts and independent pixel comparisons remain the checks.

After enabling the automatic policy and rebuilding, an additional eight-image
check compared the explicit preceding baseline, automatic default, and explicit
combined profile. All 24 outputs passed official djxl sample verification,
bringing this experiment to 368 independent decoder checks. The automatic
receipt reports Symmetric/Half/single-pass and exactly one model per domain.
Both combined entry points measured 7,970,719 bytes versus 7,985,311 baseline.
This one-pass acceptance measured 4.0151 s automatic, 3.8765 s explicit, and
3.8582 s baseline; it is not substituted for the three-repeat thread results.
Artifacts are under `out/symmetric-default-20261114/final-default` in the sibling
workspace. Release CLI/native-library builds and CLI help verification pass.
Linux runtime and full-corpus performance were not tested in this change.
