# Float32 Gradient Context Interoperability

## Failure and Cause

The 2026-11-07 compatibility investigation found a context-selection mismatch
in the Rust encoder and maintained decoder at revision `cba8c772`. They could
round-trip some float32 images together while official libjxl rejected those
streams. Conversely, the maintained Rust decoder could reject an official
float32 stream. Successful self-round-trips were insufficient evidence.

Lossless float32 samples are represented by integer words in the Modular
pipeline. A gradient MA tree tests property 9, `W + N - NW`. For example,
`W = N = 0x40000000` and `NW = 0` give `2147483648` before narrowing. The old
Rust path wrapped this to a negative i32. The libjxl gradient lookup path uses
a wide intermediate and then clamps to `[-8192, 8191]`, selecting the positive
branch instead. Different entropy contexts can cause an ANS checksum error;
prefix-coded streams may instead decode successfully to incorrect samples.
This is not evidence that ANS itself or floating-point rounding is faulty.

Small efforts expose this more often because the float32 e1/e2 routes select
a fixed gradient tree. A small image can collapse that tree to one leaf and
avoid the differing decision. Higher efforts may choose other trees, but a
learned or caller-supplied eligible gradient tree can expose the same issue.
Effort never relaxes the losslessness contract.

## Corrected Behavior

The encoder and decoder match libjxl's specialized gradient-context arithmetic
only when its dispatch conditions hold after static channel/group filtering:

- All dynamic splits test property 9.
- All leaves use the Gradient predictor, offset zero and multiplier one.
- All split thresholds are in `[-8193, 8190]` inclusive. Redundant splits and
  empty intervals are not, by themselves, disqualifying.

Other MA trees retain their existing generic i32 property semantics. Merely
widening every property or changing the predictor would not reproduce the
reference implementation. The decoder's smaller lookup-table optimization is
separate from semantic eligibility.

Reference sources are libjxl `modular/encoding/encoding.h::TreeToLookupTable`,
`modular/encoding/encoding.cc::FilterTree` and the corresponding encode/decode
gradient paths. This change targets official implementation interoperability;
it does not establish an independent ISO conformance interpretation.

## Recovering Old Files

New encodes use the corrected behavior. Decoding defaults to it as well. An old
affected file cannot be repaired merely by changing decoder defaults: recover
its original samples using the explicit compatibility option, then re-encode.

```python
import rs_jxl

pixels = rs_jxl.decode("old.jxl", legacy_gradient=True)
rs_jxl.encode(pixels, output="recovered.jxl", effort=2)
```

Check the current [Python API](PYTHON_API.md) for output and frame controls.
The example above recovers pixels of a still image, not its metadata, extra
planes or animation; preserve those through their corresponding encode APIs.
Verify recovered samples against the originals when available before replacing
any old files.
The equivalent Rust option is `DecodeOptions::legacy_gradient`; the CLI option
is `--legacy-gradient` on the decode command. This is a recovery mode, not an
encoder option or a quality setting. Do not enable it globally for official or
new files. There is no unambiguous marker for these old streams, and no automatic
fallback: even a wrong interpretation can return pixels without an error.

## Regression Coverage

`tests/gradient_interop.rs` contains frozen official and pre-fix Rust fixtures,
so ordinary tests do not require a C++ installation. It checks official input,
explicit legacy recovery, one-shot and chunked input, threaded decoding,
reusable decoder state and low-effort float32 sample preservation.

`tests/float_reference/gradient.rs` provides independent official-decoder checks
for low/high efforts, grayscale/RGB and alpha layouts, generic/gradient dispatch
boundaries, ANS/prefix coding, and channel/group specialization. Set
`RS_JXL_DJXL` to an OS-separated list of official `djxl` paths, then run:

```text
cargo test -p rs-jxl --test gradient_interop
cargo test -p rs-jxl --test float_reference gradient -- --ignored
```

Comparisons are against original float32 sample words, not newly encoded JXL
bytes. Frozen compressed fixtures exercise compatibility, not a bitstream-size
or bitstream-hash optimization target.

## Local Verification, 2026-11-07

The supplied archive was checksum-verified before replay. On Windows, the
rebuilt release DLL was checked against independent local C++ decoders:
libjxl 0.12.0 (`a1ebdf2`) and a 0.13.0 development build (`1e57a79`). The latter
is a local snapshot, not a claim about all 0.13 revisions or upstream HEAD.

- All 75 distinct retained Rust streams recovered their original float32 words
  with explicit legacy mode. Corrected default mode accepted the 40 compatible
  streams and rejected the 35 incompatible ones.
- All 38 distinct source arrays were re-encoded at e1, e2 and e9: 114 outputs
  matched original words with the Rust decoder and both C++ decoders.
- Six image-size/value boundary controls passed all three decoders.
- Six official-encoded files, e1/e2/e3 from both C++ versions, decoded exactly
  with default Rust options.
- The complete Rust workspace run passed 2,526 tests with 113 ignored and zero
  failures. The four external-decoder gradient tests were separately invoked
  and passed, including negative words/signed zero and parallel tokenization.
- Python codec/API tests passed 171 tests with nine skips. Packaging tests were
  not collected because this Python environment lacks `tomlkit`; packaging code
  was not changed. These are source-package tests against the rebuilt release
  DLL, not a newly installed-wheel or CI result.
- An additional real-interface replay passed 29 Python sample checks, including
  paths, bytes, memoryviews, streams, `decode_into`, raw layers and reusable
  contexts. All six CLI normal/incremental decode checks passed.
- Release DLL/CLI builds, workspace all-target checks, strict all-target Clippy
  (`-D warnings`), workspace formatting and whitespace checks passed.

These results concern this reported arithmetic failure. They are not a full
float-domain, cross-platform or performance certification. The frozen prefix
fixture separately demonstrates why an error-only legacy retry is unsafe.
