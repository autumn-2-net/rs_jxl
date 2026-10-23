# Official Rust Decoder Backend Audit

This document describes the unmodified upstream checkpoint before integration.
The maintained copy and current implementation status are tracked separately in
[DECODER_STATUS.md](DECODER_STATUS.md).

Date: 2026-10-23. This is a backend assessment, not a production integration.

Audited upstream: [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs), commit
`5ccf8694e3779e2931ca487d46251e3050bf849b`, package version 0.8.0.
Encoder under test: this repository at `04f12a1`.
No production dependency, encoder implementation or public API was changed.

## Conclusion

The upstream decoder is useful for ordinary pixel output and coalesced
animation, but does not cover this project's complete archival contract as-is.
Several gaps are public API limitations rather than missing entropy or Modular
algorithms. A maintained fork can reuse the existing pure-Rust decoder; a thin
wrapper alone cannot provide all missing operations. JPEG byte reconstruction
is a larger missing path, not just an output-format option.

Do not substitute a C++ runtime or a subprocess decoder without explicitly
changing the project's implementation contract. Official C++ djxl remains an
independent development oracle.

## Capability Findings

| Requirement | Finding at audited commit |
| --- | --- |
| U8/U16 Gray/GA/RGB/RGBA, invisible RGB | Focused interoperability checks passed. |
| Visible animation and references | Implemented; timed-frame crop compositing passed the local probe. |
| Visible-frame seeking | Public scan records and dependency-aware seek targets exist. Not an arbitrary raw-layer reader. |
| Raw layers / coalescing disabled | Not implemented: the coalescing field is declared/defaulted but never read. Zero-duration intermediate frames do not produce output events. |
| Frame metadata | Public headers omit crop origin, raw dimensions, blend/reference controls and timecode. Internal raw-header getter is test-only/private. |
| Mixed extra-channel pixel output | Public per-channel output formats work; mixed U16 depth and U8 mask passed. |
| Complete extra-channel metadata | Public JxlExtraChannel exposes type and alpha association only; not depth/exponent, name, dim_shift, spot values or CFA index. |
| Native subsampled extra planes | Output passes through upsampling; no public native-resolution sample export. |
| Arbitrary U31 exact output | No U32 output format; adjacent large integer samples collide in F32 output. |
| F16 raw-word preservation | Failed for signaling NaNs on tested x86 SIMD path, even with high_precision=true. |
| F32 raw-word preservation | The tested 16-word fixture passed; this is not all-pattern or reference/blending acceptance. |
| ICC and Exif/XMP/other boxes | Embedded/output profiles and requested auxiliary-box payloads are available. Brotli expansion is feature-gated. |
| Original JPEG reconstruction | jbrd extraction exists, but no complete JPEG reconstruction API or publicly exported quantized coefficient result. |
| Resource limits | sample_limit is per image/frame, not a total memory budget; auxiliary-box collection/decompression lacks a configurable byte ceiling. |

Source anchors at the frozen upstream commit:

- [Pixel and metadata types](https://github.com/libjxl/jxl-rs/blob/5ccf8694e3779e2931ca487d46251e3050bf849b/jxl/src/api/data_types.rs):
  output formats at line 102, extra metadata at line 302 and frame metadata at 316.
- [Decoder API](https://github.com/libjxl/jxl-rs/blob/5ccf8694e3779e2931ca487d46251e3050bf849b/jxl/src/api/decoder.rs):
  private raw header at line 99, canvas-sized visible output at 180 and seek API at 218.
- [Options](https://github.com/libjxl/jxl-rs/blob/5ccf8694e3779e2931ca487d46251e3050bf849b/jxl/src/api/options.rs):
  coalescing at line 23; repository-wide search finds only its declaration/default.
- [Frame rendering](https://github.com/libjxl/jxl-rs/blob/5ccf8694e3779e2931ca487d46251e3050bf849b/jxl/src/frame/render.rs):
  extra upsampling at line 639, reference saves/blending at 718 onward and
  visible-only API output at 800.
- [Sample conversion](https://github.com/libjxl/jxl-rs/blob/5ccf8694e3779e2931ca487d46251e3050bf849b/jxl/src/render/stages/convert.rs):
  half conversion uses load_f16_bits; wide integer conversion uses F32.
- [Box API](https://github.com/libjxl/jxl-rs/blob/5ccf8694e3779e2931ca487d46251e3050bf849b/jxl/src/api/box_parser.rs):
  JPEG_RECONSTRUCTION_DATA is an auxiliary payload type, not a JPEG writer.

## Executed Probe

Windows x86_64, Rust 1.97.1, release validation profile with LTO off and
16 codegen units. The isolated upstream dependency enabled sse42, avx and
brotli, not default all-simd. No ARM-specific code or testing was added.
Decoder settings: Main10 enabled, high_precision=true, orientation adjustment
and spot rendering disabled, alpha association unchanged, explicit output
formats and a four-million-sample admission limit. Output buffers use the
upstream aligned OwnedRawImage allocator. All fixtures are generated by this
repository's actual encoder, without a C++ encoder subprocess.

| Probe | Result |
| --- | --- |
| Gray/GA/RGB/RGBA at U8 and U16, 65x17 | All eight fixtures matched original sample words. |
| All 65,536 U16 words, Gray 256x256 | Zero differences. |
| All 65,536 F16 words, F16 output | 1,022 differences: signaling NaNs were quieted. |
| Same F16 codestream, F32 output | Same 1,022 differences against exact integer-defined binary16-to-binary32 widening. |
| Same F16 codestream, two official C++ djxl decoders | Each produced zero differences across all 65,536 exact widened F32 words. |
| 16 selected F32 words | Zero differences, including signed zero, infinities and selected NaN payloads. |
| U31, eight selected integers | 16,777,216 and 16,777,217 both output F32 word 0x3c000000; larger adjacent pairs also collide. |
| Gray U8 plus named U16 depth and U8 selection mask | All three output planes matched. Public metadata is still incomplete. |
| Two frames, durations 0/0 | One final composite, for both coalescing=true and false. |
| Same frame geometry, durations 1/1 | Two composed outputs, for both coalescing settings. |

The F16 failure is a raw-word archival-contract failure; it must not be
generalized to a claim that ordinary 8/16-bit integer pixel decoding is lossy.
For example 0x7c01 becomes 0x7e01. Requesting F32 cannot recover the lost
signaling bit because the conversion already quieted it. The C++ comparison
establishes that these bits were preserved in the encoder's JXL output.

Local isolated reproduction artifacts, relative to the encoder repository:

- `../jxl_python/out/decoder_audit/jxl-rs`: clean upstream checkout.
- `../jxl_python/out/decoder_audit/probe`: standalone Cargo probe, outside this workspace.
- `../jxl_python/out/decoder_audit/check_official.py`: bounded official-decoder
  cross-check using integer word comparisons, not floating arithmetic on NaNs.
- `../jxl_python/out/decoder_audit/fixtures`: generated tiny JXL/NPY fixtures.

These are local audit artifacts, not packaged dependencies or committed images.
The probe prints discrepancies as audit results; a successful process exit is
not a claim that every strict lossless contract passed.

## Maintenance Boundary

1. Start from a pinned pure-Rust upstream decoder, keeping its provenance and
   independent release cadence. Do not present a wrapper as a full decoder port.
2. Add full frame/extra metadata and raw-layer events/output, keeping composed
   visible-frame playback as a separate API. Preserve raw duration ticks.
3. Add exact typed integer/raw-float export before presentation conversions.
   Wide integer reference storage/compositing requires its own design; adding
   a U32 enum variant alone does not fix F32 reference storage.
4. Add byte-reversible JPEG reconstruction, reusing the existing Rust JPEG
   scan writer and structured reconstruction checks where applicable. It still
   requires decoding jbrd fields and recovering original quantized coefficients
   from the JXL coefficient representation; RGB output is insufficient.
   In upstream, AC integers use recyclable scratch, LF/DC integers are
   dequantized before release, and quantization tables become floating dequant
   matrices. A reconstruction mode must capture these before conversion/reuse;
   adding a getter after pixel decoding is not sufficient. Ordinary auxiliary
   box Brotli handling does not decode the separate internal jbrd suffix.
   Our existing writer also needs restored marker records/table snapshots,
   metadata sources and reconstruction hints, not only coefficient arrays.
5. Add cumulative reference/metadata/output budgets and bounded box expansion.
   Keep allocation errors distinct from unsupported format errors.
6. Run encoder/decoder cross-checks independently. Passing self-roundtrips
   cannot replace official decoder acceptance or prove compression-rate parity.

Upstream [issue 1053](https://github.com/libjxl/jxl-rs/issues/1053) separately
tracks reference-buffer accumulation beyond sample_limit, and documents that
scan_frames_only still decodes invisible frames. These observations were also
confirmed in the frozen code; this audit did not run hostile memory workloads.

No decoder backend is selected or added to the production manifests by this
audit. The encoder's bottom-level tool coverage does not establish universal
compression-ratio parity; that still needs mixed-corpus comparisons.
