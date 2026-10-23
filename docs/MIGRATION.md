# Direct Source Migration

## Scope and Baseline

- Read-only optimized C++ source: `../jxl_python/libjxl`.
- Frozen source commit: `b866b3defec13494bfbaf705db93581cda118e37`.
- Independent Rust repository: `.`, branch `main`.
- The acceptance boundary is the **complete pixel-lossless encoder**. A legal
  codestream or a few passing examples do not finish this phase.
- JPEG byte reconstruction is a separate acceptance phase, not an alternative
  to finishing pixel encoding. The user has authorized its prerequisite tools
  in parallel; marker/entropy helpers do not imply a complete transcoder.

No C++ build/link/runtime bridge, external encoder subprocess, or third-party
Rust JXL encoder may replace a source port. Rust utility crates are allowed.
The CLI uses `png` and `zune-jpeg` to obtain input samples; `rayon` supplies
owned parallel execution. Official djxl is an independent test oracle only.

## Source Map

Paths below are relative to `crates/rs-jxl/src`.

| Rust Module | Main Source | Active Responsibility |
| --- | --- | --- |
| `bit_writer.rs` | enc_bit_writer | Bounded LSB output, alignment, append |
| `entropy/` | enc_ans, enc_huffman, enc_context_map | HybridUint, prefix/RLE, rANS, normalization, clustering, context maps, source policies |
| `modular/predictor.rs`, `weighted.rs` | context_predict | All predictors, source weighted history and presets |
| `modular/predict_simd.rs`, `cost.rs` | context_predict, enc_modular | Native row kernels, RCT/WP cost analysis |
| `modular/transforms.rs`, `group.rs`, `palette.rs` | enc_rct, enc_palette, enc_palette_delta, enc_squeeze | Reversible transforms, channel bookkeeping, palette attempts and rejection |
| `modular/layout.rs`, `orchestration.rs` | enc_modular, squeeze, frame_header | Global transforms, DC/AC spatial partition, progressive passes |
| `modular/sample.rs`, `training.rs`, `ma_train.rs`, `fixed.rs`, `tree.rs` | enc_encoding, enc_ma | Sampling/quantiles, adaptive learning, constants/multipliers, predefined trees, serialization |
| `tools/lz77*` | enc_lz77, enc_lz77_bucket | Source ring/chain/bucket matching, greedy/optimal parsing and optimized filters |
| `tools/ma_entropy.rs`, `ma_simd.rs` | enc_ma_entropy, enc_ma | Incremental split costs and native SIMD |
| `tools/patch*`, `patch_frame.rs` | enc_patch_dictionary | Discovery, atlas/residual generation, dictionary and reference framing |
| `scheduler/` | enc_lossless_search and probe/scan helpers | Candidate planning, probes, feedback, retained winners and native bridge |
| `image.rs`, `metadata.rs`, `color.rs`, `icc.rs` | image_metadata, color_encoding, enc_icc_codec | Typed samples and explicit metadata without implicit color conversion |
| `modular/sample_convert.rs` | float_to_int / int_to_float | Exact raw-word mapping and checked custom-float conversion |
| `codestream.rs`, `encode.rs`, `frame.rs`, `multiframe.rs` | frame_header, encode, enc_modular | Container/TOC, layers/animation, references and all-channel crop planning |
| `jpeg/`, `modular/auxiliary.rs` | jpeg_data, enc_jpeg_data_reader, enc_jpeg_huffman_decode, enc_frame, enc_modular | Coefficient parsing/reconstruction, complete shared-model archival frames and byte/writer/CLI adapters; optimized search and metadata deduplication still pending |

An implementation can be active while a particular source optimization or
combination remains unverified. See [coverage status](PIXEL_LOSSLESS_PORT_STATUS.md)
and [scheduler status](../crates/rs-jxl/src/scheduler/PORT_STATUS.md) for actual gaps.
Experimental source-policy differences must be reported, never hidden under an
exact-candidate identity. Resource exhaustion is an error, not silent sample
truncation or replacement with a fixed tree.

## Acceptance Order

1. Complete pixel-lossless tools, optimized paths, scheduling, supported sample
   formats and frame/metadata contracts. Test independent official decoding and
   measure bytes, bpp, time and throughput on mixed real data.
2. Port JPEG coefficient parsing, entropy/progressive scans, reconstruction
   metadata and coefficient encoding. Verify **original JPEG bytes** through
   official reconstruction. Pixel decoding is not a fallback for this track.
3. Stabilize the Rust interfaces and versioned C ABI for Python CFFI/wheels.
   Add the runtime decoder separately; keep dependency/implementation choices
   explicit rather than claiming a subprocess oracle is a Rust decoder.

Independent pixel and JPEG-prerequisite work may run concurrently. Do not mark
either phase complete just because an individual primitive works.

## Validation Contract

- Pixel lossless: compare decoded sample values, including invisible RGB.
- Float lossless: compare bit patterns where the output API preserves them.
- High-depth integers: official NPY stores f32 and cannot prove arbitrary U31
  recovery. Tests must state projection-only checks, not label them full precision.
- JPEG transcoding: compare reconstructed source file bytes.
- Compressed JXL bytes may change. No whole-file hash equality with a previous
  encoder output is an optimization or migration objective.
- Primitive bit vectors, token order and SIMD/scalar parity remain valid tests
  of individual algorithm or wire contracts.

Current local environment is Rust 1.97.1, x86_64 Windows MSVC. No Linux/macOS/ARM
execution or declared-MSRV certification is implied by Windows results. SIMD
must retain scalar fallback and runtime feature checks, not require a global
`target-cpu=native`.

Central commands:

```powershell
cargo test --locked --release --workspace
$env:RS_JXL_DJXL = '..\jxl_python\out\build\windows-clang-cl-cli\libjxl\tools\djxl.exe;..\jxl_python\out\encoder_opt_baseline\djxl.exe'
cargo test --locked --release -p rs-jxl --test reference -- --ignored --nocapture
cargo test --locked --release -p rs-jxl --test float_reference -- --ignored --nocapture
```

Ignored oracle tests are not passed tests until explicitly run. Re-run affected
combinations after changing layouts, entropy wiring or sample mappings.
