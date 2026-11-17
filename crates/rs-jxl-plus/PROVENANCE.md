# PJA translation provenance

The coefficient backend is translated from
[PJA commit 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd](https://github.com/YadeWira/PJA/tree/1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd),
whose source identifies version 0.9.13. PJA's MIT license names
Copyright (c) 2026 Yade Bravo (YadeWira). Preserve
[LICENSE-PJA](LICENSE-PJA) and [NOTICE-PJA](NOTICE-PJA) with translations.
The notice is retained in full. The user has authorized the complete PJA
functionality port before pruning. Translation, integration and independent
verification are distinct checkpoints; see the module inventory in
[PJA_PORT_STATUS.md](../../docs/PJA_PORT_STATUS.md) for actual status.

| Rust module / artifact | Pinned PJA sources and scope |
| --- | --- |
| `src/arith.rs` | `pas/pjaarith.pas`: carryless arithmetic encoder/decoder and adaptive-bit update |
| `src/probability.rs` | `pas/pjav7c_loop.h`: one, avg2 and scalar mixn; `pas/pjamodel.pas`: probability initialization |
| `src/probability_tables.rs` | Frozen squash/stretch values from `pas/pjamodel.pas` InitTables definitions; checked independently with Decimal arithmetic |
| `src/predict.rs` | `pas/pjamodel.pas`: BASIS and InitUnzig; `pas/pjav7c.c`: numerical helpers and neighbor magnitude; `pas/pjav7c_loop.h`: kernel and boundary-estimate operations |
| `src/model.rs` | `pas/pjav7c_loop.h`: complete v7 component loop; `pas/pjamodel.pas`: V7Component, CodeBand and BandCount |
| `src/model/state.rs` | `pas/pjamodel.pas`: InitV7 counter dimensions and V7Weights |
| `src/arith/live.rs` | `pas/pjaarith.pas`: retained live byte blocks, concurrent arithmetic readers/writers, terminal states |
| `src/mix_simd.rs`, `src/model/parallel.rs` | Source mixer arithmetic and `CodeBand`/`BandCount`; safe disjoint Rust scheduling and x86_64 `std::arch` kernels |
| `src/jpeg_native.rs`, `src/jpeg_native/` | `pas/pjajpeg.pas`: Huffman/QM DCT and lossless JPEG parsing, scans, quirks and reconstruction; IJG attribution retained in NOTICE-PJA |
| `src/jls.rs`, `src/jls/` | `pas/pjajls.pas`: JPEG-LS entropy/parser/reconstruction; CharLS BSD attribution retained |
| `src/sample_model.rs`, `src/sample_model/` | Sample MED/blend and JPEG-LS symbol models from `pas/pjamodel.pas` |
| `src/raw_model.rs`, `src/legacy_model.rs`, `src/legacy_model/` | Raw model and legacy full/fast coefficient loop from `pas/pjamodel.pas`, `pas/pjacoef.inc` |
| `src/j2k/mod.rs`, `src/j2k/codestream.rs`, `src/j2k/codestream/` | `pas/pjaj2k.pas`, `pas/pjaj2kcs.pas`: boxes, packets, geometry, side information, segment rules and reconstruction |
| `src/j2k/tier1.rs`, `src/j2k/ht.rs`, `src/j2k/tables/` | `pas/pjaj2kt1.c/h`, `pas/pjaj2kht.c`, associated frozen tables; OpenJPH BSD-2 attribution retained |
| `src/j2k/model.rs`, `src/j2k/ht_model.rs`, `src/j2k/modes.rs` | `pas/pjaj2kmc.c`, `pas/pjaj2khm.c`, `pas/pjaj2km.pas` |
| `src/j2k/{pixels,pixel_tier1,pixel_model,transform}.rs` | `pas/pjaj2kpx.pas`, `pas/pjaj2kpm.c`: lossless coefficient/pixel conversion, wavelets/RCT, pixel models |
| `src/codec.rs`, `src/codec/`, `src/j2k/pipeline.rs` | `pas/pjacodec.pas` coefficient/sample model selection and reconstruction, reorganized into typed internal records; old archive, plugin, recursive parts, repair/patch and whole-file raw protocols removed |
| `src/model/static_model.rs`, `src/static_tests.rs` | New test-only two-pass static-probability experiment over the translated v7 decisions; no upstream PJA compatibility claim |
| `src/tool_tests.rs` | New test-only composition of existing `rs-jxl/src/jpeg/{cfl,transcode_layout}.rs` APIs with v7; CfL source is libjxl `enc_frame.cc`, `chroma_from_luma.{h,cc}`, `quantizer.h`, `dec_group.cc` at `b866b3defec13494bfbaf705db93581cda118e37`, retaining the existing BSD source ledger |
| `src/ma_tool_tests.rs` | New test-only composition using existing `rs-jxl/src/modular/{ma_train,tree,predictor,weighted}.rs` APIs; their libjxl `enc_ma`, `dec_ma`, `context_predict` and weighted-prediction source ledger/attribution remains in `rs-jxl` and `docs/PORTING_KERNELS.md`; the bounded experiment descriptor is not upstream JXL syntax |
| `src/model/{ma_context,ma_context_tree}.rs`, `src/dynamic_ma_tests.rs` | New test-only conditional-logloss context-tree learner and online probability calibration over unchanged v7 coefficient decisions/counters/mixers; uses the existing PJA frozen squash table and `rs-jxl` MA tree validation/context numbering. Causal event notifications in `arith.rs`/`model/state.rs` are test-only; the learner and descriptor are not upstream PJA/JXL syntax |
| `src/model/band_tools.rs`, `src/model/context_refine.rs`, `src/search_tools_tests.rs` | New test-only band/model search, hierarchical interpolated probability refinement and real CfL compositions. Reuses the attributed PJA v7/Full/Fast ports and existing `rs-jxl` CfL implementation; profile descriptors, selection and probability-refinement rules are experimental new work, not upstream wire syntax |
| `src/{corpus_tools_tests,speed_tools_tests,api_speed_tests,standard_corpus_tests,fast_parallel_speed_tests,parallel_corpus_tests}.rs`, `tools/build_jxl_plus_corpus.py` | New opt-in fixed-corpus evaluation, exact pre/post stream comparison and isolated stage/public-API timing harnesses; exercise the existing attributed codecs without importing a different implementation |
| `tools/pja_v7_reference.c`, `src/oracle_tests.rs` | Development-only wrapper/descriptor around original `pas/pjav7c.c` and `pas/pjav7c_loop.h`; not linked into runtime |
| `src/{pixel_model,pixel_tests,pixel_compose_tests}.rs`, `examples/pixel_lab.rs`, `tools/{build,analyze}_jxl_plus_pixel_corpus.py`, `tools/analyze_jxl_plus_pixel_composition.py` | New internal pixel experiment and fixed-corpus harness; combines existing JXL RCT/MA/WP/HybridUint/ANS/LZ with the attributed PJA adaptive binary counter/arithmetic coder. Its complete experimental framing is not final JXL extension syntax. The focused laboratory target shares the same source files; actual group selection charges complete bytes and all encoder search time. |
| `../rs-jxl/src/modular/plus_bridge.rs` | Restricted preparation/causal inverse bridge based on `enc_modular.cc`, `context_predict.h`, `rct.cc::InvRCTRow` at pinned libjxl `b866b3d`; retains BSD attribution and uses existing native Rust prediction/tree code. |
| `../rs-jxl/src/entropy/experimental_decode.rs` | Adapter to the maintained vendored jxl-rs entropy/tree reader; upstream attribution and local changes are recorded in `vendor/jxl-rs/PROVENANCE.md`. |
| `tools/pja_j2k_reference.c`, `tools/pja_pixel_reference.c` | Development-only original-C checks for tier-1/HT, their models, pixel decisions/models and transforms; never linked into runtime |

The source research copy is under
`D:/cpp_pj/jxl_python/out/pja-research-20261117/1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd/pas/`.
The development-oracle copy is under
`D:/cpp_pj/rs_jxl/out/pja-port-20261117/source/`; copied original files were
checked against the pinned Git source. These are local development artifacts,
not a runtime dependency or a source of native linkage.

Translation choices retain source decisions while changing implementation:

- Rust ownership and bounded stack copies replace aliasing counter pointers;
  model updates still occur after each bit.
- Quantized coefficients enter/leave in natural frequency order; the model
  uses PJA's zigzag order internally.
- Explicit wrapping handles modulo-2^32 sums; signed division truncates
  toward zero. Prediction division uses exact integer arithmetic instead of
  the source reciprocal approximation plus correction.
- Frozen probability tables avoid platform-dependent runtime exp/round.
- Prediction and mixer kernels use runtime-detected x86_64 AVX2 with a scalar
  fallback. This is not a claim of porting every source SIMD specialization.
- Source band partitioning, independent band models and shared chroma state
  are retained. Explicit internal v7 parallel entry points isolate band
  storage and enforce a shared output budget; the public API stays serial.

`src/container.rs` now defines an experimental **jxl_plus v1** container.
The user requested removal of content hashes: original and container digests
were removed, and version 0 is explicitly rejected. Declared sizes, budgets
and canonical arithmetic termination are checked; successful encoding still
decodes and compares every original JPEG byte. This is not the final planned
JXL-derived frame syntax and does not implement PJA's archive syntax. Reconstruction
metadata, JPEG parsing and byte writing come from the one-way `rs-jxl-plus ->
rs-jxl` dependency, including `rs-jxl/src/jpeg/{reader,coefficient_reader,
reconstruction_metadata,reconstruction_brotli,reconstruction_decode,
scan_writer}.rs`. That libjxl-derived code retains its existing BSD attribution
and source ledger; the PJA MIT grant does not relicense it.

No overall license for unrelated new repository code is selected by these
third-party notices. Utility dependencies retain their own licenses. See
[the format/status document](../../docs/JXL_PLUS.md) for supported operations,
syntax and the limited verification checkpoint.

The public API is still the Huffman-DCT experiment. The expanded
JPEG/JPEG-LS/JPEG 2000 port is integrated internally and checked separately.
Per the latest user instruction, public PJA archive compatibility is not a
goal. The eventual format will use original JXL codestreams plus extensions;
the development representations here do not define that final syntax.

J2K mode/pixel internal layout 1 and the owned composition records also omit
hashes. The old PJA archive helpers, their protocol-specific tests and the
BLAKE3 dependency have been removed. The unabridged pinned upstream notice
remains an attribution record; it is not a list of current dependencies.
Context-table hashing is statistical indexing, not integrity
checking, and is retained. Source-file identity checks remain development
provenance checks, separate from any encoded file's syntax.
