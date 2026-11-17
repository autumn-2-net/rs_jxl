# Direct Rust Port Rules

- The user additionally authorized an independent experimental `jxl_plus` format
  (extension `.jxl_plus`) in `crates/rs-jxl-plus`, depending one-way on `rs-jxl`.
  First port the PJA JPEG DCT v7 model at
  `1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd` to native Rust, then improve it.
  Standard-JXL conformance rules below continue to apply to the standard backend;
  the new backend has its own syntax and decoder. Preserve PJA MIT attribution.
  The user now requests the complete PJA functionality port before pruning:
  include arithmetic-coded and lossless JPEG, JPEG-LS, JPEG 2000 and all original
  algorithm/model paths. Do not preemptively
  replace these by a reduced subset. Track their actual support explicitly;
  the initial Huffman DCT path is not a complete PJA port. Original opaque-byte
  modes must be identified as such, never counted as coefficient recompression.
  Latest user steering: original PJA bitstream/archive compatibility is not a
  deliverable or a constraint on the final format. Stop public .pja APIs and
  CLI compatibility expansion. Retain existing original-code comparisons only
  as development checks of translated algorithms. First finish the port; the
  eventual jxl_plus format will use the original JXL codestream plus additional
  extensions. The existing experimental v0 is not that final format design.
  The user now authorizes removing format integrity hashes and testing static
  instead of adaptive probabilities. Keep structural/resource checks and exact
  original-byte tests. Static-model comparisons must include transmitted table
  costs and actual decoding; do not treat a frozen uniform prior as a fitted
  static model. Latest cleanup instruction removes old PJA archive/hash/plugin
  helpers and their protocol-only tests, implicit GivenTables, recursive archive
  repair/patch syntax, whole-file raw fallback and original-size admission gates.
  Preserve all coding tools (Full/Fast/V7, MED/Blend/symbol, dynamic/static and
  JPEG/JLS/J2K/HT paths) for selection/composition with JXL tools; a weak result
  in one experiment does not authorize deleting a coding tool. Keep original-
  source algorithm oracles, valid-input reconstruction and boundary tests.
  The user further clarifies that the final JXL-derived extension format must
  support adaptive probabilities and pixel-lossless images as well as
  byte-reversible JPEG. Treat `comr9` as an incomplete pixel-model research
  prototype, not a finished format to copy. Preserve both static/JXL entropy
  and adaptive-model options; signal deterministic model profiles, parameters
  and reset/dependency rules in the stream rather than decoder environment
  variables. JPEG-only experiment results do not establish pixel-model results.
  No large corpus run unless explicitly requested; focused local tests and
  development-only original-source oracles are allowed.
  The user now explicitly requests larger-dataset validation of compression
  gains alongside speed optimization. Fixed, documented local corpus runs are
  authorized; report development and previously unused samples separately.
  Judge compression by aggregate complete bytes on the fixed dataset; the user
  accepts individual files growing. Distinguish fixed-profile results from
  trying every tool and selecting the smallest result per file.
  Keep default JPEG encoding on V7 with its original band rule; do not enable
  probability refinement or extra parallel bands by default. Continue speed
  optimization of that default. The user next wants the pixel-lossless Plus
  route to reuse JXL tools with optional adaptive probability coding.

- Port the existing optimized libjxl C++ source, starting at commit
  `b866b3defec13494bfbaf705db93581cda118e37`. Record source files for each module.
- Do not substitute an existing Rust encoder. Do not introduce zune-jpegxl,
  jxl-oxide, a C++ wrapper, Highway bindings, native codec linkage, or subprocess
  encoding as the implementation. The rejected bootstrap has been deleted.
- Implement SIMD using Rust `std::arch`, runtime feature detection, and a tested
  portable scalar fallback. Isolate and document necessary unsafe intrinsics.
- Current user priority is x86_64 and portable Rust. Do not add further ARM
  kernels, ARM-specific optimization, or ARM-specific test work unless requested.
  Keep existing ARM code; do not remove it as an unrelated cleanup.
- Reuse suitable Rust utility crates for threading, CLI, image I/O and other
  supporting work. A dependency-free implementation is not a project goal.
  PNG and JPEG pixel input libraries are now explicitly authorized for the first
  complete pixel-lossless milestone. Keep pixel input separate from the later
  byte-reversible JPEG coefficient transcoder.
- Official `djxl` is a development-only independent decoder oracle. The user
  authorized maintaining official `jxl-rs` source under `vendor/jxl-rs` for
  runtime decoding. Keep its provenance and local patch ledger. It must not
  replace encoder migration. Native 17..31-bit integer decoding and genuine
  incremental input are now explicitly authorized beyond the rendered djxl API.
  Do not pass wide integers through floating-point storage for exact decoding.
- Work in stages. Do not expose a working encoder API until it actually emits
  complete standard-compatible JXL. Helper tests are not codec compatibility.
- The first acceptance phase is the COMPLETE PIXEL-LOSSLESS PORT, not merely
  a legal-bitstream demonstration. The user now authorizes JPEG byte-transcode
  prerequisite tools in parallel with the remaining pixel work. This does not
  mark the pixel toolset, optimized paths, scheduling or API contracts complete,
  nor does a JPEG parser constitute a working byte-reversible JXL transcoder.
  Track missing features in docs/PIXEL_LOSSLESS_PORT_STATUS.md; smoke tests are
  not phase acceptance.
- Scope: pixel lossless, byte-reversible JPEG transcode, multiframe, and
  now-authorized Python integration through a Rust C ABI. No lossy encoder
  or neural scheduling. Linux/Windows x86_64 CI for the CLI and Python wheels
  is authorized; macOS CI and local WSL remain out of scope.
- macOS runtime work is out of the current acceptance scope. Do not start WSL
  or install Linux toolchains without explicit user approval. Cross-platform
  support does not authorize changing or starting a separate runtime environment.
- For special JPEG inputs, distinguish a normative format limitation from a
  reference implementation limitation. Standard-compatible extensions are
  authorized; private archival boxes or pixel fallbacks are not a substitute for
  byte-reversible JPEG recompression.
- Pixel acceptance compares original decoded samples including invisible RGB;
  JPEG acceptance compares original reconstructed JPEG bytes. Identical JXL bytes
  are not a compression goal. Low-level primitive equivalence tests are allowed.
- Prioritize lossless pixel decoding, byte-exact JPEG reconstruction and exact
  multiframe reconstruction. Lossy VarDCT rendering need not numerically match
  C++ down to an arbitrary floating-point tolerance. Record small finite
  differences without blocking lossless work; crashes, invalid dimensions,
  missing frames/channels and substantial rendering errors still need attention.
  Do not relax lossless comparisons or infer full upstream conformance from this.
- This is an independent repository on main. Do not modify the old C++ projects.
  Do not stage rejected bootstrap artifacts, data, targets, or temporary binaries.
  Do not commit/push without the main agent's explicit instruction.
- Use disjoint subagent file ownership. Parent runs cargo tests centrally.
- Preserve upstream BSD attribution for translated code. No license for unrelated
  new project code is implicitly selected by that attribution.
