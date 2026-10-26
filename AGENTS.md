# Direct Rust Port Rules

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
- Scope: pixel lossless, byte-reversible JPEG transcode, multiframe, and a future
  Python-facing C ABI. No lossy encoder, no neural scheduling, no CI changes.
- macOS runtime work is out of the current acceptance scope. Do not start WSL
  or install Linux toolchains without explicit user approval. Cross-platform
  support does not authorize changing or starting a separate runtime environment.
- For special JPEG inputs, distinguish a normative format limitation from a
  reference implementation limitation. Standard-compatible extensions are
  authorized; private archival boxes or pixel fallbacks are not a substitute for
  byte-reversible JPEG recompression. Python integration remains deferred.
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
