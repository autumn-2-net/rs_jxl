# Frozen C++ Reference Build

This is a development-only comparison executable, not a Rust dependency.
The Rust encoder does not load or execute it.

## Build Record

Built locally on 2026-10-23, Windows x86_64:

- Source: `../jxl_python/libjxl`.
- Source HEAD: `b866b3defec13494bfbaf705db93581cda118e37`.
- Source status before/after: only pre-existing ` m third_party/zlib`.
  This is therefore not a claim of pristine dependency provenance.
- Build: `./out/cpp-b866b3d`, separate from source.
- CMake 4.3.1, Ninja, clang-cl 17.0.6, Release, static libraries.
- Configure: `JXL_PATCH_ACTION=check`, `JXL_PYTHON_ENABLE_NATIVE=OFF`,
  `CMAKE_POLICY_VERSION_MINIMUM=3.5`. No patches applied to source.
- Target: `cjxl`, eight build jobs, completed in 47.685 seconds.
- Executable: `out/cpp-b866b3d/libjxl/tools/cjxl.exe`.
- Executable SHA256:
  `ffc5ff48b4544d3b7e4fdeed6cfff9ec662974206b1fd04d7b1f96e5ad3b336d`.

The executable digest identifies the tested binary. It is not a target for
compressed output or a pixel-correctness criterion.

## Version String Caveat

The freshly built executable prints `cjxl v0.13.0 74ad654`. This is the parent
superbuild repository's HEAD, not the libjxl HEAD. Frozen `tools/CMakeLists.txt`
passes `CMAKE_SOURCE_DIR` to `tools/git_version.cmake`, so the configure log
reports libjxl `b866b3d` but the generated version header reads the outer repo.

Do not infer binary source provenance from that string, timestamps, or current
HEAD alone. This record describes the observed isolated build. The generic
comparison harness continues to label externally supplied binaries unverified;
it does not automatically certify this record.

## Acceptance

Compare original decoded samples using independent official decoders. Record
bytes, bpp, wall time and MPix/s separately for a single native candidate and
the complete scheduler. Equal effort labels do not prove equal search paths.
Metadata-free benchmark snapshots measure pixel coding, not metadata retention.
