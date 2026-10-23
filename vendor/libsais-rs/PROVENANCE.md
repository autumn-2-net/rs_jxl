# Vendored libsais-rs 0.2.2

This directory contains the pure-Rust utility approved for the rs-jxl suffix
index. It is not a native codec or a binding to C. The encoder calls only
`libsais_int` with one worker, via `tools/lz77_source_sais.rs`.

## Source

- Package: crates.io `libsais-rs` exactly `0.2.2`.
- Registry package checksum:
  `00777f770949ecbe5524eb6630699a501c9c27e4d18493705a7dad49c9bdfbd1`.
- Repository: https://github.com/henriksson-lab/libsais-rs
- Packaged `.cargo_vcs_info.json` revision:
  `cde7351048122fd14b485daf61957733acc4231e`.
- Upstream algorithm identified by the Rust source: Ilya Grebnov's libsais
  `2.10.4`. The frozen libjxl b866b3d adapter requests an external libsais;
  that checkout itself does not pin an external engine revision.
- Imported read-only from the cached registry package on 2026-10-23.
- `LICENSE` and `README.md` are unchanged. The utility and local changes use
  its Apache-2.0 license. The caller adapter retains its separate libjxl BSD
  attribution; the utility is not relicensed as BSD.

Original file SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `src/lib.rs` | `86e314fb2ccdb606484d62463bda55f0bee38bc455f3adf8e476dac3071d9d5a` |
| `src/libsais16.rs` | `a41250c1e4c8905553af065d5002b6e5d1d0a78e58b733d8fd9ef7c1e47e7365` |
| `src/libsais16x64.rs` | `81c72c4ae5e509c4a3ffbd86636c6ec317111febb12bb8b0e5d6203598f42b21` |
| `src/libsais64.rs` | `b17c08b5f0fbe1fa6414d668a016645534ed2ef7081d9dda0da18c89b578f805` |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `README.md` | `4622cd6bac73f560b7f82604c6a27a3f1400c98dcded2f158c37e87a62a1fe5a` |

## Local Changes

Five helpers in `src/lib.rs` now return through a sequential branch when their
effective worker count is one. Previously a singleton Rayon iterator could
initialize the process-global pool, even when `libsais_int` passed threads=1.

| Helper | Sequential branch |
| --- | --- |
| `partial_sorting_shift_markers_32s_6k_omp` | Identical descending bucket traversal and four-element marker scan, including its scalar remainder. |
| `mark_distinct_lms_suffixes_32s_omp` | Existing scalar kernel over `[0, n/2)`. |
| `clamp_lms_suffixes_length_32s_omp` | Existing scalar kernel over `[0, n/2)`. |
| `reconstruct_lms_suffixes_omp` | Existing scalar kernel over `[0, m)`. |
| `clear_lms_suffixes_omp` | Identical ascending bucket traversal and conditional zero fill. |

The parallel branches, thresholds, marker arithmetic and other algorithms are
unchanged. No changes were made to the three sibling Rust modules. Only the
integer serial entry point is integrated/audited for the encoder; this patch
does not claim every utility API avoids a global or auxiliary pool.

The manifest retains package identity/version, library path, empty default
features and the upstream `rayon = "1.10"` dependency. It omits examples,
benchmark profiles, the optional `cc` dependency, C comparison feature and
build script. `build = false` prevents build-script discovery. Retained source
`cfg(feature = "upstream-c")` sections are acknowledged by check-cfg but that
feature is deliberately not declared or selectable. Thus ordinary Cargo feature
selection, including all-features, cannot enable the C comparison APIs.
An empty `[workspace]` keeps this third-party dependency outside the parent
workspace's test/lint targets. No root manifest/lockfile was changed here.

The sole direct runtime dependency is Rayon, resolved by the parent lockfile.
At audit time it was Rayon 1.12.0 (declared MSRV 1.80) and rayon-core 1.13.0,
with either and crossbeam-deque/epoch/utils. No build dependencies remain.
The package uses edition 2021 and declares no upstream MSRV. Parent's Rust
1.89 all-target check passed for this vendored integration on 2026-10-23;
the required compatibility check is no longer pending. Internal unsafe code
has not received a full soundness audit; serial-only use avoids its parallel
full-buffer mutable aliases.

## Integration And Verification

The parent workspace selects this copy, keeping the existing exact-version,
default-features=false dependency declaration:

```toml
[patch.crates-io]
libsais-rs = { path = "vendor/libsais-rs" }
```

The rs-jxl adapter checks n<=2^24 and symbols<2^20, copies mutable input,
allocates the source n+6*k suffix storage, and checks its scratch ceiling before
allocating. Nonzero utility status or invalid output indices request the
existing static-index fallback. The old radix-doubling implementation is removed.

Focused rs-jxl tests cover the five helper branches against scalar kernels and
an independent marker oracle, exhaustive three-symbol suffix ordering through
length seven, recursive/wide inputs, budget/status failures, and all four suffix
parser modes through independent command decoding. An isolated unit-test process
calls all five helpers and serial SA-IS, then proves it can still initialize
Rayon's global pool. That deliberate initialization is test-only, not a runtime
workaround. Tests/builds are run centrally by the parent, not by this vendor edit.

Parent verification, 2026-10-23: the vendor-backed core suite compiled with the
current default Rust toolchain, and all SA-IS-focused tests passed, including
the isolated no-global-Rayon-pool test. Parent reports 1316 tests in that run,
1304 passed overall, and one unrelated new FJXL test failure owned by another
worker; this was not a completely passing core-suite checkpoint.

Subsequent parent verification, 2026-10-23: all three vendor-backed release
source-LZ suites passed, with 172 independent pixel comparisons including
suffix routes. Strict Clippy and the Rust 1.89 all-target check also passed.
The Rust 1.89 result is a compilation check, not a claim that tests were run
under that toolchain. SA-IS code remains frozen.
