# Module Organization Checkpoint, 2026-10-29

This maintenance pass is based on `301dc8f`. It reorganizes implementation and
build infrastructure, not encoder policy. The current responsibility map is in
[architecture and maintenance](ARCHITECTURE.md).

## Changes

| Area | New ownership |
| --- | --- |
| MA training | `modular/ma_train.rs` retains the public entry points, constraints and constants. `ma_train/` owns preparation, parent histograms, cache leases, scratch, split scoring/search and learner orchestration. |
| LZ77 | 39 existing files moved under `tools/lz77/` following their existing Rust module hierarchy. Matcher/index/parser tests live with their owning modules; root inline tests moved to `lz77/tests.rs`. |
| FFI requests | `request.rs` owns the operation envelope and validation; `request/deserialize.rs`, `image.rs` and `options.rs` own JSON parsing and wire schemas. Existing names are reexported. |
| Large test bodies | `modular/training/tests.rs` and `scheduler/adapter/*_tests.rs` separate tests from production orchestration while retaining their qualified names. |
| Python builds | `setup.py` registers commands from `build_support/`. `native.py` validates/discovers artifacts; `sources.py` checks source/license completeness; `commands.py` implements setuptools hooks. |

Entry-file line counts show where implementation moved; they are not a reduction
in codec functionality or a performance metric:

| File | Before | After |
| --- | ---: | ---: |
| `modular/ma_train.rs` | 4,025 | 673 |
| `modular/training.rs` | 3,421 | 1,128 |
| `scheduler/adapter.rs` | 1,409 | 504 |
| FFI `request.rs` | 588 | 343 |
| `setup.py` | 314 | 17 |

Public Rust/Python paths, the C ABI, parameter defaults, search ordering,
arithmetic and error precedence are retained. No new encoder limits or unsafe
code were introduced. Existing experimental controls and attribution remain.
The runtime Python package, vendored code, dependency lockfiles and workflow
configuration are unchanged. CI remains manual-only, with conditional Releases.
Developer scripts retain their existing command paths. Local data/results were
not deleted as part of source cleanup.

## Validation

Host: Windows x86-64, Rust 1.97.1, Python 3.10, default workspace features.
Cargo used `--locked --offline -j 2`; ordinary Rust tests used four test threads.

| Check | Result |
| --- | --- |
| All-target workspace compile check | Passed. Two missing imports found during extraction were corrected before the full run. |
| Complete Rust workspace suite | 2,503 passed, zero failed, 109 ignored across 53 top-level summaries; nested subprocess summaries excluded. Same counts as the earlier maintenance baseline. |
| Independent official djxl 0.12/0.13 | Six normally ignored metadata/frame interoperability cases passed; five ordinary cases were already covered by the workspace run. |
| Packaging fixtures | 89 passed, including project/non-default versions, version mismatch rejection, missing helper modules, isolated setup imports and wheel/sdist roundtrips. Two existing tooling warnings. |
| CI helper tests | 27 passed. |
| Development-tool harness tests | 181 passed; no dataset benchmark was launched. |
| Source distribution | Built successfully; all 61 files under the moved/extracted module and helper directories were present, with no generated output directories. |
| Strict workspace Clippy | Passed with `--all-targets -- -D warnings`. |
| Release workspace build | Passed; CLI and native library rebuilt from the reorganized sources. |
| Fresh installed Windows wheel | CLI/Python smoke checks and 251 Python tests passed, 12 optional skips, two existing tooling warnings. This includes the packaging cases above, not an additional disjoint total. |
| Wheel contents | Bundled DLL matches this Release build; `build_support/` is excluded. |
| Formatting and local documentation links | Passed. |

The build-helper package is included in sdists but excluded from normal/editable
wheels. Fixture tests copy real helper code and rebuild from extracted sdists;
they cannot silently fall back to the development checkout's build files.

Logs and generated distributions are under `out/maintenance-20261029/`, not
repository source. This is not a new corpus compression-ratio/throughput study,
Linux/macOS runtime certification, MSRV check or claim of full format coverage.
Ignored tests remain unverified except for the explicitly selected oracle cases.
No hosted workflow or Release publication was triggered by these local checks.
