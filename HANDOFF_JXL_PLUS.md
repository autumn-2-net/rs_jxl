# JXL Plus conversation handoff

## Archived experiment checkpoint

The user requested saving the complete Plus work on branch `jxl_plus`, then
returning `main` to its pre-Plus commit
`d003ae07e97c84d17ba616c9e93c7e1bcd15c1f7` and prioritizing standard-JXL decoding
speed. The original handoff below is historical; its documentation-only scope
and proposed filename extension were superseded by later user instructions.

Current experiment status, validation and measurements are in
`docs/JXL_PLUS.md`, `docs/PJA_PORT_STATUS.md` and
`docs/JXL_PLUS_BITSTREAM_DRAFT.md`. Source, tools, attribution and the supplied
`comr9` source archive are preserved on this branch. Test images, `target/` and
local experiment receipts under `out/` remain local and are not Git contents.
The format draft is not frozen. Further standard-JXL work belongs on `main`.

Date: 2026-11-17. This is a disposable handoff, placed in the repository root
at the user's request. It is not an approved format specification.

## Start here

The next conversation should work in **D:/cpp_pj/rs_jxl**, not
`D:/cpp_pj/jxl_python`. The editor now displays the Rust repository, but the
old conversation still reports the C++ project's directory and write scope.
This mismatch was observed; its client-side cause was not established.

Before editing, verify the tool working directory, repository root, branch,
worktree and write permissions. Merely changing the shell directory does not
change the tool's permission scope. Do not keep using the old experiment clone
as the primary repository.

```powershell
Get-Location
git rev-parse --show-toplevel
git status --short --branch
git log -1 --format="%H%n%s"
```

Verified handoff baseline:

- Repository: `D:/cpp_pj/rs_jxl`, remote `git@github.com:autumn-2-net/rs_jxl.git`.
- Branch: `main`, aligned with the locally recorded `origin/main`.
- Commit: `d003ae07e97c84d17ba616c9e93c7e1bcd15c1f7`.
- Subject: `Organize JPEG compression experiments and add entropy diagnostics`.
- Worktree was clean before adding this document.
- Latest request is documentation only. No JXL Plus implementation, new crate,
  branch, commit or push is authorized by this handoff alone.

Read `AGENTS.md` first. Some rules describe standard-JXL compatibility. The
new discussion proposes a separate, deliberately incompatible backend, not
removing compatibility requirements from the existing standard backend.
Clarify and document that scope before implementation.

## User's proposed direction

The user proposes an experimental format tentatively named **jxl_plus**, with
extension **.jxlp**, without requiring its bitstream to be readable by standard
JXL decoders. Reuse the existing Rust infrastructure through a workspace;
eventually expose both formats through the CLI and Python bindings.

This is a design discussion. The user has not yet approved the exact format,
first milestone, extraction boundaries, experimental branch or API details.
Do not treat the suggestions below as already implemented or all agreed.

The motivation is to explore decoder-visible models that cannot simply be
inserted into standard JXL. It is not proof that standard JXL has reached its
compression limit, nor a guarantee that a new format will compress better.

## Existing architecture and proposed reuse

The root `Cargo.toml` is **already a virtual Cargo workspace**, without a root
package. It currently contains:

```text
crates/rs-jxl/        Standard JXL encoder, decoder integration and tools
crates/rs-jxl-cli/    CLI and experimental native examples
crates/rs-jxl-image/  Pixel input adapters
crates/rs-jxl-ffi/    Rust C ABI used by Python
python/rs_jxl/       Python binding
vendor/jxl-rs/       Maintained official Rust decoder sources
```

Workspace metadata: edition 2024, Rust minimum 1.89, release thin LTO and one
codegen unit. Keep these as the current baseline, not a new toolchain mandate.

Suggested additions, with names and boundaries still provisional:

```text
crates/codec-core/   Only genuinely reusable primitives/types, if needed
crates/rs-jxl-plus/  Independent format encoder and decoder
```

Share algorithms, not format restrictions. Possible reuse includes JPEG
parsing and byte reconstruction, SIMD, histogram/statistical kernels, matching,
buffer handling and thread scheduling. Keep syntax, predictor semantics and
decoder state specific to each backend. A workspace alone does not provide
code sharing; dependencies must remain acyclic.

Do not rename or move the whole existing codec into a generic core. Extract
only what the first experiment actually needs. Existing image types, errors,
threading helpers and entropy writers have JXL-specific coupling. In particular,
`rs-jxl-image` currently depends on `rs_jxl::Image`; account for this before
designing the shared type layer. The existing vendor decoder cannot decode
new entropy semantics without implementing them.

Useful starting points:

- `crates/rs-jxl/src/jpeg/`: coefficient parsing, reconstruction and experiments.
- `crates/rs-jxl/src/entropy/`: ANS, histograms and diagnostics; distinguish
  reusable mathematics from JXL-specific serialization.
- `crates/rs-jxl/src/image.rs` and `execution.rs`: shared-type and scheduling
  candidates, currently coupled to this crate.
- `crates/rs-jxl-ffi/protocol.md`: existing ABI, errors and ownership contracts.
- `python/rs_jxl/encode.py`: current encode and JPEG transcode interfaces.

## Format and interface boundaries

Keep existing calls producing standard JXL by default. Proposed future API
shape, **not currently implemented**:

```python
rs_jxl.encode(array, format="jxlp")
rs_jxl.transcode_jpeg(jpeg_bytes, format="jxlp")
rs_jxl.decode(encoded_bytes)  # Detect by file signature, not just extension.
```

Retain path, in-memory bytes, NumPy and Torch input/output adapters. Actual
codec selection and image processing belong in native Rust, not Python.
Validate options separately per backend and report unsupported operations;
never silently replace byte-exact JPEG reconstruction with pixel encoding.
Evolve the C ABI additively or with explicit version/capability handling.

`jxlp` is already the standard JXL Partial Codestream box identifier. That
does not prohibit this tentative extension, but creates naming ambiguity.
Use an independent magic signature and explicit format version. Do not make
the new format look like standard JXL or claim official decoder compatibility.
The name, extension and signature remain to be finalized.

## Suggested first experiment, pending agreement

Start with **JPEG byte-exact recompression**, where parsing/reconstruction and
diagnostic infrastructure already exist. This is a suggested order, not a
confirmed user decision. Pixel lossless and multiframe remain intended scope.

1. Define a minimal versioned container, supported JPEG subset, rejection rules
   and resource limits; implement a matching native decoder.
2. Establish a correct baseline using original JPEG coefficients and the
   metadata/reconstruction information required to restore the original file.
3. Experiment with stronger neighbor/edge AC conditioning and AC-assisted DC
   prediction. Merely reordering existing JXL tokens does not implement these.
4. Compare complete size, encode/decode time, throughput and peak memory against
   current standard JXL on the same inputs. Include model and container costs.
5. Expand to pixel-lossless and multiframe after proving a useful first route.

Investigate chunk-local modeling with independent chunks before accepting a
globally serial adaptive model. Measure reset overhead against parallel speed.
Keep numerical rules deterministic: predictor arithmetic, overflow, rounding
and probability updates must agree between encoder and decoder.

Do not present a wrapper around unchanged JXL as a new compression improvement.
Experimental stream revisions may change, but each must be identifiable and
unsupported versions must fail clearly.

## Correctness and working constraints

- Standard JXL remains standard-compatible and independently checked with djxl.
  The proposed new format has its own decoder and must not claim djxl support.
- Pixel lossless means original samples, including invisible RGB and alpha,
  appropriate bit depth, frames and explicitly supported metadata semantics.
- JPEG lossless transcode means original **JPEG bytes**, not just equivalent
  coefficients or rendered pixels.
- Own encoder/decoder roundtrip is necessary but insufficient. Use originals,
  small reference implementations/known vectors, malformed and truncated input
  tests, and relevant independent checks.
- Encoded output identity is not a compression objective. Compression changes
  may change every encoded byte as long as the required reconstruction holds.
- Keep implementations native Rust. Utility crates are allowed; do not hide a
  C++ encoder or subprocess codec behind a Rust wrapper. Preserve attribution
  and check licensing before adapting external implementations.
- Priorities are Windows/Linux x86_64. No macOS, WSL, new ARM, lossy-encoder or
  neural-scheduling work unless separately requested.
- Keep tests proportional to changed behavior. Do not launch a full corpus
  until the user explicitly starts it; large runs are scheduled while the PC
  is otherwise idle. Windows background helpers must not open visible consoles.
- Preserve existing GitHub noreply identity. Do not commit/push automatically,
  alter old C++ repositories or add datasets, temporary binaries or personal data.
- User prefers economical subagents for bounded tasks, previously requesting
  6.1sol where available. Use actual available models, disjoint file ownership
  and centralized build/test execution; no need to spawn agents for this handoff.

## Latest JPEG findings and saved evidence

The latest commit organized opt-in experiments for AC histogram groups,
BlockContextMap, coefficient orders and selected entropy-model diagnostics.
These experiments did not change ordinary JPEG encoding defaults.

Read these existing reports instead of repeating the investigation:

- `docs/JPEG_API.md`
- `docs/JPEG_COMPRESSION_TUNING_20261116.md`
- `docs/JPEG_AC_MODELS_20261117.md`
- `docs/JPEG_CONTEXT_ORDER_20261117.md`
- `docs/JPEG_ENTROPY_AUDIT_20261117.md`

The 45-file audit includes 16 compatibility fixtures, nine project JPEGs and
20 CC12m JPEGs. It is not a representative sample of 45 independent photos.
Original size was 33,406,801 B; JXL size was 27,372,627 B, or 2.061961 weighted
bpp over 106,200,376 pixels. AC data accounts for 91.1193% of complete files.
The selected-table fitting gap was only 0.0378% of file size; the raw-context
versus clustered empirical gap was 0.6434%, and 3.2400% in the CC12m subset.
These are conditional diagnostic references, not guaranteed attainable savings.
They overlap and must not be added together. The audit's 22.023 s includes
4.038 s of diagnostic work and is not ordinary production encode speed.

PJA source was inspected at
`1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd` (source version 0.9.13).
It uses stronger interior/edge AC conditioning, AC-informed DC prediction and
decoder-visible adaptive arithmetic models. No PJA code was transplanted and
no PJA executable was benchmarked. Details and pinned source links are in the
entropy audit report. Do not describe PJA's published results as local results.

Last recorded post-cleanup checks, not rerun for this documentation task:

- 407 JPEG Rust tests passed; nine external-oracle tests explicitly ignored.
- Ten entropy-audit tests and two CLI tests passed.
- 29 Python receipt/accounting tests passed.
- Earlier validation covered 66 full encodes with both Rust and official djxl
  reconstructing the original JPEG bytes.
- All three saved audit summaries were reproduced from their receipts.

## Local artifacts and environment

Historical artifacts remain under the old project because of the earlier tool
workspace. They are references, not destinations for new work. Do not move or
delete them automatically. Use `rs_jxl/out/` for future artifacts once the new
conversation has the correct workspace.

```text
D:/cpp_pj/jxl_python/out/jpeg-opt-20261116/cases.json
D:/cpp_pj/jxl_python/out/jpeg-opt-20261116/profile-cases.json
D:/cpp_pj/jxl_python/out/jpeg-ac-20261117/
D:/cpp_pj/jxl_python/out/jpeg-context-order-20261117/
D:/cpp_pj/jxl_python/out/jpeg-entropy-audit-20261117/
D:/cpp_pj/jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe
```

The audit directory contains `audit45.json`, `summary45.json`,
`compat-e10-h4-lz.json`, `summary-e10-h4-lz.json`, `compat-e3-off.json`,
`summary-e3-off.json` and frozen `audit.exe`.

`D:/cpp_pj/jxl_python/out/jpeg-opt-20261116/work` is an old writable experiment
clone with staged/unstaged work. It is not the authoritative clean checkout;
do not reset, commit or push it as part of resuming this handoff.

The existing Python environment can be located without hardcoding a username:

```powershell
$python = Join-Path $env:USERPROFILE '.conda\envs\py10\python.exe'
```

## Next conversation

First confirm the workspace and current code, then resolve the first milestone,
new-format identity and dependency boundaries with the user. Do not start by
mass-moving modules, weakening standard JXL checks or launching full benchmarks.

Suggested opening request:

> Work in D:/cpp_pj/rs_jxl. Read AGENTS.md and HANDOFF_JXL_PLUS.md first. Continue
> the jxl_plus / .jxlp design discussion while keeping standard JXL intact.
> Verify the workspace and current state, then identify the decisions needed
> before implementation. Do not start a large refactor or full-corpus test yet.
