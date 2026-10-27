# Platform and Hostile-Input Validation

This is a reproducible validation procedure, not a declaration that every
platform or every JPEG XL feature has passed. It does not modify CI, install
packages, provision machines, or invoke an external codec as the implementation.
Python bindings are outside its scope.

**Current acceptance scope is Windows and Linux only.** macOS is explicitly out
of scope, not an outstanding delivery item. Do not add macOS CI or provision a
macOS runner for this work. Existing generic runner support does not change that
scope.

## Current Stop Status

The user subsequently authorized **hosted Linux/Windows CI** for CLI and Python
wheel builds. See [CI guide](CI.md). That permission does not resume local WSL
work; Linux runtime acceptance still requires an actual successful hosted run.
The stopped local preparation below remains historical evidence only.

Linux runtime acceptance is **NOT complete**. WSL work was stopped at the user's
request. Do not launch any further WSL commands, including distribution listing,
or Linux Cargo builds/tests without a new explicit instruction. Windows runs of
`tools/verify_platform.py` do not invoke or look up WSL. The commands below are
reference procedures, not permission to resume Linux work.

Before the stop, the three official Rust 1.89 Linux component archives were
downloaded, verified against their official `.sha256` files and installed solely
under ignored `out/linux-toolchain/toolchain`, with `--disable-ldconfig`. The
isolated Cargo cache contains 82 public locked registry packages, each checked
against `Cargo.lock`; `cargo fetch --locked --offline` completed. No user Cargo
configuration or credentials were copied/read. These are preparation results,
**not a Linux codec compilation or test pass**.

Download, installation and cache-preparation sessions all completed. After the
stop request, a Windows-side process query found no matching task-owned WSL
launcher. No WSL shutdown, directory deletion or unrelated process termination
was performed. Generated artifacts/logs are retained. A final Cargo-version
launcher check was interrupted; it is not counted as additional verification.

## Evidence Levels

- `probe`: record the host, Rust toolchain, installed targets, Git revision and
  worktree status. **No compilation or codec tests run.**
- `check`: offline compilation of workspace targets, optionally for a different
  target triple. **Cross-compiling is not runtime validation.**
- `test`: run the selected tests serially on the native host. The lossless suite
  covers decoder/layers, JPEG reconstruction, lossless edges, automatic
  multiframe selection, native wide integers, incremental decoding, hostile
  inputs and CLI tests. The workspace suite runs all ordinary workspace tests.
  Ignored external-data/oracle tests stay ignored.
- `binaries`: execute explicitly supplied, already-built libtest binaries on
  their actual runtime OS; no Rust installation is required there. A Linux ELF
  cross-compiled on Windows and executed by the Linux kernel is genuine Linux
  runtime evidence. Cross-compilation alone still is not.

Every command has its exact argv, working directory, UTC start, duration,
deadline, exit status, separate stdout/stderr logs, and retained log tails in
`report.json`. Reports include `rustc -vV`, `cargo -Vv`, lockfile and runner hashes,
and selected build environment variables (not all environment variables or
registry credentials). `build_host` and `runtime_host` are separate. In binaries
mode an optional `--build-report` records build-side evidence; absent evidence
leaves the build host unknown, not guessed from the runtime. Binaries have their
SHA-256 and executable format recorded. Supplied build reports are not a
cryptographically verified association with those binaries. A missing required
tool, failed command, interruption or timeout is not a pass. Optional environment
probe failures remain visible in the report.

Reports can contain local filesystem paths and tool diagnostics. Keep them in
ignored `out/`, review before publishing, and never commit private fixture paths
or credentials. Source files contain no machine-specific paths.

## Commands

Run from the repository root with Python 3.10+; use `python3` on systems where
that is the Python executable. Rust and dependencies must already be installed.
The runner always uses `--locked --offline`, never downloads missing tools or
dependencies, and refuses to overwrite an existing report directory.

```text
python tools/verify_platform.py --mode probe
python tools/verify_platform.py --mode check
python tools/verify_platform.py --mode test --suite lossless --timeout 600
python tools/verify_platform.py --mode test --suite workspace --timeout 1800
```

Use an already installed MSRV toolchain without installing anything:

```text
python tools/verify_platform.py --mode check --toolchain 1.89.0
```

A separately installed target can be compile-checked, but cannot be labeled a
runtime pass:

```text
python tools/verify_platform.py --mode check --target x86_64-unknown-linux-gnu
```

For Linux acceptance, either run the same native `--mode test` command
**on Linux** with that OS's Rust installation/cached dependencies, or
build compatible test binaries elsewhere and run them there in binaries mode:

```text
python3 tools/verify_platform.py --mode binaries --test-binary out/linux-tests/decode_hostile --build-report out/windows-build/report.json
```

Repeat `--test-binary` for each test executable. Binaries mode never compiles or
requires a matching compiler host: it validates the executable's native format
and architecture (ELF64, thin Mach-O64 or PE), executes it locally, and requires a
nonempty passing libtest summary, not merely exit code zero. A Windows PE launched
through WSL interoperability cannot be reported as a Linux test. Do not pass
untrusted third-party executables. This is a test runner, not a binary sandbox.

The combined build-and-run `test` mode rejects a mismatched compiler host and
`CARGO_BUILD_TARGET`, passing an explicit native target to Cargo. This is a
limitation of that convenience mode, not a rejection of cross-built native
runtime evidence. Custom Cargo runner/wrapper configuration is an environmental
trust boundary: record the real execution OS rather than assuming it equals the
compiler host. Nothing here adds architecture-specific kernels or linker wrappers.

Commands run sequentially, and libtest uses `--test-threads=1`; individual tests
may still intentionally exercise the library's internal parallel execution.
Do not launch this concurrently with the main agent's central cargo run. Cargo
build time is included in each command deadline. A timeout kills the process
group on POSIX or the process tree with `taskkill /T /F` on Windows, reaps the
child, records failure and stops. Descendant cleanup failure is explicitly
reported; an unkillable child is a fatal harness error, never silently ignored.

Lightweight runner tests (no cargo or codec compilation):

```text
python -B tools/test_verify_platform.py
```

## Deterministic Hostile Corpus

`crates/rs-jxl/tests/decode_hostile.rs` is self-contained. Tiny valid RGBA Modular
and grayscale JPEG-transcode seeds are constructed in memory. No private data,
large fixtures, network downloads or new dependencies are needed.

The four parent tests cover:

1. Bounded prefix truncations of raw/fragmented-container JXL, 128 fixed-seed
   single-bit mutations and 64 bounded pseudo-random byte streams. They exercise
   pixel decode, header/structure inspection and reconstruction entry points.
2. Invalid 32/64-bit box lengths, truncated/compressed metadata, legal unknown
   boxes and an 8 KiB Brotli-expanded metadata budget boundary. Selected-frame
   early stopping is explicitly distinguished from whole-file validation.
3. Original-byte JPEG reconstruction with incomplete input, missing/duplicate
   `jbrd`, mutated reconstruction payloads and malformed reconstruction box
   sizes. `allow_partial` must not permit incomplete reconstructed JPEGs.
4. Input/pixel/sample/output/frame limits and invalid thread options, including
   correct `Error::Limit` versus `Error::InvalidInput` classification.

Seed: `0x83e196d402a75b19`; generator: explicitly defined xorshift64. Every worker
logs its seed/batch/case before potentially problematic operations. Corpus sizes
are fixed/bounded: at most 176 cut positions per fixture, 64 mutations per
mutation loop, 255+1 bytes per pseudo-random input. Decoder caps are 32 KiB input,
256 KiB retained output, 16 KiB metadata, 4096 pixels, 16384 samples and 4 frames.
These are API work/output budgets, **not an OS-enforced process RSS ceiling**.

Each batch executes the current test binary in an isolated child with a fixed
45-second deadline. A panic, abort, nonzero child exit or timeout fails the parent
test; stuck decoder threads are not detached. The subprocess-only worker is
marked ignored, and parents invoke it explicitly. **Do not use `--include-ignored`
or run the worker directly.** This preserves normal `cargo test` behavior.

```text
cargo test --locked --offline --release -p rs-jxl --test decode_hostile -- --test-threads=1
```

Strictly malformed structures must return errors. Arbitrary bit mutations may
create a different valid image or legal JPEG reconstruction metadata: accepted
outputs must obey dimensions, lengths and budgets, but are not required to
match an unrelated original fixture. Unmodified valid controls are checked
against original pixels/JPEG bytes exactly. Header inspection is never treated
as an entropy-stream integrity check. Compressed JXL byte identity is not an
acceptance criterion.

This is bounded regression coverage, not an exhaustive fuzzer or security
certification. Separate large-corpus, long-running fuzzing, sanitizers and
independent official-decoder acceptance remain necessary for wider claims.

## Observed Environment, 2026-10-24

Initial read-only environment probes found (see the current stop status above
for subsequent isolated toolchain preparation):

| Environment | Observed capability | New runtime validation in this work |
| --- | --- | --- |
| Windows x86_64 | Rust 1.97.1, host `x86_64-pc-windows-msvc`; stable, nightly and 1.89.0 toolchains installed | 92 integration, 295 JPEG unit and 82 CLI tests passed; see DECODER_STATUS.md |
| Linux cross-target | `x86_64-unknown-linux-gnu` standard library target installed in Windows rustup | Not run; installed target does not establish linker or runtime availability |
| WSL2 Ubuntu x86_64 | Linux execution, GCC 13.3.0, GNU ld 2.42, glibc 2.39, Python 3.12.3 available; `rustc`/`cargo` absent | No codec tests run; Linux linking prerequisites exist |
| Container runners | Docker/Podman commands not found on Windows PATH | Not run |

Initial sandboxed WSL enumeration was denied; subsequent probes could list
Ubuntu and execute `/bin/uname` and executable availability queries. That initial
probe stage installed nothing. The later isolated preparation is recorded above.
WSL may start its existing distribution for a command; the portable runner no
longer invokes WSL at all, including enumeration.
Other registered cross-target standard libraries do not establish platform
support. These observations are not a pass/fail verdict on unrun platforms.

## Linux Execution Feasibility

Additional read-only WSL probes confirmed `gcc -dumpmachine` returns
`x86_64-linux-gnu`. `gcc -print-file-name=...` resolves existing `crt1.o`,
`Scrt1.o`, `crti.o`, `crtn.o`, `libgcc_s.so.1` and `libc.so`. Standard C headers
and `/lib64/ld-linux-x86-64.so.2` exist. The Windows Rust installation contains
Linux-target `libstd` rlib/rmeta/shared-library files. These checks establish
prerequisites, **not a successful Rust link or codec execution**.

Two routes considered during the initial inspection:

1. Windows Cargo/Rust emits Linux objects, a carefully reviewed linker wrapper
   invokes WSL GCC, then Linux executes the resulting ELF test binaries. This
   avoids installing Linux Rust, but must translate Windows paths in ordinary
   arguments, response files, `-L` and linker-script options, preserve quoting,
   propagate signals/exit status, and keep build artifacts separate. Cargo build
   scripts/procedural macros remain Windows-host tools. A single host/target
   mistake can silently test the wrong binary; record both hosts and verify ELF.
2. A portable native Linux Rust toolchain under ignored `out/` avoids linker
   translation and needs no package-manager/global install. Rust compiler,
   standard library and Cargo must be a matching release, extracted/installed
   only under the chosen local prefix; verify official SHA-256 before use.
   Existing source/cache access and execute permissions must still be checked.

WSL HTTPS **HEAD-only** requests to the official 1.89.0 x86_64 Linux archives all
returned HTTP 200 on the observation date:

| Component | Compressed bytes from Content-Length |
| --- | ---: |
| `rustc-1.89.0-x86_64-unknown-linux-gnu.tar.xz` | 81,862,808 |
| `rust-std-1.89.0-x86_64-unknown-linux-gnu.tar.xz` | 28,894,028 |
| `cargo-1.89.0-x86_64-unknown-linux-gnu.tar.xz` | 10,120,792 |

Total approximately 115.3 MiB compressed; unpacked space is additional. Origin:
`https://static.rust-lang.org/dist/`. `curl` 8.5.0 and GNU tar 1.35 are already
available in WSL. Those initial HEAD probes downloaded no archives; the later
isolated download/verification/installation is recorded in the current stop
status. No linker wrapper, Rust link or Linux codec test was executed during
preparation. Network access later may differ. The Windows-to-WSL link route
remains an unimplemented alternative, not a validated path. All Linux work is
currently stopped.
