# CLI and Python Wheel CI

The [build workflow](../.github/workflows/build.yml) builds and tests **Windows
x86_64 and Linux x86_64 only**. It runs **only** through
**Actions -> Build CLI and Python wheels -> Run workflow**. Pushes, tags and pull
requests do not trigger builds.
There are no macOS, ARM, WSL, CUDA or PyPI-publishing jobs. Build jobs use read-only
repository permissions. Only the conditional GitHub Release job requests
`contents: write`.

## Optional Version

The manual form's `version` field accepts `1.2.3` or `v1.2.3`. Leave it empty to
keep the checked-out project's version and **build artifacts only**. Supplying a
version also requests a GitHub Release after both platforms pass. Whitespace-only
input counts as empty. Only stable `X.Y.Z` release versions are
accepted; prerelease suffixes and leading zeroes are rejected before compilation.
An initial `v` and surrounding whitespace are removed.

The override updates the disposable CI checkout's Cargo workspace version,
local workspace entries in `Cargo.lock`, Python project version and
`rs_jxl.__version__`. Third-party dependency versions/checksums remain locked.
The installed-wheel check also requires the wheel metadata, Python version and
CLI `--version` to agree. Artifact names and CLI archive names include this
normalized version. The version-preparation script does not commit changes back;
the separate publication job handles tags and GitHub Releases. Nothing is uploaded
to PyPI.

## Build Outputs

| Job | Build environment | Downloadable artifact |
| --- | --- | --- |
| `build-windows` | `windows-2022`, stable Rust/MSVC, Python 3.10 x64 | `rs-jxl-<version>-windows-x86_64`: `py3-none-win_amd64.whl`, versioned CLI ZIP and `SHA256SUMS` |
| `build-linux` | `ubuntu-22.04` host with the PyPA `manylinux_2_28_x86_64` container, stable Rust, Python 3.10 | `rs-jxl-<version>-linux-x86_64`: auditwheel-repaired manylinux wheel, versioned CLI tar.gz and `SHA256SUMS` |

Download these from the completed run's **Artifacts** section. Retention is
14 days. The wheel contains the Python package and its Rust native library;
the CLI archive contains `rs-jxl`/`rs-jxl.exe`, parameter documentation and
third-party notices. Linux archives retain the executable permission. Neither
artifact needs the source checkout or a runtime Rust compiler.

Both CLI executables and the wheel's DLL/shared library come from
`cargo build --release --locked --offline --workspace` and `target/release`.
The workspace's Release profile uses thin LTO and one codegen unit. Default-profile
test binaries are separate and never become release assets.

The Linux build targets glibc 2.28 or newer, not musl/Alpine. The wheel is
audited/repaired before it is installed for testing; a raw `linux_x86_64` wheel
is not uploaded. This follows the [PyPA manylinux environment](https://github.com/pypa/manylinux#manylinux_2_28-almalinux-8-based)
and [auditwheel repair](https://github.com/pypa/auditwheel) contracts. The CLI is
built in the same container, but is not itself a wheel or subject to wheel repair.
Windows uses the standard MSVC target and can require the Microsoft Visual C++
x64 runtime. CPU dispatch remains the normal portable/runtime-detected code;
CI does not enable `target-cpu=native` or require AVX512.

The CFFI package is Python-ABI-independent (`py3-none`); it is built and tested
here on Python 3.10, not on every Python version permitted by package metadata.
This workflow does not certify the declared Rust MSRV either.

## GitHub Releases

For example, enter `1.2.3` in the manual form. After Windows and Linux builds,
tests and artifact uploads succeed, `publish-release` verifies both jobs reported
that version, downloads only this run's artifacts, verifies their checksums and
assembles five assets:

- Windows wheel.
- Linux manylinux wheel.
- `rs-jxl-1.2.3-windows-x86_64.zip`.
- `rs-jxl-1.2.3-linux-x86_64.tar.gz`.
- One combined `SHA256SUMS` covering all four downloads.

The job creates Release/tag `v1.2.3` targeting the workflow's exact source commit,
not whatever the branch points to later. Existing tags pointing elsewhere and
existing Releases (including drafts) are rejected; no force-update or asset
overwrite is attempted. Choose a new version for another release.

Assets are uploaded to a draft first, then the draft is published only after
the upload command succeeds, using the documented
[GitHub CLI create](https://cli.github.com/manual/gh_release_create) and
[edit](https://cli.github.com/manual/gh_release_edit) commands. An upload failure
may leave a draft for manual inspection; the workflow does not delete it. Tag
protection or repository policies that prohibit the token's write access cause
publication to fail rather than bypassing those restrictions.

The tag's source still contains its committed default versions: the requested
version is a build-time override in the disposable checkout. GitHub's automatic
source archives are not pre-versioned wheels or binary archives.

## Validation Gates

1. Validate/apply the optional version, fetch the exact `Cargo.lock` dependencies,
   then build the Release workspace
   offline. Only Cargo dependency downloads are cached, not native build outputs.
2. Run ordinary workspace tests with two test threads in Cargo's default test
   profile. Release CLI/library outputs are tested separately below. Explicitly
   ignored corpus/oracle/subprocess tests remain ignored; no private dataset is needed.
3. Build the wheel using the native library from that build, without a second
   implicit Cargo build or a different dependency resolution.
4. Test version synchronization, invalid input rejection, archive assembly,
   required notices, executable permissions and checksums. Packaging roundtrips
   compare wheel/sdist metadata and wheel filenames against the fixture's input
   `pyproject.toml`, covering both the current project version and explicit
   non-default versions. They must not assume a particular project version.
5. Create a fresh temporary virtual environment, install the built wheel and
   test dependencies, and clear `PYTHONPATH`, `PYTHONHOME` and native-library
   overrides. Python runs with `-I` outside the checkout. Smoke checks verify
   that both the package and its native library belong to that installation.
6. Run CLI/Python pixel-lossless, JPEG-byte-reconstruction and automatic-frame
   smoke checks, then discover all `python/tests` against the installed wheel.
   Optional torch/CUDA and official `djxl` checks are not enabled in this CI.
7. Package and upload artifacts only after every required command succeeds.
8. With an explicit version only, validate both platforms' assets and publish
   the versioned GitHub Release. Empty input skips this job entirely.

Checksums identify downloads, not compression correctness. Correctness checks
compare recovered pixel samples/frames and reconstructed JPEG bytes. Smoke
checks use small synthetic images and low effort; they are not a new corpus,
compression-ratio or throughput benchmark.

## Local Windows Reproduction

Use an existing Python 3.10+ environment and Rust toolchain. The wheel test
helper creates and removes only its own temporary virtual environment and
downloads its Python dependencies with pip. It never installs into your base
Python environment or starts WSL.

The isolated test environment also installs `tomlkit` to read and vary temporary
packaging manifests. This is a test dependency, not a wheel runtime dependency.

```powershell
cargo fetch --locked
cargo build --release --locked --offline --workspace
cargo test --locked --offline --workspace --no-fail-fast -- --test-threads=2
python -m pip install "setuptools>=68" "wheel>=0.38" "tomlkit>=0.12,<1"
$env:RS_JXL_NATIVE_LIB = "$PWD/target/release/rs_jxl_ffi.dll"
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir out/ci/wheels
Remove-Item Env:RS_JXL_NATIVE_LIB
python -B -m unittest discover -s scripts -p "test_ci_*.py" -v
python -B scripts/test_ci_wheel.py --wheel-dir out/ci/wheels --cli target/release/rs-jxl.exe
python -B scripts/package_ci.py --platform windows-x86_64 --cli target/release/rs-jxl.exe --dist out/ci/wheels
```

Use a wheel directory containing exactly one wheel. Helpers reject ambiguous
directories instead of accidentally validating or publishing a stale version.
These local commands keep the project version. `prepare_ci_version.py` is
intended for a disposable CI checkout; running it locally with a nonempty
`RS_JXL_BUILD_VERSION` deliberately edits that checkout's version files.

## Evidence Boundary

Adding the workflow is not a successful hosted CI run. The earlier
[maintenance checkpoint](ARCHITECTURE.md#multiframe-maintenance-checkpoint)
records Windows validation before these CI helpers were introduced. Linux
runtime evidence must come from the hosted Linux job after this workflow is
pushed and run; no local WSL session is authorized by this addition.

The CI helpers were exercised locally on Windows on 2026-10-27: all five
archive tests passed, a real wheel and CLI ZIP/checksums were produced, and the
fresh-venv CLI/wheel smoke checks passed. The installed-wheel pytest run passed
235 tests with 12 optional cases skipped and two non-fatal tooling warnings.
After the manual-only/version-input correction, all 13 archive/version tests
passed. Actionlint 1.7.12 accepted the workflow (shellcheck/pyflakes were not installed
or run). This is local helper validation, not a hosted Windows or Linux run.

After adding conditional GitHub Release publication, all 27 archive/version/
release-assembly tests passed on Windows, including symlink rejection and Linux
multi-tag wheel filename fixtures. Actionlint 1.7.12 accepted the updated workflow
with shellcheck/pyflakes disabled. No GitHub Release was created and no hosted
workflow was triggered during this local verification.

On 2026-10-28 the packaging test's fixed `0.1.0` expectation was replaced with
the input manifest's version. All 75 packaging tests passed, including wheel/
sdist roundtrips at the project version, `1.0.0` and `2.7.3`, plus mismatched
version rejection. The fresh-venv CI runner using the existing Windows Release
wheel/CLI passed its smoke checks and 237 Python tests (12 optional skips, two
tooling warnings); all 27 CI helper tests also passed. Native binaries were not
rebuilt, and hosted Linux/Windows runs still require manual dispatch.
