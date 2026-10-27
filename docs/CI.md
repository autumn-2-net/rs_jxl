# CLI and Python Wheel CI

The [build workflow](../.github/workflows/build.yml) builds and tests **Windows
x86_64 and Linux x86_64 only**. It runs on pushes to `main`, pull requests and
manual **Actions -> Build CLI and Python wheels -> Run workflow** requests.
There are no macOS, ARM, WSL, CUDA or PyPI-publishing jobs. The workflow needs
only read access to repository contents and does not create releases or tags.

## Build Outputs

| Job | Build environment | Downloadable artifact |
| --- | --- | --- |
| `build-windows` | `windows-2022`, stable Rust/MSVC, Python 3.10 x64 | `rs-jxl-windows-x86_64`: `py3-none-win_amd64.whl`, CLI ZIP and `SHA256SUMS` |
| `build-linux` | `ubuntu-22.04` host with the PyPA `manylinux_2_28_x86_64` container, stable Rust, Python 3.10 | `rs-jxl-linux-x86_64`: auditwheel-repaired manylinux wheel, CLI tar.gz and `SHA256SUMS` |

Download these from the completed run's **Artifacts** section. Retention is
14 days. The wheel contains the Python package and its Rust native library;
the CLI archive contains `rs-jxl`/`rs-jxl.exe`, parameter documentation and
third-party notices. Linux archives retain the executable permission. Neither
artifact needs the source checkout or a runtime Rust compiler.

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

## Validation Gates

1. Fetch the exact `Cargo.lock` dependencies, then build the Release workspace
   offline. Only Cargo dependency downloads are cached, not native build outputs.
2. Run ordinary workspace tests with two test threads in Cargo's default test
   profile. Release CLI/library outputs are tested separately below. Explicitly
   ignored corpus/oracle/subprocess tests remain ignored; no private dataset is needed.
3. Build the wheel using the native library from that build, without a second
   implicit Cargo build or a different dependency resolution.
4. Test archive assembly, required notices, executable permissions and checksums.
5. Create a fresh temporary virtual environment, install the built wheel and
   test dependencies, and clear `PYTHONPATH`, `PYTHONHOME` and native-library
   overrides. Python runs with `-I` outside the checkout. Smoke checks verify
   that both the package and its native library belong to that installation.
6. Run CLI/Python pixel-lossless, JPEG-byte-reconstruction and automatic-frame
   smoke checks, then discover all `python/tests` against the installed wheel.
   Optional torch/CUDA and official `djxl` checks are not enabled in this CI.
7. Package and upload artifacts only after every required command succeeds.

Checksums identify downloads, not compression correctness. Correctness checks
compare recovered pixel samples/frames and reconstructed JPEG bytes. Smoke
checks use small synthetic images and low effort; they are not a new corpus,
compression-ratio or throughput benchmark.

## Local Windows Reproduction

Use an existing Python 3.10+ environment and Rust toolchain. The wheel test
helper creates and removes only its own temporary virtual environment and
downloads its Python dependencies with pip. It never installs into your base
Python environment or starts WSL.

```powershell
cargo fetch --locked
cargo build --release --locked --offline --workspace
cargo test --locked --offline --workspace --no-fail-fast -- --test-threads=2
python -m pip install "setuptools>=68" "wheel>=0.38"
$env:RS_JXL_NATIVE_LIB = "$PWD/target/release/rs_jxl_ffi.dll"
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir out/ci/wheels
Remove-Item Env:RS_JXL_NATIVE_LIB
python -B -m unittest discover -s scripts -p test_ci_artifacts.py -v
python -B scripts/test_ci_wheel.py --wheel-dir out/ci/wheels --cli target/release/rs-jxl.exe
python -B scripts/package_ci.py --platform windows-x86_64 --cli target/release/rs-jxl.exe --dist out/ci/wheels
```

Use a wheel directory containing exactly one wheel. Helpers reject ambiguous
directories instead of accidentally validating or publishing a stale version.

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
Actionlint 1.7.12 accepted the workflow (shellcheck/pyflakes were not installed
or run). This is local helper validation, not a hosted Windows or Linux run.
