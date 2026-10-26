"""Offline, sequential platform validation with explicit evidence and deadlines.

Probe mode never compiles. Check mode is NOT runtime acceptance. Test mode builds
and runs on one host. Binaries mode runs prebuilt native test binaries and needs
no local Rust toolchain; cross-built binaries executed on Linux are Linux runtime
evidence. Reports distinguish build and runtime hosts. Requires Python 3.10+.
Windows probing never invokes WSL, not even distribution enumeration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time


LOSSLESS_TESTS = (
    "decode", "decode_backend_layers", "decode_jpeg", "decode_jpeg_capture",
    "decode_lossless_edges", "decode_wide_integer", "decode_streaming",
    "decode_hostile", "jpeg_dc_refinement", "multiframe_auto",
)
RECORDED_ENVIRONMENT = (
    "RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_TARGET_DIR", "CARGO_PROFILE_RELEASE_LTO", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
)


def decode_output(data: bytes) -> str:
    # WSL status queries use UTF-16LE on Windows; compiler output is UTF-8.
    if data.startswith(b"\xff\xfe") or (data and data[:512].count(b"\0") > len(data[:512]) // 5):
        return data.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    return data.decode("utf-8", errors="replace")


def tail(path: Path, maximum: int = 12000) -> str:
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        source.seek(max(0, source.tell() - maximum))
        return decode_output(source.read())


def kill_tree(process: subprocess.Popen) -> dict:
    """Terminate descendants too; a hung compiler must not outlive the report."""
    if os.name == "nt":
        command = ["taskkill", "/PID", str(process.pid), "/T", "/F"]
        try:
            killed = subprocess.run(command, capture_output=True, timeout=15, check=False)
            result = {"argv": command, "returncode": killed.returncode,
                      "stdout": decode_output(killed.stdout), "stderr": decode_output(killed.stderr)}
            if killed.returncode and process.poll() is None:
                process.kill()
                result["fallback"] = "direct process kill; descendant cleanup NOT confirmed"
        except (OSError, subprocess.TimeoutExpired) as error:
            process.kill()
            result = {"argv": command, "error": str(error),
                      "fallback": "direct process kill; descendant cleanup NOT confirmed"}
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            result = {"method": "killpg", "signal": "SIGKILL"}
        except ProcessLookupError:
            result = {"method": "killpg", "already_exited": True}
    # Never leave a child running silently or block forever during cleanup.
    try:
        result["reaped_returncode"] = process.wait(timeout=10)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("child could not be reaped after forced termination") from error
    return result


def run_command(argv: list[str], cwd: Path, out: Path, label: str, timeout: float) -> dict:
    stdout_path = out / f"{label}.stdout.log"
    stderr_path = out / f"{label}.stderr.log"
    row = {"label": label, "argv": argv, "cwd": str(cwd), "timeout_seconds": timeout,
           "stdout_log": stdout_path.name, "stderr_log": stderr_path.name,
           "started_utc": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
              if os.name == "nt" else {"start_new_session": True})
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, **kwargs)
        except OSError as error:
            row.update(status="start_failed", error=str(error), returncode=None)
        else:
            try:
                code = process.wait(timeout=timeout)
                row.update(status="passed" if code == 0 else "failed", returncode=code)
            except subprocess.TimeoutExpired:
                row.update(status="timed_out", cleanup=kill_tree(process), returncode=process.returncode)
            except KeyboardInterrupt:
                row.update(status="interrupted", cleanup=kill_tree(process), returncode=process.returncode)
    row.update(seconds=time.monotonic() - started, stdout_tail=tail(stdout_path), stderr_tail=tail(stderr_path))
    return row


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def native_binary_kind(path: Path) -> str:
    """Reject foreign executable formats, including Windows interop inside WSL."""
    with path.open("rb") as source:
        header = source.read(64)
        system, machine = platform.system(), platform.machine().lower()
        if system == "Linux":
            expected = {"amd64": 62, "x86_64": 62, "arm64": 183, "aarch64": 183}.get(machine)
            if (len(header) >= 20 and header[:6] == b"\x7fELF\x02\x01"
                    and int.from_bytes(header[18:20], "little") == expected):
                return "ELF64"
        elif system == "Darwin":
            expected = {"x86_64": 0x01000007, "arm64": 0x0100000c}.get(machine)
            if (len(header) >= 8 and header[:4] == b"\xcf\xfa\xed\xfe"
                    and int.from_bytes(header[4:8], "little") == expected):
                return "Mach-O64"
        elif system == "Windows" and len(header) >= 64 and header[:2] == b"MZ":
            source.seek(int.from_bytes(header[60:64], "little"))
            coff = source.read(6)
            expected = {"amd64": 0x8664, "x86_64": 0x8664, "arm64": 0xaa64}.get(machine)
            if (coff[:4] == b"PE\0\0" and int.from_bytes(coff[4:6], "little") == expected):
                return "PE"
    raise ValueError(f"not a supported native {system}/{machine} executable: {path}")


def validation_commands(cargo: list[str], mode: str, suite: str, target: str | None) -> list[list[str]]:
    if mode in ("probe", "binaries"):
        return []
    if mode == "check":
        return [[*cargo, "check", "--locked", "--offline", "--workspace", "--all-targets",
                 *(["--target", target] if target else [])]]
    test = [*cargo, "test", "--locked", "--offline", "--release"]
    if suite == "workspace":
        return [[*test, "--workspace", "--", "--test-threads=1"]]
    selected = [argument for name in LOSSLESS_TESTS for argument in ("--test", name)]
    return [
        [*test, "-p", "rs-jxl", *selected, "--", "--test-threads=1"],
        [*test, "-p", "rs-jxl-cli", "--", "--test-threads=1"],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "check", "test", "binaries"), default="probe")
    parser.add_argument("--suite", choices=("lossless", "workspace"), default="lossless")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, help="new report directory, normally under ignored out/")
    parser.add_argument("--toolchain", help="already installed rustup toolchain; never installs it")
    parser.add_argument("--target", help="cross-compilation target, permitted ONLY with --mode check")
    parser.add_argument("--test-binary", type=Path, action="append", default=[], help="prebuilt native libtest executable; repeat in binaries mode")
    parser.add_argument("--build-report", type=Path, help="optional build-side report to attach in binaries mode")
    parser.add_argument("--timeout", type=float, default=600, help="hard deadline per validation command, seconds")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.target and args.mode != "check":
        parser.error("--target is check-only; execute cross-built tests with --mode binaries on their runtime OS")
    if bool(args.test_binary) != (args.mode == "binaries"):
        parser.error("--mode binaries requires at least one --test-binary, and only that mode accepts binaries")
    if args.build_report and args.mode != "binaries":
        parser.error("--build-report is only used with --mode binaries")
    repo = args.repo.resolve()
    if not (repo / "Cargo.toml").is_file() or not (repo / "Cargo.lock").is_file():
        parser.error("--repo must contain Cargo.toml and Cargo.lock")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.out or repo / "out" / "platform-validation" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "report.json"
    prefix = ["rustup", "run", args.toolchain] if args.toolchain else []
    cargo = [*prefix, "cargo"]
    commands = validation_commands(cargo, args.mode, args.suite, args.target)
    build_evidence = None
    if args.build_report:
        build_evidence = json.loads(args.build_report.read_text(encoding="utf-8"))
        if not isinstance(build_evidence, dict):
            parser.error("--build-report must contain a JSON object")
    report = {
        "schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "suite": args.suite, "target": args.target,
        "validation_status": "not_run", "probe_status": "pending",
        "scope": "native runtime" if args.mode in ("test", "binaries") else "no runtime validation",
        "host": {"system": platform.system(), "release": platform.release(),
                 "version": platform.version(), "machine": platform.machine(),
                 "python": sys.version, "python_executable": sys.executable,
                 "cpu_count": os.cpu_count()},
        "script_sha256": sha256(Path(__file__)), "lockfile_sha256": sha256(repo / "Cargo.lock"),
        "build_host": None, "runtime_host": None,
        "build_evidence": ({"path": str(args.build_report.resolve()), "sha256": sha256(args.build_report),
                            "report": build_evidence} if build_evidence is not None else None),
        # An allowlist avoids leaking registry credentials, CI secrets, etc.
        "relevant_environment": {key: os.environ[key] for key in RECORDED_ENVIRONMENT if key in os.environ},
        "executables": {name: shutil.which(name) for name in ("rustc", "cargo", "rustup", "git", "docker", "podman")},
        "planned_commands": commands, "probes": [], "runs": [],
        "limitations": [
            "No packages installed, targets added, CI changed or machines provisioned.",
            "Cross-compilation checks alone are not runtime acceptance; actually executing a cross-built native binary is.",
            "Supplied build evidence is recorded, not a cryptographically verified association with test binaries.",
            "Ignored external-corpus tests are not included; no official decoder oracle is run here.",
            "No sanitizer, exhaustive fuzzing or process RSS bound is provided by this runner.",
            "Test commands stop at first failure/timeout; later commands are not passed by implication.",
        ],
    }

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    save()
    probes = [
        ("rustc", [*prefix, "rustc", "-vV"], args.mode != "binaries"),
        ("cargo", [*cargo, "-Vv"], args.mode != "binaries"),
        ("git-head", ["git", "rev-parse", "HEAD"], False),
        ("git-status", ["git", "status", "--short", "--branch"], False),
    ]
    if shutil.which("rustup"):
        probes.extend([
            ("rustup-toolchains", ["rustup", "toolchain", "list"], False),
            ("rustup-targets", ["rustup", "target", "list", "--installed",
                                *(["--toolchain", args.toolchain] if args.toolchain else [])], False),
        ])
    # Stay on this host. Do not enumerate WSL distributions, connect to Docker
    # daemons, launch VMs, or treat another environment's presence as acceptance.
    required_ok = True
    for label, command, required in probes:
        row = run_command(command, repo, out, label, 20)
        row["required"] = required
        report["probes"].append(row)
        required_ok &= not required or row["status"] == "passed"
        save()
        print(f"probe {label}: {row['status']}", flush=True)
        if row["status"] == "interrupted":
            report["probe_status"] = "interrupted"
            save()
            return 130
    report["probe_status"] = "passed" if required_ok else "incomplete"
    rust = report["probes"][0]
    compiler_host = next((line.partition(": ")[2] for line in rust["stdout_tail"].splitlines()
                          if line.startswith("host: ")), None)
    report["compiler_host"] = compiler_host
    if args.mode == "binaries":
        report["runtime_host"] = report["host"]
        report["build_host"] = (build_evidence.get("build_host") or build_evidence.get("host")) if build_evidence else None
        report["build_compiler_host"] = build_evidence.get("compiler_host") if build_evidence else None
        report["prebuilt_binaries"] = []
        for item in args.test_binary:
            path = item.resolve()
            try:
                kind = native_binary_kind(path)
                digest = sha256(path)
            except (OSError, ValueError) as error:
                report.update(validation_status="blocked", error=str(error))
                save()
                print(str(error), file=sys.stderr)
                return 1
            report["prebuilt_binaries"].append({"path": str(path), "sha256": digest, "format": kind})
            commands.append([str(path), "--test-threads=1"])
        report["planned_commands"] = commands
        save()
    else:
        report["build_host"] = report["host"]
        if args.mode == "test":
            report["runtime_host"] = report["host"]
    host_system = {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}.get(platform.system())
    host_machine = {"AMD64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(platform.machine())
    if args.mode == "test" and (
        not compiler_host or not host_system or not host_machine
        or host_system not in compiler_host or not compiler_host.startswith(host_machine + "-")
        or os.environ.get("CARGO_BUILD_TARGET")
    ):
        report["validation_status"] = "blocked"
        report["error"] = "test mode builds and runs locally; for cross-built executables use binaries mode on their runtime OS"
        save()
        print(report["error"], file=sys.stderr)
        return 1
    if args.mode not in ("probe", "binaries") and not required_ok:
        report["validation_status"] = "blocked"
        save()
        print(f"Rust toolchain unavailable; see {report_path}", file=sys.stderr)
        return 1
    if args.mode == "test":
        # Do not let a repository/user Cargo build.target setting silently turn
        # a native runtime claim into a cross-target or configured-runner check.
        for command in commands:
            command[command.index("--"):command.index("--")] = ["--target", compiler_host]
        report["runtime_target"] = compiler_host
        save()
    for number, command in enumerate(commands):
        print(json.dumps({"run": number, "argv": command}), flush=True)
        row = run_command(command, repo, out, f"run-{number:02d}", args.timeout)
        if args.mode == "binaries" and row["status"] == "passed":
            counts = re.findall(r"(?m)^test result: ok\. (\d+) passed; 0 failed;", row["stdout_tail"])
            if not counts or int(counts[-1]) == 0:
                row.update(status="failed", error="binary exited successfully without a nonempty passing libtest summary")
            else:
                row["passed_tests"] = int(counts[-1])
        report["runs"].append(row)
        report["validation_status"] = "running" if row["status"] == "passed" else row["status"]
        save()
        print(f"{row['status']} ({row['seconds']:.3f}s)", flush=True)
        if row["status"] != "passed":
            if row.get("error"):
                print(row["error"], file=sys.stderr)
            print(row["stderr_tail"], file=sys.stderr)
            print(f"Report: {report_path}")
            return 130 if row["status"] == "interrupted" else 1
    report["validation_status"] = {"probe": "not_run", "check": "compile_check_passed", "test": "tests_passed", "binaries": "tests_passed"}[args.mode]
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"{report['validation_status']}; report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
