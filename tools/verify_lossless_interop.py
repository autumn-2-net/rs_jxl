"""Opt-in bidirectional lossless audit against caller-supplied official tools.

Requires numpy, Pillow and an installed rs_jxl (or PYTHONPATH=python). No build,
download, benchmark, WSL or external codec use occurs in the library itself.
Each native task runs in a killable child. Official tools are independent
development oracles, not a fallback encoder/decoder. Failed samples are retained.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from lossless_interop_cases import exact_pixels, jpeg_cases, pfm_bytes, pixel_cases, png_bytes


def run(command, timeout, *, payload=None):
    start = time.perf_counter()
    completed = subprocess.run(command, input=payload, stdin=subprocess.DEVNULL if payload is None else None,
                               capture_output=True, timeout=timeout, check=False)
    text = completed.stdout.decode("utf-8", errors="replace")
    error = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"exit={completed.returncode}: {error[-6000:]} {text[-1000:]}")
    return time.perf_counter() - start, text, error


def native_worker(task):
    import rs_jxl

    source = Path(task["source"])
    expected = source.read_bytes() if task["kind"] == "jpeg" else np.load(source, allow_pickle=False)
    encoded_path = Path(task["encoded"])
    result = {"checks": []}
    if task["encode"]:
        start = time.perf_counter()
        options = dict(effort=task["effort"], threads=task["threads"], return_info=True, **task["options"])
        if task["kind"] == "jpeg":
            encoded, info = rs_jxl.transcode_jpeg(expected, **options)
        else:
            encoded, info = rs_jxl.encode(expected, **options)
        result.update(encode_seconds=time.perf_counter() - start, native_report=info)
        encoded_path.write_bytes(encoded)
    encoded = encoded_path.read_bytes()
    for threads in (1, 4):
        start = time.perf_counter()
        check = {"decoder": "rust", "threads": threads}
        try:
            if task["kind"] == "jpeg":
                actual = rs_jxl.reconstruct_jpeg(encoded, threads=threads)
                if actual != expected:
                    raise AssertionError(f"original JPEG bytes differ ({len(actual)} vs {len(expected)})")
            else:
                exact_pixels(rs_jxl.decode(encoded, threads=threads), expected)
            check["passed"] = True
        except Exception as error:
            check.update(passed=False, error=f"{type(error).__name__}: {error}")
        check["seconds"] = time.perf_counter() - start
        result["checks"].append(check)
    return result


def tool_record(encoder, decoder, timeout):
    tools = {}
    for role, path in (("encoder", encoder), ("decoder", decoder)):
        path = path.resolve(strict=True)
        _, out, err = run([str(path), "--version"], timeout)
        tools[role] = {"path": str(path), "version": (out + err).strip(),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return tools


def attempt(row, stage, operation):
    start = time.perf_counter()
    check = {"decoder": stage}
    try:
        operation()
        check["passed"] = True
    except Exception as error:
        check.update(passed=False, error=f"{type(error).__name__}: {error}")
    check["seconds"] = time.perf_counter() - start
    row["checks"].append(check)


def execute_job(job, oracles, root, failures, timeout):
    work = root / f"job-{job['index']:05d}"
    work.mkdir()
    encoded = work / "encoded.jxl"
    row = {key: value for key, value in job.items() if key not in ("source", "input")}
    row["checks"] = []
    start = time.perf_counter()
    try:
        task = {**job, "encoded": str(encoded), "encode": job["encoder"] == "rust"}
        if not task["encode"]:
            command = [oracles[job["encoder"]]["encoder"]["path"], job["input"], str(encoded),
                       "-d", "0", "-e", str(job["effort"]), f"--num_threads={job['threads']}"]
            command += ["--lossless_jpeg=1"] if job["kind"] == "jpeg" else ["--lossless_jpeg=0", "--keep_invisible=1"]
            if job["effort"] == 11:
                command.append("--allow_expert_options")
            command += job.get("official_options", [])
            row["encode_command"] = command
            row["stage"] = "official_encode"
            seconds, _, _ = run(command, timeout)
            row["encode_seconds"] = seconds
        row["stage"] = "native_encode_decode" if task["encode"] else "native_decode"
        _, output, _ = run([sys.executable, "-B", str(Path(__file__).resolve()), "--native"], timeout,
                           payload=json.dumps(task).encode("utf-8"))
        native = json.loads(output)
        if "error" in native:
            raise RuntimeError(native["error"])
        row["checks"].extend(native["checks"])
        if task["encode"]:
            row["encode_seconds"] = native["encode_seconds"]
            row["native_report"] = native["native_report"]
        row["encoded_bytes"] = encoded.stat().st_size
        row["stage"] = "official_decode"
        expected = (Path(job["source"]).read_bytes() if job["kind"] == "jpeg"
                    else np.load(job["source"], allow_pickle=False))
        if job["kind"] == "pixels":
            row["bpp"] = row["encoded_bytes"] * 8 / (expected.shape[0] * expected.shape[1])
        for label, pair in oracles.items():
            def verify(label=label, pair=pair):
                destination = work / (f"{label}.jpg" if job["kind"] == "jpeg" else f"{label}.npy")
                run([pair["decoder"]["path"], str(encoded), str(destination), "--num_threads=1"], timeout)
                if job["kind"] == "jpeg":
                    if destination.read_bytes() != expected:
                        raise AssertionError("original JPEG bytes differ")
                else:
                    exact_pixels(np.load(destination, allow_pickle=False), expected, normalized=True)
            attempt(row, label, verify)
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["seconds"] = time.perf_counter() - start
    row["passed"] = ("error" not in row and len(row["checks"]) == 2 + len(oracles)
                     and all(check["passed"] for check in row["checks"]))
    if row["passed"]:
        row["stage"] = "complete"
    if not row["passed"]:
        saved = failures / work.name
        saved.mkdir(parents=True, exist_ok=False)
        for path in work.iterdir():
            shutil.copy2(path, saved / path.name)
        shutil.copy2(job["source"], saved / Path(job["source"]).name)
        if job["source"] != job["input"]:
            shutil.copy2(job["input"], saved / Path(job["input"]).name)
        (saved / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        row["evidence"] = str(saved)
    return row


def fixtures(root, selected, include_jpeg):
    result = []
    for case in pixel_cases():
        if selected and case.name not in selected:
            continue
        source = root / f"{case.name}.npy"
        np.save(source, case.pixels, allow_pickle=False)
        floating = case.pixels.dtype == np.float32
        encoded_input = root / f"{case.name}.{'pfm' if floating else 'png'}"
        encoded_input.write_bytes(pfm_bytes(case.pixels) if floating else png_bytes(case.pixels))
        result.append({"case": case.name, "kind": "pixels", "source": str(source),
                       "input": str(encoded_input), "floating": floating})
    if include_jpeg:
        for name, data in jpeg_cases():
            if selected and name not in selected:
                continue
            source = root / f"{name}.jpg"
            source.write_bytes(data)
            result.append({"case": name, "kind": "jpeg", "source": str(source),
                           "input": str(source), "floating": False})
    if selected - {case["case"] for case in result}:
        raise ValueError(f"unknown/unselected fixture names: {selected - {case['case'] for case in result}}")
    return result


def jobs_for(cases, labels, efforts, tool_variants):
    jobs = []

    def add(case, encoder, effort, variant="default", options=None, official_options=None):
        jobs.append({**case, "index": len(jobs), "encoder": encoder, "effort": effort,
                     "threads": 1 if (len(jobs) // (len(labels) + 1)) % 2 == 0 else 4,
                     "variant": variant, "options": options or {},
                     "official_options": official_options or []})

    for case in cases:
        for effort in efforts:
            for label in ("rust", *labels):
                add(case, label, effort)
        if tool_variants and case["case"] == "rgba16_groups":
            for label in ("rust", *labels):
                add(case, label, 9, "weighted_small_groups",
                    {"group_size_shift": 0, "predictor": 6}, ["-g", "0", "-P", "6"])
        if tool_variants and case["case"] in ("rgba16_edges", "rgb8_tiles", "rgb32_gradient"):
            variants = [
                ("zero_no_rct", {"rct": 0, "predictor": 0, "palette_colors": 0},
                 ["-C", "0", "-P", "0", "--modular_palette_colors=0"]),
                ("weighted_small_groups", {"group_size_shift": 0, "predictor": 6}, ["-g", "0", "-P", "6"]),
                ("palette", {"palette_colors": 10000}, ["--modular_palette_colors=10000"]),
            ]
            if not case["floating"]:
                variants.append(("patches", {"patches": True}, ["--patches=1"]))
            for name, options, official_options in variants:
                for label in ("rust", *labels):
                    add(case, label, 9, name, options, official_options)
            if not case["floating"]:
                add(case, "rust", 9, "source_search", {"source_search": True})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cjxl", action="append", type=Path, required=True)
    parser.add_argument("--djxl", action="append", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--efforts", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--skip-jpeg", action="store_true")
    parser.add_argument("--skip-tool-variants", action="store_true",
                        help="skip the separate explicit-tool matrix (always effort 9)")
    args = parser.parse_args()
    if (len(args.cjxl) != len(args.djxl) or args.workers < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("supply paired cjxl/djxl paths and positive workers/timeout")
    if not args.efforts or any(effort not in range(1, 12) for effort in args.efforts):
        parser.error("efforts must be in 1..11")
    args.report = args.report.resolve()
    if args.report.exists():
        parser.error("report already exists; select a new evidence destination")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    failures = args.report.with_suffix("").with_name(args.report.stem + "-failures")
    if failures.exists():
        parser.error("failure evidence already exists; select a new report destination")
    oracles = {f"official-{i}": tool_record(encoder, decoder, args.timeout)
               for i, (encoder, decoder) in enumerate(zip(args.cjxl, args.djxl))}
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "oracles": oracles,
              "acceptance": "exact decoded samples/float words or original JPEG bytes; not encoded-byte identity",
              "efforts": args.efforts, "workers": args.workers, "timeout": args.timeout, "cases": []}
    native_library = os.environ.get("RS_JXL_NATIVE_LIB")
    if native_library:
        path = Path(native_library).resolve(strict=True)
        report["native_library"] = {"path": str(path),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rs-jxl-lossless-interop-") as directory:
        root = Path(directory)
        cases = fixtures(root, set(args.case), not args.skip_jpeg)
        if not cases:
            parser.error("no fixtures selected")
        jobs = jobs_for(cases, list(oracles), args.efforts, not args.skip_tool_variants)
        print(f"{len(cases)} fixtures, {len(jobs)} encode cases, {len(oracles)} independent decoders", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute_job, job, oracles, root, failures, args.timeout) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                report["cases"].append(row)
                failed = sum(not item["passed"] for item in report["cases"])
                if not row["passed"]:
                    print(f"FAIL {row['case']} {row['encoder']} e{row['effort']} {row['variant']}: {row.get('error', row['checks'])}", flush=True)
                if index % 25 == 0 or index == len(jobs):
                    print(f"[{index}/{len(jobs)}] failures={failed} elapsed={time.perf_counter() - started:.1f}s", flush=True)
                report["passed"] = index - failed
                report["completed"] = index
                args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["cases"].sort(key=lambda item: item["index"])
    report["seconds"] = time.perf_counter() - started
    report["checks_passed"] = sum(check["passed"] for row in report["cases"] for check in row["checks"])
    report["checks_total"] = sum(len(row["checks"]) for row in report["cases"])
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT {report['passed']}/{report['completed']} cases; {report['checks_passed']}/{report['checks_total']} decoder checks", flush=True)
    return 0 if report["passed"] == report["completed"] else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--native"]:
        try:
            result = native_worker(json.load(sys.stdin))
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(result), flush=True)
    else:
        raise SystemExit(main())
