"""Independent CLI acceptance; development-only numpy dependency.

No private paths are embedded. Use repeated --jxl/--jpeg/--djxl arguments.
Float comparison defaults to exact words. A nonzero --atol is explicit and is
reported, for independently implemented lossy rendering with rounding variation.
JPEG reconstruction ALWAYS compares original bytes exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def run(command: list[str], timeout: float) -> float:
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-6000:])
    return time.perf_counter() - started


def stack(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim == 3:
        values = values[None]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rs-jxl", type=Path, required=True)
    parser.add_argument("--djxl", type=Path, action="append", required=True)
    parser.add_argument("--cjxl", type=Path, action="append", default=[])
    parser.add_argument("--jxl", type=Path, action="append", default=[])
    parser.add_argument("--jpeg", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--allow-implicit-level10", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.jxl and not args.jpeg:
        parser.error("supply at least one --jxl or --jpeg fixture")
    if not 1 <= args.threads <= 256 or args.timeout <= 0 or args.atol < 0:
        parser.error("invalid threads, timeout or tolerance")
    rows: list[dict] = []
    rust = str(args.rs_jxl.resolve())
    compatibility = ["--allow-implicit-level10"] if args.allow_implicit_level10 else []
    with tempfile.TemporaryDirectory(prefix="rs-jxl-decoder-") as temporary:
        root = Path(temporary)
        for number, source in enumerate(args.jxl):
            row = {"kind": "pixels", "file": source.name, "atol": args.atol,
                "allow_implicit_level10": args.allow_implicit_level10}
            try:
                own = root / f"{number}-rust.npy"
                row["rust_seconds"] = run([rust, "decode", str(source.resolve()), str(own),
                    "--bits", "32", "--threads", str(args.threads), *compatibility], args.timeout)
                actual = stack(own)
                row["shape"] = list(actual.shape)
                comparisons = []
                row["comparisons"] = comparisons
                for ref_index, decoder in enumerate(args.djxl):
                    reference = root / f"{number}-{ref_index}-reference.npy"
                    seconds = run([str(decoder.resolve()), str(source.resolve()), str(reference)], args.timeout)
                    expected = stack(reference)
                    if actual.shape != expected.shape or actual.dtype != expected.dtype:
                        raise AssertionError(f"shape/type differ: {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
                    exact = actual.view(np.uint32) == expected.view(np.uint32)
                    mismatch = int(np.count_nonzero(~exact))
                    maximum = 0.0
                    rms = 0.0
                    nonfinite_mismatch = 0
                    if mismatch:
                        finite = np.isfinite(actual) & np.isfinite(expected)
                        nonfinite_mismatch = int(np.count_nonzero(~exact & ~finite))
                        if np.any(finite):
                            difference = actual[finite].astype(np.float64) - expected[finite]
                            maximum = float(np.max(np.abs(difference)))
                            rms = float(np.sqrt(np.mean(difference * difference)))
                    passed = not mismatch or (args.atol > 0 and not nonfinite_mismatch and maximum <= args.atol)
                    comparisons.append({"decoder": decoder.name, "decoder_index": ref_index, "seconds": seconds,
                        "different_words": mismatch, "max_abs_error": maximum, "rms_error": rms,
                        "nonfinite_mismatches": nonfinite_mismatch, "passed": passed})
                row["passed"] = all(comparison["passed"] for comparison in comparisons)
                if not row["passed"]:
                    row["error"] = "pixel comparison failed; see per-decoder measurements"
            except (OSError, RuntimeError, AssertionError, subprocess.TimeoutExpired) as error:
                row.update(passed=False, error=str(error))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)

        for number, source in enumerate(args.jpeg):
            original = source.read_bytes()
            for encoder_index, encoder in enumerate([None, *args.cjxl]):
                row = {"kind": "jpeg_reconstruction", "file": source.name,
                    "encoder": "rs-jxl" if encoder is None else encoder.name,
                    "encoder_index": encoder_index}
                try:
                    encoded = root / f"jpeg-{number}-{encoder_index}.jxl"
                    restored = root / f"jpeg-{number}-{encoder_index}.jpg"
                    command = ([rust, "transcode-jpeg", str(source.resolve()), str(encoded), "--effort", "3"]
                        if encoder is None else [str(encoder.resolve()), str(source.resolve()), str(encoded),
                            "--lossless_jpeg=1", "-e", "3"])
                    row["encode_seconds"] = run(command, args.timeout)
                    row["rust_seconds"] = run([rust, "decode", str(encoded), str(restored),
                        "--threads", str(args.threads), *compatibility], args.timeout)
                    if restored.read_bytes() != original:
                        raise AssertionError("Rust reconstruction differs from original JPEG")
                    row["rust_exact"] = True
                    references = []
                    for ref_index, decoder in enumerate(args.djxl):
                        reference = root / f"jpeg-{number}-{encoder_index}-{ref_index}.jpg"
                        run([str(decoder.resolve()), str(encoded), str(reference)], args.timeout)
                        references.append({"decoder_index": ref_index,
                            "exact": reference.read_bytes() == original})
                    row["references"] = references
                    if not all(reference["exact"] for reference in references):
                        raise AssertionError("an official reconstruction differs from original JPEG; see references")
                    row.update(passed=True, jpeg_bytes=len(original), jxl_bytes=encoded.stat().st_size)
                except (OSError, RuntimeError, AssertionError, subprocess.TimeoutExpired) as error:
                    row.update(passed=False, error=str(error))
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
    report = {"passed": sum(bool(row["passed"]) for row in rows), "total": len(rows), "cases": rows}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT: {report['passed']}/{report['total']} cases passed")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
