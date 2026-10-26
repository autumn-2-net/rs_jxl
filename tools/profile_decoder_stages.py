"""Collect opt-in decoder stage scopes from existing, independently checked inputs.

Build --features decoder-profile; no system profiler or elevated tracing needed.
Normal throughput comes from a separate non-instrumented timing report, never
from these diagnostic runs. All temporary decoded outputs are removed.
"""

import argparse
import json
from pathlib import Path
import tempfile

from benchmark_codec_subset import decode_command
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, file_sha256, png_samples, run_process,
)


def parse_profile(text):
    result = dict(scopes={}, samples={}, timer_pair_floor_ns=None)
    backend = False
    for line in text.splitlines():
        if line.startswith("[decoder stages]"):
            backend = True
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) == 2 and fields[0] == "timer_pair_floor_ns":
            result["timer_pair_floor_ns"] = float(fields[1])
        if len(fields) != 4 or fields[0] == "stage":
            continue
        name, calls, seconds, last = fields
        row = dict(calls=int(calls), seconds=float(seconds))
        row["mean_ns" if backend else "longest_seconds"] = float(last)
        result["samples" if name.startswith("sample.") else "scopes"][name] = row
    if result["timer_pair_floor_ns"] is None:
        raise ValueError("Missing detailed diagnostics; build with decoder-profile")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True,
                        help="Existing uninstrumented, verified timing report")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1 or args.rounds < 1:
        parser.error("reps and rounds must be positive")
    import os
    env, environment = clean_environment(os.environ)
    baseline = json.loads(args.timings.read_text(encoding="utf-8"))
    binary = args.binary.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", binary=str(binary), sha256=file_sha256(binary),
                  timings=str(args.timings), environment=environment, cases=[])
    try:
        for old in baseline["cases"]:
            case = old["case"]
            print(f"Profile: {case['label']}", flush=True)
            jxl, source = Path(case["jxl"]), Path(case["original"])
            if (file_sha256(jxl) != case["jxl_sha256"]
                    or file_sha256(source) != case["original_sha256"]):
                raise ValueError("Input differs from the timing report")
            row = dict(case=case, normal_seconds=old["seconds_per_decode"],
                       normal_mpix_s=old["mpix_s"], profiles=[])
            report["cases"].append(row)
            with tempfile.TemporaryDirectory(prefix="verify-", dir=args.out) as temp:
                output = Path(temp) / ("decoded.jpg" if case["kind"] == "jpeg" else "decoded.png")
                verified = run_process(decode_command(
                    "rust", binary, jxl, kind=case["kind"], threads=1,
                    output=output, depth=case["depth"] or 8), env, 120)
                check_process(verified)
                original, decoded = source.read_bytes(), output.read_bytes()
                if case["kind"] == "jpeg":
                    if original != decoded:
                        raise ValueError("JPEG reconstruction mismatch")
                else:
                    compare_pixels(png_samples(original), png_samples(decoded))
            row["verified"] = True
            for _ in range(args.rounds):
                run = run_process(decode_command("rust", binary, jxl, kind=case["kind"],
                                  threads=1, reps=args.reps), dict(env, RS_JXL_PROFILE="1"), 120)
                check_process(run)
                row["profiles"].append(dict(raw=run, parsed=parse_profile(run["stderr"])))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
