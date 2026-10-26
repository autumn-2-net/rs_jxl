"""Profile JPEG-transcoded JXL -> F32 pixels; never reconstruct JPEG scans.

Requires a decoder-profile build and an independently checked pixel report.
Counters add overhead. Use ordinary Release builds for throughput comparisons.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

from benchmark_jpeg_pixel_decode import command, differences
from compare_cpp_port import check_process, clean_environment, file_sha256, run_process
from profile_decoder_stages import parse_profile
from verify_decoder_features import portable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--case-label", action="append")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("reps must be positive")
    old = json.loads(args.baseline.read_text(encoding="utf-8"))
    if old["status"] not in ("completed_exact", "completed_with_pixel_differences"):
        parser.error("A completed pixel baseline is required")
    cases = old["results"]
    if args.case_label:
        wanted = set(args.case_label)
        if wanted - {r["case"]["label"] for r in cases}:
            parser.error("Unknown case label")
        cases = [r for r in cases if r["case"]["label"] in wanted]
    binary = args.binary.resolve(strict=True)
    baseline_binary = old["binaries"]["rust"]
    if file_sha256(Path(baseline_binary["path"])) != baseline_binary["sha256"]:
        raise ValueError("Changed baseline binary")
    args.out.mkdir(parents=True, exist_ok=False)
    env, environment = clean_environment(os.environ)
    report = dict(status="running", binary=str(binary), sha256=file_sha256(binary),
                  baseline=str(args.baseline), environment=environment, cases=[])
    try:
        with tempfile.TemporaryDirectory(prefix="verify-", dir=args.out) as temp:
            for row in cases:
                case = row["case"]
                source = Path(case["jxl"])
                if file_sha256(source) != case["jxl_sha256"]:
                    raise ValueError("Changed input")
                arrays = []
                for i, exe in enumerate((binary, Path(baseline_binary["path"]))):
                    path = Path(temp) / f"{i}.pfm"
                    run = run_process(command("rust", exe, source, 1, output=path, bits=32), env, 120)
                    check_process(run)
                    arrays.append(portable(path))
                    path.unlink()
                comparison = differences(*arrays)
                del arrays
                if not comparison["exact"]:
                    raise ValueError("Instrumentation changed F32 output")
                run = run_process(command("rust", binary, source, 1, bits=32, reps=args.reps),
                                  dict(env, RS_JXL_PROFILE="1"), 120)
                check_process(run)
                stages = parse_profile(run["stderr"])
                report["cases"].append(dict(case=case, f32_exact=True, run=run, parsed=stages))
                print(case["label"], flush=True)
                for name, value in stages["scopes"].items():
                    if name.startswith("vardct.") or name in ("decode.pixels.total", "decode.pixels.copy_frame"):
                        print(f"  {name}: {value['seconds'] / args.reps * 1000:.3f} ms/decode; calls={value['calls']}", flush=True)
                if file_sha256(source) != case["jxl_sha256"]:
                    raise ValueError("Changed input during profile")
        if file_sha256(binary) != report["sha256"]:
            raise ValueError("Changed binary during profile")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
