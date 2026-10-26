"""Compare native integer pixel decode checkpoints on existing PNG/JXL pairs.

Unlike djxl --disable_output (float32), this measures Rust U8/U16 output.
No encoding. Original PNG samples, including invisible RGB, must match exactly.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import tempfile

from benchmark_codec_subset import decode_command
from compare_decoder_checkpoints import load_case, trial, validate_args
from compare_cpp_port import binary_info, check_process, clean_environment, file_sha256


def decode_seconds(stderr, repetitions):
    matches = re.findall(r"^(\d+) repetition\(s\) in (\d+\.\d+) s\.$", stderr, re.MULTILINE)
    if len(matches) != 1 or int(matches[0][0]) != repetitions:
        raise ValueError("Missing or inconsistent decoder timing")
    seconds = float(matches[0][1])
    if seconds <= 0:
        raise ValueError("Nonpositive decoder timing")
    return seconds


def benchmark(args):
    validate_args(args)
    args.out.mkdir(exist_ok=False)
    env, environment = clean_environment(os.environ)
    report = dict(status="running", config=vars(args), environment=environment,
                  created=datetime.now(timezone.utc).isoformat(), binaries={},
                  measurement="CLI internal decode timer, excludes startup and input IO; includes context creation",
                  cases=[], trials=[], summaries=[], errors=[])
    try:
        binaries = {"before": args.before, "after": args.after}
        for name, path in binaries.items():
            info = binary_info(path, env, min(args.timeout, 10))
            check_process(info["version_probe"])
            report["binaries"][name] = info
            binaries[name] = Path(info["path"])
        with tempfile.TemporaryDirectory(prefix="verify-", dir=args.out) as tmp:
            output = Path(tmp) / "restored.png"
            for index, (jxl, original_path) in enumerate(args.case):
                case, original, expected = load_case(index, jxl, original_path)
                if case["kind"] != "png":
                    raise ValueError("This benchmark requires PNG originals")
                report["cases"].append(case)
                bits = 8 if case["depth"] <= 8 else 16
                for threads in args.threads:
                    for name, binary in binaries.items():
                        command = decode_command("rust", binary, case["jxl"], kind="png",
                                                 threads=threads, output=output, depth=bits)
                        trial(report, command, env, args.timeout, output=output,
                              original=original, expected=expected, case_index=index,
                              decoder=name, threads=threads, reps=1)
                    for repeat in range(-args.warmups, args.repeats):
                        names = list(binaries)
                        if (repeat + index + threads) % 2:
                            names.reverse()
                        for name in names:
                            command = [str(binaries[name]), "decode", case["jxl"],
                                       "--verbose", "--disable_output", f"--bits={bits}",
                                       f"--num_threads={threads}",
                                       f"--num_reps={args.decode_reps}"]
                            row = trial(report, command, env, args.timeout, case_index=index,
                                        decoder=name, threads=threads, repeat=repeat,
                                        warmup=repeat < 0, reps=args.decode_reps)
                            row["decode_seconds"] = decode_seconds(row["stderr"], args.decode_reps)
                    decoders = {}
                    for name in binaries:
                        times = [r["decode_seconds"] / r["reps"] for r in report["trials"]
                                 if r["phase"] == "decode" and r["case_index"] == index
                                 and r["threads"] == threads and r["decoder"] == name
                                 and not r["warmup"]]
                        sec = statistics.median(times)
                        decoders[name] = dict(seconds=sec, mpix_s=case["pixels"] / 1e6 / sec,
                                              min_seconds=min(times), max_seconds=max(times))
                    row = dict(label=case["label"], threads=threads, bits=bits,
                               bpp=case["bpp"], decoders=decoders,
                               speedup=decoders["before"]["seconds"] / decoders["after"]["seconds"])
                    report["summaries"].append(row)
                    print(f"{case['label']} {bits}bit {threads}T: "
                          f"{decoders['before']['seconds'] * 1000:.3f} -> "
                          f"{decoders['after']['seconds'] * 1000:.3f} ms, "
                          f"{row['speedup']:.3f}x", flush=True)
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        identities = [(v["path"], v["sha256"]) for v in report["binaries"].values()]
        identities += [(c[k], c[k + "_sha256"]) for c in report["cases"] for k in ("jxl", "original")]
        for path, digest in identities:
            try:
                if file_sha256(Path(path)) != digest:
                    report["errors"].append(f"Input changed during measurement: {path}")
            except OSError as error:
                report["errors"].append(str(error))
        if report["errors"]:
            report["status"] = "failed"
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n",
                                              encoding="utf-8")
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--before", type=Path, required=True)
    cli.add_argument("--after", type=Path, required=True)
    cli.add_argument("--case", type=Path, nargs=2, action="append", required=True)
    cli.add_argument("--threads", type=int, nargs="+", default=[1, 8, 16])
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--warmups", type=int, default=1)
    cli.add_argument("--decode-reps", type=int, default=10)
    cli.add_argument("--timeout", type=float, default=300)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    report = benchmark(args)
    print(report["status"], args.out / "report.json", *report["errors"], sep="\n")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
