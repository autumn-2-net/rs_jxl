"""Compare two native libraries on identical in-memory lossless inputs.

Small synthetic tiled images isolate output/adapter overhead; these are not a
representative compression corpus. Native calls, JSON, allocation and NumPy
output are timed. Process startup, file I/O, fixture encoding and verification
are excluded. Libraries execute in separate children, in alternating order.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np


def fixtures():
    y, x = np.indices((1024, 1024), dtype=np.uint32)
    tile = (x // 32 + y // 32) % 4
    rgb = np.stack([(tile * 12301 + channel * 7001 + 17) % 65536
                    for channel in range(3)], axis=-1).astype(np.uint16)
    alpha = ((tile * 12347) % 65536).astype(np.uint16)[..., None]
    return [
        ("rgb8_tiles", (rgb % 256).astype(np.uint8)),
        ("rgb16_tiles", rgb),
        ("rgba16_tiles", np.concatenate((rgb, alpha), axis=-1)),
        ("rgb32_tiles", ((rgb % 4).astype(np.float32) + 1) * np.float32(0.25)),
    ]


def exact(actual, expected):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not np.array_equal(actual.view(np.uint8), expected.view(np.uint8))):
        raise AssertionError("decoded shape, dtype or original sample words differ")


def worker(task):
    os.environ["RS_JXL_NATIVE_LIB"] = task["native"]
    import rs_jxl

    directory = Path(task["directory"])
    if task.get("prepare"):
        records = []
        for label, expected in fixtures():
            encoded = rs_jxl.encode(expected, effort=3, threads=1)
            (directory / f"{label}.jxl").write_bytes(encoded)
            exact(rs_jxl.decode(encoded), expected)
            records.append({"label": label, "shape": list(expected.shape),
                            "dtype": str(expected.dtype), "jxl_bytes": len(encoded)})
        return records
    rows = []
    with rs_jxl.Decoder(threads=task["threads"]) as decoder:
        for label, expected in fixtures():
            encoded = (directory / f"{label}.jxl").read_bytes()
            destination = np.empty_like(expected)
            for operation in ("decode", "decode_into"):
                def execute():
                    if operation == "decode_into":
                        return decoder.decode_into(encoded, destination)
                    return decoder.decode(encoded)

                exact(execute(), expected)
                for _ in range(task["warmup"]):
                    execute()
                start = time.perf_counter()
                for _ in range(task["iterations"]):
                    execute()
                seconds = (time.perf_counter() - start) / task["iterations"]
                exact(execute(), expected)
                rows.append({"case": label, "operation": operation, "verified": True,
                             "seconds": seconds,
                             "mpix_s": expected.shape[0] * expected.shape[1] / seconds / 1e6})
    return rows


def call_worker(task):
    env = dict(os.environ)
    env.pop("RS_JXL_PROFILE", None)
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
                            input=json.dumps(task).encode("utf-8"), capture_output=True,
                            env=env, timeout=180, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.rounds, args.iterations, args.threads) < 1 or args.warmup < 0:
        parser.error("rounds, iterations and threads must be positive; warmup must be nonnegative")
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("output already exists; select a new report path")
    libraries = {label: path.resolve(strict=True)
                 for label, path in (("before", args.before), ("after", args.after))}
    report = {"status": "running", "threads": args.threads, "rounds": args.rounds,
              "iterations": args.iterations, "warmup": args.warmup, "results": [],
              "libraries": {label: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for label, path in libraries.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="rs-jxl-ffi-bench-") as directory:
            task = {"directory": directory, "threads": args.threads,
                    "iterations": args.iterations, "warmup": args.warmup}
            report["fixtures"] = call_worker({**task, "native": str(libraries["before"]), "prepare": True})
            for round_index in range(args.rounds):
                order = ("before", "after") if round_index % 2 == 0 else ("after", "before")
                for label in order:
                    rows = call_worker({**task, "native": str(libraries[label])})
                    report["results"].extend({**row, "round": round_index, "library": label} for row in rows)
                print(f"round {round_index + 1}/{args.rounds}: exact sample checks passed", flush=True)
        summary = []
        for case, operation in dict.fromkeys((row["case"], row["operation"]) for row in report["results"]):
            medians = {label: statistics.median(row["seconds"] for row in report["results"]
                                               if row["case"] == case and row["operation"] == operation
                                               and row["library"] == label) for label in libraries}
            ratio = medians["after"] / medians["before"]
            summary.append({"case": case, "operation": operation, **medians, "time_ratio": ratio})
            print(f"{case:14} {operation:11}: {medians['before'] * 1000:.3f} -> "
                  f"{medians['after'] * 1000:.3f} ms; time {ratio:.2%}", flush=True)
        report.update(status="passed", summary=summary)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin))), flush=True)
    else:
        main()
