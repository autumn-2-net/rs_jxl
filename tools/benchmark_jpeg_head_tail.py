"""Fixed JPEG head/tail experiment; no builds, discovery or pixel fallbacks.

Native seconds cover read/parse+encode, excluding disk IO and reconstruction.
Process timeout is only a harness guard. The caller supplies official djxl;
its hash identifies the binary, not its provenance. Fixture results are not a
representative sampling claim. All encoded/reconstructed images are temporary.
Manifest format: {"cases": [{"original": "relative/or/absolute.jpg"}]}.
"""

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

from PIL import Image

from benchmark_fast_context import load_cases
from benchmark_ma_models import trial_order
import compare_cpp_port as port


MODES = ("baseline", "head-tail")


def number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def invalid_json_constant(value):
    raise ValueError(f"Nonfinite JSON constant in encoder receipt: {value}")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True, help="Prebuilt jpeg_entropy_lab")
    p.add_argument("--decoder", type=Path, required=True, help="Caller-attested official djxl")
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--case", type=Path, action="append", default=[])
    sources.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, required=True, help="New JSON report; no overwrite")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--effort", type=int, choices=range(1, 12), default=9)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300, help="Per-process guard only")
    args = p.parse_args(argv)
    if not 1 <= args.threads <= 256 or args.repeats < 1 or not number(args.timeout, positive=True):
        p.error("Invalid threads/repeats/timeout")
    if args.out.exists():
        p.error("Report exists; it will not be overwritten")
    return args


def command(args, source, output, mode):
    if mode not in MODES:
        raise ValueError("Unknown head/tail mode")
    result = [str(args.encoder), str(source), "--effort", str(args.effort),
              "--threads", str(args.threads), "--output", str(output)]
    if mode == "head-tail":
        result.append("--head-tail")
    return result


def validate_measurement(m, mode, size, width, height):
    if mode not in MODES or not isinstance(m, dict):
        raise ValueError("Invalid mode/measurement")
    if m.get("native_verified") is not True:
        raise ValueError("Native original-JPEG reconstruction was not verified")
    for key, expected in (("bytes", size), ("width", width), ("height", height)):
        if type(m.get(key)) is not int or m[key] <= 0 or m[key] != expected:
            raise ValueError(f"Invalid/mismatched {key}")
    if not number(m.get("seconds"), positive=True):
        raise ValueError("Invalid native encoder seconds")
    pixels = width * height
    for key, expected in (("bpp", 8 * size / pixels),
                          ("mpix_per_second", pixels / 1e6 / m["seconds"])):
        if not number(m.get(key), positive=True) or not math.isclose(m[key], expected, rel_tol=1e-5):
            raise ValueError(f"Invalid/inconsistent {key}")
    if "head_tail" not in m:
        raise ValueError("Missing head_tail activation receipt")
    work = m["head_tail"]
    if mode == "baseline":
        if work is not None:
            raise ValueError("Baseline unexpectedly enabled head/tail")
        return
    if not isinstance(work, dict):
        raise ValueError("Missing head/tail work receipt")
    for key in ("attempted", "candidate", "selected"):
        if type(work.get(key)) is not bool:
            raise ValueError(f"Invalid head/tail flag: {key}")
    for key in ("histograms", "changed_histograms", "configurations", "baseline_bytes"):
        if type(work.get(key)) is not int or work[key] < 0:
            raise ValueError(f"Invalid head/tail counter: {key}")
    if work["baseline_bytes"] <= 0 or work["changed_histograms"] > work["histograms"]:
        raise ValueError("Inconsistent histogram/baseline counters")
    for key in ("search_seconds", "write_seconds"):
        if not number(work.get(key)) or work[key] > m["seconds"] + 1e-6:
            raise ValueError(f"Invalid head/tail duration: {key}")
    if "candidate_bytes" not in work:
        raise ValueError("Missing candidate_bytes")
    candidate_size = work["candidate_bytes"]
    if work["candidate"]:
        if (not work["attempted"] or type(candidate_size) is not int or candidate_size <= 0
                or work["configurations"] == 0):
            raise ValueError("Candidate lacks attempted work/size")
    elif candidate_size is not None or work["selected"]:
        raise ValueError("Absent candidate has size or selection")
    if not work["attempted"] and (work["changed_histograms"] or work["configurations"]):
        raise ValueError("Inactive attempt has changed histograms/configurations")
    expected_size = candidate_size if work["selected"] else work["baseline_bytes"]
    if size != expected_size or size > work["baseline_bytes"]:
        raise ValueError("Selected output disagrees with receipt or exceeds baseline")
    if work["candidate"] and size != min(work["baseline_bytes"], candidate_size):
        raise ValueError("Head/tail failed to return the smaller output")


def validate_pair(measurements):
    baseline, head_tail = (measurements[mode] for mode in MODES)
    if (head_tail["bytes"] > baseline["bytes"]
            or head_tail["head_tail"]["baseline_bytes"] != baseline["bytes"]):
        raise ValueError("Same-trial head/tail size/receipt differs from standalone baseline")


def summarize(samples, pixels, original_bytes):
    sizes = [sample["measurement"]["bytes"] for sample in samples]
    seconds = statistics.median(sample["measurement"]["seconds"] for sample in samples)
    size = statistics.median(sizes)
    return dict(bytes=size, min_bytes=min(sizes), max_bytes=max(sizes), seconds=seconds,
                bpp=8 * size / pixels, original_jpeg_ratio=size / original_bytes,
                mpix_per_second=pixels / 1e6 / seconds)


def aggregate(cases):
    pixels = sum(case["pixels"] for case in cases)
    original = sum(case["original_bytes"] for case in cases)
    result = {}
    for mode in MODES:
        size = sum(case["summaries"][mode]["bytes"] for case in cases)
        seconds = sum(case["summaries"][mode]["seconds"] for case in cases)
        result[mode] = dict(bytes=size, seconds=seconds, bpp=8 * size / pixels,
                            original_jpeg_ratio=size / original,
                            mpix_per_second=pixels / 1e6 / seconds)
    return dict(cases=len(cases), pixels=pixels, original_bytes=original, modes=result)


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(), cases=[], errors=[],
                  timing="Native read/parse+encode only; excludes disk IO and reconstruction",
                  statistics="Per-image medians; aggregate sums per-image medians, not trial totals",
                  correctness="Native receipt plus official djxl; complete original JPEG bytes",
                  selection="Explicit fixed matrix; no representative sampling claim",
                  decoder_provenance="Caller-attested official djxl; path and SHA256 recorded")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            for name in ("encoder", "decoder"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                setattr(args, name, path)
                digest = port.file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                path = args.manifest.resolve(strict=True)
                digest = port.file_sha256(path)
                identities.append((path, digest))
                report["manifest"] = dict(path=str(path), sha256=digest)
            sources = load_cases(args)
            with tempfile.TemporaryDirectory(prefix="jpeg-head-tail-") as scratch:
                temporary = Path(scratch)
                for index, source in enumerate(sources):
                    if not source.is_file():
                        raise ValueError(f"Input is not a file: {source}")
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    identities.append((source, digest))
                    with Image.open(BytesIO(original)) as image:
                        if image.format != "JPEG":
                            raise ValueError(f"Not JPEG: {source}")
                        width, height = image.size
                    snapshot = temporary / "original.jpg"
                    snapshot.write_bytes(original)
                    case = dict(source=str(source), sha256=digest, original_bytes=len(original),
                                width=width, height=height, pixels=width * height,
                                variants={mode: [] for mode in MODES})
                    report["cases"].append(case)
                    for repetition in range(args.repeats):
                        measurements = {}
                        for mode in trial_order(MODES, repetition, index):
                            sample = dict(repetition=repetition, verified=False)
                            case["variants"][mode].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                jxl, jpeg = Path(trial) / "encoded.jxl", Path(trial) / "restored.jpg"
                                sample["encoder"] = port.run_process(
                                    command(args, snapshot, jxl, mode), env, args.timeout)
                                port.check_process(sample["encoder"])
                                m = json.loads(sample["encoder"]["stdout"],
                                               parse_constant=invalid_json_constant)
                                validate_measurement(m, mode, jxl.stat().st_size, width, height)
                                sample["measurement"] = m
                                sample["actual_bytes"] = jxl.stat().st_size
                                sample["decoder"] = port.run_process(
                                    [str(args.decoder), str(jxl), str(jpeg),
                                     "--num_threads=1", "--quiet"], env, args.timeout)
                                port.check_process(sample["decoder"])
                                if jpeg.read_bytes() != original:
                                    raise ValueError(f"Official original-JPEG byte mismatch: {source}")
                                sample["verified"] = True
                                measurements[mode] = m
                        validate_pair(measurements)
                    case["summaries"] = {mode: summarize(case["variants"][mode], case["pixels"],
                                                         len(original)) for mode in MODES}
                    print(f"{source.name}: {json.dumps(case['summaries'])}", flush=True)
                report["aggregate"] = aggregate(report["cases"])
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if port.file_sha256(path) == digest:
                        continue
                    error = f"Input/binary changed during measurement: {path}"
                except OSError as failure:
                    error = str(failure)
                report["errors"].append(error)
            if report["errors"] or not report["temporary_outputs_cleaned"]:
                report["status"] = "failed"
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    return report


def main(argv=None):
    report = benchmark(arguments(argv))
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
