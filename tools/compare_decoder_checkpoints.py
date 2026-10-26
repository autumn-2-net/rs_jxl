"""Compare two Rust decoder checkpoints and official djxl on existing JXL files.

No encoding/builds. Originals are checked before timing; temporary decoded files
are removed even on failure. Requires Pillow and the existing PNG oracle deps.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

from PIL import Image

from benchmark_codec_subset import decode_command
from compare_cpp_port import (
    PNG_SIGNATURE, binary_info, check_process, clean_environment, compare_pixels,
    file_sha256, png_samples, run_process,
)

NAMES = ("before", "after", "official")
WARNINGS = [
    "CPU thread counts are requests, not equal scheduling semantics or CPU use. "
    "Official djxl reuses its runner across in-process repetitions; Rust checkpoints "
    "may recreate pools per decode API call. Pool/startup overhead is included.",
    "PNG timing uses float32 buffers to match djxl --disable_output, not PNG writing. "
    "Correctness separately checks original integer samples, including hidden RGB; "
    "this does not prove timed float buffers are identical. PNG metadata is not compared. "
    "The PNG oracle supports single-frame 8/16-bit inputs (8-bit palettes included).",
    "JPEG timing is exact-byte reconstruction, not IDCT rendering or JPEG encoding. "
    "Wall/reps includes amortized process startup, input IO and allocation. "
    "Official binary provenance is caller-attested, not certified by its version/hash.",
]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "official-djxl"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--case", type=Path, nargs=2, action="append", required=True,
                        metavar=("JXL", "ORIGINAL"))
    result.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--decode-reps", type=int, default=10)
    result.add_argument("--timeout", type=float, default=300)
    result.add_argument("--out", type=Path, required=True, help="NEW output directory")
    return result


def validate_args(args):
    if (args.repeats < 1 or args.warmups < 0 or args.decode_reps < 1
            or any(not 1 <= t <= 256 for t in args.threads)
            or len(set(args.threads)) != len(args.threads)
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("Invalid measurement settings")


def load_case(index, jxl, source):
    jxl, source = jxl.resolve(strict=True), source.resolve(strict=True)
    original = source.read_bytes()
    expected = None
    if original.startswith(PNG_SIGNATURE):
        expected = png_samples(original)
        kind, width, height, depth = "png", expected.width, expected.height, expected.depth
    else:
        with Image.open(BytesIO(original)) as image:
            if image.format != "JPEG" or getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Original must be a single-frame PNG or JPEG: {source}")
            kind, width, height, depth = "jpeg", image.width, image.height, None
    size = jxl.stat().st_size
    if size <= 0 or width * height <= 0:
        raise ValueError("Empty JXL or invalid original dimensions")
    case = dict(index=index, label=source.name, jxl=str(jxl), original=str(source),
                kind=kind, width=width, height=height, depth=depth, pixels=width * height,
                jxl_bytes=size, bpp=8 * size / (width * height),
                jxl_sha256=file_sha256(jxl), original_bytes=len(original),
                original_sha256=hashlib.sha256(original).hexdigest())
    return case, original, expected


def trial(report, command, env, timeout, *, output=None, original=None, expected=None, **labels):
    row = dict(command=command, status="running", phase="verify" if output else "decode", **labels)
    report["trials"].append(row)
    try:
        if output is not None:
            output.unlink(missing_ok=True)
        row.update(run_process(command, env, timeout))
        check_process(row)
        if output is not None:
            restored = output.read_bytes()
            if expected is not None:
                compare_pixels(expected, png_samples(restored))
            elif restored != original:
                raise ValueError("Reconstructed JPEG bytes differ from original")
            row["verified"] = True
        elif not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0:
            raise ValueError("Invalid subprocess wall time")
        row["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return row


def summarize(case, threads, rows):
    decoders = {}
    for name in NAMES:
        seconds = [r["wall_seconds"] / r["reps"] for r in rows
                   if r["phase"] == "decode" and r["case_index"] == case["index"]
                   and r["threads"] == threads and r["decoder"] == name
                   and not r["warmup"] and r["status"] == "passed"]
        median = statistics.median(seconds)
        decoders[name] = dict(median_seconds=median, min_seconds=min(seconds),
                              max_seconds=max(seconds), trials=len(seconds),
                              mpix_s=case["pixels"] / 1e6 / median)
    return dict(case_index=case["index"], label=case["label"], threads=threads,
                bpp=case["bpp"], decoders=decoders,
                before_over_after_speedup=decoders["before"]["median_seconds"]
                / decoders["after"]["median_seconds"])


def markdown(report):
    config = report["config"]
    lines = ["# Decoder Checkpoints", "", f"Status: **{report['status']}**", "",
             f"Serial rotating before/after/official order; {config['warmups']} excluded warmup(s), "
             f"{config['repeats']} measured rounds, {config['decode_reps']} in-process decodes/run.",
             "Same pre-encoded input for all decoders. Every case/decoder/thread pair is "
             "verified before timing that case; failures stop the run without skipping.",
             "Times are median wall/reps; speedup = before/after (>1 is faster after). "
             "bpp = 8 * JXL bytes / original pixels. Timed commands write no decoded files.", ""]
    lines += ["- " + warning for warning in report["warnings"]]
    lines += ["", "| Case | Threads | bpp | Before ms | After ms | Official ms | "
              "Before MPix/s | After MPix/s | Official MPix/s | Speedup |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for summary in report["summaries"]:
        decoders = summary["decoders"]
        values = [f"{decoders[n]['median_seconds'] * 1000:.3f}" for n in NAMES]
        values += [f"{decoders[n]['mpix_s']:.3f}" for n in NAMES]
        label = summary["label"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {summary['case_index']}: {label} | {summary['threads']} | "
                     f"{summary['bpp']:.4f} | " + " | ".join(values)
                     + f" | {summary['before_over_after_speedup']:.3f}x |")
    lines += ["", "## Binaries", ""]
    for name, info in report["binaries"].items():
        lines.append(f"- {name}: `{info['path']}`; SHA256 `{info['sha256']}`")
    commands = dict.fromkeys(tuple(row["command"]) for row in report["trials"] if row["phase"] == "decode")
    if commands:
        lines += ["", "## Timed Commands", "", "Raw argv, deduplicated across rounds:", "", "```text"]
        lines += [json.dumps(command) for command in commands]
        lines.append("```")
    lines += ["", "All raw commands, version probes, input hashes and every trial are in `report.json`.",
              "On failure, any partial timing results are not a valid completed comparison."]
    if report["errors"]:
        lines += ["", "## Failure", "", "```text", *report["errors"], "```"]
    return "\n".join(lines) + "\n"


def benchmark(args):
    validate_args(args)
    args.out.mkdir(parents=False, exist_ok=False)
    env, environment = clean_environment(os.environ)
    report = dict(status="running", created=datetime.now(timezone.utc).isoformat(),
                  config=vars(args).copy(), environment=environment, warnings=list(WARNINGS),
                  binaries={}, cases=[], trials=[], summaries=[], errors=[],
                  temporary_outputs_cleaned=False)
    try:
        binaries = dict(before=args.before, after=args.after, official=args.official_djxl)
        for name, path in binaries.items():
            info = binary_info(path, env, min(args.timeout, 10))
            report["binaries"][name] = info
            check_process(info["version_probe"])
            binaries[name] = Path(info["path"])
        with tempfile.TemporaryDirectory(prefix="verify-", dir=args.out) as temp:
            for index, (jxl, source) in enumerate(args.case):
                case, original, expected = load_case(index, jxl, source)
                report["cases"].append(case)
                output = Path(temp) / ("restored.png" if case["kind"] == "png" else "restored.jpg")
                for threads in args.threads:
                    for name in NAMES:
                        command = decode_command("official" if name == "official" else "rust",
                            binaries[name], case["jxl"], kind=case["kind"], threads=threads,
                            output=output, depth=case["depth"])
                        trial(report, command, env, args.timeout, output=output,
                              original=original, expected=expected, case_index=index,
                              decoder=name, threads=threads, reps=1)
                output.unlink()
                for thread_index, threads in enumerate(args.threads):
                    for repeat in range(-args.warmups, args.repeats):
                        start = (index + thread_index + repeat) % len(NAMES)
                        for name in NAMES[start:] + NAMES[:start]:
                            command = decode_command("official" if name == "official" else "rust",
                                binaries[name], case["jxl"], kind=case["kind"], threads=threads,
                                reps=args.decode_reps)
                            trial(report, command, env, args.timeout, case_index=index,
                                  decoder=name, threads=threads, repeat=repeat,
                                  warmup=repeat < 0, reps=args.decode_reps)
                    report["summaries"].append(summarize(case, threads, report["trials"]))
                print(f"[{index + 1}/{len(args.case)}] verified and timed: {case['label']}",
                      flush=True)
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed")
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        report["temporary_outputs_cleaned"] = not any(args.out.glob("verify-*"))
        # Checkpoint files can be rebuilt by another worker while this harness runs.
        identities = [(info["path"], info["sha256"]) for info in report["binaries"].values()]
        identities += [(case[key], case[key + "_sha256"]) for case in report["cases"]
                       for key in ("jxl", "original")]
        for path, digest in identities:
            try:
                if file_sha256(Path(path)) != digest:
                    raise ValueError(f"File changed during run; measurements invalid: {path}")
            except OSError as error:
                report["errors"].append(f"Identity check failed: {path}: {error}")
            except ValueError as error:
                report["errors"].append(str(error))
        if not report["temporary_outputs_cleaned"]:
            report["errors"].append("Temporary output cleanup incomplete")
        if report["errors"]:
            report["status"] = "failed"
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        (args.out / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    return report


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        report = benchmark(args)
    except (OSError, ValueError) as error:
        cli.error(str(error))
    print(f"{report['status']}: {args.out / 'report.json'}")
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
