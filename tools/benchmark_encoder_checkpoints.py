"""Serial, lossless-verified comparison of two frozen Rust encoder executables.

PNG: --source-effort 9 --source-search --threads 8, using the existing PNG
oracle (numpy/OpenCV), including invisible RGB and original sample precision.
JPEG: original-byte transcode-jpeg --effort 9 --source-search --threads 8;
Pillow reads dimensions only, never supplies pixels to the encoder.
Official djxl verifies every output, including warmups, outside encode timing.
Its provenance is caller-attested, not certified by a version string or hash.

No builds, downloads, corpus discovery or subprocesses on import/help. Only
report.json and REPORT.md persist in the NEW --out directory. Timeout is a
per-process harness guard: a timeout fails the comparison, never a successful
bounded search. Coordinate with other workers before starting this harness.

Experimental arguments are literal argv tokens, not shell text; for example
--after-arg=--source-scout applies ONLY to PNG. --before-arg/--after-arg alias
--before-png-arg/--after-png-arg; JPEG options use --before-jpeg-arg/--after-jpeg-arg.
"""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile

import compare_cpp_port as port


NAMES = ("before", "after")
THREADS = 8
EFFORT = 9
RESERVED_ARGS = {
    "--threads", "--source-effort", "--effort", "--source-search",
    "--source-sample-limit", "--strip-metadata", "--force", "--help", "--version",
}


def parser():
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--before", "--baseline", type=Path, required=True)
    result.add_argument("--after", type=Path, required=True)
    result.add_argument("--official-djxl", type=Path, required=True)
    result.add_argument("--out", "--out-dir", type=Path, required=True,
                        help="NEW directory for reports and automatically cleaned scratch files")
    result.add_argument("--case", type=Path, action="append", required=True,
                        help="Original PNG or JPEG, detected by contents; repeat for each input")
    for name in NAMES:
        result.add_argument(f"--{name}-arg", f"--{name}-png-arg", action="append",
                            default=[], metavar="TOKEN", help="PNG-only experimental encode argument")
        result.add_argument(f"--{name}-jpeg-arg", action="append", default=[], metavar="TOKEN",
                            help="JPEG-only experimental transcode argument")
        result.add_argument(f"--{name}-env", action="append", default=[], metavar="RS_JXL_NAME=value",
                            help="Explicit encoder-only environment override")
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--threads", type=int, default=THREADS,
                        help="Same encoder worker limit for both checkpoints (1..256)")
    result.add_argument("--source-sample-limit", type=int, default=None,
                        help="Optional positive PNG MA sample cap; omitted uses each CLI's default")
    result.add_argument("--timeout", type=float, default=300,
                        help="Positive per-subprocess guard in seconds; timeout means failure")
    return result


def validate_extra_args(tokens):
    may_take_value = False
    for token in tokens:
        if not token or "\0" in token:
            raise ValueError("Experimental arguments must be nonempty literal argv tokens")
        if token.startswith("--"):
            name, separator, _ = token.partition("=")
            if name in RESERVED_ARGS or not re.fullmatch(r"--[a-zA-Z][a-zA-Z0-9-]*", name):
                raise ValueError(f"Experimental argument cannot override the harness: {token}")
            may_take_value = not separator
        elif may_take_value and not token.startswith("-"):
            may_take_value = False
        else:
            raise ValueError(f"Expected a long experimental option or its value: {token}")


def validate_args(args):
    if not 1 <= args.threads <= 256:
        raise ValueError("Threads must be in 1..256")
    if (args.repeats < 1 or args.warmups < 0
            or (args.source_sample_limit is not None and args.source_sample_limit < 1)
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("Repeats and explicit sample limit must be positive, warmups nonnegative, timeout finite and positive")
    for name in NAMES:
        validate_extra_args(getattr(args, name + "_arg"))
        validate_extra_args(getattr(args, name + "_jpeg_arg"))
        port.clean_environment({}, getattr(args, name + "_env"))


def jpeg_dimensions(original):
    from PIL import JpegImagePlugin

    with JpegImagePlugin.JpegImageFile(BytesIO(original)) as image:
        if image.format != "JPEG" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Expected a single-frame JPEG")
        return image.width, image.height


def load_case(index, path):
    path = path.resolve(strict=True)
    original = path.read_bytes()
    if not original:
        raise ValueError(f"Empty input: {path}")
    case = dict(index=index, path=str(path), label=path.name,
                original_bytes=len(original), original_sha256=port.sha256(original))
    if original.startswith(port.PNG_SIGNATURE):
        payload, info = port.png_payload(original)
        expected = port.png_samples(original)
        port.compare_pixels(expected, port.png_samples(payload))
        case.update(info, kind="png", depth=expected.depth)
    elif original.startswith(b"\xff\xd8\xff"):
        width, height = jpeg_dimensions(original)
        payload, expected = original, None
        case.update(kind="jpeg", width=width, height=height, depth=None)
    else:
        raise ValueError(f"Only original PNG/JPEG inputs are accepted: {path}")
    if case["width"] <= 0 or case["height"] <= 0:
        raise ValueError("Invalid original dimensions")
    case.update(pixels=case["width"] * case["height"],
                payload_bytes=len(payload), payload_sha256=port.sha256(payload))
    return case, payload, original, expected


def encoder_command(binary, source, output, kind, sample_limit=None, extra=(), threads=THREADS):
    if kind == "png":
        command = port.encoder_command("rust", binary, source, output, threads, EFFORT,
                                       "full-scheduler", sample_limit)
    elif kind == "jpeg":
        command = [str(binary), "transcode-jpeg", str(source), str(output),
                   "--effort", str(EFFORT), "--source-search", "--threads", str(threads)]
    else:
        raise ValueError(f"Unknown input kind: {kind}")
    return command + list(extra)


def search_diagnostics(kind, process):
    lines = [line for line in (process["stdout"] + "\n" + process["stderr"]).splitlines()
             if any(word in line.lower() for word in ("source", "search", "scout", "pilot"))]
    return dict(requested=True, lines=lines, diagnostics_observed=bool(lines),
                evidence=("PNG source-search evidence required on success" if kind == "png" else
                          "JPEG --source-search requested; this CLI does not guarantee search diagnostics"))


def measure(row, binary, decoder, case, source, original, expected, directory, env, args, decoder_env=None):
    output = directory / "encoded.jxl"
    try:
        command = encoder_command(binary, source, output, case["kind"],
                                  args.source_sample_limit, getattr(args, row["encoder"] +
                                      ("_arg" if case["kind"] == "png" else "_jpeg_arg")), args.threads)
        process = port.run_process(command, env, args.timeout)
        row["process"] = process
        row["source_search"] = search_diagnostics(case["kind"], process)
        port.check_process(process)
        size, seconds = output.stat().st_size, process["wall_seconds"]
        if size <= 0 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Empty encoder output or invalid subprocess wall time")
        pixels = case["pixels"]
        row.update(bytes=size, wall_seconds=seconds, bpp=8 * size / pixels,
                   mpix=pixels / 1e6, mpix_per_second=pixels / 1e6 / seconds)
        if case["kind"] == "png":
            row["source_search"]["evidence"] = port.scheduler_evidence(
                "rust", "full-scheduler", process["stderr"])
            restored = directory / "restored.png"
            decode = port.decoder_command(decoder, output, restored, expected.depth)
        else:
            restored = directory / "restored.jpg"
            decode = [str(decoder), str(output), str(restored),
                      "--reconstruct_jpeg", "--num_threads=1", "--quiet"]
        row["decode"] = port.run_process(decode, decoder_env if decoder_env is not None else env, args.timeout)
        port.check_process(row["decode"])
        if case["kind"] == "png":
            port.compare_pixels(expected, port.png_samples(restored.read_bytes()))
            row["pixels_match_including_hidden_rgb"] = True
        else:
            if restored.read_bytes() != original:
                raise ValueError("Reconstructed JPEG bytes differ from original")
            row["original_jpeg_bytes_match"] = True
        row["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def summarize(cases, rows, repeats, warmups, errors):
    summaries = []
    required = sorted((name, repeat, repeat < 0)
                      for repeat in range(-warmups, repeats) for name in NAMES)
    for case in cases:
        group = [row for row in rows if row["case_index"] == case["index"]]
        observed = sorted((row["encoder"], row["repeat"], row["warmup"]) for row in group)
        complete = observed == required and all(row["status"] == "passed" for row in group)
        item = dict(case_index=case["index"], label=case["label"], kind=case["kind"],
                    status="passed" if complete and not errors else "failed")
        if item["status"] == "passed":
            for name in NAMES:
                selected = [row for row in group if row["encoder"] == name and not row["warmup"]]
                seconds = statistics.median(row["wall_seconds"] for row in selected)
                sizes = [row["bytes"] for row in selected]
                item[name] = dict(trials=len(selected), median_wall_seconds=seconds,
                                  min_wall_seconds=min(row["wall_seconds"] for row in selected),
                                  max_wall_seconds=max(row["wall_seconds"] for row in selected),
                                  median_bytes=statistics.median(sizes), min_bytes=min(sizes),
                                  max_bytes=max(sizes),
                                  median_bpp=statistics.median(row["bpp"] for row in selected),
                                  mpix_per_second=case["pixels"] / 1e6 / seconds)
            item["before_over_after_wall_speedup"] = (
                item["before"]["median_wall_seconds"] / item["after"]["median_wall_seconds"])
            item["after_over_before_bytes"] = item["after"]["median_bytes"] / item["before"]["median_bytes"]
        summaries.append(item)
    return summaries


def markdown(report):
    lines = ["# Encoder Checkpoints", "", f"Status: **{report['status']}**", "",
             report["timing"], report["validation"], report["search_contract"],
             report["timeout_contract"], report["official_decoder_provenance"],
             f"Temporary outputs cleaned: {report['temporary_outputs_cleaned']}.", "",
             f"PNG/JPEG: effort 9, source search, {report['config']['threads']} threads per encoder.",
             "Warmups are validated but excluded from medians. Speedup = before wall / after wall.",
             "A passed comparison means verified outputs and complete measurements, not a speed or size improvement.",
             "", "| Case | Kind | Encoder | Median bytes | Median bpp | Median wall s | MPix/s |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for item in report["summaries"]:
        label = item["label"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        for name in NAMES:
            metrics = item.get(name)
            cells = (f"{metrics['median_bytes']:g} | {metrics['median_bpp']:.6f} | "
                     f"{metrics['median_wall_seconds']:.6f} | {metrics['mpix_per_second']:.3f}"
                     if metrics else "FAIL | - | - | -")
            lines.append(f"| {item['case_index']}: {label} | {item['kind']} | {name} | {cells} |")
    lines += ["", "## Binaries", ""]
    for name, info in report["binaries"].items():
        lines.append(f"- {name}: `{info['path']}`; SHA256 `{info['sha256']}`")
    lines += ["", "Exact argv, binary/version identities, source-search diagnostics, all process logs,",
              "input hashes, settings and per-image speedup/size ratios are in `report.json`.",
              "Incomplete or failed comparisons have no valid medians; raw trials remain for diagnosis."]
    if report["errors"]:
        lines += ["", "## Errors", "", "```text", *report["errors"], "```"]
    return "\n".join(lines) + "\n"


def benchmark(args):
    validate_args(args)
    out = args.out.resolve()
    out.mkdir(parents=False, exist_ok=False)
    env, environment = port.clean_environment(os.environ)
    encoder_envs = {}
    environments = {"decoder": environment}
    for name in NAMES:
        encoder_envs[name], environments[name] = port.clean_environment(os.environ, getattr(args, name + "_env"))
    report = dict(schema_version=1, status="failed", created_utc=datetime.now(timezone.utc).isoformat(),
                  config=dict(threads=args.threads, effort=EFFORT, source_search=True,
                              source_sample_limit=args.source_sample_limit, repeats=args.repeats,
                              warmups=args.warmups, timeout=args.timeout,
                              encode_args={name: dict(png=getattr(args, name + "_arg"),
                                                      jpeg=getattr(args, name + "_jpeg_arg"))
                                           for name in NAMES},
                              requested_cases=[str(path.resolve()) for path in args.case]),
                  invocation=getattr(args, "invocation", None), environment=environment, environments=environments,
                  timing="Serial CLI wall time includes startup, input/output IO and diagnostics; excludes independent decoding and validation. Before/after order rotates every pair, including warmups.",
                  validation="PNG uses identical metadata-free snapshots and original sample equality including invisible RGB at original depth. JPEG uses unchanged original bytes and official djxl reconstruction, never pixel fallback.",
                  search_contract="Search is requested for both checkpoints; this does not imply identical candidate policies or work. PNG requires source-search summary evidence; JPEG diagnostics may be absent.",
                  timeout_contract="Timeout is a harness guard, not an encoder budget; every timeout fails the comparison.",
                  official_decoder_provenance="Official djxl origin is caller-attested; version and SHA256 are recorded, not proof of provenance.",
                  binaries={}, cases=[], rows=[], summaries=[], errors=[],
                  temporary_outputs_cleaned=False)
    scratch = None
    try:
        binaries = dict(before=args.before, after=args.after, official_djxl=args.official_djxl)
        for name, path in binaries.items():
            info = port.binary_info(path, encoder_envs.get(name, env), min(args.timeout, 10))
            report["binaries"][name] = info
            port.check_process(info["version_probe"])
            binaries[name] = Path(info["path"])
        with tempfile.TemporaryDirectory(prefix="encoder-checkpoints-", dir=out) as temp:
            scratch = Path(temp)
            pair_index = 0
            for index, path in enumerate(args.case):
                case, payload, original, expected = load_case(index, path)
                report["cases"].append(case)
                with tempfile.TemporaryDirectory(prefix=f"case-{index}-", dir=scratch) as case_temp:
                    source = Path(case_temp) / ("input.png" if case["kind"] == "png" else "input.jpg")
                    source.write_bytes(payload)
                    del payload
                    for repeat in range(-args.warmups, args.repeats):
                        order = NAMES if pair_index % 2 == 0 else NAMES[::-1]
                        pair_index += 1
                        for order_index, name in enumerate(order):
                            row = dict(status="failed", case_index=index, encoder=name, repeat=repeat,
                                       warmup=repeat < 0, pair_index=pair_index - 1, order_index=order_index)
                            report["rows"].append(row)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=case_temp) as trial:
                                measure(row, binaries[name], binaries["official_djxl"], case, source,
                                        original, expected, Path(trial), encoder_envs[name], args, decoder_env=env)
                            print(f"{case['label']} {name} repeat={repeat}: {row['bytes']} B "
                                  f"{row['bpp']:.6f} bpp {row['wall_seconds']:.6f}s "
                                  f"{row['mpix_per_second']:.3f} MPix/s", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        report["temporary_outputs_cleaned"] = scratch is None or not scratch.exists()
        if not report["temporary_outputs_cleaned"]:
            report["errors"].append("Temporary output cleanup incomplete")
        identities = [(name, info, info["sha256"]) for name, info in report["binaries"].items()]
        identities += [(case["path"], case, case["original_sha256"]) for case in report["cases"]]
        for name, info, digest in identities:
            try:
                info["unchanged_during_run"] = port.file_sha256(Path(info["path"])) == digest
            except OSError as error:
                info["unchanged_during_run"] = False
                report["errors"].append(f"Identity check failed: {name}: {error}")
            if not info["unchanged_during_run"]:
                report["errors"].append(f"File changed during run: {name}; measurements are invalid")
        expected_rows = len(args.case) * len(NAMES) * (args.warmups + args.repeats)
        if len(report["cases"]) != len(args.case) or len(report["rows"]) != expected_rows:
            report["errors"].append("Incomplete comparison: not all requested trials finished")
        report["summaries"] = summarize(report["cases"], report["rows"], args.repeats,
                                         args.warmups, report["errors"])
        if (not report["errors"] and report["summaries"]
                and all(item["status"] == "passed" for item in report["summaries"])):
            report["status"] = "passed"
        for name, content in (("report.json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
                              ("REPORT.md", markdown(report))):
            with (out / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    return report


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.invocation = [sys.executable, str(Path(__file__).resolve()),
                       *(sys.argv[1:] if argv is None else argv)]
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
