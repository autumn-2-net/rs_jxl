"""Bounded, serial source-tool experiments using caller-attested official djxl.

--mode is repeatable; by default run the five cumulative stages. Only reports
persist: input snapshots, JXL files and decoded PNGs live in cleaned scratch.
baseline disables patch gather; patch enables only patch gather; sparse adds
Sparse LZ quick; prev1 adds one previous channel; screened adds palette screening.
The *-only profiles isolate a single tool with patch gather disabled.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile

import compare_cpp_port as port
from corpus_inputs import prepare


MODES = ("baseline", "patch", "sparse", "prev1", "screened",
         "sparse-only", "prev1-only", "screened-only")
DEFAULT_MODES = MODES[:5]


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("encoder", "decoder", "manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", action="append", choices=MODES,
                        help="Repeat for each profile; default: baseline,patch,sparse,prev1,screened")
    parser.add_argument("--effort", type=int, choices=(9,), default=9)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--force-source", action="store_true",
                        help="Targeted patches only: use direct candidate source, not normal corpus")
    args = parser.parse_args(argv)
    args.mode = list(DEFAULT_MODES) if args.mode is None else args.mode
    if len(set(args.mode)) != len(args.mode) or "baseline" not in args.mode:
        parser.error("Require distinct modes including baseline for size comparisons")
    if (not 1 <= args.repeats <= 100 or not math.isfinite(args.timeout)
            or args.timeout <= 0 or len(set(args.threads)) != len(args.threads)
            or not all(1 <= value <= 256 for value in args.threads)):
        parser.error("Require repeats in 1..100, positive finite timeout, distinct threads in 1..256")
    if args.out.exists():
        parser.error("Output must be a NEW directory; refusing overwrite")
    return args


def trial_order(modes, image_index, repeat):
    shift = (image_index + repeat) % len(modes)
    return modes[shift:] + modes[:shift]


def command(args, snapshot, output, threads, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown source-tools mode: {mode}")
    return [str(args.encoder), str(snapshot), "--output", str(output),
            "--effort", str(args.effort), "--threads", str(threads),
            "--direct-candidate", "source" if args.force_source else "baseline",
            "--direct-source-tools", mode]


def validate_receipt(row, args, mode, threads, size, expected):
    tools = row.get("source_tools")
    if not isinstance(tools, dict) or tools.get("profile") != mode:
        raise ValueError("Missing or mismatched source_tools profile")
    if not {"patch_parallel_gather", "previous_channels", "lz_quick", "palette_order"} <= tools.keys():
        raise ValueError("Incomplete source_tools receipt")
    patch = mode in ("patch", "sparse", "prev1", "screened")
    previous = int(mode in ("prev1", "screened", "prev1-only"))
    quick = "Sparse" if mode in ("sparse", "prev1", "screened", "sparse-only") else "Off"
    palette = "ScreenedRuns" if mode in ("screened", "screened-only") else "Default"
    if (tools["patch_parallel_gather"] is not patch
            or type(tools["previous_channels"]) is not int
            # Gray inputs have no previous channel. The requested budget is
            # independently recorded so a silently ignored switch still fails.
            or tools.get("max_previous_channels") != previous
            or not 0 <= tools["previous_channels"] <= previous
            or tools["lz_quick"] != quick or tools["palette_order"] != palette):
        raise ValueError("Source-tools switches do not match requested mode")
    if (row.get("direct") is not True or type(row.get("encode_calls")) is not int
            or row["encode_calls"] != 1
            or row.get("candidate") != ("source" if args.force_source else "baseline")
            or row.get("effort") != args.effort or row.get("threads") != threads
            or row.get("bytes") != size or size <= 0
            or (row.get("width"), row.get("height")) != (expected.width, expected.height)
            or not isinstance(row.get("seconds"), (int, float))
            or isinstance(row["seconds"], bool)
            or not math.isfinite(row["seconds"]) or row["seconds"] <= 0):
        raise ValueError("Invalid direct receipt: require exactly one native encode")


def measure(args, snapshot, expected, directory, env, row):
    """Retain logs and errors; even an invalid receipt's output is decoded."""
    errors = []
    with tempfile.TemporaryDirectory(prefix="trial-", dir=directory) as tmp:
        output, decoded = Path(tmp) / "out.jxl", Path(tmp) / "out.png"
        try:
            row["encoder_process"] = port.run_process(
                command(args, snapshot, output, row["threads"], row["mode"]), env, args.timeout)
            port.check_process(row["encoder_process"])
            receipt = json.loads(row["encoder_process"]["stdout"], parse_constant=reject_constant)
            if not isinstance(receipt, dict):
                raise ValueError("Native receipt must be an object")
            row["receipt"] = receipt
            validate_receipt(receipt, args, row["mode"], row["threads"], output.stat().st_size, expected)
            for key in ("width", "height", "bytes", "seconds", "encode_calls", "source_tools"):
                row[key] = receipt[key]
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
            errors.append(f"encode/receipt: {error}")
        if output.exists():
            try:
                row["decoder_process"] = port.run_process(
                    port.decoder_command(args.decoder, output, decoded, expected.depth), env, args.timeout)
                port.check_process(row["decoder_process"])
                port.compare_pixels(expected, port.png_samples(decoded.read_bytes()))
                row["verified"] = True
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(f"decode/pixels: {error}")
        else:
            errors.append("No JXL output to independently decode")
    row["status"] = "failed" if errors else "passed"
    if errors:
        row["errors"] = errors
    return row


def summarize(rows, sources, modes, threads, repeats):
    """No aggregate for a partial corpus or an incomplete repeat group."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"], row["mode"], row["threads"]].append(row)
    images = {}
    for source in sources:
        for mode in modes:
            for count in threads:
                trials = groups[source, mode, count]
                complete = (len(trials) == repeats
                            and sorted(r["repeat"] for r in trials) == list(range(repeats))
                            and all(r.get("status") == "passed" and r.get("verified") for r in trials))
                item = dict(source=source, status="passed" if complete else "failed",
                            trials=len(trials), errors=sum(r.get("status") != "passed" for r in trials))
                if complete:
                    item.update(pixels=trials[0]["width"] * trials[0]["height"],
                                bytes=median(r["bytes"] for r in trials),
                                seconds=median(r["seconds"] for r in trials))
                images[source, mode, count] = item
    result = []
    for mode in modes:
        for count in threads:
            items = [dict(images[source, mode, count]) for source in sources]
            for item in items:
                baseline = images[item["source"], "baseline", count]
                if item["status"] == baseline["status"] == "passed":
                    item["bytes_delta_vs_baseline"] = item["bytes"] - baseline["bytes"]
                    item["size_percent_vs_baseline"] = 100 * (item["bytes"] / baseline["bytes"] - 1)
                else:
                    item["bytes_delta_vs_baseline"] = None
                    item["size_percent_vs_baseline"] = None
            complete = bool(items) and all(item["status"] == "passed" for item in items)
            summary = dict(mode=mode, threads=count, status="passed" if complete else "failed",
                           images=len(items), per_image=items)
            if complete:
                size = sum(item["bytes"] for item in items)
                seconds = sum(item["seconds"] for item in items)
                pixels = sum(item["pixels"] for item in items)
                summary.update(bytes=size, seconds=seconds,
                               mean_bpp=sum(8 * item["bytes"] / item["pixels"] for item in items) / len(items),
                               weighted_bpp=8 * size / pixels, mpix_s=pixels / 1e6 / seconds)
            result.append(summary)
    return result


def benchmark(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status="failed", rows=[], inputs=[], errors=[],
                  config=dict(modes=args.mode, threads=args.threads, repeats=args.repeats,
                              effort=args.effort, force_source=args.force_source, timeout=args.timeout),
                  timing="Serial codec work; sum of per-image median native seconds; excludes IO and verification",
                  correctness="Official djxl vs original integer samples including hidden RGB and 16-bit",
                  official_decoder_provenance="Caller-attested, not certified by binary hash",
                  identities=[], summaries=[])
    identities, sources, scratch = {}, [], None

    def remember(path):
        path = path.resolve(strict=True)
        digest = port.file_sha256(path)
        identities[path] = digest
        report["identities"].append(dict(path=str(path), sha256=digest))
        return path

    try:
        args.encoder, args.decoder = remember(args.encoder), remember(args.decoder)
        args.manifest = remember(args.manifest)
        cases = json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
        if not isinstance(cases, list) or not cases:
            raise ValueError("Require a nonempty manifest cases list")
        sources = [str(Path(case["source"]).resolve()) for case in cases]
        if len(set(sources)) != len(sources):
            raise ValueError("Manifest sources must be distinct")
        for source in sources:
            remember(Path(source))
        env, report["environment"] = port.clean_environment(os.environ)
        with (args.out / "trials.jsonl").open("x", encoding="utf-8") as journal:
            for index, (case, source) in enumerate(zip(cases, sources)):
                try:
                    with tempfile.TemporaryDirectory(prefix="source-tools-") as tmp:
                        scratch = Path(tmp)
                        info, snapshot, expected = prepare(Path(source), scratch)
                        report["inputs"].append(dict(source=source, **info))
                        if info["kind"] != "png":
                            raise ValueError("Pixel benchmark requires integer pixel input, not JPEG transcode")
                        if info["input_sha256"] != identities[Path(source)]:
                            raise ValueError("Source changed since manifest freeze")
                        for key, actual in (("source_sha256", "input_sha256"),
                                            ("input_sha256", "input_sha256"),
                                            ("payload_sha256", "payload_sha256")):
                            if key in case and case[key] != info[actual]:
                                raise ValueError(f"Manifest hash mismatch: {key}")
                        for count in args.threads:
                            for repeat in range(args.repeats):
                                for position, mode in enumerate(trial_order(args.mode, index, repeat)):
                                    row = dict(source=source, mode=mode, threads=count,
                                               repeat=repeat, order=position, verified=False)
                                    measure(args, snapshot, expected, scratch, env, row)
                                    if port.file_sha256(snapshot) != info["payload_sha256"]:
                                        row.update(status="failed")
                                        row.setdefault("errors", []).append("Input snapshot changed")
                                    report["rows"].append(row)
                                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                                    journal.flush()
                                    print(f"[{index + 1}/{len(cases)}] t{count} r{repeat + 1} {mode}: {row['status']}", flush=True)
                except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
                    report["errors"].append(dict(source=source, error=str(error)))
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, KeyboardInterrupt) as error:
        report["errors"].append(dict(error=str(error) or "Interrupted"))
    finally:
        for path, digest in identities.items():
            try:
                if port.file_sha256(path) != digest:
                    raise ValueError("SHA256 changed")
            except (OSError, ValueError) as error:
                report["errors"].append(dict(path=str(path), error=f"Immutable input changed: {error}"))
        report["temporary_outputs_cleaned"] = scratch is None or not scratch.exists()
        report["summaries"] = summarize(report["rows"], sources, args.mode, args.threads, args.repeats)
        if (not report["errors"] and report["summaries"]
                and all(item["status"] == "passed" for item in report["summaries"])):
            report["status"] = "passed"
        with (args.out / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main(argv=None):
    args = arguments(argv)
    report = benchmark(args)
    print(f"{report['status'].upper()}: {args.out / 'summary.json'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
