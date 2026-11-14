"""Bounded serial OLD/NEW native e9 defaults comparison; no policy overrides.

Official djxl provenance is caller-attested. Only reports persist; snapshots,
compressed outputs and decoded pixels are temporary. No benchmark runs on import.
"""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import benchmark_source_tools as source_tools
import compare_cpp_port as port
from corpus_inputs import prepare


MODES = ("baseline", "candidate")
ERRORS = (OSError, ValueError, RuntimeError, TypeError, KeyError)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "decoder", "manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--force-source", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.repeats <= 100 or not math.isfinite(args.timeout)
            or args.timeout <= 0 or len(set(args.threads)) != len(args.threads)
            or not all(1 <= t <= 256 for t in args.threads)):
        parser.error("Require repeats in 1..100, positive finite timeout, distinct threads in 1..256")
    if args.out.exists():
        parser.error("Output must be a NEW directory; refusing overwrite")
    return args


def command(args, snapshot, output, threads, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown binary mode: {mode}")
    return [str(getattr(args, mode)), str(snapshot), "--output", str(output),
            "--effort", "9", "--threads", str(threads), "--direct-candidate",
            "source" if args.force_source else "baseline"]


def validate_receipt(row, args, mode, threads, size, expected, grayscale=False):
    if mode not in MODES:
        raise ValueError(f"Unknown binary mode: {mode}")
    integers = dict(encode_calls=1, effort=9, threads=threads, bytes=size,
                    width=expected.width, height=expected.height)
    if (row.get("direct") is not True or size <= 0
            or row.get("candidate") != ("source" if args.force_source else "baseline")
            or any(type(row.get(k)) is not int or row[k] != v for k, v in integers.items())
            or type(row.get("seconds")) not in (int, float)
            or not math.isfinite(row["seconds"]) or row["seconds"] <= 0):
        raise ValueError("Invalid direct receipt: require exactly one native e9 encode")
    tools = row.get("source_tools")
    if not isinstance(tools, dict) or "profile" not in tools or tools["profile"] is not None:
        raise ValueError("Require null source_tools request profile (ordinary defaults)")
    previous = tools.get("previous_channels")
    limit = 1 if mode == "candidate" and not grayscale else 0
    if (type(previous) is not int or previous != limit
            or tools.get("lz_quick") != ("Sparse" if mode == "candidate" else "Off")
            or tools.get("palette_order") != ("ScreenedRuns" if mode == "candidate" else "Default")):
        raise ValueError("Effective source tools do not match OLD/NEW defaults")
    # Older receipts need not expose any newer request/budget fields.
    if "max_previous_channels" in tools and tools["max_previous_channels"] is not None:
        raise ValueError("Unexpected explicit previous-channel request")


def measure(args, snapshot, expected, directory, env, row):
    errors = []
    with tempfile.TemporaryDirectory(prefix="trial-", dir=directory) as tmp:
        output, decoded = Path(tmp) / "out.jxl", Path(tmp) / "out.png"
        try:
            _, framing = port.png_payload(snapshot.read_bytes())
            row["encoder_process"] = port.run_process(
                command(args, snapshot, output, row["threads"], row["mode"]), env, args.timeout)
            port.check_process(row["encoder_process"])
            receipt = json.loads(row["encoder_process"]["stdout"],
                                 parse_constant=source_tools.reject_constant)
            if not isinstance(receipt, dict):
                raise ValueError("Native receipt must be an object")
            row["receipt"] = receipt
            validate_receipt(receipt, args, row["mode"], row["threads"],
                             output.stat().st_size, expected, framing["color_type"] == 0)
            for key in ("width", "height", "bytes", "seconds", "encode_calls", "source_tools"):
                row[key] = receipt[key]
        except ERRORS as error:
            errors.append(f"encode/receipt: {error}")
        # Independently verify even outputs from failed encoders/invalid receipts.
        if output.exists():
            try:
                row["decoder_process"] = port.run_process(
                    port.decoder_command(args.decoder, output, decoded, expected.depth), env, args.timeout)
                port.check_process(row["decoder_process"])
                port.compare_pixels(expected, port.png_samples(decoded.read_bytes()))
                row["verified"] = True
            except ERRORS as error:
                errors.append(f"decode/pixels: {error}")
        else:
            errors.append("No JXL output to independently decode")
    row["status"] = "failed" if errors else "passed"
    if errors:
        row["errors"] = errors
    return row


def summarize(rows, sources, threads, repeats):
    summaries = source_tools.summarize(rows, sources, MODES, threads, repeats)
    baselines = {s["threads"]: s for s in summaries if s["mode"] == "baseline"}
    for item in summaries:
        baseline = baselines[item["threads"]]
        comparable = item["status"] == baseline["status"] == "passed"
        item["changes_vs_baseline"] = {
            key: (dict(delta=item[key] - baseline[key],
                       percent=100 * (item[key] / baseline[key] - 1)) if comparable else None)
            for key in ("bytes", "seconds", "mean_bpp", "weighted_bpp", "mpix_s")}
    return summaries


def benchmark(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status="failed", rows=[], inputs=[], errors=[], identities=[], summaries=[],
                  config=dict(modes=MODES, threads=args.threads, repeats=args.repeats,
                              effort=9, force_source=args.force_source, timeout=args.timeout),
                  timing="Serial; sum of per-image median native encode seconds; excludes IO and verification",
                  correctness="Official djxl vs original integer samples including hidden RGB and 16-bit; never compressed bytes",
                  official_decoder_provenance="Caller-attested, not certified by binary hash")
    identities, sources, scratch = {}, [], None

    def remember(path, role):
        path = path.resolve(strict=True)
        digest = port.file_sha256(path)
        if path in identities and identities[path] != digest:
            raise ValueError("Identity changed during freeze")
        identities[path] = digest
        report["identities"].append(dict(role=role, path=str(path), sha256=digest))
        return path

    try:
        for role in ("baseline", "candidate", "decoder", "manifest"):
            setattr(args, role, remember(getattr(args, role), role))
        cases = json.loads(args.manifest.read_text(encoding="utf-8"),
                           parse_constant=source_tools.reject_constant)["cases"]
        if not isinstance(cases, list) or not cases:
            raise ValueError("Require a nonempty manifest cases list")
        sources = [str(Path(case["source"]).resolve()) for case in cases]
        if len(set(sources)) != len(sources):
            raise ValueError("Manifest sources must be distinct")
        for source in sources:
            remember(Path(source), "input")
        env, report["environment"] = port.clean_environment(os.environ)
        with (args.out / "trials.jsonl").open("x", encoding="utf-8") as journal:
            for index, (case, source) in enumerate(zip(cases, sources)):
                try:
                    with tempfile.TemporaryDirectory(prefix="source-defaults-") as tmp:
                        scratch = Path(tmp)
                        info, snapshot, expected = prepare(Path(source), scratch)
                        report["inputs"].append(dict(source=source, **info))
                        if info["kind"] != "png":
                            raise ValueError("Require integer pixel input, not JPEG transcode")
                        if info["input_sha256"] != identities[Path(source)]:
                            raise ValueError("Source changed since manifest freeze")
                        for key, actual in (("source_sha256", "input_sha256"),
                                            ("input_sha256", "input_sha256"),
                                            ("payload_sha256", "payload_sha256")):
                            if key in case and case[key] != info[actual]:
                                raise ValueError(f"Manifest hash mismatch: {key}")
                        for count in args.threads:
                            for repeat in range(args.repeats):
                                order = source_tools.trial_order(list(MODES), index, repeat)
                                for position, mode in enumerate(order):
                                    row = dict(source=source, mode=mode, threads=count,
                                               repeat=repeat, order=position, verified=False)
                                    measure(args, snapshot, expected, scratch, env, row)
                                    if port.file_sha256(snapshot) != info["payload_sha256"]:
                                        row["status"] = "failed"
                                        row.setdefault("errors", []).append("Input snapshot changed")
                                    report["rows"].append(row)
                                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                                    journal.flush()
                                    print(f"[{index + 1}/{len(cases)}] t{count} r{repeat + 1} {mode}: {row['status']}", flush=True)
                except ERRORS as error:
                    report["errors"].append(dict(source=source, error=str(error)))
    except (*ERRORS, KeyboardInterrupt) as error:
        report["errors"].append(dict(error=str(error) or "Interrupted"))
    finally:
        for path, digest in identities.items():
            try:
                if port.file_sha256(path) != digest:
                    raise ValueError("SHA256 changed")
            except (OSError, ValueError) as error:
                report["errors"].append(dict(path=str(path), error=f"Immutable input changed: {error}"))
        report["temporary_outputs_cleaned"] = scratch is None or not scratch.exists()
        report["summaries"] = summarize(report["rows"], sources, args.threads, args.repeats)
        if (not report["errors"] and report["summaries"]
                and all(s["status"] == "passed" for s in report["summaries"])):
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
