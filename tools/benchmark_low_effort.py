"""Bounded, interleaved low-effort comparisons; no encoder policy overrides."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import compare_cpp_port as port
from benchmark_context_corpus import parse_cjxl_stats
from corpus_inputs import prepare


def profile_rows(stderr):
    result = {}
    for line in stderr.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 4 or "<-" in fields[0]:
            continue
        try:
            result[fields[0]] = dict(calls=int(fields[1]), seconds=float(fields[2]),
                                    longest=float(fields[3]))
        except ValueError:
            continue
    return result


def encoder_command(binary, official, source, output, effort, threads):
    if official:
        return [str(binary), str(source), str(output), "-d", "0", "-e", str(effort),
                f"--num_threads={threads}", "--lossless_jpeg=0", "--keep_invisible=1", "--num_reps=1"]
    return [str(binary), str(source), "--output", str(output), "--effort", str(effort),
            "--threads", str(threads), "--direct-candidate", "baseline"]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["binary"], row["effort"], row["threads"], row["source"]].append(row)
    summaries = defaultdict(list)
    for (binary, effort, threads, source), repeats in groups.items():
        receipts = [r["receipt"] for r in repeats]
        # Report timing medians, but do not assume parallel encoding is bit-identical.
        summaries[binary, effort, threads].append(dict(
            source=source, seconds=statistics.median(r["seconds"] for r in receipts),
            bytes=statistics.median(r["bytes"] for r in receipts),
            pixels=receipts[0]["width"] * receipts[0]["height"]))
    result = []
    for (binary, effort, threads), images in sorted(summaries.items()):
        seconds = sum(r["seconds"] for r in images)
        size = sum(r["bytes"] for r in images)
        pixels = sum(r["pixels"] for r in images)
        result.append(dict(binary=binary, effort=effort, threads=threads,
                           images=len(images), seconds=seconds, bytes=size,
                           mean_bpp=statistics.mean(r["bytes"] * 8 / r["pixels"] for r in images),
                           weighted_bpp=size * 8 / pixels, mpix_s=pixels / 1e6 / seconds))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", action="append", required=True, help="NAME=portfolio_lab.exe")
    parser.add_argument("--cjxl", action="append", default=[], help="Optional NAME=official-cjxl.exe")
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--efforts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (args.out.exists() or args.repeats < 1 or not all(1 <= e <= 9 for e in args.efforts)
            or not all(1 <= t <= 256 for t in args.threads)):
        parser.error("Require a fresh output directory and valid effort/thread/repeat counts")
    binaries = {}
    official = {value.split("=", 1)[0] for value in args.cjxl}
    for value in args.binary + args.cjxl:
        name, path = value.split("=", 1)
        if not name or name in binaries:
            parser.error("Binary names must be nonempty and unique")
        binaries[name] = Path(path).resolve(strict=True)
    for values in (args.image, args.efforts, args.threads):
        if len(set(values)) != len(values):
            parser.error("Duplicate image/effort/thread entries")
    args.out.mkdir(parents=True)
    immutable = list(binaries.values()) + [args.decoder] + args.image
    hashes = {str(p): port.file_sha256(p) for p in immutable}
    env, removed = port.clean_environment(os.environ)
    if args.profile:
        env["RS_JXL_PROFILE"] = "1"
    report = dict(status="running", config=vars(args), hashes=hashes, rows=[],
                  removed_environment=removed, errors=[],
                  timing="Sum of per-image median native encode time; serial interleaved processes; excludes IO/oracle",
                  correctness="Official djxl pixels against original, including hidden RGB and original bit depth")
    count = len(args.image) * len(args.efforts) * len(args.threads) * args.repeats * len(binaries)
    try:
        with (args.out / "trials.jsonl").open("x", encoding="utf-8") as journal:
            for index, image in enumerate(args.image):
                with tempfile.TemporaryDirectory(prefix="rs-jxl-lowe-") as directory:
                    scratch = Path(directory)
                    info, snapshot, expected = prepare(image, scratch)
                    if info["kind"] != "png":
                        raise ValueError("Require PNG pixel inputs")
                    for effort in args.efforts:
                        for threads in args.threads:
                            for repeat in range(args.repeats):
                                names = list(binaries)
                                if (repeat + index + effort + threads) % 2:
                                    names.reverse()
                                for name in names:
                                    output, decoded = scratch / "output.jxl", scratch / "decoded.png"
                                    process = port.run_process(encoder_command(binaries[name], name in official,
                                        snapshot, output, effort, threads), env, 180)
                                    port.check_process(process)
                                    if name in official:
                                        receipt = dict(parse_cjxl_stats(process["stderr"], expected.width,
                                                                       expected.height, threads),
                                                       encode_calls=1, effort=effort, threads=threads,
                                                       bytes=output.stat().st_size, width=expected.width,
                                                       height=expected.height, implementation="official cjxl")
                                    else:
                                        receipt = json.loads(process["stdout"])
                                    if (receipt["encode_calls"] != 1 or receipt["effort"] != effort
                                            or receipt["threads"] != threads
                                            or receipt["bytes"] != output.stat().st_size
                                            or not math.isfinite(receipt["seconds"]) or receipt["seconds"] <= 0
                                            or (receipt["width"], receipt["height"]) != (expected.width, expected.height)):
                                        raise ValueError("Invalid native receipt")
                                    oracle = port.run_process(port.decoder_command(
                                        args.decoder, output, decoded, expected.depth), env, 180)
                                    port.check_process(oracle)
                                    port.compare_pixels(expected, port.png_samples(decoded.read_bytes()))
                                    row = dict(binary=name, source=str(image), effort=effort, threads=threads,
                                               repeat=repeat, receipt=receipt, verified=True,
                                               encoder_process=process, decoder_process=oracle,
                                               stages=profile_rows(process["stderr"]))
                                    report["rows"].append(row)
                                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                                    journal.flush()
                                    print(f"[{len(report['rows'])}/{count}] {image.name} e{effort} t{threads} {name}: "
                                          f"{receipt['seconds']:.5f}s {receipt['bytes']}B PASS", flush=True)
                                    output.unlink()
                                    decoded.unlink()
        for path, digest in hashes.items():
            if port.file_sha256(Path(path)) != digest:
                raise ValueError(f"Input or executable changed during run: {path}")
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["errors"].append(repr(error))
        raise
    finally:
        report["summaries"] = summarize(report["rows"])
        (args.out / "summary.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
