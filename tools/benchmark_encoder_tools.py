"""Serial development-tool experiments, independently checked by official djxl.

Variants are NAME=JSON_ARGV. Only reports persist; each JXL and decoded PNG is
temporary. Primary timing is the encoder call, including the entire requested
scout/pilot, but excluding CLI startup and input/output IO. Whole-process timing
is reported separately. Warmups are verified but excluded from medians; variant
order reverses on alternate rounds. This measures explicit candidates unless
the variant contains --search.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

import compare_cpp_port as port
from benchmark_encoder_checkpoints import load_case, validate_extra_args


def variant(text):
    name, separator, value = text.partition("=")
    if not name or not separator:
        raise argparse.ArgumentTypeError("Expected NAME=JSON_ARGV")
    try:
        tokens = json.loads(value)
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("argv must be a string array")
        validate_extra_args(tokens)
        if any(token.partition("=")[0] == "--output" for token in tokens):
            raise ValueError("variant cannot replace output")
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return name, tokens


def summarize(cases, rows, variants):
    groups = {}
    for row in rows:
        if not row["warmup"]:
            groups.setdefault((row["case_index"], row["variant"]), []).append(row)
    summaries = []
    for case in cases:
        for name, _ in variants:
            selected = groups[(case["index"], name)]
            seconds = statistics.median(row["seconds"] for row in selected)
            wall = statistics.median(row["encode"]["wall_seconds"] for row in selected)
            sizes = [row["bytes"] for row in selected]
            bpps = [row["bpp"] for row in selected]
            summaries.append(dict(
                case_index=case["index"], case=case["label"], variant=name, trials=len(selected),
                median_call_seconds=seconds, median_wall_seconds=wall,
                call_mpix_per_second=case["pixels"] / 1e6 / seconds,
                whole_process_mpix_per_second=case["pixels"] / 1e6 / wall,
                median_bytes=statistics.median(sizes), min_bytes=min(sizes), max_bytes=max(sizes),
                median_bpp=statistics.median(bpps), min_bpp=min(bpps), max_bpp=max(bpps)))
    return summaries


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--encoder", required=True, type=Path)
    cli.add_argument("--djxl", required=True, type=Path)
    cli.add_argument("--case", required=True, action="append", type=Path)
    cli.add_argument("--threads", type=int, default=8)
    cli.add_argument("--effort", type=int, default=9, help="Encoder-lab effort for all variants (1..10)")
    cli.add_argument("--repeats", type=int, default=1, help="Measured rounds per case (positive)")
    cli.add_argument("--warmups", type=int, default=0, help="Verified, unmeasured rounds per case")
    cli.add_argument("--encoder-env", action="append", default=[], metavar="RS_JXL_NAME=value",
                     help="Explicit encoder-only override; inherited codec variables are removed")
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--variant", action="append", type=variant)
    group.add_argument("--matrix", type=Path, help="JSON object mapping variant names to argv arrays")
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    if not 1 <= args.threads <= 64:
        cli.error("threads must be between 1 and 64")
    if not 1 <= args.effort <= 10:
        cli.error("effort must be between 1 and 10")
    if args.repeats < 1 or args.warmups < 0:
        cli.error("repeats must be positive and warmups nonnegative")
    try:
        if args.matrix:
            matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
            if not isinstance(matrix, dict) or not matrix:
                raise ValueError("matrix must be a nonempty JSON object")
            args.variant = [variant(name + "=" + json.dumps(tokens)) for name, tokens in matrix.items()]
        if len({name for name, _ in args.variant}) != len(args.variant):
            raise ValueError("variant names must be unique")
        encoder, decoder = args.encoder.resolve(strict=True), args.djxl.resolve(strict=True)
        for path in (encoder, decoder, *args.case):
            if not path.is_file():
                raise ValueError(f"Expected an existing input file: {path}")
        env, environment = port.clean_environment(os.environ, args.encoder_env)
    except (OSError, ValueError, argparse.ArgumentTypeError) as error:
        cli.error(str(error))
    decoder_env, decoder_environment = port.clean_environment(os.environ)
    args.out.mkdir(exist_ok=False)
    report = dict(status="failed", encoder=str(encoder), encoder_sha256=port.file_sha256(encoder),
                  djxl=str(decoder), djxl_sha256=port.file_sha256(decoder), threads=args.threads, effort=args.effort,
                  repeats=args.repeats, warmups=args.warmups, primary_metric="call_mpix_per_second",
                  timing="Call includes the entire requested scout/pilot, excludes CLI startup and input/output IO. "
                         "Wall includes the encoder process only, not independent decoding. "
                         "Summary rates use median durations; warmups are verified but excluded.",
                  environment=environment, decoder_environment=decoder_environment,
                  cases=[], rows=[], summaries=[])
    print("Primary: encoder-call MPix/s and ms (entire requested scout/pilot; excludes CLI startup and input/output IO). "
          "Wall: encoder-process MPix/s and ms. Warmups are verified, excluded from medians.", flush=True)
    try:
        with tempfile.TemporaryDirectory(dir=args.out) as root:
            for index, path in enumerate(args.case):
                info, payload, _, expected = load_case(index, path)
                if info["kind"] != "png":
                    raise ValueError("This candidate harness requires PNG pixel input")
                report["cases"].append(info)
                source = Path(root) / f"source-{index}.png"
                source.write_bytes(payload)
                for repeat in range(-args.warmups, args.repeats):
                    # Measured round zero always uses input order, independently of warmup count.
                    order = args.variant if repeat % 2 == 0 else args.variant[::-1]
                    for order_index, (name, tokens) in enumerate(order):
                        row = dict(case=info["label"], case_index=index, variant=name, status="failed",
                                   repeat=repeat, warmup=repeat < 0, order_index=order_index)
                        report["rows"].append(row)
                        with tempfile.TemporaryDirectory(dir=root) as temp:
                            output, restored = Path(temp) / "encoded.jxl", Path(temp) / "restored.png"
                            row["encode"] = port.run_process([str(encoder), str(source), "--output", str(output),
                                                              "--threads", str(args.threads), "--effort", str(args.effort), *tokens], env, 300)
                            port.check_process(row["encode"])
                            metrics = json.loads(row["encode"]["stdout"])
                            if (not math.isfinite(metrics["seconds"]) or metrics["seconds"] <= 0
                                    or not math.isfinite(metrics["bpp"]) or metrics["bpp"] < 0):
                                raise ValueError("Encoder reported invalid call time or bpp")
                            if not math.isfinite(row["encode"]["wall_seconds"]) or row["encode"]["wall_seconds"] <= 0:
                                raise ValueError("Invalid encoder process wall time")
                            row.update(metrics)
                            if metrics["bytes"] <= 0 or output.stat().st_size != metrics["bytes"]:
                                raise ValueError("Reported size differs from actual output or is empty")
                            row["decode"] = port.run_process(port.decoder_command(decoder, output, restored, expected.depth), decoder_env, 300)
                            port.check_process(row["decode"])
                            port.compare_pixels(expected, port.png_samples(restored.read_bytes()))
                            row["status"] = "passed"
                            row["call_mpix_per_second"] = info["pixels"] / 1e6 / row["seconds"]
                            row["whole_process_mpix_per_second"] = info["pixels"] / 1e6 / row["encode"]["wall_seconds"]
                            phase = "warmup" if row["warmup"] else "repeat"
                            ordinal = repeat + args.warmups + 1 if row["warmup"] else repeat + 1
                            count = args.warmups if row["warmup"] else args.repeats
                            print(f"{info['label']} {name} {phase}={ordinal}/{count}: "
                                  f"call={row['call_mpix_per_second']:.3f} MPix/s ({row['seconds'] * 1000:.3f} ms) "
                                  f"wall={row['whole_process_mpix_per_second']:.3f} MPix/s ({row['encode']['wall_seconds'] * 1000:.3f} ms) "
                                  f"{row['bytes']} B {row['bpp']:.6f} bpp verified", flush=True)
        if port.file_sha256(encoder) != report["encoder_sha256"]:
            raise ValueError("Encoder changed during the experiment")
        report["summaries"] = summarize(report["cases"], report["rows"], args.variant)
        for item in report["summaries"]:
            print(f"median {item['case']} {item['variant']} (n={item['trials']}): "
                  f"call={item['call_mpix_per_second']:.3f} MPix/s ({item['median_call_seconds'] * 1000:.3f} ms) "
                  f"wall={item['whole_process_mpix_per_second']:.3f} MPix/s ({item['median_wall_seconds'] * 1000:.3f} ms) "
                  f"{item['median_bytes']:g} B range=[{item['min_bytes']}, {item['max_bytes']}] "
                  f"{item['median_bpp']:.6f} bpp", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
