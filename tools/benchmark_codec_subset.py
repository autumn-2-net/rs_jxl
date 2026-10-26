"""Serial, pixel-verified codec comparison on a deterministic stratified subset.

No encoder algorithms are changed. PNG metadata is stripped identically without
recompressing IDAT. JPEG inputs use byte-reversible coefficient transcode only.
Decode measurements cross every encoder's output with every decoder, without
PNG serialization; wall time / repetitions amortizes process startup and input IO.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import struct
import tempfile

from PIL import Image

from compare_cpp_port import (
    binary_info, check_process, clean_environment, compare_pixels,
    encoder_command, png_payload, png_samples, run_process,
)

NAMES = ("official", "cpp", "rust")


def select_subset(root, seed):
    """Select without looking at timing or compression results; keep full images."""
    rng = random.Random(seed)
    result = []
    groups = [
        ("photo-kodak", "jxl_testdata/images/photo_kodak", None),
        ("photo-tecnick", "jxl_testdata/images/photo_tecnick", None),
        ("game-ui", "jxl_testdata/images/screenshot_game", None),
        ("web", "jxl_testdata/images/screenshot_web", None),
        ("texture", "jxl_testdata/images/textures_photo", None),
        ("icon-alpha", "jxl_testdata/images/icon_512", 8),
        ("gray", "jxl_testdata/testdata-main/external/wesaturate/500px", "gray"),
        ("photo16-small", "jxl_testdata/testdata-main/external/wesaturate/64px", 16),
    ]
    for label, folder, required in groups:
        candidates = []
        for path in sorted((root / folder).rglob("*.png")):
            with path.open("rb") as stream:
                header = stream.read(29)
            if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            w, h, depth, color, _, _, _ = struct.unpack(">IIBBBBB", header[16:29])
            minimum = 4096 if required == 16 else 65_536
            if not minimum <= w * h <= 2_500_000 or depth not in (8, 16):
                continue
            if required == "gray" and color not in (0, 4):
                continue
            if isinstance(required, int) and depth != required:
                continue
            candidates.append(path)
        if not candidates:
            raise ValueError(f"No eligible samples: {label} {folder}")
        result.append(dict(label=label, kind="png", path=str(rng.choice(candidates)),
                           eligible=len(candidates), selection="seeded stratified random"))
    for label, name in (
        ("pdf-regression", "9bde6aaf390109f97622d35ff1e900ee.png"),
        ("wallhaven-regression", "test_img/png/wallhaven-49yekn.png"),
    ):
        result.append(dict(label=label, kind="png", path=str((root / name).resolve(strict=True)),
                           selection="explicit historical regression, not random"))
    jpegs = sorted((root / "test_img").glob("*.jpg"))
    for i, path in enumerate(rng.sample(jpegs, 3)):
        result.append(dict(label=f"jpeg-{i + 1}", kind="jpeg", path=str(path),
                           eligible=len(jpegs), selection="seeded random"))
    return result


def decode_command(name, binary, source, *, kind, threads, reps=1, output=None, depth=8):
    args = [str(binary)] + (["decode"] if name == "rust" else []) + [str(source)]
    if output is not None:
        args.append(str(output))
    args += [f"--num_threads={threads}", "--quiet"]
    if kind == "jpeg":
        args.append("--reconstruct_jpeg")
    elif output is not None:
        args.append(f"--bits_per_sample={depth}")
    if output is None:
        args += ["--disable_output", f"--num_reps={reps}"]
        # djxl's no-output path only accepts float buffers. Match that explicitly.
        if kind == "png" and name == "rust":
            args.append("--bits=32")
    return args


def encode_command(name, binary, source, output, case, args):
    if case["kind"] == "png":
        return encoder_command("rust" if name == "rust" else "cpp", binary,
                               source, output, args.threads, 9, "full-scheduler", 33_554_432)
    if name == "rust":
        return [str(binary), "transcode-jpeg", str(source), str(output), "--effort=9",
                f"--threads={args.threads}", "--source-search"]
    return [str(binary), str(source), str(output), "--effort=9", "--lossless_jpeg=1",
            f"--num_threads={args.threads}"]


def measured(command, env, args):
    result = run_process(command, env, args.timeout)
    check_process(result)
    return result


def summarize(case, rows):
    summary = dict(**case, encode={}, decode={})
    for name in NAMES:
        enc = [r for r in rows if r["phase"] == "encode" and r["encoder"] == name
               and not r["warmup"]]
        seconds = statistics.median(r["wall_seconds"] for r in enc)
        size = statistics.median(r["bytes"] for r in enc)
        summary["encode"][name] = dict(bytes=size, bpp=8 * size / case["pixels"],
            seconds=seconds, mpix_s=case["pixels"] / 1e6 / seconds,
            ratio=size / case["input_bytes"],
            min_seconds=min(r["wall_seconds"] for r in enc),
            max_seconds=max(r["wall_seconds"] for r in enc))
        summary["decode"][name] = {}
        for decoder in NAMES:
            dec = [r for r in rows if r["phase"] == "decode" and r["encoder"] == name
                   and r["decoder"] == decoder and not r["warmup"]]
            seconds = statistics.median(r["wall_seconds"] / r["reps"] for r in dec)
            summary["decode"][name][decoder] = dict(seconds=seconds,
                mpix_s=case["pixels"] / 1e6 / seconds)
    return summary


def aggregates(summaries):
    result = {}
    for kind in ("png", "jpeg"):
        cases = [s for s in summaries if s["kind"] == kind]
        if not cases:
            continue
        pixels = sum(s["pixels"] for s in cases)
        sizes = sum(s["input_bytes"] for s in cases)
        stats = dict(count=len(cases), pixels=pixels, input_bytes=sizes, encoders={}, decoders={})
        for name in NAMES:
            size = sum(s["encode"][name]["bytes"] for s in cases)
            seconds = sum(s["encode"][name]["seconds"] for s in cases)
            stats["encoders"][name] = dict(bytes=size, bpp=8 * size / pixels,
                seconds=seconds, mpix_s=pixels / 1e6 / seconds, ratio=size / sizes)
            # Same official bitstream for every decoder in the headline comparison.
            seconds = sum(s["decode"]["official"][name]["seconds"] for s in cases)
            stats["decoders"][name] = dict(seconds=seconds, mpix_s=pixels / 1e6 / seconds)
        result[kind] = stats
    return result


def markdown(report):
    lines = ["# Lossless Codec Subset Benchmark", "", f"Status: **{report['status']}**.",
        "", "## Method", "",
        f"Windows native, serial jobs, e9, {report['config']['threads']} workers requested in each codec.",
        "Official 0.12 defaults vs frozen optimized C++ full scheduler vs Rust --source-search.",
        "Same effort labels do not imply equal search work. No resizing or crops.",
        "PNG metadata removed identically without IDAT recompression; hidden RGB preserved.",
        "JPEG uses coefficient transcode and exact original-byte reconstruction, never pixel fallback.",
        f"Seed {report['config']['seed']}; {report['config']['warmups']} excluded warmup(s), "
        f"{report['config']['repeats']} timed rounds; medians, rotated execution order.",
        "Encoding: CLI wall including startup, source input and JXL output IO; excludes verification.",
        f"Decode: CLI wall / {report['config']['decode_reps']} in-process repetitions, no output IO; "
        "includes amortized startup/read and allocation. PNG buffers are float32 in all decoders.",
        "This is application throughput, not isolated entropy-kernel timing. Worker semantics can differ.",
        "C++ reuses its runner across in-process repeats; Rust CLI checkpoints may differ in runner reuse. "
        "That overhead is included, not removed or mislabeled as core decoder cost.",
        "Every encode is checked by all three decoders outside timing. PNG compares integer samples, "
        "including transparent RGB; JPEG compares complete original bytes.",
        "Decode speed is NOT PNG output speed. For JPEG it measures original-byte reconstruction, not IDCT rendering.",
        "Aggregate bpp = 8 * total JXL bytes / total pixels; MPix/s = total pixels / sum of per-image median seconds.",
        "Small stratified sample plus two explicitly selected regressions, not a population estimate.",
        "", "## Aggregate", ""]
    for kind, stats in report.get("aggregates", {}).items():
        lines += [f"### {kind.upper()} ({stats['count']} files)", "",
            "| Codec | Bytes | bpp | Enc s | Enc MPix/s | Same-official-stream decode MPix/s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name in NAMES:
            s = stats["encoders"][name]
            lines.append(f"| {name} | {s['bytes']:.0f} | {s['bpp']:.4f} | {s['seconds']:.3f} | "
                         f"{s['mpix_s']:.3f} | {stats['decoders'][name]['mpix_s']:.3f} |")
        lines.append("")
    lines += ["## Per Image", "", "| Case | Dimensions | Codec | Bytes | bpp | Enc s | Enc MPix/s | Own-stream decode MPix/s |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in report["summaries"]:
        for name in NAMES:
            s = case["encode"][name]
            lines.append(f"| {case['label']} | {case['width']}x{case['height']} | {name} | "
                         f"{s['bytes']:.0f} | {s['bpp']:.4f} | {s['seconds']:.3f} | {s['mpix_s']:.3f} | "
                         f"{case['decode'][name][name]['mpix_s']:.3f} |")
    lines += ["", "## Same Bitstream Decode Matrix", "",
              "| Case | Encoder | Official MPix/s | C++ MPix/s | Rust MPix/s |",
              "| --- | --- | ---: | ---: | ---: |"]
    for case in report["summaries"]:
        for name in NAMES:
            d = case["decode"][name]
            lines.append(f"| {case['label']} | {name} | " + " | ".join(f"{d[n]['mpix_s']:.3f}" for n in NAMES) + " |")
    lines += ["", "## Inputs", ""]
    for case in report["cases"]:
        lines.append(f"- {case['label']}: `{case['path']}` ({case['selection']}).")
    lines += ["", "## Binaries", ""]
    for name, info in report["binaries"].items():
        lines.append(f"- {name}: `{info['path']}`; SHA256 `{info['sha256']}`; {info['version'].splitlines()[0]}")
    lines += ["", "Binary hashes identify test executables only; JXL byte identity is not a target.",
              "C++ build provenance: docs/CPP_REFERENCE_BUILD.md. JSON/JSONL contain exact commands, stderr and per-run timings."]
    if report.get("error"):
        lines += ["", "## Failure", "", "```text", report["error"], "```"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    for name in NAMES:
        parser.add_argument("--" + name, type=Path, required=True)
        if name != "rust":
            parser.add_argument("--" + name + "-decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="NEW output directory")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-reps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int, help="Smoke test first N preselected cases")
    args = parser.parse_args()
    if (args.repeats < 1 or args.warmups < 0 or args.decode_reps < 1
            or not 1 <= args.threads <= 256 or not 0 < args.timeout < float("inf")
            or (args.limit is not None and args.limit < 1)):
        parser.error("invalid measurement settings")
    encoders = {n: getattr(args, n).resolve(strict=True) for n in NAMES}
    decoders = {n: (args.rust if n == "rust" else getattr(args, n + "_decoder")).resolve(strict=True) for n in NAMES}
    cases = select_subset(args.dataset_root.resolve(strict=True), args.seed)
    if args.limit is not None:
        cases = cases[:args.limit]
    args.out.mkdir(parents=False, exist_ok=False)
    env, env_record = clean_environment(os.environ)
    # No verbose fork tracing in timed processes; command line selects full search.
    report = dict(status="running", created=datetime.now(timezone.utc).isoformat(),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        environment=env_record, binaries={}, cases=cases, summaries=[], errors=[])
    for label, binaries in (("encoder", encoders), ("decoder", decoders)):
        for name, path in binaries.items():
            report["binaries"][f"{label}-{name}"] = binary_info(path, env, 10)
    print("SUBSET " + json.dumps(cases), flush=True)
    try:
        with (args.out / "trials.jsonl").open("x", encoding="utf-8") as log:
            with tempfile.TemporaryDirectory(prefix="rsjxl-subset-") as temp:
                root = Path(temp)
                for index, chosen in enumerate(cases):
                    rows = []
                    original = Path(chosen["path"]).read_bytes()
                    case = dict(chosen, input_bytes=len(original), sha256=hashlib.sha256(original).hexdigest())
                    if case["kind"] == "png":
                        payload, _ = png_payload(original)
                        expected = png_samples(original)
                        compare_pixels(expected, png_samples(payload))
                        case.update(width=expected.width, height=expected.height, depth=expected.depth)
                    else:
                        payload, expected = original, None
                        with Image.open(case["path"]) as image:
                            case.update(width=image.width, height=image.height, depth=8)
                    case["pixels"] = case["width"] * case["height"]
                    source = root / ("input.png" if case["kind"] == "png" else "input.jpg")
                    source.write_bytes(payload)
                    print(f"[{index + 1}/{len(cases)}] {case['label']} {case['width']}x{case['height']}", flush=True)
                    for repeat in range(-args.warmups, args.repeats):
                        start = (repeat + index) % 3
                        for name in NAMES[start:] + NAMES[:start]:
                            output = root / (name + ".jxl")
                            output.unlink(missing_ok=True)
                            process = measured(encode_command(name, encoders[name], source, output, case, args), env, args)
                            row = dict(process, phase="encode", encoder=name, repeat=repeat,
                                       warmup=repeat < 0, label=case["label"], bytes=output.stat().st_size)
                            for decoder in NAMES:
                                restored = root / ("verify.png" if case["kind"] == "png" else "verify.jpg")
                                restored.unlink(missing_ok=True)
                                measured(decode_command(decoder, decoders[decoder], output,
                                    kind=case["kind"], threads=args.threads, output=restored, depth=case["depth"]), env, args)
                                if expected is not None:
                                    compare_pixels(expected, png_samples(restored.read_bytes()))
                                elif restored.read_bytes() != original:
                                    raise ValueError(f"JPEG original bytes differ: {name} -> {decoder}")
                            row["verified_by"] = list(NAMES)
                            log.write(json.dumps(row) + "\n")
                            log.flush()
                            rows.append(row)
                            print(f"  enc {name:8} round={repeat:2} {row['bytes']:9} B {row['wall_seconds']:.3f}s", flush=True)
                    # Retain only each final trial's stream; different trials need
                    # not be byte-identical. Every preceding stream was verified.
                    for repeat in range(-args.warmups, args.repeats):
                        for name in NAMES:
                            start = (repeat + index) % 3
                            for decoder in NAMES[start:] + NAMES[:start]:
                                process = measured(decode_command(decoder, decoders[decoder], root / (name + ".jxl"),
                                    kind=case["kind"], threads=args.threads, reps=args.decode_reps), env, args)
                                row = dict(process, phase="decode", encoder=name, decoder=decoder,
                                    repeat=repeat, warmup=repeat < 0, label=case["label"], reps=args.decode_reps)
                                log.write(json.dumps(row) + "\n")
                                log.flush()
                                rows.append(row)
                    summary = summarize(case, rows)
                    report["summaries"].append(summary)
                    report["aggregates"] = aggregates(report["summaries"])
                    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print("SUMMARY " + json.dumps(summary), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.out / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    print("AGGREGATE " + json.dumps(report["aggregates"]), flush=True)


if __name__ == "__main__":
    main()
