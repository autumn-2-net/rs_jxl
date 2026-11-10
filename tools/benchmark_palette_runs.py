"""Bounded palette_lab trials; official djxl verifies original PNG samples, including hidden RGB.

No builds, production-default changes, searchbest, or JXL identity comparisons.
Timing is exactly one native encode, excluding process startup and file IO.
Configured order is not activation evidence: receipts cover global tuples only.
Graph execution needs a separate profile audit, not an eligible tuple count.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import zlib

from benchmark_fast_context import load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
from benchmark_tile_patches import add_comparisons as _comparisons, aggregate as _aggregate
import compare_cpp_port as port
import corpus_inputs

MODES = ("default", "frequency", "runs", "screened", "hot0", "hot32", "hot128", "hot512",
         "adjacency", "adjacency-minla", "ezeng", "ezeng-minla")
DEFAULT_MODES = ("default", "runs", "hot0", "hot128")
MINLA_MODES = ("adjacency-minla", "ezeng-minla")
SCOPE = "global_tuple_only; excludes single-channel compaction and local/groupwise palettes"


def prepare_png(source, original, scratch):
    """Keep packed input intact; only expand the independent oracle's samples."""
    if original.startswith(port.PNG_SIGNATURE) and len(original) > 24 and original[24] < 8:
        # Reuse the corpus adapter's packed-depth/transparency validation. Its
        # expanded snapshot is for checking only, never fed to the encoder.
        info, expanded, expected = corpus_inputs.prepare(source, scratch)
        try:
            if info["input_sha256"] != port.sha256(original):
                raise ValueError("Original changed during packed PNG preparation")
            payload, _ = corpus_inputs.png_framing(original)
        finally:
            expanded.unlink()
        return payload, expected, "original-packed-png; oracle-expanded-to-8bit"
    payload, _ = port.png_payload(original)
    return payload, port.png_samples(original), "original-8/16bit-png"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True, help="Prebuilt palette_lab executable")
    p.add_argument("--decoder", "--official-djxl", "--djxl", dest="decoder", type=Path, required=True,
                   help="Caller-attested official djxl")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(DEFAULT_MODES))
    p.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--minla-rounds", type=int, choices=range(41),
                   help="Refinement rounds for MinLA modes only (native default: 4)")
    p.add_argument("--palette-colors", type=int)
    p.add_argument("--full-frame", action="store_true")
    p.add_argument("--profile", action="store_true", help="Collect inclusive native profile diagnostics")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if (not 1 <= a.threads <= 256 or not 1 <= a.repeats <= 100 or not 0 <= a.warmups <= 100
            or not positive_number(a.timeout)
            or (a.palette_colors is not None and not 0 <= a.palette_colors <= 70911)):
        p.error("Invalid threads/repeats/warmups/timeout/palette-colors")
    if len(set(a.modes)) != len(a.modes) or "default" not in a.modes:
        p.error("Use distinct modes including default")
    if a.minla_rounds is not None and not any(mode in MINLA_MODES for mode in a.modes):
        p.error("--minla-rounds requires adjacency-minla or ezeng-minla in --modes")
    if not a.case and a.manifest is None and not a.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if a.out.exists():
        p.error("Report already exists")
    return a


def mode_config(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return ("hot", int(mode[3:])) if mode.startswith("hot") else (mode, None)


def command(args, source, output, mode):
    native, hot = mode_config(mode)
    cmd = [str(args.encoder), str(source), "--output", str(output), "--mode", native,
           "--effort", str(args.effort), "--threads", str(args.threads)]
    if hot is not None:
        cmd += ["--hot-colors", str(hot)]
    if mode in MINLA_MODES and args.minla_rounds is not None:
        cmd += ["--minla-rounds", str(args.minla_rounds)]
    if args.palette_colors is not None:
        cmd += ["--palette-colors", str(args.palette_colors)]
    if args.full_frame:
        cmd += ["--full-frame"]
    return cmd


def fixtures(folder):
    """Four run-skew formats, one 300-tuple cutoff probe, and a gray control."""
    folder.mkdir()
    paths = []
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    for depth, color, colors in ((8, 2, 48), (8, 6, 48), (16, 2, 48), (16, 6, 48),
                                 (16, 6, 300), (8, 0, 0)):
        width, height = (257, 65) if colors == 300 else (193, 41)
        channels = {0: 1, 2: 3, 6: 4}[color]
        top = (1 << depth) - 1
        raw = bytearray()
        for y in range(height):
            row = []
            for x in range(width):
                # Color zero has many pixels but few runs; other colors have
                # fewer pixels and many short runs. This separates the orders.
                index = 0 if x < 113 or not colors else 1 + ((x + y * 7) % (colors - 1))
                if colors == 300:
                    # Solid top rows form one long background run. The mixed
                    # tail gives 31 tuples extra short runs among 299 tuples.
                    residue = (x + (y - 17) * width) % 449
                    index = 0 if y < 17 else 1 + (residue if residue < 299 else (residue - 299) % 31)
                for c in range(channels):
                    value = ((index * (37 + c * 18) + 11 + c * 29) * (257 if depth == 16 else 1)
                             + (index if depth == 16 else 0)) & top
                    if c == 3:
                        value = 0 if index % 3 == 0 else top
                    if color == 0:
                        value = (x * 31 + y * 17) & top
                    row.append(value)
            raw.append(0)
            raw.extend(struct.pack(">" + ("B" if depth == 8 else "H") * len(row), *row))
        header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0)
        path = folder / f"{'gray-control' if color == 0 else 'run-skew'}-{depth}-{color}-{colors}.png"
        path.write_bytes(port.PNG_SIGNATURE + chunk(b"IHDR", header)
                         + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        paths.append(path)
    return paths


def validate_measurement(m, mode, size, width, height, args):
    native, hot = mode_config(mode)
    if not isinstance(m, dict) or not positive_number(m.get("seconds")):
        raise ValueError("Invalid native duration")
    if "profile" not in m or (m["profile"] is not None and not isinstance(m["profile"], str)):
        raise ValueError("Invalid profile snapshot; expected nullable string")
    if type(m.get("bytes")) is not int or m["bytes"] != size or size <= 0:
        raise ValueError("Invalid actual codestream size")
    if (m.get("width"), m.get("height")) != (width, height) or width * height <= 0:
        raise ValueError("Mismatched dimensions")
    for key, expected in (("bpp", size * 8 / (width * height)),
                          ("mpix_per_second", width * height / 1e6 / m["seconds"])):
        if not positive_number(m.get(key)) or not math.isclose(m[key], expected, rel_tol=1e-5):
            raise ValueError(f"Invalid {key}")
    if (m.get("mode") != native or type(m.get("encode_calls")) is not int
            or m["encode_calls"] != 1 or m.get("patches_cleared") is not True
            or m.get("effort") != args.effort or m.get("threads") != args.threads):
        raise ValueError("Mode/work receipt mismatch")
    if m.get("overrides") != dict(palette_colors=args.palette_colors, full_frame=args.full_frame):
        raise ValueError("Experimental overrides mismatch")
    configured, projected = m.get("configured"), m.get("projected")
    if not isinstance(configured, dict) or not isinstance(projected, dict):
        raise ValueError("Missing configured/projected policy")
    expected_order = {"default": "Default", "frequency": "FrequencyLuma", "runs": "RunFrequency",
                      "screened": "ScreenedRuns", "adjacency": "Adjacency"}.get(native)
    rounds = None
    if hot is not None:
        expected_order = f"HotRuns {{ hot_colors: {hot} }}"
    if native in MINLA_MODES or native == "ezeng":
        rounds = 0 if native == "ezeng" else (4 if args.minla_rounds is None else args.minla_rounds)
        variant = "AdjacencyMinLa" if native == "adjacency-minla" else "EzengMinLa"
        expected_order = f"{variant} {{ rounds: {rounds} }}"
    if configured.get("order") != expected_order or configured.get("hot_colors") != hot:
        raise ValueError("Configured order mismatch")
    actual_rounds = configured.get("minla_rounds")
    if actual_rounds != rounds or (rounds is not None and type(actual_rounds) is not int):
        raise ValueError("Configured MinLA rounds mismatch")
    for policy in (configured, projected):
        if (type(policy.get("groupwise")) is not bool or type(policy.get("palette_colors")) is not int
                or not 0 <= policy["palette_colors"] <= 70911):
            raise ValueError("Invalid routing/cap policy")
    if (configured["groupwise"] != (projected["groupwise"] and not args.full_frame)
            or configured["palette_colors"] != (args.palette_colors if args.palette_colors is not None
                                                 else projected["palette_colors"])):
        raise ValueError("Unexpected non-order policy change")
    evidence = m.get("palette_evidence")
    if not isinstance(evidence, dict) or evidence.get("scope") != SCOPE:
        raise ValueError("Missing actual palette evidence/scope")
    if evidence.get("route") not in ("full_image", "groupwise"):
        raise ValueError("Invalid actual route")
    if args.full_frame and evidence["route"] != "full_image":
        raise ValueError("Full-frame override not honored")
    if (evidence.get("input_width"), evidence.get("input_height")) != (width, height):
        raise ValueError("Evidence dimensions mismatch")
    for key in ("input_channels", "spatial_channels", "meta_channels", "global_tuple_palette_count"):
        if type(evidence.get(key)) is not int or evidence[key] < 0:
            raise ValueError(f"Invalid evidence counter {key}")
    if not 1 <= evidence["input_channels"] <= 4:
        raise ValueError("Invalid input channels")
    palettes = evidence.get("global_tuple_palettes")
    if not isinstance(palettes, list) or len(palettes) != evidence["global_tuple_palette_count"]:
        raise ValueError("Palette count mismatch")
    if evidence["route"] == "groupwise" and palettes:
        raise ValueError("Local palettes cannot be claimed as global tuples")
    for palette in palettes:
        if not isinstance(palette, dict):
            raise ValueError("Invalid palette entry")
        bounds = palette.get("input_channels")
        if (not isinstance(bounds, list) or len(bounds) != 2 or any(type(x) is not int for x in bounds)
                or not 0 <= bounds[0] < bounds[1] <= evidence["input_channels"]
                or bounds[1] - bounds[0] < 2):
            raise ValueError("Invalid tuple coverage")
        if any(type(palette.get(k)) is not int or palette[k] < 0 for k in ("colors", "deltas")):
            raise ValueError("Invalid accepted palette counts")
        if not isinstance(palette.get("predictor"), str) or not palette["predictor"]:
            raise ValueError("Missing palette predictor")


def add_comparisons(summaries):
    _comparisons({"baseline" if k == "default" else k: v for k, v in summaries.items()})


def aggregate(cases, modes):
    translated = [dict(pixels=c["pixels"], summaries={"baseline" if k == "default" else k: v
                  for k, v in c["summaries"].items()}) for c in cases]
    result = _aggregate(translated, ["baseline" if m == "default" else m for m in modes])
    return {"default" if k == "baseline" else k: v for k, v in result.items()}


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(), cases=[], errors=[],
                  timing="Exactly one native encode per process; excludes startup/input/output IO",
                  profile_timing="Inclusive per-thread scope totals; nested/parallel durations overlap, not wall time",
                  correctness="Original 8/16-bit PNG samples including hidden RGB; official djxl",
                  official_decoder_provenance="Caller-attested; path and SHA256 recorded",
                  receipt_scope=SCOPE, ratio_contract="mode / default; sums of per-case medians",
                  graph_evidence="Configured modes/accepted tuple counts are not execution evidence; separate profile audit required",
                  regression_policy="Size/time regressions reported, not failures")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            if args.profile:
                env["RS_JXL_PROFILE"] = "1"
                report["environment"]["explicit"]["RS_JXL_PROFILE"] = "1"
            for name in ("encoder", "decoder"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                setattr(args, name, path)
                digest = port.file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                identities.append((args.manifest, port.file_sha256(args.manifest)))
            with tempfile.TemporaryDirectory(prefix="palette-runs-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                sources = load_cases(args) if args.case or args.manifest else []
                external = set(sources)
                if args.synthetic:
                    sources += fixtures(temporary / "synthetic")
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    payload, expected, preparation = prepare_png(source, original, temporary)
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(payload)
                    if source in external:
                        identities.append((source, port.sha256(original)))
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=port.sha256(original), pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                preparation=preparation,
                                variants={mode: [] for mode in args.modes})
                    report["cases"].append(case)
                    for repeat in range(-args.warmups, args.repeats):
                        for mode in trial_order(args.modes, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][mode].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                jxl, png = Path(trial) / "out.jxl", Path(trial) / "out.png"
                                sample["encoder"] = port.run_process(command(args, snapshot, jxl, mode), env, args.timeout)
                                port.check_process(sample["encoder"])
                                sample["decoder"] = port.run_process(
                                    port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                port.check_process(sample["decoder"])
                                port.compare_pixels(expected, port.png_samples(png.read_bytes()))
                                measurement = json.loads(sample["encoder"]["stdout"])
                                validate_measurement(measurement, mode, jxl.stat().st_size,
                                                     expected.width, expected.height, args)
                                sample.update(verified=True, measurement=measurement)
                            print(f"{source.name} {mode} repeat={repeat}: {measurement['bytes']} bytes "
                                  f"{measurement['seconds']:.6f}s route="
                                  f"{measurement['palette_evidence']['route']} verified", flush=True)
                    case["summaries"] = {mode: summarize(samples, pixels)
                                         for mode, samples in case["variants"].items()}
                    add_comparisons(case["summaries"])
                report["aggregate"] = aggregate(report["cases"], args.modes)
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
