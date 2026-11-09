"""Sidecar patch_lab experiments, verified against original PNG samples by official djxl.

Native timing includes discovery, atlas and residual encoding; guarded includes
both complete encodes. Startup, input/output IO and verification are excluded.
Tiles default to cheap screened admission before encoding; force bypasses that
screen. Guarded is an explicit dual-encode diagnostic, excluded from defaults.
Size/time regressions are reported, not rejected. No builds or JXL hash checks.
selected_mode=source means an enabled detector candidate, not proof of an emitted
dictionary. source_detector_enabled is boolean for source and null otherwise.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
import zlib

from benchmark_fast_context import add_ratios, load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
import compare_cpp_port as port


MODES = ("baseline", "source", "components", "tiles", "guarded")
DEFAULT_MODES = ("baseline", "source", "components", "tiles")
ADMISSIONS = ("screened", "force")
COUNTERS = ("calls", "tiles_examined", "constant_tiles", "repeated_groups",
            "selected_groups", "positions", "covered_pixels", "atlas_pixels", "admission_rejected")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True, help="Prebuilt patch_lab executable")
    p.add_argument("--decoder", "--official-djxl", "--djxl", dest="decoder", type=Path, required=True,
                   help="Caller-attested official djxl executable")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--modes", nargs="+", choices=(*MODES, "off"), default=list(DEFAULT_MODES))
    p.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tile-size", type=int)
    p.add_argument("--tile-admission", choices=ADMISSIONS, default="screened")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300, help="Harness process guard only")
    p.add_argument("--out", type=Path, required=True, help="New JSON report file, never overwritten")
    args = p.parse_args(argv)
    args.modes = ["baseline" if mode == "off" else mode for mode in args.modes]
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not positive_number(args.timeout)
            or (args.tile_size is not None and args.tile_size < 1)):
        p.error("Invalid threads/tile-size/repeats/warmups/timeout")
    if len(set(args.modes)) != len(args.modes) or "baseline" not in args.modes:
        p.error("Use distinct modes including baseline (off is its alias)")
    if not args.case and args.manifest is None and not args.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if args.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return args


def command(args, source, output, mode):
    if mode not in (*MODES, "off"):
        raise ValueError(f"Unknown patch mode: {mode}")
    native_mode = "off" if mode == "baseline" else mode
    result = [str(args.encoder), str(source), "--output", str(output),
              "--effort", str(args.effort), "--threads", str(args.threads), "--mode", native_mode]
    if mode in ("tiles", "guarded"):
        result += ["--tile-admission", args.tile_admission]
        if args.tile_size is not None:
            result += ["--tile-size", str(args.tile_size)]
    return result


def fixtures(folder):
    """Eight format repeats, two controls, and one three-group RGB repeat case."""
    folder.mkdir()
    paths = []
    rng = random.Random(1701)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    for depth, color, pattern in ([(d, c, "tiles") for d in (8, 16) for c in (0, 4, 2, 6)]
                                  + [(8, 2, "natural-like"), (16, 6, "noise"), (8, 2, "tiles-multi")]):
        channels = {0: 1, 4: 2, 2: 3, 6: 4}[color]
        width, height = (69, 53) if pattern.startswith("tiles") else (97, 71)
        top = (1 << depth) - 1
        raw = bytearray()
        for y in range(height):
            row = []
            for x in range(width):
                # Full 16x16 tiles repeat exactly, including every alpha sample.
                # Partial edges deliberately retain different, nonzero data.
                u, v = (x % 16, y % 16) if x < 64 and y < 48 else (x, y)
                for c in range(channels):
                    if pattern.startswith("tiles"):
                        word = (u * 0x45d9f3b) ^ (v * 0x119de1f3) ^ ((c + 1) * 0x9e3779b9)
                        value = ((word ^ (word >> 16)) * 0x45d9f3b) & top
                        if pattern == "tiles-multi":
                            # Three distinct textures, each present in four full tiles.
                            group = (x // 16 + y // 16) % 3
                            value = (value + group * 67) & top
                    elif pattern == "natural-like":
                        value = int(top * (0.45 + 0.2 * math.sin(x / 21 + c)
                                           + 0.15 * math.cos(y / 17 + c)
                                           + 0.08 * math.sin((x + y) / 9)))
                    else:
                        value = rng.randrange(top + 1)
                    if color in (4, 6) and c == channels - 1 and (u + v) % 5 == 0:
                        value = 0
                    row.append(value)
            raw.append(0)
            raw.extend(struct.pack(">" + ("B" if depth == 8 else "H") * len(row), *row))
        header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0)
        data = (port.PNG_SIGNATURE + chunk(b"IHDR", header)
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        path = folder / f"{pattern}-{depth}-{color}.png"
        path.write_bytes(data)
        paths.append(path)
    return paths


def nonnegative_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_measurement(m, mode, size, width, height, tile_size=16, tile_admission="screened"):
    if not isinstance(m, dict) or not positive_number(m.get("seconds")):
        raise ValueError("Missing/invalid native encoder time")
    if type(m.get("bytes")) is not int or m["bytes"] <= 0 or m["bytes"] != size:
        raise ValueError("Reported size differs from actual nonempty codestream")
    if width <= 0 or height <= 0 or (m.get("width"), m.get("height")) != (width, height):
        raise ValueError("Invalid/mismatched image dimensions")
    pixels = width * height
    for key, expected in (("bpp", size * 8 / pixels),
                          ("mpix_per_second", pixels / 1e6 / m["seconds"])):
        if not positive_number(m.get(key)) or not math.isclose(m[key], expected, rel_tol=1e-5):
            raise ValueError(f"Missing/inconsistent {key}")
    native_mode = "off" if mode == "baseline" else mode
    if mode not in (*MODES, "off") or m.get("mode") != native_mode:
        raise ValueError("Requested mode differs from encoder receipt")
    expected_admission = tile_admission if mode in ("tiles", "guarded") else None
    if (tile_admission not in ADMISSIONS or "tile_admission" not in m
            or m["tile_admission"] != expected_admission):
        raise ValueError("Missing/mismatched tile admission policy")
    if "tile_work" not in m or "guard_work" not in m:
        raise ValueError("Missing work receipt (inactive receipts must be null)")
    if "source_detector_enabled" not in m:
        raise ValueError("Missing source detector activation receipt")
    source_enabled = m["source_detector_enabled"]
    if mode == "source":
        if type(source_enabled) is not bool:
            raise ValueError("Source detector activation must be boolean")
    elif source_enabled is not None:
        raise ValueError("Inactive source detector activation must be null")
    work = m["tile_work"]
    if mode not in ("tiles", "guarded"):
        selected_mode = "off" if mode == "source" and not source_enabled else native_mode
        if work is not None or m.get("selected_mode") != selected_mode:
            raise ValueError("Unexpected detector or selected mode")
    else:
        if not isinstance(work, dict) or work.get("tile_size") != tile_size:
            raise ValueError("Missing/mismatched tile work receipt")
        if any(type(work.get(k)) is not int or work[k] < 0 for k in COUNTERS):
            raise ValueError("Invalid tile work counter")
        if (work["calls"] == 0 or work["constant_tiles"] > work["tiles_examined"]
                or work["selected_groups"] > work["repeated_groups"]
                or not nonnegative_number(work.get("seconds"))
                or work["seconds"] > m["seconds"] + 1e-6):
            raise ValueError("Inconsistent tile work counters/time")
        planned = work["selected_groups"] > 0
        rejected = work["admission_rejected"] > 0
        saved = work.get("estimated_saved_bits")
        if (type(saved) not in (int, float) or not math.isfinite(saved)
                or not nonnegative_number(work.get("estimated_baseline_bits"))):
            raise ValueError("Invalid tile cost estimates")
        if (work["admission_rejected"] > work["calls"]
                or (rejected and (tile_admission == "force" or planned or work["repeated_groups"] == 0))):
            raise ValueError("Inconsistent tile admission rejection")
        note = ("rejected_by_estimate" if rejected else "plan_selected" if planned else
                "no_selection" if work["repeated_groups"] else "no_repeated_tiles")
        if work.get("plan_selected") is not planned or work.get("note") != note:
            raise ValueError("Tile plan selection receipt is inconsistent")
        if planned != all(work[k] > 0 for k in ("positions", "covered_pixels", "atlas_pixels")):
            raise ValueError("Tile selection lacks positions/coverage/atlas")
        if not planned and any(work[k] != 0 for k in ("positions", "covered_pixels", "atlas_pixels")):
            raise ValueError("No-plan receipt contains selected work")
        selected = m.get("selected_mode")
        if selected not in ("off", "tiles") or (not planned and selected != "off"):
            raise ValueError("Tile selected mode disagrees with plan")
        if mode == "tiles" and selected != ("tiles" if planned else "off"):
            raise ValueError("Unguarded tile mode did not use its selected plan")
    guard = m["guard_work"]
    if mode != "guarded":
        if guard is not None:
            raise ValueError("Unexpected guarded dual encode")
        return
    if (not isinstance(guard, dict)
            or any(type(guard.get(k)) is not int or guard[k] <= 0 for k in ("off_bytes", "tiles_bytes"))
            or any(not positive_number(guard.get(k)) for k in ("off_seconds", "tiles_seconds"))):
        raise ValueError("Missing/invalid guarded dual encode receipt")
    expected_mode = "tiles" if guard["tiles_bytes"] < guard["off_bytes"] and work["plan_selected"] else "off"
    if (size != min(guard["off_bytes"], guard["tiles_bytes"])
            or not math.isclose(m["seconds"], guard["off_seconds"] + guard["tiles_seconds"], rel_tol=1e-6)
            or m["selected_mode"] != expected_mode):
        raise ValueError("Guard must return min bytes and include both full encode times")


def add_comparisons(summaries):
    add_ratios(summaries)
    for summary in summaries.values():
        summary["regressions"] = {key: summary["ratios_to_baseline"][key] > 1
                                  for key in ("bytes", "seconds")}


def aggregate(cases, modes):
    if not cases or not modes or "baseline" not in modes:
        raise ValueError("Cannot aggregate without cases and baseline")
    if any(not positive_number(case["pixels"]) for case in cases):
        raise ValueError("Aggregate requires positive pixel counts")
    pixels = sum(case["pixels"] for case in cases)
    result = {}
    for mode in modes:
        summaries = [case["summaries"][mode] for case in cases]
        if any(not positive_number(s[k]) for s in summaries for k in ("bytes", "seconds")):
            raise ValueError("Aggregate requires positive sizes and durations")
        size, seconds = (sum(s[k] for s in summaries) for k in ("bytes", "seconds"))
        result[mode] = dict(bytes=size, seconds=seconds, bpp=8 * size / pixels,
                            mpix_per_second=pixels / 1e6 / seconds)
    add_comparisons(result)
    return result


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(), cases=[], errors=[],
                  timing="Native discovery+admission+encode; explicit guarded sums both full encodes; no startup/IO",
                  correctness="Official djxl; original 8/16-bit PNG samples including alpha=0 colors",
                  official_decoder_provenance="Caller-attested; executable path and SHA256 recorded",
                  regression_policy="Size/time regressions reported, not failures",
                  ratio_contract="mode / baseline; corpus ratios use sums of per-case medians")
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
                identities.append((args.manifest, port.file_sha256(args.manifest)))
            with tempfile.TemporaryDirectory(prefix="tile-patches-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                sources = load_cases(args) if args.case or args.manifest else []
                external = set(sources)
                if args.synthetic:
                    sources += fixtures(temporary / "synthetic")
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    expected = port.png_samples(original)
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(port.png_payload(original)[0])
                    if source in external:
                        identities.append((source, port.sha256(original)))
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=port.sha256(original), pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
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
                                                     expected.width, expected.height, args.tile_size or 16,
                                                     args.tile_admission)
                                sample.update(verified=True, measurement=measurement)
                            print(f"{source.name} {mode} repeat={repeat}: {measurement['bytes']} bytes "
                                  f"{measurement['seconds']:.6f}s selected={measurement['selected_mode']} verified",
                                  flush=True)
                    case["summaries"] = {mode: summarize(samples, pixels)
                                         for mode, samples in case["variants"].items()}
                    add_comparisons(case["summaries"])
                report["aggregate"] = aggregate(report["cases"], args.modes)
                print(json.dumps(report["aggregate"], allow_nan=False), flush=True)
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
