"""Compare complete E9 Patch outputs with two independent pixel decoders."""
import argparse
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_WORK_DIR = ROOT / "out" / "patch-near-20261125"
sys.path.insert(0, str(ROOT / "tools"))
import compare_cpp_port as port
import numpy as np
from PIL import Image

MODES = {
    "ordinary": ["--mode", "ordinary"],
    "off": ["--mode", "off"],
    "d1": ["--mode", "source"],
    "d2": ["--mode", "source", "--source-fingerprint-divisor", "2"],
    "d4": ["--mode", "source", "--source-fingerprint-divisor", "4"],
    "near": ["--mode", "source", "--near-match"],
    "near-d2": ["--mode", "source", "--near-match", "--source-fingerprint-divisor", "2"],
    "near-d4": ["--mode", "source", "--near-match", "--source-fingerprint-divisor", "4"],
    "components": ["--mode", "components"],
    "grid16": ["--mode", "tiles", "--tile-admission", "force"],
    "offgrid16": ["--mode", "tiles", "--tile-admission", "force", "--off-grid"],
    "grid16+expand": ["--mode", "tiles", "--tile-admission", "force", "--expand-rectangles"],
    "offgrid16+expand": ["--mode", "tiles", "--tile-admission", "force", "--off-grid", "--expand-rectangles"],
}
EXPERIMENTS = {"near", "near-d2", "near-d4", "grid16+expand", "offgrid16+expand"}


def args_from_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--frozen-lab", type=Path, help="Audit identical output of frozen d1/off")
    parser.add_argument("--official", type=Path, default=ROOT / "out/cpp-b866b3d/libjxl/tools/djxl.exe")
    parser.add_argument("--rust", type=Path, default=ROOT / "target/release/rs-jxl.exe")
    parser.add_argument("--png-reader", type=Path, help="FFmpeg for exact 16-bit PNG sample I/O")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--manifest", type=Path, help="Input manifest, default WORK_DIR/manifest.json")
    parser.add_argument("--out", type=Path, help="New report, default WORK_DIR/report.json")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--cohorts", nargs="+", choices=["known", "holdout", "synthetic"],
                        default=["known", "holdout", "synthetic"])
    parser.add_argument("--case", action="append", help="Select a manifest label or label prefix")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    args.manifest = args.manifest or args.work_dir / "manifest.json"
    args.out = args.out or args.work_dir / "report.json"
    if args.out.suffix != ".json":
        parser.error("Report path must end in .json")
    if not (1 <= args.threads <= 256 and args.repeats > 0 and args.timeout > 0):
        parser.error("Invalid threads, repeats or timeout")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Specify distinct modes")
    if args.out.exists() or args.out.with_suffix(".jsonl").exists():
        parser.error("Report or checkpoint already exists")
    return args


def png_samples(blob, path, png_reader, timeout):
    _, info = port.png_payload(blob)
    width, height, depth = info["width"], info["height"], info["bit_depth"]
    if depth == 8:
        with Image.open(BytesIO(blob)) as image:
            rgba = np.asarray(image.convert("RGBA"))
    else:
        result = subprocess.run(
            [str(png_reader), "-v", "error", "-i", str(path), "-map", "0:v:0",
             "-frames:v", "1", "-threads", "1", "-f", "rawvideo",
             "-pix_fmt", "rgba64le", "-"],
            capture_output=True, timeout=timeout, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode or len(result.stdout) != width * height * 8:
            raise ValueError(f"16-bit PNG output invalid: {result.stderr.decode(errors='replace')}")
        rgba = np.frombuffer(result.stdout, dtype="<u2").reshape(height, width, 4)
    return port.Pixels(width, height, depth, rgba)


def verify(original, restored):
    if (original.width, original.height, original.depth) != (
            restored.width, restored.height, restored.depth):
        raise ValueError("Decoded dimensions or bit depth differ from source")
    sample_errors = [0] * 4
    pixel_errors = max_delta = hidden_errors = hidden_samples = 0
    for a, b in zip(original.rgba, restored.rgba):
        mismatched = a != b
        hidden = a[:, 3] == 0
        sample_errors = [i + int(n) for i, n in zip(sample_errors, mismatched.sum(axis=0))]
        pixel_errors += int(mismatched.any(axis=1).sum())
        hidden_samples += int(hidden.sum()) * 3
        hidden_errors += int(mismatched[hidden, :3].sum())
        if mismatched.any():
            max_delta = max(max_delta, int(np.abs(a.astype("int32") - b.astype("int32")).max()))
    return {"width": restored.width, "height": restored.height, "bit_depth": restored.depth,
            "pixel_errors": pixel_errors, "sample_errors_rgba": sample_errors,
            "hidden_rgb_samples": hidden_samples, "hidden_rgb_errors": hidden_errors,
            "max_abs_delta": max_delta}


def encode(args, binary, run, source, output, mode):
    command = [str(binary), str(source), "--output", str(output), "--effort", "9",
               "--threads", str(args.threads), *MODES[mode]]
    result = run(command)
    record = json.loads(result["stdout"])
    if record["bytes"] != output.stat().st_size or record["mode"] != MODES[mode][1]:
        raise ValueError(f"Mismatched lab receipt {mode}")
    if not math.isfinite(record["seconds"]) or record["seconds"] <= 0:
        raise ValueError(f"Invalid native encode duration {mode}")
    if (mode in ("d2", "d4", "near-d2", "near-d4") and record["source_detector_enabled"]
            and record.get("fingerprint_divisor") != int(mode[-1])):
        raise ValueError(f"Mismatched fingerprint divisor for {mode}")
    if mode in ("near", "near-d2", "near-d4") and record.get("near_match") is not True:
        raise ValueError("Missing requested near-match receipt")
    if mode.endswith("+expand") and not (record.get("tile_work") or {}).get("expand_rectangles"):
        raise ValueError(f"Missing rectangle expansion receipt for {mode}")
    return record


def session(args, row, temp, env):
    source = Path(row["original"])
    if not source.is_absolute():
        source = args.manifest.resolve().parent / source
    blob = source.read_bytes()
    if hashlib.sha256(blob).hexdigest() != row["sha256"]:
        raise ValueError("Frozen input hash mismatch")
    clean, _ = port.png_payload(blob)
    input_png = temp / "source.png"
    input_png.write_bytes(clean)
    expected = png_samples(clean, input_png, args.png_reader, args.timeout)
    output = temp / "sample.jxl"
    row["results"] = {}

    def run(command):
        result = port.run_process(command, env, args.timeout)
        port.check_process(result)
        return result

    for repeat in range(args.repeats):
        # Rotate order for repeated experiments to reduce systematic drift.
        for mode in args.modes[repeat % len(args.modes):] + args.modes[:repeat % len(args.modes)]:
            trials = row["results"].setdefault(mode, [])
            trial = {"repetition": repeat + 1}
            trials.append(trial)
            try:
                trial["measurement"] = measurement = encode(args, args.lab, run, input_png, output, mode)
                if (measurement["width"], measurement["height"]) != (expected.width, expected.height):
                    raise ValueError("Lab encoded unexpected dimensions")
                trial["sha256"] = port.file_sha256(output)
                trial["decoders"] = {}
                cpp, rust = temp / "official.png", temp / "rust.png"
                for name, png, command in [
                    ("official", cpp, port.decoder_command(args.official, output, cpp, expected.depth)),
                    ("rust", rust, [str(args.rust), "decode", str(output), str(rust),
                                    "--bits", str(expected.depth), "--threads", "1", "--quiet"]),
                ]:
                    try:
                        timing = run(command)
                        decoded = png.read_bytes()
                        _, decoded_header = port.png_payload(decoded)
                        stats = verify(expected, png_samples(decoded, png, args.png_reader, args.timeout))
                        stats["png_color_type"] = decoded_header["color_type"]
                        stats["seconds"] = timing["wall_seconds"]
                        trial["decoders"][name] = stats
                        if stats["pixel_errors"]:
                            raise ValueError(f"{name} decoded {stats['pixel_errors']} unequal pixels")
                        if row["color_type"] in (4, 6) and decoded_header["color_type"] not in (4, 6):
                            raise ValueError(f"{name} dropped explicit alpha")
                    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                        trial["decoders"].setdefault(name, {})["error"] = str(error)
                    finally:
                        png.unlink(missing_ok=True)
                regressed = [n for n, c in trial["decoders"].items() if "error" in c]
                if regressed:
                    raise ValueError("Decoder verification failed for " + ", ".join(regressed))
                if args.frozen_lab and repeat == 0 and mode in ("d1", "off"):
                    frozen = temp / "frozen.jxl"
                    trial["frozen_lab"] = encode(args, args.frozen_lab, run, input_png, frozen, mode)
                    trial["frozen_sha256"] = port.file_sha256(frozen)
                    frozen.unlink()
                    trial["frozen_parity"] = trial["frozen_sha256"] == trial["sha256"]
                    if not trial["frozen_parity"]:
                        raise ValueError("Frozen native output differs")
                if trial["sha256"] != trials[0].get("sha256"):
                    raise ValueError("Repeated output is not byte stable")
            except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                trial["error"] = str(error)
            finally:
                output.unlink(missing_ok=True)


def aggregate(cases, mode, repeats):
    bytes_ = []
    seconds = []
    pixels = []
    deltas = []
    patch_active = 0
    for row in cases:
        trial = row.get("results", {}).get(mode, [])
        if len(trial) != repeats or any("error" in t for t in trial):
            return None
        measurements = [t["measurement"] for t in trial]
        patch_active += int(measurements[0].get("source_patch_selected") is True or
                            (measurements[0].get("tile_work") or {}).get("selected_groups", 0) > 0)
        bytes_.append(measurements[0]["bytes"])
        seconds.append(statistics.median(m["seconds"] for m in measurements))
        pixels.append(row["width"] * row["height"])
        off = row["results"].get("off", [])
        if off and "error" not in off[0]:
            deltas.append(bytes_[-1] - off[0]["measurement"]["bytes"])
    return {"images": len(cases), "bytes": sum(bytes_), "seconds": sum(seconds),
            "mpix_per_second": sum(pixels) / 1e6 / sum(seconds),
            "mean_bpp": statistics.mean(8*b/n for b, n in zip(bytes_, pixels)),
            "weighted_bpp": 8 * sum(bytes_) / sum(pixels),
            "patch_active_images": None if mode == "components" else patch_active,
            "vs_off": ({"win": sum(d < 0 for d in deltas), "tie": sum(d == 0 for d in deltas),
                        "loss": sum(d > 0 for d in deltas), "max_delta_bytes": max(deltas)}
                       if deltas else None)}


def offline_comparison(cases, modes, totals):
    controls = [mode for mode in modes if mode not in EXPERIMENTS]
    if not controls or len(controls) == len(modes) or any(totals[m] is None for m in modes):
        return None
    old_bytes = new_bytes = improved = 0
    for case in cases:
        sizes = {m: case["results"][m][0]["measurement"]["bytes"] for m in modes}
        old = min(sizes[m] for m in controls)
        new = min(sizes.values())
        old_bytes += old
        new_bytes += new
        improved += new < old
    return {"controls": controls, "controls_per_image_min_bytes": old_bytes,
            "all_per_image_min_bytes": new_bytes, "improved_images": improved,
            "all_candidates_serial_encode_seconds": sum(totals[m]["seconds"] for m in modes)}


def main():
    args = args_from_cli()
    env, env_config = port.clean_environment(os.environ)
    frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [c for c in frozen["cases"] if c["cohort"] in args.cohorts and
             (not args.case or any(c["label"].startswith(p) for p in args.case))]
    if not cases:
        raise ValueError("No matching cases")
    if any(c["bit_depth"] == 16 for c in cases) and args.png_reader is None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise ValueError("16-bit PNG verification requires FFmpeg or --png-reader")
        args.png_reader = Path(executable)
    identities = {}
    for name in ("lab", "frozen_lab", "official", "rust", "manifest", "png_reader"):
        path = getattr(args, name)
        if path is not None:
            path = path.resolve(strict=True)
            setattr(args, name, path)
            identities[name] = {"path": str(path), "sha256": port.file_sha256(path)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with args.out.with_suffix(".jsonl").open("x", encoding="utf-8") as checkpoint:
        for row in cases:
            with tempfile.TemporaryDirectory(dir=args.out.resolve().parent, prefix="trial-") as directory:
                try:
                    session(args, row, Path(directory), env)
                except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                    row["error"] = str(error)
            print(json.dumps(row, ensure_ascii=False), file=checkpoint, flush=True)
            size_map = {m: (t[0].get("error") or t[0]["measurement"]["bytes"])
                        for m, t in row.get("results", {}).items()}
            print(row["label"], row.get("error") or size_map, flush=True)
            failed |= "error" in row or any("error" in t for trials in row.get("results", {}).values()
                                            for t in trials)
    summary, comparisons = {}, {}
    for cohort in ["all", "real", *args.cohorts]:
        subset = [c for c in cases if cohort in ("all", c["cohort"])
                  or (cohort == "real" and c["cohort"] != "synthetic")]
        if subset:
            summary[cohort] = {m: aggregate(subset, m, args.repeats) for m in args.modes}
            comparisons[cohort] = offline_comparison(subset, args.modes, summary[cohort])
    with args.out.open("x", encoding="utf-8") as report:
        json.dump({"schema": 1, "status": "failed" if failed else "passed",
                   "config": vars(args), "identities": identities,
                   "machine": {**platform.uname()._asdict(), "cpu": os.environ.get("PROCESSOR_IDENTIFIER"),
                               "logical_cpus": os.cpu_count()},
                   "environment": env_config, "seconds": "native encoder time; median per image",
                   "summary": summary, "offline_comparison": comparisons, "cases": cases},
                  report, default=str, ensure_ascii=False, indent=2)
        report.write("\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
