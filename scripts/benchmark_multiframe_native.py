"""Benchmark native multiframe encoding on small, externally supplied folders.

Each folder is one sequence, ordered naturally by filename. PNG/JPEG sources are
read and decoded once before encoding. All pixel modes receive the same native
arrays and supported sample/color metadata; no Pillow or subprocess codec is used.
One invocation loads one DLL. Run a separate invocation with --legacy --modes
crop auto to measure an older DLL without sending auto_options.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter


DEFAULT_MODES = ("crop", "auto-legacy", "auto-pruned", "auto-adaptive", "independent", "jpeg")
MODES = (*DEFAULT_MODES[:-2], "auto-retained", "auto-lookahead", "auto-multiscale", *DEFAULT_MODES[-2:])
AUTO_OPTIONS = {
    "auto-legacy": {"prune": False, "adaptive_rectangles": False},
    "auto-pruned": {"prune": True, "adaptive_rectangles": False},
    "auto-adaptive": {"prune": True, "adaptive_rectangles": True},
    "auto-retained": {"prune": True, "adaptive_rectangles": True, "retained_reference": True},
    "auto-lookahead": {"prune": True, "adaptive_rectangles": True,
                       "retained_reference": True, "reference_lookahead": True},
    "auto-multiscale": {"prune": True, "adaptive_rectangles": True,
                        "retained_reference": True, "reference_lookahead": True,
                        "extra_tile_sizes": [8, 16, 64]},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--folders", action="append", nargs="+", type=Path, default=[])
    parser.add_argument("--native", type=Path, help="DLL/shared library; otherwise RS_JXL_NATIVE_LIB or package discovery")
    parser.add_argument("--effort", type=int, default=9)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reference", choices=("auto", "previous", "first", "full"), default="auto")
    parser.add_argument("--modes", nargs="+", help="space/comma-separated modes: " + ", ".join(MODES) + "; auto aliases auto-legacy")
    parser.add_argument("--legacy", action="store_true", help="omit auto_options for an older native library")
    parser.add_argument("--out", type=Path, required=True, help="JSON/MD filename (writes both siblings), or report directory")
    args = parser.parse_args(argv)
    args.folders = [*args.paths, *(path for group in args.folders for path in group)]
    if not args.folders:
        parser.error("supply at least one folder, positionally or with --folders")
    if not 1 <= args.effort <= 9 or args.threads < 1:
        parser.error("effort must be 1..9 and threads must be positive")
    modes = args.modes or (["crop", "auto", "independent", "jpeg"] if args.legacy else DEFAULT_MODES)
    args.modes = []
    for group in modes:
        for mode in group.split(","):
            mode = mode.replace("_", "-")
            mode = "auto-legacy" if mode == "auto" else mode
            if mode not in MODES:
                parser.error(f"unknown mode: {mode}")
            if mode not in args.modes:
                args.modes.append(mode)
    if args.legacy and any(mode in ("auto-pruned", "auto-adaptive", "auto-retained", "auto-lookahead", "auto-multiscale") for mode in args.modes):
        parser.error("--legacy cannot measure pruned/adaptive/retained modes")
    if any(mode in ("auto-retained", "auto-lookahead", "auto-multiscale") for mode in args.modes) and args.reference != "auto":
        parser.error("auto-retained/auto-lookahead/auto-multiscale require --reference auto")
    return args


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", path.name)) + ((0, path.name),)


def pixel_settings(info):
    """Refuse metadata that the common still/multiframe array API cannot retain."""
    if info.get("orientation", 1) != 1:
        raise ValueError("source orientation must be 1 for common multiframe array metadata")
    settings = {"layout": "hwc", "bits_per_sample": info["bits_per_sample"]}
    if info.get("icc"):
        settings["icc_profile"] = bytes(info["icc"])
    else:
        color = info.get("color_encoding") or {}
        name = color.get("name")
        if name not in ("srgb", "linear_srgb"):
            # Native input metadata formats optional primaries with Rust Debug.
            primaries = str(color.get("primaries", "")).casefold()
            standard = (str(color.get("white_point", "")).casefold() == "d65"
                        and (color.get("color_space") == "gray"
                             or primaries in ("srgb", "some(srgb)")))
            name = {"srgb": "srgb", "linear": "linear_srgb"}.get(color.get("transfer_function")) if standard else None
        if name is None:
            raise ValueError("source color encoding is not representable by the array API; provide ICC-tagged or sRGB inputs")
        settings["color_encoding"] = name
    extras = info.get("extra_channels", [])
    if len(extras) > 1 or any(
        extra["type"] != "alpha" or extra.get("dim_shift", 0)
        or extra.get("alpha_premultiplied", False)
        or extra["bits_per_sample"] != info["bits_per_sample"]
        or extra.get("exponent_bits_per_sample", 0) != info.get("exponent_bits_per_sample", 0)
        for extra in extras
    ):
        raise ValueError("PNG/JPEG sequence requires ordinary interleaved alpha without additional planes")
    return settings


def load_folder(codec, folder, threads):
    folder = folder.expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError(f"not a folder: {folder}")
    paths = sorted((p for p in folder.iterdir() if p.is_file()
                    and p.suffix.lower() in (".png", ".jpg", ".jpeg")), key=natural_key)
    if not paths:
        raise ValueError(f"no PNG/JPEG inputs: {folder}")
    sources = []
    settings = None
    signature = None
    started = perf_counter()
    for path in paths:
        raw = path.read_bytes()
        pixels, info = codec.decode(raw, return_info=True, return_extra_channels=True,
                                    include_alpha_extra=True, threads=threads)
        current = pixel_settings(info)
        current_signature = (pixels.shape, pixels.dtype.str, current,
                             info.get("exponent_bits_per_sample", 0))
        if signature is not None and current_signature != signature:
            raise ValueError(f"geometry, sample type or color metadata differs: {path}")
        settings, signature = current, current_signature
        sources.append({"path": path, "raw": raw, "pixels": pixels, "info": info,
                        "jpeg": raw.startswith(b"\xff\xd8")})
    height, width = sources[0]["pixels"].shape[:2]
    return {"folder": str(folder), "sources": sources, "settings": settings,
            "width": width, "height": height, "input_seconds": perf_counter() - started}


def exact_array(actual, expected, label):
    # Byte equality includes hidden RGB, signed zero and floating-point NaN payloads.
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual.tobytes(order="C") != expected.tobytes(order="C")):
        raise ValueError(f"sample mismatch: {label}")


def verify_pixels(values, info, sources, animated):
    frames = info["frames"]
    if len(values) != len(sources) or len(frames) != len(sources):
        raise ValueError("decoded frame count does not match the original sequence")
    for actual, details, source in zip(values, frames, sources):
        label = source["path"].name
        exact_array(actual, source["pixels"], label)
        if animated and details["duration"] != 1:
            raise ValueError(f"duration mismatch: {label}")
        expected_extras = source["info"].get("extra_channels", [])
        actual_extras = details.get("extra_channels", [])
        if len(actual_extras) != len(expected_extras):
            raise ValueError(f"extra-channel count mismatch: {label}")
        for actual_extra, expected_extra in zip(actual_extras, expected_extras):
            for key in ("type", "bits_per_sample", "exponent_bits_per_sample", "dim_shift"):
                if actual_extra.get(key) != expected_extra.get(key):
                    raise ValueError(f"extra-channel {key} mismatch: {label}")
            exact_array(actual_extra["data"], expected_extra["data"], label + " extra plane")


def search_statistics(info, require_new=False):
    frames = info.get("frames", [])
    if require_new and (not frames or any(
        not {"candidates", "pruned", "analysis_seconds"} <= frame.keys() for frame in frames
    )):
        raise ValueError("native automatic-search statistics unavailable; use --legacy for an old DLL")
    completed = [candidate for frame in frames for candidate in frame.get("candidates", [])]
    pruned = [candidate for frame in frames for candidate in frame.get("pruned", [])]
    attempts = completed + pruned
    if require_new and any(not {"trial_pixels", "elapsed_seconds"} <= item.keys() for item in attempts):
        raise ValueError("native automatic-search work/timing statistics unavailable")
    return {
        "candidates_completed": len(completed) if frames else None,
        "candidates_pruned": len(pruned) if frames and all("pruned" in f for f in frames) else None,
        "trial_pixels": sum(c["trial_pixels"] for c in attempts) if attempts and all("trial_pixels" in c for c in attempts) else None,
        "analysis_seconds": sum(f["analysis_seconds"] for f in frames) if frames and all("analysis_seconds" in f for f in frames) else None,
        "candidate_seconds": sum(c["elapsed_seconds"] for c in attempts) if attempts and all("elapsed_seconds" in c for c in attempts) else None,
        **{key: sum(f[key] for f in frames) if frames and all(key in f for f in frames) else None
           for key in ("lookahead_comparisons", "reused_payloads", "reused_payload_pixels")},
    }


def measure(codec, dataset, mode, args):
    sources = dataset["sources"]
    if mode == "jpeg":
        sources = [source for source in sources if source["jpeg"]]
        if not sources:
            return {"mode": mode, "status": "skipped", "reason": "no JPEG inputs"}
    pixels = dataset["width"] * dataset["height"] * len(sources)
    common = {"effort": args.effort, "threads": args.threads}
    pixel_options = {**common, **dataset["settings"]}
    frames = [source["pixels"] for source in sources]
    options = {}
    if mode in AUTO_OPTIONS and not args.legacy:
        options["auto_options"] = AUTO_OPTIONS[mode].copy()
    info = {}
    started = perf_counter()
    if mode == "jpeg":
        encoded = [codec.transcode_jpeg(source["raw"], **common) for source in sources]
    elif mode == "independent":
        encoded = [codec.encode(frame, **pixel_options) for frame in frames]
    else:
        data, info = codec.encode_multiframe(
            frames, strategy="crop" if mode == "crop" else "auto",
            reference=args.reference, durations=1, tps=(1000, 1), loops=0,
            return_info=True, **pixel_options, **options,
        )
        encoded = [data]
    encode_seconds = perf_counter() - started
    started = perf_counter()
    if mode == "jpeg":
        decoded = [codec.reconstruct_jpeg(data, threads=args.threads) for data in encoded]
    else:
        decoded = [codec.decode(data, frame=None, return_info=True,
                                return_extra_channels=True, include_alpha_extra=True,
                                threads=args.threads, adjust_orientation=False) for data in encoded]
    decode_seconds = perf_counter() - started
    if mode == "jpeg":
        if any(actual != source["raw"] for actual, source in zip(decoded, sources)):
            raise ValueError("JPEG reconstruction differs from original bytes")
    elif mode == "independent":
        for (values, metadata), source in zip(decoded, sources):
            verify_pixels(values, metadata, [source], animated=False)
    else:
        verify_pixels(*decoded[0], sources, animated=True)
    total_bytes = sum(map(len, encoded))
    return {
        "mode": mode, "status": "passed", "frames": len(sources),
        "total_bytes": total_bytes, "bpp": total_bytes * 8 / pixels,
        "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
        "encode_mpix_s": pixels / 1e6 / encode_seconds,
        "decode_mpix_s": pixels / 1e6 / decode_seconds,
        "correctness": "original JPEG bytes" if mode == "jpeg" else "all original native samples",
        "auto_options": options.get("auto_options"),
        **search_statistics(info, require_new=mode in AUTO_OPTIONS and not args.legacy),
        "search_frames": info.get("frames", []),
    }


def markdown(report):
    lines = ["# Native Multiframe Benchmark", "", f"Status: {report['status']}", "",
             f"Native library: `{report['native']}`", "",
             f"Effort: {report['effort']}; threads: {report['threads']}; reference: {report['reference']}; legacy API: {report['legacy']}", "",
             "Times are public native API wall time, including Python packing, FFI and codec setup; "
             "source IO/initial decoding and correctness checks are excluded. No warmups or repeats.", "",
             "bpp = total bytes * 8 / (width * height * frame count). Independent modes sum complete "
             "still-image containers; JPEG rows use only JPEG inputs and decode time means byte reconstruction. "
             "Unavailable old-DLL statistics are shown as n/a.", ""]
    for dataset in report["datasets"]:
        lines += [f"## {Path(dataset['folder']).name}", ""]
        if "error" in dataset:
            lines += [dataset["error"], ""]
            continue
        input_seconds = dataset.get("input_seconds")
        input_time = "not measured" if input_seconds is None else f"{input_seconds:.6f} s"
        lines += [f"{dataset['width']} x {dataset['height']}, {dataset['frames']} frames; "
                  f"input read/decode: {input_time}", "",
                  "| Mode | Bytes | bpp | Encode s | Decode s | Encode MPix/s | Decode MPix/s | Completed | Pruned | Trial pixels | Analysis s | Status |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        notes = []
        for row in dataset["results"]:
            cells = [row["mode"]]
            for key in ("total_bytes", "bpp", "encode_seconds", "decode_seconds", "encode_mpix_s",
                        "decode_mpix_s", "candidates_completed", "candidates_pruned", "trial_pixels", "analysis_seconds"):
                value = row.get(key)
                cells.append("n/a" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value))
            cells.append(row["status"])
            lines.append("| " + " | ".join(cells) + " |")
            if row.get("error") or row.get("reason"):
                notes.append(f"{row['mode']}: {row.get('error', row.get('reason'))}")
        lines.extend(["", *notes, ""])
    return "\n".join(lines) + "\n"


def write_reports(report, out):
    if out.suffix.lower() in (".json", ".md"):
        json_path, md_path = out.with_suffix(".json"), out.with_suffix(".md")
    else:
        json_path, md_path = out / "report.json", out / "report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    return json_path, md_path


def main(argv=None):
    args = parse_args(argv)
    if args.native:
        os.environ["RS_JXL_NATIVE_LIB"] = str(args.native.expanduser().resolve(strict=True))
    # Select the DLL before importing/loading rs_jxl; never swap a live CFFI handle.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import rs_jxl
    from rs_jxl._ffi import _library_path

    report = {"created": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "native": str(_library_path()), "python_api_version": rs_jxl.__version__,
              "effort": args.effort, "threads": args.threads, "reference": args.reference,
              "legacy": args.legacy, "modes": args.modes, "datasets": []}
    for folder in args.folders:
        entry = {"folder": str(folder.expanduser().resolve()), "results": []}
        report["datasets"].append(entry)
        try:
            dataset = load_folder(rs_jxl, folder, args.threads)
            entry.update({key: dataset[key] for key in ("folder", "width", "height", "input_seconds")})
            entry["frames"] = len(dataset["sources"])
            entry["files"] = [source["path"].name for source in dataset["sources"]]
            entry["sample_dtype"] = dataset["sources"][0]["pixels"].dtype.str
            entry["bits_per_sample"] = dataset["settings"]["bits_per_sample"]
            for mode in args.modes:
                try:
                    row = measure(rs_jxl, dataset, mode, args)
                except Exception as error:
                    row = {"mode": mode, "status": "failed", "error": f"{type(error).__name__}: {error}"}
                entry["results"].append(row)
                if row["status"] == "failed":
                    report["status"] = "failed"
                print(f"{folder.name}: {mode}: {row['status']}", file=sys.stderr, flush=True)
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            report["status"] = "failed"
    paths = write_reports(report, args.out)
    print(*paths, sep="\n")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
