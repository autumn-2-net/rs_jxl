"""Measure JPEG-transcoded JXL -> pixels, NOT original-JPEG reconstruction.

Uses existing verified JPEG/JXL pairs from compare_decoder_checkpoints reports.
Pixel differences are reported exactly, never relabeled as lossless equality.
Temporary PAM/PFM/JPEG output is used only for validation, not timed runs.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import tempfile

import numpy as np

from compare_cpp_port import binary_info, check_process, clean_environment, file_sha256, run_process
from verify_decoder_features import portable


def command(name, binary, source, threads, *, output=None, bits=32, reps=1, jpeg=False,
            dither=None):
    if name == "official" and output is None and not jpeg and bits != 32:
        raise ValueError("djxl --disable_output selects F32 regardless of --bits_per_sample")
    args = [str(binary)] + (["decode"] if name != "official" else []) + [str(source)]
    if output is not None:
        args.append(str(output))
    args += [f"--num_threads={threads}", "--quiet"]
    if jpeg:
        args.append("--reconstruct_jpeg")
    else:
        args.append(f"--bits_per_sample={bits}")
        if name == "rust" and dither is not None:
            args += ["--dither", dither]
    if output is None:
        args += ["--disable_output", f"--num_reps={reps}"]
    return args


def differences(actual, expected, *, expected_name="official"):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"Pixel shape/type mismatch: {actual.shape}/{actual.dtype} vs "
                         f"{expected.shape}/{expected.dtype}")
    if actual.ndim != 3 or actual.shape[-1] not in (1, 3):
        raise ValueError("Expected one gray/RGB frame, not alpha or extra planes")
    floating = actual.dtype == np.float32
    maximum, absolute, squared = 0.0, 0.0, 0.0
    changed_samples, changed_pixels, nonfinite = 0, 0, 0
    first = None
    for y in range(actual.shape[0]):
        a, b = actual[y], expected[y]
        unequal = (a.view(np.uint32) != b.view(np.uint32)) if floating else (a != b)
        changed_samples += int(np.count_nonzero(unequal))
        changed_pixels += int(np.count_nonzero(np.any(unequal, axis=-1)))
        finite = np.isfinite(a) & np.isfinite(b)
        nonfinite += int(np.count_nonzero(~finite))
        delta = a[finite].astype(np.float64) - b[finite].astype(np.float64)
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
        absolute += float(np.sum(np.abs(delta)))
        squared += float(np.sum(delta * delta))
        if first is None and np.any(unequal):
            x, channel = np.argwhere(unequal)[0]
            first = dict(x=int(x), y=y, channel=int(channel), rust=float(a[x, channel]))
            first[expected_name] = float(b[x, channel])
    if nonfinite:
        raise ValueError(f"Unexpected nonfinite pixel samples: {nonfinite}")
    pixels = actual.shape[0] * actual.shape[1]
    return dict(exact=changed_samples == 0, samples=int(actual.size), pixels=pixels,
                channels=actual.shape[-1], different_samples=changed_samples,
                different_pixels=changed_pixels, different_pixel_percent=100 * changed_pixels / pixels,
                max_abs_error=maximum, mean_abs_error=absolute / actual.size,
                rmse=(squared / actual.size) ** 0.5, first_difference=first)


def markdown(report):
    sample_type = "uint8" if report.get("config", {}).get("bits", 32) == 8 else "float32"
    timed_format = ("float32 pixels" if sample_type == "float32" else
                    "mixed sample types (Rust U8, official F32; NOT COMPARABLE)")
    lines = ["# JPEG-Transcoded JXL Pixel Decoding", "", f"Status: {report['status']}", "",
             f"Timed: JXL -> {timed_format}, one thread, no output-file encoding/writes. "
             "Original JPEG reconstruction is verified separately and is NOT the timed path.",
             "Wall/reps includes amortized startup, input IO, allocation and decode. "
             "Same color/orientation defaults, no added destination color profile.", "",
             f"Rust output dither: {report.get('config', {}).get('dither') or 'binary default'}. "
             "This selection does not change F32 samples or original JPEG bytes.", "",
             "| Input | MPix | bpp | Previous Rust ms | Rust ms | Official ms | Rust MPix/s | Official MPix/s | "
             "U8 changed pixels | U8 max error | F32 max error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["results"]:
        r, o = row["timing"]["rust"], row["timing"]["official"]
        u, f = row["u8"], row["f32"]
        before = row["timing"].get("before")
        before_ms = f"{before['seconds'] * 1000:.3f}" if before else "-"
        lines.append(f"| {row['case']['label']} | {row['mpix']:.4f} | {row['bpp']:.4f} | "
                     f"{before_ms} | {r['seconds'] * 1000:.3f} | {o['seconds'] * 1000:.3f} | "
                     f"{r['mpix_s']:.3f} | {o['mpix_s']:.3f} | "
                     f"{u['different_pixels']} ({u['different_pixel_percent']:.5f}%) | "
                     f"{u['max_abs_error']:.0f} | {f['max_abs_error']:.9g} |")
    if report["results"]:
        mpix = sum(row["mpix"] for row in report["results"])
        lines += ["", "## Aggregate", ""]
        for name in report["results"][0]["timing"]:
            seconds = sum(row["timing"][name]["seconds"] for row in report["results"])
            lines.append(f"- {name}: {seconds:.6f} s; {mpix / seconds:.3f} MPix/s.")
    lines += ["", "## Interpretation", "",
              "Pixel equality is reported independently for U8 and F32. A nonzero difference "
              "is not an exact match, even when visually negligible. Byte-exact JPEG "
              "reconstruction preserves compressed coefficients/metadata, not a unique "
              "choice of IDCT, upsampling, color conversion and final pixel rounding.",
              "This bounded corpus is not full conformance. Float equality counts IEEE words; "
              "U8 metrics include every channel without rescaling or alignment correction.", "",
              "All commands, hashes, raw trials and numerical comparisons are in report.json.",
              "No new encoding, model training, builds or network access occur in this tool."]
    if report.get("error"):
        lines += ["", "## Failure", "", report["error"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--before", type=Path,
                        help="Optional older Rust checkpoint, using its default dither, in alternating timing")
    parser.add_argument("--djxl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(32,), default=32,
                        help="Only F32 is comparable: djxl --disable_output forces F32; U8 files are always verified separately")
    parser.add_argument("--dither", choices=("jxl-rs", "libjxl"),
                        help="Rust U8 output dither; omit for compatibility with older binaries")
    parser.add_argument("--case-label", action="append",
                        help="Select a JPEG baseline label; repeat for multiple cases")
    parser.add_argument("--decode-reps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if min(args.decode_reps, args.repeats) < 1 or args.warmups < 0 or not 0 < args.timeout < float("inf"):
        parser.error("Invalid repetition or timeout settings")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline["status"] != "passed":
        parser.error("Baseline must be a completed verified report")
    cases = [case for case in baseline["cases"] if case["kind"] == "jpeg"]
    if not cases:
        parser.error("No JPEG cases in baseline")
    if args.case_label:
        selected = set(args.case_label)
        unknown = selected - {case["label"] for case in cases}
        if unknown:
            parser.error(f"Unknown JPEG case labels: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["label"] in selected]
    args.out.mkdir(parents=True, exist_ok=False)
    env, environment = clean_environment(os.environ)
    report = dict(status="running", created=datetime.now(timezone.utc).isoformat(),
                  config=vars(args), environment=environment, binaries={}, trials=[], results=[])

    def execute(argv, **labels):
        trial = dict(command=argv, **labels)
        report["trials"].append(trial)
        trial.update(run_process(argv, env, args.timeout))
        check_process(trial)
        return trial["wall_seconds"]

    try:
        binaries = {"rust": args.rust.resolve(strict=True), "official": args.djxl.resolve(strict=True)}
        if args.before:
            binaries["before"] = args.before.resolve(strict=True)
        for name, binary in binaries.items():
            report["binaries"][name] = binary_info(binary, env, 10)
            check_process(report["binaries"][name]["version_probe"])
        with tempfile.TemporaryDirectory(prefix="pixels-", dir=args.out) as temp:
            folder = Path(temp)
            for index, case in enumerate(cases):
                jxl, original = Path(case["jxl"]), Path(case["original"])
                for path, digest in [(jxl, case["jxl_sha256"]), (original, case["original_sha256"])]:
                    if file_sha256(path) != digest:
                        raise ValueError(f"Input changed since baseline: {path}")
                comparisons = {}
                for name, binary in binaries.items():
                    output = folder / f"{name}.jpg"
                    execute(command(name, binary, jxl, 1, output=output, jpeg=True),
                            phase="verify_jpeg", case=index, decoder=name)
                    if output.read_bytes() != original.read_bytes():
                        raise ValueError(f"JPEG byte reconstruction mismatch: {case['label']}/{name}")
                    output.unlink()
                for bits, extension, key in [(8, "pam", "u8"), (32, "pfm", "f32")]:
                    arrays = {}
                    for name, binary in binaries.items():
                        output = folder / f"{name}.{extension}"
                        execute(command(name, binary, jxl, 1, output=output, bits=bits,
                                        dither=args.dither),
                                phase="verify_pixels", bits=bits, case=index, decoder=name)
                        arrays[name] = portable(output)
                        output.unlink()
                    if arrays["rust"].shape[:2] != (case["height"], case["width"]):
                        raise ValueError("Unexpected dimensions/orientation; not silently realigning")
                    if arrays["rust"].dtype != (np.uint8 if bits == 8 else np.float32):
                        raise ValueError("Unexpected sample depth")
                    comparisons[key] = differences(arrays["rust"], arrays["official"])
                    if "before" in arrays:
                        comparisons["before_" + key] = differences(
                            arrays["rust"], arrays["before"], expected_name="before")
                    del arrays
                seconds = {name: [] for name in binaries}
                for repeat in range(-args.warmups, args.repeats):
                    names = list(binaries)
                    if len(names) == 2:
                        if (index + repeat) % 2:
                            names.reverse()
                    else:
                        start = (index + repeat) % len(names)
                        names = names[start:] + names[:start]
                    for name in names:
                        elapsed = execute(command(name, binaries[name], jxl, 1,
                                                  bits=args.bits, reps=args.decode_reps,
                                                  dither=args.dither),
                                          phase="timing", case=index, decoder=name, repeat=repeat,
                                          warmup=repeat < 0, reps=args.decode_reps, bits=args.bits)
                        if repeat >= 0:
                            seconds[name].append(elapsed / args.decode_reps)
                mpix = case["pixels"] / 1e6
                timing = {name: dict(seconds=statistics.median(values), min_seconds=min(values),
                                    max_seconds=max(values), mpix_s=mpix / statistics.median(values))
                          for name, values in seconds.items()}
                report["results"].append(dict(case=case, mpix=mpix, bpp=case["bpp"],
                                               jpeg_bytes_exact=True, timing=timing, **comparisons))
                print(f"[{index + 1}/{len(cases)}] {case['label']}: "
                      f"Rust={timing['rust']['seconds']:.4f}s official={timing['official']['seconds']:.4f}s "
                      f"U8 max={comparisons['u8']['max_abs_error']} "
                      f"changed={comparisons['u8']['different_pixel_percent']:.5f}%", flush=True)
        for name, binary in binaries.items():
            if file_sha256(binary) != report["binaries"][name]["sha256"]:
                raise ValueError("Binary changed during measurement")
        for case in cases:
            for path_key, digest_key in (("jxl", "jxl_sha256"), ("original", "original_sha256")):
                path = Path(case[path_key])
                if file_sha256(path) != case[digest_key]:
                    raise ValueError(f"Input changed during measurement: {path}")
        report["status"] = ("completed_exact" if all(r["u8"]["exact"] and r["f32"]["exact"]
                                                     for r in report["results"])
                            else "completed_with_pixel_differences")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["temporary_outputs_cleaned"] = not any(args.out.glob("pixels-*"))
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        (args.out / "REPORT.md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
