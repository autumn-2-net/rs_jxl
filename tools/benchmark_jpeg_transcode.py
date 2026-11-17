"""Paired native JPEG transcode measurements with byte-exact djxl validation.

No build or source discovery. Inputs are explicit originals from a manifest.
Native time excludes file IO, process startup, and reconstruction. Use --profile
only for diagnostics: inclusive stage times overlap across worker threads.
"""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import statistics
import tempfile

from PIL import Image

import compare_cpp_port as port
from benchmark_fast_context import load_cases
from benchmark_jpeg_head_tail import validate_measurement
from benchmark_ma_models import trial_order
from jpeg_lab_receipts import (
    EXPERIMENT_FLAGS, AC_LZ_FLAGS, AC_HISTOGRAM_FLAGS, BLOCK_CONTEXT_FLAGS, AC_ORDER_FLAGS,
    SIZE_BREAKDOWN_FIELDS, ac_lz_mode, ac_histograms, representation_modes,
    validate_experiments, validate_size_breakdown, validate_audit,
)


def summary(cases, name):
    size = sum(c["summaries"][name]["bytes"] for c in cases)
    seconds = sum(c["summaries"][name]["seconds"] for c in cases)
    pixels = sum(c["pixels"] for c in cases)
    original = sum(c["original_bytes"] for c in cases)
    return dict(bytes=size, seconds=seconds, bpp=8 * size / pixels,
                mean_bpp=statistics.mean(c["summaries"][name]["bpp"] for c in cases),
                mpix_per_second=pixels / 1e6 / seconds, jpeg_ratio=size / original)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True, type=Path)
    p.add_argument("--baseline", type=Path, help="Optional paired jpeg_entropy_lab; may match --encoder")
    p.add_argument("--decoder", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--effort", type=int, choices=range(1, 12), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300)
    for name in ("baseline", "candidate"):
        p.add_argument(f"--{name}-arg", action="append", default=[],
                       choices=EXPERIMENT_FLAGS + AC_LZ_FLAGS + AC_HISTOGRAM_FLAGS + BLOCK_CONTEXT_FLAGS + AC_ORDER_FLAGS,
                       help=f"Repeat as --{name}-arg=--flag for each experiment")
    p.add_argument("--profile", action="store_true")
    p.set_defaults(case=[])
    a = p.parse_args(argv)
    for name in ("baseline", "candidate"):
        # Clap arguments are single-valued; keep harmless identical repeats once.
        setattr(a, f"{name}_arg", list(dict.fromkeys(getattr(a, f"{name}_arg"))))
        try:
            ac_lz_mode(getattr(a, f"{name}_arg"))
            ac_histograms(getattr(a, f"{name}_arg"))
            representation_modes(getattr(a, f"{name}_arg"))
        except ValueError as error:
            p.error(f"{name}: {error}")
    if not 1 <= a.threads <= 256 or a.repeats < 1 or not 0 < a.timeout < float("inf"):
        p.error("Invalid execution limits")
    if a.out.exists():
        p.error("Output exists")
    return a


def benchmark(args):
    env, removed = port.clean_environment(os.environ)
    if args.profile:
        env["RS_JXL_PROFILE"] = "1"
    variants = {"candidate": args.encoder.resolve(strict=True)}
    if args.baseline:
        variants = {"baseline": args.baseline.resolve(strict=True), **variants}
    decoder = args.decoder.resolve(strict=True)
    identities = {p: port.file_sha256(p) for p in [*variants.values(), decoder, args.manifest]}
    report = dict(status="running", config=vars(args), environment=removed,
                  binaries={k: dict(path=str(p), sha256=identities[p]) for k, p in variants.items()},
                  decoder=dict(path=str(decoder), sha256=identities[decoder]),
                  timing="Native parse and encode only; median per image then sum",
                  correctness="Native and official djxl complete original JPEG byte equality",
                  cases=[])
    if any("--audit-ac" in getattr(args, f"{name}_arg") for name in variants):
        report["timing"] = "Diagnostic run: audited variants include AC audit work; not a speed benchmark"
    sources = load_cases(args)
    if not sources:
        raise ValueError("Empty case manifest")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        try:
            with tempfile.TemporaryDirectory(prefix="jpeg-transcode-") as temp:
                temp = Path(temp)
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    with Image.open(BytesIO(original)) as image:
                        if image.format != "JPEG":
                            raise ValueError(f"Not JPEG: {source}")
                        width, height = image.size
                        sampling = [list(layer[1:3]) for layer in image.layer]
                    case = dict(source=str(source), sha256=port.sha256(original),
                                original_bytes=len(original), pixels=width * height,
                                width=width, height=height, sampling=sampling,
                                cfl_eligible=len(sampling) == 3 and all(s == sampling[0] for s in sampling),
                                variants={n: [] for n in variants})
                    report["cases"].append(case)
                    snapshot = temp / "input.jpg"
                    snapshot.write_bytes(original)
                    for repeat in range(args.repeats):
                        for name in trial_order(tuple(variants), repeat, index):
                            encoded = temp / f"{index}-{repeat}-{name}.jxl"
                            restored = encoded.with_suffix(".jpg")
                            cmd = [str(variants[name]), str(snapshot), "--output", str(encoded),
                                   "--effort", str(args.effort), "--threads", str(args.threads)]
                            flags = getattr(args, f"{name}_arg")
                            cmd += flags
                            sample = dict(repeat=repeat, encoder=port.run_process(cmd, env, args.timeout))
                            case["variants"][name].append(sample)
                            port.check_process(sample["encoder"])
                            m = json.loads(sample["encoder"]["stdout"])
                            validate_measurement(m, "baseline", encoded.stat().st_size, width, height)
                            validate_experiments(m, flags, channels=len(sampling))
                            validate_size_breakdown(m, encoded.stat().st_size)
                            if "--audit-ac" in flags:
                                validate_audit(m)
                            if m["jpeg_bytes"] != len(original):
                                raise ValueError("Input byte count mismatch")
                            sample["measurement"] = m
                            sample["decoder"] = port.run_process(
                                [str(decoder), str(encoded), str(restored), "--num_threads=1", "--quiet"],
                                env, args.timeout)
                            port.check_process(sample["decoder"])
                            if restored.read_bytes() != original:
                                raise ValueError(f"Official JPEG byte mismatch: {source}")
                            sample["verified"] = True
                            encoded.unlink()
                            restored.unlink()
                    case["summaries"] = {}
                    for name, samples in case["variants"].items():
                        size = statistics.median(s["measurement"]["bytes"] for s in samples)
                        seconds = statistics.median(s["measurement"]["seconds"] for s in samples)
                        case["summaries"][name] = dict(bytes=size, seconds=seconds, bpp=size * 8 / case["pixels"])
                    print(f"[{index + 1}/{len(sources)}] {source.name}: {case['summaries']}", flush=True)
            report["aggregate"] = {name: summary(report["cases"], name) for name in variants}
            for path, digest in identities.items():
                if port.file_sha256(path) != digest:
                    raise ValueError(f"Executable or manifest changed: {path}")
            report["status"] = "passed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    benchmark(arguments())
