"""Independent seven-mode native e9 palette ablation, checked by official djxl.

Retain native grouping for the primary comparison. Use --full-frame only for
separate controls: it can greatly increase memory/time on large originals.
Repeat in NEW reports at 4096/10000 on a preselected sensitivity subset.
Never attribute grouping/cap changes to delta. No builds or winner filtering.
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

from benchmark_fast_context import positive_number, summarize
from representation_inputs import load_verified_cases as load_cases
from benchmark_ma_models import trial_order
from benchmark_palette_runs import prepare_png
from benchmark_tile_patches import add_comparisons as _comparisons, aggregate as _aggregate
import compare_cpp_port as port

MODES = ("absolute", "exact-delta-gradient", "exact-delta-left", "exact-delta-top",
         "exact-zero-gradient", "exact-zero-left", "exact-zero-top")
SCOPE = "invocation_policy_attempts_global_and_local"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True, help="Prebuilt palette_delta_lab")
    p.add_argument("--decoder", "--official-djxl", "--djxl", dest="decoder", type=Path, required=True,
                   help="Caller-attested official djxl; recorded with SHA256")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--effort", type=int, choices=(9,), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--full-frame", action="store_true")
    p.add_argument("--palette-colors", type=int, choices=(1024, 4096, 10000))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if (not 1 <= a.threads <= 256 or not 1 <= a.repeats <= 100 or not 0 <= a.warmups <= 100
            or not positive_number(a.timeout)):
        p.error("Invalid threads/repeats/warmups/timeout")
    if len(set(a.modes)) != len(a.modes) or "absolute" not in a.modes:
        p.error("Use distinct modes including absolute")
    if not a.case and a.manifest is None and not a.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if a.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return a


def command(args, source, output, mode):
    if mode not in MODES or args.effort != 9:
        raise ValueError("Expected one of the seven e9 palette modes")
    cmd = [str(args.encoder), str(source), "--output", str(output), "--mode", mode,
           "--effort", "9", "--threads", str(args.threads)]
    if args.palette_colors is not None:
        cmd += ["--palette-colors", str(args.palette_colors)]
    if args.full_frame:
        cmd += ["--full-frame"]
    return cmd


def native_mode(mode):
    return "Absolute" if mode == "absolute" else (
        f"{'ExactDelta' if mode.startswith('exact-delta-') else 'ExactZero'}({mode.rsplit('-', 1)[1].title()})")


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_attempt(a, mode):
    if not isinstance(a, dict) or a.get("requested_mode") != native_mode(mode):
        raise ValueError("Attempt requested mode mismatch")
    if a.get("scope") not in ("global", "local"):
        raise ValueError("Missing global/local attempt scope")
    for key in ("begin", "limit", "colors", "deltas", "delta_pixels", "zero_mask_matches"):
        if not _integer(a.get(key)):
            raise ValueError(f"Invalid attempt {key}")
    for key in ("width", "height", "channels"):
        if not _integer(a.get(key), 1):
            raise ValueError(f"Invalid attempt {key}")
    flags = ("exact_delta_attempted", "exact_delta_produced", "exact_zero_attempted",
             "zero_mask_eligible", "absolute_attempted", "absolute_produced", "cost_bypassed")
    if any(type(a.get(key)) is not bool for key in flags):
        raise ValueError("Missing/invalid kernel or cost flags")
    is_delta, is_zero = mode.startswith("exact-delta-"), mode.startswith("exact-zero-")
    reached_kernel = any(a[key] for key in ("exact_delta_attempted", "exact_zero_attempted", "absolute_attempted"))
    if ((a["exact_delta_attempted"] and not is_delta) or (a["exact_zero_attempted"] and not is_zero)
            or (reached_kernel and (a["exact_delta_attempted"] != is_delta or a["exact_zero_attempted"] != is_zero))):
        raise ValueError("Inconsistent requested/attempted kernel")
    if ((a["exact_delta_produced"] and not a["exact_delta_attempted"])
            or (a["zero_mask_eligible"] and not a["exact_zero_attempted"])
            or (reached_kernel and a["absolute_attempted"] == a["exact_delta_produced"])
            or (a["absolute_produced"] and not a["absolute_attempted"])):
        raise ValueError("Inconsistent kernel production/fallback")
    pixels = a["width"] * a["height"]
    if (a["delta_pixels"] > pixels or a["zero_mask_matches"] > pixels
            or (not a["zero_mask_eligible"] and a["zero_mask_matches"])):
        raise ValueError("Inconsistent actual delta references")
    outcome = a.get("outcome")
    if outcome not in ("accepted", "cost_rejected", "kernel_declined") or not _number(a.get("cost_before")):
        raise ValueError("Missing policy outcome/cost")
    produced = a["exact_delta_produced"] or a["absolute_produced"]
    if produced != (outcome != "kernel_declined"):
        raise ValueError("Kernel/policy outcome mismatch")
    if not produced and any(a[key] for key in ("colors", "deltas", "delta_pixels")):
        raise ValueError("Declined attempt claims a candidate table")
    after = a.get("cost_after")
    if a["cost_bypassed"] or not produced:
        if after is not None or outcome == "cost_rejected":
            raise ValueError("Bypassed/unreached cost claims rejection")
    elif not _number(after) or (after > a["cost_before"]) != (outcome == "cost_rejected"):
        raise ValueError("Inconsistent policy cost rejection")
    predictor = mode.rsplit("-", 1)[-1].title()
    if a["exact_delta_produced"]:
        if (a["colors"] != 0 or a["deltas"] < 1 or a["delta_pixels"] != pixels
                or a.get("predictor") != predictor):
            raise ValueError("Invalid exact delta candidate")
    elif produced and is_zero and a["zero_mask_eligible"]:
        if (a["deltas"] != 1 or a["delta_pixels"] != a["zero_mask_matches"]
                or a.get("predictor") != predictor):
            raise ValueError("Invalid exact zero candidate/use")
    elif a["deltas"] or a["delta_pixels"] or a.get("predictor") != "Zero":
        raise ValueError("Absolute fallback falsely claims deltas")


def validate_measurement(m, mode, size, width, height, args):
    if not isinstance(m, dict) or not positive_number(m.get("seconds")):
        raise ValueError("Missing/invalid native seconds")
    if not _integer(m.get("bytes"), 1) or m["bytes"] != size:
        raise ValueError("Reported bytes differ from actual output")
    if (m.get("width"), m.get("height")) != (width, height):
        raise ValueError("Image dimensions mismatch")
    for key, expected in (("bpp", size * 8 / (width * height)),
                          ("mpix_per_second", width * height / 1e6 / m["seconds"])):
        if not positive_number(m.get(key)) or not math.isclose(m[key], expected, rel_tol=1e-6):
            raise ValueError(f"Inconsistent {key}")
    if (m.get("mode") != mode or type(m.get("effort")) is not int or m["effort"] != 9
            or type(m.get("threads")) is not int or m["threads"] != args.threads
            or type(m.get("encode_calls")) is not int or m["encode_calls"] != 1
            or m.get("search") is not False or m.get("patches_cleared") is not True):
        raise ValueError("Not the requested single native e9 encode")
    if m.get("overrides") != dict(palette_colors=args.palette_colors, full_frame=args.full_frame):
        raise ValueError("Override mismatch")
    c, p = m.get("configured"), m.get("projected")
    for policy in (c, p):
        if (not isinstance(policy, dict) or type(policy.get("groupwise")) is not bool
                or not _integer(policy.get("palette_colors"), 1) or policy["palette_colors"] > 70911):
            raise ValueError("Missing/invalid palette policy")
    if (c.get("mode") != native_mode(mode)
            or c["groupwise"] != (p["groupwise"] and not args.full_frame)
            or c["palette_colors"] != (args.palette_colors if args.palette_colors is not None else p["palette_colors"])):
        raise ValueError("Unexpected non-mode policy override")
    if not isinstance(m.get("invariant_options"), str) or not m["invariant_options"]:
        raise ValueError("Missing non-mode option fingerprint")
    e = m.get("palette_evidence")
    if not isinstance(e, dict) or e.get("scope") != SCOPE or not e.get("limitations"):
        raise ValueError("Missing invocation-local evidence/limitations")
    if e.get("route") not in ("full_image", "groupwise") or (args.full_frame and e["route"] != "full_image"):
        raise ValueError("Invalid actual route")
    if not isinstance(e.get("attempts"), list) or not isinstance(e.get("global_tuple_palettes"), list):
        raise ValueError("Missing actual palette attempts/global tuples")
    for attempt in e["attempts"]:
        validate_attempt(attempt, mode)
    tuples = [(a["colors"], a["deltas"], a["predictor"]) for a in e["attempts"]
              if a["scope"] == "global" and a["channels"] > 1 and a["outcome"] == "accepted"]
    actual = e["global_tuple_palettes"]
    if any(not isinstance(t, dict) for t in actual) or sorted(tuples) != sorted(
            (t.get("colors"), t.get("deltas"), t.get("predictor")) for t in actual):
        raise ValueError("Global tuple receipt contradicts invocation telemetry")
    if e["route"] == "groupwise" and any(a["scope"] == "global" for a in e["attempts"]):
        raise ValueError("Groupwise route falsely claims global work")


def activation(measurement):
    """Derived from completed attempts, never from the configured mode alone."""
    result = {}
    for scope in ("global", "local"):
        events = [a for a in measurement["palette_evidence"]["attempts"] if a["scope"] == scope]
        accepted = [a for a in events if a["outcome"] == "accepted"]
        result[scope] = dict(
            attempts=len(events), exact_delta_attempted=sum(a["exact_delta_attempted"] for a in events),
            exact_zero_attempted=sum(a["exact_zero_attempted"] for a in events),
            exact_delta_produced=sum(a["exact_delta_produced"] for a in events),
            exact_delta_accepted=sum(a["exact_delta_produced"] for a in accepted),
            exact_zero_accepted=sum(a["deltas"] > 0 and a["exact_zero_attempted"] for a in accepted),
            absolute_fallback_accepted=sum(a["deltas"] == 0 and a["requested_mode"] != "Absolute" for a in accepted),
            absolute_accepted=sum(a["deltas"] == 0 for a in accepted),
            kernel_declined=sum(a["outcome"] == "kernel_declined" for a in events),
            cost_rejected=sum(a["outcome"] == "cost_rejected" for a in events),
            delta_accepted=sum(a["deltas"] > 0 for a in accepted),
            delta_used=sum(a["delta_pixels"] > 0 for a in accepted),
            delta_pixels=sum(a["delta_pixels"] for a in accepted))
    return result


def content_features(expected):
    """Bounded deterministic source-RGBA descriptors, NOT post-RCT activation."""
    _, np = port.pixel_libraries()
    data = expected.rgba
    total = expected.width * expected.height
    positions = np.arange(0, total, max(1, math.ceil(total / 65536)))
    samples = data.reshape(-1, 4)[positions].astype(np.int64)
    result = dict(scope="source RGBA; <=65536 raster-stride samples; interior-only residuals; excludes RCT/group borders",
                  sampled_pixels=len(positions), unique_tuples=int(len(np.unique(samples, axis=0))),
                  alpha_zero_fraction=float(np.mean(samples[:, 3] == 0)),
                  hidden_rgb_nonzero_fraction=float(np.mean((samples[:, 3] == 0) & np.any(samples[:, :3] != 0, axis=1))),
                  residuals={})
    y, x = positions // expected.width, positions % expected.width
    interior = (x > 0) & (y > 0)
    x, y = x[interior], y[interior]
    if len(x):
        v = data[y, x].astype(np.int64)
        l, t, tl = (data[yy, xx].astype(np.int64) for yy, xx in ((y, x-1), (y-1, x), (y-1, x-1)))
        for name, prediction in (("left", l), ("top", t),
                                  ("gradient", np.clip(l + t - tl, np.minimum(l, t), np.maximum(l, t)))):
            residual = v - prediction
            result["residuals"][name] = dict(samples=len(x), unique_tuples=int(len(np.unique(residual, axis=0))),
                                              exact_zero_fraction=float(np.mean(np.all(residual == 0, axis=1))))
    return result


def fixtures(folder):
    folder.mkdir()
    paths = []
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    patterns = (("gradient", 193, 67), ("left", 193, 67), ("top", 193, 67),
                ("mixed", 193, 67), ("natural-like", 193, 67), ("border", 2, 129),
                ("single-row", 257, 1), ("single-column", 1, 257), ("constant", 1, 1))
    for depth in (8, 16):
        top = (1 << depth) - 1
        for pattern, width, height in patterns:
            raw = bytearray()
            for y in range(height):
                row = []
                for x in range(width):
                    for c in range(4):
                        if pattern == "natural-like":
                            # Smooth deterministic texture plus fine detail; not a natural-photo claim.
                            v = round(top * (0.45 + 0.17 * math.sin(x / 19 + c)
                                             + 0.12 * math.cos(y / 11 + c) + 0.08 * math.sin((x+y)/7)))
                            v = (v + ((x * 17 + y * 31 + c * 13) % 37)) & top
                        elif pattern == "mixed" and x < width // 2:
                            v = 23 + c * 17
                        elif pattern == "constant":
                            v = 19 + c * 31
                        else:
                            v = ((0 if pattern == "top" else x) * (c + 1)
                                 + (0 if pattern in ("left", "single-row") else y) * (c + 2) + 11 + c * 7)
                            if depth == 16:
                                v = v * 41 + (x % 7)  # Genuine low bits, not x257 expansion.
                            v &= top
                        if c == 3:
                            v = 0 if pattern == "mixed" or (x + y) % 5 == 0 else top
                        row.append(v)
                raw.append(0)
                raw.extend(struct.pack(">" + ("B" if depth == 8 else "H") * len(row), *row))
            header = struct.pack(">IIBBBBB", width, height, depth, 6, 0, 0, 0)
            path = folder / f"{pattern}-{depth}.png"
            with path.open("xb") as out:
                out.write(port.PNG_SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
            paths.append(path)
    return paths


def add_comparisons(summaries):
    _comparisons({"baseline" if mode == "absolute" else mode: summary for mode, summary in summaries.items()})


def aggregate(cases, modes):
    translated = [dict(pixels=c["pixels"], summaries={"baseline" if m == "absolute" else m: s
                  for m, s in c["summaries"].items()}) for c in cases]
    result = _aggregate(translated, ["baseline" if m == "absolute" else m for m in modes])
    return {"absolute" if m == "baseline" else m: s for m, s in result.items()}


def run_trial(args, snapshot, expected, mode, trial, env, sample):
    jxl, png = trial / "out.jxl", trial / "decoded.png"
    sample["encoder"] = port.run_process(command(args, snapshot, jxl, mode), env, args.timeout)
    # Even a nonzero encoder exit or invalid receipt must not skip an emitted JXL.
    if jxl.exists():
        sample["decoder"] = port.run_process(port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
        port.check_process(sample["decoder"])
        port.compare_pixels(expected, port.png_samples(png.read_bytes()))
        sample["pixels_verified"] = True
    port.check_process(sample["encoder"])
    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    measurement = json.loads(sample["encoder"]["stdout"], parse_constant=reject_nonfinite)
    sample["measurement"] = measurement
    validate_measurement(measurement, mode, jxl.stat().st_size, expected.width, expected.height, args)
    if not sample.get("pixels_verified"):
        raise ValueError("Missing official sample verification")
    sample.update(verified=True, activation=activation(measurement))


def benchmark(args):
    report = dict(schema_version=1, status="failed", baseline_mode="absolute", config=vars(args).copy(),
                  cases=[], errors=[],
                  timing="One complete native e9 encode including opt-in telemetry; input/output IO, startup, descriptors and djxl excluded",
                  correctness="Official djxl; exact original 8/16-bit samples including hidden RGB",
                  official_decoder_provenance="Caller-attested; executable path and SHA256 recorded",
                  receipt_scope=SCOPE, regression_policy="Declined/larger/slower modes retained, not failures",
                  ratio_contract="mode / absolute within this cap/buffering only; sums of per-case medians",
                  correlation_contract="Descriptive source features and observed routes/activation, not a trained router or heldout claim",
                  failure_contract="All trials attempted; errors/stdout/stderr/receipts retained in JSON; temporary binary artifacts removed")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        def checkpoint(final=False):
            output.seek(0)
            # Receipts can be large; pretty-print only the final report.
            json.dump(report, output, indent=2 if final else None, default=str, allow_nan=False)
            output.write("\n")
            output.truncate()
            output.flush()
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            for name in ("encoder", "decoder"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not a binary file: {path}")
                setattr(args, name, path)
                digest = port.file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                digest = port.file_sha256(args.manifest)
                identities.append((args.manifest, digest))
                report["manifest"] = dict(path=str(args.manifest.resolve()), sha256=digest)
            with tempfile.TemporaryDirectory(prefix="palette-delta-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                sources = load_cases(args) if args.case or args.manifest else []
                external = set(sources)
                if args.synthetic:
                    sources += fixtures(temporary / "synthetic")
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    if source in external:
                        identities.append((source, digest))
                    payload, expected, preparation = prepare_png(source, original, temporary)
                    snapshot = temporary / f"input-{index}.png"
                    snapshot.write_bytes(payload)
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                preparation=preparation, features=content_features(expected),
                                variants={m: [] for m in args.modes}, summaries={})
                    report["cases"].append(case)
                    invariant = None
                    for repeat in range(-args.warmups, args.repeats):
                        for mode in trial_order(args.modes, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][mode].append(sample)
                            try:
                                with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                    run_trial(args, snapshot, expected, mode, Path(trial), env, sample)
                                    options = sample["measurement"]["invariant_options"]
                                    if invariant is None:
                                        invariant = options
                                    if invariant != options:
                                        raise ValueError("Non-mode configuration changed within paired ablation")
                            except Exception as error:
                                sample.update(verified=False, error=f"{type(error).__name__}: {error}")
                                report["errors"].append(f"{source.name} {mode} repeat={repeat}: {sample['error']}")
                            checkpoint()
                    # No survivor-only summaries: incomplete modes stay as raw failed trials.
                    for mode, samples in case["variants"].items():
                        if all(s["verified"] for s in samples):
                            case["summaries"][mode] = summarize(samples, pixels)
                    if "absolute" in case["summaries"]:
                        add_comparisons(case["summaries"])
                if report["cases"] and all(set(c["summaries"]) == set(args.modes) for c in report["cases"]):
                    report["aggregate"] = aggregate(report["cases"], args.modes)
            if not report["errors"]:
                report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if port.file_sha256(path) == digest:
                        continue
                    error = f"Input/binary changed during benchmark: {path}"
                except OSError as failure:
                    error = str(failure)
                report["errors"].append(error)
            if report["errors"] or not report["temporary_outputs_cleaned"]:
                report["status"] = "failed"
            checkpoint(final=True)
    return report


def main(argv=None):
    report = benchmark(arguments(argv))
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
