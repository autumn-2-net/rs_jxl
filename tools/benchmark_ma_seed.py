"""Compare explicit encoder_lab MA seed and fast-context experiments.

Native timings exclude startup, PNG I/O and official djxl verification. No build,
binary discovery, encoded-byte identity goal or production-default claim.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from benchmark_fast_context import (
    add_ratios, load_cases, summarize, validate_measurement as validate_common,
)
from benchmark_ma_models import trial_order
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process, sha256,
)


VARIANTS = ("baseline", "fast-basic", "fast-full", "seed-only", "seed-refine", "seed-repair")
SEED_COUNTERS = ("learners", "prepared_records", "seed_splits", "refined_leaves",
                 "repaired_branches", "merged_branches", "predictor_changes", "histogram_reuses")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", "--official-djxl", dest="decoder", type=Path, required=True)
    parser.add_argument("--case", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--efforts", nargs="+", type=int, choices=range(1, 11), default=[2, 3, 5, 9])
    parser.add_argument("--ma-seed-depth", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.case and args.manifest is None:
        parser.error("At least one --case or --manifest is required")
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid threads/repeats/warmups/timeout")
    if len(set(args.variants)) != len(args.variants) or "baseline" not in args.variants:
        parser.error("Variants must be distinct and include baseline")
    if len(set(args.efforts)) != len(args.efforts):
        parser.error("Efforts must be distinct")
    if args.out.exists():
        parser.error("Output already exists; reports are never overwritten")
    return args


def variant_parameters(variant, depth=3):
    if variant not in VARIANTS or type(depth) is not int or not 1 <= depth <= 8:
        raise ValueError("Unknown variant or invalid seed depth")
    fast, seed = None, None
    if variant.startswith("fast-"):
        fast = dict(mode="auto12", bias="static", predictors=variant[5:])
    if variant.startswith("seed-"):
        seed = dict(mode=variant[5:], depth=depth)
    return dict(fast_context=fast, ma_seed=seed)


def encoder_command(args, source, output, effort, variant):
    command = [str(args.encoder), str(source), "--effort", str(effort),
               "--threads", str(args.threads), "--model-costs", "--output", str(output)]
    parameters = variant_parameters(variant, args.ma_seed_depth)
    if parameters["fast_context"] is not None:
        command += ["--fast-context", "auto12", "--context-bias", "static",
                    "--context-predictors", parameters["fast_context"]["predictors"]]
    if parameters["ma_seed"] is not None:
        command += ["--ma-seed", parameters["ma_seed"]["mode"],
                    "--ma-seed-depth", str(args.ma_seed_depth)]
    return command


def validate_measurement(measurement, variant, depth, encoded_size, pixels):
    parameters = variant_parameters(variant, depth)
    # Reuse the existing numeric/model-cost and fast-context counter contract.
    fast_variant = "auto12-static" if parameters["fast_context"] is not None else "baseline"
    validate_common(measurement, fast_variant, "auto", encoded_size, pixels)
    fast = measurement["fast_context"]
    if fast is not None and "predictors" in fast:
        if fast["predictors"] != parameters["fast_context"]["predictors"]:
            raise ValueError("Fast-context predictors differ from requested experiment")
    if "ma_seed" not in measurement:
        raise ValueError("Missing ma_seed receipt (inactive modes must explicitly report null)")
    receipt = measurement["ma_seed"]
    expected = parameters["ma_seed"]
    if expected is None:
        if receipt is not None:
            raise ValueError("Non-seed variant unexpectedly used MA seed")
        return
    if (not isinstance(receipt, dict) or receipt.get("mode") != expected["mode"]
            or type(receipt.get("depth")) is not int or receipt["depth"] != depth):
        raise ValueError("MA seed parameters differ from requested experiment")
    for key in SEED_COUNTERS:
        if type(receipt.get(key)) is not int or receipt[key] < 0:
            raise ValueError(f"Invalid MA seed counter: {key}")
    if pixels > 0 and receipt["learners"] == 0:
        raise ValueError("Nonempty source has no MA seed learners")


def matrix_order(args, repetition, case_index):
    # Complete a local effort comparison before moving on; rotate both axes.
    efforts = trial_order(args.efforts, repetition, case_index)
    return [(effort, variant) for index, effort in enumerate(efforts)
            for variant in trial_order(args.variants, repetition, case_index + index)]


def benchmark(args):
    report = dict(schema_version=1, status="failed", created_utc=datetime.now(timezone.utc).isoformat(),
                  config=vars(args).copy(), cases=[], errors=[],
                  timing="Native encoder call only; startup/PNG I/O/verification excluded; warmups excluded",
                  pixel_contract="Original PNG samples, including hidden RGB and 16-bit depth",
                  ratio_contract="variant / same-image same-effort baseline; medians; not cross-effort",
                  scope="Explicit experimental matrix, not production defaults or corpus-wide acceptance",
                  receipt_contract="Raw native receipts retained; zero prepared_records means no prepared work",
                  predictors_contract="Optional native predictors checked when present; requested value always recorded",
                  hash_contract="Inputs/binaries/manifest hashed for provenance only; no encoded hash goal",
                  official_decoder_provenance="Caller-attested official djxl; path and SHA256 recorded",
                  timeout_contract="Harness process guard, not an encoder heuristic")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = clean_environment(os.environ)
            for name in ("encoder", "decoder"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                setattr(args, name, path)
                digest = file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                digest = file_sha256(args.manifest)
                identities.append((args.manifest, digest))
                report["manifest"] = dict(path=str(args.manifest.resolve()), sha256=digest)
            sources = load_cases(args)
            with tempfile.TemporaryDirectory(prefix="ma-seed-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = sha256(original)
                    identities.append((source, digest))
                    expected = png_samples(original)
                    pixels = expected.width * expected.height
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(original)
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                efforts={str(e): dict(variants={v: [] for v in args.variants}, summaries={})
                                         for e in args.efforts})
                    report["cases"].append(case)
                    for repetition in range(-args.warmups, args.repeats):
                        for effort, variant in matrix_order(args, repetition, index):
                            sample = dict(repetition=repetition, warmup=repetition < 0, verified=False,
                                          parameters=variant_parameters(variant, args.ma_seed_depth))
                            case["efforts"][str(effort)]["variants"][variant].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                jxl, png = Path(trial) / "output.jxl", Path(trial) / "decoded.png"
                                encoded = run_process(encoder_command(args, snapshot, jxl, effort, variant),
                                                      env, args.timeout)
                                sample["encoder"] = encoded
                                check_process(encoded)
                                decoded = run_process(decoder_command(args.decoder, jxl, png, expected.depth),
                                                      env, args.timeout)
                                sample["decoder"] = decoded
                                check_process(decoded)
                                compare_pixels(expected, png_samples(png.read_bytes()))
                                sample["verified"] = True
                                measurement = json.loads(encoded["stdout"])
                                validate_measurement(measurement, variant, args.ma_seed_depth,
                                                     jxl.stat().st_size, pixels)
                                sample["measurement"] = measurement
                                seed = measurement["ma_seed"]
                                if seed is not None:
                                    sample["seed_prepared_work"] = seed["prepared_records"] > 0
                            print(f"{source.name} effort={effort} {variant} warmup={sample['warmup']}: "
                                  f"{measurement['bytes']} bytes {measurement['seconds']:.6f}s verified",
                                  flush=True)
                    for result in case["efforts"].values():
                        result["summaries"] = {v: summarize(samples, pixels)
                                               for v, samples in result["variants"].items()}
                        add_ratios(result["summaries"])
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if file_sha256(path) == digest:
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
    try:
        report = benchmark(arguments(argv))
    except (OSError, ValueError) as error:
        print(f"Benchmark/report failure: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(report["status"], flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
