"""Controlled single-Palette-decision exploration using a prebuilt native lab.

No builds, resizing, winner selection, or corpus aggregation across baselines.
Official djxl provenance is caller-attested; exact original samples are checked.
"""
import argparse
import json
import math
import os
from pathlib import Path, PureWindowsPath
import statistics
import sys
import tempfile

from benchmark_palette_delta import MODES
from benchmark_palette_runs import prepare_png
import compare_cpp_port as port

CHOICES = ("skip", *MODES)
SELECTOR_FIELDS = {"stage", "width", "height", "begin", "channels", "limit", "colors", "deltas", "outcome"}
OUTCOMES = ("accepted", "cost_rejected", "kernel_declined")


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def read_json(text):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(text, parse_constant=reject)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("encoder", "decoder", "manifest", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args(argv)
    if not 1 <= a.threads <= 256 or not 1 <= a.repeats <= 100 or not number(a.timeout, True):
        p.error("Invalid threads/repeats/timeout")
    if a.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return a


def load_manifest(path):
    document = read_json(path.read_text(encoding="utf-8"))
    cases = document.get("cases") if isinstance(document, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest needs a nonempty cases list")
    seen, result = set(), []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in seen:
            raise ValueError("Case IDs must be nonempty and unique")
        seen.add(case["id"])
        choices, selector = case.get("choices"), case.get("selector")
        if (not isinstance(choices, list) or not choices or any(c not in CHOICES for c in choices)
                or len(set(choices)) != len(choices) or case.get("baseline_mode") not in MODES):
            raise ValueError("Invalid baseline mode or duplicate/invalid choices")
        if not isinstance(selector, dict) or set(selector) - SELECTOR_FIELDS or selector.get("stage") not in ("global", "local"):
            raise ValueError("Invalid selector")
        for key, value in selector.items():
            if key == "outcome" and value not in OUTCOMES:
                raise ValueError("Invalid selector outcome")
            if key not in ("stage", "outcome") and not integer(value, 1 if key in ("width", "height", "channels") else 0):
                raise ValueError(f"Invalid selector {key}")
        maximum = case.get("max_targets", 1)
        if not integer(maximum, 1) or maximum > 100:
            raise ValueError("max_targets must be 1..100")
        if not isinstance(case.get("path"), str) or not case["path"]:
            raise ValueError("Missing original path")
        source = Path(case["path"])
        source = (source if source.is_absolute() else path.resolve().parent / source).resolve(strict=True)
        if not source.is_file():
            raise ValueError("Original is not a file")
        result.append(dict(case, path=str(source), max_targets=maximum))
    return result


def identity(event):
    if (not isinstance(event, dict) or event.get("stage") not in ("global", "local")
            or not integer(event.get("stream_id")) or not integer(event.get("ordinal"))):
        raise ValueError("Invalid event/target identity")
    return event["stage"], event["stream_id"], event["ordinal"]


def target_dict(event):
    stage, stream, ordinal = identity(event)
    return dict(stage=stage, stream_id=stream, ordinal=ordinal)


def select_targets(events, selector, maximum):
    matches = [e for e in events if isinstance(e.get("attempt"), dict)
               and all((e.get(k) if k == "stage" else e["attempt"].get(k)) == v for k, v in selector.items())]
    matches.sort(key=lambda e: (0 if identity(e)[0] == "global" else 1, *identity(e)[1:]))
    return dict(matched_count=len(matches), selected=[target_dict(e) for e in matches[:maximum]])


def output_path(folder, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or Path(name).is_absolute() or PureWindowsPath(name).is_absolute()
            or any(p in (".", "..") for p in name.split("/"))):
        raise ValueError("Unsafe output path")
    path = folder / name
    if path.suffix.lower() != ".jxl" or path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Unsafe output path")
    return path


def validate_kernel(attempt, choice, unavailable):
    flags = ("exact_delta_attempted", "exact_delta_produced", "exact_zero_attempted",
             "zero_mask_eligible", "absolute_attempted", "absolute_produced")
    if any(type(attempt.get(k)) is not bool for k in flags):
        raise ValueError("Missing/invalid kernel evidence")
    if (not integer(attempt.get("zero_mask_matches"))
            or attempt["zero_mask_matches"] > attempt["width"] * attempt["height"]):
        raise ValueError("Invalid zero mask evidence")
    produced = attempt["exact_delta_produced"] or attempt["absolute_produced"]
    if ((attempt["exact_delta_produced"] and not attempt["exact_delta_attempted"])
            or (attempt["absolute_produced"] and not attempt["absolute_attempted"])
            or (attempt["zero_mask_eligible"] and not attempt["exact_zero_attempted"])
            or (not unavailable and produced != (attempt["outcome"] != "kernel_declined"))):
        raise ValueError("Contradictory kernel production/outcome")
    if unavailable:
        # A declined forced kernel may have produced an unused natural fallback.
        if (attempt["outcome"] != "kernel_declined"
                or any(attempt[k] for k in ("colors", "deltas", "delta_pixels"))):
            raise ValueError("Unavailable branch claims an accepted candidate table")
        if ((choice.startswith("exact-delta-") and attempt["exact_delta_produced"])
                or (choice.startswith("exact-zero-") and attempt["zero_mask_eligible"] and attempt["absolute_produced"])
                or (choice == "absolute" and attempt["absolute_produced"])):
            raise ValueError("Unavailable branch contradicts requested kernel evidence")
        return
    if choice in ("keep", "skip"):
        return
    if attempt["outcome"] != "accepted":
        raise ValueError("Successful forced candidate was not accepted")
    if choice.startswith("exact-delta-"):
        if (not attempt["exact_delta_attempted"] or not attempt["exact_delta_produced"]
                or attempt["absolute_attempted"] or attempt["absolute_produced"]
                or attempt["exact_zero_attempted"] or attempt["colors"] != 0
                or attempt["deltas"] < 1 or attempt["delta_pixels"] != attempt["width"] * attempt["height"]):
            raise ValueError("Forced delta lacks exact-delta production")
    elif choice.startswith("exact-zero-"):
        if (not attempt["exact_zero_attempted"] or not attempt["zero_mask_eligible"]
                or not attempt["absolute_attempted"] or not attempt["absolute_produced"]
                or attempt["exact_delta_attempted"] or attempt["exact_delta_produced"]
                or attempt["deltas"] != 1 or attempt["delta_pixels"] != attempt["zero_mask_matches"]):
            raise ValueError("Forced zero lacks zero-mask production")
    elif (not attempt["absolute_attempted"] or not attempt["absolute_produced"]
          or attempt["exact_delta_attempted"] or attempt["exact_zero_attempted"]
          or attempt["exact_delta_produced"] or attempt["deltas"] or attempt["delta_pixels"]):
        raise ValueError("Forced absolute lacks absolute production")


def validate_events(events, name, target, unavailable):
    if not isinstance(events, list):
        raise ValueError("Missing events")
    seen, changed = set(), []
    for event in events:
        key = identity(event)
        if key in seen or type(event.get("unavailable")) is not bool:
            raise ValueError("Duplicate event or invalid unavailable flag")
        seen.add(key)
        choice = event.get("choice")
        if type(event.get("cost_forced")) is not bool or event["cost_forced"] != (choice in MODES):
            raise ValueError("cost_forced does not match event choice")
        if choice != "keep":
            changed.append(key)
            if target is None or key != identity(target) or choice != name or name == "baseline":
                raise ValueError("Changed event is not the requested target/choice")
        elif event["unavailable"]:
            raise ValueError("Natural event claims unavailable")
        if choice != "keep" and event["unavailable"] != unavailable:
            raise ValueError("Unavailable branch/event mismatch")
        if "attempt" not in event:
            raise ValueError("Missing attempt evidence")
        attempt = event["attempt"]
        if attempt is None:
            if choice not in ("keep", "skip") and not unavailable:
                raise ValueError("Forced candidate missing attempt")
            continue
        if not isinstance(attempt, dict) or attempt.get("mode") not in MODES or attempt.get("outcome") not in OUTCOMES:
            raise ValueError("Invalid attempt mode/outcome")
        if choice not in ("keep", "skip") and attempt["mode"] != choice:
            raise ValueError("Forced attempt mode mismatch")
        for key in ("width", "height", "channels", "begin", "limit", "colors", "deltas", "delta_pixels"):
            if not integer(attempt.get(key), 1 if key in ("width", "height", "channels") else 0):
                raise ValueError(f"Invalid attempt {key}")
        if attempt["delta_pixels"] > attempt["width"] * attempt["height"]:
            raise ValueError("Invalid delta pixel count")
        if not number(attempt.get("cost_before")) or type(attempt.get("cost_bypassed")) is not bool:
            raise ValueError("Invalid cost metadata")
        if "cost_after" not in attempt or (attempt["cost_after"] is not None and not number(attempt["cost_after"])):
            raise ValueError("Invalid cost_after")
        validate_kernel(attempt, choice, event["unavailable"])
        # Forced outcomes intentionally need not agree with the natural cost test.
    if target is not None and identity(target) not in seen:
        raise ValueError("Requested target missing from branch")
    if changed != ([] if name == "baseline" else [identity(target)]):
        raise ValueError("Expected exactly one changed event at target")


def validate_receipt(receipt, case, target, expected, threads, folder, verified):
    if not isinstance(receipt, dict):
        raise ValueError("Receipt is not an object")
    for key, value in dict(schema=1, width=expected.width, height=expected.height,
                           bits=expected.depth, effort=9, threads=threads).items():
        if type(receipt.get(key)) is not int or receipt[key] != value:
            raise ValueError(f"Receipt {key} mismatch")
    if receipt.get("baseline_mode") != case["baseline_mode"] or receipt.get("target") != target:
        raise ValueError("Baseline mode/target mismatch")
    if target is not None:
        identity(receipt["target"])
    configured = receipt.get("configured")
    if (receipt.get("search") is not False or receipt.get("patches_cleared") is not True
            or not isinstance(configured, dict) or type(configured.get("palette_colors")) is not int
            or configured["palette_colors"] != 1024 or type(configured.get("groupwise")) is not bool):
        raise ValueError("Expected native grouping policy, cap 1024, no search, and cleared patches")
    trials = receipt.get("trials")
    names = ["baseline"] + (case["choices"] if target is not None else [])
    if (not isinstance(trials, list) or any(not isinstance(t, dict) for t in trials)
            or sorted(t.get("name", "") for t in trials) != sorted(names)):
        raise ValueError("Incomplete or duplicate branch list")
    files = set()
    for trial in trials:
        unavailable = trial.get("status") == "unavailable"
        if trial.get("status") not in ("ok", "unavailable") or (unavailable and trial["name"] == "baseline"):
            raise ValueError("Invalid branch status")
        if not number(trial.get("seconds"), not unavailable):
            raise ValueError("Invalid native duration")
        if (type(trial.get("fixed_input_verified")) is not bool
                or trial["fixed_input_verified"] != (target is not None)):
            raise ValueError("Missing fixed-input verification")
        validate_events(trial.get("events"), trial["name"], target, unavailable)
        if unavailable:
            if any(trial.get(k) is not None for k in ("file", "bytes", "bpp", "mpix_per_second")):
                raise ValueError("Unavailable branch claims output")
            continue
        path = output_path(folder, trial.get("file"))
        if path in files or path not in verified or not integer(trial.get("bytes"), 1) or trial["bytes"] != path.stat().st_size:
            raise ValueError("Duplicate, unverified or inconsistent output")
        files.add(path)
        pixels = expected.width * expected.height
        for key, value in (("bpp", trial["bytes"] * 8 / pixels),
                           ("mpix_per_second", pixels / 1e6 / trial["seconds"])):
            if not number(trial.get(key), True) or not math.isclose(trial[key], value, rel_tol=1e-6):
                raise ValueError(f"Invalid {key}")
    if files != set(verified):
        raise ValueError("Emitted JXL missing from receipt")


def invoke(args, case, snapshot, expected, folder, env, sample, target=None, configured=None):
    if folder.exists():
        raise ValueError("Encoder output directory must not already exist")
    command = [str(args.encoder), str(snapshot), "--out-dir", str(folder),
               "--baseline-mode", case["baseline_mode"], "--threads", str(args.threads)]
    if target is not None:
        command += ["--target", ":".join(map(str, identity(target))), "--choices", *case["choices"]]
    sample["encoder"] = port.run_process(command, env, args.timeout)
    verified, errors = set(), []
    sample["outputs"] = []
    # Inspect outputs independently of exit status, JSON validity and branch claims.
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() != ".jxl":
            continue
        result = dict(file=str(path.relative_to(folder)), verified=False)
        sample["outputs"].append(result)
        try:
            output_path(folder, result["file"])
            png = folder / f"oracle-{len(sample['outputs'])}.png"
            result["decoder"] = port.run_process(port.decoder_command(args.decoder, path, png, expected.depth), env, args.timeout)
            port.check_process(result["decoder"])
            port.compare_pixels(expected, port.png_samples(png.read_bytes()))
            result["verified"] = True
            verified.add(path)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            errors.append(result["error"])
    try:
        sample["receipt"] = read_json(sample["encoder"]["stdout"])
        validate_receipt(sample["receipt"], case, target, expected, args.threads, folder, verified)
        if configured is not None and sample["receipt"]["configured"] != configured:
            raise ValueError("Native policy changed between discovery and target replay")
        port.check_process(sample["encoder"])
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    if errors:
        raise ValueError("; ".join(errors))
    sample["verified"] = True
    return sample["receipt"]


def summarize_target(samples, choices, repeats):
    complete = len(samples) == repeats and all(s.get("verified") for s in samples)
    baselines = [next(t for t in s["receipt"]["trials"] if t["name"] == "baseline")
                 for s in samples if s.get("verified")]
    result = dict(complete=complete, baseline=None, alternatives={})
    if complete:
        result["baseline"] = dict(bytes=statistics.median(t["bytes"] for t in baselines),
                                  seconds=statistics.median(t["seconds"] for t in baselines))
    for choice in choices:
        entry = dict(status="failed" if not complete else "unavailable", scored=False,
                     observations=[], bytes=None, seconds=None, byte_delta=None,
                     byte_ratio=None, seconds_ratio=None)
        result["alternatives"][choice] = entry
        for sample in samples:
            if not sample.get("verified"):
                continue
            trials = {t["name"]: t for t in sample.get("receipt", {}).get("trials", [])}
            if choice in trials:
                entry["observations"].append({k: trials[choice].get(k) for k in ("status", "bytes", "seconds")})
        if complete and all(o["status"] == "ok" for o in entry["observations"]):
            size = statistics.median(o["bytes"] for o in entry["observations"])
            seconds = statistics.median(o["seconds"] for o in entry["observations"])
            entry.update(status="ok", scored=True,
                         bytes=size, seconds=seconds, byte_delta=size - result["baseline"]["bytes"],
                         byte_ratio=size / result["baseline"]["bytes"],
                         seconds_ratio=seconds / result["baseline"]["seconds"])
    return result


def benchmark(args):
    report = dict(schema=1, status="running", config=vars(args).copy(), cases=[], errors=[],
                  scope="Controlled per-target alternatives; no corpus winners or cross-baseline aggregate",
                  correctness="Caller-attested official djxl; exact original samples including hidden RGB",
                  timing="Reported native branch seconds; no discovery, startup, IO or oracle timing in ratios")
    identities, scratch = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        def checkpoint():
            output.seek(0)
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
            output.truncate()
            output.flush()
        def failure(sample, error):
            sample.update(verified=False, error=f"{type(error).__name__}: {error}")
            report["errors"].append(sample["error"])
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            for name in ("encoder", "decoder", "manifest"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not a file: {path}")
                setattr(args, name, path)
                digest = port.file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            cases = load_manifest(args.manifest)
            with tempfile.TemporaryDirectory(prefix="palette-admission-", dir=args.out.parent) as temporary:
                scratch = Path(temporary)
                for index, spec in enumerate(cases):
                    print(f"case {index + 1}/{len(cases)}: {spec['id']} discovery", flush=True)
                    case = dict(spec=spec, targets=[])
                    report["cases"].append(case)
                    try:
                        source = Path(spec["path"])
                        original = source.read_bytes()
                        case["sha256"] = port.sha256(original)
                        identities.append((source, case["sha256"]))
                        payload, expected, case["preparation"] = prepare_png(source, original, scratch)
                        case.update(width=expected.width, height=expected.height, bits=expected.depth)
                        snapshot = scratch / f"input-{index}.png"
                        snapshot.write_bytes(payload)
                        discovery = case["discovery"] = dict(verified=False)
                        with tempfile.TemporaryDirectory(dir=scratch) as directory:
                            receipt = invoke(args, spec, snapshot, expected, Path(directory) / "trials", env, discovery)
                        case["selection"] = select_targets(receipt["trials"][0]["events"], spec["selector"], spec["max_targets"])
                        if not case["selection"]["selected"]:
                            raise ValueError("No discovery events match selector")
                        for target in case["selection"]["selected"]:
                            print(f"case {spec['id']} target {':'.join(map(str, identity(target)))} ({args.repeats} replay(s))", flush=True)
                            item = dict(target=target, samples=[])
                            case["targets"].append(item)
                            for repeat in range(args.repeats):
                                sample = dict(repetition=repeat, verified=False)
                                item["samples"].append(sample)
                                try:
                                    with tempfile.TemporaryDirectory(dir=scratch) as directory:
                                        invoke(args, spec, snapshot, expected, Path(directory) / "trials", env,
                                               sample, target, receipt["configured"])
                                except Exception as error:
                                    failure(sample, error)
                                checkpoint()
                            item["summary"] = summarize_target(item["samples"], spec["choices"], args.repeats)
                    except Exception as error:
                        failure(case, error)
                    checkpoint()
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            for case in report["cases"]:
                for item in case["targets"]:
                    item["summary"] = summarize_target(item["samples"], case["spec"]["choices"], args.repeats)
            report["temporary_outputs_cleaned"] = scratch is None or not scratch.exists()
            for path, digest in identities:
                try:
                    if port.file_sha256(path) != digest:
                        raise ValueError(f"Input/binary changed during run: {path}")
                except Exception as error:
                    report["errors"].append(str(error))
            if not report["temporary_outputs_cleaned"]:
                report["errors"].append("Temporary outputs not cleaned")
            report["status"] = "passed" if report["cases"] and not report["errors"] else "failed"
            checkpoint()
    return report


def main(argv=None):
    report = benchmark(arguments(argv))
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
