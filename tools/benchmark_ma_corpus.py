"""Serial, one-trial native MA experiments on an explicit PNG manifest.

The caller selects the corpus and supplies an independent official djxl. Native
receipt seconds cover the full encode, not PNG I/O; subprocess wall time is only
diagnostic. No builds, sampling, JPEG transcoding, or replacement inputs.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

from benchmark_ma_seed import (
    SEED_COUNTERS, VARIANTS, encoder_command, matrix_order,
    validate_measurement, variant_parameters,
)
from compare_cpp_port import (
    PNG_SIGNATURE, check_process, clean_environment, compare_pixels,
    decoder_command, file_sha256, png_samples, run_process,
)
from corpus_inputs import prepare


SCRIPTS = ("benchmark_ma_corpus.py", "benchmark_ma_seed.py", "benchmark_ma_models.py",
           "benchmark_fast_context.py", "compare_cpp_port.py", "corpus_inputs.py")
FAST_COUNTERS = ("trees", "leaves", "pixels", "offset_leaves", "horizontal", "vertical", "single")


def now():
    return datetime.now(timezone.utc).isoformat()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("encoder", "decoder", "manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--effort", type=int, choices=range(1, 11), default=9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 256 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Invalid threads or timeout")
    if "baseline" not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("Variants must be distinct and include baseline")
    args.ma_seed_depth = 3
    args.efforts = [args.effort]
    return args


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return result


def loads(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def load_cases(manifest):
    spec = loads(manifest.read_text(encoding="utf-8-sig"))
    cases = spec.get("cases") if isinstance(spec, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain a nonempty cases list")
    seen, result = set(), []
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("original"), str)
                or not case["original"] or not isinstance(case.get("stratum"), str)
                or not _digest(case.get("sha256"))):
            raise ValueError("Each case requires original, stratum and lowercase SHA256")
        path = Path(case["original"])
        if not path.is_absolute() or path.suffix.lower() != ".png":
            raise ValueError("Manifest original must be an absolute PNG path")
        path = path.resolve()
        if path in seen:
            raise ValueError(f"Duplicate manifest source: {path}")
        seen.add(path)
        if "known_case" in case and type(case["known_case"]) is not bool:
            raise ValueError("known_case must be boolean")
        for name in ("width", "height", "depth", "bit_depth"):
            if name in case and (type(case[name]) is not int or case[name] <= 0):
                raise ValueError(f"Invalid manifest {name}")
        result.append(dict(case, original=str(path)))
    return result


def identity(args):
    def hashed(path):
        path = path.resolve(strict=True)
        return dict(path=str(path), sha256=file_sha256(path))

    args.encoder = args.encoder.resolve(strict=True)
    args.decoder = args.decoder.resolve(strict=True)
    args.manifest = args.manifest.resolve(strict=True)
    return dict(encoder=hashed(args.encoder), decoder=hashed(args.decoder),
                manifest=hashed(args.manifest),
                scripts=[hashed(Path(__file__).resolve().with_name(name)) for name in SCRIPTS],
                config=dict(effort=args.effort, threads=args.threads, timeout=args.timeout,
                            variants=list(args.variants), ma_seed_depth=args.ma_seed_depth,
                            trials_per_variant=1, serial=True))


def _serialize(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"


def atomic_write(path, values):
    """Replace only a derived file, or journal statuses invalidated by integrity."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=path.name + "-", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for value in values:
                stream.write(_serialize(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def receipt_validation(measurement, variant, size, pixels):
    """Keep raw receipts; permit a well-formed, entirely idle experiment bypass."""
    parameters = variant_parameters(variant)
    fast = measurement.get("fast_context") if isinstance(measurement, dict) else None
    if parameters["fast_context"] is not None:
        if not isinstance(fast, dict) or fast.get("predictors") != parameters["fast_context"]["predictors"]:
            raise ValueError("Fast-context receipt must confirm requested predictors (full means all 14)")
    try:
        validate_measurement(measurement, variant, 3, size, pixels)
    except ValueError as error:
        seed = measurement.get("ma_seed") if isinstance(measurement, dict) else None
        seed_idle = (str(error) == "Nonempty source has no MA seed learners"
                     and isinstance(seed, dict)
                     and all(type(seed.get(k)) is int and seed[k] == 0 for k in SEED_COUNTERS))
        fast_idle = (str(error) == "Fast-context experiment has a zero work receipt"
                     and isinstance(fast, dict)
                     and all(type(fast.get(k)) is int and fast[k] == 0 for k in FAST_COUNTERS)
                     and measurement.get("ma_seed", "missing") is None)
        if not (seed_idle or fast_idle):
            raise
        return dict(status="bypassed", reason="Experiment reports zero work; original pixels verified separately")
    return dict(status="passed")


def read_journal(path, cases, args):
    rows, seen = [], set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError(f"Torn journal record at line {number}")
            row = loads(line)
            required = {"index", "source", "stratum", "variant", "effort", "status", "pixels",
                        "width", "height", "depth", "original_sha256", "payload_sha256",
                        "measurement", "encoder", "decoder", "receipt_validation", "pixels_verified"}
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError(f"Malformed journal record at line {number}")
            index = row["index"]
            if type(index) is not int or not 0 <= index < len(cases):
                raise ValueError("Invalid journal case index")
            case = cases[index]
            if (row["source"] != case["original"] or row["stratum"] != case["stratum"]
                    or row["original_sha256"] != case["sha256"]
                    or row["variant"] not in args.variants or type(row["effort"]) is not int
                    or row["effort"] != args.effort or row["status"] not in ("passed", "failed")
                    or row.get("known_case") != case.get("known_case")):
                raise ValueError("Journal identity/config differs from manifest/run")
            key = (index, row["variant"])
            if key in seen:
                raise ValueError(f"Duplicate journal trial: {key}")
            seen.add(key)
            if row["payload_sha256"] is not None and not _digest(row["payload_sha256"]):
                raise ValueError("Invalid journal payload SHA256")
            for name in ("width", "height", "depth", "pixels"):
                value = row[name]
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(f"Invalid journal {name}")
            if type(row["pixels_verified"]) is not bool:
                raise ValueError("Invalid journal pixel verification")
            for name in ("encoder", "decoder", "measurement", "receipt_validation"):
                if row[name] is not None and not isinstance(row[name], dict):
                    raise ValueError(f"Invalid journal {name}")
            if row.get("preparation") is not None and not isinstance(row["preparation"], dict):
                raise ValueError("Invalid journal preparation")
            for name in ("encoder", "decoder"):
                record = row[name]
                if record is None:
                    continue
                wall = record.get("wall_seconds")
                if (not isinstance(record.get("command"), list) or not record["command"]
                        or any(not isinstance(token, str) for token in record["command"])
                        or (record.get("returncode") is not None and type(record["returncode"]) is not int)
                        or "returncode" not in record
                        or not isinstance(record.get("stdout"), str)
                        or not isinstance(record.get("stderr"), str)
                        or type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0):
                    raise ValueError(f"Malformed journal {name} subprocess record")
            receipt = row["receipt_validation"]
            if receipt is not None and receipt.get("status") not in ("passed", "failed", "bypassed"):
                raise ValueError("Invalid journal receipt validation status")
            dimensions = [row[k] for k in ("width", "height", "depth", "pixels")]
            if any(value is not None for value in dimensions):
                if (any(value is None for value in dimensions) or row["depth"] not in (8, 16)
                        or row["pixels"] != row["width"] * row["height"]):
                    raise ValueError("Inconsistent journal dimensions/depth")
            if "integrity_error" in row and (
                    not isinstance(row["integrity_error"], str) or not row["integrity_error"]
                    or row["status"] != "failed"):
                raise ValueError("Invalid journal integrity failure")
            if row["status"] == "failed":
                if not isinstance(row.get("error"), str) or not row["error"]:
                    raise ValueError("Failed journal row requires an error")
            else:
                if (not row["pixels_verified"] or not row["payload_sha256"]
                        or any(value is None for value in dimensions)
                        or any(not row[k] or row[k].get("returncode") != 0 for k in ("encoder", "decoder"))):
                    raise ValueError("Passed journal row lacks pixel/process evidence")
                measurement = row["measurement"]
                if not isinstance(measurement, dict) or row["receipt_validation"] != receipt_validation(
                        measurement, row["variant"], measurement.get("bytes"), row["pixels"]):
                    raise ValueError("Invalid passed journal measurement receipt")
            rows.append(row)
    return rows


def source_error(case):
    try:
        if file_sha256(Path(case["original"])) == case["sha256"]:
            return None
        return "Source changed: SHA256 differs from manifest"
    except OSError as error:
        return f"Source integrity failure: {error}"


def base_row(index, case, variant, args, info=None):
    info = info or {}
    row = dict(index=index, source=case["original"], stratum=case["stratum"], variant=variant,
               effort=args.effort, status="failed", pixels=info.get("pixels"),
               width=info.get("width"), height=info.get("height"), depth=info.get("depth"),
               original_sha256=case["sha256"], payload_sha256=info.get("payload_sha256"),
               preparation=info or None, parameters=variant_parameters(variant),
               measurement=None, encoder=None, decoder=None, receipt_validation=None,
               pixels_verified=False)
    if "known_case" in case:
        row["known_case"] = case["known_case"]
    return row


def invalidate(rows, index, error, journal):
    changed = False
    for row in rows:
        if row["index"] == index and row.get("integrity_error") != error:
            row["status"] = "failed"
            row["integrity_error"] = error
            row["error"] = (row.get("error", "") + "; " + error).lstrip("; ")
            changed = True
    if changed:
        atomic_write(journal, rows)


def benchmark(args):
    frozen = identity(args)
    cases = load_cases(args.manifest)
    if file_sha256(args.manifest) != frozen["manifest"]["sha256"]:
        raise ValueError("Manifest changed while loading")
    env, environment = clean_environment(os.environ)
    run_path, journal = args.out / "run.json", args.out / "trials.jsonl"
    if args.resume:
        saved = loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or saved.get("identity") != frozen:
            raise ValueError("Resume requires unchanged binaries, manifest, scripts and config")
        rows = read_journal(journal, cases, args)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        with run_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_serialize(dict(schema_version=1, created_utc=now(), identity=frozen,
                                         environment=environment, total_cases=len(cases),
                                         timing="Native full encode only; PNG I/O excluded; wall diagnostic separate",
                                         metadata="Metadata-free snapshots; pixel acceptance, not metadata acceptance",
                                         integrity="Rehash after variants/case; atomically fail all mutated-case rows")))
            stream.flush()
            os.fsync(stream.fileno())
        with journal.open("x", encoding="utf-8") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        rows = []
    completed = {(r["index"], r["variant"]) for r in rows}
    active = dict(index=None, source=None, variant=None, phase="starting")

    def counts():
        by_case = Counter(row["index"] for row in rows)
        return dict(total_cases=len(cases), completed_cases=sum(
            count == len(args.variants) for count in by_case.values()), completed_trials=len(rows))

    def progress(phase, *, index=None, source=None, variant=None, status="running"):
        active.update(index=index, source=source, variant=variant, phase=phase)
        atomic_write(args.out / "progress.json", [dict(active, **counts(), pid=os.getpid(),
                                                       updated_utc=now(), status=status)])

    def append(row):
        with journal.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_serialize(row))
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
        completed.add((row["index"], row["variant"]))

    errors, stopped = [], False
    scratch = None
    try:
        progress("starting")
        with tempfile.TemporaryDirectory(prefix="ma-corpus-", dir=args.out) as directory:
            scratch = Path(directory)
            for index, case in enumerate(cases):
                source = case["original"]
                remaining = [(effort, variant) for effort, variant in matrix_order(args, 0, index)
                             if (index, variant) not in completed]
                integrity = source_error(case)
                if integrity:
                    invalidate(rows, index, integrity, journal)
                if not remaining:
                    continue
                if (args.out / "STOP").exists():
                    stopped = True
                    break
                with tempfile.TemporaryDirectory(prefix="case-", dir=scratch) as case_directory:
                    info, snapshot, expected, preparation_error = None, None, None, integrity
                    progress("prepare", index=index, source=source)
                    try:
                        if integrity:
                            raise ValueError(integrity)
                        with Path(source).open("rb") as stream:
                            if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                                raise ValueError("Only original PNG input is allowed; no JPEG fallback")
                        info, snapshot, expected = prepare(Path(source), Path(case_directory))
                        if info["kind"] != "png" or info["input_sha256"] != case["sha256"]:
                            raise ValueError("Prepared input identity differs from PNG manifest")
                        for name in ("width", "height"):
                            if name in case and case[name] != info[name]:
                                raise ValueError(f"Manifest {name} differs from prepared original")
                        original_depth = case.get("bit_depth", case.get("depth"))
                        if original_depth is not None and max(8, original_depth) != info["depth"]:
                            raise ValueError("Manifest depth differs from prepared original")
                        old = [r for r in rows if r["index"] == index and r["payload_sha256"]]
                        if any(r["payload_sha256"] != info["payload_sha256"] for r in old):
                            integrity = "Prepared payload changed across resume"
                            invalidate(rows, index, integrity, journal)
                            raise ValueError(integrity)
                    except Exception as error:
                        preparation_error = f"{type(error).__name__}: {error}"
                    for effort, variant in remaining:
                        row = base_row(index, case, variant, args, info)
                        try:
                            if preparation_error or integrity:
                                raise ValueError(preparation_error or integrity)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=case_directory) as trial:
                                jxl, png = Path(trial) / "output.jxl", Path(trial) / "decoded.png"
                                progress("encode", index=index, source=source, variant=variant)
                                encoded = run_process(encoder_command(args, snapshot, jxl, effort, variant),
                                                      env, args.timeout)
                                row["encoder"] = encoded
                                check_process(encoded)
                                measurement = loads(encoded["stdout"])
                                if not isinstance(measurement, dict):
                                    raise ValueError("Encoder measurement must be an object")
                                row["measurement"] = measurement
                                receipt_error = None
                                try:
                                    row["receipt_validation"] = receipt_validation(
                                        measurement, variant, jxl.stat().st_size, info["pixels"])
                                except ValueError as error:
                                    receipt_error = str(error)
                                    row["receipt_validation"] = dict(status="failed", error=receipt_error)
                                progress("decode", index=index, source=source, variant=variant)
                                row["decoder"] = run_process(
                                    decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                check_process(row["decoder"])
                                progress("verify_pixels", index=index, source=source, variant=variant)
                                compare_pixels(expected, png_samples(png.read_bytes()))
                                row["pixels_verified"] = True
                                if receipt_error:
                                    raise ValueError(receipt_error)
                                row["status"] = "passed"
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                        progress("verify_source", index=index, source=source, variant=variant)
                        integrity = source_error(case) or integrity
                        if integrity:
                            row["status"] = "failed"
                            row["integrity_error"] = integrity
                            row["error"] = (row.get("error", "") + "; " + integrity).lstrip("; ")
                            invalidate(rows, index, integrity, journal)
                        append(row)
                        progress("trial_complete", index=index, source=source, variant=variant)
                    progress("case_integrity", index=index, source=source)
                    integrity = source_error(case) or integrity
                    if integrity:
                        invalidate(rows, index, integrity, journal)
                    progress("case_complete", index=index, source=source)
                if (args.out / "STOP").exists():
                    stopped = True
                    break
    except KeyboardInterrupt:
        stopped = True
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    report = dict(schema_version=1, status="stopped" if stopped else "completed", **counts(),
                  total_trials=len(cases) * len(args.variants),
                  passed_trials=sum(r["status"] == "passed" for r in rows),
                  failed_trials=sum(r["status"] == "failed" for r in rows),
                  failed_cases=len({r["index"] for r in rows if r["status"] == "failed"}),
                  integrity_failed_cases=len({r["index"] for r in rows if r.get("integrity_error")}),
                  errors=errors, temporary_outputs_cleaned=scratch is None or not scratch.exists(),
                  updated_utc=now())
    if errors:
        report["status"] = "failed"
    progress(report["status"], status=report["status"], **{k: active[k] for k in ("index", "source", "variant")})
    atomic_write(args.out / "report.json", [report])
    return report


def main(argv=None):
    try:
        report = benchmark(arguments(argv))
    except (OSError, ValueError) as error:
        print(f"MA corpus runner: {error}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {report['completed_trials']} trials, {report['failed_trials']} failed", flush=True)
    return int(report["status"] != "completed" or report["failed_trials"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
