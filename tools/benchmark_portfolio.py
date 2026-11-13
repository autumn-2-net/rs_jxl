"""Prepare an original PNG corpus; encode serially only with explicit --run.

No builds or discovery. An existing encoder is required even for preparation so
its identity can be frozen. Resume a prepared directory with --resume --run.
Corpus cases run serially; native probe roots run concurrently by default.
Candidate probe_seconds sum elapsed probe work and can exceed scout_seconds,
which measures scout wall time rather than summed concurrent work.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process,
)
from corpus_inputs import png_framing, prepare


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def loads(text):
    value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    # Also reject exponent overflow such as 1e999.
    canonical(value)
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_write(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoder", type=Path,
                        default=Path("target/release/examples") / ("portfolio_lab.exe" if os.name == "nt" else "portfolio_lab"))
    parser.add_argument("--djxl", type=Path)
    parser.epilog = ("Supply the intended --djxl (or --skip-verify) during preparation too; "
                     "resume freezes verification settings as well as native parameters.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="Explicit prepare-only mode")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Explicit limit on unique PNG attempts")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--serial-probes", action="store_true",
                        help="Run native probe roots serially; probe_seconds sums can otherwise exceed scout wall time")
    parser.add_argument("--preparation-cache-mib", type=int, default=128,
                        help="Native preparation cache budget in MiB (0 disables reuse)")
    parser.add_argument("--no-finalist-bound", action="store_true",
                        help="Disable safe output-bound exits for losing full finalists")
    parser.add_argument("--effort", type=int, choices=(8, 9, 10), default=9)
    parser.add_argument("--candidates", help="Comma-separated native candidate names; baseline is always retained")
    parser.add_argument("--max-probe-encodes", type=int, default=48)
    parser.add_argument("--max-finalists", type=int, default=2)
    parser.add_argument("--probe-side", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 256 or args.max_probe_encodes < 0
            or args.preparation_cache_mib < 0
            or args.max_finalists < 0 or not 32 <= args.probe_side <= 512
            or (args.limit is not None and args.limit < 1)
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid resource parameters")
    if args.run and not args.skip_verify and args.djxl is None:
        parser.error("--run requires --djxl unless --skip-verify is explicit")
    if args.skip_verify and args.djxl is not None:
        parser.error("--skip-verify and --djxl are mutually exclusive")
    if args.candidates is not None:
        names = [name.strip() for name in args.candidates.split(",")]
        if any(not name for name in names) or len(set(names)) != len(names):
            parser.error("--candidates requires distinct nonempty comma-separated names")
        args.candidates = ",".join(names)
    return args


def hashed(path):
    path = path.resolve(strict=True)
    return dict(path=str(path), sha256=file_sha256(path))


def identity(args):
    return dict(schema=1, manifest=hashed(args.manifest), encoder=hashed(args.encoder),
                djxl=hashed(args.djxl) if args.djxl else None,
                scripts=[hashed(Path(__file__).with_name(name)) for name in
                         ("benchmark_portfolio.py", "corpus_inputs.py", "compare_cpp_port.py")],
                config={name: getattr(args, name) for name in
                        ("threads", "effort", "max_probe_encodes", "max_finalists",
                         "probe_side", "timeout", "skip_verify", "limit", "candidates",
                         "serial_probes", "preparation_cache_mib", "no_finalist_bound")})


def inventory(manifest, limit=None):
    spec = loads(manifest.read_text(encoding="utf-8-sig"))
    rows = spec.get("inputs", spec.get("cases")) if isinstance(spec, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest requires nonempty inputs[] or cases[]")
    cases, excluded, aliases, seen = [], [], [], {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {index} must be an object")
        source = row.get("path", row.get("original", row.get("source")))
        if not isinstance(source, str) or not source:
            raise ValueError(f"Manifest row {index} has no source path")
        path = Path(source)
        path = (path if path.is_absolute() else manifest.parent / path).resolve()
        base = dict(source_index=index, original=str(path), dataset=row.get("dataset"),
                    stratum=row.get("stratum"), relative=row.get("relative"))
        if path.suffix.lower() != ".png":
            reason = ("JPEG not in pixel-lossless PNG scope" if path.suffix.lower() in
                      (".jpg", ".jpeg", ".jpe") else "Non-PNG input not in scope")
            excluded.append(dict(base, reason=reason))
            continue
        if limit is not None and len(cases) >= limit:
            excluded.append(dict(base, reason="Explicit --limit; not attempted"))
            continue
        case = dict(base, sha256=None, status="ready", error=None)
        try:
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            case.update(sha256=sha, input_bytes=len(data))
            for key in ("sha256", "input_sha256", "original_sha256"):
                if key in row and row[key] != sha:
                    raise ValueError(f"Declared original {key} mismatch")
            if sha in seen:
                aliases.append(dict(base, sha256=sha, duplicate_of=seen[sha]))
                continue
            seen[sha] = str(path)
            _, info = png_framing(data)
            for key in ("width", "height", "bit_depth", "color_type"):
                if key in row and row[key] != info[key]:
                    raise ValueError(f"Declared original {key} mismatch")
            case.update(**{k: info[k] for k in ("width", "height", "bit_depth", "color_type")})
            case.update(pixels=info["width"] * info["height"],
                        large_case=info["width"] * info["height"] > 8_000_000)
        except (OSError, ValueError) as error:
            case.update(status="failed", error=str(error))
        cases.append(case)
    return dict(schema=1, source_rows=len(rows), cases=cases, excluded=excluded,
                duplicate_aliases=aliases,
                validation="Original SHA256 and PNG framing/CRC/APNG guards; no pixel decode yet",
                scope="Original PNG only; no resize, crop, replacement or default limit")


def number(value, name, positive=False, integer=False):
    if ((type(value) is not int if integer else type(value) not in (int, float))
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"Invalid receipt {name}")


def validate_receipt(receipt, case, args, size=None):
    if not isinstance(receipt, dict) or type(receipt.get("schema")) is not int or receipt["schema"] != 1:
        raise ValueError("Invalid native receipt schema")
    for key in ("effort", "threads"):
        if key in receipt and (type(receipt[key]) is not int or receipt[key] != getattr(args, key)):
            raise ValueError(f"Native receipt {key} differs from configured value")
    for key, expected in (("parallel_probes", not args.serial_probes),
                          ("preparation_cache_bytes", args.preparation_cache_mib << 20),
                          ("bound_finalists", not args.no_finalist_bound)):
        if key in receipt and (type(receipt[key]) is not type(expected) or receipt[key] != expected):
            raise ValueError(f"Native receipt {key} differs from configured value")
    pixels = case["pixels"]
    for key in ("width", "height"):
        if type(receipt.get(key)) is not int or receipt[key] != case[key]:
            raise ValueError("Native dimensions differ from original")
    number(receipt.get("bytes"), "bytes", positive=True, integer=True)
    if size is not None and receipt["bytes"] != size:
        raise ValueError("Native bytes differ from chosen output size")
    for key in ("seconds", "bpp", "mpix_per_second", "scout_seconds"):
        number(receipt.get(key), key, positive=key == "seconds")
    for key in ("probe_encodes", "probe_pixels", "full_encodes"):
        number(receipt.get(key), key, integer=True)
    if "probe_workers" in receipt:
        number(receipt["probe_workers"], "probe_workers", integer=True)
        if receipt["probe_workers"] > min(args.threads, receipt["probe_encodes"],
                                          1 if args.serial_probes else args.threads):
            raise ValueError("Native receipt probe_workers exceeds configured work budget")
    if "preparation_reuse" in receipt:
        reuse = receipt["preparation_reuse"]
        if not isinstance(reuse, dict):
            raise ValueError("Invalid receipt preparation_reuse")
        for key in ("hits", "misses", "duplicate_builds", "bypasses", "entries",
                    "retained_bytes", "peak_bytes"):
            number(reuse.get(key), "preparation_reuse." + key, integer=True)
        if reuse["retained_bytes"] > reuse["peak_bytes"]:
            raise ValueError("Invalid receipt preparation_reuse retained/peak bytes")
    if (receipt["probe_encodes"] > args.max_probe_encodes
            or not 1 <= receipt["full_encodes"] <= 1 + args.max_finalists
            or receipt["scout_seconds"] > receipt["seconds"]):
        raise ValueError("Native receipt exceeds configured work budget")
    if not math.isclose(receipt["bpp"], receipt["bytes"] * 8 / pixels, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Native bpp does not match bytes/pixels")
    if not math.isclose(receipt["mpix_per_second"], pixels / 1e6 / receipt["seconds"], rel_tol=1e-4, abs_tol=1e-8):
        raise ValueError("Native throughput does not match total seconds")
    baseline = receipt.get("baseline")
    if not isinstance(baseline, dict):
            raise ValueError("Missing ordinary baseline receipt")
    for key in ("bytes", "seconds", "bpp"):
        number(baseline.get(key), "baseline." + key, positive=key == "bytes", integer=key == "bytes")
    if (baseline["seconds"] > receipt["seconds"] or receipt["bytes"] > baseline["bytes"]
            or not math.isclose(baseline["bpp"], baseline["bytes"] * 8 / pixels, rel_tol=1e-5, abs_tol=1e-8)):
        raise ValueError("Inconsistent baseline receipt")
    if (not isinstance(receipt.get("winner"), str) or not receipt["winner"]
            or not isinstance(receipt.get("regions"), list)
            or not isinstance(receipt.get("features"), dict)
            or not isinstance(receipt.get("candidates"), list)):
        raise ValueError("Missing routing evidence")
    for candidate in receipt["candidates"]:
        if not isinstance(candidate, dict):
            raise ValueError("Invalid candidate record")
        for key in ("name", "family", "decision"):
            if not isinstance(candidate.get(key), str):
                raise ValueError("Invalid candidate " + key)
        number(candidate.get("priority"), "priority")
        number(candidate.get("probe_seconds"), "probe_seconds")
        if not isinstance(candidate.get("probe_bytes"), list):
            raise ValueError("Invalid candidate probe_bytes")
        for value in candidate["probe_bytes"]:
            if value is not None:
                number(value, "probe_bytes", integer=True)
        for key in ("full_bytes", "full_seconds"):
            if key not in candidate:
                raise ValueError("Missing candidate " + key)
            if candidate[key] is not None:
                number(candidate[key], key, integer=key == "full_bytes")
        if "error" not in candidate or (candidate["error"] is not None and not isinstance(candidate["error"], str)):
            raise ValueError("Invalid candidate error")
        if (candidate["decision"] in ("winner", "full-rejected")
                or candidate["name"] in (receipt["winner"], "baseline")):
            number(candidate["full_bytes"], "completed full_bytes", positive=True, integer=True)
            number(candidate["full_seconds"], "completed full_seconds")
            if candidate["full_seconds"] > receipt["seconds"]:
                raise ValueError("Completed candidate time exceeds total seconds")
        if candidate["decision"] == "winner" and candidate["name"] != receipt["winner"]:
            raise ValueError("Candidate winner decision differs from receipt winner")
        lower_bound = candidate.get("full_lower_bound")
        if lower_bound is not None:
            number(lower_bound, "full_lower_bound", integer=True)
            if candidate["decision"] != "full-bounded":
                raise ValueError("Candidate full_lower_bound requires full-bounded decision")
        if candidate["decision"] == "full-bounded":
            # A prior incumbent may have been larger than the eventual winner.
            if (lower_bound is None or lower_bound < receipt["bytes"]
                    or candidate["full_bytes"] is not None or candidate["full_seconds"] is None
                    or candidate["error"] is not None or args.no_finalist_bound
                    or candidate["name"] == receipt["winner"]):
                raise ValueError("Invalid candidate full-bounded receipt")
    winners = [c for c in receipt["candidates"] if c["name"] == receipt["winner"]]
    if len(winners) != 1:
        raise ValueError("Receipt winner must identify exactly one candidate")
    winner = winners[0]
    # The baseline can retain a complete output even when its later scout fails.
    scout_failed_baseline = (winner["name"] == "baseline"
                             and winner["decision"] == "probe-error"
                             and winner["error"] is not None)
    # Preserve schema-1 finalist labels when complete evidence identifies the winner.
    if (winner["full_bytes"] != receipt["bytes"]
            or (not scout_failed_baseline and
                (winner["decision"] not in ("winner", "finalist") or winner["error"] is not None))):
        raise ValueError("Inconsistent complete winner evidence")
    baselines = [c for c in receipt["candidates"] if c["name"] == "baseline"]
    if len(baselines) != 1:
        raise ValueError("Receipt baseline must identify exactly one candidate")
    ordinary = baselines[0]
    if (ordinary["decision"] not in ("winner", "full-rejected", "probe-error")
            or ordinary["full_bytes"] != baseline["bytes"]
            or ordinary["full_seconds"] != baseline["seconds"]):
        raise ValueError("Inconsistent baseline candidate evidence")


def read_journal(path, cases, args, run_id):
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Torn journal record; refusing resume")
            row = loads(line)
            if not isinstance(row, dict):
                raise ValueError("Malformed journal record")
            checksum = row.pop("checksum", None)
            if digest(row) != checksum or row.get("run_id") != run_id:
                raise ValueError("Journal checksum/run identity mismatch")
            index = len(rows)
            if index >= len(cases) or row.get("index") != index or type(row.get("index")) is not int:
                raise ValueError("Journal order/index mismatch")
            if row.get("case") != cases[index] or row.get("status") not in ("passed", "unverified", "failed"):
                raise ValueError("Journal source identity/status mismatch")
            if type(row.get("pixels_verified")) is not bool:
                raise ValueError("Invalid verification record")
            if type(row.get("receipt_validated")) is not bool:
                raise ValueError("Invalid native receipt validation record")
            for key in ("encoder", "decoder"):
                process = row.get(key)
                if process is not None:
                    if (not isinstance(process, dict) or "returncode" not in process
                            or (process["returncode"] is not None and type(process["returncode"]) is not int)
                            or not isinstance(process.get("stdout"), str)
                            or not isinstance(process.get("stderr"), str)
                            or not isinstance(process.get("command"), list)
                            or not process["command"]
                            or any(not isinstance(token, str) for token in process["command"])):
                        raise ValueError("Malformed subprocess evidence")
                    number(process.get("wall_seconds"), "wall_seconds")
            if row["receipt_validated"]:
                validate_receipt(row.get("measurement"), cases[index], args)
                if (not isinstance(row.get("encoder"), dict)
                        or row["encoder"].get("returncode") != 0
                        or loads(row["encoder"]["stdout"]) != row["measurement"]):
                    raise ValueError("Native stdout differs from validated receipt")
            if row["status"] == "failed":
                if not isinstance(row.get("error"), str) or not row["error"]:
                    raise ValueError("Failed record requires an error")
            else:
                if (not row["receipt_validated"] or cases[index]["status"] != "ready" or row.get("error") is not None
                        or row["pixels_verified"] != (not args.skip_verify)
                        or row["status"] != ("unverified" if args.skip_verify else "passed")
                        or not isinstance(row.get("payload_sha256"), str)
                        or len(row["payload_sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in row["payload_sha256"])):
                    raise ValueError("Invalid success evidence")
                validate_receipt(row.get("measurement"), cases[index], args)
                for key in (("encoder",) if args.skip_verify else ("encoder", "decoder")):
                    if not isinstance(row.get(key), dict) or row[key].get("returncode") != 0:
                        raise ValueError("Missing successful subprocess evidence")
            rows.append(row)
    return rows


def summarize(rows, population, mode):
    successes = [r for r in rows if r["status"] in ("passed", "unverified")]
    decision_counts, error_counts, errors_by_name = Counter(), Counter(), Counter()
    decisions_by_name = {}
    cases_with_errors = 0
    for row in rows:
        if not row.get("receipt_validated"):
            continue
        has_error = False
        for candidate in row["measurement"]["candidates"]:
            name, decision = candidate["name"], candidate["decision"]
            decision_counts[decision] += 1
            decisions_by_name.setdefault(name, Counter())[decision] += 1
            # Structural ineligibility can also carry an error string; it is
            # not an attempted encode failure and must stay in decision counts.
            if decision in ("probe-error", "full-error"):
                error_counts[decision] += 1
                errors_by_name[name] += 1
                has_error = True
        cases_with_errors += has_error
    def totals(items):
        pixels = sum(r["case"]["pixels"] for r in items)
        size = sum(r["measurement"]["bytes"] for r in items)
        baseline_size = sum(r["measurement"]["baseline"]["bytes"] for r in items)
        native_seconds = sum(r["measurement"]["seconds"] for r in items)
        return dict(count=len(items), pixels=pixels, bytes=size,
                    arithmetic_mean_bpp=(sum(r["measurement"]["bpp"] for r in items) / len(items) if items else None),
                    aggregate_weighted_bpp=8 * size / pixels if pixels else None,
                    baseline_bytes=baseline_size,
                    baseline_arithmetic_mean_bpp=(sum(r["measurement"]["baseline"]["bpp"] for r in items) / len(items) if items else None),
                    baseline_aggregate_weighted_bpp=8 * baseline_size / pixels if pixels else None,
                    byte_change_percent=100 * (size - baseline_size) / baseline_size if baseline_size else None,
                    overall_mpix_per_second=pixels / 1e6 / native_seconds if native_seconds else None,
                    native_seconds=native_seconds,
                    baseline_seconds=sum(r["measurement"]["baseline"]["seconds"] for r in items),
                    scout_seconds=sum(r["measurement"]["scout_seconds"] for r in items),
                    full_encode_seconds=sum(sum(c["full_seconds"] or 0 for c in r["measurement"]["candidates"])
                                            for r in items))
    return dict(schema=1, mode=mode, total_cases=len(population["cases"]), completed=len(rows),
                pending=len(population["cases"]) - len(rows),
                statuses=dict(Counter(r["status"] for r in rows)),
                candidate_decision_counts=dict(decision_counts),
                candidate_decision_counts_by_name={name: dict(counts) for name, counts in decisions_by_name.items()},
                candidate_error_counts=dict(error_counts),
                candidate_error_counts_by_name=dict(errors_by_name),
                candidate_errors=sum(error_counts.values()),
                cases_with_candidate_errors=cases_with_errors,
                preparation_failures=sum(c["status"] == "failed" for c in population["cases"]),
                excluded=len(population["excluded"]), duplicate_aliases=len(population["duplicate_aliases"]),
                successful=totals(successes),
                observed_native=totals([r for r in rows if r.get("receipt_validated")]),
                encoder_wall_seconds=sum(r["encoder"]["wall_seconds"] for r in rows if r.get("encoder")),
                native_time_unavailable_calls=sum(bool(r.get("encoder")) and not r.get("receipt_validated") for r in rows),
                large_cases=dict(planned=sum(c.get("large_case", False) for c in population["cases"]),
                                 failed=sum(r["status"] == "failed" and r["case"].get("large_case", False) for r in rows),
                                 successful=totals([r for r in successes if r["case"].get("large_case")])),
                timing="Native seconds include routing, baseline and finalists; full/scout counters overlap total. observed_native includes valid receipts even when pixel verification failed; successful excludes failures.",
                temporary_outputs_cleaned=True)


def trial(index, case, args, frozen_id, env):
    row = dict(run_id=frozen_id, index=index, case=case, status="failed", error=None,
               measurement=None, encoder=None, decoder=None, pixels_verified=False,
               payload_sha256=None, receipt_validated=False)
    try:
        if case["status"] != "ready":
            raise ValueError(case["error"])
        if file_sha256(Path(case["original"])) != case["sha256"]:
            raise ValueError("Source SHA256 changed before trial")
        with tempfile.TemporaryDirectory(prefix="portfolio-") as directory:
            directory = Path(directory)
            info, snapshot, expected = prepare(Path(case["original"]), directory)
            if (info["kind"] != "png" or info["input_sha256"] != case["sha256"]
                    or (info["width"], info["height"], info["depth"]) !=
                    (case["width"], case["height"], max(8, case["bit_depth"]))):
                raise ValueError("Prepared input identity/precision mismatch")
            row["payload_sha256"] = info["payload_sha256"]
            output, restored = directory / "chosen.jxl", directory / "restored.png"
            command = [str(args.encoder.resolve()), str(snapshot), "--output", str(output)]
            for name in ("threads", "effort", "max_probe_encodes", "max_finalists", "probe_side",
                         "preparation_cache_mib"):
                command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
            for name in ("serial_probes", "no_finalist_bound"):
                if getattr(args, name):
                    command.append("--" + name.replace("_", "-"))
            if args.candidates is not None:
                command.extend(["--candidates", args.candidates])
            row["encoder"] = run_process(command, env, args.timeout)
            check_process(row["encoder"])
            row["measurement"] = loads(row["encoder"]["stdout"])
            validate_receipt(row["measurement"], case, args, output.stat().st_size)
            row["receipt_validated"] = True
            if not args.skip_verify:
                row["decoder"] = run_process(decoder_command(args.djxl.resolve(), output, restored,
                                                              expected.depth), env, args.timeout)
                check_process(row["decoder"])
                compare_pixels(expected, png_samples(restored.read_bytes()))
                row["pixels_verified"] = True
            if file_sha256(Path(case["original"])) != case["sha256"]:
                raise ValueError("Source SHA256 changed during trial")
            row["status"] = "unverified" if args.skip_verify else "passed"
    except (OSError, ValueError, RuntimeError) as error:
        row.update(status="failed", error=str(error))
    return row


def benchmark(args):
    frozen = identity(args)
    population = inventory(args.manifest.resolve(), args.limit)
    if identity(args) != frozen:
        raise ValueError("Manifest or executable changed during preparation")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / "run.lock"
    # Exclusive writer; a stale lock after a hard kill requires explicit removal.
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        run_path, manifest_path = args.out / "run.json", args.out / "manifest.json"
        journal = args.out / "trials.jsonl"
        run_identity = dict(identity=frozen, inventory_sha256=digest(population))
        run_id = digest(run_identity)
        if args.resume:
            saved = loads(run_path.read_text(encoding="utf-8"))
            saved_population = loads(manifest_path.read_text(encoding="utf-8"))
            if saved.get("run_id") != run_id or saved.get("frozen") != run_identity or saved_population != population:
                raise ValueError("Resume requires unchanged config, executables, manifest and input SHA256")
            rows = read_journal(journal, population["cases"], args, run_id)
        else:
            if any(p != lock for p in args.out.iterdir()):
                raise ValueError("Output directory is not empty; use --resume")
            atomic_write(manifest_path, population)
            atomic_write(run_path, dict(schema=1, run_id=run_id, frozen=run_identity,
                                        created_utc=datetime.now(timezone.utc).isoformat()))
            with journal.open("x", encoding="utf-8") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            rows = []
        mode = "run" if args.run else "prepare-only"
        def progress(phase, active=None):
            atomic_write(args.out / "progress.json", dict(schema=1, mode=mode, phase=phase,
                         completed=len(rows), total=len(population["cases"]), active=active,
                         updated_utc=datetime.now(timezone.utc).isoformat()))
        progress("prepared")
        if args.run:
            env, _ = clean_environment(os.environ)
            active = None
            try:
                for index in range(len(rows), len(population["cases"])):
                    if identity(args) != frozen:
                        raise ValueError("Executable/config identity changed during run")
                    active = index
                    progress("encoding", index)
                    row = trial(index, population["cases"][index], args, run_id, env)
                    sealed = dict(row, checksum=digest(row))
                    with journal.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(canonical(sealed) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(row)
                    active = None
                    atomic_write(args.out / "summary.json", summarize(rows, population, mode))
                    progress("checkpointed")
            except KeyboardInterrupt:
                atomic_write(args.out / "summary.json", summarize(rows, population, mode))
                progress("interrupted", active)
                raise
            except Exception:
                progress("failed", active)
                raise
        result = summarize(rows, population, mode)
        atomic_write(args.out / "summary.json", result)
        progress("complete" if args.run else "prepared")
        return result
    finally:
        lock.unlink()


def main(argv=None):
    args = arguments(argv)
    try:
        result = benchmark(args)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
