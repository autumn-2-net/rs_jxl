"""Refresh only native effort curves on a previously verified, unchanged corpus.

Cached rs_jxl/cjxl measurements are validated, never executed or overwritten.
One complete native encode per new trial; official djxl checks original pixels.
"""

import argparse
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import benchmark_context_corpus as corpus
from benchmark_palette_runs import prepare_png
import compare_cpp_port as port
import plot_context_corpus as old_plot


SCHEMA = "effort-refresh-v1"
TIMING = "recipe projection and one complete native encode, including pool setup; excludes image IO"
require = corpus.require


def atomic_json(path, value):
    """Retry Windows replace conflicts; durable evidence must still fail on IO errors."""
    delays = (.025, .05, .1, .2, .4)
    for attempt in range(len(delays) + 1):
        try:
            corpus.atomic_json(path, value)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


class ProgressWriter:
    """The replaceable display snapshot is advisory, never the result journal."""

    def __init__(self, path):
        self.path = path
        self.failures = 0
        self.warning_active = False

    def write(self, value):
        try:
            atomic_json(self.path, dict(value, progress_write_failures=self.failures))
        except OSError as error:
            self.failures += 1
            if not self.warning_active:
                print(f"Progress snapshot unavailable; results remain in trials.jsonl: {error}",
                      file=sys.stderr, flush=True)
            self.warning_active = True
            return False
        self.warning_active = False
        return True


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("previous", "encoder", "decoder", "outdir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--label", default="rs_jxl current (new)")
    parser.add_argument("--efforts", type=int, nargs="+", choices=range(1, 10),
                        default=list(range(1, 10)))
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-harness-update", action="store_true",
                        help="Explicitly permit source-only runner repairs on resume; preserve prior receipts")
    parser.add_argument("--plot", action="store_true", help="Render both charts after success")
    args = parser.parse_args(argv)
    if len(set(args.efforts)) != len(args.efforts) or not corpus.positive(args.timeout):
        parser.error("Require distinct efforts and positive finite timeout")
    if args.resume_harness_update and not args.resume:
        parser.error("--resume-harness-update requires --resume")
    return args


def previous_report(info):
    data = Path(info["path"]).read_bytes()
    require(port.sha256(data) == info["sha256"], "Cached report changed")
    report = corpus.loads(data)
    _, totals = old_plot.validate_report(report)
    return report, totals


def freeze(args):
    previous = corpus.hashed(args.previous)
    old, _ = previous_report(previous)
    config = old["identity"]["config"]
    require(set(args.efforts) <= set(config["efforts"]), "Efforts missing from cached baseline")
    decoder = corpus.hashed(args.decoder)
    require(decoder["sha256"] == old["identity"]["binaries"]["decoder"]["sha256"],
            "Use the same official decoder as the cached comparison")
    sources = {Path(__file__).resolve(), Path(old_plot.__file__).resolve(),
               Path(__file__).with_name("plot_effort_refresh.py").resolve(),
               *corpus.helper_sources()}
    identity = dict(schema=SCHEMA, previous_report=previous, previous_fingerprint=old["fingerprint"],
                    inputs=old["identity"]["inputs"], label=args.label,
                    binaries=dict(encoder=corpus.hashed(args.encoder), decoder=decoder),
                    sources=[corpus.hashed(p) for p in sorted(sources)],
                    config=dict(efforts=sorted(args.efforts), threads=config["threads"],
                                repeats=config["repeats"], timeout=args.timeout, serial=True,
                                patches="auto", search=False))
    check_frozen(identity)
    return identity


def check_frozen(identity):
    for info in [identity["previous_report"], *identity["inputs"],
                 *identity["binaries"].values(), *identity["sources"]]:
        require(port.file_sha256(Path(info["path"])) == info["sha256"],
                f"Frozen input changed: {info['path']}")


def measurement_identity(identity):
    # Only an explicitly accepted harness repair may vary; binaries, inputs,
    # commands, timing configuration and cached baselines must not change.
    return {k: v for k, v in identity.items() if k != "sources"}


def resume_identity(saved, current, allow_harness_update=False):
    original = saved["identity"]
    require(saved["fingerprint"] == corpus.fingerprint(original), "Stored identity checksum mismatch")
    require(original == current or (allow_harness_update
            and measurement_identity(original) == measurement_identity(current)),
            "Resume identity changed; source-only repairs require --resume-harness-update")
    return original


def validate_runtimes(identity, runtimes, rows):
    digest = corpus.fingerprint(identity)
    require(runtimes.get(digest) == identity, "Original runtime identity missing")
    for runtime_digest, runtime in runtimes.items():
        require(runtime_digest == corpus.fingerprint(runtime)
                and measurement_identity(runtime) == measurement_identity(identity),
                "Runtime changed benchmark inputs, binaries or settings")
    require(all(r.get("runtime_fingerprint", digest) in runtimes for r in rows),
            "Unknown trial runtime")


def command(identity, effort, source, output):
    return [identity["binaries"]["encoder"]["path"], str(source), "--output", str(output),
            "--effort", str(effort), "--threads", str(identity["config"]["threads"]),
            "--direct-candidate", "baseline"]


def pure_gray(framing):
    # Native PNG expansion turns a grayscale transparency key into GrayAlpha.
    return framing["color_type"] == 0 and "transparency_key" not in framing


def validate_native(m, identity, row):
    case = identity["inputs"][row["case_index"]]
    require(isinstance(m, dict), "Missing native receipt")
    for key, expected in dict(effort=row["effort"], threads=identity["config"]["threads"],
                              encode_calls=1, bytes=row["bytes"], width=case["width"],
                              height=case["height"]).items():
        require(type(m.get(key)) is int and m[key] == expected, f"Native {key} mismatch")
    require(m.get("direct") is True and m.get("candidate") == "baseline"
            and m.get("timing") == TIMING, "Not one ordinary native encode")
    require(corpus.positive(m.get("seconds")) and m["seconds"] <= row["wall_seconds"] * 1.01 + .001,
            "Invalid native timer")
    pixels = case["width"] * case["height"]
    for key, expected in (("bpp", 8 * row["bytes"] / pixels),
                          ("mpix_per_second", pixels / 1e6 / m["seconds"])):
        require(corpus.positive(m.get(key)) and math.isclose(m[key], expected, rel_tol=1e-5),
                f"Native {key} mismatch")
    tools = m.get("source_tools")
    require(isinstance(tools, dict) and "profile" in tools and tools["profile"] is None
            and tools.get("max_previous_channels") is None, "Unexpected experimental tool profile")
    if row["effort"] == 9:
        require(tools.get("lz_quick") == "Sparse" and tools.get("palette_order") == "ScreenedRuns"
                and type(tools.get("previous_channels")) is int
                and tools["previous_channels"] == int(not row["pure_gray"])
                and m.get("direct_ma_profile") == "half-single-pass"
                and m.get("direct_tree_learning") == "symmetric", "New e9 defaults missing")
    else:
        require(tools.get("lz_quick") in (None, "Off")
                and tools.get("palette_order") == "Default"
                and tools.get("previous_channels") == 0, "Unexpected non-e9 defaults")


def key(row):
    return row["case_index"], row["effort"], row["repetition"]


def validate_row(row, identity, digest):
    require(row.get("fingerprint") == digest and row.get("row_sha256") == corpus.fingerprint(
        {k: v for k, v in row.items() if k != "row_sha256"}), "Journal fingerprint/checksum mismatch")
    index, effort, repeat = key(row)
    cfg = identity["config"]
    require(type(index) is int and 0 <= index < len(identity["inputs"])
            and type(effort) is int and effort in cfg["efforts"]
            and type(repeat) is int and 0 <= repeat < cfg["repeats"], "Journal index mismatch")
    require(row.get("source_sha256") == identity["inputs"][index]["sha256"], "Journal source mismatch")
    require(type(row.get("verified")) is bool and row.get("status") in ("verified", "failed")
            and row["verified"] == (row["status"] == "verified"), "Journal status mismatch")
    if not row["verified"]:
        return
    require(type(row.get("bytes")) is int and row["bytes"] > 0
            and type(row.get("pure_gray")) is bool and corpus.positive(row.get("wall_seconds")),
            "Invalid trial size/time/layout")
    ep, dp = row["encoder_process"], row["decoder_process"]
    require(ep.get("returncode") == dp.get("returncode") == 0
            and ep.get("wall_seconds") == row["wall_seconds"], "Failed verification processes")
    argv = ep["command"]
    require(argv == command(identity, effort, argv[1], argv[3]), "Nondefault encoder command")
    evidence = row.get("verification", {})
    require(evidence.get("original_samples_equal") is True and evidence.get("includes_hidden_rgb") is True
            and evidence.get("depth") in (8, 16), "Missing pixel verification")
    require(dp["command"] == port.decoder_command(identity["binaries"]["decoder"]["path"],
                argv[3], dp["command"][2], evidence["depth"]), "Wrong oracle command")
    validate_native(row["measurement"], identity, row)


def latest_rows(rows, identity, digest):
    result = {}
    for row in rows:
        validate_row(row, identity, digest)
        require(key(row) not in result or not result[key(row)]["verified"], "Duplicate verified trial")
        result[key(row)] = row
    return result


def series(identity, rows, old, old_totals):
    cfg = identity["config"]
    require(identity["inputs"] == old["identity"]["inputs"]
            and identity["previous_fingerprint"] == old["fingerprint"]
            and cfg["threads"] == old["identity"]["config"]["threads"]
            and cfg["repeats"] == old["identity"]["config"]["repeats"], "Unmatched cached cohort/settings")
    trials = latest_rows(rows, identity, corpus.fingerprint(identity))
    require(len(trials) == len(identity["inputs"]) * len(cfg["efforts"]) * cfg["repeats"]
            and all(r["verified"] for r in trials.values()), "Incomplete verified matrix")
    images = {}
    for effort in cfg["efforts"]:
        values = []
        for index, case in enumerate(identity["inputs"]):
            repeats = [trials[index, effort, r] for r in range(cfg["repeats"])]
            size = statistics.median(r["bytes"] for r in repeats)
            seconds = statistics.median(r["measurement"]["seconds"] for r in repeats)
            pixels = case["width"] * case["height"]
            values.append(dict(bytes=size, pixels=pixels, seconds=seconds, bpp=8 * size / pixels,
                               native_seconds_lower=seconds, native_seconds_upper=seconds))
        images[effort] = values
    totals = corpus.aggregate(images)
    return dict(new=dict(label=identity["label"], points=[dict(effort=e, **totals[e]) for e in cfg["efforts"]]),
                previous=dict(label="rs_jxl previous (2026-11-12, cached)", points=[
                    dict(effort=e, **old_totals[f"rust-e{e}"]) for e in cfg["efforts"]]),
                cjxl=dict(label="cjxl 0.12 (cached)", points=[
                    dict(effort=e, **old_totals[f"cjxl-e{e}"]) for e in cfg["efforts"]]))


def validate_report(report):
    require(report.get("schema") == SCHEMA and report.get("status") == "complete"
            and report.get("integrity_verified") is True and report.get("errors") == [],
            "Cannot plot an incomplete or integrity-failed refresh")
    identity = report["identity"]
    require(report["fingerprint"] == corpus.fingerprint(identity), "Report fingerprint mismatch")
    cfg = identity["config"]
    require(identity["schema"] == SCHEMA and cfg["serial"] is True and cfg["search"] is False
            and cfg["patches"] == "auto" and cfg["efforts"]
            and cfg["efforts"] == sorted(set(cfg["efforts"]))
            and all(type(e) is int and 1 <= e <= 9 for e in cfg["efforts"]), "Invalid refresh policy")
    old, totals = previous_report(identity["previous_report"])
    validate_runtimes(identity, report.get("runtime_identities", {report["fingerprint"]: identity}),
                      report["trials"])
    result = series(identity, report["trials"], old, totals)
    expected = len(identity["inputs"]) * len(cfg["efforts"]) * cfg["repeats"]
    require(report["verified_trials"] == report["expected_trials"] == expected
            and report["failed_trials"] == 0 and report["series"] == result,
            "Incorrect report aggregate/completeness")
    return result


def run_trial(identity, digest, index, effort, repetition, snapshot, expected, pure_gray, folder, env):
    row = dict(fingerprint=digest, case_index=index, effort=effort, repetition=repetition,
               source_sha256=identity["inputs"][index]["sha256"], pure_gray=pure_gray,
               verified=False, status="failed", recorded_at=corpus.now())
    with tempfile.TemporaryDirectory(prefix="trial-", dir=folder) as tmp:
        jxl, png = Path(tmp) / "output.jxl", Path(tmp) / "decoded.png"
        errors = []
        try:
            ep = port.run_process(command(identity, effort, snapshot, jxl), env, identity["config"]["timeout"])
            row.update(encoder_process=ep, wall_seconds=ep["wall_seconds"])
            port.check_process(ep)
            row["bytes"] = jxl.stat().st_size
            row["measurement"] = corpus.loads(ep["stdout"])
            validate_native(row["measurement"], identity, row)
        except Exception as error:
            errors.append(f"encode: {error}")
        if jxl.exists():
            try:
                dp = port.run_process(port.decoder_command(identity["binaries"]["decoder"]["path"],
                    jxl, png, expected.depth), env, identity["config"]["timeout"])
                row["decoder_process"] = dp
                port.check_process(dp)
                port.compare_pixels(expected, port.png_samples(png.read_bytes()))
                row["verification"] = dict(original_samples_equal=True, includes_hidden_rgb=True,
                    depth=expected.depth, snapshot_sha256=port.file_sha256(snapshot),
                    jxl_sha256=port.file_sha256(jxl), decoded_png_sha256=port.file_sha256(png))
            except Exception as error:
                errors.append(f"decode/pixels: {error}")
        else:
            errors.append("Missing JXL output")
        if errors:
            row["error"] = "; ".join(errors)
        else:
            row.update(verified=True, status="verified")
    return corpus.seal_row(row)


def benchmark(args):
    with corpus.run_lock(args.outdir):
        current = freeze(args)
        runtime_digest = corpus.fingerprint(current)
        identity_path, journal_path = args.outdir / "identity.json", args.outdir / "trials.jsonl"
        if args.resume:
            identity = resume_identity(corpus.loads(identity_path.read_bytes()), current,
                                       args.resume_harness_update)
        else:
            identity = current
            require(not any(p.name != ".run.lock" for p in args.outdir.iterdir()), "Output directory must be new")
            atomic_json(identity_path, dict(identity=identity, fingerprint=runtime_digest))
        digest = corpus.fingerprint(identity)
        runtimes = {digest: identity}
        for path in sorted(args.outdir.glob("runtime-*.json")):
            stored = corpus.loads(path.read_bytes())
            require(stored["fingerprint"] == corpus.fingerprint(stored["identity"]),
                    "Runtime identity checksum mismatch")
            runtimes[stored["fingerprint"]] = stored["identity"]
        rows = []
        if journal_path.exists():
            data = journal_path.read_bytes()
            if data and not data.endswith(b"\n"):
                end = data.rfind(b"\n") + 1
                backup = journal_path.with_name(f"trials.torn-{time.time_ns()}.bin")
                backup.write_bytes(data[end:])
                with journal_path.open("r+b") as stream:
                    stream.truncate(end)
                data = data[:end]
            rows = [corpus.loads(line) for line in data.splitlines()]
        latest = latest_rows(rows, identity, digest)
        validate_runtimes(identity, runtimes, rows)
        runtimes[runtime_digest] = current
        atomic_json(args.outdir / f"runtime-{runtime_digest}.json",
                    dict(identity=current, fingerprint=runtime_digest))
        cfg = identity["config"]
        expected_trials = len(identity["inputs"]) * len(cfg["efforts"]) * cfg["repeats"]
        env, environment = port.clean_environment(os.environ)
        started, errors, integrity = time.perf_counter(), [], True
        writer = ProgressWriter(args.outdir / "progress.json")

        def progress(status, active=None):
            elapsed = time.perf_counter() - started
            done = len(rows) - initial_attempts
            remaining = expected_trials - sum(r["verified"] for r in latest.values())
            writer.write(dict(status=status, pid=os.getpid(),
                updated_at=corpus.now(), expected_trials=expected_trials, verified_trials=sum(
                    r["verified"] for r in latest.values()), failed_trials=sum(
                    not r["verified"] for r in latest.values()), active=active, elapsed_seconds=elapsed,
                eta_seconds=elapsed / done * remaining if done else None))

        initial_attempts = len(rows)
        try:
            port.pixel_libraries()
            progress("running")
            with journal_path.open("a", encoding="utf-8", newline="\n") as journal:
                for index, case in enumerate(identity["inputs"]):
                    if all(latest.get((index, e, r), {}).get("verified") for e in cfg["efforts"]
                           for r in range(cfg["repeats"])):
                        continue
                    with tempfile.TemporaryDirectory(prefix="effort-refresh-", dir=args.outdir) as tmp:
                        folder = Path(tmp)
                        original = Path(case["path"]).read_bytes()
                        require(port.sha256(original) == case["sha256"], "Original input changed")
                        payload, pixels, _ = prepare_png(Path(case["path"]), original, folder)
                        _, framing = corpus.png_framing(original)
                        snapshot = folder / "input.png"
                        snapshot.write_bytes(payload)
                        for repeat in range(cfg["repeats"]):
                            for effort in corpus.trial_order(cfg["efforts"], index, repeat):
                                if latest.get((index, effort, repeat), {}).get("verified"):
                                    continue
                                progress("running", dict(image=index + 1, source=case["path"], effort=effort))
                                row = run_trial(identity, digest, index, effort, repeat, snapshot,
                                                pixels, pure_gray(framing), folder, env)
                                row["runtime_fingerprint"] = runtime_digest
                                row = corpus.seal_row(row)
                                validate_row(row, identity, digest)
                                journal.write(corpus.serialized(row) + "\n")
                                journal.flush()
                                os.fsync(journal.fileno())
                                rows.append(row)
                                latest[key(row)] = row
                                progress("running")
                                print(f"[{index + 1}/{len(identity['inputs'])}] new e{effort}: {row['status']}"
                                      + (" " + row["error"] if "error" in row else ""), flush=True)
        except (Exception, KeyboardInterrupt) as error:
            errors.append(f"{type(error).__name__}: {error}")
            print(f"Benchmark interrupted: {errors[-1]}", file=sys.stderr, flush=True)
        finally:
            try:
                check_frozen(current)
            except (OSError, ValueError) as error:
                errors.append(str(error))
                integrity = False
            complete = integrity and not errors and len(latest) == expected_trials and all(
                r["verified"] for r in latest.values())
            report = dict(schema=SCHEMA, status="complete" if complete else "incomplete", identity=identity,
                fingerprint=digest, trials=list(latest.values()), expected_trials=expected_trials,
                verified_trials=sum(r["verified"] for r in latest.values()),
                failed_trials=sum(not r["verified"] for r in latest.values()), errors=errors,
                integrity_verified=integrity, elapsed_seconds=time.perf_counter() - started,
                environment=environment, updated_at=corpus.now(),
                runtime_identities=runtimes, progress_write_failures=writer.failures,
                timing_note="Historical baselines reused; runs are not simultaneous; native time excludes IO.")
            if complete:
                old, totals = previous_report(identity["previous_report"])
                report["series"] = series(identity, report["trials"], old, totals)
            atomic_json(args.outdir / "summary.json", report)
            progress(report["status"])
        return report


def main(argv=None):
    args = arguments(argv)
    report = benchmark(args)
    if report["status"] == "complete" and args.plot:
        import plot_effort_refresh
        writer = ProgressWriter(args.outdir / "progress.json")
        progress = dict(status="plotting", pid=os.getpid(), updated_at=corpus.now(), active=None,
                        expected_trials=report["expected_trials"], verified_trials=report["verified_trials"],
                        failed_trials=report["failed_trials"], elapsed_seconds=report["elapsed_seconds"])
        writer.write(progress)
        for metric, name in (("mean-bpp", "bpp-time.png"), ("total-mib", "size-time.png")):
            status = plot_effort_refresh.main([str(args.outdir / "summary.json"), "--metric", metric,
                                              "--out", str(args.outdir.parent / name)])
            if status:
                progress.update(status="plot_failed", updated_at=corpus.now(), failed_plot=name)
                writer.write(progress)
                return status
        progress.update(status="complete", plots_complete=True, updated_at=corpus.now())
        writer.write(progress)
    print(f"{report['status']}: {report['verified_trials']}/{report['expected_trials']} new outputs verified")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
