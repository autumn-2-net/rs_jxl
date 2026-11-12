"""Serial direct-native context corpus experiment; prebuilt binaries only.

Defaults: 200-case admission cap, Rust/cjxl e1..9 plus 12 independent e9
experiments, one repeat. No sampling, resizing, builds, portfolio or RCT search.
Official binaries are caller-attested, hashed, and cjxl must identify as 0.12.
"""

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile

from benchmark_palette_runs import prepare_png
from corpus_inputs import png_framing
import compare_cpp_port as port


EXPERIMENT_MODES = (
    "auto12", "activity25", "activity81", "activity25-refine",
    "activity81-refine", "activity81-repair", "offset-distribution",
    "auto12-offset", "full14-static", "full14-static-offset", "lookahead2", "lookahead4",
)
FAST_MODES = {"auto12", "activity25", "activity81", "auto12-offset",
              "full14-static", "full14-static-offset"}
SEED_MODES = {"activity25-refine", "activity81-refine", "activity81-repair"}
OFFSET_MODES = {"offset-distribution", "auto12-offset", "full14-static-offset"}
FAST_COUNTERS = ("trees", "leaves", "pixels", "offset_leaves", "horizontal", "vertical",
                 "single", "joint", "cross_channel_plans")
SEED_COUNTERS = ("learners", "prepared_records", "seed_splits", "refined_leaves",
                 "repaired_branches", "merged_branches", "predictor_changes", "histogram_reuses")
NATIVE_TIMING = "native projection+analysis+encode; excludes startup and input/output IO"
MODEL_COST_CONTRACT = "emitted MA tree and entropy-header bits only; not estimated scores"
SCHEMA = "context-corpus-v1"
CJXL_TIMING = {
    "scope": "EncodeImageJXL only; excludes PNG decoding, process startup and file IO",
    "sources": ["tools/cjxl_main.cc: EncodeImageJXL, NotifyElapsed, num_reps",
                "tools/speed_stats.cc: SummaryStat %.3f, Print"],
    "rate_decimal_places": 3,
    "binary_source_relationship": "caller-attested; source hashes are not build provenance",
}
FINAL_STATS = re.compile(r"^([1-9][0-9]*) x ([1-9][0-9]*), ([0-9]+\.[0-9]{3}) MP/s, "
                         r"([0-9]+) threads\.$", re.MULTILINE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_number(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def loads(value):
    def finite_float(token):
        number = float(token)
        require(math.isfinite(number), "Nonfinite JSON float")
        return number
    return json.loads(value, object_pairs_hook=_pairs, parse_constant=_bad_number,
                      parse_float=finite_float)


def serialized(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(serialized(value).encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=path.name, suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "encoder", "cjxl", "decoder", "outdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--efforts", type=int, nargs="+", choices=range(1, 10),
                        default=list(range(1, 10)))
    parser.add_argument("--experiment-modes", nargs="+", choices=EXPERIMENT_MODES,
                        default=list(EXPERIMENT_MODES))
    parser.add_argument("--max-cases", type=int, default=200,
                        help="Admission cap, not truncation; raising requires an explicit value")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Per-process harness guard only, never an encoder route")
    parser.add_argument("--cjxl-source", type=Path,
                        help="Optional official source root; freeze CLI/timer source hashes")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 256 or not 1 <= args.repeats <= 100
            or args.max_cases < 1 or not positive(args.timeout)):
        parser.error("Invalid threads/repeats/max-cases/timeout")
    if (len(set(args.efforts)) != len(args.efforts)
            or len(set(args.experiment_modes)) != len(args.experiment_modes)):
        parser.error("Efforts and experiment modes must be distinct")
    return args


def variants(config):
    return ([dict(id=f"rust-e{e}", family="rust", mode="baseline", effort=e)
             for e in config["efforts"]]
            + [dict(id=f"cjxl-e{e}", family="cjxl", mode="official", effort=e)
               for e in config["efforts"]]
            + [dict(id=f"experiment-{mode}-e9", family="experiment", mode=mode, effort=9)
               for mode in config["experiment_modes"]])


def trial_order(points, case_index, repetition):
    shift = (case_index + repetition) % len(points)
    return points[shift:] + points[:shift]


def command(identity, variant, source, output):
    config = identity["config"]
    if variant["family"] == "cjxl":
        return [identity["binaries"]["cjxl"]["path"], str(source), str(output), "-d", "0",
                "-e", str(variant["effort"]), f"--num_threads={config['threads']}",
                "--lossless_jpeg=0", "--keep_invisible=1", "--num_reps=1"]
    return [identity["binaries"]["encoder"]["path"], str(source), "--output", str(output),
            "--mode", variant["mode"], "--effort", str(variant["effort"]),
            "--threads", str(config["threads"]), "--patches", "auto"]


def parse_cjxl_stats(stderr, width, height, threads):
    text = stderr.replace("\r\n", "\n")
    matches = list(FINAL_STATS.finditer(text))
    # Reject malformed/extra final-stat lines rather than selecting a convenient one.
    candidates = [line for line in text.splitlines() if re.match(r"^\d+ x \d+,", line)]
    require(len(matches) == len(candidates) == 1, "Missing or ambiguous cjxl final statistics")
    match = matches[0]
    require(tuple(map(int, (match[1], match[2], match[4]))) == (width, height, threads),
            "cjxl dimensions/threads mismatch")
    rate = float(match[3])
    require(positive(rate), "Zero/nonfinite cjxl native rate")
    mpix = width * height / 1e6
    # %.3f rounds the rate, not the duration. Endpoints are conservative closed bounds.
    low_rate, high_rate = rate - 0.0005, rate + 0.0005
    require(low_rate > 0, "cjxl rate quantization has no finite time upper bound")
    return dict(seconds=mpix / rate, mpix_per_second=rate,
                native_seconds_lower=mpix / high_rate, native_seconds_upper=mpix / low_rate,
                rate_lower=low_rate, rate_upper=high_rate,
                timing_quantization="MP/s rounded to nearest 0.001; bounds are not statistical CI",
                final_stats_line=match[0])


def validate_native(m, variant, size, case, threads, wall_seconds):
    require(isinstance(m, dict), "Native receipt must be an object")
    require(type(m.get("bytes")) is int and m["bytes"] == size and size > 0,
            "Native bytes differ from actual output file")
    require(all(type(m.get(k)) is int and m[k] == case[k] for k in ("width", "height")),
            "Native dimensions mismatch")
    require(positive(m.get("seconds")), "Invalid native seconds")
    require(positive(wall_seconds) and m["seconds"] <= wall_seconds + max(.001, .01 * wall_seconds),
            "Native time exceeds process wall time")
    pixels = case["width"] * case["height"]
    for key, expected in (("bpp", size * 8 / pixels),
                          ("mpix_per_second", pixels / 1e6 / m["seconds"])):
        require(positive(m.get(key)) and math.isclose(m[key], expected, rel_tol=1e-5),
                f"Invalid native {key}")
    require(m.get("mode") == variant["mode"] and m.get("search") is False
            and type(m.get("native_encode_calls")) is int and m["native_encode_calls"] == 1
            and m.get("timing") == NATIVE_TIMING, "Not the requested single direct native encode")
    for key, expected in (("effort", variant["effort"]), ("threads", threads), ("previous_channels", 0)):
        require(type(m.get(key)) is int and m[key] == expected, f"Native {key} mismatch")
    require(m.get("patches") == "Auto", "Whole-pipeline comparison requires patches Auto")
    require(all(key in m for key in ("fast_context", "seed", "model_costs")),
            "Missing nullable native receipts")
    mode, fast, seed = variant["mode"], m["fast_context"], m["seed"]
    if mode in FAST_MODES:
        full = mode in ("full14-static", "full14-static-offset")
        expected_fast = dict(mode=mode if mode in ("activity25", "activity81") else "auto12",
                             bias="static" if full else "off", predictors="full" if full else "basic",
                             scan="auto", cross_channel=False)
        require(isinstance(fast, dict) and all(fast.get(k) == v for k, v in expected_fast.items())
                and fast.get("cross_channel") is False,
                "Fast-context experiment identity mismatch")
        require(all(type(fast.get(k)) is int and fast[k] >= 0 for k in FAST_COUNTERS),
                "Invalid fast-context work counters")
        require(fast["leaves"] >= fast["trees"] and fast["cross_channel_plans"] == 0,
                "Invalid fast-context execution receipt")
        require(seed is None, "Fast mode unexpectedly enabled seed refinement")
    elif mode in SEED_MODES:
        require(fast is None and isinstance(seed, dict), "Seed experiment receipt mismatch")
        topology, method = mode.split("-")
        options = m.get("seed_options")
        require(isinstance(options, dict) and options.get("topology") == topology
                and options.get("mode") == method and options.get("depth") == 3,
                "Seed experiment identity mismatch")
        require(all(type(seed.get(k)) is int and seed[k] >= 0 for k in SEED_COUNTERS),
                "Invalid seed work counters")
    else:
        require(fast is None and seed is None, "Unexpected fast/seed experiment activation")
    if mode not in SEED_MODES:
        require(m.get("seed_options") is None, "Unexpected seed settings")
    offsets, lookahead = m.get("offset_options"), m.get("lookahead_options")
    if mode in OFFSET_MODES:
        require(isinstance(offsets, dict) and offsets.get("mode") == "distribution"
                and type(offsets.get("samples_per_leaf")) is int and offsets["samples_per_leaf"] > 0,
                "Offset-distribution experiment identity mismatch")
        require(isinstance(m.get("refinement"), dict)
                and m["refinement"].get("offset_mode") == "distribution", "Missing offset work receipt")
    else:
        require(offsets is None and m.get("refinement") is None, "Unexpected offset/refinement activation")
    if mode in ("lookahead2", "lookahead4"):
        require(isinstance(lookahead, dict) and set(lookahead) == {"width", "min_records", "relative_window"}
                and type(lookahead["width"]) is int and lookahead["width"] == int(mode[-1])
                and type(lookahead["min_records"]) is int and lookahead["min_records"] == 8192
                and positive(lookahead["relative_window"])
                and math.isclose(lookahead["relative_window"], .01, rel_tol=1e-6)
                and isinstance(m.get("lookahead"), dict), "Lookahead experiment identity mismatch")
        require(m.get("lookahead_score_contract") == "estimated local scoring only; not emitted wire bits",
                "Invalid lookahead score contract")
    else:
        require(lookahead is None and m.get("lookahead") is None, "Unexpected lookahead activation")
    signature = m.get("options_signature")
    require(isinstance(signature, dict) and signature.get("schema") == "static_context_lab.options.v1",
            "Missing native options signature")
    for key in ("mode", "effort", "threads", "patches", "previous_channels"):
        require(signature.get(key) == m[key], f"Native signature mismatch: {key}")
    require(all(key in signature for key in ("fast_context", "seed", "offsets", "lookahead")),
            "Missing experiment signature settings")
    require(signature["offsets"] == offsets and signature["lookahead"] == lookahead
            and signature["seed"] == m.get("seed_options"), "Experiment options signature mismatch")
    if fast is not None:
        require(signature["fast_context"] == {k: fast[k] for k in expected_fast},
                "Fast-context signature mismatch")
    else:
        require(signature["fast_context"] is None, "Unexpected fast-context signature")
    patch_options = m.get("patch_options")
    require(isinstance(patch_options, dict) and all(type(patch_options.get(k)) is bool
            for k in ("legacy", "source", "source_reference", "tiles"))
            and signature.get("patch_options") == patch_options, "Invalid patch options signature")
    for name, receipt in (("fast_context", fast), ("seed", seed)):
        if receipt is not None:
            for key, value in receipt.items():
                if type(value) in (int, float):
                    require(type(value) is int and value >= 0, f"Invalid {name} counter {key}")
    costs = m["model_costs"]
    if costs is not None:
        require(m.get("model_cost_contract") == MODEL_COST_CONTRACT and isinstance(costs, dict),
                "Invalid emitted model cost contract")
        require(all(type(costs.get(k)) is int and costs[k] >= 0
                    for k in ("tree_bits", "entropy_header_bits", "total_bits")), "Invalid model costs")
        require(costs["total_bits"] == costs["tree_bits"] + costs["entropy_header_bits"]
                and costs["total_bits"] <= size * 8, "Invalid model cost sum")


def hashed(path):
    path = path.resolve(strict=True)
    require(path.is_file(), f"Not a file: {path}")
    return dict(path=str(path), sha256=port.file_sha256(path))


def helper_sources():
    """Freeze the transitive local Python imports actually used by the harness."""
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                local = path.with_name(name.split(".")[0] + ".py")
                if local.is_file() and local not in seen:
                    pending.append(local)
    return sorted(seen)


def freeze_identity(args):
    manifest = hashed(args.manifest)
    spec = loads(Path(manifest["path"]).read_text(encoding="utf-8-sig"))
    cases = spec.get("cases") if isinstance(spec, dict) else None
    require(isinstance(cases, list) and 0 < len(cases) <= args.max_cases,
            "Manifest cases missing or exceed --max-cases; no silent truncation")
    inputs, seen = [], set()
    for entry in cases:
        require(isinstance(entry, dict) and isinstance(entry.get("original"), str)
                and entry["original"], "Manifest case requires original PNG path")
        path = Path(entry["original"])
        if not path.is_absolute():
            path = Path(manifest["path"]).parent / path
        info = hashed(path)
        require(info["path"] not in seen, "Duplicate manifest original")
        seen.add(info["path"])
        original = Path(info["path"]).read_bytes()
        require(port.sha256(original) == info["sha256"], "Input changed during admission")
        _, framing = png_framing(original)
        info.update({k: framing[k] for k in ("width", "height")})
        if "sha256" in entry:
            require(entry["sha256"] == info["sha256"], "Manifest input hash mismatch")
        for key in ("width", "height"):
            if key in entry:
                require(type(entry[key]) is int and entry[key] == info[key], "Manifest dimensions mismatch")
        info["stratum"] = entry.get("stratum")
        inputs.append(info)
    root = Path(__file__).resolve().parent.parent
    sources = set(helper_sources())
    sources.add(Path(__file__).resolve().with_name("plot_context_corpus.py"))
    for folder in (root / "crates", root / "vendor" / "jxl-rs"):
        if folder.exists():
            sources.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".rs", ".toml"))
    sources.update(p for p in (root / "Cargo.toml", root / "Cargo.lock") if p.exists())
    if args.cjxl_source is not None:
        sources.update(args.cjxl_source / "tools" / name
                       for name in ("cjxl_main.cc", "speed_stats.cc", "cmdline.cc"))
    identity = dict(schema=SCHEMA, manifest=manifest, inputs=inputs,
                    binaries={name: hashed(getattr(args, name)) for name in ("encoder", "cjxl", "decoder")},
                    sources=[hashed(p) for p in sorted(sources)],
                    config=dict(threads=args.threads, efforts=list(args.efforts),
                                experiment_modes=list(args.experiment_modes), repeats=args.repeats,
                                max_cases=args.max_cases, timeout=args.timeout, patches="auto",
                                backend="direct native; no outer portfolio", serial=True),
                    cjxl_timing=CJXL_TIMING)
    require(port.file_sha256(Path(manifest["path"])) == manifest["sha256"],
            "Manifest changed during admission")
    return identity


def check_frozen(identity):
    for item in [identity["manifest"], *identity["inputs"],
                 *identity["binaries"].values(), *identity["sources"]]:
        require(port.file_sha256(Path(item["path"])) == item["sha256"],
                f"Frozen input/binary/source changed: {item['path']}")


def probe_official(identity, env):
    binary = identity["binaries"]["cjxl"]["path"]
    timeout = identity["config"]["timeout"]
    version = port.run_process([binary, "--version"], env, timeout)
    port.check_process(version)
    require(re.search(r"\bcjxl v?0\.12\.[0-9]+\b", version["stdout"] + version["stderr"]),
            "Official cjxl must report version 0.12.x")
    help_record = port.run_process([binary, "--help", "-v", "-v", "-v"], env, timeout)
    port.check_process(help_record)
    help_text = help_record["stdout"] + help_record["stderr"]
    for flag in ("--num_reps", "--keep_invisible", "--lossless_jpeg", "--num_threads", "--effort"):
        require(flag in help_text, f"Official cjxl CLI missing {flag}")
    return dict(version=version, help=help_record)


def row_key(row):
    return row["case_index"], row["variant_id"], row["repetition"]


def seal_row(row):
    row["row_sha256"] = fingerprint({k: v for k, v in row.items() if k != "row_sha256"})
    return row


def validate_row(row, identity, digest):
    require(isinstance(row, dict) and row.get("fingerprint") == digest, "Checkpoint identity mismatch")
    require(row.get("row_sha256") == fingerprint({k: v for k, v in row.items() if k != "row_sha256"}),
            "Checkpoint row checksum mismatch")
    index, point_id, repetition = row_key(row)
    require(type(index) is int and 0 <= index < len(identity["inputs"])
            and type(repetition) is int and 0 <= repetition < identity["config"]["repeats"],
            "Checkpoint trial index out of range")
    point = next((p for p in variants(identity["config"]) if p["id"] == point_id), None)
    require(point is not None, "Unknown checkpoint variant")
    case = identity["inputs"][index]
    require(row.get("source_sha256") == case["sha256"], "Checkpoint source mismatch")
    require(type(row.get("verified")) is bool and row.get("status") in ("verified", "failed"),
            "Invalid checkpoint status")
    require(row["verified"] == (row["status"] == "verified"), "Inconsistent verification status")
    if not row["verified"]:
        return
    require(type(row.get("bytes")) is int and row["bytes"] > 0, "Invalid checkpoint byte size")
    require(positive(row.get("wall_seconds")), "Invalid checkpoint wall time")
    require(all(isinstance(row.get(k), dict) and row[k].get("returncode") == 0
                for k in ("encoder_process", "decoder_process")), "Unsuccessful verification processes")
    process = row["encoder_process"]
    require(process.get("wall_seconds") == row["wall_seconds"], "Encoder wall-time receipt mismatch")
    argv = process.get("command")
    require(isinstance(argv, list) and len(argv) >= (3 if point["family"] == "cjxl" else 4),
            "Missing encoder invocation receipt")
    output = argv[2] if point["family"] == "cjxl" else argv[3]
    require(argv == command(identity, point, argv[1], output), "Encoder invocation does not match frozen trial")
    evidence = row.get("verification")
    require(isinstance(evidence, dict) and evidence.get("original_samples_equal") is True
            and evidence.get("includes_hidden_rgb") is True and evidence.get("depth") in (8, 16)
            and all(isinstance(evidence.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", evidence[k])
                    for k in ("jxl_sha256", "decoded_png_sha256", "snapshot_sha256")),
            "Missing strict independent pixel verification evidence")
    decoder_argv = row["decoder_process"].get("command")
    require(isinstance(decoder_argv, list) and len(decoder_argv) >= 3,
            "Missing independent decoder invocation")
    require(decoder_argv == port.decoder_command(identity["binaries"]["decoder"]["path"], output,
                                                 decoder_argv[2], evidence["depth"]),
            "Decoder invocation does not match the frozen oracle/output")
    if point["family"] == "cjxl":
        expected = parse_cjxl_stats(row["encoder_process"]["stderr"], case["width"], case["height"],
                                    identity["config"]["threads"])
        require(row.get("measurement") == expected, "Checkpoint cjxl timing mismatch")
        require(expected["native_seconds_lower"] <= row["wall_seconds"] + .001,
                "cjxl native time exceeds process wall time")
    else:
        validate_native(row.get("measurement"), point, row["bytes"], case,
                        identity["config"]["threads"], row["wall_seconds"])


def read_journal(path, identity, digest, repair=False):
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        require(repair, "Torn final checkpoint line; use --resume for recovery")
        end = data.rfind(b"\n") + 1
        # Preserve the incomplete write as evidence, then discard only that final fragment.
        backup = path.with_name(f"{path.name}.torn-{now().replace(':', '-')}.bin")
        backup.write_bytes(data[end:])
        with path.open("r+b") as stream:
            stream.truncate(end)
            stream.flush()
            os.fsync(stream.fileno())
        data = data[:end]
    rows = []
    for line in data.splitlines():
        row = loads(line)
        validate_row(row, identity, digest)
        rows.append(row)
    return rows


def latest_rows(rows):
    latest = {}
    for row in rows:
        key = row_key(row)
        require(key not in latest or not latest[key]["verified"],
                "Checkpoint attempts to overwrite an already verified trial")
        latest[key] = row
    return latest


def per_image(identity, rows):
    latest = latest_rows(rows)
    result = {}
    for point in variants(identity["config"]):
        values = []
        for index, case in enumerate(identity["inputs"]):
            samples = [latest.get((index, point["id"], r)) for r in range(identity["config"]["repeats"])]
            require(all(s and s["verified"] for s in samples), "Incomplete case/variant/repetition matrix")
            size = statistics.median(s["bytes"] for s in samples)
            seconds = statistics.median(s["measurement"]["seconds"] for s in samples)
            pixels = case["width"] * case["height"]
            lower = statistics.median(s["measurement"].get("native_seconds_lower", s["measurement"]["seconds"])
                                      for s in samples)
            upper = statistics.median(s["measurement"].get("native_seconds_upper", s["measurement"]["seconds"])
                                      for s in samples)
            values.append(dict(case_index=index, bytes=size, pixels=pixels, bpp=size * 8 / pixels,
                               seconds=seconds, native_seconds_lower=lower, native_seconds_upper=upper))
        result[point["id"]] = values
    return result


def aggregate(images):
    result = {}
    for name, values in images.items():
        size = sum(v["bytes"] for v in values)
        pixels = sum(v["pixels"] for v in values)
        seconds = math.fsum(v["seconds"] for v in values)
        result[name] = dict(n=len(values), mean_bpp=statistics.mean(v["bpp"] for v in values),
                            pixel_weighted_bpp=size * 8 / pixels, summed_bytes=size,
                            summed_pixels=pixels, summed_mpix=pixels / 1e6,
                            native_seconds=seconds, mpix_per_second=pixels / 1e6 / seconds,
                            native_seconds_lower=math.fsum(v["native_seconds_lower"] for v in values),
                            native_seconds_upper=math.fsum(v["native_seconds_upper"] for v in values))
    return result


def make_report(identity, digest, rows, errors=(), integrity=True, probes=None, environment=None):
    latest = latest_rows(rows)
    points = variants(identity["config"])
    expected = len(identity["inputs"]) * len(points) * identity["config"]["repeats"]
    verified = sum(r["verified"] for r in latest.values())
    complete = verified == expected and integrity and not errors
    report = dict(schema=SCHEMA, status="complete" if complete else "incomplete", updated_at=now(),
                  fingerprint=digest, identity=identity, variants=points, trials=list(latest.values()),
                  expected_trials=expected, verified_trials=verified, attempts=len(rows),
                  failed_trials=sum(not r["verified"] for r in latest.values()),
                  missing_trials=expected - len(latest), errors=list(errors), integrity_verified=integrity,
                  official_probes=probes, environment=environment,
                  y_contract="arithmetic mean of per-image bpp; repeats reduced by per-image median",
                  x_contract="sum of per-image median native encode seconds; never process wall time",
                  correctness="official djxl vs original samples, including hidden RGB; 16-bit preserved",
                  comparison_scope="ordinary whole-pipeline patches Auto; differs from earlier patches-Off MA isolation",
                  timeout_contract="process guard only, no encoder selection or fallback",
                  case_statuses=[dict(case_index=i, status="verified" if all(
                      latest.get((i, p["id"], r), {}).get("verified", False)
                      for p in points for r in range(identity["config"]["repeats"])) else "incomplete")
                      for i in range(len(identity["inputs"]))])
    if complete:
        report["aggregate"] = aggregate(per_image(identity, rows))
    return report


def run_trial(identity, digest, index, point, repetition, snapshot, expected, temporary, env):
    case = identity["inputs"][index]
    row = dict(fingerprint=digest, case_index=index, variant_id=point["id"], repetition=repetition,
               source_sha256=case["sha256"], status="failed", verified=False, recorded_at=now())
    try:
        with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
            jxl, png = Path(trial) / "output.jxl", Path(trial) / "oracle.png"
            process = port.run_process(command(identity, point, snapshot, jxl), env, identity["config"]["timeout"])
            row["encoder_process"] = process
            row["wall_seconds"] = process["wall_seconds"]
            port.check_process(process)
            size = jxl.stat().st_size
            require(size > 0, "Empty native output")
            row["bytes"] = size
            decoder = port.run_process(port.decoder_command(identity["binaries"]["decoder"]["path"],
                                       jxl, png, expected.depth), env, identity["config"]["timeout"])
            row["decoder_process"] = decoder
            port.check_process(decoder)
            port.compare_pixels(expected, port.png_samples(png.read_bytes()))
            row["verification"] = dict(original_samples_equal=True, includes_hidden_rgb=True,
                                       depth=expected.depth, jxl_sha256=port.file_sha256(jxl),
                                       decoded_png_sha256=port.file_sha256(png),
                                       snapshot_sha256=port.file_sha256(snapshot))
            if point["family"] == "cjxl":
                measurement = parse_cjxl_stats(process["stderr"], expected.width, expected.height,
                                               identity["config"]["threads"])
            else:
                measurement = loads(process["stdout"])
                validate_native(measurement, point, size, case, identity["config"]["threads"], process["wall_seconds"])
            row["measurement"] = measurement
            row.update(verified=True, status="verified")
            seal_row(row)
            validate_row(row, identity, digest)
    except Exception as error:
        row.update(verified=False, status="failed", error=f"{type(error).__name__}: {error}")
    return seal_row(row)


@contextmanager
def run_lock(outdir):
    """OS-owned lock survives the file but not a killed process, allowing crash recovery."""
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / ".run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def benchmark(args):
    with run_lock(args.outdir):
        return _benchmark_locked(args)


def _benchmark_locked(args):
    identity = freeze_identity(args)
    digest = fingerprint(identity)
    args.outdir.mkdir(parents=True, exist_ok=True)
    identity_path = args.outdir / "identity.json"
    journal_path = args.outdir / "trials.jsonl"
    if args.resume:
        previous = loads(identity_path.read_text(encoding="utf-8"))
        require(previous == dict(fingerprint=digest, identity=identity),
                "Resume configuration/binary/input/source identity changed; use a new outdir")
    else:
        require(not any(p.name != ".run.lock" for p in args.outdir.iterdir()),
                "Output directory must be empty unless --resume")
        atomic_json(identity_path, dict(fingerprint=digest, identity=identity))
    rows = read_journal(journal_path, identity, digest, repair=args.resume)
    latest = latest_rows(rows)
    env, environment = port.clean_environment(os.environ)
    errors, probes, integrity = [], None, True

    def progress(status="running"):
        current = latest_rows(rows)
        atomic_json(args.outdir / "progress.json", dict(
            fingerprint=digest, updated_at=now(), status=status, attempts=len(rows),
            expected_trials=len(identity["inputs"]) * len(variants(identity["config"])) * args.repeats,
            verified_trials=sum(r["verified"] for r in current.values()),
            failed_trials=sum(not r["verified"] for r in current.values())))

    try:
        port.pixel_libraries()
        probes = probe_official(identity, env)
        check_frozen(identity)
        progress()
        with journal_path.open("a", encoding="utf-8", newline="\n") as journal:
            for index, case in enumerate(identity["inputs"]):
                points = variants(identity["config"])
                if all(latest.get((index, p["id"], r), {}).get("verified", False)
                       for p in points for r in range(args.repeats)):
                    continue
                with tempfile.TemporaryDirectory(prefix="context-corpus-", dir=args.outdir) as folder:
                    temporary = Path(folder)
                    preparation_error = None
                    try:
                        source = Path(case["path"])
                        original = source.read_bytes()
                        require(port.sha256(original) == case["sha256"], "Frozen input changed")
                        payload, expected, preparation = prepare_png(source, original, temporary)
                        require((expected.width, expected.height) == (case["width"], case["height"]),
                                "Prepared dimensions mismatch")
                        snapshot = temporary / "input.png"
                        snapshot.write_bytes(payload)
                    except Exception as error:
                        preparation_error = f"{type(error).__name__}: {error}"
                    for repetition in range(args.repeats):
                        for point in trial_order(points, index, repetition):
                            key = index, point["id"], repetition
                            if latest.get(key, {}).get("verified", False):
                                continue
                            if preparation_error is None:
                                row = run_trial(identity, digest, index, point, repetition,
                                                snapshot, expected, temporary, env)
                                row["preparation"] = preparation
                                seal_row(row)
                            else:
                                row = seal_row(dict(fingerprint=digest, case_index=index,
                                    variant_id=point["id"], repetition=repetition,
                                    source_sha256=case["sha256"], verified=False, status="failed",
                                    recorded_at=now(), error=preparation_error))
                            journal.write(serialized(row) + "\n")
                            journal.flush()
                            os.fsync(journal.fileno())
                            rows.append(row)
                            latest[key] = row
                            progress()
                            print(f"{index + 1}/{len(identity['inputs'])} {point['id']} "
                                  f"repeat={repetition}: {row['status']}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        try:
            check_frozen(identity)
        except (OSError, ValueError) as error:
            integrity = False
            errors.append(f"{type(error).__name__}: {error}")
        report = make_report(identity, digest, rows, errors, integrity, probes, environment)
        atomic_json(args.outdir / "summary.json", report)
        progress(report["status"])
    return report


def main(argv=None):
    try:
        args = arguments(argv)
        report = benchmark(args)
        print(f"{report['status']}: {report['verified_trials']}/{report['expected_trials']} verified; "
              f"{args.outdir / 'summary.json'}")
        return 0 if report["status"] == "complete" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Context corpus failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
