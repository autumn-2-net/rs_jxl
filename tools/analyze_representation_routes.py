"""Offline input-feature hypothesis study; never builds or invokes a codec.

Run `extract` only outside benchmark timing windows. `analyze` reads cached
features and verified experiment measurements, fits a calibration-only byte-cost
stump and evaluates holdout once. This is not a production classifier/C61 port.
"""

import argparse
from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
import random
import statistics
import struct
from time import perf_counter

from prepare_representation_corpus import (checked_original, digest_file,
                                            original_path, provenance, read_json)
from corpus_inputs import png_framing


FEATURES = (
    "horizontal_equality", "vertical_equality", "horizontal_change", "vertical_change",
    "flat_block_rate", "left_residual_mean_abs", "top_residual_mean_abs",
    "gradient_residual_mean_abs", "left_residual_cardinality",
    "top_residual_cardinality", "gradient_residual_cardinality", "color_count_proxy",
    "horizontal_contrast", "vertical_contrast", "directional_contrast",
    "left_residual_tuple_cardinality", "top_residual_tuple_cardinality",
    "gradient_residual_tuple_cardinality", "left_residual_zero_tuple_fraction",
    "top_residual_zero_tuple_fraction", "gradient_residual_zero_tuple_fraction",
)


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write("\n")


def windows(width, height, seed=20261113, size=64, grid=3):
    """One unresized window per spatial stratum; duplicate rectangles removed."""
    if min(width, height, size, grid) < 1:
        raise ValueError("Positive window parameters required")
    rng, result = random.Random(seed), []
    w, h = min(size, width), min(size, height)
    for gy in range(grid):
        for gx in range(grid):
            x0, x1 = (width - w) * gx // grid, (width - w) * (gx + 1) // grid
            y0, y1 = (height - h) * gy // grid, (height - h) * (gy + 1) // grid
            rectangle = (rng.randint(x0, x1), rng.randint(y0, y1), w, h)
            if rectangle not in result:
                result.append(rectangle)
    return result


def decode_samples(data):
    """Canonical straight RGBA integers, no ICC/orientation or alpha compositing.

    Packed gray/palette expands to eight-bit samples. Pillow cannot preserve
    multichannel 16-bit PNG: use the existing optional OpenCV oracle for that
    case, or fail explicitly rather than silently narrowing to eight bits.
    """
    import numpy as np
    from PIL import PngImagePlugin

    payload, info = png_framing(data)
    depth = info["bit_depth"]
    if depth == 16 and info["color_type"] != 0:
        from compare_cpp_port import png_samples
        try:
            pixels = png_samples(payload)
        except RuntimeError as error:
            raise ValueError("16-bit multichannel extraction needs optional opencv-python; no narrowing permitted") from error
        rgba, sample_depth = pixels.rgba, pixels.depth
    else:
        with PngImagePlugin.PngImageFile(BytesIO(payload)) as image:
            if depth == 16:
                gray = np.asarray(image)
                if gray.dtype.kind not in "ui" or gray.max() > 65535 or gray.min() < 0:
                    raise ValueError("Invalid 16-bit gray decode")
                rgba = np.empty((*gray.shape, 4), dtype=np.uint16)
                rgba[:, :, :3] = gray[:, :, None]
                key = info.get("transparency_key")
                rgba[:, :, 3] = np.where(gray == key[0], 0, 65535) if key else 65535
                sample_depth = 16
            else:
                if depth < 8 and info["color_type"] == 0:
                    image = image.convert("L")
                    key = info.get("transparency_key")
                    if key:
                        image.info["transparency"] = key[0] * 255 // ((1 << depth) - 1)
                rgba, sample_depth = np.asarray(image.convert("RGBA")), 8
    if rgba.shape != (info["height"], info["width"], 4):
        raise ValueError("Decoded dimensions/channels changed")
    offset, has_trns = 8, False
    while offset < len(payload):
        length, kind = struct.unpack_from(">I4s", payload, offset)
        has_trns |= kind == b"tRNS"
        offset += length + 12
    return rgba, dict(width=info["width"], height=info["height"],
                      source_bit_depth=depth, source_color_type=info["color_type"],
                      sample_depth=sample_depth,
                      alpha_present=info["color_type"] in (4, 6) or has_trns,
                      sample_layout="straight RGBA; synthesized opaque alpha if absent; gray replicated",
                      hidden_rgb="retained, including zero-alpha pixels")


def content_features(sample_windows, sample_depth):
    import numpy as np

    if sample_depth not in (8, 16) or not sample_windows:
        raise ValueError("Eight/sixteen-bit nonempty integer windows required")
    maximum = (1 << sample_depth) - 1
    horizontal, vertical, gradient, colors = [], [], [], []
    heq = veq = hn = vn = flat = blocks = 0
    for raw in sample_windows:
        a = np.asarray(raw)
        if a.ndim != 3 or a.shape[2] != 4 or a.dtype.kind not in "ui" or a.size == 0:
            raise ValueError("Expected nonempty integer RGBA windows")
        if a.min() < 0 or a.max() > maximum:
            raise ValueError("Samples exceed declared precision")
        a = a.astype(np.int64)
        colors.append(a.reshape(-1, 4))
        dx, dy = a[:, 1:] - a[:, :-1], a[1:] - a[:-1]
        horizontal.append(dx.reshape(-1, 4))
        vertical.append(dy.reshape(-1, 4))
        hn += dx.shape[0] * dx.shape[1]
        vn += dy.shape[0] * dy.shape[1]
        heq += int(np.all(dx == 0, axis=2).sum())
        veq += int(np.all(dy == 0, axis=2).sum())
        left, top = a[1:, :-1], a[:-1, 1:]
        prediction = np.clip(left + top - a[:-1, :-1],
                             np.minimum(left, top), np.maximum(left, top))
        gradient.append((a[1:, 1:] - prediction).reshape(-1, 4))
        bh, bw = a.shape[0] // 4, a.shape[1] // 4
        if bh and bw:
            tiles = a[:bh * 4, :bw * 4].reshape(bh, 4, bw, 4, 4).transpose(0, 2, 1, 3, 4)
            flat += int(np.all(tiles == tiles[:, :, :1, :1], axis=(2, 3, 4)).sum())
            blocks += bh * bw

    def residual(values):
        values = np.concatenate(values)
        if not values.size:
            return 0.0, 0.0, 0.0, 0.0
        return (float(np.abs(values).mean() / maximum),
                float(np.unique(values).size / values.size),
                float(np.unique(values, axis=0).shape[0] / len(values)),
                float(np.all(values == 0, axis=1).mean()))

    left, lc, ltc, lzf = residual(horizontal)
    top, tc, ttc, tzf = residual(vertical)
    grad, gc, gtc, gzf = residual(gradient)
    colors = np.concatenate(colors)
    h, v = heq / hn if hn else 1.0, veq / vn if vn else 1.0
    return dict(horizontal_equality=h, vertical_equality=v,
                horizontal_change=1 - h, vertical_change=1 - v,
                flat_block_rate=flat / blocks if blocks else 0.0,
                left_residual_mean_abs=left, top_residual_mean_abs=top,
                gradient_residual_mean_abs=grad, left_residual_cardinality=lc,
                top_residual_cardinality=tc, gradient_residual_cardinality=gc,
                left_residual_tuple_cardinality=ltc, top_residual_tuple_cardinality=ttc,
                gradient_residual_tuple_cardinality=gtc,
                left_residual_zero_tuple_fraction=lzf, top_residual_zero_tuple_fraction=tzf,
                gradient_residual_zero_tuple_fraction=gzf,
                color_count_proxy=float(np.unique(colors, axis=0).shape[0] / len(colors)),
                horizontal_contrast=left, vertical_contrast=top,
                directional_contrast=(left - top) / (left + top) if left + top else 0.0)


def extract_case(case, manifest, seed, size, grid):
    started = perf_counter()
    checked = checked_original(case, manifest)
    pixels, semantics = decode_samples(Path(checked["original"]).read_bytes())
    decoded = perf_counter()
    rectangles = windows(semantics["width"], semantics["height"], seed, size, grid)
    slices = [pixels[y:y + h, x:x + w] for x, y, w, h in rectangles]
    features = content_features(slices, semantics["sample_depth"])
    ended = perf_counter()
    return dict(sha256=checked["sha256"], original=checked["original"],
                semantics=semantics, windows=rectangles, features=features,
                timing=dict(read_validate_decode_seconds=decoded - started,
                            feature_seconds=ended - decoded, total_seconds=ended - started))


def extract(manifest, seed=20261113, size=64, grid=3):
    # No report argument: extraction cannot see labels, output sizes or timings.
    cases = [extract_case(case, manifest, seed, size, grid) for case in read_json(manifest)["cases"]]
    return dict(schema="representation-features-v1", manifest=provenance(manifest),
                configuration=dict(seed=seed, size=size, grid=grid, resizing=False,
                                   residual="left/top and JXL Gradient: clamp(left+top-top_left, min(left,top), max(left,top)); all straight RGBA channels",
                                   cardinality="unique scalar residuals / scalar samples; exact on sampled windows",
                                   tuple_cardinality="unique signed RGBA residual tuples / residual pixels; no cross-window edges; zero if no residual pixels",
                                   zero_tuple_fraction="fraction of residual pixels with all four signed RGBA residuals zero, including alpha; zero if no residual pixels",
                                   flat_blocks="4x4, within each window only",
                                   color_proxy="unique RGBA tuples / sampled pixels; overlaps count again"),
                feature_names=list(FEATURES), cases=cases)


def finite_number(value, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Expected finite" + (" positive" if positive else "") + " number")
    return float(value)


def validate_study(study):
    cases = study["cases"]
    if not cases:
        raise ValueError("Empty study")
    hashes, paths, groups = set(), set(), {}
    for case in cases:
        sha, path, split = case["sha256"], str(Path(case["original"]).resolve()), case["split"]
        if split not in ("calibration", "holdout") or sha in hashes or path in paths:
            raise ValueError("Invalid split or duplicate input")
        group = case["source_group"]
        if group in groups and groups[group] != split:
            raise ValueError("Source group crosses calibration/holdout")
        hashes.add(sha)
        paths.add(path)
        groups[group] = split
    if {c["split"] for c in cases} != {"calibration", "holdout"}:
        raise ValueError("Both fixed splits required")
    return cases


def load_feature_rows(document, cases):
    if document.get("schema") != "representation-features-v1" or document.get("feature_names") != list(FEATURES):
        raise ValueError("Unknown or contaminated feature schema")
    rows = {}
    for row in document["cases"]:
        sha = row["sha256"]
        if sha in rows or set(row["features"]) != set(FEATURES):
            raise ValueError("Duplicate or non-whitelisted feature fields")
        for value in row["features"].values():
            finite_number(value)
        for key in ("read_validate_decode_seconds", "feature_seconds", "total_seconds"):
            if finite_number(row["timing"][key]) < 0:
                raise ValueError("Negative feature overhead")
        if row["timing"]["total_seconds"] + 1e-9 < (row["timing"]["feature_seconds"] + row["timing"]["read_validate_decode_seconds"]):
            raise ValueError("Inconsistent feature timing")
        rows[sha] = row
    for case in cases:
        if case["sha256"] not in rows:
            raise ValueError("Missing input features; partial holdout evaluation is forbidden")
        semantics = rows[case["sha256"]]["semantics"]
        for key in ("width", "height"):
            if semantics[key] != case[key]:
                raise ValueError("Feature dimension mismatch")
        if (semantics["source_bit_depth"] != case["bit_depth"] or
                semantics["source_color_type"] != case["color_type"] or
                semantics["sample_depth"] != (16 if case["bit_depth"] == 16 else 8)):
            raise ValueError("Feature sample/depth semantics mismatch")
    return rows


def at_path(value, path):
    for key in path:
        value = value[key]
    return value


def normalize_samples(value):
    """Explicit path must point to a benchmark variant sample array."""
    if not isinstance(value, list):
        raise ValueError("Route path must point to a sample array")
    samples = [r for r in value if not r.get("warmup", False)]
    if not samples or any(r.get("verified") is not True for r in samples):
        raise ValueError("Route needs non-warmup independently verified samples")
    sizes, seconds = [], []
    for sample in samples:
        m = sample["measurement"]
        if type(m["bytes"]) is not int or m["bytes"] <= 0:
            raise ValueError("Output size must be positive integer bytes")
        sizes.append(m["bytes"])
        seconds.append(finite_number(m["seconds"], positive=True))
    if len(set(sizes)) != 1:
        raise ValueError("Nondeterministic byte counts across repetitions")
    return dict(bytes=sizes[0], seconds=statistics.median(seconds), repetitions=len(samples))


def load_labels(reports, route_specs, cases):
    """Route specs: {alias: {report: integer index, path: [keys, ...]}}.

    E.g. native evidence may use ["evidence", "delta", "variants", "native"].
    No recursive result discovery: caller must name each experimental candidate.
    """
    by_sha = {c["sha256"]: c for c in cases}
    by_path = {str(Path(c["original"]).resolve()): c for c in cases}
    indexes, path_only = [], set()
    for report_index, report in enumerate(reports):
        index = {}
        for row in report["cases"]:
            sha = row.get("sha256")
            source = row.get("source", row.get("original"))
            path_case = by_path.get(str(Path(source).resolve())) if source else None
            if sha is not None and path_case and sha != path_case["sha256"]:
                raise ValueError("Report input SHA256 disagrees with manifest path")
            case = by_sha.get(sha) if sha else path_case
            if case is None:
                continue
            key = case["sha256"]
            if key in index:
                raise ValueError("Duplicate case in report")
            if not sha:
                path_only.add((report_index, key))
            for dimension in ("width", "height"):
                if dimension in row and row[dimension] != case[dimension]:
                    raise ValueError("Report dimension mismatch")
            if "pixels" in row and row["pixels"] != case["width"] * case["height"]:
                raise ValueError("Report pixel count mismatch")
            if "depth" in row and row["depth"] != (16 if case["bit_depth"] == 16 else 8):
                raise ValueError("Report decoded precision mismatch")
            index[key] = row
        indexes.append(index)
    labels = {sha: {} for sha in by_sha}
    for route, spec in route_specs.items():
        if type(spec["report"]) is not int or not 0 <= spec["report"] < len(indexes):
            raise ValueError("Invalid route report index")
        for sha, case in by_sha.items():
            try:
                samples = at_path(indexes[spec["report"]][sha], spec["path"])
                labels[sha][route] = normalize_samples(samples)
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError(f"Missing route {route} for input {sha}") from error
            for sample in samples:
                if sample.get("warmup", False):
                    continue
                for dimension in ("width", "height"):
                    m = sample["measurement"]
                    if dimension in m and m[dimension] != case[dimension]:
                        raise ValueError("Measurement dimension mismatch")
    return labels, len(path_only)


def route_order(routes, baseline):
    if baseline not in routes:
        raise ValueError("Baseline route missing")
    return [baseline] + sorted(r for r in routes if r != baseline)


def best_route(ids, labels, routes):
    return min(routes, key=lambda route: sum(labels[sha][route]["bytes"] for sha in ids))


def fit_stump(cases, features, labels, baseline):
    ids = [c["sha256"] for c in cases if c["split"] == "calibration"]
    routes = route_order(labels[ids[0]], baseline)
    constant = best_route(ids, labels, routes)
    model = dict(kind="constant", route=constant)
    best_cost = sum(labels[sha][constant]["bytes"] for sha in ids)
    for feature in FEATURES:
        values = sorted({features[sha]["features"][feature] for sha in ids})
        for low, high in zip(values, values[1:]):
            threshold = low + (high - low) / 2
            left = [sha for sha in ids if features[sha]["features"][feature] <= threshold]
            right = [sha for sha in ids if features[sha]["features"][feature] > threshold]
            if not left or not right:
                continue
            lr, rr = best_route(left, labels, routes), best_route(right, labels, routes)
            cost = sum(labels[sha][lr]["bytes"] for sha in left) + sum(labels[sha][rr]["bytes"] for sha in right)
            if cost < best_cost:
                best_cost = cost
                model = dict(kind="stump", feature=feature, threshold=threshold,
                             left_route=lr, right_route=rr, left_calibration_count=len(left),
                             right_calibration_count=len(right))
    winners = Counter(best_route([sha], labels, routes) for sha in ids)
    any_gain = any(labels[sha][best_route([sha], labels, routes)]["bytes"] < labels[sha][baseline]["bytes"] for sha in ids)
    status = ("no_candidate_byte_gains_on_calibration" if not any_gain else
              "collapsed_calibration_labels_constant_route" if len(winners) == 1 else
              "calibration_fit_only")
    return dict(**model, calibration_byte_cost=best_cost, calibration_oracle_winners=dict(winners),
                status=status, objective="sum of actual output bytes; baseline wins ties; no runtime feature")


def predict(model, features):
    if model["kind"] == "constant":
        return model["route"]
    return model["left_route"] if features[model["feature"]] <= model["threshold"] else model["right_route"]


def evaluate(cases, features, labels, model, baseline):
    routes = route_order(labels[cases[0]["sha256"]], baseline)
    started = perf_counter()
    predictions = {c["sha256"]: predict(model, features[c["sha256"]]["features"]) for c in cases}
    decision_seconds = perf_counter() - started
    rows = []
    for case in cases:
        sha = case["sha256"]
        chosen = predictions[sha]
        oracle = best_route([sha], labels, routes)
        m, b, o = labels[sha][chosen], labels[sha][baseline], labels[sha][oracle]
        rows.append(dict(sha256=sha, chosen_route=chosen, oracle_route=oracle,
                         chosen_bytes=m["bytes"], baseline_bytes=b["bytes"], oracle_bytes=o["bytes"],
                         byte_regret=m["bytes"] - o["bytes"],
                         false_skip=chosen == baseline and o["bytes"] < b["bytes"],
                         miss=m["bytes"] > o["bytes"],
                         false_enable=chosen != baseline and m["bytes"] >= b["bytes"],
                         chosen_route_seconds_estimate=m["seconds"],
                         oracle_route_seconds_estimate=o["seconds"], baseline_seconds=b["seconds"],
                         exhaustive_seconds=sum(v["seconds"] for v in labels[sha].values()),
                         feature_overhead_seconds=features[sha]["timing"]["total_seconds"]))
    total = lambda key: sum(r[key] for r in rows)
    chosen, base, oracle = total("chosen_bytes"), total("baseline_bytes"), total("oracle_bytes")
    estimated = total("chosen_route_seconds_estimate")
    overhead = total("feature_overhead_seconds")
    exhaustive = total("exhaustive_seconds")
    return dict(count=len(rows), baseline_bytes=base, chosen_bytes=chosen, oracle_bytes=oracle,
                byte_savings_vs_baseline=base - chosen,
                byte_regret=chosen - oracle, relative_byte_regret=(chosen - oracle) / oracle,
                max_case_byte_regret=max(r["byte_regret"] for r in rows),
                false_skips=total("false_skip"), misses=total("miss"), false_enables=total("false_enable"),
                chosen_route_counts=dict(Counter(predictions.values())),
                oracle_route_counts=dict(Counter(r["oracle_route"] for r in rows)),
                baseline_seconds=total("baseline_seconds"),
                exhaustive_seconds=exhaustive, oracle_single_route_seconds_estimate=total("oracle_route_seconds_estimate"),
                chosen_route_seconds_estimate=estimated,
                independently_measured_feature_overhead_seconds=overhead,
                measured_python_decision_seconds=decision_seconds,
                chosen_plus_offline_overhead_seconds_estimate=estimated + overhead + decision_seconds,
                saved_seconds_vs_exhaustive_estimate=exhaustive - estimated - overhead - decision_seconds,
                actual_chosen_route_runs=None, cases=rows)


def attach_measured_runs(evaluation, document):
    rows = {}
    for row in document["cases"]:
        if row["sha256"] in rows:
            raise ValueError("Duplicate chosen-route run")
        rows[row["sha256"]] = row
    seconds = 0.0
    for case in evaluation["cases"]:
        if case["sha256"] not in rows:
            raise ValueError("Chosen-route runs do not cover complete holdout")
        row = rows[case["sha256"]]
        if row.get("verified") is not True or row["route"] != case["chosen_route"] or row["bytes"] != case["chosen_bytes"]:
            raise ValueError("Chosen-route verification/route/bytes mismatch")
        seconds += finite_number(row["seconds"], positive=True)
    evaluation["actual_chosen_route_runs"] = dict(
        count=evaluation["count"], measured_route_seconds=seconds,
        plus_independent_offline_feature_seconds=seconds + evaluation["independently_measured_feature_overhead_seconds"],
        note="Caller-supplied separate chosen-route runs; feature overhead measured independently, not an end-to-end production timing")


def analyze(study, feature_document, reports, baseline="baseline", route_specs=None,
            measured_runs=None):
    cases = validate_study(study)
    features = load_feature_rows(feature_document, cases)
    if route_specs is None:
        if len(reports) != 1:
            raise ValueError("Multiple reports require explicit route specs")
        variants = reports[0]["cases"][0]["variants"]
        route_specs = {name: dict(report=0, path=["variants", name]) for name in variants}
    if not route_specs:
        raise ValueError("At least a baseline route is required")
    labels, path_only_count = load_labels(reports, route_specs, cases)
    model = fit_stump(cases, features, labels, baseline)
    evaluations = {split: evaluate([c for c in cases if c["split"] == split], features, labels, model, baseline)
                   for split in ("calibration", "holdout")}
    if measured_runs is not None:
        attach_measured_runs(evaluations["holdout"], measured_runs)
    return dict(schema="representation-routes-v1", model=model, evaluations=evaluations,
                routes=route_specs, path_only_report_rows=path_only_count,
                limitations=["Offline hypothesis test, not C61 or a production port/API",
                             "Feature extraction is separate from codec timing; decoding may dominate overhead",
                             "Sum of per-route median seconds estimates exhaustive/chosen cost; not measured end-to-end selected-route timing",
                             "Holdout labels never fit thresholds; repeated study revisions after viewing holdout invalidate untouched-holdout claims",
                             "Exact-file SHA and declared-family separation do not prove absence of perceptual near-duplicates",
                             "Path-only report rows assume frozen original paths; prefer reports carrying original SHA256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extraction = sub.add_parser("extract", help="Input-only offline extraction; do not overlap benchmarks")
    extraction.add_argument("--manifest", type=Path, required=True)
    extraction.add_argument("--seed", type=int, default=20261113)
    extraction.add_argument("--size", type=int, default=64)
    extraction.add_argument("--grid", type=int, default=3)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--features", type=Path, required=True)
    analysis.add_argument("--report", type=Path, action="append", required=True)
    analysis.add_argument("--baseline", default="baseline")
    analysis.add_argument("--route-spec", type=Path)
    analysis.add_argument("--chosen-runs", type=Path)
    for command in (extraction, analysis):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output exists; frozen results are never overwritten")
    if args.command == "extract":
        result = extract(args.manifest, args.seed, args.size, args.grid)
    else:
        feature_document = read_json(args.features)
        if feature_document.get("manifest", {}).get("sha256") != digest_file(args.manifest):
            parser.error("Feature cache was extracted for a different frozen manifest")
        result = analyze(read_json(args.manifest), feature_document,
                         [read_json(p) for p in args.report], args.baseline,
                         read_json(args.route_spec) if args.route_spec else None,
                         read_json(args.chosen_runs) if args.chosen_runs else None)
        inputs = dict(manifest=provenance(args.manifest), features=provenance(args.features),
                      reports=[provenance(p) for p in args.report])
        if args.route_spec:
            inputs["route_spec"] = provenance(args.route_spec)
        if args.chosen_runs:
            inputs["chosen_runs"] = provenance(args.chosen_runs)
        result["inputs"] = inputs
    write_json(args.out, result)
    print(json.dumps(dict(out=str(args.out.resolve()), schema=result["schema"])))


if __name__ == "__main__":
    main()
