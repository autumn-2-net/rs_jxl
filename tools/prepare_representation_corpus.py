"""Freeze original-only, source-group-separated representation study manifests.

No decoding, resizing, encoder execution or experiment-result inputs. Validation
checks original SHA256 and PNG framing/CRC, not decoded-pixel conformance.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from corpus_inputs import png_framing


QUOTAS = dict(document_web=40, manga=24, game_ui=24, icon_render=24,
              natural_photo=32)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(path):
    return dict(path=str(Path(path).resolve()), sha256=digest_file(path))


def original_path(row, manifest):
    path = Path(row.get("original", row.get("source", "")))
    if not str(path) or str(path) == ".":
        raise ValueError("Missing original/source path")
    return (path if path.is_absolute() else Path(manifest).parent / path).resolve()


def study_stratum(row):
    name = row.get("stratum")
    if name in ("document_web", "manga", "game_ui", "icon_render", "natural_photo"):
        return name
    if name in ("photo_highres", "photo_other", "urban"):
        return "natural_photo"
    relative = row.get("relative", "").lower().replace("\\", "/")
    if name == "alpha_icon" and ("/icon_" in relative or "kenney_particle" in relative):
        return "icon_render"
    return None


def source_group(row):
    """Conservative collections, not file hashes, decide independence units."""
    if row.get("source_group"):
        return row["source_group"]
    text = row.get("relative", "").lower().replace("\\", "/")
    parts = text.split("/")
    dataset = row.get("dataset", "unknown").lower()
    if "screenshot_web" in parts:
        return "qoi-web-screenshot-collection"  # includes mirrors
    if "screenshot_game" in parts:
        return "qoi-game-screenshot-collection"
    if any(p.startswith("icon_") for p in parts):
        return "breeze-icon-render-collection"  # both 64px and 512px
    if dataset == "manga109":
        series = re.sub(r"[_-]?vol\d+.*$", "", Path(text).stem)
        return "manga109/" + series
    if dataset == "diverse-downloads":
        # imazen collections and asset packs stay intact, including subfolders.
        return "/".join(parts[:3] if len(parts) > 2 and parts[1] == "imazen26" else parts[:2])
    if dataset.startswith("local-screenshot"):
        return "local-screenshot-collection"
    if dataset in ("div2k", "urban100", "bsds100", "set14", "set5"):
        return dataset
    if dataset == "jxl-testdata" and len(parts) > 2:
        return "/".join(parts[:3])
    return dataset + "/" + str(Path(text).parent)


def checked_original(row, manifest):
    path = original_path(row, manifest)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if row.get("sha256") and sha != row["sha256"].lower():
        raise ValueError(f"Original SHA256 mismatch: {path}")
    _, info = png_framing(data)
    for key in ("width", "height", "bit_depth", "color_type"):
        if key in row and row[key] != info[key]:
            raise ValueError(f"Original {key} mismatch: {path}")
    return dict(original=str(path), sha256=sha,
                **{key: info[key] for key in ("width", "height", "bit_depth", "color_type")})


def fair_sample(rows, count, rng):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["original"].casefold()):
        groups[row["source_group"]].append(row)
    names = sorted(groups)
    rng.shuffle(names)
    for items in groups.values():
        rng.shuffle(items)
    selected = []
    while len(selected) < count:
        changed = False
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
                changed = True
        if not changed:
            raise ValueError("Insufficient population for quota")
    return selected


def prepare(population, known, references=(), seed=20261113, quotas=None,
            diagnostic_count=24):
    quotas = dict(QUOTAS if quotas is None else quotas)
    if not quotas or any(type(n) is not int or n < 2 or n % 2 for n in quotas.values()):
        raise ValueError("Positive even quotas required for equal split")
    if type(seed) is not int or type(diagnostic_count) is not int or diagnostic_count < 0:
        raise ValueError("Invalid seed or diagnostic count")
    known_cases = []
    for row in read_json(known)["cases"]:
        checked = checked_original(row, known)
        known_cases.append(dict(checked, split="known_positive_control",
                                stratum="known_patch", source_group=source_group(row)))
    known_hashes = {r["sha256"] for r in known_cases}
    if len(known_hashes) != len(known_cases):
        raise ValueError("Known cohort has duplicate original hashes")
    rows, paths, hashes, exclusions = [], set(), set(), []
    raw_rows = read_json(population)["cases"]
    for row in raw_rows:
        path = str(original_path(row, population))
        sha = row.get("sha256", "").lower()
        if not re.fullmatch("[0-9a-f]{64}", sha):
            raise ValueError("Population must supply original SHA256, not experiment sizes")
        stratum = study_stratum(row)
        reason = ("duplicate_path" if path in paths else
                  "duplicate_sha256" if sha in hashes else
                  "known12_sha256" if sha in known_hashes else
                  "outside_declared_strata" if stratum not in quotas else None)
        paths.add(path)
        hashes.add(sha)
        if reason:
            exclusions.append(dict(original=path, sha256=sha, reason=reason))
            continue
        rows.append(dict(original=path, sha256=sha, stratum=stratum,
                         dataset=row.get("dataset"), relative=row.get("relative"),
                         source_group=source_group(row)))
    # Try whole-source assignments; feasibility depends only on declared inputs.
    names = sorted({r["source_group"] for r in rows})
    rng = random.Random(seed)
    assignment = None
    for attempt in range(4096):
        trial = {name: ("holdout" if rng.random() < 0.5 else "calibration") for name in names}
        capacities = Counter((r["stratum"], trial[r["source_group"]]) for r in rows)
        if all(capacities[s, split] >= n // 2 for s, n in quotas.items()
               for split in ("calibration", "holdout")):
            assignment = trial
            break
    if assignment is None:
        raise ValueError("No feasible whole-source split; change input-only quotas or source inventory")
    selected = []
    for stratum, count in sorted(quotas.items()):
        for split in ("calibration", "holdout"):
            pool = [r for r in rows if r["stratum"] == stratum and assignment[r["source_group"]] == split]
            for row in fair_sample(pool, count // 2, rng):
                checked = checked_original(row, population)
                selected.append(dict(row, **checked, split=split))
    rng.shuffle(selected)
    holdout = [r for r in selected if r["split"] == "holdout"]
    if diagnostic_count > len(holdout):
        raise ValueError("Diagnostic count exceeds holdout")
    # Fixed stratified round-robin, not a Patch activation/win filter.
    pools = defaultdict(list)
    for row in holdout:
        pools[row["stratum"]].append(row)
    diagnostic = []
    for items in pools.values():
        rng.shuffle(items)
    while len(diagnostic) < diagnostic_count:
        for name in sorted(pools):
            if pools[name] and len(diagnostic) < diagnostic_count:
                diagnostic.append(pools[name].pop())
    overlaps = {}
    for ref in references:
        ref_hashes = set()
        for row in read_json(ref)["cases"]:
            sha = row.get("sha256")
            ref_hashes.add(sha.lower() if sha else digest_file(original_path(row, ref)))
        overlaps[str(Path(ref).resolve())] = dict(
            **provenance(ref), reference_unique_hashes=len(ref_hashes),
            main=sum(r["sha256"] in ref_hashes for r in selected),
            calibration=sum(r["sha256"] in ref_hashes and r["split"] == "calibration" for r in selected),
            holdout=sum(r["sha256"] in ref_hashes and r["split"] == "holdout" for r in selected),
            known12=len(known_hashes & ref_hashes),
            patch_diagnostic=sum(r["sha256"] in ref_hashes for r in diagnostic))
    protocol = dict(
        schema="representation-corpus-v1", seed=seed, quotas=quotas,
        population=provenance(population), known=provenance(known),
        count=len(selected), split_counts=dict(Counter(r["split"] for r in selected)),
        population_rows=len(raw_rows), eligible_unique_rows=len(rows),
        population_counts=dict(Counter(r["stratum"] for r in rows)),
        exclusion_counts=dict(Counter(r["reason"] for r in exclusions)), exclusions=exclusions,
        source_assignment=assignment, feasible_assignment_attempt=attempt + 1,
        selected_group_counts={split: len({r["source_group"] for r in selected if r["split"] == split})
                               for split in ("calibration", "holdout")},
        pixel_format_counts=dict(Counter(f"{r['bit_depth']}bit-type{r['color_type']}" for r in selected)),
        overlaps=overlaps, known_main_hash_overlap=len(known_hashes & {r["sha256"] for r in selected}),
        patch_diagnostic_count=len(diagnostic),
        sampling="Seeded whole-source random assignment until quota-feasible, then source-balanced round-robin; no outcome inputs",
        rationale="112/144 screenshot/document, manga, game and icon inputs; 32 meaningful natural-photo negatives. Quotas fixed before labels. Rendered icons are not asserted to be original vector files.",
        independence="Calibration and holdout have disjoint original hashes and declared source groups. Existing benchmark overlaps are audited, not excluded by wins. This is NOT a never-benchmarked or perceptual-dedup claim; manga titles are series units, other repeated collections are kept whole.",
        validation="SHA256 plus PNG framing/CRC, dimensions, depth, color type and single-frame checks on selected originals; no pixel decode, re-encode or image copies. Full decoded sample verification belongs to experiments.",
        diagnostic="patch24 is a fixed stratified subset of holdout, excludes all known12 hashes, not selected for Patch activation; do not tune rules on it before final holdout evaluation.",
        analysis="Input-only windows/features separately timed. Calibration-only one-split byte-cost decision stump; baseline ties win. Do not force wins with collapsed labels. No PNG byte-count or codec result feature. Offline estimated chosen timing is NOT measured chosen-route runtime, nor a deployable C61 port.")
    return dict(protocol=protocol, study=selected, calibration=[r for r in selected if r["split"] == "calibration"],
                holdout=holdout, patch24=diagnostic, known12=known_cases)


def write_manifests(result, output):
    output = Path(output)
    targets = [output / (name + ".json") for name in result]
    if any(p.exists() for p in targets):
        raise FileExistsError("Frozen study outputs already exist")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in result.items():
        payload = content if name == "protocol" else dict(
            schema="representation-cases-v1", seed=result["protocol"]["seed"],
            count=len(content), cases=content)
        with (output / (name + ".json")).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--known", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.population, args.known, args.reference, args.seed)
    write_manifests(result, args.out)
    print(json.dumps({k: result["protocol"][k] for k in
                      ("count", "split_counts", "quotas", "exclusion_counts", "patch_diagnostic_count")}))


if __name__ == "__main__":
    main()
