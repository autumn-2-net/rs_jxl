"""Seeded proportional sampling of original files only; no codec or result inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys


def largest_remainder(sizes, count):
    if (type(count) is not int or count < 1 or not sizes
            or any(type(n) is not int or n < 1 for n in sizes.values())):
        raise ValueError("Count and stratum populations must be positive integers")
    total = sum(sizes.values())
    if count > total:
        raise ValueError("Requested count exceeds population")
    quotas = {name: count * sizes[name] // total for name in sorted(sizes)}
    ranked = sorted(sizes, key=lambda name: (-(count * sizes[name] % total), name))
    for name in ranked[:count - sum(quotas.values())]:
        quotas[name] += 1
    return quotas


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_manifest(manifest, count, seed, out):
    manifest, out = Path(manifest).resolve(strict=True), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output already exists: {out}")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    raw = manifest.read_bytes()
    source = json.loads(raw)
    cases = source.get("cases") if isinstance(source, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain a nonempty cases list")
    groups, paths, hashes = {}, set(), set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        name, original, sha = (case.get(key) for key in ("stratum", "original", "sha256"))
        if (not isinstance(name, str) or not name.strip()
                or not isinstance(original, str) or not original
                or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha)):
            raise ValueError("Each case needs stratum, original and SHA256")
        path = Path(original)
        path = (path if path.is_absolute() else manifest.parent / path).resolve()
        if path in paths or sha.lower() in hashes:
            raise ValueError(f"Duplicate resolved path or SHA256: {path}")
        paths.add(path)
        hashes.add(sha.lower())
        groups.setdefault(name, []).append(dict(case, original=str(path)))
    sizes = {name: len(groups[name]) for name in sorted(groups)}
    quotas = largest_remainder(sizes, count)
    rng, selected = random.Random(seed), []
    for name, quota in quotas.items():
        population = sorted(groups[name], key=lambda case: Path(case["original"]).as_posix())
        selected.extend(rng.sample(population, quota))
    rng.shuffle(selected)
    for case in selected:
        path = Path(case["original"])
        if not path.is_file():
            raise FileNotFoundError(f"Selected original is not a file: {path}")
        if file_sha256(path) != case["sha256"].lower():
            raise ValueError(f"Selected original SHA256 mismatch: {path}")
    result = dict(
        schema=1, seed=seed, count=count, quotas=quotas,
        source_manifest=dict(path=str(manifest), sha256=hashlib.sha256(raw).hexdigest(),
                             count=len(cases), metadata={k: v for k, v in source.items() if k != "cases"}),
        population_counts=sizes,
        sampling="Largest remainder (alphabetical ties); sorted strata/paths; seeded sample then shuffle",
        verified_originals=len(selected), cases=selected,
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = sample_manifest(args.manifest, args.count, args.seed, args.out)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(dict(out=str(args.out.resolve()), count=result["count"],
                          verified_originals=result["verified_originals"], quotas=result["quotas"],
                          source_manifest_sha256=result["source_manifest"]["sha256"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
