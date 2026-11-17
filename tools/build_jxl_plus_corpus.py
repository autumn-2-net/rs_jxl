#!/usr/bin/env python3
"""Build a fixed, content-disjoint JPEG experiment manifest; never encode inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=repo.parent / "jxl_python/out/jpeg-opt-20261116/cases.json")
    parser.add_argument("--development", type=Path, default=repo / "out/pja-port-20261117/smoke7-size-comparison.json")
    parser.add_argument("--root", type=Path, help="CC12m directory; defaults to prior manifest's CC12m root")
    parser.add_argument("--output", type=Path, default=repo / "out/pja-port-20261117/corpus256-manifest.json")
    args = parser.parse_args()
    prior = read_json(args.prior)["cases"]
    development = read_json(args.development)["cases"]
    root = args.root or next(Path(c["original"]).parent for c in prior if "cc12" in c["original"].lower())
    cache = {}

    def info(path):
        path = path.resolve(strict=True)
        key = str(path)
        if key not in cache:
            cache[key] = (path.stat().st_size, digest(path))
        return path, *cache[key]

    cases, excluded = [], set()
    counts = {"development_input": len(development), "prior_input": len(prior),
              "development_content_duplicates": 0, "prior_content_duplicates_or_development": 0}
    for source, split, duplicate_key in [(development, "development7", "development_content_duplicates"),
                                          (prior, "prior_audit", "prior_content_duplicates_or_development")]:
        for item in source:
            path, size, sha = info(Path(item.get("source", item.get("original"))))
            if sha in excluded:
                counts[duplicate_key] += 1
                continue
            excluded.add(sha)
            cases.append(dict(label=item["label"], source=str(path), split=split,
                              sha256=sha, original_bytes=size, stratum=None))

    files = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")),
                   key=lambda p: (p.stat().st_size, p.relative_to(root).as_posix()))
    selected_hashes = set(excluded)
    strata = []
    counts.update(dataset_files=len(files), dataset_excluded_prior_content=0,
                  dataset_duplicate_content=0, selected_new=0)
    for s in range(4):
        population = files[len(files) * s // 4:len(files) * (s + 1) // 4]
        ordered = sorted(population, key=lambda p: (hashlib.sha256(p.relative_to(root).as_posix().encode("utf-8")).hexdigest(), p.relative_to(root).as_posix()))
        unique, local_seen = [], set()
        exclusions, duplicates = 0, 0
        # Scan every member so exclusion/duplicate counts describe the whole stratum.
        for p in ordered:
            path, size, sha = info(p)
            if sha in excluded:
                exclusions += 1
            elif sha in selected_hashes or sha in local_seen:
                duplicates += 1
            else:
                local_seen.add(sha)
                unique.append((path, size, sha))
        if len(unique) < 64:
            raise ValueError(f"stratum {s} has only {len(unique)} eligible unique files")
        chosen = unique[:64]
        for n, (path, size, sha) in enumerate(chosen):
            selected_hashes.add(sha)
            cases.append(dict(label=f"cc12m-q{s + 1}-{n + 1:03d}", source=str(path), split="validation256",
                              sha256=sha, original_bytes=size, stratum=s + 1))
        counts["dataset_excluded_prior_content"] += exclusions
        counts["dataset_duplicate_content"] += duplicates
        counts["selected_new"] += len(chosen)
        strata.append(dict(stratum=s + 1, population=len(population),
                           population_min_bytes=min(p.stat().st_size for p in population),
                           population_max_bytes=max(p.stat().st_size for p in population),
                           excluded_prior_content=exclusions, duplicate_content=duplicates,
                           eligible_unique=len(unique), selected=len(chosen),
                           selected_bytes=sum(v[1] for v in chosen),
                           selected_min_bytes=min(v[1] for v in chosen), selected_max_bytes=max(v[1] for v in chosen)))
    counts["split_counts"] = {split: sum(c["split"] == split for c in cases)
                              for split in ("development7", "prior_audit", "validation256")}
    result = dict(schema="jxl-plus-corpus-v1", selection=dict(dataset_root=str(root.resolve()),
                  recursive=False, extensions=[".jpg", ".jpeg"],
                  strata="four equal-rank file-size quartiles before exclusions; ties by relative POSIX path",
                  within_stratum="ascending SHA256(UTF-8 relative POSIX path), tie by path; select first 64 eligible unique contents",
                  exclusion="exclude development and prior content SHA256; exclude duplicate selected contents across strata",
                  support="unfiltered JPEG candidates; codec support/rejections must be recorded by evaluator, no replacement based on compression results",
                  prior_manifest=str(args.prior.resolve()), development_receipt=str(args.development.resolve())),
                  counts=counts, strata=strata, validation_bytes=sum(c["original_bytes"] for c in cases if c["split"] == "validation256"), cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": counts, "strata": strata,
                      "validation_bytes": result["validation_bytes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
