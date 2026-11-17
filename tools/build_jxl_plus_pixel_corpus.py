"""Build a fixed, metadata-only PNG pilot corpus for pixel-lossless research.

No image decoder, codec, network access, cropping or resizing is used. Selection
is fixed by content strata and names, before compression measurements. Saved
run records are consulted only to identify prior use, never to rank outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import zlib

WORKSPACE = Path(__file__).resolve().parents[1]
TESTDATA = Path("D:/cpp_pj/jxl_python/jxl_testdata")
DIVERSE = Path("D:/cpp_pj/jxl_python/out/datasets/lossless_diverse_v1")
MAX_IMAGE_PIXELS = 2_500_000
MAX_TOTAL_PIXELS = 20_000_000

# source root, relative path, label, group, coverage tags, canonical fixture
# Numbers/names are fixed identifiers, not compression-ranking selections.
SELECTION = [
    ("testdata", "images/photo_kodak/kodim01.png", "photo-kodak-01", "photo", ["photo"], False),
    ("testdata", "images/photo_kodak/kodim02.png", "photo-kodak-02", "photo", ["photo"], False),
    ("testdata", "images/photo_tecnick/RGB_OR_1200x1200_001.png", "photo-tecnick-001", "photo", ["photo"], False),
    ("testdata", "images/photo_wikipedia/001.png", "photo-wikipedia-001", "photo", ["photo"], False),
    ("diverse", "images/gb82/sc-png/gui.png", "ui-gui", "ui", ["ui", "alpha"], False),
    ("diverse", "images/gb82/sc-png/terminal.png", "text-terminal", "text", ["text", "ui"], False),
    ("testdata", "images/screenshot_web/news.ycombinator.com.png", "web-hacker-news", "web", ["web", "text", "alpha"], False),
    ("diverse", "images/imazen26/8100-lilith-web-screenshots/8277.png", "web-mobile-8277", "web", ["web", "text"], False),
    ("diverse", "images/imazen26/9094-lilith-ai-illustrations/9104.png", "illustration-9104", "illustration", ["illustration"], False),
    ("diverse", "images/imazen26/7000-lilith-plots/7006.png", "lineart-7006", "lineart", ["lineart", "text"], False),
    ("testdata", "images/textures_photo/grass1.png", "texture-grass", "texture", ["texture"], False),
    ("diverse", "images/polyhaven/rough_linen/rough_linen_Diffuse_1k.png", "texture-linen-16", "texture", ["texture", "16bit"], False),
    ("testdata", "images/icon_512/actions-bookmark-new.png", "alpha-ui-icon", "alpha", ["alpha", "ui"], False),
    ("testdata", "images/textures_plants/plant_01.png", "alpha-plant", "alpha", ["alpha", "texture"], False),
    ("testdata", "testdata-main/external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png", "gray-photo", "gray", ["gray", "photo"], True),
    ("testdata", "testdata-main/jxl/flower/flower_small.ga.png", "gray-alpha-flower", "gray", ["gray", "alpha", "photo"], True),
    ("testdata", "testdata-main/external/raw.pixls/DJI-FC6310-16bit_709_v4_krita.png", "rgb16-camera", "16bit", ["16bit", "photo"], True),
    ("diverse", "images/polyhaven/blue_floor_tiles_01/blue_floor_tiles_01_Rough_1k.png", "gray16-roughness", "16bit", ["16bit", "gray", "texture"], False),
    ("testdata", "images/pngimg/viber_PNG2.png", "rgba16-viber", "16bit", ["16bit", "alpha", "ui"], False),
    ("testdata", "images/screenshot_game/3dbrick2.png", "palette-game", "palette", ["palette", "illustration", "ui"], False),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def png_header(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(33)
    if len(header) != 33 or header[:16] != b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR":
        raise ValueError(f"Invalid PNG IHDR: {path}")
    if zlib.crc32(header[12:29]) != int.from_bytes(header[29:33], "big"):
        raise ValueError(f"Invalid PNG IHDR CRC: {path}")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", header[16:29]
    )
    allowed = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
    if not width or not height or depth not in allowed.get(color, set()) or compression or filtering or interlace > 1:
        raise ValueError(f"Invalid PNG IHDR parameters: {path}")
    return dict(width=width, height=height, bit_depth=depth,
                png_color_type=color, interlaced=bool(interlace))


def normalize(path: str | Path) -> str:
    return str(path).replace("\\", "/").casefold()


def prior_runs(workspace: Path) -> list[Path]:
    return [
        workspace / "out/static-context-corpus200-20261112/run/summary.json",
        workspace / "out/effort-png200-20261114/summary.json",
        workspace / "out/pixel_port_corpus_20261023.jsonl",
        Path("D:/cpp_pj/jxl_python/out/scheduler_gather_20261020/stratified32_trial.json"),
        Path("D:/cpp_pj/jxl_python/out/scheduler_gather_20261020/stratified_trial.json"),
    ]


def prior_index(paths: list[Path]) -> tuple[dict, dict, list]:
    seen_paths, seen_hashes, receipts = {}, {}, []

    def collect(value, receipt):
        if isinstance(value, dict):
            for key, item in value.items():
                # Read identifiers only; no bytes, scores, speed or ranking
                # fields from previous experiments influence selection.
                if isinstance(item, str) and len(item) == 64 and ("sha256" in key or key == "identity"):
                    seen_hashes.setdefault(item.casefold(), set()).add(receipt)
                collect(item, receipt)
        elif isinstance(value, list):
            for item in value:
                collect(item, receipt)
        elif isinstance(value, str) and value.casefold().endswith(".png"):
            seen_paths.setdefault(normalize(value), set()).add(receipt)

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required prior-use record missing: {path}")
        data = path.read_bytes()
        receipt = path.resolve().as_posix()
        receipts.append(dict(source=receipt, sha256=hashlib.sha256(data).hexdigest(), original_bytes=len(data)))
        if path.suffix == ".jsonl":
            encoding = "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
            for line in data.decode(encoding).splitlines():
                if line.strip():
                    collect(json.loads(line), receipt)
        else:
            collect(json.loads(data), receipt)
    return seen_paths, seen_hashes, receipts


def build(testdata: Path, diverse: Path, workspace: Path) -> dict:
    seen_paths, seen_hashes, prior_receipts = prior_index(prior_runs(workspace))
    # This existing stratified inventory supplies semantic categories and
    # provenance, not outcome-based selection. Whole files stay untouched.
    inventory = diverse / "manifest.jsonl"
    inventory_rows = {}
    with inventory.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            inventory_rows[normalize(diverse / row["path"])] = row
    roots = {"testdata": testdata, "diverse": diverse}
    cases = []
    for root, relative, label, group, tags, canonical_fixture in SELECTION:
        path = (roots[root] / relative).resolve()
        header = png_header(path)
        pixels = header["width"] * header["height"]
        if pixels > MAX_IMAGE_PIXELS:
            raise ValueError(f"Individual whole-file pixel cap exceeded: {label}")
        digest = sha256(path)
        evidence = sorted(seen_paths.get(normalize(path), set()) | seen_hashes.get(digest, set()))
        if canonical_fixture:
            evidence.append("conservative-development-classification: existing canonical codec fixture")
        case = dict(source=path.as_posix(), label=label, group=group, sha256=digest,
                    **header, original_bytes=path.stat().st_size,
                    previously_used=bool(evidence), previously_used_evidence=evidence,
                    coverage_tags=tags, pixels=pixels)
        row = inventory_rows.get(normalize(path))
        if row:
            if (row.get("width"), row.get("height"), row.get("bit_depth"), row.get("png_color_type")) != (
                    header["width"], header["height"], header["bit_depth"], header["png_color_type"]):
                raise ValueError(f"Inventory/header disagreement: {label}")
            if row.get("sha256") and row["sha256"] != digest:
                raise ValueError(f"Inventory/source SHA disagreement: {label}")
            case["inventory_category"] = row.get("category")
            case["inventory_group_id"] = row.get("group_id")
            case["source_license"] = row.get("license")
        cases.append(case)
    if not 12 <= len(cases) <= 24 or sum(row["pixels"] for row in cases) > MAX_TOTAL_PIXELS:
        raise ValueError("Pilot corpus count/total pixel cap exceeded")
    if len({row["sha256"] for row in cases}) != len(cases):
        raise ValueError("Duplicate source bytes in fixed selection")
    tags = set().union(*(set(row["coverage_tags"]) for row in cases))
    if not {"photo", "ui", "web", "text", "illustration", "texture", "alpha", "gray", "16bit"} <= tags:
        raise ValueError("Required content strata missing")
    return dict(
        schema="rs-jxl-plus.pixel-corpus.v1",
        selection="Fixed named content-stratified whole PNGs; no cropping, decoding, encoding or compression ranking.",
        inspection="IHDR/CRC, file size and SHA256 only. Alpha tags identify alpha-bearing PNG color types; actual transparency and invisible-RGB sample statistics were not decoded.",
        acceptance="Original decoded samples including invisible RGB; PNG file-byte equality is not the pixel acceptance contract.",
        previously_used_scope="True means present by path/SHA in the listed prior-use records, or conservatively classified canonical development fixture. False means absent from those records, not proof of never being used anywhere.",
        inventory=dict(source=inventory.resolve().as_posix(), sha256=sha256(inventory)),
        prior_use_records=prior_receipts,
        limits=dict(max_cases=24, max_image_pixels=MAX_IMAGE_PIXELS, max_total_pixels=MAX_TOTAL_PIXELS),
        summary=dict(cases=len(cases), pixels=sum(row["pixels"] for row in cases),
                     original_bytes=sum(row["original_bytes"] for row in cases),
                     by_group=dict(sorted(Counter(row["group"] for row in cases).items())),
                     previously_used=sum(row["previously_used"] for row in cases),
                     absent_from_prior_records=sum(not row["previously_used"] for row in cases)),
        cases=cases,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata-root", type=Path, default=TESTDATA)
    parser.add_argument("--diverse-root", type=Path, default=DIVERSE)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "out/pja-port-20261117/pixel-manifest.json")
    args = parser.parse_args()
    manifest = build(args.testdata_root, args.diverse_root, WORKSPACE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
