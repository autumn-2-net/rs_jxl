"""Freeze prior real cohorts and generate mechanism controls without codec tools."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from compare_cpp_port import png_payload

DEFAULT_WORK_DIR = ROOT / "out" / "patch-near-20261125"
OLD = ROOT / "out" / "patch-ma-followup-20261113"
# Exclude the Lilith web collection, wallhaven and Kenney Pixel Platformer
# from the holdout since those source families occur in known12.
HOLDOUT = {
    1: ("sprite", "Kenney Particle Pack"),
    11: ("web", "Creative Commons site"),
    18: ("game", "Nuclear Throne"),
    24: ("manga", "Manga109 SeisinkiVulnus"),
    42: ("photo-negative", "Set14 Monarch"),
    43: ("photo-negative", "Tecnick RGB_OR_1200x1200_023"),
    48: ("texture", "jxl-testdata clover texture"),
    54: ("urban", "Urban100"),
}
VARIANTS = (
    ("exact-repeat", False, 8, False),
    ("aa-boundary-rgb8", True, 8, False),
    ("aa-boundary-hidden-rgba8", True, 8, True),
    ("aa-boundary-hidden-rgba16", True, 16, True),
    ("exact-repeat-hidden-rgba16", False, 16, True),
)


def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def fixture(vary, depth, alpha):
    width, height = 513, 273
    shape_w, shape_h = 19, 21
    rng = random.Random(20261125)
    glyph = {}
    for y in range(shape_h):
        for x in range(shape_w):
            if 2 <= x <= 5 or 1 <= y <= 4 or 9 <= y <= 12 or 17 <= y <= 19:
                glyph[x, y] = tuple(rng.randrange(100, 225) for _ in range(3))
    # This AA fringe is connected and never enters the background flood.
    edge = [(x, 20) for x in range(4)]
    glyph.update({p: (99, 124, 124) for p in edge})
    patch_origins = [(17 + 61 * col + row % 3, 17 + 48 * row + col % 3)
                     for row in range(2) for col in range(8)]
    background = (20, 20, 20)
    color = [background] * (width * height)
    fingerprint_keys = []
    scale = 257 if depth == 16 else 1
    for i, (x0, y0) in enumerate(patch_origins):
        patch = glyph.copy()
        if vary:
            for bit, p in enumerate(edge):
                if i & (1 << bit):
                    patch[p] = (101, 124, 124)
        for (x, y), samples in patch.items():
            color[(y0 + y) * width + x0 + x] = samples
        fingerprint_keys.append([int((patch[p][0] - 20) / 20) for p in edge])
    # The changed R deltas 79 and 81 straddle q=4 and remain distinct even
    # with divisor 4. The 16-bit test adds non-8-bit words inside the glyph.
    assert len({tuple(q // 4 for q in key) for key in fingerprint_keys}) == (16 if vary else 1)
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            rgb = color[y * width + x]
            words = [v * scale for v in rgb]
            if depth == 16 and rgb != background:
                words[2] += 3
            if alpha:
                words.append(0 if (x // 43 + y // 37) % 3 else 255 * scale)
            raw.extend(bytes(words) if depth == 8 else struct.pack(">"+"H"*len(words), *words))
    header = struct.pack(">IIBBBBB", width, height, depth, 6 if alpha else 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return png, {"glyph_size": [shape_w, shape_h], "occurrences": patch_origins,
                 "red_fringe_keys_d1": fingerprint_keys, "distinct_fringe_keys_d4": 16 if vary else 1}


def tile_fixture():
    width, height = 225, 161
    rng = random.Random(20261126)
    color = bytearray(width * height * 3)
    rects = []
    for kind, origins in [("grid", [(16, 16), (80, 16), (144, 16)]),
                          ("offgrid", [(16, 80), (79, 86), (147, 99)])]:
        tile = rng.randbytes(48 * 32 * 3)
        rects.append({"kind": kind, "size": [48, 32], "origins": origins})
        for x, y in origins:
            for dy in range(32):
                dst = ((y + dy) * width + x) * 3
                color[dst:dst + 144] = tile[dy * 144:(dy + 1) * 144]
    raw = b"".join(b"\0" + color[y:y + width * 3]
                   for y in range(0, len(color), width * 3))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return png, {"exact_rectangles": rects}


def case(path, cohort, label, relative_to=None, **fields):
    blob = path.read_bytes()
    _, header = png_payload(blob)
    if relative_to is not None and path.is_relative_to(relative_to):
        path = path.relative_to(relative_to)
    return dict(original=path.as_posix(), cohort=cohort, label=label,
                sha256=hashlib.sha256(blob).hexdigest(), input_bytes=len(blob), **header, **fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--manifest", type=Path, help="Output manifest, default WORK_DIR/manifest.json")
    parser.add_argument("--known12", type=Path, default=OLD / "patch12.json")
    parser.add_argument("--holdout60", type=Path, default=OLD / "holdout60.json")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Generate only the six deterministic fixtures")
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    target = args.manifest.resolve() if args.manifest else work_dir / "manifest.json"
    fixture_dir = work_dir / "fixtures"
    fixture_names = [name for name, *_ in VARIANTS] + ["exact-tile-rectangles"]
    for candidate in [target, *(fixture_dir / f"{name}.png" for name in fixture_names)]:
        if candidate.exists():
            parser.error(f"Will not overwrite {candidate}")

    cases = []
    real_manifests = [(12, "known", args.known12), (60, "holdout", args.holdout60)]
    for count, label, upstream in [] if args.synthetic_only else real_manifests:
        upstream = upstream.resolve(strict=True)
        published = json.loads(upstream.read_text(encoding="utf-8-sig"))
        assert len(published["cases"]) == count
        for index, entry in enumerate(published["cases"]):
            if label == "holdout" and index not in HOLDOUT:
                continue
            path = upstream.parent / entry["original"]
            path = path.resolve(strict=True)
            category, family = (HOLDOUT[index] if label == "holdout"
                                else ("source-positive-control", None))
            frozen = case(path, label, f"{label}{index+1:02}-{path.stem}",
                          category=category, source_family=family, upstream_index=index,
                          upstream_manifest=str(upstream))
            if "sha256" in entry:
                assert frozen["sha256"] == entry["sha256"]
            cases.append(frozen)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    fixtures = [(name, fixture(vary, depth, alpha)) for name, vary, depth, alpha in VARIANTS]
    fixtures.append(("exact-tile-rectangles", tile_fixture()))
    for name, (data, design) in fixtures:
        path = fixture_dir / f"{name}.png"
        with path.open("xb") as handle:
            handle.write(data)
        cases.append(case(path, "synthetic", name, relative_to=target.parent, design=design))
    assert len({c["sha256"] for c in cases}) == len(cases)
    sampling = "Six deterministic synthetic controls."
    if not args.synthetic_only:
        sampling = ("All known12 controls and eight source-family-separated holdout60 cases, "
                    "selected by type and size before near matching was benchmarked. " + sampling)
    with target.open("x", encoding="utf-8") as handle:
        json.dump({"schema": 1, "count": len(cases), "cases": cases, "sampling": sampling},
                  handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
