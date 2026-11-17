"""Freeze small development sequences for the multiframe rate-feedback benchmark.

Synthetic cases are deterministic; photographic edits are labelled as derived,
not real animation. Original JPEG input is decoded once with the frozen DLL.
No encoding, resizing, corpus search, or external-directory writes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.out / "manifest.json"
    if manifest.exists():
        parser.error("use a new manifest path")
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures_dir = args.out / "fixtures"
    fixtures_dir.mkdir(exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "python"))
    sys.path.insert(0, str(project / "scripts"))
    import rs_jxl
    from rs_jxl import _ffi
    from benchmark_multiframe_reuse import fixtures

    cases = []

    def save(label, family, arrays, *, reference="auto", auto_options=None, icc=None):
        paths = []
        for index, array in enumerate(arrays):
            path = fixtures_dir / f"{label}-{index}.npy"
            np.save(path, array, allow_pickle=False)
            paths.append(str(path.relative_to(args.out)))
        cases.append({"label": label, "family": family, "frames": paths,
                      "options": {"strategy": "auto", "reference": reference,
                                  "layout": "hwc", "bits_per_sample": 8,
                                  "color_encoding": "srgb", "durations": 1,
                                  "auto_options": auto_options or {}}})
        if icc is not None:
            path = fixtures_dir / f"{label}.icc"
            path.write_bytes(icc)
            cases[-1]["icc"] = str(path.relative_to(args.out))
            cases[-1]["options"].pop("color_encoding")

    for label, arrays in fixtures().items():
        save(label, "synthetic-control", arrays)

    rng = np.random.default_rng(20261117)
    textured = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    blank = np.zeros_like(textured)

    def edits(array, tile, xor=127):
        result = array.copy()
        for y in range(1, result.shape[0] - 3, tile):
            for x in range(1, result.shape[1] - 3, tile):
                result[y:y + 3, x:x + 3, :3] ^= np.uint8(xor)
        return result

    partition = {"tile_size": 16, "max_rectangles": 16}
    save("flat-to-textured", "synthetic-development",
         [blank, textured, edits(textured, 16)], reference="previous", auto_options=partition)
    save("textured-to-flat", "synthetic-control",
         [textured, blank, edits(blank, 16)], reference="previous", auto_options=partition)
    alpha = np.indices((128, 128)).sum(axis=0).astype(np.uint8) % 3 * np.uint8(127)
    rgba = np.dstack((textured, alpha))
    rgba_blank = np.dstack((blank, np.zeros_like(alpha)))
    save("hidden-rgb-transition", "synthetic-development",
         [rgba_blank, rgba, edits(rgba, 16)], reference="previous", auto_options=partition)

    library_path = args.before.resolve(strict=True)
    library_hash = digest(library_path)
    lib = _ffi.ffi.dlopen(str(library_path))
    _ffi._validate_abi(lib, library_path)
    sources = []
    photographs = []
    profiles = []
    with patch.object(_ffi, "library", return_value=lib):
        decoder = rs_jxl.Decoder(threads=4)
    with decoder:
        for filename in ("146810648_p1.jpg", "146810648_p2.jpg"):
            path = (args.real_dir / filename).resolve(strict=True)
            before = digest(path)
            image, info = decoder.decode(path.read_bytes(), return_info=True)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("expected original RGB8 photographic controls")
            if info.get("orientation", 1) != 1:
                raise ValueError("fixed photographic controls require unrotated sources")
            profile = bytes(info.get("icc") or [])
            if not profile:
                raise ValueError("fixed photographic controls require their original ICC")
            assert digest(path) == before
            sources.append({"path": str(path), "sha256": before})
            photographs.append(image)
            profiles.append(profile)
    if photographs[0].shape != photographs[1].shape:
        raise ValueError("original control dimensions must match")
    if profiles[0] != profiles[1]:
        raise ValueError("original control profiles must match")
    save("t3-matching-pair", "real-control", photographs, icc=profiles[0])
    photo = photographs[0]
    save("photo-with-flat-intro-and-edits", "derived-development",
         [np.zeros_like(photo), photo, edits(photo, 64)], reference="previous",
         auto_options={"tile_size": 64, "max_rectangles": 16}, icc=profiles[0])
    assert digest(library_path) == library_hash
    report = {"description": "Fixed development cases; derived edits are not real animation",
              "before": {"path": str(library_path), "sha256": library_hash},
              "sources": sources, "cases": cases}
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(cases)} sequences: {manifest}")


if __name__ == "__main__":
    main()
