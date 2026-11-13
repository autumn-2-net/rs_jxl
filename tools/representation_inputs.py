"""Frozen-manifest input preflight, without image decoding or codec execution.

The returned list/order is benchmark_fast_context.load_cases's contract. Only
manifest entries declaring sha256 are hash-verified; legacy entries remain
accepted with a warning, and explicit --case inputs have no declared hash.
This is a preflight, not a file lock: harnesses must still record/check the
actual original or snapshot they encode and join final reports by original SHA.
"""

import json
from pathlib import Path
import re
import warnings

from benchmark_fast_context import load_cases
from compare_cpp_port import file_sha256


def load_verified_cases(args):
    """Load paths and reject any supplied manifest SHA256 mismatch before trials.

    SHA256 must be a 64-digit hexadecimal string when its key is present;
    null/empty/malformed declarations are errors, not legacy opt-outs. Relative
    original paths resolve against the manifest directory, as in load_cases.
    """
    if args.manifest is None:
        return load_cases(args)
    manifest = Path(args.manifest)
    frozen = manifest.read_bytes()
    sources = load_cases(args)
    if manifest.read_bytes() != frozen:
        raise ValueError(f"Manifest changed during input loading: {manifest}")
    cases = json.loads(frozen.decode("utf-8"))["cases"]
    expected_paths, declared, legacy = [], [], 0
    for case in cases:
        path = Path(case["original"])
        path = (path if path.is_absolute() else manifest.resolve().parent / path).resolve(strict=True)
        expected_paths.append(path)
        if "sha256" not in case:
            legacy += 1
            continue
        sha = case["sha256"]
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
            raise ValueError(f"Invalid manifest sha256 declaration for {path}")
        declared.append((path, sha.lower()))
    if sources[len(args.case):] != expected_paths:
        raise ValueError(f"Manifest paths changed during input loading: {manifest}")
    for path, expected in declared:
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen manifest SHA256 mismatch for {path}: expected {expected}, actual {actual}")
    if legacy:
        warnings.warn(
            f"Manifest {manifest}: {legacy} legacy case(s) without sha256 accepted; "
            "these originals are not frozen-hash verified",
            UserWarning, stacklevel=2)
    return sources
