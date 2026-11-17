"""Alternate before/after native multiframe encoding on a fixed .npy manifest.

Manifest: {"cases":[{"label":"clip","family":"real","frames":["0.npy",
"1.npy"],"options":{"strategy":"auto","reference":"previous",
"durations":[40,60],"effort":9,"threads":4}}]}. Paths are manifest-relative.
No builds, discovery or codec mocks. Only context library selection is patched.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
from compare_cpp_port import clean_environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(actual, expected, metadata, durations):
    actual = actual if isinstance(actual, list) else [actual]
    if len(actual) != len(expected) or len(metadata["frames"]) != len(expected):
        raise AssertionError("decoded frame count differs")
    for value, source, frame, duration in zip(actual, expected, metadata["frames"], durations):
        if (value.shape != source.shape or value.dtype != source.dtype
                or not np.array_equal(value.view(np.uint8), source.view(np.uint8))):
            raise AssertionError("original sample words/shape/dtype differ (including hidden RGB)")
        if frame["duration"] != duration:
            raise AssertionError("original frame duration differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("before", "after", "manifest", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, help="override manifest threads (otherwise 4)")
    parser.add_argument("--effort", type=int, help="override manifest effort (otherwise 9)")
    parser.add_argument("--after-rate-feedback", choices=("on", "off"),
                        help="override the after build's adaptive_rate_feedback only")
    parser.add_argument("--after-adaptive-splits", choices=("on", "off"))
    parser.add_argument("--after-multiply-mask", choices=("on", "off"))
    parser.add_argument("--streams-dir", type=Path, help="NEW directory for first encoded streams")
    args = parser.parse_args()
    if (args.out.exists() or args.repeats < 1 or args.warmups < 0
            or (args.threads is not None and not 1 <= args.threads <= 256)
            or (args.effort is not None and not 1 <= args.effort <= 9)):
        parser.error("use a new output and valid repetition/thread/effort settings")
    if args.streams_dir:
        args.streams_dir.mkdir(parents=True, exist_ok=False)
    env, environment = clean_environment(os.environ)
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import rs_jxl
    from rs_jxl import _ffi

    manifest = args.manifest.resolve(strict=True)
    spec = json.loads(manifest.read_text(encoding="utf-8-sig"))
    identities = {manifest: digest(manifest)}
    report = dict(status="running", config=vars(args), environment=environment,
                  measurement="reusable public Encoder.encode_multiframe / Decoder.decode, including FFI/output copies",
                  verification="every warmup and measured call; all original frame words, shapes, dtypes and durations outside timers",
                  binaries={}, cases=[], summaries=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        libraries = {}
        for name in ("before", "after"):
            path = getattr(args, name).resolve(strict=True)
            identities[path] = digest(path)
            lib = _ffi.ffi.dlopen(str(path))
            _ffi._validate_abi(lib, path)
            libraries[name] = lib
            report["binaries"][name] = dict(path=str(path), sha256=identities[path])
        labels = set()
        for index, case in enumerate(spec["cases"]):
            label, family = case["label"], case["family"]
            if label in labels or not isinstance(family, str) or not family.strip():
                raise ValueError("unique labels and nonempty string family required")
            labels.add(label)
            print(f"Case {index + 1}/{len(spec['cases'])}: {label} [{family}] loading", flush=True)
            frames, source_records = [], []
            for item in case["frames"]:
                path = (manifest.parent / item).resolve(strict=True)
                identities[path] = digest(path)
                frame = np.load(path, allow_pickle=False)
                if frame.ndim != 3 or not frame.flags.c_contiguous:
                    raise ValueError("frames must be C-contiguous HWC .npy arrays")
                frames.append(frame)
                source_records.append(dict(path=str(path), sha256=identities[path], shape=frame.shape, dtype=str(frame.dtype)))
            if not frames:
                raise ValueError("case needs frames")
            options = {"strategy": "auto", "reference": "auto", "effort": 9, "threads": 4, **case.get("options", {})}
            if args.threads is not None:
                options["threads"] = args.threads
            if args.effort is not None:
                options["effort"] = args.effort
            if not 1 <= options["effort"] <= 9:
                raise ValueError("effort must be 1..9")
            icc, icc_record = None, None
            if case.get("icc") is not None:
                path = (manifest.parent / case["icc"]).resolve(strict=True)
                identities[path] = digest(path)
                icc = path.read_bytes()
                options["icc_profile"] = icc
                options.pop("color_encoding", None)
                icc_record = dict(path=str(path), sha256=identities[path], bytes=len(icc))
            if "output" in options or "return_info" in options:
                raise ValueError("output and return_info are harness-controlled")
            durations = options.get("durations", 1)
            durations = [1 if durations is None else durations] * len(frames) if durations is None or np.isscalar(durations) else list(durations)
            if len(durations) != len(frames):
                raise ValueError("one duration per frame required")
            pixels = sum(f.shape[0] * f.shape[1] for f in frames)
            recorded_options = dict(options)
            if icc_record is not None:
                recorded_options["icc_profile"] = icc_record
            row = dict(label=label, family=family, options=recorded_options, frames=source_records,
                       pixels=pixels, durations=durations, builds={})
            report["cases"].append(row)
            build_options = {name: dict(options) for name in libraries}
            for field, value in (("adaptive_rate_feedback", args.after_rate_feedback),
                                 ("adaptive_splits", args.after_adaptive_splits),
                                 ("allow_multiply_mask", args.after_multiply_mask)):
                if value is not None:
                    build_options["after"]["auto_options"] = {
                        **build_options["after"].get("auto_options", {}), field: value == "on"}
            row["after_rate_feedback"] = args.after_rate_feedback
            row["after_adaptive_splits"] = args.after_adaptive_splits
            row["after_multiply_mask"] = args.after_multiply_mask
            with ExitStack() as stack:
                encoders, decoders = {}, {}
                for name, lib in libraries.items():
                    with patch.object(_ffi, "library", return_value=lib):
                        encoders[name] = stack.enter_context(rs_jxl.Encoder(threads=options["threads"]))
                        decoders[name] = stack.enter_context(rs_jxl.Decoder(threads=options["threads"]))
                    row["builds"][name] = dict(samples=[], verified_calls=0)
                for repetition in range(-args.warmups, args.repeats):
                    names = ["before", "after"]
                    if (repetition + index) % 2:
                        names.reverse()
                    for name in names:
                        build = row["builds"][name]
                        start = time.perf_counter()
                        encoded, info = encoders[name].encode_multiframe(frames, return_info=True, **build_options[name])
                        encode_seconds = time.perf_counter() - start
                        start = time.perf_counter()
                        actual, metadata = decoders[name].decode(encoded, frame=None, return_info=True)
                        decode_seconds = time.perf_counter() - start
                        exact(actual, frames, metadata, durations)
                        if icc is not None and bytes(metadata.get("icc") or []) != icc:
                            raise AssertionError("original ICC profile differs")
                        size, sha = len(encoded), hashlib.sha256(encoded).hexdigest()
                        if "complete_bytes" in build and build["complete_bytes"] != size:
                            raise AssertionError(f"nondeterministic encoded size: {label}/{name}")
                        if "first_sha256" in build and build["first_sha256"] != sha:
                            raise AssertionError(f"nondeterministic encoded bytes: {label}/{name}")
                        if "complete_bytes" not in build:
                            build.update(complete_bytes=size, first_sha256=sha, search_reports=info.get("search_reports", info.get("frames")), first_encode_info=info)
                            if args.streams_dir:
                                path = args.streams_dir / f"{index:03d}-{name}.jxl"
                                with path.open("xb") as stream:
                                    stream.write(encoded)
                                build["saved_stream"] = str(path.resolve())
                        build["verified_calls"] += 1
                        if repetition >= 0:
                            build["samples"].append(dict(repetition=repetition, order=names, encode_seconds=encode_seconds, decode_seconds=decode_seconds, complete_bytes=size, sha256=sha))
                    phase = "warmup" if repetition < 0 else "measured"
                    print(f"{label}: {phase} round {repetition + args.warmups + 1}/{args.warmups + args.repeats} complete", flush=True)
                for build in row["builds"].values():
                    for operation in ("encode", "decode"):
                        seconds = statistics.median(s[operation + "_seconds"] for s in build["samples"])
                        build[operation] = dict(median_seconds=seconds, mpix_s=pixels / 1e6 / seconds)
            before, after = (row["builds"][name] for name in ("before", "after"))
            row["size_change_percent"] = 100 * (after["complete_bytes"] / before["complete_bytes"] - 1)
            print(f'{label}: {before["complete_bytes"]} -> {after["complete_bytes"]} B ({row["size_change_percent"]:+.3f}%); '
                  f'encode {before["encode"]["mpix_s"]:.3f} -> {after["encode"]["mpix_s"]:.3f} MPix/s; '
                  f'decode {before["decode"]["mpix_s"]:.3f} -> {after["decode"]["mpix_s"]:.3f} MPix/s', flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            for path, before in identities.items():
                if digest(path) != before:
                    raise AssertionError(f"input/library changed: {path}")
        except BaseException as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            report["files"] = [dict(path=str(path), sha256=value) for path, value in identities.items()]
            with args.out.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, default=str)
                output.write("\n")


if __name__ == "__main__":
    main()
