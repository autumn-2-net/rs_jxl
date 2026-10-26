"""Opt-in PNG pixel comparison against the frozen optimized C++ port source.

Requires numpy and OpenCV for independent, non-narrowing PNG sample reads.
No build, download, corpus discovery or benchmark runs happen on import/help.
Pass an independently obtained official djxl explicitly (repeat for multiple
oracles). Its origin is caller-attested, not certified from its version string.

Both encoders receive the SAME metadata-free PNG snapshot. PNG compressed pixels,
palette and transparency are retained; samples are checked against the ORIGINAL
PNG, without color conversion, orientation changes or alpha compositing. This is
pixel acceptance only, not metadata acceptance or proof of port completeness.

single-candidate: Rust --source-effort; C++ --iterations=50 explicitly retains
nb_repeats=0.5 but clears ma_tree_learning_percent_is_default, disabling
HasLosslessSearchOptions (b866b3d: encode.cc and enc_lossless_search.cc).
full-scheduler: Rust also --source-search; C++ defaults, with required search
start/end trace evidence. Optional Rust scheduler routes are still incomplete;
matching effort labels do NOT imply equivalent policies or implementation.

Wall time is serial CLI latency including startup, PNG I/O, encoding, output I/O
and captured diagnostics, excluding independent decoding and sample comparison.
Order alternates Rust/C++ and C++/Rust. Warmups are validated but not aggregated.
All transient inputs/JXL/decoded PNGs are removed, including on failure.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import tempfile
from time import perf_counter
import zlib


CPP_COMMIT = "b866b3defec13494bfbaf705db93581cda118e37"
CPP_SOURCE = Path(__file__).resolve().parents[2] / "jxl_python" / "libjxl"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MODES = ("single-candidate", "full-scheduler")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_libraries():
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("PNG oracle requires numpy and opencv-python in this Python environment") from error
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    return cv2, np


def png_payload(data):
    """Validate framing/CRC and remove metadata, never recompress pixel data."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Only PNG input is allowed; JPEG is never accepted")
    offset = 8
    chunks, removed, seen = [], [], set()
    info = None
    ended_idat = False
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError("Truncated PNG chunk")
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + length + 12
        if length > 0x7fffffff or end > len(data):
            raise ValueError("Invalid or truncated PNG chunk payload")
        if any(not (65 <= c <= 90 or 97 <= c <= 122) for c in kind) or kind[2] & 32:
            raise ValueError("Invalid PNG chunk type")
        body = data[offset + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + body) != crc:
            raise ValueError(f"PNG CRC mismatch: {kind!r}")
        if info is None and kind != b"IHDR":
            raise ValueError("PNG must start with IHDR")
        if kind in (b"acTL", b"fcTL", b"fdAT"):
            raise ValueError("APNG is outside single-frame PNG pixel acceptance")
        if kind == b"IHDR":
            if info is not None or length != 13:
                raise ValueError("Invalid or repeated PNG IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
            if (depth not in (8, 16) or color not in (0, 2, 3, 4, 6)
                    or (color == 3 and depth != 8)):
                raise ValueError("Only 8/16-bit gray/GA/RGB/RGBA and 8-bit palette PNG are supported")
            if not 0 < width <= 0x7fffffff or not 0 < height <= 0x7fffffff:
                raise ValueError("PNG dimensions exceed the nonzero 31-bit format range")
            if compression or filtering or interlace not in (0, 1):
                raise ValueError("Unsupported PNG compression/filter/interlace method")
            info = dict(width=width, height=height, bit_depth=depth, color_type=color,
                        interlaced=bool(interlace))
        elif kind == b"IDAT":
            if ended_idat or (info["color_type"] == 3 and b"PLTE" not in seen):
                raise ValueError("Invalid PNG IDAT/palette ordering")
        elif kind in (b"PLTE", b"tRNS"):
            if kind in seen or b"IDAT" in seen:
                raise ValueError("Invalid PNG palette/transparency ordering")
            if kind == b"tRNS" and info["color_type"] != 3:
                count = {0: 1, 2: 3}.get(info["color_type"])
                if count is None or length != count * 2:
                    raise ValueError("Invalid PNG transparency key")
                info["transparency_key"] = list(struct.unpack(">" + "H" * count, body))
        elif kind == b"IEND":
            if length or end != len(data) or b"IDAT" not in seen:
                raise ValueError("Invalid PNG IEND or trailing data")
        elif not kind[0] & 32:
            raise ValueError(f"Unknown critical PNG chunk: {kind!r}")
        if kind != b"IDAT" and b"IDAT" in seen:
            ended_idat = True
        if kind in (b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"):
            chunks.append(data[offset:end])
        else:
            removed.append(kind.decode("ascii", errors="replace"))
        seen.add(kind)
        offset = end
    if b"IEND" not in seen:
        raise ValueError("Missing PNG IEND")
    return PNG_SIGNATURE + b"".join(chunks), {**info, "removed_metadata_chunks": removed}


@dataclass
class Pixels:
    width: int
    height: int
    depth: int
    rgba: object


def png_samples(data):
    _, info = png_payload(data)
    cv2, np = pixel_libraries()
    try:
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise ValueError(f"PNG oracle failed: {error}") from error
    depth = info["bit_depth"]
    dtype = np.uint8 if depth == 8 else np.uint16
    if (decoded is None or decoded.dtype != dtype
            or decoded.shape[:2] != (info["height"], info["width"])):
        raise ValueError("PNG oracle failed, changed dimensions or narrowed sample precision")
    channels = 1 if decoded.ndim == 2 else decoded.shape[2]
    if channels not in (1, 3, 4):
        raise ValueError(f"Unexpected PNG oracle channel count: {channels}")
    rgba = np.empty((info["height"], info["width"], 4), dtype=dtype)
    rgba[:, :, 3] = (1 << depth) - 1
    if channels == 1:
        rgba[:, :, :3] = decoded[:, :, None]
    else:
        rgba[:, :, :3] = decoded[:, :, 2::-1]
        if channels == 4:
            rgba[:, :, 3] = decoded[:, :, 3]
    # OpenCV/libpng builds can leave grayscale tRNS unexpanded. Apply the key
    # from the PNG, without changing any color samples (including hidden RGB).
    if "transparency_key" in info:
        transparent = np.all(rgba[:, :, :3] == info["transparency_key"], axis=2)
        rgba[:, :, 3] = np.where(transparent, 0, (1 << depth) - 1)
    return Pixels(info["width"], info["height"], depth, rgba)


def compare_pixels(expected, actual):
    if (expected.width, expected.height, expected.depth) != (actual.width, actual.height, actual.depth):
        raise ValueError("Decoded dimensions or bit depth differ from source")
    _, np = pixel_libraries()
    # Row-wise comparison bounds the mismatch allocation on large inputs.
    for y in range(expected.height):
        mismatch = np.flatnonzero(expected.rgba[y] != actual.rgba[y])
        if mismatch.size:
            x, channel = divmod(int(mismatch[0]), 4)
            raise ValueError(
                f"Pixel mismatch at x={x} y={y} channel={'RGBA'[channel]}: "
                f"source={int(expected.rgba[y, x, channel])} "
                f"decoded={int(actual.rgba[y, x, channel])}; hidden RGB is included"
            )


def clean_environment(parent, overrides=(), *, cpp=False):
    removed = sorted(key for key in parent if key.upper().startswith(("JXL_", "RS_JXL_")))
    environment = {key: value for key, value in parent.items() if key not in removed}
    explicit = {}
    prefix = "JXL_" if cpp else "RS_JXL_"
    for assignment in overrides:
        name, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(prefix + r"[A-Z0-9_]+", name):
            raise ValueError(f"Expected {prefix}NAME=value, got {assignment!r}")
        if "\0" in value or name in explicit:
            raise ValueError(f"Invalid or duplicate environment override: {name}")
        explicit[name] = value
    if cpp:
        if explicit.get("JXL_LOSSLESS_SEARCH_TRACE", "1") != "1":
            raise ValueError("JXL_LOSSLESS_SEARCH_TRACE=1 is required for scheduler evidence")
        explicit["JXL_LOSSLESS_SEARCH_TRACE"] = "1"
    environment.update(explicit)
    return environment, {"removed_inherited_names": removed, "explicit": explicit}


def encoder_command(encoder, binary, source, output, threads, effort, mode, sample_limit=None):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("Explicit sample limit must be positive")
    if encoder == "rust":
        command = [str(binary), "encode", str(source), str(output), "--threads", str(threads),
                   "--strip-metadata", "--source-effort", str(effort)]
        if sample_limit is not None:
            command += ["--source-sample-limit", str(sample_limit)]
        if mode == "full-scheduler":
            command.append("--source-search")
        return command
    if encoder != "cpp":
        raise ValueError(f"Unknown encoder: {encoder}")
    command = [str(binary), str(source), str(output), "--distance=0", f"--effort={effort}",
               f"--num_threads={threads}", "--lossless_jpeg=0", "--keep_invisible=1"]
    if mode == "single-candidate":
        command.append("--iterations=50")
    return command


def decoder_command(binary, jxl, png, depth):
    return [str(binary), str(jxl), str(png), f"--bits_per_sample={depth}",
            "--num_threads=1", "--quiet"]


def run_process(command, environment, timeout):
    start = perf_counter()
    try:
        result = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=timeout, check=False)
        record = dict(returncode=result.returncode, stdout=result.stdout.decode(errors="replace"),
                      stderr=result.stderr.decode(errors="replace"))
    except subprocess.TimeoutExpired as error:
        record = dict(returncode=None, error=f"Process timeout after {timeout}s",
                      stdout=(error.stdout or b"").decode(errors="replace"),
                      stderr=(error.stderr or b"").decode(errors="replace"))
    except OSError as error:
        record = dict(returncode=None, error=str(error), stdout="", stderr="")
    record.update(command=list(command), wall_seconds=perf_counter() - start)
    return record


def check_process(record):
    if record["returncode"] != 0:
        raise RuntimeError(record.get("error") or
                           f"Exit {record['returncode']}: {record['stderr'][-2000:]}")


def scheduler_evidence(encoder, mode, stderr):
    if encoder == "cpp":
        started = "lossless-search event=start " in stderr
        ended = "lossless-search event=end " in stderr
        if mode == "single-candidate":
            if started or ended:
                raise ValueError("C++ search ran in single-candidate mode")
            return "--iterations=50 disables HasLosslessSearchOptions; no search trace observed"
        if not started or not ended:
            raise ValueError("C++ full scheduler has no complete start/end trace; refuse to label it full-scheduler")
        return "lossless-search start/end observed"
    requested_search = mode == "full-scheduler"
    marker = f"search={str(requested_search).lower()} "
    if marker not in stderr or (requested_search and "source search: full_encodes=" not in stderr):
        raise ValueError("Rust source mode diagnostics do not confirm the requested scheduler mode")
    if not requested_search and "source search: full_encodes=" in stderr:
        raise ValueError("Rust search ran in single-candidate mode")
    return "source-search summary observed" if requested_search else "source search=false observed"


def binary_info(path, environment, timeout):
    path = path.resolve(strict=True)
    identity = dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))
    version = run_process([str(path), "--version"], environment, timeout)
    identity["version_probe"] = version
    identity["version"] = (version["stdout"] + version["stderr"]).strip()
    return identity


def cpp_provenance(binary, source):
    """Collect evidence, but never infer binary provenance from current HEAD/mtime."""
    evidence = []
    for parent in (binary.parent, *list(binary.parents)[1:5]):
        for name in ("CMakeCache.txt", "tool_version_git.h", ".ninja_log"):
            path = parent / name
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if name == "CMakeCache.txt":
                lines = [line for line in lines if re.match(
                    r"(CMAKE_HOME_DIRECTORY|CMAKE_BUILD_TYPE|JXL_SOURCE_DIR|LIBJXL_SOURCE_DIR):", line)]
            elif name == ".ninja_log":
                lines = [line for line in lines if "cjxl.exe" in line or "tools/cjxl\t" in line][-8:]
            evidence.append(dict(path=str(path), sha256=file_sha256(path), relevant_lines=lines))
    environment, _ = clean_environment(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    source_info = {"path": str(source)}
    for name, args in (("head", ["rev-parse", "HEAD"]),
                       ("status", ["status", "--porcelain", "--untracked-files=no"])):
        result = run_process(["git", "-C", str(source), *args], environment, 10)
        source_info[name] = result["stdout"].strip() if result["returncode"] == 0 else None
    return dict(status="unverified", expected_commit=CPP_COMMIT, source_checkout=source_info,
                reason="No audited commit-to-binary build attestation. Version/cache/link logs and current HEAD alone do not prove a clean build of the target commit.",
                build_evidence=evidence)


def measure(encoder, binary, decoders, source, expected, directory, threads, effort,
            mode, sample_limit, environment, decoder_environment, timeout):
    output = directory / "encoded.jxl"
    record = {"status": "failed", "decodes": []}
    try:
        command = encoder_command(encoder, binary, source, output, threads, effort, mode, sample_limit)
        process = run_process(command, environment, timeout)
        record["process"] = process
        check_process(process)
        size = output.stat().st_size
        if not size:
            raise ValueError("Encoder produced an empty JXL")
        pixels = expected.width * expected.height
        seconds = process["wall_seconds"]
        record.update(bytes=size, bpp=8 * size / pixels, wall_seconds=seconds,
                      mpix=pixels / 1e6, mpix_per_second=pixels / 1e6 / seconds)
        record["scheduler_evidence"] = scheduler_evidence(encoder, mode, process["stderr"])
        for index, decoder in enumerate(decoders):
            decoded = directory / f"decoded-{index}.png"
            result = run_process(decoder_command(decoder, output, decoded, expected.depth),
                                 decoder_environment, timeout)
            record["decodes"].append(result)
            check_process(result)
            compare_pixels(expected, png_samples(decoded.read_bytes()))
            result["pixels_match_including_hidden_rgb"] = True
        record["status"] = "passed"
    except (OSError, ValueError, RuntimeError) as error:
        record["error"] = str(error)
    return record


def summarize(rows):
    groups = {}
    for row in rows:
        if row["warmup"]:
            continue
        key = (row["input_index"], row["threads"], row["effort"], row["mode"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (index, threads, effort, mode), group in groups.items():
        item = dict(input_index=index, threads=threads, effort=effort, mode=mode)
        rust_repeats = [row["repeat"] for row in group if row["encoder"] == "rust"]
        cpp_repeats = [row["repeat"] for row in group if row["encoder"] == "cpp"]
        complete = bool(rust_repeats) and sorted(rust_repeats) == sorted(cpp_repeats)
        item["status"] = "passed" if complete and all(row["status"] == "passed" for row in group) else "failed"
        if item["status"] == "passed":
            item["mpix"] = group[0]["mpix"]
            for encoder in ("rust", "cpp"):
                selected = [row for row in group if row["encoder"] == encoder]
                seconds = statistics.median(row["wall_seconds"] for row in selected)
                sizes = [row["bytes"] for row in selected]
                item[encoder] = dict(median_bytes=statistics.median(sizes), min_bytes=min(sizes),
                                     max_bytes=max(sizes), median_wall_seconds=seconds,
                                     median_bpp=statistics.median(row["bpp"] for row in selected),
                                     mpix_per_second=selected[0]["mpix"] / seconds)
            item["rust_over_cpp_bytes"] = item["rust"]["median_bytes"] / item["cpp"]["median_bytes"]
            item["cpp_over_rust_wall_speedup"] = item["cpp"]["median_wall_seconds"] / item["rust"]["median_wall_seconds"]
        summaries.append(item)
    return summaries


def markdown_report(report):
    lines = ["# Rust / optimized C++ PNG pixel comparison", "",
             f"Status: **{report['status']}**. C++ provenance: **unverified**.",
             f"Target C++ commit: `{CPP_COMMIT}`.",
             "Pixel matches are not port-completeness or source-equivalence acceptance.", "",
             "Serial CLI wall time includes startup, input/output and diagnostic collection;",
             "independent decoding/comparison is excluded. Warmups are excluded from medians.",
             "Both encoders receive the same metadata-stripped PNG; original samples including hidden RGB are compared.",
             "Full scheduler means the requested entry point ran, not parity of optional policies.",
             "Official decoder origin is caller-attested, not certified from its version string.",
             f"Temporary outputs cleaned: {report.get('temporary_outputs_cleaned', False)}.", "",
             "## Binaries", ""]
    for name, binary in report.get("binaries", {}).items():
        lines.extend([f"- {name}: `{binary['path']}`; SHA256 `{binary['sha256']}`",
                      "```text", binary["version"], "```"])
    lines.extend(["", "## Provenance", "", "```json",
                  json.dumps(report.get("cpp_provenance", {}), indent=2), "```",
                  "", "## Configuration And Inputs", "", "```json",
                  json.dumps({"config": report.get("config", {}), "inputs": report["inputs"]}, indent=2), "```",
                  "", "## Explicit Environments", "", "```json",
                  json.dumps(report.get("environments", {}), indent=2), "```", "",
                  "## Measurements", "",
                  "| Input | Mode | Effort | Threads | Encoder | MPix | Bytes (median) | bpp | Wall s (median) | MPix/s |",
                  "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for item in report.get("summaries", []):
        name = report["inputs"][item["input_index"]]["path"].replace("|", "\\|")
        for encoder in ("rust", "cpp"):
            metrics = item.get(encoder)
            cells = (f"{item['mpix']:.6f} | {metrics['median_bytes']:g} | {metrics['median_bpp']:.6f} | "
                     f"{metrics['median_wall_seconds']:.6f} | {metrics['mpix_per_second']:.3f}"
                     if metrics else "- | FAIL | - | - | -")
            lines.append(f"| {name} | {item['mode']} | {item['effort']} | {item['threads']} | {encoder} | {cells} |")
    lines.extend(["", "## Commands And Failures", ""])
    for row in report.get("rows", []):
        lines.extend([f"- input={row['input_index']} {row['mode']} {row['encoder']} "
                      f"threads={row['threads']} effort={row['effort']} repeat={row['repeat']} "
                      f"warmup={row['warmup']} status={row['status']}",
                      "```json", json.dumps(dict(
                          encoder=row.get("process", {}).get("command", []),
                          decoders=[result["command"] for result in row["decodes"]],
                          scheduler_evidence=row.get("scheduler_evidence")), indent=2), "```"])
        if row.get("error"):
            lines.extend(["```text", row["error"], "```"])
    for error in report.get("errors", []):
        lines.extend(["```text", error, "```"])
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--rust", type=Path, required=True, help="Already-built Rust CLI")
    result.add_argument("--cpp", type=Path, required=True, help="Explicit optimized cjxl; always unverified without audited build attestation")
    result.add_argument("--official-djxl", type=Path, action="append", required=True,
                        help="Independent official decoder, caller-attested provenance; repeatable")
    result.add_argument("--cpp-source", type=Path, default=CPP_SOURCE, help="Read-only source checkout evidence, NOT binary provenance")
    result.add_argument("--mode", choices=(*MODES, "both"), required=True)
    result.add_argument("--efforts", type=int, nargs="+", default=[9])
    result.add_argument("--threads", type=int, nargs="+", default=[1])
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--warmups", type=int, default=0)
    result.add_argument("--source-sample-limit", type=int, default=None,
                        help="Optional positive Rust MA sample cap; omitted uses the CLI default")
    result.add_argument("--timeout", type=float, default=120, help="Per subprocess seconds; no fallback after failure")
    result.add_argument("--cpp-env", action="append", default=[], metavar="JXL_NAME=value")
    result.add_argument("--rust-env", action="append", default=[], metavar="RS_JXL_NAME=value")
    result.add_argument("--temp-root", type=Path, help="Parent for automatically cleaned scratch directory")
    result.add_argument("--json", type=Path, help="Optional NEW JSON report (includes every process log)")
    result.add_argument("--md", type=Path, help="Optional NEW Markdown report")
    result.add_argument("images", type=Path, nargs="+")
    return result


def validate_args(args):
    if any(not 1 <= value <= 256 for value in args.threads):
        raise ValueError("--threads must be in 1..256")
    if any(not 1 <= value <= 10 for value in args.efforts):
        raise ValueError("--efforts must be in 1..10")
    if (args.repeats < 1 or args.warmups < 0
            or (args.source_sample_limit is not None and args.source_sample_limit < 1)):
        raise ValueError("repeats and explicit sample-limit must be positive; warmups must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite and positive")
    for values in (args.threads, args.efforts):
        if len(values) != len(set(values)):
            raise ValueError("Duplicate thread/effort values are not allowed")
    outputs = [path.resolve() for path in (args.json, args.md) if path is not None]
    if len(outputs) != len(set(outputs)) or any(path.exists() for path in outputs):
        raise ValueError("Reports must use distinct NEW paths; refusing to overwrite any existing file")
    if any(not path.parent.is_dir() for path in outputs):
        raise ValueError("Report parent directories must already exist")


def benchmark(args):
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  status="failed", inputs=[], rows=[], errors=[], binaries={},
                  pixel_contract="original samples including hidden RGB, canonical RGBA at source depth",
                  metadata_policy="same metadata-free PNG for both; no metadata acceptance",
                  scheduling_contract="single-candidate and full-scheduler are distinct; no policy parity or port-completeness claim",
                  timing="serial CLI wall seconds; decoding/validation excluded; alternating order",
                  single_candidate_guard="C++ --iterations=50 clears ma_tree_learning_percent_is_default in target source; no search trace allowed. Binary correspondence remains unverified.",
                  official_decoder_provenance="caller-attested, version and SHA256 recorded",
                  config=dict(threads=args.threads, efforts=args.efforts, mode=args.mode,
                              repeats=args.repeats, warmups=args.warmups, timeout=args.timeout,
                              source_sample_limit=args.source_sample_limit))
    scratch = None
    try:
        cv2, np = pixel_libraries()
        report["sample_reader"] = dict(opencv=cv2.__version__, numpy=np.__version__, threads=1)
        cpp_env, cpp_record = clean_environment(os.environ, args.cpp_env, cpp=True)
        rust_env, rust_record = clean_environment(os.environ, args.rust_env)
        decoder_env, decoder_record = clean_environment(os.environ)
        report["environments"] = dict(cpp=cpp_record, rust=rust_record, decoder=decoder_record)
        for name, environment in report["environments"].items():
            print(f"Environment {name}: removed={json.dumps(environment['removed_inherited_names'])} "
                  f"explicit={json.dumps(environment['explicit'], sort_keys=True)}", flush=True)
        executables = dict(rust=args.rust, cpp=args.cpp)
        executables.update({f"official_djxl_{i}": path for i, path in enumerate(args.official_djxl)})
        for name, path in executables.items():
            report["binaries"][name] = binary_info(path, decoder_env, min(args.timeout, 10))
        binaries = {name: Path(info["path"]) for name, info in report["binaries"].items()}
        report["cpp_provenance"] = cpp_provenance(binaries["cpp"], args.cpp_source)
        print(f"C++ commit provenance=unverified; target={CPP_COMMIT}; "
              f"binary={binaries['cpp']}; sha256={report['binaries']['cpp']['sha256']}", flush=True)
        decoders = [binaries[f"official_djxl_{i}"] for i in range(len(args.official_djxl))]
        modes = MODES if args.mode == "both" else (args.mode,)
        with tempfile.TemporaryDirectory(prefix="rs-jxl-cpp-port-", dir=args.temp_root) as temp:
            scratch = Path(temp)
            pair_index = 0
            for input_index, path in enumerate(args.images):
                path = path.resolve(strict=True)
                original = path.read_bytes()
                payload, info = png_payload(original)
                expected = png_samples(original)
                compare_pixels(expected, png_samples(payload))
                report["inputs"].append(dict(path=str(path), bytes=len(original), sha256=sha256(original),
                                             payload_bytes=len(payload), payload_sha256=sha256(payload), **info))
                source = scratch / "input.png"
                source.write_bytes(payload)
                del original, payload
                for mode in modes:
                    for effort in args.efforts:
                        for threads in args.threads:
                            for repeat in range(args.warmups + args.repeats):
                                order = ("rust", "cpp") if pair_index % 2 == 0 else ("cpp", "rust")
                                pair_index += 1
                                for order_index, encoder in enumerate(order):
                                    with tempfile.TemporaryDirectory(prefix="trial-", dir=scratch) as trial:
                                        row = measure(encoder, binaries[encoder], decoders, source, expected,
                                                      Path(trial), threads, effort, mode, args.source_sample_limit,
                                                      rust_env if encoder == "rust" else cpp_env, decoder_env, args.timeout)
                                    row.update(input_index=input_index, encoder=encoder, threads=threads,
                                               effort=effort, mode=mode, repeat=repeat - args.warmups,
                                               warmup=repeat < args.warmups, order_index=order_index)
                                    report["rows"].append(row)
                                    detail = (f"bytes={row['bytes']} bpp={row['bpp']:.6f} "
                                              f"wall={row['wall_seconds']:.6f}s MPix={row['mpix']:.6f} "
                                              f"MPix/s={row['mpix_per_second']:.3f}"
                                              if row["status"] == "passed" else row["error"])
                                    print(f"{path.name} {mode} e{effort} t{threads} {encoder} "
                                          f"warmup={row['warmup']} {row['status']}: {detail}", flush=True)
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        report["errors"].append(str(error) or "Interrupted")
    finally:
        report["temporary_outputs_cleaned"] = scratch is None or not scratch.exists()
        for name, binary in report["binaries"].items():
            try:
                binary["unchanged_during_run"] = file_sha256(Path(binary["path"])) == binary["sha256"]
            except OSError:
                binary["unchanged_during_run"] = False
            if not binary["unchanged_during_run"]:
                report["errors"].append(f"Binary changed during run: {name}; measurements are invalid")
        report["summaries"] = summarize(report["rows"])
        if not report["errors"] and report["rows"] and all(row["status"] == "passed" for row in report["rows"]):
            report["status"] = "passed"
    return report


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        validate_args(args)
        clean_environment(os.environ, args.cpp_env, cpp=True)
        clean_environment(os.environ, args.rust_env)
    except ValueError as error:
        arguments.error(str(error))
    report = benchmark(args)
    try:
        for path, render in ((args.json, lambda: json.dumps(report, indent=2, allow_nan=False) + "\n"),
                             (args.md, lambda: markdown_report(report))):
            if path is not None:
                with path.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(render())
    except OSError as error:
        print(f"Report write failed: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(f"{report['status'].upper()}; C++ commit provenance=unverified; "
          f"temporary outputs cleaned={report['temporary_outputs_cleaned']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
