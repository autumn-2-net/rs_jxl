"""Shared JPEG experiment choices and native measurement validation.

Pure report helpers: no codec execution, image loading, or CLI entry points.
"""

import math


EXPERIMENT_FLAGS = ("--cooperative", "--symmetric-ma", "--cfl-peak", "--paired-luma", "--audit-ac")
AC_LZ_FLAGS = tuple(f"--ac-lz={mode}" for mode in (
    "off", "fast", "optimal", "source", "fast-forced", "optimal-forced"))
AC_HISTOGRAM_FLAGS = tuple(f"--ac-histograms={count}" for count in range(1, 5))
BLOCK_CONTEXT_FLAGS = tuple(f"--block-context={mode}" for mode in (
    "source", "channel", "shared-chroma", "paired-luma", "quantile4", "quantile5", "activity4", "activity5"))
AC_ORDER_FLAGS = tuple(f"--ac-order={mode}" for mode in (
    "source", "zigzag", "edge-first", "edge-last", "band-edge-first", "band-edge-last"))
SIZE_BREAKDOWN_FIELDS = (
    "global_bits", "aux_model_bits", "dc_bits", "ac_metadata_bits", "quant_bits",
    "ac_model_bits", "ac_data_bits", "padding_bits", "header_bits",
    "reconstruction_bits", "metadata_bits", "container_bits",
)


def ac_lz_mode(flags):
    modes = {flag.split("=", 1)[1] for flag in flags if flag.startswith("--ac-lz=")}
    if len(modes) > 1:
        raise ValueError("Conflicting --ac-lz arguments")
    return next(iter(modes), "source")


def ac_histograms(flags):
    values = {int(flag.split("=", 1)[1]) for flag in flags if flag.startswith("--ac-histograms=")}
    if len(values) > 1:
        raise ValueError("Conflicting --ac-histograms arguments")
    return next(iter(values), 1)


def validate_experiments(measurement, flags, channels=None):
    modes = representation_modes(flags)
    for key, mode in modes.items():
        prefix = "--" + key.replace("_", "-") + "="
        default = None if any(f.startswith(prefix) for f in flags) else "source"
        if measurement.get(key, default) != mode:
            raise ValueError(f"Experiment receipt mismatch: {key}")
    for key, lo, hi in [("ac_block_contexts", 1, 16), ("dc_thresholds", 0, 15)]:
        if key in measurement and (type(measurement[key]) is not int or not lo <= measurement[key] <= hi):
            raise ValueError(f"Invalid actual {key}")
    if any(f.startswith("--block-context=") for f in flags):
        if not all(k in measurement for k in ("ac_block_contexts", "dc_thresholds")):
            raise ValueError("Missing actual block-context counts")
        policy = modes["block_context"]
        cuts = measurement["dc_thresholds"]
        bins = cuts + 1
        if policy == "channel" and cuts != 0:
            raise ValueError("Channel-only policy unexpectedly has DC thresholds")
        if policy.startswith(("quantile", "activity")) and bins > int(policy[-1]):
            raise ValueError("Too many quantile/activity DC bins")
        expected = None
        if policy == "channel" or policy.startswith(("quantile", "activity")):
            expected = (bins + 1, 3 * bins)
        elif policy == "shared-chroma":
            expected = (bins + 1, bins + (bins + 1) // 2)
        elif policy == "paired-luma":
            expected = ((bins + 1) // 2 + 1, 3 * ((bins + 1) // 2))
        if expected is not None:
            allowed = expected if channels is None else (expected[channels != 1],)
            if measurement["ac_block_contexts"] not in allowed:
                raise ValueError("Actual block contexts do not match the selected policy")
    expected_ac_lz = ac_lz_mode(flags)
    for flag in EXPERIMENT_FLAGS:
        key = flag[2:].replace("-", "_")
        if measurement.get(key, False) is not (flag in flags):
            raise ValueError(f"Experiment receipt mismatch: {key}")
    # Old binaries may omit ac_lz when no override was requested.
    default = None if any(flag in AC_LZ_FLAGS for flag in flags) else "source"
    if measurement.get("ac_lz", default) != expected_ac_lz:
        raise ValueError("Experiment receipt mismatch: ac_lz")
    if expected_ac_lz.endswith("-forced") and measurement.get("ac_lz_enabled") is not True:
        raise ValueError("Forced AC LZ did not enable the wire model")
    if expected_ac_lz == "off" and measurement.get("ac_lz_enabled") is not False:
        raise ValueError("Disabled AC LZ did not disable the wire model")
    requested = ac_histograms(flags)
    default = None if any(flag in AC_HISTOGRAM_FLAGS for flag in flags) else 1
    actual = measurement.get("ac_histograms", default)
    if type(actual) is not int or actual != requested:
        raise ValueError("Experiment receipt mismatch: ac_histograms")
    if "ac_histograms_used" in measurement:
        used = measurement["ac_histograms_used"]
        if type(used) is not int or not 1 <= used <= requested:
            raise ValueError("Invalid actual AC histogram-group count")


def validate_size_breakdown(measurement, size):
    if "size_breakdown" not in measurement:
        return
    breakdown = measurement["size_breakdown"]
    if not isinstance(breakdown, dict) or set(breakdown) != set(SIZE_BREAKDOWN_FIELDS):
        raise ValueError("Invalid size_breakdown fields")
    if any(type(value) is not int or value < 0 for value in breakdown.values()):
        raise ValueError("Invalid size_breakdown bit count")
    if sum(breakdown.values()) != size * 8:
        raise ValueError("size_breakdown does not sum to output bytes * 8")


def representation_modes(flags):
    modes = {}
    for key in ("block_context", "ac_order"):
        prefix = "--" + key.replace("_", "-") + "="
        values = {f[len(prefix):] for f in flags if f.startswith(prefix)}
        if len(values) > 1:
            raise ValueError(f"Conflicting {prefix} arguments")
        if key == "block_context" and values and "--paired-luma" in flags:
            raise ValueError("Conflicting paired-luma and block-context overrides")
        modes[key] = next(iter(values), "source")
    return modes


AUDIT_COUNTS = (
    "streams", "nonempty_streams", "tokens", "raw_contexts", "active_contexts",
    "histograms", "extra_bits", "selector_bits",
)
AUDIT_COSTS = (
    "model_symbol_bits", "cluster_entropy_bits", "raw_context_entropy_bits",
    "extra_bit_entropy_bits", "seconds",
)


def validate_audit(measurement):
    a = measurement.get("ac_audit")
    if measurement.get("audit_ac") is not True or not isinstance(a, dict):
        raise ValueError("Missing requested AC audit")
    if any(type(a.get(k)) is not int or a[k] < 0 for k in AUDIT_COUNTS):
        raise ValueError("Invalid AC audit count")
    if any(type(a.get(k)) not in (int, float) or not math.isfinite(a[k]) or a[k] < 0
           for k in AUDIT_COSTS):
        raise ValueError("Invalid AC audit cost")
    if any(type(a.get(k)) is not bool for k in ("ans", "lz77")):
        raise ValueError("Invalid AC audit coding state")
    if not (0 < a["histograms"] <= 256 and 0 < a["raw_contexts"] and
            a["active_contexts"] <= a["raw_contexts"] and
            a["nonempty_streams"] <= a["streams"]):
        raise ValueError("Inconsistent AC audit dimensions")
    epsilon = max(1e-6, a["model_symbol_bits"] * 1e-10)
    if (a["raw_context_entropy_bits"] > a["cluster_entropy_bits"] + epsilon or
            a["cluster_entropy_bits"] > a["model_symbol_bits"] + epsilon or
            a["extra_bit_entropy_bits"] > a["extra_bits"] + epsilon):
        raise ValueError("Inconsistent AC audit entropy ordering")
    if a["seconds"] > measurement["seconds"] + 1e-6:
        raise ValueError("Audit time exceeds enclosing encode time")
    if a["lz77"] != measurement.get("ac_lz_enabled"):
        raise ValueError("Audit LZ state disagrees with writer")
    wire = measurement["size_breakdown"]["ac_data_bits"]
    if a["extra_bits"] + a["selector_bits"] > wire:
        raise ValueError("Audit extra/selector bits exceed actual AC data")
