# Resource Limit Audit

Date: 2026-10-26. Scope: the independent Rust repository, not the old C++ project.

## Policy

Defaults must not reject a valid input merely because it crosses a private
migration-era count or byte budget. No automatic replacement ceiling, retry at
lower effort, sample truncation, or changed source sampling fraction is used.

Explicit caller limits remain enforced. `ModularOptions::sample_limit` is now
`Option<usize>`: `None` means no extra ceiling, `Some(N)` is a positive hard
limit. Numeric resource fields elsewhere keep their existing types and default
to the type's maximum. These maxima are bounds, never allocation requests.

## Removed Defaults And Hidden Ceilings

| Path | Removed policy |
| --- | --- |
| Pixel MA gathering | 1,048,576 selected channel samples; CLI and library now default to `None` |
| `encoder_lab` | Its private 33,554,432 override; ordinary CLI and laboratory now agree |
| JPEG MA gathering | 4,194,304 selected samples; finite explicit sample limits still apply |
| MA configuration/grouping | 16,384 maximum configurable learner leaves, 131,072 merged leaves, and maximum legacy sample target of 2^20 |
| Pixel/FJXL encoding | Default pixel, output-byte and group-count budgets |
| Modular transformed layout | 8,192 total transformed channels and 2^20 layout streams; per-transform wire fields are distinct and retained |
| Manual transform options | 70,000 palette budget ceiling and forced-palette 65,536 ceiling now use the 70,911 wire field; removed 30-item squeeze request cap, retaining validation of actual emitted transforms |
| CLI input/output | 512 MiB input/output/retained-frame data, 16,384 per-axis and 64-million-pixel restrictions |
| PNG/ICC input | Private 16 MiB ICC and 64 MiB PNG-reader buffer budgets |
| JPEG pixel input | 100-scan ceiling and pixel-decoder size/scan defaults |
| JPEG archival helpers | Default input, coefficient, token, output, record, scan, hint, padding, table-definition and block-work budgets |
| JPEG metadata | Hidden 16,384 reconstruction markers and redundant ICC clamp |
| JPEG coefficient ordering | Private 2^34-block ceiling; ordering now handles wide counts without packed-key overflow |
| Decoding | Default byte, expanded-metadata, pixel, sample and frame-count budgets |
| Decoder ICC | Independent 16 MiB compressed ICC cap; hidden CMS CLUT/TRC ceilings |
| Multiframe/streaming | Default logical frames, trial count/pixels, index entries/index bytes, total stream bytes; absolute 4096 rectangle-option ceiling |
| Scheduler | Default probe/full-encode count, pixel work, output and retained-byte budgets that previously returned errors |
| Group sampling | Private 2^20-cell rejection; Morton ordering covers complete u32 axes |
| Experimental transcripts | Record/value/byte defaults, absolute name-length cap; parsing grows only for actual received values |
| Benchmark/oracle tools | Forced sample-budget overrides and private image-size/byte/count caps; corpus `--max-*` budgets are now opt-in, and caller-specified timeout/worker settings are not clipped to a private smaller range |

These changes do not change configured quality presets, candidate generation,
sampling fractions, predictor search or search stopping criteria.

## Retained Checks

- JXL Level 5/10 dimensions, extra-channel/ICC limits, tree-node and
  geometry-dependent local/global tree bounds. These are required for the
  supported official decoders, not substitutes for the removed sample cap.
- Transform field ranges (for example palette channel count and squeeze
  counts), TOC section widths, reference slots, and JPEG reconstruction fields.
- JPEG's 16-bit dimensions, component/table IDs, marker lengths, and the
  distinct `jbrd` Huffman-entry/padding/tail fields. Removing cumulative table
  definition caps does not remove those wire constraints.
- Integer/index representability, checked size arithmetic, buffer validation,
  fallible allocation, malformed/truncated input and unsupported-mode checks.
- Explicit caller budgets, including tiny limits used by regression tests.
- Cache capacities, concurrency/grain thresholds and bounded search presets
  that fall back safely or control optional work rather than reject legal input.
- Existing supported feature domains and worker-count configuration ranges.
  Removing resource caps is not an implementation of unsupported JPEG modes.
- Test-runner default timeouts and diagnostic-log guards, distinct from codec
  input/output restrictions. Finite positive timeouts remain caller-adjustable.

The ICC helper's source-derived `(u32::MAX >> 2)` arithmetic guard remains;
the supported JXL Level 10 ICC limit is already smaller (2^28 bytes).

## Operational Consequences

Without an explicit budget, large inputs can consume substantially more memory
and time. Arithmetic safety is not protection from decompression bombs or a
process-wide OOM/time guarantee. Applications handling untrusted data should
set finite input, metadata, output, pixel/sample and work budgets appropriate
to their service, and use process isolation where needed. The CLI no longer
silently imposes the old defaults.

Changing the Modular sample field to `Option<usize>` is a Rust API change:
replace `options.modular.sample_limit = n` with `Some(n)` for an intentional cap,
or `None` for the default. JPEG's numeric `sample_limit` keeps its existing type;
`usize::MAX` maps to Modular `None`.

## Verification

Regression coverage includes crossing the former million-sample boundary,
explicit-cap failures, large allocation-free geometry checks, PNG beyond the
old width/ICC limits, a valid 190-scan progressive JPEG, many-marker JPEG
reconstruction, metadata boundaries, incremental decode limits, stream index
budgets, and transcript parsing without declared-count preallocation.

Workspace Rust regression: **2313 passed, 0 failed, 107 ignored**, excluding
duplicate subprocess-helper summaries. The ignored cases remain explicitly
opt-in oracle/benchmark checks, not claimed passes. Command:

```text
cargo test --workspace --release --locked --config "profile.release.lto=false" --config "profile.release.codegen-units=16" --lib --bins --tests -- --test-threads=8
```

The LTO/codegen overrides apply only to the test invocation to reduce test
compilation cost; the repository's normal Release profile is unchanged.
The CLI is rebuilt separately with its normal `cargo build --release` settings.

Python tool regression: 178 unit tests passed with `python -B -m unittest
discover -s tools -p 'test_*.py'`, plus 18 `verify_port_corpus.py --self-test`
checks. The libpng diagnostic during the unit run is from a deliberately
corrupted input fixture; the suite exits successfully.

Independent runtime checks used the rebuilt release CLI and the existing
official libjxl `djxl` on Windows, without any sample-budget override:

| Case | Output bytes | Check |
| --- | ---: | --- |
| DIV2K 0716, 2040x2040 RGB8, source e9 | 4,187,169 | Official decoded samples exactly match |
| Same input, source e10 (previously failed) | 4,117,787 | Official decoded samples exactly match; 7.915777 bpp |
| Same e10 with explicit sample limit 1048576 | No output | Expected sample-budget error |
| Synthetic RGBA8 with nonzero invisible RGB, e10 | 1,096 | Official exact sample comparison including alpha=0 RGB |
| Synthetic RGBA16 with nonzero invisible RGB, e10 | 587 | Official exact 16-bit sample comparison |
| 16385x1 Gray16 PNG, e9 | 1,211 | Official and Rust decoders both reproduce every sample |
| t2/145702319_p0.jpg, JPEG transcode e9 | 934,487 | Official and Rust both reconstruct original JPEG bytes |
| RGB16 with 70,911 distinct tuples, manual global palette budget 70,911 | 416,777 | Official and Rust exact sample comparison |
| RGB8, 64 manual horizontal squeeze requests | 348 | Official and Rust exact sample comparison; no-op requests do not consume transform fields |

Encoded JXL bytes are not compared against an earlier encoding as a correctness
goal. Temporary outputs were removed. Encoding overlapped the test build, so
these runs are correctness checks, not uncontended performance measurements.
The original 67-image report corpus has not been rerun.
