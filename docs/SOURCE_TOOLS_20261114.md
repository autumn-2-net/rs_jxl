# Source Tools Integration (2026-11-14)

## Scope

Integrate Source Add's channel-parallel sparse MA gathering after a real patch
plan, then test Sparse LZ77, prev1 cross-channel MA and ScreenedRuns palette
ordering in that order. Keep the existing Symmetric + Half + single-pass
Distribution Offset baseline and the current SIMD work. No full corpus run,
lossy encoder, JPEG byte-transcode changes or compressed-byte identity goal.

## Default Patch Gathering

`EncodeOptions::source_patch_parallel_gather` defaults to true. The native frame
writer resolves the atlas's own policy, then supplies `GatherMode::Channels`,
`sparse=true`, at most `min(threads, 64)` workers and the existing gather memory
budget for each frame without an explicit `source_gather` setting. This happens
only for an actual Source Add plan, with source sampling and `threads > 1`.

- No dictionary: use the original full-frame path without changing gathering.
- Tile/Replace patches and single-thread calls: unchanged.
- Explicit main and atlas gathers: preserved separately, even with opt-out.
- No new pool, image-level race, fallback encode, or output-size tournament.
- Parallel collection can change learning-time deduplication and file size.
  Pixel losslessness, not identical compressed bytes, is the contract.

Opt-out:

```rust
base.source_patch_parallel_gather = false;
```

```python
data = rs_jxl.encode(image, effort=9, threads=8,
                     source_patch_parallel_gather=False)
```

CLI: `--no-source-patch-parallel-gather` with `--source-effort` or manual
`--source-patches`. Python still/multiframe/layers accept the boolean; JPEG
byte-transcode rejects it. Existing default effort remains 7.

## Native Experiments

The other three switches are explicit native compositions, not an expansion of
the automatic portfolio tournament. `scheduler::source_tools::Options::build`
performs one source projection, preserving the current MA and LZ Batch defaults.
It does not encode. Sparse and palette policy apply to main and deferred atlas;
prev1 is projected before sampling so both domains resolve correct properties.

```rust
use rs_jxl::scheduler::{CandidateParams, source_tools};
let params = CandidateParams::default().backend(9)?;
let options = source_tools::Options {
    sparse_lz77: true,
    previous_channel: true,
    screened_runs: true,
}.build(&params, &image, &base)?;
let bytes = rs_jxl::encode(&image, &options)?;
```

This experiment API requires e9. It leaves JPEG and lower-effort policy alone.
An enabled switch explicitly chooses that tool; disabled switches preserve the
underlying projected choice. No private JXL decoder features are introduced.
For a forced Source Add experiment set `params.patches = Override::On` before
building. The ordinary recipe keeps automatic patch policy, rather than forcing
patch detection on every image.

## Measurement

`portfolio_lab --direct-candidate baseline --direct-source-tools MODE` encodes
exactly once. `--direct-candidate source` separately forces the source route for
targeted patch tests; it is not silently used for the ordinary corpus.

| Mode | Patch gather | Sparse seed | Prev1 | ScreenedRuns |
| --- | --- | --- | --- | --- |
| baseline | off | off | off | off |
| patch | on after detection | off | off | off |
| sparse | on after detection | on | off | off |
| prev1 | on after detection | on | on | off |
| screened | on after detection | on | on | on |
| sparse-only | off | on | off | off |
| prev1-only | off | off | on | off |
| screened-only | off | off | off | on |

```powershell
cargo build --release --offline -p rs-jxl-cli --example portfolio_lab
python -B tools/benchmark_source_tools.py --encoder target/release/examples/portfolio_lab.exe --decoder path/to/official/djxl.exe --manifest cases.json --out fresh-results --threads 1 4 16 --repeats 2
```

To try the combination on one file without a candidate tournament:

```powershell
target/release/examples/portfolio_lab.exe input.png --output output.jxl --effort 9 --threads 8 --direct-candidate baseline --direct-source-tools screened
```

Use `--direct-candidate source` only to explicitly enable the Source Add route.
These are lab controls; the regular CLI/Python API gains the patch-gather default
and its opt-out, not a new default Sparse/prev1/ScreenedRuns policy.

The runner serializes codec jobs and rotates mode order. It reports sums of
per-image median native times, MPix/s, mean image bpp and pixel-weighted bpp;
timings exclude image I/O, process startup and decoder verification. Each
temporary output is decoded by official djxl and compared with original samples,
including hidden RGB and 16-bit values. Failed/missing trials remain failures,
never a smaller successful-only corpus. Outputs are removed; JSON evidence
remains. No dataset starts automatically.

## Results

Current e9 baseline includes the existing SIMD work, Symmetric learning, Half
description scoring and single-pass Distribution Offset. Both sides use the
same Release executable; the baseline below disables only the new patch gather
default. No official encoder baseline was rerun. Official djxl is the decoder
oracle, not a competing encoder in this experiment.

Native times exclude CLI startup, PNG I/O and verification, but include policy
projection, pool setup and complete JXL encoding. Different runs are not pooled:
8-image and patch results use two rotated repeats and per-image median times;
the broader check is one pass. Small timing differences are not significant.

### Eight Mixed Images

The same 8-image mixed subset as the previous SIMD check, at 1/4/16 threads.
Mean bpp is the arithmetic average of per-image bpp, not weighted by pixels.

| Cumulative stage | Total bytes | Mean bpp | 1T seconds | 4T seconds | 16T seconds | 16T MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 7,970,719 | 4.971681 | 20.8249 | 6.6841 | 3.4210 | 2.451 |
| Patch gather | 7,970,719 | 4.971681 | 22.0292 | 6.5561 | 3.4065 | 2.462 |
| + Sparse | 7,970,812 | 4.973100 | 19.2860 | 5.9855 | 3.3024 | 2.539 |
| + Prev1 | 7,912,523 | 4.937176 | 22.8114 | 7.0051 | 3.6685 | 2.286 |
| + ScreenedRuns | 7,912,266 | 4.936859 | 22.4919 | 7.0696 | 3.6817 | 2.278 |

Sparse adds 93 bytes (+0.0012% total, all on `pkw_tiles01b.png`, +0.4192%
for that image), and measures 7.4%/10.5%/3.5% faster than baseline. The complete
combination saves 0.7333% total bytes but costs 8.0%/5.8%/7.6% more time.
These stages are cumulative; their totals are not isolated per-tool effects.
Patch gather at 1T is a no-op: its +5.8% time is an observable noise/control
spread, not evidence that parallel sampling slows the single-thread encoder.

### Five Source Add Targets

Four screenshots (paper9bde, screenshot7dba, editor3dd, image.png) and
wallhaven-49yekn, using the explicit Source Add recipe. The latter also checks
fallback behavior. This targeted group is not representative of all photos.

| Stage | Total bytes | Mean bpp | 4T seconds | 16T seconds | 16T MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original gather | 1,380,800 | 1.250585 | 13.7205 | 11.6421 | 0.666 |
| Default parallel gather | 1,380,800 | 1.250585 | 12.6008 | 10.2606 | 0.756 |
| + Sparse + Prev1 + ScreenedRuns | 1,305,606 | 1.174605 | 12.5703 | 10.5718 | 0.734 |

Parallel gather saves 8.16%/11.87% time with unchanged sizes in this sample.
The optional combination saves another 5.4457% bytes; against the new parallel
gather default it is effectively tied at 4T and 3.03% slower at 16T.
All five files shrink with the combination, but attribution requires separate
ablations; the 17.15% reduction on editor3dd is not credited to just one tool.

| Source Add input | Baseline/parallel bytes | Combination bytes |
| --- | ---: | ---: |
| paper9bde | 262,565 | 236,214 |
| screenshot7dba | 9,281 | 7,709 |
| editor3dd | 102,440 | 84,876 |
| image.png | 329,634 | 304,609 |
| wallhaven-49yekn | 676,880 | 672,198 |

### Thirty-Two Mixed Images

16 threads, one rotated pass, using the previous stratified 32-image manifest.
It includes the eight paired images, photos, manga, city textures, icons,
low-bit-depth gray, 16-bit RGB and previously adverse synthetic cases. The
first three experimental rows isolate one tool with patch gather disabled;
the final row is the complete cumulative combination. No images were dropped.

| Profile | Total bytes | Mean bpp | Seconds | MPix/s | Size change | Time change | Smaller / equal / larger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 25,972,953 | 5.757505 | 10.4193 | 2.232 | - | - | - |
| Sparse only | 25,973,034 | 5.757843 | 9.8074 | 2.371 | +0.0003% | -5.87% | 1 / 30 / 1 |
| Prev1 only | 25,781,876 | 5.714068 | 11.7348 | 1.982 | -0.7357% | +12.63% | 19 / 12 / 1 |
| ScreenedRuns only | 25,972,332 | 5.736756 | 10.4552 | 2.224 | -0.0024% | +0.34% | 3 / 27 / 2 |
| All four | 25,781,211 | 5.693475 | 11.2492 | 2.067 | -0.7382% | +7.97% | 20 / 8 / 4 |

Pixel-weighted bpp is respectively 8.934526, 8.934554, 8.868797, 8.934312
and 8.868568. Small images contribute equally to the mean but very little to
total bytes; in particular ScreenedRuns changes mean bpp more than total bytes.
Its +0.34% time is within measurement noise, not an established slowdown.

Adverse cases remain visible:

- Sparse: `pkw_tiles01b.png` grows 93 B (+0.4192%); the 16-bit circles input
  saves 12 B, giving the aggregate +81 B.
- Prev1: `img_480x360_3x16bit_RGB_circles_progressive_a_corner_tl.png` grows
  1,023 B (+11.8705%), although the other 19 changed files shrink.
- ScreenedRuns: two very small 16-bit bars files grow 11 B (+8.5938%) and
  16 B (+7.3733%). Three inputs save 648 B combined, net saving 621 B.
- All four: 20 shrink, 8 unchanged, 4 grow. The circles regression is reduced
  to 578 B (+6.7069%), not eliminated. There is no per-image size safeguard.

## Decision

- Enable Source Patch parallel gathering by default with the narrow native
  actual-plan/multithread/sampling gates. Keep CLI/Python opt-out and explicit
  gather overrides. These tests show useful throughput improvement without
  measured size growth, not a universal performance or byte-identity guarantee.
- Keep Sparse, prev1 and ScreenedRuns optional in `scheduler::source_tools`
  and the direct lab controls. Do not silently change all ordinary e9 encodes
  or add a multi-encode fallback tournament.
- Sparse is the strongest speed candidate here. Prev1 trades more MA work for
  smaller files, particularly on Source Add residuals. ScreenedRuns remains
  usable but has only a tiny aggregate gain on this 32-image sample.
- The Source Add combination merits use as an explicit size-oriented recipe:
  -5.45% bytes for roughly +3% time versus the new default at 16 threads in
  the five-target group. This is not evidence of the same tradeoff everywhere.

## Validation and Evidence

- 37 distinct original images; 460 complete JXL outputs decoded by official
  djxl, all exactly equal to the original integer pixels. This includes 8/16-bit
  sources and hidden RGB under zero alpha. No timeout, failed case or pixel
  mismatch was excluded. Temporary encoded/decoded images were removed.
- 27 focused Rust tests passed, including one integration test covering 128
  layout/depth/thread/tool combinations. Source Patch tests also cover actual
  atlas/residual activation, supplied pools, explicit per-frame overrides,
  no-plan fallback and opt-out.
- 19 focused Python tests passed: six binding/option tests and thirteen benchmark
  harness tests. Still/multiframe/layer encoding, null/type/operation validation
  and JPEG option rejection are included.
- Release workspace build, focused rustfmt check and `git diff --check` passed.
  This is not a full workspace or full-corpus regression claim. No WSL or macOS
  run was performed; existing unrelated SIMD work remains in the working tree.

Local evidence under `../jxl_python/out/source-tools-20261114/`:

| Directory | Work |
| --- | --- |
| `paired8` | 8 images x 5 stages x 3 thread counts x 2 repeats = 240 encodes |
| `patch5` | 5 Source Add targets x 3 stages x 2 thread counts x 2 repeats = 60 encodes |
| `broad32` | 32 images x 5 profiles x 1 repeat, all at 16 threads = 160 encodes |

Each directory contains `summary.json` and `trials.jsonl`, including per-image
sizes/times, original-input and binary hashes, mode receipts, and decoder logs.
The tested lab executable SHA256 is
`a1c3260cb9c8d7f289a3538e6743156cc96a588741e4515651ce2ea43fae5832`.
The official decoder SHA256 is
`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.
These hashes identify the executables; compressed JXL hash equality is neither
a correctness requirement nor the optimization target.
