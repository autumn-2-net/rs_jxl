# Ordinary Pixel e9 Tool Defaults (2026-11-14)

## Correction and Scope

The previous change enabled Source Patch parallel gathering by default, but left
Sparse LZ77, cross-channel MA prev1 and Palette ScreenedRuns opt-in. The intended
change was to enable these three together in ordinary pixel-lossless e9 and then
measure time and size, not only benchmark Source Patch inputs.

The shared native source projection now enables all three at effort 9. This
applies to normal full-frame and streaming/groupwise encoding, the Source Patch
atlas and residual, and source-backed Rust/CLI/Python still, multiframe and layer
encoding. No patch dictionary is required. Existing transform and palette
eligibility rules still apply; pure grayscale has no previous color channel.
The manual low-level Tools API is not an effort preset. Other efforts and JPEG
byte-lossless transcoding keep their existing defaults. Public Python effort
still defaults to 7; callers select effort=9 to use this preset.

This is one native encoding plan. It does not add a portfolio tournament,
baseline-versus-candidate full encoding or a per-image non-growth guarantee.
Aggregate size and speed are evaluated; individual images may get larger.

## Controls

```python
data = rs_jxl.encode(image, effort=9, threads=8)
# Ablate only these three automatic additions:
control = rs_jxl.encode(image, effort=9, threads=8,
                        source_tool_defaults=False)
```

```powershell
rs-jxl encode input.png output.jxl --source-effort 9
rs-jxl encode input.png control.jxl --source-effort 9 --no-source-tool-defaults
```

The Rust opt-out is `base.modular.source_tool_defaults = false`. It is separate
from `source_ma_defaults` (Symmetric, Half, Distribution Offset) and
`source_patch_parallel_gather`. Explicit SourceConfig and PaletteOptions retain
their requested settings; explicit previous-channel counts also take precedence.

`CandidateParams.max_properties` changes from `u8` to `Option<u8>` to distinguish
automatic selection from explicit zero. Rust callers migrate `n` to `Some(n)`;
`None` means the effort default. CLI omission means auto,
`--source-max-properties 0` explicitly disables prev1, and a nonzero count selects
that count subject to available transformed channels. Sampling and training
resolve the same count, including the deferred atlas. The C ABI is unchanged.

The existing `source_tools::Options::build` remains an explicit ablation helper:
it suppresses the new automatic tool bundle before applying individual flags.
It is not used for the OLD/NEW default measurements below.

## Measurement Method

- Baseline: release artifacts from main `b8b8f43`, frozen before edits. These
  already include Source Patch parallel gathering and the earlier MA defaults.
- Candidate: this change, same release profile. Both use native source effort 9
  with automatic patch eligibility, no explicit tool or MA experiment profile.
- Each measurement performs exactly one full encode. Old/new execution order is
  rotated per image/repeat, with no concurrent benchmark jobs or builds.
- Report per-image median native time, summed across the set. It includes native
  projection/pool/encoding, excludes process startup, file IO and verification.
- Decode every output with the existing official libjxl djxl, comparing original
  integer samples including alpha=0 RGB and 16-bit low bits. Compressed JXL byte
  identity is not a correctness target.
- Report both arithmetic mean image bpp and pixel-weighted bpp; MPix/s uses total
  pixels divided by summed native seconds.

`tools/benchmark_source_defaults.py` validates effective settings in each native
receipt. New RGB/GA/RGBA must report Sparse, ScreenedRuns and prev1; pure grayscale
reports prev0. It rejects experiment profiles, partial result summaries and
changed inputs/binaries. Only JSON results persist; compressed and decoded images
are temporary.

## Ordinary e9: Thread Scaling

Eight existing mixed inputs: game screenshot, CLIC photo, DIV2K photo, manga,
16-bit synthetic, Urban100, indexed texture and RGBA icon. Two repeats each,
96 OLD/NEW encodes and official pixel checks, all passed. This is a bounded
integration comparison, not an independent full-corpus quality claim.

At every thread count, total size changes from 7,970,719 to 7,912,266 bytes
(-0.7333%). Mean image bpp changes from 4.971681 to 4.936859; pixel-weighted bpp
changes from 7.604396 to 7.548630.

| Threads | Old seconds | New seconds | Time change | Old MPix/s | New MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.702686 | 22.876182 | +10.50% | 0.405038 | 0.366555 |
| 4 | 6.508548 | 7.256922 | +11.50% | 1.288364 | 1.155501 |
| 16 | 3.490230 | 3.780016 | +8.30% | 2.402530 | 2.218345 |

Enabling the bundle is a size-oriented tradeoff, not an overall speedup. The
timing percentages are local two-repeat measurements, not confidence intervals.
Raw evidence: `../jxl_python/out/source-defaults-20261114/paired8/summary.json`.

## Regression Checks

- Workspace/all-targets compile check and release CLI/FFI/lab build pass.
- 398 scheduler tests pass, including all three default tools at e9 across
  full-frame, streaming, responsive and Patch atlas projection, independent
  opt-out, explicit prev0/prev2, and unchanged other efforts.
- The 144-case source tool integration loop passes Rust decoding for Gray, GA,
  RGB and RGBA, 8/16-bit, serial/multithread and explicit/default policies.
- CLI: 98 passed, 2 existing ignored. FFI: 30 unit tests passed, 1 existing
  ignored; 36 ABI/analysis/error/layer integration tests passed.
- Six Python option/native tests pass, including still, multiframe and layers,
  strict option types and JPEG/nonpixel option rejection.
- Ten benchmark runner tests pass, covering receipt controls, no partial
  aggregates, immutable inputs and detection of hidden-RGB/16-bit changes.

## Ordinary e9: Wider Input Coverage

All entries below use 16 threads and two rotated OLD/NEW repeats. No forced
Source Patch, experiment profile or search tournament is used. The 32-image
existing manifest includes photos, manga, urban scenes, textures, game UI,
indexed/transparent inputs, low-bit grayscale and 16-bit synthetic/texture images.
The five additional inputs are the familiar four document/editor screenshots and
wallhaven-49yekn illustration. The thread-scaling eight are a subset of the 32;
they are not counted twice in the 37-image total.

| Set | Old bytes | New bytes | Size change | Old seconds | New seconds | Time change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed 32 | 25,972,953 | 25,781,211 | -0.7382% | 10.481252 | 11.313676 | +7.94% |
| Screenshot/illustration 5 | 1,603,843 | 1,512,132 | -5.7182% | 3.190005 | 3.228402 | +1.20% |
| Unique total 37 | 27,576,796 | 27,293,343 | -1.0279% | 13.671257 | 14.542078 | +6.37% |

| Set | Old mean bpp | New mean bpp | Old weighted bpp | New weighted bpp | Old MPix/s | New MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed 32 | 5.757505 | 5.693475 | 8.934526 | 8.868568 | 2.218843 | 2.055588 |
| Screenshot/illustration 5 | 1.485990 | 1.392315 | 1.653564 | 1.559010 | 2.432424 | 2.403495 |
| Unique total 37 | 5.180274 | 5.112237 | 7.112990 | 7.039878 | 2.268680 | 2.132825 |

The mixed set has 20 smaller, 8 equal and 4 larger outputs. The increases are
578 B on the 16-bit corner circles (+6.71%), 16 B on the 16-bit diagonal bars
(+7.37%), 93 B on pkw_tiles01b (+0.42%), and 11 B on a 16-bit square-bar fixture
(+8.59%). These are retained, not routed through a second full encode.
All five additional inputs get smaller:

| Input | Old bytes | New bytes | Old seconds | New seconds |
| --- | ---: | ---: | ---: | ---: |
| Paper9bde | 393,022 | 354,791 | 1.160002 | 1.082391 |
| Screenshot7dba | 13,190 | 12,588 | 0.135522 | 0.140362 |
| Editor3dd | 125,823 | 117,483 | 0.375417 | 0.335702 |
| image.png | 379,193 | 341,295 | 0.983778 | 1.139155 |
| wallhaven-49yekn | 692,615 | 685,975 | 0.535286 | 0.530792 |

Raw evidence is under `../jxl_python/out/source-defaults-20261114/`:

- `paired8/summary.json`: 96 encodes (8 images x 3 thread counts x 2 binaries x 2 repeats).
- `broad32/summary.json`: 128 encodes, including ordinary non-Patch cases.
- `screenshots5/summary.json`: 20 encodes through the same ordinary entry.

All 244 ordinary-entry outputs pass official djxl pixel verification. A separate
`patch5-check/summary.json` forces Source Add on the five targets for one repeat
per binary (10 more outputs), also all pixel-exact with official djxl. This checks
the deferred atlas/residual integration but is not included in the ordinary
timing/size tables above. There are 254 successful official decode checks total.
No full corpus run was
started. The image selection comes from prior experiments and is not a fresh
random holdout; these numbers establish integration behavior, not corpus-wide
optimality. The 1.2% screenshot timing difference is small enough to treat as
near parity rather than a precisely established slowdown.

## Reproduction

```powershell
python tools/benchmark_source_defaults.py `
  --baseline ../jxl_python/out/source-defaults-20261114/baseline/portfolio_lab.exe `
  --candidate target/release/examples/portfolio_lab.exe `
  --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe `
  --manifest ../jxl_python/out/ma-kernels-20261114/final-broad32/manifest.json `
  --out ../jxl_python/out/source-defaults-repeat --threads 16 --repeats 2
```

Use a new output directory. The CLI driver is a measurement wrapper around the
shared native default projection; it does not execute the portfolio scout when
called with its direct baseline recipe. The runner forbids explicit experiment
settings in its command and requires the expected effective tools in receipts.
The ordinary CLI and Python projections are also covered by the regression tests.
