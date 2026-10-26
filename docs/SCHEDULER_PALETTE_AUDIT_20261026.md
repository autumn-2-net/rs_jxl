# Scheduler Palette Audit: 2026-10-26

## Status and Scope

**Source-palette alone restores 13 of the 17 historical size regressions exactly
by byte count. Four remain; this is not an all-fixed result.** All 68 matrix
outputs and all 12 fixed-candidate outputs passed official `djxl` decoding and
exact original-pixel comparison, including invisible RGB. Equal sizes do not
imply identical JXL bytes.

**Both same-binary 500-input runs have completed: 496 passed and four unsupported
inputs per run, with no pixel/reconstruction verification failures.** The four
unsupported inputs are non-full-range Netpbm samples in the input adapter; they
are not successful codec tests. Global source-palette selection does not solve
the regressions: 17 PNGs shrink but 40 grow compared with the Morton policy.

Both `PilotOptions::source_palette` and
`PilotOptions::family_tournament` remain experimental, independent opt-ins,
**not enabled by default**.

## Evidence and Commands

Paths are relative to the repository root:

- Historical pairs: `out/encoder-random500-20261026/analysis/comparison.json`,
  specifically its 17 positive `bytes_delta` entries against
  `out/encoder-corpus-20261025-v23`.
- Four-policy matrix: `out/scheduler-repair-20261026/matrix.json`;
  results: `out/scheduler-repair-20261026/regressions/report.json`
  (17 inputs x 4 policies = 68 outputs).
- Direct-candidate matrix: `out/scheduler-repair-20261026/fixed.json`;
  results: `out/scheduler-repair-20261026/remaining-fixed/report.json`
  (4 inputs x 3 candidates = 12 outputs).
- Encoder: `out/scheduler-repair-20261026/encoder.exe`,
  SHA256 `640c8e05007e8f40a1d7c1c2de0aaa99523c98cec5cf3cd59e4a1333f19d4945`.
- Official decoder: `D:/cpp_pj/jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe`,
  SHA256 `c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.

All 17 original and prepared-payload SHA256 values match the archived random-500
records. The new matrix's `current` sizes reproduce all 17 archived current
sizes. Both reports use effort 9, 16 threads, one measured pass, and no warmups.
They retain the actual encoder and decoder argv in each row.

The common search flags in `matrix.json` are `--search --content-pilot
--fast-ranges --subtrees --ma-workers 16 --concat-batches
--pilot-parallel-initial --pilot-overlap-automatic --pilot-near-tie-consensus`.
`source-palette` adds `--pilot-source-palette`; `family-source` adds both
that flag and `--pilot-family-tournament`; `family-morton` adds only the latter.
`fixed.json` selects `cross6/routed`, `cross6/source`, and `automatic/routed`,
without search. None of these matrices enables progressive groups.

Reproduction commands below reuse each report's exact input list and write to
new directories; they were not run for this document:

```powershell
$root = "out/scheduler-repair-20261026"
$djxl = "D:/cpp_pj/jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe"
foreach ($suite in @("regressions", "remaining-fixed")) {
    $saved = Get-Content "$root/$suite/report.json" -Raw | ConvertFrom-Json
    $caseArgs = @($saved.cases | ForEach-Object { "--case"; $_.path })
    $matrix = if ($suite -eq "regressions") { "matrix.json" } else { "fixed.json" }
    python tools/benchmark_encoder_tools.py --encoder "$root/encoder.exe" `
        --djxl $djxl --threads 16 --repeats 1 --warmups 0 `
        --matrix "$root/$matrix" --out "$root/$suite-rerun" @caseArgs
}
```

## Seventeen Historical Size Regressions

Sizes are bytes. Basenames below uniquely identify these cases; full source paths
and hashes are retained in the reports.

| Input | Old v23 | Current | Source-palette | Residual vs v23 |
| --- | ---: | ---: | ---: | ---: |
| `img_480x360_3x16bit_RGB_bars_vertical_square_0032_1xhalfPI.png` | 94 | 95 | 94 | 0 |
| `pk02_floor11_S.png` | 51233 | 51930 | 51233 | 0 |
| `pkf_wall1n.png` | 49649 | 50180 | 49649 | 0 |
| `pkf_wall1i.png` | 51228 | 51292 | 51228 | 0 |
| `mod_labcab2a.png` | 4206 | 4231 | 4206 | 0 |
| `pk02_wall08_N.png` | 195809 | 195825 | 195825 | 16 |
| `status-weather-overcast.png` | 22182 | 22353 | 22353 | 171 |
| `football_player_PNG54.png` | 149269 | 164626 | 149269 | 0 |
| `pkf_wall5f.png` | 42636 | 43065 | 42636 | 0 |
| `img_480x360_3x16bit_RGB_bars_vertical_square_0064_0xhalfPI.png` | 93 | 96 | 93 | 0 |
| `places-network-server.png` | 6012 | 6015 | 6015 | 3 |
| `pk_chcobb2c.png` | 54424 | 54426 | 54424 | 0 |
| `apps-accessories-text-editor.png` | 19980 | 20084 | 20084 | 104 |
| `img_480x360_3x16bit_RGB_grid_rectangle_0016x0012_1.png` | 190 | 191 | 190 | 0 |
| `pke_tuwall01b.png` | 56094 | 56201 | 56094 | 0 |
| `mod_labcon1a.png` | 23001 | 23491 | 23001 | 0 |
| `mod_metfloor1b.png` | 57977 | 60160 | 57977 | 0 |
| **Total** | **784077** | **804261** | **784371** | **294** |

Source-palette removes 19890 of the original 20184 excess bytes. The 13 zero
residuals above are exact size restorations, not merely improvements.

## Remaining Four and Family Results

| Input | Old v23 | Source-palette | Direct cross6 routed / source | Interpretation |
| --- | ---: | ---: | ---: | --- |
| status-weather-overcast.png | 22182 | 22353 | 22182 / 22182 | The old size is reachable; pilot selection still misses it. |
| apps-accessories-text-editor.png | 19980 | 20084 | 19966 / 19978 | Both direct candidates beat v23; routed saves 14 bytes versus v23. |
| places-network-server.png | 6012 | 6015 | 6015 / 6015 | Residual backend-size difference of 3 bytes persists. |
| pk02_wall08_N.png | 195809 | 195825 | 282685 / 282652 | Residual backend-size difference of 16 bytes persists; cross6 is not the winning representation. |

The last two remain backend investigation items, not established palette fixes;
these measurements do not identify their exact lower-level cause.

| Matrix policy | Total bytes | Delta vs v23 | Equal / smaller / larger vs v23 | Total call seconds |
| --- | ---: | ---: | ---: | ---: |
| `current` | 804261 | 20184 | 0/0/17 | 12.935 |
| `source-palette` | 784371 | 294 | 13/0/4 | 13.923 |
| `family-source` | 784041 | -36 | 9/3/5 | 13.863 |
| `family-morton` | 778852 | -5225 | 0/4/13 | 13.708 |

Family results are mixed. `family-source` is 36 bytes smaller in aggregate but
still has five inputs larger than v23, including weather at 22781 bytes and
pkf_wall1i at 51229 (+1). `family-morton` improves football-player to 141180
and weather to 21425 bytes, but 13 inputs remain larger than v23. Neither
aggregate establishes an across-the-board repair or justifies a default change.

The call totals are rough one-pass observations, not statistically established
speedups or regressions. Calls include the requested search/pilot, exclude CLI
startup and input/output IO; process wall excludes independent decoding.
One repeat and fixed first-round variant order do not control noise or order
effects. Historical v23 timings are not a controlled performance baseline.

## Owned Code Paths and Representation Caveat

- `crates/rs-jxl/src/scheduler/pilot.rs` maps the independent opt-ins.
  Source-palette disables `literal_palette_tools`, not probe-only palette
  forcing or explicit caller-supplied base palette policies.
- `crates/rs-jxl/src/scheduler/native.rs` applies the literal override to
  eligible Zero-predictor candidates: `force_palette=true`,
  `FrequencyBandsMorton` order, and flat mapping.
  `scheduler/adapter.rs` projects each candidate's source palette policy and
  preserves applicable explicit base overrides.
- `crates/rs-jxl/src/modular/palette.rs` makes `force_palette` bypass the
  derived multichannel color cap and palette cost rejection. It still uses the
  explicit palette-color limit and validation. This is more than an ordering
  change: palette admission and the resulting representation can change.
- `crates/rs-jxl/src/scheduler/scout_pilot.rs` owns family ranking and
  confirmation: retain the best original Variable-RCT anchor and the overall
  Variable-RCT winner without removing natural/literal families; suppress wide
  witnesses for a natural leader without competitive Zero; confirm a learned
  literal on 8-bit input with at least 50% sampled flatness.
  `compact_pilot.rs` and `finalists.rs` own the small-image tournament and
  finalist dispatch.
- `crates/rs-jxl-cli/examples/encoder_lab.rs` and
  `tools/benchmark_encoder_corpus.py` expose the flags;
  `tools/benchmark_encoder_tools.py` runs the reported matrices and official
  pixel comparisons.

A `CandidateParams` label is not a complete representation identity.
Probe and full trials, explicit base options, and source-palette versus forced
Morton policy can produce different accepted transforms for the same parameters.
Do not treat equal family labels, RCTs, or learning fractions as identical work.
`native.rs` can allocate a fresh representation receipt for each trial;
`modular/receipt.rs` describes accepted transforms for that invocation, not
palette-table/index identity or a replay recipe. These reports verify pixels and
sizes; they do not prove per-candidate representation equivalence.

## Same-Binary Random-500 Comparison

`out/scheduler-repair-20261026/morton-palette-500` and `source-palette-500`
use the same frozen encoder listed above, effort 9, 16 threads, and
`--pilot-progressive-groups`. The only policy difference is
`--pilot-source-palette`. Original and prepared input hashes match for every
paired PNG, and both outputs passed exact pixel checks with official `djxl`.
The 182 JPEGs were verified through the same frozen JPEG encoder, but are not
labels for this pixel/palette experiment. No JPEG-to-pixel re-encoding is used.

The table below covers all 314 paired PNGs, not the JPEG inputs:

| Policy | Bytes | bpp | Encoder-call seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Morton literal tools | 237432642 | 6.454430 | 373.822 | 0.787242 |
| Source palette | 237466828 | 6.455359 | 371.288 | 0.792614 |
| Per-image two-policy size oracle | 237412692 | 6.453888 | Not a runtime policy | - |

Source-palette is smaller on 17 images, equal on 257, and larger on 40.
The total grows only 0.0144%, but hides substantial individual regressions,
including `hlog` (6039 -> 9901 bytes) and a Trek73 screenshot (991 -> 1584).
Switching the entire corpus to the old palette policy is therefore rejected.

Even perfect selection between these two policies would save only 19950 bytes
(0.0084%) versus fixed Morton. This small oracle headroom bounds this particular
experiment; it says nothing about the headroom of a broader representation/MA/
LZ77 candidate portfolio. The policies still run their existing internal pilot
search. Selecting a policy is not equivalent to predicting a complete final plan.

Timing is one-pass observation, not a statistically established difference.
Compared with the earlier 394.399-second run, the new 373.822-second Morton run
also includes incumbent-first witness scheduling; that historical difference
must not be attributed to palette selection alone.
