# Bounded Palette MinLA Experiment

## Scope And Sources

This is a native Rust, default-off ordering experiment. It changes the explicit
absolute palette permutation and uses the existing exact remapping and standard
JXL encoder. It does not add WebP, PNG or LZP coding syntax, change predictors,
deduplicate transparent colors, or require a modified decoder.

Primary references inspected:

- [libwebp palette.c](https://github.com/webmproject/libwebp/blob/main/src/utils/palette.c):
  `PaletteMinLARefine`, `MinLABestSlot`, and spatial co-occurrence construction.
- [Oxipng palette.rs](https://github.com/oxipng/oxipng/blob/master/src/reduction/palette.rs):
  EZeng chooses insertion slots accounting for both the new vertex and existing
  edges crossing the slot. This is different from placing only at either end.

The new Rust graph math is an independent implementation of weighted linear
arrangement identities, not copied C code or an external codec dependency.
It uses incremental vertex balances to avoid rebuilding every cut after a move.
It does not reproduce Oxipng's PNG-specific edge-color policy or WebP's exact
iteration order. No improvement percentage from either codec is assumed here.

## Implementation

`modular/palette_minla.rs` contains the pure graph algorithms and separate
brute-force tests. `palette_graph.rs` adapts the existing full-tuple map and
spatial sampling. `palette_order.rs` only dispatches and applies permutations.

For symmetric edge weights, the objective is:

```text
C(order) = sum over a < b of weight[a,b] * abs(position[a] - position[b])
balance[v] = weight from v to its right - weight from v to its left
delta(swap adjacent a,b) = balance[b] - balance[a] + 2*weight[a,b]
```

Chaining those swap deltas evaluates every reinsertion slot in O(n). An accepted
move updates balances only for crossed vertices. A refinement sweep is O(n^2),
including position updates, and accepts only strictly decreasing graph cost.
Equal-cost moves retain the current position. The seed grows by strongest
connection to placed vertices and full-cost insertion; it is also O(n^2).
Intermediate arithmetic uses i128; sampled final graph costs fit in u64.

Controls:

- `Adjacency`: existing frozen-source graph seed, unchanged.
- `AdjacencyMinLa { rounds }`: that seed followed by bounded refinement.
- `EzengMinLa { rounds: 0 }`: the insertion-based seed alone.
- `EzengMinLa { rounds }`: insertion seed followed by refinement.

The development `palette_lab` exposes these as `--mode adjacency`,
`adjacency-minla`, `ezeng`, and `ezeng-minla`. `--minla-rounds` accepts 0..40,
defaults to 4 for refinement modes, and is rejected for unrelated modes.
Existing source numeric modes stay 0..9; C/Python modes and production defaults
are unchanged. This is not a new automatic scheduler candidate.

## Boundaries

- Only 2..256 distinct explicit tuples with 3 or 4 channels enter graph math.
  This is an experimental work bound, not a JXL color limit. Larger tables use
  existing adjacency/default ordering rather than rejecting the image.
- Repeated promoted implicit tuples bypass the new graph path. Every original
  slot and the last-alias-wins remapping remain intact.
- Unpromoted implicit colors are not graph vertices. Alpha and invisible RGB
  participate in complete tuple identity. There is no alpha-zero normalization.
- Single/dual-channel tables, unordered palettes and effective geometry settings
  keep their existing precedence. Successful exact-delta palettes bypass the
  absolute reorderer; zero-delta insertion remains after absolute reordering.
- The graph uses the same right/down, bounded-stride sampling as existing
  adjacency: at most 262,144 anchor pixels. This isolates refinement from a
  sampler change but retains that sampler's possible periodic bias.
- The graph does not include palette-entry coding cost, actual MA choices,
  LZ77 decisions or ANS/table headers. Lower graph cost is not a guarantee of
  smaller complete JXL. No baseline re-encode is hidden behind this experiment.

## Measurement

`out/palette-minla-20261110/cases36.json` is a pre-encoding coverage selection:
all 31 cases in the earlier 200-image baseline report with an accepted global
3/4-channel tuple palette of 2..256 colors and no deltas, plus five high-color
or alternate-route controls. Selection does not use compression improvements.
This is deliberately enriched for relevant palettes, not a random population.

Compare default, runs, adjacency, adjacency-minla, ezeng and ezeng-minla using
one complete native fixed-plan encode per trial at e9 / eight threads. Record
actual file bytes, pixel-weighted bpp, native time and MPix/s, preserving all
regressions. Native time excludes startup, PNG input decode and output I/O.
The main scheduler and patches are not exercised by this lab.

```powershell
cargo build --release -p rs-jxl-cli --example palette_lab
python -B tools/benchmark_palette_runs.py --encoder target/release/examples/palette_lab.exe --decoder path/to/official/djxl.exe --manifest out/palette-minla-20261110/cases36.json --modes default runs adjacency adjacency-minla ezeng ezeng-minla --effort 9 --threads 8 --repeats 3 --out out/palette-minla-20261110/repeat-e9-t8.json
```

The existing harness validates original samples through official `djxl`,
including packed PNG inputs, hidden RGB and 16-bit values. Generated images
are temporary. A configured order or accepted palette count alone does not
prove graph execution; a separate `--profile` run records graph build/seed/
refinement work and `palette.graph.result` with moves and before/after costs.
Attempted palettes may later be rejected by ordinary palette admission.

## Results: 2026-11-10

Release build on Windows, e9, eight encoder threads, three rotated-order trials
per image/order, no profiler in the main timing run. The six orders cover 36
images / 14.504616 MPix. Every one of the 648 outputs passed original-sample
comparison through libjxl `djxl v0.12.0 a1ebdf2`, including alpha and hidden RGB.
There were no encode/decode failures. The two separate profile audits add 40
verified outputs, for 688 official-decoder checks in total.

Times below sum per-image medians for one corpus pass; they are not the wall
time of all three repeats or the process harness. bpp is pixel-weighted.
No minimum-of-six result is presented as an automatically achieved result.

### All 36 Images

| Order | Bytes | bpp | Native seconds | MPix/s | Size change | Time change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 9212731 | 5.081268 | 10.3671 | 1.3991 | +0.000% | +0.000% |
| RunFrequency | 9464753 | 5.220271 | 12.3983 | 1.1699 | +2.736% | +19.593% |
| Adjacency | 9306053 | 5.132740 | 12.1250 | 1.1963 | +1.013% | +16.956% |
| Adjacency + MinLA(4) | 9199774 | 5.074122 | 11.6482 | 1.2452 | -0.141% | +12.357% |
| EZeng | 9213804 | 5.081860 | 11.2411 | 1.2903 | +0.012% | +8.430% |
| EZeng + MinLA(4) | 9178007 | 5.062117 | 10.8915 | 1.3317 | -0.377% | +5.058% |

For EZeng + MinLA, 17 files became smaller, five stayed the same size, and 14
became larger. The large controls dominate pooled bytes, so this table must
not substitute for the targeted palette comparison below. Three controls
(PDF screenshot, DIV2K 0049, RGBA16 floor texture) have unchanged file sizes
in all modes. The editor screenshot and illustration use local palettes and
are not bypass controls.

### The 31 Prior Global-Palette Cases

These account for 5.022530 MPix. This subset was fixed from previous palette
receipts before running the experiment; it is not selected by the new gains.

| Order | Bytes | bpp | Native seconds | MPix/s | Size change | Time change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 658891 | 1.049497 | 3.9823 | 1.2612 | +0.000% | +0.000% |
| RunFrequency | 910395 | 1.450098 | 5.9359 | 0.8461 | +38.171% | +49.057% |
| Adjacency | 744778 | 1.186299 | 5.9341 | 0.8464 | +13.035% | +49.011% |
| Adjacency + MinLA(4) | 648512 | 1.032965 | 5.2791 | 0.9514 | -1.575% | +32.563% |
| EZeng | 665253 | 1.059630 | 5.0864 | 0.9874 | +0.966% | +27.724% |
| EZeng + MinLA(4) | 630196 | 1.003791 | 4.6373 | 1.0831 | -4.355% | +16.448% |

EZeng + MinLA: 16 smaller / two equal / 13 larger against Default.
Adjacency + MinLA: 14 smaller / three equal / 14 larger against Default.

Refinement itself is useful relative to its seed in aggregate:

- Adjacency + MinLA versus Adjacency: -12.925% bytes and -11.038% native time;
  17 smaller, seven equal, seven larger.
- EZeng + MinLA versus EZeng: -5.270% bytes and -8.828% native time;
  16 smaller, ten equal, five larger.

Those comparisons do not imply a win over Default. In particular,
`head.png`, `face.png`, and `baboon.png` contribute 29,322 saved bytes,
more than the subset's net 28,695-byte saving. The remaining 28 cases together
grow from 396,434 to 397,061 bytes (+0.158%). Do not generalize the -4.355%
subset result into an all-screenshot or all-image policy.

Selected complete-encode results:

| Image | Default bytes -> EZeng+MinLA bytes | bpp before -> after | Median seconds before -> after |
| --- | ---: | ---: | ---: |
| head.png | 36920 -> 32503 | 3.767347 -> 3.316633 | 0.1256 -> 0.1201 |
| face.png | 35767 -> 31663 | 3.756249 -> 3.325247 | 0.1227 -> 0.1189 |
| baboon.png | 189770 -> 168969 | 6.325667 -> 5.632300 | 0.3709 -> 0.3805 |
| editor screenshot 3dd595... | 126447 -> 116640 | 0.473243 -> 0.436539 | 0.4349 -> 0.4260 |
| window_03.png (regression) | 2837 -> 3093 | 0.086578 -> 0.094391 | 0.1038 -> 0.1361 |

The editor screenshot's best tested order is actually Adjacency + MinLA:
116,276 bytes / 0.435176 bpp / 0.4291 s, versus Default's 126,447 bytes.
This is an 8.044% size reduction. Small observed timing differences, especially
on unchanged routes, should not be treated as established speedups: even the
unchanged large controls show run-to-run timing variation. The stronger
evidence is reproducible byte counts plus exact decoded pixels.

### Per-Image Sizes

All measured cases are retained below, including every size regression.
The JSON report additionally retains all timing repetitions, receipts and
verification results. Columns are complete JXL bytes, not palette proxy cost.

| Image | Default | Runs | Adjacency | Adjacency+MinLA | EZeng | EZeng+MinLA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| img_480x360_3x16bit_RGB_grid_triangle_a_0080x0060_1.png | 814 | 868 | 790 | 918 | 797 | 827 |
| EmpireOriginal.png | 5723 | 5330 | 5215 | 5410 | 5438 | 5438 |
| tile_0178.png | 126 | 129 | 126 | 126 | 126 | 126 |
| img_480x360_3x16bit_RGB_bars_horizontal_square_0008_3xhalfPI.png | 152 | 149 | 149 | 149 | 149 | 149 |
| muzzle_05.png | 21641 | 40749 | 30179 | 21716 | 24670 | 21777 |
| trace_07.png | 13515 | 27106 | 15955 | 13578 | 14588 | 14037 |
| img_480x360_3x16bit_RGB_bars_vertical_triangle_progressive.png | 1351 | 1387 | 767 | 800 | 817 | 807 |
| img_480x360_3x16bit_RGB_grid_triangle_a_0024x0018_1.png | 682 | 695 | 628 | 690 | 786 | 745 |
| head.png | 36920 | 43266 | 35781 | 34266 | 32824 | 32503 |
| muzzle_01.png | 40062 | 63181 | 53453 | 41386 | 43443 | 40391 |
| s04i3p01.png | 84 | 84 | 84 | 84 | 91 | 91 |
| face.png | 35767 | 42075 | 35615 | 34057 | 32225 | 31663 |
| window_03.png | 2837 | 6366 | 5653 | 3109 | 3841 | 3093 |
| circle_03.png | 30280 | 63536 | 49173 | 35579 | 38341 | 31208 |
| pk01_floor02_s.png | 45989 | 50323 | 49051 | 46549 | 47551 | 46547 |
| spark_05_rotated.png | 33475 | 49138 | 43906 | 33635 | 37579 | 33391 |
| cropped_traffic_light_frame-2.png | 260 | 270 | 274 | 277 | 275 | 280 |
| tilemap-backgrounds.png | 543 | 501 | 505 | 519 | 531 | 532 |
| trace_03.png | 19425 | 32010 | 23243 | 20666 | 20911 | 18948 |
| star_05.png | 16797 | 32747 | 19272 | 17016 | 16854 | 16797 |
| tile_0128.png | 164 | 158 | 159 | 157 | 156 | 156 |
| tile_0056.png | 133 | 139 | 126 | 126 | 126 | 126 |
| bridge.png | 112029 | 137308 | 111967 | 111967 | 111962 | 111962 |
| cropped_traffic_light_frame-1.png | 440 | 448 | 439 | 421 | 449 | 460 |
| baboon.png | 189770 | 203875 | 179054 | 171325 | 172054 | 168969 |
| star_09.png | 19079 | 58229 | 35795 | 23810 | 23771 | 19381 |
| img_480x360_3x16bit_RGB_bars_vertical_square_0128_0xhalfPI.png | 167 | 144 | 167 | 167 | 144 | 144 |
| scratch_01.png | 14248 | 34817 | 32500 | 14962 | 19682 | 14556 |
| img_480x360_3x16bit_RGB_bars_vertical_square_0128_1xhalfPI.png | 144 | 138 | 138 | 138 | 138 | 138 |
| windows95.png | 11406 | 10567 | 10333 | 10731 | 10731 | 10731 |
| tilemap_packed.png | 4868 | 4662 | 4281 | 4178 | 4203 | 4223 |
| 9bde6aaf390109f97622d35ff1e900ee.png | 412539 | 412539 | 412539 | 412539 | 412539 | 412539 |
| 3dd595a30a79d3ba6d107fa6f60c5d3d.png | 126447 | 120731 | 122574 | 116276 | 118156 | 116640 |
| wallhaven-49yekn.png | 707442 | 713676 | 718750 | 715035 | 710444 | 711220 |
| 0049.png | 2834292 | 2834292 | 2834292 | 2834292 | 2834292 | 2834292 |
| blue_floor_tiles_01_Diffuse_1k.png | 4473120 | 4473120 | 4473120 | 4473120 | 4473120 | 4473120 |

### Execution And Cost Audit

The six synthetic fixtures exercise RGB8, RGBA8, RGB16, RGBA16, a 300-color
RGBA16 bypass, and a grayscale bypass. All 24 outputs passed official decode.
The 300-color Adjacency+MinLA result uses the existing Adjacency fallback, not
MinLA; EZeng modes use Default fallback. Its byte improvement is therefore
not attributed to the new refinement.

Graph objectives and final bytes are distinct. On the 48-color RGB8 fixture,
EZeng has graph cost 44,350 and emits 419 bytes. Refinement lowers graph cost
to 44,070 but emits 430 bytes. This is a measured counterexample to a
"lower MinLA cost means smaller JXL" admission rule.

The four-image real profile audit also passed all 16 official checks.
It confirms a global graph execution for each new mode on head/muzzle and
27 local graph executions on the editor screenshot. The illustration has
local executions too. Profile counts record attempted graph work; they are
not a guarantee that every attempted palette is admitted.

For the two full-image examples, EZeng+MinLA profiling gives:

| Image | Graph build | EZeng seed | MinLA refinement | Graph total |
| --- | ---: | ---: | ---: | ---: |
| head.png (252 colors) | 12.609 ms | 0.309 ms | 0.495 ms | 13.501 ms |
| muzzle_05.png (256 colors) | 36.120 ms | 0.313 ms | 0.417 ms | 36.901 ms |

Most direct work is spatial tuple lookup / graph construction, not local
optimization. This is not a CPU-stall diagnosis. Local group profile totals
are inclusive sums across workers and cannot be added to native wall time.

### Decision

Retain both native Rust strategies as default-off experiments. Do not promote
either to automatic selection or C/Python numeric modes based on this sample.
The graph objective helps existing poor orders but ignores MA prediction,
palette-table coding and actual entropy-model effects. It can misrank even
tiny two-color tables where reversing the order leaves graph cost unchanged.

The next useful work is either a cheaper graph-construction path or a
JXL-specific admission score including index residuals and palette-table cost.
Do not add a hidden full baseline re-encode, or claim the smallest result from
six measured encodes is available for one encode's cost.

## Validation And Reproduction Artifacts

- `cargo check --workspace`: passed, including FFI and CLI.
- `cargo test -p rs-jxl --lib palette_order`: 25 passed; brute-force
  insertion/delta tests cover crossed old edges and incremental updates.
- `cargo test -p rs-jxl --test palette_run_order --test palette_tools_reference`:
  16 passed, five external-reference tests ignored because their separate
  decoder environment was not configured. The official checks above did run.
- `cargo test -p rs-jxl-cli --example palette_lab`: one passed.
- Python `test_benchmark_palette_runs.py`: 12 passed.
- `cargo build --release -p rs-jxl-cli --example palette_lab`: passed.
- Not run: a fresh full corpus, main automatic scheduler benchmark, or
  Linux execution. The implementation adds no platform-specific code.

Base revision: `c00dd93`, plus this working-tree experiment.
Ignored local artifacts (no corpus images or encoded outputs are committed):

- `out/palette-minla-20261110/cases36.json`
- `out/palette-minla-20261110/repeat-e9-t8.json`
- `out/palette-minla-20261110/synthetic-profile-e9.json`
- `out/palette-minla-20261110/real-profile-e9.json`

SHA256 identities, used only for artifact provenance, not as a compressed
output correctness requirement:

```text
manifest  052a80196ebf803ba2785cc3b0433fdfffea7cc080c75b45b94b6c60773240a5
report    12d51e7e042f7306c08047dbfa1b8eae6e69e5ead58c10b5ed379b630c0db1b4
encoder   7cbb581d13bd454e9cdf70eaa4c400caaabcb8b8515fcde6493c47b020855c2a
djxl      c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481
```

## Other Attachment Directions

FXZ was audited separately, not bundled into this palette timing comparison.
Its [batched radix pulls](https://github.com/peacherwu969/fxz/blob/8236dd5fdb030c2e8a47419b760c664b6c4cabb5/src/liblzma/radix/radix_pull_batch.h)
suggest interleaving independent immutable Chain queries within each worker.
Current `source/chain_batch.rs` processes each position sequentially within its
worker. This is distinct from batching candidates at a single position, but
there is no measured Rust speedup yet. StaticIndex's advancing MatchCache
cannot be shared unchanged between interleaved positions. Any future version
must retain per-position budgets, branch dependencies and overflow requery.

Do not transplant FXZ code without resolving its per-file licensing. No new
LZP model, confidence gate, LZ format or decoder changes are included here.
