# MA Lookahead and RCT / Cross-Channel Joint Search

Two independent, opt-in native Rust experiments. Neither changes the default
encoder, decoder, Python ABI, bitstream syntax, lossy path, or JPEG reconstruction.
The experiment baselines are direct source-projected native e9 with previous
channels zero and **patches explicitly disabled**, not the outer scheduler and
not official libjxl. Mixing patch-enabled and patch-disabled baselines would
confound this comparison.

## Two-Ply MA Lookahead

Entry: `ModularOptions::source_ma_lookahead`, forwarded to
`ma_train::TrainOptions::lookahead`. Implementation lives in
`modular/ma_train/lookahead.rs`; ordinary split orchestration stays in its
existing module. `LookaheadOptions` defaults:

| Field | Default | Meaning |
| --- | ---: | --- |
| `width` | 3 | 2..4 actual split proposals, including the greedy proposal |
| `min_records` | 8192 | Minimum distinct prepared records, not weighted population |
| `relative_window` | 0.01 | Maximum relative one-step cost gap, plus the split threshold |
| `receipt` | None | Optional invocation-scoped work/estimated-gain counters |

The existing threshold sweep retains a bounded shortlist of actual property /
threshold / predictor choices. The existing six preference slots are **not**
treated as a six-wide beam. Only initial learner roots are reconsidered,
including separate channel roots when explicitly configured. Descendants use
the existing greedy learner. This is a restricted two-ply experiment, not a
whole-tree beam search or global optimization.

Each close root proposal is scored by partitioning private record-index lists
and searching both children once. Training columns remain immutable during
speculation. Both children may retain their leaf. The root decision compares
their residual/token costs and split overhead. Legacy mode uses the existing
split threshold as a model-overhead proxy; `TreeLearning::Cost` uses the PR5007
threshold/alphabet overhead estimate. Neither is exact final tree/context-map
signalling cost. Greedy wins exact score ties. No extra tree leaf ceiling is
introduced: speculation is skipped if the existing allowance cannot fit three
additional leaves.

The real complete encoding is still the authority. A lower estimated score may
produce a larger file after later training, clustering, LZ77 and serialization.
The lab reports raw independent variants, without hiding regressions behind a
second greedy encode. A future byte-size fallback must charge for both encodes.

Seed/fixed trees, symmetric PR5007 splitting, alternative split search/scoring,
and candidate-cache sharing are rejected for this initial experiment. It does
not speculate recursively or change the existing predictors and properties.

## RCT Joint Search

Native entries in `scheduler::rct_joint`:

- `screen_rcts`: bounded all-42 analysis only.
- `run_rct_joint`: screening, full trials and smallest complete JXL selection.
- `run_rct_joint_with_pool`: reuse a caller-owned bounded executor.

The initial domain is RGB/RGBA U8/U16, with 8/16-bit full-resolution integer
extra channels. Unsupported gray, float, wide, shifted or sub-8/16-bit domains
return an explicit error; they are not silently narrowed. Main alpha and
invisible RGB are preserved. This is an experimental search restriction, not
a new restriction on ordinary encoding.

The screen uses all pixels of small images, otherwise up to nine original
contiguous tiles and at most 8,192 sample positions by default. It applies all
42 actual native reversible transforms. Native gradient / HybridUint cost is
combined with a cheap previous-channel residual sign/zero/magnitude and value
sign conditional-histogram proxy. The proxy charges for its split and measures
sampled tokens, not encoded crops. It is not full MA training and does not model
palette changes, arbitrary MA combinations, Weighted prediction or LZ77.

Rankings for previous-channel counts 0/1/2 contribute to a common shortlist
(default three transforms). Each shortlisted transform is fully encoded with
all three counts. The ordinary auto-RCT 0/1/2 controls are also encoded. The
zero-channel control aliases the identical baseline rather than rerunning it.
Default cost: **12 actual full encodes**, 13 diagnostic rows including the alias.
Only the current smallest complete output is retained; ties prefer the first.
Baseline inclusion prevents size regression among successful encodes, but does
not prove the screen found the best of all 126 combinations.

The result exposes all screen scores, actual projected options, candidate
bytes/errors/times, the winning row, and complete elapsed native search time.
Screening, losing trials, failed trials, pool setup and bookkeeping are included.
The existing decoder is unchanged. Experimental candidate errors must be
inspected in the receipt, even when another candidate succeeds.

## Reproduction

```powershell
cargo build --release -p rs-jxl-cli --example ma_lookahead_lab --example rct_joint_lab -j 4
python -B tools/benchmark_ma_rct.py --encoder target/release/examples/ma_lookahead_lab.exe --rct-encoder target/release/examples/rct_joint_lab.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --case INPUT.png --effort 9 --threads 8 --repeats 2 --max-pixels 16000000 --out out/ma-rct-local.json
python -B tools/plot_patch_tradeoffs.py out/ma-rct-local.json --out out/ma-rct-local.png --svg --per-case --pareto
```

Repeat `--case`, or provide `--manifest`. A PNG is never resized to meet the
harness admission limit; an over-limit invocation errors. `--synthetic` adds
six fixed RGB/RGBA 8/16-bit fixtures; use `--min-records 4` to exercise small
roots. Normal real-image trials use 8,192. `--tree-learning cost` is available
for MA-only experiments, with its own matched greedy control. The combined
harness requires e9 / legacy learning for both experiments' common baseline.

The native labs load/write files outside the measured interval. Harnesses run
serially in rotated trial order, fingerprint inputs/binaries, and decode each
exported output using official djxl 0.12. Validation compares all original
integer samples, including RGB below alpha zero. Encoded byte identity is not
the correctness criterion. RCT losing/control rows are diagnostic statistics;
only its exported winner is externally pixel-verified. They are not presented
as separately verified chart points.

Charts use bpp against total native time. Corpus bpp is
`8 * sum(compressed_bytes) / sum(width * height)`, not an unweighted per-image
average. Corpus time sums each image's median native time; RCT points include
the entire tournament, never only the winner's execution time.

## Initial Verification

- 259 MA learner tests passed, including cache/reordering/parallel regressions.
- New XOR fixture: greedy stops at one leaf, two-ply finds four useful leaves
  under both Legacy and Cost scoring without mutating caller samples.
- Seven RCT tests passed: all-42 arithmetic, bounded deterministic sampling,
  previous-channel activation and lossless hidden-RGBA16 / extra-plane checks.
- Two MA lab CLI tests, 17 harness tests and 20 plotting tests passed.
- Six synthetic images x four exported variants: 24 official pixel checks
  passed. This includes 90 actual native encodes because RCT runs 12 per image.

Synthetic observations are separate from real-image compression claims. With
`min_records=4`, lookahead4 reduced one fixture by 23 bytes; lookahead2 changed
none. RCT joint reduced their aggregate 87,319 bytes to 86,699 bytes, but took
3.568s versus 0.282s for raw greedy. These tiny-image timings are illustrative,
not a speed guarantee.

## Real-Image Results

Report: `out/ma-rct-20261112/focused8.json`. Eight original-size images,
10,225,638 pixels, e9, eight threads, two rotated-order observations per variant.
The sample includes a document screenshot, illustration, urban photo, Tecnick
photo, DIV2K photo, RGBA plant texture, game screenshot and transparent particle.
No cropping, scaling, content-specific settings or per-file routing was used.

| Variant | Bytes | Weighted bpp | Native seconds | MPix/s | Size vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 8,177,833 | 6.397905 | 6.7802 | 1.5082 | baseline |
| Two-ply, width 2 | 8,178,261 | 6.398240 | 7.2908 | 1.4025 | +0.005234% |
| Two-ply, width 4 | 8,178,539 | 6.398458 | 7.0262 | 1.4554 | +0.008633% |
| RCT joint, shortlist 3 | 8,074,423 | 6.317003 | 86.5515 | 0.1181 | -1.264516% |

Every pair of repeated output sizes agreed. That is measurement stability, not
an output-hash optimization objective. Timing has only two observations; do not
interpret the width-4 versus width-2 ordering as a proven algorithmic speedup.

Both lookahead variants made real root changes on all eight images, so these
are not dormant-switch results. They each improved the game screenshot but
increased the other seven files. Width 2 saves 340 bytes on the game screenshot;
width 4 saves 89. For the PDF screenshot, local two-ply score improves by about
4,792 estimated bits, yet the final file grows by 13 bytes. The proxy sees two
levels, while subsequent full training and entropy coding can reverse that
ranking. This result does not establish that deeper beam search is ineffective;
it does reject promoting this particular root-only heuristic from this sample.

### Separating RCT Gains from Existing Cross-Channel Gains

The best of the ordinary auto-RCT controls 0/1/2 totals **8,082,371 bytes**.
Joint search adds only **7,948 bytes (0.09834%)** of aggregate savings beyond
those controls. Candidate statistics are diagnostic, as described above;
every exported joint winner and independent greedy output was pixel-verified.

| Image | Greedy bytes | Best auto-RCT 0/1/2 bytes | Joint bytes | Joint winner | Full search seconds |
| --- | ---: | ---: | ---: | --- | ---: |
| Paper9bde | 412,539 | 379,961 | 379,961 | auto RCT, prev2 | 15.6409 |
| Wallhaven49 | 707,442 | 698,485 | 698,485 | auto RCT, prev2 | 12.3235 |
| Urban001 | 728,708 | 716,539 | 716,539 | auto RCT, prev1 | 6.9921 |
| Tecnick024 | 1,697,292 | 1,679,823 | 1,671,875 | RCT26, prev2 | 13.8110 |
| DIV2K0380 | 3,684,044 | 3,679,841 | 3,679,841 | auto RCT, prev2 | 22.9573 |
| Plant14 RGBA | 312,585 | 305,975 | 305,975 | auto RCT, prev1 | 6.2758 |
| Minetest | 612,223 | 598,747 | 598,747 | auto RCT, prev2 | 6.8233 |
| RGBA particle | 23,000 | 23,000 | 23,000 | baseline | 1.7277 |

Tecnick024 improves by another 0.47315% beyond its best auto-RCT cross-channel
control. However, **RCT26 already belongs to the existing 19-transform set**.
This is evidence for a joint selection mismatch, not evidence that the extra
23 transforms supplied a new winning tool. No newly added transform won this
small sample. Because only a sampled shortlist was fully tried, this does not
prove those transforms could never win, or that the screen has perfect recall.

Screening itself took roughly 15-24ms per image in the first repetitions. The
expensive part was the twelve full encodes, making joint search about 12.77x
the baseline time. A future useful route must predict which combinations merit
encoding, rather than enable this tournament indiscriminately.

### Existing Cost Scoring Control

Report: `out/ma-rct-20261112/cost3.json`. Paper9bde, Tecnick024 and Minetest were
retested with PR5007 `Cost`, two repetitions, otherwise the same e9 settings.
This separates non-greedy search from the already-existing cost formulation.

| Variant | Bytes | Weighted bpp | Native seconds | Size vs Cost greedy |
| --- | ---: | ---: | ---: | ---: |
| Cost greedy | 2,717,523 | 6.203711 | 2.5475 | baseline |
| Cost + lookahead2 | 2,717,321 | 6.203249 | 2.7130 | -0.007433% |
| Cost + lookahead4 | 2,717,476 | 6.203603 | 2.8346 | -0.001730% |

Width 2 saves 202 aggregate bytes at 6.50% more observed time; width 4 saves
47 bytes at 11.27% more time. Both still worsen the paper/photo and improve
only the game screenshot. This does not meet a useful speed/size tradeoff.
The separate improvement from switching Legacy to Cost must not be attributed
to the new lookahead implementation.

## Artifacts and Decision

- Reports: `out/ma-rct-20261112/{synthetic,focused8,cost3}.json`.
- Aggregate chart: `out/ma-rct-20261112/tradeoffs.png` (also SVG and audit CSV).
- Per-image chart: `out/ma-rct-20261112/tradeoffs_per_case.png` (also SVG).
- Native MA lab SHA256:
  `467eb46a3f58dd49b81b71e5585640f2bfd65f3962435b9169038030bf6e3745`.
- Native RCT lab SHA256:
  `10dcb440650c931e6376bb7231a91d840ee8f41708c48dd8b93478c3669b797e`.
- Official djxl 0.12 SHA256:
  `c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.

All **106 exported outputs** passed exact official pixel checks: 24 synthetic,
64 main real-image observations, and 18 Cost-control observations. There were
348 native encodes including RCT's losing candidates; none returned an encoding
error. Losing candidates were not all independently decoded. All temporary
encoded/decoded outputs were cleaned. `cargo check --workspace` also passed.
Execution was on local Windows x64; this is not a new Linux runtime acceptance.

Keep both tools opt-in. Do not enable root lookahead or the twelve-encode RCT
tournament as an e9 default. The most concrete remaining lead is the Tecnick
joint-RCT selection miss, but it needs a cheaper, independently validated
decision rule. These results are not evidence for enabling all 42 transforms
or combining both experiments automatically.
