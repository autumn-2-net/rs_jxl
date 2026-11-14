# Single-Pass MA Offsets and Half Description Cost

## Scope

The ordinary source-policy **effort 9** pixel encoder enables both:

- `DescriptionCost::Half` during MA split selection.
- `OffsetMode::DistributionSinglePass` after tree learning, before LZ77 and
  entropy coding.

This does not change the default public effort (7), other effort mappings,
JPEG coefficient transcoding, the decoder or the JXL format. Automatic settings
do not replace explicit fixed/custom/fast/block/seed/lookahead tree plans,
alternative learners or explicit channel-tree/search experiments. Explicit
offset settings and non-disabled description weights take precedence.

The acceptance objective is aggregate bytes/bpp and runtime, with exact decoded
samples. Individual compressed files may grow. Compressed-byte identity and
per-image non-regression are not requirements.

## One Backend Encode

The old checked Distribution path remains available for experiments. It builds
the original model, fits offsets using that model's HybridUint configurations,
encodes the proposal, and retains the smaller actual encoded domain.

Single-pass fitting does not build that baseline:

1. Train the MA tree once, using Half scoring when automatically selected.
2. Generate raw residual literals and reuse them to collect leaf statistics
   when the integer/routing contract allows it.
3. Score offset proposals using sampled HybridUint entropy/extra-bit costs
   and the shared offset-signaling histogram. Adaptive mappings use source
   `(4,2,0)` as the proxy; an explicit fixed mapping is honored.
4. Keep the original offset as a cheap score candidate. Apply the selected
   offsets directly, without serializing a competing baseline.
5. Remap cached literals when safe, then parse LZ77, build the final entropy
   model and serialize once. The final adaptive entropy mapping remains free
   to differ from the proxy.

Unsafe-for-reuse routing, wide Average4 prediction and nonunit multipliers
retain the original pixel fitter/tokenizer. These are correctness fallbacks,
not hidden full-output races. The raw cache is consumed on its first direct
parse; explicitly requested subsequent predictor trials must not shift an
already remapped cache a second time.

Single-pass does not disable other explicitly requested refinement experiments:
entropy, context-map or predictor comparisons can still add work if requested.
They are not enabled by this default policy.

`offset_single_pass_domains` and `offset_applied` count direct decisions.
`offset_wins` and `saved_bits` only describe measured comparisons: direct
offset application must not fabricate either kind of measured gain.

## Controls

- Rust source projection: `base.modular.source_ma_defaults = false` suppresses
  automatic additions but retains explicit tools. Manual `EncodeOptions`
  without source projection does not automatically enable the profile.
- CLI source encoding: `--source-effort 9 --no-source-ma-defaults` restores
  the former learned-tree defaults. `--source-refine-offsets` keeps its existing
  explicit Entropy/checked meaning.
- Python pixel encoding: `encode(image, effort=9, source_ma_defaults=False)`;
  also accepted in `encoder_options`, for multiframe/layer encoding and with
  still source search. JPEG byte transcoding rejects this pixel-only option.
- `encoder_lab --leaf-offset distribution-single-pass` requests the new mode
  explicitly. `distribution` retains the old checked mode. This ablation tool
  disables automatic MA defaults so omitted experiments remain off; use
  `--description-cost half` to combine the two explicit tools.
- `portfolio_lab --direct-candidate baseline --direct-ma-profile ...` isolates
  `legacy`, `half`, `checked`, `half-checked`, `single-pass` and
  `half-single-pass`. Omission exercises the actual current default.

`DescriptionCost::Disabled` is not presence-aware in the Rust scoring struct;
use `source_ma_defaults=false` to keep it explicitly disabled at e9. Half is a
model-description penalty weight, not half precision, half the tree, or half
effort.

## Measurement

`tools/benchmark_offset_single_pass.py` consumes an explicit original-image
manifest, alternates profile order deterministically, freezes the encoder,
compares decoded samples with official djxl, and removes temporary images.
It checks that every default/direct-offset domain builds exactly one model and
does not claim a measured fallback win. Output records retain native seconds,
mean per-image bpp, pixel-weighted bpp, bytes, MPix/s and growth counts.

This change does not automatically launch the full corpus. The focused cohort
reuses the prior 32-original manifest: 24 outcome-independent stratified cases,
four 16-bit cases and four known adverse controls. Synthetic 8/16-bit
gray/gray-alpha/RGB/RGBA fixtures are reported separately. Thread comparisons
reuse the same eight preselected originals at 1, 4 and 16 workers.

## Focused Results

Final Release build, 32 originals, 16 threads, effort 9. These are summed
native-call times, including pool setup and projection, excluding CLI startup,
input/output and verification. Each profile ran once per image in mixed order;
timing differences are descriptive, not confidence intervals.
`legacy` uses the same final binary with the two automatic additions disabled,
not a separately built historical executable.

| Profile | Total bytes | Mean image bpp | Weighted bpp | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Former e9 | 26,367,387 | 5.84519 | 9.07021 | 12.606 | 1.845 |
| Checked Distribution only | 26,229,054 | 5.81518 | 9.02262 | 15.136 | 1.536 |
| Single-pass Distribution only | 26,232,284 | 5.81901 | 9.02373 | 13.501 | 1.723 |
| New default: Half + single-pass Distribution | 26,010,580 | 5.76512 | 8.94747 | 11.375 | 2.045 |

The combined default saves **1.353% aggregate bytes** and **9.765% native
time** versus former e9. Twenty files shrink, five grow and seven tie; the
largest individual growth is 3.985% on a preselected adverse control. No
individual-output fallback is added. On the 24 stratified originals alone,
bytes improve 1.376% and time improves 11.356%; three files grow.

Removing the checked offset encode alone sacrifices 3,230 aggregate bytes
(0.0123% relative to checked Distribution) while reducing time by 10.802%.
Compared with former e9, offsets alone still cost 7.104% extra time. This is
not a claim that fitting offsets is free: Half scoring contributes substantially
to the combined speed gain. The preliminary six-profile run measured Half
alone at 26,175,018 bytes, 0.730% below former e9; adding direct offsets to Half
then removes another 164,438 bytes. That preliminary binary preceded the
cache-consumption fix for explicit follow-up predictor trials, so its timing
is not substituted for the final build above.

The final direct/default profiles each record 388 refinement domains and
388 model/output evaluations, versus 759 evaluations for checked offsets.
They reuse 65,977,569 raw residual tokens. Direct decisions record zero
`offset_wins` and zero `saved_bits`: measured savings come from this external
paired benchmark, not fabricated internal comparisons. Summed per-domain
timers can overlap across workers and are not wall-clock phase totals.

All 160 final 16-thread outputs (32 originals plus eight synthetic layouts,
four profiles) pass official djxl exact-sample checks. The synthetic set is
excluded from the original-image table: the combined profile shrinks it by
9.899%, while its total native time rises from 0.145s to 0.171s. Small-input
overheads have not disappeared.

### Thread Scaling

Same eight originals, same four profiles; the 16-thread entries below are the
matching subset of the 32-original run, not its full total.

| Threads | Former e9 seconds | Checked offsets seconds | Single-pass offsets seconds | Combined default seconds | Default MPix/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.698 | 29.584 | 26.261 | 22.631 | 0.371 |
| 4 | 7.941 | 8.841 | 8.091 | 6.902 | 1.215 |
| 16 | 3.990 | 4.658 | 4.317 | 3.737 | 2.244 |

Combined-default time improves 11.93%, 13.08% and 6.34% respectively against
the matched former e9 policy. Its measured throughput scales 3.28x from one
to four threads and 6.06x from one to sixteen. More threads are not assumed to
give linear scaling. The eight-image combined total is 7,985,311 bytes versus
8,048,552 before (-0.786%); mean bpp is 4.98635 versus 5.02111. Every profile's
per-image byte count is unchanged across these thread counts. This observation
is not an encoded-byte-identity constraint on future optimizations.

All **224** final-build outputs across these three runs pass official decoder
exact-sample verification. Source Patch/portfolio/multiframe automatic search
was not part of the timed matrix; the new default is a single source candidate,
not an automatic portfolio launch.

## Verification and Reproduction

Release CLI, FFI library and `portfolio_lab` were built on Windows. Focused
tests cover single-pass execution counts/error propagation, cached versus
fresh tokenization, wide residuals and routing fallbacks, default/explicit
option precedence, shared/streaming groups, previous-channel inputs, and
8/16-bit gray/gray-alpha/RGB/RGBA samples. The native Python check also covers
e9 still images, crop animations and layers with hidden RGB, for omitted,
enabled and disabled defaults. No Linux/macOS run was performed here.

```powershell
cargo build --release -p rs-jxl-cli -p rs-jxl-ffi --bins --lib --example portfolio_lab
python -B tools/benchmark_offset_single_pass.py `
  --manifest <original-case-manifest.json> `
  --encoder target/release/examples/portfolio_lab.exe `
  --decoder <official-djxl.exe> --threads 16 --synthetic `
  --profiles legacy checked single-pass default --out <new-output-directory>
```

For thread comparisons, add `--thread-subset`, omit `--synthetic` and use a
fresh output directory per thread count. Frozen binaries, input hashes,
per-output verification and exact native times are in the local artifacts:
`../jxl_python/out/offset-single-pass-20261114/`. `t16/` is preliminary;
`final-t16/`, `final-t1/` and `final-t4/` use the final native implementation.
No full-corpus test has been launched by this change.
