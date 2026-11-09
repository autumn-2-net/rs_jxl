# Post-Training Refinement Experiments

## Scope

Native Rust, pixel-lossless encoder experiments inspired by the fixed-topology
leaf refinement and entropy candidate selection in
[koni_jxl](https://github.com/zenbaku/koni_jxl/blob/4506c3be14f1d3bf381c181631183d0ae0528f89/packages/koni_jxl/lib/src/encode/encoder.dart).
The implementation uses this repository's own traversal, predictors, tree,
tokenizer and entropy writer. No Dart code, external codec or decoder change is
introduced. The prior experimental LZ77 kernels are independent and disabled.

Both switches default to off, independently of effort. These are development
experiments, not new default effort policies or Python options. No lossy or
JPEG-transcode policy changes are made.

## Controls

```text
cargo build --release -p rs-jxl-cli --example encoder_lab
target/release/examples/encoder_lab input.png --effort 9 --threads 8 --refine-entropy
target/release/examples/encoder_lab input.png --effort 9 --threads 8 --refine-leaves
target/release/examples/encoder_lab input.png --effort 9 --threads 8 --refine-entropy --refine-leaves
```

The flags cannot be combined with `--search` or `--scan-only`: the initial
experiment measures fixed candidates, not a new outer scheduler. Leaf refitting
also conflicts with `--fast-context`. Explicit fixed/custom trees remain intact.

Native controls live in `modular::encode::ModularOptions::refinement`, whose
type is `modular::refinement::Options`. The optional caller-owned `Receipt`
aggregates completed domains, emitted model trials, proposed and winning refits,
alternate entropy wins, saved cost bits and stage timings. Stage times are sums
of work and can overlap across independently encoded domains; they are not a
wall-clock breakdown. Use a fresh receipt per invocation.

## Algorithms

`entropy` retains the source-mode baseline and tests the opposite Prefix/ANS
mode with its own valid model and HybridUint selection. It reuses the exact
tree, transforms, residual tokens and LZ77 parse. It does not launch another
MA learner, RCT/WP search or LZ77 matcher. The winning candidate is selected by
serialized size, not merely the entropy estimate. This is deliberately a
two-mode test, not Koni's entire configuration Cartesian product.

`leaves` freezes the learned tree topology, then traverses all input samples
with the existing causal predictor/property state. For eligible leaves, it
scores the original predictor and permitted Gradient/Weighted alternatives by
HybridUint(4,1,2) token entropy plus extra bits. Source predictor restrictions
remain in force. Leaves with an offset or non-unit multiplier, leaves without
samples, and exact ties remain unchanged. The proposal is only a heuristic;
the original encoded candidate remains available for actual-size fallback.

No dual full-image residual arrays are retained. Histograms scale with eligible
leaf count; WP row state and channel/stream histories are preserved. WP advances
even at non-WP leaves, and property 15 keeps WP state active when necessary.
Reference-channel eligibility and stream IDs match actual tokenization.

A successful proposal repeats tokenization, LZ77 parsing and entropy encoding,
but not image preparation or tree learning. Enabling both flags tests the
entropy pair for each proposed tree. This can cost substantially more time;
no speedup or worthwhile size gain is assumed.

Local streams are compared by actual unpadded payload bits before appending to
the caller's possibly unaligned writer. Shared streams include finalized
payload bytes and unpadded TOC size-entry costs; unchanged entries do not affect
ordering. Only winning sections charge output budgets and model-cost receipts.
No encoded-byte equality requirement is imposed on an optimization.

## Validation

```text
python -B tools/benchmark_refinement.py --encoder PATH_TO_ENCODER_LAB --decoder PATH_TO_OFFICIAL_DJXL --case image.png --synthetic --effort 9 --threads 8 --repeats 3 --out out/refinement/results.json
```

Variants are `baseline`, `entropy`, `leaves`, and `combined`; optional `frozen`
uses `--frozen PATH` as a pre-change control. `--buffering` can explicitly test
full-frame and streaming paths. Synthetic cases cover 8/16-bit gray, gray-alpha,
RGB and RGBA, including hidden colors under zero and partial alpha.

Trials run serially in rotating order. Reported encode seconds exclude input
decoding, CLI startup, output I/O and oracle decoding. Reports retain actual
bytes, bpp, MPix/s, per-image medians, commands, executable provenance and work
receipts. Official djxl reconstructs every output for comparison against original
samples. Temporary encoded/decoded outputs are removed by the harness. The
process timeout is only a test guard, never an encoder selection algorithm.

## Results

**Not promoted.** Six real images, effort 9, eight threads, three order-rotated
repetitions per variant. Times below sum per-image medians, not whole-process
wall times. The corpus contains 15,274,438 pixels. Baseline is the new binary
with both flags off; a separately preserved pre-change binary is also measured.

| Variant | Total bytes | Saving vs baseline | Encode seconds | MPix/s | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 17,281,764 | 0 | 8.9550 | 1.7057 | 9.051339 |
| Frozen pre-change | 17,281,764 | 0 | 9.4025 | 1.6245 | 9.051339 |
| Entropy comparison | 17,281,731 | 0.000191% | 9.5470 | 1.5999 | 9.051321 |
| Leaf refit | 17,277,066 | 0.027185% | 11.8776 | 1.2860 | 9.048878 |
| Both | 17,277,033 | 0.027376% | 12.8754 | 1.1863 | 9.048861 |

Both switches save only 4,731 bytes while increasing measured encoding time by
43.8%. Entropy comparison alone saves 33 bytes on the 16-bit texture; none of
the other five real images benefits. Leaf refit wins in some domains on all
six images, but the final gain is too small for its extra work. Timings include
diagnostic receipts; they have not been separately measured without receipts.
The frozen/current control also shows timing variation, so incidental faster
runs must not be claimed as a default-path speed improvement.

| Input | Baseline bytes | Both bytes | Baseline s | Both s | Baseline MPix/s | Both MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `9bde6aaf...png` | 412,541 | 412,385 | 1.6046 | 2.6413 | 0.9075 | 0.5513 |
| `3dd595a3...png` | 126,449 | 126,360 | 0.4787 | 0.7329 | 4.4652 | 2.9166 |
| `wallhaven-49yekn.png` | 707,442 | 707,132 | 0.9997 | 1.3570 | 2.0742 | 1.5281 |
| `wallhaven-mlzoy1.png` | 14,179,178 | 14,175,119 | 5.4233 | 7.5080 | 1.5294 | 1.1047 |
| `rough_linen_Displacement_1k.png` | 1,841,476 | 1,841,373 | 0.2788 | 0.4118 | 3.7683 | 2.5513 |
| `cursor_PNG65.png` | 14,678 | 14,664 | 0.1698 | 0.2244 | 1.5437 | 1.1681 |

These observations do not reproduce the large gains reported against Koni's
single-predictor baseline. This encoder already chooses predictors per leaf
during MA learning, so the remaining full-population correction is small on
this sample. This does not prove that no specialized inputs could benefit.

Eight synthetic fixtures additionally exercise small 8/16-bit inputs. The
16-bit gray fixture improves from 1,227 to 1,206 bytes with both switches; this
21-byte result is correctness/activation evidence, not a practical speed claim.

### Acceptance

- 170 official djxl pixel checks passed, with no increased output size against
  the corresponding baseline: 90 real-image paired runs, 32 synthetic runs,
  40 single-threaded streaming runs and eight forced full-frame shared-tree runs.
- The streaming test actually entered multiple independent domains: 36 for the
  editor screenshot and 20 for the 16-bit texture, not just an ignored flag.
- Forced full-frame checks retain one shared tree across multiple sections;
  their output totals are compared only against the same buffering policy.
- 583 Modular tests passed, one existing test was ignored. The focused CLI
  control test and three Python harness tests passed.
- An additional shared-tree budget regression passed: discarded model trials
  do not charge section bytes, and the selected output still honors pruning.
- All workspace targets, including the native Python bridge and existing
  optional `experimental-lz77` feature, passed `cargo check`. Edited Rust files
  passed formatting checks; `git diff --check` passed.
- Strict Clippy remains blocked by four pre-existing `needless_borrow` findings
  in `modular/fast_context.rs` (500, 503, 506, 514); they are not suppressed or
  changed as part of this experiment.

Reports in ignored `out/refinement-20261109/`:

- `paired-e9.json`: real-image measurements and executable identities.
- `synthetic-e9-r1.json`: eight-bit/sixteen-bit and hidden-color checks.
- `streaming-single-e9.json`: explicit streaming, one encoder thread.
- `shared-full-e9.json`: explicit full-frame, shared-tree multi-section checks.

An earlier `synthetic-e9.json` invocation was rejected for a missing refinement
receipt: it had picked up the pre-build executable. It is not included in the
completed measurements above. The receipt check prevented that stale binary
from being presented as a successful experiment.
