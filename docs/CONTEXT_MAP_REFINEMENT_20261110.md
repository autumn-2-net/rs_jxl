# Context-Map-Aware ANS Clustering Experiment

## Scope

This native Rust experiment adds local merge proposals after the existing
histogram clustering. It is disabled by default. It changes only the encoder's
choice of standard histograms and context-map entries; the decoder, MA tree,
predictors, transforms and LZ77 parse are unchanged. JPEG reconstruction and
lossy encoding are outside this experiment.

The original merge heuristic includes ANS population and histogram-table costs,
but not context-map signaling. That is inherited behavior, not a missing Rust
port. Context-map writing already compares simple, raw-coded and MTF-coded
representations. The new experiment changes the clustering fed to that writer.

## Proposal And Selection

1. Start with the existing final model and raw-context-to-histogram map.
   Recount the unchanged token streams using the common initial HybridUint
   config, and aggregate counts according to that map.
2. Rank extra histogram merges using ANSPopulationCost plus an H0 estimate of
   map-label savings. Use the existing 16-bin entropy bound to discard pairs
   whose estimated population loss cannot be covered even by removing the map.
3. For up to eight shortlisted pairs, serialize the actual context map,
   including its nested coding headers, RLE and entropy-coded payload.
   Accept only a strictly improving population-plus-map proposal score.
4. Continue for at most four extra merges. Return at most two complete-model
   proposals: after the first merge and at the end of the chain.
5. Rebuild the HybridUint configs and ANS tables for those proposals, retaining
   every original stream boundary and the final raw LZ77 distance context.
   Histogram IDs are densely renumbered in first-occurrence order, with tables
   reordered consistently; raw context positions are never permuted.
6. Serialize each complete candidate using the existing refinement selection.
   Count tree/model headers, payload and relevant section/TOC overhead. Keep
   the original on ties or larger output. Actual bytes, not a predicted score
   or JXL hash identity, decide the winner. The benchmark separately checks
   the complete file size and original decoded pixels.

The population proxy uses common initial configs and Fast ANS normalization,
not the final per-table adaptive config and normalization. The bound and H0
ranking prune *proposals* and do not prove globally optimal final clustering.
This first version only adds merges: it cannot undo old greedy merges, split a
cluster or reassign individual contexts.

The shortlist bounds exact map trials, not all work. Pair ranking can still
visit `O(merges * K^2)` pairs, with up to 256 histograms, and map scoring builds
nested raw/MTF models. Two complete payload candidates are not two total model
constructions. This is a compression experiment, not a promised speedup.

Prefix models, one-histogram models, incrementally appended models and models
with injected fixed/missing-symbol tables retain their baseline. No arbitrary
image or MA-training limits were introduced. Proposal errors are not silently
swallowed; ordinary clustering's total-weight validation also bounds unions.

The source e9 policy also uses streaming-style table/config choices for closed
groups. Those single-batch models are eligible before any output is committed;
the experiment retains that table/config policy. A model with appended batches
is skipped because its old tables and payloads are pinned.

## Controls

```rust
use rs_jxl::modular::refinement::ContextMapOptions;

options.modular.refinement.context_map = Some(ContextMapOptions {
    shortlist: 8,
    max_merges: 4,
});
```

Both budgets must be positive. They are local-search budgets, not limits on
input size or the number of learned MA leaves. `None` preserves default behavior.

The development example exposes the experiment without changing the production
CLI or Python ABI:

```powershell
cargo build --release -p rs-jxl-cli --example encoder_lab
target/release/examples/encoder_lab.exe input.png --effort 9 --threads 8 --refine-context-map --context-map-shortlist 8 --context-map-merges 4 --output output.jxl
```

`--search` and `--scan-only` conflict with this flag. Explicit tree/entropy
refinement flags can be combined for investigation, but the measurements below
isolate context-map refinement without offset fitting or other experiments.

The JSON receipt reports `context_map_proposals`, `context_map_wins`,
`context_map_proposed_merges`, `context_map_selected_merges`,
`context_map_pairs`, `context_map_map_trials` and `context_map_skipped`.
`context_map_seconds` includes proposal search, map trial serialization and
candidate model construction. Final candidate-output serialization is counted
under `write_seconds`. Timers aggregated across concurrent domains are work
times and must not be confused with whole-encode wall time.

## Verification

Focused tests cover positive ANS population loss offset by map savings, dense
renumbering and table order, bounded proposal chains, deterministic ties,
one-context versus one-histogram syntax, 8/9 and 255/256 histogram boundaries,
LZ length/distance semantics, fixed/adaptive configs, separate empty ANS
streams, every initial bit offset, unchanged appended models and exact-output
selection that tokenizes only once.

The reusable harness compares full original PNG samples, including invisible
RGB, against official `djxl`, not against another encode's JXL bytes:

```powershell
python -B tools/benchmark_refinement.py --encoder target/release/examples/encoder_lab.exe --decoder <official-djxl-path> --manifest <cases.json> --synthetic --variants baseline context-map --effort 9 --threads 8 --repeats 3 --out out/context-map-new-run.json
```

Encoded files are temporary and deleted after checking. Native-call timing
excludes process launch, image input decoding and output I/O. Decoder timings
are process wall times including I/O, not decoder-kernel benchmarks.

`--lz 1` / `--lz 2` and `--buffering 0` / `--buffering 2` can check LZ77
and full-frame/grouped policies. Both sides receive identical policy flags.

## Results, 2026-11-10

The corrected implementation was measured at e9, eight encoder threads,
shortlist 8 and at most four extra merges. The main test uses the existing
eight-case mixed manifest plus eight synthetic 8/16-bit gray, GA, RGB and RGBA
fixtures. Each variant is run three times, serially with rotated ordering.
Times below are per-image medians. No warmup was excluded; small differences
must not be interpreted as demonstrated speedups.

| Real input | Baseline bytes | Trial bytes | Saving | Baseline s | Trial s |
|---|---:|---:|---:|---:|---:|
| PDF screenshot `9bde6a...png` | 412,539 | 412,386 | 0.03709% | 1.352538 | 1.769182 |
| Code screenshot `3dd595...png` | 126,447 | 126,385 | 0.04903% | 0.448574 | 0.442784 |
| `wallhaven-n6qzmx.png` | 459,153 | 459,048 | 0.02287% | 2.375914 | 2.449703 |
| `Celeste_screenshot_02.png` | 177,669 | 177,601 | 0.03827% | 1.309766 | 1.288229 |
| `en.wikipedia.org.png` | 750,804 | 750,500 | 0.04049% | 1.898753 | 2.142106 |
| `wallhaven-mlzoy1.png` | 14,178,604 | 14,178,048 | 0.00392% | 5.332746 | 5.501607 |
| `rough_linen_Displacement_1k.png` | 1,841,476 | 1,840,935 | 0.02938% | 0.324096 | 0.348123 |
| `cursor_PNG65.png` | 14,678 | 14,678 | 0% | 0.147858 | 0.164078 |

Weighted aggregates use total bytes / total pixels, not the mean of image
ratios. Throughput is total pixels / sum of per-image median encode times.

| Eight real images, 19,939,406 pixels | Baseline | Trial |
|---|---:|---:|
| Bytes | 17,961,370 | 17,959,581 |
| bpp | 7.206381 | 7.205663 |
| Encode seconds | 13.190245 | 14.105812 |
| MPix/s | 1.511678 | 1.413560 |

This saves **1,789 bytes (0.00996%)** for **6.94% more measured encode time**.
Seven real images improve and one ties; none grows. Across all sixteen images,
nine improve and seven tie: 1,792 bytes saved, bpp 7.206177 -> 7.205459, time
13.342773 -> 14.275994 seconds (+6.99%). The two synthetic gains are 1 and 2 bytes.

The experiment genuinely ran: the first measured repetition across real images
had 344 domains, 244,351 pair visits, 5,245 exact map trials and 322 complete
candidate models; 193 domains selected a proposal. There were 412 proposed
chain merges and 388 merges in selected models. This illustrates why a small
shortlist per domain is still meaningful work over a complete image.

Additional one-repeat compatibility checks:

- LZ policy 1, full-frame/shared model, two real plus eight synthetic images:
  20/20 official checks pass. The 16-bit displacement image saves 169 bytes,
  synthetic 16-bit GA saves 8 bytes, all other cases tie.
- LZ policy 2, forced grouped streaming, two real plus eight synthetic images:
  20/20 checks pass. The code screenshot saves 50 bytes and synthetic 16-bit
  RGBA saves 3 bytes, all other cases tie.
- Together with the main 96 checks: **136/136 official `djxl` pixel comparisons
  pass**, including invisible RGB, with no file-size growth. These are focused
  tests, not a claim of full-format conformance or a large-corpus result.
- 161 entropy tests, 7 refinement tests, 28 example CLI tests and 13 benchmark
  harness tests passed. No decoder changes or special decoder flags were needed.

**Decision:** retain as an explicit experiment, disabled by default. The idea
does produce compatible smaller streams, but the current gains are too small
for the added work. Do not expand the brute-force shortlist or promote it into
normal e9 based on this sample. A future attempt should first find cases with
material context-map overhead or avoid repeated map/model construction.

## Reproducibility

Source base: `1c249d9`, plus this experiment's working-tree changes.
Raw local reports (temporary JXL/PNG outputs are not retained):

- `out/context-map-20261110/closed-groups-e9.json`: main corrected measurements.
- `out/context-map-20261110/lz1-fullframe-e9.json`: additional shared/LZ checks.
- `out/context-map-20261110/lz2-streaming-e9.json`: additional grouped/LZ checks.
- `out/context-map-20261110/focused-e9.json`: earlier eligibility diagnostic,
  **not the reported performance result**. That version excluded all streaming
  table policies and skipped the real images; this was corrected to distinguish
  closed single-batch models from truly incremental appended models.

The corrected reports pin these SHA256 identities for reproducibility only,
not as a compression objective:

```text
encoder_lab.exe
f94ad397d4f7b5ce65aefd3f6ca36fc2bbb01c9cb59e444cd4fd0525ad9e94ee
official libjxl 0.12 djxl.exe
c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481
eight-real-image manifest
5bcb45bcab68bd428cf5fa2220b2b1dedb139a3ef16b87963e3e1628e89ad4b9
```
