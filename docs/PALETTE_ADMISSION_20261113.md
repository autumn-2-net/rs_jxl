# Fixed-Input Palette Admission Experiment

## Question

Does the existing Palette cost gate rank Skip, Absolute, and Delta correctly
once MA training, LZ77, entropy tables, and the complete JXL output are counted?
This is an opt-in diagnostic experiment, not a new production e9 policy.

The existing policy contains two separate decisions:

- A successful ExactDelta kernel does not try Absolute after a cost rejection.
  Only a kernel decline permits the normal Absolute fallback.
- Acceptance changes later channel attempts. ExactZero can therefore change the
  final encoding even if none of its delta tables is eventually accepted.

Whole-image seven-mode comparisons cannot separate these effects. This lab
overrides exactly one decision in each complete encode.

## Method

`modular::palette::counterfactual` assigns each attempt a stable
`(stage, stream_id, ordinal)` identity. Ordinals belong to a stream traversal,
not to the arrival order of parallel worker records.

1. Discovery performs one ordinary native e9 encode and records attempts.
2. A manifest selects attempts by geometry/kernel outcome and stable ID order,
   before any counterfactual byte results are examined.
3. The baseline captures the target's complete channel planes, channel metadata,
   Palette state, bit depth, accepted transform prefix, and attempt arguments.
4. Each alternative replays the original encode with otherwise identical
   options. The native hook compares the exact target input with that snapshot
   before changing anything. This is sample/metadata equality, not a hash match.
5. At that point only, choose Skip, force Absolute, or force a specified Delta
   mode. Force bypasses the estimated-cost gate, not the format/kernel limits.
6. Finish normal downstream encoding and compare complete JXL file sizes.

Later policy choices may respond to the intervention. This measures the total
effect of one admission decision and its normal continuation, not the isolated
entropy cost of one table. Other streams cannot be assigned an override.

The three choices are not always available. A forced ExactDelta that cannot
build a table is `unavailable`; an unused Absolute fallback is not mislabeled
as a successful Delta result. Such trials have no output bytes and cannot win.

Missing/repeated targets or changed replay inputs are errors, not skips.
Experiments are invocation-owned and require explicit `validate_complete()`.
Use fresh instances and disable candidate caches, as the example does. The
identity scheme is for a single frame/native encode, not a multi-frame or
multi-candidate search session.

## Implementation

- `crates/rs-jxl/src/modular/palette_counterfactual.rs`: snapshot/replay and
  invocation-owned records; no environment variables or process-global state.
- `crates/rs-jxl/src/modular/palette.rs`: optional attempt hook; default is None.
- `crates/rs-jxl-cli/examples/palette_admission_lab.rs`: complete native e9 lab;
  ordinary MA, native grouping and palette cap 1024, patches disabled.
- `tools/benchmark_palette_admission.py`: discovery, stable target selection,
  repetitions, independent official djxl verification, and JSON report.
- `crates/rs-jxl/tests/palette_counterfactual.rs`: 8/16-bit RGB/RGBA roundtrips,
  invisible RGB, unavailable fallback, input mismatch, and thread stability.

No public CLI preset, Python binding, decoder, or default e9 admission was
changed. With the hook absent, no experiment snapshot or event list is built.

## Reproduction

Build the development-only example:

```powershell
cargo build --release --locked --offline -p rs-jxl-cli --example palette_admission_lab -j 4
```

Example manifest (paths are relative to the manifest directory):

```json
{
  "cases": [{
    "id": "example-local-rejection",
    "path": "original.png",
    "baseline_mode": "exact-zero-gradient",
    "choices": ["skip", "absolute", "exact-delta-gradient", "exact-zero-gradient"],
    "selector": {
      "stage": "local", "width": 256, "height": 256,
      "begin": 0, "channels": 3, "outcome": "cost_rejected"
    },
    "max_targets": 1
  }]
}
```

Selectors also support `limit`, `colors`, and `deltas`. No matches is an error.
Modes are `absolute`, `exact-delta-{gradient,left,top}` and
`exact-zero-{gradient,left,top}`; `skip` is an alternative, not a baseline mode.

```powershell
python -B tools/benchmark_palette_admission.py `
  --encoder target/release/examples/palette_admission_lab.exe `
  --decoder PATH/TO/OFFICIAL/djxl.exe `
  --manifest cases.json --threads 8 --repeats 2 `
  --out out/palette-admission/report.json
```

The report path must be new. Temporary complete JXL outputs and decoded PNGs are
removed after verification. Every emitted JXL, including outputs preceding an
encoder error, is checked against original pixel samples. No rendered-only or
alpha-masked comparisons are used. Original and binary hashes are recorded for
provenance and checked for changes during the run, never used as a compressed
bitstream optimization objective.

`cost_forced` belongs to the event and identifies experimental admission.
`attempt.cost_bypassed` retains the original policy's bypass flag. A forced
candidate may legitimately be accepted with `cost_after > cost_before`.

Timing is full native encoding, excluding input decode, process startup, file
I/O, and oracle decoding, but including instrumentation and snapshot checks.
It is not a forecast of a production single-pass policy's speed. Do not add
independent per-target savings: entropy models and later decisions interact.

## Results

Base revision: `a4743c3`, with this opt-in instrumentation applied. Release
build, native e9, 8 encoder threads, cap 1024, default native grouping, no
patches/searchbest, original resolution. Six originals, seven baseline-mode
configurations, eight decision points, two repetitions per branch.

Local artifacts (ignored by Git):

- `out/palette-admission-20261113/cases.json`: selectors fixed before results.
- `out/palette-admission-20261113/results-v2.json`: complete validated report.
- `results.json`: aborted harness-debug run, NOT the results source. Its
  validator incorrectly applied one unavailable target's status to normal
  worker events. The event-local fix has a regression test; the full matrix
  was rerun, not repaired by dropping failing trials.

The final run passed: **77 complete JXL outputs**, each independently decoded
by official libjxl 0.12 djxl and checked against all original samples. Four
unavailable branch executions (two targets, two repetitions) are not encoding
failures or byte results. All repeated successful byte counts agreed. Temporary
JXL/PNG outputs were cleaned. Native time across ALL branches and discovery,
including unavailable attempts: **184.833 s**; summed encoder process wall
time: **188.298 s**, excluding oracle decoding and Python sample comparison.

### Complete Output Bytes

`G0` = global stream 0; `Lxx` = local stream xx; all selected ordinals are zero.
Each cell is the complete image size, not just that local stream. `N/A` means
that kernel could not construct a table under the unchanged target budget.

| Image / target / baseline policy | Baseline B | Skip B | Absolute B | Delta B | Zero Gradient B |
| --- | ---: | ---: | ---: | ---: | ---: |
| window_03 / G0 / Absolute | 2,837 | 3,149 | 2,837 | **2,541 (Gradient)** | 6,327 |
| 6826 / L30 / Delta Left | 6,057,315 | **6,047,285** | N/A | 6,057,315 (Left) | not tested |
| 6826 / L31 / Delta Left | 6,057,315 | **6,045,971** | N/A | 6,057,315 (Left) | not tested |
| 6826 / L98 / Zero Gradient | 3,992,577 | **3,992,577** | 4,014,720 | 4,009,224 (Gradient) | 4,026,863 |
| 1550 / L23 / Delta Top | 1,065,338 | **1,058,948** | 1,081,570 | 1,065,338 (Top) | not tested |
| CLIC f67f11de / L62 / Zero Gradient | 2,370,017 | **2,370,017** | 2,375,242 | 2,372,583 (Gradient) | 2,379,939 |
| OhWareraRettouSeitokai / L32 / Zero Gradient | 735,004 | **735,004** | 736,649 | 736,760 (Gradient) | 739,443 |
| 6835 / L131 / Zero Gradient | 4,904,964 | **4,904,964** | 4,905,764 | 4,906,373 (Gradient) | 4,907,249 |

The first two 6826 targets were the first two stable-ID accepted 256x256
three-channel attempts, not selected by their measured savings. The other
local targets match the previously recorded geometry/table signatures.

### Proxy Cost Versus Reality

| Target | Original estimated cost | Candidate estimated cost | Effect versus Skip on complete file |
| --- | ---: | ---: | ---: |
| 6826 L30 Delta Left | 327,294 | 312,723 | **+10,030 B**, despite a lower estimate |
| 6826 L31 Delta Left | 361,891 | 332,446 | **+11,344 B**, despite a lower estimate |
| 1550 L23 Delta Top | 749,124 | 335,606 | **+6,390 B**, despite a much lower estimate |
| 6826 L98 Absolute | 473,211 | 425,923 | **+22,143 B** |
| CLIC L62 Absolute | 160,044 | 116,103 | **+5,225 B** |
| Manga L32 Absolute | 57,887 | 52,694 | **+1,645 B** |
| 6835 L131 Absolute | 35,505 | 30,092 | **+800 B** |

This is not merely a threshold/tie problem: 1550's Delta estimate drops by
more than half and still produces a larger final file than Skip. The source
estimator in `modular/cost.rs` uses fixed gradient/context/HybridUint scoring;
it does not execute the final MA search, LZ77 parse, or entropy clustering.
The experiment proves misranking, not which downstream stage causes each byte
difference. It does not establish that adding one fixed table-overhead penalty
would fix these cases.

### Bpp And Native Time

`bpp = 8 * complete_file_bytes / (width * height)`. Time is the median of two
complete native encodes, with instrumentation. Small timing differences are
not claimed as speedups. A row's best choice was selected retrospectively from
real bytes; these are diagnostic oracles, not a deployable automatic policy.

| Target | Pixels | Baseline bpp | Best bpp | Baseline s | Best s | Best MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| window G0 (Delta Gradient) | 512x512 | 0.086578 | 0.077545 | 0.0918 | 0.0752 | 3.484 |
| 6826 L30 (Skip) | 2415x3528 | 5.687540 | 5.678122 | 4.1875 | 4.0933 | 2.081 |
| 6826 L31 (Skip) | 2415x3528 | 5.687540 | 5.676888 | 4.1948 | 4.0707 | 2.093 |
| 6826 L98 (Skip/tie) | 2415x3528 | 3.748846 | 3.748846 | 3.7137 | 3.6527 | 2.333 |
| 1550 L23 (Skip) | 1079x1439 | 5.489024 | 5.456101 | 0.7020 | 0.7087 | 2.191 |
| CLIC L62 (Skip/tie) | 1632x2048 | 5.672720 | 5.672720 | 1.5182 | 1.4336 | 2.332 |
| Manga L32 (Skip/tie) | 826x1170 | 6.084344 | 6.084344 | 0.6250 | 0.6292 | 1.536 |
| 6835 L131 (Skip/tie) | 2844x4410 | 3.128655 | 3.128655 | 5.4101 | 5.4090 | 2.319 |

### Interpretation And Next Step

1. **The admission problem is real.** Keeping exactly the same input and
   prefix still exposes harmful accepted tables. Not all whole-mode gains or
   regressions can be attributed to the Delta representation itself.
2. **Do not blindly add Absolute after a Delta/Zero cost rejection.** At the
   selected Zero-rejected sites, Absolute satisfies the existing estimated-cost
   gate but is worse than Skip in the final output. These controls explain part
   of how a Zero mode can help without an accepted delta table. They do not
   assign the entire earlier whole-image saving to one selected site.
3. **A whole-mode winner contains locally bad decisions.** Delta Top previously
   helped 1550 as a whole; removing its selected accepted table saves another
   6,390 B. Likewise the two independent 6826 interventions must not be summed
   or extrapolated to all 139 earlier accepted tables.
4. **Delta remains useful.** window_03 has a controlled 296 B reduction
   (10.43%) from changing just the global table to Delta Gradient. Zero Gradient
   at that identical input is much worse. There is no universally best mode.
5. Next investigate a cheap downstream-aware score for the same frozen inputs:
   transformed index/table residual token costs, a bounded native MA pilot,
   and native LZ77 profitability evidence, retaining Skip as a first-class
   candidate. Validate rankings against these complete-output labels before
   adding any policy. No classifier, new default threshold, or seven-way e9
   exhaustive search was introduced here.

This is a deliberately selected counterexample study, not a random-corpus
compression gain claim or proof of a globally optimal Palette policy. Broader
held-out testing belongs after a candidate scoring method exists.

### Verification

- `cargo check --workspace --all-targets --locked --offline -j 4`: passed.
- 112 Palette-focused library tests: passed.
- Five native counterfactual integration tests: passed, including U8/U16
  RGB/RGBA and exact invisible RGB preservation.
- Experiment CLI argument test: passed.
- Eleven Python harness tests: passed, including unavailable-target versus
  independent-worker records and output validation after process failure.
- 77 original-image outputs verified with official djxl; zero pixel failures.
