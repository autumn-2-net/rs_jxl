# JPEG AC entropy diagnostics and PJA modeling

This investigation separates the selected JXL AC stream's symbol cost, raw
HybridUint extra bits and model-fitting losses. It does not introduce a new
coding mode, alter ordinary JPEG encoding defaults or establish a global
compression bound. JPEG correctness remains original-file byte reconstruction.

## Native diagnostic

`jpeg::tuning::transcode_audited` returns encoded bytes, the existing exact
`SizeBreakdown`, and `AcAudit`. The CLI example exposes `--audit-ac`.

The audit reads the final selected model and token streams, after any LZ parse.
It uses the same HybridUint mapping, length-symbol offset and context mapping
as the writer, without changing them. It measures:

- Exact extra-bit count and AC histogram-set selector bits.
- Selected ANS table symbol cross-entropy, or actual prefix-code lengths.
- Empirical symbol entropy per selected histogram.
- Empirical symbol entropy per original context, using the final configurations.
- Empirical binary entropy of extra bits conditioned on selected histogram,
  mapped symbol and appended bit position.

For a histogram with counts `n_s`, the empirical reference is
`sum_s n_s * log2(N / n_s)`. Empty contexts contribute zero, not fabricated
observations. ANS table cost uses the actual normalized frequencies out of
4096. Actual AC payload bits minus selector bits, extra bits and table cross-
entropy are reported separately as a coder/framing residual; this includes
finite ANS state effects and is not model overhead.

Three different gaps must not be conflated:

1. Selected-table cost minus clustered empirical entropy: probability
   approximation or prefix coding cost at the current fixed partition.
2. Clustered minus raw-context entropy: optimistic benefit of separating the
   current contexts, ignoring extra table/context-map signaling and table limits.
3. Raw extra-bit count minus conditional bit entropy: observed marginal bit
   bias. JXL does not offer an arbitrary learned coder for those appended bits.

All empirical statistics fit and score the same data. Small-population contexts
can look especially attractive before signaling their distributions. These are
diagnostic references, not achievable file savings, generalization estimates,
or a lower bound over all legal JXL representations. The extra-bit calculation
also omits dependencies between bits and richer neighbor conditioning.

The audit is opt-in and adds counting work; ordinary encoding does not run it.
Audited wall time is explicitly labeled as diagnostic, not normal encode speed.
Source-search caches and head/tail multi-model selection cannot be combined
with this audit: a report must refer to the model actually being emitted.

## Local measurements

The e9 run used eight threads, cooperative work, symmetric DC MA and actual-
maximum CfL fitting, matching the preceding JPEG experiments. The 45 originals
comprise 16 format/compatibility inputs (some share image content), nine project
JPEGs and twenty CC12m JPEGs. They are a diagnostic set, not 45 independent
camera photographs or a representative population sample.

All 45 outputs reconstructed the original JPEG bytes with both our Rust
decoder and official djxl 0.12. Each output size also matched the corresponding
cached control in `full-order-zigzag.json` by input path and SHA256. That check
tests the observer, not a compression requirement: future compression changes
need not preserve encoded JXL bytes or sizes.

- Original JPEGs: 33,406,801 B; output: **27,372,627 B** (81.9373%).
- Pixels: 106,200,376; aggregate pixel-weighted rate: **2.061961 bpp**.
- 44 ANS models and one prefix model; no AC LZ admission in this control.
- Native encoding including audit: 22.023 s; the nested audit alone: 4.038 s.
  These diagnostic times are not a production speed comparison.

| Quantity | Byte equivalent | Fraction of complete JXL files |
| --- | ---: | ---: |
| Actual AC data | 24,941,754.250 | 91.1193% |
| Exact appended HybridUint bits | 1,531,458.250 | 5.5949% |
| Actual AC symbols and coder state | 23,410,296.000 | 85.5245% |
| Selected table symbol cross-entropy | 23,403,950.735 | 85.5013% |
| Table-fit gap at fixed partition | 10,341.531 | 0.0378% |
| Clustered minus raw-context empirical entropy | 176,103.304 | 0.6434% |
| Appended-bit marginal bias reference | 22,783.988 | 0.0832% |
| Actual coder/framing residual | 6,345.265 | 0.0232% |

Rows overlap and must not be summed as independent file sections. In particular,
the three gaps are already included in the actual AC payload, not additional
bytes. Appended-bit bias excludes richer spatial conditioning and is not an
estimate of what PJA saves.

The nine project JPEGs account for 82.64% of output bytes. Report the strata
separately before drawing conclusions:

| Stratum | Files | JXL bytes | Weighted bpp | Table-fit gap | Cluster gap | Extra-bit bias reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Format/compatibility | 16 | 2,579,213 | 1.396817 | 0.0345% | 0.9576% | 0.0886% |
| Project images | 9 | 22,620,524 | 2.266687 | 0.0292% | 0.3581% | 0.0453% |
| CC12m | 20 | 2,172,890 | 1.499576 | 0.1308% | 3.2400% | 0.4721% |

Each percentage uses that stratum's complete JXL size as denominator. Several
tiny format fixtures show a 13-17% raw-context gap, but only hundreds of bytes
in absolute terms, before signaling the extra tables. The CC12m gap deserves
more attention than these tiny fixtures; it is still not a promised 3.24%
file reduction. The previous multiple-model trial already demonstrated how
additional headers can consume payload savings.

### Interpretation

The selected probability tables are already close to their empirical symbol
entropy. Making ANS execution faster does not remove that entropy; merely
improving frequency normalization has little visible aggregate room here.
There is more evidence for selective context partitioning on the CC12m subset.
Any proposed partition still needs a complete data-plus-model cost comparison.

This is **not a JXL-format optimum**. These diagnostics keep tokenization,
HybridUint configurations and available contexts fixed. Other legal mappings,
orders and DC models can change those observations. Conversely, PJA's richer
decoder-visible prediction and adaptive conditioning can change the coding
problem itself, rather than only fitting the same tables more accurately.

### Validation and artifacts

- 11 native audit tests and one CLI argument test passed.
- 29 Python receipt/accounting tests passed.
- 45 e9 controls passed native and official byte reconstruction.
- Seven e10 images with forced AC LZ and up to four histogram sets passed twice
  at one thread. Repeated counts, entropy values and actual wire costs matched;
  timing was permitted to differ.
- Seven e3 images with AC LZ disabled passed at eight threads.
- Total: 66 full encodes with two independent JPEG reconstruction checks each.

Artifacts live in the experiment workspace
`jxl_python/out/jpeg-entropy-audit-20261117/`: `audit45.json`, `summary45.json`,
`compat-e10-h4-lz.json`, `summary-e10-h4-lz.json`, `compat-e3-off.json` and
`summary-e3-off.json`. The encoder executable is frozen as `audit.exe`; reports
include its identity and that of the official decoder. No default changed.

## PJA is a different statistical model

The source inspection pins PJA
[`1058f63`](https://github.com/YadeWira/PJA/commit/1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd),
whose source identifies version 0.9.13. PJA emits its own archive format, not
a JXL codestream. Its v7 coefficient decoder implements mechanisms that merely
changing a JXL coefficient permutation cannot reproduce:

- Interior AC magnitudes are conditioned on same-frequency neighboring blocks,
  intra-block activity and earlier color components. The neighborhood and
  coding logic are in
  [pjav7c.c](https://github.com/YadeWira/PJA/blob/1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd/pas/pjav7c.c#L315-L328)
  and [the interior loop](https://github.com/YadeWira/PJA/blob/1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd/pas/pjav7c_loop.h#L237-L315).
- The fourteen axis-edge ACs have separate row/column counts and
  quantization-aware boundary predictions. Prediction information conditions
  their zero/exponent/mantissa/sign probabilities; these are still original AC
  coefficients, not edge-prediction residuals.
  [Edge loop](https://github.com/YadeWira/PJA/blob/1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd/pas/pjav7c_loop.h#L321-L450).
- DC is predicted after AC using block-boundary estimates, then coded as a
  residual. This is a decoder-visible dependency, not an encoder-only search.
  [DC loop](https://github.com/YadeWira/PJA/blob/1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd/pas/pjav7c_loop.h#L455-L533).
- Binary arithmetic probabilities and context-mixer weights adapt after bits.
  The decoder reproduces this learning, instead of receiving only JXL's static
  ANS tables. [Mixer](https://github.com/YadeWira/PJA/blob/1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd/pas/pjav7c_loop.h#L30-L92).

This explains why the previous interior-first permutation experiment is not a
PJA-equivalent model. It changed order while retaining JXL's existing contexts.
No PJA source was transplanted and no PJA binary was benchmarked in this audit.

## Code layout

- `entropy/stream_audit.rs` observes the selected generic entropy model;
  its fixtures and invariance tests live in `stream_audit_tests.rs`.
- `jpeg/ac_audit.rs` carries JPEG-specific selector counts and the diagnostic
  receipt; `jpeg/tuning.rs` exposes the opt-in entry points, with codec tests
  in `tuning_tests.rs`.
- `jpeg_entropy_lab.rs` maps CLI arguments to native policies and emits receipts.
- `tools/jpeg_lab_receipts.py` owns shared experiment choices and measurement
  validation, without importing image loaders or executing codecs.
- `benchmark_jpeg_transcode.py` runs and verifies trials;
  `summarize_jpeg_entropy.py` analyzes saved receipts without running a codec.

Post-cleanup checks passed: 407 JPEG unit tests (nine explicitly external-oracle
tests remain ignored), ten entropy-audit tests, two CLI tests and 29 Python
tests. Re-reading all three saved audit reports reproduced their stored
summaries exactly. No compression policy or report schema changed in cleanup.

## Published comparison scope

The author reports output/input ratios of 72.60% for PJA and 79.80% for JXL on
151 Huffman camera JPEGs. This corresponds to about 9.0% less output than that
JXL result, not 27.4% less than JXL. The JXL baseline is 0.11.2, and the reported
command does not specify an effort sweep. Unsupported inputs in the other
sets count at their original size. Those mixed-set numbers combine feature
coverage with recompression effectiveness. The timing includes different
verification and disk-sync behavior.
[Author's methodology](https://github.com/YadeWira/PJA/wiki/Benchmarks#how-it-was-measured).

These are author-reported measurements, not results on our corpus/current e9.
The source architecture supports a richer-model explanation, but there is no
local ablation attributing a percentage of PJA's advantage to each mechanism.

## Reproduce

```powershell
cargo build --release -p rs-jxl-cli --example jpeg_entropy_lab
target/release/examples/jpeg_entropy_lab.exe input.jpg --effort 9 --threads 8 --cooperative --symmetric-ma --cfl-peak --audit-ac
python tools/benchmark_jpeg_transcode.py --encoder lab.exe --decoder djxl.exe --manifest cases.json --out audit.json --effort 9 --threads 8 --repeats 1 --candidate-arg=--cooperative --candidate-arg=--symmetric-ma --candidate-arg=--cfl-peak --candidate-arg=--audit-ac
python tools/summarize_jpeg_entropy.py audit.json --out summary.json
```

The harness retains original input/binary identities, exact writer costs and
native/official byte-reconstruction checks. It deletes temporary JXL/JPEG
outputs after validation. The summary counts each original once across repeats.
