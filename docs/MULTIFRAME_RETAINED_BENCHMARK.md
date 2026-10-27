# Retained-Reference Multiframe Measurements

## Scope

Measured on Windows x86-64 on 2026-10-27 with the release native Rust FFI,
effort 9 and four encoder/decoder threads. The checkpoint preceding this work
is `1a4e85a`; these results include the subsequent retained-reference changes.
Inputs are decoded once, then the same sample arrays and supported metadata
are supplied to every pixel-encoding mode. JPEG inputs here are pixel-lossless
relative to the input decoder, not reversible JPEG transcodes.

- `auto-adaptive`: existing byte-selected crop/mask/tile/merged candidates,
  first/previous references and proven-byte pruning.
- `auto-retained`: the same tools with `retained_reference=true`. Slots 1/2
  rotate complete canvas states; slot 3 remains private patch storage.
- `independent`: sum of complete still-image JXL files at the same effort.

Times include Python input packing, FFI, codec setup and actual codec work.
They exclude initial source reading/decoding and subsequent correctness checks.
Each mode ran once, without warmups; small timing differences are not evidence
of a speedup. bpp uses all logical input pixels, including repeated canvases:
`bytes * 8 / (width * height * logical_frames)`. Encoded data stayed in memory.

## Results

| Sequence | Mode | Bytes | bpp | Encode s | Encode MPix/s | Decode s | Decode MPix/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t1, 4 x 4096 x 4096 RGBA | auto-adaptive | 12,478 | 0.001487 | 18.606 | 3.607 | 2.896 | 23.176 |
| t1 | auto-retained | 12,478 | 0.001487 | 17.626 | 3.807 | 2.805 | 23.922 |
| t1 | independent | 45,714 | 0.005450 | 20.610 | 3.256 | 1.229 | 54.584 |
| t2, 4 x 2000 x 2183 RGB | auto-adaptive | 3,767,749 | 1.725950 | 33.906 | 0.515 | 3.427 | 5.097 |
| t2 | auto-retained | 3,767,749 | 1.725950 | 34.478 | 0.507 | 3.975 | 4.394 |
| t2 | independent | 4,098,261 | 1.877353 | 10.480 | 1.666 | 0.503 | 34.694 |
| Synthetic recurrence, 4 x 256 x 192 RGB | auto-adaptive | 645,549 | 26.267456 | 4.081 | 0.048 | 0.0468 | 4.200 |
| Synthetic recurrence | auto-retained | 484,436 | 19.711751 | 3.195 | 0.062 | 0.0351 | 5.601 |
| Synthetic recurrence | independent | 645,712 | 26.274089 | 1.817 | 0.108 | 0.0457 | 4.302 |

t1/t2 have **no additional size benefit** from the retained policy. Their
multiframe outputs are respectively 72.70% and 8.06% smaller than independent
stills, but that gain already exists with the default policy. In particular,
t2 still spends much longer encoding and coalescing than independent stills.
This change is reference reuse, not a fix for all candidate-search/decoder cost.

The synthetic sequence is `X, A, B, A2`, where A2 differs from A only in an 8x8
region. Default references at A2 contain X and B. The retained policy has A and
B, selects `CropRetained` against logical frame 1, and encodes that final
candidate in 258 bytes instead of a 161,371-byte full-frame candidate. Complete
sequence size drops by **24.96%**. This deliberately constructed recurrence
demonstrates the new capability; its random image content and small dimensions
are not a representative throughput workload.

Every row passed exact original-sample recovery. Comparisons include decoded
shape/dtype, hidden color, alpha planes where present, and animation durations.
No test requires new and old compressed bytes to be identical. Independent
official djxl 0.12/0.13 acceptance uses focused recurrence fixtures in the Python
suite, not every benchmark row. Full-corpus and cross-platform runtime testing
have not been done for this change.

## Reproduction

With the new release library selected through `RS_JXL_NATIVE_LIB`, supply the
two existing sequence folders:

```powershell
python -B scripts/benchmark_multiframe_native.py path/to/t1 path/to/t2 --effort 9 --threads 4 --modes auto-adaptive auto-retained independent --out out/retained-reference-e9/report.json
```

The benchmark's default modes remain unchanged; `auto-retained` is explicit.
The synthetic input arrays were generated as follows, then encoded through the
same `benchmark_multiframe_native.measure` function and modes:

```python
import numpy as np

rng = np.random.default_rng(20261027)
x, a, b = [rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
           for _ in range(3)]
a2 = a.copy()
a2[64:72, 80:88] ^= np.uint8(0x7f)
frames = [x, a, b, a2]
```

Use HWC, 8-bit sRGB, durations 1, ticks per second 1000/1 and loops 0. The
public API switch is:

```python
data, info = rs_jxl.encode_multiframe(
    frames, strategy="auto", reference="auto", effort=9, threads=4,
    durations=1, tps=(1000, 1), loops=0,
    auto_options={"retained_reference": True}, return_info=True,
)
```

Generated full reports are locally under
`out/retained-reference-e9-20261027/report.{json,md}` and
`out/retained-reference-e9-20261027/synthetic.{json,md}`. They include candidate
bytes, pruning work and reference-slot transitions. These runtime artifacts
are not source files and are not committed.

## Remaining Limits

This policy retains two complete states, not arbitrary historical images. It
can evict a future-useful first frame and produce a larger sequence than the
default policy; therefore it remains opt-in. It does not find translated
regions or add inter-frame patch matching. Geometry-only analysis continues
to use static first/previous references because it has no encoded winners to
drive reference allocation. Automatic differences still require equal input
geometry; explicit different-size rectangles use `encode_layers`.
