# LZ77 matched-length, price-cache and DP experiments

## Scope

**Not promoted to default.** The three prototypes are retained behind the
non-default `experimental-lz77-hotspots` Cargo feature. Normal builds compile
out the SIMD helper, DP pruning and cached length fields/lookup paths. No new
runtime flag or default search policy is introduced.

Baseline: `a1e12d0` on `main`. Native Rust pixel-lossless encoding only; no
changes to the codestream, decoder, search budget, effort mapping or public API.
This is a focused hotspot experiment, not a new corpus-wide performance claim.

Four proposed directions were checked against the actual source:

1. **First mismatch in an eight-token block.** Ordinary `MatchCache` compared
   eight `u32` tokens for equality, then reread a failed block one token at a
   time. The experiment uses the x86_64 SSE2 baseline to retain lane equality
   bits and identify the first mismatch. Equal blocks still advance eight
   tokens; short tails, prefix/cache bounds and other architectures retain the
   scalar behavior. There is no per-query ISA dispatch or diagnostic counter
   in production. This is distinct from the earlier opt-in AVX2/AVX512
   extension experiment, which did not demonstrate stable overall gains.
2. **Compiled length intervals/prices.** Build contiguous HybridUint interval
   ends once per fitted model, and precompute each cached context's length
   prices. The first interval still uses the original HybridUint token mapping.
   All subsequent intervals reuse boundaries and `f32` prices, including the
   original addition order. Cache limits fall back to the original sparse path;
   they do not reject inputs or cap match lengths. High symbol offsets with a
   small configured match-length cap must not overflow while compiling unused
   full-domain prices.
3. **Index reuse across refits.** Not applicable to ordinary e9: the default
   uses one fitted `SymbolId` pass with static/shared indexes disabled. Ordinary
   static indexing requires `Frontier`, also a one-pass route. The optional
   shared seed index serves the seed/first pass; explicit refit switches its
   second pass to mutable matching rather than rebuilding the same index.
   Retaining that index would change the second pass's search route, not remove
   an existing duplicate build. `RangeReplay` already supports match replay.
   No production change was made for this proposal.
4. **Narrow DP coverage pruning.** `RangeParser` already caches a best offer
   covering the entire current 16-endpoint block. That offer can certify that
   a new interval's leading scalar fringe cannot win. Only that covered fringe
   is skipped; future blocks and uncovered tails are processed normally. The
   certificate must be finite and strictly better under the existing tie rule.
   Merely comparing the first endpoint, or treating an equal-cost pending offer
   as already materialized in legacy mode, is not sufficient.

## Measurement Contract

- Frozen pre-change `palette_lab`: `out/lz-length-20261110/palette-before.exe`.
  SHA256: `4548825fc7960ea459604d90ff46cebff8329ca0e66a694f0ebc584ec84ea06d`.
- Effort 9, eight threads, default palette ordering, patches disabled in both
  binaries. This measures one projected native encode, not outer `searchbest`.
- Eight cases chosen before new timings: five LZ77-heavy screenshots/graphics
  from the previous 200-image profile, a large photo, 16-bit gray texture, and
  gray-alpha cursor. Manifest: `out/lz-length-20261110/cases.json`.
- Three alternating-order trials per binary/case for the initial screen.
  Report medians, aggregate bytes, bpp and aggregate pixels / summed medians.
  Input/output IO and process startup are excluded by the native timer.
- Every encoded artifact is decoded by the caller-supplied official libjxl
  0.12 `djxl`, then compared with original samples, including invisible RGB.
  Output JXL byte identity is not an acceptance target. Source/binary hashes
  only detect changed inputs or changed executables during measurement.
- Profile-enabled runs are separate diagnostics. Inclusive per-thread scopes
  overlap; sampled matching/DP times are not wall-time percentages.

The harness is `tools/benchmark_lz77_hotspots.py`. It checks native receipts,
cross-binary policy equality, actual codestream size, original pixels and input
stability. Temporary JXL/decoded PNG files are removed, including failure paths.

## Initial Combined Screen

Artifact: `out/lz-length-20261110/combined.json`.
All 48 independent decode checks passed; all eight measured sizes were unchanged.

| Image | Before s | Combined s | Time change |
| --- | ---: | ---: | ---: |
| PDF `9bde...` | 1.4443 | 1.3135 | -9.06% |
| VS Code `3dd...` | 0.3968 | 0.3921 | -1.18% |
| `wallhaven-n6qzmx` | 2.2888 | 1.9277 | -15.78% |
| Celeste | 1.1636 | 1.0754 | -7.58% |
| Wikipedia | 1.9129 | 1.8619 | -2.66% |
| `wallhaven-mlzoy1` | 4.9394 | 4.9068 | -0.66% |
| 16-bit rough linen | 0.2502 | 0.3083 | +23.21% |
| Gray-alpha cursor | 0.1775 | 0.1739 | -2.00% |

Aggregate: 17,961,370 bytes, 7.206381 bpp for both; 12.5734 -> 11.9596 s,
1.58584 -> 1.66723 MPix/s. The aggregate improvement does not excuse the texture
regression, and one combined screen does not attribute gains to each change.

### Texture Investigation

`texture-profile.json` shows no optimal DP or compiled-length calls for this
image. Its LZ77 scopes decreased, while the longest MA task increased. Thus
the profile does not support blaming the new length table. One-thread controls
in `serial-controls.json` showed approximately 1.528 -> 1.511 s for the texture.
This suggests a parallel-tail/timing effect, but does not prove the eight-thread
regression harmless. A repeated final control is required before acceptance.

## Final Disposition

### Ablations

These are independent alternating-order runs; compare within a run, not raw
times between separate sessions. All samples, including slow outliers, remain
in the reports. The same three-repeat native-timing and pixel-check rules apply.

| Run | Scope | Combined s | Ablated s | Ablated time change |
| --- | --- | ---: | ---: | ---: |
| `ablation.json` | Eight cases; disable compiled lengths and DP fringe pruning | 12.2112 | 12.7715 | +4.59% |
| `no-simd-ablation.json` | PDF, n6qzmx, Celeste, texture; restore original matched-length loop | 4.3858 | 4.8659 | +10.95% |

All 72 ablation artifacts passed official pixel checks, and sizes stayed equal.
The first ablation supports the **cache/prune combination**, not separate speed
claims for those two changes. The second supports the new matcher on the three
LZ77-heavy cases; the texture is faster without it in this run. The texture has
no optimal DP/compiled-length work and no stable single-thread slowdown, so its
parallel regression cannot be attributed to a specific inner kernel from these
measurements alone.

Selected for final validation: all three changes, with additional guards for
symbol-offset overflow and routes that do not consume the interval cache.

### Five-Repeat Final Control

`out/lz-length-20261110/final.json`: five measured trials plus one predetermined
warmup per binary/image, alternating order. All 96 artifacts, including warmups,
passed official pixel checks. The timer includes all native encoding work.

| Image | Before s | Experiment s | Time change | Before MPix/s | Experiment MPix/s | Both bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PDF `9bde...` | 1.38825 | 1.48400 | +6.90% | 1.0489 | 0.9812 | 2.266499 |
| VS Code `3dd...` | 0.46163 | 0.45945 | -0.47% | 4.6304 | 4.6524 | 0.473243 |
| `wallhaven-n6qzmx` | 2.46951 | 2.16010 | -12.53% | 0.8397 | 0.9600 | 1.771424 |
| Celeste | 1.26466 | 1.14767 | -9.25% | 0.6405 | 0.7058 | 1.754756 |
| Wikipedia | 1.99336 | 1.91494 | -3.93% | 1.9339 | 2.0131 | 1.558102 |
| `wallhaven-mlzoy1` | 5.15831 | 5.33802 | +3.48% | 1.6080 | 1.5538 | 13.675351 |
| 16-bit rough linen | 0.31223 | 0.32060 | +2.68% | 3.3649 | 3.2771 | 14.021960 |
| Gray-alpha cursor | 0.18080 | 0.18389 | +1.70% | 1.4499 | 1.4256 | 0.447937 |

Aggregate: **13.22876 -> 13.00866 s (-1.66%)**, **1.50728 -> 1.53278 MPix/s**.
Both produce 17,961,370 bytes, **7.206381 bpp**. The sizeable n6qzmx improvement
persisted, but the PDF benefit reversed and the initial texture regression
shrank substantially. Timing variation is large enough that the small aggregate
gain is not convincing evidence for a universally faster default.

**Decision:** retain a compile-time experiment, do not change normal builds.
These techniques have useful local wins but have not eliminated the LZ77
hotspot. In particular, no new general whole-match DP bound or extra index-reuse
machinery is justified by these results. A future experiment should isolate
actual match-length distributions and matching/frontier work on the remaining
slow streams, rather than extrapolating one screenshot's result to the corpus.

## Reproduction And Validation

Measured binaries and JSON are ignored local artifacts, with exact hashes in
each report. The final timing above was collected before adding compile-time
isolation, with all three paths active; it is not a timing of the final
feature-disabled/default build. No source sample was selected by its new
compression outcome, and there was no new full-corpus run.

Build the opt-in experiment:

```powershell
cargo build --release -p rs-jxl-cli --example palette_lab --features rs-jxl/experimental-lz77-hotspots
python -B tools/benchmark_lz77_hotspots.py --before out/lz-length-20261110/palette-before.exe --after target/release/examples/palette_lab.exe --decoder D:/cpp_pj/jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --manifest out/lz-length-20261110/cases.json --repeats 5 --warmups 1 --out out/lz-length-20261110/recheck.json
```

The optional `--profile` is for a separate diagnostic run, not the timed control.
Reports are never overwritten. Use a new output filename for another run.

Correctness coverage includes wide symbols, configured short maximum lengths,
sparse contexts, cache-cap fallback, every bit/lane of a mismatching block,
unaligned/tail and overlapping matches, legacy/source tie rules, finite and
non-finite DP certificates, resolved steps and independent scalar endpoint DP.
Current runtime acceptance is Windows x86_64 only; no Linux or macOS runtime
validation is implied.

Validation completed:

- 281 focused LZ77 tests in the default build.
- 286 focused LZ77 tests with both `experimental-lz77` and
  `experimental-lz77-hotspots` enabled.
- Four benchmark-harness tests, including cross-binary policy mismatch and
  pixel mismatch rejection, rotation, cleanup and report overwrite protection.
- 230 official `djxl` pixel comparisons across the six recorded runs. These
  are repeated configurations of eight images, **not 230 distinct images**.
  Every report passed and confirmed temporary-output cleanup.
- `cargo fmt --all -- --check` and `git diff --check`.
- `cargo clippy -p rs-jxl --lib --features experimental-lz77-hotspots --
  -D warnings -A clippy::needless_borrow`; the last exemption is pre-existing.
