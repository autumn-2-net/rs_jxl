# Complete PJA port status

Target: PJA 0.9.13, commit `1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd`.
The user requested the complete functionality port first, then cleanup of the
excluded archive mechanisms. All coefficient/sample coding tools are retained
for selection and composition with JXL tools. Native Rust is the runtime implementation. Original C or
Pascal binaries are development-only verification oracles. The user has now
authorized larger local corpus validation and speed optimization. No commit
or push has been requested.

The 44 source files in the pinned `pas/` directory have been retrieved and
verified against their Git blob hashes. The ignored local manifest is
`out/pja-port-20261117/source-manifest.json`. PJA's MIT license and full
third-party notice are retained in `crates/rs-jxl-plus/`.

| Source family | Rust destination | Current checkpoint |
| --- | --- | --- |
| `pjaarith.pas` | `arith.rs`, `arith/live.rs` | In-memory and concurrent live coder, bounded storage, completion/cancellation and producer-failure wakeups implemented and tested |
| `pjav7.pas`, `pjav7c.c`, `pjav7c_loop.h`, v7 parts of `pjamodel.pas` | `model/`, `predict.rs`, `probability.rs` | Full v7 DCT model; source-shaped independent bands; internal bounded parallel encode/decode and scalar/runtime prediction checks |
| `pjajpeg.pas` | `jpeg_native/` | Huffman/QM sequential and progressive, 8/12-bit DCT, lossless 2..16-bit, DNL/restarts/quirks; original-byte reconstruction tested |
| `pjajls.pas`, sample models in `pjamodel.pas` | `jls/`, `sample_model/` | JPEG-LS modes, NEAR/runs/restarts/subsampling; MED, blend and symbol models implemented |
| Raw and legacy coefficient portions of `pjamodel.pas`, `pjacoef.inc` | `raw_model.rs`, `legacy_model/` | Order-0/1/2 byte model and full/fast DCT tools retained; implicit external tables removed |
| `pjaj2k.pas` | `j2k/mod.rs` | Bounded box traversal, codestream locations, HT detection, JPX/MJ2 multiple-stream handling |
| `pjaj2kcs.pas` | `j2k/codestream/` | Headers, geometry, all progression orders, packets/tag-trees, segment rules, PLT/TLM and exact rebuilding implemented |
| `pjaj2kt1.c/h` | `j2k/tier1.rs` | MQ, bypass, passes and termination rules; original-C comparisons |
| `pjaj2kht.c`, `pjaj2kht_tab.h` | `j2k/ht.rs`, `j2k/tables/` | HT cleanup/refinement coding and decoding; original-C candidate-byte comparisons |
| `pjaj2km.pas`, `pjaj2kmc.c`, `pjaj2km_tab.h` | `j2k/model.rs`, `j2k/modes.rs` | Classic decision model, slices and side information; original-C stream comparisons |
| `pjaj2khm.c` | `j2k/ht_model.rs` | HT model including parent/cross-component contexts; original-C stream comparisons |
| `pjaj2kpx.pas`, `pjaj2kpm.c` | `j2k/pixels.rs`, `pixel_tier1.rs`, `pixel_model.rs`, `transform.rs` | Reversible 5/3/RCT, coefficient/pixel bridge, bitonal and multibit models, classic/HT pixel paths; independent C checks and linked reconstruction tests |
| Model vector loops in Pascal/C | `mix_simd.rs` | Runtime x86_64 AVX2 and portable scalar dot/weight-update kernels, wrapping/truncation equivalence tests |
| `pjacodec.pas` algorithm selection | `codec/`, `j2k/pipeline.rs`, `ported.rs` | Typed internal model routing and reconstruction; all DCT/sample/J2K tools retained; old archive/parts/repair wrappers removed |
| `pjaformat.pas`, archive-specific `pjalimits.pas` | Removed | Old archive descriptors, hashes, extension checks and protocol-only tests removed |
| `pjapack.pas`, `pjafs.pas`, CLI/info/console and source-specific instrumentation | Existing Rust bounded/atomic I/O; internal checks | Original PJA frontend/archive parity explicitly dropped by latest user steering; no new PJA CLI/API |
| `pjablake3.pas`, `pblake3.pas`, SIMD helpers | Removed | Integrity hashes, archive helpers and BLAKE3 dependency removed; source provenance checks remain development tools |
| Upstream selftest vectors, test fixtures/scripts | Rust tests and development-oracle checks | Source algorithm checks retained; original command-line selftest parity is not a deliverable |

Earlier baseline: experimental `.jxl_plus` v0, 8-bit Huffman DCT JPEG,
coefficient recompression with exact original-byte verification. The seven
focused real JPEGs total
3,138,885 bytes against the recorded standard-JXL baseline of 3,473,378
bytes, a 9.63% reduction. This is a small, selected comparison, not a claim
about other codecs or general compression performance.

Current cleanup checkpoint: the typed internal entry reconstructs **61** pinned
upstream files exactly (30 JPEG/MPO, 12 JPEG-LS, 19 JPEG 2000 family files).
Five damaged JPEGs that previously needed repair/patch or raw fallback are now
explicitly rejected, with their original parser failures checked. The J2K
files contain 21 codestreams and exercise all four model paths. Whole-codestream
raw fallback and the original-size admission gate are removed; structured
encoding may expand the original. This is focused coverage, not exhaustive
conformance. MPO/gain-map/embedded-JPEG bytes are still preserved in the source
reconstruction metadata; their former recursive independent-recompression
wrapper has been removed pending the new format's object-reference design.

Cleanup validation: **89 release library tests and 10 integration tests passed**.
Only the unchanged seven-input static experiment was filtered from this run.
Old archive compatibility, Pjax, GivenTables and repair/PATCH-specific tests
were removed with their obsolete implementation, rather than left disabled.
Source algorithm oracles, all coding-tool tests, valid-byte reconstruction and
resource/framing tests remain. Receipt: `out/pja-port-20261117/cleanup-tests.log`.

Historical pre-cleanup verification on 2026-11-17: release tests passed **77 library tests and
10 integration tests**, including the original-C algorithm oracles, all 66
upstream files, malformed/limit cases, live coder and parallel/SIMD checks,
and seven existing local real JPEGs. The three original-archive compatibility
tests were explicitly excluded under the latest scope. Rust 1.89 checking
passed for the crate and `jxl-plus` CLI. The ignored local receipts are
`out/pja-port-20261117/release-complete-port.log` and
`out/pja-port-20261117/msrv-complete-port.log`.

Execution limits remain per buffer/work stage, not a process-wide memory
guarantee. The live arithmetic coders and v7 parallel model entry points are
implemented; the public API still uses its serial scheduling. Legacy,
sample and JPEG 2000 traversal remains serial. Source fast header/scan reuse
and profiling/plugin-cost instrumentation are not ported as performance
features. The core algorithms are present; this is not a claim of matching
all upstream SIMD specializations or whole-program speed.

Latest user steering: PJA archive/bitstream compatibility is **not a target**.
Public `.pja` APIs and command-line compatibility work have stopped. Existing
algorithm comparisons are development evidence of the ported algorithms;
they do not constrain the eventual syntax. Finish the algorithm/functionality
port first, then build `jxl_plus` from the original JXL codestream plus extra
extensions. The experimental hashless v1 is not that final layout.

Current cleanup: the public experiment is version 1, with both 32-byte content
digests removed (64 bytes saved per file); old v0 is rejected. Internal JPEG
and J2K owned records and J2K mode bodies no longer store or validate hashes.
Canonical four-byte arithmetic termination, stream lengths and resource checks
remain, and original bytes are still compared during encoding/tests. This
detects framing failures, not every modification of file content. BLAKE3 and
the PJA archive/Pjax entry have now been deleted, including their dedicated
tests. Context-index hashes are retained.

The historical pre-cleanup hashless checkpoint passed 85 release library tests and 10 integration
tests, including all 66 upstream originals and the seven local JPEGs. Three
historical archive compatibility tests were excluded. Receipt:
`out/pja-port-20261117/hashless-static-tests.log`.

The user explicitly retains Full/Fast/V7, MED/Blend/symbol and dynamic/static
tools for selection/composition. A weak aggregate result does not authorize
deleting a coding tool. Old archive versions, implicit GivenTables and recursive
repair/patch syntax have been removed. Source information needed by valid JPEG,
JPEG-LS and J2K reconstruction remains. `ported` now uses typed `codec::Encoded`
records; it no longer routes through historical PJA streams 5/6/7/8. The J2K
chooser compares actual candidate output sizes, without the old floor shortcut.

The target Plus format explicitly includes adaptive probabilities and exact
pixel-lossless images as well as JPEG byte reconstruction. The user-supplied
`comr9` archive is an incomplete RGB pixel CM prototype to inform future tools,
not another PJA module or an already integrated pixel backend. Its algorithms,
model-state requirements and the target extension layers are described in
JXL_PLUS.md. No prototype code or its environment-dependent container has
been imported into the runtime.

Static-probability follow-up: 3 context projections times 4 finite capacities
were tested on the same 7 JPEGs, with full table costs and original-byte decode
checks. Best single configuration totals 3,729,671 equivalent complete bytes
(+18.8385% vs adaptive v1's 3,138,437); selecting the best tested configuration
per file totals 3,687,998 (+17.5107%). This single-probability-table family is
not the original multi-expert mixer with only learning disabled, nor a bound
on static methods. It remains test-only; see JXL_PLUS.md for assumptions and
`out/pja-port-20261117/static-probabilities-profiles.json` for all 84 variants.

The first test-only JXL/PJA composition now combines reversible AC CfL with
v7, retaining the original v7 candidate. Two of the same seven inputs qualify;
both fitting policies reconstruct original coefficients and JPEG bytes.
Including selectors, transmitted maps and all existing framing/metadata,
selection totals 3,130,892 bytes against 3,138,444 for the baseline candidate
set (7,552 bytes / 0.2406% smaller). Only the RGB input selects CfL; the other
eligible input remains on original v7. The zero-factor control and focused
release experiment passed. This is neither a new public format nor an
evaluation of JXL ANS/MA/LZ combinations; see JXL_PLUS.md and
`out/pja-port-20261117/cfl-v7-tools.json` for accounting and limitations.
Rust 1.89 checking passed again for the production crate and CLI after cleanup.

The subsequent MA/Weighted-to-v7 JPEG experiment checked two candidates on
each of the same seven originals: DC-only and all-frequency prediction by a
learned tree, followed by unchanged adaptive v7 coding. With actual tree/WP
parameters, modes and complete framing counted, the totals were 3,250,631
bytes (+3.5746%) and 3,348,627 (+6.6970%) against the selector-inclusive
3,138,444-byte baseline. All 14 candidates restored every original JPEG byte;
all seven inputs select original v7. Three focused unit tests and the release
experiment passed, as did Rust 1.89 test compilation and format checking.
The result is specific to the bounded coefficient-residual composition; it
does not evaluate pixel MA/CM or direct MA-leaf probability contexts. Tools
remain available. See `out/pja-port-20261117/ma-v7-tools.json` and JXL_PLUS.md.

The direct dynamic-context follow-up leaves original coefficient prediction
and v7 expert updates intact. A bounded learned tree chooses online
probability-calibration states (16 bins per leaf); one-leaf and fixed-six-leaf
controls use the same update rule. Across the same seven originals, the best
single learned configuration totals 3,134,967 equivalent complete bytes
(-0.1108%), versus fixed-six-leaf's 3,133,633 (-0.1533%) and the mode-inclusive
v7 baseline of 3,138,444. Per-input selection from all six calibration variants
and original v7 totals 3,133,469 (4,975 bytes / 0.1585% saved). Learned contexts
only add 14 bytes of savings beyond the best fixed-context/original pool in
this sample; they are not consistently superior. Actual tree/profile/length
costs are included. All 42 candidates reconstructed original bytes, nine
focused tests and the unchanged-v7 original-C check passed, along with Rust
1.89 test checking and formatting. This is probability calibration, not a
replacement v7 expert bank or a pixel experiment. See JXL_PLUS.md and
`out/pja-port-20261117/dynamic-ma-tools.json` for the finite search and controls.

The broader compression search then checked V7 band counts, original-band
Full/Fast, bounded interpolated bank/selector probability refinement and real
CfL compositions. All 75 candidates restored the seven original JPEGs exactly.
The selected total is 3,111,998 equivalent complete bytes, 26,446 bytes
(0.8426%) below the mode-inclusive original v7 baseline, and 21,471 bytes below
the preceding dynamic-MA pool. The nested candidate totals are 3,124,608 after
band/model choice, 3,115,603 after refinement, and 3,111,998 after CfL
combinations. Full/Fast did not win this sample and remain intact. All relevant
framing, actual band lengths and transform/model descriptors are included.
Seven focused tests, the release experiment and Rust 1.89 test checking passed.
See JXL_PLUS.md and `out/pja-port-20261117/compression-tool-search.json`; this is
a finite JPEG-only search, with no public defaults or final syntax changed.

The speed/larger-corpus follow-up now validates a fixed set of 301 originals:
seven development, 38 other prior-audit and 256 previously unused CC12m JPEGs.
All 1,228 tool candidates restored original bytes with no skips. On the new
256, fixed probability-refinement profile 2 saves 65,167 bytes (0.22049%)
against the 29,556,007-byte mode-inclusive V7 baseline; a per-file search saves
only 21 additional bytes. Actual standard Rust JXL E3/E9 totals are
32,386,114/31,398,801 bytes, all 512 outputs independently restored. Original
V7 is 8.7386%/5.8690% smaller than those standard outputs.

Fixed-expert array loops, fixed-width AVX2/scalar mixer helpers and exact
refinement-state simplifications preserve every stream. On seven inputs,
source-V7 coefficient-stage throughput improves about 23% encode / 24% decode.
The further scalar-prefix and unsigned-counter trials were slower and reverted.
Four/eight parallel bands also restore all 512 new-set candidates, but add
2.22346%/4.10521% vs original V7. The user keeps original-band, single-thread
V7 as the default: refinement, extra bands and Fast remain optional internal
tools. Strict parallel termination is checked before transactional coefficient
publication. See JXL_PLUS.md for API timing, cost accounting and receipts.

The first pixel-lossless experiment now has an internal encoder and actual
causal decoder: shared JXL RCT, predictors, weighted state and MA tree, with
static ANS or fitted/adaptive zero gates plus static ANS for nonzero residuals.
All models/lengths are serialized and counted; decoding does not borrow encoder
models or original-context traces. Equal-depth interleaved integer channels
including hidden RGB and 31-bit words are supported in this restricted profile.
It remains test-only experimental framing, not the final JXL extension format.
The fixed 20-PNG speed/size evaluation is documented in JXL_PLUS.md. Standard
JXL behavior and remaining pixel-port acceptance are unchanged. On the first
20-PNG / 15.83 MP fixed corpus, adaptive zero gating saves 1.05286% versus the
matched static experiment, adding 0.793% encode time and 4.507% decode time.
It is still 0.72226% larger than the complete default standard `rs-jxl` output.
All 240 measured outputs are sample-exact, including invisible RGB; this
restricted tool profile is not a replacement for the complete JXL toolset.

The subsequent `JXLPPX02` research stream adds existing native residual LZ,
strict bounded LZ decoding and actual per-group static/adaptive selection.
On the same 20 images, fixed Adaptive+LZ is 0.85926% below standard default;
Selected is 1.14359% below it with all candidate-search time charged. Both
prior-used and absent-prior-record subsets improve in aggregate. All 480
measured outputs are sample-exact. The new JXL-derived bitstream draft has
working identification/reset/dispatch decisions, but actual image/frame
extension integration and final vectors remain required before wire freeze.
See JXL_PLUS.md and JXL_PLUS_BITSTREAM_DRAFT.md for measurements and boundaries.

Completion of this port step requires the algorithm paths to be integrated
and checked, including arithmetic JPEG, high precision/lossless JPEG, JPEG-LS,
JPEG 2000 and HTJ2K, original-byte equality and malformed-input limits.
Local raw codeblock exceptions and metadata byte coding remain reconstruction
tools; arbitrary whole-file storage is excluded. A complete JXL-plus-extension design is subsequent
work, not implied by PJA model tests.
