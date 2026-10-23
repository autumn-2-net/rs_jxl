# FJXL Port Status

Source: immutable libjxl `b866b3defec13494bfbaf705db93581cda118e37`,
`lib/jxl/enc_fast_lossless.{cc,h}`. Translated code retains the JPEG XL Project
Authors BSD-3-Clause notice; full notice is in `docs/PORTING_KERNELS.md`.

## Checkpoint

Complete scalar encoding, runtime AVX2 row/entropy kernels, region leases,
prepared states and incremental/seek output are connected. Parent registered
the module and typed direct-candidate dispatch.

The latest stage-2 release checkpoint passed **60 ordinary FJXL tests**, including
the six packed-narrow tests and five transformed-input tests (with Windows guard
pages). The two external oracle suites were ignored in that ordinary run, then
**explicitly rerun successfully**: standalone plus region/output suites passed
**108 exact pixel checks across both official djxl versions**. The two typed F16
suites were also explicitly rerun and passed **32 exact word checks**. These
results were reported by parent for the integrated frozen stage-2 snapshot.

Stage 2 connects source packed-i16 prediction/residual/prefix counting through
ordinary <=14-bit sampling/encoding, palette indices and palette dictionary
encoding. Archimedes' additive fused input/YCoCg helper is wired through
`Kernel::load_transformed_row` into sampling and encoding; the former second
YCoCg pass is removed. The raw loader remains available and unchanged. Existing
public APIs, region lifecycles, entropy algorithms, backend chunk sizes and
valid-input results are unchanged. No ARM or AVX512 work is included.

### Earlier Validation

The frozen baseline before the wide-prediction slice passed **1416 tests in both debug and optimized
validation**, all affected official checks, strict all-target Clippy, Rust 1.89
checks and current x86 Linux compilation, as reported by parent.

That baseline includes **108 FJXL official exact checks = 76 standalone + 32
region** with two decoders. Parent additionally reports 32 typed U8/U16 pixel
checks, 16 finite F16-word checks and 16 special F16-word checks including
NaN/Inf/payloads. These are exact sample-word checks, not floating-point equality.
The reported 2.32-second FJXL oracle run after a 1 minute 22 second core build
is test-run timing, not an encoder performance benchmark.

The subsequent wide-prediction/input-deinterleaving checkpoint passed **49 FJXL
unit tests, zero failures, two ignored** in central debug validation, followed by
**1450 release-workspace tests, 108 FJXL official checks and 32 typed F16-word
checks**, as reported by parent. Parent corrected one test-only Clippy modulo
lint; no implementation change was needed. Those results cover the active
15/16-bit fused prediction/prefix kernel and all eight input loaders.

These checkpoints establish the stated correctness coverage, not measured
performance or identical internal representation to C++.
This owner ran only targeted rustfmt, not Cargo/build/tests, and made no native
source, dependency, parent module or Cargo registration changes. Parent owns
typed Image conversion, eligibility, dispatch, metadata/container wrapping,
general region orchestration and central checks. Historical review questions
about late-backpatch failure, output-callback panic and little-endian callback
test coverage were outside this kernel slice and are not re-audited here.

## Entry Contract

- `encode(ImageView, &Options) -> Result<Vec<u8>>`: standalone raw JXL, integer
  sRGB or grayscale, optional unassociated alpha, last full-canvas Replace frame.
- `encode_frame(ImageView, &Options, is_last) -> Result<Vec<u8>>`: frame header,
  TOC and sections only. Caller must provide matching dimensions, channel layout,
  integer sample-word width and non-XYB metadata. The typed parent wrapper can
  also supply IEEE binary16 metadata for raw 16-bit words; standalone `encode`
  always writes integer metadata. Header has no animation duration,
  timecode, crop, name or saved-reference ID. Do not use it unchanged in an
  animation or custom-origin/reference context.
- `encode_with_pool(image, options, pool)` and
  `encode_frame_with_pool(image, options, is_last, pool)` use the caller's
  `rayon::ThreadPool`. The two non-pool APIs establish the existing strict serial
  execution scope and never enter the global Rayon pool.
- `ImageView` borrows interleaved bytes, with `usize` width, height, row_stride;
  `u8` channels and bit_depth; `bool` big_endian. Channels 1/2/3/4 mean G/GA/RGB/RGBA.
  Depths 1..=8 use one byte; 9..=16 use two. Strides may include padding; final
  row need only contain active samples. Narrow-depth values are checked.
- `Options { effort, oneshot, backend, limits }`: FJXL effort is **0..=127**, default
  **2**, not general encoder effort 1..11 or Modular tier. `oneshot` defaults true;
  false selects source non-oneshot sampling/palette policy. This policy is
  independent of whether pixels come from ImageView or a region provider.
- `Backend::Auto` uses detected AVX2, otherwise scalar. `Scalar` fixes eight-sample
  chunks. AVX2 uses the source sixteen-sample chunk policy. Consequently sampling,
  RLE boundaries, trees and bytes may differ across these profiles.
- Default `Limits`: 64 Mi pixels, 512 MiB returned output, 1 Mi total TOC entries.
  Dimensions must be nonzero and at most 2^30. Output accounting includes header
  and TOC. Up to sixteen AC groups are processed concurrently; this is not a
  strict total-allocation cap (retained sections, final output, DP scratch and
  per-worker scratch also coexist). Violations return errors, not silent retries.

## Region / Output Contract

Source anchors: `enc_fast_lossless.h:67..119` lifecycle and output API,
`enc_fast_lossless.cc:345` reservation, `:3788` LLPrepare, `:4027` LLProcess,
`:939` WriteOutput, `:4419` OutputFrame; `encode.cc:1331` outer finalization.
All new entry points are re-exported from `fast_lossless`; they are FJXL-specific,
not replacements for the parent's general metadata/region/streaming interfaces.

```rust
prepare_frame<'a, S: RegionSource + ?Sized>(source: &'a S, options: &Options)
    -> Result<PreparedFrame<'a, S>>
PreparedFrame::process(self) -> Result<ProcessedFrame>
PreparedFrame::process_with_pool(self, pool: &rayon::ThreadPool)
    -> Result<ProcessedFrame>
ProcessedFrame::prepare_header(self, add_image_header: bool, is_last: bool)
    -> Result<EncodedFrame>
EncodedFrame::write_output(&mut self, destination: &mut [u8]) -> Result<usize>
PreparedFrame::process_to(self, output: &mut impl OutputProcessor,
    add_image_header: bool, is_last: bool) -> Result<usize>
```

- `FrameInfo { width, height, channels, bit_depth, big_endian }` is metadata only,
  with the same integer-word constraints as ImageView. `Region { x, y, width,
  height }` is in pixels. `RegionSource: Sync` supplies `info()` and
  `get_region(Region) -> Result<Self::Buffer<'_>>`; its GAT buffer implements
  `RegionBuffer::{bytes, row_stride}`. The provider's buffer Drop releases the
  lease. The encoder never retains that borrowed pixel view beyond the lease.
  ImageView implements the trait without copying pixels; its full stride is
  checked before cropping a region.
- `info()` is queried once per prepare. Source format/content must remain
  immutable throughout preparation and processing. Requests are entire AC
  groups, origins divisible by 256, dimensions 1..=256, with partial sizes only
  at right/bottom edges. Even a zero-row sampling pass acquires/releases its full
  group, as source does. This is a **re-readable region API**, not a one-way row
  iterator. A successful get is dropped once on success, validation failure,
  encoding error or unwinding. Errors before get returns a buffer remain the
  provider's cleanup responsibility; Drop/release callbacks must not panic.
- Prepare is serial. At depth 8, effort >=2 and oneshot=true it scans regions for
  the existing palette gate until collision; histogram sampling is unchanged:
  all groups for oneshot or effort >=64, center group otherwise. Preparing
  retains format, codes, optional palette and global compressed bits, no input
  leases or whole-image sample buffer. Before palette index encoding, each
  fetched pixel is checked against its dictionary entry, preventing a changed
  source from silently aliasing a palette index. Other content changes violate
  the immutable-provider contract and are not detected by a full-frame hash.
- `process` and `process_to` consume the prepared state, including on failure.
  The pool variants use only the explicit caller pool and respect an enclosing
  serial scope. Input leases are acquired inside workers and need not be Send;
  at most sixteen can coexist. `process_to_with_pool(output, add_image_header,
  is_last, pool)` additionally requires Send for the output processor. Output
  callbacks are serialized, never concurrently invoked by group workers.
- `process` retains all **compressed** sections. `prepare_header` detaches output
  from the provider; `output_size` is exact, `remaining_size` tracks its cursor,
  `write_output` accepts any nonempty slice and returns zero only at EOF. An
  empty slice before EOF errors without advancing. No C++ 32-byte destination
  minimum or +32 spill padding is necessary. `write_to` emits the same remaining
  bytes through callbacks, without seek. This buffered-compressed-output route
  is explicitly not bounded-memory streaming encoding.
- `EncodedFrame::reset_header(add_image_header, is_last)` implements source
  prepared-header reuse: rebuild header/reset cursor without rereading pixels
  or re-encoding sections. Use a fresh/rewound destination. `into_bytes` collects
  the complete stored frame, irrespective of cursor progress. Neither operation
  silently retries a failed input/output transaction.
- `OutputProcessor` provides absolute `position: u64`,
  `get_buffer(suggested_size) -> Result<&mut [u8]>`, `release_buffer(written)`,
  `seek(position)` and `set_finalized_position(position)`. One successful get
  receives exactly one release; early failure/unwind before committing uses
  release(0). A release error is not retried. A zero-length acquired buffer is a
  terminal stop, not pending/backpressure. Released byte counts and seek targets
  are checked against the processor's resulting position, with checked offsets.
  Callback failure/unwind poisons an EncodedFrame cursor until explicit header
  reset; partial external output is invalid and must be discarded or rewound.
- `process_to` is the truly bounded-output path. For multiple groups it requires
  a seekable target, reserves frame header/TOC/global space, emits AC groups in
  batches of <=16, backpatches the prefix and restores position to frame end.
  It retains O(TOC entries) sizes, small global/palette data, <=16 compressed AC
  group results and per-worker coding scratch, not the entire input or output.
  The provider and output processor own any additional storage. One-group frames
  need no seek and retain only that single <=256x256 group's compressed payload.
  There is no implicit no-seek fallback buffering a whole multi-group output.
- Reservation is source `ComputeAcGroupDataOffset`: TOC byte-size boundaries
  1024, 17408, 4211712; encoded widths 12/16/24/32 bits including selectors;
  max AC width 24 bits; empty DC/AC-global width 12 bits; max frame-header size
  five bytes. Global minimum is advanced to the next bucket as needed to absorb
  `1 + ceil(12 * ac_groups / 8)` padding without changing its TOC width. Actual
  frame/global padding is backpatched after group sizes are known. Seek output
  may contain more global padding than collected output; pixels, not whole-JXL
  hashes, are the acceptance target.
- `max_output_bytes` counts the entire newly written frame/codestream, including
  image header when requested, frame header, TOC and reservation padding, but
  excludes an existing outer prefix. A budget failure may occur after writing
  AC data; output is then invalid, never finalized and never silently retried.
  Limits are not a strict total-allocation cap. Allocation/scratch/provider
  storage and buffered compressed-output copies may coexist.
- **FJXL never calls set_finalized_position.** Source LLProcess likewise leaves
  that to the outer encoder, which first backpatches container box headers.
  Parent must finalize only after its own outstanding prefix writes complete.
  `add_image_header=false` writes only a frame at the current output offset and
  preserves the same metadata/full-canvas restrictions as encode_frame. Generic
  image geometry, container muxing, output atomicity and public source selection
  remain the parent's integration responsibility.

## Implemented Routes

- Source gradient predictor with signed 16-bit intermediate wrapping through
  depth 14 and 32-bit profile above 14, exact signed packing, reversible YCoCg,
  gray/GA/RGB/RGBA conversion, endian handling, group boundaries and narrow tails.
- Source chunk-prefix zero-run accumulation, minimum run eight, literal-zero
  plus distance-one LZ representation, HybridUint 000/400, cross-row runs and
  omission of open final runs from sampled histograms.
- Sampling every group for oneshot or effort >=64, otherwise center-group rows;
  source frequency priors, x256 sampled counts, per-depth symbol limits, four
  independent constrained Huffman tables, DP reverse-trace tie order, canonical
  codes, code-length tree serialization and 205-zero alphabet gap.
- Palette attempt is default-enabled at depth 8, effort >=2, oneshot true. Hash
  collisions and more than 511 nonzero colors reject it; all-zero is always the
  implicit first entry. Source dense-grayscale gate, alpha*luma comparison,
  dictionary gradient coding, index-channel shift for single versus multiple
  groups, and source palette warm-up-row sampling behavior are retained.
- Standalone metadata, fixed global tree/entropy headers, RCT/palette signaling,
  256x256 AC groups, 2048x2048 DC geometry, unpadded channel concatenation, byte
  padding at section boundaries, TOC sizes and empty DC/AC-global entries.
- Native AVX2 YCoCg and prediction/packing kernels are active, not standalone
  helpers. Runtime feature detection guards all intrinsics; tails are scalar
  with no padded reads. Native lanes explicitly preserve narrow wrapping.
- `simd.rs` registers `input_simd.rs` as an x86-only child; `Kernel::load_row`
  runtime-gates the source `SIMDVec16::LoadG/GA/RGB/RGBA8/16` ports from
  `enc_fast_lossless.cc:1666..1830`, `:1831` SwapEndian, `:1655` Upcast and
  `:3193..3413` FillRow tails. All eight storage/layout variants are active in
  both histogram sampling and encoding. The native helper checks format and
  address bounds before mutation, loads only complete sixteen-pixel blocks,
  handles exact tails and zero-extends unsigned words including 0x8000..0xffff.
  Both paths preserve the left border, unused channels and cells outside the
  selected segment. The previous safe scalar loader remains the non-AVX2 or
  forced-Scalar fallback. Existing caller validation checks narrow-depth values;
  loading does not clamp or mask them. Palette input still bypasses this loader.
  No whole-image buffer or extra region acquisition was introduced.
- Active `Kernel::load_transformed_row` connects the additive helper's source
  `StoreYCoCg`/`FillRowRGB(A)` fusion (`enc_fast_lossless.cc:3272..3413`) to the
  shared sampling/encoding route. Complete AVX2 input blocks transform RGB in
  registers before storing Y/Co/Cg, with endian conversion first. Depths <=14
  use packed i16 arithmetic and signed Co/Cg widening; depths 15/16 zero-extend
  input words before i32 YCoCg. G/GA and alpha remain unchanged. Scalar fallback
  uses the previous raw-load/transform sequence. No second YCoCg pass follows
  the helper, and the raw `load_row` contract remains unchanged.
- Active AVX2 entropy now ports source `TokenizeSIMD`, `HuffmanSIMDUpTo13`,
  `HuffmanSIMD14`, `HuffmanSIMDAbove14`, `StoreSIMD*`, `Bits16/32::Merge` and
  `StoreToWriter`. Sixteen-bit tokens use the four nibble lookup tables;
  wide tokens extract the exponent of an exact f32 conversion for residuals
  <=262143, with no approximate logarithm. Source depth-14 and >14 paired
  eight-bit Huffman codes retain their high-bit reconstruction/remapping.
- The <=8 profile merges 16->32->64 bits; depths 9..14 interleave Huffman and
  extra-bit fields before merging; >14 combines 32-bit fields then merges to
  64-bit words. Tail lanes are initialized and clipped. A sliced skipped prefix
  changes lane grouping, not bit order or the existing chunk/RLE policy.
- Source short RLE cache is active for actual run lengths 8..39. Longer runs
  merge literal zero, LZ code and extra bits into one bounded write. Scalar
  literals likewise merge Huffman and extra bits. These are default-enabled;
  `Backend::Scalar` still bypasses every intrinsic but uses the scalar cache.
- Active 15/16-bit `Kernel::predict_chunk_wide` ports
  `enc_fast_lossless.cc:2620` PredictPixels<SIMDVec32>, `:1475`
  Mask32::CountPrefix and `:3109` ChannelRowProcessor::ProcessChunk. Each native
  chunk is still sixteen samples, comprising two eight-lane vectors. Prediction,
  residual packing and equality-mask generation share the same vector values;
  zero-extension of the eight-bit movemask before inversion yields a prefix of
  eight for an all-zero vector. The second prefix contributes only when the
  first vector was entirely zero. No separate residual scan is performed for
  full native vectors. Short tails use bounded scalar prediction/counting,
  without external row padding or extra reads. Scalar still uses eight-sample
  chunks and the existing scalar prediction arithmetic.
- `Processor::predicted_wide_row` feeds the returned prefix into the unchanged
  source RLE state rules. It uses a sixteen-value residual scratch buffer and
  is active through `pixels::process` for both sampled histograms and encoded
  pixels. Run accumulation crosses rows exactly as before; sampled final runs
  remain omitted, while encoder final runs are emitted. This does not fuse input
  loading with prediction or change Huffman/bit-packing code.
- Active <=14-bit `Kernel::predict_chunk_narrow` ports
  `enc_fast_lossless.cc:2620` PredictPixels<SIMDVec16> and `:1537`
  Mask16::CountPrefix. Each full native chunk uses sixteen packed i16 lanes for
  subtraction, addition, signed comparison, residual subtraction and zigzag.
  The existing i32 row storage is packed with signed `packs_epi32`, followed by
  `permute4x64(0xd8)` to restore order. All inputs are signed-i16-representable,
  so this adapter never saturates. Residuals widen as unsigned u16 words for the
  existing sink; no residual scan is needed for full vectors. Zero-extension of
  the 32-bit byte movemask to u64 precedes inversion and trailing-zero count,
  divided by two; an all-zero vector returns sixteen, not zero or thirty-two.
- `Processor::predicted_narrow_row` shares the existing chunk/RLE implementation
  with the wide route. It is active for ordinary pixels and palette indices in
  `pixels::process`, and for `palette_data` dictionary rows, including widths
  greater than 256. Palette sampling still includes its warm-up row. Bounds are
  checked before native loads, only complete vectors are read, and short tails
  emulate the packed lane operations without reading padding. Scalar continues
  to use eight-sample chunks. Previous widened prediction and residual-row
  processing are retained only as test baselines, not duplicate active routes.

## Deliberate Safety / Representation Differences

- A nonzero color hashing to slot zero rejects palette: the source uses zero as
  its empty sentinel and can alias the mandatory zero entry for this input.
  Rejection retains exact pixels via nonpalette FJXL, including hidden RGB.
- Equal alpha*luma scores have no source secondary key; `std::sort` tie order is
  unspecified. Rust keeps hash enumeration order on equal scores. No new numeric
  tie key, float precision upgrade or approximate score is introduced.
- Source `Write(8, 256)` in alphabet declarations spills a high bit onto the next
  mandatory one bit. The bounded Rust writer emits its low eight zero bits and
  lets that following field supply the one, preserving the effective bitstream.
- Safe bounded BitWriter and merged group sections replace unsafe spill stores
  and four retained C++ channel writers. This is a representation change, not
  a replacement entropy algorithm. No whole-JXL byte equality goal applies.
- Source `WriteMultiple` accepts 64-bit merged words. The shared bounded writer
  accepts at most 56 bits per call, so the local adapter splits only words wider
  than 56 into low/high writes. This preserves bit order without exposing shared
  writer internals or requiring changes outside the owned subtree.
- Packed and promoted scalar residual arithmetic differ on arbitrary signed
  i16 inputs: `pixel=32767, prediction=-32768` produces packed residual 1, but
  source scalar's promoted subtraction followed by u16 assignment produces
  65534. The new packed-chunk scalar emulation preserves the former; the legacy
  scalar predictor remains separately tested for the latter. This is not a
  public pixel difference: for valid <=14-bit samples, with M=(1<<depth)-1,
  Y/gray/alpha lie in [0,M], Co/Cg in [-M,M], and M<=16383. Pairwise neighbor
  differences fit i16, clamped prediction stays in the channel's interval, and
  residual magnitude is <=32766. Existing public/region sample validation
  establishes this precondition. The 15/16-bit and F16-word routes stay wide.
- Current i32 rows and u32 sink residuals require packing before narrow
  prediction, unsigned widening afterward, and repacking for narrow entropy.
  These are representation overhead, not missing prediction, prefix, RLE or
  entropy algorithms. This slice deliberately does not overhaul storage or
  sink interfaces; performance impact has not been measured.

## Input Transform Helper Checkpoint

The additive x86-only `input_simd::load_transformed_row(image, x, y, width,
rows) -> Result<()>` ports frozen `enc_fast_lossless.cc:3193..3413` FillRow and
`:3272..3310` StoreYCoCg. Its unsafe precondition is detected AVX2; format,
row/segment bounds, checked offsets and width <=256 are verified before any
destination write. It preserves the existing `[[i32; 257]; 4]` interface,
left border, unused channels and cells past the segment. Width zero is valid.

- G/GA retain unsigned samples; RGB/RGBA return Y/Co/Cg[/A]. Full sixteen-pixel
  blocks stay in deinterleaved registers through the color transform, with
  endian swapping before arithmetic. There is no intermediate raw RGB row
  store/reload and no fusion with prediction.
- Depths <=14 use the source wrapping i16 add/subtract and arithmetic shifts.
  Negative Co/Cg are sign-extended into Rust's i32 rows; Y and alpha use unsigned
  extension. Depths 15/16 zero-extend input words, including 0x8000..0xffff,
  BEFORE i32 subtraction/shifts. Source scalar equations handle bounded tails.
  Existing per-region validation still enforces declared sample ranges; the
  helper does not clamp bad samples or duplicate that scan.
- `load_row` retains its raw unsigned contract. `Kernel::load_transformed_row`
  and `pixels::process` now connect the helper to both sampling and encoding,
  replacing load-plus-YCoCg without a second transform. Palette input bypasses
  the helper; the existing scalar load/transform fallback remains intact.
- Four new AVX2-gated tests cover every valid sample value at depths 1..16 in
  every channel/endian layout, all widths 0..256 and byte alignments at storage
  and 14/15-bit boundaries, extreme/odd signed chroma, hidden RGB and all 65536
  signed-extension words. Existing strided-subrow and invalid-bound tests now
  exercise both entries. One new Windows guard-page test covers exact input
  and output boundaries for G/GA/RGB/RGBA, depths 1/8/9/14/15/16, every width,
  both byte orders and both protected edges. The oracle uses independent byte
  decoding and nonnegative integer averages, not production YCoCg routines.

At the original helper handoff, its owner ran scoped rustfmt only, with no
Cargo/build/tests. The subsequent integrated stage-2 central release run passed
all 60 ordinary FJXL tests, including these five helper additions, followed by
108 official pixel checks and 32 typed F16-word checks. Thus the fused helper has
both primitive and active-codec evidence, not merely earlier raw-loader results.
No new ARM, AVX512, dependencies, C++ linkage or whole-JXL identity goal is introduced.

## Verification

- The registered files include the unit suites below and two ignored oracle
  suites. All 60 ordinary tests passed at stage 2. The standalone suite has 38
  encodes per decoder; all 76 official checks passed in the explicit stage-2
  rerun. The oracle takes the OS path list
  `RS_JXL_DJXL`; invoke the `fast_lossless::reference_tests` filter with `--ignored`.
  It checks all components including transparent RGB, stride/endian handling,
  depths 1/7/8/9/13/14/15/16, palette and nonpalette, one/multiple AC and DC groups,
  effort 0/1/2/64/127 and oneshot/non-oneshot. Depths 8/16 compare exact PNG samples;
  other depths use official NPY and exact recovered original integers, including
  all hidden RGB. There is no one-unit tolerance or discarded-bit comparison.
- Fifteen new `fast_lossless::regions::tests` cover full-group callback geometry,
  effort 0/2/63/64/127 sampling, palette dictionaries, release on errors/unwind,
  changed-palette rejection, 1/7/31/32/33-byte cursors, EOF and header reuse,
  balanced output get/release, terminal stop/failure, absolute position overflow,
  seek/backpatch at nonzero positions, caller-controlled finalization, output
  budgets, TOC bucket boundaries, serial/pool equality and live-region bounds.
  Providers allocate only requested regions; whole expected images exist solely
  in test comparison helpers. These tests, including the header-reuse correction,
  first passed in the earlier central 1325-pass, zero-failure core run and are
  also covered by the successful stage-2 FJXL run.
- New ignored oracle:
  `fast_lossless::reference_tests::official_region_incremental_and_seek_output`.
  Four fixtures x Scalar/Auto x incremental/seek = 16 encodes per decoder
  (32 exact checks with two decoders), all passed in the explicit stage-2 release
  rerun. Together with the standalone suite this gives 108 exact pixel checks.
  It covers palette
  and hidden RGBA, GA7/RGB14 NPY, RGBA8/16 PNG, partial 256x256 edges, multiple
  DC groups, a second sixteen-AC-group batch and frame-only output after parent
  metadata at a nonzero seek position. It uses a caller four-thread pool and
  generated region leases. Existing regression oracle remains
  `fast_lossless::reference_tests::official_scalar_and_native_fjxl_profiles`.
- Six entropy-slice tests compare independent bit-by-bit output across all code
  profiles, skewed frequencies, tail/skip combinations and initial bit offsets;
  verify RLE cache/long-run boundaries, output limits and invalid residuals;
  compare complete encoder sinks with fixed chunk boundaries; check full 64-bit
  merged words and carries; and enumerate all 65536 narrow plus 262144 wide
  residual tokens against integer leading-zero arithmetic. Public APIs and
  source chunk-size selection were unchanged by this slice.
- New filter `fast_lossless::tests::wide_prediction` contains five tests:
  independent clamp/zigzag lanes for every prefix and length 0..16, destination
  canaries and invalid slice bounds, fixed-chunk RLE events and histogram parity,
  open final-run omission, all four layouts at depths 15/16 in both byte orders,
  first-row/left-boundary and sampling warm-up contexts, fixed-chunk encoded-bit
  equivalence against the previous residual-row route, and output-budget errors.
  Existing standalone/region official suites and parent typed F16-word suites
  remain the independent pixel oracles. The wide/input checkpoint has passed the
  central runs recorded above, and stage 2 passed the explicit official reruns.
  Timing/performance is unmeasured.
- Registered helper filter `fast_lossless::simd::input_simd::tests` contains the
  original six portable-x86 AVX2-gated tests and two Windows guarded-memory
  tests, plus four new transformed-input tests and one new Windows guard test.
  They cover every u8/u16 word in each channel and byte order, every width and
  alignment, exact tails, absolute strided subrows, invalid format/buffer bounds
  without destination mutation, endian/upcast semantics, unchanged raw words,
  and protected input/output boundaries. The fused additions cover every valid
  per-channel sample range, widths/alignment/depths, extreme colors, odd negative
  chroma, every signed-i16 widening value and protected transformed-row bounds.
  The integration owner did not modify helper files or run their tests.
- `fast_lossless::tests::input_dispatch_preserves_guards_and_sampled_encoded_row_residuals`
  exercises the actual Scalar/Auto dispatch for all layouts at depths
  1/8/9/14/15/16, both byte orders, widths 0/1/15/16/17/255/256, nonzero x/y,
  padding and an exact unpadded final row. Independent expected samples verify
  border/remainder/unused-channel canaries for both raw and transformed dispatch;
  an independent average/difference oracle checks YCoCg. Expanded residuals agree through
  `pixels::process` with and without sampling warm-up, including hidden RGB.
- New filter `fast_lossless::tests::narrow_prediction` contains six tests:
  independent modulo-65536 arithmetic across every residual word and arbitrary
  neighbor lanes; explicit distinction from promoted scalar subtraction;
  independent valid-domain clamp/zigzag and previous-kernel equivalence for all
  prefixes/tails at depths 1/7/8/9/13/14; destination canaries and invalid lengths;
  fixed-chunk RLE traces, histograms, final-run omission, exact encoded bits and
  output limits; all layouts/endian/sample contexts; and palette dictionary
  widths 2/7/8/15/16/17/255/256/257/512 plus index sampling with warm-up retained.
  The tests do not weaken the existing arbitrary-i16 widened-kernel assertions.
  All six passed in the central stage-2 release run.

## Validation And Integration Boundaries

Historical central follow-up: the combined input-loader/wide-prediction slice passed all
49 affected debug unit tests, including Windows guard-page tests, and the full
debug/optimized workspace. Both official djxl versions passed 108 standalone
and region/output pixel comparisons, plus 32 typed F16-word comparisons.
Strict all-target Clippy passed after one test-only modulo-expression cleanup.
These are correctness checks, not whole-encoder performance measurements.

- That earlier combined slice has central Clippy/MSRV validation and a current-Rust
  x86 Linux compile check; no Linux runtime execution or whole-encoder
  performance measurement is claimed here. The stage-2 results recorded here
  are its 60 release unit tests, 108 explicit official pixel checks and 32
  explicit typed F16-word checks; earlier strict-check results are historical.
- Parent owns source tier-9 eligibility/explicit-override and metadata
  contracts beyond this kernel slice. Source search's backend
  baseline is separate. FLOAT16 raw-word/metadata integration is now implemented
  and has finite and NaN/infinity/payload word evidence above. No float mapping
  is claimed for standalone integer metadata.
- Input deinterleaving and wide prediction/prefix fusion are active and
  centrally verified. Packed narrow prediction/prefix fusion and source
  `FillRowRGB(A)` input/YCoCg fusion are also active and passed the combined
  stage-2 central unit and official verification. They are no longer missing
  source algorithm/entry-route blockers. The i32-row/u32-sink conversion costs are representation
  overhead, not missing algorithms. No packed-storage overhaul, ARM or AVX512
  extension is included, and no performance improvement is claimed without
  measurement.
- User explicitly limited subsequent work to x86 plus scalar: no additional ARM
  kernels or ARM tests. AArch64 currently uses scalar; NEON is not a completion
  blocker under this revised architecture requirement.
- Prepared-state, callback-input, seekable streaming/backpatch and incremental
  output now have active typed Rust entry points, with central unit and official
  pixel verification and strict lint/MSRV passing at the prior frozen checkpoint.
  Parent owns general ImageDescriptor/provider
  adapters, format conversion/eligibility and public container/geometry routes;
  this records ownership and call-contract boundaries, not a claim that those
  separate integrations are absent.

## Narrow-Depth Oracle Correction

Frozen `render_pipeline/stage_write.cc:264` explicitly adds ordered dither for
UINT8 output in `MakeUnsigned`, after multiplying normalized values by 255.
For the failed case-9 sample at x=1, y=0, channel=1, the original integer is 82
at depth 7. `dec_modular.cc:92` uses an f32 reciprocal and multiply, yielding
float word `0x3f254a95`; multiplying by 255 gives about 164.6457. Source dither
index `(y + c*13)%32 * 48 + (x + c*23)%32` is 648, whose table value is -0.44200.
The correct dithered PNG integer is therefore 164, not the old test's 165.

The corrected oracle uses `extras/enc/npy.cc:234`/`:312` float32 output for
non-storage depths, avoiding UINT8 quantization/dither altogether. Decode uses
`dec_modular.cc:584`'s reciprocal of `(2^depth-1)`. Multiplying each finite NPY
value by that exact integer range in f64 and rounding recovers the original
integer, which is compared with strict equality. The helper test enumerates
every value at every depth 1..16, proves distinct source normalized values stay
distinct, and checks exact recovery for both source f32-multiply and accurately
rounded f64-division projections. Another regression fixes the failed sample's
float bits and both dithered/undithered outcomes. A checked fixed-format NPY
parser rejects malformed dtype, shape, truncation and surplus payload.

No runtime subprocess, C++ binding, native codec linkage or substitute encoder
crate is used. Official djxl appears only in ignored development tests.
