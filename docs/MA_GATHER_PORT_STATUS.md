# MA Gather Port Status

Source: frozen read-only libjxl
`b866b3defec13494bfbaf705db93581cda118e37` under
`../jxl_python/libjxl`. Only `modular/training.rs`, `modular/sample.rs`,
the new local `modular/gather_samples.rs`, and this document are owned here.
No changes to `ma_train.rs`, `ma_samples.rs`, or `ma_columns.rs` are authorized.

## Default Storage

- `enc_ma.cc::PrepareForSamples/InitTable`: gather now reserves at every
  channel/stripe boundary using truncation of `fraction * rows * width + 64`.
  Tables use `next_power_of_two(floor(3 * (distinct + expected) / 2))`, with no
  512-slot floor or 1<<20 ceiling. Equal size retains the table. Changed size,
  including shrink, rebuilds empty slots in record order, omitting saturation.
  Column growth uses checked, fallible source 1.5x reservations. Underestimated
  random sample counts grow columns, never silently cap the population or resize
  the table mid-channel. Empty untouched populations allocate no table.
- `AddToTableAndMerge/AddWeightedToTableAndMerge`: selected pixels are quantized
  directly into byte dynamic-property columns, u32 static-property columns,
  `[token, nbits]` byte pairs, and saturated u16 counts. Only two probe positions
  are tried. Collisions retain records; u16 saturation starts another record.
  Per-pixel temporary keys/heap vectors and retained row keys are eliminated.
- `AppendSamples`: one prepare covers all distinct records in a completed wave.
  Shared quantizers and predictor order are checked before mutation. Weighted
  rows are merged directly from columns in ordered batch order; consumed input
  collectors and their tables drop individually. Selection allowance is charged
  once in gathering, never again in merging. The master table drops at finish.
- Source learning now consumes `PreparedSamples` directly through Parfit's
  crate-private `ma_train::train_prepared`. It does not reconstruct `Sample`
  rows, requantize properties, or deduplicate the gathered population again.
  The former compatibility adapter and its `[i32; 2]` raw static-ID column
  exist only under `cfg(test)` for comparisons against public `train`. Production
  gathering no longer allocates that extra 8 bytes per distinct record. The
  source record-size scheduling estimate remains an estimate, not an RSS bound.
  The learner safely converts residual pair columns to its existing token type;
  property, count, and threshold buffers transfer without that conversion.

## Parent Integration Contract

`training::gather_samples::PreparedSamples` is crate-visible, with owned fields:

- `predictors: Vec<Predictor>` in configured order.
- `quantization: PropertyQuantization` with property IDs and unchanged thresholds.
- `properties: Vec<GatherColumn>` in that order; `Packed(Vec<u8>)` for dynamic
  properties and `Static(Vec<u32>)` for IDs 0/1, even when interleaved.
- `residuals: Vec<Vec<[u8; 2]>>`, one column per predictor, `[token, nbits]`.
- `counts: Vec<u16>`, nonzero; `total: u64`, exact selected-pixel multiplicity.
- All column lengths equal `counts.len()`. Collision duplicates are intentional.
  No gather hash table or per-record key is part of the handoff.

Coordinated with Parfit through the parent, now wired:

1. Parfit owns `ma_train::train_prepared(samples:
   training::gather_samples::PreparedSamples, options: &TrainOptions) -> Result<Tree>`
   and the local `ma_prepared.rs` consumer. Public `train(&[Sample], ...)` is
   unchanged. The source caller was not switched until that hook was present
   and the parent reported its frozen checkpoint ready.
2. The consumer checks predictor/quantizer identity, property widths and bin
   bounds, all column lengths, nonzero u16 weights, exact total, i32::MAX weight
   bound, canonical residual token/extra-bit-count shape, and absence of raw
   multiplier constraints. Static bins must overlap the variable static range;
   quantized bins are not substituted for proof of raw IDs.
3. The consumer reuses constant-suffix assembly, variable static ranges,
   complete-search/leaf-budget behavior, and the empty-population seed. The
   caller still performs full-plane constant proof before excluding the suffix.
4. `SourceGather::collect_prepared` returns `master.finish()` by ownership.
   `learn_with_budget_and_gather` forwards it after `policy.apply_to_train`,
   with debug assertions for selected multiplicity and quantizer identity.
   Default serial, configured serial, channel and wave source collection share
   that path. No gather option still means serial collection; turning off
   parallel search preserves requested sparse/stripe semantics.
5. Legacy gathering still calls public `train` with original samples. The test
   wrappers `SourceGather::collect`, `gather_source`, and `into_samples` retain
   the old comparison interface only under `cfg(test)`. No raw rows or raw
   static-ID column exist in the production source gather-to-learner path.

No learner implementation was edited here. Removal or test-gating of the old
`ma_train::SampleBatchCollector` remains with its owner. No parent messaging
tool was available; coordination has been through the parent's turn messages.

## Optional Gather

New public `GatherOptions` fields, both source opt-ins:

- `stripes: usize`, default 1, validated 1..=16. Source gates: height >=512 and
  area >=2^18, then `min(stripes, height / 128)` integer-partitioned row ranges.
  Each range uses the whole-channel sampling fraction, skips the exact RNG
  prefix, warms WP for the preceding 16 rows, and selects only within its own
  range. Global neighbor/property/reference coordinates and IDs are retained.
  This intentionally approximates learning-only WP history, exactly as the
  source experiment does; it never changes pixel coding state or sample sites.
- `sparse: bool`, default false. Unselected pixels advance the RNG, WP prediction
  and error state, and previous unclamped gradient, but skip unused property
  calculations. References were already loaded only at selected pixels in Rust;
  that exact lazy behavior is retained at all densities, rather than regressing
  to the source's dense reference-row materialization above 0.25.

Existing `mode`, `workers`, and `batch_budget_bytes` keep their defaults. A null
pool or one-worker execution retains selected stripe/sparse semantics with
serial tasks. Optional controls are not discarded when parallel search is off.
Sampling fractions, +1024 expected-site policy, quantiles, population statistics,
candidate order, and local/shared selected-sample error limits are unchanged.
Public struct literals without `..GatherOptions::default()` need the two new
fields supplied by their owners. The parent owns the corresponding literal
update in `crates/rs-jxl/tests/reference.rs`; that test file was not edited here.
No parent option file requires new forwarding:
these fields travel inside the existing `ModularOptions::source_gather` value.

## Verification

Ten focused tests were added in the accepted gather checkpoint:

- `gather_samples::tests::source_table_floor_caps_and_prepare_lifetime`
- `gather_samples::tests::source_columns_saturation_append_and_adapter_preserve_objective`
- `gather_samples::tests::source_adapter_roundtrips_all_residual_token_shapes`
- `gather_samples::tests::source_append_rejects_mismatched_quantizers_and_total_limits`
- `gather_samples::tests::source_hash_order_and_collision_misses_do_not_evict_or_probe_further`
- `sample::tests::gather_stripes_use_source_gates_and_checked_row_cuts`
- `sample::tests::stripe_rng_warmup_does_not_change_selected_coordinates`
- `training::tests::sparse_gather_preserves_generic_samples_with_references_and_wp_presets`
- `training::tests::gather_stripe_warmup_matches_source_window_and_excludes_budget_charges`
- `training::tests::stripe_channel_wave_and_null_pool_paths_preserve_population_and_limits`

The direct-caller handoff adds
`training::tests::prepared_source_handoff_preserves_schema_constants_and_selected_budget`.
It checks actual collector/consumer schema agreement and exact totals, compares
the prepared learner and full source caller with the public-row oracle, covers
zero/fractional/full sampling and constant bypass on/off, and verifies both
default serial and explicit sparse gather charge the selected budget once.

Only scoped edition-2024 rustfmt (`skip_children=true`) is run locally on the
three owned Rust files. No Cargo, build, test, benchmark, decoder, or commit
command is run here. Parent reports the accepted checkpoint passed 1,194 core,
44 CLI, and 47 integration/doc tests, including every new gather test. Parent
also reports 30 official JPEG metadata checks passing; these are separate from
pixel-oracle coverage. Later caller and active stripe/sparse results are recorded
below. No performance or complete-migration claim is made.

Parent's first central compile reported only test closure lifetime errors in
`stripe_channel_wave_and_null_pool_paths_preserve_population_and_limits` and no
production errors. The test now explicitly types the callback's borrowed
`&SampleBudget` parameter and uses named budget bindings instead of temporaries.
The successful central run reported above includes that lifetime fix.

Parent subsequently reported 1,211 core passes including Parfit's prepared
consumer tests, before this caller was wired. Parfit confirmed that consumer
checkpoint stable. This caller handoff changes only `training.rs`, the owned
`gather_samples.rs`, and this document; neither `sample.rs` nor any learner
file was changed in this handoff. The 1,211-core result predates the caller
wiring and its new test. Only scoped edition-2024 rustfmt/check is run here;
no local build or test command is run.

### Accepted Direct-Caller Checkpoint

Parent subsequently reports the wired direct-prepared caller passed the full
workspace: 1,262 total core tests, 11 ignored. All 12 active 513x513 stripe/sparse
official pixel-oracle checks passed, as did 48 new scoring checks. These results
cover the caller handoff rather than only the standalone prepared consumer.
They are parent-reported central results, not commands executed by this agent.

The gather table floor/cap, retained row-key adapter, production row conversion,
stripe RNG/warm-up, and sparse traversal gaps described in this document are
closed by this validated checkpoint. Remaining implementation differences are
optimization-only: eager computation of both gather hashes instead of source
fast dedup's deferred second hash; scalar insertion/append instead of the source
opt-in sample-batch and column-batch merge tools; and the consumer's safe residual
column conversion rather than a common residual element type. No missing sample
budget is replaced with a sampling fallback. Full pixel-port acceptance and
performance parity are not inferred from this checkpoint. No additional ARM
work or tests are performed.

The subsequent central clippy report identified three test-only lints. The two
single-range expectations in `sample.rs` now collect `std::iter::once(range)`;
the sparse traversal comparison in `training.rs` enumerates the first two channel
references instead of indexing them with a range loop. Scoped edition-2024
rustfmt check passes for those two owned files. Production code is unchanged;
central clippy confirmation remains with the parent.
