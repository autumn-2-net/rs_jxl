// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; the complete upstream notice is in docs/PORTING_KERNELS.md.

//! Shared/local LearnTree integration from frozen b866b3defec13494bfbaf705db93581cda118e37
//! enc_encoding.cc and enc_ma.cc. The caller supplies every transformed stream
//! in the learning domain; IDs and metadata channels are never renumbered.
//!
//! Source mode performs the geometric quantile pre-pass before opt-in constant bypass,
//! then per-channel random training with complete predictor-state traversal.
//! An explicit selected-sample limit is an error ceiling, not a sampling target.
//! By default there is no additional selected-sample count cap.
//! Selected samples enter compact columns directly, using source-sized tables
//! prepared at channel/stripe and wave-append boundaries. Source learning
//! transfers those columns directly to ma_train::train_prepared; only legacy
//! learning and test oracles use raw Sample rows.
//! Local sample charging is unnecessary when the checked visited-pixel upper
//! bound fits the local limit. Shared ceilings still charge each selected pixel;
//! tighter local ceilings retain the original per-sample failure accounting.
//! Optional sparse traversal preserves causal state; optional stripes use the
//! source sixteen-row learning-only WP warm-up. Both are disabled by default.
//! Legacy mode retains collect_samples' explicit per-stream midpoint target.
//! Its local and shared selected-sample budgets are reserved before collector
//! allocation/traversal, using encode::legacy_sample_count's checked target.
//! Source gathering dispatches the exact Weighted/Gradient candidate pair once,
//! retaining generic candidate order for all other sets and unchanged WP state.
//! Serial and channel/wave gathering share source no-edge row segmentation:
//! y > 1, width > 8, no configured reference slots, and x in 2..width-2.
//! Safe row slices replace per-pixel edge tests only in that interior. Borders,
//! narrow planes and reference-enabled rows keep the general neighbor loader.
//! Source AddSample narrows the wide predictor difference to i32 before token
//! quantization. This is not the encoder's multiplier path: that path must do
//! wide subtraction/division before narrowing its quotient for PackSigned.
//!
//! A zero source fraction is LearnTree's no-samples case: predictor zero in the
//! configured candidate order, plus proven constant leaves when opted in. PredefinedTree's
//! WPFixedDC/GradientFixedDC trees are separate tree-kind choices requiring a
//! bit depth; zero learning fraction alone does not select those source trees.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::execution;

use crate::{Error, Result};

use super::encode::{self, ModularOptions};
use super::group::Channel;
#[cfg(test)]
use super::ma_train::TrainOptions;
use super::ma_train::{self, ChannelView, PropertyQuantization, Sample};
use super::predictor::{Neighbors, Predictor, clamped_gradient, predict_one};
use super::sample::{self, ChannelSampler, GatherOptions, SamplingPolicy, SamplingStats, TreeMode};
#[cfg(test)]
use super::tree::Properties;
use super::tree::Tree;
use super::weighted;

#[cfg(test)]
#[path = "training_budget_mode_tests.rs"]
mod budget_mode_tests;

#[path = "gather_samples.rs"]
pub(crate) mod gather_samples;
use gather_samples::{PreparedSamples, SourceSampleCollector};

pub(super) struct TrainingStream<'a> {
    pub channels: &'a [Channel],
    pub wp: weighted::Header,
    pub stream_id: usize,
}

/// One selected-sample ceiling shared by independent learners. Deduplication
/// must not make additional selected pixels disappear from this accounting.
pub(super) struct SampleBudget {
    remaining: AtomicUsize,
}

impl SampleBudget {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(limit),
        }
    }

    fn consume(&self, count: usize) -> Result<()> {
        self.remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                remaining.checked_sub(count)
            })
            .map(|_| ())
            .map_err(|_| Error::Limit("grouped MA selected-sample budget exceeded"))
    }
}

/// Learn from ALL supplied streams, sorted by actual Modular stream ID.
/// Caller-owned planes stay borrowed throughout preparation and collection.
pub(super) fn learn(streams: &[TrainingStream<'_>], options: &ModularOptions) -> Result<Tree> {
    learn_with_gather(streams, options, options.source_gather.as_ref())
}

/// ModularGenericCompress writes a local tree, including in a global stream
/// without a shared tree. Its decoder bound uses transformed local samples.
#[cfg(test)]
pub(super) fn learn_local(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
) -> Result<Tree> {
    learn_local_with_history(streams, options, None)
}

pub(super) fn learn_local_with_history(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    history: Option<&super::weighted_history::History<'_>>,
) -> Result<Tree> {
    let samples =
        streams
            .iter()
            .flat_map(|stream| stream.channels)
            .try_fold(0usize, |sum, channel| {
                sum.checked_add(channel.samples.len())
                    .ok_or(Error::Limit("local MA tree population overflow"))
            })?;
    let mut local = options.clone();
    local.max_merged_tree_leaves = Some(
        options
            .tree_leaf_limit()
            .min(ma_train::local_tree_leaf_limit(samples)),
    );
    if let Some(history) = history {
        if let Some(gather) = options.source_gather.as_ref() {
            gather.validate()?;
        }
        let tree = learn_impl(
            streams,
            &local,
            None,
            true,
            options.source_gather.as_ref(),
            &[history],
        )?;
        if tree.leaves() > local.tree_leaf_limit() {
            return Err(Error::Limit("merged MA tree exceeds leaf budget"));
        }
        Ok(tree)
    } else {
        learn(streams, &local)
    }
}

/// Explicit source gather policy passed through local and grouped learning.
/// None retains the original serial gather, including grouped learning.
pub(super) fn learn_with_gather(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    gather: Option<&GatherOptions>,
) -> Result<Tree> {
    learn_with_gather_and_histories(streams, options, gather, &[])
}

pub(super) fn learn_with_histories(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    histories: &[&super::weighted_history::History<'_>],
) -> Result<Tree> {
    learn_with_gather_and_histories(streams, options, options.source_gather.as_ref(), histories)
}

fn learn_with_gather_and_histories(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    gather: Option<&GatherOptions>,
    histories: &[&super::weighted_history::History<'_>],
) -> Result<Tree> {
    if let Some(gather) = gather {
        gather.validate()?;
        if options.sampling.is_none() {
            return Err(Error::InvalidInput(
                "MA gather scheduling requires source sampling".into(),
            ));
        }
    }
    // A block plan owns its complete bounded stream/channel dispatch. Optional
    // learner sharding must not impose mandatory leaves or duplicate its scan.
    let tree = if options.block_context.is_some() {
        learn_with_budget_and_gather(streams, options, None, false, gather)?
    } else if let Some(grouping) = &options.tree_groups {
        if let Some(gather) = gather {
            super::tree_groups::learn_with_gather(
                streams,
                options,
                &options.tree_group_splits,
                grouping,
                Some(gather),
            )?
        } else {
            super::tree_groups::learn(streams, options, &options.tree_group_splits, grouping)?
        }
    } else {
        learn_impl(streams, options, None, true, gather, histories)?
    };
    if (options.max_merged_tree_leaves.is_some() || options.ma_max_leaves.is_some())
        && tree.leaves() > options.tree_leaf_limit()
    {
        return Err(Error::Limit("merged MA tree exceeds leaf budget"));
    }
    Ok(tree)
}

pub(super) fn learn_with_budget(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    budget: Option<&SampleBudget>,
    parallel_search: bool,
) -> Result<Tree> {
    learn_with_budget_and_gather(streams, options, budget, parallel_search, None)
}

pub(super) fn learn_with_budget_and_gather(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    budget: Option<&SampleBudget>,
    parallel_search: bool,
    gather: Option<&GatherOptions>,
) -> Result<Tree> {
    learn_impl(streams, options, budget, parallel_search, gather, &[])
}

fn learn_impl(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    budget: Option<&SampleBudget>,
    parallel_search: bool,
    gather: Option<&GatherOptions>,
    histories: &[&super::weighted_history::History<'_>],
) -> Result<Tree> {
    options.validate()?;
    let _learn_scope = crate::profile::scope("ma.learn");
    if streams.is_empty() {
        return Err(Error::InvalidInput(
            "empty MA training stream domain".into(),
        ));
    }
    let mut ordered: Vec<_> = streams.iter().collect();
    ordered.sort_unstable_by_key(|stream| stream.stream_id);
    let mut views = Vec::new();
    let mut previous_views = super::fast_context::Previous::new();
    let cross_channel = options.previous_channels > 0
        && options
            .fast_context
            .as_ref()
            .is_some_and(|fast| fast.cross_channel);
    let mut max_channels = 0;
    let mut previous_id = None;
    for stream in &ordered {
        if stream.stream_id > i32::MAX as usize || previous_id == Some(stream.stream_id) {
            return Err(Error::InvalidInput(
                "invalid/duplicate MA training stream ID".into(),
            ));
        }
        previous_id = Some(stream.stream_id);
        if stream.channels.len() > i32::MAX as usize {
            return Err(Error::Limit("MA training channel IDs exceed i32"));
        }
        max_channels = max_channels.max(stream.channels.len());
        for (channel, plane) in stream.channels.iter().enumerate() {
            let area = plane
                .width
                .checked_mul(plane.height)
                .ok_or(Error::Limit("MA training plane dimensions overflow"))?;
            if plane.samples.len() != area {
                return Err(Error::InvalidInput(
                    "MA training plane dimensions/buffer mismatch".into(),
                ));
            }
            if cross_channel {
                if let Some(reference) = super::fast_context::previous_view(
                    stream.channels,
                    channel,
                    stream.stream_id as u32,
                ) {
                    previous_views.insert((stream.stream_id as u32, channel as u32), reference);
                }
            }
            views.push(ChannelView {
                channel: channel as u32,
                group: stream.stream_id as u32,
                width: plane.width,
                height: plane.height,
                stride: plane.width,
                pixels: &plane.samples,
                is_meta: plane.hshift < 0 || plane.vshift < 0,
            });
        }
    }

    if let Some(block) = &options.block_context {
        // Keep the existing explicit group-domain contract even though this
        // experiment compiles one plan rather than separate shard learners.
        let splits = &options.tree_group_splits;
        if !splits.is_empty()
            && (splits.len() < 2
                || splits.iter().any(|&id| id > i32::MAX as usize + 1)
                || splits.windows(2).any(|p| p[0] >= p[1])
                || ordered[0].stream_id < splits[0]
                || ordered.last().expect("nonempty domain").stream_id
                    >= *splits.last().expect("nonempty splits"))
        {
            return Err(Error::InvalidInput(
                "invalid block-context MA stream group domain".into(),
            ));
        }
        let samples = views.iter().try_fold(0usize, |n, v| {
            n.checked_add(v.width * v.height)
                .ok_or(Error::Limit("block-context population overflow"))
        })?;
        if options.sample_limit.is_some_and(|limit| samples > limit) {
            return Err(Error::Limit(
                "block-context full-scan sample budget exceeded",
            ));
        }
        if let Some(budget) = budget {
            budget.consume(samples)?;
        }
        let limit = options
            .tree_groups
            .as_ref()
            .map_or(options.tree_leaf_limit(), |groups| {
                options.tree_leaf_limit().min(groups.max_merged_leaves)
            });
        return super::block_context::learn(&views, limit, block);
    }

    if let Some(fast) = &options.fast_context {
        let samples = views.iter().try_fold(0usize, |n, v| {
            n.checked_add(v.width * v.height)
                .ok_or(Error::Limit("fast-context population overflow"))
        })?;
        if options.sample_limit.is_some_and(|limit| samples > limit) {
            return Err(Error::Limit(
                "fast-context full-scan sample budget exceeded",
            ));
        }
        if let Some(budget) = budget {
            budget.consume(samples)?;
        }
        return super::fast_context::learn_with_previous(
            &views,
            max_channels,
            options.tree_leaf_limit(),
            options.ma_budget_mode == Some(ma_train::MaBudgetMode::Strict),
            &ordered
                .iter()
                .map(|stream| (stream.stream_id as u32, stream.wp))
                .collect::<Vec<_>>(),
            fast,
            &previous_views,
        );
    }

    let mut train = encode::train_options(options);
    train.parallel = parallel_search;
    let policy = options.sampling.as_ref();
    if let Some(policy) = policy {
        if policy
            .properties
            .iter()
            .any(|&p| p >= 16 + 4 * options.previous_channels)
        {
            return Err(Error::InvalidInput(
                "MA sampling property exceeds available previous channels".into(),
            ));
        }
        train.predictors = source_predictors(options.predictor, policy)?;
        train.require_complete_search = true;
    }
    // Quantile populations include verified constants and every provided group.
    crate::profile::note(
        "ma.domain",
        format_args!(
            "streams={} views={} predictors={} properties={} sample_limit={:?} max_leaves={} max_merged_leaves={}",
            ordered.len(),
            views.len(),
            train.predictors.len(),
            policy.map_or(train.properties.len(), |p| p.properties.len()),
            options.sample_limit,
            train.max_leaves,
            train.tree_leaf_limit(),
        ),
    );
    let quantiles = {
        let _scope = crate::profile::scope("ma.quantiles");
        policy
            .map(|p| sample::collect_pixel_samples(&views, p))
            .transpose()?
    };
    let max_channel_size = policy.map_or(usize::MAX, |p| p.max_channel_size);
    if max_channels != 0 {
        let _scope = crate::profile::scope("ma.constants");
        let range = [
            [0, max_channels as u32],
            [
                ordered[0].stream_id as u32,
                ordered[ordered.len() - 1].stream_id as u32 + 1,
            ],
        ];
        train.static_property_range = Some(range);
        if policy.is_none() || options.source_constant_channels {
            train.constant_channels =
                ma_train::find_constant_channels(&views, range, max_channel_size)?;
        }
        let required = train.constant_channels.len()
            + usize::from(train.constant_channels.len() < max_channels);
        if required > train.tree_leaf_limit() {
            if policy.is_some() || options.ma_budget_mode.is_some() {
                return Err(Error::Limit(
                    "MA constant suffix exceeds explicit leaf budget",
                ));
            }
            // Preserve legacy behavior: constant bypass is optional in this mode.
            train.constant_channels.clear();
        }
    }
    let variable_channels = train
        .constant_channels
        .first()
        .map_or(max_channels, |constant| constant.channel as usize);

    let gather_scope = crate::profile::scope("ma.gather");
    if let Some(policy) = policy {
        if !train.multiplier_info.is_empty() {
            return Err(Error::InvalidInput(
                "source prepared gathering cannot validate raw multipliers".into(),
            ));
        }
        let population = quantiles.as_ref().expect("source quantiles collected");
        let quantization =
            population.quantization(&policy.properties, policy.max_property_values)?;
        // No gather option retains serial collection. A serial learner still
        // honors requested sparse/stripe semantics, but disables task dispatch.
        let serial_gather = GatherOptions {
            workers: 1,
            ..gather.cloned().unwrap_or_default()
        };
        let gathered = SourceGather {
            streams: &ordered,
            policy,
            predictors: &train.predictors,
            variable_channels,
            sample_limit: options.sample_limit.unwrap_or(usize::MAX),
            quantization: &quantization,
            budget,
        }
        .collect_prepared_with_histories(
            gather.filter(|_| parallel_search).unwrap_or(&serial_gather),
            histories,
        )?;
        let samples = gathered.samples;
        let stats = gathered.stats;
        debug_assert_eq!(samples.total, stats.selected_samples);
        crate::profile::note(
            "ma.gather_counts",
            format_args!(
                "visited={} selected={} records={} source_total_pixels={} variable_channels={}",
                stats.visited_pixels,
                stats.selected_samples,
                samples.len(),
                stats.source_total_pixels,
                variable_channels,
            ),
        );
        policy.apply_to_train(population, stats, &mut train)?;
        debug_assert_eq!(
            train.property_quantization.as_ref(),
            Some(&samples.quantization)
        );
        drop(gather_scope);
        return ma_train::train_prepared(samples, &train);
    }
    let samples = gather_legacy(&ordered, options, budget, |stream| {
        encode::collect_samples(stream.channels, options, &stream.wp, stream.stream_id)
    })?;
    drop(gather_scope);
    ma_train::train(&samples, &train)
}

/// Reserve selected-sample allowance before the collector allocates or visits
/// pixels. Like source collection, a downstream failure aborts the invocation;
/// already consumed shared allowance is not refunded.
fn gather_legacy(
    streams: &[&TrainingStream<'_>],
    options: &ModularOptions,
    budget: Option<&SampleBudget>,
    mut collect: impl FnMut(&TrainingStream<'_>) -> Result<Vec<Sample>>,
) -> Result<Vec<Sample>> {
    let mut samples = Vec::new();
    for stream in streams {
        let target = encode::legacy_sample_count(stream.channels, options)?;
        if options
            .sample_limit
            .is_some_and(|limit| target > limit.saturating_sub(samples.len()))
        {
            return Err(Error::Limit("legacy MA selected-sample budget exceeded"));
        }
        if let Some(budget) = budget {
            budget.consume(target)?;
        }
        samples
            .try_reserve(target)
            .map_err(|_| Error::Limit("MA sample allocation"))?;
        let mut batch = collect(stream)?;
        debug_assert_eq!(batch.len(), target);
        samples.append(&mut batch);
    }
    Ok(samples)
}

/// TreeSamples::SetPredictor. GradientOnly by itself restricts properties;
/// enc_modular.cc's decoding tier independently supplies predictor_override.
pub(super) fn source_predictors(
    mode: Predictor,
    policy: &SamplingPolicy,
) -> Result<Vec<Predictor>> {
    if policy.tree_mode == TreeMode::WeightedOnly {
        return Ok(vec![Predictor::Weighted]);
    }
    let mut candidates = encode::predictors(policy.predictor_override.unwrap_or(mode));
    if policy.tree_mode == TreeMode::NoWeighted {
        candidates.retain(|&p| p != Predictor::Weighted);
    }
    if candidates.is_empty() {
        return Err(Error::InvalidInput(
            "MA tree mode excludes its only predictor".into(),
        ));
    }
    Ok(candidates)
}

fn channel_prefix(
    stream: &TrainingStream<'_>,
    policy: &SamplingPolicy,
    variable_channels: usize,
) -> usize {
    stream
        .channels
        .iter()
        .position(|channel| {
            channel.hshift >= 0
                && channel.vshift >= 0
                && (channel.width > policy.max_channel_size
                    || channel.height > policy.max_channel_size)
        })
        .unwrap_or(stream.channels.len())
        .min(variable_channels)
}

fn prefix_population(stream: &TrainingStream<'_>, end: usize) -> Result<u64> {
    stream.channels[..end]
        .iter()
        .try_fold(0u64, |sum, channel| {
            let pixels = channel
                .width
                .checked_mul(channel.height)
                .ok_or(Error::Limit("MA training plane dimensions overflow"))?;
            sum.checked_add(pixels as u64)
                .ok_or(Error::Limit("MA training population overflow"))
        })
}

#[cfg(test)]
fn gather_source(
    streams: &[&TrainingStream<'_>],
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    variable_channels: usize,
    sample_limit: usize,
    quantization: &PropertyQuantization,
    budget: Option<&SampleBudget>,
) -> Result<(Vec<Sample>, SamplingStats)> {
    let gathered = SourceGather {
        streams,
        policy,
        predictors,
        variable_channels,
        sample_limit,
        quantization,
        budget,
    }
    .collect(&GatherOptions {
        workers: 1,
        ..GatherOptions::default()
    })?;
    Ok((gathered.samples, gathered.stats))
}

fn uses_predictor_pair(predictors: &[Predictor]) -> bool {
    predictors == [Predictor::Weighted, Predictor::Gradient]
}

/// enc_predict.h::PredictLearnBest reuses the WP/property traversal and only
/// evaluates Gradient alongside Weighted. Candidate order is part of the gate.
#[inline]
#[cfg(test)]
fn sample_residuals<const PAIR: bool>(
    predictors: &[Predictor],
    neighbors: Neighbors,
    wp_prediction: i32,
    value: i32,
) -> Result<Vec<i32>> {
    if PAIR {
        debug_assert!(uses_predictor_pair(predictors));
        return Ok(vec![
            value.wrapping_sub(wp_prediction),
            value.wrapping_sub(clamped_gradient(
                neighbors.left,
                neighbors.top,
                neighbors.topleft,
            )),
        ]);
    }
    predictors
        .iter()
        .map(|&predictor| {
            let guess = predict_one(predictor, neighbors, Some(i64::from(wp_prediction)))
                .map_err(|e| Error::Encoding(e.to_string()))?;
            // TreeSamples::AddSample assigns the wide difference to pixel_type
            // before PackSigned, retaining its low 32 bits.
            Ok((i64::from(value) - guess) as i32)
        })
        .collect()
}

#[cfg(test)]
fn visit_source_samples(
    streams: &[&TrainingStream<'_>],
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    variable_channels: usize,
    sample_limit: usize,
    budget: Option<&SampleBudget>,
    collect: impl FnMut(Sample) -> Result<()>,
) -> Result<SamplingStats> {
    if uses_predictor_pair(predictors) {
        visit_source_samples_impl::<true>(
            streams,
            policy,
            predictors,
            variable_channels,
            sample_limit,
            budget,
            collect,
        )
    } else {
        visit_source_samples_impl::<false>(
            streams,
            policy,
            predictors,
            variable_channels,
            sample_limit,
            budget,
            collect,
        )
    }
}

#[cfg(test)]
fn visit_source_samples_impl<const PAIR: bool>(
    streams: &[&TrainingStream<'_>],
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    variable_channels: usize,
    sample_limit: usize,
    budget: Option<&SampleBudget>,
    mut collect: impl FnMut(Sample) -> Result<()>,
) -> Result<SamplingStats> {
    let first_pixels = match streams.first() {
        Some(first) => prefix_population(first, channel_prefix(first, policy, variable_channels))?,
        None => 0,
    };
    let mut stats = SamplingStats {
        source_total_pixels: first_pixels.max(1),
        ..SamplingStats::default()
    };
    for stream in streams {
        let end = channel_prefix(stream, policy, variable_channels);
        let pixels = prefix_population(stream, end)?;
        stats.visited_pixels = stats
            .visited_pixels
            .checked_add(pixels)
            .ok_or(Error::Limit("MA training population overflow"))?;
        stats.source_total_pixels = stats
            .source_total_pixels
            .checked_add(pixels)
            .ok_or(Error::Limit("MA training population overflow"))?;
        // Zero source fraction never produces training samples.
        if policy.nb_repeats == 0.0 {
            continue;
        }
        for channel in 0..end {
            visit_source_channel_impl::<PAIR, true>(
                stream,
                channel,
                policy,
                predictors,
                || {
                    // Preserve serial gathering's pre-allocation sample charge.
                    if stats.selected_samples == sample_limit as u64 {
                        return Err(Error::Limit("source MA selected-sample budget exceeded"));
                    }
                    if let Some(budget) = budget {
                        budget.consume(1)?;
                    }
                    stats.selected_samples += 1;
                    Ok(())
                },
                &mut collect,
            )?;
        }
    }
    Ok(stats)
}

/// Detail-only trace gate. Mirrors the MA histogram detail scopes: one
/// `RS_JXL_PROFILE_DETAIL=1` switch turns on every fine-grained note, and the
/// default path pays a single `OnceLock` read.
fn gather_detail_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("RS_JXL_PROFILE_DETAIL").as_deref() == Ok("1"))
        && crate::profile::enabled()
}

struct GatherTask {
    stream: usize,
    channel: usize,
    rows: std::ops::Range<usize>,
}

struct GatheredSamples<S = Vec<Sample>> {
    samples: S,
    stats: SamplingStats,
    #[cfg(test)]
    parallel_waves: usize,
    #[cfg(test)]
    waves: usize,
    #[cfg(test)]
    serial_tasks: usize,
}

/// Shared quantization/domain inputs for all independent gather chunks.
struct SourceGather<'a, 'pixels> {
    streams: &'a [&'a TrainingStream<'pixels>],
    policy: &'a SamplingPolicy,
    predictors: &'a [Predictor],
    variable_channels: usize,
    sample_limit: usize,
    quantization: &'a PropertyQuantization,
    budget: Option<&'a SampleBudget>,
}

/// Emits the detail-only domain line for one `collect_prepared` call.
///
/// The policy fingerprint hashes every input that decides which pixels are
/// selected and which sample columns are built, so two learners sharing a
/// fingerprint visit comparable domains. `sid`/`chmax`/`rows` name the covered
/// region, `px` the visited population, and `dispatch` the per-wave
/// `(tasks, chunks, estimated MB)` that decides the batch branch.
fn note_gather_domain(
    domain: &SourceGather<'_, '_>,
    options: &GatherOptions,
    tasks: &[GatherTask],
    wave_trace: &[(usize, usize, f64)],
    stats: &SamplingStats,
    serial_dispatch: bool,
    started: Option<std::time::Instant>,
) {
    let Some(started) = started else { return };
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{:?}", domain.policy).hash(&mut hasher);
    format!("{:?}", domain.predictors).hash(&mut hasher);
    format!("{:?}", options).hash(&mut hasher);
    domain.variable_channels.hash(&mut hasher);
    domain.sample_limit.hash(&mut hasher);
    let fingerprint = hasher.finish();
    let mut row_lo = usize::MAX;
    let mut row_hi = 0usize;
    let mut channel_max = 0usize;
    for task in tasks {
        row_lo = row_lo.min(task.rows.start);
        row_hi = row_hi.max(task.rows.end);
        channel_max = channel_max.max(task.channel + 1);
    }
    if tasks.is_empty() {
        row_lo = 0;
    }
    let props: Vec<String> = domain
        .policy
        .properties
        .iter()
        .map(|value| value.to_string())
        .collect();
    let line = format!(
        "fp={fingerprint:016x} sid={}..{}n{} tasks={} chmax={channel_max} rows={row_lo}..{row_hi} px={} selected={} preds={} props={}:{} prev={} reps={} limit={} waves={} dispatch={:?} serial={serial_dispatch} wcount={} gworkers={} ms={:.3}",
        domain.streams.first().map_or(0, |stream| stream.stream_id),
        domain.streams.last().map_or(0, |stream| stream.stream_id),
        domain.streams.len(),
        tasks.len(),
        stats.visited_pixels,
        stats.selected_samples,
        domain.predictors.len(),
        domain.policy.properties.len(),
        props.join(","),
        domain.policy.previous_channels,
        domain.policy.nb_repeats,
        domain.sample_limit,
        wave_trace.len(),
        wave_trace,
        execution::worker_count(),
        options.workers,
        started.elapsed().as_secs_f64() * 1000.0,
    );
    crate::profile::note("ma.gather_domain", line);
}

impl SourceGather<'_, '_> {
    #[cfg(test)]
    fn collect(&self, options: &GatherOptions) -> Result<GatheredSamples> {
        let output = self.collect_prepared(options)?;
        Ok(GatheredSamples {
            samples: output.samples.into_samples()?,
            stats: output.stats,
            parallel_waves: output.parallel_waves,
            waves: output.waves,
            serial_tasks: output.serial_tasks,
        })
    }

    #[cfg(test)]
    fn collect_prepared(
        &self,
        options: &GatherOptions,
    ) -> Result<GatheredSamples<PreparedSamples>> {
        self.collect_prepared_with_history(options, None)
    }

    #[cfg(test)]
    fn collect_prepared_with_history(
        &self,
        options: &GatherOptions,
        history: Option<&super::weighted_history::History<'_>>,
    ) -> Result<GatheredSamples<PreparedSamples>> {
        self.collect_prepared_with_histories(options, history.as_slice())
    }

    fn collect_prepared_with_histories(
        &self,
        options: &GatherOptions,
        histories: &[&super::weighted_history::History<'_>],
    ) -> Result<GatheredSamples<PreparedSamples>> {
        options.validate()?;
        // Stream ordering can differ from the writer's ordering. Bind once by
        // immutable plane/header identity, never by a coincident stream index.
        let histories: Vec<_> = if histories.is_empty() {
            Vec::new()
        } else {
            self.streams
                .iter()
                .map(|stream| {
                    histories
                        .iter()
                        .copied()
                        .find(|cache| cache.matches(stream.channels, &stream.wp))
                })
                .collect()
        };
        // Domain-level trace: one line per learner records the policy
        // fingerprint, the covered domain and the dispatch shape. It answers
        // whether two learners gather the same pixels twice, and why the batch
        // branch stays closed, without touching the collected samples.
        let detail = gather_detail_enabled();
        let started = detail.then(std::time::Instant::now);
        let serial_dispatch = detail.then(execution::is_serial).unwrap_or(false);
        let mut wave_trace: Vec<(usize, usize, f64)> = Vec::new();
        let mut master =
            SourceSampleCollector::new(self.predictors, self.quantization, options.cache_hashes)?;
        let first_pixels = match self.streams.first() {
            Some(first) => prefix_population(
                first,
                channel_prefix(first, self.policy, self.variable_channels),
            )?,
            None => 0,
        };
        let mut output = GatheredSamples {
            samples: (),
            stats: SamplingStats {
                source_total_pixels: first_pixels.max(1),
                ..SamplingStats::default()
            },
            #[cfg(test)]
            parallel_waves: 0,
            #[cfg(test)]
            waves: 0,
            #[cfg(test)]
            serial_tasks: 0,
        };
        let mut tasks = Vec::new();
        let mut work = Vec::new();
        let mut expected = Vec::new();
        for (stream_index, stream) in self.streams.iter().enumerate() {
            let end = channel_prefix(stream, self.policy, self.variable_channels);
            let pixels = prefix_population(stream, end)?;
            output.stats.visited_pixels = output
                .stats
                .visited_pixels
                .checked_add(pixels)
                .ok_or(Error::Limit("MA training population overflow"))?;
            output.stats.source_total_pixels = output
                .stats
                .source_total_pixels
                .checked_add(pixels)
                .ok_or(Error::Limit("MA training population overflow"))?;
            if self.policy.nb_repeats == 0.0 {
                continue;
            }
            for (channel, plane) in stream.channels[..end].iter().enumerate() {
                let area = plane
                    .width
                    .checked_mul(plane.height)
                    .ok_or(Error::Limit("MA training plane dimensions overflow"))?;
                if area == 0 {
                    continue;
                }
                let fraction = sample::tree_sample_fraction(area, self.policy.nb_repeats)?;
                for rows in options.rows(plane.width, plane.height)? {
                    let pixels = plane.width * rows.len();
                    work.push(pixels as u64);
                    expected.push(fraction * pixels as f64 + 64.0);
                    tasks.push(GatherTask {
                        stream: stream_index,
                        channel,
                        rows,
                    });
                }
            }
        }
        let waves = options.plan(
            &work,
            &expected,
            self.predictors.len(),
            &self.policy.properties,
            execution::worker_count() > 1,
        )?;
        if crate::profile::enabled() {
            for wave in &waves {
                crate::profile::note(
                    "gather.plan",
                    format!(
                        "mode={:?} tasks={} chunks={} estimated_bytes={:.0} budget_bytes={}",
                        options.mode,
                        wave.tasks.len(),
                        wave.chunks.len(),
                        wave.estimated_bytes,
                        options.batch_budget_bytes,
                    ),
                );
            }
        }
        // Tasks partition each eligible plane; warm-up rows never collect samples.
        // Each visited pixel can therefore consume at most one local allowance.
        let local_budget = (output.stats.visited_pixels > self.sample_limit as u64)
            .then(|| SampleBudget::new(self.sample_limit));
        let concat_batches = options.concat_batches && waves.len() == 1;
        for wave in waves {
            #[cfg(test)]
            {
                output.waves += 1;
            }
            if detail {
                wave_trace.push((
                    wave.tasks.len(),
                    wave.chunks.len(),
                    wave.estimated_bytes / 1_000_000.0,
                ));
            }
            if !execution::is_serial()
                && wave.chunks.len() > 1
                && wave.estimated_bytes <= options.batch_budget_bytes as f64
            {
                #[cfg(test)]
                {
                    output.parallel_waves += 1;
                }
                // Indexed collection retains source chunk order irrespective of
                // completion order. Only one wave's extra batches stay live.
                let batches: Vec<Result<_>> = {
                    let _scope = crate::profile::scope("gather.batch_collect");
                    execution::map(&wave.chunks, |range| {
                        let _batch_new_scope = gather_detail_enabled()
                            .then(|| crate::profile::scope("gather.batch_new"));
                        let mut batch = SourceSampleCollector::new(
                            self.predictors,
                            self.quantization,
                            options.cache_hashes,
                        )?;
                        drop(_batch_new_scope);
                        let _tasks_scope = gather_detail_enabled()
                            .then(|| crate::profile::scope("gather.collect_tasks"));
                        let selected = self.collect_tasks(
                            &tasks[range.clone()],
                            local_budget.as_ref(),
                            &mut batch,
                            options.sparse,
                            &histories,
                        )?;
                        drop(_tasks_scope);
                        Ok((batch, selected))
                    })
                };
                let mut prepared_batches = Vec::new();
                for batch in batches {
                    let (samples, selected) = batch?;
                    output.stats.selected_samples += selected;
                    prepared_batches.push(samples);
                }
                // Prepare once for the complete wave, not once per batch/row.
                // The selected pixels were already charged in the gather tasks.
                let _scope = crate::profile::scope("gather.append");
                if concat_batches {
                    let samples = master.finish_appended(prepared_batches)?;
                    debug_assert_eq!(samples.total, output.stats.selected_samples);
                    note_gather_domain(
                        self,
                        options,
                        &tasks,
                        &wave_trace,
                        &output.stats,
                        serial_dispatch,
                        started,
                    );
                    return Ok(GatheredSamples {
                        samples,
                        stats: output.stats,
                        #[cfg(test)]
                        parallel_waves: output.parallel_waves,
                        #[cfg(test)]
                        waves: output.waves,
                        #[cfg(test)]
                        serial_tasks: output.serial_tasks,
                    });
                }
                master.append(prepared_batches)?;
            } else {
                #[cfg(test)]
                {
                    output.serial_tasks += wave.tasks.len();
                }
                let _scope = crate::profile::scope("gather.batch_collect");
                let _tasks_scope =
                    gather_detail_enabled().then(|| crate::profile::scope("gather.collect_tasks"));
                output.stats.selected_samples += self.collect_tasks(
                    &tasks[wave.tasks],
                    local_budget.as_ref(),
                    &mut master,
                    options.sparse,
                    &histories,
                )?;
                drop(_tasks_scope);
            }
        }
        let samples = master.finish();
        debug_assert_eq!(samples.total, output.stats.selected_samples);
        note_gather_domain(
            self,
            options,
            &tasks,
            &wave_trace,
            &output.stats,
            serial_dispatch,
            started,
        );
        Ok(GatheredSamples {
            samples,
            stats: output.stats,
            #[cfg(test)]
            parallel_waves: output.parallel_waves,
            #[cfg(test)]
            waves: output.waves,
            #[cfg(test)]
            serial_tasks: output.serial_tasks,
        })
    }

    fn collect_tasks(
        &self,
        tasks: &[GatherTask],
        local_budget: Option<&SampleBudget>,
        collector: &mut SourceSampleCollector,
        sparse: bool,
        histories: &[Option<&super::weighted_history::History<'_>>],
    ) -> Result<u64> {
        let mut selected = 0;
        // Amortize table rebuilds only across small, bounded task batches.
        // push grows columns, not the table: refresh prepare between batches
        // using actual retained rows, and keep per-task hints for heavy batches.
        const PREPARE_TASKS: usize = 8;
        const PREPARE_RECORDS: usize = 65_536;
        for batch in tasks.chunks(PREPARE_TASKS) {
            let mut expected = [0usize; PREPARE_TASKS];
            let mut batch_expected = 0usize;
            for (task, hint) in batch.iter().zip(&mut expected) {
                let plane = &self.streams[task.stream].channels[task.channel];
                let area = plane
                    .width
                    .checked_mul(plane.height)
                    .ok_or(Error::Limit("MA training plane dimensions overflow"))?;
                let fraction = sample::tree_sample_fraction(area, self.policy.nb_repeats)?;
                *hint = gather_samples::expected_samples(fraction, plane.width, task.rows.len())?;
                batch_expected = batch_expected.saturating_add(*hint);
            }
            let prepare_once = batch_expected <= PREPARE_RECORDS;
            if prepare_once {
                let _scope =
                    gather_detail_enabled().then(|| crate::profile::scope("gather.task_prepare"));
                collector.prepare(batch_expected)?;
            }
            for (task, hint) in batch.iter().zip(expected) {
                if !prepare_once {
                    let _scope = gather_detail_enabled()
                        .then(|| crate::profile::scope("gather.task_prepare"));
                    collector.prepare(hint)?;
                }
                let stream = self.streams[task.stream];
                let history = histories
                    .get(task.stream)
                    .copied()
                    .flatten()
                    .filter(|_| task.rows == (0..stream.channels[task.channel].height));
                let mut captured = history.and_then(|cache| cache.start(task.channel));
                let _visit_scope =
                    gather_detail_enabled().then(|| crate::profile::scope("gather.visit_source"));
                visit_source_range(
                    stream,
                    task.channel,
                    self.policy,
                    self.predictors,
                    task.rows.clone(),
                    sparse,
                    || {
                        if let Some(budget) = local_budget {
                            budget.consume(1)?;
                        }
                        if let Some(budget) = self.budget {
                            budget.consume(1)?;
                        }
                        selected += 1;
                        Ok(())
                    },
                    |properties, residuals| collector.push(properties, residuals, 1),
                    |prediction, error| {
                        if let Some(values) = &mut captured {
                            values.push([prediction, error]);
                        }
                    },
                )?;
                if let (Some(history), Some(values)) = (history, captured) {
                    history.finish(task.channel, values);
                }
                drop(_visit_scope);
            }
        }
        Ok(selected)
    }
}

/// Whole-channel GatherTreeData: independent RNG/WP state, original channel
/// index and references into the complete stream (never a renumbered slice).
#[cfg(test)]
fn visit_source_channel(
    stream: &TrainingStream<'_>,
    channel_index: usize,
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    collect: impl FnMut(Sample) -> Result<()>,
) -> Result<()> {
    if uses_predictor_pair(predictors) {
        visit_source_channel_impl::<true, true>(
            stream,
            channel_index,
            policy,
            predictors,
            || Ok(()),
            collect,
        )
    } else {
        visit_source_channel_impl::<false, true>(
            stream,
            channel_index,
            policy,
            predictors,
            || Ok(()),
            collect,
        )
    }
}

/// GatherTreeData's references.w is the configured property-slot count, not
/// the number of earlier matching channels found for this particular plane.
#[inline]
fn has_no_edge_interior(width: usize, y: usize, previous_channels: usize) -> bool {
    y > 1 && width > 8 && previous_channels == 0
}

#[inline(always)]
fn visit_source_row<const NO_EDGE: bool>(
    channel: &Channel,
    y: usize,
    columns: std::ops::Range<usize>,
    pixel: &mut impl FnMut(usize, i32, Neighbors, bool) -> Result<()>,
) -> Result<()> {
    let width = channel.width;
    if NO_EDGE {
        debug_assert!(has_no_edge_interior(width, y, 0));
        debug_assert!(columns.start >= 2 && columns.end <= width - 2);
        let begin = y * width;
        let row = &channel.samples[begin..begin + width];
        let top = &channel.samples[begin - width..begin];
        let toptop = &channel.samples[begin - width - width..begin - width];
        let start = columns.start;
        let end = columns.end;
        // Every lane below is a slice already offset into its row, so the hot
        // interior loop has no per-pixel edge tests and no per-index bounds
        // checks. The centre lane carries the sample the edge loader re-reads.
        let lanes = row[start..end]
            .iter()
            .zip(&row[start - 1..end - 1])
            .zip(&row[start - 2..end - 2])
            .zip(&top[start..end])
            .zip(&top[start - 1..end - 1])
            .zip(&top[start + 1..end + 1])
            .zip(&toptop[start..end])
            .zip(&top[start + 2..end + 2]);
        for (
            offset,
            (((((((value, left), leftleft), top), topleft), topright), toptop), toprightright),
        ) in lanes.enumerate()
        {
            pixel(
                start + offset,
                *value,
                Neighbors {
                    left: *left,
                    top: *top,
                    toptop: *toptop,
                    topleft: *topleft,
                    topright: *topright,
                    leftleft: *leftleft,
                    toprightright: *toprightright,
                },
                true,
            )?;
        }
    } else {
        for x in columns {
            let value = channel.samples[y * width + x];
            let neighbors = Neighbors::from_plane(&channel.samples, width, width, x, y)
                .map_err(|e| Error::Encoding(e.to_string()))?;
            pixel(x, value, neighbors, false)?;
        }
    }
    Ok(())
}

// Keep the raw-sample oracle interface test-only; production collects borrowed
// fields into compact columns without allocating two vectors per selected pixel.
#[cfg(test)]
fn visit_source_channel_impl<const PAIR: bool, const INTERIOR: bool>(
    stream: &TrainingStream<'_>,
    channel_index: usize,
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    before_collect: impl FnMut() -> Result<()>,
    mut collect: impl FnMut(Sample) -> Result<()>,
) -> Result<()> {
    visit_source_range_impl::<PAIR, INTERIOR>(
        stream,
        channel_index,
        policy,
        predictors,
        0..stream.channels[channel_index].height,
        false,
        before_collect,
        |properties, residuals| {
            collect(Sample {
                properties: properties.to_vec(),
                residuals: residuals.to_vec(),
                weight: 1,
            })
        },
        |_, _| {},
    )
}

#[allow(clippy::too_many_arguments)]
fn visit_source_range(
    stream: &TrainingStream<'_>,
    channel_index: usize,
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    rows: std::ops::Range<usize>,
    sparse: bool,
    before_collect: impl FnMut() -> Result<()>,
    collect: impl FnMut(&[i32], &[i32]) -> Result<()>,
    capture: impl FnMut(i32, i32),
) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    match super::kernel_isa::selected() {
        super::kernel_isa::Isa::Avx2 => {
            // SAFETY: selection checks CPU and OS support once per process.
            return unsafe {
                visit_source_range_avx2(
                    stream,
                    channel_index,
                    policy,
                    predictors,
                    rows,
                    sparse,
                    before_collect,
                    collect,
                    capture,
                )
            };
        }
        super::kernel_isa::Isa::Avx512 => {
            // SAFETY: selection checks every feature required by this wrapper.
            return unsafe {
                visit_source_range_avx512(
                    stream,
                    channel_index,
                    policy,
                    predictors,
                    rows,
                    sparse,
                    before_collect,
                    collect,
                    capture,
                )
            };
        }
        super::kernel_isa::Isa::Portable => {}
    }
    visit_source_range_portable(
        stream,
        channel_index,
        policy,
        predictors,
        rows,
        sparse,
        before_collect,
        collect,
        capture,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn visit_source_range_avx2(
    stream: &TrainingStream<'_>,
    channel_index: usize,
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    rows: std::ops::Range<usize>,
    sparse: bool,
    before_collect: impl FnMut() -> Result<()>,
    collect: impl FnMut(&[i32], &[i32]) -> Result<()>,
    capture: impl FnMut(i32, i32),
) -> Result<()> {
    visit_source_range_portable(
        stream,
        channel_index,
        policy,
        predictors,
        rows,
        sparse,
        before_collect,
        collect,
        capture,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn visit_source_range_avx512(
    stream: &TrainingStream<'_>,
    channel_index: usize,
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    rows: std::ops::Range<usize>,
    sparse: bool,
    before_collect: impl FnMut() -> Result<()>,
    collect: impl FnMut(&[i32], &[i32]) -> Result<()>,
    capture: impl FnMut(i32, i32),
) -> Result<()> {
    visit_source_range_portable(
        stream,
        channel_index,
        policy,
        predictors,
        rows,
        sparse,
        before_collect,
        collect,
        capture,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn visit_source_range_portable(
    stream: &TrainingStream<'_>,
    channel_index: usize,
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    rows: std::ops::Range<usize>,
    sparse: bool,
    before_collect: impl FnMut() -> Result<()>,
    collect: impl FnMut(&[i32], &[i32]) -> Result<()>,
    capture: impl FnMut(i32, i32),
) -> Result<()> {
    if uses_predictor_pair(predictors) {
        visit_source_range_impl::<true, true>(
            stream,
            channel_index,
            policy,
            predictors,
            rows,
            sparse,
            before_collect,
            collect,
            capture,
        )
    } else {
        visit_source_range_impl::<false, true>(
            stream,
            channel_index,
            policy,
            predictors,
            rows,
            sparse,
            before_collect,
            collect,
            capture,
        )
    }
}

// INTERIOR=false is a test-only traversal oracle. The warm-up changes learning
// state only; pixel coding never calls this path with a truncated history.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn visit_source_range_impl<const PAIR: bool, const INTERIOR: bool>(
    stream: &TrainingStream<'_>,
    channel_index: usize,
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    rows: std::ops::Range<usize>,
    sparse: bool,
    mut before_collect: impl FnMut() -> Result<()>,
    mut collect: impl FnMut(&[i32], &[i32]) -> Result<()>,
    mut capture: impl FnMut(i32, i32),
) -> Result<()> {
    let channel = &stream.channels[channel_index];
    let width = channel.width;
    let height = channel.height;
    if rows.start > rows.end || rows.end > height || predictors.len() > 14 {
        return Err(Error::InvalidInput(
            "invalid MA gather row/predictor range".into(),
        ));
    }
    if width == 0 || height == 0 || rows.is_empty() || policy.nb_repeats == 0.0 {
        return Ok(());
    }
    let area = width
        .checked_mul(height)
        .ok_or(Error::Limit("MA training plane dimensions overflow"))?;
    if channel.samples.len() != area {
        return Err(Error::InvalidInput(
            "MA training plane dimensions/buffer mismatch".into(),
        ));
    }
    let references: Vec<_> = stream.channels[..channel_index]
        .iter()
        .rev()
        .filter(|p| {
            p.width == width
                && p.height == height
                && p.hshift == channel.hshift
                && p.vshift == channel.vshift
        })
        .take(policy.previous_channels)
        .collect();
    if references
        .iter()
        .any(|reference| reference.samples.len() != area)
    {
        return Err(Error::InvalidInput(
            "MA reference plane dimensions/buffer mismatch".into(),
        ));
    }
    let use_wp = predictors.contains(&Predictor::Weighted) || policy.properties.contains(&15);
    let mut state = if use_wp {
        Some(weighted::State::new(stream.wp, width)?)
    } else {
        None
    };
    let mut sampler = ChannelSampler::new(width, height, policy.nb_repeats)?;
    let warm_begin = rows.start.saturating_sub(16);
    sampler.skip_pixels(warm_begin * width)?;
    let channel_id = channel_index as i32;
    let stream_id = stream.stream_id as i32;
    let mut previous_gradient = 0i32;
    let mut properties = vec![0; 16 + 4 * policy.previous_channels];
    let mut residuals = [0; 14];
    #[cfg(feature = "encoder-kernel-profile")]
    let mut kernel_profile = super::kernel_profile::Profile::new(
        "gather",
        [
            "rng_weighted",
            "properties",
            "reference_residual",
            "collector",
        ],
        (stream_id as u64).wrapping_mul(131)
            ^ channel_index as u64
            ^ (width as u64).rotate_left(17)
            ^ (rows.start as u64).rotate_left(7),
    );
    for y in warm_begin..rows.end {
        let in_rows = y >= rows.start;
        let mut pixel = {
            #[inline(always)]
            |x: usize, value: i32, n: Neighbors, interior: bool| {
                #[cfg(feature = "encoder-kernel-profile")]
                let mut clock = kernel_profile.start();
                // Draw even in warm-up rows, matching the source's operand order.
                let selected =
                    sampler.next().expect("one sample draw per channel pixel") && in_rows;
                let (prediction, error) = match state.as_mut() {
                    // The validated rectangle walks every pixel, including warm-up
                    // rows. Local history needs no per-pixel coordinate validation;
                    // shifting row parity only renames its initially zero buffers.
                    Some(state) => {
                        if interior {
                            state.predict_and_update_raster_interior::<false>(
                                value,
                                x,
                                y - warm_begin,
                                n.top,
                                n.left,
                                n.topright,
                                n.topleft,
                                n.toptop,
                            )
                        } else {
                            state.predict_and_update_raster::<false>(
                                value,
                                x,
                                y - warm_begin,
                                n.top,
                                n.left,
                                n.topright,
                                n.topleft,
                                n.toptop,
                            )
                        }
                    }
                    None => (0, 0),
                };
                capture(prediction, error);
                #[cfg(feature = "encoder-kernel-profile")]
                kernel_profile.lap(0, &mut clock);
                // Same values as Properties::calculate, with one gradient instead
                // of two and the previous-gradient reset fed by this loop's raster
                // tracker. That tracker also advances over pixels whose properties
                // the sparse traversal never materializes.
                let gradient = n.left.wrapping_add(n.top).wrapping_sub(n.topleft);
                if !sparse || selected {
                    let previous = if x == 0 { 0 } else { previous_gradient };
                    let values = [
                        channel_id,
                        stream_id,
                        y as i32,
                        x as i32,
                        n.top.wrapping_abs(),
                        n.left.wrapping_abs(),
                        n.top,
                        n.left,
                        n.left.wrapping_sub(previous),
                        gradient,
                        n.left.wrapping_sub(n.topleft),
                        n.topleft.wrapping_sub(n.top),
                        n.top.wrapping_sub(n.topright),
                        n.top.wrapping_sub(n.toptop),
                        n.left.wrapping_sub(n.leftleft),
                        error,
                    ];
                    properties[..16].copy_from_slice(&values);
                }
                previous_gradient = gradient;
                #[cfg(feature = "encoder-kernel-profile")]
                kernel_profile.lap(1, &mut clock);
                if selected {
                    before_collect()?;
                    let position = y * width + x;
                    // Reference properties are stateless; skipped samples need no loads.
                    for (i, reference) in references.iter().enumerate() {
                        let v = reference.samples[position];
                        let left = if x > 0 {
                            reference.samples[position - 1]
                        } else {
                            0
                        };
                        let top = if y > 0 {
                            reference.samples[position - width]
                        } else {
                            left
                        };
                        let nw = if x > 0 && y > 0 {
                            reference.samples[position - width - 1]
                        } else {
                            left
                        };
                        let residual = v.wrapping_sub(clamped_gradient(left, top, nw));
                        properties[16 + 4 * i..20 + 4 * i].copy_from_slice(&[
                            v.wrapping_abs(),
                            v,
                            residual.wrapping_abs(),
                            residual,
                        ]);
                    }
                    if PAIR {
                        debug_assert!(uses_predictor_pair(predictors));
                        residuals[0] = value.wrapping_sub(prediction);
                        residuals[1] =
                            value.wrapping_sub(clamped_gradient(n.left, n.top, n.topleft));
                    } else {
                        for (residual, &predictor) in residuals.iter_mut().zip(predictors) {
                            let guess = predict_one(predictor, n, Some(i64::from(prediction)))
                                .map_err(|e| Error::Encoding(e.to_string()))?;
                            *residual = (i64::from(value) - guess) as i32;
                        }
                    }
                    #[cfg(feature = "encoder-kernel-profile")]
                    kernel_profile.lap(2, &mut clock);
                    let result = collect(&properties, &residuals[..predictors.len()]);
                    #[cfg(feature = "encoder-kernel-profile")]
                    kernel_profile.lap(3, &mut clock);
                    result?;
                }
                Ok(())
            }
        };
        if INTERIOR && has_no_edge_interior(width, y, policy.previous_channels) {
            visit_source_row::<false>(channel, y, 0..2, &mut pixel)?;
            visit_source_row::<true>(channel, y, 2..width - 2, &mut pixel)?;
            visit_source_row::<false>(channel, y, width - 2..width, &mut pixel)?;
        } else {
            visit_source_row::<false>(channel, y, 0..width, &mut pixel)?;
        }
    }
    debug_assert_eq!(sampler.len(), (height - rows.end) * width);
    Ok(())
}

#[cfg(test)]
mod tests;
