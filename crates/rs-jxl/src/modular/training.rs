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
use super::tree::{Properties, Tree};
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
pub(super) fn learn_local(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
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
    learn(streams, &local)
}

/// Explicit source gather policy passed through local and grouped learning.
/// None retains the original serial gather, including grouped learning.
pub(super) fn learn_with_gather(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    gather: Option<&GatherOptions>,
) -> Result<Tree> {
    if let Some(gather) = gather {
        gather.validate()?;
        if options.sampling.is_none() {
            return Err(Error::InvalidInput(
                "MA gather scheduling requires source sampling".into(),
            ));
        }
    }
    let tree = if let Some(grouping) = &options.tree_groups {
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
        learn_with_budget_and_gather(streams, options, None, true, gather)?
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
        .collect_prepared(gather.filter(|_| parallel_search).unwrap_or(&serial_gather))?;
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
fn source_predictors(mode: Predictor, policy: &SamplingPolicy) -> Result<Vec<Predictor>> {
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

    fn collect_prepared(
        &self,
        options: &GatherOptions,
    ) -> Result<GatheredSamples<PreparedSamples>> {
        options.validate()?;
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
                        let mut batch = SourceSampleCollector::new(
                            self.predictors,
                            self.quantization,
                            options.cache_hashes,
                        )?;
                        let selected = self.collect_tasks(
                            &tasks[range.clone()],
                            local_budget.as_ref(),
                            &mut batch,
                            options.sparse,
                        )?;
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
                output.stats.selected_samples += self.collect_tasks(
                    &tasks[wave.tasks],
                    local_budget.as_ref(),
                    &mut master,
                    options.sparse,
                )?;
            }
        }
        let samples = master.finish();
        debug_assert_eq!(samples.total, output.stats.selected_samples);
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
    ) -> Result<u64> {
        let mut selected = 0;
        for task in tasks {
            let stream = self.streams[task.stream];
            let plane = &stream.channels[task.channel];
            let area = plane
                .width
                .checked_mul(plane.height)
                .ok_or(Error::Limit("MA training plane dimensions overflow"))?;
            let fraction = sample::tree_sample_fraction(area, self.policy.nb_repeats)?;
            collector.prepare(gather_samples::expected_samples(
                fraction,
                plane.width,
                task.rows.len(),
            )?)?;
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
            )?;
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

#[inline]
fn visit_source_row<const NO_EDGE: bool>(
    channel: &Channel,
    y: usize,
    columns: std::ops::Range<usize>,
    mut pixel: impl FnMut(usize, Neighbors) -> Result<()>,
) -> Result<()> {
    let width = channel.width;
    if NO_EDGE {
        debug_assert!(has_no_edge_interior(width, y, 0));
        debug_assert!(columns.start >= 2 && columns.end <= width - 2);
        let begin = y * width;
        let row = &channel.samples[begin..begin + width];
        let top = &channel.samples[begin - width..begin];
        let toptop = &channel.samples[begin - width - width..begin - width];
        for x in columns {
            pixel(
                x,
                Neighbors {
                    left: row[x - 1],
                    top: top[x],
                    toptop: toptop[x],
                    topleft: top[x - 1],
                    topright: top[x + 1],
                    leftleft: row[x - 2],
                    toprightright: top[x + 2],
                },
            )?;
        }
    } else {
        for x in columns {
            let neighbors = Neighbors::from_plane(&channel.samples, width, width, x, y)
                .map_err(|e| Error::Encoding(e.to_string()))?;
            pixel(x, neighbors)?;
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
        )
    }
}

// INTERIOR=false is a test-only traversal oracle. The warm-up changes learning
// state only; pixel coding never calls this path with a truncated history.
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
    let mut property_state = Properties::default();
    let mut previous_gradient = 0i32;
    let mut properties = vec![0; 16 + 4 * policy.previous_channels];
    let mut residuals = [0; 14];
    for y in warm_begin..rows.end {
        let mut pixel = |x: usize, n: Neighbors| {
            let position = y * width + x;
            let value = channel.samples[position];
            // Draw even in warm-up rows, matching the source's operand order.
            let selected =
                sampler.next().expect("one sample draw per channel pixel") && y >= rows.start;
            let (prediction, error) = if let Some(state) = &mut state {
                // State's checked raster coordinates start at zero. Shifting
                // both parity buffers by warm_begin only renames zeroed rows;
                // all spatial neighbors/properties retain global coordinates.
                // The local state is discarded if collection fails later.
                state.predict_and_update(
                    value,
                    x,
                    y - warm_begin,
                    n.top,
                    n.left,
                    n.topright,
                    n.topleft,
                    n.toptop,
                )?
            } else {
                (0, 0)
            };
            if !sparse || selected {
                properties[..16].copy_from_slice(&property_state.calculate(
                    n,
                    x,
                    y,
                    channel_index,
                    stream.stream_id,
                    error,
                ));
                // AdvanceLearnState updates the previous *unclamped* gradient
                // even when the other properties were not materialized.
                properties[8] = n
                    .left
                    .wrapping_sub(if x == 0 { 0 } else { previous_gradient });
            }
            previous_gradient = n.left.wrapping_add(n.top).wrapping_sub(n.topleft);
            if selected {
                before_collect()?;
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
                    residuals[1] = value.wrapping_sub(clamped_gradient(n.left, n.top, n.topleft));
                } else {
                    for (residual, &predictor) in residuals.iter_mut().zip(predictors) {
                        let guess = predict_one(predictor, n, Some(i64::from(prediction)))
                            .map_err(|e| Error::Encoding(e.to_string()))?;
                        *residual = (i64::from(value) - guess) as i32;
                    }
                }
                collect(&properties, &residuals[..predictors.len()])?;
            }
            Ok(())
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
