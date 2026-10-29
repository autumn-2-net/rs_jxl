// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Node split orchestration: parent/cache admission, property proofs and task dispatch.
//! Ordered reduction publishes a cache and redundant-property proofs only after success.

#[cfg(test)]
use super::InitialCacheStage;
use super::{
    Arc, Candidates, DefaultScorer, DenseHistogram, HistogramBudget, HistogramCache, Mutex,
    Ordering, ParentHistogram, PredictorPlan, PropertyColumn, PropertyHistogram, PropertyScratch,
    PropertyValue, SampleStorage, Samples, ScratchPool, SearchOptions, SparseHistogramPlan, Split,
    TrainOptions, TrainingStats, WP_PROPERTY, choose_split, collect_property_marginals,
    compact_parent_histograms_with_layout, evaluate_property_values, execution, histogram_stride,
    keep_better, ma_partition_proofs, ma_search, parent_histogram,
};
use crate::Result;

pub(super) struct SeenProperty {
    pub(super) signature: u64,
    pub(super) property: usize,
}

pub(super) fn property_signature(used: &[usize]) -> u64 {
    let mut signature = 14695981039346656037u64;
    for (value, &count) in used.iter().enumerate() {
        if count != 0 {
            signature = (signature ^ value as u64).wrapping_mul(1099511628211);
            signature = (signature ^ count as u64).wrapping_mul(1099511628211);
        }
    }
    signature
}

// enc_ma.cc::SeenProperty / enc_ma.h::SameProperty. Signatures only filter
// candidates; equality must cover every quantized value in the live node.
#[allow(clippy::too_many_arguments)]
pub(super) fn duplicate_property<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    property: usize,
    values: &[T],
    used: &[usize],
    seen: &mut Vec<SeenProperty>,
    remap: &mut Vec<usize>,
) -> bool {
    // Search experiments may select thresholds by their original bin index.
    // Only a complete sweep permits ignoring strictly monotone bin relabeling.
    let relabel = options.search == SearchOptions::default();
    let signature = if relabel {
        ma_partition_proofs::signature(used)
    } else {
        property_signature(used)
    };
    let id = options.properties[property];
    for previous in seen.iter() {
        let previous_id = options.properties[previous.property];
        if previous.signature != signature
            || (previous_id < 2) != (id < 2)
            || (previous_id == WP_PROPERTY) != (id == WP_PROPERTY)
        {
            continue;
        }
        let same = with_property_values!(&samples.properties[previous.property], |other| {
            if relabel {
                ma_partition_proofs::same_partition(indices, values, other, used.len(), remap)
            } else {
                indices
                    .iter()
                    .all(|&i| values[i].index() == other[i].index())
            }
        });
        if same {
            return true;
        }
    }
    seen.push(SeenProperty {
        signature,
        property,
    });
    false
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn find_best_split(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    parent_predictor: usize,
    options: &TrainOptions,
    parents: &mut Option<Vec<ParentHistogram>>,
    cache: &mut Option<HistogramCache>,
    budget: &Arc<HistogramBudget>,
    stats: &TrainingStats,
) -> Result<Option<Split>> {
    find_best_split_with_scratch(
        samples,
        indices,
        parent_predictor,
        options,
        parents,
        cache,
        budget,
        stats,
        &ScratchPool::new(1),
        &[],
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn find_best_split_with_scratch(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    parent_predictor: usize,
    options: &TrainOptions,
    parents: &mut Option<Vec<ParentHistogram>>,
    cache: &mut Option<HistogramCache>,
    budget: &Arc<HistogramBudget>,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
    removed: &[usize],
) -> Result<Option<Split>> {
    find_best_split_with_properties(
        samples,
        indices,
        parent_predictor,
        options,
        parents,
        cache,
        budget,
        stats,
        scratch_pool,
        removed,
        &mut Vec::new(),
    )
}

// FindBestSplit keeps node proofs separate from invocation-owned scratch/cache.
#[allow(clippy::too_many_arguments)]
pub(super) fn find_best_split_with_properties(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    parent_predictor: usize,
    options: &TrainOptions,
    parents: &mut Option<Vec<ParentHistogram>>,
    cache: &mut Option<HistogramCache>,
    budget: &Arc<HistogramBudget>,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
    removed: &[usize],
    redundant_properties: &mut Vec<u8>,
) -> Result<Option<Split>> {
    let _scope = crate::profile::scope("ma.find_best_split");
    debug_assert!(removed.is_empty() || cache.is_some());
    if indices.len() < 2 {
        *cache = None;
        return Ok(None);
    }
    let mut leaf_parent = None;
    if parents.is_none() {
        let _scope = crate::profile::scope("ma.leaf_histogram");
        leaf_parent = Some(parent_histogram(samples, indices, parent_predictor));
    }
    let parent = parents.as_ref().map_or_else(
        || leaf_parent.as_ref().expect("uncached leaf histogram"),
        |parents| &parents[parent_predictor],
    );
    let total: u32 = parent.counts.iter().sum();
    let scorer = DefaultScorer::default();
    let base_bits = options.scoring.leaf_bits(&scorer, &parent.counts, total)
        + parent.extra_bits as f32
        + options
            .scoring
            .description
            .cost(&scorer, &parent.counts, total);
    if base_bits <= options.split_threshold as f32 {
        *cache = None;
        return Ok(None);
    }
    let parents = {
        let _scope = crate::profile::scope("ma.parent_histograms");
        parents.get_or_insert_with(|| {
            let leaf_parent = leaf_parent.expect("uncached leaf histogram");
            let collect = |predictor| {
                if predictor == parent_predictor {
                    leaf_parent.clone()
                } else {
                    parent_histogram(samples, indices, predictor)
                }
            };
            let predictors = samples.residuals.len();
            if options.parallel
                && predictors > 2
                && indices.len() >= 8192
                && indices.len().saturating_mul(predictors - 1) >= 65536
            {
                scratch_pool
                    .dispatch_slots
                    .store(execution::worker_count(), Ordering::Relaxed);
                execution::map_with_state(predictors, || (), |_, p| collect(p))
            } else {
                (0..predictors).map(collect).collect()
            }
        })
    };
    let plan = PredictorPlan::new(samples, indices, options, parents, parent_predictor);
    stats
        .equivalent_predictors
        .fetch_add(parents.len() - plan.active_count, Ordering::Relaxed);
    // Only publish a fully evaluated cache. A task error or unwind drops the
    // local cache/lease after all borrowed worker tasks have finished, leaving
    // no partially filled or subtracted cache visible to the caller.
    let mut working_cache = cache.take();
    #[cfg(test)]
    let inherited_cache = working_cache.is_some();
    if working_cache.is_some() {
        stats.cache_reuses.fetch_add(1, Ordering::Relaxed);
    } else if options.search.histogram_bins == 0
        && indices.len() >= options.histogram_cache_thresholds().0
    {
        let _scope = crate::profile::scope("ma.histogram_cache");
        let stride = options.padded_histogram.then(|| {
            histogram_stride(
                parents.iter().map(|p| p.counts.len()).max().unwrap_or(1),
                scorer.lanes(),
                true,
            )
        });
        working_cache = HistogramCache::reserve_with_stride(samples, parents, budget, stride);
        if working_cache.is_some() {
            stats.cache_builds.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[cfg(test)]
    let initialize_cache = !inherited_cache && working_cache.is_some();
    let _sweep_scope = crate::profile::scope("ma.split_sweep");
    let work = indices
        .len()
        .saturating_mul(options.predictors.len())
        .saturating_mul(options.properties.len());
    let parallel =
        options.parallel && indices.len() >= 1024 && work >= 65536 && execution::worker_count() > 1;
    // Publish proofs only when every property succeeds, alongside its cache.
    let mut working_redundant = if options.property_dedup {
        if redundant_properties.is_empty() {
            vec![0; options.properties.len()]
        } else {
            debug_assert_eq!(redundant_properties.len(), options.properties.len());
            redundant_properties.clone()
        }
    } else {
        Vec::new()
    };
    let compact_limit = if options.dense_histogram == DenseHistogram::Extended {
        4096
    } else {
        256
    };
    let compact = if !options.scoring.fast_entropy
        && working_cache.is_none()
        && indices.len() <= compact_limit
    {
        compact_parent_histograms_with_layout(
            parents,
            scorer.lanes(),
            options.dense_histogram,
            options.padded_histogram,
        )
    } else {
        None
    };
    if compact.is_some() {
        stats.compact_nodes.fetch_add(1, Ordering::Relaxed);
    }
    // Preserve raw parents for child subtraction and raw-token cache lookup.
    let scoring_parents = compact
        .as_ref()
        .map_or(parents.as_slice(), |c| c.parents.as_slice());
    let sparse_plan = SparseHistogramPlan::new(
        scoring_parents,
        plan.active(),
        scorer.lanes(),
        options.sparse_entropy && !options.scoring.fast_entropy,
    );
    let race = ma_search::race(
        samples,
        indices,
        options,
        histogram_stride(
            parents.iter().map(|p| p.counts.len()).max().unwrap_or(1),
            scorer.lanes(),
            options.padded_histogram,
        ),
        working_cache
            .as_ref()
            .is_some_and(|cache| cache.properties.iter().any(|p| p.valid)),
    )?;
    if !race.keep.is_empty() {
        #[cfg(test)]
        stats.race_ranges.lock().unwrap().push((
            samples.base() + indices[0],
            samples.base() + indices[0] + indices.len(),
        ));
        stats.race_nodes.fetch_add(1, Ordering::Relaxed);
        stats.race_draws.fetch_add(race.draws, Ordering::Relaxed);
        stats.race_pruned.fetch_add(
            race.keep.iter().filter(|&&keep| keep == 0).count(),
            Ordering::Relaxed,
        );
        if let Some(cache) = &mut working_cache {
            for (histogram, &keep) in cache.properties.iter_mut().zip(&race.keep) {
                if keep == 0 {
                    histogram.valid = false;
                }
            }
        }
    }
    let mut winners = [None; 4];
    let candidates: Vec<Result<(Candidates, bool)>> = {
        let cache_stride = working_cache.as_ref().and_then(|cache| cache.stride);
        let collection_stride = histogram_stride(
            parents.iter().map(|p| p.counts.len()).max().unwrap_or(1),
            scorer.lanes(),
            options.padded_histogram,
        );
        // Each indexed task owns a different mutable property. These temporary
        // mutexes adapt disjoint borrows to execution's shared Fn callback;
        // no cache-wide lock, copying, or new scheduler is needed.
        let property_jobs: Option<Vec<_>> = working_cache
            .as_mut()
            .map(|cache| cache.properties.iter_mut().map(Mutex::new).collect());
        let evaluate = |scratch: &mut PropertyScratch,
                        property: usize,
                        seen: Option<&mut Vec<SeenProperty>>| {
            if options.property_dedup && working_redundant[property] != 0 {
                stats.inherited_properties.fetch_add(1, Ordering::Relaxed);
                return Ok(([None; 4], true));
            }
            if !race.keep.is_empty() && race.keep[property] == 0 {
                return Ok(([None; 4], false));
            }
            with_property_values!(&samples.properties[property], |values| {
                let mut histogram = property_jobs
                    .as_ref()
                    .map(|jobs| jobs[property].lock().expect("MA property cache poisoned"));
                let cached_property = histogram.as_ref().is_some_and(|h| h.valid);
                let cache_scope = histogram.as_ref().map(|_| {
                    crate::profile::scope(if cached_property {
                        "ma.cache_subtract"
                    } else {
                        "ma.cache_collect"
                    })
                });
                if options.property_dedup {
                    if let Some(histogram) = histogram.as_mut() {
                        if cached_property {
                            histogram.subtract_marginals_values(samples, removed, values)?;
                        } else {
                            ***histogram = PropertyHistogram::collect_marginals_values(
                                samples, indices, property, values,
                            );
                        }
                    } else {
                        collect_property_marginals(
                            samples,
                            indices,
                            values,
                            samples.thresholds[property].len() + 1,
                            &mut scratch.used,
                            &mut scratch.weights,
                        );
                    }
                    let used = histogram
                        .as_ref()
                        .map_or(scratch.used.as_slice(), |h| h.used.as_slice());
                    if used.iter().filter(|&&n| n != 0).take(2).count() < 2 {
                        stats.constant_properties.fetch_add(1, Ordering::Relaxed);
                        return Ok(([None; 4], true));
                    }
                    if indices.len() >= 64
                        && let Some(seen) = seen
                        && duplicate_property(
                            samples,
                            indices,
                            options,
                            property,
                            values,
                            used,
                            seen,
                            &mut scratch.property_bins,
                        )
                    {
                        // Source Invalidate retains storage but forbids reuse of
                        // token tables that still describe the parent range.
                        if let Some(histogram) = histogram.as_mut() {
                            histogram.valid = false;
                        }
                        stats.duplicate_properties.fetch_add(1, Ordering::Relaxed);
                        return Ok(([None; 4], true));
                    }
                    if let Some(histogram) = histogram.as_mut() {
                        if cached_property {
                            if options.paired_cache && indices.len() >= 256 {
                                let pairs = histogram.subtract_paired_predictors_values(
                                    samples, removed, values, &plan,
                                )?;
                                stats
                                    .paired_cache_subtractions
                                    .fetch_add(pairs, Ordering::Relaxed);
                            } else {
                                histogram
                                    .subtract_predictors_values(samples, removed, values, &plan)?;
                            }
                        } else if options.histogram_shards {
                            histogram.collect_sharded_predictors_values(
                                samples,
                                indices,
                                parents,
                                values,
                                &plan,
                                cache_stride,
                                collection_stride,
                                &mut scratch.shards,
                            )?;
                        } else {
                            histogram.collect_predictors_values(
                                samples,
                                indices,
                                parents,
                                values,
                                &plan,
                                cache_stride,
                            )?;
                        }
                    }
                } else if let Some(histogram) = histogram.as_mut() {
                    if !cached_property {
                        // enc_ma.cc fills and scores on the same property worker.
                        // No second dispatch or all-properties fill barrier.
                        if options.histogram_shards {
                            ***histogram = PropertyHistogram::collect_marginals_values(
                                samples, indices, property, values,
                            );
                            histogram.collect_sharded_predictors_values(
                                samples,
                                indices,
                                parents,
                                values,
                                &plan,
                                cache_stride,
                                collection_stride,
                                &mut scratch.shards,
                            )?;
                        } else {
                            ***histogram = PropertyHistogram::collect_values_with_stride(
                                samples,
                                indices,
                                parents,
                                property,
                                values,
                                &plan,
                                cache_stride,
                            )?;
                        }
                    } else {
                        if options.paired_cache && indices.len() >= 256 {
                            histogram.subtract_marginals_values(samples, removed, values)?;
                            let pairs = histogram.subtract_paired_predictors_values(
                                samples, removed, values, &plan,
                            )?;
                            stats
                                .paired_cache_subtractions
                                .fetch_add(pairs, Ordering::Relaxed);
                        } else {
                            histogram.subtract_values(samples, removed, values, &plan)?;
                        }
                    }
                }
                drop(cache_scope);
                if cached_property && !removed.is_empty() {
                    stats.cache_subtractions.fetch_add(1, Ordering::Relaxed);
                }
                #[cfg(test)]
                if initialize_cache && let Some(hook) = &stats.initial_cache_hook {
                    hook(property, InitialCacheStage::Filled)?;
                }
                let candidates = evaluate_property_values(
                    samples,
                    indices,
                    options,
                    scoring_parents,
                    &plan,
                    &sparse_plan,
                    total,
                    property,
                    values,
                    histogram.as_ref().map(|histogram| &***histogram),
                    options.property_dedup,
                    compact.as_ref().map(|c| &c.mapping),
                    scratch,
                )?;
                if scratch.dominated_predictors != 0 {
                    stats
                        .dominated_predictors
                        .fetch_add(scratch.dominated_predictors, Ordering::Relaxed);
                }
                #[cfg(test)]
                if initialize_cache && let Some(hook) = &stats.initial_cache_hook {
                    hook(property, InitialCacheStage::Scored)?;
                }
                Ok((candidates, false))
            })
        };
        if parallel {
            scratch_pool
                .dispatch_slots
                .store(execution::worker_count(), Ordering::Relaxed);
            execution::map_with_state(
                options.properties.len(),
                || scratch_pool.acquire(),
                |lease, property| evaluate(&mut lease.scratch, property, None),
            )
        } else if !options.property_dedup {
            // Tiny serial nodes dominate the tree count. Reduce their results
            // immediately instead of allocating a vector of four-way candidates.
            let mut lease = scratch_pool.acquire();
            let mut first_error = None;
            for property in 0..options.properties.len() {
                match evaluate(&mut lease.scratch, property, None) {
                    Ok((candidates, _)) => {
                        for (best, candidate) in winners.iter_mut().zip(candidates) {
                            if let Some(candidate) = candidate {
                                keep_better(best, candidate);
                            }
                        }
                    }
                    Err(error) => {
                        first_error.get_or_insert(error);
                    }
                }
            }
            // Like indexed parallel collection, drain the batch before
            // returning its first ordered error; never publish partial caches.
            if let Some(error) = first_error {
                return Err(error);
            }
            Vec::new()
        } else {
            let mut lease = scratch_pool.acquire();
            let mut seen = Vec::new();
            (0..options.properties.len())
                .map(|property| evaluate(&mut lease.scratch, property, Some(&mut seen)))
                .collect()
        }
    };
    // Indexed collection followed by strict ordered reduction preserves ties
    // independently of Rayon completion order. No worker mutates samples.
    for (property, candidates) in candidates.into_iter().enumerate() {
        let (candidates, redundant) = candidates?;
        if options.property_dedup {
            working_redundant[property] = u8::from(redundant);
        }
        for (best, candidate) in winners.iter_mut().zip(candidates) {
            if let Some(candidate) = candidate {
                keep_better(best, candidate);
            }
        }
    }
    let decision = choose_split(winners, base_bits, options);
    *cache = working_cache;
    *redundant_properties = working_redundant;
    Ok(decision)
}
