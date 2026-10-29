// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Property histogram population/subtraction and invocation-wide cache leases.
//! Cache admission and publication policy belongs to `split_search`, not this storage.

#[cfg(test)]
use super::Mutex;
use super::{
    Arc, AtomicUsize, HistogramShards, OnceLock, Ordering, ParentHistogram, PredictorPlan,
    PropertyColumn, PropertyValue, ResidualToken, SampleStorage, Samples, TrainOptions,
    collect_mapped, execution,
};
use crate::{Error, Result};

fn cache_pair14_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("RS_JXL_MA_CACHE_PAIR14").as_deref() == Ok("1"))
}

pub(super) struct HistogramBudget {
    pub(super) limit: usize,
    pub(super) used: AtomicUsize,
    pub(super) peak: AtomicUsize,
}

impl HistogramBudget {
    pub(super) fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }
    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> Option<HistogramLease> {
        let previous = self
            .used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes).filter(|&next| next <= self.limit)
            })
            .ok()?;
        self.peak.fetch_max(previous + bytes, Ordering::Relaxed);
        Some(HistogramLease {
            budget: Arc::clone(self),
            bytes,
        })
    }
}

pub(super) struct HistogramLease {
    pub(super) budget: Arc<HistogramBudget>,
    pub(super) bytes: usize,
}
impl Drop for HistogramLease {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

pub(super) struct BinHistogram {
    pub(super) symbols: usize,
    pub(super) counts: Vec<u32>,
    pub(super) extra: Vec<u64>,
}

impl BinHistogram {
    #[inline]
    fn subtract(&mut self, residual: ResidualToken, bin: usize, weight: u32) -> Result<()> {
        let cell = &mut self.counts[bin * self.symbols + usize::from(residual.token)];
        *cell = cell
            .checked_sub(weight)
            .ok_or_else(|| Error::Encoding("MA cached token subtraction underflow".into()))?;
        self.extra[bin] = self.extra[bin]
            .checked_sub(u64::from(weight) * u64::from(residual.nbits))
            .ok_or_else(|| Error::Encoding("MA cached extra-bit subtraction underflow".into()))?;
        Ok(())
    }

    pub(super) fn copy_predictor_from(&mut self, source: &Self) -> Result<()> {
        debug_assert_eq!(self.extra.len(), source.extra.len());
        let shared = self.symbols.min(source.symbols);
        for (destination, source) in self
            .counts
            .chunks_exact_mut(self.symbols)
            .zip(source.counts.chunks_exact(source.symbols))
        {
            // Source uses a common stride; Rust retains per-predictor strides.
            // After subtraction, equal live tokens imply zero discarded tails.
            // Keep both allocations unchanged under the original cache lease.
            if source[shared..].iter().any(|&count| count != 0) {
                return Err(Error::Encoding(
                    "MA predictor cache copy has a nonzero tail".into(),
                ));
            }
            destination[..shared].copy_from_slice(&source[..shared]);
            destination[shared..].fill(0);
        }
        self.extra.copy_from_slice(&source.extra);
        Ok(())
    }
}
#[derive(Default)]
pub(super) struct PropertyHistogram {
    pub(super) used: Vec<usize>,
    pub(super) weights: Vec<u32>,
    pub(super) predictors: Vec<BinHistogram>,
    pub(super) valid: bool,
}

pub(super) fn collect_property_marginals<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    bins: usize,
    used: &mut Vec<usize>,
    weights: &mut Vec<u32>,
) {
    used.clear();
    used.resize(bins, 0);
    weights.clear();
    weights.resize(bins, 0);
    for &i in indices {
        let bin = values[i].index();
        used[bin] += 1;
        weights[bin] += u32::from(samples.counts[i]);
    }
}

impl PropertyHistogram {
    #[cfg(test)]
    pub(super) fn collect(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        property: usize,
    ) -> Self {
        with_property_values!(&samples.properties[property], |values| {
            Self::collect_values(
                samples,
                indices,
                parents,
                property,
                values,
                &PredictorPlan::identity(parents.len()),
            )
            .expect("identity predictor cache collection")
        })
    }

    #[cfg(test)]
    pub(super) fn collect_values<T: PropertyValue>(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        property: usize,
        values: &[T],
        plan: &PredictorPlan,
    ) -> Result<Self> {
        Self::collect_values_with_stride(samples, indices, parents, property, values, plan, None)
    }

    pub(super) fn collect_values_with_stride<T: PropertyValue>(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        property: usize,
        values: &[T],
        plan: &PredictorPlan,
        stride: Option<usize>,
    ) -> Result<Self> {
        let mut histogram = Self::collect_marginals_values(samples, indices, property, values);
        histogram.collect_predictors_values(samples, indices, parents, values, plan, stride)?;
        Ok(histogram)
    }

    pub(super) fn collect_marginals_values<T: PropertyValue>(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        property: usize,
        values: &[T],
    ) -> Self {
        let bins = samples.thresholds[property].len() + 1;
        let mut histogram = Self::default();
        collect_property_marginals(
            samples,
            indices,
            values,
            bins,
            &mut histogram.used,
            &mut histogram.weights,
        );
        histogram
    }

    pub(super) fn collect_predictors_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        values: &[T],
        plan: &PredictorPlan,
        stride: Option<usize>,
    ) -> Result<()> {
        self.collect_predictors_values_with_pair14(
            samples,
            indices,
            parents,
            values,
            plan,
            stride,
            parents.len() == 14 && cache_pair14_enabled(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn collect_predictors_values_with_pair14<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        values: &[T],
        plan: &PredictorPlan,
        stride: Option<usize>,
        pair14: bool,
    ) -> Result<()> {
        if !self.allocate_predictors(parents, stride) {
            return Ok(());
        }
        if pair14 && plan.cache_pair14_eligible(self.predictors.len()) {
            // Touch two histogram columns per pass without extra allocations.
            for predictor in (0..14).step_by(2) {
                let (first, rest) = self.predictors.split_at_mut(predictor + 1);
                let (first, second) = (&mut first[predictor], &mut rest[0]);
                let (a, b) = (
                    &samples.residuals[predictor][..],
                    &samples.residuals[predictor + 1][..],
                );
                for &i in indices {
                    let bin = values[i].index();
                    let weight = u32::from(samples.counts[i]);
                    let residual = a[i];
                    first.counts[bin * first.symbols + usize::from(residual.token)] += weight;
                    first.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
                    let residual = b[i];
                    second.counts[bin * second.symbols + usize::from(residual.token)] += weight;
                    second.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
                }
            }
        } else if self.predictors.len() == 2 && plan.active() == [0, 1] {
            // Hoist the two column/histogram lookups out of the record loop.
            let (first, rest) = self.predictors.split_at_mut(1);
            let (first, second) = (&mut first[0], &mut rest[0]);
            let (a, b) = (&samples.residuals[0][..], &samples.residuals[1][..]);
            for &i in indices {
                let bin = values[i].index();
                let weight = u32::from(samples.counts[i]);
                let residual = a[i];
                first.counts[bin * first.symbols + usize::from(residual.token)] += weight;
                first.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
                let residual = b[i];
                second.counts[bin * second.symbols + usize::from(residual.token)] += weight;
                second.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
            }
        } else {
            // Source fused collection: share property/weight loads.
            for &i in indices {
                let bin = values[i].index();
                let weight = u32::from(samples.counts[i]);
                for &predictor in plan.active() {
                    let histogram = &mut self.predictors[predictor];
                    let residual = samples.residuals[predictor][i];
                    histogram.counts[bin * histogram.symbols + usize::from(residual.token)] +=
                        weight;
                    histogram.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
                }
            }
        }
        self.copy_predictors(plan)?;
        self.valid = true;
        Ok(())
    }

    pub(super) fn allocate_predictors(
        &mut self,
        parents: &[ParentHistogram],
        stride: Option<usize>,
    ) -> bool {
        // A subset of a constant property stays constant. Retain only its
        // partition counts; neither this node nor descendants need tokens.
        if self.used.iter().filter(|&&n| n != 0).take(2).count() < 2 {
            self.valid = true;
            return false;
        }
        let bins = self.used.len();
        self.predictors = parents
            .iter()
            .map(|parent| BinHistogram {
                symbols: stride.unwrap_or(parent.counts.len()),
                counts: vec![0; bins * stride.unwrap_or(parent.counts.len())],
                extra: vec![0; bins],
            })
            .collect();
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn collect_sharded_predictors_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        values: &[T],
        plan: &PredictorPlan,
        stride: Option<usize>,
        node_stride: usize,
        scratch: &mut HistogramShards,
    ) -> Result<usize> {
        if !self.allocate_predictors(parents, stride) {
            return Ok(0);
        }
        let bins = self.used.len();
        let mut sharded = 0;
        for &predictor in plan.active() {
            if indices.len() >= 256 && plan.pair_available(predictor, parents.len()) {
                if predictor % 2 == 1 {
                    continue;
                }
                let (first, rest) = self.predictors.split_at_mut(predictor + 1);
                let (first, second) = (&mut first[predictor], &mut rest[0]);
                for &i in indices {
                    let bin = values[i].index();
                    let weight = u32::from(samples.counts[i]);
                    let a = samples.residuals[predictor][i];
                    let b = samples.residuals[predictor + 1][i];
                    first.counts[bin * first.symbols + usize::from(a.token)] += weight;
                    second.counts[bin * second.symbols + usize::from(b.token)] += weight;
                    first.extra[bin] += u64::from(weight) * u64::from(a.nbits);
                    second.extra[bin] += u64::from(weight) * u64::from(b.nbits);
                }
            } else {
                let histogram = &mut self.predictors[predictor];
                if indices.len() >= 4096 && indices.len() >= 4 * bins * node_stride {
                    scratch.collect(
                        samples,
                        indices,
                        values,
                        &samples.residuals[predictor],
                        histogram.symbols,
                        bins,
                        &mut histogram.counts,
                        &mut histogram.extra,
                    )?;
                    sharded += 1;
                } else {
                    collect_mapped::<T, false, false>(
                        samples,
                        indices,
                        values,
                        &samples.residuals[predictor],
                        histogram.symbols,
                        &[],
                        None,
                        &mut histogram.counts,
                        &mut histogram.extra,
                    );
                }
            }
        }
        self.copy_predictors(plan)?;
        self.valid = true;
        Ok(sharded)
    }

    #[cfg(test)]
    pub(super) fn subtract(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        property: usize,
        removed: &[usize],
    ) -> Result<()> {
        with_property_values!(&samples.properties[property], |values| {
            self.subtract_values(
                samples,
                removed,
                values,
                &PredictorPlan::identity(samples.residuals.len()),
            )
        })
    }

    pub(super) fn subtract_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        values: &[T],
        plan: &PredictorPlan,
    ) -> Result<()> {
        if self.predictors.len() == 2 && plan.active() == [0, 1] {
            let (first, rest) = self.predictors.split_at_mut(1);
            let (first, second) = (&mut first[0], &mut rest[0]);
            let (a, b) = (&samples.residuals[0][..], &samples.residuals[1][..]);
            for &i in removed {
                let bin = values[i].index();
                let weight = u32::from(samples.counts[i]);
                // Preserve record/weight/first/second failure order without
                // introducing the separate marginal pass of paired_cache.
                self.used[bin] = self.used[bin].checked_sub(1).ok_or_else(|| {
                    Error::Encoding("MA cached record subtraction underflow".into())
                })?;
                self.weights[bin] = self.weights[bin].checked_sub(weight).ok_or_else(|| {
                    Error::Encoding("MA cached weight subtraction underflow".into())
                })?;
                first.subtract(a[i], bin, weight)?;
                second.subtract(b[i], bin, weight)?;
            }
        } else {
            for &i in removed {
                let bin = values[i].index();
                let weight = u32::from(samples.counts[i]);
                self.subtract_marginal(bin, weight)?;
                self.subtract_predictors(samples, i, bin, weight, plan)?;
            }
        }
        self.copy_predictors(plan)?;
        Ok(())
    }

    pub(super) fn subtract_marginal(&mut self, bin: usize, weight: u32) -> Result<()> {
        self.used[bin] = self.used[bin]
            .checked_sub(1)
            .ok_or_else(|| Error::Encoding("MA cached record subtraction underflow".into()))?;
        self.weights[bin] = self.weights[bin]
            .checked_sub(weight)
            .ok_or_else(|| Error::Encoding("MA cached weight subtraction underflow".into()))?;
        Ok(())
    }

    pub(super) fn subtract_predictors(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        i: usize,
        bin: usize,
        weight: u32,
        plan: &PredictorPlan,
    ) -> Result<()> {
        let predictor_count = self.predictors.len();
        for &predictor in plan.active().iter().filter(|&&p| p < predictor_count) {
            self.predictors[predictor].subtract(samples.residuals[predictor][i], bin, weight)?;
        }
        Ok(())
    }

    pub(super) fn subtract_marginals_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        values: &[T],
    ) -> Result<()> {
        for &i in removed {
            self.subtract_marginal(values[i].index(), u32::from(samples.counts[i]))?;
        }
        Ok(())
    }

    pub(super) fn subtract_predictors_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        values: &[T],
        plan: &PredictorPlan,
    ) -> Result<()> {
        for &i in removed {
            self.subtract_predictors(
                samples,
                i,
                values[i].index(),
                u32::from(samples.counts[i]),
                plan,
            )?;
        }
        self.copy_predictors(plan)?;
        Ok(())
    }

    pub(super) fn subtract_paired_predictors_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        values: &[T],
        plan: &PredictorPlan,
    ) -> Result<usize> {
        let count = self.predictors.len();
        let mut pairs = 0;
        for &predictor in plan.active().iter().filter(|&&p| p < count) {
            if plan.pair_available(predictor, count) {
                if predictor % 2 == 1 {
                    continue;
                }
                let (first, rest) = self.predictors.split_at_mut(predictor + 1);
                let (first, second) = (&mut first[predictor], &mut rest[0]);
                for &i in removed {
                    let bin = values[i].index();
                    let weight = u32::from(samples.counts[i]);
                    first.subtract(samples.residuals[predictor][i], bin, weight)?;
                    second.subtract(samples.residuals[predictor + 1][i], bin, weight)?;
                }
                pairs += 1;
            } else {
                for &i in removed {
                    self.predictors[predictor].subtract(
                        samples.residuals[predictor][i],
                        values[i].index(),
                        u32::from(samples.counts[i]),
                    )?;
                }
            }
        }
        self.copy_predictors(plan)?;
        Ok(pairs)
    }

    pub(super) fn copy_predictors(&mut self, plan: &PredictorPlan) -> Result<()> {
        for predictor in 0..self.predictors.len() {
            let representative = plan.representative[predictor];
            if representative != predictor {
                let (previous, current) = self.predictors.split_at_mut(predictor);
                current[0].copy_predictor_from(&previous[representative])?;
            }
        }
        Ok(())
    }
}

pub(super) struct HistogramCache {
    pub(super) properties: Vec<PropertyHistogram>,
    pub(super) _lease: HistogramLease,
    pub(super) stride: Option<usize>,
}

impl HistogramCache {
    // Reserve the complete conservative lease before any property allocation.
    // Entries stay empty until their scoring tasks fill them. This is separate
    // from find_best_split's record-count creation gate (8192 by default).
    #[cfg(test)]
    pub(super) fn reserve(
        samples: &Samples<impl SampleStorage>,
        parents: &[ParentHistogram],
        budget: &Arc<HistogramBudget>,
    ) -> Option<Self> {
        Self::reserve_with_stride(samples, parents, budget, None)
    }

    pub(super) fn reserve_with_stride(
        samples: &Samples<impl SampleStorage>,
        parents: &[ParentHistogram],
        budget: &Arc<HistogramBudget>,
        stride: Option<usize>,
    ) -> Option<Self> {
        if budget.limit == 0 {
            return None;
        }
        let mut bytes = std::mem::size_of::<Self>();
        for thresholds in samples.thresholds.iter() {
            let bins = thresholds.len() + 1;
            bytes = bytes
                .checked_add(std::mem::size_of::<PropertyHistogram>())?
                .checked_add(bins.checked_mul(std::mem::size_of::<usize>() + 4)?)?;
            for parent in parents {
                bytes = bytes
                    .checked_add(std::mem::size_of::<BinHistogram>())?
                    .checked_add(
                        bins.checked_mul(
                            stride
                                .unwrap_or(parent.counts.len())
                                .checked_mul(4)?
                                .checked_add(8)?,
                        )?,
                    )?;
            }
        }
        let lease = budget.reserve(bytes)?;
        Some(Self {
            properties: (0..samples.thresholds.len())
                .map(|_| PropertyHistogram::default())
                .collect(),
            _lease: lease,
            stride,
        })
    }

    // Eager population is retained only as a test reference for the fused path
    // and for constructing inherited-cache fixtures below the creation gate.
    #[cfg(test)]
    pub(super) fn create(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        budget: &Arc<HistogramBudget>,
        parallel: bool,
    ) -> Option<Self> {
        let mut cache = Self::reserve(samples, parents, budget)?;
        let work = indices
            .len()
            .saturating_mul(parents.len())
            .saturating_mul(samples.thresholds.len());
        if parallel && indices.len() >= 1024 && work >= 65536 {
            let jobs: Vec<_> = cache.properties.iter_mut().map(Mutex::new).collect();
            let _: Vec<()> = execution::map_with_state(
                jobs.len(),
                || (),
                |_, property| {
                    **jobs[property].lock().expect("MA property cache poisoned") =
                        PropertyHistogram::collect(samples, indices, parents, property);
                },
            );
        } else {
            for (property, histogram) in cache.properties.iter_mut().enumerate() {
                *histogram = PropertyHistogram::collect(samples, indices, parents, property);
            }
        }
        Some(cache)
    }

    #[cfg(test)]
    pub(super) fn subtract(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
    ) -> Result<()> {
        for (property, histogram) in self.properties.iter_mut().enumerate() {
            histogram.subtract(samples, property, removed)?;
        }
        Ok(())
    }

    pub(super) fn normalize_for_handoff(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        options: &TrainOptions,
        redundant: &[u8],
    ) -> Result<()> {
        let plan = PredictorPlan::identity(samples.residuals.len());
        let subtract = |property: usize, histogram: &mut PropertyHistogram| {
            // Redundant entries may contain only updated marginals, or token
            // tables from an ancestor. They must never be subtracted/reused.
            if options.property_dedup && redundant.get(property).is_some_and(|&v| v != 0) {
                histogram.valid = false;
            }
            if histogram.valid {
                with_property_values!(&samples.properties[property], |values| {
                    histogram.subtract_values(samples, removed, values, &plan)
                })
            } else {
                Ok(())
            }
        };
        if options.parallel && removed.len() >= 1024 {
            for result in execution::map_mut(&mut self.properties, subtract) {
                result?;
            }
        } else {
            for (property, histogram) in self.properties.iter_mut().enumerate() {
                subtract(property, histogram)?;
            }
        }
        Ok(())
    }
}
