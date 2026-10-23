// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

//! Lossless transform/predictor costs from libjxl
//! b866b3defec13494bfbaf705db93581cda118e37:
//! - `lib/jxl/enc_modular_simd.cc::EstimateCost`, including ContextMap, wrapped
//!   gradient residuals, HybridUint(4,2,0), and integer/fraction accumulation.
//! - `lib/jxl/enc_modular.cc::EstimateWPCost` (line 282), WPCostWork, and
//!   `PrepareStreamParams` RCT/preset candidate order and strict comparisons.
//! - `lib/jxl/enc_cluster.cc::{Entropy, HistogramEntropy}`,
//!   `base/fast_math-inl.h::FastLog2f`, `base/rational_polynomial-inl.h`, and
//!   Highway `generic_ops-inl.h::SumOfLanes` (translated to native intrinsics).
//!
//! These are the SOURCE cost functions, not a replacement Shannon estimate of
//! raw residuals and not MA's probability-floored entropy. The histogram kernel
//! keeps source float-count equality masking, rational log approximation,
//! per-lane accumulation and horizontal reduction. Native x86_64 dispatch uses
//! SSE2, AVX2, or AVX2+FMA; other targets use the source scalar profile. Source
//! scalar EstimateCost reverses context numbering relative to SIMD; this is
//! retained because fraction accumulation order can change the final integer.
//! Different source ISA profiles can disagree at float ties. No native-oracle
//! bit-identity claim is made before parent-run validation.
//!
//! Parent integration: construct ChannelView values (including metadata/alpha),
//! call select_rct before final transforms, apply only the returned RCT to the
//! first three color channels, then call select_weighted_preset on the resulting
//! channels. The parent owns eligibility (speed, lossless, responsive, color
//! mode, matching channel shifts) and candidate budgets: RCT 4/5/7/9/19; WP 2/5.
//! Functions never mutate caller samples or alter encode/group registration.
//!
//! Counts are limited to i32::MAX pixels per channel, matching Histogram's
//! signed counters/conversion. Gradient scoring wraps its residual to i32, as
//! the source SIMD path does; this is NOT permission to narrow encode residuals.
//! WP subtraction overflow rejects the mode with f32::MAX, as in the source.
//! Extra-bit totals exceeding the source SIMD u32 reduction are explicit errors
//! in EstimateCost; WP uses checked usize totals. Histograms reset per channel.
//!
//! Omitted: source flat-WP shortcut (ordinary State gives the same predictions),
//! parallel/wave scheduling, environment settings, direct-plan replay/audits,
//! and source speed-tier eligibility policy. Pruning is explicit, not implicit.
//! Gradient cost fuses i32 prediction, context spread, signed packing and token
//! classification in native SIMD. Histogram updates and context-map lookups
//! remain scalar, preserving source channel/context accumulation order.

#[path = "cost_simd.rs"]
mod cost_simd;

#[cfg(test)]
#[path = "cost_bench.rs"]
mod cost_bench;

use super::predictor::Neighbors;
use super::{transforms, weighted};
use crate::entropy::HybridUintConfig;
use crate::{Error, Result};

/// Borrowed visible plane; final-row padding is not required. Empty channels
/// are permitted and ignored. Stride is measured in samples, not bytes.
#[derive(Clone, Copy, Debug)]
pub struct ChannelView<'a> {
    pub pixels: &'a [i32],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl<'a> ChannelView<'a> {
    fn validate(self) -> Result<usize> {
        let size = self
            .width
            .checked_mul(self.height)
            .ok_or(Error::Limit("cost channel dimensions"))?;
        if size == 0 {
            return Ok(0);
        }
        if size > i32::MAX as usize {
            return Err(Error::Limit("cost histogram population exceeds i32::MAX"));
        }
        let end = (self.height - 1)
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add(self.width))
            .ok_or(Error::Limit("cost channel addressing"))?;
        if self.stride < self.width || end > self.pixels.len() {
            return Err(Error::InvalidInput(
                "invalid cost channel buffer/stride".into(),
            ));
        }
        Ok(size)
    }

    fn neighbors(self, x: usize, y: usize) -> Neighbors {
        Neighbors::from_plane(self.pixels, self.width, self.stride, x, y)
            .expect("validated cost view and raster coordinates")
    }

    fn row(self, y: usize) -> &'a [i32] {
        &self.pixels[y * self.stride..y * self.stride + self.width]
    }
}

fn validate(channels: &[ChannelView<'_>]) -> Result<usize> {
    channels.iter().try_fold(0usize, |total, &ch| {
        total
            .checked_add(ch.validate()?)
            .ok_or(Error::Limit("cost image population"))
    })
}

fn zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| Error::Limit("cost scratch allocation"))?;
    v.resize(len, T::default());
    Ok(v)
}

const GRADIENT_CUTOFFS: [u32; 16] = [
    1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
];
const WP_CUTOFFS: [i32; 33] = [
    -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -4, -3, -1, 0, 1, 3, 5, 7,
    11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
];

/// The source's 19 distinct RCT candidates, in first-wins tie order.
pub const RCT_CANDIDATES: [u32; 19] = [
    0, 6, 5, 10, 26, 40, 12, 19, 8, 4, 9, 15, 16, 17, 32, 33, 2, 1, 3,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
enum Backend {
    Scalar,
    Sse2,
    Avx2,
    Fma,
}

fn backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return if std::is_x86_feature_detected!("fma") {
                Backend::Fma
            } else {
                Backend::Avx2
            };
        }
        if std::is_x86_feature_detected!("sse2") {
            return Backend::Sse2;
        }
    }
    Backend::Scalar
}

#[derive(Clone)]
struct Histogram {
    counts: [u32; 128],
    total: u32,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            counts: [0; 128],
            total: 0,
        }
    }
}

impl Histogram {
    fn add(&mut self, symbol: u32) {
        // HybridUint(4,2,0) maps all u32 inputs to 0..=127; the validated
        // channel population bounds every signed source count and total.
        self.counts[symbol as usize] += 1;
        self.total += 1;
    }

    fn entropy(&self, mode: Backend) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: modes originate in runtime detection (tests check support).
        // Counts are a full 128-element array and all values fit signed i32.
        unsafe {
            match mode {
                Backend::Sse2 => return x86::sse2::entropy(&self.counts, self.total),
                Backend::Avx2 => return x86::avx2::entropy(&self.counts, self.total),
                Backend::Fma => return x86::fma::entropy(&self.counts, self.total),
                Backend::Scalar => {}
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = mode;
        self.counts
            .iter()
            .fold(0.0, |sum, &c| sum + entropy_term(c, self.total, false))
    }
}

#[allow(clippy::excessive_precision)]
const P: [f32; 3] = [
    -1.8503833400518310E-06,
    1.4287160470083755E+00,
    7.4245873327820566E-01,
];
#[allow(clippy::excessive_precision)]
const Q: [f32; 3] = [
    9.9032814277590719E-01,
    1.0096718572241148E+00,
    1.7409343003366853E-01,
];

fn fast_log2(value: f32, fused: bool) -> f32 {
    let bits = value.to_bits() as i32;
    let exp = bits.wrapping_sub(0x3f2aaaab) >> 23;
    let x = f32::from_bits(bits.wrapping_sub(exp.wrapping_shl(23)) as u32) - 1.0;
    let (p, q) = if fused {
        (
            P[2].mul_add(x, P[1]).mul_add(x, P[0]),
            Q[2].mul_add(x, Q[1]).mul_add(x, Q[0]),
        )
    } else {
        ((P[2] * x + P[1]) * x + P[0], (Q[2] * x + Q[1]) * x + Q[0])
    };
    p / q + exp as f32
}

fn entropy_term(count: u32, total: u32, fused: bool) -> f32 {
    // HistogramEntropy compares FLOAT counts, unlike MA EstimateBits. Preserve
    // that source distinction at counts above the f32 exact-integer boundary.
    if count == 0 || count as f32 == total as f32 {
        return 0.0;
    }
    0.0 - count as f32 * fast_log2((1.0 / total as f32) * count as f32, fused)
}

#[cfg(test)]
fn gradient_context(n: Neighbors, mode: Backend) -> usize {
    let max = n.left.max(n.top).max(n.topleft);
    let min = n.left.min(n.top).min(n.topleft);
    let difference = (i64::from(max) - i64::from(min)) as u32;
    let ascending = GRADIENT_CUTOFFS.partition_point(|&cutoff| cutoff <= difference);
    if mode == Backend::Scalar {
        16 - ascending
    } else {
        ascending
    }
}

fn wp_context(property: i32) -> usize {
    WP_CUTOFFS.len() - WP_CUTOFFS.partition_point(|&cutoff| cutoff < property)
}

fn add_residual(histogram: &mut Histogram, residual: i32, extra: &mut usize) -> Result<()> {
    let packed = ((residual as u32) << 1) ^ ((residual >> 31) as u32);
    let encoded = HybridUintConfig::default().encode(packed);
    histogram.add(encoded.token);
    *extra = extra
        .checked_add(encoded.nbits as usize)
        .ok_or(Error::Limit("cost extra-bit total"))?;
    Ok(())
}

#[derive(Clone)]
struct ChannelCost {
    histograms: [f32; 17],
    extra: usize,
}

fn channel_cost(ch: ChannelView<'_>, mode: Backend) -> Result<ChannelCost> {
    let mut histograms: [Histogram; 17] = std::array::from_fn(|_| Histogram::default());
    let mut extra = 0usize;
    if ch.width != 0 && ch.height != 0 {
        let mut differences = zeroed(ch.width)?;
        let mut tokens = zeroed(ch.width)?;
        for y in 0..ch.height {
            let row_extra = cost_simd::row(
                ch.row(y),
                (y != 0).then(|| ch.row(y - 1)),
                mode,
                &mut differences,
                &mut tokens,
            )?;
            extra = extra
                .checked_add(row_extra)
                .ok_or(Error::Limit("cost extra-bit total"))?;
            for (&difference, &token) in differences.iter().zip(&tokens) {
                histograms[cost_simd::context(difference, mode)].add(token);
            }
        }
    }
    Ok(ChannelCost {
        histograms: histograms.map(|h| h.entropy(mode)),
        extra,
    })
}

fn combine_costs(channels: &[ChannelCost]) -> Result<f32> {
    let (mut integral, mut extra) = (0usize, 0usize);
    let mut fraction = 0.0f32;
    for ch in channels {
        extra = extra
            .checked_add(ch.extra)
            .ok_or(Error::Limit("gradient cost extra bits"))?;
        for &cost in &ch.histograms {
            let integer = cost as usize;
            integral = integral
                .checked_add(integer)
                .ok_or(Error::Limit("gradient histogram cost"))?;
            fraction += cost - integer as f32;
        }
    }
    if extra > u32::MAX as usize {
        return Err(Error::Limit("gradient cost source u32 extra-bit reduction"));
    }
    let total = integral
        .checked_add(extra)
        .and_then(|v| v.checked_add(fraction as usize))
        .ok_or(Error::Limit("gradient cost total"))?;
    Ok(total as f32)
}

/// Source EstimateCost: gradient-context histograms + hybrid extra bits, with
/// the source's final integer truncation. Includes every supplied channel.
pub fn estimate_cost(channels: &[ChannelView<'_>]) -> Result<f32> {
    validate(channels)?;
    let mode = backend();
    let costs = channels
        .iter()
        .map(|&ch| channel_cost(ch, mode))
        .collect::<Result<Vec<_>>>()?;
    combine_costs(&costs)
}

#[derive(Clone, Debug)]
pub struct RctSelection {
    pub rct_type: u32,
    pub cost: f32,
    /// Source order; includes the unchanged-image baseline (RCT 0).
    pub candidates: Vec<(u32, f32)>,
}

/// Score a prefix of RCT_CANDIDATES (1..=19), retaining the first strict minimum.
/// The three color views must have equal dimensions; shifts/eligibility belong
/// to the caller. All other channel costs are cached without changing the
/// source's per-histogram global fraction accumulation order. Samples are never
/// mutated. Zero candidates is invalid: skipping search is the parent's policy.
pub fn select_rct(
    channels: &[ChannelView<'_>],
    first_color: usize,
    candidate_count: usize,
) -> Result<RctSelection> {
    validate(channels)?;
    let end = first_color
        .checked_add(3)
        .ok_or(Error::Limit("RCT color channel range"))?;
    if candidate_count == 0 || candidate_count > RCT_CANDIDATES.len() || end > channels.len() {
        return Err(Error::InvalidInput(
            "invalid RCT candidate count or color channel range".into(),
        ));
    }
    let shape = (channels[first_color].width, channels[first_color].height);
    if channels[first_color..end]
        .iter()
        .any(|ch| (ch.width, ch.height) != shape)
    {
        return Err(Error::InvalidInput(
            "RCT color channels have different dimensions".into(),
        ));
    }
    let mode = backend();
    let mut costs = channels
        .iter()
        .map(|&ch| channel_cost(ch, mode))
        .collect::<Result<Vec<_>>>()?;
    let baseline = combine_costs(&costs)?;
    let mut result = RctSelection {
        rct_type: 0,
        cost: baseline,
        candidates: vec![(0, baseline)],
    };
    if candidate_count == 1 {
        return Ok(result);
    }
    let mut original = Vec::with_capacity(3);
    for &ch in &channels[first_color..end] {
        let mut plane = zeroed(ch.width * ch.height)?;
        if ch.width != 0 {
            for y in 0..ch.height {
                plane[y * ch.width..(y + 1) * ch.width].copy_from_slice(ch.row(y));
            }
        }
        original.push(plane);
    }
    let mut trial: Vec<Vec<i32>> = (0..3)
        .map(|_| zeroed(shape.0 * shape.1))
        .collect::<Result<_>>()?;
    for &rct_type in &RCT_CANDIDATES[1..candidate_count] {
        for c in 0..3 {
            trial[c].copy_from_slice(&original[c]);
        }
        transforms::rct_forward(&mut trial, 0, rct_type)?;
        for c in 0..3 {
            costs[first_color + c] = channel_cost(
                ChannelView {
                    pixels: &trial[c],
                    width: shape.0,
                    height: shape.1,
                    stride: shape.0,
                },
                mode,
            )?;
        }
        let cost = combine_costs(&costs)?;
        result.candidates.push((rct_type, cost));
        if cost < result.cost {
            result.rct_type = rct_type;
            result.cost = cost;
        }
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WpCostWork {
    pub pixels: usize,
    pub pruned_modes: usize,
    pub checkpoints: usize,
}

impl WpCostWork {
    fn add(&mut self, other: Self) -> Result<()> {
        self.pixels = self
            .pixels
            .checked_add(other.pixels)
            .ok_or(Error::Limit("WP work population"))?;
        self.pruned_modes = self
            .pruned_modes
            .checked_add(other.pruned_modes)
            .ok_or(Error::Limit("WP work pruned modes"))?;
        self.checkpoints = self
            .checkpoints
            .checked_add(other.checkpoints)
            .ok_or(Error::Limit("WP work checkpoints"))?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WpCost {
    pub cost: f32,
    pub work: WpCostWork,
    /// Source SubOverflow rejected this candidate. A pruned result instead has
    /// work.pruned_modes == 1; either reason returns the source f32::MAX sentinel.
    pub unhealthy: bool,
}

/// Source EstimateWPCost, with signed WP-error contexts and a fresh predictor
/// state per channel. Finite limits below f32::MAX enable pruning only when the
/// WHOLE image has at least 65536 samples; margin uses all channels even when
/// selected_channel limits work. Checkpoints occur after rows 63,127,... within
/// each selected channel. NaN/infinite/max limits disable pruning as in source.
pub fn estimate_wp_cost(
    channels: &[ChannelView<'_>],
    preset: u8,
    cost_limit: f32,
    selected_channel: Option<usize>,
) -> Result<WpCost> {
    let total_pixels = validate(channels)?;
    if selected_channel.is_some_and(|c| c >= channels.len()) {
        return Err(Error::InvalidInput(
            "WP selected channel is out of bounds".into(),
        ));
    }
    let header = weighted::Header::preset(u32::from(preset))?;
    let mode = backend();
    let bounded = cost_limit.is_finite() && cost_limit < f32::MAX && total_pixels >= 65536;
    let margin = 16.0 + total_pixels as f64 * 0.001;
    let (mut extra_bits, mut histo_cost) = (0usize, 0.0f32);
    let mut work = WpCostWork::default();
    for (c, &ch) in channels.iter().enumerate() {
        if selected_channel.is_some_and(|selected| c != selected) || ch.width == 0 || ch.height == 0
        {
            continue;
        }
        let mut histograms: [Histogram; 34] = std::array::from_fn(|_| Histogram::default());
        let mut state = weighted::State::new(header, ch.width)?;
        let mut unhealthy = false;
        for y in 0..ch.height {
            for x in 0..ch.width {
                let n = ch.neighbors(x, y);
                let (guess, property) =
                    state.predict(x, y, n.top, n.left, n.topright, n.topleft, n.toptop)?;
                let value = ch.row(y)[x];
                let (residual, overflow) = value.overflowing_sub(guess);
                unhealthy |= overflow;
                add_residual(
                    &mut histograms[wp_context(property)],
                    residual,
                    &mut extra_bits,
                )?;
                state.update(value, x, y)?;
            }
            work.pixels += ch.width;
            if bounded && y & 63 == 63 {
                let mut lower_bound = f64::from(histo_cost) + extra_bits as f64;
                for h in &histograms {
                    lower_bound += f64::from(h.entropy(mode));
                }
                work.checkpoints += 1;
                if lower_bound > f64::from(cost_limit) + margin {
                    work.pruned_modes += 1;
                    return Ok(WpCost {
                        cost: f32::MAX,
                        work,
                        unhealthy,
                    });
                }
            }
        }
        if unhealthy {
            return Ok(WpCost {
                cost: f32::MAX,
                work,
                unhealthy: true,
            });
        }
        for h in &histograms {
            histo_cost += h.entropy(mode);
        }
    }
    Ok(WpCost {
        cost: histo_cost + extra_bits as f32,
        work,
        unhealthy: false,
    })
}

#[derive(Clone, Debug)]
pub struct WeightedPresetSelection {
    pub preset: u8,
    pub cost: f32,
    pub candidates: Vec<WpCost>,
    pub work: WpCostWork,
}

/// Source sequential preset search over modes 0..mode_count (1..=5), first
/// strict minimum wins. Set bounded only for the source's bound-enabled policy.
/// With no accepted candidate, returns preset 0 and f32::MAX as in the source;
/// the caller can inspect each candidate's unhealthy/pruned state.
pub fn select_weighted_preset(
    channels: &[ChannelView<'_>],
    mode_count: usize,
    bounded: bool,
) -> Result<WeightedPresetSelection> {
    validate(channels)?;
    if !(1..=5).contains(&mode_count) {
        return Err(Error::InvalidInput(
            "WP candidate count must be 1..=5".into(),
        ));
    }
    let mut result = WeightedPresetSelection {
        preset: 0,
        cost: f32::MAX,
        candidates: Vec::with_capacity(mode_count),
        work: WpCostWork::default(),
    };
    for preset in 0..mode_count {
        let limit = if bounded { result.cost } else { f32::INFINITY };
        let candidate = estimate_wp_cost(channels, preset as u8, limit, None)?;
        result.work.add(candidate.work)?;
        if candidate.cost < result.cost {
            result.preset = preset as u8;
            result.cost = candidate.cost;
        }
        result.candidates.push(candidate);
    }
    Ok(result)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{P, Q};

    // Same Histogram::kRounding-capped lanes, Horner evaluation and SumOfLanes
    // tree as the source. FMA is selected separately from AVX2 support.
    macro_rules! entropy_kernel {
        ($feature:literal, $lanes:literal) => {
            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn entropy(counts: &[u32; 128], total: u32) -> f32 {
                let inverse = splat(1.0 / total as f32);
                let total = splat(total as f32);
                let mut sum = zero();
                // SAFETY: runtime features checked by caller; counts has 128
                // readable u32s, exactly divisible by both native lane widths.
                // Counts fit i32, the output stack array holds exactly LANES.
                unsafe {
                    for i in (0..128).step_by($lanes) {
                        let cf = convert(load(counts.as_ptr().add(i).cast()));
                        let bits = cast_i(mul(cf, inverse));
                        let exp = sar::<23>(sub_i(bits, splat_i(0x3f2aaaab)));
                        let x = sub(cast_f(sub_i(bits, shl::<23>(exp))), splat(1.0));
                        let p = madd(madd(splat(P[2]), x, splat(P[1])), x, splat(P[0]));
                        let q = madd(madd(splat(Q[2]), x, splat(Q[1])), x, splat(Q[0]));
                        let logs = add(div(p, q), convert(exp));
                        let term = andnot(eq(cf, total), sub(zero(), mul(cf, logs)));
                        sum = add(sum, term);
                    }
                    let mut lanes = [0.0f32; $lanes];
                    store(lanes.as_mut_ptr(), sum);
                    reduce(&lanes)
                }
            }
        };
    }

    fn reduce(v: &[f32]) -> f32 {
        if v.len() == 8 {
            let a = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
            (a[0] + a[3]) + (a[1] + a[2])
        } else {
            (v[0] + v[3]) + (v[1] + v[2])
        }
    }

    pub(super) mod sse2 {
        use super::{P, Q, reduce};
        use std::arch::x86_64::{
            __m128, _mm_add_ps as add, _mm_andnot_ps as andnot, _mm_castps_si128 as cast_i,
            _mm_castsi128_ps as cast_f, _mm_cmpeq_ps as eq, _mm_cvtepi32_ps as convert,
            _mm_div_ps as div, _mm_loadu_si128 as load, _mm_mul_ps as mul,
            _mm_set1_epi32 as splat_i, _mm_set1_ps as splat, _mm_setzero_ps as zero,
            _mm_slli_epi32 as shl, _mm_srai_epi32 as sar, _mm_storeu_ps as store,
            _mm_sub_epi32 as sub_i, _mm_sub_ps as sub,
        };
        #[target_feature(enable = "sse2")]
        unsafe fn madd(a: __m128, b: __m128, c: __m128) -> __m128 {
            add(mul(a, b), c)
        }
        entropy_kernel!("sse2", 4);
    }

    pub(super) mod avx2 {
        use super::{P, Q, reduce};
        use std::arch::x86_64::{__m256, _CMP_EQ_OQ, _mm256_cmp_ps};
        pub(super) use std::arch::x86_64::{
            _mm256_add_ps as add, _mm256_andnot_ps as andnot, _mm256_castps_si256 as cast_i,
            _mm256_castsi256_ps as cast_f, _mm256_cvtepi32_ps as convert, _mm256_div_ps as div,
            _mm256_loadu_si256 as load, _mm256_mul_ps as mul, _mm256_set1_epi32 as splat_i,
            _mm256_set1_ps as splat, _mm256_setzero_ps as zero, _mm256_slli_epi32 as shl,
            _mm256_srai_epi32 as sar, _mm256_storeu_ps as store, _mm256_sub_epi32 as sub_i,
            _mm256_sub_ps as sub,
        };
        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn eq(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b)
        }
        #[target_feature(enable = "avx2")]
        unsafe fn madd(a: __m256, b: __m256, c: __m256) -> __m256 {
            add(mul(a, b), c)
        }
        entropy_kernel!("avx2", 8);
    }

    pub(super) mod fma {
        use super::avx2::{
            add, andnot, cast_f, cast_i, convert, div, eq, load, mul, sar, shl, splat, splat_i,
            store, sub, sub_i, zero,
        };
        use super::{P, Q, reduce};
        use std::arch::x86_64::_mm256_fmadd_ps as madd;
        entropy_kernel!("avx2,fma", 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(pixels: &[i32], width: usize, height: usize) -> ChannelView<'_> {
        ChannelView {
            pixels,
            width,
            height,
            stride: width,
        }
    }

    pub(super) fn supported() -> Vec<Backend> {
        #[allow(unused_mut)]
        let mut modes = vec![Backend::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("sse2") {
                modes.push(Backend::Sse2);
            }
            if std::is_x86_feature_detected!("avx2") {
                modes.push(Backend::Avx2);
                if std::is_x86_feature_detected!("fma") {
                    modes.push(Backend::Fma);
                }
            }
        }
        modes
    }

    // Scalar lane simulation of the source HistogramEntropy target profile,
    // independent of native loads, masks, polynomial and horizontal reduction.
    fn entropy_reference(counts: &[u32; 128], mode: Backend) -> f32 {
        let total: u32 = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let lanes = match mode {
            Backend::Scalar => 1,
            Backend::Sse2 => 4,
            _ => 8,
        };
        let mut sums = [0.0f32; 8];
        for (i, &count) in counts.iter().enumerate() {
            sums[i % lanes] += entropy_term(count, total, mode == Backend::Fma);
        }
        match lanes {
            1 => sums[0],
            4 => (sums[0] + sums[3]) + (sums[1] + sums[2]),
            _ => {
                ((sums[0] + sums[4]) + (sums[3] + sums[7]))
                    + ((sums[1] + sums[5]) + (sums[2] + sums[6]))
            }
        }
    }

    fn token_reference(residual: i32) -> (usize, usize) {
        let packed = if residual >= 0 {
            (residual as u32) * 2
        } else {
            ((-(i64::from(residual)) * 2) - 1) as u32
        };
        if packed < 16 {
            return (packed as usize, 0);
        }
        let bits = 31 - packed.leading_zeros() - 2;
        (
            (8 + 4 * bits + ((packed >> bits) & 3)) as usize,
            bits as usize,
        )
    }

    fn cost_reference(channels: &[ChannelView<'_>], mode: Backend) -> f32 {
        let (mut integral, mut fraction, mut extra) = (0usize, 0.0f32, 0usize);
        for &ch in channels {
            let mut histograms = [[0u32; 128]; 17];
            for y in 0..ch.height {
                for x in 0..ch.width {
                    let left = if x > 0 {
                        ch.row(y)[x - 1]
                    } else if y > 0 {
                        ch.row(y - 1)[0]
                    } else {
                        0
                    };
                    let top = if y > 0 { ch.row(y - 1)[x] } else { left };
                    let tl = if x > 0 && y > 0 {
                        ch.row(y - 1)[x - 1]
                    } else {
                        left
                    };
                    let spread =
                        i64::from(left.max(top).max(tl)) - i64::from(left.min(top).min(tl));
                    let scalar_context = [
                        0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
                    ]
                    .iter()
                    .filter(|&&cutoff| spread < cutoff)
                    .count();
                    let context = if mode == Backend::Scalar {
                        scalar_context
                    } else {
                        16 - scalar_context
                    };
                    let prediction = (i64::from(left) + i64::from(top) - i64::from(tl))
                        .clamp(i64::from(left.min(top)), i64::from(left.max(top)))
                        as i32;
                    let (token, bits) = token_reference(ch.row(y)[x].wrapping_sub(prediction));
                    histograms[context][token] += 1;
                    extra += bits;
                }
            }
            for h in &histograms {
                let cost = entropy_reference(h, mode);
                integral += cost as usize;
                fraction += cost - (cost as usize) as f32;
            }
        }
        (integral + extra + fraction as usize) as f32
    }

    #[test]
    fn histogram_kernel_matches_each_source_lane_profile() {
        let mut state = 0xb866b3de_u32;
        for iteration in 0..256 {
            let mut h = Histogram::default();
            for c in &mut h.counts {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *c = match iteration % 4 {
                    0 => state & 255,
                    1 => state & 65535,
                    2 => state & 0x7fffff,
                    _ => u32::from(state & 31 == 0),
                };
                h.total += *c;
            }
            for mode in supported() {
                assert_eq!(
                    h.entropy(mode).to_bits(),
                    entropy_reference(&h.counts, mode).to_bits(),
                    "{mode:?}"
                );
            }
        }
        let mut h = Histogram::default();
        h.counts[0] = 1 << 24;
        h.counts[7] = 1;
        h.total = (1 << 24) + 1;
        // HistogramEntropy's FLOAT equality drops the dominant bin here;
        // MA's INTEGER equality must not be reused for this source estimator.
        assert_eq!(entropy_term(1 << 24, h.total, false), 0.0);
        for mode in supported() {
            assert_eq!(
                h.entropy(mode).to_bits(),
                entropy_reference(&h.counts, mode).to_bits()
            );
        }
        for exponent in -31..=31 {
            let x = 2.0_f32.powi(exponent) * 1.125;
            for fused in [false, true] {
                assert!((f64::from(fast_log2(x, fused)) - f64::from(x).log2()).abs() < 0.000006);
            }
        }
    }

    #[test]
    fn source_cost_contexts_padding_tails_and_wrapping_samples() {
        let mut state = 0x174a78_u32;
        for width in 1..=35 {
            for height in [1, 2, 4] {
                let stride = width + 3;
                let mut storage = vec![i32::MIN; stride * (height - 1) + width];
                for y in 0..height {
                    for x in 0..width {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        storage[y * stride + x] = if width % 2 == 0 {
                            state as i32
                        } else {
                            (state & 65535) as i32 - 32768
                        };
                    }
                }
                let channels = [ChannelView {
                    pixels: &storage,
                    width,
                    height,
                    stride,
                }];
                for mode in supported() {
                    let actual =
                        combine_costs(&[channel_cost(channels[0], mode).unwrap()]).unwrap();
                    assert_eq!(actual.to_bits(), cost_reference(&channels, mode).to_bits());
                }
                assert_eq!(
                    estimate_cost(&channels).unwrap(),
                    cost_reference(&channels, backend())
                );
            }
        }
        for (pixel, cost) in [(0, 0.0), (8, 2.0), (i32::MIN, 29.0), (i32::MAX, 29.0)] {
            assert_eq!(estimate_cost(&[view(&[pixel], 1, 1)]).unwrap(), cost);
        }
        for difference in 0..=510 {
            let n = Neighbors {
                left: difference,
                ..Neighbors::default()
            };
            let scalar = GRADIENT_CUTOFFS
                .iter()
                .filter(|&&c| difference < c as i32)
                .count();
            assert_eq!(gradient_context(n, Backend::Scalar), scalar);
            assert_eq!(gradient_context(n, Backend::Sse2), 16 - scalar);
        }
        for property in -510..=510 {
            assert_eq!(
                wp_context(property),
                WP_CUTOFFS.iter().filter(|&&c| c >= property).count()
            );
        }
    }

    #[test]
    fn fused_cost_all_rcts_preserves_channel_and_context_accumulation_order() {
        let extremes = [i32::MIN, i32::MAX, -1, 0, 1, -65535, 65535];
        let mut random = 0xb866_b3d0u32;
        let originals: Vec<Vec<i32>> = (0..3)
            .map(|c| {
                (0..17 * 5)
                    .map(|i| {
                        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                        if i % 2 == 0 {
                            extremes[(i + c) % extremes.len()]
                        } else {
                            random as i32
                        }
                    })
                    .collect()
            })
            .collect();
        let metadata = [i32::MIN, 3, i32::MAX, -7, 0, 9];
        let extra: Vec<_> = (0..9 * 3).map(|i| extremes[i % extremes.len()]).collect();
        for kind in 0..42 {
            let mut trial = originals.clone();
            transforms::rct_forward(&mut trial, 0, kind).unwrap();
            let channels = [
                view(&metadata, 3, 2),
                view(&trial[0], 17, 5),
                view(&trial[1], 17, 5),
                view(&trial[2], 17, 5),
                view(&extra, 9, 3),
            ];
            for mode in supported() {
                let costs: Vec<_> = channels
                    .iter()
                    .map(|&c| channel_cost(c, mode).unwrap())
                    .collect();
                assert_eq!(
                    combine_costs(&costs).unwrap().to_bits(),
                    cost_reference(&channels, mode).to_bits(),
                    "RCT {kind}, {mode:?}",
                );
            }
        }
    }

    #[test]
    fn rct_candidates_use_original_planes_preserve_other_channels_and_ties() {
        let originals: Vec<Vec<i32>> = (0..3)
            .map(|c| {
                (0..45)
                    .map(|x| (x * 187 + c * 53 + x * x * (c + 1)) % 65536)
                    .collect()
            })
            .collect();
        let metadata = [13, -9, 11, -2, 8, 5];
        let alpha: Vec<_> = (0..45).map(|x| x * 7).collect();
        let channels = [
            view(&metadata, 2, 3),
            view(&originals[0], 9, 5),
            view(&originals[1], 9, 5),
            view(&originals[2], 9, 5),
            view(&alpha, 9, 5),
        ];
        for count in [1, 4, 5, 7, 9, 19] {
            let result = select_rct(&channels, 1, count).unwrap();
            let mut expected = (0, f32::MAX);
            for (i, &rct) in RCT_CANDIDATES[..count].iter().enumerate() {
                let mut transformed = originals.clone();
                transforms::rct_forward(&mut transformed, 0, rct).unwrap();
                let trial = [
                    channels[0],
                    view(&transformed[0], 9, 5),
                    view(&transformed[1], 9, 5),
                    view(&transformed[2], 9, 5),
                    channels[4],
                ];
                let cost = cost_reference(&trial, backend());
                assert_eq!(estimate_cost(&trial).unwrap(), cost);
                assert_eq!(result.candidates[i], (rct, cost));
                if cost < expected.1 {
                    expected = (rct, cost);
                }
            }
            assert_eq!((result.rct_type, result.cost), expected);
        }
        let zero = [0; 45];
        let ties = [view(&zero, 9, 5); 3];
        let rct = select_rct(&ties, 0, 19).unwrap();
        assert_eq!((rct.rct_type, rct.cost), (0, 0.0));
        let wp = select_weighted_preset(&ties, 5, true).unwrap();
        assert_eq!((wp.preset, wp.cost), (0, 0.0));
        assert_eq!(wp.work.pixels, 45 * 3 * 5);
    }

    fn wp_reference(channels: &[ChannelView<'_>], preset: u8) -> f32 {
        let mut extra = 0usize;
        let mut histo_cost = 0.0f32;
        for &ch in channels {
            if ch.width == 0 || ch.height == 0 {
                continue;
            }
            let mut histograms = [[0u32; 128]; 34];
            let mut state = weighted::State::new(
                weighted::Header::preset(u32::from(preset)).unwrap(),
                ch.width,
            )
            .unwrap();
            let mut unhealthy = false;
            for y in 0..ch.height {
                for x in 0..ch.width {
                    let n = ch.neighbors(x, y);
                    let (guess, prop) = state
                        .predict(x, y, n.top, n.left, n.topright, n.topleft, n.toptop)
                        .unwrap();
                    let (residual, overflow) = ch.row(y)[x].overflowing_sub(guess);
                    unhealthy |= overflow;
                    let ctx = WP_CUTOFFS.iter().filter(|&&cutoff| cutoff >= prop).count();
                    let (token, bits) = token_reference(residual);
                    histograms[ctx][token] += 1;
                    extra += bits;
                    state.update(ch.row(y)[x], x, y).unwrap();
                }
            }
            if unhealthy {
                return f32::MAX;
            }
            for h in &histograms {
                histo_cost += entropy_reference(h, backend());
            }
        }
        histo_cost + extra as f32
    }

    #[test]
    fn weighted_cost_presets_unhealthy_and_independent_channels() {
        let pixels: Vec<_> = (0..255).map(|i| ((i * i * 397) & 65535) - 32768).collect();
        let channels = [view(&pixels, 17, 15), view(&pixels[..63], 9, 7)];
        let mut best = (0, f32::MAX);
        for preset in 0..5 {
            let cost = estimate_wp_cost(&channels, preset, f32::INFINITY, None).unwrap();
            assert_eq!(
                cost.cost.to_bits(),
                wp_reference(&channels, preset).to_bits()
            );
            assert_eq!(cost.work.pixels, 255 + 63);
            assert_eq!(cost.work.checkpoints, 0);
            assert!(!cost.unhealthy);
            if cost.cost < best.1 {
                best = (preset, cost.cost);
            }
        }
        let selected = select_weighted_preset(&channels, 5, true).unwrap();
        assert_eq!((selected.preset, selected.cost), best);
        for preset in 0..5 {
            let rejected =
                estimate_wp_cost(&[view(&[1, i32::MIN], 2, 1)], preset, f32::INFINITY, None)
                    .unwrap();
            assert_eq!(rejected.cost, f32::MAX);
            assert!(rejected.unhealthy);
            assert_eq!(rejected.work.pixels, 2);
        }
    }

    #[test]
    fn pruning_population_gate_checkpoints_margin_and_selected_channel() {
        let pixels: Vec<_> = (0..131072_u32)
            .map(|i| (i.wrapping_mul(1664525).wrapping_add(1013904223) & 65535) as i32)
            .collect();
        let channels = [view(&pixels, 1024, 128)];
        let pruned = estimate_wp_cost(&channels, 0, 0.0, None).unwrap();
        assert_eq!(pruned.cost, f32::MAX);
        assert_eq!(
            pruned.work,
            WpCostWork {
                pixels: 65536,
                pruned_modes: 1,
                checkpoints: 1
            }
        );
        let one_checkpoint = [view(&pixels[..65536], 1024, 64)];
        let full = estimate_wp_cost(&one_checkpoint, 0, f32::INFINITY, None).unwrap();
        let near = estimate_wp_cost(&one_checkpoint, 0, full.cost - 8.0, None).unwrap();
        assert_eq!(
            near.cost, full.cost,
            "a near tie must not prune within the 81.536-bit margin"
        );
        assert_eq!(near.work.checkpoints, 1);
        assert_eq!(near.work.pruned_modes, 0);
        let below = [view(&pixels[..65472], 1023, 64)];
        let small = estimate_wp_cost(&below, 0, 0.0, None).unwrap();
        assert_eq!(small.work.checkpoints, 0);
        assert!(small.cost < f32::MAX);
        for limit in [f32::NAN, f32::INFINITY, f32::MAX] {
            assert_eq!(
                estimate_wp_cost(&one_checkpoint, 0, limit, None)
                    .unwrap()
                    .work
                    .checkpoints,
                0
            );
        }
        let split = [view(&pixels[..32768], 512, 64); 2];
        let selected = estimate_wp_cost(&split, 0, 0.0, Some(1)).unwrap();
        assert_eq!(
            selected.work,
            WpCostWork {
                pixels: 32768,
                pruned_modes: 1,
                checkpoints: 1
            }
        );
        let bounded = select_weighted_preset(&channels, 2, true).unwrap();
        let unbounded = select_weighted_preset(&channels, 2, false).unwrap();
        assert_eq!(
            (bounded.preset, bounded.cost),
            (unbounded.preset, unbounded.cost)
        );
    }

    #[test]
    fn invalid_views_options_and_source_limits_are_explicit() {
        assert_eq!(estimate_cost(&[]).unwrap(), 0.0);
        assert_eq!(estimate_cost(&[view(&[], 0, usize::MAX)]).unwrap(), 0.0);
        for ch in [
            view(&[0; 3], 2, 2),
            ChannelView {
                pixels: &[0; 4],
                width: 2,
                height: 2,
                stride: 1,
            },
            view(&[], usize::MAX, 2),
            view(&[], i32::MAX as usize + 1, 1),
        ] {
            assert!(estimate_cost(&[ch]).is_err());
            assert!(estimate_wp_cost(&[ch], 0, f32::INFINITY, None).is_err());
        }
        let channels = [view(&[0; 4], 2, 2); 3];
        for count in [0, 20, usize::MAX] {
            assert!(select_rct(&channels, 0, count).is_err());
        }
        for count in [0, 6, usize::MAX] {
            assert!(select_weighted_preset(&channels, count, false).is_err());
        }
        assert!(select_rct(&channels, usize::MAX, 19).is_err());
        assert!(select_rct(&[channels[0], channels[1], view(&[0; 4], 1, 4)], 0, 19).is_err());
        assert!(estimate_wp_cost(&channels, 5, 0.0, None).is_err());
        assert!(estimate_wp_cost(&channels, 0, 0.0, Some(3)).is_err());
        let too_many_bits = ChannelCost {
            histograms: [0.0; 17],
            extra: u32::MAX as usize,
        };
        let one_bit = ChannelCost {
            histograms: [0.0; 17],
            extra: 1,
        };
        assert!(combine_costs(&[too_many_bits, one_bit]).is_err());
    }
}
