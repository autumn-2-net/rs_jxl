// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

//! Native x86_64 SSE2/AVX2/AVX2+FMA probability-floored MA scoring paths from
//! libjxl b866b3defec13494bfbaf705db93581cda118e37:
//! - `lib/jxl/modular/encoding/enc_ma.cc::{EstimateBits, EntropyContribution,
//!   SmallHistogramEntropy, MoveCountsAndEstimateBitsImpl, MoveCountsAndLogBits}`.
//! - `lib/jxl/modular/encoding/enc_ma_entropy.h::{MaCountLogTable,
//!   MaClippedEntropy}` and `enc_ma.cc::{CountEntropyTable, CountEntropy,
//!   HistogramCountEntropy, MoveCountsAndShannonBits}` for the f64 cache.
//! - `lib/jxl/base/fast_math-inl.h::FastLog2f` and
//!   `lib/jxl/base/rational_polynomial-inl.h::EvalRationalPolynomial`.
//! - `lib/jxl/ans_params.h::ANS_LOG_TAB_SIZE` (12).
//!
//! `DefaultScorer` is the internal source-default f32 trainer path: caller-
//! validated weighted totals, immutable cached deltas, fused movement/scoring,
//! and a shared triangular contribution table for each side with total <=256.
//! Its parent estimate uses FastLog2f directly, never a log(count) cache.
//! Native sweeps with combined mass <=256 specialize to that table throughout,
//! hoisting block/tail layout without changing any split or accumulation order.
//! `RS_JXL_MA_AVX512=1` opts into 16-bin dense computation with the same eight
//! logical accumulation lanes. The flag and CPU/OS support are checked once
//! when constructing the first DefaultScorer. Small/sparse sweeps keep AVX2.
//! `RS_JXL_MA_RCP=1` instead opts into an AVX2/FMA reciprocal experiment. Its
//! rational log uses one Newton correction instead of vector division. Integer
//! counts, clipping, and lookup tables stay exact; tree scoring may differ by
//! rounding. It takes precedence over the AVX512 experiment and defaults OFF.
//!
//! Construct one `LogTable` per trainer/worker and reuse it for dense predictor
//! histograms and split sweeps. `move_counts_and_log_bits` moves all weighted
//! increments, clears them, and scores both resulting histograms in one SIMD
//! pass after validation. Each side independently falls back to FastLog2f when
//! its total exceeds the allocated log cache. `LogTable::new(0)` selects that
//! source estimator for every nonempty population, without a large cache.
//! Do not mutate a separate `ClippedEntropy`'s counts behind its cached sums.
//!
//! Counts/transfers are exact. Both profiles approximate the CLIPPED model
//! sum(c * min(log2(N/c), 12)), excluding token extra bits, tree signaling, and
//! predictor penalties. The cache uses log2(N)-log2(c); fallback uses the source
//! -FastLog2f(max(c * (1/N), 1/4096)) with an INTEGER c == N zero mask.
//! FastLog2f uses the source coefficients/range reduction and real division,
//! with fused Horner operations on the default AVX2+FMA backend and non-fused
//! Horner operations on its fallbacks and the optional LogTable path. Cache and
//! fallback are not bit-identical, nor are different reduction orders/FMA
//! targets/math libraries. A cost tie may select a different lossless tree.
//!
//! LogTable is NOT `TreeSamples::FastEntropy` / `MoveCountsAndShannonBits`: that
//! source profile caches c*log2(c) and does NOT apply the probability floor.
//! Never substitute an uncorrected Shannon cache for these clipped costs.
//! The log-cache allocation remains bounded by 65536; histogram/post-move
//! totals may reach i32::MAX, the source signed-count/conversion domain. Larger
//! totals are errors, not wrapped sums or signed float conversions. No readable
//! padding is needed, including for all-zero histograms and scalar tails.
//! The separate internal CountEntropyTable stores f64 c*log2(c), with an exact
//! log2 fallback for the full u32 count domain. It actively initializes
//! ma_entropy::ClippedEntropy's sums and updates both sums during a split move.
//! AVX2 gathers four cached terms; SSE2 uses scalar loads and paired arithmetic.
//! The caller retains its rare-count correction: the shared cache does NOT
//! switch the active trainer to unclipped Shannon or approximate FastLog2f.
//! Omitted: configurable small-table limits, sparse-block
//! scoring, Shannon-profile selection, scheduling, and non-x86 SIMD.
//! Non-x86 targets use the safe scalar implementation.

use std::fmt;
use std::sync::OnceLock;

pub const MAX_LOG_TABLE_TOTAL: u32 = 65536;
pub const MAX_HISTOGRAM_TOTAL: u32 = i32::MAX as u32;
const PROBABILITY_BITS: f32 = 12.0;
const MIN_PROBABILITY: f32 = 1.0 / 4096.0;

// Keep the frozen source literals rather than substitute another log fit.
#[allow(clippy::excessive_precision)]
const LOG_P: [f32; 3] = [
    -1.8503833400518310E-06,
    1.4287160470083755E+00,
    7.4245873327820566E-01,
];
#[allow(clippy::excessive_precision)]
const LOG_Q: [f32; 3] = [
    9.9032814277590719E-01,
    1.0096718572241148E+00,
    1.7409343003366853E-01,
];

/// Source FastLog2f restricted to the normal, positive MA probability domain.
/// Only the four Horner multiply-adds are fused in the FMA profile. Both profiles
/// use actual division (`FastDivision<float>`'s JXL_TRUE branch), not reciprocals.
#[inline]
fn fast_log2<const FMA: bool>(value: f32) -> f32 {
    let bits = value.to_bits() as i32;
    let exponent = bits.wrapping_sub(0x3f2aaaab) >> 23;
    let mantissa = f32::from_bits(bits.wrapping_sub(exponent.wrapping_shl(23)) as u32);
    let x = mantissa - 1.0;
    let (numerator, denominator) = if FMA {
        (
            LOG_P[2].mul_add(x, LOG_P[1]).mul_add(x, LOG_P[0]),
            LOG_Q[2].mul_add(x, LOG_Q[1]).mul_add(x, LOG_Q[0]),
        )
    } else {
        (
            (LOG_P[2] * x + LOG_P[1]) * x + LOG_P[0],
            (LOG_Q[2] * x + LOG_Q[1]) * x + LOG_Q[0],
        )
    };
    numerator / denominator + exponent as f32
}

#[inline]
fn inverse_total(total: u32) -> f32 {
    // Source masks the zero-total lanes after evaluating them. Avoid its
    // intermediate 0 * infinity / NaN without changing the all-zero cost.
    if total == 0 { 0.0 } else { 1.0 / total as f32 }
}

#[inline]
fn fast_contribution(count: u32, total: u32, inverse: f32) -> f32 {
    if count == total {
        return 0.0;
    }
    let probability = (count as f32 * inverse).max(MIN_PROBABILITY);
    -(count as f32 * fast_log2::<false>(probability))
}

const SMALL_HISTOGRAM_TOTAL: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DefaultBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    #[cfg(target_arch = "x86_64")]
    Avx2Rcp,
    #[cfg(target_arch = "x86_64")]
    Avx512Fma8,
}

impl DefaultBackend {
    fn detect(avx512: bool) -> Self {
        #[cfg(not(target_arch = "x86_64"))]
        let _ = avx512;
        let backend = Self::Scalar;
        #[cfg(target_arch = "x86_64")]
        let backend = if std::is_x86_feature_detected!("avx2") {
            if std::is_x86_feature_detected!("fma") {
                if avx512
                    && std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512dq")
                {
                    Self::Avx512Fma8
                } else {
                    Self::Avx2Fma
                }
            } else {
                Self::Avx2
            }
        } else if std::is_x86_feature_detected!("sse2") {
            Self::Sse2
        } else {
            backend
        };
        backend
    }

    fn uses_fma(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            matches!(self, Self::Avx2Fma | Self::Avx2Rcp | Self::Avx512Fma8)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

/// Source-default f32 MA scoring for the trainer's validated histograms.
///
/// The caller must ensure counts sum to the supplied total, all totals are at
/// most i32::MAX, and each delta fits its source bin. Movement totals describe
/// the POST-move histograms; destination addition must not overflow. Weighted
/// populations, not unique-record counts, are required. These numerical
/// invariants are not rescanned here. Invalid inputs have no scoring guarantee.
/// Lengths are checked before mutation; even invalid counts cannot cause an
/// out-of-bounds native gather. No histogram allocation, copy, or delta clearing
/// occurs during scoring. Each used numerical profile initializes its own
/// shared 132612-byte table once; a normal process selects only one profile.
///
/// Numerical profile: source FastLog2f with fused Horner multiply-add on
/// AVX2+FMA, non-fused fallbacks, real division, integer count==total masking,
/// and the 1/4096 probability floor. Contribution multiplication and accumulator
/// subtraction stay separate, including on FMA; source does not use NegMulAdd.
/// SSE2 reduces (lane0+lane3)+(lane1+lane2); AVX2 first adds opposite 128-bit
/// halves, then applies that tree. This follows frozen Highway's
/// generic_ops-inl.h::{ReduceAcrossBlocks, ReduceWithinBlocks}. Tail bins enter
/// the appropriate lane BEFORE reduction, equivalent to zero padding. Scalar
/// fallback accumulates serially. FMA/other vector widths can round differently;
/// neither all-target bit parity nor identical tree selection is promised.
/// AVX2 tails use fault-suppressing masked loads/stores when a side needs
/// FastLog2f, avoiding up to fourteen scalar log evaluations per split. SSE2
/// uses bounded stack copies for its partial vectors. When both sides use the
/// small table, scalar tail lookups avoid gather overhead. No padding is needed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DefaultScorer {
    small: &'static [f32],
    backend: DefaultBackend,
}

/// One complete-search threshold, in the caller's stable bin order. `row`
/// addresses immutable deltas and may differ from `bin` after property packing.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HistogramSplit {
    pub(crate) bin: usize,
    pub(crate) row: usize,
    pub(crate) totals: (u32, u32),
}

impl Default for DefaultScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultScorer {
    /// Integer-only skipped-quantile movement; immutable histogram deltas.
    pub(crate) fn move_counts(&self, above: &mut [u32], below: &mut [u32], delta: &[u32]) {
        assert_eq!(above.len(), below.len());
        assert_eq!(above.len(), delta.len());
        #[cfg(target_arch = "x86_64")]
        match self.backend {
            DefaultBackend::Avx512Fma8 => {
                // SAFETY: detected AVX512F/DQ, AVX2/FMA and matching slices.
                unsafe { x86::avx512_fma8::move_counts(above, below, delta) };
                return;
            }
            DefaultBackend::Avx2 | DefaultBackend::Avx2Fma | DefaultBackend::Avx2Rcp => {
                // SAFETY: detected AVX2 and matching bounded slices.
                unsafe { x86::avx2::move_counts(above, below, delta) };
                return;
            }
            DefaultBackend::Sse2 => {
                // SAFETY: detected SSE2 and matching bounded slices.
                unsafe { x86::sse2::move_counts(above, below, delta) };
                return;
            }
            DefaultBackend::Scalar => {}
        }
        for ((r, l), &d) in above.iter_mut().zip(below).zip(delta) {
            *r = r.wrapping_sub(d);
            *l = l.wrapping_add(d);
        }
    }

    /// Source FastLog2f for positive finite description-penalty operands.
    pub(crate) fn log2(&self, value: f32) -> f32 {
        if self.backend.uses_fma() {
            fast_log2::<true>(value)
        } else {
            fast_log2::<false>(value)
        }
    }

    /// Lane width used by the source-order histogram compactor. Removing zero
    /// symbols must preserve both lane identity and each lane's summation order.
    pub(crate) fn lanes(&self) -> usize {
        match self.backend {
            DefaultBackend::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Sse2 => 4,
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2
            | DefaultBackend::Avx2Fma
            | DefaultBackend::Avx2Rcp
            | DefaultBackend::Avx512Fma8 => 8,
        }
    }

    pub(crate) fn new() -> Self {
        static BACKEND: OnceLock<DefaultBackend> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| {
            let detected =
                DefaultBackend::detect(std::env::var("RS_JXL_MA_AVX512").as_deref() == Ok("1"));
            #[cfg(target_arch = "x86_64")]
            if detected.uses_fma() && std::env::var("RS_JXL_MA_RCP").as_deref() == Ok("1") {
                return DefaultBackend::Avx2Rcp;
            }
            detected
        });
        Self::with_backend(*backend)
    }

    // Callers selecting a native backend must detect its CPU features first.
    fn with_backend(backend: DefaultBackend) -> Self {
        static SMALL: OnceLock<Box<[f32]>> = OnceLock::new();
        static SMALL_FMA: OnceLock<Box<[f32]>> = OnceLock::new();
        let cache = if backend.uses_fma() {
            &SMALL_FMA
        } else {
            &SMALL
        };
        let small = cache.get_or_init(|| {
            let limit = SMALL_HISTOGRAM_TOTAL as usize;
            let mut table = vec![0.0; (limit + 1) * (limit + 2) / 2];
            for total in 1..=SMALL_HISTOGRAM_TOTAL {
                let offset = total as usize * (total as usize + 1) / 2;
                let inverse = inverse_total(total);
                for count in 0..=total {
                    table[offset + count as usize] = if backend.uses_fma() {
                        source_entropy::<true>(count, total, inverse)
                    } else {
                        source_entropy::<false>(count, total, inverse)
                    };
                }
            }
            table.into_boxed_slice()
        });
        Self { small, backend }
    }

    /// Source EstimateBits using the already validated weighted population.
    /// Empty and all-zero histograms have total zero and return zero.
    pub(crate) fn estimate_bits(&self, counts: &[u32], total: u32) -> f32 {
        debug_assert!(total <= MAX_HISTOGRAM_TOTAL);
        match self.backend {
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2Rcp => {
                // SAFETY: AVX2/FMA detected; the kernel bounds every load.
                unsafe { x86::avx2_rcp::default_estimate(counts, total) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx512Fma8 => {
                // SAFETY: runtime feature/OS detection and internally bounded loads.
                unsafe { x86::avx512_fma8::default_estimate(counts, total) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2Fma => {
                // SAFETY: both AVX2/OS support and FMA detected; bounded loads.
                unsafe { x86::avx2_fma::default_estimate(counts, total) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2 => {
                // SAFETY: backend chosen by runtime detection; all loads are
                // bounded by counts.len(), independently of numerical inputs.
                unsafe { x86::avx2::default_estimate(counts, total) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Sse2 => {
                // SAFETY: runtime SSE2 detection and bounded loads.
                unsafe { x86::sse2::default_estimate(counts, total) }
            }
            DefaultBackend::Scalar => {
                let inverse = inverse_total(total);
                counts.iter().fold(0.0, |bits, &count| {
                    bits - source_entropy::<false>(count, total, inverse)
                })
            }
        }
    }

    /// Move immutable weighted deltas from above to below and score both in
    /// one pass. `totals` is (post_move_above, post_move_below). Panics before
    /// mutation if the three lengths differ. Numerical invariants are those
    /// documented on DefaultScorer; this method does not recompute totals.
    pub(crate) fn move_and_score(
        &self,
        above: &mut [u32],
        below: &mut [u32],
        delta: &[u32],
        totals: (u32, u32),
    ) -> (f32, f32) {
        assert_eq!(above.len(), below.len(), "MA histogram length mismatch");
        assert_eq!(above.len(), delta.len(), "MA delta length mismatch");
        debug_assert!(totals.0 <= MAX_HISTOGRAM_TOTAL && totals.1 <= MAX_HISTOGRAM_TOTAL);
        match self.backend {
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2Rcp => {
                // SAFETY: detected AVX2/FMA and matching bounded slices.
                unsafe { x86::avx2_rcp::default_move(self, above, below, delta, totals) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx512Fma8 => {
                // SAFETY: detected features and matching bounded slices.
                unsafe { x86::avx512_fma8::default_move(self, above, below, delta, totals) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2Fma => {
                // SAFETY: AVX2+FMA detected, matching lengths and bounded gathers.
                unsafe { x86::avx2_fma::default_move(self, above, below, delta, totals) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2 => {
                // SAFETY: runtime detection, matching lengths, and internally
                // bounded gathers. Numerical invariants are caller-owned.
                unsafe { x86::avx2::default_move(self, above, below, delta, totals) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Sse2 => {
                // SAFETY: runtime detection and the same slice guarantees.
                unsafe { x86::sse2::default_move(self, above, below, delta, totals) }
            }
            DefaultBackend::Scalar => {
                let row_r = self.small_row(totals.0);
                let row_l = self.small_row(totals.1);
                let ir = inverse_total(totals.0);
                let il = inverse_total(totals.1);
                let (mut br, mut bl) = (0.0, 0.0);
                for ((r, l), &d) in above.iter_mut().zip(below).zip(delta) {
                    *r = r.wrapping_sub(d);
                    *l = l.wrapping_add(d);
                    br -= cached_source_entropy::<false>(row_r, *r, totals.0, ir);
                    bl -= cached_source_entropy::<false>(row_l, *l, totals.1, il);
                }
                (br, bl)
            }
        }
    }

    /// Source active-block sweep. The node proves omitted blocks are zero for
    /// every descendant; validate block addresses before entering native code.
    pub(crate) fn move_and_score_sparse(
        &self,
        above: &mut [u32],
        below: &mut [u32],
        delta: &[u32],
        totals: (u32, u32),
        blocks: &[usize],
    ) -> (f32, f32) {
        assert_eq!(above.len(), below.len(), "MA histogram length mismatch");
        assert_eq!(above.len(), delta.len(), "MA delta length mismatch");
        let lanes = self.lanes();
        assert!(blocks.iter().all(|&b| b < above.len() && b % lanes == 0));
        assert!(blocks.windows(2).all(|b| b[0] < b[1]));
        match self.backend {
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2Rcp => {
                // SAFETY: detected AVX2/FMA, checked slices and block addresses.
                unsafe { x86::avx2_rcp::sparse_move(self, above, below, delta, totals, blocks) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2Fma | DefaultBackend::Avx512Fma8 => {
                // SAFETY: detected CPU features, matching slices, bounded aligned blocks.
                unsafe { x86::avx2_fma::sparse_move(self, above, below, delta, totals, blocks) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Avx2 => {
                // SAFETY: same checked block/slice contract and detected AVX2.
                unsafe { x86::avx2::sparse_move(self, above, below, delta, totals, blocks) }
            }
            #[cfg(target_arch = "x86_64")]
            DefaultBackend::Sse2 => {
                // SAFETY: same checked block/slice contract and detected SSE2.
                unsafe { x86::sse2::sparse_move(self, above, below, delta, totals, blocks) }
            }
            DefaultBackend::Scalar => {
                let (row_r, row_l) = (self.small_row(totals.0), self.small_row(totals.1));
                let (ir, il) = (inverse_total(totals.0), inverse_total(totals.1));
                let (mut br, mut bl) = (0.0, 0.0);
                for &i in blocks {
                    above[i] = above[i].wrapping_sub(delta[i]);
                    below[i] = below[i].wrapping_add(delta[i]);
                    br -= cached_source_entropy::<false>(row_r, above[i], totals.0, ir);
                    bl -= cached_source_entropy::<false>(row_l, below[i], totals.1, il);
                }
                (br, bl)
            }
        }
    }

    /// Dispatch once per predictor, not once per threshold. Deltas and sparse
    /// addresses are checked before native access; sparse proofs are immutable
    /// for the whole sweep. No allocation or change in lane arithmetic/order.
    /// `population` is the caller's already-known combined weighted total. It
    /// selects a kernel, not a resource limit; per-split totals bound lookups.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn split_sweep(
        &self,
        above: &mut [u32],
        below: &mut [u32],
        deltas: &[u32],
        stride: usize,
        population: u32,
        sparse: Option<&[usize]>,
        splits: impl Iterator<Item = HistogramSplit>,
        mut emit: impl FnMut(HistogramSplit, f32, f32),
    ) {
        assert_eq!(above.len(), below.len(), "MA histogram length mismatch");
        assert!(stride != 0 && stride >= above.len(), "MA delta stride");
        assert_eq!(deltas.len() % stride, 0, "MA delta rows");
        if let Some(blocks) = sparse {
            let lanes = self.lanes();
            assert!(blocks.iter().all(|&b| b < above.len() && b % lanes == 0));
            assert!(blocks.windows(2).all(|b| b[0] < b[1]));
        }
        #[cfg(target_arch = "x86_64")]
        match self.backend {
            DefaultBackend::Avx2Rcp => {
                // SAFETY: detected AVX2/FMA and checked histogram layout.
                unsafe {
                    x86::avx2_rcp::default_sweep(
                        self, above, below, deltas, stride, population, sparse, splits, emit,
                    )
                };
                return;
            }
            DefaultBackend::Avx512Fma8 => {
                // SAFETY: detected features and checked slice/block layout.
                unsafe {
                    x86::avx512_fma8::default_sweep(
                        self, above, below, deltas, stride, population, sparse, splits, emit,
                    )
                };
                return;
            }
            DefaultBackend::Avx2Fma => {
                // SAFETY: detected CPU features and checked slice/block layout.
                // The native sweep bounds each row before calling its kernel.
                unsafe {
                    x86::avx2_fma::default_sweep(
                        self, above, below, deltas, stride, population, sparse, splits, emit,
                    )
                };
                return;
            }
            DefaultBackend::Avx2 => {
                // SAFETY: same checked layout and detected AVX2.
                unsafe {
                    x86::avx2::default_sweep(
                        self, above, below, deltas, stride, population, sparse, splits, emit,
                    )
                };
                return;
            }
            DefaultBackend::Sse2 => {
                // SAFETY: same checked layout and detected SSE2.
                unsafe {
                    x86::sse2::default_sweep(
                        self, above, below, deltas, stride, population, sparse, splits, emit,
                    )
                };
                return;
            }
            DefaultBackend::Scalar => {}
        }
        for split in splits {
            let delta = split_delta(deltas, stride, above.len(), split.row);
            let bits = if let Some(blocks) = sparse {
                self.move_and_score_sparse(above, below, delta, split.totals, blocks)
            } else {
                self.move_and_score(above, below, delta, split.totals)
            };
            emit(split, bits.0, bits.1);
        }
    }

    fn small_row(&self, total: u32) -> &[f32] {
        if total > SMALL_HISTOGRAM_TOTAL {
            return &[];
        }
        let total = total as usize;
        let offset = total * (total + 1) / 2;
        &self.small[offset..offset + total + 1]
    }
}

#[inline]
fn split_delta(deltas: &[u32], stride: usize, symbols: usize, row: usize) -> &[u32] {
    assert!(row < deltas.len() / stride, "MA delta row out of bounds");
    &deltas[row * stride..row * stride + symbols]
}

#[inline]
fn source_entropy<const FMA: bool>(count: u32, total: u32, inverse: f32) -> f32 {
    let cf = count as f32;
    let log = if count == total {
        0.0
    } else {
        fast_log2::<FMA>((cf * inverse).max(MIN_PROBABILITY))
    };
    cf * log
}

#[inline]
fn cached_source_entropy<const FMA: bool>(
    row: &[f32],
    count: u32,
    total: u32,
    inverse: f32,
) -> f32 {
    if row.is_empty() {
        source_entropy::<FMA>(count, total, inverse)
    } else {
        row[count as usize]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaSimdError {
    TableLimitTooLarge,
    LengthMismatch,
    TotalTooLarge,
    CountUnderflow,
    CountOverflow,
}

impl fmt::Display for MaSimdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TableLimitTooLarge => "MA log table limit exceeds 65536",
            Self::LengthMismatch => "MA histograms and increments have different lengths",
            Self::TotalTooLarge => "MA histogram total exceeds the source i32::MAX limit",
            Self::CountUnderflow => "MA increment exceeds its source count",
            Self::CountOverflow => "MA destination count exceeds u32",
        })
    }
}

impl std::error::Error for MaSimdError {}

/// Reusable immutable `log2(count)` table; share it between workers if desired.
#[derive(Clone, Debug)]
pub struct LogTable {
    logs: Vec<f32>,
}

impl LogTable {
    pub(crate) fn shared() -> &'static Self {
        static TABLE: OnceLock<LogTable> = OnceLock::new();
        TABLE.get_or_init(|| Self::new(MAX_LOG_TABLE_TOTAL).expect("source log cache limit"))
    }

    /// Learner counterpart with immutable cache deltas and validated totals.
    /// Counts obey DefaultScorer's contract; gathers remain address-bounded even
    /// for invalid numerical input. Both totals must fit the count-log table.
    pub(crate) fn move_known_totals(
        &self,
        above: &mut [u32],
        below: &mut [u32],
        delta: &[u32],
        totals: (u32, u32),
    ) -> (f32, f32) {
        assert_eq!(above.len(), below.len());
        assert_eq!(above.len(), delta.len());
        assert!(totals.0 <= self.maximum_total() && totals.1 <= self.maximum_total());
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: detected CPU support, equal slices, bounded gathers.
                return unsafe { x86::avx2::move_readonly(self, above, below, delta, totals) };
            }
            if std::is_x86_feature_detected!("sse2") {
                // SAFETY: detected CPU support, equal slices, bounded gathers.
                return unsafe { x86::sse2::move_readonly(self, above, below, delta, totals) };
            }
        }
        self.move_readonly_scalar(above, below, delta, totals)
    }

    fn move_readonly_scalar(
        &self,
        above: &mut [u32],
        below: &mut [u32],
        delta: &[u32],
        totals: (u32, u32),
    ) -> (f32, f32) {
        let (mut rbits, mut lbits) = (0.0, 0.0);
        for ((r, l), &d) in above.iter_mut().zip(below).zip(delta) {
            *r -= d;
            *l += d;
            rbits += self.contribution(*r, totals.0);
            lbits += self.contribution(*l, totals.1);
        }
        (rbits, lbits)
    }

    /// Allocate only the bounded log cache (0..=65536). This is not a bound on
    /// accepted histogram populations; larger totals use the FastLog2f path.
    pub fn new(maximum_total: u32) -> Result<Self, MaSimdError> {
        if maximum_total > MAX_LOG_TABLE_TOTAL {
            return Err(MaSimdError::TableLimitTooLarge);
        }
        let mut logs = vec![0.0; maximum_total as usize + 1];
        for count in 1..=maximum_total {
            logs[count as usize] = f64::from(count).log2() as f32;
        }
        Ok(Self { logs })
    }

    /// Largest total served by the cache, NOT the accepted histogram limit.
    pub fn maximum_total(&self) -> u32 {
        (self.logs.len() - 1) as u32
    }

    /// Dense clipped cost, using the log cache when the total fits, otherwise
    /// the source FastLog2f estimator. Totals up to MAX_HISTOGRAM_TOTAL are valid.
    /// Empty/all-zero and one-symbol histograms return exactly zero.
    pub fn estimate_bits(&self, counts: &[u32]) -> Result<f32, MaSimdError> {
        let total = self.total(counts)?;
        #[cfg(target_arch = "x86_64")]
        {
            if counts.len() >= 8 && std::is_x86_feature_detected!("avx2") {
                // SAFETY: runtime AVX2 support; counts fit signed conversion.
                // The backend uses gathers only when the total fits the cache.
                return Ok(unsafe { x86::avx2::estimate(self, counts, total) });
            }
            if counts.len() >= 4 && std::is_x86_feature_detected!("sse2") {
                // SAFETY: runtime SSE2 support; backend checks the cache bound.
                return Ok(unsafe { x86::sse2::estimate(self, counts, total) });
            }
        }
        Ok(self.estimate_scalar_validated(counts, total))
    }

    /// Portable cost reference; same bounds and f32 operations, but a serial
    /// accumulation order rather than per-vector-lane accumulators.
    pub fn estimate_bits_scalar(&self, counts: &[u32]) -> Result<f32, MaSimdError> {
        Ok(self.estimate_scalar_validated(counts, self.total(counts)?))
    }

    /// Fused dense split update, returning `(bits_above, bits_below)`.
    /// All slices must have equal lengths. For every bin, `increase <= above`
    /// and `below + increase` must fit u32; both POST-move totals must be at most
    /// MAX_HISTOGRAM_TOTAL. A side exceeding the cache uses FastLog2f. On error,
    /// all slices are untouched. On success every increase is cleared. Totals
    /// are computed, not trusted caller hints. The method name is retained for
    /// compatibility with the original bounded-cache entry point.
    pub fn move_counts_and_log_bits(
        &self,
        above: &mut [u32],
        below: &mut [u32],
        increase: &mut [u32],
    ) -> Result<(f32, f32), MaSimdError> {
        let totals = self.validate_move(above, below, increase)?;
        #[cfg(target_arch = "x86_64")]
        {
            if above.len() >= 8 && std::is_x86_feature_detected!("avx2") {
                // SAFETY: runtime AVX2/OS support; validation bounds all slices,
                // additions/subtractions and float conversions. The backend
                // selects cache/fallback independently for each post-move total.
                return Ok(unsafe {
                    x86::avx2::move_and_score(self, above, below, increase, totals)
                });
            }
            if above.len() >= 4 && std::is_x86_feature_detected!("sse2") {
                // SAFETY: runtime SSE2 support and the same validated contract.
                return Ok(unsafe {
                    x86::sse2::move_and_score(self, above, below, increase, totals)
                });
            }
        }
        Ok(self.move_scalar_validated(above, below, increase, totals))
    }

    fn total(&self, counts: &[u32]) -> Result<u32, MaSimdError> {
        let mut total = 0u32;
        for &count in counts {
            total = total.checked_add(count).ok_or(MaSimdError::TotalTooLarge)?;
            if total > MAX_HISTOGRAM_TOTAL {
                return Err(MaSimdError::TotalTooLarge);
            }
        }
        Ok(total)
    }

    fn validate_move(
        &self,
        above: &[u32],
        below: &[u32],
        increase: &[u32],
    ) -> Result<(u32, u32), MaSimdError> {
        if above.len() != below.len() || above.len() != increase.len() {
            return Err(MaSimdError::LengthMismatch);
        }
        let (mut total_r, mut total_l) = (0u32, 0u32);
        for ((&r, &l), &delta) in above.iter().zip(below).zip(increase) {
            let r = r.checked_sub(delta).ok_or(MaSimdError::CountUnderflow)?;
            let l = l.checked_add(delta).ok_or(MaSimdError::CountOverflow)?;
            total_r = total_r.checked_add(r).ok_or(MaSimdError::TotalTooLarge)?;
            total_l = total_l.checked_add(l).ok_or(MaSimdError::TotalTooLarge)?;
            if total_r > MAX_HISTOGRAM_TOTAL || total_l > MAX_HISTOGRAM_TOTAL {
                return Err(MaSimdError::TotalTooLarge);
            }
        }
        Ok((total_r, total_l))
    }

    #[inline]
    fn contribution(&self, count: u32, total: u32) -> f32 {
        if total <= self.maximum_total() {
            count as f32
                * (self.logs[total as usize] - self.logs[count as usize]).min(PROBABILITY_BITS)
        } else {
            fast_contribution(count, total, inverse_total(total))
        }
    }

    fn estimate_scalar_validated(&self, counts: &[u32], total: u32) -> f32 {
        if total <= self.maximum_total() {
            counts
                .iter()
                .fold(0.0, |bits, &count| bits + self.contribution(count, total))
        } else {
            let inverse = inverse_total(total);
            counts.iter().fold(0.0, |bits, &count| {
                bits + fast_contribution(count, total, inverse)
            })
        }
    }

    fn move_scalar_validated(
        &self,
        above: &mut [u32],
        below: &mut [u32],
        increase: &mut [u32],
        totals: (u32, u32),
    ) -> (f32, f32) {
        let (mut bits_r, mut bits_l) = (0.0, 0.0);
        let ir = inverse_total(totals.0);
        let il = inverse_total(totals.1);
        for ((r, l), delta) in above.iter_mut().zip(below).zip(increase) {
            *r -= *delta;
            *l += *delta;
            *delta = 0;
            bits_r += if totals.0 <= self.maximum_total() {
                self.contribution(*r, totals.0)
            } else {
                fast_contribution(*r, totals.0, ir)
            };
            bits_l += if totals.1 <= self.maximum_total() {
                self.contribution(*l, totals.1)
            } else {
                fast_contribution(*l, totals.1, il)
            };
        }
        (bits_r, bits_l)
    }
}

#[derive(Clone, Copy)]
enum CountEntropyBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

/// Source CountEntropyTable, deliberately distinct from the f32 log cache.
/// The 512 KiB allocation and runtime feature detection are shared by workers.
pub(crate) struct CountEntropyTable {
    values: Box<[f64]>,
    backend: CountEntropyBackend,
}

pub(crate) struct CountEntropyUpdate {
    pub before: [f64; 2],
    pub after: [f64; 2],
    pub sums: [f64; 2],
}

impl CountEntropyTable {
    pub(crate) fn shared() -> &'static Self {
        static TABLE: OnceLock<CountEntropyTable> = OnceLock::new();
        TABLE.get_or_init(|| {
            let values = (0..=MAX_LOG_TABLE_TOTAL)
                .map(uncached_count_entropy)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let backend = CountEntropyBackend::Scalar;
            #[cfg(target_arch = "x86_64")]
            let backend = if std::is_x86_feature_detected!("avx2") {
                CountEntropyBackend::Avx2
            } else if std::is_x86_feature_detected!("sse2") {
                CountEntropyBackend::Sse2
            } else {
                backend
            };
            Self { values, backend }
        })
    }

    #[inline]
    pub(crate) fn get(&self, count: u32) -> f64 {
        self.values
            .get(count as usize)
            .copied()
            .unwrap_or_else(|| uncached_count_entropy(count))
    }

    pub(crate) fn sum(&self, counts: &[u32]) -> f64 {
        match self.backend {
            #[cfg(target_arch = "x86_64")]
            CountEntropyBackend::Avx2 => {
                if counts.len() < 4 {
                    return self.sum_scalar(counts);
                }
                // SAFETY: shared() detects AVX2/OS support once. The backend
                // bounds every gather independently and handles scalar tails.
                unsafe { count_x86::sum_avx2(self, counts) }
            }
            #[cfg(target_arch = "x86_64")]
            CountEntropyBackend::Sse2 => {
                if counts.len() < 2 {
                    return self.sum_scalar(counts);
                }
                // SAFETY: runtime SSE2 support; no readable padding required.
                unsafe { count_x86::sum_sse2(self, counts) }
            }
            CountEntropyBackend::Scalar => self.sum_scalar(counts),
        }
    }

    fn sum_scalar(&self, counts: &[u32]) -> f64 {
        counts.iter().map(|&count| self.get(count)).sum()
    }

    /// Two independent c*log2(c) accumulator updates, used for both sides of a
    /// validated split transfer. This is NOT a complete entropy calculation.
    /// It returns the same terms needed for the caller's rare-count updates.
    pub(crate) fn replace_pair(
        &self,
        before: [u32; 2],
        after: [u32; 2],
        sums: [f64; 2],
    ) -> CountEntropyUpdate {
        match self.backend {
            #[cfg(target_arch = "x86_64")]
            CountEntropyBackend::Avx2 => {
                // SAFETY: runtime AVX2 support and checked/bounded gathers.
                unsafe { count_x86::pair_avx2(self, before, after, sums) }
            }
            #[cfg(target_arch = "x86_64")]
            CountEntropyBackend::Sse2 => {
                // SAFETY: runtime SSE2 support; fixed two-element arrays.
                unsafe { count_x86::pair_sse2(self, before, after, sums) }
            }
            CountEntropyBackend::Scalar => self.replace_pair_scalar(before, after, sums),
        }
    }

    fn replace_pair_scalar(
        &self,
        before: [u32; 2],
        after: [u32; 2],
        sums: [f64; 2],
    ) -> CountEntropyUpdate {
        let before = before.map(|c| self.get(c));
        let after = after.map(|c| self.get(c));
        CountEntropyUpdate {
            before,
            after,
            sums: [
                sums[0] + (after[0] - before[0]),
                sums[1] + (after[1] - before[1]),
            ],
        }
    }
}

#[inline]
fn uncached_count_entropy(count: u32) -> f64 {
    if count == 0 {
        0.0
    } else {
        f64::from(count) * f64::from(count).log2()
    }
}

#[cfg(target_arch = "x86_64")]
mod count_x86 {
    use super::{CountEntropyTable, CountEntropyUpdate, MAX_LOG_TABLE_TOTAL};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn gather4(table: &CountEntropyTable, counts: [u32; 4]) -> __m256d {
        if counts.iter().all(|&count| count <= MAX_LOG_TABLE_TOTAL) {
            let indices = _mm_set_epi32(
                counts[3] as i32,
                counts[2] as i32,
                counts[1] as i32,
                counts[0] as i32,
            );
            // SAFETY: every index is <= 65536 and the shared table has 65537
            // entries. In particular, u32 counts above i32::MAX NEVER gather.
            unsafe { _mm256_i32gather_pd::<8>(table.values.as_ptr(), indices) }
        } else {
            // A mixed block keeps cached lanes cached; only large counts log.
            _mm256_set_pd(
                table.get(counts[3]),
                table.get(counts[2]),
                table.get(counts[1]),
                table.get(counts[0]),
            )
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sum_avx2(table: &CountEntropyTable, counts: &[u32]) -> f64 {
        let mut sum = _mm256_setzero_pd();
        let mut chunks = counts.chunks_exact(4);
        for chunk in &mut chunks {
            // SAFETY: AVX2 propagated, gather4 checks each cache index.
            let terms = unsafe { gather4(table, [chunk[0], chunk[1], chunk[2], chunk[3]]) };
            sum = _mm256_add_pd(sum, terms);
        }
        let mut lanes = [0.0; 4];
        // SAFETY: four writable f64 values; unaligned stores are supported.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), sum) };
        lanes.iter().sum::<f64>() + table.sum_scalar(chunks.remainder())
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sum_sse2(table: &CountEntropyTable, counts: &[u32]) -> f64 {
        let mut sum = _mm_setzero_pd();
        let mut chunks = counts.chunks_exact(2);
        for chunk in &mut chunks {
            sum = _mm_add_pd(sum, _mm_set_pd(table.get(chunk[1]), table.get(chunk[0])));
        }
        let mut lanes = [0.0; 2];
        // SAFETY: two writable f64 values.
        unsafe { _mm_storeu_pd(lanes.as_mut_ptr(), sum) };
        lanes[0] + lanes[1] + table.sum_scalar(chunks.remainder())
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn update_pair(before: __m128d, after: __m128d, sums: [f64; 2]) -> CountEntropyUpdate {
        let mut result = CountEntropyUpdate {
            before: [0.0; 2],
            after: [0.0; 2],
            sums: [0.0; 2],
        };
        // SAFETY: SSE2 propagated; every array has exactly two f64 values.
        unsafe {
            let updated = _mm_add_pd(_mm_loadu_pd(sums.as_ptr()), _mm_sub_pd(after, before));
            _mm_storeu_pd(result.before.as_mut_ptr(), before);
            _mm_storeu_pd(result.after.as_mut_ptr(), after);
            _mm_storeu_pd(result.sums.as_mut_ptr(), updated);
        }
        result
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pair_avx2(
        table: &CountEntropyTable,
        before: [u32; 2],
        after: [u32; 2],
        sums: [f64; 2],
    ) -> CountEntropyUpdate {
        // SAFETY: AVX2 propagated, gather4 independently checks all four counts;
        // AVX2 implies SSE2, and update_pair only accesses fixed-size arrays.
        unsafe {
            let terms = gather4(table, [before[0], before[1], after[0], after[1]]);
            update_pair(
                _mm256_castpd256_pd128(terms),
                _mm256_extractf128_pd::<1>(terms),
                sums,
            )
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn pair_sse2(
        table: &CountEntropyTable,
        before: [u32; 2],
        after: [u32; 2],
        sums: [f64; 2],
    ) -> CountEntropyUpdate {
        // SAFETY: SSE2 propagated; scalar lookups accept the full u32 domain.
        unsafe {
            update_pair(
                _mm_set_pd(table.get(before[1]), table.get(before[0])),
                _mm_set_pd(table.get(after[1]), table.get(after[0])),
                sums,
            )
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{
        DefaultScorer, HistogramSplit, LOG_P, LOG_Q, LogTable, MIN_PROBABILITY, PROBABILITY_BITS,
        cached_source_entropy, inverse_total, split_delta,
    };

    macro_rules! default_kernel {
        ($feature:literal, $lanes:literal, $int:ty, $float:ty, $fma:literal) => {
            #[inline]
            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn fast_log(value: $float) -> $float {
                let bits = cast_i(value);
                let exponent = sar_i::<23>(sub_i(bits, splat_i(0x3f2aaaab)));
                let mantissa = cast_f(sub_i(bits, shl_i::<23>(exponent)));
                let x = sub(mantissa, splat(1.0));
                // SAFETY: mul_add has this backend's feature requirements.
                // Only these Horner steps fuse; the final division/add do not.
                let (numerator, denominator) = unsafe {
                    (
                        mul_add(
                            mul_add(splat(LOG_P[2]), x, splat(LOG_P[1])),
                            x,
                            splat(LOG_P[0]),
                        ),
                        mul_add(
                            mul_add(splat(LOG_Q[2]), x, splat(LOG_Q[1])),
                            x,
                            splat(LOG_Q[0]),
                        ),
                    )
                };
                add(div(numerator, denominator), convert(exponent))
            }

            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn default_entropy<const LOOKUP: bool>(
                row: &[f32],
                counts: $int,
                inverse: $float,
                total: $int,
            ) -> $float {
                if LOOKUP {
                    // SAFETY: same CPU feature. The lookup specialization has
                    // a nonempty row; gather_entropy bounds its own indices.
                    unsafe { gather_entropy(row, counts) }
                } else {
                    let cf = convert(counts);
                    let probability = max(mul(cf, inverse), splat(MIN_PROBABILITY));
                    // SAFETY: same CPU feature and positive normal input.
                    let logs = unsafe { fast_log(probability) };
                    mul(cf, andnot_f(cast_f(eq_i(counts, total)), logs))
                }
            }

            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn default_estimate(counts: &[u32], total: u32) -> f32 {
                let inverse = inverse_total(total);
                let inv = splat(inverse);
                let tv = splat_i(total as i32);
                let end = counts.len() / $lanes * $lanes;
                let mut sum = zero();
                let mut lanes = [0.0; $lanes];
                // SAFETY: full loads are bounded; partial loads enable/copy
                // only the remaining elements. No readable padding is needed.
                unsafe {
                    for i in (0..end).step_by($lanes) {
                        let c = load(counts.as_ptr().add(i).cast());
                        sum = sub(sum, default_entropy::<false>(&[], c, inv, tv));
                    }
                    if end != counts.len() {
                        let mask = tail_mask(counts.len() - end);
                        let c = load_tail(counts.as_ptr().add(end), mask);
                        sum = sub(sum, default_entropy::<false>(&[], c, inv, tv));
                    }
                    store_f(lanes.as_mut_ptr(), sum);
                }
                reduce_default(lanes)
            }

            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn default_move(
                scorer: &DefaultScorer,
                above: &mut [u32],
                below: &mut [u32],
                delta: &[u32],
                totals: (u32, u32),
            ) -> (f32, f32) {
                // SAFETY: propagated CPU/slice contract; dense mode ignores blocks.
                unsafe { move_dispatch::<false>(scorer, above, below, delta, totals, &[]) }
            }

            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn sparse_move(
                scorer: &DefaultScorer,
                above: &mut [u32],
                below: &mut [u32],
                delta: &[u32],
                totals: (u32, u32),
                blocks: &[usize],
            ) -> (f32, f32) {
                // SAFETY: caller validates block alignment/order/bounds and CPU.
                unsafe { move_dispatch::<true>(scorer, above, below, delta, totals, blocks) }
            }

            #[allow(clippy::too_many_arguments)]
            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn default_sweep(
                scorer: &DefaultScorer,
                above: &mut [u32],
                below: &mut [u32],
                deltas: &[u32],
                stride: usize,
                population: u32,
                sparse: Option<&[usize]>,
                splits: impl Iterator<Item = HistogramSplit>,
                mut emit: impl FnMut(HistogramSplit, f32, f32),
            ) {
                // Transfers preserve combined weighted mass. If it fits the
                // existing table, both sides fit for every subsequent split.
                if population <= super::super::SMALL_HISTOGRAM_TOTAL {
                    // SAFETY: the wrapper checked lengths and sparse addresses;
                    // the specialized sweep still bounds every row and gather.
                    unsafe {
                        if let Some(blocks) = sparse {
                            small_sweep::<true>(
                                scorer, above, below, deltas, stride, blocks, splits, emit,
                            );
                        } else {
                            small_sweep::<false>(
                                scorer,
                                above,
                                below,
                                deltas,
                                stride,
                                &[],
                                splits,
                                emit,
                            );
                        }
                    }
                    return;
                }
                // Keep the sparse branch and address validation outside the
                // threshold loop. Row slicing remains checked for every move.
                if let Some(blocks) = sparse {
                    for split in splits {
                        debug_assert!(split.totals.0 <= i32::MAX as u32);
                        debug_assert!(split.totals.1 <= i32::MAX as u32);
                        let delta = split_delta(deltas, stride, above.len(), split.row);
                        // SAFETY: same target features; wrapper checked lengths
                        // and block addresses, and delta is a bounded row slice.
                        let (r, l) = unsafe {
                            move_dispatch::<true>(scorer, above, below, delta, split.totals, blocks)
                        };
                        emit(split, r, l);
                    }
                } else {
                    for split in splits {
                        debug_assert!(split.totals.0 <= i32::MAX as u32);
                        debug_assert!(split.totals.1 <= i32::MAX as u32);
                        let delta = split_delta(deltas, stride, above.len(), split.row);
                        // SAFETY: same checked layout; dense mode ignores blocks.
                        let (r, l) = unsafe {
                            move_dispatch::<false>(scorer, above, below, delta, split.totals, &[])
                        };
                        emit(split, r, l);
                    }
                }
            }

            #[allow(clippy::too_many_arguments)]
            #[target_feature(enable = $feature)]
            unsafe fn small_sweep<const SPARSE: bool>(
                scorer: &DefaultScorer,
                above: &mut [u32],
                below: &mut [u32],
                deltas: &[u32],
                stride: usize,
                blocks: &[usize],
                splits: impl Iterator<Item = HistogramSplit>,
                mut emit: impl FnMut(HistogramSplit, f32, f32),
            ) {
                let end = above.len() / $lanes * $lanes;
                let tail_end = if !SPARSE || blocks.last() == Some(&end) {
                    above.len()
                } else {
                    end
                };
                let block_count = if SPARSE {
                    blocks.partition_point(|&b| b < end)
                } else {
                    end / $lanes
                };
                let row = |total: u32| {
                    // On x86_64 these offsets fit usize for every u32 total.
                    // Checked slicing independently bounds the nonempty rows
                    // supplied to gathers, even for malformed numerical inputs.
                    let total = total as usize;
                    let offset = total * (total + 1) / 2;
                    &scorer.small[offset..offset + total + 1]
                };
                for split in splits {
                    debug_assert!(split.totals.0 <= i32::MAX as u32);
                    debug_assert!(split.totals.1 <= i32::MAX as u32);
                    let delta = split_delta(deltas, stride, above.len(), split.row);
                    // A valid sweep conserves mass. Keep the primitive's old
                    // behavior if later supplied totals break that invariant.
                    if u64::from(split.totals.0) + u64::from(split.totals.1)
                        > u64::from(super::super::SMALL_HISTOGRAM_TOTAL)
                    {
                        // SAFETY: same features and checked row/block layout.
                        let (r, l) = unsafe {
                            move_dispatch::<SPARSE>(
                                scorer,
                                above,
                                below,
                                delta,
                                split.totals,
                                blocks,
                            )
                        };
                        emit(split, r, l);
                        continue;
                    }
                    let row_r = row(split.totals.0);
                    let row_l = row(split.totals.1);
                    let (mut br, mut bl) = (zero(), zero());
                    let mut lanes_r = [0.0; $lanes];
                    let mut lanes_l = [0.0; $lanes];
                    // SAFETY: checked sparse addresses and matching slices bound
                    // full blocks. Both nonempty rows use bounded gathers.
                    unsafe {
                        for block in 0..block_count {
                            let i = if SPARSE {
                                blocks[block]
                            } else {
                                block * $lanes
                            };
                            let d = load(delta.as_ptr().add(i).cast());
                            let r = sub_i(load(above.as_ptr().add(i).cast()), d);
                            let l = add_i(load(below.as_ptr().add(i).cast()), d);
                            store_i(above.as_mut_ptr().add(i).cast(), r);
                            store_i(below.as_mut_ptr().add(i).cast(), l);
                            br = sub(br, gather_entropy(row_r, r));
                            bl = sub(bl, gather_entropy(row_l, l));
                        }
                        store_f(lanes_r.as_mut_ptr(), br);
                        store_f(lanes_l.as_mut_ptr(), bl);
                    }
                    for i in end..tail_end {
                        above[i] = above[i].wrapping_sub(delta[i]);
                        below[i] = below[i].wrapping_add(delta[i]);
                        lanes_r[i - end] -= row_r[above[i] as usize];
                        lanes_l[i - end] -= row_l[below[i] as usize];
                    }
                    emit(split, reduce_default(lanes_r), reduce_default(lanes_l));
                }
            }

            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn move_dispatch<const SPARSE: bool>(
                scorer: &DefaultScorer,
                above: &mut [u32],
                below: &mut [u32],
                delta: &[u32],
                totals: (u32, u32),
                blocks: &[usize],
            ) -> (f32, f32) {
                // SAFETY: caller checks equal lengths and runtime feature.
                // Each side specializes independently, outside the bin loop.
                unsafe {
                    match (
                        !scorer.small_row(totals.0).is_empty(),
                        !scorer.small_row(totals.1).is_empty(),
                    ) {
                        (true, true) => default_move_impl::<true, true, SPARSE>(
                            scorer, above, below, delta, totals, blocks,
                        ),
                        (true, false) => default_move_impl::<true, false, SPARSE>(
                            scorer, above, below, delta, totals, blocks,
                        ),
                        (false, true) => default_move_impl::<false, true, SPARSE>(
                            scorer, above, below, delta, totals, blocks,
                        ),
                        (false, false) => default_move_impl::<false, false, SPARSE>(
                            scorer, above, below, delta, totals, blocks,
                        ),
                    }
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn default_move_impl<
                const CACHE_R: bool,
                const CACHE_L: bool,
                const SPARSE: bool,
            >(
                scorer: &DefaultScorer,
                above: &mut [u32],
                below: &mut [u32],
                delta: &[u32],
                totals: (u32, u32),
                blocks: &[usize],
            ) -> (f32, f32) {
                let row_r = scorer.small_row(totals.0);
                let row_l = scorer.small_row(totals.1);
                let ir = inverse_total(totals.0);
                let il = inverse_total(totals.1);
                let (vr, vl) = (splat(ir), splat(il));
                let (tr, tl) = (splat_i(totals.0 as i32), splat_i(totals.1 as i32));
                let (mut br, mut bl) = (zero(), zero());
                let end = above.len() / $lanes * $lanes;
                let tail_end = if !SPARSE || blocks.last() == Some(&end) {
                    above.len()
                } else {
                    end
                };
                // Small-table tails need only bounded scalar loads. Vectorize
                // when at least one side would otherwise repeat FastLog2f.
                let vector_tail = !(CACHE_R && CACHE_L);
                let mut lanes_r = [0.0; $lanes];
                let mut lanes_l = [0.0; $lanes];
                // SAFETY: matching lengths ensure all complete-block accesses
                // are in bounds. Cached rows are nonempty, and native gather
                // bounds do not depend on caller-owned numerical invariants.
                unsafe {
                    let block_count = if SPARSE {
                        blocks.partition_point(|&b| b < end)
                    } else {
                        end / $lanes
                    };
                    for block in 0..block_count {
                        let i = if SPARSE {
                            blocks[block]
                        } else {
                            block * $lanes
                        };
                        let d = load(delta.as_ptr().add(i).cast());
                        let r = sub_i(load(above.as_ptr().add(i).cast()), d);
                        let l = add_i(load(below.as_ptr().add(i).cast()), d);
                        store_i(above.as_mut_ptr().add(i).cast(), r);
                        store_i(below.as_mut_ptr().add(i).cast(), l);
                        br = sub(br, default_entropy::<CACHE_R>(row_r, r, vr, tr));
                        bl = sub(bl, default_entropy::<CACHE_L>(row_l, l, vl, tl));
                    }
                    if vector_tail && end != tail_end {
                        let mask = tail_mask(tail_end - end);
                        let d = load_tail(delta.as_ptr().add(end), mask);
                        let r = sub_i(load_tail(above.as_ptr().add(end), mask), d);
                        let l = add_i(load_tail(below.as_ptr().add(end), mask), d);
                        store_tail(above.as_mut_ptr().add(end), mask, r);
                        store_tail(below.as_mut_ptr().add(end), mask, l);
                        br = sub(br, default_entropy::<CACHE_R>(row_r, r, vr, tr));
                        bl = sub(bl, default_entropy::<CACHE_L>(row_l, l, vl, tl));
                    }
                    store_f(lanes_r.as_mut_ptr(), br);
                    store_f(lanes_l.as_mut_ptr(), bl);
                }
                if !vector_tail {
                    for i in end..tail_end {
                        above[i] = above[i].wrapping_sub(delta[i]);
                        below[i] = below[i].wrapping_add(delta[i]);
                        lanes_r[i - end] -=
                            cached_source_entropy::<$fma>(row_r, above[i], totals.0, ir);
                        lanes_l[i - end] -=
                            cached_source_entropy::<$fma>(row_l, below[i], totals.1, il);
                    }
                }
                (reduce_default(lanes_r), reduce_default(lanes_l))
            }
        };
    }

    macro_rules! log_kernel {
        ($feature:literal, $lanes:literal, $int:ty, $float:ty) => {
            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn move_counts(
                above: &mut [u32],
                below: &mut [u32],
                delta: &[u32],
            ) {
                let end = above.len() / $lanes * $lanes;
                // SAFETY: caller checks CPU/slice lengths; complete blocks only.
                unsafe {
                    for i in (0..end).step_by($lanes) {
                        let d = load(delta.as_ptr().add(i).cast());
                        let r = sub_i(load(above.as_ptr().add(i).cast()), d);
                        let l = add_i(load(below.as_ptr().add(i).cast()), d);
                        store_i(above.as_mut_ptr().add(i).cast(), r);
                        store_i(below.as_mut_ptr().add(i).cast(), l);
                    }
                }
                for i in end..above.len() {
                    above[i] = above[i].wrapping_sub(delta[i]);
                    below[i] = below[i].wrapping_add(delta[i]);
                }
            }

            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn move_readonly(
                table: &LogTable,
                above: &mut [u32],
                below: &mut [u32],
                delta: &[u32],
                totals: (u32, u32),
            ) -> (f32, f32) {
                let (mut sr, mut sl) = (zero(), zero());
                let pr = splat(table.logs[totals.0 as usize]);
                let pl = splat(table.logs[totals.1 as usize]);
                let tr = splat_i(totals.0 as i32);
                let tl = splat_i(totals.1 as i32);
                let end = above.len() / $lanes * $lanes;
                let (mut rlanes, mut llanes) = ([0.0; $lanes], [0.0; $lanes]);
                // SAFETY: CPU and matching slices checked by the safe wrapper;
                // complete unaligned blocks, and gather() bounds every address.
                unsafe {
                    for i in (0..end).step_by($lanes) {
                        let d = load(delta.as_ptr().add(i).cast());
                        let r = sub_i(load(above.as_ptr().add(i).cast()), d);
                        let l = add_i(load(below.as_ptr().add(i).cast()), d);
                        store_i(above.as_mut_ptr().add(i).cast(), r);
                        store_i(below.as_mut_ptr().add(i).cast(), l);
                        sr = add(sr, contribution::<true>(table, r, pr, tr));
                        sl = add(sl, contribution::<true>(table, l, pl, tl));
                    }
                    store_f(rlanes.as_mut_ptr(), sr);
                    store_f(llanes.as_mut_ptr(), sl);
                }
                for i in end..above.len() {
                    above[i] -= delta[i];
                    below[i] += delta[i];
                    rlanes[i - end] += table.contribution(above[i], totals.0);
                    llanes[i - end] += table.contribution(below[i], totals.1);
                }
                (reduce_default(rlanes), reduce_default(llanes))
            }

            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn contribution<const LOOKUP: bool>(
                table: &LogTable,
                counts: $int,
                parameter: $float,
                total: $int,
            ) -> $float {
                if LOOKUP {
                    // SAFETY: this specialization is called only when the
                    // integer total (and therefore every count) fits the cache.
                    let log_counts = unsafe { gather(table, counts) };
                    mul(
                        convert(counts),
                        min(splat(PROBABILITY_BITS), sub(parameter, log_counts)),
                    )
                } else {
                    let cf = convert(counts);
                    let probability = max(mul(cf, parameter), splat(MIN_PROBABILITY));
                    // SAFETY: same CPU features; finite positive probabilities
                    // in the source estimator's normal-float domain.
                    let logs = unsafe { fast_log(probability) };
                    // Comparing converted floats would incorrectly zero bins
                    // near 2^24/2^31 that round to the same f32 as the total.
                    let logs = andnot_f(cast_f(eq_i(counts, total)), logs);
                    sub(zero(), mul(cf, logs))
                }
            }

            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn estimate(
                table: &LogTable,
                counts: &[u32],
                total: u32,
            ) -> f32 {
                // SAFETY: propagated CPU/validated signed-total contract; the
                // cache decision occurs once, outside the vector loop.
                unsafe {
                    if total <= table.maximum_total() {
                        estimate_impl::<true>(table, counts, total)
                    } else {
                        estimate_impl::<false>(table, counts, total)
                    }
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn estimate_impl<const LOOKUP: bool>(
                table: &LogTable,
                counts: &[u32],
                total: u32,
            ) -> f32 {
                let mut sum = zero();
                let parameter = splat(if LOOKUP {
                    table.logs[total as usize]
                } else {
                    inverse_total(total)
                });
                let total_v = splat_i(total as i32);
                let end = counts.len() / $lanes * $lanes;
                // SAFETY: runtime feature and table/count contract supplied by
                // caller. Full blocks read exactly LANES in-bounds u32s; tail
                // remains scalar. Reduction stores exactly LANES f32s.
                unsafe {
                    for i in (0..end).step_by($lanes) {
                        let counts = load(counts.as_ptr().add(i).cast());
                        sum = add(
                            sum,
                            contribution::<LOOKUP>(table, counts, parameter, total_v),
                        );
                    }
                    let mut lanes = [0.0f32; $lanes];
                    store_f(lanes.as_mut_ptr(), sum);
                    let mut bits: f32 = lanes.iter().sum();
                    for &count in &counts[end..] {
                        bits += table.contribution(count, total);
                    }
                    bits
                }
            }

            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn move_and_score(
                table: &LogTable,
                above: &mut [u32],
                below: &mut [u32],
                increase: &mut [u32],
                totals: (u32, u32),
            ) -> (f32, f32) {
                // SAFETY: propagated CPU/validated slice and arithmetic
                // contract. Each side chooses cache/fallback independently,
                // with no inner-loop path test or speculative out-of-bounds
                // gather on the side whose total exceeds the cache.
                unsafe {
                    match (
                        totals.0 <= table.maximum_total(),
                        totals.1 <= table.maximum_total(),
                    ) {
                        (true, true) => {
                            move_impl::<true, true>(table, above, below, increase, totals)
                        }
                        (true, false) => {
                            move_impl::<true, false>(table, above, below, increase, totals)
                        }
                        (false, true) => {
                            move_impl::<false, true>(table, above, below, increase, totals)
                        }
                        (false, false) => {
                            move_impl::<false, false>(table, above, below, increase, totals)
                        }
                    }
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn move_impl<const CACHE_R: bool, const CACHE_L: bool>(
                table: &LogTable,
                above: &mut [u32],
                below: &mut [u32],
                increase: &mut [u32],
                totals: (u32, u32),
            ) -> (f32, f32) {
                let (mut sum_r, mut sum_l) = (zero(), zero());
                let pr = splat(if CACHE_R {
                    table.logs[totals.0 as usize]
                } else {
                    inverse_total(totals.0)
                });
                let pl = splat(if CACHE_L {
                    table.logs[totals.1 as usize]
                } else {
                    inverse_total(totals.1)
                });
                let tr = splat_i(totals.0 as i32);
                let tl = splat_i(totals.1 as i32);
                let end = above.len() / $lanes * $lanes;
                // SAFETY: caller validated identical slice lengths, arithmetic,
                // and post-move totals; those also bound every gather index.
                // All loads/stores are unaligned complete blocks, no padding.
                unsafe {
                    for i in (0..end).step_by($lanes) {
                        let delta = load(increase.as_ptr().add(i).cast());
                        let r = sub_i(load(above.as_ptr().add(i).cast()), delta);
                        let l = add_i(load(below.as_ptr().add(i).cast()), delta);
                        store_i(above.as_mut_ptr().add(i).cast(), r);
                        store_i(below.as_mut_ptr().add(i).cast(), l);
                        store_i(increase.as_mut_ptr().add(i).cast(), zero_i());
                        sum_r = add(sum_r, contribution::<CACHE_R>(table, r, pr, tr));
                        sum_l = add(sum_l, contribution::<CACHE_L>(table, l, pl, tl));
                    }
                    let mut lanes_r = [0.0f32; $lanes];
                    let mut lanes_l = [0.0f32; $lanes];
                    store_f(lanes_r.as_mut_ptr(), sum_r);
                    store_f(lanes_l.as_mut_ptr(), sum_l);
                    let (mut bits_r, mut bits_l): (f32, f32) =
                        (lanes_r.iter().sum(), lanes_l.iter().sum());
                    for i in end..above.len() {
                        above[i] -= increase[i];
                        below[i] += increase[i];
                        increase[i] = 0;
                        bits_r += table.contribution(above[i], totals.0);
                        bits_l += table.contribution(below[i], totals.1);
                    }
                    (bits_r, bits_l)
                }
            }
        };
    }

    pub(super) mod sse2 {
        use super::{
            DefaultScorer, HistogramSplit, LOG_P, LOG_Q, LogTable, MIN_PROBABILITY,
            PROBABILITY_BITS, cached_source_entropy, inverse_total, split_delta,
        };
        use std::arch::x86_64::*;
        use std::arch::x86_64::{
            _mm_add_epi32 as add_i, _mm_add_ps as add, _mm_andnot_ps as andnot_f,
            _mm_castps_si128 as cast_i, _mm_castsi128_ps as cast_f, _mm_cmpeq_epi32 as eq_i,
            _mm_cvtepi32_ps as convert, _mm_div_ps as div, _mm_loadu_si128 as load,
            _mm_max_ps as max, _mm_min_ps as min, _mm_mul_ps as mul, _mm_set1_epi32 as splat_i,
            _mm_set1_ps as splat, _mm_setzero_ps as zero, _mm_setzero_si128 as zero_i,
            _mm_slli_epi32 as shl_i, _mm_srai_epi32 as sar_i, _mm_storeu_ps as store_f,
            _mm_storeu_si128 as store_i, _mm_sub_epi32 as sub_i, _mm_sub_ps as sub,
        };

        #[inline]
        fn tail_mask(len: usize) -> usize {
            debug_assert!(len < 4);
            len
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn load_tail(ptr: *const u32, len: usize) -> __m128i {
            let mut lanes = [0u32; 4];
            // SAFETY: the caller supplies len readable elements, with len < 4.
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, lanes.as_mut_ptr(), len);
                load(lanes.as_ptr().cast())
            }
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn store_tail(ptr: *mut u32, len: usize, value: __m128i) {
            let mut lanes = [0u32; 4];
            // SAFETY: the caller supplies len writable elements, with len < 4.
            unsafe {
                store_i(lanes.as_mut_ptr().cast(), value);
                std::ptr::copy_nonoverlapping(lanes.as_ptr(), ptr, len);
            }
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn gather(table: &LogTable, counts: __m128i) -> __m128 {
            let mut indices = [0u32; 4];
            // SAFETY: exactly four initialized/writable u32 lanes. SSE2 lacks
            // gather; bounded scalar table loads feed vector float arithmetic.
            unsafe { _mm_storeu_si128(indices.as_mut_ptr().cast(), counts) };
            _mm_set_ps(
                table.logs[indices[3] as usize],
                table.logs[indices[2] as usize],
                table.logs[indices[1] as usize],
                table.logs[indices[0] as usize],
            )
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn gather_entropy(row: &[f32], counts: __m128i) -> __m128 {
            let mut indices = [0u32; 4];
            // SAFETY: the store writes exactly four initialized u32s. Scalar
            // indexing keeps even invalid caller counts memory-safe.
            unsafe { _mm_storeu_si128(indices.as_mut_ptr().cast(), counts) };
            _mm_set_ps(
                row[indices[3] as usize],
                row[indices[2] as usize],
                row[indices[1] as usize],
                row[indices[0] as usize],
            )
        }

        #[inline]
        fn reduce_default(v: [f32; 4]) -> f32 {
            (v[0] + v[3]) + (v[1] + v[2])
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn mul_add(a: __m128, b: __m128, c: __m128) -> __m128 {
            add(mul(a, b), c)
        }

        default_kernel!("sse2", 4, __m128i, __m128, false);
        log_kernel!("sse2", 4, __m128i, __m128);
    }

    pub(super) mod avx2 {
        use super::{
            DefaultScorer, HistogramSplit, LOG_P, LOG_Q, LogTable, MIN_PROBABILITY,
            PROBABILITY_BITS, cached_source_entropy, inverse_total, split_delta,
        };
        use std::arch::x86_64::*;
        use std::arch::x86_64::{
            _mm256_add_epi32 as add_i, _mm256_add_ps as add, _mm256_andnot_ps as andnot_f,
            _mm256_castps_si256 as cast_i, _mm256_castsi256_ps as cast_f,
            _mm256_cmpeq_epi32 as eq_i, _mm256_cvtepi32_ps as convert, _mm256_div_ps as div,
            _mm256_loadu_si256 as load, _mm256_max_ps as max, _mm256_min_ps as min,
            _mm256_mul_ps as mul, _mm256_set1_epi32 as splat_i, _mm256_set1_ps as splat,
            _mm256_setzero_ps as zero, _mm256_setzero_si256 as zero_i, _mm256_slli_epi32 as shl_i,
            _mm256_srai_epi32 as sar_i, _mm256_storeu_ps as store_f,
            _mm256_storeu_si256 as store_i, _mm256_sub_epi32 as sub_i, _mm256_sub_ps as sub,
        };

        #[inline]
        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn tail_mask(len: usize) -> __m256i {
            debug_assert!(len < 8);
            _mm256_cmpgt_epi32(
                _mm256_set1_epi32(len as i32),
                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            )
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn load_tail(ptr: *const u32, mask: __m256i) -> __m256i {
            // SAFETY: enabled lanes are within the caller's slice. AVX2 masked
            // loads suppress memory accesses for disabled lanes and return zero.
            unsafe { _mm256_maskload_epi32(ptr.cast(), mask) }
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn store_tail(ptr: *mut u32, mask: __m256i, value: __m256i) {
            // SAFETY: the same mask bounds stores; adjacent bins/cache rows are
            // never written, including when the tail ends at a page boundary.
            unsafe { _mm256_maskstore_epi32(ptr.cast(), mask, value) };
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn gather(table: &LogTable, counts: __m256i) -> __m256 {
            let last = _mm256_set1_epi32(table.maximum_total() as i32);
            let bounded = _mm256_min_epu32(counts, last);
            // SAFETY: bound addresses independently of numerical invariants.
            // Valid counts are unchanged, not clamped as a scoring policy.
            unsafe { _mm256_i32gather_ps::<4>(table.logs.as_ptr(), bounded) }
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn gather_entropy(row: &[f32], counts: __m256i) -> __m256 {
            // The selected triangular row is nonempty and at most 257 floats.
            // Clamp only gather addresses, so this safe crate API cannot read
            // out of bounds even if its numerical contract is violated. This
            // is a no-op for valid counts, not histogram validation or repair.
            let last = _mm256_set1_epi32((row.len() - 1) as i32);
            let bounded = _mm256_min_epu32(counts, last);
            // SAFETY: every unsigned index now lies within this nonempty row.
            unsafe { _mm256_i32gather_ps::<4>(row.as_ptr(), bounded) }
        }

        #[inline]
        pub(super) fn reduce_default(v: [f32; 8]) -> f32 {
            let a = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
            (a[0] + a[3]) + (a[1] + a[2])
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn mul_add(a: __m256, b: __m256, c: __m256) -> __m256 {
            add(mul(a, b), c)
        }

        default_kernel!("avx2", 8, __m256i, __m256, false);
        log_kernel!("avx2", 8, __m256i, __m256);
    }

    pub(super) mod avx2_fma {
        use super::avx2::{gather_entropy, load_tail, reduce_default, store_tail, tail_mask};
        use super::{
            DefaultScorer, HistogramSplit, LOG_P, LOG_Q, MIN_PROBABILITY, cached_source_entropy,
            inverse_total, split_delta,
        };
        use std::arch::x86_64::*;
        use std::arch::x86_64::{
            _mm256_add_epi32 as add_i, _mm256_add_ps as add, _mm256_andnot_ps as andnot_f,
            _mm256_castps_si256 as cast_i, _mm256_castsi256_ps as cast_f,
            _mm256_cmpeq_epi32 as eq_i, _mm256_cvtepi32_ps as convert, _mm256_div_ps as div,
            _mm256_loadu_si256 as load, _mm256_max_ps as max, _mm256_mul_ps as mul,
            _mm256_set1_epi32 as splat_i, _mm256_set1_ps as splat, _mm256_setzero_ps as zero,
            _mm256_slli_epi32 as shl_i, _mm256_srai_epi32 as sar_i, _mm256_storeu_ps as store_f,
            _mm256_storeu_si256 as store_i, _mm256_sub_epi32 as sub_i, _mm256_sub_ps as sub,
        };

        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn mul_add(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fmadd_ps(a, b, c)
        }

        default_kernel!("avx2,fma", 8, __m256i, __m256, true);
    }

    pub(super) mod avx2_rcp {
        use super::avx2::{gather_entropy, load_tail, reduce_default, store_tail, tail_mask};
        use super::{
            DefaultScorer, HistogramSplit, LOG_P, LOG_Q, MIN_PROBABILITY, cached_source_entropy,
            inverse_total, split_delta,
        };
        use std::arch::x86_64::*;
        use std::arch::x86_64::{
            _mm256_add_epi32 as add_i, _mm256_add_ps as add, _mm256_andnot_ps as andnot_f,
            _mm256_castps_si256 as cast_i, _mm256_castsi256_ps as cast_f,
            _mm256_cmpeq_epi32 as eq_i, _mm256_cvtepi32_ps as convert, _mm256_loadu_si256 as load,
            _mm256_max_ps as max, _mm256_mul_ps as mul, _mm256_set1_epi32 as splat_i,
            _mm256_set1_ps as splat, _mm256_setzero_ps as zero, _mm256_slli_epi32 as shl_i,
            _mm256_srai_epi32 as sar_i, _mm256_storeu_ps as store_f,
            _mm256_storeu_si256 as store_i, _mm256_sub_epi32 as sub_i, _mm256_sub_ps as sub,
        };

        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn mul_add(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fmadd_ps(a, b, c)
        }

        #[inline]
        #[target_feature(enable = "avx2,fma")]
        fn div(a: __m256, b: __m256) -> __m256 {
            // FastLog2's reduced rational denominator is positive and bounded
            // away from zero. This is not a general IEEE division replacement.
            let r = _mm256_rcp_ps(b);
            let error = _mm256_fnmadd_ps(b, r, _mm256_set1_ps(1.0));
            _mm256_mul_ps(a, _mm256_fmadd_ps(r, error, r))
        }

        default_kernel!("avx2,fma", 8, __m256i, __m256, true);
    }

    pub(super) mod avx512_fma8 {
        use super::super::SMALL_HISTOGRAM_TOTAL;
        use super::{
            DefaultScorer, HistogramSplit, LOG_P, LOG_Q, MIN_PROBABILITY, avx2, avx2_fma,
            inverse_total, split_delta,
        };
        use std::arch::x86_64::*;

        #[inline]
        #[target_feature(enable = "avx2,fma,avx512f,avx512dq")]
        pub(in super::super) unsafe fn fast_log(value: __m512) -> __m512 {
            let bits = _mm512_castps_si512(value);
            let exponent =
                _mm512_srai_epi32::<23>(_mm512_sub_epi32(bits, _mm512_set1_epi32(0x3f2aaaab)));
            let mantissa =
                _mm512_castsi512_ps(_mm512_sub_epi32(bits, _mm512_slli_epi32::<23>(exponent)));
            let x = _mm512_sub_ps(mantissa, _mm512_set1_ps(1.0));
            let numerator = _mm512_fmadd_ps(
                _mm512_fmadd_ps(_mm512_set1_ps(LOG_P[2]), x, _mm512_set1_ps(LOG_P[1])),
                x,
                _mm512_set1_ps(LOG_P[0]),
            );
            let denominator = _mm512_fmadd_ps(
                _mm512_fmadd_ps(_mm512_set1_ps(LOG_Q[2]), x, _mm512_set1_ps(LOG_Q[1])),
                x,
                _mm512_set1_ps(LOG_Q[0]),
            );
            _mm512_add_ps(
                _mm512_div_ps(numerator, denominator),
                _mm512_cvtepi32_ps(exponent),
            )
        }

        #[inline]
        #[target_feature(enable = "avx2,fma,avx512f,avx512dq")]
        unsafe fn entropy(counts: __m512i, inverse: __m512, total: __m512i) -> __m512 {
            let cf = _mm512_cvtepi32_ps(counts);
            let probability =
                _mm512_max_ps(_mm512_mul_ps(cf, inverse), _mm512_set1_ps(MIN_PROBABILITY));
            // SAFETY: the same features are enabled; probabilities are normal.
            let logs = unsafe { fast_log(probability) };
            let unequal = !_mm512_cmpeq_epi32_mask(counts, total);
            _mm512_mul_ps(cf, _mm512_maskz_mov_ps(unequal, logs))
        }

        #[inline]
        #[target_feature(enable = "avx2,fma,avx512f,avx512dq")]
        unsafe fn subtract_terms(sum: __m256, terms: __m512, high: bool) -> __m256 {
            // Keep one accumulator: (sum - low) - high, never sum - (low + high)
            // or separate 16-lane partial sums. This also preserves compaction.
            let sum = _mm256_sub_ps(sum, _mm512_castps512_ps256(terms));
            if high {
                _mm256_sub_ps(sum, _mm512_extractf32x8_ps::<1>(terms))
            } else {
                sum
            }
        }

        #[inline]
        #[target_feature(enable = "avx2,fma,avx512f,avx512dq")]
        unsafe fn reduce(sum: __m256) -> f32 {
            let mut lanes = [0.0; 8];
            // SAFETY: eight writable f32 values and AVX2 enabled.
            unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
            avx2::reduce_default(lanes)
        }

        #[target_feature(enable = "avx2,fma,avx512f,avx512dq")]
        pub(in super::super) unsafe fn default_estimate(counts: &[u32], total: u32) -> f32 {
            if counts.len() < 16 {
                // SAFETY: AVX2/FMA enabled and the original bounded slice.
                return unsafe { avx2_fma::default_estimate(counts, total) };
            }
            let inverse = _mm512_set1_ps(inverse_total(total));
            let total = _mm512_set1_epi32(total as i32);
            let end = counts.len() / 16 * 16;
            let mut sum = _mm256_setzero_ps();
            // SAFETY: full vectors fit the slice. Tail masks suppress every
            // out-of-bounds load, without requiring readable padding.
            unsafe {
                for i in (0..end).step_by(16) {
                    let c = _mm512_loadu_si512(counts.as_ptr().add(i).cast());
                    sum = subtract_terms(sum, entropy(c, inverse, total), true);
                }
                let tail = counts.len() - end;
                if tail != 0 {
                    let mask = ((1u32 << tail) - 1) as u16;
                    let c = _mm512_maskz_loadu_epi32(mask, counts.as_ptr().add(end).cast());
                    sum = subtract_terms(sum, entropy(c, inverse, total), tail > 8);
                }
                reduce(sum)
            }
        }

        #[target_feature(enable = "avx2,fma,avx512f,avx512dq")]
        pub(in super::super) unsafe fn move_counts(
            above: &mut [u32],
            below: &mut [u32],
            delta: &[u32],
        ) {
            let end = above.len() / 16 * 16;
            // SAFETY: wrapper checks equal lengths. Each full load/store fits;
            // AVX2 handles the remaining bounded slices with wrapping arithmetic.
            unsafe {
                for i in (0..end).step_by(16) {
                    let d = _mm512_loadu_si512(delta.as_ptr().add(i).cast());
                    let r = _mm512_sub_epi32(_mm512_loadu_si512(above.as_ptr().add(i).cast()), d);
                    let l = _mm512_add_epi32(_mm512_loadu_si512(below.as_ptr().add(i).cast()), d);
                    _mm512_storeu_si512(above.as_mut_ptr().add(i).cast(), r);
                    _mm512_storeu_si512(below.as_mut_ptr().add(i).cast(), l);
                }
                avx2::move_counts(&mut above[end..], &mut below[end..], &delta[end..]);
            }
        }

        #[target_feature(enable = "avx2,fma,avx512f,avx512dq")]
        pub(in super::super) unsafe fn default_move(
            scorer: &DefaultScorer,
            above: &mut [u32],
            below: &mut [u32],
            delta: &[u32],
            totals: (u32, u32),
        ) -> (f32, f32) {
            if above.len() < 16
                || totals.0 <= SMALL_HISTOGRAM_TOTAL
                || totals.1 <= SMALL_HISTOGRAM_TOTAL
            {
                // SAFETY: AVX2/FMA enabled, equal slices; shared small rows are
                // exactly the original FMA rows, including mixed lookup/log sides.
                return unsafe { avx2_fma::default_move(scorer, above, below, delta, totals) };
            }
            let vr = _mm512_set1_ps(inverse_total(totals.0));
            let vl = _mm512_set1_ps(inverse_total(totals.1));
            let tr = _mm512_set1_epi32(totals.0 as i32);
            let tl = _mm512_set1_epi32(totals.1 as i32);
            let (mut br, mut bl) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let end = above.len() / 16 * 16;
            // SAFETY: equal checked lengths bound full accesses. Tail masks
            // bound all three loads and both stores; delta is never mutated.
            unsafe {
                for i in (0..end).step_by(16) {
                    let d = _mm512_loadu_si512(delta.as_ptr().add(i).cast());
                    let r = _mm512_sub_epi32(_mm512_loadu_si512(above.as_ptr().add(i).cast()), d);
                    let l = _mm512_add_epi32(_mm512_loadu_si512(below.as_ptr().add(i).cast()), d);
                    _mm512_storeu_si512(above.as_mut_ptr().add(i).cast(), r);
                    _mm512_storeu_si512(below.as_mut_ptr().add(i).cast(), l);
                    br = subtract_terms(br, entropy(r, vr, tr), true);
                    bl = subtract_terms(bl, entropy(l, vl, tl), true);
                }
                let tail = above.len() - end;
                if tail != 0 {
                    let mask = ((1u32 << tail) - 1) as u16;
                    let d = _mm512_maskz_loadu_epi32(mask, delta.as_ptr().add(end).cast());
                    let r = _mm512_sub_epi32(
                        _mm512_maskz_loadu_epi32(mask, above.as_ptr().add(end).cast()),
                        d,
                    );
                    let l = _mm512_add_epi32(
                        _mm512_maskz_loadu_epi32(mask, below.as_ptr().add(end).cast()),
                        d,
                    );
                    _mm512_mask_storeu_epi32(above.as_mut_ptr().add(end).cast(), mask, r);
                    _mm512_mask_storeu_epi32(below.as_mut_ptr().add(end).cast(), mask, l);
                    br = subtract_terms(br, entropy(r, vr, tr), tail > 8);
                    bl = subtract_terms(bl, entropy(l, vl, tl), tail > 8);
                }
                (reduce(br), reduce(bl))
            }
        }

        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = "avx2,fma,avx512f,avx512dq")]
        pub(in super::super) unsafe fn default_sweep(
            scorer: &DefaultScorer,
            above: &mut [u32],
            below: &mut [u32],
            deltas: &[u32],
            stride: usize,
            population: u32,
            sparse: Option<&[usize]>,
            splits: impl Iterator<Item = HistogramSplit>,
            mut emit: impl FnMut(HistogramSplit, f32, f32),
        ) {
            if sparse.is_some() || above.len() < 16 {
                // SAFETY: original validated 8-lane sparse layout and AVX2/FMA.
                unsafe {
                    avx2_fma::default_sweep(
                        scorer, above, below, deltas, stride, population, sparse, splits, emit,
                    )
                };
                return;
            }
            for split in splits {
                debug_assert!(split.totals.0 <= i32::MAX as u32);
                debug_assert!(split.totals.1 <= i32::MAX as u32);
                let delta = split_delta(deltas, stride, above.len(), split.row);
                // SAFETY: same features and bounded, equal-length row slices.
                let (r, l) = unsafe { default_move(scorer, above, below, delta, split.totals) };
                emit(split, r, l);
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "ma_simd_avx512_tests.rs"]
mod avx512_tests;

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "ma_simd_rcp_tests.rs"]
mod rcp_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // Independent FMA expression with the frozen literals rounded to f32 bits.
    // Do not call the production polynomial or share its coefficient arrays.
    fn fma_log_oracle(value: f32) -> f32 {
        let bits = value.to_bits() as i32;
        let exponent = (bits - 0x3f2aaaab) >> 23;
        let x = f32::from_bits((bits - (exponent << 23)) as u32) - 1.0;
        let numerator = f32::from_bits(0x3f3e11c7)
            .mul_add(x, f32::from_bits(0x3fb6e02b))
            .mul_add(x, f32::from_bits(0xb5f85ab0));
        let denominator = f32::from_bits(0x3e32458c)
            .mul_add(x, f32::from_bits(0x3f813ced))
            .mul_add(x, f32::from_bits(0x3f7d8625));
        numerator / denominator + exponent as f32
    }

    // Independent scalar expression and lane schedule from frozen enc_ma.cc
    // and Highway generic_ops-inl.h; compare each backend to its own profile.
    fn default_oracle_term(count: u32, total: u32, fused: bool) -> f32 {
        if count == total {
            return 0.0;
        }
        let p = (count as f32 * (1.0 / total as f32)).max(1.0 / 4096.0);
        if fused {
            return count as f32 * fma_log_oracle(p);
        }
        let bits = p.to_bits() as i32;
        let exponent = (bits - 0x3f2aaaab) >> 23;
        let x = f32::from_bits((bits - (exponent << 23)) as u32) - 1.0;
        let numerator = LOG_P[0] + x * (LOG_P[1] + x * LOG_P[2]);
        let denominator = LOG_Q[0] + x * (LOG_Q[1] + x * LOG_Q[2]);
        count as f32 * (exponent as f32 + numerator / denominator)
    }

    fn default_oracle(counts: &[u32], total: u32, width: usize, fused: bool) -> f32 {
        let mut sums = [0.0f32; 8];
        for (i, &count) in counts.iter().enumerate() {
            sums[i % width] -= default_oracle_term(count, total, fused);
        }
        oracle_reduce(sums, width)
    }

    fn oracle_reduce(mut sums: [f32; 8], width: usize) -> f32 {
        if width == 1 {
            return sums[0];
        }
        if width == 8 {
            for i in 0..4 {
                sums[i] += sums[i + 4];
            }
        }
        (sums[0] + sums[3]) + (sums[1] + sums[2])
    }

    fn default_profiles() -> Vec<(DefaultScorer, usize)> {
        #[allow(unused_mut)]
        let mut profiles = vec![(DefaultScorer::with_backend(DefaultBackend::Scalar), 1)];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("sse2") {
                profiles.push((DefaultScorer::with_backend(DefaultBackend::Sse2), 4));
            }
            if std::is_x86_feature_detected!("avx2") {
                profiles.push((DefaultScorer::with_backend(DefaultBackend::Avx2), 8));
                if std::is_x86_feature_detected!("fma") {
                    profiles.push((DefaultScorer::with_backend(DefaultBackend::Avx2Fma), 8));
                    if DefaultBackend::detect(true) == DefaultBackend::Avx512Fma8 {
                        profiles.push((DefaultScorer::with_backend(DefaultBackend::Avx512Fma8), 8));
                    }
                }
            }
        }
        profiles
    }

    #[test]
    fn fma_horner_matches_independent_f32_and_native_lane_oracles() {
        let mut values = Vec::new();
        for exponent in -12..=0 {
            for step in 0..=1024 {
                values.push((1.0 + step as f32 / 1024.0) * 2.0_f32.powi(exponent));
            }
        }
        // Either side of the range-reduction breakpoint, plus exact one.
        values.extend([0x3f2aaaaa, 0x3f2aaaab, 0x3f2aaaac, 0x3f800000].map(f32::from_bits));
        let mut differences = 0;
        for &value in &values {
            let expected = fma_log_oracle(value);
            assert_eq!(fast_log2::<true>(value).to_bits(), expected.to_bits());
            differences += usize::from(expected.to_bits() != fast_log2::<false>(value).to_bits());
            assert!((f64::from(expected) - f64::from(value).log2()).abs() <= 0.000005);
        }
        assert!(differences > 0, "fixture must distinguish FMA from non-FMA");

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
            for chunk in values.chunks(8) {
                let mut input = [1.0; 8];
                input[..chunk.len()].copy_from_slice(chunk);
                let mut fused = [0.0; 8];
                let mut unfused = [0.0; 8];
                // SAFETY: both CPU/OS features detected; all accesses span
                // exactly eight initialized/writable f32 lanes.
                unsafe {
                    let lanes = _mm256_loadu_ps(input.as_ptr());
                    _mm256_storeu_ps(fused.as_mut_ptr(), x86::avx2_fma::fast_log(lanes));
                    _mm256_storeu_ps(unfused.as_mut_ptr(), x86::avx2::fast_log(lanes));
                }
                for ((&value, actual), non_fma) in input.iter().zip(fused).zip(unfused) {
                    assert_eq!(actual.to_bits(), fma_log_oracle(value).to_bits());
                    assert_eq!(non_fma.to_bits(), fast_log2::<false>(value).to_bits());
                }
            }
        }
    }

    #[test]
    fn default_runtime_dispatch_and_profile_specific_tables() {
        let expected = DefaultBackend::Scalar;
        #[cfg(target_arch = "x86_64")]
        let expected = if std::is_x86_feature_detected!("avx2") {
            if std::is_x86_feature_detected!("fma") {
                if std::env::var("RS_JXL_MA_RCP").as_deref() == Ok("1") {
                    DefaultBackend::Avx2Rcp
                } else if std::env::var("RS_JXL_MA_AVX512").as_deref() == Ok("1")
                    && std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512dq")
                {
                    DefaultBackend::Avx512Fma8
                } else {
                    DefaultBackend::Avx2Fma
                }
            } else {
                DefaultBackend::Avx2
            }
        } else if std::is_x86_feature_detected!("sse2") {
            DefaultBackend::Sse2
        } else {
            expected
        };
        assert_eq!(DefaultScorer::new().backend, expected);
        let unfused = DefaultScorer::with_backend(DefaultBackend::Scalar);
        for (profile, _) in default_profiles() {
            let again = DefaultScorer::with_backend(profile.backend);
            assert!(std::ptr::eq(profile.small, again.small));
            if profile.backend.uses_fma() {
                assert!(!std::ptr::eq(profile.small, unfused.small));
                assert!(
                    profile
                        .small
                        .iter()
                        .zip(unfused.small)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                );
            } else {
                assert!(std::ptr::eq(profile.small, unfused.small));
            }
        }
    }

    #[test]
    fn fma_scoring_preserves_unfused_entropy_accumulation_and_lane_reduction() {
        let mut seed = 0xb866_b3de_u32;
        let (mut contraction_differs, mut serial_differs) = (false, false);
        for _ in 0..32 {
            let counts: Vec<u32> = (0..257)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    1 + (seed >> 8) % 50000
                })
                .collect();
            let total = counts.iter().sum::<u32>();
            let expected = default_oracle(&counts, total, 8, true);
            let mut contracted = [0.0f32; 8];
            for (i, &count) in counts.iter().enumerate() {
                let probability = (count as f32 * (1.0 / total as f32)).max(1.0 / 4096.0);
                // Deliberately WRONG for this source: catches accidental
                // NegMulAdd contraction of Sub(accumulator, Mul(count, log)).
                contracted[i % 8] =
                    (-(count as f32)).mul_add(fma_log_oracle(probability), contracted[i % 8]);
            }
            contraction_differs |= oracle_reduce(contracted, 8).to_bits() != expected.to_bits();
            serial_differs |=
                default_oracle(&counts, total, 1, true).to_bits() != expected.to_bits();
            for (scorer, width) in default_profiles() {
                assert_eq!(
                    scorer.estimate_bits(&counts, total).to_bits(),
                    default_oracle(&counts, total, width, scorer.backend.uses_fma()).to_bits()
                );
            }
        }
        assert!(
            contraction_differs,
            "fixture must reject fused accumulation"
        );
        assert!(serial_differs, "fixture must reject serial lane reduction");
    }

    fn check_default_move(above: &[u32], below: &[u32], delta: &[u32]) {
        let expected_r: Vec<u32> = above.iter().zip(delta).map(|(&r, &d)| r - d).collect();
        let expected_l: Vec<u32> = below.iter().zip(delta).map(|(&l, &d)| l + d).collect();
        let totals = (expected_r.iter().sum(), expected_l.iter().sum());
        for offset in 0..8 {
            let end = offset + delta.len();
            let mut cached = vec![u32::MAX; end + 1];
            cached[offset..end].copy_from_slice(delta);
            let snapshot = cached.clone();
            // All profiles reuse this same immutable, possibly unaligned cache.
            for (scorer, width) in default_profiles() {
                let mut r = vec![u32::MAX; end + 1];
                let mut l = vec![u32::MAX; end + 1];
                r[offset..end].copy_from_slice(above);
                l[offset..end].copy_from_slice(below);
                let actual = scorer.move_and_score(
                    &mut r[offset..end],
                    &mut l[offset..end],
                    &cached[offset..end],
                    totals,
                );
                assert_eq!(&r[offset..end], &expected_r);
                assert_eq!(&l[offset..end], &expected_l);
                assert_eq!(cached, snapshot);
                for (bits, counts, total) in [
                    (actual.0, expected_r.as_slice(), totals.0),
                    (actual.1, expected_l.as_slice(), totals.1),
                ] {
                    assert!(bits.is_finite() && bits >= 0.0);
                    assert_eq!(
                        bits.to_bits(),
                        default_oracle(counts, total, width, scorer.backend.uses_fma()).to_bits()
                    );
                    assert_eq!(
                        bits.to_bits(),
                        scorer.estimate_bits(counts, total).to_bits()
                    );
                    close(bits, default_oracle(counts, total, 1, false));
                }
                for storage in [&r, &l] {
                    assert!(storage[..offset].iter().all(|&c| c == u32::MAX));
                    assert_eq!(storage[end], u32::MAX);
                }
            }
        }
    }

    #[test]
    fn sparse_blocks_match_dense_native_profiles_and_reject_invalid_addresses() {
        for (scorer, lanes) in default_profiles() {
            for len in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 255, 256] {
                for scale in [1, 65535] {
                    let parent: Vec<u32> = (0..len)
                        .map(|i| {
                            if i % 23 == 0 || i + 1 == len {
                                scale
                            } else {
                                0
                            }
                        })
                        .collect();
                    let blocks: Vec<_> = parent
                        .chunks(lanes)
                        .enumerate()
                        .filter_map(|(i, b)| b.iter().any(|&n| n != 0).then_some(i * lanes))
                        .collect();
                    let delta: Vec<_> = parent.iter().map(|&n| n / 2).collect();
                    let total: u32 = parent.iter().sum();
                    let mut sparse_r = parent.clone();
                    let mut sparse_l = vec![0; len];
                    let mut dense_r = parent.clone();
                    let mut dense_l = vec![0; len];
                    let mut moved = 0;
                    for _ in 0..2 {
                        moved += delta.iter().sum::<u32>();
                        let totals = (total - moved, moved);
                        let a = scorer.move_and_score_sparse(
                            &mut sparse_r,
                            &mut sparse_l,
                            &delta,
                            totals,
                            &blocks,
                        );
                        let b = scorer.move_and_score(&mut dense_r, &mut dense_l, &delta, totals);
                        assert_eq!(
                            (a.0.to_bits(), a.1.to_bits()),
                            (b.0.to_bits(), b.1.to_bits())
                        );
                        assert_eq!(sparse_r, dense_r);
                        assert_eq!(sparse_l, dense_l);
                    }
                }
            }
            for blocks in [vec![lanes], vec![0, 0]] {
                let mut above = vec![1; lanes];
                let mut below = vec![0; lanes];
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scorer
                        .move_and_score_sparse(
                            &mut above,
                            &mut below,
                            &vec![0; lanes],
                            (lanes as u32, 0),
                            &blocks
                        )))
                    .is_err()
                );
                assert_eq!(above, vec![1; lanes]);
                assert_eq!(below, vec![0; lanes]);
            }
        }
    }

    #[test]
    fn default_split_sweep_matches_each_threshold_all_native_profiles() {
        for (scorer, lanes) in default_profiles() {
            for len in [0, 1, 3, 4, 7, 8, 9, 15, 17, 31, 33, 65, 255, 256] {
                for scale in [1u32, 257, 65535, 4_000_000] {
                    for padded in [false, true] {
                        let stride = len.max(1) + if padded { lanes } else { 0 };
                        // Nonadjacent rows and unrelated bin tags exercise packed
                        // layouts. Neighboring storage must stay immutable.
                        let mut storage = vec![0; 1 + 9 * stride + 1];
                        let mut parent = vec![0; len];
                        let mut splits = Vec::new();
                        let mut moved = 0;
                        for row in [0, 2, 5, 8] {
                            let mut weight = 0;
                            for symbol in 0..len {
                                if symbol % 23 == 0 || symbol + 1 == len {
                                    let count = scale + ((symbol + row) % 3) as u32;
                                    storage[1 + row * stride + symbol] = count;
                                    parent[symbol] += count;
                                    weight += count;
                                }
                            }
                            moved += weight;
                            splits.push(HistogramSplit {
                                bin: row * 3 + 7,
                                row,
                                totals: (0, moved),
                            });
                        }
                        let total: u32 = parent.iter().sum();
                        for split in &mut splits {
                            split.totals.0 = total - split.totals.1;
                        }
                        let blocks: Vec<_> = parent
                            .chunks(lanes)
                            .enumerate()
                            .filter_map(|(i, b)| b.iter().any(|&n| n != 0).then_some(i * lanes))
                            .collect();
                        let original = storage.clone();
                        for sparse in [None, Some(blocks.as_slice())] {
                            let mut above = vec![123; len + 2];
                            above[1..len + 1].copy_from_slice(&parent);
                            let mut below = vec![123; len + 2];
                            below[1..len + 1].fill(0);
                            let mut expected_above = parent.clone();
                            let mut expected_below = vec![0; len];
                            let mut expected = Vec::new();
                            for &split in &splits {
                                let start = 1 + split.row * stride;
                                let bits = scorer.move_and_score(
                                    &mut expected_above,
                                    &mut expected_below,
                                    &storage[start..start + len],
                                    split.totals,
                                );
                                expected.push((split.bin, bits.0.to_bits(), bits.1.to_bits()));
                            }
                            let mut actual = Vec::new();
                            scorer.split_sweep(
                                &mut above[1..len + 1],
                                &mut below[1..len + 1],
                                &storage[1..1 + 9 * stride],
                                stride,
                                total,
                                sparse,
                                splits.iter().copied(),
                                |split, r, l| actual.push((split.bin, r.to_bits(), l.to_bits())),
                            );
                            assert_eq!(
                                actual, expected,
                                "backend={:?}, len={len}, scale={scale}",
                                scorer.backend
                            );
                            assert_eq!(&above[1..len + 1], expected_above);
                            assert_eq!(&below[1..len + 1], expected_below);
                            assert_eq!(
                                (above[0], above[len + 1], below[0], below[len + 1]),
                                (123, 123, 123, 123)
                            );
                            assert_eq!(storage, original);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn default_small_sweep_matches_weighted_moves_and_table_boundaries() {
        for (scorer, lanes) in default_profiles() {
            for len in [1, 3, 4, 7, 8, 9, 15, 16, 17, 33] {
                for total in (0u32..=257).chain([511, 512, 513, 65535, i32::MAX as u32]) {
                    let mut parent = vec![0; len];
                    let last = if total % 2 == 0 && len > lanes {
                        len / lanes * lanes - 1
                    } else {
                        len - 1
                    };
                    parent[0] += total / 7;
                    parent[len / 2] += total / 3;
                    parent[last] += total - total / 7 - total / 3;
                    let initial_below: Vec<_> = parent.iter().map(|&n| n / 3).collect();
                    let initial_above: Vec<_> = parent
                        .iter()
                        .zip(&initial_below)
                        .map(|(&n, &l)| n - l)
                        .collect();
                    let stride = len + if total % 2 == 0 { lanes } else { 0 };
                    let mut storage = vec![u32::MAX; 1 + 9 * stride + 1];
                    let mut remaining = initial_above.clone();
                    let mut splits = Vec::new();
                    // Packed rows need not follow the stable threshold order.
                    for (step, row) in [5, 0, 7, 2, 8].into_iter().enumerate() {
                        let start = 1 + row * stride;
                        for (i, r) in remaining.iter_mut().enumerate() {
                            let d = *r / (5 - step) as u32;
                            storage[start + i] = d;
                            *r -= d;
                        }
                        let right = remaining.iter().sum::<u32>();
                        splits.push(HistogramSplit {
                            bin: step * 17 + 3,
                            row,
                            totals: (right, total - right),
                        });
                    }
                    let blocks: Vec<_> = parent
                        .chunks(lanes)
                        .enumerate()
                        .filter_map(|(i, b)| b.iter().any(|&n| n != 0).then_some(i * lanes))
                        .collect();
                    let snapshot = storage.clone();
                    let mut expected_above = initial_above.clone();
                    let mut expected_below = initial_below.clone();
                    let mut expected = Vec::new();
                    for &split in &splits {
                        let start = 1 + split.row * stride;
                        let bits = scorer.move_and_score(
                            &mut expected_above,
                            &mut expected_below,
                            &storage[start..start + len],
                            split.totals,
                        );
                        expected.push((split.bin, split.row, bits.0.to_bits(), bits.1.to_bits()));
                    }
                    for sparse in [None, Some(blocks.as_slice())] {
                        let offset = 1 + len % lanes;
                        let end = offset + len;
                        let mut above = vec![u32::MAX; end + 1];
                        let mut below = vec![u32::MAX; end + 1];
                        above[offset..end].copy_from_slice(&initial_above);
                        below[offset..end].copy_from_slice(&initial_below);
                        let mut actual = Vec::new();
                        scorer.split_sweep(
                            &mut above[offset..end],
                            &mut below[offset..end],
                            &storage[1..1 + 9 * stride],
                            stride,
                            total,
                            sparse,
                            splits.iter().copied(),
                            |split, r, l| {
                                actual.push((split.bin, split.row, r.to_bits(), l.to_bits()));
                            },
                        );
                        assert_eq!(
                            actual, expected,
                            "backend={:?}, len={len}, total={total}, sparse={sparse:?}",
                            scorer.backend
                        );
                        assert_eq!(&above[offset..end], expected_above);
                        assert_eq!(&below[offset..end], expected_below);
                        assert_eq!(below[offset..end].iter().sum::<u32>(), total);
                        for counts in [&above, &below] {
                            assert!(counts[..offset].iter().all(|&n| n == u32::MAX));
                            assert_eq!(counts[end], u32::MAX);
                        }
                        assert_eq!(storage, snapshot);
                    }
                }
            }
        }
    }

    #[test]
    fn default_small_sweep_requires_combined_mass_to_fit_table() {
        for (scorer, _) in default_profiles() {
            let mut above = [128; 4];
            let mut below = [0; 4];
            let mut expected_above = above;
            let mut expected_below = below;
            let splits = [
                HistogramSplit {
                    bin: 3,
                    row: 0,
                    totals: (256, 256),
                },
                HistogramSplit {
                    bin: 7,
                    row: 1,
                    totals: (0, 512),
                },
            ];
            let mut emitted = 0;
            scorer.split_sweep(
                &mut above,
                &mut below,
                &[64; 8],
                4,
                512,
                None,
                splits.into_iter(),
                |split, r, l| {
                    let expected = scorer.move_and_score(
                        &mut expected_above,
                        &mut expected_below,
                        &[64; 4],
                        split.totals,
                    );
                    assert_eq!(r.to_bits(), expected.0.to_bits());
                    assert_eq!(l.to_bits(), expected.1.to_bits());
                    emitted += 1;
                },
            );
            assert_eq!(emitted, 2);
            assert_eq!(above, expected_above);
            assert_eq!(below, expected_below);
        }
    }

    #[test]
    fn default_small_sweep_empty_and_unwind_preserve_completed_moves() {
        for (scorer, _) in default_profiles() {
            for callback_panics in [false, true] {
                let mut above = [9; 9];
                let mut below = [0; 9];
                scorer.split_sweep(
                    &mut above,
                    &mut below,
                    &[1; 18],
                    9,
                    81,
                    None,
                    std::iter::empty(),
                    |_, _, _| panic!("empty sweep emitted a split"),
                );
                assert_eq!(above, [9; 9]);
                assert_eq!(below, [0; 9]);
                let splits = [
                    HistogramSplit {
                        bin: 3,
                        row: 0,
                        totals: (72, 9),
                    },
                    HistogramSplit {
                        bin: 7,
                        row: usize::MAX,
                        totals: (63, 18),
                    },
                ];
                let mut emitted = 0;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    scorer.split_sweep(
                        &mut above,
                        &mut below,
                        &[1; 18],
                        9,
                        81,
                        None,
                        splits.into_iter(),
                        |_, _, _| {
                            emitted += 1;
                            assert!(!callback_panics, "injected callback panic");
                        },
                    );
                }));
                assert!(result.is_err());
                assert_eq!(emitted, 1);
                assert_eq!(above, [8; 9]);
                assert_eq!(below, [1; 9]);
            }
        }
    }

    #[test]
    fn default_small_sweep_changed_totals_keep_general_fallback() {
        for (scorer, lanes) in default_profiles() {
            let mut above = [9; 9];
            let mut below = [0; 9];
            let mut expected_above = above;
            let mut expected_below = below;
            let blocks: Vec<_> = (0..above.len()).step_by(lanes).collect();
            let totals = [(72, 9), (257, 18), (54, 257), (256, 256), (36, 45)];
            let mut emitted = 0;
            scorer.split_sweep(
                &mut above,
                &mut below,
                &[1; 45],
                9,
                81,
                Some(&blocks),
                totals
                    .into_iter()
                    .enumerate()
                    .map(|(row, totals)| HistogramSplit {
                        bin: row * 3,
                        row,
                        totals,
                    }),
                |split, r, l| {
                    let expected = scorer.move_and_score(
                        &mut expected_above,
                        &mut expected_below,
                        &[1; 9],
                        split.totals,
                    );
                    assert_eq!(r.to_bits(), expected.0.to_bits());
                    assert_eq!(l.to_bits(), expected.1.to_bits());
                    emitted += 1;
                },
            );
            assert_eq!(emitted, totals.len());
            assert_eq!(above, expected_above);
            assert_eq!(below, expected_below);
        }
    }

    #[test]
    fn default_split_sweep_rejects_invalid_layout_before_mutation() {
        for (scorer, lanes) in default_profiles() {
            let len = lanes * 2;
            for (stride, cells, below_len, row, blocks) in [
                (0, len, len, 0, None),
                (len - 1, len - 1, len, 0, None),
                (len, len + 1, len, 0, None),
                (len, len, len - 1, 0, None),
                (len, len, len, usize::MAX, None),
                (len, len, len, 0, Some(vec![len])),
                (len, len, len, 0, Some(vec![0, 0])),
                (len, len, len, 0, Some(vec![lanes, 0])),
            ] {
                let mut above = vec![1; len];
                let mut below = vec![0; below_len];
                let deltas = vec![0; cells];
                let mut emitted = false;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    scorer.split_sweep(
                        &mut above,
                        &mut below,
                        &deltas,
                        stride,
                        len as u32,
                        blocks.as_deref(),
                        std::iter::once(HistogramSplit {
                            bin: 0,
                            row,
                            totals: (len as u32, 0),
                        }),
                        |_, _, _| emitted = true,
                    );
                }));
                assert!(result.is_err());
                assert_eq!(above, vec![1; len]);
                assert_eq!(below, vec![0; below_len]);
                assert!(!emitted);
            }
        }
    }

    #[test]
    fn default_small_table_is_shared_source_expression_not_log_cache() {
        let scorer = DefaultScorer::default();
        assert!(std::ptr::eq(scorer.small, DefaultScorer::new().small));
        assert_eq!(std::mem::size_of_val(scorer.small), 132612);
        assert_eq!(scorer.small_row(257), &[]);
        for (profile, width) in default_profiles() {
            assert_eq!(profile.lanes(), width);
            for total in 0..=256 {
                let row = profile.small_row(total);
                assert_eq!(row.len(), total as usize + 1);
                for count in 0..=total {
                    assert_eq!(
                        row[count as usize].to_bits(),
                        default_oracle_term(count, total, profile.backend.uses_fma()).to_bits(),
                        "backend={:?}, count={count}, total={total}",
                        profile.backend
                    );
                }
            }
            if profile.backend == scorer.backend {
                let counts = [1, 9, 0, 3, 0, 0, 8, 1, 17];
                assert_eq!(
                    scorer.estimate_bits(&counts, 39).to_bits(),
                    default_oracle(&counts, 39, width, scorer.backend.uses_fma()).to_bits()
                );
            }
        }
    }

    #[test]
    fn default_weighted_moves_immutable_cache_unaligned_tails() {
        let mut state = 0xb866_b3de_u32;
        for len in (0..=33).chain([63, 64, 65, 127, 128, 129]) {
            let mut above = vec![0; len];
            let mut below = vec![0; len];
            let mut delta = vec![0; len];
            for i in 0..len {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                above[i] = state % 300;
                below[i] = (state >> 8) % 90;
                delta[i] = (state >> 16) % (above[i] + 1);
            }
            check_default_move(&above, &below, &delta);
            check_default_move(&above, &vec![0; len], &above);
            check_default_move(&above, &vec![0; len], &vec![0; len]);
        }
    }

    #[test]
    fn default_masked_tails_cover_each_lane_and_independent_table_routes() {
        // Put mass in both ends of the partial vector, not just lane zero.
        // check_default_move also surrounds every unaligned slice with canaries.
        for len in (1..=16).chain(25..=31) {
            for (right, left) in [
                (0, 0),
                (1, 0),
                (256, 256),
                (256, 257),
                (257, 256),
                (4097, 65535),
                ((1 << 24) + 1, 257),
                (MAX_HISTOGRAM_TOTAL - 257, 257),
            ] {
                let tail = len / 8 * 8;
                let first = if tail == len { len - 1 } else { tail };
                let mut r = vec![0; len];
                let mut l = vec![0; len];
                r[first] = right / 3;
                r[len - 1] += right - right / 3;
                l[first] = left / 2;
                l[len - 1] += left - left / 2;
                let delta: Vec<_> = l.iter().map(|&n| n / 2).collect();
                let above: Vec<_> = r.iter().zip(&delta).map(|(&r, &d)| r + d).collect();
                let below: Vec<_> = l.iter().zip(&delta).map(|(&l, &d)| l - d).collect();
                check_default_move(&above, &below, &delta);
            }
        }
    }

    #[test]
    fn default_masked_tails_match_full_rescans_through_weighted_split_sweeps() {
        // Each row is an immutable cached property bin. Include empty rows and
        // saturated sample weights, then reuse the same cache for another sweep.
        for symbols in 1..=33 {
            let rows: Vec<Vec<u32>> = (0..37)
                .map(|bin| {
                    (0..symbols)
                        .map(|symbol| {
                            if bin % 7 == 0 || (bin + symbol) % 5 != 0 {
                                0
                            } else if bin == 18 {
                                65535
                            } else {
                                1 + (bin * 13 + symbol * 19) as u32 % 31
                            }
                        })
                        .collect()
                })
                .collect();
            let snapshot = rows.clone();
            let parent: Vec<u32> = (0..symbols)
                .map(|symbol| rows.iter().map(|row| row[symbol]).sum())
                .collect();
            for (scorer, width) in default_profiles() {
                for _ in 0..2 {
                    // Exact-sized slices exercise kernels without padding.
                    let mut above = parent.clone().into_boxed_slice();
                    let mut below = vec![0; symbols].into_boxed_slice();
                    for bin in 0..rows.len() {
                        let expected_l: Vec<u32> = (0..symbols)
                            .map(|s| rows[..=bin].iter().map(|row| row[s]).sum())
                            .collect();
                        let expected_r: Vec<u32> = (0..symbols)
                            .map(|s| rows[bin + 1..].iter().map(|row| row[s]).sum())
                            .collect();
                        let totals = (expected_r.iter().sum(), expected_l.iter().sum());
                        let actual =
                            scorer.move_and_score(&mut above, &mut below, &rows[bin], totals);
                        assert_eq!(&*above, expected_r.as_slice());
                        assert_eq!(&*below, expected_l.as_slice());
                        for (bits, counts, total) in [
                            (actual.0, expected_r.as_slice(), totals.0),
                            (actual.1, expected_l.as_slice(), totals.1),
                        ] {
                            assert_eq!(
                                bits.to_bits(),
                                default_oracle(counts, total, width, scorer.backend.uses_fma())
                                    .to_bits(),
                                "symbols={symbols}, bin={bin}, backend={:?}",
                                scorer.backend,
                            );
                        }
                    }
                }
            }
            assert_eq!(rows, snapshot);
        }
    }

    #[test]
    fn default_masked_sparse_tail_is_optional_and_preserves_neighboring_storage() {
        for (scorer, lanes) in default_profiles() {
            for tail in 1..lanes {
                let len = 2 * lanes + tail;
                for include_tail in [false, true] {
                    let mut parent = vec![0; len];
                    parent[0] = 65535;
                    let mut blocks = vec![0];
                    if include_tail {
                        parent[len - 1] = 257;
                        blocks.push(2 * lanes);
                    }
                    let delta: Vec<_> = parent.iter().map(|&n| n / 2).collect();
                    let mut above = parent.clone();
                    let mut below = vec![0; len];
                    // The following elements are outside the borrowed histogram.
                    above.extend([u32::MAX; 8]);
                    below.extend([u32::MAX; 8]);
                    let moved = delta.iter().sum::<u32>();
                    let totals = (parent.iter().sum::<u32>() - moved, moved);
                    let actual = scorer.move_and_score_sparse(
                        &mut above[..len],
                        &mut below[..len],
                        &delta,
                        totals,
                        &blocks,
                    );
                    assert_eq!(&above[len..], &[u32::MAX; 8]);
                    assert_eq!(&below[len..], &[u32::MAX; 8]);
                    for i in 0..len {
                        assert_eq!(above[i], parent[i] - delta[i]);
                        assert_eq!(below[i], delta[i]);
                    }
                    for (bits, counts, total) in [
                        (actual.0, &above[..len], totals.0),
                        (actual.1, &below[..len], totals.1),
                    ] {
                        assert_eq!(
                            bits.to_bits(),
                            default_oracle(counts, total, lanes, scorer.backend.uses_fma())
                                .to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn default_small_table_boundary_is_independent_per_side() {
        for total_r in [0, 1, 255, 256, 257, 4096, 4097] {
            for total_l in [0, 1, 255, 256, 257, 4096, 4097] {
                let mut after_r = [0; 9];
                let mut after_l = [0; 9];
                after_r[0] = total_r / 2;
                after_r[8] = total_r - after_r[0];
                after_l[0] = total_l / 3;
                after_l[8] = total_l - after_l[0];
                let delta = after_l.map(|c| c / 2);
                let above = std::array::from_fn::<_, 9, _>(|i| after_r[i] + delta[i]);
                let below = std::array::from_fn::<_, 9, _>(|i| after_l[i] - delta[i]);
                check_default_move(&above, &below, &delta);
            }
        }
    }

    #[test]
    fn default_floor_4096_and_integer_mask_near_float_precision_limit() {
        for total in [
            4095,
            4096,
            4097,
            8191,
            8192,
            8193,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            MAX_HISTOGRAM_TOTAL,
        ] {
            for rare in [1, 2, 7] {
                let mut counts = [0; 17];
                counts[0] = rare;
                counts[16] = total - rare;
                check_default_move(&counts, &[0; 17], &[0; 17]);
                let mut delta = [0; 17];
                delta[0] = rare;
                delta[16] = 251;
                check_default_move(&counts, &[0; 17], &delta);
            }
            for (scorer, _) in default_profiles() {
                let mut single = [0; 17];
                single[16] = total;
                assert_eq!(scorer.estimate_bits(&single, total), 0.0);
            }
        }
        for (scorer, width) in default_profiles() {
            // Adjacent integers can convert to equal f32s, but must not take
            // the exact-integer zero mask. FastLog2f(1) is slightly negative.
            let counts = [1 << 24, 1, 0, 0, 0, 0, 0, 0, 0];
            let total = (1 << 24) + 1;
            assert!(scorer.estimate_bits(&counts, total) > 13.0);
            let total = 1 << 20;
            let mut counts = [1; 257];
            counts[256] = total - 256;
            let actual = scorer.estimate_bits(&counts, total);
            assert_eq!(
                actual.to_bits(),
                default_oracle(&counts, total, width, scorer.backend.uses_fma()).to_bits()
            );
            let shannon: f64 = counts
                .iter()
                .map(|&c| f64::from(c) * (f64::from(total) / f64::from(c)).log2())
                .sum();
            assert!(shannon - f64::from(actual) > 2000.0);
        }
    }

    #[test]
    fn default_length_checks_precede_mutation() {
        for (scorer, _) in default_profiles() {
            for (r_len, l_len, d_len) in [(8, 7, 8), (8, 8, 7), (0, 1, 0), (1, 1, 0)] {
                let mut above = vec![2; r_len];
                let mut below = vec![0; l_len];
                let delta = vec![1; d_len];
                let before = (above.clone(), below.clone());
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    scorer.move_and_score(&mut above, &mut below, &delta, (8, 8));
                }));
                assert!(result.is_err());
                assert_eq!((above, below), before);
                assert_eq!(delta, vec![1; d_len]);
            }
        }
    }

    #[test]
    fn count_entropy_cache_and_full_u32_fallback() {
        let table = CountEntropyTable::shared();
        assert!(std::ptr::eq(table, CountEntropyTable::shared()));
        assert_eq!(table.values.len(), MAX_LOG_TABLE_TOTAL as usize + 1);
        for count in (0..=MAX_LOG_TABLE_TOTAL).chain([
            65537,
            1 << 20,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            i32::MAX as u32,
            i32::MAX as u32 + 1,
            u32::MAX,
        ]) {
            assert_eq!(table.get(count), uncached_count_entropy(count));
        }
    }

    #[test]
    fn count_entropy_native_sums_cover_tails_and_large_unsigned_counts() {
        let table = CountEntropyTable::shared();
        let values = [0, 1, 2, 3, 65535, 65536, 65537, 1 << 24, u32::MAX];
        for len in 0..=137 {
            for offset in 0..4 {
                let mut storage = vec![u32::MAX; offset + len];
                for (i, count) in storage[offset..].iter_mut().enumerate() {
                    *count = values[(i + len) % values.len()];
                }
                let counts = &storage[offset..];
                let expected: f64 = counts.iter().map(|&c| uncached_count_entropy(c)).sum();
                let check = |actual: f64| {
                    assert!(actual.is_finite());
                    assert!((actual - expected).abs() <= 1e-9 + expected * 64.0 * f64::EPSILON);
                };
                check(table.sum(counts));
                check(table.sum_scalar(counts));
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("sse2") {
                        // SAFETY: feature detected; backend accepts all lengths.
                        check(unsafe { count_x86::sum_sse2(table, counts) });
                    }
                    if std::is_x86_feature_detected!("avx2") {
                        // SAFETY: feature detected, including OS AVX support.
                        check(unsafe { count_x86::sum_avx2(table, counts) });
                    }
                }
            }
        }
    }

    #[test]
    fn count_entropy_native_pair_updates_match_scalar_operations() {
        let table = CountEntropyTable::shared();
        let edges = [0, 1, 4096, 65535, 65536, 65537, i32::MAX as u32, u32::MAX];
        for &r in &edges {
            for &l in &edges {
                for weight in [0, 1, 997, 65536, u32::MAX] {
                    let weight = weight.min(r).min(u32::MAX - l);
                    let before = [r, l];
                    let after = [r - weight, l + weight];
                    let sums = [table.get(r) + 19.0, table.get(l) + 23.0];
                    let expected = table.replace_pair_scalar(before, after, sums);
                    let check = |actual: CountEntropyUpdate| {
                        assert_eq!(actual.before, expected.before);
                        assert_eq!(actual.after, expected.after);
                        assert_eq!(actual.sums, expected.sums);
                    };
                    check(table.replace_pair(before, after, sums));
                    #[cfg(target_arch = "x86_64")]
                    {
                        if std::is_x86_feature_detected!("sse2") {
                            // SAFETY: feature detected; fixed arrays, full u32 inputs.
                            check(unsafe { count_x86::pair_sse2(table, before, after, sums) });
                        }
                        if std::is_x86_feature_detected!("avx2") {
                            // SAFETY: feature detected; gather bounds checked inside.
                            check(unsafe { count_x86::pair_avx2(table, before, after, sums) });
                        }
                    }
                }
            }
        }
    }

    fn close(actual: f32, expected: f32) {
        assert!(actual.is_finite() && actual >= 0.0);
        assert!(
            (actual - expected).abs() <= 0.00001 * expected.max(1.0),
            "{actual} != {expected}"
        );
    }

    fn check_estimate(table: &LogTable, counts: &[u32]) {
        let total = table.total(counts).unwrap();
        let expected = table.estimate_bits_scalar(counts).unwrap();
        // Independent f64 mathematical probability-floor oracle.
        let model: f64 = counts
            .iter()
            .filter(|&&c| c != 0)
            .map(|&c| f64::from(c) * (f64::from(total) / f64::from(c)).log2().min(12.0))
            .sum();
        // The f32 log table loses absolute precision near log2(65536).
        // Multiplying that error by a large count is not a relative-cost error.
        let per_sample_error = if total <= table.maximum_total() {
            2.0_f64.powi(-19)
        } else {
            0.000008
        };
        let bound = f64::from(total) * per_sample_error + model.max(1.0) * 0.00001;
        assert!(
            (f64::from(expected) - model).abs() <= bound,
            "{expected} != {model}"
        );
        close(table.estimate_bits(counts).unwrap(), expected);
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("sse2") {
                // SAFETY: detected CPU and validated counts/table total.
                close(
                    unsafe { x86::sse2::estimate(table, counts, total) },
                    expected,
                );
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: detected CPU and validated counts/table total.
                close(
                    unsafe { x86::avx2::estimate(table, counts, total) },
                    expected,
                );
            }
        }
    }

    fn check_move(table: &LogTable, above: &[u32], below: &[u32], increase: &[u32]) {
        let totals = table.validate_move(above, below, increase).unwrap();
        let expected_r: Vec<_> = above.iter().zip(increase).map(|(&a, &d)| a - d).collect();
        let expected_l: Vec<_> = below.iter().zip(increase).map(|(&a, &d)| a + d).collect();
        let bits = (
            table.estimate_bits_scalar(&expected_r).unwrap(),
            table.estimate_bits_scalar(&expected_l).unwrap(),
        );
        for backend in 0..4 {
            let offset = (above.len() + backend) % 8;
            let end = offset + above.len();
            let mut rs = vec![u32::MAX; end + 1];
            let mut ls = vec![u32::MAX; end + 1];
            let mut ds = vec![u32::MAX; end + 1];
            let r = &mut rs[offset..end];
            let l = &mut ls[offset..end];
            let d = &mut ds[offset..end];
            r.copy_from_slice(above);
            l.copy_from_slice(below);
            d.copy_from_slice(increase);
            let actual = match backend {
                0 => table.move_scalar_validated(r, l, d, totals),
                1 => table.move_counts_and_log_bits(r, l, d).unwrap(),
                #[cfg(target_arch = "x86_64")]
                2 if std::is_x86_feature_detected!("sse2") => {
                    // SAFETY: detected CPU, validated matching slices/totals.
                    unsafe { x86::sse2::move_and_score(table, r, l, d, totals) }
                }
                #[cfg(target_arch = "x86_64")]
                3 if std::is_x86_feature_detected!("avx2") => {
                    // SAFETY: detected CPU, validated matching slices/totals.
                    unsafe { x86::avx2::move_and_score(table, r, l, d, totals) }
                }
                _ => continue,
            };
            assert_eq!(r, expected_r.as_slice());
            assert_eq!(l, expected_l.as_slice());
            assert!(d.iter().all(|&c| c == 0));
            close(actual.0, bits.0);
            close(actual.1, bits.1);
            for storage in [&rs, &ls, &ds] {
                assert!(storage[..offset].iter().all(|&v| v == u32::MAX));
                assert_eq!(storage[end], u32::MAX);
            }
        }
    }

    #[test]
    fn lengths_tails_unaligned_counts_and_weighted_transfers() {
        let table = LogTable::new(MAX_LOG_TABLE_TOTAL).unwrap();
        let mut state = 0xb866_b3de_u32;
        for len in (0..=35).chain([63, 64, 65, 127, 128, 129, 255, 256, 257]) {
            for offset in 0..8 {
                let mut storage = vec![u32::MAX; offset + len];
                let above = &mut storage[offset..];
                let mut below = vec![0; len];
                let mut increase = vec![0; len];
                for i in 0..len {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    above[i] = state % 128;
                    below[i] = (state >> 8) % 64;
                    increase[i] = (state >> 16) % (above[i] + 1);
                }
                check_estimate(&table, above);
                check_move(&table, above, &below, &increase);
                above.fill(0);
                check_estimate(&table, above);
                if len > 0 {
                    above[len - 1] = MAX_LOG_TABLE_TOTAL;
                    check_estimate(&table, above);
                    assert_eq!(table.estimate_bits(above).unwrap(), 0.0);
                    below.fill(0);
                    increase.copy_from_slice(above);
                    check_move(&table, above, &below, &increase);
                }
            }
        }
    }

    #[test]
    fn probability_floor_boundary_and_empty_sides() {
        let table = LogTable::new(MAX_LOG_TABLE_TOTAL).unwrap();
        // Pre-move population may exceed the table when both resulting sides
        // fit. Each moved count remains an exact multiplicity.
        check_move(&table, &[65536, 65536], &[0, 0], &[0, 65536]);
        for total in [4095, 4096, 4097, 8192, 65536] {
            for rare in [1, 2, 15, 16, 17] {
                let mut counts = [0; 17];
                counts[0] = rare;
                counts[8] = total - rare;
                check_estimate(&table, &counts);
                check_move(&table, &counts, &[0; 17], &counts);
                check_move(&table, &counts, &[0; 17], &[0; 17]);
            }
        }
        let zero = LogTable::new(0).unwrap();
        check_estimate(&zero, &[]);
        check_estimate(&zero, &[0; 17]);
        check_move(&zero, &[0; 17], &[0; 17], &[0; 17]);
    }

    #[test]
    fn invalid_limits_and_moves_never_partially_mutate() {
        assert!(matches!(
            LogTable::new(65537),
            Err(MaSimdError::TableLimitTooLarge)
        ));
        assert!(matches!(
            LogTable::new(u32::MAX),
            Err(MaSimdError::TableLimitTooLarge)
        ));
        let table = LogTable::new(65536).unwrap();
        for counts in [
            &[MAX_HISTOGRAM_TOTAL + 1][..],
            &[MAX_HISTOGRAM_TOTAL, 1],
            &[u32::MAX, 1],
        ] {
            assert_eq!(table.estimate_bits(counts), Err(MaSimdError::TotalTooLarge));
            assert_eq!(
                table.estimate_bits_scalar(counts),
                Err(MaSimdError::TotalTooLarge)
            );
        }
        for (a, b, d, error) in [
            (
                vec![1; 17],
                vec![0; 16],
                vec![0; 17],
                MaSimdError::LengthMismatch,
            ),
            (
                vec![1; 17],
                vec![0; 17],
                vec![0; 16],
                MaSimdError::LengthMismatch,
            ),
            (
                vec![1; 17],
                vec![0; 17],
                {
                    let mut d = vec![1; 17];
                    d[16] = 2;
                    d
                },
                MaSimdError::CountUnderflow,
            ),
            (vec![1], vec![u32::MAX], vec![1], MaSimdError::CountOverflow),
            (
                vec![MAX_HISTOGRAM_TOTAL, 1],
                vec![0; 2],
                vec![0; 2],
                MaSimdError::TotalTooLarge,
            ),
            (
                vec![1; 17],
                {
                    let mut b = vec![0; 17];
                    b[16] = MAX_HISTOGRAM_TOTAL;
                    b
                },
                vec![1; 17],
                MaSimdError::TotalTooLarge,
            ),
        ] {
            let (mut above, mut below, mut increase) = (a.clone(), b.clone(), d.clone());
            assert_eq!(
                table.move_counts_and_log_bits(&mut above, &mut below, &mut increase),
                Err(error)
            );
            assert_eq!((above, below, increase), (a, b, d));
        }
    }

    #[test]
    fn fast_log_polynomial_matches_source_accuracy_and_clipped_floor() {
        for exponent in -12..=0 {
            for step in 0..=1024 {
                let value = (1.0 + step as f32 / 1024.0) * 2.0_f32.powi(exponent);
                let error = (f64::from(fast_log2::<false>(value)) - f64::from(value).log2()).abs();
                assert!(error <= 0.000005, "{value}, error={error}");
            }
        }
        let total = 1 << 20;
        let clipped_rare = fast_contribution(1, total, inverse_total(total));
        assert!((clipped_rare - 12.0).abs() < 0.00001);
        assert!(
            20.0 - clipped_rare > 7.99,
            "Shannon rare-symbol cost must not replace the floor"
        );
        // FastLog2f(1) is deliberately not zero. The source masks exact
        // INTEGER equality, but must not mask adjacent counts rounded to 1.0.
        assert_ne!(fast_log2::<false>(1.0), 0.0);
        assert_eq!(fast_contribution(total, total, inverse_total(total)), 0.0);
        assert!(fast_contribution(1 << 24, (1 << 24) + 1, inverse_total((1 << 24) + 1)) > 1.0);
    }

    #[test]
    fn cache_boundary_large_counts_and_integer_equality_mask() {
        for cache in [0, 256, 65536] {
            let table = LogTable::new(cache).unwrap();
            for total in [
                65535,
                65536,
                65537,
                1 << 20,
                (1 << 24) - 1,
                1 << 24,
                (1 << 24) + 1,
                MAX_HISTOGRAM_TOTAL,
            ] {
                for rare in [1, 2, 15, 16, 17, 255] {
                    let mut counts = [0; 17];
                    counts[0] = rare;
                    counts[8] = total - rare;
                    check_estimate(&table, &counts);
                    check_move(&table, &counts, &[0; 17], &counts);
                    check_move(&table, &counts, &[0; 17], &[0; 17]);
                }
                for position in 0..17 {
                    let mut counts = [0; 17];
                    counts[position] = total;
                    check_estimate(&table, &counts);
                    assert_eq!(table.estimate_bits(&counts).unwrap(), 0.0);
                }
            }
        }
    }

    #[test]
    fn mixed_cache_fallback_splits_and_large_unaligned_tails() {
        let table = LogTable::new(65536).unwrap();
        // Both directions of the cache boundary and both-fallback splits.
        for (a, b, d) in [
            (131072, 0, 65536),
            (131073, 0, 65536),
            (131073, 0, 65537),
            (262144, 0, 131072),
            (MAX_HISTOGRAM_TOTAL, 0, 65536),
        ] {
            let mut above = [0; 17];
            let mut below = [0; 17];
            let mut increase = [0; 17];
            above[8] = a;
            below[8] = b;
            increase[8] = d;
            check_move(&table, &above, &below, &increase);
        }
        let mut state = 0x19371b_u32;
        for len in (0..=35).chain([63, 64, 65, 127, 128, 129, 257]) {
            for offset in 0..8 {
                let mut storage = vec![u32::MAX; offset + len];
                let above = &mut storage[offset..];
                let mut below = vec![0; len];
                let mut increase = vec![0; len];
                for i in 0..len {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    above[i] = state & 0x3fffff;
                    below[i] = (state >> 8) & 0x1fffff;
                    increase[i] = state % (above[i] + 1);
                }
                check_estimate(&table, above);
                check_move(&table, above, &below, &increase);
            }
        }
    }
    #[test]
    fn count_log_readonly_matches_source_lane_order_and_preserves_cache_deltas() {
        let table = LogTable::shared();
        let width = DefaultScorer::new().lanes();
        for len in [0, 1, 3, 4, 7, 8, 9, 31, 32, 33, 255, 256] {
            let original: Vec<u32> = (0..len).map(|i| (i % 37 + 1) as u32).collect();
            let delta: Vec<_> = original.iter().map(|&n| n / 2).collect();
            let mut above = original.clone();
            let mut below = vec![0; len];
            let totals = (
                original.iter().sum::<u32>() - delta.iter().sum::<u32>(),
                delta.iter().sum(),
            );
            let actual = table.move_known_totals(&mut above, &mut below, &delta, totals);
            for (counts, total, bits) in [
                (above.as_slice(), totals.0, actual.0),
                (below.as_slice(), totals.1, actual.1),
            ] {
                let mut lanes = [0.0; 8];
                for (i, &count) in counts.iter().enumerate() {
                    lanes[i % width] += table.contribution(count, total);
                }
                assert_eq!(bits.to_bits(), oracle_reduce(lanes, width).to_bits());
            }
            assert_eq!(below, delta);
            assert_eq!(
                above
                    .iter()
                    .zip(&below)
                    .map(|(&r, &l)| r + l)
                    .collect::<Vec<_>>(),
                original
            );
        }
        let mut above = [9];
        let mut below = [0];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                table.move_known_totals(&mut above, &mut below, &[], (9, 0));
            }))
            .is_err()
        );
        assert_eq!(above, [9]);
        assert_eq!(below, [0]);
    }
}
