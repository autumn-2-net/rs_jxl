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
#[path = "ma_simd_count_x86.rs"]
mod count_x86;

#[cfg(target_arch = "x86_64")]
#[path = "ma_simd_x86.rs"]
mod x86;

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "ma_simd_avx512_tests.rs"]
mod avx512_tests;

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "ma_simd_rcp_tests.rs"]
mod rcp_tests;

#[cfg(test)]
#[path = "ma_simd_tests.rs"]
mod tests;
