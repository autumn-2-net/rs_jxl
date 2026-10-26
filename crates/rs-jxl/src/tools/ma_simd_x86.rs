// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

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
                        move_dispatch::<SPARSE>(scorer, above, below, delta, split.totals, blocks)
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
                    (true, true) => move_impl::<true, true>(table, above, below, increase, totals),
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
        DefaultScorer, HistogramSplit, LOG_P, LOG_Q, LogTable, MIN_PROBABILITY, PROBABILITY_BITS,
        cached_source_entropy, inverse_total, split_delta,
    };
    use std::arch::x86_64::*;
    use std::arch::x86_64::{
        _mm_add_epi32 as add_i, _mm_add_ps as add, _mm_andnot_ps as andnot_f,
        _mm_castps_si128 as cast_i, _mm_castsi128_ps as cast_f, _mm_cmpeq_epi32 as eq_i,
        _mm_cvtepi32_ps as convert, _mm_div_ps as div, _mm_loadu_si128 as load, _mm_max_ps as max,
        _mm_min_ps as min, _mm_mul_ps as mul, _mm_set1_epi32 as splat_i, _mm_set1_ps as splat,
        _mm_setzero_ps as zero, _mm_setzero_si128 as zero_i, _mm_slli_epi32 as shl_i,
        _mm_srai_epi32 as sar_i, _mm_storeu_ps as store_f, _mm_storeu_si128 as store_i,
        _mm_sub_epi32 as sub_i, _mm_sub_ps as sub,
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
        DefaultScorer, HistogramSplit, LOG_P, LOG_Q, LogTable, MIN_PROBABILITY, PROBABILITY_BITS,
        cached_source_entropy, inverse_total, split_delta,
    };
    use std::arch::x86_64::*;
    use std::arch::x86_64::{
        _mm256_add_epi32 as add_i, _mm256_add_ps as add, _mm256_andnot_ps as andnot_f,
        _mm256_castps_si256 as cast_i, _mm256_castsi256_ps as cast_f, _mm256_cmpeq_epi32 as eq_i,
        _mm256_cvtepi32_ps as convert, _mm256_div_ps as div, _mm256_loadu_si256 as load,
        _mm256_max_ps as max, _mm256_min_ps as min, _mm256_mul_ps as mul,
        _mm256_set1_epi32 as splat_i, _mm256_set1_ps as splat, _mm256_setzero_ps as zero,
        _mm256_setzero_si256 as zero_i, _mm256_slli_epi32 as shl_i, _mm256_srai_epi32 as sar_i,
        _mm256_storeu_ps as store_f, _mm256_storeu_si256 as store_i, _mm256_sub_epi32 as sub_i,
        _mm256_sub_ps as sub,
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
        _mm256_castps_si256 as cast_i, _mm256_castsi256_ps as cast_f, _mm256_cmpeq_epi32 as eq_i,
        _mm256_cvtepi32_ps as convert, _mm256_div_ps as div, _mm256_loadu_si256 as load,
        _mm256_max_ps as max, _mm256_mul_ps as mul, _mm256_set1_epi32 as splat_i,
        _mm256_set1_ps as splat, _mm256_setzero_ps as zero, _mm256_slli_epi32 as shl_i,
        _mm256_srai_epi32 as sar_i, _mm256_storeu_ps as store_f, _mm256_storeu_si256 as store_i,
        _mm256_sub_epi32 as sub_i, _mm256_sub_ps as sub,
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
        _mm256_castps_si256 as cast_i, _mm256_castsi256_ps as cast_f, _mm256_cmpeq_epi32 as eq_i,
        _mm256_cvtepi32_ps as convert, _mm256_loadu_si256 as load, _mm256_max_ps as max,
        _mm256_mul_ps as mul, _mm256_set1_epi32 as splat_i, _mm256_set1_ps as splat,
        _mm256_setzero_ps as zero, _mm256_slli_epi32 as shl_i, _mm256_srai_epi32 as sar_i,
        _mm256_storeu_ps as store_f, _mm256_storeu_si256 as store_i, _mm256_sub_epi32 as sub_i,
        _mm256_sub_ps as sub,
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
