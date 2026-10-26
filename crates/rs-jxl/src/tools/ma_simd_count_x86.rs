// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

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
