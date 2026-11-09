//! Unaligned, bounded token loads. No padding, BMI1, AVX-512VL, or AVX-512DQ.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(test)]
use super::scalar_u16;
use super::scalar_u32;

/// # Safety
/// AVX2 must be available, and both slices must have equal lengths.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn length_u32_avx2(a: &[u32], b: &[u32]) -> usize {
    const LANES: usize = 8;
    let mut offset = 0;
    while a.len() - offset >= LANES {
        // SAFETY: offset <= len and a full vector remains in both equal-length
        // slices. loadu needs no vector alignment; shared slices may overlap.
        let (current, candidate) = unsafe {
            (
                _mm256_loadu_si256(a.as_ptr().add(offset).cast()),
                _mm256_loadu_si256(b.as_ptr().add(offset).cast()),
            )
        };
        let equal = _mm256_cmpeq_epi32(current, candidate);
        let different = !(_mm256_movemask_epi8(equal) as u32);
        if different != 0 {
            // Each unequal token contributes four consecutive mismatch bits.
            return offset + different.trailing_zeros() as usize / 4;
        }
        offset += LANES;
    }
    offset + scalar_u32(&a[offset..], &b[offset..])
}

/// # Safety
/// AVX2 must be available, and both slices must have equal lengths.
#[target_feature(enable = "avx2")]
#[cfg(test)]
pub(super) unsafe fn length_u16_avx2(a: &[u16], b: &[u16]) -> usize {
    const LANES: usize = 16;
    let mut offset = 0;
    while a.len() - offset >= LANES {
        // SAFETY: the loop guard proves a full unaligned vector is in each
        // equal-length slice. Reads through shared slices may overlap.
        let (current, candidate) = unsafe {
            (
                _mm256_loadu_si256(a.as_ptr().add(offset).cast()),
                _mm256_loadu_si256(b.as_ptr().add(offset).cast()),
            )
        };
        let equal = _mm256_cmpeq_epi16(current, candidate);
        let different = !(_mm256_movemask_epi8(equal) as u32);
        if different != 0 {
            return offset + different.trailing_zeros() as usize / 2;
        }
        offset += LANES;
    }
    offset + scalar_u16(&a[offset..], &b[offset..])
}

/// # Safety
/// AVX-512F/BW must be available, and both slices must have equal lengths.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn length_u32_avx512(a: &[u32], b: &[u32]) -> usize {
    const LANES: usize = 16;
    let mut offset = 0;
    while a.len() - offset >= LANES {
        // SAFETY: only full vectors inside both equal-length slices are read.
        // No SIMD alignment or non-overlap is required by loadu.
        let (current, candidate) = unsafe {
            (
                _mm512_loadu_si512(a.as_ptr().add(offset).cast()),
                _mm512_loadu_si512(b.as_ptr().add(offset).cast()),
            )
        };
        // Invert at the native u16 mask width: one bit per u32 token.
        let different = !_mm512_cmpeq_epi32_mask(current, candidate);
        if different != 0 {
            return offset + different.trailing_zeros() as usize;
        }
        offset += LANES;
    }
    offset + scalar_u32(&a[offset..], &b[offset..])
}

/// # Safety
/// AVX-512F/BW must be available, and both slices must have equal lengths.
#[target_feature(enable = "avx512f,avx512bw")]
#[cfg(test)]
pub(super) unsafe fn length_u16_avx512(a: &[u16], b: &[u16]) -> usize {
    const LANES: usize = 32;
    let mut offset = 0;
    while a.len() - offset >= LANES {
        // SAFETY: only full vectors inside both equal-length slices are read.
        // No SIMD alignment or non-overlap is required by loadu.
        let (current, candidate) = unsafe {
            (
                _mm512_loadu_si512(a.as_ptr().add(offset).cast()),
                _mm512_loadu_si512(b.as_ptr().add(offset).cast()),
            )
        };
        let different = !_mm512_cmpeq_epi16_mask(current, candidate);
        if different != 0 {
            return offset + different.trailing_zeros() as usize;
        }
        offset += LANES;
    }
    offset + scalar_u16(&a[offset..], &b[offset..])
}

/// # Safety
/// AVX2 must be available, and all five slices must have equal lengths.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn lengths4_u32_avx2(a: &[u32], b: [&[u32]; 4]) -> [usize; 4] {
    const LANES: usize = 8;
    if a.len() < LANES {
        return b.map(|candidate| scalar_u32(a, candidate));
    }

    // SAFETY: each equal-length slice contains at least LANES tokens. All
    // loads are unaligned reads, so aliasing between any inputs is allowed.
    // Expose four independent comparisons before following any long match.
    let different = unsafe {
        let current = _mm256_loadu_si256(a.as_ptr().cast());
        let candidate0 = _mm256_loadu_si256(b[0].as_ptr().cast());
        let equal0 = _mm256_cmpeq_epi32(current, candidate0);
        let candidate1 = _mm256_loadu_si256(b[1].as_ptr().cast());
        let equal1 = _mm256_cmpeq_epi32(current, candidate1);
        let candidate2 = _mm256_loadu_si256(b[2].as_ptr().cast());
        let equal2 = _mm256_cmpeq_epi32(current, candidate2);
        let candidate3 = _mm256_loadu_si256(b[3].as_ptr().cast());
        let equal3 = _mm256_cmpeq_epi32(current, candidate3);
        [
            !(_mm256_movemask_epi8(equal0) as u32),
            !(_mm256_movemask_epi8(equal1) as u32),
            !(_mm256_movemask_epi8(equal2) as u32),
            !(_mm256_movemask_epi8(equal3) as u32),
        ]
    };
    std::array::from_fn(|i| {
        if different[i] != 0 {
            different[i].trailing_zeros() as usize / 4
        } else {
            // SAFETY: AVX2 was proved by the caller. Removing the same full
            // first vector preserves equal lengths; no prefix is rescanned.
            LANES + unsafe { length_u32_avx2(&a[LANES..], &b[i][LANES..]) }
        }
    })
}

/// # Safety
/// AVX-512F/BW must be available, and all five slices must have equal lengths.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn lengths4_u32_avx512(a: &[u32], b: [&[u32]; 4]) -> [usize; 4] {
    const LANES: usize = 16;
    if a.len() < LANES {
        return b.map(|candidate| scalar_u32(a, candidate));
    }

    // SAFETY: all slices contain the complete first vector. loadu does not
    // require vector alignment or disjointness. Reuse current in all four
    // independent comparisons before allowing any continuation to begin.
    let different = unsafe {
        let current = _mm512_loadu_si512(a.as_ptr().cast());
        let candidate0 = _mm512_loadu_si512(b[0].as_ptr().cast());
        let equal0 = _mm512_cmpeq_epi32_mask(current, candidate0);
        let candidate1 = _mm512_loadu_si512(b[1].as_ptr().cast());
        let equal1 = _mm512_cmpeq_epi32_mask(current, candidate1);
        let candidate2 = _mm512_loadu_si512(b[2].as_ptr().cast());
        let equal2 = _mm512_cmpeq_epi32_mask(current, candidate2);
        let candidate3 = _mm512_loadu_si512(b[3].as_ptr().cast());
        let equal3 = _mm512_cmpeq_epi32_mask(current, candidate3);
        [!equal0, !equal1, !equal2, !equal3]
    };
    std::array::from_fn(|i| {
        if different[i] != 0 {
            different[i].trailing_zeros() as usize
        } else {
            // SAFETY: caller proved AVX-512F/BW; the equal-length suffixes
            // begin after the already-compared, complete first vector.
            LANES + unsafe { length_u32_avx512(&a[LANES..], &b[i][LANES..]) }
        }
    })
}
