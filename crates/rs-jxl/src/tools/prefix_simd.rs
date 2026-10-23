// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from the BSD-3-Clause sources identified in docs/PORTING_KERNELS.md.
// The complete upstream notice and license are retained in that document.

//! Native byte-mask port of `lib/jxl/enc_lz77_bucket.cc::FingerprintMask`
//! at frozen commit b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Bit i denotes input byte i, independently of host byte order. Only x86_64
//! has SIMD backends: runtime-detected AVX2 or SSE2. All other targets use the
//! scalar implementation. Full unaligned vectors and scalar tails never need
//! padding. This helper alone does not establish codec equivalence or speed.

/// Internal bounded entry point; the public API returns an error above 64.
pub(super) fn fingerprint_mask(tags: &[u8], wanted: u8) -> u64 {
    assert!(tags.len() <= 64, "a fingerprint mask holds at most 64 tags");
    #[cfg(target_arch = "x86_64")]
    {
        if tags.len() >= 32 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection includes OS support for AVX2. The slice
            // has at most 64 bytes, and the backend loads only complete chunks.
            return unsafe { x86_64::avx2_mask(tags, wanted) };
        }
        if tags.len() >= 16 && std::is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 was detected; the bounded slice is read only in
            // complete chunks, with any remaining bytes handled scalarly.
            return unsafe { x86_64::sse2_mask(tags, wanted) };
        }
    }
    scalar_mask(tags, wanted)
}

/// Caller supplies at most 64 bytes, so every shift is strictly below 64.
fn scalar_mask(tags: &[u8], wanted: u8) -> u64 {
    tags.iter()
        .enumerate()
        .fold(0, |mask, (i, &tag)| mask | (u64::from(tag == wanted) << i))
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::{
        _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8, _mm256_cmpeq_epi8,
        _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
    };

    /// Requires runtime AVX2 support and at most 64 input bytes.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avx2_mask(tags: &[u8], wanted: u8) -> u64 {
        let wanted_vector = _mm256_set1_epi8(wanted as i8);
        let mut result = 0;
        let mut chunks = tags.chunks_exact(32);
        for (block, bytes) in chunks.by_ref().enumerate() {
            // SAFETY: chunks_exact yields 32 readable bytes inside tags;
            // loadu permits any alignment, and this function requires AVX2.
            let values = unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) };
            // movemask returns i32: narrow to u32 before widening so lane 31
            // cannot sign-extend into the upper half of the 64-bit result.
            let bits = _mm256_movemask_epi8(_mm256_cmpeq_epi8(values, wanted_vector)) as u32;
            result |= u64::from(bits) << (block * 32);
        }
        let tail = chunks.remainder();
        if !tail.is_empty() {
            // An empty tail after 64 bytes must not cause a shift by 64.
            result |= super::scalar_mask(tail, wanted) << (tags.len() - tail.len());
        }
        result
    }

    /// Requires runtime SSE2 support and at most 64 input bytes.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sse2_mask(tags: &[u8], wanted: u8) -> u64 {
        let wanted_vector = _mm_set1_epi8(wanted as i8);
        let mut result = 0;
        let mut chunks = tags.chunks_exact(16);
        for (block, bytes) in chunks.by_ref().enumerate() {
            // SAFETY: chunks_exact yields 16 readable bytes inside tags;
            // loadu permits any alignment, and this function requires SSE2.
            let values = unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) };
            let bits = _mm_movemask_epi8(_mm_cmpeq_epi8(values, wanted_vector)) as u32;
            result |= u64::from(bits) << (block * 16);
        }
        let tail = chunks.remainder();
        if !tail.is_empty() {
            result |= super::scalar_mask(tail, wanted) << (tags.len() - tail.len());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_mask(tags: &[u8], wanted: u8) -> u64 {
        let mut mask = 0;
        for (i, &tag) in tags.iter().enumerate() {
            if tag == wanted {
                mask |= 1u64 << i;
            }
        }
        mask
    }

    fn check_lengths_alignments_and_targets(mask: impl Fn(&[u8], u8) -> u64) {
        let bytes: [u8; 96] = std::array::from_fn(|i| (i as u8).wrapping_mul(73));
        for len in 0..=64 {
            for offset in 0..32 {
                let tags = &bytes[offset..offset + len];
                for wanted in 0..=u8::MAX {
                    let expected = reference_mask(tags, wanted);
                    assert_eq!(scalar_mask(tags, wanted), expected);
                    assert_eq!(
                        mask(tags, wanted),
                        expected,
                        "len={len}, offset={offset}, wanted={wanted}"
                    );
                }
            }
        }
    }

    fn check_matching_positions_and_tails(mask: impl Fn(&[u8], u8) -> u64) {
        for len in 0..=64 {
            // No readable padding is part of this allocation's slice contract.
            let mut tags = vec![0; len].into_boxed_slice();
            let all = if len == 64 {
                u64::MAX
            } else {
                (1u64 << len) - 1
            };
            for wanted in 0..=u8::MAX {
                tags.fill(wanted);
                assert_eq!(mask(&tags, wanted), all);
                tags.fill(wanted ^ 0xff);
                assert_eq!(mask(&tags, wanted), 0);
                for position in 0..len {
                    tags[position] = wanted;
                    assert_eq!(
                        mask(&tags, wanted),
                        1u64 << position,
                        "len={len}, position={position}, wanted={wanted}"
                    );
                    tags[position] = wanted ^ 0xff;
                }
                for (i, tag) in tags.iter_mut().enumerate() {
                    if i % 3 != 1 {
                        *tag = wanted;
                    }
                }
                assert_eq!(mask(&tags, wanted), reference_mask(&tags, wanted));
            }
        }

        let positions = [0, 3, 15, 16, 31, 32, 47, 48, 63];
        let mut tags = [0; 64];
        for position in positions {
            tags[position] = 0x80;
        }
        let mut bits = mask(&tags, 0x80);
        let mut ascending = Vec::new();
        while bits != 0 {
            ascending.push(bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
        assert_eq!(ascending, positions);
        let mut bits = mask(&tags, 0x80);
        let mut descending = Vec::new();
        while bits != 0 {
            let position = 63 - bits.leading_zeros() as usize;
            descending.push(position);
            bits &= !(1u64 << position);
        }
        assert_eq!(descending, positions.into_iter().rev().collect::<Vec<_>>());
    }

    #[test]
    fn scalar_and_dispatch_cover_all_lengths_alignments_and_bytes() {
        check_lengths_alignments_and_targets(fingerprint_mask);
    }

    #[test]
    fn scalar_and_dispatch_cover_matching_positions_tails_and_order() {
        check_matching_positions_and_tails(scalar_mask);
        check_matching_positions_and_tails(fingerprint_mask);
    }

    #[test]
    #[should_panic(expected = "a fingerprint mask holds at most 64 tags")]
    fn internal_dispatch_rejects_oversized_input() {
        fingerprint_mask(&[0; 65], 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_for_all_lengths_bytes_positions_and_tails() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 parity cases skipped: AVX2 is unavailable");
            return;
        }
        let mask = |tags: &[u8], wanted| {
            assert!(tags.len() <= 64);
            // SAFETY: AVX2 was detected above, every input is bounded, and the
            // backend reads complete chunks only. Force it even for short tails.
            unsafe { x86_64::avx2_mask(tags, wanted) }
        };
        check_lengths_alignments_and_targets(mask);
        check_matching_positions_and_tails(mask);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse2_matches_scalar_for_all_lengths_bytes_positions_and_tails() {
        if !std::is_x86_feature_detected!("sse2") {
            eprintln!("SSE2 parity cases skipped: SSE2 is unavailable");
            return;
        }
        let mask = |tags: &[u8], wanted| {
            assert!(tags.len() <= 64);
            // SAFETY: SSE2 was detected above and every input is bounded;
            // force this backend even on AVX2 hosts and for short tails.
            unsafe { x86_64::sse2_mask(tags, wanted) }
        };
        check_lengths_alignments_and_targets(mask);
        check_matching_positions_and_tails(mask);
    }
}
