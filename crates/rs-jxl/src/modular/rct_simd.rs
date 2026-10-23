// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

//! Native forward RCT kernels from libjxl
//! b866b3defec13494bfbaf705db93581cda118e37:
//! `lib/jxl/modular/transform/enc_rct.cc::{RctPermute, FwdRctRow, FwdRctImpl}`.
//! `rct.cc::InvRCTRow` supplies the independent test inverse.
//!
//! All 42 types are supported: six input permutations times seven transforms.
//! The transform is specialized outside the pixel loop. On x86_64, runtime
//! AVX2/SSE2 detection selects eight/four signed-i32 lanes; AArch64 NEON selects
//! four lanes. Other platforms and short slices use safe scalar code.
//! Every add/subtract wraps BEFORE any arithmetic right shift, matching
//! Highway rather than widened averages or
//! truncating signed division. The full i32 domain is valid, including raw
//! binary32 sample words. All three inputs are loaded before in-place stores.
//!
//! Integration: replace the body/call of transforms::rct_forward with forward
//! after parent registration. This module never calls that wrapper, so such
//! delegation cannot recurse. No registration or parent file is modified here.
//! Selected planes must have equal lengths; the caller still owns checking
//! equal image dimensions/shifts and writing the transform header. Stateless
//! RCT may traverse flattened planes across row boundaries without changing
//! results. No allocation, channel-buffer swap, or readable padding is needed.
//! Type zero is a validated successful no-op, matching the Rust wrapper API
//! (source FwdRct(Image&) returns false to indicate 'not applied').
//!
//! Borrowed row scheduling reuses the caller's pool and respects null scopes.
//! Omitted: aligned/padded overrun paths, AVX-512,
//! ARMv7/SVE SIMD, inverse production kernels, candidate selection, and metadata.
//! Native parity tests are not an end-to-end speed or codec-compatibility claim.

use crate::{Error, Result};

#[cfg(target_arch = "aarch64")]
#[path = "rct_neon.rs"]
mod neon;

/// Transform exactly planes[begin..begin+3] in place. Types are 0..=41.
/// Validate type/range/all three lengths before any mutation, including type
/// zero. Empty equal-length planes are valid. Other planes remain untouched.
pub fn forward(planes: &mut [Vec<i32>], begin: usize, rct_type: u32) -> Result<()> {
    forward_impl(planes, begin, rct_type, true)
}

/// Portable scalar counterpart with identical validation and wrapping rules.
/// It is also useful for parent-side backend parity checks.
pub fn forward_scalar(planes: &mut [Vec<i32>], begin: usize, rct_type: u32) -> Result<()> {
    forward_impl(planes, begin, rct_type, false)
}

fn forward_impl(planes: &mut [Vec<i32>], begin: usize, rct_type: u32, native: bool) -> Result<()> {
    if rct_type >= 42 {
        return Err(Error::InvalidInput("RCT type must be 0..=41".into()));
    }
    let end = begin
        .checked_add(3)
        .ok_or(Error::Limit("RCT channel range overflow"))?;
    if end > planes.len() {
        return Err(Error::InvalidInput(
            "RCT requires three selected planes".into(),
        ));
    }
    let len = planes[begin].len();
    if planes[begin..end].iter().any(|plane| plane.len() != len) {
        return Err(Error::InvalidInput("RCT plane lengths differ".into()));
    }
    if rct_type == 0 || len == 0 {
        return Ok(());
    }
    let p = (rct_type / 7) as usize;
    let order = [p % 3, (p + 1 + p / 3) % 3, (p + 2 - p / 3) % 3];
    let [a, b, c] = &mut planes[begin..end] else {
        unreachable!("validated three-plane range")
    };
    let rows = [a.as_mut_slice(), b.as_mut_slice(), c.as_mut_slice()];
    dispatch(rows, order, rct_type, native);
    Ok(())
}

/// FwdRctImpl's row-parallel, in-place contract. The caller has already checked
/// channel shifts and metadata. No sample buffer is cloned or replaced.
pub(crate) fn forward_rows(rows: [&mut [i32]; 3], width: usize, rct_type: u32) -> Result<()> {
    if rct_type >= 42 {
        return Err(Error::InvalidInput("RCT type must be 0..=41".into()));
    }
    let len = rows[0].len();
    if rows.iter().any(|row| row.len() != len)
        || (width == 0 && len != 0)
        || (width != 0 && !len.is_multiple_of(width))
    {
        return Err(Error::InvalidInput("invalid RCT row geometry".into()));
    }
    if rct_type == 0 || len == 0 {
        return Ok(());
    }
    let p = (rct_type / 7) as usize;
    let order = [p % 3, (p + 1 + p / 3) % 3, (p + 2 - p / 3) % 3];
    if crate::execution::is_serial() || crate::execution::worker_count() == 1 || len == width {
        dispatch(rows, order, rct_type, true);
    } else {
        let [a, b, c] = rows;
        let mut rows: Vec<_> = a
            .chunks_mut(width)
            .zip(b.chunks_mut(width))
            .zip(c.chunks_mut(width))
            .collect();
        crate::execution::for_each_mut(&mut rows, |_, ((a, b), c)| {
            dispatch([a, b, c], order, rct_type, true);
        });
    }
    Ok(())
}

fn dispatch(rows: [&mut [i32]; 3], order: [usize; 3], rct_type: u32, native: bool) {
    match rct_type % 7 {
        0 => run::<0>(rows, order, native),
        1 => run::<1>(rows, order, native),
        2 => run::<2>(rows, order, native),
        3 => run::<3>(rows, order, native),
        4 => run::<4>(rows, order, native),
        5 => run::<5>(rows, order, native),
        6 => run::<6>(rows, order, native),
        _ => unreachable!(),
    }
}

#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(unused_mut)
)]
fn run<const T: u32>(mut rows: [&mut [i32]; 3], order: [usize; 3], native: bool) {
    #[allow(unused_mut)]
    let mut next = 0;
    #[cfg(target_arch = "x86_64")]
    if native {
        if rows[0].len() >= 8 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2/OS support detected; validated disjoint slices
            // have equal lengths and order is a permutation of 0,1,2.
            next = unsafe { x86::avx2::run::<T>(&mut rows, order) };
        } else if rows[0].len() >= 4 && std::is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 detected; same validated slice/order contract.
            next = unsafe { x86::sse2::run::<T>(&mut rows, order) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if native && rows[0].len() >= 4 && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: NEON detected (baseline on supported std AArch64 targets);
        // validated disjoint, equal-length slices and a permutation of 0,1,2.
        next = unsafe { neon::run::<T>(&mut rows, order) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = native;
    for x in next..rows[0].len() {
        let [a, mut b, mut c] = order.map(|channel| rows[channel][x]);
        let result = if T == 6 {
            let co = a.wrapping_sub(c);
            let tmp = c.wrapping_add(co >> 1);
            let cg = b.wrapping_sub(tmp);
            [tmp.wrapping_add(cg >> 1), co, cg]
        } else {
            if T >> 1 == 1 {
                b = b.wrapping_sub(a);
            } else if T >> 1 == 2 {
                b = b.wrapping_sub(a.wrapping_add(c) >> 1);
            }
            if T & 1 != 0 {
                c = c.wrapping_sub(a);
            }
            [a, b, c]
        };
        rows[0][x] = result[0];
        rows[1][x] = result[1];
        rows[2][x] = result[2];
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    // The seven source templates become concrete native kernels, without
    // transform branches or permutation lookup inside the vector loop.
    macro_rules! kernel {
        ($feature:literal, $lanes:literal) => {
            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn run<const T: u32>(
                rows: &mut [&mut [i32]; 3],
                order: [usize; 3],
            ) -> usize {
                let end = rows[0].len() / $lanes * $lanes;
                let out = [
                    rows[0].as_mut_ptr(),
                    rows[1].as_mut_ptr(),
                    rows[2].as_mut_ptr(),
                ];
                let input = order.map(|channel| out[channel].cast_const());
                // SAFETY: caller detects this target feature and provides
                // equal-length, disjoint slices plus a valid permutation.
                // Each unaligned load/store covers exactly LANES elements;
                // x+LANES <= end <= len, so no padding is read or overwritten.
                // Permuted inputs can alias the output of another channel:
                // load all three before any stores, just as FwdRctRow does.
                unsafe {
                    for x in (0..end).step_by($lanes) {
                        let a = load(input[0].add(x).cast());
                        let b = load(input[1].add(x).cast());
                        let c = load(input[2].add(x).cast());
                        let (a, b, c) = if T == 6 {
                            let co = sub(a, c);
                            let tmp = add(c, shr::<1>(co));
                            let cg = sub(b, tmp);
                            (add(tmp, shr::<1>(cg)), co, cg)
                        } else {
                            let b = if T >> 1 == 1 {
                                sub(b, a)
                            } else if T >> 1 == 2 {
                                sub(b, shr::<1>(add(a, c)))
                            } else {
                                b
                            };
                            let c = if T & 1 != 0 { sub(c, a) } else { c };
                            (a, b, c)
                        };
                        store(out[0].add(x).cast(), a);
                        store(out[1].add(x).cast(), b);
                        store(out[2].add(x).cast(), c);
                    }
                }
                end
            }
        };
    }

    pub(super) mod sse2 {
        use std::arch::x86_64::{
            _mm_add_epi32 as add, _mm_loadu_si128 as load, _mm_srai_epi32 as shr,
            _mm_storeu_si128 as store, _mm_sub_epi32 as sub,
        };
        kernel!("sse2", 4);
    }

    pub(super) mod avx2 {
        use std::arch::x86_64::{
            _mm256_add_epi32 as add, _mm256_loadu_si256 as load, _mm256_srai_epi32 as shr,
            _mm256_storeu_si256 as store, _mm256_sub_epi32 as sub,
        };
        kernel!("avx2", 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 2, 1],
        [1, 0, 2],
        [2, 1, 0],
    ];

    #[test]
    fn borrowed_rows_match_all_transforms_with_and_without_parallelism() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for width in [1, 7, 8, 17, 257] {
            let source: Vec<Vec<i32>> = (0..3)
                .map(|c| {
                    (0..width * 9)
                        .map(|i| ((i as u32).wrapping_mul(0x9e37_79b9) ^ (c * 0x1234_5678)) as i32)
                        .collect()
                })
                .collect();
            for kind in 0..42 {
                let mut expected = source.clone();
                forward_scalar(&mut expected, 0, kind).unwrap();
                for serial in [false, true] {
                    let mut actual = source.clone();
                    let pointers: Vec<_> = actual.iter().map(|v| v.as_ptr()).collect();
                    pool.install(|| {
                        let mut apply = || {
                            let [a, b, c] = actual.as_mut_slice() else {
                                unreachable!()
                            };
                            forward_rows([a, b, c], width, kind).unwrap();
                        };
                        if serial {
                            crate::execution::serial(apply)
                        } else {
                            apply()
                        }
                    });
                    assert_eq!(
                        actual, expected,
                        "width={width} kind={kind} serial={serial}"
                    );
                    assert_eq!(
                        actual.iter().map(|v| v.as_ptr()).collect::<Vec<_>>(),
                        pointers
                    );
                }
            }
        }
    }

    #[test]
    fn borrowed_row_validation_precedes_mutation() {
        for (width, kind, short) in [(0, 1, false), (4, 1, false), (3, 42, false), (3, 1, true)] {
            let mut input = [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
            if short {
                input[2].pop();
            }
            let original = input.clone();
            let [a, b, c] = &mut input;
            assert!(forward_rows([a, b, c], width, kind).is_err());
            assert_eq!(input, original);
        }
        forward_rows([&mut [], &mut [], &mut []], 0, 6).unwrap();
    }

    // Independent mathematical lane oracle: do arithmetic in i64, reduce
    // modulo 2^32, and use floor division instead of the kernel's signed shift.
    fn lane(value: i64) -> i64 {
        (value + (1i64 << 31)).rem_euclid(1i64 << 32) - (1i64 << 31)
    }

    fn reference(input: [i32; 3], rct_type: u32) -> [i32; 3] {
        let [a, b, c] = ORDERS[(rct_type / 7) as usize].map(|i| i64::from(input[i]));
        let result = match rct_type % 7 {
            0 => [a, b, c],
            1 => [a, b, lane(c - a)],
            2 => [a, lane(b - a), c],
            3 => [a, lane(b - a), lane(c - a)],
            4 => [a, lane(b - lane(a + c).div_euclid(2)), c],
            5 => [a, lane(b - lane(a + c).div_euclid(2)), lane(c - a)],
            6 => {
                let co = lane(a - c);
                let tmp = lane(c + co.div_euclid(2));
                let cg = lane(b - tmp);
                [lane(tmp + cg.div_euclid(2)), co, cg]
            }
            _ => unreachable!(),
        };
        result.map(|v| i32::try_from(v).unwrap())
    }

    fn inverse(input: [i32; 3], rct_type: u32) -> [i32; 3] {
        let [a, b, c] = input.map(i64::from);
        let ordered = if rct_type % 7 == 6 {
            let tmp = lane(a - c.div_euclid(2));
            let blue = lane(tmp - b.div_euclid(2));
            [lane(blue + b), lane(c + tmp), blue]
        } else {
            let c = if (rct_type % 7) & 1 != 0 {
                lane(c + a)
            } else {
                c
            };
            let b = match (rct_type % 7) >> 1 {
                1 => lane(b + a),
                2 => lane(b + lane(a + c).div_euclid(2)),
                _ => b,
            };
            [a, b, c]
        };
        let mut original = [0; 3];
        for (i, value) in ORDERS[(rct_type / 7) as usize].into_iter().zip(ordered) {
            original[i] = i32::try_from(value).unwrap();
        }
        original
    }

    fn check_wrappers(source: &[Vec<i32>], begin: usize) {
        for rct_type in 0..42 {
            let mut expected = source.to_vec();
            for x in 0..source[begin].len() {
                let input = [source[begin][x], source[begin + 1][x], source[begin + 2][x]];
                let output = reference(input, rct_type);
                assert_eq!(inverse(output, rct_type), input);
                for channel in 0..3 {
                    expected[begin + channel][x] = output[channel];
                }
            }
            for apply in [forward, forward_scalar] {
                let mut actual = source.to_vec();
                let pointers: Vec<_> = actual.iter().map(|v| v.as_ptr()).collect();
                let capacities: Vec<_> = actual.iter().map(Vec::capacity).collect();
                apply(&mut actual, begin, rct_type).unwrap();
                assert_eq!(actual, expected, "RCT {rct_type}");
                assert_eq!(
                    actual.iter().map(|v| v.as_ptr()).collect::<Vec<_>>(),
                    pointers
                );
                assert_eq!(
                    actual.iter().map(Vec::capacity).collect::<Vec<_>>(),
                    capacities
                );
            }
        }
    }

    #[test]
    fn all_types_signed_extremes_random_words_and_unrelated_channels() {
        let edges = [
            i32::MIN,
            i32::MIN + 1,
            -65535,
            -9,
            -1,
            0,
            1,
            7,
            65535,
            i32::MAX - 1,
            i32::MAX,
        ];
        let mut selected = [Vec::new(), Vec::new(), Vec::new()];
        for &a in &edges {
            for &b in &edges {
                for &c in &edges {
                    for (plane, value) in selected.iter_mut().zip([a, b, c]) {
                        plane.push(value);
                    }
                }
            }
        }
        let mut random = 0xb866_b3deu32;
        for _ in 0..1024 {
            for plane in &mut selected {
                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                plane.push(random as i32);
            }
        }
        let alpha = vec![0; selected[0].len()];
        let [a, b, c] = selected;
        let source = vec![vec![123], vec![], a, b, c, alpha, vec![i32::MIN, 55]];
        check_wrappers(&source, 2);
    }

    #[test]
    fn all_lengths_tails_empty_and_single_pixel_planes() {
        for len in (0..=35).chain([63, 64, 65, 127, 128, 129, 255, 256, 257]) {
            let source: Vec<Vec<i32>> = (0..3u32)
                .map(|c| {
                    (0..len)
                        .map(|x| {
                            (x as u32)
                                .wrapping_mul(0x9e37_79b9)
                                .rotate_left(c * 7)
                                .wrapping_add(c) as i32
                        })
                        .collect()
                })
                .collect();
            check_wrappers(&source, 0);
        }
    }

    #[test]
    fn wrapping_before_shift_and_negative_odd_rounding_goldens() {
        for (input, rct_type, expected) in [
            ([i32::MAX, 0, i32::MAX], 4, [i32::MAX, 1, i32::MAX]),
            ([i32::MIN, 0, i32::MAX], 5, [i32::MIN, 1, -1]),
            ([i32::MAX, i32::MIN, i32::MIN], 6, [i32::MAX, -1, 1]),
            (
                [i32::MIN, i32::MAX, 0],
                6,
                [-1610612737, i32::MIN, -1073741825],
            ),
            ([-2, 0, -1], 4, [-2, 2, -1]),
            ([17, -9, 4], 6, [0, 13, -19]),
        ] {
            // Enough samples for each native width, plus a scalar tail.
            let mut planes = input.map(|value| vec![value; 17]);
            forward(&mut planes, 0, rct_type).unwrap();
            for x in 0..17 {
                assert_eq!(planes.each_ref().map(|p| p[x]), expected);
            }
        }
    }

    #[test]
    fn invalid_type_range_or_length_never_mutates_even_noop() {
        let original = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7]];
        for apply in [forward, forward_scalar] {
            for (begin, rct_type) in [(0, 42), (0, u32::MAX), (1, 6), (2, 0), (usize::MAX, 0)] {
                let mut planes = original.clone();
                assert!(apply(&mut planes, begin, rct_type).is_err());
                assert_eq!(planes, original);
            }
            let mut mismatched = vec![vec![1], vec![], vec![2]];
            let copy = mismatched.clone();
            assert!(apply(&mut mismatched, 0, 0).is_err());
            assert_eq!(mismatched, copy);
            assert!(apply(&mut [], 0, 0).is_err());
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_all_types_match_integer_oracle_with_unaligned_tails() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for rct_type in 0..42 {
            for len in (0..=19).chain([31, 32, 33, 63, 64, 65, 257]) {
                for offset in 0..4 {
                    let offsets = [offset, (offset + 1) % 4, (offset + 3) % 4];
                    let mut storage = offsets.map(|o| vec![0x1357_2468; o + len + 1]);
                    for c in 0..3 {
                        for x in 0..len {
                            storage[c][offsets[c] + x] = match (x + c) % 7 {
                                0 => i32::MIN,
                                1 => i32::MAX,
                                2 => -1,
                                3 => -3,
                                _ => {
                                    ((x as u32).wrapping_mul(0x8765_4321)
                                        ^ (c as u32).wrapping_mul(0x8000_0001))
                                        as i32
                                }
                            };
                        }
                    }
                    let original = storage.clone();
                    let [a, b, c] = &mut storage;
                    let mut rows = [
                        &mut a[offsets[0]..offsets[0] + len],
                        &mut b[offsets[1]..offsets[1] + len],
                        &mut c[offsets[2]..offsets[2] + len],
                    ];
                    let order = ORDERS[(rct_type / 7) as usize];
                    // SAFETY: NEON checked; disjoint equal-length slices and
                    // every valid permutation/transform are covered here.
                    let end = unsafe {
                        match rct_type % 7 {
                            0 => neon::run::<0>(&mut rows, order),
                            1 => neon::run::<1>(&mut rows, order),
                            2 => neon::run::<2>(&mut rows, order),
                            3 => neon::run::<3>(&mut rows, order),
                            4 => neon::run::<4>(&mut rows, order),
                            5 => neon::run::<5>(&mut rows, order),
                            6 => neon::run::<6>(&mut rows, order),
                            _ => unreachable!(),
                        }
                    };
                    assert_eq!(end, len / 4 * 4);
                    let mut expected = original.clone();
                    for x in 0..end {
                        let input = std::array::from_fn(|c| original[c][offsets[c] + x]);
                        let values = reference(input, rct_type);
                        assert_eq!(inverse(values, rct_type), input);
                        for c in 0..3 {
                            expected[c][offsets[c] + x] = values[c];
                        }
                    }
                    assert_eq!(
                        storage, expected,
                        "RCT {rct_type}, len {len}, offset {offset}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn each_native_backend_unaligned_independent_offsets_and_no_tail_writes() {
        macro_rules! dispatch {
            ($kernel:ident, $rows:expr, $order:expr, $transform:expr) => {
                match $transform {
                    0 => $kernel::<0>($rows, $order),
                    1 => $kernel::<1>($rows, $order),
                    2 => $kernel::<2>($rows, $order),
                    3 => $kernel::<3>($rows, $order),
                    4 => $kernel::<4>($rows, $order),
                    5 => $kernel::<5>($rows, $order),
                    6 => $kernel::<6>($rows, $order),
                    _ => unreachable!(),
                }
            };
        }
        for lanes in [4, 8] {
            if (lanes == 4 && !std::is_x86_feature_detected!("sse2"))
                || (lanes == 8 && !std::is_x86_feature_detected!("avx2"))
            {
                continue;
            }
            for rct_type in 0..42 {
                for len in (0..=19).chain([31, 32, 33, 63, 64, 65]) {
                    for offset in 0..8 {
                        let offsets = [offset, (offset + 3) % 8, (offset + 5) % 8];
                        let mut storage = offsets.map(|o| vec![0x1357_2468; o + len + 1]);
                        for c in 0..3 {
                            for x in 0..len {
                                storage[c][offsets[c] + x] = ((x as u32).wrapping_mul(0x8765_4321)
                                    ^ (c as u32).wrapping_mul(0x8000_0001))
                                    as i32;
                            }
                        }
                        let original = storage.clone();
                        let [a, b, c] = &mut storage;
                        let mut rows = [
                            &mut a[offsets[0]..offsets[0] + len],
                            &mut b[offsets[1]..offsets[1] + len],
                            &mut c[offsets[2]..offsets[2] + len],
                        ];
                        let order = ORDERS[(rct_type / 7) as usize];
                        // SAFETY: feature detected above; disjoint equal-length
                        // slices with every lane permutation tested explicitly.
                        let end = unsafe {
                            if lanes == 4 {
                                use x86::sse2::run as kernel;
                                dispatch!(kernel, &mut rows, order, rct_type % 7)
                            } else {
                                use x86::avx2::run as kernel;
                                dispatch!(kernel, &mut rows, order, rct_type % 7)
                            }
                        };
                        assert_eq!(end, len / lanes * lanes);
                        let mut expected = original.clone();
                        for x in 0..end {
                            let input = std::array::from_fn(|c| original[c][offsets[c] + x]);
                            let values = reference(input, rct_type);
                            for c in 0..3 {
                                expected[c][offsets[c] + x] = values[c];
                            }
                        }
                        assert_eq!(
                            storage, expected,
                            "{lanes} lanes, RCT {rct_type}, len {len}, offset {offset}"
                        );
                    }
                }
            }
        }
    }
}
