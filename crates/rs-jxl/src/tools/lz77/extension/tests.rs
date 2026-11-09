use super::{Extension, MatchExtension};
use crate::Error;

fn supported(mode: MatchExtension) -> bool {
    match mode {
        MatchExtension::Legacy | MatchExtension::Scalar => true,
        MatchExtension::Avx2 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                std::is_x86_feature_detected!("avx2")
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                false
            }
        }
        MatchExtension::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                false
            }
        }
    }
}

fn extensions() -> Vec<(MatchExtension, Extension)> {
    [
        MatchExtension::Scalar,
        MatchExtension::Avx2,
        MatchExtension::Avx512,
    ]
    .into_iter()
    .filter_map(|mode| {
        let result = Extension::new(mode);
        if supported(mode) {
            Some((mode, result.unwrap().expect("explicit mode needs a kernel")))
        } else {
            assert!(matches!(result, Err(Error::InvalidInput(_))), "{mode:?}");
            None
        }
    })
    .collect()
}

// Deliberately independent of the implementation's scalar while loops.
fn oracle<T: Eq>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .position(|(left, right)| left != right)
        .unwrap_or(a.len())
}

fn check_u32(extensions: &[(MatchExtension, Extension)], a: &[u32], b: &[u32]) {
    let expected = oracle(a, b);
    for &(mode, extension) in extensions {
        assert_eq!(extension.length_u32(a, b), expected, "{mode:?}, u32");
        assert_eq!(extension.length_u32(b, a), expected, "{mode:?}, reversed");
    }
}

fn check_u16(extensions: &[(MatchExtension, Extension)], a: &[u16], b: &[u16]) {
    let expected = oracle(a, b);
    for &(mode, extension) in extensions {
        assert_eq!(extension.length_u16(a, b), expected, "{mode:?}, u16");
        assert_eq!(extension.length_u16(b, a), expected, "{mode:?}, reversed");
    }
}

fn check_four(extensions: &[(MatchExtension, Extension)], a: &[u32], b: [&[u32]; 4]) {
    let expected = b.map(|candidate| oracle(a, candidate));
    for &(mode, extension) in extensions {
        assert_eq!(extension.lengths4_u32(a, b), expected, "{mode:?}, four-way");
    }
}

fn token(i: usize) -> u32 {
    match i % 4 {
        0 => 0,
        1 => u32::MAX,
        2 => 0x8000_8000,
        _ => (i as u32).wrapping_mul(0x9e37_79b9),
    }
}

#[test]
fn construction_keeps_legacy_and_rejects_unsupported_isa() {
    assert_eq!(MatchExtension::default(), MatchExtension::Legacy);
    assert!(Extension::new(MatchExtension::Legacy).unwrap().is_none());
    assert!(!extensions().is_empty());
}

#[test]
#[cfg(not(feature = "experimental-lz77"))]
fn production_build_rejects_experiments_instead_of_silently_ignoring_them() {
    assert!(MatchExtension::Legacy.validate().is_ok());
    for mode in [
        MatchExtension::Scalar,
        MatchExtension::Avx2,
        MatchExtension::Avx512,
    ] {
        assert!(mode.validate().is_err());
    }
}

#[test]
fn every_u32_mismatch_lane_and_bit_including_four_way() {
    let extensions = extensions();
    let a = [u32::MAX; 65];
    let mut b = a;
    for position in 0..a.len() {
        for bit in 0..u32::BITS {
            b[position] ^= 1u32 << bit;
            assert_eq!(oracle(&a, &b), position);
            check_u32(&extensions, &a, &b);
            for slot in 0..4 {
                let mut candidates = [a.as_slice(); 4];
                candidates[slot] = &b;
                check_four(&extensions, &a, candidates);
            }
            b[position] = a[position];
        }
    }
}

#[test]
fn every_u16_mismatch_lane_and_bit() {
    let extensions = extensions();
    let a = [u16::MAX; 97];
    let mut b = a;
    for position in 0..a.len() {
        for bit in 0..u16::BITS {
            b[position] ^= 1u16 << bit;
            assert_eq!(oracle(&a, &b), position);
            check_u16(&extensions, &a, &b);
            b[position] = a[position];
        }
    }
}

#[test]
fn every_short_length_and_first_mismatch() {
    let extensions = extensions();
    for length in 0..=129 {
        let a: Vec<u32> = (0..length).map(token).collect();
        let a16: Vec<u16> = a.iter().map(|&value| value as u16).collect();
        let mut b = a.clone();
        let mut b16 = a16.clone();
        for first in 0..=length {
            b.clone_from_slice(&a);
            b16.clone_from_slice(&a16);
            if first < length {
                b[first] ^= u32::MAX;
                b16[first] ^= u16::MAX;
                // A later difference must not hide the first one.
                b[length - 1] = a[length - 1] ^ 1;
                b16[length - 1] = a16[length - 1] ^ 1;
            }
            assert_eq!(oracle(&a, &b), first);
            assert_eq!(oracle(&a16, &b16), first);
            check_u32(&extensions, &a, &b);
            check_u16(&extensions, &a16, &b16);
            check_four(&extensions, &a, [&a, &b, &a, &b]);
        }
    }
}

#[test]
fn long_lengths_tails_and_all_equal_maximum_values() {
    let extensions = extensions();
    for length in [255usize, 256, 257, 511, 512, 513, 4095, 4096, 4097, 16385] {
        let a = vec![u32::MAX; length];
        let a16 = vec![u16::MAX; length];
        let mut b = a.clone();
        let mut b16 = a16.clone();
        for first in [0, 7, 8, 15, 16, 31, 32, length / 2, length - 1, length] {
            b.clone_from_slice(&a);
            b16.clone_from_slice(&a16);
            if first < length {
                b[first] = 0;
                b16[first] = 0;
            }
            check_u32(&extensions, &a, &b);
            check_u16(&extensions, &a16, &b16);
            check_four(&extensions, &a, [&b, &a, &b, &a]);
        }
        check_u32(&extensions, &a, &a);
        check_u16(&extensions, &a16, &a16);
        check_four(&extensions, &a, [&a; 4]);
    }
}

#[test]
fn independent_u32_alignments_and_exact_slice_ends() {
    let extensions = extensions();
    for a_offset in 0..16 {
        for b_offset in 0..16 {
            for length in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
                let mut a = vec![0; a_offset + length];
                for (i, value) in a[a_offset..].iter_mut().enumerate() {
                    *value = token(i);
                }
                let a = &a[a_offset..];
                let mut storage = vec![0; b_offset + length];
                let b = &mut storage[b_offset..];
                for first in [0, length / 2, length.saturating_sub(1), length] {
                    b.clone_from_slice(a);
                    if first < length {
                        b[first] ^= 0x8000_0000;
                    }
                    check_u32(&extensions, a, b);
                }
            }
        }
    }
}

#[test]
fn independent_u16_alignments_and_exact_slice_ends() {
    let extensions = extensions();
    for a_offset in 0..32 {
        for b_offset in 0..32 {
            for length in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
                let mut a = vec![0u16; a_offset + length];
                for (i, value) in a[a_offset..].iter_mut().enumerate() {
                    *value = token(i) as u16;
                }
                let a = &a[a_offset..];
                let mut storage = vec![0; b_offset + length];
                let b = &mut storage[b_offset..];
                for first in [0, length / 2, length.saturating_sub(1), length] {
                    b.clone_from_slice(a);
                    if first < length {
                        b[first] ^= 0x8000;
                    }
                    check_u16(&extensions, a, b);
                }
            }
        }
    }
}

#[test]
fn immutable_overlaps_from_distance_one() {
    let extensions = extensions();
    for distance in (1usize..=65).chain([127, 128, 129, 255]) {
        for length in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 129, 1025] {
            for first in [0, length / 2, length.saturating_sub(1), length] {
                let mut values = vec![u32::MAX; distance + length];
                let mut values16 = vec![u16::MAX; distance + length];
                if first < length {
                    // The shifted slice encounters the marker first, even
                    // when the marker also falls inside the current slice.
                    values[distance + first] = 0;
                    values16[distance + first] = 0;
                }
                let a = &values[..length];
                let b = &values[distance..];
                let a16 = &values16[..length];
                let b16 = &values16[distance..];
                assert_eq!(oracle(a, b), first);
                assert_eq!(oracle(a16, b16), first);
                check_u32(&extensions, a, b);
                check_u16(&extensions, a16, b16);
                check_four(&extensions, a, [a, b, b, a]);
            }
        }
    }
}

#[test]
fn four_way_divergent_matches_with_independent_alignments() {
    let extensions = extensions();
    for a_offset in 0..16 {
        for shift in 0..16 {
            let offsets = [shift, (shift + 3) % 16, (shift + 7) % 16, (shift + 11) % 16];
            for length in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129, 1025] {
                let mut storage = vec![0; a_offset + length];
                for (i, value) in storage[a_offset..].iter_mut().enumerate() {
                    *value = token(i);
                }
                let a = &storage[a_offset..];
                let mut candidates: [Vec<u32>; 4] =
                    std::array::from_fn(|i| vec![0; offsets[i] + length]);
                let firsts = [0, length.min(8), length.saturating_sub(1), length];
                for rotation in 0..4 {
                    for i in 0..4 {
                        let b = &mut candidates[i][offsets[i]..];
                        b.clone_from_slice(a);
                        let first = firsts[(i + rotation) % 4];
                        if first < length {
                            b[first] ^= 1u32 << (i * 8);
                        }
                    }
                    let b = std::array::from_fn(|i| &candidates[i][offsets[i]..]);
                    check_four(&extensions, a, b);
                }
            }
        }
    }
}

#[test]
fn four_way_overlapping_candidates_with_divergent_continuations() {
    let extensions = extensions();
    for length in [
        0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129, 1025, 16385,
    ] {
        let mut values = vec![u32::MAX; length + 129];
        values[(length + 129) / 2] = 0;
        let a = &values[..length];
        for distances in [[1, 2, 7, 33], [8, 16, 64, 129]] {
            let b = distances.map(|distance| &values[distance..distance + length]);
            check_four(&extensions, a, b);
        }
    }
}

#[test]
fn safe_entries_reject_all_unequal_lengths() {
    use std::panic::catch_unwind;

    let current = [u32::MAX; 65];
    let candidate = [0u32; 65];
    for (_, extension) in extensions() {
        for (left, right) in [(0, 1), (1, 0), (63, 64), (64, 63), (64, 65)] {
            assert!(
                catch_unwind(|| { extension.length_u32(&current[..left], &candidate[..right]) })
                    .is_err()
            );
            assert!(
                catch_unwind(|| {
                    extension.length_u16(&[u16::MAX; 65][..left], &[0u16; 65][..right])
                })
                .is_err()
            );
        }
        for slot in 0..4 {
            for length in [0, 63, 65] {
                let mut b = [&candidate[..64]; 4];
                b[slot] = &candidate[..length];
                // Even an earlier candidate's immediate mismatch must not
                // skip validation of a later candidate's invalid length.
                assert!(catch_unwind(|| extension.lengths4_u32(&current[..64], b)).is_err());
            }
        }
    }
}
