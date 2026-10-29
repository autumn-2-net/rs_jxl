// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Adapter derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.
// The separately licensed libsais-rs utility retains its own Apache-2.0 notice.

//! enc_lz77_suffix.cc::BuildLosslessSuffixArray and the allocation/gates in
//! enc_lz77_suffix.h::LZ77SuffixIndex::Build at b866b3d. Only the integer serial
//! API is used, with the source 6*k scratch tail. No byte/BWT/OMP entry points.
//! Uses vendor/libsais-rs 0.2.2 with five true-serial helper branches; the
//! unpatched registry release can initialize global Rayon on this API.

use super::matchers::filled;
use super::reserved;
use crate::Result;

#[derive(Clone, Copy, Debug)]
struct Layout {
    alphabet: i32,
    extra: i32,
    suffix_entries: usize,
    scratch_bytes: usize,
}

impl Layout {
    fn new(size: usize, largest: u32) -> Option<Self> {
        if size == 0 || size > 1 << 24 || largest >= 1 << 20 {
            return None;
        }
        let alphabet = usize::try_from(largest).ok()?.checked_add(1)?;
        let extra = alphabet.checked_mul(6)?;
        let suffix_entries = size.checked_add(extra)?;
        // Covers mutable text, SA/result, recursive utility workspace, retained
        // neighbors/ranks/LCP minima and allocator slack. The source SA
        // workspace needs six i32 entries per alphabet symbol.
        let scratch_bytes = size
            .checked_mul(80)?
            .checked_add(extra.checked_mul(std::mem::size_of::<i32>())?)?
            .checked_add(1 << 20)?;
        Some(Self {
            alphabet: i32::try_from(alphabet).ok()?,
            extra: i32::try_from(extra).ok()?,
            suffix_entries,
            scratch_bytes,
        })
    }
}

pub(super) fn suffix_array(values: &[u32], budget: usize) -> Result<Option<Vec<u32>>> {
    build_with(values, budget, libsais_rs::libsais_int)
}

fn build_with(
    values: &[u32],
    budget: usize,
    backend: impl FnOnce(&mut [i32], &mut [i32], i32, i32) -> i32,
) -> Result<Option<Vec<u32>>> {
    if values.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if values.len() > 1 << 24 {
        return Ok(None);
    }
    let largest = *values.iter().max().expect("nonempty source values");
    let Some(layout) = Layout::new(values.len(), largest) else {
        return Ok(None);
    };
    if layout.scratch_bytes > budget {
        return Ok(None);
    }
    // libsais_int mutates its input and only promises restoration on success.
    // Never expose caller tokens to that mutation, including on failure.
    let mut text = reserved(values.len())?;
    text.extend(values.iter().map(|&value| value as i32));
    let mut suffixes = filled(layout.suffix_entries, 0i32)?;
    if backend(&mut text, &mut suffixes, layout.alphabet, layout.extra) != 0 {
        return Ok(None);
    }
    drop(text);
    let mut output = reserved(values.len())?;
    for &position in &suffixes[..values.len()] {
        let Ok(position) = u32::try_from(position) else {
            return Ok(None);
        };
        if position as usize >= values.len() {
            return Ok(None);
        }
        output.push(position);
    }
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_order(values: &[u32]) -> Vec<u32> {
        let mut order: Vec<_> = (0..values.len() as u32).collect();
        order.sort_by(|&a, &b| values[a as usize..].cmp(&values[b as usize..]));
        order
    }

    #[test]
    fn source_sais_layout_preserves_limits_six_alphabet_tail_and_budget_boundary() {
        for (size, largest) in [(1, 0), (2, 1), (8193, 65535), (1 << 24, (1 << 20) - 1)] {
            let layout = Layout::new(size, largest).unwrap();
            assert_eq!(layout.alphabet as u32, largest + 1);
            assert_eq!(layout.extra, 6 * layout.alphabet);
            assert_eq!(layout.suffix_entries, size + layout.extra as usize);
            assert_eq!(
                layout.scratch_bytes,
                80 * size + 24 * (largest as usize + 1) + (1 << 20)
            );
            assert!(layout.suffix_entries <= i32::MAX as usize);
        }
        for (size, largest) in [
            (0, 0),
            ((1 << 24) + 1, 0),
            (usize::MAX, 0),
            (1, 1 << 20),
            (1, u32::MAX),
        ] {
            assert!(Layout::new(size, largest).is_none());
        }
        let values = [0, 3, 0, 3, 0];
        let bytes = Layout::new(values.len(), 3).unwrap().scratch_bytes;
        assert!(
            build_with(&values, bytes - 1, |_, _, _, _| panic!(
                "over budget backend"
            ))
            .unwrap()
            .is_none()
        );
        let result = build_with(&values, bytes, |text, suffixes, alphabet, extra| {
            assert_eq!(text, [0, 3, 0, 3, 0]);
            assert_eq!((alphabet, extra), (4, 24));
            assert_eq!(suffixes.len(), values.len() + extra as usize);
            suffixes[..5].copy_from_slice(&[4, 2, 0, 3, 1]);
            0
        })
        .unwrap()
        .unwrap();
        assert_eq!(result, scalar_order(&values));
        assert!(
            build_with(&[u32::MAX], usize::MAX, |_, _, _, _| panic!("wide backend"))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            build_with(&[], 0, |_, _, _, _| panic!("empty backend")).unwrap(),
            Some(vec![])
        );
    }

    #[test]
    fn source_sais_status_and_checked_mapping_preserve_static_fallback() {
        let values = [3, 1, 3, 1, 0];
        for status in [-1, -2] {
            assert!(
                build_with(&values, usize::MAX, |text, _, _, _| {
                    text.fill(i32::MIN);
                    status
                })
                .unwrap()
                .is_none()
            );
            assert_eq!(values, [3, 1, 3, 1, 0]);
        }
        for invalid in [-1, values.len() as i32, i32::MAX] {
            assert!(
                build_with(&values, usize::MAX, |_, suffixes, _, _| {
                    suffixes[0] = invalid;
                    0
                })
                .unwrap()
                .is_none()
            );
        }
    }

    #[test]
    fn serial_integer_sais_matches_scalar_suffix_order_on_recursive_and_wide_inputs() {
        let mut state = 0x159a_bcd3u32;
        for size in [
            1usize, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 255, 1025, 8193, 65537, 131073,
        ] {
            let values: Vec<_> = (0..size)
                .map(|p| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    if p % 97 < 16 { 0 } else { (state % 17) * 4097 }
                })
                .collect();
            let before = values.clone();
            let expected = scalar_order(&values);
            assert_eq!(
                suffix_array(&values, usize::MAX).unwrap().unwrap(),
                expected
            );
            assert_eq!(values, before);
        }
        for values in [
            vec![7; 1025],
            (0..1025).map(|p| p % 3).collect(),
            (0..1025).collect(),
            (0..1025).rev().collect(),
            vec![(1 << 20) - 1, 0, (1 << 20) - 1],
        ] {
            assert_eq!(
                suffix_array(&values, usize::MAX).unwrap().unwrap(),
                scalar_order(&values)
            );
        }
    }

    fn check_serial_helpers() {
        use libsais_rs::{
            clamp_lms_suffixes_length_32s, clamp_lms_suffixes_length_32s_omp,
            clear_lms_suffixes_omp, mark_distinct_lms_suffixes_32s,
            mark_distinct_lms_suffixes_32s_omp, partial_sorting_shift_markers_32s_6k_omp,
            reconstruct_lms_suffixes, reconstruct_lms_suffixes_omp,
        };

        for size in [0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 63, 64, 65, 131] {
            for threads in [1, 2] {
                // Both explicit serial calls and small-work serial fallbacks
                // must avoid Rayon, including empty buckets and four-lane tails.
                let initial: Vec<_> = (0..size)
                    .map(|i| i as i32 | if i.is_multiple_of(3) { i32::MIN } else { 0 })
                    .collect();
                let ends = [0, size / 3, 2 * size / 3, size];
                let mut buckets = [0; 24];
                let mut expected = initial.clone();
                for (bucket, bounds) in ends.windows(2).enumerate().rev() {
                    buckets[4 * (bucket + 1)] = bounds[1] as i32;
                    buckets[16 + 2 * bucket] = bounds[0] as i32;
                    let mut previous_sign = i32::MIN;
                    for value in expected[bounds[0]..bounds[1]].iter_mut().rev() {
                        let sign = *value & i32::MIN;
                        *value = (*value & i32::MAX) | previous_sign;
                        previous_sign = sign;
                    }
                }
                let mut actual = initial;
                partial_sorting_shift_markers_32s_6k_omp(&mut actual, 4, &buckets, threads);
                assert_eq!(actual, expected, "marker size={size} threads={threads}");

                let initial: Vec<_> = (0..size + 3)
                    .map(|i| match i % 4 {
                        0 => 0,
                        1 => i as i32,
                        _ => i as i32 | i32::MIN,
                    })
                    .collect();
                let n = (2 * size + 1) as i32;
                let mut expected = initial.clone();
                let mut actual = initial.clone();
                mark_distinct_lms_suffixes_32s(&mut expected, 3, 0, size as isize);
                mark_distinct_lms_suffixes_32s_omp(&mut actual, n, 3, threads);
                assert_eq!(actual, expected, "mark size={size} threads={threads}");
                let mut expected = initial.clone();
                let mut actual = initial;
                clamp_lms_suffixes_length_32s(&mut expected, 3, 0, size as isize);
                clamp_lms_suffixes_length_32s_omp(&mut actual, n, 3, threads);
                assert_eq!(actual, expected, "clamp size={size} threads={threads}");

                let n = 3 * size + 3;
                let mut initial = vec![-1; n];
                for (i, value) in initial[..size].iter_mut().enumerate() {
                    *value = (size - i - 1) as i32;
                }
                for (i, value) in initial[n - size..].iter_mut().enumerate() {
                    *value = (2 * i + 1) as i32;
                }
                let mut expected = initial.clone();
                let mut actual = initial;
                reconstruct_lms_suffixes(&mut expected, n as i32, size as i32, 0, size as isize);
                reconstruct_lms_suffixes_omp(&mut actual, n as i32, size as i32, threads);
                assert_eq!(
                    actual, expected,
                    "reconstruct size={size} threads={threads}"
                );

                let mut actual = [5, 4, 3, 2, 1, 9];
                clear_lms_suffixes_omp(&mut actual, 6, 3, &[1, 4, 5], &[3, 5, 5], threads);
                assert_eq!(actual, [5, 0, 0, 2, 0, 9]);
            }
        }
    }

    #[test]
    fn serial_integer_helpers_match_scalar_kernels_and_marker_order() {
        check_serial_helpers();
    }

    #[test]
    fn serial_sais_does_not_initialize_global_rayon_in_an_isolated_process() {
        const CHILD: &str = "RS_JXL_SAIS_SERIAL_PROBE";
        const VERIFIED: &str = "source-sais-global-pool-remained-uninitialized";
        if std::env::var_os(CHILD).is_some() {
            check_serial_helpers();
            // A fresh unit-test process prevents unrelated parallel tests from
            // initializing Rayon before this probe. Only after all SA calls do
            // we deliberately initialize a one-thread global pool as evidence.
            let mut state = 0x1357_abd1u32;
            let large: Vec<_> = (0..131073)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state % 131
                })
                .collect();
            for values in [
                vec![2, 1, 3, 1, 0],
                (0..8193).map(|p| ((p * 37) ^ (p / 11)) % 131).collect(),
                (0..8193).map(|p| p % 7).collect(),
                large,
            ] {
                let expected = scalar_order(&values);
                let actual = crate::execution::serial(|| suffix_array(&values, usize::MAX))
                    .unwrap()
                    .unwrap();
                assert_eq!(actual, expected);
            }
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build_global()
                .expect("serial SA-IS initialized the global Rayon pool");
            println!("{VERIFIED}");
            return;
        }
        let module = module_path!()
            .split_once("::")
            .expect("crate module path")
            .1;
        let test = format!(
            "{module}::serial_sais_does_not_initialize_global_rayon_in_an_isolated_process"
        );
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &test, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .output()
            .expect("start isolated SA-IS unit-test process");
        assert!(
            child.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        assert!(String::from_utf8_lossy(&child.stdout).contains(VERIFIED));
    }
}
