use super::*;
use crate::jpeg::coefficient_reader::read_coefficients;
use crate::jpeg::transcode_reference_tests::fixtures;

#[test]
fn partition_boundaries_do_not_overflow_intermediate_products() {
    for total in [8_388_608, u32::MAX as usize, usize::MAX] {
        for parts in [1, 8, 512] {
            let mut previous = 0;
            for part in 0..=parts {
                let end = partition_end(total, part, parts);
                assert!(end >= previous && end <= total);
                let width = end - previous;
                if part > 0 {
                    assert!(width == total / parts || width == total / parts + 1);
                }
                previous = end;
            }
            assert_eq!(previous, total);
        }
    }
}

#[test]
fn fragmented_bits_join_across_every_pending_alignment_and_stuffing_boundary() {
    for prefix in 0..8 {
        for width in 0..=32 {
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            for word in [0, u32::MAX, 0x55ff_017f, 0xffff_ffff, 0xff00_ff00] {
                let mut original = EntropyBitWriter::new(4096);
                original.write_bits((1 << prefix) - 1, prefix).unwrap();
                let mut joined = original.clone();
                for repeat in 0..7 {
                    let value = word.rotate_left(repeat) & mask;
                    original.write_bits(value, width).unwrap();
                    let mut part = EntropyBitWriter::new(4096);
                    part.write_bits(value, width).unwrap();
                    let tail_bits = part.pending_bits();
                    part.align_with(0, part.padding_bits_needed()).unwrap();
                    append_fragment(
                        &mut joined,
                        &Fragment {
                            bytes: part.into_bytes().unwrap(),
                            tail_bits,
                            padding_bits: 0,
                        },
                    )
                    .unwrap();
                }
                assert_eq!(joined, original);
            }
        }
    }
}

#[test]
fn word_merge_handles_long_runs_and_new_stuffing_after_bit_shifts() {
    for pending in 0..8 {
        for tail in 0..8 {
            for value in [0, u32::MAX, 0xfefe_fefe, 0x0101_0101, 0x1234_5678] {
                let mut expected = EntropyBitWriter::new(4096);
                expected.write_bits((1 << pending) - 1, pending).unwrap();
                let mut joined = expected.clone();
                let mut part = EntropyBitWriter::new(4096);
                for _ in 0..64 {
                    part.write_bits(value, 32).unwrap();
                    expected.write_bits(value, 32).unwrap();
                }
                part.write_bits((1 << tail) - 1, tail).unwrap();
                expected.write_bits((1 << tail) - 1, tail).unwrap();
                part.align_with(0, part.padding_bits_needed()).unwrap();
                let fragment = Fragment {
                    bytes: part.into_bytes().unwrap(),
                    tail_bits: tail,
                    padding_bits: 0,
                };
                append_fragment(&mut joined, &fragment).unwrap();
                assert_eq!(joined, expected);
                let mut limited = EntropyBitWriter::new(expected.bytes().len() - 1);
                limited.write_bits((1 << pending) - 1, pending).unwrap();
                assert!(append_fragment(&mut limited, &fragment).is_err());
            }
        }
    }
}

#[test]
fn fragments_restore_dc_history_across_rows_and_subsampled_mcus() {
    for (name, original) in fixtures::fixtures() {
        let mut jpeg = read_coefficients(&original).unwrap();
        if jpeg.coding_process == Some(CodingProcess::Progressive) {
            continue;
        }
        // Exercise each sampling geometry as one continuous scan here; the
        // independent restart-boundary test covers reset predictors and markers.
        jpeg.scan_info[0].restart_interval = 0;
        jpeg.padding_bits.clear();
        jpeg.has_zero_padding_bit = false;
        let scan = &jpeg.scan_info[0];
        let mcus = scan.mcu_cols as usize * scan.mcu_rows as usize;
        let dc = std::array::from_fn(|slot| {
            scan.dc_huffman_indices[slot].map(|i| table(&jpeg.huffman_code[i]).unwrap())
        });
        let ac = std::array::from_fn(|slot| {
            scan.ac_huffman_indices[slot].map(|i| table(&jpeg.huffman_code[i]).unwrap())
        });
        let expected = encode_scan(&jpeg, 0, &mut 0, &WriteLimits::default(), 2 << 20).unwrap();
        for jobs in [1, 2, 3, 8, 17] {
            let mut joined = EntropyBitWriter::new(2 << 20);
            for job in 0..jobs {
                let part = fragment(
                    &jpeg,
                    0,
                    &dc,
                    &ac,
                    job * mcus / jobs..(job + 1) * mcus / jobs,
                    2 << 20,
                )
                .unwrap();
                append_fragment(&mut joined, &part).unwrap();
            }
            joined.align_ones().unwrap();
            assert_eq!(
                joined.into_bytes().unwrap(),
                expected,
                "{name}, jobs={jobs}"
            );
        }
    }
}

#[test]
fn progressive_scans_keep_original_bytes_restarts_and_padding_budget() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for (name, original) in fixtures::fixtures()
        .into_iter()
        .chain(fixtures::tier_fixtures())
    {
        let mut jpeg = read_coefficients(&original).unwrap();
        if jpeg.coding_process != Some(CodingProcess::Progressive) {
            continue;
        }
        if jpeg.padding_bits.iter().any(|&bit| bit != 1) {
            continue;
        }
        let padding = jpeg.padding_bits.len();
        jpeg.padding_bits.clear();
        jpeg.has_zero_padding_bit = false;
        let limits = WriteLimits::default();
        validate(&jpeg, &limits).unwrap();
        let output = progressive_scans(&jpeg, &limits, &pool).unwrap().unwrap();
        assert_eq!(output, original, "{name}");
        let exact = WriteLimits {
            max_output_bytes: original.len(),
            ..limits.clone()
        };
        assert_eq!(
            reconstruct_with_pool(&jpeg, &exact, Some(&pool)).unwrap(),
            original
        );
        let short = WriteLimits {
            max_output_bytes: original.len() - 1,
            ..limits.clone()
        };
        assert!(reconstruct_with_pool(&jpeg, &short, Some(&pool)).is_err());
        if padding > 0 {
            let short = WriteLimits {
                max_padding_bits: padding - 1,
                ..limits
            };
            assert!(reconstruct_with_pool(&jpeg, &short, Some(&pool)).is_err());
        }
    }
}

fn large_zero_fixture(progressive: bool) -> Vec<u8> {
    large_zero_fixture_with_ac_bits(progressive, 1)
}

fn large_zero_fixture_with_ac_bits(progressive: bool, ac_bits: usize) -> Vec<u8> {
    fn segment(out: &mut Vec<u8>, marker: u8, data: &[u8]) {
        out.extend([255, marker]);
        out.extend(((data.len() + 2) as u16).to_be_bytes());
        out.extend(data);
    }
    let mut out = vec![255, 0xd8];
    let mut quant = vec![0];
    quant.extend([1; 64]);
    segment(&mut out, 0xdb, &quant);
    segment(
        &mut out,
        if progressive { 0xc2 } else { 0xc0 },
        &[8, 2, 0, 2, 0, 1, 1, 0x11, 0],
    );
    let mut tables = Vec::new();
    for selector in [0, 0x10] {
        tables.push(selector);
        let mut counts = [0; 16];
        counts[if selector == 0 { 0 } else { ac_bits - 1 }] = 1;
        tables.extend(counts);
        tables.push(0);
    }
    segment(&mut out, 0xc4, &tables);
    if progressive {
        segment(&mut out, 0xda, &[1, 1, 0, 0, 0, 0]);
        out.extend([0; 512]);
        segment(&mut out, 0xda, &[1, 1, 0, 1, 63, 0]);
        out.extend(vec![0; 512 * ac_bits]);
    } else {
        segment(&mut out, 0xda, &[1, 1, 0, 0, 63, 0]);
        out.extend(vec![0; 512 * (1 + ac_bits)]);
    }
    out.extend([255, 0xd9]);
    out
}

#[test]
fn skewed_progressive_scan_budget_falls_back_without_rejecting_valid_output() {
    let original = large_zero_fixture_with_ac_bits(true, 8);
    let jpeg = read_coefficients(&original).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let limits = WriteLimits {
        max_output_bytes: original.len(),
        ..Default::default()
    };
    assert!(progressive(&jpeg, &limits, &pool).unwrap().is_none());
    assert_eq!(
        reconstruct_with_pool(&jpeg, &limits, Some(&pool)).unwrap(),
        original
    );
}

#[test]
fn restart_fragments_reset_nonzero_dc_in_each_sampling_geometry() {
    let mut covered = 0;
    for (name, original) in fixtures::fixtures() {
        let mut jpeg = read_coefficients(&original).unwrap();
        let scan = &jpeg.scan_info[0];
        let interval = scan.restart_interval as usize;
        if jpeg.coding_process == Some(CodingProcess::Progressive) || interval == 0 {
            continue;
        }
        if jpeg
            .components
            .iter()
            .all(|c| c.coefficients.chunks_exact(64).all(|b| b[0] == 0))
        {
            continue;
        }
        covered += 1;
        let mcus = scan.mcu_cols as usize * scan.mcu_rows as usize;
        let units = mcus.div_ceil(interval);
        let dc = std::array::from_fn(|slot| {
            scan.dc_huffman_indices[slot].map(|i| table(&jpeg.huffman_code[i]).unwrap())
        });
        let ac = std::array::from_fn(|slot| {
            scan.ac_huffman_indices[slot].map(|i| table(&jpeg.huffman_code[i]).unwrap())
        });
        jpeg.padding_bits.clear();
        jpeg.has_zero_padding_bit = false;
        let mut expected_padding = 0;
        let expected = encode_scan(
            &jpeg,
            0,
            &mut expected_padding,
            &WriteLimits::default(),
            2 << 20,
        )
        .unwrap();
        for jobs in [1, 2, 3, 8] {
            let jobs = jobs.min(units);
            let mut joined = Vec::new();
            let mut padding = 0;
            for job in 0..jobs {
                let part = fragment(
                    &jpeg,
                    0,
                    &dc,
                    &ac,
                    partition_end(units, job, jobs) * interval
                        ..(partition_end(units, job + 1, jobs) * interval).min(mcus),
                    2 << 20,
                )
                .unwrap();
                joined.extend_from_slice(&part.bytes);
                padding += part.padding_bits;
            }
            assert_eq!(joined, expected, "{name}, jobs={jobs}");
            assert_eq!(padding, expected_padding);
        }
    }
    assert!(covered > 0);
}

#[test]
fn production_parallel_gates_and_tight_limits_preserve_original_bytes() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for progressive_mode in [false, true] {
        let original = large_zero_fixture(progressive_mode);
        let jpeg = read_coefficients(&original).unwrap();
        assert!(implicit_one_padding(&jpeg));
        validate(&jpeg, &WriteLimits::default()).unwrap();
        if progressive_mode {
            assert!(
                progressive(&jpeg, &WriteLimits::default(), &pool)
                    .unwrap()
                    .is_some()
            );
        } else {
            assert!(
                sequential(&jpeg, 0, &mut 0, &WriteLimits::default(), 2048, &pool)
                    .unwrap()
                    .is_some()
            );
            assert!(
                sequential(&jpeg, 0, &mut 0, &WriteLimits::default(), 1, &pool)
                    .unwrap()
                    .is_none()
            );
        }
        for budget in [original.len(), original.len() + 1, 2 << 20] {
            let limits = WriteLimits {
                max_output_bytes: budget,
                ..Default::default()
            };
            assert_eq!(
                reconstruct_with_pool(&jpeg, &limits, Some(&pool)).unwrap(),
                original
            );
        }
        let limits = WriteLimits {
            max_output_bytes: original.len() - 1,
            ..Default::default()
        };
        assert!(reconstruct_with_pool(&jpeg, &limits, Some(&pool)).is_err());
        let mut invalid = jpeg.clone();
        invalid.components[0].coefficients[64 * 3000 + 1] = 2048;
        assert!(reconstruct_with_pool(&invalid, &WriteLimits::default(), Some(&pool)).is_err());
    }
}

#[test]
fn parallel_restart_segments_keep_fill_order_partial_interval_and_padding_limits() {
    let baseline = large_zero_fixture(false);
    let sos = baseline
        .windows(2)
        .position(|pair| pair == [255, 0xda])
        .unwrap();
    let mut original = baseline[..sos].to_vec();
    original.extend([255, 0xdd, 0, 4, 0, 31]);
    original.extend_from_slice(&baseline[sos..sos + 10]);
    let first_entropy = original.len();
    for (index, first) in (0usize..4096).step_by(31).enumerate() {
        let count = (4096 - first).min(31);
        original.extend(vec![0; count / 4]);
        if count % 4 != 0 {
            original.push((1u8 << (8 - (count % 4) * 2)) - 1);
        }
        if first + count != 4096 {
            original.extend([255, 255, 0xd0 + (index % 8) as u8]);
        }
    }
    original.extend([255, 0xd9]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut jpeg = read_coefficients(&original).unwrap();
    let required_padding = jpeg.padding_bits.len();
    assert!(required_padding > 0);
    assert!(
        sequential(&jpeg, 0, &mut 0, &WriteLimits::default(), 2 << 20, &pool)
            .unwrap()
            .is_none()
    );
    jpeg.padding_bits.clear();
    jpeg.has_zero_padding_bit = false;
    let mut padding = 0;
    assert!(
        sequential(
            &jpeg,
            0,
            &mut padding,
            &WriteLimits::default(),
            2 << 20,
            &pool
        )
        .unwrap()
        .is_some()
    );
    assert_eq!(padding, required_padding);
    for max_output_bytes in [original.len(), 2 << 20] {
        let limits = WriteLimits {
            max_output_bytes,
            ..Default::default()
        };
        assert_eq!(
            reconstruct_with_pool(&jpeg, &limits, Some(&pool)).unwrap(),
            original
        );
    }
    let limits = WriteLimits {
        max_padding_bits: required_padding - 1,
        ..Default::default()
    };
    assert!(reconstruct_with_pool(&jpeg, &limits, Some(&pool)).is_err());
    // A mixed padding bit retains the original serial fallback, never normalizes.
    original[first_entropy + 7] &= !1;
    let jpeg = read_coefficients(&original).unwrap();
    assert!(jpeg.has_zero_padding_bit);
    assert_eq!(
        reconstruct_with_pool(&jpeg, &WriteLimits::default(), Some(&pool)).unwrap(),
        original
    );
}
