//! Small block/bit fixtures, not a complete JPEG-to-JXL encoder test.

use super::*;
use crate::jpeg::bitstream::{EntropyBitReader, PaddingPolicy};
use crate::jpeg::coefficients::{BlockInfo, BlockState, decode_initial_block, refine_block};

fn table(symbols: &[u8], length: usize) -> HuffmanTable {
    let mut counts = [0; 16];
    counts[length - 1] = symbols.len() as u8;
    HuffmanTable::new(&counts, symbols).unwrap()
}

fn dc_table() -> HuffmanTable {
    table(&(0..12).collect::<Vec<_>>(), 4)
}

fn ac_table() -> HuffmanTable {
    let mut counts = [0; 16];
    counts[8] = 255;
    counts[9] = 1;
    HuffmanTable::new(&counts, &(0..=255).collect::<Vec<_>>()).unwrap()
}

// Independent MSB packing for expected bytes. Symbols below use explicitly
// known canonical codewords, not the implementation's Huffman emission helpers.
fn packed(fields: &[(u32, u8)]) -> Vec<u8> {
    let mut bits = Vec::new();
    for &(value, count) in fields {
        for bit in (0..count).rev() {
            bits.push(((value >> bit) & 1) as u8);
        }
    }
    while !bits.len().is_multiple_of(8) {
        bits.push(1);
    }
    let mut bytes = Vec::new();
    for chunk in bits.chunks_exact(8) {
        let byte = chunk.iter().fold(0, |value, bit| (value << 1) | bit);
        bytes.push(byte);
        if byte == 255 {
            bytes.push(0);
        }
    }
    bytes
}

fn aligned(mut writer: EntropyBitWriter) -> Vec<u8> {
    writer.align_ones().unwrap();
    writer.into_bytes().unwrap()
}

fn terminated(bytes: &[u8]) -> Vec<u8> {
    let mut bytes = bytes.to_vec();
    bytes.extend([0xff, 0xd9]);
    bytes
}

fn ac_band(low_bit: u8) -> ScanBand {
    ScanBand {
        start: 1,
        end: 63,
        low_bit,
    }
}

#[test]
fn sequential_known_negative_difference_and_ac_bits() {
    let dc = table(&[0, 1, 2], 2);
    let ac = table(&[0, 1, 0xf0], 2);
    let mut block = [0; BLOCK_SIZE];
    block[0] = 2;
    block[NATURAL_ORDER[1]] = 1;
    let mut previous = 4;
    let mut writer = EntropyBitWriter::new(32);
    encode_sequential_block(&mut writer, &dc, &ac, 0, &mut previous, &block).unwrap();
    assert_eq!(previous, 2);
    let bytes = aligned(writer);
    assert_eq!(bytes, packed(&[(2, 2), (1, 2), (1, 2), (1, 1), (0, 2)]));
    let bytes = terminated(&bytes);
    let mut reader = EntropyBitReader::new(&bytes);
    let mut state = BlockState {
        last_dc: 4,
        ..Default::default()
    };
    let mut decoded = [0; BLOCK_SIZE];
    assert_eq!(
        decode_initial_block(
            &mut reader,
            Some(&dc),
            Some(&ac),
            ScanBand {
                start: 0,
                end: 63,
                low_bit: 0
            },
            &mut state,
            &mut decoded
        )
        .unwrap(),
        BlockInfo::default()
    );
    assert_eq!(decoded, block);
    reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
}

#[test]
fn explicit_trailing_zrls_include_the_frozen_readers_four_run_overshoot() {
    let dc = table(&[0], 1);
    let ac = table(&[0, 0xf0], 2);
    for extra in 0..=4 {
        let mut writer = EntropyBitWriter::new(32);
        encode_sequential_block(&mut writer, &dc, &ac, extra, &mut 0, &[0; BLOCK_SIZE]).unwrap();
        let mut fields = vec![(0, 1)];
        fields.extend(std::iter::repeat_n((1, 2), extra as usize));
        if extra < 4 {
            fields.push((0, 2));
        }
        let bytes = aligned(writer);
        assert_eq!(bytes, packed(&fields));
        let bytes = terminated(&bytes);
        let mut reader = EntropyBitReader::new(&bytes);
        let mut block = [0; BLOCK_SIZE];
        let info = decode_initial_block(
            &mut reader,
            Some(&dc),
            Some(&ac),
            ScanBand {
                start: 0,
                end: 63,
                low_bit: 0,
            },
            &mut BlockState::default(),
            &mut block,
        )
        .unwrap();
        assert_eq!(info.extra_zero_runs, extra);
        assert_eq!(block, [0; BLOCK_SIZE]);
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }
    // A narrow progressive band also permits exactly one final overshooting ZRL.
    let band = ScanBand {
        start: 8,
        end: 8,
        low_bit: 0,
    };
    let mut writer = EntropyBitWriter::new(32);
    let mut state = DctCodingState::new(0);
    encode_initial_block(
        &mut writer,
        None,
        Some(&ac),
        band,
        1,
        &mut state,
        &mut 0,
        &[0; BLOCK_SIZE],
    )
    .unwrap();
    state.flush(&mut writer).unwrap();
    assert_eq!(aligned(writer), packed(&[(1, 2)]));
}

#[test]
fn initial_dc_rounds_negative_down_while_ac_truncates_magnitude() {
    let dc = table(&[0, 1, 2], 2);
    let ac = table(&[0, 1], 2);
    let mut block = [0; BLOCK_SIZE];
    block[0] = -3;
    block[NATURAL_ORDER[1]] = -3;
    let mut state = DctCodingState::new(0);
    let mut writer = EntropyBitWriter::new(32);
    let mut previous = 0;
    encode_initial_block(
        &mut writer,
        Some(&dc),
        None,
        ScanBand {
            start: 0,
            end: 0,
            low_bit: 1,
        },
        0,
        &mut state,
        &mut previous,
        &block,
    )
    .unwrap();
    assert_eq!(previous, -2);
    assert_eq!(aligned(writer), packed(&[(2, 2), (1, 2)]));
    let mut writer = EntropyBitWriter::new(32);
    encode_initial_block(
        &mut writer,
        None,
        Some(&ac),
        ac_band(1),
        0,
        &mut state,
        &mut 0,
        &block,
    )
    .unwrap();
    state.flush(&mut writer).unwrap();
    assert_eq!(aligned(writer), packed(&[(1, 2), (0, 1), (0, 2)]));
}

#[test]
fn progressive_initial_and_two_refinement_levels_recover_final_coefficients() {
    let dc = dc_table();
    let ac = ac_table();
    let mut expected = [[0; BLOCK_SIZE]; 5];
    for (block, value) in expected.iter_mut().zip([-7, 9, 1, -3, 0]) {
        block[0] = value;
    }
    for (b, k, value) in [
        (0, 1, 5),
        (0, 2, -7),
        (0, 4, 1),
        (0, 20, -1),
        (0, 63, 3),
        (1, 1, -3),
        (1, 31, 1),
        (1, 62, -1),
        (4, 2, 7),
        (4, 16, -5),
    ] {
        expected[b][NATURAL_ORDER[k]] = value;
    }
    let mut decoded = [[0; BLOCK_SIZE]; 5];
    for low_bit in (0..=2).rev() {
        for band in [
            ScanBand {
                start: 0,
                end: 0,
                low_bit,
            },
            ac_band(low_bit),
        ] {
            let initial = low_bit == 2;
            let mut writer = EntropyBitWriter::new(4096);
            let mut coding = DctCodingState::new(512);
            let mut previous = 0;
            for block in &expected {
                if initial {
                    encode_initial_block(
                        &mut writer,
                        Some(&dc),
                        Some(&ac),
                        band,
                        0,
                        &mut coding,
                        &mut previous,
                        block,
                    )
                    .unwrap();
                } else {
                    encode_refinement_block(&mut writer, Some(&ac), band, &mut coding, block)
                        .unwrap();
                }
            }
            coding.flush(&mut writer).unwrap();
            let bytes = terminated(&aligned(writer));
            let mut reader = EntropyBitReader::new(&bytes);
            let mut state = BlockState::default();
            for block in &mut decoded {
                if initial {
                    decode_initial_block(
                        &mut reader,
                        Some(&dc),
                        Some(&ac),
                        band,
                        &mut state,
                        block,
                    )
                    .unwrap();
                } else {
                    refine_block(&mut reader, Some(&ac), band, &mut state, block).unwrap();
                }
            }
            assert!(state.eob_run <= 0);
            reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
            if band.start == 1 {
                for (actual, expected) in decoded.iter().zip(&expected) {
                    assert_eq!(actual[0], (expected[0] >> low_bit) << low_bit);
                    for (k, (&actual, &expected)) in actual.iter().zip(expected).enumerate().skip(1)
                    {
                        let magnitude = (i32::from(expected).abs() >> low_bit) << low_bit;
                        let value = if expected < 0 { -magnitude } else { magnitude };
                        assert_eq!(
                            i32::from(actual),
                            value,
                            "low_bit={low_bit}, natural coefficient={k}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(decoded, expected);
}

#[test]
fn reset_point_flush_preserves_adjacent_eob_symbols_instead_of_grouping() {
    let ac = table(&[0, 0x10], 2);
    for reset in [false, true] {
        let mut writer = EntropyBitWriter::new(32);
        let mut state = DctCodingState::new(0);
        for block in 0..2 {
            if reset && block == 1 {
                state.flush(&mut writer).unwrap();
            }
            encode_initial_block(
                &mut writer,
                None,
                Some(&ac),
                ac_band(0),
                0,
                &mut state,
                &mut 0,
                &[0; BLOCK_SIZE],
            )
            .unwrap();
        }
        state.flush(&mut writer).unwrap();
        let bytes = aligned(writer);
        let fields = if reset {
            vec![(0, 2), (0, 2)]
        } else {
            vec![(1, 2), (0, 1)]
        };
        assert_eq!(bytes, packed(&fields));
        let bytes = terminated(&bytes);
        let mut reader = EntropyBitReader::new(&bytes);
        let mut state = BlockState::default();
        for block in 0..2 {
            let info = decode_initial_block(
                &mut reader,
                None,
                Some(&ac),
                ac_band(0),
                &mut state,
                &mut [0; BLOCK_SIZE],
            )
            .unwrap();
            assert_eq!(info.reset_eob_state, reset && block == 1);
        }
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }
}

#[test]
fn buffered_refinement_bits_pack_across_words_and_multiple_blocks() {
    let ac = ac_table();
    let mut expected = [[0; BLOCK_SIZE]; 5];
    let mut decoded = [[0; BLOCK_SIZE]; 5];
    let mut fields = vec![(0x20, 9), (1, 2)]; // Five-block EOB run: symbol 20, suffix 01.
    for (b, (expected, decoded)) in expected.iter_mut().zip(&mut decoded).enumerate() {
        for (k, &natural) in NATURAL_ORDER[..BLOCK_SIZE].iter().enumerate().skip(1) {
            let bit = ((b + k) & 1) as i16;
            let sign = if k % 3 == 0 { -1 } else { 1 };
            expected[natural] = sign * (2 + bit);
            decoded[natural] = sign * 2;
            fields.push((bit as u32, 1));
        }
    }
    let mut writer = EntropyBitWriter::new(1024);
    let mut coding = DctCodingState::new(5 * 63);
    for block in &expected {
        encode_refinement_block(&mut writer, Some(&ac), ac_band(0), &mut coding, block).unwrap();
    }
    assert!(writer.bytes().is_empty());
    assert_eq!(coding.pending_eob_run(), 5);
    assert_eq!(coding.pending_refinement_bits(), 315);
    coding.flush(&mut writer).unwrap();
    assert_eq!(
        (coding.pending_eob_run(), coding.pending_refinement_bits()),
        (0, 0)
    );
    let bytes = aligned(writer);
    assert_eq!(bytes, packed(&fields));
    let bytes = terminated(&bytes);
    let mut reader = EntropyBitReader::new(&bytes);
    let mut state = BlockState::default();
    for block in &mut decoded {
        refine_block(&mut reader, Some(&ac), ac_band(0), &mut state, block).unwrap();
    }
    assert_eq!(decoded, expected);
    reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
}

#[test]
fn refinement_zrl_sign_and_correction_order_matches_known_bits() {
    let ac = ac_table();
    let band = ScanBand {
        start: 1,
        end: 20,
        low_bit: 0,
    };
    let mut block = [0; BLOCK_SIZE];
    block[NATURAL_ORDER[1]] = 3;
    block[NATURAL_ORDER[18]] = -1;
    block[NATURAL_ORDER[20]] = -3;
    let mut writer = EntropyBitWriter::new(128);
    let mut coding = DctCodingState::new(64);
    encode_refinement_block(&mut writer, Some(&ac), band, &mut coding, &block).unwrap();
    coding.flush(&mut writer).unwrap();
    let bytes = aligned(writer);
    assert_eq!(
        bytes,
        packed(&[(0xf0, 9), (1, 1), (1, 9), (0, 1), (0, 9), (1, 1)])
    );
    let bytes = terminated(&bytes);
    let mut reader = EntropyBitReader::new(&bytes);
    let mut decoded = [0; BLOCK_SIZE];
    decoded[NATURAL_ORDER[1]] = 2;
    decoded[NATURAL_ORDER[20]] = -2;
    refine_block(
        &mut reader,
        Some(&ac),
        band,
        &mut BlockState::default(),
        &mut decoded,
    )
    .unwrap();
    assert_eq!(decoded, block);
    reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
}

#[test]
fn dc_refinement_needs_no_huffman_table_and_preserves_signed_bit_semantics() {
    let mut block = [0; BLOCK_SIZE];
    block[0] = -3;
    let band = ScanBand {
        start: 0,
        end: 0,
        low_bit: 0,
    };
    let mut writer = EntropyBitWriter::new(8);
    let mut coding = DctCodingState::new(0);
    encode_refinement_block(&mut writer, None, band, &mut coding, &block).unwrap();
    coding.flush(&mut writer).unwrap();
    let bytes = aligned(writer);
    assert_eq!(bytes, [0xff, 0]);
    let bytes = terminated(&bytes);
    let mut reader = EntropyBitReader::new(&bytes);
    let mut decoded = [0; BLOCK_SIZE];
    decoded[0] = -4;
    refine_block(
        &mut reader,
        None,
        band,
        &mut BlockState::default(),
        &mut decoded,
    )
    .unwrap();
    assert_eq!(decoded, block);
    reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
}

#[test]
fn maximum_eob_run_flushes_at_source_boundary_without_large_allocation() {
    let ac = table(&[0xe0], 1);
    let mut writer = EntropyBitWriter::new(16);
    let mut coding = DctCodingState::new(0);
    for _ in 0..MAX_EOB_RUN {
        encode_initial_block(
            &mut writer,
            None,
            Some(&ac),
            ac_band(0),
            0,
            &mut coding,
            &mut 0,
            &[0; BLOCK_SIZE],
        )
        .unwrap();
    }
    assert_eq!(coding.pending_eob_run(), 0);
    assert!(coding.refinement_words.is_empty());
    coding.flush(&mut writer).unwrap();
    assert_eq!(aligned(writer), packed(&[(0, 1), (0x3fff, 14)]));
}

#[test]
fn buffered_eob_snapshot_outlives_table_and_detects_midrun_redefinition() {
    let mut writer = EntropyBitWriter::new(32);
    let mut coding = DctCodingState::new(0);
    {
        let original = table(&[0x10, 0], 2);
        encode_initial_block(
            &mut writer,
            None,
            Some(&original),
            ac_band(0),
            0,
            &mut coding,
            &mut 0,
            &[0; BLOCK_SIZE],
        )
        .unwrap();
    }
    coding.flush(&mut writer).unwrap();
    assert_eq!(aligned(writer), packed(&[(1, 2)]));
    let mut writer = EntropyBitWriter::new(32);
    let old = table(&[0x10, 0], 2);
    let new = table(&[0, 0x10], 2);
    encode_initial_block(
        &mut writer,
        None,
        Some(&old),
        ac_band(0),
        0,
        &mut coding,
        &mut 0,
        &[0; BLOCK_SIZE],
    )
    .unwrap();
    assert!(matches!(
        encode_initial_block(
            &mut writer,
            None,
            Some(&new),
            ac_band(0),
            0,
            &mut coding,
            &mut 0,
            &[0; BLOCK_SIZE]
        ),
        Err(CoefficientWriteError::Invalid(_))
    ));
    assert_eq!(
        coding.flush(&mut writer),
        Err(CoefficientWriteError::FailedState)
    );
}

#[test]
fn refinement_limit_rejects_instead_of_changing_eob_grouping() {
    let ac = table(&[0], 1);
    let block = [3; BLOCK_SIZE];
    for limit in [0, 62, 63] {
        let mut writer = EntropyBitWriter::new(128);
        let mut coding = DctCodingState::new(limit);
        if limit == 63 {
            encode_refinement_block(&mut writer, Some(&ac), ac_band(0), &mut coding, &block)
                .unwrap();
            assert_eq!(
                (coding.pending_eob_run(), coding.pending_refinement_bits()),
                (1, 63)
            );
        }
        assert_eq!(
            encode_refinement_block(&mut writer, Some(&ac), ac_band(0), &mut coding, &block),
            Err(CoefficientWriteError::RefinementLimit { limit })
        );
        assert!(writer.bytes().is_empty());
        assert_eq!(writer.pending_bits(), 0);
        assert_eq!(
            coding.flush(&mut writer),
            Err(CoefficientWriteError::FailedState)
        );
    }
    let mut coding = DctCodingState::new(usize::MAX);
    assert_eq!(coding.refinement_words.capacity(), 0);
    encode_refinement_block(
        &mut EntropyBitWriter::new(0),
        Some(&ac),
        ac_band(0),
        &mut coding,
        &block,
    )
    .unwrap();
    assert!(coding.refinement_words.capacity() <= MAX_BUFFERED_BITS.div_ceil(16));
}

#[test]
fn output_failure_poisoning_prevents_partial_flush_retry() {
    let ac = table(&[0], 1);
    let mut coding = DctCodingState::new(63);
    let mut writer = EntropyBitWriter::new(0);
    encode_refinement_block(
        &mut writer,
        Some(&ac),
        ac_band(0),
        &mut coding,
        &[3; BLOCK_SIZE],
    )
    .unwrap();
    assert!(matches!(
        coding.flush(&mut writer),
        Err(CoefficientWriteError::Entropy(WriteError::OutputLimit {
            limit: 0
        }))
    ));
    // The one-bit EOB fits the pending byte; the subsequent corrections fail.
    assert_eq!(writer.pending_bits(), 1);
    assert!(writer.bytes().is_empty());
    assert_eq!(
        coding.flush(&mut EntropyBitWriter::new(1024)),
        Err(CoefficientWriteError::FailedState)
    );
    assert_eq!(
        encode_refinement_block(
            &mut writer,
            Some(&ac),
            ac_band(0),
            &mut coding,
            &[0; BLOCK_SIZE]
        ),
        Err(CoefficientWriteError::FailedState)
    );
}

#[test]
fn missing_eob_symbol_is_an_error_not_an_implicit_split_or_early_flush() {
    let ac = table(&[0], 1);
    let mut writer = EntropyBitWriter::new(64);
    let mut coding = DctCodingState::new(0);
    for _ in 0..2 {
        encode_initial_block(
            &mut writer,
            None,
            Some(&ac),
            ac_band(0),
            0,
            &mut coding,
            &mut 0,
            &[0; BLOCK_SIZE],
        )
        .unwrap();
    }
    assert_eq!(
        coding.flush(&mut writer),
        Err(CoefficientWriteError::Entropy(
            WriteError::MissingHuffmanSymbol(0x10)
        ))
    );
    assert!(writer.bytes().is_empty());
}

#[test]
fn invalid_bands_tables_categories_and_hints_fail_before_emission() {
    let dc = dc_table();
    let ac = ac_table();
    for band in [
        ScanBand {
            start: 8,
            end: 7,
            low_bit: 0,
        },
        ScanBand {
            start: 1,
            end: 64,
            low_bit: 0,
        },
        ScanBand {
            start: 0,
            end: 0,
            low_bit: 11,
        },
    ] {
        let mut writer = EntropyBitWriter::new(64);
        assert!(
            encode_initial_block(
                &mut writer,
                Some(&dc),
                Some(&ac),
                band,
                0,
                &mut DctCodingState::new(64),
                &mut 0,
                &[0; BLOCK_SIZE]
            )
            .is_err()
        );
        assert_eq!(writer, EntropyBitWriter::new(64));
        assert!(
            encode_refinement_block(
                &mut writer,
                Some(&ac),
                band,
                &mut DctCodingState::new(64),
                &[0; BLOCK_SIZE]
            )
            .is_err()
        );
        assert_eq!(writer, EntropyBitWriter::new(64));
    }
    for (dc, ac) in [(None, Some(&ac)), (Some(&dc), None)] {
        let mut writer = EntropyBitWriter::new(64);
        assert!(
            encode_initial_block(
                &mut writer,
                dc,
                ac,
                ScanBand {
                    start: 0,
                    end: 63,
                    low_bit: 0
                },
                0,
                &mut DctCodingState::new(64),
                &mut 0,
                &[0; BLOCK_SIZE]
            )
            .is_err()
        );
        assert_eq!(writer, EntropyBitWriter::new(64));
    }
    for (dc_value, ac_value, previous, extra) in [
        (i16::MAX, 0, i16::MIN, 0),
        (0, 2048, 0, 0),
        (0, i16::MIN, 0, 0),
        (0, 0, 0, 5),
        (0, 0, 0, u32::MAX),
    ] {
        let mut block = [0; BLOCK_SIZE];
        block[0] = dc_value;
        block[NATURAL_ORDER[1]] = ac_value;
        let mut last = previous;
        let mut writer = EntropyBitWriter::new(64);
        assert!(encode_sequential_block(&mut writer, &dc, &ac, extra, &mut last, &block).is_err());
        assert_eq!(last, previous);
        assert_eq!(writer, EntropyBitWriter::new(64));
    }
}

#[test]
fn signed_i16_dc_endpoints_work_when_the_difference_fits() {
    let dc = dc_table();
    let ac = ac_table();
    for value in [i16::MIN, i16::MAX] {
        let previous = if value < 0 { value + 1 } else { value - 1 };
        let mut block = [0; BLOCK_SIZE];
        block[0] = value;
        block[NATURAL_ORDER[1]] = -2047;
        let mut writer = EntropyBitWriter::new(64);
        let mut predictor = previous;
        encode_sequential_block(&mut writer, &dc, &ac, 0, &mut predictor, &block).unwrap();
        let bytes = terminated(&aligned(writer));
        let mut reader = EntropyBitReader::new(&bytes);
        let mut decoded = [0; BLOCK_SIZE];
        decode_initial_block(
            &mut reader,
            Some(&dc),
            Some(&ac),
            ScanBand {
                start: 0,
                end: 63,
                low_bit: 0,
            },
            &mut BlockState {
                last_dc: previous,
                ..Default::default()
            },
            &mut decoded,
        )
        .unwrap();
        assert_eq!(decoded, block);
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }
}
