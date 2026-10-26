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

fn sequential_matches_initial(
    writer: &mut EntropyBitWriter,
    dc: &HuffmanTable,
    ac: &HuffmanTable,
    extra_zero_runs: u32,
    previous: &mut i16,
    block: &[i16; BLOCK_SIZE],
) -> Result<(), CoefficientWriteError> {
    let mut expected_writer = writer.clone();
    let mut expected_previous = *previous;
    let expected = encode_initial_block(
        &mut expected_writer,
        Some(dc),
        Some(ac),
        ScanBand {
            start: 0,
            end: 63,
            low_bit: 0,
        },
        extra_zero_runs,
        &mut DctCodingState::new(0),
        &mut expected_previous,
        block,
    );
    let actual = encode_sequential_block(writer, dc, ac, extra_zero_runs, previous, block);
    assert_eq!(actual, expected, "extra={extra_zero_runs}, block={block:?}");
    assert_eq!(*previous, expected_previous);
    assert_eq!(*writer, expected_writer);
    // Compare exact stuffed JPEG bytes, including pending bits and mixed padding,
    // rather than just decoded coefficients. Also compare alignment-limit errors.
    let count = writer.padding_bits_needed();
    let mask = ((1u16 << count) - 1) as u8;
    for pattern in [0, 0x55, 0xff] {
        let mut padded = writer.clone();
        let mut expected_padded = expected_writer.clone();
        assert_eq!(
            padded.align_with(pattern & mask, count),
            expected_padded.align_with(pattern & mask, count)
        );
        assert_eq!(padded, expected_padded);
    }
    actual
}

// Pre-batch emission references. Fixtures below use validated coefficients;
// all entropy writes still happen at their original symbol boundaries.
fn scalar_sequential(
    writer: &mut EntropyBitWriter,
    dc: &HuffmanTable,
    ac: &HuffmanTable,
    extra: u32,
    previous: &mut i16,
    block: &[i16; BLOCK_SIZE],
) -> Result<(), CoefficientWriteError> {
    let (count, bits) = magnitude_bits(i32::from(block[0]) - i32::from(*previous))?;
    writer.write_symbol_bits(dc, count, u32::from(bits), count)?;
    *previous = block[0];
    let mut zeros = 0i32;
    for &natural in &NATURAL_ORDER[1..BLOCK_SIZE] {
        let value = i32::from(block[natural]);
        if value == 0 {
            zeros += 1;
            continue;
        }
        while zeros > 15 {
            writer.write_symbol(ac, 0xf0)?;
            zeros -= 16;
        }
        let (count, bits) = magnitude_bits(value)?;
        writer.write_symbol_bits(ac, ((zeros as u8) << 4) | count, u32::from(bits), count)?;
        zeros = 0;
    }
    for _ in 0..extra {
        writer.write_symbol(ac, 0xf0)?;
        zeros -= 16;
    }
    if zeros > 0 {
        writer.write_symbol(ac, 0)?;
    }
    Ok(())
}

fn scalar_flush(
    state: &mut DctCodingState,
    writer: &mut EntropyBitWriter,
) -> Result<(), CoefficientWriteError> {
    if state.eob_run == 0 {
        return Ok(());
    }
    let count = (15 - state.eob_run.leading_zeros()) as u8;
    let code =
        state.eob_codes[usize::from(count)].ok_or(WriteError::MissingHuffmanSymbol(count << 4))?;
    let suffix = u32::from(state.eob_run) & ((1 << count) - 1);
    writer.write_bits(
        (u32::from(code.bits) << count) | suffix,
        code.bit_length + count,
    )?;
    let mut remaining = state.refinement_bits;
    for &word in &state.refinement_words {
        let count = remaining.min(16) as u8;
        writer.write_bits(u32::from(word), count)?;
        remaining -= usize::from(count);
    }
    state.eob_run = 0;
    state.eob_codes = [None; 15];
    state.refinement_words.clear();
    state.refinement_bits = 0;
    Ok(())
}

fn scalar_buffer_end_of_band(
    state: &mut DctCodingState,
    writer: &mut EntropyBitWriter,
    ac: &HuffmanTable,
    bits: u64,
    count: u8,
) -> Result<(), CoefficientWriteError> {
    let codes = std::array::from_fn(|i| ac.code((i as u8) << 4));
    if state.eob_run != 0 && codes != state.eob_codes {
        return Err(CoefficientWriteError::Invalid(
            "AC table changed inside a pending EOB run",
        ));
    }
    let total = state
        .refinement_bits
        .checked_add(usize::from(count))
        .ok_or(CoefficientWriteError::RefinementLimit {
            limit: state.max_refinement_bits,
        })?;
    if total > state.max_refinement_bits {
        return Err(CoefficientWriteError::RefinementLimit {
            limit: state.max_refinement_bits,
        });
    }
    let words = total.div_ceil(16);
    if words > state.refinement_words.len() {
        state
            .refinement_words
            .try_reserve_exact(words - state.refinement_words.len())
            .map_err(|_| CoefficientWriteError::AllocationFailed)?;
    }
    if state.eob_run == 0 {
        state.eob_codes = codes;
    }
    state.eob_run += 1;
    // Intentionally bit-at-a-time, independent of the production chunk packer.
    for bit in (0..count).rev() {
        if state.refinement_bits.is_multiple_of(16) {
            state.refinement_words.push(0);
        }
        let word = state.refinement_words.last_mut().unwrap();
        *word = (*word << 1) | ((bits >> bit) & 1) as u16;
        state.refinement_bits += 1;
    }
    if state.eob_run == MAX_EOB_RUN {
        scalar_flush(state, writer)?;
    }
    Ok(())
}

fn scalar_prepare_refinement_band(
    block: &[i16; BLOCK_SIZE],
    band: ScanBand,
    magnitudes: &mut [i32; BLOCK_SIZE],
) -> Result<usize, CoefficientWriteError> {
    let mut last_new = 0;
    for k in usize::from(band.start.max(1))..=usize::from(band.end) {
        let value = shifted_ac(block[NATURAL_ORDER[k]], band.low_bit);
        let (category, _) = magnitude_bits(value)?;
        if category != 0 && category + band.low_bit >= 12 {
            return Err(CoefficientWriteError::Invalid(
                "AC coefficient exceeds supported precision",
            ));
        }
        magnitudes[k] = value.abs();
        if magnitudes[k] == 1 {
            last_new = k;
        }
    }
    Ok(last_new)
}

fn assert_refinement_preparation_matches_scalar(block: &[i16; BLOCK_SIZE], band: ScanBand) {
    let mut magnitudes = [0; BLOCK_SIZE];
    let expected = scalar_prepare_refinement_band(block, band, &mut magnitudes);
    match (prepare_refinement_band(block, band), expected) {
        (Err(actual), Err(expected)) => assert_eq!(actual, expected),
        (Ok((masks, last_new, tail)), Ok(expected_last)) => {
            let mut nonzero = 0u64;
            let mut newly_nonzero = 0u64;
            let mut correction = 0u64;
            let mut positive = 0u64;
            for k in usize::from(band.start.max(1))..=usize::from(band.end) {
                nonzero |= u64::from(magnitudes[k] != 0) << k;
                newly_nonzero |= u64::from(magnitudes[k] == 1) << k;
                correction |= ((magnitudes[k] & 1) as u64) << k;
                positive |= u64::from(block[NATURAL_ORDER[k]] > 0) << k;
            }
            assert_eq!(
                (
                    masks.nonzero,
                    masks.newly_nonzero,
                    masks.correction,
                    masks.positive,
                ),
                (nonzero, newly_nonzero, correction, positive),
                "band={band:?}, block={block:?}"
            );
            assert_eq!(last_new, expected_last);
            let tail_start = (expected_last + 1).max(usize::from(band.start.max(1)));
            let mut zeros = 0;
            let mut bits = 0u64;
            let mut count = 0;
            for &value in &magnitudes[tail_start..=usize::from(band.end)] {
                if value == 0 {
                    zeros += 1;
                } else {
                    assert!(value > 1);
                    bits = (bits << 1) | (value as u64 & 1);
                    count += 1;
                }
            }
            assert_eq!((tail.zeros, tail.bits, tail.count), (zeros, bits, count));
        }
        _ => panic!("refinement validation differs for band={band:?}, block={block:?}"),
    }
}

fn scalar_refinement_ac(
    writer: &mut EntropyBitWriter,
    ac: &HuffmanTable,
    band: ScanBand,
    state: &mut DctCodingState,
    block: &[i16; BLOCK_SIZE],
) -> Result<(), CoefficientWriteError> {
    state.healthy()?;
    let result = (|| {
        validate_band(band)?;
        if band.start == 0 && state.eob_run != 0 {
            return Err(CoefficientWriteError::Invalid(
                "pending AC state before DC refinement",
            ));
        }
        let mut magnitudes = [0; BLOCK_SIZE];
        let last_new = if band.end == 0 {
            0
        } else {
            scalar_prepare_refinement_band(block, band, &mut magnitudes)?
        };
        if band.start == 0 {
            writer.write_bits(((i32::from(block[0]) >> band.low_bit) & 1) as u32, 1)?;
        }
        if band.end == 0 {
            return Ok(());
        }
        let mut zeros = 0u8;
        let mut bits = 0u64;
        let mut count = 0u8;
        for k in usize::from(band.start.max(1))..=usize::from(band.end) {
            let magnitude = magnitudes[k];
            if magnitude == 0 {
                zeros += 1;
                continue;
            }
            while zeros > 15 && k <= last_new {
                scalar_flush(state, writer)?;
                writer.write_symbol(ac, 0xf0)?;
                zeros -= 16;
                write_corrections(writer, bits, count)?;
                bits = 0;
                count = 0;
            }
            if magnitude > 1 {
                bits = (bits << 1) | (magnitude as u64 & 1);
                count += 1;
                continue;
            }
            scalar_flush(state, writer)?;
            let sign = u32::from(block[NATURAL_ORDER[k]] > 0);
            writer.write_symbol_bits(ac, (zeros << 4) | 1, sign, 1)?;
            write_corrections(writer, bits, count)?;
            bits = 0;
            count = 0;
            zeros = 0;
        }
        if zeros != 0 || count != 0 {
            scalar_buffer_end_of_band(state, writer, ac, bits, count)?;
            if band.start == 0 {
                scalar_flush(state, writer)?;
            }
        }
        Ok(())
    })();
    state.finish_operation(result)
}

#[test]
fn initial_precision_bound_matches_category_validation_for_every_i16() {
    for low_bit in 0..=10 {
        let band = ScanBand {
            start: 23,
            end: 23,
            low_bit,
        };
        // Invalid coefficients outside the selected band must remain ignored.
        let mut block = [i16::MIN; BLOCK_SIZE];
        for value in i16::MIN..=i16::MAX {
            block[NATURAL_ORDER[23]] = value;
            let shifted = shifted_ac(value, low_bit);
            let (category, _) = magnitude_bits(shifted).unwrap();
            let expected = if category != 0 && category + low_bit >= 12 {
                Err(CoefficientWriteError::Invalid(
                    "AC coefficient exceeds supported precision",
                ))
            } else {
                Ok(usize::from(shifted == 0))
            };
            assert_eq!(validate_ac_band(&block, band), expected);
        }
    }
}

#[test]
fn refinement_preparation_preserves_band_validation_and_tail_order() {
    let mut rng = 0x42ec_9765u32;
    for low_bit in 0..=10 {
        for (start, end) in [(1, 63), (2, 19), (17, 47), (63, 63)] {
            let band = ScanBand {
                start,
                end,
                low_bit,
            };
            for sample in 0..80 {
                let mut block = [i16::MIN; BLOCK_SIZE];
                for &natural in &NATURAL_ORDER[usize::from(start)..=usize::from(end)] {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    block[natural] = (rng % 4095) as i16 - 2047;
                }
                if sample < 4 {
                    block[NATURAL_ORDER[usize::from(end)]] =
                        [-2048, 2048, i16::MIN, i16::MAX][sample];
                }
                assert_refinement_preparation_matches_scalar(&block, band);
            }
        }
    }
}

#[test]
fn correction_word_append_matches_bitwise_reference_at_every_word_offset() {
    let ac = ac_table();
    for prefix in 0..16u8 {
        for count in 0..=63u8 {
            for pattern in [0, u64::MAX, 0x5a5a_ff00_1234_abcd] {
                let bits = pattern & ((1u64 << count) - 1);
                let total = usize::from(prefix) + usize::from(count);
                for limit in [total, total.saturating_sub(1)] {
                    if limit < usize::from(prefix) {
                        continue;
                    }
                    let mut scalar = EntropyBitWriter::new(128);
                    scalar.write_bits(5, 3).unwrap();
                    let mut expected = DctCodingState::new(limit);
                    scalar_buffer_end_of_band(
                        &mut expected,
                        &mut scalar,
                        &ac,
                        (1u64 << prefix) - 1,
                        prefix,
                    )
                    .unwrap();
                    let mut actual = expected.clone();
                    let mut writer = scalar.clone();
                    let expected_result =
                        scalar_buffer_end_of_band(&mut expected, &mut scalar, &ac, bits, count);
                    let actual_result = actual.buffer_end_of_band(&mut writer, &ac, bits, count);
                    assert_eq!(actual_result, expected_result);
                    assert_eq!(actual, expected);
                    assert_eq!(writer, scalar);
                    if actual_result.is_ok() {
                        assert_eq!(
                            actual.flush(&mut writer),
                            scalar_flush(&mut expected, &mut scalar)
                        );
                        assert_eq!(actual, expected);
                        assert_eq!(writer, scalar);
                    }
                }
            }
        }
    }
}

#[test]
fn refinement_suffix_fast_path_matches_scalar_limits_and_maximum_eob_run() {
    let ac = ac_table();
    for low_bit in [0, 3, 10] {
        for (start, end) in [(0, 63), (1, 63), (17, 47), (63, 63)] {
            let band = ScanBand {
                start,
                end,
                low_bit,
            };
            let first = usize::from(start.max(1));
            let end = usize::from(end);
            for last_new in [None, Some(first), Some((first + end) / 2), Some(end)] {
                let mut block = [i16::MIN; BLOCK_SIZE];
                block[0] = -3;
                for (k, &natural) in NATURAL_ORDER.iter().enumerate().take(end + 1).skip(first) {
                    let value = [0i32, 2, -3, 3][k % 4] << low_bit;
                    block[natural] = if value.abs() < 2048 { value as i16 } else { 0 };
                }
                if let Some(k) = last_new {
                    block[NATURAL_ORDER[k]] = -(1 << low_bit);
                }
                for prior_run in [0, 1, MAX_EOB_RUN - 1] {
                    if start == 0 && prior_run != 0 {
                        continue;
                    }
                    for refinement_limit in [0, 63] {
                        let mut seed = DctCodingState::new(refinement_limit);
                        if prior_run != 0 {
                            scalar_buffer_end_of_band(
                                &mut seed,
                                &mut EntropyBitWriter::new(0),
                                &ac,
                                0,
                                0,
                            )
                            .unwrap();
                            seed.eob_run = prior_run;
                        }
                        for pending in [0, 3, 7] {
                            for limit in [0, 1, 2, 7, 8, 15, 16, 31, 128] {
                                let mut scalar = EntropyBitWriter::new(limit);
                                scalar.write_bits((1 << pending) - 1, pending).unwrap();
                                let mut writer = scalar.clone();
                                let mut expected = seed.clone();
                                let mut actual = seed.clone();
                                let expected_result = scalar_refinement_ac(
                                    &mut scalar,
                                    &ac,
                                    band,
                                    &mut expected,
                                    &block,
                                );
                                let actual_result = encode_refinement_block(
                                    &mut writer,
                                    Some(&ac),
                                    band,
                                    &mut actual,
                                    &block,
                                );
                                assert_eq!(actual_result, expected_result);
                                assert_eq!(writer, scalar);
                                assert_eq!(actual, expected);
                                if actual_result.is_ok() {
                                    let result = scalar_flush(&mut expected, &mut scalar);
                                    let result = expected.finish_operation(result);
                                    assert_eq!(actual.flush(&mut writer), result);
                                    assert_eq!(writer, scalar);
                                    assert_eq!(actual, expected);
                                } else {
                                    assert_eq!(
                                        actual.flush(&mut writer),
                                        Err(CoefficientWriteError::FailedState)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn pending_eob_checks_every_code_even_after_same_variable_table_replacement() {
    for pending_corrections in [0, 63] {
        for changed in 0..15 {
            let symbols: Vec<_> = (0..15).map(|i| i << 4).collect();
            let mut ac = table(&symbols, 4);
            let mut writer = EntropyBitWriter::new(128);
            writer.write_bits(5, 3).unwrap();
            let mut state = DctCodingState::new(pending_corrections);
            let coefficient = if pending_corrections == 0 { 0 } else { 3 };
            encode_refinement_block(
                &mut writer,
                Some(&ac),
                ac_band(0),
                &mut state,
                &[coefficient; BLOCK_SIZE],
            )
            .unwrap();
            assert_eq!(state.pending_refinement_bits(), pending_corrections);
            let before = state.clone();
            let before_writer = writer.clone();
            let mut changed_symbols = symbols;
            changed_symbols[changed] = 1;
            ac = table(&changed_symbols, 4);
            let mut scalar = writer.clone();
            let mut scalar_state = state.clone();
            let expected_result = scalar_refinement_ac(
                &mut scalar,
                &ac,
                ac_band(0),
                &mut scalar_state,
                &[3; BLOCK_SIZE],
            );
            // Table mismatch must win over the exhausted refinement budget,
            // without flushing or discarding any pending correction words.
            assert_eq!(
                expected_result,
                Err(CoefficientWriteError::Invalid(
                    "AC table changed inside a pending EOB run"
                ))
            );
            assert_eq!(
                encode_refinement_block(
                    &mut writer,
                    Some(&ac),
                    ac_band(0),
                    &mut state,
                    &[3; BLOCK_SIZE]
                ),
                expected_result
            );
            assert_eq!(state, scalar_state);
            assert_eq!(writer, scalar);
            let mut expected = before;
            expected.failed = true;
            assert_eq!(state, expected);
            assert_eq!(writer, before_writer);
            assert_eq!(
                state.flush(&mut writer),
                Err(CoefficientWriteError::FailedState)
            );
            assert_eq!(writer, before_writer);
        }
    }
    let mut ac = table(&[0, 0x10, 1, 2], 3);
    let mut writer = EntropyBitWriter::new(128);
    let mut state = DctCodingState::new(0);
    encode_refinement_block(
        &mut writer,
        Some(&ac),
        ac_band(0),
        &mut state,
        &[0; BLOCK_SIZE],
    )
    .unwrap();
    ac = table(&[0, 0x10, 2, 1], 3);
    encode_refinement_block(
        &mut writer,
        Some(&ac),
        ac_band(0),
        &mut state,
        &[0; BLOCK_SIZE],
    )
    .unwrap();
    state.flush(&mut writer).unwrap();
    assert_eq!(aligned(writer), packed(&[(1, 3), (0, 1)]));
}

#[test]
fn batched_sequential_matches_scalar_at_limits_and_missing_code_boundaries() {
    let dc = dc_table();
    let ac = ac_table();
    let incomplete = table(&[0, 1, 0x0b, 0xf0], 16);
    let mut sparse = [0; BLOCK_SIZE];
    sparse[0] = -1023;
    sparse[NATURAL_ORDER[1]] = 2047;
    sparse[NATURAL_ORDER[19]] = -1;
    sparse[NATURAL_ORDER[31]] = 255;
    let mut dense = [2047; BLOCK_SIZE];
    for k in (1..BLOCK_SIZE).step_by(2) {
        dense[k] = -2047;
    }
    for ac in [&ac, &incomplete] {
        for (block, extra) in [(&dense, 0), (&sparse, 2), (&[0; BLOCK_SIZE], 4)] {
            for pending in 0..8 {
                for limit in 0..=520 {
                    let mut scalar = EntropyBitWriter::new(limit);
                    scalar.write_bits((1 << pending) - 1, pending).unwrap();
                    let mut batched = scalar.clone();
                    let mut expected_dc = 0;
                    let mut actual_dc = 0;
                    let expected =
                        scalar_sequential(&mut scalar, &dc, ac, extra, &mut expected_dc, block);
                    let actual = encode_sequential_block(
                        &mut batched,
                        &dc,
                        ac,
                        extra,
                        &mut actual_dc,
                        block,
                    );
                    assert_eq!(actual, expected, "pending={pending}, limit={limit}");
                    assert_eq!(actual_dc, expected_dc);
                    assert_eq!(batched, scalar);
                }
            }
        }
    }
}

#[test]
fn batched_refinement_matches_scalar_with_pending_runs_and_partial_errors() {
    let full_ac = ac_table();
    let incomplete = table(&[0, 0x10, 0x20, 0xf0], 16);
    for ac in [&full_ac, &incomplete] {
        for low_bit in [0, 1, 6] {
            let band = ac_band(low_bit);
            let mut seed = DctCodingState::new(1024);
            let mut unused = EntropyBitWriter::new(0);
            // Pending EOB plus 126 correction bits exercises batching during
            // flush as well as the new block's sign/ZRL/correction emissions.
            for _ in 0..2 {
                encode_refinement_block(
                    &mut unused,
                    Some(ac),
                    band,
                    &mut seed,
                    &[3 << low_bit; BLOCK_SIZE],
                )
                .unwrap();
            }
            for layout in 0..3 {
                let mut block = [0; BLOCK_SIZE];
                for (k, &natural) in NATURAL_ORDER[..BLOCK_SIZE].iter().enumerate().skip(1) {
                    let magnitude = match layout {
                        0 => [0, 1, 3, -2, -1][k % 5],
                        1 => {
                            if k == 18 || k == 63 {
                                -1
                            } else {
                                0
                            }
                        }
                        _ => {
                            if k == 62 {
                                1
                            } else {
                                3
                            }
                        }
                    };
                    block[natural] = magnitude << low_bit;
                }
                for pending in 0..8 {
                    for limit in 0..=160 {
                        let mut scalar = EntropyBitWriter::new(limit);
                        scalar.write_bits((1 << pending) - 1, pending).unwrap();
                        let mut batched = scalar.clone();
                        let mut expected_state = seed.clone();
                        let mut actual_state = seed.clone();
                        let expected = scalar_refinement_ac(
                            &mut scalar,
                            ac,
                            band,
                            &mut expected_state,
                            &block,
                        );
                        let actual = encode_refinement_block(
                            &mut batched,
                            Some(ac),
                            band,
                            &mut actual_state,
                            &block,
                        );
                        assert_eq!(
                            actual, expected,
                            "layout={layout}, pending={pending}, limit={limit}"
                        );
                        assert_eq!(batched, scalar);
                        assert_eq!(actual_state, expected_state);
                    }
                }
            }
        }
    }
}

#[test]
fn refinement_adversarial_subbands_match_scalar_across_all_bit_planes() {
    let ac = ac_table();
    for low_bit in 0..=10 {
        let unit = 1i16 << low_bit;
        for (start, end) in [(1, 63), (2, 62), (7, 55), (16, 49), (32, 63), (63, 63)] {
            let band = ScanBand {
                start,
                end,
                low_bit,
            };
            let width = usize::from(end - start + 1);
            for negative_first in [false, true] {
                let mut writer = EntropyBitWriter::new(4096);
                writer.write_bits(0x7f, 7).unwrap();
                let mut scalar = writer.clone();
                let mut actual_state = DctCodingState::new(1024);
                let mut expected_state = actual_state.clone();
                for layout in 0..9 {
                    let last_new = match layout {
                        3 | 6 | 8 => Some(width - 1),
                        4 => Some(0),
                        5 => Some(width / 2),
                        _ => None,
                    };
                    let mut block = [i16::MIN; BLOCK_SIZE];
                    for (offset, &natural) in NATURAL_ORDER[usize::from(start)..=usize::from(end)]
                        .iter()
                        .enumerate()
                    {
                        let old = 2 + (offset & 1) as i16;
                        let mut magnitude = match layout {
                            0 | 7 | 8 => 0,
                            1 | 6 => old,
                            2 | 4 | 5 => {
                                if offset % 3 == 1 {
                                    0
                                } else {
                                    old
                                }
                            }
                            // Old coefficients straddle 15/16-zero boundaries;
                            // the full band needs three ZRLs before its new value.
                            3 => {
                                if [0, 16, 18, 35, 37, 53].contains(&offset) {
                                    old
                                } else {
                                    0
                                }
                            }
                            _ => unreachable!(),
                        };
                        // At bit 10 no old coefficient fits the AC precision.
                        if low_bit == 10 && magnitude > 1 {
                            magnitude = 0;
                        }
                        if last_new == Some(offset) {
                            magnitude = 1;
                        }
                        let residue = if (offset + layout) % 3 == 0 {
                            0
                        } else {
                            unit - 1
                        };
                        let value = magnitude * unit + residue;
                        block[natural] = if ((offset + layout) % 2 == 0) == negative_first {
                            -value
                        } else {
                            value
                        };
                    }
                    let expected =
                        scalar_refinement_ac(&mut scalar, &ac, band, &mut expected_state, &block);
                    let actual = encode_refinement_block(
                        &mut writer,
                        Some(&ac),
                        band,
                        &mut actual_state,
                        &block,
                    );
                    assert_eq!(
                        actual, expected,
                        "band={band:?}, layout={layout}, negative_first={negative_first}"
                    );
                    assert_eq!(actual, Ok(()));
                    assert_eq!(writer, scalar, "band={band:?}, layout={layout}");
                    assert_eq!(
                        actual_state, expected_state,
                        "band={band:?}, layout={layout}"
                    );
                }
                actual_state.flush(&mut writer).unwrap();
                scalar_flush(&mut expected_state, &mut scalar).unwrap();
                assert_eq!(actual_state, expected_state);
                assert_eq!(aligned(writer), aligned(scalar));
            }
        }
    }
}

#[test]
fn refinement_precision_bounds_validate_before_pending_flush_or_dc_emission() {
    let ac = ac_table();
    for low_bit in 0..=10 {
        for (start, end) in [(0, 63), (1, 63), (9, 42), (63, 63)] {
            let band = ScanBand {
                start,
                end,
                low_bit,
            };
            let first = usize::from(start.max(1));
            let end = usize::from(end);
            for position in first..=end {
                if position != first && position != (first + end) / 2 && position != end {
                    continue;
                }
                for value in [-2047, 2047, -2048, 2048, i16::MIN, i16::MAX] {
                    let mut block = [i16::MIN; BLOCK_SIZE];
                    for (k, &natural) in NATURAL_ORDER.iter().enumerate().take(end + 1).skip(first)
                    {
                        block[natural] = if k % 2 == 0 {
                            1 << low_bit
                        } else {
                            -(1 << low_bit)
                        };
                    }
                    block[NATURAL_ORDER[position]] = value;
                    let mut writer = EntropyBitWriter::new(1024);
                    writer.write_bits(0x7f, 7).unwrap();
                    let mut actual_state = DctCodingState::new(128);
                    if start != 0 {
                        scalar_buffer_end_of_band(
                            &mut actual_state,
                            &mut writer,
                            &ac,
                            0x5a5a_ffff_1234_5678,
                            63,
                        )
                        .unwrap();
                    }
                    let before_writer = writer.clone();
                    let before_state = actual_state.clone();
                    let mut scalar = writer.clone();
                    let mut expected_state = actual_state.clone();
                    let expected =
                        scalar_refinement_ac(&mut scalar, &ac, band, &mut expected_state, &block);
                    let actual = encode_refinement_block(
                        &mut writer,
                        Some(&ac),
                        band,
                        &mut actual_state,
                        &block,
                    );
                    assert_eq!(
                        actual, expected,
                        "band={band:?}, k={position}, value={value}"
                    );
                    assert_eq!(writer, scalar);
                    assert_eq!(actual_state, expected_state);
                    if (-2047..=2047).contains(&value) {
                        assert_eq!(actual, Ok(()));
                        actual_state.flush(&mut writer).unwrap();
                        scalar_flush(&mut expected_state, &mut scalar).unwrap();
                        assert_eq!(actual_state, expected_state);
                        assert_eq!(aligned(writer), aligned(scalar));
                    } else {
                        assert_eq!(
                            actual,
                            Err(CoefficientWriteError::Invalid(
                                "AC coefficient exceeds supported precision"
                            ))
                        );
                        let mut poisoned = before_state;
                        poisoned.failed = true;
                        assert_eq!(actual_state, poisoned);
                        assert_eq!(writer, before_writer);
                        assert_eq!(
                            actual_state.flush(&mut writer),
                            Err(CoefficientWriteError::FailedState)
                        );
                        assert_eq!(
                            encode_refinement_block(
                                &mut writer,
                                Some(&ac),
                                band,
                                &mut actual_state,
                                &[0; BLOCK_SIZE],
                            ),
                            Err(CoefficientWriteError::FailedState)
                        );
                        assert_eq!(actual_state, poisoned);
                        assert_eq!(writer, before_writer);
                    }
                }
            }
        }
    }
}

#[test]
fn sequential_randomized_dense_sparse_and_zero_blocks_match_initial_bytes() {
    let dc = dc_table();
    let ac = ac_table();
    let long_dc = table(&(0..12).rev().collect::<Vec<_>>(), 16);
    let mut counts = [0; 16];
    counts[7] = 254;
    counts[8] = 1;
    counts[15] = 1;
    let mut symbols: Vec<_> = (1..=255).collect();
    symbols.push(0); // The EOB takes the longest (16-bit) code.
    let long_ac = HuffmanTable::new(&counts, &symbols).unwrap();
    let mut rng = 0x42ec_9765u32;
    let mut random = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng
    };
    let mut saw_stuffing = false;
    for (dc, ac) in [(&dc, &ac), (&long_dc, &long_ac)] {
        for layout in 0..3 {
            for last in 0..BLOCK_SIZE {
                let mut block = [0; BLOCK_SIZE];
                block[0] = match last % 4 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    2 => 0,
                    _ => random() as i16,
                };
                let previous = (i32::from(block[0]) + (random() % 4095) as i32 - 2047)
                    .clamp(i32::from(i16::MIN), i32::from(i16::MAX))
                    as i16;
                for (k, &natural) in NATURAL_ORDER[..BLOCK_SIZE].iter().enumerate().skip(1) {
                    if layout == 2 || k > last || (layout == 1 && k != last && random() % 8 != 0) {
                        continue;
                    }
                    let high_bit = 1 << (random() % 11);
                    let magnitude = (high_bit | (random() & (high_bit - 1))) as i16;
                    block[natural] = if random() & 1 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                }
                let trailing = if layout == 2 { 63 } else { 63 - last };
                for extra in 0..=trailing.div_ceil(16) as u32 {
                    for pending in 0..8 {
                        let mut writer = EntropyBitWriter::new(4096);
                        writer.write_bits(0x12ff, 16).unwrap();
                        let mask = (1 << pending) - 1;
                        let prefix = match last % 3 {
                            0 => 0,
                            1 => mask,
                            _ => random() & mask,
                        };
                        writer.write_bits(prefix, pending).unwrap();
                        let mut predictor = previous;
                        sequential_matches_initial(
                            &mut writer,
                            dc,
                            ac,
                            extra,
                            &mut predictor,
                            &block,
                        )
                        .unwrap();
                        writer.align_ones().unwrap();
                        saw_stuffing |= writer.bytes()[3..].windows(2).any(|pair| pair == [255, 0]);
                    }
                }
            }
        }
    }
    assert!(saw_stuffing);
}

#[test]
fn sequential_consumed_or_overshot_trailing_zeros_need_no_eob_symbol() {
    let dc = dc_table();
    let ac = table(&(1..=255).collect::<Vec<_>>(), 9);
    for last in 0..BLOCK_SIZE {
        let mut block = [0; BLOCK_SIZE];
        if last != 0 {
            block[NATURAL_ORDER[last]] = 1;
        }
        let extra = (63 - last).div_ceil(16) as u32;
        sequential_matches_initial(
            &mut EntropyBitWriter::new(128),
            &dc,
            &ac,
            extra,
            &mut 0,
            &block,
        )
        .unwrap();
    }
}

#[test]
fn sequential_invalid_categories_and_hints_leave_existing_bits_and_predictor_untouched() {
    let dc = dc_table();
    let ac = ac_table();
    let check = |block: &[i16; BLOCK_SIZE], previous: i16, extra, reason| {
        let mut writer = EntropyBitWriter::new(0);
        writer.write_bits(5, 3).unwrap();
        let before = writer.clone();
        let mut predictor = previous;
        assert_eq!(
            sequential_matches_initial(&mut writer, &dc, &ac, extra, &mut predictor, block),
            Err(CoefficientWriteError::Invalid(reason))
        );
        assert_eq!(predictor, previous);
        assert_eq!(writer, before);
    };
    for (value, previous) in [
        (2048, 0),
        (-2048, 0),
        (i16::MAX, i16::MIN),
        (i16::MIN, i16::MAX),
    ] {
        let mut block = [0; BLOCK_SIZE];
        block[0] = value;
        check(&block, previous, 0, "DC difference exceeds 11 bits");
    }
    for natural in 1..BLOCK_SIZE {
        for value in [-2048, 2048, i16::MIN, i16::MAX] {
            let mut block = [0; BLOCK_SIZE];
            block[0] = 18;
            block[natural] = value;
            check(&block, 17, 0, "AC coefficient exceeds supported precision");
        }
    }
    for last in 0..BLOCK_SIZE {
        let mut block = [0; BLOCK_SIZE];
        block[0] = 18;
        if last != 0 {
            block[NATURAL_ORDER[last]] = 1;
        }
        for extra in [(63 - last).div_ceil(16) as u32 + 1, u32::MAX] {
            check(
                &block,
                17,
                extra,
                "extra zero runs exceed trailing spectral band",
            );
        }
    }
    // Preserve DC-before-AC-before-hint validation even when all are malformed.
    let mut block = [2048; BLOCK_SIZE];
    check(&block, 0, u32::MAX, "DC difference exceeds 11 bits");
    block[0] = 1;
    check(
        &block,
        0,
        u32::MAX,
        "AC coefficient exceeds supported precision",
    );
    block[1..].fill(0);
    check(
        &block,
        0,
        u32::MAX,
        "extra zero runs exceed trailing spectral band",
    );
}

#[test]
fn sequential_missing_symbols_preserve_initial_partial_failure_state() {
    let dc = dc_table();
    let ac = ac_table();
    let dc_zero = table(&[0], 1);
    let ac_zero = table(&[0], 1);
    let ac_no_eob = table(&[1, 0xf0], 2);
    let mut empty = [0; BLOCK_SIZE];
    empty[0] = 1;
    let mut nonzero = empty;
    nonzero[NATURAL_ORDER[1]] = 1;
    let mut interior_zrl = empty;
    interior_zrl[NATURAL_ORDER[17]] = 1;
    for (dc, ac, block, extra, missing, expected_previous) in [
        (&dc_zero, &ac, &empty, 0, 1, 0),
        (&dc, &ac_zero, &nonzero, 0, 1, 1),
        (&dc, &ac_zero, &interior_zrl, 0, 0xf0, 1),
        (&dc, &ac_zero, &empty, 1, 0xf0, 1),
        (&dc, &ac_no_eob, &nonzero, 0, 0, 1),
    ] {
        let mut writer = EntropyBitWriter::new(128);
        writer.write_bits(0xff, 8).unwrap();
        writer.write_bits(5, 3).unwrap();
        let mut previous = 0;
        assert_eq!(
            sequential_matches_initial(&mut writer, dc, ac, extra, &mut previous, block),
            Err(CoefficientWriteError::Entropy(
                WriteError::MissingHuffmanSymbol(missing)
            ))
        );
        assert_eq!(previous, expected_previous);
    }
}

#[test]
fn sequential_output_limits_match_initial_at_every_byte_boundary() {
    let dc = dc_table();
    let ac = ac_table();
    let positive = [2047; BLOCK_SIZE];
    let negative = [-2047; BLOCK_SIZE];
    let mut sparse = [0; BLOCK_SIZE];
    sparse[0] = -12;
    sparse[NATURAL_ORDER[18]] = -1024;
    sparse[NATURAL_ORDER[47]] = 2047;
    let mut empty = [0; BLOCK_SIZE];
    empty[0] = 1;
    for (block, extra, previous) in [
        (&positive, 0, 0),
        (&negative, 0, 0),
        (&sparse, 1, 7),
        (&empty, 0, -2046),
        (&empty, 3, 0),
        (&empty, 4, 0),
    ] {
        for pending in 0..8 {
            let prefix = (1 << pending) - 1;
            let mut complete = EntropyBitWriter::new(4096);
            complete.write_bits(prefix, pending).unwrap();
            let mut predictor = previous;
            sequential_matches_initial(&mut complete, &dc, &ac, extra, &mut predictor, block)
                .unwrap();
            let emitted = complete.bytes().len();
            for limit in 0..=emitted + 2 {
                let mut writer = EntropyBitWriter::new(limit);
                writer.write_bits(prefix, pending).unwrap();
                let mut predictor = previous;
                let result =
                    sequential_matches_initial(&mut writer, &dc, &ac, extra, &mut predictor, block);
                let expected = if limit < emitted {
                    Err(CoefficientWriteError::Entropy(WriteError::OutputLimit {
                        limit,
                    }))
                } else {
                    Ok(())
                };
                assert_eq!(
                    result, expected,
                    "limit={limit}, pending={pending}, extra={extra}"
                );
            }
        }
    }
}

#[test]
fn prepared_refinement_matches_category_validation_for_every_i16_and_bit_plane() {
    for low_bit in 0..=10 {
        let band = ScanBand {
            start: 17,
            end: 17,
            low_bit,
        };
        for value in i16::MIN..=i16::MAX {
            let mut block = [0; BLOCK_SIZE];
            block[NATURAL_ORDER[17]] = value;
            assert_refinement_preparation_matches_scalar(&block, band);
        }
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
fn empty_flush_is_a_noop_before_and_after_buffering_but_checks_failed_state() {
    let ac = ac_table();
    let mut coding = DctCodingState::new(63);
    let mut writer = EntropyBitWriter::new(128);
    writer.write_bits(5, 3).unwrap();
    for buffered in [false, true] {
        if buffered {
            encode_refinement_block(
                &mut writer,
                Some(&ac),
                ac_band(0),
                &mut coding,
                &[3; BLOCK_SIZE],
            )
            .unwrap();
            assert_eq!(coding.pending_eob_run(), 1);
            assert_eq!(coding.pending_refinement_bits(), 63);
            coding.flush(&mut writer).unwrap();
            assert!(coding.refinement_words.capacity() > 0);
        }
        assert_eq!(coding.pending_eob_run(), 0);
        assert_eq!(coding.pending_refinement_bits(), 0);
        assert!(coding.refinement_words.is_empty());
        let before_writer = writer.clone();
        let before_coding = coding.clone();
        coding.flush(&mut writer).unwrap();
        assert_eq!(writer, before_writer);
        assert_eq!(coding, before_coding);
    }
    let before = writer.clone();
    assert_eq!(
        encode_refinement_block(&mut writer, None, ac_band(0), &mut coding, &[0; BLOCK_SIZE]),
        Err(CoefficientWriteError::Invalid(
            "missing refinement AC table"
        ))
    );
    assert_eq!(coding.pending_eob_run(), 0);
    assert_eq!(
        coding.flush(&mut writer),
        Err(CoefficientWriteError::FailedState)
    );
    assert_eq!(
        encode_initial_block(
            &mut writer,
            None,
            Some(&ac),
            ac_band(0),
            0,
            &mut coding,
            &mut 0,
            &[0; BLOCK_SIZE],
        ),
        Err(CoefficientWriteError::FailedState)
    );
    assert_eq!(
        encode_refinement_block(
            &mut writer,
            Some(&ac),
            ac_band(0),
            &mut coding,
            &[0; BLOCK_SIZE],
        ),
        Err(CoefficientWriteError::FailedState)
    );
    assert_eq!(writer, before);
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
