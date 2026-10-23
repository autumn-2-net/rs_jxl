// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! DCT scans from enc_jpeg_data_reader.cc::{DecodeDCTBlock, RefineDCTBlock}
//! at b866b3d.
//! Quantized coefficients remain integers in natural 8x8 order. No inverse DCT,
//! pixel conversion, quantization change, or JXL coefficient coding occurs here.

use super::bitstream::{EntropyBitReader, EntropyError};
use super::data::{BLOCK_SIZE, NATURAL_ORDER};
use super::huffman::{HuffmanError, HuffmanTable};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoefficientError {
    Entropy(EntropyError),
    Huffman(HuffmanError),
    Invalid(&'static str),
}

impl fmt::Display for CoefficientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Entropy(error) => error.fmt(f),
            Self::Huffman(error) => error.fmt(f),
            Self::Invalid(reason) => f.write_str(reason),
        }
    }
}
impl std::error::Error for CoefficientError {}
impl From<EntropyError> for CoefficientError {
    fn from(error: EntropyError) -> Self {
        Self::Entropy(error)
    }
}
impl From<HuffmanError> for CoefficientError {
    fn from(error: HuffmanError) -> Self {
        Self::Huffman(error)
    }
}

/// Spectral indices are zigzag indices. `low_bit` is the scan's Al value.
#[derive(Clone, Copy, Debug)]
pub struct ScanBand {
    pub start: u8,
    pub end: u8,
    pub low_bit: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockState {
    /// Unshifted DC predictor for one component. Reset at restart boundaries.
    pub last_dc: i16,
    /// Shared across the blocks of an AC scan, initially -1 as in the source.
    pub eob_run: i32,
}
impl Default for BlockState {
    fn default() -> Self {
        Self {
            last_dc: 0,
            eob_run: -1,
        }
    }
}

/// Reconstruction hints. They are not redundant even if coefficients are equal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockInfo {
    pub reset_eob_state: bool,
    pub extra_zero_runs: u32,
}

/// Decode a sequential block or the FIRST pass of a progressive spectral band.
/// Refinement scans (Ah != 0) need the separate refinement algorithm. The caller
/// owns MCU/component traversal, scan progression, restart order, and recording
/// BlockInfo at the correct scan block index. Errors leave all state unchanged.
pub fn decode_initial_block(
    reader: &mut EntropyBitReader<'_>,
    dc: Option<&HuffmanTable>,
    ac: Option<&HuffmanTable>,
    band: ScanBand,
    state: &mut BlockState,
    coefficients: &mut [i16; BLOCK_SIZE],
) -> Result<BlockInfo, CoefficientError> {
    if band.start > band.end || band.end >= 64 || band.low_bit > 10 {
        return Err(CoefficientError::Invalid(
            "invalid initial JPEG spectral band",
        ));
    }
    if state.eob_run > 32767 {
        return Err(CoefficientError::Invalid("invalid JPEG EOB state"));
    }
    let mut trial = *reader;
    let mut next = *state;
    let mut block = *coefficients;
    let mut info = BlockInfo::default();
    let scale = 1i32 << band.low_bit;
    let eob_allowed = band.start > 0;
    let mut k = usize::from(band.start);
    let end = usize::from(band.end);
    if k == 0 {
        let dc = dc.ok_or(CoefficientError::Invalid("missing JPEG DC table"))?;
        let category = dc.decode(&mut trial)?;
        if category >= 12 {
            return Err(CoefficientError::Invalid("invalid JPEG DC category"));
        }
        let value = i32::from(next.last_dc) + trial.receive_and_extend(category)?;
        block[0] = i16::try_from(value * scale)
            .map_err(|_| CoefficientError::Invalid("JPEG DC coefficient exceeds i16"))?;
        next.last_dc = i16::try_from(value)
            .map_err(|_| CoefficientError::Invalid("JPEG DC predictor exceeds i16"))?;
        k += 1;
    }
    if k <= end {
        if next.eob_run > 0 {
            next.eob_run -= 1;
        } else {
            let ac = ac.ok_or(CoefficientError::Invalid("missing JPEG AC table"))?;
            let first_ac = k;
            while k <= end {
                let symbol = ac.decode(&mut trial)?;
                let run = symbol >> 4;
                let category = symbol & 15;
                if category != 0 {
                    k += usize::from(run);
                    if k > end || category + band.low_bit >= 12 {
                        return Err(CoefficientError::Invalid("out-of-band JPEG AC coefficient"));
                    }
                    block[NATURAL_ORDER[k]] = (trial.receive_and_extend(category)? * scale) as i16;
                    info.extra_zero_runs = 0;
                } else if run == 15 {
                    // Retain superfluous trailing ZRLs; a canonical coefficient
                    // encoder would otherwise omit these original JPEG bits.
                    k += 15;
                    info.extra_zero_runs += 1;
                } else {
                    info.reset_eob_state = eob_allowed && k == first_ac && next.eob_run == 0;
                    if run != 0 && !eob_allowed {
                        return Err(CoefficientError::Invalid(
                            "JPEG EOB run crosses DC coefficients",
                        ));
                    }
                    next.eob_run = (1i32 << run) + trial.read_bits(run)? as i32;
                    break;
                }
                k += 1;
            }
            next.eob_run = next
                .eob_run
                .checked_sub(1)
                .ok_or(CoefficientError::Invalid("JPEG EOB state underflow"))?;
        }
    }
    *reader = trial;
    *state = next;
    *coefficients = block;
    Ok(info)
}

/// Apply a progressive refinement pass (Ah != 0). The caller validates scan
/// ordering with ScanProgression. DC refinement is bitwise; negative AC values
/// refine away from zero. No pixel arithmetic or floating point is involved.
/// Like initial decoding, a failure leaves the reader, state and block intact.
pub fn refine_block(
    reader: &mut EntropyBitReader<'_>,
    ac: Option<&HuffmanTable>,
    band: ScanBand,
    state: &mut BlockState,
    coefficients: &mut [i16; BLOCK_SIZE],
) -> Result<BlockInfo, CoefficientError> {
    if band.start > band.end || band.end >= 64 || band.low_bit > 10 || state.eob_run > 32767 {
        return Err(CoefficientError::Invalid(
            "invalid JPEG refinement parameters",
        ));
    }
    let mut trial = *reader;
    let mut next = *state;
    let mut block = *coefficients;
    let mut info = BlockInfo::default();
    let bit = 1i16 << band.low_bit;
    let mut k = usize::from(band.start);
    let end = usize::from(band.end);
    if k == 0 {
        block[0] |= if trial.read_bit()? { bit } else { 0 };
        k += 1;
    }
    if k <= end {
        let first_ac = k;
        let mut in_zero_run = false;
        if next.eob_run <= 0 {
            let ac = ac.ok_or(CoefficientError::Invalid(
                "missing JPEG refinement AC table",
            ))?;
            while k <= end {
                let symbol = ac.decode(&mut trial)?;
                let mut run = i32::from(symbol >> 4);
                let category = symbol & 15;
                let value = if category != 0 {
                    if category != 1 {
                        return Err(CoefficientError::Invalid(
                            "invalid JPEG refinement AC category",
                        ));
                    }
                    in_zero_run = false;
                    if trial.read_bit()? { bit } else { -bit }
                } else {
                    if run != 15 {
                        info.reset_eob_state = band.start > 0 && k == first_ac && next.eob_run == 0;
                        if run != 0 && band.start == 0 {
                            return Err(CoefficientError::Invalid(
                                "JPEG refinement EOB crosses DC",
                            ));
                        }
                        next.eob_run = (1i32 << run) + trial.read_bits(run as u8)? as i32;
                        break;
                    }
                    in_zero_run = true;
                    0
                };
                loop {
                    let coefficient = &mut block[NATURAL_ORDER[k]];
                    if *coefficient != 0 {
                        refine_nonzero(&mut trial, coefficient, bit)?;
                    } else {
                        run -= 1;
                        if run < 0 {
                            break;
                        }
                    }
                    k += 1;
                    if k > end {
                        break;
                    }
                }
                if value != 0 {
                    if k > end {
                        return Err(CoefficientError::Invalid(
                            "out-of-band JPEG refinement coefficient",
                        ));
                    }
                    block[NATURAL_ORDER[k]] = value;
                }
                k += 1;
            }
        }
        if in_zero_run {
            return Err(CoefficientError::Invalid(
                "extra zero run before JPEG refinement end-of-block",
            ));
        }
        if next.eob_run > 0 {
            while k <= end {
                let coefficient = &mut block[NATURAL_ORDER[k]];
                if *coefficient != 0 {
                    refine_nonzero(&mut trial, coefficient, bit)?;
                }
                k += 1;
            }
        }
        next.eob_run = next
            .eob_run
            .checked_sub(1)
            .ok_or(CoefficientError::Invalid("JPEG EOB state underflow"))?;
    }
    *reader = trial;
    *state = next;
    *coefficients = block;
    Ok(info)
}

fn refine_nonzero(
    reader: &mut EntropyBitReader<'_>,
    value: &mut i16,
    bit: i16,
) -> Result<(), CoefficientError> {
    if reader.read_bit()? && *value & bit == 0 {
        *value = value
            .checked_add(if *value >= 0 { bit } else { -bit })
            .ok_or(CoefficientError::Invalid(
                "JPEG refined coefficient exceeds i16",
            ))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::bitstream::PaddingPolicy;

    fn table(symbols: &[u8], length: usize) -> HuffmanTable {
        let mut counts = [0; 16];
        counts[length - 1] = symbols.len() as u8;
        HuffmanTable::new(&counts, symbols).unwrap()
    }

    fn entropy(bits: &str) -> Vec<u8> {
        let mut output = Vec::new();
        for chunk in bits.as_bytes().chunks(8) {
            let mut byte = 0u8;
            for position in 0..8 {
                byte = (byte << 1) | u8::from(chunk.get(position).copied().unwrap_or(b'1') == b'1');
            }
            output.push(byte);
            if byte == 0xff {
                output.push(0);
            }
        }
        output.extend_from_slice(&[0xff, 0xd9]);
        output
    }

    #[test]
    fn sequential_negative_dc_and_ac_keep_natural_order() {
        let dc = table(&[0, 1, 2], 2);
        let ac = table(&[0, 1, 0xf0], 2);
        // DC category 2, magnitude 01 => -2; AC category 1 => +1; EOB.
        let bytes = entropy("100101100");
        let mut reader = EntropyBitReader::new(&bytes);
        let mut state = BlockState {
            last_dc: 4,
            ..Default::default()
        };
        let mut coefficients = [0; BLOCK_SIZE];
        let info = decode_initial_block(
            &mut reader,
            Some(&dc),
            Some(&ac),
            ScanBand {
                start: 0,
                end: 63,
                low_bit: 0,
            },
            &mut state,
            &mut coefficients,
        )
        .unwrap();
        assert_eq!(coefficients[0], 2);
        assert_eq!(coefficients[NATURAL_ORDER[1]], 1);
        assert!(coefficients[2..].iter().all(|&value| value == 0));
        assert_eq!(
            state,
            BlockState {
                last_dc: 2,
                eob_run: 0
            }
        );
        assert_eq!(info, BlockInfo::default());
        assert_eq!(
            reader
                .finish_scan(PaddingPolicy::RequireOnes)
                .unwrap()
                .marker
                .code,
            0xd9
        );
    }

    #[test]
    fn progressive_eob_run_and_trailing_zero_run_hints_are_preserved() {
        let ac = table(&[0x10, 0, 0xf0], 2);
        let bytes = entropy("000011001");
        let mut reader = EntropyBitReader::new(&bytes);
        let band = ScanBand {
            start: 1,
            end: 63,
            low_bit: 2,
        };
        let mut state = BlockState::default();
        let mut block = [0; BLOCK_SIZE];
        block[0] = -24;
        decode_initial_block(&mut reader, None, Some(&ac), band, &mut state, &mut block).unwrap();
        assert_eq!(state.eob_run, 1);
        let position = (reader.position(), reader.bits_left());
        decode_initial_block(&mut reader, None, None, band, &mut state, &mut block).unwrap();
        assert_eq!((reader.position(), reader.bits_left()), position);
        assert_eq!(state.eob_run, 0);
        let info = decode_initial_block(&mut reader, None, Some(&ac), band, &mut state, &mut block)
            .unwrap();
        assert!(info.reset_eob_state);
        let info = decode_initial_block(&mut reader, None, Some(&ac), band, &mut state, &mut block)
            .unwrap();
        assert_eq!(info.extra_zero_runs, 1);
        assert!(!info.reset_eob_state);
        assert_eq!(block[0], -24);
        assert!(block[1..].iter().all(|&value| value == 0));
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }

    #[test]
    fn full_spectral_bands_without_eob_retain_source_negative_state() {
        let ac = table(&[1], 1);
        let bytes = entropy("010101");
        let mut reader = EntropyBitReader::new(&bytes);
        let mut state = BlockState::default();
        for expected in [-2, -3, -4] {
            let mut block = [0; BLOCK_SIZE];
            decode_initial_block(
                &mut reader,
                None,
                Some(&ac),
                ScanBand {
                    start: 8,
                    end: 8,
                    low_bit: 3,
                },
                &mut state,
                &mut block,
            )
            .unwrap();
            assert_eq!(block[NATURAL_ORDER[8]], 8);
            assert_eq!(state.eob_run, expected);
        }
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }

    #[test]
    fn failures_do_not_change_coefficients_predictor_or_input_position() {
        let dc = table(&[2], 1);
        let ac = table(&[0x10], 1);
        for (bytes, initial_dc, end) in [
            (entropy("011"), i16::MAX, 0),
            (entropy("0110"), 0, 63),
            (vec![], 0, 63),
        ] {
            let mut reader = EntropyBitReader::new(&bytes);
            let mut state = BlockState {
                last_dc: initial_dc,
                ..Default::default()
            };
            let before = state;
            let mut block = [7; BLOCK_SIZE];
            assert!(
                decode_initial_block(
                    &mut reader,
                    Some(&dc),
                    Some(&ac),
                    ScanBand {
                        start: 0,
                        end,
                        low_bit: 0
                    },
                    &mut state,
                    &mut block
                )
                .is_err()
            );
            assert_eq!(state, before);
            assert_eq!(block, [7; BLOCK_SIZE]);
            assert_eq!((reader.position(), reader.bits_left()), (0, 0));
        }
    }

    #[test]
    fn refinement_distinguishes_signed_dc_bits_from_signed_ac_magnitudes() {
        let mut state = BlockState::default();
        let mut block = [0; BLOCK_SIZE];
        block[0] = -4;
        let dc_bits = entropy("1");
        let mut dc_reader = EntropyBitReader::new(&dc_bits);
        refine_block(
            &mut dc_reader,
            None,
            ScanBand {
                start: 0,
                end: 0,
                low_bit: 1,
            },
            &mut state,
            &mut block,
        )
        .unwrap();
        assert_eq!(block[0], -2);
        block[NATURAL_ORDER[1]] = 4;
        block[NATURAL_ORDER[2]] = -4;
        let ac = table(&[0], 1);
        let ac_bits = entropy("011");
        let mut ac_reader = EntropyBitReader::new(&ac_bits);
        refine_block(
            &mut ac_reader,
            Some(&ac),
            ScanBand {
                start: 1,
                end: 63,
                low_bit: 1,
            },
            &mut state,
            &mut block,
        )
        .unwrap();
        assert_eq!(block[NATURAL_ORDER[1]], 6);
        assert_eq!(block[NATURAL_ORDER[2]], -6);
        assert_eq!(state.eob_run, 0);
        ac_reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }

    #[test]
    fn refinement_skips_existing_nonzeros_and_introduces_signed_coefficients() {
        let ac = table(&[0x11, 0], 2);
        // One zero before a new negative coefficient; refine the existing +4.
        let bytes = entropy("000101");
        let mut reader = EntropyBitReader::new(&bytes);
        let mut state = BlockState::default();
        let mut block = [0; BLOCK_SIZE];
        block[NATURAL_ORDER[1]] = 4;
        refine_block(
            &mut reader,
            Some(&ac),
            ScanBand {
                start: 1,
                end: 63,
                low_bit: 1,
            },
            &mut state,
            &mut block,
        )
        .unwrap();
        assert_eq!(block[NATURAL_ORDER[1]], 6);
        assert_eq!(block[NATURAL_ORDER[2]], 0);
        assert_eq!(block[NATURAL_ORDER[3]], -2);
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }

    #[test]
    fn full_refinement_band_breaks_consecutive_eob_reset_hint() {
        let ac = table(&[0, 1], 2);
        // EOB, a full one-coefficient band (+1, no EOB), then EOB again.
        let bytes = entropy("0001100");
        let mut reader = EntropyBitReader::new(&bytes);
        let mut state = BlockState::default();
        for expected_eob in [0, -1, 0] {
            let mut block = [0; BLOCK_SIZE];
            let info = refine_block(
                &mut reader,
                Some(&ac),
                ScanBand {
                    start: 1,
                    end: 1,
                    low_bit: 0,
                },
                &mut state,
                &mut block,
            )
            .unwrap();
            assert_eq!(state.eob_run, expected_eob);
            assert!(!info.reset_eob_state);
            assert_eq!(block[NATURAL_ORDER[1]], i16::from(expected_eob == -1));
        }
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }

    #[test]
    fn malformed_refinement_is_atomic() {
        for symbol in [0x02, 0xf0, 0xf1] {
            let ac = table(&[symbol], 1);
            let bytes = entropy("01");
            let mut reader = EntropyBitReader::new(&bytes);
            let mut state = BlockState::default();
            let mut block = [0; BLOCK_SIZE];
            assert!(
                refine_block(
                    &mut reader,
                    Some(&ac),
                    ScanBand {
                        start: 1,
                        end: 1,
                        low_bit: 0
                    },
                    &mut state,
                    &mut block
                )
                .is_err()
            );
            assert_eq!(state, BlockState::default());
            assert_eq!(block, [0; BLOCK_SIZE]);
            assert_eq!((reader.position(), reader.bits_left()), (0, 0));
        }
    }
}
