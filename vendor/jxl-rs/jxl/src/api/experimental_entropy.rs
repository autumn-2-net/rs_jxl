// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Bounded, explicitly experimental access to the native JXL entropy decoder.
//! This does not change the image decoder or define an image format. The MA
//! token parser follows `frame/modular/tree.rs::Tree::read`, without consuming
//! the residual histogram header that normally follows a standard MA tree.

use std::fmt;

use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::{Histograms, SymbolReader, unpack_signed};

/// Absolute resource bounds for the experimental bridge.
pub const MAX_EXPERIMENTAL_CONTEXTS: usize = 4096;
pub const MAX_EXPERIMENTAL_SYMBOLS: usize = 1 << 30;
pub const MAX_EXPERIMENTAL_BYTES: usize = 256 << 20;
pub const MAX_EXPERIMENTAL_TREE_NODES: usize = 8191;

#[derive(Debug)]
pub enum ExperimentalEntropyError {
    Invalid(&'static str),
    Codec(crate::error::Error),
}

impl fmt::Display for ExperimentalEntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => f.write_str(reason),
            Self::Codec(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for ExperimentalEntropyError {}
impl From<crate::error::Error> for ExperimentalEntropyError {
    fn from(error: crate::error::Error) -> Self {
        Self::Codec(error)
    }
}
type Result<T> = std::result::Result<T, ExperimentalEntropyError>;

fn invalid(reason: &'static str) -> ExperimentalEntropyError {
    ExperimentalEntropyError::Invalid(reason)
}

fn validate_section(bytes: &[u8], bits: usize) -> Result<()> {
    if bytes.len() > MAX_EXPERIMENTAL_BYTES || bits.div_ceil(8) != bytes.len() {
        return Err(invalid("invalid experimental entropy section length"));
    }
    if bits % 8 != 0 && bytes[bytes.len() - 1] >> (bits % 8) != 0 {
        return Err(invalid("nonzero experimental entropy padding"));
    }
    Ok(())
}

fn check_position(br: &BitReader<'_>, bits: usize, exact: bool) -> Result<()> {
    br.check_for_error()?;
    let consumed = br.total_bits_read();
    if consumed > bits || (exact && consumed != bits) {
        return Err(invalid("experimental entropy bit consumption mismatch"));
    }
    Ok(())
}

/// Decoder for one `Model::write_header` / `Model::write_stream` pair.
/// Header and data are separately byte padded; bit lengths exclude the padding.
/// The symbol count is a required budget and must be consumed exactly.
pub struct ExperimentalEntropyDecoder<'a> {
    histograms: Histograms,
    reader: SymbolReader,
    bits: BitReader<'a>,
    data_bits: usize,
    contexts: usize,
    remaining: usize,
    bounded_lz: bool,
    failed: bool,
}

impl<'a> ExperimentalEntropyDecoder<'a> {
    pub fn new(
        header: &[u8],
        header_bits: usize,
        data: &'a [u8],
        data_bits: usize,
        contexts: usize,
        symbols: usize,
    ) -> Result<Self> {
        Self::new_internal(
            header,
            header_bits,
            data,
            data_bits,
            contexts,
            symbols,
            None,
        )
    }

    /// Opt into bounded LZ77 expansion. Contexts exclude the appended distance
    /// context; symbols count expanded values. Zero multiplier means ordinary
    /// distance-minus-one symbols; nonzero selects JXL's row-distance mapping.
    pub fn new_with_lz(
        header: &[u8],
        header_bits: usize,
        data: &'a [u8],
        data_bits: usize,
        contexts: usize,
        symbols: usize,
        distance_multiplier: u32,
    ) -> Result<Self> {
        Self::new_internal(
            header,
            header_bits,
            data,
            data_bits,
            contexts,
            symbols,
            Some(distance_multiplier),
        )
    }

    fn new_internal(
        header: &[u8],
        header_bits: usize,
        data: &'a [u8],
        data_bits: usize,
        contexts: usize,
        symbols: usize,
        distance_multiplier: Option<u32>,
    ) -> Result<Self> {
        if contexts == 0 || contexts > MAX_EXPERIMENTAL_CONTEXTS {
            return Err(invalid("experimental entropy context budget exceeded"));
        }
        if symbols > MAX_EXPERIMENTAL_SYMBOLS {
            return Err(invalid("experimental entropy symbol budget exceeded"));
        }
        validate_section(header, header_bits)?;
        validate_section(data, data_bits)?;
        let mut scratch = Vec::new();
        let mut header_reader = BitReader::new(header);
        let histograms = Histograms::decode(
            contexts,
            &mut header_reader,
            distance_multiplier.is_some(),
            &mut scratch,
        )?;
        check_position(&header_reader, header_bits, true)?;
        if histograms.lz77_params().enabled {
            scratch.clear();
            scratch
                .try_reserve(symbols.min(1 << 20))
                .map_err(|_| invalid("experimental LZ77 history allocation failed"))?;
        }
        let mut bits = BitReader::new(data);
        let reader = SymbolReader::new(
            &histograms,
            &mut bits,
            distance_multiplier.map(|distance| distance as usize),
            &mut scratch,
        )?;
        check_position(&bits, data_bits, false)?;
        Ok(Self {
            histograms,
            reader,
            bits,
            data_bits,
            contexts,
            remaining: symbols,
            bounded_lz: distance_multiplier.is_some(),
            failed: false,
        })
    }

    pub fn has_lz77(&self) -> bool {
        self.reader.has_lz77()
    }

    pub fn read(&mut self, context: usize) -> Result<u32> {
        if self.failed || context >= self.contexts || self.remaining == 0 {
            return Err(invalid(
                "experimental entropy context or symbol out of bounds",
            ));
        }
        // A partially consumed bad command cannot subsequently finalize as a
        // successful stream, even when its optimistic bit reads stayed in range.
        self.failed = true;
        let value = if self.bounded_lz {
            self.reader
                .read_unsigned_bounded(&self.histograms, &mut self.bits, context, self.remaining)
                .map_err(invalid)?
        } else {
            self.reader
                .read_unsigned(&self.histograms, &mut self.bits, context)
        };
        self.reader.check_for_error()?;
        check_position(&self.bits, self.data_bits, false)?;
        self.remaining -= 1;
        self.failed = false;
        Ok(value)
    }

    /// Checks the ANS final state, declared symbol count and exact data length.
    pub fn finish(mut self) -> Result<()> {
        if self.failed || self.remaining != 0 {
            return Err(invalid("experimental entropy symbols remain unread"));
        }
        self.reader.check_bounded_lz77_complete().map_err(invalid)?;
        self.reader
            .check_final_state(&self.histograms, &mut self.bits)?;
        check_position(&self.bits, self.data_bits, true)
    }
}

/// Plain breadth-first MA nodes. Predictor IDs use standard JXL values 0..13.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperimentalMaNode {
    Split {
        property: u8,
        threshold: i32,
        greater: usize,
        less_equal: usize,
    },
    Leaf {
        predictor: u8,
        offset: i32,
        multiplier: u32,
    },
}

/// Parses one no-LZ77 `Tree::write` stream (tree entropy header plus tokens).
/// The node limit is enforced before adding nodes or pending children.
pub fn parse_experimental_ma_tree(
    bytes: &[u8],
    bits: usize,
    max_nodes: usize,
) -> Result<Vec<ExperimentalMaNode>> {
    if max_nodes == 0 || max_nodes > MAX_EXPERIMENTAL_TREE_NODES {
        return Err(invalid("invalid experimental MA node budget"));
    }
    validate_section(bytes, bits)?;
    let mut br = BitReader::new(bytes);
    let mut scratch = Vec::new();
    let histograms = Histograms::decode(6, &mut br, false, &mut scratch)?;
    check_position(&br, bits, false)?;
    let mut reader = SymbolReader::new(&histograms, &mut br, None, &mut scratch)?;
    check_position(&br, bits, false)?;
    let mut nodes = Vec::new();
    let mut pending = 1usize;
    while pending != 0 {
        if nodes.len() >= max_nodes {
            return Err(invalid("experimental MA node budget exceeded"));
        }
        pending -= 1;
        let property = read_tree_value(&mut reader, &histograms, &mut br, bits, 1)?;
        let node = if property != 0 {
            if property > 256 || nodes.len() + pending + 3 > max_nodes {
                return Err(invalid("experimental MA property or node budget invalid"));
            }
            let threshold =
                unpack_signed(read_tree_value(&mut reader, &histograms, &mut br, bits, 0)?);
            let greater = nodes.len() + pending + 1;
            pending += 2;
            ExperimentalMaNode::Split {
                property: (property - 1) as u8,
                threshold,
                greater,
                less_equal: greater + 1,
            }
        } else {
            let predictor = read_tree_value(&mut reader, &histograms, &mut br, bits, 2)?;
            let offset =
                unpack_signed(read_tree_value(&mut reader, &histograms, &mut br, bits, 3)?);
            let log = read_tree_value(&mut reader, &histograms, &mut br, bits, 4)?;
            let mul_bits = read_tree_value(&mut reader, &histograms, &mut br, bits, 5)?;
            if predictor >= 14 || log >= 31 {
                return Err(invalid("invalid experimental MA predictor or multiplier"));
            }
            let multiplier = (u64::from(mul_bits) + 1) << log;
            if multiplier > i32::MAX as u64 {
                return Err(invalid("experimental MA multiplier exceeds pixel domain"));
            }
            ExperimentalMaNode::Leaf {
                predictor: predictor as u8,
                offset,
                multiplier: multiplier as u32,
            }
        };
        nodes
            .try_reserve(1)
            .map_err(|_| invalid("experimental MA allocation failed"))?;
        nodes.push(node);
    }
    reader.check_final_state(&histograms, &mut br)?;
    check_position(&br, bits, true)?;
    validate_tree_ranges(&nodes)?;
    Ok(nodes)
}

fn read_tree_value(
    reader: &mut SymbolReader,
    histograms: &Histograms,
    br: &mut BitReader<'_>,
    bits: usize,
    context: usize,
) -> Result<u32> {
    let value = reader.read_unsigned(histograms, br, context);
    reader.check_for_error()?;
    check_position(br, bits, false)?;
    Ok(value)
}

// Iterative DFS keeps memory proportional to depth, following the maintained
// decoder's range-validation algorithm rather than allocating ranges per node.
fn validate_tree_ranges(nodes: &[ExperimentalMaNode]) -> Result<()> {
    let mut ranges = [(i32::MIN, i32::MAX); 256];
    // Stage 0 enters; stages 1/2 restore the left/right branch's changed range.
    let mut stack = vec![(0usize, 0usize, 0u8, (0i32, 0i32))];
    while let Some((index, depth, stage, old)) = stack.pop() {
        if depth >= 2048 {
            return Err(invalid("experimental MA tree depth exceeded"));
        }
        let ExperimentalMaNode::Split {
            property,
            threshold,
            greater,
            less_equal,
        } = nodes[index]
        else {
            continue;
        };
        let p = usize::from(property);
        if stage != 0 {
            ranges[p] = old;
        }
        if stage == 2 {
            continue;
        }
        let (lo, hi) = ranges[p];
        if lo > threshold || hi <= threshold {
            return Err(invalid("experimental MA split has an unreachable branch"));
        }
        stack.push((index, depth, stage + 1, ranges[p]));
        if stage == 0 {
            ranges[p] = (threshold + 1, hi);
            stack.push((greater, depth + 1, 0, (0, 0)));
        } else {
            ranges[p] = (lo, threshold);
            stack.push((less_equal, depth + 1, 0, (0, 0)));
        }
    }
    Ok(())
}
