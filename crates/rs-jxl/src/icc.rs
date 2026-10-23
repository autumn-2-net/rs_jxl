// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! PredictICC/WriteICC wire port from enc_icc_codec.cc, icc_codec_common.{h,cc}
//! and enc_fields.cc (U64Coder), libjxl b866b3d. Includes all source prediction
//! commands, tag folding, wrapping arithmetic and the 41 byte contexts.
//! WriteICC retains the source's four actual-bit trials below 16,384 predicted
//! bytes: Best/Fast clustering crossed with Best/fixed-000 uint coding, all
//! using prefix codes and LZ77. Larger streams use the single baseline trial.
//!
//! All trials use source Optc256 below that threshold and b3w3t otherwise,
//! through the native source parser. Its acceptance flag, not match presence,
//! controls the entropy header. Parser costs and wire share length000 and
//! min-symbol512, with no legacy 256-symbol/match-length restriction.
//!
//! Remaining adaptations: source parser/SIMD floating-point ties can differ
//! across targets; environment-selected parser experiments, tracing, AuxOut,
//! and CMS/profile synthesis are outside this writer. The source baseline
//! policy is explicit, not read from process environment. See tools::lz77 for
//! its parser coverage; this is not a claim of byte-identical compressed output.

#![forbid(unsafe_code)]

use crate::bit_writer::BitWriter;
use crate::entropy::HybridUintConfig;
use crate::entropy::options::{ClusterStrategy, EntropyOptions, UintConfigStrategy};
use crate::entropy::stream::{Coding, Model, Token};
use crate::tools::lz77::{self, Command, Literal, SourceConfig, SourceLz77Method};
use crate::{Error, Result};
use std::collections::BTreeMap;

const SIZE_LIMIT: usize = (u32::MAX >> 2) as usize;
const ICC_CONTEXTS: usize = 41;
const ICC_SEARCH_LIMIT: usize = 16_384;
const TAGS: [[u8; 4]; 17] = [
    *b"cprt", *b"wtpt", *b"bkpt", *b"rXYZ", *b"gXYZ", *b"bXYZ", *b"kXYZ", *b"rTRC", *b"gTRC",
    *b"bTRC", *b"kTRC", *b"chad", *b"desc", *b"chrm", *b"dmnd", *b"dmdd", *b"lumi",
];
const TYPES: [[u8; 4]; 8] = [
    *b"XYZ ", *b"desc", *b"text", *b"mluc", *b"para", *b"curv", *b"sf32", *b"gbd ",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IccColorSpace {
    Rgb,
    Gray,
    Cmyk,
}

/// Structural guard for explicit color metadata, not a full ICC conformance or
/// CMS check. Unknown spaces, malformed lengths/signature/tag bounds are rejected.
/// Use predict_icc/write_icc directly to preserve arbitrary non-profile bytes.
pub fn validate_profile(profile: &[u8]) -> Result<IccColorSpace> {
    if profile.len() < 132
        || profile.len() > SIZE_LIMIT
        || be32(profile, 0) as usize != profile.len()
        || keyword(profile, 36) != *b"acsp"
    {
        return Err(Error::InvalidInput(
            "invalid ICC size or acsp signature".into(),
        ));
    }
    let space = match &profile[16..20] {
        b"RGB " => IccColorSpace::Rgb,
        b"GRAY" => IccColorSpace::Gray,
        b"CMYK" => IccColorSpace::Cmyk,
        _ => {
            return Err(Error::InvalidInput(
                "only RGB, GRAY or CMYK ICC profiles are supported".into(),
            ));
        }
    };
    if !matches!(&profile[20..24], b"XYZ " | b"Lab ") {
        return Err(Error::InvalidInput("ICC PCS must be XYZ or Lab".into()));
    }
    let count = be32(profile, 128) as usize;
    if count > (profile.len() - 132) / 12 {
        return Err(Error::InvalidInput("truncated ICC tag directory".into()));
    }
    let table_end = 132 + count * 12;
    for i in 0..count {
        let start = be32(profile, 132 + i * 12 + 4) as usize;
        let len = be32(profile, 132 + i * 12 + 8) as usize;
        if start < table_end || !in_bounds(start, len, profile.len()) {
            return Err(Error::InvalidInput(
                "ICC tag lies outside profile data".into(),
            ));
        }
    }
    Ok(space)
}

/// Byte-exact reversible pretransform; like PredictICC accepts arbitrary bytes,
/// including empty input, truncated tags and invalid ICC. Size limit is 2^30-1.
pub fn predict_icc(icc: &[u8]) -> Result<Vec<u8>> {
    let size = icc.len();
    if size > SIZE_LIMIT {
        return Err(Error::Limit("ICC profile is too large"));
    }
    let mut result = Vec::new();
    varint(size as u64, &mut result);
    let mut commands = Vec::new();
    let mut data = Vec::new();
    data.try_reserve(size)
        .map_err(|_| Error::Limit("ICC allocation"))?;
    let mut header = initial_header(size as u32);
    for i in 0..128.min(size) {
        predict_header(icc, &mut header, i);
        data.push(icc[i].wrapping_sub(header[i]));
    }
    if size <= 128 {
        varint(0, &mut result);
        result.extend_from_slice(&data);
        return Ok(result);
    }

    let mut tagmap = BTreeMap::new();
    let mut pos = 128;
    if in_bounds(pos, 4, size) {
        let count = u64::from(be32(icc, pos));
        pos += 4;
        varint(count + 1, &mut commands);
        let mut previous_start = 128 + count * 12;
        let mut previous_size = 0u32;
        let mut i = 0;
        while i < count && in_bounds(pos, 12, size) {
            let tag = keyword(icc, pos);
            let start = be32(icc, pos + 4);
            let len = be32(icc, pos + 8);
            pos += 12;
            tagmap.insert(start as usize, len as usize);
            let mut code = TAGS
                .iter()
                .position(|t| *t == tag)
                .map_or(1, |j| j as u8 + 4);
            // Source intentionally uses strict < here, not <=.
            if tag == *b"rTRC"
                && pos + 24 < size
                && keyword(icc, pos) == *b"gTRC"
                && keyword(icc, pos + 12) == *b"bTRC"
                && icc[pos - 8..pos] == icc[pos + 4..pos + 12]
                && icc[pos - 8..pos] == icc[pos + 16..pos + 24]
            {
                code = 2;
                pos += 24;
                i += 2;
            }
            if tag == *b"rXYZ"
                && pos + 24 < size
                && keyword(icc, pos) == *b"gXYZ"
                && keyword(icc, pos + 12) == *b"bXYZ"
                && len == 20
                && be32(icc, pos + 8) == 20
                && be32(icc, pos + 20) == 20
                && be32(icc, pos + 4) == start.wrapping_add(20)
                && be32(icc, pos + 16) == start.wrapping_add(40)
            {
                code = 3;
                pos += 24;
                i += 2;
            }
            let mut command = code;
            if previous_start + u64::from(previous_size) != u64::from(start) {
                command |= 64;
            }
            let predicted_size = if [
                *b"rXYZ", *b"gXYZ", *b"bXYZ", *b"kXYZ", *b"wtpt", *b"bkpt", *b"lumi",
            ]
            .contains(&tag)
            {
                20
            } else {
                previous_size
            };
            if predicted_size != len {
                command |= 128;
            }
            commands.push(command);
            if code == 1 {
                data.extend_from_slice(&tag);
            }
            if command & 64 != 0 {
                varint(u64::from(start), &mut commands);
            }
            if command & 128 != 0 {
                varint(u64::from(len), &mut commands);
            }
            previous_start = u64::from(start);
            previous_size = len;
            i += 1;
        }
    }
    commands.push(0);

    let mut tag = [0; 4];
    let mut tagstart = 0;
    let mut tagsize = 0;
    let mut clutstart = 0;
    let mut last0 = pos;
    while pos <= size {
        let mut last1 = pos;
        let mut add = Vec::new();
        let mut bytes = Vec::new();
        if pos as u64 > tagstart as u64 + tagsize as u64 && tagsize < SIZE_LIMIT {
            tag = [0; 4];
        }
        if let Some(&len) = tagmap.get(&pos).filter(|_| in_bounds(pos, 4, size)) {
            tag = keyword(icc, pos);
            tagstart = pos;
            tagsize = len;
            if tag == *b"mluc"
                && sane(tagsize)
                && in_bounds(pos, tagsize, size)
                && be32(icc, pos + 4) == 0
            {
                let num = tagsize - 8;
                add.push(19);
                pos += 8;
                add.push(2);
                varint(num as u64, &mut add);
                bytes.extend_from_slice(&icc[pos..pos + num]);
                pos += num;
                unshuffle(&mut bytes, 2);
            }
            if tag == *b"curv"
                && sane(tagsize)
                && in_bounds(pos, tagsize, size)
                && be32(icc, pos + 4) == 0
            {
                let num = tagsize - 8;
                if num > 16 && num < 1 << 28 && in_bounds(pos, num, size) && pos > 0 {
                    add.push(21);
                    pos += 8;
                    add.extend_from_slice(&[4, 5]);
                    varint(num as u64, &mut add);
                    predict_and_shuffle(icc, &mut pos, num, 2, 2, 1, &mut bytes)?;
                }
            }
        }
        if tag == *b"mAB " || tag == *b"mBA " {
            let subtag = keyword(icc, pos);
            if pos + 12 < size
                && (subtag == *b"curv" || subtag == *b"vcgt")
                && be32(icc, pos + 4) == 0
            {
                let num = be32(icc, pos + 8).wrapping_mul(2) as usize;
                if num > 16 && num < 1 << 28 && in_bounds(pos + 12, num, size) {
                    pos += 12;
                    last1 = pos;
                    add.extend_from_slice(&[4, 5]);
                    varint(num as u64, &mut add);
                    predict_and_shuffle(icc, &mut pos, num, 2, 2, 1, &mut bytes)?;
                }
            }
            if pos == tagstart + 24 && pos + 4 < size {
                clutstart = tagstart.saturating_add(be32(icc, pos) as usize);
            }
            if pos == clutstart && in_bounds(clutstart, 17, size) {
                let numi = usize::from(icc[tagstart + 8]);
                let numo = usize::from(icc[tagstart + 9]);
                let width = usize::from(icc[clutstart + 16]);
                let stride = width * numo;
                let mut num = stride;
                for i in 0..numi.min(size - clutstart) {
                    num = num.wrapping_mul(usize::from(icc[clutstart + i]));
                }
                if (width == 1 || width == 2)
                    && num > 64
                    && num < 1 << 28
                    && in_bounds(pos, num, size)
                    && pos > stride * 4
                {
                    let flags = 4 | (width as u8 - 1) | if stride == width { 0 } else { 16 };
                    add.extend_from_slice(&[4, flags]);
                    if flags & 16 != 0 {
                        varint(stride as u64, &mut add);
                    }
                    varint(num as u64, &mut add);
                    predict_and_shuffle(icc, &mut pos, num, stride, width, 1, &mut bytes)?;
                }
            }
        }
        if add.is_empty()
            && bytes.is_empty()
            && tag == *b"gbd "
            && sane(tagsize)
            && pos == tagstart + 8
            && in_bounds(pos, tagsize - 8, size)
            && pos > 16
        {
            let num = tagsize - 8;
            add.extend_from_slice(&[4, 3]);
            varint(num as u64, &mut add);
            predict_and_shuffle(icc, &mut pos, num, 4, 4, 0, &mut bytes)?;
        }
        if add.is_empty()
            && bytes.is_empty()
            && in_bounds(pos, 20, size)
            && keyword(icc, pos) == *b"XYZ "
            && be32(icc, pos + 4) == 0
        {
            add.push(10);
            pos += 8;
            bytes.extend_from_slice(&icc[pos..pos + 12]);
            pos += 12;
        }
        if add.is_empty()
            && bytes.is_empty()
            && in_bounds(pos, 8, size)
            && be32(icc, pos + 4) == 0
            && let Some(i) = TYPES.iter().position(|&t| t == keyword(icc, pos))
        {
            add.push(16 + i as u8);
            pos += 8;
        }
        let emitted = !add.is_empty() || !bytes.is_empty();
        if emitted || pos == size {
            if last0 < last1 {
                commands.push(1);
                varint((last1 - last0) as u64, &mut commands);
                data.extend_from_slice(&icc[last0..last1]);
            }
            commands.extend_from_slice(&add);
            data.extend_from_slice(&bytes);
            last0 = pos;
        }
        if !emitted {
            pos += 1;
        }
    }
    varint(commands.len() as u64, &mut result);
    result.extend_from_slice(&commands);
    result.extend_from_slice(&data);
    Ok(result)
}

/// Complete JPEG XL ICC stream. Call after ImageMetadata and custom transform
/// data, before byte padding. Enables source lossless four-way search for small
/// predicted streams. Does not pad. Empty input is an error. Errors leave the
/// destination unchanged; the chosen candidate is appended without re-encoding.
pub fn write_icc(icc: &[u8], writer: &mut BitWriter) -> Result<()> {
    write_icc_with_search(icc, writer, true)
}

/// Explicit source WriteICC lossless_search flag. False keeps the baseline
/// LZ77 + prefix trial, not the previous no-LZ77 writer. The threshold uses
/// PredictICC output size, not the input profile size. See module-level gaps.
pub fn write_icc_with_search(
    icc: &[u8],
    writer: &mut BitWriter,
    lossless_search: bool,
) -> Result<()> {
    if icc.is_empty() {
        return Err(Error::InvalidInput("ICC must be non-empty".into()));
    }
    let enc = predict_icc(icc)?;
    let literals = icc_literals(&enc)?;
    let (_, selected) = select_icc_candidate(&literals, lossless_search)?;
    writer.append_unaligned(&selected)?;
    Ok(())
}

fn icc_literals(enc: &[u8]) -> Result<Vec<Literal>> {
    let mut literals = Vec::new();
    literals
        .try_reserve(enc.len())
        .map_err(|_| Error::Limit("ICC token allocation"))?;
    for (i, &byte) in enc.iter().enumerate() {
        let context = icc_context(
            i,
            if i > 0 { enc[i - 1] } else { 0 },
            if i > 1 { enc[i - 2] } else { 0 },
        );
        literals.push(Literal {
            context: context as u32,
            value: u32::from(byte),
        });
    }
    Ok(literals)
}

fn icc_entropy_options(choice: usize) -> EntropyOptions {
    debug_assert!(choice < 4);
    EntropyOptions {
        clustering: if choice & 1 == 0 {
            ClusterStrategy::Best
        } else {
            ClusterStrategy::Fast
        },
        uint_config: if choice & 2 == 0 {
            UintConfigStrategy::Best
        } else {
            UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).expect("constant"))
        },
        prefix_singleton_guard: true,
        lz77_length_config: HybridUintConfig::new(0, 0, 0).expect("constant"),
        lz77_min_symbol: 512,
        ..EntropyOptions::default()
    }
}

fn icc_lz77_options(input_len: usize, choice: usize) -> SourceConfig {
    let entropy = icc_entropy_options(choice);
    SourceConfig {
        method: if input_len < ICC_SEARCH_LIMIT {
            SourceLz77Method::Optc256
        } else {
            SourceLz77Method::Lz77b3w3t
        },
        literal_uint_config: match entropy.uint_config {
            UintConfigStrategy::Fixed(config) => config,
            UintConfigStrategy::Fast | UintConfigStrategy::Best => HybridUintConfig::default(),
        },
        length_uint_config: entropy.lz77_length_config,
        min_symbol: Some(entropy.lz77_min_symbol),
        force_huffman: true,
        // WriteICC's HistogramParams does not enable lossless_lz77_cost_spans.
        range_search: false,
        ..SourceConfig::default()
    }
}

fn icc_tokens(literals: &[Literal], options: &SourceConfig) -> Result<(Vec<Token>, bool)> {
    // ICC has no image-width multiplier, so distance symbols are distance - 1.
    let parsed = lz77::parse_source_with_state(literals, 0, options)?;
    let count = parsed
        .commands
        .iter()
        .try_fold(0usize, |sum, command| {
            sum.checked_add(if matches!(command, Command::Match { .. }) {
                2
            } else {
                1
            })
        })
        .ok_or(Error::Limit("ICC parsed token count"))?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("ICC parsed token allocation"))?;
    for command in parsed.commands {
        match command {
            Command::Literal(literal) => {
                tokens.push(Token::literal(literal.context as usize, literal.value))
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                tokens.push(Token {
                    context: context as usize,
                    value: length
                        .checked_sub(lz77::MIN_MATCH as u32)
                        .ok_or_else(|| Error::Encoding("invalid ICC LZ77 length".into()))?,
                    is_length: true,
                });
                tokens.push(Token::literal(
                    ICC_CONTEXTS,
                    lz77::distance_symbol(distance as usize, 0)?,
                ));
            }
        }
    }
    Ok((tokens, parsed.enabled))
}

fn icc_candidate(literals: &[Literal], choice: usize, max_bytes: usize) -> Result<BitWriter> {
    let (tokens, lz77) = icc_tokens(literals, &icc_lz77_options(literals.len(), choice))?;
    let model = Model::build_with_options(
        std::slice::from_ref(&tokens),
        ICC_CONTEXTS + usize::from(lz77),
        lz77,
        Coding::Prefix,
        &icc_entropy_options(choice),
    )?;
    let mut trial = BitWriter::new(max_bytes);
    // The common U64 prefix is included in every trial, so comparison still
    // matches the source and the final append can be atomic at any bit offset.
    write_u64(literals.len() as u64, &mut trial)?;
    model.write_header(&mut trial)?;
    model.write_stream(&tokens, &mut trial)?;
    Ok(trial)
}

fn select_icc_candidate(literals: &[Literal], lossless_search: bool) -> Result<(usize, BitWriter)> {
    // At most one entropy symbol and 31 extra bits per literal; a match uses
    // two tokens for at least three literals. Leave room for all 42 histograms
    // and their nested map. Trial limits do not depend on destination capacity:
    // a large losing trial must not prevent a smaller fitting trial from winning.
    let max_bytes = literals
        .len()
        .checked_mul(8)
        .and_then(|n| n.checked_add(65_536))
        .ok_or(Error::Limit("ICC candidate output bound"))?;
    let choices = if lossless_search && literals.len() < ICC_SEARCH_LIMIT {
        4
    } else {
        1
    };
    let mut best = (0, icc_candidate(literals, 0, max_bytes)?);
    for choice in 1..choices {
        let candidate = icc_candidate(literals, choice, max_bytes)?;
        if candidate.bits_written() < best.1.bits_written() {
            best = (choice, candidate);
        }
    }
    Ok(best)
}

/// ICCANSContext, including the source's inclusive i <= 128 boundary.
pub fn icc_context(i: usize, previous: u8, previous_previous: u8) -> usize {
    if i <= 128 {
        return 0;
    }
    let a = match previous {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0 => 2,
        1 => 3,
        2..=15 => 4,
        255 => 6,
        241..=254 => 5,
        _ => 7,
    };
    let b = match previous_previous {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0..=15 => 2,
        241..=255 => 3,
        _ => 4,
    };
    1 + a + b * 8
}

fn in_bounds(pos: usize, len: usize, size: usize) -> bool {
    pos <= size && len <= size - pos
}
fn sane(len: usize) -> bool {
    len > 8 && len < SIZE_LIMIT
}
fn be32(data: &[u8], pos: usize) -> u32 {
    if !in_bounds(pos, 4, data.len()) {
        return 0;
    }
    u32::from_be_bytes(data[pos..pos + 4].try_into().expect("four bytes"))
}
fn keyword(data: &[u8], pos: usize) -> [u8; 4] {
    if !in_bounds(pos, 4, data.len()) {
        return *b"    ";
    }
    data[pos..pos + 4].try_into().expect("four bytes")
}
fn varint(mut value: u64, data: &mut Vec<u8>) {
    while value > 127 {
        data.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    data.push(value as u8);
}
fn initial_header(size: u32) -> [u8; 128] {
    let mut h = [0; 128];
    h[..4].copy_from_slice(&size.to_be_bytes());
    h[8] = 4;
    h[12..24].copy_from_slice(b"mntrRGB XYZ ");
    h[36..40].copy_from_slice(b"acsp");
    h[68..80].copy_from_slice(&[0, 0, 246, 214, 0, 1, 0, 0, 0, 0, 211, 45]);
    h
}
fn predict_header(icc: &[u8], header: &mut [u8; 128], i: usize) {
    if i == 8 {
        header[80..84].copy_from_slice(&icc[4..8]);
    }
    if i == 41 {
        if icc[40] == b'A' {
            header[41..44].copy_from_slice(b"PPL");
        }
        if icc[40] == b'M' {
            header[41..44].copy_from_slice(b"SFT");
        }
    }
    if i == 42 {
        if &icc[40..42] == b"SG" {
            header[42..44].copy_from_slice(b"I ");
        }
        if &icc[40..42] == b"SU" {
            header[42..44].copy_from_slice(b"NW");
        }
    }
}
fn unshuffle(data: &mut [u8], width: usize) {
    let height = data.len().div_ceil(width);
    let mut result = vec![0; data.len()];
    let mut s = 0;
    let mut j = 0;
    for &byte in data.iter() {
        result[j] = byte;
        j += height;
        if j >= data.len() {
            s += 1;
            j = s;
        }
    }
    data.copy_from_slice(&result);
}
fn linear_predict(
    data: &[u8],
    start: usize,
    i: usize,
    stride: usize,
    width: usize,
    order: u8,
) -> u8 {
    let pos = start + i;
    let base = start + (i & !(width - 1));
    let read = |n: usize| -> u32 {
        let p = base - n * stride;
        match width {
            1 => u32::from(data[p]),
            2 => u32::from(u16::from_be_bytes([data[p], data[p + 1]])),
            _ => be32(&data[..pos], p),
        }
    };
    let (a, b, c) = (read(1), read(2), read(3));
    let pred = match order {
        0 => a,
        1 => a.wrapping_mul(2).wrapping_sub(b),
        2 => a
            .wrapping_mul(3)
            .wrapping_sub(b.wrapping_mul(3))
            .wrapping_add(c),
        _ => 0,
    };
    (pred >> ((width - 1 - (i & (width - 1))) * 8)) as u8
}
fn predict_and_shuffle(
    data: &[u8],
    pos: &mut usize,
    num: usize,
    stride: usize,
    width: usize,
    order: u8,
    result: &mut Vec<u8>,
) -> Result<()> {
    if !in_bounds(*pos, num, data.len()) || *pos == 0 || ((*pos - 1) >> 2) < stride {
        return Err(Error::Encoding("ICC predictor range/stride".into()));
    }
    let start = result.len();
    for i in 0..num {
        result
            .push(data[*pos + i].wrapping_sub(linear_predict(data, *pos, i, stride, width, order)));
    }
    *pos += num;
    if width > 1 {
        unshuffle(&mut result[start..], width);
    }
    Ok(())
}
fn write_u64(mut value: u64, writer: &mut BitWriter) -> Result<()> {
    if value == 0 {
        writer.write(2, 0)?;
    } else if value <= 16 {
        writer.write(2, 1)?;
        writer.write(4, value - 1)?;
    } else if value <= 272 {
        writer.write(2, 2)?;
        writer.write(8, value - 17)?;
    } else {
        writer.write(2, 3)?;
        writer.write(12, value & 4095)?;
        value >>= 12;
        let mut shift = 12;
        while value != 0 && shift < 60 {
            writer.write(1, 1)?;
            writer.write(8, value & 255)?;
            value >>= 8;
            shift += 8;
        }
        if value != 0 {
            writer.write(1, 1)?;
            writer.write(4, value & 15)?;
        } else {
            writer.write(1, 0)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_varint(bytes: &[u8], pos: &mut usize) -> u64 {
        let mut result = 0;
        for shift in (0..64).step_by(7) {
            let b = bytes[*pos];
            *pos += 1;
            result |= u64::from(b & 127) << shift;
            if b < 128 {
                return result;
            }
        }
        panic!("invalid varint");
    }
    fn command_bytes(enc: &[u8]) -> &[u8] {
        let mut pos = 0;
        read_varint(enc, &mut pos);
        let len = read_varint(enc, &mut pos) as usize;
        &enc[pos..pos + len]
    }
    fn shuffle(bytes: &[u8], width: usize) -> Vec<u8> {
        let mut result = Vec::new();
        let height = bytes.len().div_ceil(width);
        for start in 0..height {
            for i in (start..bytes.len()).step_by(height) {
                result.push(bytes[i]);
            }
        }
        result
    }

    // Test-only command interpreter, following icc_codec.cc::UnpredictICC.
    // Scalar prediction is deliberately independent of linear_predict.
    fn inverse(enc: &[u8]) -> Vec<u8> {
        let mut p = 0;
        let size = read_varint(enc, &mut p) as usize;
        let count = read_varint(enc, &mut p) as usize;
        let mut c = p;
        let end = p + count;
        p = end;
        let mut result = Vec::new();
        let mut header = initial_header(size as u32);
        for i in 0..size.min(128) {
            predict_header(&result, &mut header, i);
            result.push(enc[p].wrapping_add(header[i]));
            p += 1;
        }
        if size > 128 {
            let tags = read_varint(enc, &mut c);
            if tags != 0 {
                result.extend_from_slice(&((tags - 1) as u32).to_be_bytes());
                let mut previous_start = 128 + (tags - 1) * 12;
                let mut previous_size = 0;
                while c < end {
                    let cmd = enc[c];
                    c += 1;
                    let code = cmd & 63;
                    if code == 0 {
                        break;
                    }
                    let tag = match code {
                        1 => {
                            let t = keyword(enc, p);
                            p += 4;
                            t
                        }
                        2 => *b"rTRC",
                        3 => *b"rXYZ",
                        _ => TAGS[usize::from(code - 4)],
                    };
                    let start = if cmd & 64 != 0 {
                        read_varint(enc, &mut c)
                    } else {
                        previous_start + previous_size
                    };
                    let mut len = if [
                        *b"rXYZ", *b"gXYZ", *b"bXYZ", *b"kXYZ", *b"wtpt", *b"bkpt", *b"lumi",
                    ]
                    .contains(&tag)
                    {
                        20
                    } else {
                        previous_size
                    };
                    if cmd & 128 != 0 {
                        len = read_varint(enc, &mut c);
                    }
                    result.extend_from_slice(&tag);
                    result.extend_from_slice(&(start as u32).to_be_bytes());
                    result.extend_from_slice(&(len as u32).to_be_bytes());
                    previous_start = start;
                    previous_size = len;
                    if code == 2 || code == 3 {
                        for (i, second) in if code == 2 {
                            [*b"gTRC", *b"bTRC"]
                        } else {
                            [*b"gXYZ", *b"bXYZ"]
                        }
                        .into_iter()
                        .enumerate()
                        {
                            let offset = start + if code == 3 { (i as u64 + 1) * len } else { 0 };
                            result.extend_from_slice(&second);
                            result.extend_from_slice(&(offset as u32).to_be_bytes());
                            result.extend_from_slice(&(len as u32).to_be_bytes());
                        }
                    }
                }
            }
        }
        while c < end {
            let cmd = enc[c];
            c += 1;
            match cmd {
                1..=3 => {
                    let len = read_varint(enc, &mut c) as usize;
                    if cmd == 1 {
                        result.extend_from_slice(&enc[p..p + len]);
                    } else {
                        result.extend(shuffle(&enc[p..p + len], if cmd == 2 { 2 } else { 4 }));
                    }
                    p += len;
                }
                4 => {
                    let flags = enc[c];
                    c += 1;
                    let width = usize::from((flags & 3) + 1);
                    let order = (flags >> 2) & 3;
                    let stride = if flags & 16 != 0 {
                        read_varint(enc, &mut c) as usize
                    } else {
                        width
                    };
                    let len = read_varint(enc, &mut c) as usize;
                    let residual = shuffle(&enc[p..p + len], width);
                    p += len;
                    let start = result.len();
                    assert!(start > stride * 4);
                    for (i, byte) in residual.into_iter().enumerate() {
                        let base = start + i / width * width;
                        let read = |distance: usize| -> i64 {
                            let pos = base - stride * distance;
                            if pos + width > result.len() {
                                return 0;
                            }
                            result[pos..pos + width]
                                .iter()
                                .fold(0, |v, &b| (v << 8) | i64::from(b))
                        };
                        let prediction = match order {
                            0 => read(1),
                            1 => 2 * read(1) - read(2),
                            2 => 3 * read(1) - 3 * read(2) + read(3),
                            _ => panic!("order"),
                        };
                        let predicted = (prediction >> ((width - 1 - i % width) * 8)) as u8;
                        result.push(byte.wrapping_add(predicted));
                    }
                }
                10 => {
                    result.extend_from_slice(b"XYZ \0\0\0\0");
                    result.extend_from_slice(&enc[p..p + 12]);
                    p += 12;
                }
                16..=23 => {
                    result.extend_from_slice(&TYPES[usize::from(cmd - 16)]);
                    result.extend_from_slice(&[0; 4]);
                }
                _ => panic!("unexpected command {cmd}"),
            }
        }
        assert_eq!(result.len(), size);
        assert_eq!(p, enc.len());
        result
    }

    fn profile(tags: &[([u8; 4], u32, u32)], body: &[u8]) -> Vec<u8> {
        let size = 132 + tags.len() * 12 + body.len();
        let mut result = initial_header(size as u32).to_vec();
        result.extend_from_slice(&(tags.len() as u32).to_be_bytes());
        for &(tag, start, len) in tags {
            result.extend_from_slice(&tag);
            result.extend_from_slice(&start.to_be_bytes());
            result.extend_from_slice(&len.to_be_bytes());
        }
        result.extend_from_slice(body);
        result
    }
    fn single(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut body = kind.to_vec();
        body.extend_from_slice(&[0; 4]);
        body.extend_from_slice(payload);
        profile(&[(*b"desc", 144, body.len() as u32)], &body)
    }
    fn roundtrip(input: &[u8]) {
        assert_eq!(inverse(&predict_icc(input).unwrap()), input);
    }

    fn stream_bits(mut writer: BitWriter) -> Vec<u8> {
        let count = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        let bytes = writer.into_bytes().unwrap();
        (0..count).map(|i| (bytes[i / 8] >> (i % 8)) & 1).collect()
    }

    fn read_bits(bits: &[u8], position: &mut usize, width: usize) -> u32 {
        let mut result = 0;
        for shift in 0..width {
            result |= u32::from(bits[*position]) << shift;
            *position += 1;
        }
        result
    }

    fn assert_source_lz77_header(bits: &[u8], position: &mut usize) {
        assert_eq!(read_bits(bits, position, 1), 1); // enabled
        assert_eq!(read_bits(bits, position, 2), 1); // min_symbol=512
        assert_eq!(read_bits(bits, position, 2), 0); // min_length=3
        assert_eq!(read_bits(bits, position, 4), 0); // length config 000
    }

    // Independent expansion of the entropy input, including overlapping copies
    // and the decoder's recomputed byte contexts at each non-copied position.
    fn expand_icc_tokens(tokens: &[Token]) -> Vec<u8> {
        let mut decoded = Vec::new();
        let mut index = 0;
        while index < tokens.len() {
            let token = tokens[index];
            let pos = decoded.len();
            let context = icc_context(
                pos,
                if pos > 0 { decoded[pos - 1] } else { 0 },
                if pos > 1 { decoded[pos - 2] } else { 0 },
            );
            assert_eq!(token.context, context);
            index += 1;
            if token.is_length {
                let distance = tokens[index];
                index += 1;
                assert!(!distance.is_length);
                assert_eq!(distance.context, ICC_CONTEXTS);
                let distance = distance.value as usize + 1;
                assert!(distance <= decoded.len());
                assert!(distance <= lz77::WINDOW_SIZE);
                let length = token.value as usize + 3;
                for _ in 0..length {
                    decoded.push(decoded[decoded.len() - distance]);
                }
            } else {
                decoded.push(u8::try_from(token.value).unwrap());
            }
        }
        decoded
    }

    #[test]
    fn writer_trials_have_source_order_and_size_boundary() {
        for choice in 0..4 {
            let options = icc_entropy_options(choice);
            assert_eq!(
                options.clustering,
                if choice & 1 == 0 {
                    ClusterStrategy::Best
                } else {
                    ClusterStrategy::Fast
                }
            );
            assert_eq!(
                options.uint_config,
                if choice & 2 == 0 {
                    UintConfigStrategy::Best
                } else {
                    UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap())
                }
            );
            assert!(options.prefix_singleton_guard);
            // Source clamps its default 256 cap to the raw contexts AFTER
            // adding the distance context, so an enabled trial may use 42.
            assert_eq!(options.max_histograms, 256);
            assert_eq!(options.lz77_min_symbol, 512);
            assert_eq!(
                options.lz77_length_config,
                HybridUintConfig::new(0, 0, 0).unwrap()
            );
            let small = icc_lz77_options(ICC_SEARCH_LIMIT - 1, choice);
            let large = icc_lz77_options(ICC_SEARCH_LIMIT, choice);
            assert_eq!(small.method, SourceLz77Method::Optc256);
            assert_eq!(large.method, SourceLz77Method::Lz77b3w3t);
            assert_eq!(
                small.literal_uint_config,
                if choice & 2 == 0 {
                    HybridUintConfig::default()
                } else {
                    HybridUintConfig::new(0, 0, 0).unwrap()
                }
            );
            assert_eq!(large.literal_uint_config, small.literal_uint_config);
            for config in [&small, &large] {
                assert!(config.force_huffman);
                assert_eq!(config.min_symbol(), options.lz77_min_symbol);
                assert_eq!(config.length_uint_config, options.lz77_length_config);
                assert_eq!(config.max_match_length, u32::MAX as usize);
                assert!(!config.bucket_matcher);
                assert!(!config.careful_first_pass);
                assert!(config.require_global_savings);
                assert!(!config.range_search);
                assert_eq!(config.match_cost, lz77::SourceMatchCost::SymbolId);
                assert_eq!(config.prefix_filter, lz77::FilterPolicy::Disabled);
            }
        }
    }

    #[test]
    fn parsed_icc_values_contexts_and_overlapping_copies_roundtrip() {
        for size in [4096, ICC_SEARCH_LIMIT] {
            let bytes: Vec<u8> = (0..=255).cycle().take(size).collect();
            let literals = icc_literals(&bytes).unwrap();
            for choice in [0, 2] {
                let (tokens, enabled) =
                    icc_tokens(&literals, &icc_lz77_options(size, choice)).unwrap();
                assert!(enabled);
                assert_eq!(expand_icc_tokens(&tokens), bytes);
                assert!(
                    tokens
                        .iter()
                        .any(|token| token.is_length && token.value as usize + 3 > 256)
                );
            }
        }
        let bytes = b"ab";
        let literals = icc_literals(bytes).unwrap();
        let (tokens, enabled) = icc_tokens(&literals, &icc_lz77_options(bytes.len(), 0)).unwrap();
        assert!(!enabled);
        assert_eq!(expand_icc_tokens(&tokens), bytes);
    }

    #[test]
    fn accepted_literal_only_parse_keeps_source_lz77_header() {
        let bytes = b"ab";
        let literals = icc_literals(bytes).unwrap();
        for choice in 0..4 {
            let mut config = icc_lz77_options(bytes.len(), choice);
            let (_, rejected_enabled) = icc_tokens(&literals, &config).unwrap();
            assert!(!rejected_enabled);
            // Exercise the accepted-but-literal-only state independently of
            // source savings thresholds. Production keeps this gate enabled.
            config.require_global_savings = false;
            let expected = lz77::parse_source_with_state(&literals, 0, &config).unwrap();
            assert!(expected.enabled);
            assert!(
                expected
                    .commands
                    .iter()
                    .all(|c| matches!(c, Command::Literal(_)))
            );
            let (tokens, enabled) = icc_tokens(&literals, &config).unwrap();
            assert_eq!(enabled, expected.enabled);
            assert!(tokens.iter().all(|t| !t.is_length));
            assert_eq!(expand_icc_tokens(&tokens), bytes);
            let streams = [tokens];
            let model = Model::build_with_options(
                &streams,
                ICC_CONTEXTS + usize::from(enabled),
                enabled,
                Coding::Prefix,
                &icc_entropy_options(choice),
            )
            .unwrap();
            let mut writer = BitWriter::new(8192);
            model.write_header(&mut writer).unwrap();
            model.write_stream(&streams[0], &mut writer).unwrap();
            assert_source_lz77_header(&stream_bits(writer), &mut 0);
        }
    }

    #[test]
    fn source_match_symbols_use_wide_prefix_alphabet_and_length000_wire() {
        let bytes: Vec<u8> = (0..=255).cycle().take(4096).collect();
        let literals = icc_literals(&bytes).unwrap();
        let (tokens, enabled) = icc_tokens(&literals, &icc_lz77_options(bytes.len(), 2)).unwrap();
        assert!(enabled);
        assert!(tokens.iter().any(|t| t.is_length));
        assert_eq!(expand_icc_tokens(&tokens), bytes);
        let streams = [tokens];
        let mut options = icc_entropy_options(2);
        options.max_histograms = 1;
        let model = Model::build_with_options(
            &streams,
            ICC_CONTEXTS + 1,
            enabled,
            Coding::Prefix,
            &options,
        )
        .unwrap();
        assert!(!model.uses_ans());
        assert_eq!(model.log_alpha_size(), 15);
        let mut writer = BitWriter::new(65536);
        model.write_header(&mut writer).unwrap();
        model.write_stream(&streams[0], &mut writer).unwrap();
        let bits = stream_bits(writer);
        let mut position = 0;
        assert_source_lz77_header(&bits, &mut position);
        assert_eq!(read_bits(&bits, &mut position, 1), 1); // simple context map
        assert_eq!(read_bits(&bits, &mut position, 2), 0); // one histogram
        assert_eq!(read_bits(&bits, &mut position, 1), 1); // prefix mode
        assert_eq!(read_bits(&bits, &mut position, 4), 0); // literal config 000
        assert_eq!(read_bits(&bits, &mut position, 1), 1); // nonzero alphabet-1
        let width = read_bits(&bits, &mut position, 4) as usize;
        let alphabet = 1 + (1u32 << width) + read_bits(&bits, &mut position, width);
        let maximum = streams[0]
            .iter()
            .map(|t| {
                if t.is_length {
                    options.lz77_min_symbol + options.lz77_length_config.encode(t.value).token
                } else {
                    HybridUintConfig::new(0, 0, 0)
                        .unwrap()
                        .encode(t.value)
                        .token
                }
            })
            .max()
            .unwrap();
        assert!(alphabet > 512);
        assert_eq!(alphabet, maximum + 1);
    }

    #[test]
    fn four_way_winner_measures_complete_serialized_bits() {
        let repeated = (0..=255).cycle().take(2048).collect::<Vec<u8>>();
        let input = single(*b"text", &repeated);
        let predicted = predict_icc(&input).unwrap();
        let literals = icc_literals(&predicted).unwrap();
        let mut trials = Vec::new();
        for choice in 0..4 {
            let (tokens, _) =
                icc_tokens(&literals, &icc_lz77_options(literals.len(), choice)).unwrap();
            assert_eq!(inverse(&expand_icc_tokens(&tokens)), input);
            trials.push(stream_bits(
                icc_candidate(&literals, choice, 1 << 20).unwrap(),
            ));
        }
        let mut expected = 0;
        for choice in 1..4 {
            if trials[choice].len() < trials[expected].len() {
                expected = choice;
            }
        }
        let (choice, selected) = select_icc_candidate(&literals, true).unwrap();
        assert_eq!(choice, expected);
        assert_eq!(stream_bits(selected), trials[expected]);
        let mut actual = BitWriter::new(1 << 20);
        write_icc(&input, &mut actual).unwrap();
        assert_eq!(stream_bits(actual), trials[expected]);
        let mut baseline = BitWriter::new(1 << 20);
        write_icc_with_search(&input, &mut baseline, false).unwrap();
        assert_eq!(stream_bits(baseline), trials[0]);
        assert!(trials[expected].len() <= trials[0].len());
    }

    #[test]
    fn ties_keep_first_trial_and_large_streams_skip_search() {
        // With one zero, all trial choices select the same config and histogram.
        let literals = icc_literals(&[0]).unwrap();
        let trials: Vec<Vec<u8>> = (0..4)
            .map(|choice| stream_bits(icc_candidate(&literals, choice, 8192).unwrap()))
            .collect();
        assert!(trials.iter().all(|bits| *bits == trials[0]));
        assert_eq!(select_icc_candidate(&literals, true).unwrap().0, 0);
        let bytes: Vec<u8> = (0..=255).cycle().take(ICC_SEARCH_LIMIT).collect();
        let literals = icc_literals(&bytes).unwrap();
        let (choice, searched) = select_icc_candidate(&literals, true).unwrap();
        let (baseline_choice, baseline) = select_icc_candidate(&literals, false).unwrap();
        assert_eq!((choice, baseline_choice), (0, 0));
        assert_eq!(stream_bits(searched), stream_bits(baseline));
    }

    #[test]
    fn writer_limit_errors_are_atomic_and_exact_size_output_fits() {
        let input = single(*b"text", b"profile contents");
        let mut reference = BitWriter::new(1 << 20);
        write_icc(&input, &mut reference).unwrap();
        let expected = stream_bits(reference);
        for prefix in 0..8u8 {
            let total = expected.len() + usize::from(prefix);
            let mut exact = BitWriter::new(total.div_ceil(8));
            exact.write(prefix, (1u64 << prefix) - 1).unwrap();
            write_icc(&input, &mut exact).unwrap();
            let bits = stream_bits(exact);
            assert!(bits[..usize::from(prefix)].iter().all(|&bit| bit == 1));
            assert_eq!(&bits[usize::from(prefix)..], expected);
            let mut too_small = BitWriter::new(total.div_ceil(8) - 1);
            too_small.write(prefix, (1u64 << prefix) - 1).unwrap();
            assert!(write_icc(&input, &mut too_small).is_err());
            assert_eq!(too_small.bits_written(), usize::from(prefix));
            assert_eq!(stream_bits(too_small), vec![1; usize::from(prefix)]);
        }
    }

    #[test]
    fn source_derived_header_and_command_goldens() {
        for (input, expected) in [
            (b"".as_slice(), vec![0, 0]),
            (b"a", vec![1, 0, 97]),
            (b"ab", vec![2, 0, 97, 98]),
            (b"aaaa", vec![4, 0, 97, 97, 97, 93]),
        ] {
            assert_eq!(predict_icc(input).unwrap(), expected);
            roundtrip(input);
        }
        let mut body = b"XYZ \0\0\0\0".to_vec();
        body.extend_from_slice(&[1; 12]);
        let input = profile(&[], &body);
        let mut expected = vec![152, 1, 3, 1, 0, 10];
        expected.extend_from_slice(&[0; 128]);
        expected.extend_from_slice(&[1; 12]);
        assert_eq!(predict_icc(&input).unwrap(), expected);
        roundtrip(&input);
        let input = single(*b"mluc", &[1, 2, 3, 4, 5]);
        let enc = predict_icc(&input).unwrap();
        assert_eq!(command_bytes(&enc), &[2, 208, 144, 1, 13, 0, 19, 2, 5]);
        assert_eq!(&enc[enc.len() - 5..], &[1, 3, 5, 2, 4]);
        roundtrip(&input);
    }

    #[test]
    fn source_trc_and_xyz_tag_folding_goldens() {
        let mut xyz = b"XYZ \0\0\0\0".to_vec();
        xyz.extend_from_slice(&[255; 12]);
        let input = profile(
            &[
                (*b"rXYZ", 168, 20),
                (*b"gXYZ", 188, 20),
                (*b"bXYZ", 208, 20),
            ],
            &xyz.repeat(3),
        );
        assert_eq!(
            command_bytes(&predict_icc(&input).unwrap()),
            &[4, 67, 168, 1, 0, 10, 10, 10]
        );
        roundtrip(&input);
        let mut curve = b"curv\0\0\0\0".to_vec();
        curve.extend_from_slice(&[255; 24]);
        let input = profile(
            &[
                (*b"rTRC", 168, 32),
                (*b"gTRC", 168, 32),
                (*b"bTRC", 168, 32),
            ],
            &curve,
        );
        assert_eq!(
            command_bytes(&predict_icc(&input).unwrap()),
            &[4, 194, 168, 1, 32, 0, 21, 4, 5, 24]
        );
        roundtrip(&input);
        // Strict '< size' folding condition: no payload means no folding.
        let truncated = &input[..168];
        roundtrip(truncated);
    }

    #[test]
    fn curves_gamut_lut_and_embedded_curves_roundtrip() {
        let payload: Vec<u8> = (0..257).map(|i| (i * 73 + 253) as u8).collect();
        for kind in [*b"mluc", *b"curv", *b"gbd "] {
            for len in [1, 16, 17, 25, 256, 257] {
                roundtrip(&single(kind, &payload[..len]));
            }
        }
        for kind in [*b"mAB ", *b"mBA "] {
            let mut payload = vec![0; 432];
            payload[0] = 2;
            payload[1] = 3;
            payload[16..20].copy_from_slice(&32u32.to_be_bytes());
            payload[24] = 8;
            payload[25] = 8;
            payload[40] = 2;
            let input = single(kind, &payload);
            assert!(
                command_bytes(&predict_icc(&input).unwrap())
                    .windows(5)
                    .any(|w| w == [4, 21, 6, 128, 3])
            );
            roundtrip(&input);
            payload = vec![0; 56];
            payload[24..28].copy_from_slice(b"curv");
            payload[32..36].copy_from_slice(&10u32.to_be_bytes());
            payload[36..].copy_from_slice(&[255; 20]);
            let input = single(kind, &payload);
            assert!(
                command_bytes(&predict_icc(&input).unwrap())
                    .windows(3)
                    .any(|w| w == [4, 5, 20])
            );
            roundtrip(&input);
        }
    }

    #[test]
    fn arbitrary_truncated_data_and_wrapping_prediction() {
        let mut state = 42u32;
        let data: Vec<u8> = (0..513)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect();
        for len in 0..data.len() {
            roundtrip(&data[..len]);
        }
        let tagged = single(*b"curv", &data);
        for len in 0..tagged.len() {
            roundtrip(&tagged[..len]);
        }
        for width in [1, 2, 4] {
            let mut input = vec![0; 32];
            input[16 - width..16].fill(255);
            input[16 - width * 2..16 - width].fill(0);
            input[15 - width] = 1;
            for i in 0..width {
                assert_eq!(
                    linear_predict(&input, 16, i, width, width, 1),
                    if i + 1 == width { 253 } else { 255 }
                );
            }
        }
        let mut pos = 8;
        assert!(predict_and_shuffle(&[0; 32], &mut pos, 8, 2, 2, 1, &mut Vec::new()).is_err());
        assert_eq!(pos, 8);
        // ICCPredictHeader predicts bytes 41..43 after A/M, but only bytes
        // 42..43 after SG/SU. Preserve the unpredicted platform prefix exactly.
        for (platform, residual) in [
            (*b"APPL", [b'A', 0, 0, 0]),
            (*b"MSFT", [b'M', 0, 0, 0]),
            (*b"SGI ", [b'S', b'G', 0, 0]),
            (*b"SUNW", [b'S', b'U', 0, 0]),
        ] {
            let mut input = initial_header(128);
            input[40..44].copy_from_slice(&platform);
            input[4..8].copy_from_slice(b"test");
            input[80..84].copy_from_slice(b"test");
            roundtrip(&input);
            let enc = predict_icc(&input).unwrap();
            assert_eq!(&enc[3 + 40..3 + 44], &residual);
            assert_eq!(&enc[3 + 80..3 + 84], &[0; 4]);
        }
    }

    #[test]
    fn profile_validation_contexts_and_unaligned_streams() {
        let good = single(*b"text", b"profile");
        assert_eq!(validate_profile(&good).unwrap(), IccColorSpace::Rgb);
        for index in [0, 16, 20, 36, 128, 136, 140] {
            let mut bad = good.clone();
            bad[index] = 255;
            assert!(validate_profile(&bad).is_err());
        }
        for b1 in 0..=255u8 {
            for b2 in 0..=255u8 {
                assert_eq!(icc_context(128, b1, b2), 0);
                assert!((1..=40).contains(&icc_context(129, b1, b2)));
            }
        }
        assert_eq!(icc_context(129, 255, 255), 31);
        assert_eq!(icc_context(129, 240, 240), 40);
        for prefix in 0..8u8 {
            let mut w = BitWriter::new(16384);
            w.write(prefix, (1u64 << prefix) - 1).unwrap();
            let mut reference = BitWriter::new(16384);
            write_icc(&good, &mut reference).unwrap();
            let n = reference.bits_written();
            write_icc(&good, &mut w).unwrap();
            assert_eq!(w.bits_written(), n + usize::from(prefix));
            w.zero_pad_to_byte().unwrap();
            reference.zero_pad_to_byte().unwrap();
            let (actual, expected) = (w.bytes().unwrap(), reference.bytes().unwrap());
            for i in 0..n {
                let j = i + usize::from(prefix);
                assert_eq!(
                    (actual[j / 8] >> (j % 8)) & 1,
                    (expected[i / 8] >> (i % 8)) & 1
                );
            }
        }
        let mut w = BitWriter::new(8);
        w.write(3, 7).unwrap();
        assert!(write_icc(&[], &mut w).is_err());
        assert_eq!(w.bits_written(), 3);
    }
}
