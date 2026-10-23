// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Streaming TOC helpers from libjxl b866b3defec13494bfbaf705db93581cda118e37:
//! enc_frame.cc::ComputePermutationForStreaming/ComputeGroupDataOffset/
//! ComputeDcGlobalPadding, enc_toc.cc, enc_coeff_order.cc::EncodePermutation,
//! coeff_order.cc::CoeffOrderContext and lehmer_code.h::ComputeLehmerCode.
//!
//! Permutations map canonical section IDs to physical positions. Sizes are in
//! PHYSICAL order. Offsets are relative to the frame start, not a container box.
//! The header byte count includes WriteTocPermutation's final byte padding but
//! excludes TOC sizes. The caller owns output limits, seeking and group emission.
//! No out-of-order jxlp/version-1 framing is introduced here.

use crate::bit_writer::BitWriter;
use crate::entropy::{
    HybridUintConfig,
    options::EntropyOptions,
    stream::{Coding, Model, Token},
};
use crate::tools::lz77::{self, Literal, SourceConfig};
use crate::{Error, Result};

const DC_GROUP_FACTOR: usize = 8;
const PERMUTATION_CONTEXTS: usize = 8;
const SIZE_OFFSETS: [usize; 4] = [0, 1024, 17408, 4211712];
const SIZE_BITS: [usize; 4] = [12, 16, 24, 32];
pub(crate) const MAX_SECTION_SIZE: usize = 4211712 + (1 << 30) - 1;

#[derive(Debug)]
#[allow(dead_code)] // Seekable input currently supplies FrameAreaPlan's permutation.
pub(crate) struct StreamingTocPlan {
    pub(crate) permutation: Vec<u32>,
    pub(crate) dc_group_order: Vec<usize>,
}

#[allow(dead_code)] // Keep the source geometry-only planner independently usable.
impl StreamingTocPlan {
    /// The input dimensions must already be in the coded frame's coordinate
    /// system; group_size is the actual group dimension, including Modular's
    /// group_size_shift. Caller geometry must agree with FrameDimensions' grid;
    /// do not pass upsampled display sizes or substitute an extra-channel grid.
    /// Frame-header pass-count/geometry legality remains the caller's contract.
    ///
    /// Rejects the single-group/single-pass collapsed TOC: the source streaming
    /// route only uses separate DC/AC sections, not the collapsed representation.
    pub(crate) fn new(
        width: usize,
        height: usize,
        group_size: usize,
        passes: usize,
        max_sections: usize,
    ) -> Result<Self> {
        if width == 0 || height == 0 || group_size == 0 || passes == 0 {
            return Err(Error::InvalidInput(
                "zero streaming TOC dimension/count".into(),
            ));
        }
        let dc_size = group_size
            .checked_mul(DC_GROUP_FACTOR)
            .ok_or(Error::Limit("streaming DC group dimension"))?;
        let groups_x = width.div_ceil(group_size);
        let groups_y = height.div_ceil(group_size);
        let dc_x = width.div_ceil(dc_size);
        let dc_y = height.div_ceil(dc_size);
        let groups = groups_x
            .checked_mul(groups_y)
            .ok_or(Error::Limit("streaming group count"))?;
        let dc_groups = dc_x
            .checked_mul(dc_y)
            .ok_or(Error::Limit("streaming DC group count"))?;
        if groups == 1 && passes == 1 {
            return Err(Error::InvalidInput(
                "single-group TOC is collapsed, not streaming".into(),
            ));
        }
        let ac_start = dc_groups
            .checked_add(2)
            .ok_or(Error::Limit("streaming section count"))?;
        let sections = groups
            .checked_mul(passes)
            .and_then(|n| n.checked_add(ac_start))
            .ok_or(Error::Limit("streaming section count"))?;
        validate_count(sections, max_sections)?;
        let mut permutation = zeroes::<u32>(sections)?;
        let mut dc_group_order = reserved(dc_groups)?;
        let mut position = 1usize; // DC Global is at physical position zero.
        for y in 0..dc_y {
            for x in 0..dc_x {
                let dc = y * dc_x + x;
                dc_group_order.push(dc);
                permutation[1 + dc] = position as u32;
                position += 1;
                let y0 = y * DC_GROUP_FACTOR;
                let x0 = x * DC_GROUP_FACTOR;
                let y1 = y0 + (groups_y - y0).min(DC_GROUP_FACTOR);
                let x1 = x0 + (groups_x - x0).min(DC_GROUP_FACTOR);
                for pass in 0..passes {
                    for gy in y0..y1 {
                        for gx in x0..x1 {
                            let canonical = ac_start + pass * groups + gy * groups_x + gx;
                            permutation[canonical] = position as u32;
                            position += 1;
                        }
                    }
                }
            }
        }
        // AC Global is last, including the empty Modular AC Global section.
        permutation[1 + dc_groups] = position as u32;
        if position + 1 != sections {
            return Err(Error::Encoding("streaming TOC section coverage".into()));
        }
        Ok(Self {
            permutation,
            dc_group_order,
        })
    }
}

fn reserved<T>(count: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| Error::Limit("streaming TOC scratch allocation"))?;
    Ok(out)
}

fn zeroes<T: Default + Clone>(count: usize) -> Result<Vec<T>> {
    let mut out = reserved(count)?;
    out.resize(count, T::default());
    Ok(out)
}

fn validate_count(count: usize, maximum: usize) -> Result<()> {
    // The shared source RLE/cost engine has signed-32-bit stream counts. This
    // also bounds Fenwick increments and all uint32 permutation token values.
    if count == 0 {
        return Err(Error::InvalidInput("empty permutation".into()));
    }
    if count > maximum || count > i32::MAX as usize {
        return Err(Error::Limit("TOC permutation entries"));
    }
    Ok(())
}

fn permutation_context(value: u32) -> usize {
    HybridUintConfig::new(0, 0, 0)
        .expect("constant config")
        .encode(value)
        .token
        .min(PERMUTATION_CONTEXTS as u32 - 1) as usize
}

fn tokenize_permutation(order: &[u32], skip: usize, max_entries: usize) -> Result<Vec<Token>> {
    validate_count(order.len(), max_entries)?;
    if skip > order.len() {
        return Err(Error::InvalidInput("permutation skip exceeds size".into()));
    }
    let mut seen = zeroes::<u8>(order.len().div_ceil(8))?;
    let mut tree = zeroes::<u32>(order.len() + 1)?;
    let mut code = reserved(order.len())?;
    for &symbol in order {
        let symbol_index = symbol as usize;
        if symbol_index >= order.len() || seen[symbol_index / 8] & (1 << (symbol_index % 8)) != 0 {
            return Err(Error::InvalidInput(
                "TOC permutation is not a bijection".into(),
            ));
        }
        seen[symbol_index / 8] |= 1 << (symbol_index % 8);
        let mut penalty = 0;
        let mut i = symbol_index + 1;
        while i != 0 {
            penalty += tree[i];
            i &= i - 1;
        }
        code.push(symbol - penalty);
        i = symbol_index + 1;
        while i < tree.len() {
            tree[i] += 1;
            // count <= i32::MAX makes this safe even on 32-bit hosts.
            i += i & i.wrapping_neg();
        }
    }
    if code[..skip].iter().any(|&digit| digit != 0) {
        return Err(Error::InvalidInput(
            "skipped permutation prefix is not identity".into(),
        ));
    }
    let mut end = order.len();
    while end > skip && code[end - 1] == 0 {
        end -= 1;
    }
    let mut tokens = reserved(1 + end - skip)?;
    tokens.push(Token::literal(
        permutation_context(order.len() as u32),
        (end - skip) as u32,
    ));
    let mut last = 0;
    for &digit in &code[skip..end] {
        tokens.push(Token::literal(permutation_context(last), digit));
        last = digit;
    }
    Ok(tokens)
}

fn permutation_model(tokens: Vec<Token>) -> Result<(Model, Vec<Token>)> {
    let mut literals = reserved(tokens.len())?;
    literals.extend(tokens.iter().map(|t| Literal {
        context: t.context as u32,
        value: t.value,
    }));
    drop(tokens);
    // HistogramParams() uses Best uint/clustering, Precise histograms and RLE;
    // its LZ77Params length config is 000, not EntropyOptions' legacy 400.
    let options = EntropyOptions {
        // BuildAndEncodeHistograms fixes this cap before adding LZ distance.
        max_histograms: PERMUTATION_CONTEXTS,
        lz77_length_config: HybridUintConfig::new(0, 0, 0).expect("constant config"),
        ..EntropyOptions::default()
    };
    let parsed = lz77::parse_source_rle_with_state(&literals, 0, &SourceConfig::default())?;
    drop(literals);
    let enabled = parsed.enabled;
    let (tokens, _) =
        crate::modular::encode::commands_to_tokens(parsed.commands, PERMUTATION_CONTEXTS, 0)?;
    let model = Model::build_with_options(
        std::slice::from_ref(&tokens),
        PERMUTATION_CONTEXTS + usize::from(enabled),
        enabled,
        Coding::Source,
        &options,
    )?;
    Ok((model, tokens))
}

/// Generic EncodePermutation: no presence flag or byte padding. Unlike the
/// DCT8-only helper, this supports arbitrary sizes up to the explicit budget.
/// Errors from the bounded writer can leave a partial output; discard it.
pub(crate) fn encode_permutation(
    order: &[u32],
    skip: usize,
    writer: &mut BitWriter,
    max_entries: usize,
) -> Result<()> {
    let (model, tokens) = permutation_model(tokenize_permutation(order, skip, max_entries)?)?;
    model.write_header(writer)?;
    model.write_stream(&tokens, writer)
}

/// Append the source presence flag, entropy-coded permutation and byte padding
/// to an UNPADDED frame-header writer. None and Some(&[]) mean no permutation;
/// a nonempty identity permutation is explicitly encoded, as in enc_toc.cc.
/// max_entries bounds scratch; BitWriter bounds the combined header output.
/// On any failure the caller must discard this partially written header.
pub(crate) fn write_permutation_with_limit(
    order: Option<&[u32]>,
    writer: &mut BitWriter,
    max_entries: usize,
) -> Result<()> {
    let order = order.unwrap_or_default();
    writer.write(1, u64::from(!order.is_empty()))?;
    if !order.is_empty() {
        encode_permutation(order, 0, writer, max_entries)?;
    }
    writer.zero_pad_to_byte()?;
    Ok(())
}

/// Parent seek-writer hook. The caller has already bounded its section vector;
/// this uses that vector's length as the scratch-entry budget. Empty means no
/// permutation. It appends to an unpadded header and pads before TOC sizes.
pub(crate) fn write_permutation(writer: &mut BitWriter, order: &[u32]) -> Result<()> {
    write_permutation_with_limit(Some(order), writer, order.len())
}

fn bucket(size: usize) -> Result<usize> {
    if size > MAX_SECTION_SIZE {
        return Err(Error::Limit("TOC section size"));
    }
    Ok(if size < SIZE_OFFSETS[1] {
        0
    } else if size < SIZE_OFFSETS[2] {
        1
    } else if size < SIZE_OFFSETS[3] {
        2
    } else {
        3
    })
}

/// Source TOCSize, excluding the permutation prefix. Requires nonempty sizes.
pub(crate) fn toc_size(sizes: &[usize]) -> Result<usize> {
    let (&first, rest) = sizes
        .split_first()
        .ok_or_else(|| Error::InvalidInput("empty TOC sizes".into()))?;
    toc_size_with_first(first, rest)
}

fn toc_size_with_first(first: usize, rest: &[usize]) -> Result<usize> {
    let mut bits = SIZE_BITS[bucket(first)?];
    for &size in rest {
        bits = bits
            .checked_add(SIZE_BITS[bucket(size)?])
            .ok_or(Error::Limit("TOC bit count"))?;
    }
    Ok(bits.div_ceil(8))
}

/// Writes physical-order sizes after a byte-aligned permutation prefix and
/// pads before the first group. All size-domain errors precede output writes.
pub(crate) fn write_sizes(writer: &mut BitWriter, sizes: &[usize]) -> Result<()> {
    if !writer.bits_written().is_multiple_of(8) {
        return Err(Error::InvalidInput(
            "TOC sizes require a byte-aligned prefix".into(),
        ));
    }
    toc_size(sizes)?;
    for &size in sizes {
        let b = bucket(size)?;
        writer.write(2, b as u64)?;
        writer.write((SIZE_BITS[b] - 2) as u8, (size - SIZE_OFFSETS[b]) as u64)?;
    }
    writer.zero_pad_to_byte()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupDataOffset {
    pub(crate) min_dc_global_size: usize,
    pub(crate) group_data_offset: usize,
}

/// Parent seek-writer hook: (minimum DC Global size, frame-relative offset).
pub(crate) fn group_data_offset(
    header_len: usize,
    dc_global_len: usize,
    count: usize,
) -> Result<(usize, usize)> {
    let result = compute_group_data_offset(header_len, dc_global_len, count)?;
    Ok((result.min_dc_global_size, result.group_data_offset))
}

/// Reserve the largest possible TOC before DC-group/AC-group output. The DC
/// Global bucket is stabilized so later padding cannot change its TOC width.
/// The returned offset is frame-relative; the sink base must be added checked.
pub(crate) fn compute_group_data_offset(
    frame_header_size: usize,
    dc_global_size: usize,
    num_sections: usize,
) -> Result<GroupDataOffset> {
    let others = num_sections
        .checked_sub(1)
        .ok_or_else(|| Error::InvalidInput("zero TOC section count".into()))?;
    let mut max_bits = others
        .checked_mul(32)
        .ok_or(Error::Limit("TOC bit count"))?;
    let min_bits = others
        .checked_mul(12)
        .ok_or(Error::Limit("TOC bit count"))?;
    let max_padding = (max_bits - min_bits).div_ceil(8);
    let mut minimum = dc_global_size;
    let mut b = bucket(minimum)?;
    loop {
        let padded = minimum
            .checked_add(max_padding)
            .ok_or(Error::Limit("DC Global padding"))?;
        let next = bucket(padded)?;
        if next == b {
            break;
        }
        b = next;
        minimum = SIZE_OFFSETS[b];
    }
    max_bits = max_bits
        .checked_add(SIZE_BITS[b])
        .ok_or(Error::Limit("TOC bit count"))?;
    let group_data_offset = frame_header_size
        .checked_add(max_bits.div_ceil(8))
        .and_then(|n| n.checked_add(minimum))
        .ok_or(Error::Limit("streaming group offset"))?;
    Ok(GroupDataOffset {
        min_dc_global_size: minimum,
        group_data_offset,
    })
}

/// Computes bytes to append to DC Global, without cloning sizes. The caller
/// adds this result to sizes[0] BEFORE write_sizes. Rejects stale/inconsistent
/// reservations, underflow, overflow, and padding outside the reserved bucket.
pub(crate) fn compute_dc_global_padding(
    group_sizes: &[usize],
    frame_header_size: usize,
    group_data_offset: usize,
    min_dc_global_size: usize,
) -> Result<usize> {
    let (&dc_size, rest) = group_sizes
        .split_first()
        .ok_or_else(|| Error::InvalidInput("empty TOC sizes".into()))?;
    bucket(dc_size)?;
    if min_dc_global_size < dc_size {
        return Err(Error::InvalidInput(
            "DC Global exceeds its minimum reservation".into(),
        ));
    }
    let size = toc_size_with_first(min_dc_global_size, rest)?;
    let actual_offset = frame_header_size
        .checked_add(size)
        .and_then(|n| n.checked_add(dc_size))
        .ok_or(Error::Limit("streaming group offset"))?;
    let padding = group_data_offset
        .checked_sub(actual_offset)
        .ok_or_else(|| {
            Error::InvalidInput("streaming group offset precedes DC Global end".into())
        })?;
    let padded = dc_size
        .checked_add(padding)
        .ok_or(Error::Limit("DC Global padding"))?;
    if padded < min_dc_global_size || bucket(padded)? != bucket(min_dc_global_size)? {
        return Err(Error::InvalidInput(
            "DC Global padding changes the reserved TOC bucket".into(),
        ));
    }
    Ok(padding)
}

/// Parent seek-writer hook; sizes are in physical emission order.
pub(crate) fn dc_global_padding(
    sizes: &[usize],
    header_len: usize,
    group_offset: usize,
    min_dc_global: usize,
) -> Result<usize> {
    compute_dc_global_padding(sizes, header_len, group_offset, min_dc_global)
}

#[cfg(test)]
mod tests {
    use super::*;

    // O(n^2) order-statistics oracle, deliberately independent of Fenwick sums.
    fn reference_tokens(order: &[u32], skip: usize) -> Vec<Token> {
        let context = |v: u32| (u32::BITS - v.leading_zeros()).min(7) as usize;
        let mut remaining: Vec<_> = (0..order.len() as u32).collect();
        let mut code: Vec<_> = order
            .iter()
            .map(|s| {
                let index = remaining.iter().position(|v| v == s).unwrap();
                remaining.remove(index);
                index as u32
            })
            .collect();
        while code.len() > skip && code.last() == Some(&0) {
            code.pop();
        }
        let mut out = vec![Token::literal(
            context(order.len() as u32),
            (code.len() - skip) as u32,
        )];
        let mut last = 0;
        for &digit in &code[skip..] {
            out.push(Token::literal(context(last), digit));
            last = digit;
        }
        out
    }

    fn shuffled(size: u32) -> Vec<u32> {
        let mut order: Vec<_> = (0..size).collect();
        let mut state = 7u32;
        for i in (1..order.len()).rev() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            order.swap(i, state as usize % (i + 1));
        }
        order
    }

    #[test]
    fn source_streaming_raster_order_golden_and_multiple_passes() {
        let plan = StreamingTocPlan::new(9 * 128, 128, 128, 2, 23).unwrap();
        assert_eq!(plan.dc_group_order, [0, 1]);
        assert_eq!(
            plan.permutation,
            [
                0, 1, 18, 21, 2, 3, 4, 5, 6, 7, 8, 9, 19, 10, 11, 12, 13, 14, 15, 16, 17, 20
            ]
        );
        for group in [128, 256, 512, 1024] {
            for passes in [1, 2, 7] {
                let p = StreamingTocPlan::new(16 * group + 1, 8 * group + 1, group, passes, 1200)
                    .unwrap();
                assert_eq!(p.dc_group_order, [0, 1, 2, 3, 4, 5]);
                let mut physical = vec![usize::MAX; p.permutation.len()];
                for (canonical, &position) in p.permutation.iter().enumerate() {
                    assert_eq!(physical[position as usize], usize::MAX);
                    physical[position as usize] = canonical;
                }
                let mut expected = vec![0];
                for dc in 0..6 {
                    expected.push(dc + 1);
                    for pass in 0..passes {
                        for g in 0..17 * 9 {
                            if (g / 17) / 8 * 3 + (g % 17) / 8 == dc {
                                expected.push(8 + pass * 17 * 9 + g);
                            }
                        }
                    }
                }
                expected.push(7);
                assert_eq!(physical, expected);
            }
        }
    }

    #[test]
    fn plan_rejects_collapsed_invalid_and_unbounded_geometry() {
        assert!(StreamingTocPlan::new(128, 128, 128, 1, 99).is_err());
        assert!(StreamingTocPlan::new(128, 128, 128, 2, 5).is_ok());
        for args in [
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
            (usize::MAX, usize::MAX, 1, 1),
            (2, 1, usize::MAX, 1),
            (2, 1, 1, usize::MAX),
        ] {
            assert!(StreamingTocPlan::new(args.0, args.1, args.2, args.3, usize::MAX).is_err());
        }
        assert!(StreamingTocPlan::new(256, 128, 128, 1, 4).is_err());
    }

    #[test]
    fn generic_lehmer_contexts_trimming_and_skip_match_rank_oracle() {
        for n in [1, 2, 7, 64, 65, 127, 128, 257, 1024] {
            let order = shuffled(n);
            assert_eq!(
                tokenize_permutation(&order, 0, n as usize).unwrap(),
                reference_tokens(&order, 0)
            );
            let identity: Vec<_> = (0..n).collect();
            assert_eq!(
                tokenize_permutation(&identity, 0, n as usize).unwrap(),
                reference_tokens(&identity, 0)
            );
        }
        let mut order: Vec<_> = (0..257).collect();
        order[11..].reverse();
        for skip in [0, 1, 11] {
            assert_eq!(
                tokenize_permutation(&order, skip, 257).unwrap(),
                reference_tokens(&order, skip)
            );
        }
        assert!(tokenize_permutation(&order, 12, 257).is_err());
        assert_eq!(
            tokenize_permutation(&[0, 1, 2], 3, 3).unwrap(),
            [Token::literal(2, 0)]
        );
        for order in [&[][..], &[0, 0], &[1], &[0, 2], &[u32::MAX]] {
            assert!(tokenize_permutation(order, 0, 10).is_err());
        }
        assert!(tokenize_permutation(&[0], 2, 10).is_err());
        assert!(tokenize_permutation(&[0, 1], 0, 1).is_err());
    }

    #[test]
    fn source_no_permutation_padding_and_explicit_identity() {
        for alignment in 0..8 {
            for absent in [None, Some(&[][..])] {
                let mut w = BitWriter::new(1);
                w.write(alignment, (1u64 << alignment) - 1).unwrap();
                write_permutation_with_limit(absent, &mut w, 0).unwrap();
                assert_eq!(w.bytes().unwrap(), &[(1u8 << alignment) - 1]);
            }
            let identity: Vec<_> = (0..65).collect();
            let mut w = BitWriter::new(100);
            w.write(alignment, 0).unwrap();
            write_permutation(&mut w, &identity).unwrap();
            assert_eq!(
                (w.bytes().unwrap()[alignment as usize / 8] >> (alignment % 8)) & 1,
                1
            );
            assert!(w.bits_written() > 8);
        }
    }

    #[test]
    fn actual_entropy_writer_matches_independent_tokens_and_is_bounded() {
        let mut remaining: Vec<_> = (0..512).collect();
        let repeated: Vec<_> = (0..512)
            .map(|i| {
                let rank = if i < 480 { 2 + (i / 32) % 2 } else { 0 };
                remaining.remove(rank)
            })
            .collect();
        let orders = [shuffled(65), shuffled(257), repeated];
        let mut saw_ans = false;
        let mut saw_rle = false;
        for order in orders {
            let raw = reference_tokens(&order, 0);
            let literals: Vec<_> = raw
                .iter()
                .map(|t| Literal {
                    context: t.context as u32,
                    value: t.value,
                })
                .collect();
            let parsed =
                lz77::parse_source_rle_with_state(&literals, 0, &SourceConfig::default()).unwrap();
            saw_rle |= parsed.enabled;
            let (tokens, _) =
                crate::modular::encode::commands_to_tokens(parsed.commands, 8, 0).unwrap();
            let options = EntropyOptions {
                max_histograms: 8,
                lz77_length_config: HybridUintConfig::new(0, 0, 0).unwrap(),
                ..Default::default()
            };
            let model = Model::build_with_options(
                std::slice::from_ref(&tokens),
                8 + usize::from(parsed.enabled),
                parsed.enabled,
                Coding::Source,
                &options,
            )
            .unwrap();
            saw_ans |= model.uses_ans();
            let mut expected = BitWriter::new(65536);
            expected.write(4, 13).unwrap();
            expected.write(1, 1).unwrap();
            model.write_header(&mut expected).unwrap();
            model.write_stream(&tokens, &mut expected).unwrap();
            expected.zero_pad_to_byte().unwrap();
            let expected = expected.into_bytes().unwrap();
            let mut actual = BitWriter::new(expected.len());
            actual.write(4, 13).unwrap();
            write_permutation(&mut actual, &order).unwrap();
            assert_eq!(actual.bytes().unwrap(), expected);
            let mut short = BitWriter::new(expected.len() - 1);
            short.write(4, 13).unwrap();
            assert!(write_permutation(&mut short, &order).is_err());
            assert!(
                write_permutation_with_limit(
                    Some(&order),
                    &mut BitWriter::new(65536),
                    order.len() - 1
                )
                .is_err()
            );
        }
        assert!(saw_ans);
        assert!(saw_rle);
        let mut w = BitWriter::new(128);
        encode_permutation(&[0, 1, 3, 2], 2, &mut w, 4).unwrap();
        assert!(w.bits_written() > 0);
    }

    fn read_bits(bytes: &[u8], cursor: &mut usize, bits: usize) -> usize {
        let mut value = 0;
        for i in 0..bits {
            value |= usize::from((bytes[*cursor / 8] >> (*cursor % 8)) & 1) << i;
            *cursor += 1;
        }
        value
    }

    #[test]
    fn toc_wire_bucket_goldens_and_separate_size_alignment() {
        for (size, expected) in [
            (0, vec![0, 0]),
            (1023, vec![0xfc, 0x0f]),
            (1024, vec![1, 0]),
            (17407, vec![0xfd, 0xff]),
            (17408, vec![2, 0, 0]),
            (4211711, vec![0xfe, 0xff, 0xff]),
            (4211712, vec![3, 0, 0, 0]),
            (MAX_SECTION_SIZE, vec![0xff; 4]),
        ] {
            let mut w = BitWriter::new(4);
            write_sizes(&mut w, &[size]).unwrap();
            assert_eq!(w.bytes().unwrap(), expected);
            assert_eq!(toc_size(&[size]).unwrap(), expected.len());
        }
        let sizes = [
            0,
            1023,
            1024,
            17407,
            17408,
            4211711,
            4211712,
            MAX_SECTION_SIZE,
        ];
        let mut w = BitWriter::new(100);
        w.write(5, 21).unwrap();
        write_permutation(&mut w, &[]).unwrap();
        let prefix = w.bits_written();
        write_sizes(&mut w, &sizes).unwrap();
        assert_eq!(w.bytes().unwrap()[0], 21);
        assert_eq!(w.bits_written(), prefix + toc_size(&sizes).unwrap() * 8);
        let mut cursor = prefix;
        for size in sizes {
            let selector = read_bits(w.bytes().unwrap(), &mut cursor, 2);
            let widths = [10, 14, 22, 30];
            let offsets = [0, 1024, 17408, 4211712];
            assert_eq!(
                offsets[selector] + read_bits(w.bytes().unwrap(), &mut cursor, widths[selector]),
                size
            );
        }
        let mut short = BitWriter::new(toc_size(&sizes).unwrap() - 1);
        assert!(write_sizes(&mut short, &sizes).is_err());
        let mut w = BitWriter::new(100);
        assert!(write_sizes(&mut w, &[0, MAX_SECTION_SIZE + 1]).is_err());
        assert_eq!(w.bits_written(), 0);
        assert!(write_sizes(&mut w, &[]).is_err());
        w.write(1, 0).unwrap();
        assert!(write_sizes(&mut w, &[0]).is_err());
    }

    #[test]
    fn offset_and_dc_padding_match_bucket_boundary_math() {
        for dc in [0, 1016, 1023, 1024, 17407, 17408, 4211711, 4211712] {
            for sections in [1, 2, 3, 17, 1000] {
                let reserved = compute_group_data_offset(19, dc, sections).unwrap();
                assert_eq!(
                    group_data_offset(19, dc, sections).unwrap(),
                    (reserved.min_dc_global_size, reserved.group_data_offset)
                );
                let max_padding = ((sections - 1) * 20).div_ceil(8);
                // Independent brute minimum: no smaller candidate can keep its
                // width unchanged across all possible TOC-size differences.
                let width = |s| {
                    if s < 1024 {
                        12
                    } else if s < 17408 {
                        16
                    } else if s < 4211712 {
                        24
                    } else {
                        32
                    }
                };
                let mut minimum = dc;
                while width(minimum) != width(minimum + max_padding) {
                    minimum += 1;
                }
                assert_eq!(reserved.min_dc_global_size, minimum);
                assert_eq!(
                    reserved.group_data_offset,
                    19 + ((sections - 1) * 32 + width(minimum)).div_ceil(8) + minimum
                );
                for other in [0, 1023, 1024, 17408, 4211712, MAX_SECTION_SIZE] {
                    let mut sizes = vec![other; sections];
                    sizes[0] = dc;
                    let padding =
                        dc_global_padding(&sizes, 19, reserved.group_data_offset, minimum).unwrap();
                    sizes[0] += padding;
                    assert!(sizes[0] >= minimum);
                    assert_eq!(width(sizes[0]), width(minimum));
                    assert_eq!(
                        19 + toc_size(&sizes).unwrap() + sizes[0],
                        reserved.group_data_offset
                    );
                    let mut w = BitWriter::new(toc_size(&sizes).unwrap());
                    write_sizes(&mut w, &sizes).unwrap();
                    assert_eq!(
                        19 + w.bytes().unwrap().len() + sizes[0],
                        reserved.group_data_offset
                    );
                }
            }
        }
        // Crossing a bucket can require more than one stabilization iteration.
        let reservation = compute_group_data_offset(0, 0, 6801).unwrap();
        assert_eq!(reservation.min_dc_global_size, 17408);

        // Mixed-width entries exercise half-byte TOC rounding independently of
        // the all-small/all-large extrema used to reserve the output position.
        for sections in 2..19 {
            let (minimum, offset) = group_data_offset(7, 1023, sections).unwrap();
            let mut sizes: Vec<_> = (0..sections)
                .map(|i| [0, 1024, 17408, 4211712][i % 4])
                .collect();
            sizes[0] = 1023;
            let padding = dc_global_padding(&sizes, 7, offset, minimum).unwrap();
            sizes[0] += padding;
            assert_eq!(7 + toc_size(&sizes).unwrap() + sizes[0], offset);
        }
    }

    #[test]
    fn reservations_reject_overflow_underflow_and_unrepresentable_padding() {
        assert!(compute_group_data_offset(0, 0, 0).is_err());
        assert!(compute_group_data_offset(usize::MAX, 0, 1).is_err());
        assert!(compute_group_data_offset(0, 0, usize::MAX).is_err());
        assert!(compute_group_data_offset(0, MAX_SECTION_SIZE + 1, 1).is_err());
        assert!(compute_group_data_offset(0, MAX_SECTION_SIZE, 2).is_err());
        assert!(compute_group_data_offset(0, MAX_SECTION_SIZE, 1).is_ok());
        assert!(compute_dc_global_padding(&[], 0, 0, 0).is_err());
        assert!(compute_dc_global_padding(&[1], 0, 100, 0).is_err());
        assert!(compute_dc_global_padding(&[1], 0, 0, 1).is_err());
        assert!(compute_dc_global_padding(&[1], usize::MAX, usize::MAX, 1).is_err());
        assert!(compute_dc_global_padding(&[1], 0, 1026, 1).is_err());
        assert!(compute_dc_global_padding(&[0], 0, usize::MAX, 0).is_err());
        assert!(compute_dc_global_padding(&[0, MAX_SECTION_SIZE + 1], 0, 100, 0).is_err());
    }
}
