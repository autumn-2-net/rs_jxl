// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Context-map representations and move-to-front transform from
//! lib/jxl/enc_context_map.cc, baseline
//! b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Tries the source's simple, raw-coded, and MTF-coded representations. Unlike
//! the source's dry histogram estimates, selection measures complete serialized
//! candidates, including mode flags, nested headers, extra bits, and ANS state.
//! Each coded candidate uses source RLE and source prefix/ANS selection, not
//! Model::Auto's estimate. The winning bits are appended without re-encoding.
//!
//! A coded map has one literal context plus a distance context when RLE is
//! accepted. That model's own map has at most two entries: source RLE cannot
//! accept a match in two entries, so its nested model has one context and ends
//! recursion. This bound is on map LENGTH, not the number of distinct symbols.
//! The caller retains stream.rs's `map.len() > 1` guard around map emission.
//!
//! Nested models use source kContextMap's fixed HybridUint(2,0,1), length000,
//! min-symbol224 and no force_huffman. RLE's source cost estimator deliberately
//! uses literal420. Direct-audit/final-plan replay, profiling, and the unrelated
//! DCT EncodeBlockCtxMap wrapper are not ported here. Exact serialized selection
//! can differ from C++'s dry-cost selection; this is not compressed-byte parity.

use super::hybrid_uint::HybridUintConfig;
use super::options::{EntropyOptions, UintConfigStrategy};
use super::stream::{Coding, Model, Token};
use crate::bit_writer::BitWriter;
use crate::tools::lz77::{self, Command, Literal, SourceConfig};
use crate::{Error, Result};

const MAX_HISTOGRAMS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Representation {
    Simple,
    Raw,
    MoveToFront,
}

struct Candidate {
    writer: BitWriter,
    representation: Representation,
}

fn validate(context_map: &[u8], num_histograms: usize) -> Result<()> {
    if context_map.is_empty() {
        return Err(Error::InvalidInput("context map must not be empty".into()));
    }
    if !(1..=MAX_HISTOGRAMS).contains(&num_histograms) {
        return Err(Error::InvalidInput(
            "context map needs 1..=256 histograms".into(),
        ));
    }
    if context_map.len() > i32::MAX as usize {
        return Err(Error::Limit(
            "context map exceeds the ANS histogram weight limit",
        ));
    }
    let mut seen = [false; MAX_HISTOGRAMS];
    for &symbol in context_map {
        if usize::from(symbol) >= num_histograms {
            return Err(Error::InvalidInput(
                "context-map histogram index is out of range".into(),
            ));
        }
        seen[usize::from(symbol)] = true;
    }
    // DecodeContextMap infers the table count from the maximum symbol and then
    // requires every table to be referenced. Do not emit a map it would reject.
    if seen[..num_histograms].iter().any(|&present| !present) {
        return Err(Error::InvalidInput(
            "context map has unused histogram identifiers".into(),
        ));
    }
    Ok(())
}

fn move_to_front_transform(context_map: &[u8]) -> Result<Vec<u8>> {
    let mut transformed = Vec::new();
    transformed
        .try_reserve_exact(context_map.len())
        .map_err(|_| Error::Limit("move-to-front context map"))?;
    let Some(&maximum) = context_map.iter().max() else {
        return Ok(transformed);
    };
    let mut order = [0u8; MAX_HISTOGRAMS];
    for (symbol, entry) in order.iter_mut().enumerate() {
        *entry = symbol as u8;
    }
    let alphabet_size = usize::from(maximum) + 1;
    for &symbol in context_map {
        let index = order[..alphabet_size]
            .iter()
            .position(|&entry| entry == symbol)
            .expect("MTF order remains a permutation of its alphabet");
        transformed.push(index as u8);
        order[..=index].rotate_right(1);
    }
    Ok(transformed)
}

fn token_stream(symbols: &[u8]) -> Result<[Vec<Token>; 1]> {
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(symbols.len())
        .map_err(|_| Error::Limit("context-map tokens"))?;
    for &symbol in symbols {
        tokens.push(Token::literal(0, u32::from(symbol)));
    }
    Ok([tokens])
}

fn entry_bits(num_histograms: usize) -> u8 {
    if num_histograms == 1 {
        0
    } else {
        (usize::BITS - (num_histograms - 1).leading_zeros()) as u8
    }
}

fn simple_candidate(context_map: &[u8], width: u8) -> Result<Candidate> {
    debug_assert!(width <= 3);
    let bits = context_map
        .len()
        .checked_mul(usize::from(width))
        .and_then(|bits| bits.checked_add(3))
        .ok_or(Error::Limit("simple context-map bit count"))?;
    let mut writer = BitWriter::new(bits.div_ceil(8));
    writer.write(1, 1)?;
    writer.write(2, u64::from(width))?;
    if width != 0 {
        for &symbol in context_map {
            writer.write(width, u64::from(symbol))?;
        }
    }
    Ok(Candidate {
        writer,
        representation: Representation::Simple,
    })
}

fn rle_tokens(streams: &[Vec<Token>; 1]) -> Result<(Vec<Token>, bool)> {
    debug_assert!(
        streams[0]
            .iter()
            .all(|token| token.context == 0 && !token.is_length)
    );
    let mut literals = Vec::new();
    literals
        .try_reserve_exact(streams[0].len())
        .map_err(|_| Error::Limit("context-map RLE input"))?;
    literals.extend(streams[0].iter().map(|token| Literal {
        context: 0,
        value: token.value,
    }));
    // Default HistogramParams uses RLE, not the optimal or fast matchers.
    // The source RLE entry point retains its literal420 cost estimator.
    let config = SourceConfig {
        literal_uint_config: HybridUintConfig::new(2, 0, 1).expect("constant"),
        ..SourceConfig::default()
    };
    let parsed = lz77::parse_source_rle_with_state(&literals, 0, &config)?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(streams[0].len())
        .map_err(|_| Error::Limit("context-map RLE tokens"))?;
    for command in parsed.commands {
        match command {
            Command::Literal(literal) => tokens.push(Token::literal(0, literal.value)),
            Command::Match {
                length, distance, ..
            } => {
                debug_assert_eq!(distance, 1);
                tokens.push(Token {
                    context: 0,
                    value: length - 3,
                    is_length: true,
                });
                tokens.push(Token::literal(1, 0));
            }
        }
    }
    Ok((tokens, parsed.enabled))
}

fn coded_candidate(streams: &[Vec<Token>; 1], representation: Representation) -> Result<Candidate> {
    debug_assert!(representation != Representation::Simple);
    let max_bytes = streams[0]
        .len()
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(8192))
        .ok_or(Error::Limit("coded context-map output limit"))?;
    let options = EntropyOptions {
        uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(2, 0, 1).expect("constant")),
        lz77_length_config: HybridUintConfig::new(0, 0, 0).expect("constant"),
        ..EntropyOptions::default()
    };
    let (tokens, enabled) = rle_tokens(streams)?;
    let streams = [tokens];
    let model = Model::build_with_options(
        &streams,
        1 + usize::from(enabled),
        enabled,
        Coding::Source,
        &options,
    )?;
    let mut writer = BitWriter::new(max_bytes);
    writer.write(1, 0)?;
    writer.write(1, u64::from(representation == Representation::MoveToFront))?;
    model.write_header(&mut writer)?;
    model.write_stream(&streams[0], &mut writer)?;
    Ok(Candidate {
        writer,
        representation,
    })
}

fn choose(context_map: &[u8], num_histograms: usize) -> Result<Candidate> {
    validate(context_map, num_histograms)?;
    if num_histograms == 1 {
        return simple_candidate(context_map, 0);
    }
    let raw = token_stream(context_map)?;
    let transformed = move_to_front_transform(context_map)?;
    let mtf = token_stream(&transformed)?;
    let mut best = coded_candidate(&raw, Representation::Raw)?;
    let candidate = coded_candidate(&mtf, Representation::MoveToFront)?;
    if candidate.writer.bits_written() < best.writer.bits_written() {
        best = candidate;
    }
    let width = entry_bits(num_histograms);
    if width < 4 {
        let simple = simple_candidate(context_map, width)?;
        if simple.writer.bits_written() < best.writer.bits_written() {
            best = simple;
        }
    }
    // Strict comparisons preserve source tie preferences: raw over MTF and
    // coded over simple. Prefix/ANS selection is the source heuristic.
    Ok(best)
}

/// Write one complete context map. The map must reference every histogram in
/// `0..num_histograms`, where num_histograms is 1..=256. The outer stream must
/// call this only when it has more than one raw context; a one-histogram map
/// shared by several contexts is encoded in exactly three bits.
///
/// Candidate comparison is exact and excludes byte padding. Validation and
/// candidate-building errors do not modify the destination. The append reserves
/// all output space before writing, so output-limit failures are also atomic.
pub fn write_context_map(
    context_map: &[u8],
    num_histograms: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    let selected = choose(context_map, num_histograms)?;
    debug_assert!(
        selected.representation != Representation::Simple || entry_bits(num_histograms) <= 3
    );
    writer.append_unaligned(&selected.writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::ans_histogram::test_decode::Reader;
    use super::super::prefix_code::tests::TestBits;
    use super::*;

    #[derive(Default)]
    struct WireStats {
        maximum_depth: usize,
        rle: usize,
        prefix: usize,
        ans: usize,
    }

    fn read_config(reader: &mut Reader<'_>, log: u8) -> (u32, u32, u32) {
        let split = reader.read((u8::BITS - log.leading_zeros()) as u8);
        if split == u32::from(log) {
            return (split, 0, 0);
        }
        let msb = reader.read((u32::BITS - split.leading_zeros()) as u8);
        let lsb = reader.read((u32::BITS - (split - msb).leading_zeros()) as u8);
        (split, msb, lsb)
    }

    fn read_uint(reader: &mut Reader<'_>, token: u32, config: (u32, u32, u32)) -> u32 {
        let (split, msb, lsb) = config;
        if token < 1 << split {
            return token;
        }
        let token = token - (1 << split);
        let width = split - msb - lsb + (token >> (msb + lsb));
        let high = (1 << msb) | ((token >> lsb) & ((1 << msb) - 1));
        (((high << width) | reader.read(width as u8)) << lsb) | (token & ((1 << lsb) - 1))
    }

    enum WireCodes {
        Prefix(Vec<(Vec<u8>, Option<usize>)>),
        Ans {
            counts: Vec<Vec<u16>>,
            tables: Vec<Vec<(usize, u32)>>,
            state: u32,
        },
    }

    impl WireCodes {
        fn symbol(&mut self, reader: &mut Reader<'_>, context: usize) -> u32 {
            match self {
                Self::Prefix(codes) => {
                    let (depths, singleton) = &codes[context];
                    if let Some(symbol) = singleton {
                        return *symbol as u32;
                    }
                    let mut bits = TestBits {
                        bytes: reader.bytes,
                        position: reader.position,
                        nbits: reader.bytes.len() * 8,
                        repeat_codes: [0; 2],
                    };
                    let symbol = bits.read_canonical(depths);
                    reader.position = bits.position;
                    symbol as u32
                }
                Self::Ans {
                    counts,
                    tables,
                    state,
                } => {
                    let (symbol, offset) = tables[context][(*state & 4095) as usize];
                    *state = u32::from(counts[context][symbol]) * (*state >> 12) + offset;
                    if *state < 65536 {
                        *state = (*state << 16) | reader.read(16);
                    }
                    symbol as u32
                }
            }
        }
    }

    // Full wire readback, using the existing independent histogram and alias
    // test decoders, not the encoder's tables, token list or selected model.
    fn decode_map(
        reader: &mut Reader<'_>,
        count: usize,
        depth: usize,
        stats: &mut WireStats,
    ) -> Vec<u8> {
        stats.maximum_depth = stats.maximum_depth.max(depth);
        assert!(
            depth <= 1,
            "context-map RLE recursion must terminate after a two-entry map"
        );
        if reader.read(1) != 0 {
            let width = reader.read(2) as u8;
            return (0..count).map(|_| reader.read(width) as u8).collect();
        }
        let mtf = reader.read(1) != 0;
        let lz77 = reader.read(1) != 0;
        let mut length_config = (0, 0, 0);
        if lz77 {
            stats.rle += 1;
            assert_eq!(reader.read(2), 0); // min_symbol224, not forced Huffman512
            assert_eq!(reader.read(2), 0); // min_length3
            length_config = read_config(reader, 8);
            assert_eq!(length_config, (0, 0, 0));
        }
        let map = if lz77 {
            decode_map(reader, 2, depth + 1, stats)
        } else {
            vec![0]
        };
        let histograms = usize::from(*map.iter().max().unwrap()) + 1;
        let prefix = reader.read(1) != 0;
        let log = if prefix { 15 } else { 5 + reader.read(2) as u8 };
        let configs: Vec<_> = (0..histograms).map(|_| read_config(reader, log)).collect();
        assert!(configs.iter().all(|&c| c == (2, 0, 1)));
        let mut codes = if prefix {
            stats.prefix += 1;
            let alphabets: Vec<_> = (0..histograms)
                .map(|_| {
                    1 + if reader.read(1) == 0 {
                        0
                    } else {
                        let width = reader.read(4) as u8;
                        (1usize << width) + reader.read(width) as usize
                    }
                })
                .collect();
            let mut bits = TestBits {
                bytes: reader.bytes,
                position: reader.position,
                nbits: reader.bytes.len() * 8,
                repeat_codes: [0; 2],
            };
            let tables = alphabets
                .into_iter()
                .map(|n| bits.read_histogram(n))
                .collect();
            reader.position = bits.position;
            WireCodes::Prefix(tables)
        } else {
            stats.ans += 1;
            let counts: Vec<_> = (0..histograms).map(|_| reader.histogram()).collect();
            let tables = counts
                .iter()
                .map(|c| super::super::ans::tests::decoder_table(c, log))
                .collect();
            WireCodes::Ans {
                counts,
                tables,
                state: reader.read(32),
            }
        };
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let context = usize::from(map[0]);
            let symbol = codes.symbol(reader, context);
            if lz77 && symbol >= 224 {
                let length = read_uint(reader, symbol - 224, length_config) as usize + 3;
                let context = usize::from(map[1]);
                let symbol = codes.symbol(reader, context);
                assert_eq!(read_uint(reader, symbol, configs[context]), 0); // distance1
                let repeated = *values.last().expect("RLE requires an earlier literal");
                assert!(values.len() + length <= count);
                values.resize(values.len() + length, repeated);
            } else {
                values.push(u8::try_from(read_uint(reader, symbol, configs[context])).unwrap());
            }
        }
        if let WireCodes::Ans { state, .. } = codes {
            assert_eq!(state, 0x13_0000);
        }
        if mtf { inverse_mtf(&values) } else { values }
    }

    fn wire_roundtrip(map: &[u8], candidate: Candidate) -> WireStats {
        let mut writer = BitWriter::new(map.len() * 8 + 16384);
        writer.write(3, 5).unwrap();
        writer.append_unaligned(&candidate.writer).unwrap();
        let end = writer.bits_written();
        writer.write(5, 0x15).unwrap();
        writer.zero_pad_to_byte().unwrap();
        let bytes = writer.into_bytes().unwrap();
        let mut reader = Reader {
            bytes: &bytes,
            position: 0,
        };
        assert_eq!(reader.read(3), 5);
        let mut stats = WireStats::default();
        assert_eq!(decode_map(&mut reader, map.len(), 0, &mut stats), map);
        assert_eq!(reader.position, end);
        assert_eq!(reader.read(5), 0x15);
        stats
    }

    fn serialized_bits(mut writer: BitWriter) -> Vec<u8> {
        let length = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        let bytes = writer.into_bytes().unwrap();
        (0..length).map(|i| (bytes[i / 8] >> (i % 8)) & 1).collect()
    }

    fn inverse_mtf(indices: &[u8]) -> Vec<u8> {
        let mut order: Vec<u8> = (0..=255).collect();
        indices
            .iter()
            .map(|&index| {
                let symbol = order.remove(usize::from(index));
                order.insert(0, symbol);
                symbol
            })
            .collect()
    }

    fn all_candidate_costs(map: &[u8], histograms: usize) -> Vec<usize> {
        let mut costs = Vec::new();
        for (values, representation) in [
            (map.to_vec(), Representation::Raw),
            (
                move_to_front_transform(map).unwrap(),
                Representation::MoveToFront,
            ),
        ] {
            let streams = token_stream(&values).unwrap();
            let candidate = coded_candidate(&streams, representation).unwrap();
            costs.push(candidate.writer.bits_written());
        }
        if entry_bits(histograms) <= 3 {
            costs.push(
                simple_candidate(map, entry_bits(histograms))
                    .unwrap()
                    .writer
                    .bits_written(),
            );
        }
        costs
    }

    #[test]
    fn move_to_front_known_vectors_and_full_byte_alphabet() {
        assert_eq!(move_to_front_transform(&[]).unwrap(), []);
        assert_eq!(
            move_to_front_transform(&[0, 1, 0, 1, 1]).unwrap(),
            [0, 1, 1, 1, 0]
        );
        assert_eq!(
            move_to_front_transform(&[2, 2, 0, 2, 1]).unwrap(),
            [2, 0, 1, 1, 2]
        );
        assert_eq!(
            move_to_front_transform(&[255, 255, 0, 255]).unwrap(),
            [255, 0, 1, 1]
        );
        let values: Vec<u8> = (0..=255)
            .chain((0..=255).rev())
            .chain((0..=255).cycle().take(4096))
            .collect();
        assert_eq!(
            inverse_mtf(&move_to_front_transform(&values).unwrap()),
            values
        );
    }

    #[test]
    fn simple_known_bit_vectors() {
        for (map, histogram_count, bit_count, expected) in [
            (vec![0, 0, 0], 1, 3, vec![0x01]),
            (vec![0, 1], 2, 5, vec![0x13]),
            (vec![2, 0, 1], 3, 9, vec![0x95, 0x00]),
        ] {
            let selected = choose(&map, histogram_count).unwrap();
            assert_eq!(selected.representation, Representation::Simple);
            assert_eq!(selected.writer.bits_written(), bit_count);
            let mut writer = selected.writer;
            writer.zero_pad_to_byte().unwrap();
            assert_eq!(writer.into_bytes().unwrap(), expected);
        }
        assert_eq!(choose(&vec![0; 4096], 1).unwrap().writer.bits_written(), 3);
    }

    #[test]
    fn exact_selection_includes_mode_headers_extra_bits_and_ans_state() {
        let cases: Vec<(Vec<u8>, usize)> = vec![
            (vec![0, 1], 2),
            ((0..8).cycle().take(96).collect(), 8),
            ((0..17).cycle().take(256).collect(), 17),
            ((0..=255).collect(), 256),
        ];
        for (map, histogram_count) in cases {
            let costs = all_candidate_costs(&map, histogram_count);
            let selected = choose(&map, histogram_count).unwrap();
            assert_eq!(selected.writer.bits_written(), *costs.iter().min().unwrap());
            let expected = serialized_bits(selected.writer);
            let mut writer = BitWriter::new(16384);
            write_context_map(&map, histogram_count, &mut writer).unwrap();
            assert_eq!(serialized_bits(writer), expected);
        }
    }

    #[test]
    fn source_rle_and_source_coding_roundtrip_complete_wire() {
        let cases: Vec<Vec<u8>> = vec![
            vec![0, 1],
            (0..=255).cycle().take(512).collect(),
            (0..16).flat_map(|v| std::iter::repeat_n(v, 128)).collect(),
            (0..=255).flat_map(|v| std::iter::repeat_n(v, 64)).collect(),
            [vec![0; 1024], vec![1; 1024]].concat(),
        ];
        for (index, map) in cases.iter().enumerate() {
            let histograms = usize::from(*map.iter().max().unwrap()) + 1;
            let streams = token_stream(map).unwrap();
            let enabled = rle_tokens(&streams).unwrap().1;
            assert_eq!(enabled, index >= 2);
            let stats =
                wire_roundtrip(map, coded_candidate(&streams, Representation::Raw).unwrap());
            assert_eq!(stats.rle != 0, enabled);
            if index == 1 || index == 3 {
                assert!(
                    stats.ans > 0,
                    "source selects ANS for these >100-token streams"
                );
            } else {
                assert!(stats.prefix > 0);
            }
            let transformed = token_stream(&move_to_front_transform(map).unwrap()).unwrap();
            wire_roundtrip(
                map,
                coded_candidate(&transformed, Representation::MoveToFront).unwrap(),
            );
            let selected = choose(map, histograms).unwrap();
            assert_eq!(
                selected.writer.bits_written(),
                *all_candidate_costs(map, histograms).iter().min().unwrap()
            );
            wire_roundtrip(map, selected);
        }
    }

    #[test]
    fn distance_context_map_is_bounded_and_two_entry_rle_always_declines() {
        for map in [[0, 0], [0, 1], [1, 0], [255, 255]] {
            let streams = token_stream(&map).unwrap();
            assert!(!rle_tokens(&streams).unwrap().1);
            let stats = wire_roundtrip(
                &map,
                coded_candidate(&streams, Representation::Raw).unwrap(),
            );
            assert_eq!(stats.maximum_depth, 0);
            assert_eq!(stats.rle, 0);
        }
        // Two distinct symbols are NOT the termination condition: a long
        // binary map can use RLE, but its distance map still has two entries.
        let map = [vec![0; 1024], vec![1; 1024]].concat();
        let stats = wire_roundtrip(
            &map,
            coded_candidate(&token_stream(&map).unwrap(), Representation::Raw).unwrap(),
        );
        assert_eq!(stats.rle, 1);
        assert_eq!(stats.maximum_depth, 1);
    }

    #[test]
    fn raw_and_mtf_candidates_remain_distinct_after_source_rle() {
        let cyclic: Vec<u8> = (0..=255).cycle().take(4096).collect();
        let mtf = token_stream(&move_to_front_transform(&cyclic).unwrap()).unwrap();
        let bits = serialized_bits(
            coded_candidate(&mtf, Representation::MoveToFront)
                .unwrap()
                .writer,
        );
        assert_eq!(&bits[..2], [0, 1]);

        let mut skewed: Vec<u8> = (0..=255).collect();
        skewed.extend((0..8192).map(|i| {
            if i % 16 == 0 {
                (1 + (i / 16 * 73) % 255) as u8
            } else {
                0
            }
        }));
        let raw = token_stream(&skewed).unwrap();
        let bits = serialized_bits(coded_candidate(&raw, Representation::Raw).unwrap().writer);
        assert_eq!(&bits[..2], [0, 0]);
    }

    #[test]
    fn append_preserves_every_initial_bit_offset_without_padding() {
        let map: Vec<u8> = (0..16).cycle().take(128).collect();
        let expected = serialized_bits(choose(&map, 16).unwrap().writer);
        for prefix in 0..8u8 {
            let mut writer = BitWriter::new(16384);
            writer.write(prefix, (1u64 << prefix) - 1).unwrap();
            write_context_map(&map, 16, &mut writer).unwrap();
            writer.write(5, 0x15).unwrap();
            let bits = serialized_bits(writer);
            assert!(bits[..usize::from(prefix)].iter().all(|&bit| bit == 1));
            assert_eq!(
                &bits[usize::from(prefix)..usize::from(prefix) + expected.len()],
                expected
            );
            assert_eq!(
                &bits[usize::from(prefix) + expected.len()..],
                [1, 0, 1, 0, 1]
            );
        }
    }

    #[test]
    fn short_nested_models_decline_rle_and_use_source_prefix_policy() {
        let streams = token_stream(&[0, 255, 4, 4, 0]).unwrap();
        assert_eq!(streams.len(), 1);
        assert!(
            streams[0]
                .iter()
                .all(|token| token.context == 0 && !token.is_length)
        );
        let candidate = coded_candidate(&streams, Representation::Raw).unwrap();
        let bits = serialized_bits(candidate.writer);
        assert_eq!(&bits[..4], [0, 0, 0, 1]); // raw, no MTF, LZ77 off, prefix
        for map in [[0, 0], [0, 1], [1, 0], [255, 255]] {
            assert!(!rle_tokens(&token_stream(&map).unwrap()).unwrap().1);
        }
    }

    #[test]
    fn malformed_maps_and_output_limits_leave_destination_unchanged() {
        for (map, histograms) in [
            (vec![], 1),
            (vec![0], 0),
            (vec![0], 257),
            (vec![0, 2], 2),
            (vec![0, 2], 3),
            (vec![0, 0], 2),
        ] {
            let mut writer = BitWriter::new(64);
            writer.write(3, 5).unwrap();
            assert!(write_context_map(&map, histograms, &mut writer).is_err());
            assert_eq!(writer.bits_written(), 3);
            assert_eq!(serialized_bits(writer), [1, 0, 1]);
        }
        let mut writer = BitWriter::new(1);
        writer.write(7, 0x55).unwrap();
        assert!(write_context_map(&[0, 0], 1, &mut writer).is_err());
        assert_eq!(writer.bits_written(), 7);
        assert_eq!(serialized_bits(writer), [1, 0, 1, 0, 1, 0, 1]);
    }
}
