//! Experimental decoder for serialized native JXL entropy tools, backed by the
//! maintained Rust decoder. No encoder histogram state is passed to decoding.
//! This is a tool bridge, not a standard image decoder or a new image format.

use crate::{Error, Result};
use jxl_decoder::api::experimental_entropy::{
    ExperimentalEntropyDecoder, ExperimentalEntropyError, ExperimentalMaNode,
    parse_experimental_ma_tree,
};

pub use jxl_decoder::api::experimental_entropy::{
    MAX_EXPERIMENTAL_BYTES, MAX_EXPERIMENTAL_CONTEXTS, MAX_EXPERIMENTAL_SYMBOLS,
    MAX_EXPERIMENTAL_TREE_NODES,
};

fn decoding_error(error: ExperimentalEntropyError) -> Error {
    Error::Decoding(error.to_string())
}

/// One fitted serialized JXL entropy model and one independently padded stream.
pub struct EntropyDecoder<'a>(ExperimentalEntropyDecoder<'a>);

impl<'a> EntropyDecoder<'a> {
    /// `header_bits` and `data_bits` exclude zero byte padding. The symbol count
    /// is exact; `finish` rejects underconsumption and an invalid ANS final state.
    /// This constructor rejects outer LZ77; use `new_with_lz` to opt in.
    pub fn new(
        header: &[u8],
        header_bits: usize,
        data: &'a [u8],
        data_bits: usize,
        contexts: usize,
        symbols: usize,
    ) -> Result<Self> {
        ExperimentalEntropyDecoder::new(header, header_bits, data, data_bits, contexts, symbols)
            .map(Self)
            .map_err(decoding_error)
    }

    /// Decode with bounded optional LZ expansion. `contexts` counts literal
    /// contexts, excluding the appended distance context. `symbols` counts
    /// expanded outputs; overlap is supported within the native 2^20 window.
    /// A zero multiplier uses plain distance-minus-one, suitable for a filtered
    /// nonzero residual sequence with no image-row distance geometry.
    pub fn new_with_lz(
        header: &[u8],
        header_bits: usize,
        data: &'a [u8],
        data_bits: usize,
        contexts: usize,
        symbols: usize,
        distance_multiplier: u32,
    ) -> Result<Self> {
        ExperimentalEntropyDecoder::new_with_lz(
            header,
            header_bits,
            data,
            data_bits,
            contexts,
            symbols,
            distance_multiplier,
        )
        .map(Self)
        .map_err(decoding_error)
    }

    pub fn has_lz77(&self) -> bool {
        self.0.has_lz77()
    }

    pub fn read(&mut self, context: usize) -> Result<u32> {
        self.0.read(context).map_err(decoding_error)
    }

    pub fn finish(self) -> Result<()> {
        self.0.finish().map_err(decoding_error)
    }
}

/// Decode a `modular::tree::Tree::write` stream into the reusable encoder-side
/// tree representation. No-LZ77 tree streams only; all tree structure is parsed
/// from the serialized bytes and checked independently of the original tree.
pub fn decode_tree(
    bytes: &[u8],
    bits: usize,
    max_nodes: usize,
) -> Result<crate::modular::tree::Tree> {
    use crate::modular::{
        predictor::Predictor,
        tree::{Node, Tree},
    };
    let parsed = parse_experimental_ma_tree(bytes, bits, max_nodes).map_err(decoding_error)?;
    let nodes = parsed
        .into_iter()
        .map(|node| match node {
            ExperimentalMaNode::Split {
                property,
                threshold,
                greater,
                less_equal,
            } => Ok(Node::Split {
                property: usize::from(property),
                threshold,
                greater,
                less_equal,
            }),
            ExperimentalMaNode::Leaf {
                predictor,
                offset,
                multiplier,
            } => Ok(Node::Leaf {
                predictor: Predictor::try_from(u32::from(predictor))
                    .map_err(|error| Error::Decoding(error.to_string()))?,
                offset,
                multiplier,
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    Tree::new(nodes).map_err(|error| Error::Decoding(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_writer::BitWriter;
    use crate::entropy::stream::{Coding, Model, Token};
    use crate::modular::{
        predictor::Predictor,
        tree::{Node, Tree},
    };

    fn bytes(mut writer: BitWriter) -> (Vec<u8>, usize) {
        let bits = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        (writer.into_bytes().unwrap(), bits)
    }

    fn encoded(
        tokens: &[Token],
        contexts: usize,
        coding: Coding,
    ) -> (Vec<u8>, usize, Vec<u8>, usize) {
        let model = Model::build(&[tokens.to_vec()], contexts, false, coding).unwrap();
        let mut header = BitWriter::new(1 << 20);
        model.write_header(&mut header).unwrap();
        let mut data = BitWriter::new(1 << 20);
        model.write_stream(tokens, &mut data).unwrap();
        let (header, header_bits) = bytes(header);
        let (data, data_bits) = bytes(data);
        (header, header_bits, data, data_bits)
    }

    fn encoded_lz(
        tokens: &[Token],
        contexts: usize,
        coding: Coding,
    ) -> (Vec<u8>, usize, Vec<u8>, usize) {
        let model = Model::build(&[tokens.to_vec()], contexts + 1, true, coding).unwrap();
        let mut header = BitWriter::new(1 << 20);
        model.write_header(&mut header).unwrap();
        let mut data = BitWriter::new(1 << 20);
        model.write_stream(tokens, &mut data).unwrap();
        let (header, hb) = bytes(header);
        let (data, db) = bytes(data);
        (header, hb, data, db)
    }

    fn copy(context: usize, length: u32) -> Token {
        Token {
            context,
            value: length - 3,
            is_length: true,
        }
    }

    #[test]
    fn bounded_lz_prefix_ans_overlap_and_wide_values() {
        for coding in [Coding::Prefix, Coding::Ans] {
            for multiplier in [0, 1, 7] {
                let distance = |distance| {
                    crate::tools::lz77::distance_symbol(distance, multiplier as usize).unwrap()
                };
                let a = u32::MAX;
                let b = 1u32 << 31;
                let tokens = vec![
                    Token::literal(0, a),
                    Token::literal(1, b),
                    copy(0, 17),
                    Token::literal(2, distance(2)),
                    Token::literal(1, 0x71234567),
                    copy(0, 31),
                    Token::literal(2, distance(1)),
                ];
                let mut expected = vec![a, b];
                for _ in 0..17 {
                    expected.push(expected[expected.len() - 2]);
                }
                expected.extend(std::iter::repeat_n(0x71234567, 32));
                let (header, hb, data, db) = encoded_lz(&tokens, 2, coding);
                assert!(EntropyDecoder::new(&header, hb, &data, db, 2, expected.len()).is_err());
                let mut decoder = EntropyDecoder::new_with_lz(
                    &header,
                    hb,
                    &data,
                    db,
                    2,
                    expected.len(),
                    multiplier,
                )
                .unwrap();
                assert!(decoder.has_lz77());
                for (i, &value) in expected.iter().enumerate() {
                    assert_eq!(decoder.read(i % 2).unwrap(), value);
                }
                assert!(decoder.read(0).is_err());
                decoder.finish().unwrap();
            }
        }
    }

    #[test]
    fn bounded_lz_rejects_overrun_missing_history_and_oversized_distance() {
        for coding in [Coding::Prefix, Coding::Ans] {
            for (tokens, outputs, multiplier, valid_prefix) in [
                (vec![copy(0, 3), Token::literal(1, 0)], 3, 0, 0),
                (
                    vec![Token::literal(0, 9), copy(0, 3), Token::literal(1, 1)],
                    4,
                    0,
                    1,
                ),
                (
                    vec![Token::literal(0, 9), copy(0, 3), Token::literal(1, 1 << 20)],
                    4,
                    0,
                    1,
                ),
                (
                    vec![
                        Token::literal(0, 9),
                        copy(0, 3),
                        Token::literal(1, u32::MAX),
                    ],
                    4,
                    0,
                    1,
                ),
                (
                    vec![Token::literal(0, 9), copy(0, 3), Token::literal(1, 4)],
                    4,
                    u32::MAX,
                    1,
                ),
                (
                    vec![Token::literal(0, 9), copy(0, 17), Token::literal(1, 0)],
                    17,
                    0,
                    1,
                ),
            ] {
                let (header, hb, data, db) = encoded_lz(&tokens, 1, coding);
                let mut decoder =
                    EntropyDecoder::new_with_lz(&header, hb, &data, db, 1, outputs, multiplier)
                        .unwrap();
                for _ in 0..valid_prefix {
                    assert_eq!(decoder.read(0).unwrap(), 9);
                }
                assert!(decoder.read(0).is_err());
                assert!(decoder.finish().is_err());
            }
        }
    }

    #[test]
    fn bounded_lz_empty_literal_only_truncation_and_unfinished_copy() {
        for coding in [Coding::Prefix, Coding::Ans] {
            let (header, hb, data, db) = encoded_lz(&[], 1, coding);
            let decoder = EntropyDecoder::new_with_lz(&header, hb, &data, db, 1, 0, 0).unwrap();
            assert!(decoder.has_lz77());
            decoder.finish().unwrap();

            let literal = [Token::literal(0, u32::MAX)];
            let (header, hb, data, db) = encoded(&literal, 1, coding);
            let mut decoder = EntropyDecoder::new_with_lz(&header, hb, &data, db, 1, 1, 0).unwrap();
            assert!(!decoder.has_lz77());
            assert_eq!(decoder.read(0).unwrap(), u32::MAX);
            decoder.finish().unwrap();

            let tokens = vec![
                Token::literal(0, 0x71234567),
                copy(0, 999),
                Token::literal(1, 0),
            ];
            let (header, hb, data, db) = encoded_lz(&tokens, 1, coding);
            let mut decoder =
                EntropyDecoder::new_with_lz(&header, hb, &data, db, 1, 1000, 0).unwrap();
            for _ in 0..2 {
                assert_eq!(decoder.read(0).unwrap(), 0x71234567);
            }
            assert!(decoder.finish().is_err());

            let truncated_bits = db - 1;
            let mut truncated = data[..truncated_bits.div_ceil(8)].to_vec();
            if truncated_bits % 8 != 0 {
                *truncated.last_mut().unwrap() &= (1 << (truncated_bits % 8)) - 1;
            }
            let result =
                EntropyDecoder::new_with_lz(&header, hb, &truncated, truncated_bits, 1, 1000, 0)
                    .and_then(|mut decoder| {
                        for _ in 0..1000 {
                            decoder.read(0)?;
                        }
                        decoder.finish()
                    });
            assert!(result.is_err());
        }
    }

    #[test]
    fn bounded_lz_wraps_its_fixed_window_and_preserves_narrow_row_aliases() {
        let window = 1u32 << 20;
        let tokens = vec![
            Token::literal(0, u32::MAX),
            copy(0, window - 1),
            Token::literal(1, 0),
            copy(0, 3),
            Token::literal(1, window - 1),
        ];
        let (header, hb, data, db) = encoded_lz(&tokens, 1, Coding::Ans);
        let outputs = window as usize + 3;
        let mut decoder =
            EntropyDecoder::new_with_lz(&header, hb, &data, db, 1, outputs, 0).unwrap();
        for _ in 0..outputs {
            assert_eq!(decoder.read(0).unwrap(), u32::MAX);
        }
        decoder.finish().unwrap();

        // Special code 9 is (-2,+1): at row width one the signed offset
        // is nonpositive and JXL's defined alias maps it to distance one.
        let tokens = vec![Token::literal(0, 77), copy(0, 3), Token::literal(1, 9)];
        for coding in [Coding::Prefix, Coding::Ans] {
            let (header, hb, data, db) = encoded_lz(&tokens, 1, coding);
            let mut decoder = EntropyDecoder::new_with_lz(&header, hb, &data, db, 1, 4, 1).unwrap();
            for _ in 0..4 {
                assert_eq!(decoder.read(0).unwrap(), 77);
            }
            decoder.finish().unwrap();
        }
    }

    #[test]
    fn parsed_prefix_and_ans_models_decode_multicontext_uints() {
        let tokens: Vec<_> = (0..2000)
            .map(|i| {
                Token::literal(
                    i % 4,
                    match i % 9 {
                        0 => u32::MAX,
                        1 => 0,
                        2 => 1 << 30,
                        _ => (i * i) as u32,
                    },
                )
            })
            .collect();
        for coding in [Coding::Prefix, Coding::Ans] {
            let (header, hb, data, db) = encoded(&tokens, 4, coding);
            let mut decoder = EntropyDecoder::new(&header, hb, &data, db, 4, tokens.len()).unwrap();
            for token in &tokens {
                assert_eq!(decoder.read(token.context).unwrap(), token.value);
            }
            decoder.finish().unwrap();
        }
    }

    #[test]
    fn singleton_and_empty_streams_use_exact_symbol_budgets() {
        for coding in [Coding::Prefix, Coding::Ans] {
            for count in [0, 1, 100] {
                let tokens = vec![Token::literal(0, 0); count];
                let (header, hb, data, db) = encoded(&tokens, 1, coding);
                let mut decoder = EntropyDecoder::new(&header, hb, &data, db, 1, count).unwrap();
                for _ in 0..count {
                    assert_eq!(decoder.read(0).unwrap(), 0);
                }
                assert!(decoder.read(0).is_err());
                decoder.finish().unwrap();
                if count != 0 {
                    assert!(
                        EntropyDecoder::new(&header, hb, &data, db, 1, count)
                            .unwrap()
                            .finish()
                            .is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn exact_lengths_padding_and_context_bounds_are_enforced() {
        let tokens: Vec<_> = (0..1000)
            .map(|i| Token::literal(i % 2, (i * 37) as u32))
            .collect();
        for coding in [Coding::Prefix, Coding::Ans] {
            let (header, hb, data, db) = encoded(&tokens, 2, coding);
            assert!(EntropyDecoder::new(&header, hb, &data, db, 0, tokens.len()).is_err());
            assert!(
                EntropyDecoder::new(
                    &header,
                    hb,
                    &data,
                    db,
                    MAX_EXPERIMENTAL_CONTEXTS + 1,
                    tokens.len()
                )
                .is_err()
            );
            assert!(
                EntropyDecoder::new(&header, hb, &data, db, 2, MAX_EXPERIMENTAL_SYMBOLS + 1)
                    .is_err()
            );
            assert!(EntropyDecoder::new(&header, hb + 1, &data, db, 2, tokens.len()).is_err());
            let mut extra = data.clone();
            extra.push(0);
            assert!(EntropyDecoder::new(&header, hb, &extra, db, 2, tokens.len()).is_err());
            let mut decoder = EntropyDecoder::new(&header, hb, &data, db, 2, tokens.len()).unwrap();
            assert!(decoder.read(2).is_err());
            for token in &tokens {
                assert_eq!(decoder.read(token.context).unwrap(), token.value);
            }
            decoder.finish().unwrap();
            let truncated_bits = db - 1;
            let mut truncated = data[..truncated_bits.div_ceil(8)].to_vec();
            if truncated_bits % 8 != 0 {
                *truncated.last_mut().unwrap() &= (1 << (truncated_bits % 8)) - 1;
            }
            let decoded =
                EntropyDecoder::new(&header, hb, &truncated, truncated_bits, 2, tokens.len())
                    .and_then(|mut decoder| {
                        for token in &tokens {
                            decoder.read(token.context)?;
                        }
                        decoder.finish()
                    });
            assert!(decoded.is_err());
            for (section, bits, is_header) in [(&header, hb, true), (&data, db, false)] {
                if bits % 8 != 0 {
                    let mut bad = section.clone();
                    *bad.last_mut().unwrap() |= 1 << (bits % 8);
                    assert!(if is_header {
                        EntropyDecoder::new(&bad, hb, &data, db, 2, tokens.len()).is_err()
                    } else {
                        EntropyDecoder::new(&header, hb, &bad, db, 2, tokens.len()).is_err()
                    });
                }
            }
        }
    }

    #[test]
    fn ans_terminal_state_and_trailing_declared_bits_are_checked() {
        let tokens = vec![Token::literal(0, 0); 17];
        let (header, hb, mut data, db) = encoded(&tokens, 1, Coding::Ans);
        assert_eq!(db, 32);
        data[0] ^= 1;
        let mut decoder = EntropyDecoder::new(&header, hb, &data, db, 1, tokens.len()).unwrap();
        for _ in &tokens {
            assert_eq!(decoder.read(0).unwrap(), 0);
        }
        assert!(decoder.finish().is_err());

        let (header, hb, data, db) = encoded(&tokens, 1, Coding::Prefix);
        assert_eq!(db, 0);
        assert!(data.is_empty());
        let mut decoder = EntropyDecoder::new(&header, hb, &[0], 1, 1, tokens.len()).unwrap();
        for _ in &tokens {
            assert_eq!(decoder.read(0).unwrap(), 0);
        }
        assert!(decoder.finish().is_err());
    }

    #[test]
    fn unreachable_ma_split_and_outer_lz77_are_rejected() {
        let mut tokens = vec![Token::literal(1, 1), Token::literal(0, u32::MAX - 1)];
        for _ in 0..2 {
            tokens.extend([
                Token::literal(1, 0),
                Token::literal(2, 0),
                Token::literal(3, 0),
                Token::literal(4, 0),
                Token::literal(5, 0),
            ]);
        }
        // The root threshold is i32::MAX, so its greater branch is empty.
        let model = Model::build(&[tokens.clone()], 6, false, Coding::Prefix).unwrap();
        let mut writer = BitWriter::new(1 << 20);
        model.write_header(&mut writer).unwrap();
        model.write_stream(&tokens, &mut writer).unwrap();
        let (wire, bits) = bytes(writer);
        assert!(decode_tree(&wire, bits, 3).is_err());

        let tokens = vec![Token::literal(0, 0)];
        let model = Model::build(&[tokens.clone()], 2, true, Coding::Prefix).unwrap();
        let mut header = BitWriter::new(1 << 20);
        model.write_header(&mut header).unwrap();
        let (header, hb) = bytes(header);
        assert!(EntropyDecoder::new(&header, hb, &[], 0, 1, 1).is_err());
    }

    #[test]
    fn serialized_ma_tree_reconstructs_predictors_offsets_multipliers_and_bfs_contexts() {
        let tree = Tree::new(vec![
            Node::Split {
                property: 0,
                threshold: 1,
                greater: 1,
                less_equal: 2,
            },
            Node::Leaf {
                predictor: Predictor::Weighted,
                offset: -13,
                multiplier: 12,
            },
            Node::Split {
                property: 0,
                threshold: 0,
                greater: 3,
                less_equal: 4,
            },
            Node::Leaf {
                predictor: Predictor::Average4,
                offset: 17,
                multiplier: 3,
            },
            Node::Leaf {
                predictor: Predictor::Gradient,
                offset: i32::MIN,
                multiplier: 1,
            },
        ])
        .unwrap();
        let mut writer = BitWriter::new(1 << 20);
        tree.write(&mut writer).unwrap();
        let (wire, bits) = bytes(writer);
        let parsed = decode_tree(&wire, bits, 5).unwrap();
        assert_eq!(tree.tokens(), parsed.tokens());
        assert!(decode_tree(&wire, bits, 4).is_err());
        assert!(decode_tree(&wire, bits, 0).is_err());
        assert!(decode_tree(&wire, bits - 1, 5).is_err());
    }
}
