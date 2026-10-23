// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Focused checks of enc_ans.cc's retained histogram/context lifecycle.

use super::*;
use crate::entropy::ans_histogram::test_decode::Reader;
use crate::entropy::prefix_code::tests::TestBits;

fn finish(mut writer: BitWriter) -> (usize, Vec<u8>) {
    let bits = writer.bits_written();
    writer.zero_pad_to_byte().unwrap();
    (bits, writer.into_bytes().unwrap())
}

fn header(model: &Model) -> (usize, Vec<u8>) {
    let mut writer = BitWriter::new(1 << 20);
    writer.write(3, 5).unwrap();
    model.write_header(&mut writer).unwrap();
    finish(writer)
}

fn payload(model: &Model, tokens: &[Token], offset: usize) -> (usize, Vec<u8>) {
    let mut writer = BitWriter::new(1 << 20);
    writer.write(3, 5).unwrap();
    model.write_stream_at(tokens, offset, &mut writer).unwrap();
    finish(writer)
}

fn caches(model: &Model) -> Vec<(usize, Vec<u8>)> {
    model
        .streaming
        .as_ref()
        .unwrap()
        .encoded_histograms
        .iter()
        .map(|cache| {
            let mut writer = BitWriter::new(1 << 20);
            writer.append_unaligned(cache).unwrap();
            finish(writer)
        })
        .collect()
}

fn streaming_options() -> EntropyOptions {
    EntropyOptions {
        streaming_mode: true,
        uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(8, 0, 0).unwrap()),
        clustering: ClusterStrategy::Fast,
        ..EntropyOptions::default()
    }
}

#[test]
fn missing_symbols_are_explicit_and_add_to_every_context() {
    let tokens = [vec![
        Token::literal(0, 7),
        Token::literal(0, 7),
        Token::literal(2, 1),
    ]];
    let mut options = streaming_options();
    let (raw, count) = raw_histograms(&tokens, 3, true, &options).unwrap();
    assert_eq!(count, 3);
    assert_eq!(raw[0][7], 2);
    assert!(raw[1].is_empty());
    assert_eq!(raw[2], [0, 1]);
    assert!(!options.add_missing_symbols && !options.add_fixed_histograms);
    options.add_missing_symbols = true;
    let (injected, count) = raw_histograms(&tokens, 3, true, &options).unwrap();
    assert_eq!(count, 3);
    for (context, histogram) in injected.iter().enumerate() {
        assert_eq!(histogram.len(), 256);
        for (symbol, &weight) in histogram.iter().enumerate() {
            assert_eq!(weight, 1 + raw[context].get(symbol).copied().unwrap_or(0));
        }
    }
    for coding in [Coding::Prefix, Coding::Ans] {
        let mut model = Model::build_with_options(&tokens, 3, true, coding, &options).unwrap();
        let before = caches(&model);
        options.max_histograms = 0;
        let added = [vec![Token::literal(0, 223), Token::literal(1, 222)]];
        assert_eq!(
            model
                .append_with_options(&added, 2, true, &options)
                .unwrap(),
            3..5
        );
        assert_eq!(caches(&model), before);
        payload(&model, &added[0], 3);
        options.max_histograms = 256;
    }
}

#[test]
fn fixed_table_and_appended_payloads_decode_from_wire() {
    for coding in [Coding::Prefix, Coding::Ans] {
        let mut options = EntropyOptions {
            add_fixed_histograms: true,
            max_histograms: 0,
            ..streaming_options()
        };
        let first = [vec![Token::literal(0, 7), Token::literal(0, 255)]];
        let mut model = Model::build_with_options(&first, 1, false, coding, &options).unwrap();
        assert_eq!(model.histogram_count(), 1);
        let before = caches(&model);
        options.add_fixed_histograms = false;
        let next = [(0..256)
            .map(|v| Token::literal(v as usize % 2, v))
            .collect::<Vec<_>>()];
        assert_eq!(
            model
                .append_with_options(&next, 2, false, &options)
                .unwrap(),
            1..3
        );
        assert_eq!(caches(&model), before);
        assert_eq!(model.context_count(), 3);
        let mut writer = BitWriter::new(1 << 20);
        writer.write(3, 5).unwrap();
        model.write_header(&mut writer).unwrap();
        model.write_stream_at(&first[0], 0, &mut writer).unwrap();
        model.write_stream_at(&next[0], 1, &mut writer).unwrap();
        let end = writer.bits_written();
        writer.write(5, 21).unwrap();
        let (_, bytes) = finish(writer);
        let mut reader = Reader {
            bytes: &bytes,
            position: 0,
        };
        assert_eq!(reader.read(3), 5);
        assert_eq!(reader.read(1), 0); // LZ77 disabled.
        assert_eq!(reader.read(1), 1); // Simple context map, all three IDs zero.
        assert_eq!(reader.read(2), 0);
        let prefix = reader.read(1) != 0;
        assert_eq!(prefix, coding == Coding::Prefix);
        let log = if prefix { 15 } else { 5 + reader.read(2) as u8 };
        assert_eq!(reader.read(4), 8); // Direct uint8 configuration.
        if prefix {
            assert_eq!(reader.read(4), 0);
            assert_eq!(reader.read(4), 0);
            assert_eq!(reader.read(1), 1); // Prefix alphabet-minus-one = 255.
            assert_eq!(reader.read(4), 7);
            assert_eq!(reader.read(7), 127);
            let mut bits = TestBits {
                bytes: &bytes,
                position: reader.position,
                nbits: bytes.len() * 8,
                repeat_codes: [0; 2],
            };
            let (depths, singleton) = bits.read_histogram(256);
            assert_eq!(singleton, None);
            assert_eq!(depths, vec![8; 256]);
            for token in first[0].iter().chain(&next[0]) {
                assert_eq!(bits.read_canonical(&depths), token.value as usize);
            }
            reader.position = bits.position;
        } else {
            assert_eq!(log, 8);
            let counts = reader.histogram();
            assert_eq!(counts, vec![16; 256]);
            let table = crate::entropy::ans::tests::decoder_table(&counts, log);
            for tokens in [&first[0], &next[0]] {
                let mut state = reader.read(32);
                for token in tokens {
                    let (symbol, offset) = table[(state & 4095) as usize];
                    state = u32::from(counts[symbol]) * (state >> 12) + offset;
                    if state < 65536 {
                        state = (state << 16) | reader.read(16);
                    }
                    assert_eq!(symbol, token.value as usize);
                }
                assert_eq!(state, 0x13_0000);
            }
        }
        assert_eq!(reader.position, end);
        assert_eq!(reader.read(5), 21);
    }
}

#[test]
fn append_preserves_old_payloads_and_cached_bits_through_compaction() {
    for coding in [Coding::Prefix, Coding::Ans] {
        let mut options = EntropyOptions {
            add_fixed_histograms: true,
            max_histograms: 1,
            ..streaming_options()
        };
        let first = [vec![Token::literal(0, 17); 200]];
        let mut model = Model::build_with_options(&first, 1, false, coding, &options).unwrap();
        assert_eq!(model.histogram_count(), 2);
        assert_eq!(model.map, [1]); // Fixed center remains available but unused.
        let before = caches(&model);
        let old_payload = payload(&model, &first[0], 0);
        options.add_fixed_histograms = false;
        let next = [vec![Token::literal(0, 19); 200]];
        let range = model
            .append_with_options(&next, 1, false, &options)
            .unwrap();
        assert_eq!(range, 1..2);
        assert_eq!(&caches(&model)[..before.len()], before);
        assert_eq!(payload(&model, &first[0], 0), old_payload);
        let next_payload = payload(&model, &next[0], range.start);
        let before_header = header(&model);
        assert_eq!(model.remove_unused_histograms(), 1);
        assert_eq!(model.map, [0, 1]);
        assert_eq!(header(&model), before_header);
        assert_eq!(payload(&model, &first[0], 0), old_payload);
        assert_eq!(payload(&model, &next[0], range.start), next_payload);
        // Cached and direct histogram emission agree at a non-byte boundary.
        let saved = model.streaming.take();
        assert_eq!(header(&model), before_header);
        model.streaming = saved;
        assert_eq!(model.remove_unused_histograms(), 0);
    }
}

#[test]
fn failed_appends_leave_model_unchanged_and_check_batch_boundaries() {
    let mut options = streaming_options();
    let initial = [vec![Token::literal(0, 0), Token::literal(0, 1)]];
    let mut model = Model::build_with_options(&initial, 1, false, Coding::Ans, &options).unwrap();
    let before = header(&model);
    let cache = caches(&model);
    options.max_histograms = 0;
    assert!(
        model
            .append_with_options(&[vec![Token::literal(0, 2)]], 1, false, &options)
            .is_err()
    );
    let mut invalid = vec![options; 5];
    invalid[0].streaming_mode = false;
    invalid[1].add_fixed_histograms = true;
    invalid[2].uint_config = UintConfigStrategy::Fixed(HybridUintConfig::default());
    invalid[3].lz77_min_symbol = 512;
    invalid[4].max_histograms = 257;
    for options in invalid {
        assert!(
            model
                .append_with_options(&initial, 1, false, &options)
                .is_err()
        );
        assert_eq!(header(&model), before);
        assert_eq!(caches(&model), cache);
        assert_eq!(model.context_count(), 1);
    }
    assert!(
        model
            .append_with_options(&[vec![Token::literal(1, 0)]], 1, false, &options)
            .is_err()
    );
    assert_eq!(header(&model), before);
    let next = [vec![Token::literal(1, 1)]];
    assert_eq!(
        model
            .append_with_options(&next, 2, false, &options)
            .unwrap(),
        1..3
    );
    for (tokens, offset) in [
        (&next[0][..], 0),
        (&initial[0][..], 2),
        (&initial[0][..], usize::MAX),
    ] {
        let mut writer = BitWriter::new(1024);
        assert!(model.write_stream_at(tokens, offset, &mut writer).is_err());
        assert_eq!(writer.bits_written(), 0);
    }
}

#[test]
fn prefix_zero_depth_weights_and_initial_mode_are_retained() {
    let mut options = streaming_options();
    let first = [vec![Token::literal(0, 0)]];
    let mut model = Model::build_with_options(&first, 1, false, Coding::Source, &options).unwrap();
    assert!(!model.uses_ans());
    options.max_histograms = 0;
    // HistogramFromSymbolInfo maps a singleton's depth zero to weight zero.
    assert!(
        model
            .append_with_options(&first, 1, false, &options)
            .is_err()
    );
    options.max_histograms = 1;
    let next = [(0..200)
        .map(|v| Token::literal(0, v % 2))
        .collect::<Vec<_>>()];
    model
        .append_with_options(&next, 1, false, &options)
        .unwrap();
    assert!(!model.uses_ans());
    payload(&model, &next[0], 1);
}

#[test]
fn each_appended_lz77_batch_has_its_own_distance_context() {
    for coding in [Coding::Prefix, Coding::Ans] {
        let mut options = EntropyOptions {
            add_fixed_histograms: true,
            max_histograms: 0,
            ..streaming_options()
        };
        let tokens = vec![
            Token::literal(0, 2),
            Token {
                context: 0,
                value: 0,
                is_length: true,
            },
            Token::literal(1, 0),
        ];
        let mut model =
            Model::build_with_options(std::slice::from_ref(&tokens), 2, true, coding, &options)
                .unwrap();
        options.add_fixed_histograms = false;
        assert_eq!(
            model
                .append_with_options(std::slice::from_ref(&tokens), 3, true, &options)
                .unwrap(),
            2..5
        );
        payload(&model, &tokens, 0);
        payload(&model, &tokens, 2);
        let length = Token {
            context: 1,
            value: 0,
            is_length: true,
        };
        let mut writer = BitWriter::new(1024);
        assert!(model.write_stream_at(&[length], 0, &mut writer).is_err());
        model.write_stream_at(&[length], 2, &mut writer).unwrap();
        assert!(
            model
                .write_stream_at(
                    &[Token {
                        context: 2,
                        ..length
                    }],
                    2,
                    &mut writer
                )
                .is_err()
        );
    }
}

#[test]
fn injection_and_initialization_option_bounds() {
    let mut options = streaming_options();
    options.max_histograms = 0;
    assert!(Model::build_with_options(&[], 1, false, Coding::Ans, &options).is_err());
    options.add_fixed_histograms = true;
    assert!(Model::build_with_options(&[], 1, false, Coding::Ans, &options).is_ok());
    options.streaming_mode = false;
    assert!(options.validate().is_err());
    options.add_fixed_histograms = false;
    options.max_histograms = 1;
    options.add_missing_symbols = true;
    options.uint_config = UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap());
    options.max_log_alpha_size = 7;
    assert!(Model::build_with_options(&[], 1, false, Coding::Ans, &options).is_err());
    options.max_log_alpha_size = 8;
    let model = Model::build_with_options(&[], 1, false, Coding::Ans, &options).unwrap();
    assert_eq!(model.log_alpha_size(), 8);
    let Tables::Ans(tables) = &model.tables else {
        unreachable!()
    };
    assert_eq!(tables[0].normalized_counts(), vec![16; 256]);
}
