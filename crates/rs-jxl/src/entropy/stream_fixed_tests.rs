use super::*;

#[test]
fn dense_literal420_counts_match_scalar_with_empty_contexts_and_wide_values() {
    for contexts in [1, 7, 256] {
        let tokens: Vec<_> = (0..8193)
            .map(|i| {
                Token::literal(
                    i % contexts,
                    [
                        0,
                        1,
                        15,
                        16,
                        255,
                        65535,
                        u32::MAX,
                        (i as u32).wrapping_mul(1664525),
                    ][i % 8],
                )
            })
            .collect();
        let streams = [vec![], tokens[..1000].to_vec(), tokens[1000..].to_vec()];
        let mut expected = vec![Vec::new(); contexts];
        for token in streams.iter().flatten() {
            add_count(
                &mut expected[token.context],
                HybridUintConfig::default().encode(token.value).token as usize,
            )
            .unwrap();
        }
        assert_eq!(literal420_histograms(&streams, contexts).unwrap(), expected);
        let options = EntropyOptions {
            streaming_mode: true,
            ..Default::default()
        };
        let (actual, count) = raw_histograms(&streams, contexts, false, &options).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(count, 8193);
    }
    assert_eq!(
        literal420_histograms(&[vec![Token::literal(0, 0)]], 3).unwrap(),
        [vec![1], vec![], vec![]]
    );
    for token in [
        Token::literal(3, 0),
        Token {
            context: 0,
            value: 0,
            is_length: true,
        },
    ] {
        assert!(literal420_histograms(&[vec![token]], 3).is_err());
    }
}

#[test]
fn counted_fixed_plans_match_validated_token_scan() {
    for config in [(4, 2, 0), (2, 0, 1), (0, 0, 0), (8, 0, 0), (10, 2, 1)] {
        for streaming_mode in [false, true] {
            for lz77 in [false, true] {
                for empty in [false, true] {
                    let options = EntropyOptions {
                        uint_config: UintConfigStrategy::Fixed(
                            HybridUintConfig::new(config.0, config.1, config.2).unwrap(),
                        ),
                        streaming_mode,
                        ..Default::default()
                    };
                    let tokens = if empty {
                        vec![]
                    } else {
                        (0..1024)
                            .map(|i| {
                                if lz77 && i % 11 == 0 {
                                    Token {
                                        context: 0,
                                        value: i as u32 * 7,
                                        is_length: true,
                                    }
                                } else {
                                    Token::literal(
                                        i % 3,
                                        if lz77 {
                                            i as u32 % 31
                                        } else {
                                            (i as u32).wrapping_mul(16777619)
                                        },
                                    )
                                }
                            })
                            .collect()
                    };
                    let streams = [tokens];
                    let (raw, _) = raw_histograms(&streams, 4, lz77, &options).unwrap();
                    let map = [0, 1, 0, 1];
                    let mut counts = vec![Vec::new(); 2];
                    for (histogram, id) in raw.iter().zip(map) {
                        let length = counts[id as usize].len().max(histogram.len());
                        counts[id as usize].resize(length, 0);
                        for (dest, &n) in counts[id as usize].iter_mut().zip(histogram) {
                            *dest += n;
                        }
                    }
                    for prefix in [false, true] {
                        let reference =
                            choose_config_plan(&streams, &map, 2, lz77, prefix, &options);
                        let actual = super::super::config_search::fixed_config_plan_from_counts(
                            &counts, lz77, prefix, &options,
                        );
                        match (reference, actual) {
                            (Ok(a), Ok(b)) => {
                                assert_eq!(a.configs, b.configs);
                                assert_eq!(a.log_alpha_size, b.log_alpha_size);
                            }
                            (Err(_), Err(_)) => {}
                            (a, b) => panic!(
                                "plan mismatch {config:?} streaming={streaming_mode} lz={lz77} empty={empty} prefix={prefix}: {:?}/{:?}",
                                a.err(),
                                b.err()
                            ),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fixed420_streams_match_general_mapping_for_prefix_and_ans() {
    let tokens: Vec<_> = (0..8193)
        .map(|i| Token::literal(i % 5, [0, 16, 255, 65535, u32::MAX][i % 5]))
        .collect();
    let options = EntropyOptions {
        uint_config: UintConfigStrategy::Fixed(HybridUintConfig::default()),
        ..Default::default()
    };
    for coding in [Coding::Prefix, Coding::Ans] {
        let model =
            Model::build_with_options(std::slice::from_ref(&tokens), 5, false, coding, &options)
                .unwrap();
        let expanded: Vec<_> = tokens
            .iter()
            .map(|&t| model.map_token(t).unwrap())
            .collect();
        for alignment in 0..8 {
            let mut actual = BitWriter::new(1 << 20);
            let mut expected = BitWriter::new(1 << 20);
            actual.write(alignment, 0).unwrap();
            expected.write(alignment, 0).unwrap();
            model.write_stream(&tokens, &mut actual).unwrap();
            match &model.tables {
                Tables::Ans(tables) => {
                    super::super::ans::write_tokens(tables, &expanded, &mut expected).unwrap()
                }
                Tables::Prefix(tables) => {
                    for t in &expanded {
                        tables[t.context]
                            .write_symbol(t.symbol, &mut expected)
                            .unwrap();
                        expected.write(t.nbits, u64::from(t.bits)).unwrap();
                    }
                }
            }
            assert_eq!(actual.bits_written(), expected.bits_written());
            actual.zero_pad_to_byte().unwrap();
            expected.zero_pad_to_byte().unwrap();
            assert_eq!(actual.into_bytes().unwrap(), expected.into_bytes().unwrap());
        }
        for bad in [
            Token::literal(5, 0),
            Token {
                context: 0,
                value: 0,
                is_length: true,
            },
            Token::literal(1, 13),
        ] {
            assert!(
                model
                    .write_stream(&[bad], &mut BitWriter::new(1 << 20))
                    .is_err()
            );
        }
    }
}

#[test]
fn fixed420_appended_batches_keep_context_boundaries() {
    let options = EntropyOptions {
        streaming_mode: true,
        ..Default::default()
    };
    let first = vec![Token::literal(0, 17); 300];
    let second: Vec<_> = (0..1000)
        .map(|i| Token::literal(i % 2, [255, u32::MAX][i % 2]))
        .collect();
    for coding in [Coding::Ans, Coding::Prefix] {
        let mut model =
            Model::build_with_options(std::slice::from_ref(&first), 1, false, coding, &options)
                .unwrap();
        let range = model
            .append_with_options(std::slice::from_ref(&second), 2, false, &options)
            .unwrap();
        assert_eq!(range, 1..3);
        let expanded: Vec<_> = second
            .iter()
            .map(|&token| model.map_token_in(token, &range).unwrap())
            .collect();
        let mut actual = BitWriter::new(1 << 20);
        let mut expected = BitWriter::new(1 << 20);
        model
            .write_stream_at(&second, range.start, &mut actual)
            .unwrap();
        match &model.tables {
            Tables::Ans(tables) => {
                super::super::ans::write_tokens(tables, &expanded, &mut expected).unwrap()
            }
            Tables::Prefix(tables) => {
                for token in expanded {
                    tables[token.context]
                        .write_symbol(token.symbol, &mut expected)
                        .unwrap();
                    expected.write(token.nbits, u64::from(token.bits)).unwrap();
                }
            }
        }
        assert_eq!(actual.bits_written(), expected.bits_written());
        actual.zero_pad_to_byte().unwrap();
        expected.zero_pad_to_byte().unwrap();
        assert_eq!(actual.into_bytes().unwrap(), expected.into_bytes().unwrap());
        for (offset, context) in [(0, 1), (1, 2), (2, 0)] {
            assert!(
                model
                    .write_stream_at(
                        &[Token::literal(context, 17)],
                        offset,
                        &mut BitWriter::new(1024),
                    )
                    .is_err()
            );
        }
    }
}
