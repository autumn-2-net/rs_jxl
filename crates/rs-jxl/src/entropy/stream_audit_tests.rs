use super::super::{AnsCode, Coding, EntropyOptions, HybridUintConfig, PrefixCode};
use super::*;
use crate::bit_writer::BitWriter;

fn fixture(map: Vec<u8>, configs: Vec<HybridUintConfig>, counts: &[Vec<u64>], ans: bool) -> Model {
    let tables = if ans {
        Tables::Ans(
            counts
                .iter()
                .map(|h| AnsCode::from_counts(h, 8).unwrap())
                .collect(),
        )
    } else {
        Tables::Prefix(
            counts
                .iter()
                .map(|h| PrefixCode::from_counts(h).unwrap())
                .collect(),
        )
    };
    Model {
        map,
        configs,
        tables,
        lz77: false,
        length_config: HybridUintConfig::new(0, 0, 0).unwrap(),
        min_symbol: 240,
        log_alpha_size: if ans { 8 } else { 15 },
        streaming: None,
    }
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn assert_ordered(audit: &Audit) {
    assert!(audit.raw_context_entropy_bits <= audit.cluster_entropy_bits + 1e-9);
    assert!(audit.cluster_entropy_bits <= audit.model_symbol_bits + 1e-9);
    assert!(audit.extra_bit_entropy_bits <= audit.extra_bits as f64 + 1e-9);
}

#[test]
fn hand_entropy_keeps_raw_and_clustered_observations_separate() {
    for ans in [false, true] {
        let model = fixture(
            vec![0, 0, 0],
            vec![HybridUintConfig::default()],
            &[vec![2, 2]],
            ans,
        );
        let streams = [
            vec![Token::literal(0, 0); 2],
            vec![],
            vec![Token::literal(1, 1); 2],
        ];
        let audit = model.audit(&streams).unwrap();
        assert_eq!(
            (audit.streams, audit.nonempty_streams, audit.tokens),
            (3, 2, 4)
        );
        assert_eq!(
            (audit.raw_contexts, audit.active_contexts, audit.histograms),
            (3, 2, 1)
        );
        assert_eq!(audit.ans, ans);
        assert!(!audit.lz77);
        assert_eq!(audit.extra_bits, 0);
        near(audit.raw_context_entropy_bits, 0.0);
        near(audit.cluster_entropy_bits, 4.0);
        near(audit.model_symbol_bits, 4.0);
        near(audit.extra_bit_entropy_bits, 0.0);
        assert_ordered(&audit);
    }
}

#[test]
fn selected_ans_probability_and_prefix_lengths_are_not_empirical_costs() {
    let streams = [vec![Token::literal(0, 0), Token::literal(0, 1)]];
    for ans in [false, true] {
        // Amortize signaling so ANS chooses nonuniform rather than flat.
        let model = fixture(
            vec![0],
            vec![HybridUintConfig::default()],
            &[vec![30_000, 10_000]],
            ans,
        );
        let audit = model.audit(&streams).unwrap();
        near(audit.raw_context_entropy_bits, 2.0);
        near(audit.cluster_entropy_bits, 2.0);
        if ans {
            let Tables::Ans(tables) = &model.tables else {
                unreachable!()
            };
            let n = tables[0].normalized_counts();
            let expected = (4096.0 / f64::from(n[0])).log2() + (4096.0 / f64::from(n[1])).log2();
            near(audit.model_symbol_bits, expected);
            assert!(audit.model_symbol_bits > 2.0);
        } else {
            near(audit.model_symbol_bits, 2.0);
        }
        assert_ordered(&audit);
    }
}

#[test]
fn extra_bit_bias_is_measured_from_actual_payload_not_value_entropy() {
    let config = HybridUintConfig::new(0, 0, 0).unwrap();
    for ans in [false, true] {
        let model = fixture(vec![0], vec![config], &[vec![0, 0, 0, 4]], ans);
        // All values have final symbol 3 and two extra bits. Only bit zero
        // varies (three zeros, one one); bit one is always zero.
        let streams = [vec![
            Token::literal(0, 4),
            Token::literal(0, 4),
            Token::literal(0, 4),
            Token::literal(0, 5),
        ]];
        let audit = model.audit(&streams).unwrap();
        assert_eq!(audit.extra_bits, 8);
        near(audit.cluster_entropy_bits, 0.0);
        near(audit.raw_context_entropy_bits, 0.0);
        near(audit.model_symbol_bits, 0.0);
        near(
            audit.extra_bit_entropy_bits,
            3.0 * (4.0f64 / 3.0).log2() + 2.0,
        );
        assert_ordered(&audit);
    }
}

#[test]
fn bit_diagnostic_conditions_on_selected_histogram_and_final_symbol() {
    let config = HybridUintConfig::new(0, 0, 0).unwrap();
    let model = fixture(
        vec![0, 1],
        vec![config; 2],
        &[vec![0, 0, 0, 2], vec![0, 0, 0, 2]],
        false,
    );
    let streams = [vec![
        Token::literal(0, 4),
        Token::literal(0, 5),
        Token::literal(1, 6),
        Token::literal(1, 7),
    ]];
    let audit = model.audit(&streams).unwrap();
    // High bit is constant within each table (opposite between tables).
    // Pooling tables incorrectly would add four more bits of entropy.
    assert_eq!(audit.extra_bits, 8);
    near(audit.extra_bit_entropy_bits, 4.0);
    let model = fixture(vec![0], vec![config], &[vec![0, 0, 2, 2]], false);
    let streams = [vec![
        Token::literal(0, 2),
        Token::literal(0, 2),
        Token::literal(0, 5),
        Token::literal(0, 5),
    ]];
    // Bit zero is constant but opposite for symbols 2 and 3.
    near(model.audit(&streams).unwrap().extra_bit_entropy_bits, 0.0);
}

#[test]
fn each_context_uses_its_own_final_selected_config() {
    let config = HybridUintConfig::new(0, 0, 0).unwrap();
    let direct = HybridUintConfig::new(4, 0, 0).unwrap();
    let model = fixture(
        vec![0, 1],
        vec![config, direct],
        &[vec![0, 0, 0, 2], vec![0, 0, 0, 0, 1, 1]],
        false,
    );
    let streams = [vec![
        Token::literal(0, 4),
        Token::literal(0, 5),
        Token::literal(1, 4),
        Token::literal(1, 5),
    ]];
    let audit = model.audit(&streams).unwrap();
    assert_eq!(audit.extra_bits, 4);
    near(audit.raw_context_entropy_bits, 2.0);
    near(audit.cluster_entropy_bits, 2.0);
    near(audit.model_symbol_bits, 2.0);
    near(audit.extra_bit_entropy_bits, 2.0);
    assert_ordered(&audit);
}

#[test]
fn empty_streams_and_unused_raw_contexts_do_not_invent_singletons() {
    for ans in [false, true] {
        // Stored tables are valid fallbacks, not audit observations.
        let model = fixture(
            vec![0, 1, 0, 1],
            vec![HybridUintConfig::default(); 2],
            &[vec![1], vec![1]],
            ans,
        );
        for streams in [vec![], vec![vec![], vec![]]] {
            let audit = model.audit(&streams).unwrap();
            assert_eq!(audit.streams, streams.len());
            assert_eq!(
                (audit.nonempty_streams, audit.tokens, audit.active_contexts),
                (0, 0, 0)
            );
            assert_eq!(
                (audit.raw_contexts, audit.histograms, audit.extra_bits),
                (4, 2, 0)
            );
            near(audit.raw_context_entropy_bits, 0.0);
            near(audit.cluster_entropy_bits, 0.0);
            near(audit.model_symbol_bits, 0.0);
            near(audit.extra_bit_entropy_bits, 0.0);
        }
    }
}

fn lz_fixture(ans: bool) -> Model {
    let mut literal = vec![0; 246];
    literal[4] = 1;
    literal[245] = 2;
    let mut model = fixture(
        vec![0, 1],
        vec![HybridUintConfig::new(2, 0, 0).unwrap(); 2],
        &[literal, vec![1]],
        ans,
    );
    model.lz77 = true;
    model
}

#[test]
fn length_tokens_use_length_config_and_reserved_symbol_offset_exactly() {
    for ans in [false, true] {
        let model = lz_fixture(ans);
        let streams = [vec![
            Token::literal(0, 7),
            Token {
                context: 0,
                value: 16,
                is_length: true,
            },
            Token {
                context: 0,
                value: 17,
                is_length: true,
            },
            Token::literal(1, 0),
        ]];
        let mapped = model.map_token(streams[0][1]).unwrap();
        assert_eq!((mapped.symbol, mapped.nbits, mapped.bits), (245, 4, 0));
        let audit = model.audit(&streams).unwrap();
        assert!(audit.lz77);
        assert_eq!((audit.tokens, audit.extra_bits), (4, 10));
        let entropy = (3.0f64).log2() + 2.0 * (1.5f64).log2();
        near(audit.raw_context_entropy_bits, entropy);
        near(audit.cluster_entropy_bits, entropy);
        near(audit.extra_bit_entropy_bits, 2.0);
        assert_ordered(&audit);
    }
}

#[test]
fn invalid_context_length_collision_and_absent_symbols_are_errors() {
    for ans in [false, true] {
        let model = lz_fixture(ans);
        for token in [
            Token::literal(2, 0),
            Token::literal(usize::MAX, 0),
            Token {
                context: 1,
                value: 0,
                is_length: true,
            },
            Token {
                context: 0,
                value: u32::MAX,
                is_length: true,
            },
            Token::literal(0, 0), // Inside alphabet but absent from its table.
        ] {
            assert!(model.audit(&[vec![token]]).is_err(), "{token:?}");
        }
        let mut collision = fixture(
            vec![0, 0],
            vec![HybridUintConfig::new(8, 0, 0).unwrap()],
            &[vec![1]],
            ans,
        );
        collision.lz77 = true;
        assert!(collision.audit(&[vec![Token::literal(0, 240)]]).is_err());
        let mut no_lz = lz_fixture(ans);
        no_lz.lz77 = false;
        assert!(
            no_lz
                .audit(&[vec![Token {
                    context: 0,
                    value: 16,
                    is_length: true
                }]])
                .is_err()
        );
        let alphabet = fixture(
            vec![0],
            vec![HybridUintConfig::new(15, 0, 0).unwrap()],
            &[vec![1]],
            ans,
        );
        assert!(
            alphabet
                .audit(&[vec![Token::literal(0, u32::MAX)]])
                .is_err()
        );
    }
}

fn wire(model: &Model, streams: &[Vec<Token>]) -> (usize, Vec<u8>) {
    let mut writer = BitWriter::new(1 << 20);
    model.write_header(&mut writer).unwrap();
    for stream in streams {
        model.write_stream(stream, &mut writer).unwrap();
    }
    let bits = writer.bits_written();
    writer.zero_pad_to_byte().unwrap();
    (bits, writer.into_bytes().unwrap())
}

#[test]
fn observation_does_not_change_real_model_tokens_or_writing() {
    let streams = [
        (0..128)
            .map(|i| Token::literal(i % 3, (i * 17 % 251) as u32))
            .collect(),
        vec![],
    ];
    let saved = streams.clone();
    for coding in [Coding::Prefix, Coding::Ans] {
        let model =
            Model::build_with_options(&streams, 4, false, coding, &EntropyOptions::default())
                .unwrap();
        let before = wire(&model, &streams);
        let audit = model.audit(&streams).unwrap();
        let again = model.audit(&streams).unwrap();
        assert_ordered(&audit);
        assert_eq!(audit.model_symbol_bits, again.model_symbol_bits);
        assert_eq!(
            audit.raw_context_entropy_bits,
            again.raw_context_entropy_bits
        );
        assert_eq!(audit.cluster_entropy_bits, again.cluster_entropy_bits);
        assert_eq!(audit.extra_bit_entropy_bits, again.extra_bit_entropy_bits);
        assert_eq!(before, wire(&model, &streams));
        assert_eq!(streams, saved);
    }
}

#[test]
fn sparse_counter_overflow_returns_resource_error_without_wrapping() {
    let mut counts = HashMap::new();
    add_observations(&mut counts, (0, 0), u64::MAX).unwrap();
    assert!(matches!(
        add_observations(&mut counts, (0, 0), 1),
        Err(Error::Limit(_))
    ));
    assert_eq!(counts[&(0, 0)], u64::MAX);
    let rows = [((0, 0), u64::MAX), ((0, 1), 1)];
    assert!(matches!(
        empirical_entropy(&rows, |&n| n),
        Err(Error::Limit(_))
    ));
}
