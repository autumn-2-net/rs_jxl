use super::*;

#[test]
fn head_tail_keeps_contexts_lz_configuration_and_stream_boundaries() {
    let mut trials = 0;
    for lz in [false, true] {
        for spread in [3, 7, 13, 29, 101, 509, 2047] {
            let mut tokens: Vec<_> = (0..2000)
                .map(|i| {
                    Token::literal(
                        i % 3,
                        if i % 20 < 17 {
                            ((i * 17) % spread) as u32
                        } else {
                            (i * 31) as u32
                        },
                    )
                })
                .collect();
            if lz {
                tokens.push(Token {
                    context: 1,
                    value: 2048,
                    is_length: true,
                });
                tokens.push(Token::literal(3, 100));
            }
            let streams = vec![tokens[..1000].to_vec(), vec![], tokens[1000..].to_vec()];
            let options = EntropyOptions::default();
            let base =
                Model::build_with_options(&streams, 3 + usize::from(lz), lz, Coding::Ans, &options)
                    .unwrap();
            let (trial, count, changed) = base.head_tail_candidate(&streams, &options).unwrap();
            assert_eq!(count, 125);
            if let Some(trial) = trial {
                trials += 1;
                assert_eq!(base.map, trial.map);
                assert_eq!(base.lz77, trial.lz77);
                assert_eq!(base.length_config, trial.length_config);
                assert_eq!(base.min_symbol, trial.min_symbol);
                assert!(changed != 0 || base.log_alpha_size != trial.log_alpha_size);
                for alignment in 0..8 {
                    let mut writer = BitWriter::new(1 << 20);
                    writer.write(alignment, 0).unwrap();
                    trial.write_header(&mut writer).unwrap();
                    for stream in &streams {
                        let before = writer.bits_written();
                        trial.write_stream(stream, &mut writer).unwrap();
                        if stream.is_empty() {
                            assert_eq!(writer.bits_written() - before, 32);
                        }
                    }
                }
            }
        }
    }
    assert!(
        trials != 0,
        "fixtures must exercise changed representations"
    );
}

#[test]
fn head_tail_leaves_prefix_and_fixed_or_incremental_policies_alone() {
    let streams = [vec![Token::literal(0, 17); 700]];
    for (coding, options) in [
        (Coding::Prefix, EntropyOptions::default()),
        (
            Coding::Ans,
            EntropyOptions {
                streaming_mode: true,
                ..Default::default()
            },
        ),
        (
            Coding::Ans,
            EntropyOptions {
                uint_config: UintConfigStrategy::Fixed(HybridUintConfig::default()),
                ..Default::default()
            },
        ),
    ] {
        let base = Model::build_with_options(&streams, 1, false, coding, &options).unwrap();
        let (trial, count, changed) = base.head_tail_candidate(&streams, &options).unwrap();
        assert!(trial.is_none());
        assert_eq!((count, changed), (0, 0));
    }
}
