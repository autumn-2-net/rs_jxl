use super::*;
use crate::entropy::cluster::neighborhood::ContextMapOptions;

fn fixture(lz: bool) -> (Vec<Vec<Token>>, Vec<u8>, Vec<Vec<u64>>) {
    let mut state = 19u32;
    let mut map = Vec::new();
    let mut tokens = Vec::new();
    for context in 0..1024 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let id = state >> 31;
        map.push(id as u8);
        tokens.extend([
            Token::literal(context, id),
            Token::literal(context, id),
            Token::literal(context, 1 - id),
        ]);
    }
    if lz {
        map.push(0);
        tokens.push(Token {
            context: 0,
            value: 0,
            is_length: true,
        });
        tokens.push(Token::literal(1024, 0));
    }
    let streams = vec![tokens[..1200].to_vec(), vec![], tokens[1200..].to_vec()];
    let (raw, _) = raw_histograms(&streams, map.len(), lz, &EntropyOptions::default()).unwrap();
    let mut counts = vec![vec![0; 256]; 2];
    for (raw, &id) in raw.iter().zip(&map) {
        for (dest, &n) in counts[id as usize].iter_mut().zip(raw) {
            *dest += n;
        }
    }
    (streams, map, counts)
}

#[test]
fn context_map_models_keep_streams_lz_and_alignment_and_can_save_actual_bits() {
    for lz in [false, true] {
        for (uint_config, streaming_mode) in [
            (UintConfigStrategy::Best, false),
            (
                UintConfigStrategy::Fixed(HybridUintConfig::default()),
                false,
            ),
            (UintConfigStrategy::Best, true),
        ] {
            let options = EntropyOptions {
                uint_config,
                streaming_mode,
                ..Default::default()
            };
            let (streams, map, counts) = fixture(lz);
            let base = Model::from_clustering(&streams, counts, map, lz, false, &options).unwrap();
            let (models, work) = base
                .context_map_candidates(&streams, &options, &ContextMapOptions::default())
                .unwrap()
                .unwrap();
            assert_eq!(work.merges, 1);
            assert_eq!(models.len(), 1);
            let candidate = &models[0].0;
            assert_eq!(candidate.map.len(), base.map.len());
            assert_eq!(candidate.lz77, base.lz77);
            assert_eq!(candidate.min_symbol, base.min_symbol);
            assert_eq!(candidate.length_config, base.length_config);
            for offset in 0..8 {
                let mut sizes = Vec::new();
                for model in [&base, candidate] {
                    let mut writer = BitWriter::new(1 << 20);
                    writer.write(offset, 0).unwrap();
                    model.write_header(&mut writer).unwrap();
                    for stream in &streams {
                        let before = writer.bits_written();
                        model.write_stream(stream, &mut writer).unwrap();
                        if stream.is_empty() {
                            assert_eq!(writer.bits_written() - before, 32);
                        }
                    }
                    sizes.push(writer.bits_written() - offset as usize);
                }
                assert!(sizes[1] < sizes[0], "lz={lz}: {sizes:?}");
            }
        }
    }
}

#[test]
fn context_map_skips_prefix_single_and_appended_models() {
    let (streams, map, counts) = fixture(false);
    for (prefix, streaming_mode) in [(true, false), (false, true)] {
        let options = EntropyOptions {
            streaming_mode,
            ..Default::default()
        };
        let mut model = Model::from_clustering(
            &streams,
            counts.clone(),
            map.clone(),
            false,
            prefix,
            &options,
        )
        .unwrap();
        if streaming_mode {
            model
                .append_with_options(&[vec![Token::literal(0, 1)]], 1, false, &options)
                .unwrap();
        }
        assert!(
            model
                .context_map_candidates(&streams, &options, &ContextMapOptions::default())
                .unwrap()
                .is_none()
        );
    }
    let streams = [vec![Token::literal(0, 0); 300]];
    let options = EntropyOptions::default();
    let model = Model::build_with_options(&streams, 1, false, Coding::Ans, &options).unwrap();
    assert!(
        model
            .context_map_candidates(&streams, &options, &ContextMapOptions::default())
            .unwrap()
            .is_none()
    );
}
