// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Packed-value and coordinate-layout oracles for enc_lz77_index.h.

use super::super::search::{match_length, word_length_u16};
use super::super::{
    Command, SourceBlockMode, SourceMatchCost, SourceSuffixMode, Work, blocks, run,
};
use super::*;

fn scalar_length<Value: PartialEq>(a: &[Value], b: &[Value]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(left, right)| left == right)
        .count()
}

#[test]
fn packed_word_localizes_every_bit_lane_alignment_and_tail_without_sentinels() {
    let values: Vec<u16> = (0..53u16).map(|v| v.wrapping_mul(0x9e37)).collect();
    for left_offset in 0..4 {
        for right_offset in 0..4 {
            for size in 0..=41 {
                let left = &values[left_offset..left_offset + size];
                let mut right = vec![u16::MAX; right_offset + size];
                right[right_offset..].copy_from_slice(left);
                assert_eq!(word_length_u16(left, &right[right_offset..]), size);
                for mismatch in 0..size {
                    for bit in 0..16 {
                        right[right_offset + mismatch] ^= 1u16 << bit;
                        let right_slice = &right[right_offset..];
                        assert_eq!(word_length_u16(left, right_slice), mismatch);
                        assert_eq!(
                            match_length(left, right_slice),
                            scalar_length(left, right_slice)
                        );
                        right[right_offset + mismatch] ^= 1u16 << bit;
                    }
                }
            }
        }
    }
    let mut repeated = [u16::MAX; 97];
    repeated[91] = 0;
    for distance in 1..8 {
        let a = &repeated[distance..];
        let b = &repeated[..repeated.len() - distance];
        assert_eq!(word_length_u16(a, b), scalar_length(a, b));
    }
}

#[test]
fn narrow_storage_is_exact_and_a_single_wide_value_anywhere_forces_full_width() {
    let narrow = [0, 1, 32768, 65534, 65535, 65535, 0];
    for input in [&[][..], &narrow[..]] {
        let packed = Values::new(input, true).unwrap();
        let Values::Narrow(buffer) = &packed else {
            panic!("u16 values must pack")
        };
        assert_eq!(buffer.len() * size_of::<u16>(), 2 * input.len());
        assert!(matches!(
            Values::new(input, false).unwrap(),
            Values::Wide(_)
        ));
        for (position, &value) in input.iter().enumerate() {
            assert_eq!(packed.get(position), value);
            assert_eq!(packed.hash(position), hash(input, position));
        }
    }
    for wide in [65536, 0x1ffff, 0x8000_0000, u32::MAX] {
        for position in 0..narrow.len() {
            let mut input = narrow;
            input[position] = wide;
            let storage = Values::new(&input, true).unwrap();
            assert!(matches!(storage, Values::Wide(_)));
            for (p, &value) in input.iter().enumerate() {
                assert_eq!(storage.get(p), value);
                assert_eq!(storage.hash(p), hash(&input, p));
            }
        }
    }
}

fn patterned_values(size: usize) -> Vec<u32> {
    (0..size)
        .map(|i| match i % 113 {
            0..=31 => 0,
            32 => 65535,
            33 => 32768,
            34 => 65534,
            p => 1 + (p as u32 * 19 % 37),
        })
        .collect()
}

fn tuples(
    index: &StaticIndex<'_>,
    position: usize,
    cap: usize,
    cache: &mut MatchCache,
) -> Vec<(usize, usize, u32)> {
    let mut found = Vec::new();
    index.find_bounded_matches(
        position,
        cap,
        |s| (s.wrapping_mul(17) % 31) as f32,
        |m| {
            found.push((m.length, m.distance, m.symbol));
        },
        cache,
    );
    found
}

#[test]
fn packed_indexes_preserve_scalar_candidate_order_cache_prefixes_tags_and_deep_caps() {
    let mut input = patterned_values(713);
    input.extend([0, 0, 0, 0, 65535, 65535, 0]);
    for bucket_mode in [SourceBucketMode::Default, SourceBucketMode::WideRing] {
        for index_run_keys in [
            SourceRunKeys::Disabled,
            SourceRunKeys::LengthAndSuffix,
            SourceRunKeys::SharedSuffix,
        ] {
            for index_deep_prefix in [SourceDeepPrefix::Disabled, SourceDeepPrefix::Eight] {
                for index_tags in [
                    SourceIndexTags::Disabled,
                    SourceIndexTags::Scalar,
                    SourceIndexTags::Simd,
                ] {
                    let baseline = SourceConfig {
                        bucket_mode,
                        index_run_keys,
                        index_deep_prefix,
                        index_tags,
                        max_match_length: 79,
                        ..SourceConfig::default()
                    };
                    let wide = StaticIndex::new(&input, 37, &baseline).unwrap();
                    for word in [false, true] {
                        let config = SourceConfig {
                            index_compact_values: true,
                            index_coordinates: word,
                            index_word_match: word,
                            ..baseline.clone()
                        };
                        let packed = StaticIndex::new(&input, 37, &config).unwrap();
                        assert_eq!(packed.layout(), (true, word));
                        assert_eq!(packed.short.slots, wide.short.slots);
                        assert_eq!(packed.short.positions, wide.short.positions);
                        assert_eq!(packed.long.slots, wide.long.slots);
                        assert_eq!(packed.long.positions, wide.long.positions);
                        assert_eq!(packed.short_tags, wide.short_tags);
                        assert_eq!(packed.long_tags, wide.long_tags);
                        for cache_kind in 0..3 {
                            let cache_config = SourceConfig {
                                index_match_cache: cache_kind == 1,
                                block_sparse_cache: cache_kind == 2,
                                ..config.clone()
                            };
                            let mut cache =
                                MatchCache::for_blocks(packed.window, &cache_config).unwrap();
                            for position in 0..=input.len() + 1 {
                                let cap = [2, 3, 7, 16, 79, usize::MAX][position % 6];
                                let expected =
                                    tuples(&wide, position, cap, &mut MatchCache::uncached());
                                let actual = tuples(&packed, position, cap, &mut cache);
                                assert_eq!(actual, expected, "position={position}, cap={cap}");
                                for (length, distance, _) in actual {
                                    assert!(
                                        distance > 0
                                            && distance <= position
                                            && distance < packed.window
                                    );
                                    let limit = 79.min(cap).min(input.len() - position);
                                    assert_eq!(
                                        length,
                                        scalar_length(
                                            &input[position..position + limit],
                                            &input
                                                [position - distance..position - distance + limit]
                                        )
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn coordinate_counts_include_current_saturate_before_deep_cap_and_reuse_run_slots() {
    let mut input = vec![0; 161];
    input.extend([65535, 1, 2]);
    input.extend([0; 67]);
    input.extend([65535, 1, 2, 0, 0]);
    let config = SourceConfig {
        bucket_mode: SourceBucketMode::WideRing,
        index_run_keys: SourceRunKeys::SharedSuffix,
        index_deep_prefix: SourceDeepPrefix::Eight,
        index_coordinates: true,
        index_compact_values: true,
        ..SourceConfig::default()
    };
    let index = StaticIndex::new(&input, 0, &config).unwrap();
    assert_eq!(index.counts.len() * size_of::<u16>(), 2 * input.len());
    let mut shared = 0;
    for (p, &counts) in index.counts.iter().enumerate() {
        let key = hash(&input, p);
        let expected_short = index.short.recent(p, key, 32).len();
        assert_eq!(counts as usize & 255, expected_short);
        if p + 8 <= input.len() {
            let key = run_hash(
                &input,
                &index.zeros,
                p,
                index.window,
                true,
                config.index_run_keys,
            );
            let expected_long = index.long.recent(p, key, 32).len();
            assert_eq!(counts as usize >> 8, expected_long);
            if shared_run(&index.zeros, p, index.window, config.index_run_keys) {
                shared += 1;
                assert_eq!(counts & 0xff00, index.counts[p - 1] & 0xff00);
            }
        } else {
            assert_eq!(counts >> 8, 0);
        }
    }
    assert!(shared > 100);
    let repeated = vec![65535; 71];
    let index = StaticIndex::new(
        &repeated,
        0,
        &SourceConfig {
            index_run_keys: SourceRunKeys::Disabled,
            ..config
        },
    )
    .unwrap();
    assert_eq!(index.counts[63], 0x2020); // Stored long count is 32, not deep's query cap 9.
    for size in 0..=8 {
        let tiny = StaticIndex::new(
            &repeated[..size],
            0,
            &SourceConfig {
                index_coordinates: true,
                index_compact_values: true,
                ..SourceConfig::default()
            },
        )
        .unwrap();
        assert_eq!(tiny.counts.len(), size);
        assert!(tuples(&tiny, size, usize::MAX, &mut MatchCache::uncached()).is_empty());
    }
}

// Independent command replay: no matcher, hash, distance mapper or parser is
// used to reconstruct values. Overlap copies advance one physical value at a time.
fn decode(input: &[Literal], commands: &[Command], maximum: usize) {
    let mut output = Vec::new();
    for command in commands {
        let origin = output.len();
        match *command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[origin]);
                output.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[origin].context);
                assert!((3..=maximum).contains(&(length as usize)));
                assert!(distance != 0 && distance as usize <= origin);
                assert!((distance as usize) < window(input.len()));
                assert!(origin + length as usize <= input.len());
                for _ in 0..length {
                    output.push(output[output.len() - distance as usize]);
                }
            }
        }
    }
    assert_eq!(output, input.iter().map(|t| t.value).collect::<Vec<_>>());
}

fn block_config() -> SourceConfig {
    SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_matcher: true,
        index_mode: SourceIndexMode::All,
        small_index: true,
        match_cost: SourceMatchCost::Frontier,
        block_mode: SourceBlockMode::Intervals,
        max_match_length: 127,
        require_global_savings: false,
        ..SourceConfig::default()
    }
}

fn literals_from(values: &[u32]) -> Vec<Literal> {
    values
        .iter()
        .enumerate()
        .map(|(p, &value)| Literal {
            value,
            context: if p.is_multiple_of(127) {
                u32::MAX
            } else {
                (p % 3) as u32
            },
        })
        .collect()
}

#[test]
fn compact_block_dispatch_decodes_wide_narrow_and_tail_sentinels_with_unchanged_commands() {
    assert!(!SourceConfig::default().index_compact_values);
    assert!(!SourceConfig::default().index_coordinates);
    let mut values = patterned_values(8192);
    for sentinel in [65535, 65536, u32::MAX] {
        *values.last_mut().unwrap() = sentinel;
        let input = literals_from(&values);
        for width in [0, 37] {
            for (block_mode, block_lazy) in [
                (SourceBlockMode::Endpoints, false),
                (SourceBlockMode::Intervals, false),
                (SourceBlockMode::Intervals, true),
            ] {
                let baseline = SourceConfig {
                    block_mode,
                    block_lazy,
                    ..block_config()
                };
                let mut expected_work = Work::default();
                let expected =
                    run(&[&input, &[]], &[width], &baseline, &mut expected_work).unwrap();
                assert_eq!(expected_work.compact_index_streams, 0);
                for index_word_match in [false, true] {
                    let config = SourceConfig {
                        index_compact_values: true,
                        index_coordinates: true,
                        index_word_match,
                        ..baseline.clone()
                    };
                    let mut work = Work::default();
                    let result = run(&[&input, &[]], &[width], &config, &mut work).unwrap();
                    assert_eq!((work.block_streams, work.coordinate_index_streams), (1, 1));
                    assert_eq!(work.compact_index_streams, usize::from(sentinel <= 65535));
                    assert!(result.enabled && result.commands[1].is_empty());
                    assert!(
                        result.commands[0]
                            .iter()
                            .any(|c| matches!(c, Command::Match { .. }))
                    );
                    decode(&input, &result.commands[0], config.max_match_length);
                    assert_eq!(result, expected);
                }
            }
        }
    }
}

#[test]
fn compact_dispatch_bypasses_nonblocks_suffixes_and_follows_suffix_fallback() {
    let input = literals_from(&patterned_values(8192));
    for (block_mode, suffix_mode, suffix_scratch_bytes, expected) in [
        (
            SourceBlockMode::Disabled,
            SourceSuffixMode::Disabled,
            0,
            (0, 0, 1),
        ),
        (
            SourceBlockMode::Intervals,
            SourceSuffixMode::SlidingLcp,
            64 << 20,
            (1, 0, 0),
        ),
        (
            SourceBlockMode::Intervals,
            SourceSuffixMode::SlidingLcp,
            0,
            (0, 1, 1),
        ),
    ] {
        let config = SourceConfig {
            index_compact_values: true,
            index_coordinates: true,
            index_word_match: true,
            block_mode,
            suffix_mode,
            suffix_scratch_bytes,
            ..block_config()
        };
        let mut work = Work::default();
        let result = run(&[&input], &[37], &config, &mut work).unwrap();
        assert_eq!(
            (
                work.suffix_streams,
                work.compact_index_streams,
                work.coordinate_index_streams
            ),
            expected
        );
        decode(&input, &result.commands[0], config.max_match_length);
        let ordinary = SourceConfig {
            index_compact_values: false,
            index_coordinates: false,
            index_word_match: false,
            ..config
        };
        assert_eq!(
            result,
            run(&[&input], &[37], &ordinary, &mut Work::default()).unwrap()
        );
    }
    let rejected_config = SourceConfig {
        index_compact_values: true,
        index_coordinates: true,
        index_word_match: true,
        ..SourceConfig::default()
    };
    let empty = super::super::parse_source_with_state(&[], 0, &rejected_config).unwrap();
    assert!(!empty.enabled && empty.commands.is_empty());
}

#[test]
fn shared_seed_auto_narrows_without_compact_switch_but_disables_words_and_coordinates() {
    let mut streams = vec![patterned_values(65536)];
    for sentinel in [65535, u32::MAX] {
        *streams[0].last_mut().unwrap() = sentinel;
        let config = SourceConfig {
            shared_index: true,
            index_compact_values: false,
            index_coordinates: true,
            index_word_match: true,
            ..block_config()
        };
        let (owner, index) = blocks::shared_index(&streams, &[37], &config)
            .unwrap()
            .unwrap();
        assert_eq!(owner, 0);
        assert_eq!(index.layout(), (sentinel <= 65535, false));
        assert!(!index.word_match);
        let input = literals_from(&streams[0]);
        let costs = super::super::Costs::raw(&[&input], &config);
        let seed = blocks::indexed_seed(&input, &index, &config, &costs).unwrap();
        decode(&input, &seed.0, config.max_match_length);
        let wide = StaticIndex::new(
            &streams[0],
            37,
            &SourceConfig {
                index_compact_values: false,
                index_coordinates: false,
                index_word_match: false,
                index_tags: SourceIndexTags::Scalar,
                index_anchors_first: true,
                ..config.clone()
            },
        )
        .unwrap();
        assert_eq!(
            seed,
            blocks::indexed_seed(&input, &wide, &config, &costs).unwrap()
        );
        let mut work = Work::default();
        let result = run(&[&input], &[37], &config, &mut work).unwrap();
        assert_eq!(work.shared_index_builds, 1);
        assert_eq!(work.compact_index_streams, usize::from(sentinel <= 65535));
        assert_eq!(work.coordinate_index_streams, 0);
        decode(&input, &result.commands[0], config.max_match_length);
    }
}
