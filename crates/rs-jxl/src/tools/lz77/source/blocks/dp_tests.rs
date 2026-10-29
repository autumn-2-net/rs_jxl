// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::super::{
    SourceBlockDyadic, SourceBlockEnvelope, SourceSuffixMode, StaticIndex, suffix::SuffixIndex,
};
use super::*;

fn configure(variant: usize, config: &mut SourceConfig) {
    match variant {
        0 => {}
        1 => config.block_calendar = true,
        2 => config.block_vector_dp = true,
        3 => config.block_range_dominance = true,
        4 => config.block_dp_jump = true,
        5 => config.block_dyadic = SourceBlockDyadic::Endpoints,
        6 => config.block_dyadic = SourceBlockDyadic::Blocks64,
        7 => config.block_dyadic = SourceBlockDyadic::Blocks16,
        8 => config.block_envelope = SourceBlockEnvelope::Blocks64,
        9 => config.block_envelope = SourceBlockEnvelope::Blocks16,
        10 => config.block_envelope = SourceBlockEnvelope::Blocks256,
        11 => {
            config.block_calendar = true;
            config.block_vector_dp = true;
            config.block_range_dominance = true;
            config.block_dp_jump = true;
            config.block_dyadic = SourceBlockDyadic::Blocks16;
            config.block_envelope = SourceBlockEnvelope::Blocks64;
        }
        _ => unreachable!(),
    }
}

fn replay(commands: &[Command], input: &[Literal], end: usize, output: &mut Vec<u32>) -> usize {
    let mut copies = 0;
    for command in commands {
        let position = output.len();
        match *command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[position]);
                output.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[position].context);
                assert!((3..=513).contains(&length));
                assert!(distance > 0 && distance as usize <= position);
                assert!(position + length as usize <= end);
                for _ in 0..length {
                    output.push(output[output.len() - distance as usize]);
                }
                copies += 1;
            }
        }
    }
    assert_eq!(output.len(), end);
    copies
}

#[test]
fn block_parser_variants_decode_across_non_aligned_cuts_with_static_and_suffix_search() {
    let values: Vec<u32> = (0..1025)
        .map(|i| {
            if i < 600 {
                u32::MAX
            } else {
                (i % 17 + 65530) as u32
            }
        })
        .collect();
    let input: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &value)| Literal {
            value,
            context: (i % 3) as u32,
        })
        .collect();
    for suffix_mode in [SourceSuffixMode::Disabled, SourceSuffixMode::SlidingLcp] {
        for block_mode in [SourceBlockMode::Endpoints, SourceBlockMode::Intervals] {
            for variant in 0..12 {
                let mut config = SourceConfig {
                    suffix_mode,
                    block_mode,
                    index_word_match: true,
                    block_sparse_cache: true,
                    block_frontier_carry: true,
                    block_shared_frontier: true,
                    block_distance_table: true,
                    block_distance_classes: true,
                    block_literal_bound: true,
                    block_zero_tail: true,
                    max_match_length: 513,
                    ..SourceConfig::default()
                };
                configure(variant, &mut config);
                let costs = Costs::raw(&[&input], &config);
                let prices =
                    DistancePrices::new(window(input.len()), 37, &config, |d| costs.distance(d))
                        .unwrap();
                let index = StaticIndex::new(&values, 37, &config).unwrap();
                let suffix = SuffixIndex::new(&values, 37, &config).unwrap();
                let find = |p,
                            query: Query<'_>,
                            cache: &mut MatchCache,
                            price: &dyn Fn(u32) -> f32,
                            visit: &mut dyn FnMut(Match)| {
                    if let Some(index) = &suffix {
                        index.find_query(p, query, price, visit, cache);
                    } else {
                        index.find_query(p, query, price, visit, cache);
                    }
                };
                let mut output = Vec::new();
                let mut copies = 0;
                for cut in [0, 337, 713, input.len()].windows(2) {
                    let commands = parse_part(
                        &input,
                        cut[0],
                        cut[1],
                        37,
                        &config,
                        &costs,
                        &prices,
                        &[],
                        &find,
                    )
                    .unwrap();
                    copies += replay(&commands, &input, cut[1], &mut output);
                }
                assert!(copies > 0);
                assert_eq!(output, values);
            }
        }
    }
}

#[test]
fn lazy_parser_bypasses_all_new_dp_variants() {
    let values: Vec<u32> = (0..257).map(|i| i % 7).collect();
    let input: Vec<_> = values
        .iter()
        .map(|&value| Literal { value, context: 0 })
        .collect();
    let mut config = SourceConfig {
        block_mode: SourceBlockMode::Intervals,
        block_lazy: true,
        ..SourceConfig::default()
    };
    let costs = Costs::raw(&[&input], &config);
    let prices =
        DistancePrices::new(window(input.len()), 0, &config, |d| costs.distance(d)).unwrap();
    let index = StaticIndex::new(&values, 0, &config).unwrap();
    let find =
        |p,
         query: Query<'_>,
         cache: &mut MatchCache,
         price: &dyn Fn(u32) -> f32,
         visit: &mut dyn FnMut(Match)| index.find_query(p, query, price, visit, cache);
    let baseline = parse_part(
        &input,
        0,
        input.len(),
        0,
        &config,
        &costs,
        &prices,
        &[],
        &find,
    )
    .unwrap();
    configure(11, &mut config);
    let changed = parse_part(
        &input,
        0,
        input.len(),
        0,
        &config,
        &costs,
        &prices,
        &[],
        &find,
    )
    .unwrap();
    assert_eq!(changed, baseline);
    let mut output = Vec::new();
    assert!(replay(&changed, &input, input.len(), &mut output) > 0);
    assert_eq!(output, values);
}
