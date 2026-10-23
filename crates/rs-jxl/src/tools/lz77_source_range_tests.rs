// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;

fn variants() -> Vec<Options> {
    let mut variants = vec![
        Options::default(),
        Options {
            calendar: true,
            ..Options::default()
        },
        Options {
            vector: true,
            ..Options::default()
        },
        Options {
            dominance: true,
            ..Options::default()
        },
    ];
    for dyadic in [
        SourceBlockDyadic::Endpoints,
        SourceBlockDyadic::Blocks64,
        SourceBlockDyadic::Blocks16,
    ] {
        variants.push(Options {
            dyadic,
            ..Options::default()
        });
        variants.push(Options {
            dyadic,
            calendar: true,
            vector: true,
            dominance: true,
            envelope: SourceBlockEnvelope::Blocks16,
        });
    }
    for envelope in [
        SourceBlockEnvelope::Blocks64,
        SourceBlockEnvelope::Blocks16,
        SourceBlockEnvelope::Blocks256,
    ] {
        variants.push(Options {
            envelope,
            ..Options::default()
        });
        variants.push(Options {
            envelope,
            calendar: true,
            vector: true,
            dominance: true,
            ..Options::default()
        });
    }
    variants
}

#[test]
fn all_block_variants_match_unpruned_scalar_costs_at_alignment_boundaries() {
    for options in variants() {
        for size in [1usize, 15, 16, 17, 63, 64, 65, 255, 256, 257, 513] {
            for limit in [0, 2, 1 << 16] {
                let mut parser = HeapParser::with_options(size, limit, options).unwrap();
                let mut oracle = vec![f32::INFINITY; size + 1];
                oracle[0] = 0.0;
                for origin in 0..size {
                    parser.advance(origin);
                    assert_eq!(
                        parser.at(origin).cost,
                        oracle[origin],
                        "size={size} origin={origin}"
                    );
                    let literal = oracle[origin] + 11.0;
                    parser.relax_context(origin + 1, origin, 0, origin as u32, literal);
                    oracle[origin + 1] = oracle[origin + 1].min(literal);
                    for (offset, span) in [(1, 2), (3, 17), (7, 81), (11, 300)] {
                        let end = (size - origin + 1).min(offset + span);
                        if offset >= end {
                            continue;
                        }
                        let cost = oracle[origin] + ((origin + span) % 13) as f32;
                        parser.interval_context(origin, offset, end, origin + 1, 0x8765_4321, cost);
                        for best in &mut oracle[origin + offset..origin + end] {
                            *best = best.min(cost);
                        }
                    }
                }
                parser.advance(size);
                assert_eq!(parser.at(size).cost, oracle[size]);
            }
        }
    }
}

#[test]
fn calendar_recycles_slots_counts_active_offers_and_expires_exclusively() {
    let mut parser = HeapParser::with_options(
        512,
        2,
        Options {
            calendar: true,
            ..Options::default()
        },
    )
    .unwrap();
    for origin in 0..450 {
        parser.advance(origin);
        parser.interval_context(origin, 1, 40, 1, 7, (origin % 3) as f32);
        assert!(parser.queued() <= 2);
        assert!(parser.events.len() <= 2);
        assert_eq!(parser.pending.len(), 0);
    }
    assert_eq!(parser.events.len(), 1);
    parser.advance(512);
    assert_eq!(parser.queued(), 0);
    assert_eq!(parser.pending_events, 0);
    assert!(parser.at(512).cost.is_infinite());
}

#[test]
fn calendar_equal_cost_offers_keep_existing_explicit_endpoints() {
    let mut parser = HeapParser::with_options(
        80,
        65536,
        Options {
            calendar: true,
            ..Options::default()
        },
    )
    .unwrap();
    parser.interval_context(0, 3, 70, 1, 13, 5.0);
    parser.interval_context(1, 2, 69, 2, 17, 5.0);
    parser.interval_context(2, 18, 20, 3, 19, 5.0);
    parser.advance(20);
    assert_eq!((parser.at(20).distance, parser.at(20).context), (3, 19));
    parser.advance(22);
    assert_eq!((parser.at(22).distance, parser.at(22).context), (1, 13));
    parser.advance(70);
    assert!(parser.at(70).cost.is_infinite());
}

#[test]
fn dominance_only_certifies_full_blocks_with_four_block_query_and_32_block_update_limits() {
    let mut parser = HeapParser::with_options(
        2048,
        0,
        Options {
            dominance: true,
            ..Options::default()
        },
    )
    .unwrap();
    parser.interval(0, 17, 1056, 1, 7.0);
    assert!(parser.cover_costs[1].is_infinite());
    assert_eq!(parser.cover_costs[2], 7.0);
    assert_eq!(parser.cover_costs[33], 7.0);
    assert!(parser.cover_costs[34].is_infinite());
    parser.interval(0, 32, 96, 2, 7.0); // Exactly four wholly certified blocks.
    assert_eq!(parser.dominated, 1);
    parser.interval(0, 32, 97, 2, 7.0); // Five touched blocks: no query pruning.
    assert_eq!(parser.dominated, 1);
    parser.interval(0, 17, 33, 2, 7.0); // Partial uncertified block.
    assert_eq!(parser.dominated, 1);
    parser.interval(0, 32, 96, 2, 6.0); // Cheaper offers must not be dropped.
    assert_eq!(parser.dominated, 1);
    assert_eq!(parser.at(64).cost, 6.0);
}

#[test]
fn tree_and_envelope_keep_source_strict_ties_and_precedence() {
    let mut parser = HeapParser::with_options(
        256,
        65536,
        Options {
            dyadic: SourceBlockDyadic::Blocks16,
            envelope: SourceBlockEnvelope::Blocks64,
            calendar: true,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(parser.dyadic_base, 32); // Strictly greater than size/16.
    parser.interval_context(2, 30, 126, 9, 7, 3.0);
    parser.interval_context(1, 31, 127, 8, 6, 3.0);
    for pos in 32..128 {
        parser.advance(pos);
        assert_eq!((parser.at(pos).distance, parser.at(pos).context), (9, 7));
    }
    assert!(
        parser
            .block_best
            .iter()
            .all(|offer| offer.cost.is_infinite())
    );
    assert!(parser.events.is_empty());
    parser.advance(128);
    assert!(parser.at(128).cost.is_infinite());
}

#[test]
fn literal_jump_stops_before_events_and_explicit_destinations_and_obeys_storage_gates() {
    let mut parser = HeapParser::new(128, 65536).unwrap();
    parser.interval(0, 3, 100, 1, 5.0);
    parser.interval(1, 39, 119, 2, 4.0);
    parser.advance(3);
    assert_eq!(parser.jump_literal_only(3, 90), 39);
    assert_eq!(parser.at(39).length, 39);
    parser.advance(39);
    assert_eq!(parser.jump_literal_only(39, 90), 39);
    parser.advance(40);
    assert_eq!(parser.jump_literal_only(40, 127), 119);

    let mut explicit = HeapParser::new(128, 65536).unwrap();
    explicit.interval(0, 3, 100, 1, 5.0);
    explicit.interval(0, 20, 22, 2, 10.0);
    explicit.advance(3);
    assert_eq!(explicit.jump_literal_only(3, 90), 3);
    explicit.advance(21);
    assert_eq!(explicit.jump_literal_only(21, 90), 90);

    for options in [
        Options {
            vector: true,
            ..Options::default()
        },
        Options {
            calendar: true,
            ..Options::default()
        },
        Options {
            dyadic: SourceBlockDyadic::Endpoints,
            ..Options::default()
        },
        Options {
            envelope: SourceBlockEnvelope::Blocks16,
            ..Options::default()
        },
    ] {
        let mut parser = HeapParser::with_options(128, 65536, options).unwrap();
        parser.interval(0, 3, 100, 1, 5.0);
        parser.advance(3);
        assert_eq!(parser.jump_literal_only(3, 90), 3);
    }
}

#[test]
fn block_options_default_off_and_endpoint_mode_gates_are_source_exact() {
    let default = SourceConfig::default();
    assert!(
        !default.bucket_distances
            && !default.block_calendar
            && !default.block_vector_dp
            && !default.block_range_dominance
            && !default.block_dp_jump
    );
    assert_eq!(default.block_dyadic, SourceBlockDyadic::Disabled);
    assert_eq!(default.block_envelope, SourceBlockEnvelope::Disabled);
    let config = SourceConfig {
        block_mode: SourceBlockMode::Endpoints,
        block_calendar: true,
        block_vector_dp: true,
        block_range_dominance: true,
        block_dyadic: SourceBlockDyadic::Endpoints,
        block_envelope: SourceBlockEnvelope::Blocks16,
        ..default
    };
    let Parser::Heap(parser) = Parser::blocks(257, &config).unwrap() else {
        unreachable!()
    };
    assert!(parser.vector.is_some() && !parser.cover_costs.is_empty());
    assert!(parser.heads.is_empty() && parser.dyadic.is_empty() && parser.block_best.is_empty());
    assert_eq!(parser.queue_limit, 0);
    assert_eq!(std::mem::size_of::<Step>(), 16);
    assert_eq!(std::mem::size_of::<DyadicOffer>(), 16);
    assert!(
        (1 << 16) * (2 * std::mem::size_of::<Offer>() + std::mem::size_of::<Event>()) <= 8 << 20
    );
    assert!(HeapParser::new(i32::MAX as usize + 1, 0).is_err());
    assert!(HeapParser::new(1, u32::MAX as usize).is_err());
}

#[test]
fn dyadic_and_combined_arrays_are_included_in_bounded_worker_scratch() {
    let mut config = SourceConfig {
        block_mode: SourceBlockMode::Intervals,
        ..SourceConfig::default()
    };
    let baseline = Parser::block_scratch(256, &config);
    config.block_dyadic = SourceBlockDyadic::Endpoints;
    assert_eq!(Parser::block_scratch(256, &config) - baseline, 2 * 512 * 16);
    assert_eq!(Parser::block_scratch(usize::MAX, &config), usize::MAX);
    config.block_calendar = true;
    config.block_range_dominance = true;
    config.block_envelope = SourceBlockEnvelope::Blocks16;
    assert_eq!(
        Parser::block_scratch(256, &config) - baseline,
        2 * 512 * 16 + 257 * 4 + 17 * 4 + 17 * std::mem::size_of::<Offer>()
    );
}
