// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.

use super::*;
use crate::bit_writer::BitWriter;
use crate::entropy::options::{AnsHistogramStrategy, ClusterStrategy, UintConfigStrategy};
use crate::jpeg::dc_streams::{CflMaps, PrepareLimits, prepare_ac_metadata_group};
use crate::jpeg::frame_header::FrameGeometry;
use crate::jpeg::transcode_layout::ChromaSubsampling;
use crate::modular::encode::write_prepared_stream;
use crate::modular::group::Channel;
use crate::modular::tree::{Node, Tree};

fn stream_bits(planes: &[Channel], options: &ModularOptions, id: usize, offset: u8) -> Vec<u8> {
    let mut writer = BitWriter::new(1 << 20);
    writer.write(offset, (1u64 << offset) - 1).unwrap();
    write_prepared_stream(planes, &[], options, &mut writer, id, 8).unwrap();
    writer.write(3, 5).unwrap();
    let size = writer.bits_written();
    writer.zero_pad_to_byte().unwrap();
    let bytes = writer.into_bytes().unwrap();
    (0..size).map(|i| (bytes[i / 8] >> (i % 8)) & 1).collect()
}

fn channels(width: usize, height: usize) -> Vec<Channel> {
    (0..3)
        .map(|c| Channel {
            width,
            height,
            hshift: 0,
            vshift: 0,
            samples: (0..width * height)
                .map(|i| ((i * 173 + c * 37) % 20001) as i32 - 10000)
                .collect(),
        })
        .collect()
}

fn split_values(tree: &Tree, property: usize, predictor: Predictor) -> Vec<i32> {
    let mut cutoffs = Vec::new();
    for node in tree.nodes() {
        match *node {
            Node::Split {
                property: actual,
                threshold,
                ..
            } => {
                assert_eq!(actual, property);
                cutoffs.push(threshold);
            }
            Node::Leaf {
                predictor: actual,
                offset,
                multiplier,
            } => assert_eq!((actual, offset, multiplier), (predictor, 0, 1)),
        }
    }
    cutoffs.sort_unstable();
    cutoffs
}

#[test]
fn every_tier_keeps_source_dc_predictor_and_fixed_tree_precedence() {
    for tier in -1..=9 {
        for decode_speed in 0..=4 {
            let dc = options_for_dc(tier, decode_speed).unwrap();
            assert_eq!(
                dc.predictor,
                if tier < 2 {
                    Predictor::Variable
                } else if tier == 2 {
                    Predictor::Best
                } else {
                    Predictor::Weighted
                }
            );
            assert_eq!(
                dc.fixed_tree,
                if decode_speed >= 1 {
                    Some(FixedTree::Gradient)
                } else if tier >= 3 {
                    Some(FixedTree::Weighted)
                } else {
                    None
                }
            );
            assert!(dc.custom_tree.is_none());
            let sampling = dc.sampling.as_ref().unwrap();
            assert_eq!(
                sampling.tree_mode,
                if tier >= 3 {
                    TreeMode::WeightedOnly
                } else {
                    TreeMode::Default
                }
            );
            assert_eq!(sampling.predictor_override, None);
            assert_eq!(sampling.fast_decode_multiplier, 1.0);
            assert_eq!(sampling.max_channel_size, 0x00ff_ffff);
            assert_eq!(
                sampling.node_threshold,
                (75 + 14 * i32::from(tier) + 10 * i32::from(decode_speed)) as f32
            );
        }
    }
}

#[test]
fn jpeg_metadata_is_one_zero_leaf_except_slow_forced_cfl() {
    for tier in -1..=9 {
        for decode_speed in 0..=4 {
            for forced in [false, true] {
                let options = StreamContext {
                    force_cfl_jpeg_recompression: forced,
                    ..StreamContext::default()
                }
                .options_for_ac_metadata(tier, decode_speed)
                .unwrap();
                assert_eq!(options.predictor, Predictor::Gradient);
                assert_eq!(options.fixed_tree, None);
                assert_eq!(options.custom_tree.is_none(), forced && tier < 3);
                assert_eq!(
                    options.sampling.as_ref().unwrap().tree_mode,
                    TreeMode::NoWeighted
                );
                if let Some(tree) = options.custom_tree {
                    let signature: Vec<_> =
                        tree.tokens().iter().map(|t| (t.context, t.value)).collect();
                    assert_eq!(signature, [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
                    // Nonzero CfL remains ordinary data even when the fixed tree
                    // was selected: Zero is a predictor, not a zero-fill command.
                    for channel in 0..4 {
                        assert_eq!(
                            tree.lookup(&[channel, 99]).unwrap(),
                            (0, Predictor::Zero, 0, 1)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn source_default_forced_cfl_changes_only_the_slow_metadata_tree() {
    let source_default = StreamContext::default();
    assert!(source_default.force_cfl_jpeg_recompression);
    let disabled = StreamContext {
        force_cfl_jpeg_recompression: false,
        ..source_default
    };
    for tier in -1..=9 {
        for decode_speed in 0..=4 {
            let default_ac = options_for_ac_metadata(tier, decode_speed).unwrap();
            let disabled_ac = disabled
                .options_for_ac_metadata(tier, decode_speed)
                .unwrap();
            assert_eq!(default_ac.custom_tree.is_none(), tier < 3);
            assert!(disabled_ac.custom_tree.is_some());
            assert_eq!(default_ac.fixed_tree, None);
            assert_eq!(default_ac.predictor, disabled_ac.predictor);
            assert_eq!(default_ac.entropy_options, disabled_ac.entropy_options);
            let sampling = default_ac.sampling.as_ref().unwrap();
            assert_eq!(sampling.tree_mode, TreeMode::NoWeighted);
            assert_eq!(
                sampling.properties,
                disabled_ac.sampling.unwrap().properties
            );
            // Decode speed 4 suppresses sampling but does not restore the fixed
            // tree kind after AddACMetadata has selected Learn.
            if decode_speed == 4 {
                assert_eq!(sampling.nb_repeats, 0.0);
            }
            for (default_options, disabled_options) in [
                (
                    source_default.options_for_dc(tier, decode_speed),
                    disabled.options_for_dc(tier, decode_speed),
                ),
                (
                    source_default.options_for_quant(tier, decode_speed),
                    disabled.options_for_quant(tier, decode_speed),
                ),
            ] {
                let default_options = default_options.unwrap();
                let disabled_options = disabled_options.unwrap();
                assert_eq!(default_options.predictor, disabled_options.predictor);
                assert_eq!(default_options.fixed_tree, disabled_options.fixed_tree);
                assert!(default_options.custom_tree.is_none());
                assert!(disabled_options.custom_tree.is_none());
                assert_eq!(
                    default_options.entropy_options,
                    disabled_options.entropy_options
                );
                assert_eq!(
                    default_options.sampling.unwrap().properties,
                    disabled_options.sampling.unwrap().properties
                );
            }
        }
    }
}

#[test]
fn sampling_preserves_stream_count_fraction_and_override_order() {
    let small = StreamContext {
        num_streams: 29,
        ..StreamContext::default()
    };
    let large = StreamContext {
        num_streams: 30,
        ..small
    };
    let small_quant = small.options_for_quant(3, 0).unwrap().sampling.unwrap();
    let large_quant = large.options_for_quant(3, 0).unwrap().sampling.unwrap();
    assert_eq!(small_quant.properties, [0, 15, 9, 10, 11, 12, 13]);
    assert_eq!(large_quant.properties, [0, 1, 15, 9, 10, 11, 12]);
    assert_eq!(small_quant.max_property_values, 96);
    assert_eq!(small_quant.max_channel_size, 256);
    assert_eq!(
        small
            .options_for_quant(3, 3)
            .unwrap()
            .sampling
            .unwrap()
            .properties,
        [9]
    );
    assert_eq!(
        small
            .options_for_ac_metadata(3, 3)
            .unwrap()
            .sampling
            .unwrap()
            .properties,
        [0, 9, 10, 11, 12, 13]
    );
    assert_eq!(
        small
            .options_for_dc(3, 3)
            .unwrap()
            .sampling
            .unwrap()
            .properties,
        [15]
    );
    for (tier, scale, bins) in [
        (-1, 0.3, 32),
        (0, 1.3, 256),
        (1, 1.3, 256),
        (2, 1.1, 128),
        (3, 1.0, 96),
        (4, 0.7, 64),
        (5, 0.5, 48),
        (6, 0.3, 32),
        (9, 0.3, 32),
    ] {
        let context = StreamContext {
            learning_fraction: 0.9,
            ..small
        };
        let sampling = context
            .options_for_quant(tier, 0)
            .unwrap()
            .sampling
            .unwrap();
        assert_eq!(
            sampling.nb_repeats.to_bits(),
            (0.9f32 * scale).min(1.0).to_bits()
        );
        assert_eq!(sampling.max_property_values, bins);
        for options in [
            context.options_for_dc(tier, 4),
            context.options_for_ac_metadata(tier, 4),
            context.options_for_quant(tier, 4),
        ] {
            assert_eq!(options.unwrap().sampling.unwrap().nb_repeats, 0.0);
        }
    }
}

#[test]
fn shared_sampling_scales_once_and_small_quant_channels_remain_full_samples() {
    use crate::modular::sample::{ChannelSampler, tree_sample_fraction};

    for tier in -1..=9 {
        for decode_speed in 0..=4 {
            let quant = options_for_quant(tier, decode_speed).unwrap();
            let repeats = quant.sampling.unwrap().nb_repeats;
            let sampler = ChannelSampler::new(8, 8, repeats).unwrap();
            assert_eq!(
                sampler.fraction(),
                if decode_speed == 4 { 0.0 } else { 1.0 }
            );
            assert_eq!(
                sampler.filter(|&selected| selected).count(),
                if decode_speed == 4 { 0 } else { 64 }
            );
        }
    }
    let context = StreamContext {
        learning_fraction: 0.5,
        ..StreamContext::default()
    };
    let options = context.options_for_quant(5, 0).unwrap();
    let repeats = options.sampling.as_ref().unwrap().nb_repeats;
    assert_eq!(repeats.to_bits(), 0.25f32.to_bits());
    for (area, fraction) in [
        (0, 0.0),
        (1024, 1.0),
        (1025, 1024.0 / 1025.0),
        (4096, 0.25),
        (4097, 0.25),
    ] {
        assert_eq!(tree_sample_fraction(area, repeats).unwrap(), fraction);
    }
    let mut capped = options;
    capped.sample_limit = Some(1);
    assert_eq!(
        capped.sampling.unwrap().nb_repeats.to_bits(),
        repeats.to_bits()
    );
    for fraction in [0.0, 1.01, 10.0, f32::MAX] {
        let options = StreamContext {
            learning_fraction: fraction,
            ..context
        }
        .options_for_quant(5, 0)
        .unwrap();
        assert_eq!(
            options.sampling.unwrap().nb_repeats.to_bits(),
            (fraction * 0.5).min(1.0).to_bits()
        );
    }
}

#[test]
fn fixed_dc_cutoffs_population_boundaries_and_properties_match_source() {
    let full = [
        -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -4, -3, -1, 0, 1, 3,
        5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
    ];
    for (population, expected) in [
        (0, &[][..]),
        (512, &[][..]),
        (513, &[0][..]),
        (4096, &[0][..]),
        (4097, &[-31, 0, 47][..]),
        (8192, &[-31, 0, 47][..]),
        (8193, &full[..]),
        (usize::MAX, &full[..]),
    ] {
        for decode_speed in 0..=4 {
            let options = StreamContext {
                dc_tree_pixels: Some(population),
                ..StreamContext::default()
            }
            .options_for_dc(3, decode_speed)
            .unwrap();
            assert_eq!(options.fixed_tree, None);
            let tree = options.custom_tree.unwrap();
            let (property, predictor) = if decode_speed == 0 {
                (15, Predictor::Weighted)
            } else {
                (9, Predictor::Gradient)
            };
            assert_eq!(split_values(&tree, property, predictor), expected);
            assert_eq!(tree.leaves(), expected.len() + 1);
            // Check strict greater-child semantics independently of serialization.
            let mut properties = [0; 20];
            properties[property] = i32::MIN;
            let mut below = tree.lookup(&properties).unwrap();
            for &cutoff in expected {
                properties[property] = cutoff;
                assert_eq!(tree.lookup(&properties).unwrap(), below);
                properties[property] = cutoff + 1;
                let above = tree.lookup(&properties).unwrap();
                assert_ne!(above.0, below.0);
                below = above;
            }
        }
    }
    let learned = StreamContext {
        dc_tree_pixels: Some(8193),
        ..StreamContext::default()
    }
    .options_for_dc(2, 0)
    .unwrap();
    assert!(learned.custom_tree.is_none());
    assert!(learned.fixed_tree.is_none());
}

#[test]
fn entropy_lz_and_quant_inheritance_do_not_leak_pixel_defaults() {
    for tier in -1..=9 {
        for decode_speed in 0..=4 {
            let dc = options_for_dc(tier, decode_speed).unwrap();
            let ac = options_for_ac_metadata(tier, decode_speed).unwrap();
            let quant = options_for_quant(tier, decode_speed).unwrap();
            assert_eq!(dc.entropy_options, ac.entropy_options);
            assert_eq!(dc.lz77, tier <= 2);
            assert_eq!(
                dc.source_lz77.as_ref().map(|c| c.method),
                if tier <= 1 {
                    Some(SourceLz77Method::Optc256)
                } else if tier == 2 {
                    Some(SourceLz77Method::Lz77b3w3t)
                } else {
                    None
                }
            );
            assert_eq!(
                dc.entropy_options.max_histograms,
                if decode_speed >= 2 { 12 } else { 256 }
            );
            if tier >= 3 {
                assert_eq!(
                    dc.entropy_options.uint_config,
                    UintConfigStrategy::Fixed(Default::default())
                );
            }
            assert_eq!(quant.predictor, Predictor::Gradient);
            assert_eq!(quant.fixed_tree, None);
            assert!(quant.custom_tree.is_none());
            assert!(quant.lz77);
            assert_eq!(quant.lz77_mode, Lz77Mode::Rle);
            assert!(quant.source_lz77.is_none());
            assert_eq!(quant.entropy_options.clustering, ClusterStrategy::Best);
            assert_eq!(
                quant.entropy_options.ans_histogram,
                AnsHistogramStrategy::Precise
            );
            assert_eq!(quant.entropy_options.uint_config, UintConfigStrategy::Best);
            assert_eq!(quant.entropy_options.max_histograms, 256);
            for options in [&dc, &ac, &quant] {
                options.validate().unwrap();
                assert_eq!(options.entropy, Coding::Source);
                assert!(options.source_tree_entropy);
                assert_eq!(
                    options.entropy_options.lz77_length_config.split_exponent(),
                    0
                );
                assert_eq!(options.rct, None);
                assert_eq!(options.previous_channels, 0);
                assert_eq!(options.weighted_preset, 0);
                assert_eq!(options.weighted_candidates, None);
                assert_eq!(options.rct_candidates, None);
                assert!(options.squeeze.is_empty());
                assert!(options.palette_policy.is_none());
                assert_eq!(options.palette_colors, None);
                assert!(!options.source_constant_channels);
                assert!(!options.parallel_tokens);
                assert!(options.source_gather.is_none());
                assert!(options.tree_groups.is_none());
                assert_eq!(options.max_tree_leaves, MAX_LEAVES);
            }
        }
    }
}

#[test]
fn all_tier_global_entropy_policies_match_source_without_domain_trees_or_rle() {
    for tier in -1..=9 {
        for decode_speed in 0..=4 {
            let global = options_for_global_entropy(tier, decode_speed).unwrap();
            global.validate().unwrap();
            assert!(global.sampling.is_none());
            assert!(global.custom_tree.is_none());
            assert!(global.fixed_tree.is_none());
            assert!(global.tree_groups.is_none());
            assert!(global.tree_group_splits.is_empty());
            assert_eq!(global.predictor, Predictor::Gradient);
            assert_eq!(global.rct, None);
            assert_eq!(global.entropy, Coding::Source);
            assert!(global.source_tree_entropy);
            assert!(!global.entropy_options.streaming_mode);
            assert!(!global.entropy_options.add_missing_symbols);
            assert!(!global.entropy_options.add_fixed_histograms);
            assert_eq!(
                global.entropy_options.max_histograms,
                if decode_speed >= 2 { 12 } else { 256 }
            );
            assert_eq!(
                global.entropy_options.clustering,
                if tier <= 2 {
                    ClusterStrategy::Best
                } else {
                    ClusterStrategy::Fast
                }
            );
            assert_eq!(
                global.entropy_options.ans_histogram,
                if tier <= 2 {
                    AnsHistogramStrategy::Precise
                } else if tier <= 8 {
                    AnsHistogramStrategy::Approximate
                } else {
                    AnsHistogramStrategy::Fast
                }
            );
            assert_eq!(
                global.entropy_options.uint_config,
                if tier <= 2 {
                    UintConfigStrategy::Best
                } else {
                    UintConfigStrategy::Fixed(Default::default())
                }
            );
            let length = global.entropy_options.lz77_length_config;
            assert_eq!(
                (
                    length.split_exponent(),
                    length.msb_in_token(),
                    length.lsb_in_token()
                ),
                (0, 0, 0)
            );
            assert_eq!(global.entropy_options.lz77_min_symbol, 224);
            assert_eq!(global.lz77, tier <= 2);
            if let Some(config) = &global.source_lz77 {
                assert_eq!(
                    config.method,
                    if tier <= 1 {
                        SourceLz77Method::Optc256
                    } else {
                        SourceLz77Method::Lz77b3w3t
                    }
                );
                assert_eq!(
                    global.lz77_mode,
                    if tier <= 1 {
                        Lz77Mode::Optimal
                    } else {
                        Lz77Mode::Greedy
                    }
                );
                assert_eq!(config.length_uint_config, length);
                assert_eq!(config.min_symbol(), 224);
                assert_eq!(config.literal_uint_config, Default::default());
                assert!(!config.bucket_matcher);
                assert!(!config.careful_first_pass);
                assert!(config.parallelism.is_none());
            } else {
                assert!(!global.lz77);
            }
            assert!(!global.lz77 || global.source_lz77.is_some());
            for local in [
                options_for_dc(tier, decode_speed),
                options_for_ac_metadata(tier, decode_speed),
            ] {
                let local = local.unwrap();
                assert_eq!(global.entropy_options, local.entropy_options);
                assert_eq!(
                    global.source_lz77.as_ref().map(|c| c.method),
                    local.source_lz77.as_ref().map(|c| c.method)
                );
            }
            // AddQuantTable's untouched HistogramParams are local-only, even
            // at slow tiers where its numerical entropy strategies coincide.
            let local_quant = options_for_quant(tier, decode_speed).unwrap();
            assert!(local_quant.lz77 && local_quant.source_lz77.is_none());
            assert_eq!(local_quant.lz77_mode, Lz77Mode::Rle);
            assert_eq!(local_quant.entropy_options.max_histograms, 256);
        }
    }
}

#[test]
fn heterogeneous_shared_writer_accepts_global_policy_across_tier_boundaries() {
    use crate::modular::auxiliary::{Domain, Shared, Stream};

    let dc = channels(2, 1);
    let geometry = FrameGeometry::new(
        16,
        8,
        ChromaSubsampling::from_sampling([1; 3], [1; 3]).unwrap(),
    )
    .unwrap();
    let metadata = prepare_ac_metadata_group(
        geometry,
        0,
        CflMaps {
            width: 1,
            height: 1,
            ytox: &[0],
            ytob: &[0],
        },
        &PrepareLimits::default(),
    )
    .unwrap();
    let quant = crate::jpeg::quantization::RawDct8Quantization::from_source_shift(
        std::array::from_fn(|c| std::array::from_fn(|i| 1 + ((i * 97 + c * 11) % 65535) as i32)),
        0,
    )
    .unwrap()
    .prepare_modular_channels()
    .unwrap();
    let dc_streams = [Stream {
        id: 1,
        channels: &dc,
        transforms: &[],
    }];
    let ac_streams = [Stream {
        id: 3,
        channels: &metadata.channels,
        transforms: &[],
    }];
    let quant_streams = [Stream {
        id: 4,
        channels: &quant,
        transforms: &[],
    }];
    for (tier, decode_speed) in [(-1, 0), (1, 2), (2, 0), (3, 0), (8, 1), (9, 4)] {
        let context = StreamContext {
            dc_tree_pixels: Some(dc.iter().map(|c| c.samples.len()).sum()),
            ..StreamContext::default()
        };
        let dc_options = context.options_for_dc(tier, decode_speed).unwrap();
        let ac_options = context.options_for_ac_metadata(tier, decode_speed).unwrap();
        let quant_options = context.options_for_quant(tier, decode_speed).unwrap();
        let domains = [
            Domain {
                start: 1,
                streams: &dc_streams,
                options: &dc_options,
            },
            Domain {
                start: 3,
                streams: &ac_streams,
                options: &ac_options,
            },
            Domain {
                start: 4,
                streams: &quant_streams,
                options: &quant_options,
            },
        ];
        let global = options_for_global_entropy(tier, decode_speed).unwrap();
        let shared = Shared::build(&domains, context.num_streams, &global, 8, 1024).unwrap();
        let mut outputs = Vec::new();
        for offset in [0u8, 7] {
            let mut writer = BitWriter::new(1 << 20);
            writer.write(offset, (1u64 << offset) - 1).unwrap();
            shared.write_global_info(&mut writer).unwrap();
            let before_empty = writer.bits_written();
            for id in [0, 2, 5, 21] {
                shared.write_stream(id, &mut writer).unwrap();
            }
            assert_eq!(writer.bits_written(), before_empty);
            for id in [1, 3, 4] {
                let before = writer.bits_written();
                shared.write_stream(id, &mut writer).unwrap();
                assert!(writer.bits_written() > before);
            }
            let size = writer.bits_written();
            writer.zero_pad_to_byte().unwrap();
            let bytes = writer.into_bytes().unwrap();
            outputs.push(
                (usize::from(offset)..size)
                    .map(|i| (bytes[i / 8] >> (i % 8)) & 1)
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(outputs[0], outputs[1], "tier={tier} decode={decode_speed}");
    }
}

#[test]
fn prepared_writer_uses_deferred_and_shared_population_dc_trees() {
    let planes = channels(171, 1); // 513 samples, not the spatial area 171.
    for decode_speed in [0, 1] {
        let local = options_for_dc(3, decode_speed).unwrap();
        let explicit = StreamContext {
            dc_tree_pixels: Some(513),
            ..StreamContext::default()
        }
        .options_for_dc(3, decode_speed)
        .unwrap();
        assert_eq!(
            stream_bits(&planes, &local, 1, 0),
            stream_bits(&planes, &explicit, 1, 0)
        );
        let shared = StreamContext {
            dc_tree_pixels: Some(8193),
            ..StreamContext::default()
        }
        .options_for_dc(3, decode_speed)
        .unwrap();
        assert_ne!(
            stream_bits(&planes, &local, 1, 0),
            stream_bits(&planes, &shared, 1, 0)
        );
    }
}

#[test]
fn prepared_metadata_keeps_source_channel_order_and_nonzero_cfl() {
    let geometry = FrameGeometry::new(
        65,
        17,
        ChromaSubsampling::from_sampling([1; 3], [1; 3]).unwrap(),
    )
    .unwrap();
    let group = prepare_ac_metadata_group(
        geometry,
        0,
        CflMaps {
            width: 2,
            height: 1,
            ytox: &[-8, 127],
            ytob: &[3, -128],
        },
        &PrepareLimits::default(),
    )
    .unwrap();
    assert_eq!(group.ac_metadata_size, 27);
    assert_eq!(group.channels[0].samples, [-8, 127]);
    assert_eq!(group.channels[1].samples, [3, -128]);
    assert_eq!((group.channels[0].hshift, group.channels[1].vshift), (3, 3));
    assert_eq!((group.channels[2].width, group.channels[2].height), (27, 2));
    assert_eq!((group.channels[3].width, group.channels[3].height), (9, 3));
    assert!(
        group.channels[2..]
            .iter()
            .all(|c| c.samples.iter().all(|&x| x == 0))
    );
    let options = options_for_ac_metadata(3, 0).unwrap();
    let original = stream_bits(&group.channels, &options, 3, 0);
    let mut zeroed = group.channels.clone();
    zeroed[0].samples.fill(0);
    zeroed[1].samples.fill(0);
    assert_ne!(original, stream_bits(&zeroed, &options, 3, 0));
}

#[test]
fn every_policy_writes_unaligned_prepared_streams_without_mutating_samples() {
    let dc = channels(5, 3);
    let quant = crate::jpeg::quantization::RawDct8Quantization::from_source_shift(
        std::array::from_fn(|c| std::array::from_fn(|i| 1 + ((i * 97 + c * 11) % 65535) as i32)),
        0,
    )
    .unwrap()
    .prepare_modular_channels()
    .unwrap();
    for (planes, options, id) in [
        (dc.as_slice(), options_for_dc(3, 0).unwrap(), 1),
        (dc.as_slice(), options_for_dc(3, 1).unwrap(), 1),
        (dc.as_slice(), options_for_dc(2, 0).unwrap(), 1),
        (dc.as_slice(), options_for_ac_metadata(3, 0).unwrap(), 3),
        (quant.as_slice(), options_for_quant(3, 0).unwrap(), 4),
    ] {
        let before: Vec<_> = planes.iter().map(|c| c.samples.clone()).collect();
        let aligned = stream_bits(planes, &options, id, 0);
        for offset in [1, 7] {
            let unaligned = stream_bits(planes, &options, id, offset);
            assert!(unaligned[..usize::from(offset)].iter().all(|&b| b == 1));
            assert_eq!(&unaligned[usize::from(offset)..], aligned);
        }
        assert_eq!(
            planes.iter().map(|c| c.samples.clone()).collect::<Vec<_>>(),
            before
        );
    }
}

#[test]
fn invalid_tiers_stream_counts_and_fractions_fail_before_encoding() {
    for tier in [i8::MIN, -2, 10, i8::MAX] {
        assert!(options_for_dc(tier, 0).is_err());
        assert!(options_for_ac_metadata(tier, 0).is_err());
        assert!(options_for_quant(tier, 0).is_err());
        assert!(options_for_global_entropy(tier, 0).is_err());
    }
    for decode_speed in [5, u8::MAX] {
        assert!(options_for_dc(3, decode_speed).is_err());
        assert!(options_for_ac_metadata(3, decode_speed).is_err());
        assert!(options_for_quant(3, decode_speed).is_err());
        assert!(options_for_global_entropy(3, decode_speed).is_err());
    }
    let empty = StreamContext {
        num_streams: 0,
        ..StreamContext::default()
    };
    assert!(empty.options_for_dc(3, 0).is_err());
    assert!(empty.options_for_ac_metadata(3, 0).is_err());
    assert!(empty.options_for_quant(3, 0).is_err());
    for fraction in [-0.1, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let context = StreamContext {
            learning_fraction: fraction,
            ..StreamContext::default()
        };
        assert!(context.options_for_dc(3, 0).is_err());
        assert!(context.options_for_ac_metadata(3, 0).is_err());
        assert!(context.options_for_quant(3, 0).is_err());
    }
    for fraction in [1.01, 10.0] {
        let context = StreamContext {
            learning_fraction: fraction,
            ..StreamContext::default()
        };
        for options in [
            context.options_for_dc(3, 0),
            context.options_for_ac_metadata(3, 0),
            context.options_for_quant(3, 0),
        ] {
            assert!(options.unwrap().sampling.unwrap().nb_repeats <= 1.0);
        }
    }
}
