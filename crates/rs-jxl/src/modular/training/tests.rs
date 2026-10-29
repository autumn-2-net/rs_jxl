// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; the complete upstream notice is in docs/PORTING_KERNELS.md.

//! Shared/local LearnTree integration from frozen b866b3defec13494bfbaf705db93581cda118e37
//! enc_encoding.cc and enc_ma.cc. The caller supplies every transformed stream
//! in the learning domain; IDs and metadata channels are never renumbered.
//!
//! Source mode performs the geometric quantile pre-pass before opt-in constant bypass,
//! then per-channel random training with complete predictor-state traversal.
//! An explicit selected-sample limit is an error ceiling, not a sampling target.
//! By default there is no additional selected-sample count cap.
//! Selected samples enter compact columns directly, using source-sized tables
//! prepared at channel/stripe and wave-append boundaries. Source learning
//! transfers those columns directly to ma_train::train_prepared; only legacy
//! learning and test oracles use raw Sample rows.
//! Local sample charging is unnecessary when the checked visited-pixel upper
//! bound fits the local limit. Shared ceilings still charge each selected pixel;
//! tighter local ceilings retain the original per-sample failure accounting.
//! Optional sparse traversal preserves causal state; optional stripes use the
//! source sixteen-row learning-only WP warm-up. Both are disabled by default.
//! Legacy mode retains collect_samples' explicit per-stream midpoint target.
//! Its local and shared selected-sample budgets are reserved before collector
//! allocation/traversal, using encode::legacy_sample_count's checked target.
//! Source gathering dispatches the exact Weighted/Gradient candidate pair once,
//! retaining generic candidate order for all other sets and unchanged WP state.
//! Serial and channel/wave gathering share source no-edge row segmentation:
//! y > 1, width > 8, no configured reference slots, and x in 2..width-2.
//! Safe row slices replace per-pixel edge tests only in that interior. Borders,
//! narrow planes and reference-enabled rows keep the general neighbor loader.
//! Source AddSample narrows the wide predictor difference to i32 before token
//! quantization. This is not the encoder's multiplier path: that path must do
//! wide subtraction/division before narrowing its quotient for PackSigned.
//!
//! A zero source fraction is LearnTree's no-samples case: predictor zero in the
//! configured candidate order, plus proven constant leaves when opted in. PredefinedTree's
//! WPFixedDC/GradientFixedDC trees are separate tree-kind choices requiring a
//! bit depth; zero learning fraction alone does not select those source trees.

use super::super::sample::SourcePolicy;
use super::*;

fn channel(width: usize, height: usize, offset: i32) -> Channel {
    Channel {
        width,
        height,
        hshift: 0,
        vshift: 0,
        samples: (0..width * height)
            .map(|i| ((i * 71 + i / width * 13) % 251) as i32 + offset)
            .collect(),
    }
}
fn options() -> ModularOptions {
    ModularOptions {
        sampling: Some(SamplingPolicy::from_source(&SourcePolicy::default()).unwrap()),
        max_tree_leaves: ma_train::MAX_LEAVES,
        ..ModularOptions::default()
    }
}
fn stream(channels: &[Channel], stream_id: usize) -> TrainingStream<'_> {
    TrainingStream {
        channels,
        stream_id,
        wp: weighted::Header::default(),
    }
}

fn gather_source_raw(
    streams: &[&TrainingStream<'_>],
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    variable_channels: usize,
    sample_limit: usize,
) -> Result<(Vec<Sample>, SamplingStats)> {
    let mut samples = Vec::new();
    let stats = visit_source_samples(
        streams,
        policy,
        predictors,
        variable_channels,
        sample_limit,
        None,
        |sample| {
            samples.push(sample);
            Ok(())
        },
    )?;
    Ok((samples, stats))
}

// Independent oracle: the original generic visit-based serial gather.
// Neither its neighbor traversal nor its sampling loop uses the new kernel.
fn visit_source_samples_generic(
    streams: &[&TrainingStream<'_>],
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    variable_channels: usize,
    sample_limit: usize,
    budget: Option<&SampleBudget>,
    mut collect: impl FnMut(Sample) -> Result<()>,
) -> Result<SamplingStats> {
    let first_pixels = match streams.first() {
        Some(first) => prefix_population(first, channel_prefix(first, policy, variable_channels))?,
        None => 0,
    };
    let mut stats = SamplingStats {
        source_total_pixels: first_pixels.max(1),
        ..SamplingStats::default()
    };
    let use_wp = predictors.contains(&Predictor::Weighted) || policy.properties.contains(&15);
    for stream in streams {
        let end = channel_prefix(stream, policy, variable_channels);
        let pixels = prefix_population(stream, end)?;
        stats.visited_pixels += pixels;
        stats.source_total_pixels += pixels;
        if policy.nb_repeats == 0.0 {
            continue;
        }
        let mut samplers = stream.channels[..end]
            .iter()
            .map(|c| ChannelSampler::new(c.width, c.height, policy.nb_repeats))
            .collect::<Result<Vec<_>>>()?;
        encode::visit(
            &stream.channels[..end],
            &stream.wp,
            use_wp,
            policy.previous_channels,
            stream.stream_id,
            |channel, _, neighbors, wp, properties, value| {
                if !samplers[channel]
                    .next()
                    .expect("generic visitor sample draw")
                {
                    return Ok(());
                }
                if stats.selected_samples == sample_limit as u64 {
                    return Err(Error::Limit("source MA selected-sample budget exceeded"));
                }
                if let Some(budget) = budget {
                    budget.consume(1)?;
                }
                collect(Sample {
                    properties: properties.to_vec(),
                    residuals: sample_residuals::<false>(predictors, neighbors, wp, value)?,
                    weight: 1,
                })?;
                stats.selected_samples += 1;
                Ok(())
            },
        )?;
        assert!(samplers.iter().all(|sampler| sampler.len() == 0));
    }
    Ok(stats)
}

fn assert_sample_sequences(actual: &[Sample], expected: &[Sample]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.properties, expected.properties);
        assert_eq!(actual.residuals, expected.residuals);
        assert_eq!(actual.weight, expected.weight);
    }
}

fn deep_channel(width: usize, height: usize) -> Channel {
    let mut plane = channel(width, height, 0);
    let levels = [
        i32::MIN,
        i32::MAX,
        -1,
        0,
        1,
        255,
        1023,
        65535,
        -65535,
        (1 << 23) - 1,
        -(1 << 23),
    ];
    for (i, value) in plane.samples.iter_mut().enumerate() {
        *value = levels[(i * 7 + i / width * 13) % levels.len()];
    }
    plane
}

fn gather_quantization(
    streams: &[&TrainingStream<'_>],
    policy: &SamplingPolicy,
) -> PropertyQuantization {
    let views: Vec<_> = streams
        .iter()
        .flat_map(|stream| {
            stream
                .channels
                .iter()
                .enumerate()
                .map(move |(channel, plane)| ChannelView {
                    channel: channel as u32,
                    group: stream.stream_id as u32,
                    width: plane.width,
                    height: plane.height,
                    stride: plane.width,
                    pixels: &plane.samples,
                    is_meta: plane.hshift < 0 || plane.vshift < 0,
                })
        })
        .collect();
    sample::collect_pixel_samples(&views, policy)
        .unwrap()
        .quantization(&policy.properties, policy.max_property_values)
        .unwrap()
}

// Compare the learner's complete objective, including every predictor's
// extra-bit count, rather than the collector's representative raw payload.
fn weighted_population(
    samples: &[Sample],
    quantization: &PropertyQuantization,
) -> std::collections::BTreeMap<Vec<u32>, u64> {
    let config = crate::entropy::hybrid_uint::HybridUintConfig::new(4, 1, 2).unwrap();
    let mut population = std::collections::BTreeMap::new();
    for sample in samples {
        let mut key: Vec<_> = quantization
            .properties
            .iter()
            .zip(&quantization.thresholds)
            .map(|(&property, thresholds)| {
                let value = sample.properties[property];
                let value = if property < 2 {
                    value
                } else {
                    value.clamp(-511, 511)
                };
                thresholds.partition_point(|&t| value > t) as u32
            })
            .collect();
        for &residual in &sample.residuals {
            let encoded = config.encode(super::super::tree::pack_signed(residual));
            key.extend([encoded.token, encoded.nbits]);
        }
        *population.entry(key).or_default() += u64::from(sample.weight);
    }
    population
}

#[test]
fn prepared_source_handoff_preserves_schema_constants_and_selected_budget() {
    let mut variable = channel(64, 64, 0);
    for (i, value) in variable.samples.iter_mut().enumerate() {
        *value = if i % 64 < 32 { 0 } else { 4 };
    }
    let channels = [
        variable,
        Channel {
            samples: vec![37; 4096],
            ..channel(64, 64, 0)
        },
    ];
    let streams = [stream(&channels, 917)];
    let ordered = [&streams[0]];
    let views: Vec<_> = channels
        .iter()
        .enumerate()
        .map(|(channel, plane)| ChannelView {
            channel: channel as u32,
            group: 917,
            width: plane.width,
            height: plane.height,
            stride: plane.width,
            pixels: &plane.samples,
            is_meta: false,
        })
        .collect();
    let signature = |tree: &Tree| {
        tree.tokens()
            .iter()
            .map(|token| (token.context, token.value))
            .collect::<Vec<_>>()
    };
    execution::serial(|| {
        for fraction in [0.0, 0.125, 1.0] {
            for constants in [false, true] {
                let mut options = options();
                options.predictor = Predictor::Zero;
                options.max_tree_leaves = 64;
                options.source_constant_channels = constants;
                let policy = options.sampling.as_mut().unwrap();
                policy.nb_repeats = fraction;
                policy.node_threshold = 0.0;
                policy.properties = vec![0, 1, 3];
                let policy = options.sampling.as_ref().unwrap();
                let population = sample::collect_pixel_samples(&views, policy).unwrap();
                let quantization = population
                    .quantization(&policy.properties, policy.max_property_values)
                    .unwrap();
                let variable_channels = if constants { 1 } else { 2 };
                let (rows, stats) = gather_source(
                    &ordered,
                    policy,
                    &[Predictor::Zero],
                    variable_channels,
                    usize::MAX,
                    &quantization,
                    None,
                )
                .unwrap();
                let selected = stats.selected_samples as usize;
                let mut train = encode::train_options(&options);
                train.parallel = false;
                train.require_complete_search = true;
                train.static_property_range = Some([[0, 2], [917, 918]]);
                if constants {
                    train.constant_channels = ma_train::find_constant_channels(
                        &views,
                        train.static_property_range.unwrap(),
                        policy.max_channel_size,
                    )
                    .unwrap();
                }
                policy
                    .apply_to_train(&population, stats, &mut train)
                    .unwrap();
                let expected = ma_train::train(&rows, &train).unwrap();
                let gather = GatherOptions {
                    workers: 1,
                    sparse: true,
                    ..GatherOptions::default()
                };
                let shared = SampleBudget::new(selected);
                let prepared = SourceGather {
                    streams: &ordered,
                    policy,
                    predictors: &train.predictors,
                    variable_channels,
                    sample_limit: selected,
                    quantization: &quantization,
                    budget: Some(&shared),
                }
                .collect_prepared(&gather)
                .unwrap();
                assert_eq!(prepared.stats, stats);
                assert_eq!(prepared.samples.total, stats.selected_samples);
                assert_eq!(prepared.samples.predictors, train.predictors);
                assert_eq!(
                    Some(&prepared.samples.quantization),
                    train.property_quantization.as_ref()
                );
                assert_eq!(
                    prepared
                        .samples
                        .counts
                        .iter()
                        .map(|&n| u64::from(n))
                        .sum::<u64>(),
                    stats.selected_samples
                );
                let direct = ma_train::train_prepared(prepared.samples, &train).unwrap();
                assert_eq!(signature(&direct), signature(&expected));
                assert_eq!(shared.remaining.load(Ordering::Relaxed), 0);
                for selected_gather in [None, Some(&gather)] {
                    let shared = SampleBudget::new(selected);
                    let actual = learn_with_budget_and_gather(
                        &streams,
                        &options,
                        Some(&shared),
                        false,
                        selected_gather,
                    )
                    .unwrap();
                    assert_eq!(signature(&actual), signature(&expected));
                    assert_eq!(shared.remaining.load(Ordering::Relaxed), 0);
                }
            }
        }
    });
}

#[test]
fn sparse_gather_preserves_generic_samples_with_references_and_wp_presets() {
    let channels = [deep_channel(33, 129), channel(33, 129, -41)];
    let mut input = stream(&channels, 917);
    let mut policy = options().sampling.unwrap();
    policy.previous_channels = 2;
    policy.properties = (0..24).collect();
    for mode in [Predictor::Best, Predictor::Variable, Predictor::Gradient] {
        let predictors = encode::predictors(mode);
        for fraction in [0.0, 0.125, 0.25, 1.0] {
            policy.nb_repeats = fraction;
            for preset in 0..5 {
                input.wp = weighted::Header::preset(preset).unwrap();
                let mut expected = Vec::new();
                visit_source_samples_generic(
                    &[&input],
                    &policy,
                    &predictors,
                    2,
                    usize::MAX,
                    None,
                    |sample| {
                        expected.push(sample);
                        Ok(())
                    },
                )
                .unwrap();
                let mut actual = Vec::new();
                let mut charges = 0;
                for (channel, plane) in channels.iter().enumerate().take(2) {
                    visit_source_range(
                        &input,
                        channel,
                        &policy,
                        &predictors,
                        0..plane.height,
                        true,
                        || {
                            charges += 1;
                            Ok(())
                        },
                        |properties, residuals| {
                            actual.push(Sample {
                                properties: properties.to_vec(),
                                residuals: residuals.to_vec(),
                                weight: 1,
                            });
                            Ok(())
                        },
                    )
                    .unwrap();
                }
                assert_eq!(charges, expected.len());
                assert_sample_sequences(&actual, &expected);
            }
        }
    }
}

// Independent full-property traversal of the source's warm-up window.
// It does not call visit_source_range/row or skip unselected property work.
fn stripe_reference(
    input: &TrainingStream<'_>,
    policy: &SamplingPolicy,
    predictors: &[Predictor],
    rows: std::ops::Range<usize>,
) -> Vec<Sample> {
    let channel = &input.channels[0];
    let warm = rows.start.saturating_sub(16);
    let mut sampler =
        ChannelSampler::new(channel.width, channel.height, policy.nb_repeats).unwrap();
    for _ in 0..warm * channel.width {
        sampler.next().unwrap();
    }
    let mut wp = weighted::State::new(input.wp, channel.width).unwrap();
    let mut properties = Properties::default();
    let mut result = Vec::new();
    for y in warm..rows.end {
        for x in 0..channel.width {
            let n = Neighbors::from_plane(&channel.samples, channel.width, channel.width, x, y)
                .unwrap();
            let (prediction, error) = wp
                .predict(x, y - warm, n.top, n.left, n.topright, n.topleft, n.toptop)
                .unwrap();
            let props = properties.calculate(n, x, y, 0, input.stream_id, error);
            let value = channel.samples[y * channel.width + x];
            if sampler.next().unwrap() && y >= rows.start {
                result.push(Sample {
                    properties: props.to_vec(),
                    residuals: sample_residuals::<false>(predictors, n, prediction, value).unwrap(),
                    weight: 1,
                });
            }
            wp.update(value, x, y - warm).unwrap();
        }
    }
    result
}

#[test]
fn gather_stripe_warmup_matches_source_window_and_excludes_budget_charges() {
    let channels = [deep_channel(17, 257)];
    let mut input = stream(&channels, 71);
    let mut policy = options().sampling.unwrap();
    policy.properties = (0..16).collect();
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    for preset in 0..5 {
        input.wp = weighted::Header::preset(preset).unwrap();
        for fraction in [0.0, 0.125, 1.0] {
            policy.nb_repeats = fraction;
            for begin in [0, 1, 16, 17, 33, 34, 128] {
                let rows = begin..begin + 31;
                let expected = stripe_reference(&input, &policy, &predictors, rows.clone());
                for sparse in [false, true] {
                    let budget = SampleBudget::new(expected.len());
                    let mut actual = Vec::new();
                    visit_source_range(
                        &input,
                        0,
                        &policy,
                        &predictors,
                        rows.clone(),
                        sparse,
                        || budget.consume(1),
                        |properties, residuals| {
                            actual.push(Sample {
                                properties: properties.to_vec(),
                                residuals: residuals.to_vec(),
                                weight: 1,
                            });
                            Ok(())
                        },
                    )
                    .unwrap();
                    assert_sample_sequences(&actual, &expected);
                    assert_eq!(budget.remaining.load(Ordering::Relaxed), 0);
                }
            }
        }
    }
    policy.nb_repeats = 1.0;
    let budget = SampleBudget::new(1);
    let mut collected = 0;
    assert!(matches!(
        visit_source_range(
            &input,
            0,
            &policy,
            &predictors,
            128..159,
            true,
            || budget.consume(1),
            |_, _| {
                collected += 1;
                Ok(())
            }
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(collected, 1);
    let mut called = false;
    assert!(
        visit_source_range(
            &input,
            0,
            &policy,
            &predictors,
            250..258,
            true,
            || {
                called = true;
                Ok(())
            },
            |_, _| Ok(())
        )
        .is_err()
    );
    assert!(!called);
}

#[test]
fn stripe_channel_wave_and_null_pool_paths_preserve_population_and_limits() {
    let channels = [channel(512, 513, -17)];
    let input = stream(&channels, 41);
    let streams = [&input];
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 0.125;
    policy.properties = vec![0, 1, 2, 3, 8, 9, 15];
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    let quantization = gather_quantization(&streams, &policy);
    let mut gather = GatherOptions {
        stripes: 4,
        sparse: true,
        workers: 2,
        ..GatherOptions::default()
    };
    let raw: Vec<_> = gather
        .rows(512, 513)
        .unwrap()
        .into_iter()
        .flat_map(|rows| stripe_reference(&input, &policy, &predictors, rows))
        .collect();
    let expected = weighted_population(&raw, &quantization);
    let selected = raw.len();
    assert_eq!(
        selected,
        ChannelSampler::new(512, 513, policy.nb_repeats)
            .unwrap()
            .filter(|&s| s)
            .count()
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for mode in [sample::GatherMode::Channels, sample::GatherMode::Waves] {
            gather.mode = mode;
            gather.batch_budget_bytes = if mode == sample::GatherMode::Waves {
                900_000
            } else {
                1 << 30
            };
            let run =
                |limit: usize, shared: &SampleBudget, serial: bool| -> Result<GatheredSamples> {
                    let source = SourceGather {
                        streams: &streams,
                        policy: &policy,
                        predictors: &predictors,
                        variable_channels: 1,
                        sample_limit: limit,
                        quantization: &quantization,
                        budget: Some(shared),
                    };
                    if serial {
                        execution::serial(|| source.collect(&gather))
                    } else {
                        source.collect(&gather)
                    }
                };
            for serial in [false, true] {
                let shared = SampleBudget::new(selected);
                let actual = run(selected, &shared, serial).unwrap();
                assert_eq!(actual.stats.visited_pixels, 512 * 513);
                assert_eq!(actual.stats.selected_samples, selected as u64);
                assert_eq!(
                    weighted_population(&actual.samples, &quantization),
                    expected
                );
                assert_eq!(shared.remaining.load(Ordering::Relaxed), 0);
                assert_eq!(actual.parallel_waves > 0, !serial);
                let local_limit_budget = SampleBudget::new(selected);
                assert!(matches!(
                    run(selected - 1, &local_limit_budget, serial),
                    Err(Error::Limit(_))
                ));
                let shared_limit_budget = SampleBudget::new(selected - 1);
                assert!(matches!(
                    run(selected, &shared_limit_budget, serial),
                    Err(Error::Limit(_))
                ));
            }
        }
    });
}

#[test]
fn no_edge_rows_match_generic_neighbors_and_exact_source_guards() {
    for width in [0, 1, 2, 3, 7, 8, 9, 10, 17, 33] {
        let plane = deep_channel(width, 7);
        for y in 0..plane.height {
            for previous in [0, 1, 2] {
                assert_eq!(
                    has_no_edge_interior(width, y, previous),
                    width >= 9 && y >= 2 && previous == 0
                );
            }
            let mut actual = Vec::new();
            let mut pixel = |x, neighbors| {
                actual.push((x, neighbors));
                Ok(())
            };
            if has_no_edge_interior(width, y, 0) {
                visit_source_row::<false>(&plane, y, 0..2, &mut pixel).unwrap();
                visit_source_row::<true>(&plane, y, 2..width - 2, &mut pixel).unwrap();
                visit_source_row::<false>(&plane, y, width - 2..width, &mut pixel).unwrap();
            } else {
                visit_source_row::<false>(&plane, y, 0..width, &mut pixel).unwrap();
            }
            assert_eq!(actual.len(), width);
            for (x, &(position, neighbors)) in actual.iter().enumerate() {
                assert_eq!(position, x);
                assert_eq!(
                    neighbors,
                    Neighbors::from_plane(&plane.samples, width, width, x, y).unwrap()
                );
            }
        }
    }
}

#[test]
fn no_edge_gather_matches_generic_borders_narrow_planes_depths_and_predictors() {
    let mut policy = options().sampling.unwrap();
    policy.max_channel_size = 1024;
    policy.nb_repeats = 0.125;
    for (width, height) in [
        (0, 4),
        (4, 0),
        (1, 65),
        (2, 65),
        (3, 5),
        (8, 5),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 65),
        (17, 7),
        (33, 67),
    ] {
        let channels = [deep_channel(width, height)];
        let mut input = stream(&channels, 1_000_000);
        input.wp = weighted::Header::preset(4).unwrap();
        for (mode, wp_property) in [
            (Predictor::Best, true),
            (Predictor::Variable, true),
            (Predictor::Weighted, false),
            (Predictor::Gradient, false),
            (Predictor::Gradient, true),
            (Predictor::Zero, false),
        ] {
            let predictors = encode::predictors(mode);
            policy.properties = (0..if wp_property { 16 } else { 15 }).collect();
            let mut expected = Vec::new();
            let expected_stats = visit_source_samples_generic(
                &[&input],
                &policy,
                &predictors,
                1,
                20_000,
                None,
                |sample| {
                    expected.push(sample);
                    Ok(())
                },
            )
            .unwrap();
            let (actual, stats) =
                gather_source_raw(&[&input], &policy, &predictors, 1, 20_000).unwrap();
            assert_eq!(stats, expected_stats);
            assert_sample_sequences(&actual, &expected);
            let mut channel_samples = Vec::new();
            visit_source_channel(&input, 0, &policy, &predictors, |sample| {
                channel_samples.push(sample);
                Ok(())
            })
            .unwrap();
            assert_sample_sequences(&channel_samples, &expected);
            let mut without_interior = Vec::new();
            visit_source_channel_impl::<false, false>(
                &input,
                0,
                &policy,
                &predictors,
                || Ok(()),
                |sample| {
                    without_interior.push(sample);
                    Ok(())
                },
            )
            .unwrap();
            assert_sample_sequences(&without_interior, &expected);
        }
    }
}

#[test]
fn no_edge_gather_preserves_rng_and_wp_through_rejected_pixels_and_row_transitions() {
    let channels = [deep_channel(17, 257)];
    let mut input = stream(&channels, 923);
    let mut policy = options().sampling.unwrap();
    policy.max_channel_size = 1024;
    policy.properties = (0..16).collect();
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    for preset in 0..=4 {
        input.wp = weighted::Header::preset(preset).unwrap();
        for fraction in [0.0, 0.03, 0.5, 1.0] {
            policy.nb_repeats = fraction;
            let mut expected = Vec::new();
            let expected_stats = visit_source_samples_generic(
                &[&input],
                &policy,
                &predictors,
                1,
                20_000,
                None,
                |sample| {
                    expected.push(sample);
                    Ok(())
                },
            )
            .unwrap();
            let (actual, stats) =
                gather_source_raw(&[&input], &policy, &predictors, 1, 20_000).unwrap();
            assert_eq!(stats, expected_stats);
            assert_sample_sequences(&actual, &expected);
            let selected: Vec<_> = if fraction == 0.0 {
                Vec::new()
            } else {
                ChannelSampler::new(17, 257, fraction)
                    .unwrap()
                    .enumerate()
                    .filter_map(|(position, selected)| selected.then_some(position))
                    .collect()
            };
            assert_eq!(actual.len(), selected.len());
            for (sample, position) in actual.iter().zip(&selected) {
                assert_eq!(sample.properties[2], (position / 17) as i32);
                assert_eq!(sample.properties[3], (position % 17) as i32);
            }
            if fraction > 0.0 && fraction < 1.0 {
                assert!(actual.len() < 17 * 257);
                // These samples expose state carried across both interior
                // transitions and the row reset after discarded pixels.
                for x in [0, 1, 2, 14, 15, 16] {
                    assert!(
                        actual
                            .iter()
                            .any(|s| s.properties[3] == x && s.properties[2] > 2)
                    );
                }
            }
        }
    }
}

#[test]
fn no_edge_reference_slot_guard_preserves_missing_and_mismatched_references() {
    let mut channels = [
        deep_channel(17, 97),
        deep_channel(9, 97),
        deep_channel(17, 97),
        channel(17, 97, 31),
        channel(17, 97, -9),
    ];
    channels[2].hshift = 1;
    channels[2].vshift = 1;
    let input = stream(&channels, 329);
    let predictors = encode::predictors(Predictor::Variable);
    let mut policy = options().sampling.unwrap();
    policy.max_channel_size = 1024;
    policy.nb_repeats = 0.125;
    for previous in [0, 1, 2] {
        policy.previous_channels = previous;
        policy.properties = (0..16 + 4 * previous).collect();
        assert_eq!(has_no_edge_interior(17, 2, previous), previous == 0);
        let mut expected = Vec::new();
        let expected_stats = visit_source_samples_generic(
            &[&input],
            &policy,
            &predictors,
            channels.len(),
            20_000,
            None,
            |sample| {
                expected.push(sample);
                Ok(())
            },
        )
        .unwrap();
        let (actual, stats) =
            gather_source_raw(&[&input], &policy, &predictors, channels.len(), 20_000).unwrap();
        assert_eq!(stats, expected_stats);
        assert_sample_sequences(&actual, &expected);
        let mut channel_samples = Vec::new();
        for channel in 0..channels.len() {
            visit_source_channel(&input, channel, &policy, &predictors, |sample| {
                channel_samples.push(sample);
                Ok(())
            })
            .unwrap();
        }
        assert_sample_sequences(&channel_samples, &expected);
        if previous != 0 {
            assert!(
                actual
                    .iter()
                    .filter(|s| s.properties[0] < 3)
                    .all(|s| s.properties[16..].iter().all(|&value| value == 0))
            );
            assert!(
                actual
                    .iter()
                    .filter(|s| s.properties[0] >= 3)
                    .any(|s| s.properties[16..].iter().any(|&value| value != 0))
            );
        }
    }
}

#[test]
fn no_edge_serial_budget_and_callback_failures_match_generic_at_segment_boundaries() {
    let channels = [deep_channel(17, 257), channel(17, 257, 13)];
    let first = stream(&channels, 23);
    let second = stream(&channels, 991);
    let streams = [&first, &second];
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    let mut policy = options().sampling.unwrap();
    policy.max_channel_size = 1024;
    policy.properties = (0..16).collect();
    for fraction in [0.0, 0.125, 1.0] {
        policy.nb_repeats = fraction;
        let total = if fraction == 0.0 {
            0
        } else {
            4 * ChannelSampler::new(17, 257, fraction)
                .unwrap()
                .filter(|&v| v)
                .count()
        };
        let run = |generic, local_limit, shared_limit, fail_after: Option<usize>| {
            let budget = SampleBudget::new(shared_limit);
            let mut samples = Vec::new();
            let mut collect = |sample| {
                if fail_after == Some(samples.len()) {
                    return Err(Error::Encoding("injected gather callback failure".into()));
                }
                samples.push(sample);
                Ok(())
            };
            let result = if generic {
                visit_source_samples_generic(
                    &streams,
                    &policy,
                    &predictors,
                    2,
                    local_limit,
                    Some(&budget),
                    &mut collect,
                )
            } else {
                visit_source_samples(
                    &streams,
                    &policy,
                    &predictors,
                    2,
                    local_limit,
                    Some(&budget),
                    &mut collect,
                )
            };
            (
                samples,
                result.map_err(|e| e.to_string()),
                budget.remaining.load(Ordering::Relaxed),
            )
        };
        for (local, shared, fail_after) in [
            (0, total, None),
            (1, total, None),
            (2, total, None),
            (15, total, None),
            (17, total, None),
            (36, total, None),
            (37, total, None),
            (49, total, None),
            (total, 37, None),
            (total, total.saturating_sub(1), None),
            (total, total, None),
            (total, total, Some(36)),
            (total, total, Some(49)),
        ] {
            let (expected, expected_result, expected_remaining) =
                run(true, local, shared, fail_after);
            let (actual, result, remaining) = run(false, local, shared, fail_after);
            assert_sample_sequences(&actual, &expected);
            assert_eq!(result, expected_result);
            assert_eq!(remaining, expected_remaining);
        }
    }
}

#[test]
fn no_edge_parallel_gather_preserves_population_and_local_shared_limits() {
    let channels = [deep_channel(256, 256), channel(256, 256, 19)];
    let input = stream(&channels, 51);
    let streams = [&input];
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 0.125;
    policy.properties = vec![0, 1, 9, 15];
    let quantization = gather_quantization(&streams, &policy);
    let mut reference = Vec::new();
    let stats = visit_source_samples_generic(
        &streams,
        &policy,
        &predictors,
        2,
        usize::MAX,
        None,
        |sample| {
            reference.push(sample);
            Ok(())
        },
    )
    .unwrap();
    let expected = weighted_population(&reference, &quantization);
    let selected = stats.selected_samples as usize;
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    workers.install(|| {
        for mode in [sample::GatherMode::Channels, sample::GatherMode::Waves] {
            let gather = GatherOptions {
                mode,
                workers: 2,
                ..GatherOptions::default()
            };
            let budget = SampleBudget::new(selected);
            let source = SourceGather {
                streams: &streams,
                policy: &policy,
                predictors: &predictors,
                variable_channels: 2,
                sample_limit: selected,
                quantization: &quantization,
                budget: Some(&budget),
            };
            let actual = source.collect(&gather).unwrap();
            assert!(actual.parallel_waves > 0);
            assert_eq!(actual.stats, stats);
            assert_eq!(
                weighted_population(&actual.samples, &quantization),
                expected
            );
            assert_eq!(budget.remaining.load(Ordering::Relaxed), 0);
            let budget = SampleBudget::new(selected);
            let serial_source = SourceGather {
                budget: Some(&budget),
                ..source
            };
            let serial = execution::serial(|| serial_source.collect(&gather)).unwrap();
            assert_eq!(serial.parallel_waves, 0);
            assert_eq!(serial.stats, stats);
            assert_eq!(
                weighted_population(&serial.samples, &quantization),
                expected
            );
            assert_eq!(budget.remaining.load(Ordering::Relaxed), 0);
            for (local, shared) in [(selected - 1, selected), (selected, selected - 1)] {
                let budget = SampleBudget::new(shared);
                let limited = SourceGather {
                    sample_limit: local,
                    budget: Some(&budget),
                    ..serial_source
                };
                assert!(matches!(limited.collect(&gather), Err(Error::Limit(_))));
                assert_eq!(
                    budget.remaining.load(Ordering::Relaxed),
                    shared - (selected - 1)
                );
            }
        }
    });
}

#[test]
fn gather_concat_only_finishes_one_parallel_wave_and_keeps_sample_budgets() {
    let plane = channel(256, 256, 7);
    let channels = [plane.clone(), plane];
    let input = stream(&channels, 23);
    let streams = [&input];
    let predictors = [Predictor::Zero];
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 0.125;
    policy.previous_channels = 0;
    policy.properties = vec![2, 3];
    // Exact x/y bins make each local batch unique, but identical channels
    // still have duplicate quantized keys across the two worker batches.
    let quantization = PropertyQuantization {
        properties: policy.properties.clone(),
        thresholds: vec![(0..255).collect(), (0..255).collect()],
    };
    let selected = 2 * ChannelSampler::new(256, 256, policy.nb_repeats)
        .unwrap()
        .filter(|&selected| selected)
        .count();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for cache_hashes in [false, true] {
            for (serial, mode, batch_budget_bytes) in [
                (false, sample::GatherMode::Channels, 1 << 30),
                (true, sample::GatherMode::Channels, 1 << 30),
                (false, sample::GatherMode::Waves, 250_000),
            ] {
                let run = |concat_batches| {
                    let shared = SampleBudget::new(selected + 7);
                    let source = SourceGather {
                        streams: &streams,
                        policy: &policy,
                        predictors: &predictors,
                        variable_channels: 2,
                        sample_limit: 2 * 256 * 256,
                        quantization: &quantization,
                        budget: Some(&shared),
                    };
                    let gather = GatherOptions {
                        mode,
                        workers: 2,
                        batch_budget_bytes,
                        cache_hashes,
                        concat_batches,
                        ..GatherOptions::default()
                    };
                    let result = if serial {
                        execution::serial(|| source.collect(&gather))
                    } else {
                        source.collect(&gather)
                    }
                    .unwrap();
                    assert_eq!(shared.remaining.load(Ordering::Relaxed), 7);
                    result
                };
                let expected = run(false);
                let actual = run(true);
                assert_eq!(actual.stats, expected.stats);
                assert_eq!(actual.stats.selected_samples, selected as u64);
                assert_eq!(actual.waves, expected.waves);
                assert_eq!(actual.parallel_waves, expected.parallel_waves);
                if !serial && mode == sample::GatherMode::Channels {
                    assert_eq!(actual.parallel_waves, 1);
                    assert_eq!(actual.samples.len(), selected);
                    assert!(expected.samples.len() < actual.samples.len());
                    assert_eq!(
                        weighted_population(&actual.samples, &quantization),
                        weighted_population(&expected.samples, &quantization)
                    );
                } else {
                    if mode == sample::GatherMode::Waves {
                        assert_eq!(actual.waves, 2);
                    }
                    assert_sample_sequences(&actual.samples, &expected.samples);
                }
            }
        }
    });
}

#[test]
fn gather_local_budget_upper_bound_preserves_sample_sequence() {
    let channels = [deep_channel(256, 256), channel(256, 256, 19)];
    let input = stream(&channels, 51);
    let streams = [&input];
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 0.125;
    policy.properties = vec![0, 1, 9, 15];
    let quantization = gather_quantization(&streams, &policy);
    let selected = 2 * ChannelSampler::new(256, 256, policy.nb_repeats)
        .unwrap()
        .filter(|&selected| selected)
        .count();
    let visited = 2 * 256 * 256;
    assert!(selected > 0 && selected < visited);
    let source = SourceGather {
        streams: &streams,
        policy: &policy,
        predictors: &predictors,
        variable_channels: 2,
        sample_limit: selected,
        quantization: &quantization,
        budget: None,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        let gather = GatherOptions {
            workers: 2,
            ..GatherOptions::default()
        };
        for serial in [false, true] {
            let run = |source: &SourceGather<'_, '_>| {
                if serial {
                    execution::serial(|| source.collect(&gather))
                } else {
                    source.collect(&gather)
                }
            };
            // The tight but sufficient limit exercises the original CAS path.
            let expected = run(&source).unwrap();
            assert_eq!(expected.stats.selected_samples, selected as u64);
            assert_eq!(expected.stats.visited_pixels, visited as u64);
            assert_eq!(expected.parallel_waves > 0, !serial);
            for limit in [visited, usize::MAX] {
                let actual = run(&SourceGather {
                    sample_limit: limit,
                    ..source
                })
                .unwrap();
                assert_eq!(actual.stats, expected.stats);
                assert_eq!(actual.parallel_waves, expected.parallel_waves);
                assert_sample_sequences(&actual.samples, &expected.samples);
            }
            let shared = SampleBudget::new(selected + 11);
            let actual = run(&SourceGather {
                sample_limit: visited,
                budget: Some(&shared),
                ..source
            })
            .unwrap();
            assert_eq!(actual.stats, expected.stats);
            assert_sample_sequences(&actual.samples, &expected.samples);
            assert_eq!(shared.remaining.load(Ordering::Relaxed), 11);
        }
    });
}

#[test]
fn gather_local_budget_upper_bound_tight_limit_keeps_failure_debits() {
    let channels = [channel(32, 129, 7)];
    let input = stream(&channels, 23);
    let streams = [&input];
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 0.125;
    policy.properties = vec![0, 1, 9, 15];
    let quantization = gather_quantization(&streams, &policy);
    let selected = ChannelSampler::new(32, 129, policy.nb_repeats)
        .unwrap()
        .filter(|&selected| selected)
        .count();
    assert!(selected > 1 && selected < 32 * 129);
    let gather = GatherOptions {
        workers: 1,
        ..GatherOptions::default()
    };
    for limit in [0, 1, selected - 1] {
        let shared = SampleBudget::new(selected + 11);
        let source = SourceGather {
            streams: &streams,
            policy: &policy,
            predictors: &predictors,
            variable_channels: 1,
            sample_limit: limit,
            quantization: &quantization,
            budget: Some(&shared),
        };
        assert!(matches!(
            source.collect_prepared(&gather),
            Err(Error::Limit("grouped MA selected-sample budget exceeded"))
        ));
        // Local failure must occur before charging the next shared sample.
        assert_eq!(
            shared.remaining.load(Ordering::Relaxed),
            selected + 11 - limit
        );
    }
}

#[test]
fn gather_local_budget_upper_bound_still_charges_shared_samples() {
    let channels = [channel(32, 129, 7)];
    let input = stream(&channels, 23);
    let streams = [&input];
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 0.125;
    policy.properties = vec![0, 1, 9, 15];
    let quantization = gather_quantization(&streams, &policy);
    let selected = ChannelSampler::new(32, 129, policy.nb_repeats)
        .unwrap()
        .filter(|&selected| selected)
        .count();
    assert!(selected > 1 && selected < 32 * 129);
    let gather = GatherOptions {
        workers: 1,
        ..GatherOptions::default()
    };
    for local_limit in [selected, 32 * 129, usize::MAX] {
        for shared_limit in [0, 1, selected - 1, selected, selected + 11] {
            let shared = SampleBudget::new(shared_limit);
            let source = SourceGather {
                streams: &streams,
                policy: &policy,
                predictors: &predictors,
                variable_channels: 1,
                sample_limit: local_limit,
                quantization: &quantization,
                budget: Some(&shared),
            };
            let result = source.collect_prepared(&gather);
            if shared_limit < selected {
                assert!(matches!(
                    result,
                    Err(Error::Limit("grouped MA selected-sample budget exceeded"))
                ));
            } else {
                let actual = result.unwrap();
                assert_eq!(actual.stats.selected_samples, selected as u64);
                assert_eq!(actual.samples.total, selected as u64);
            }
            assert_eq!(
                shared.remaining.load(Ordering::Relaxed),
                shared_limit.saturating_sub(selected)
            );
        }
    }
}

#[test]
fn no_edge_rejects_invalid_plane_shapes_before_sampling_or_collection() {
    for channel in [
        Channel {
            samples: vec![0; 26],
            ..channel(9, 3, 0)
        },
        Channel {
            width: usize::MAX,
            height: 2,
            samples: Vec::new(),
            ..channel(0, 0, 0)
        },
    ] {
        let channels = [channel];
        let input = stream(&channels, 0);
        let policy = options().sampling.unwrap();
        assert!(
            visit_source_channel_impl::<true, true>(
                &input,
                0,
                &policy,
                &[Predictor::Weighted, Predictor::Gradient],
                || panic!("invalid shape reached sampling budget"),
                |_| panic!("invalid shape reached collector"),
            )
            .is_err()
        );
    }
    let channels = [
        Channel {
            samples: vec![0; 26],
            ..channel(9, 3, 0)
        },
        channel(9, 3, 0),
    ];
    let input = stream(&channels, 0);
    let mut policy = options().sampling.unwrap();
    policy.previous_channels = 1;
    assert!(
        visit_source_channel_impl::<false, true>(
            &input,
            1,
            &policy,
            &[Predictor::Gradient],
            || panic!("invalid reference reached sampling budget"),
            |_| panic!("invalid reference reached collector"),
        )
        .is_err()
    );
}

#[test]
fn predictor_pair_gate_and_narrowing_match_generic_predictions() {
    let pair = [Predictor::Weighted, Predictor::Gradient];
    assert!(uses_predictor_pair(&pair));
    for candidates in [
        vec![],
        vec![Predictor::Weighted],
        vec![Predictor::Gradient, Predictor::Weighted],
        vec![Predictor::Weighted, Predictor::Gradient, Predictor::Zero],
    ] {
        assert!(!uses_predictor_pair(&candidates));
    }
    let extremes = [i32::MIN, i32::MAX, -1, 0, 1];
    for left in extremes {
        for top in extremes {
            for topleft in extremes {
                let neighbors = Neighbors {
                    left,
                    top,
                    topleft,
                    leftleft: left,
                    toptop: top,
                    topright: top,
                    toprightright: top,
                };
                for wp in extremes {
                    for value in extremes {
                        assert_eq!(
                            sample_residuals::<true>(&pair, neighbors, wp, value).unwrap(),
                            sample_residuals::<false>(&pair, neighbors, wp, value).unwrap(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn serial_pair_gather_matches_generic_with_edges_references_and_rejected_samples() {
    let mut channels = Vec::new();
    let extremes = [i32::MIN, i32::MAX, 0, -1, 1, 7];
    for width in [0, 1, 2, 8, 9, 32] {
        let mut first = channel(width, 65, 0);
        for (i, value) in first.samples.iter_mut().enumerate() {
            *value = extremes[(i * 73 + i / width * 13) % extremes.len()];
        }
        channels.push(first);
        channels.push(channel(width, 65, 7));
    }
    let input = stream(&channels, 1_000_000);
    let predictors = [Predictor::Weighted, Predictor::Gradient];
    let mut policy = options().sampling.unwrap();
    for previous in [0, 2] {
        policy.previous_channels = previous;
        policy.properties = (0..16 + 4 * previous).collect();
        for repeats in [0.0, 0.125, 1.0] {
            policy.nb_repeats = repeats;
            let mut reference = Vec::new();
            let expected_stats = visit_source_samples_generic(
                &[&input],
                &policy,
                &predictors,
                channels.len(),
                20_000,
                None,
                |sample| {
                    reference.push(sample);
                    Ok(())
                },
            )
            .unwrap();
            let (actual, stats) =
                gather_source_raw(&[&input], &policy, &predictors, channels.len(), 20_000).unwrap();
            assert_eq!(stats, expected_stats);
            assert_eq!(actual.len(), reference.len());
            for (actual, expected) in actual.iter().zip(&reference) {
                assert_eq!(actual.properties, expected.properties);
                assert_eq!(actual.residuals, expected.residuals);
                assert_eq!(actual.weight, expected.weight);
            }
            if repeats == 0.125 {
                assert!(stats.selected_samples < stats.visited_pixels);
            }
        }
    }
}

#[test]
fn channel_gather_matches_serial_raster_with_references_and_wide_predictors() {
    let mut channels = vec![
        channel(32, 64, 0),
        channel(16, 32, 0),
        channel(32, 64, 7),
        channel(32, 64, -9),
        channel(1, 33, 0),
        channel(0, 4, 0),
    ];
    channels[1].hshift = 1;
    let extremes = [i32::MIN, i32::MAX, 0, -1, 1, 7];
    for (i, value) in channels[0].samples.iter_mut().enumerate() {
        *value = extremes[(i * 73 + i / 32 * 13) % extremes.len()];
    }
    let first = stream(&channels, 0);
    let second = stream(&channels, 1_000_000);
    let streams = [&first, &second];
    let mut policy = options().sampling.unwrap();
    policy.previous_channels = 2;
    policy.properties = (0..24).collect();
    policy.nb_repeats = 0.125;
    for mode in [Predictor::Variable, Predictor::Gradient, Predictor::Best] {
        let predictors = encode::predictors(mode);
        let mut reference = Vec::new();
        let stats = visit_source_samples_generic(
            &streams,
            &policy,
            &predictors,
            channels.len(),
            30_000,
            None,
            |sample| {
                reference.push(sample);
                Ok(())
            },
        )
        .unwrap();
        let mut actual = Vec::new();
        for stream in &streams {
            for c in 0..channels.len() {
                visit_source_channel(stream, c, &policy, &predictors, |sample| {
                    actual.push(sample);
                    Ok(())
                })
                .unwrap();
            }
        }
        assert_eq!(actual.len(), reference.len());
        assert!(stats.selected_samples < stats.visited_pixels);
        for (actual, expected) in actual.iter().zip(&reference) {
            assert_eq!(actual.properties, expected.properties);
            assert_eq!(actual.residuals, expected.residuals);
            assert_eq!(actual.weight, expected.weight);
        }
    }
}

#[test]
fn parallel_channels_and_waves_preserve_weighted_population_and_stats() {
    let channels: Vec<_> = (0..4).map(|c| channel(256, 256, c * 31)).collect();
    let first = stream(&channels, 0);
    let second = stream(&channels, 21);
    let streams = [&first, &second];
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 0.1;
    policy.previous_channels = 2;
    policy.properties = vec![0, 1, 15, 9, 19, 23];
    let predictors = encode::predictors(Predictor::Best);
    let quantization = gather_quantization(&streams, &policy);
    let (raw, reference_stats) =
        gather_source_raw(&streams, &policy, &predictors, 4, 100_000).unwrap();
    let reference = weighted_population(&raw, &quantization);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for mode in [sample::GatherMode::Channels, sample::GatherMode::Waves] {
            let gather = GatherOptions {
                mode,
                batch_budget_bytes: if mode == sample::GatherMode::Waves {
                    1 << 20
                } else {
                    1 << 30
                },
                ..GatherOptions::default()
            };
            let budget = SampleBudget::new(reference_stats.selected_samples as usize);
            let actual = SourceGather {
                streams: &streams,
                policy: &policy,
                predictors: &predictors,
                variable_channels: 4,
                sample_limit: reference_stats.selected_samples as usize,
                quantization: &quantization,
                budget: Some(&budget),
            }
            .collect(&gather)
            .unwrap();
            assert_eq!(actual.stats, reference_stats);
            assert_eq!(
                weighted_population(&actual.samples, &quantization),
                reference
            );
            assert_eq!(budget.remaining.load(Ordering::Relaxed), 0);
            assert_eq!(
                actual.parallel_waves,
                if mode == sample::GatherMode::Waves {
                    3
                } else {
                    1
                }
            );
            assert_eq!(actual.waves, actual.parallel_waves);
            assert_eq!(actual.serial_tasks, 0);
            let null_budget = SampleBudget::new(reference_stats.selected_samples as usize);
            let null = execution::serial(|| {
                SourceGather {
                    streams: &streams,
                    policy: &policy,
                    predictors: &predictors,
                    variable_channels: 4,
                    sample_limit: reference_stats.selected_samples as usize,
                    quantization: &quantization,
                    budget: Some(&null_budget),
                }
                .collect(&gather)
                .unwrap()
            });
            assert_eq!(null.parallel_waves, 0);
            assert_eq!(null.serial_tasks, 8);
            assert_eq!(null.stats, reference_stats);
            assert_eq!(weighted_population(&null.samples, &quantization), reference);
            assert_eq!(null_budget.remaining.load(Ordering::Relaxed), 0);
        }
    });
}

#[test]
fn wave_budget_fallbacks_and_shared_sample_limits_never_drop_pixels() {
    let channels = [Channel {
        samples: vec![7; 65_536],
        ..channel(256, 256, 0)
    }];
    let first = stream(&channels, 21);
    let second = stream(&channels, 22);
    let streams = [&first, &second];
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 1.0;
    policy.properties = vec![0];
    let quantization = gather_quantization(&streams, &policy);
    let predictors = [Predictor::Zero];
    let count = 2 * 65_536;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        let mut gather = GatherOptions {
            mode: sample::GatherMode::Waves,
            ..GatherOptions::default()
        };
        let mut input = SourceGather {
            streams: &streams,
            policy: &policy,
            predictors: &predictors,
            variable_channels: 1,
            sample_limit: count,
            quantization: &quantization,
            budget: None,
        };
        let reference = input.collect(&gather).unwrap();
        assert_eq!(reference.parallel_waves, 1);
        assert_eq!(reference.stats.selected_samples, count as u64);
        assert!(reference.samples.len() < 10);
        let budget = SampleBudget::new(count - 1);
        input.budget = Some(&budget);
        assert!(matches!(input.collect(&gather), Err(Error::Limit(_))));
        let null_budget = SampleBudget::new(count - 1);
        input.budget = Some(&null_budget);
        assert!(execution::serial(|| matches!(
            input.collect(&gather),
            Err(Error::Limit(_))
        )));
        input.budget = None;
        input.sample_limit = count - 1;
        assert!(matches!(input.collect(&gather), Err(Error::Limit(_))));
        assert!(execution::serial(|| matches!(
            input.collect(&gather),
            Err(Error::Limit(_))
        )));
        input.sample_limit = count;
        gather.batch_budget_bytes = 1;
        let serial = input.collect(&gather).unwrap();
        assert_eq!(serial.waves, 2);
        assert_eq!(serial.parallel_waves, 0);
        assert_eq!(serial.serial_tasks, 2);
        assert_eq!(serial.stats, reference.stats);
        assert_eq!(
            weighted_population(&serial.samples, &quantization),
            weighted_population(&reference.samples, &quantization)
        );
    });
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    single.install(|| {
        let actual = SourceGather {
            streams: &streams,
            policy: &policy,
            predictors: &predictors,
            variable_channels: 1,
            sample_limit: count,
            quantization: &quantization,
            budget: None,
        }
        .collect(&GatherOptions::default())
        .unwrap();
        assert_eq!(actual.parallel_waves, 0);
        assert_eq!(actual.serial_tasks, 2);
        assert_eq!(actual.stats.selected_samples, count as u64);
    });
}

#[test]
fn source_gather_opt_in_reaches_learning_and_preserves_group_routes_and_constants() {
    let mut plane = channel(256, 256, 0);
    for (i, value) in plane.samples.iter_mut().enumerate() {
        *value = i32::from(i % 256 >= 128);
    }
    let channels = [
        plane.clone(),
        plane,
        Channel {
            samples: vec![37; 65_536],
            ..channel(256, 256, 0)
        },
    ];
    let streams = [stream(&channels, 0), stream(&channels, 21)];
    let mut options = options();
    options.source_constant_channels = true;
    options.predictor = Predictor::Zero;
    options.max_tree_leaves = 64;
    let policy = options.sampling.as_mut().unwrap();
    policy.nb_repeats = 0.1;
    policy.properties = vec![0, 1, 3];
    policy.max_property_values = 256;
    policy.node_threshold = 0.0;
    let signature = |tree: &Tree| {
        tree.tokens()
            .iter()
            .map(|t| (t.context, t.value))
            .collect::<Vec<_>>()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for grouping in [
            None,
            Some(super::super::tree_groups::TreeGroupOptions {
                requested_shards: 2,
                min_work: 1,
                ..Default::default()
            }),
        ] {
            options.tree_groups = grouping;
            options.source_gather = None;
            let reference = learn(&streams, &options).unwrap();
            options.source_gather = Some(GatherOptions {
                mode: sample::GatherMode::Waves,
                batch_budget_bytes: 1 << 20,
                ..GatherOptions::default()
            });
            let actual = learn(&streams, &options).unwrap();
            let null = execution::serial(|| learn(&streams, &options).unwrap());
            assert!(actual.leaves() >= 3);
            assert_eq!(signature(&actual), signature(&reference));
            assert_eq!(signature(&null), signature(&reference));
            for id in [0, 21] {
                let (_, predictor, offset, multiplier) = actual.lookup(&[2, id, 0, 0]).unwrap();
                assert_eq!((predictor, offset, multiplier), (Predictor::Zero, 37, 1));
                assert_eq!(
                    null.lookup(&[2, id, 0, 0]).unwrap(),
                    actual.lookup(&[2, id, 0, 0]).unwrap()
                );
            }
        }
    });
}

#[test]
fn source_gather_resets_per_channel_and_preserves_all_stream_ids() {
    let channels = [channel(64, 64, 0), channel(64, 64, 17)];
    let a = stream(&channels, 25);
    let b = stream(&channels, 1_000_000);
    let options = options();
    let p = options.sampling.as_ref().unwrap();
    let predictors = encode::predictors(Predictor::Best);
    let (samples, stats) = gather_source_raw(&[&a, &b], p, &predictors, 2, 20_000).unwrap();
    let expected = ChannelSampler::new(64, 64, p.nb_repeats)
        .unwrap()
        .filter(|&v| v)
        .count();
    assert_eq!(samples.len(), 4 * expected);
    assert_eq!(stats.visited_pixels, 16_384);
    assert_eq!(stats.source_total_pixels, 24_576);
    for (block, &(group, channel)) in
        samples
            .chunks_exact(expected)
            .zip(&[(25, 0), (25, 1), (1_000_000, 0), (1_000_000, 1)])
    {
        assert!(block.iter().all(|sample| sample.properties[0] == channel
            && sample.properties[1] == group
            && sample.weight == 1));
    }
}

#[test]
fn weighted_state_runs_through_rejected_pixels_and_wp_property_without_wp_predictor() {
    let channels = [channel(64, 64, -100)];
    let stream = stream(&channels, 42);
    let mut policy = options().sampling.unwrap();
    policy.properties = vec![15, 9];
    let (samples, _) =
        gather_source_raw(&[&stream], &policy, &[Predictor::Gradient], 1, 8192).unwrap();
    let flags: Vec<_> = ChannelSampler::new(64, 64, policy.nb_repeats)
        .unwrap()
        .collect();
    let mut index = 0;
    encode::visit(
        &channels,
        &stream.wp,
        true,
        0,
        42,
        |_, position, n, wp, properties, value| {
            if flags[position] {
                assert_eq!(samples[index].properties, properties);
                assert_eq!(
                    samples[index].residuals,
                    [value
                        - predict_one(Predictor::Gradient, n, Some(i64::from(wp))).unwrap() as i32]
                );
                index += 1;
            }
            Ok(())
        },
    )
    .unwrap();
    assert!(flags.iter().any(|&selected| !selected));
    assert_eq!(index, samples.len());
}

#[test]
fn full_training_includes_all_groups_constants_and_quantile_static_ids() {
    let first = [
        channel(32, 32, 0),
        Channel {
            samples: vec![37; 1024],
            ..channel(32, 32, 0)
        },
    ];
    let last = [
        channel(16, 32, 100),
        Channel {
            samples: vec![37; 512],
            ..channel(16, 32, 0)
        },
    ];
    let mut options = options();
    options.source_constant_channels = true;
    options.sampling.as_mut().unwrap().properties = vec![0, 1, 15, 9];
    let tree = learn(&[stream(&last, 2000), stream(&first, 1000)], &options).unwrap();
    for group in [1000, 2000] {
        let leaf = tree.lookup(&[1, group]).unwrap();
        assert_eq!((leaf.1, leaf.2, leaf.3), (Predictor::Zero, 37, 1));
    }
    options.sample_limit = Some(1535);
    assert!(matches!(
        learn(&[stream(&first, 1000), stream(&last, 2000)], &options),
        Err(Error::Limit(_))
    ));
    options.sample_limit = Some(1536);
    assert!(learn(&[stream(&first, 1000), stream(&last, 2000)], &options).is_ok());
}

#[test]
fn source_default_samples_cross_old_million_cap_without_truncation() {
    let count = (1 << 20) + 1024;
    let channels = [Channel {
        width: 1025,
        height: 1024,
        hshift: 0,
        vshift: 0,
        samples: vec![37; count],
    }];
    let streams = [stream(&channels, 0)];
    let mut options = options();
    assert_eq!(options.sample_limit, None);
    options.predictor = Predictor::Zero;
    let policy = options.sampling.as_mut().unwrap();
    policy.nb_repeats = 1.0;
    policy.properties = vec![0];
    assert!(!options.source_constant_channels);
    let tree = learn(&streams, &options).unwrap();
    assert_eq!(tree.leaves(), 1);
    // The same population must fail at an explicitly requested ceiling.
    options.sample_limit = Some(1 << 20);
    assert!(matches!(learn(&streams, &options), Err(Error::Limit(_))));
    options.sample_limit = Some(count);
    assert!(learn(&streams, &options).is_ok());
}

#[test]
fn source_constant_bypass_defaults_off_and_requires_explicit_opt_in() {
    let channels = [Channel {
        samples: vec![37; 64],
        ..channel(8, 8, 0)
    }];
    let streams = [stream(&channels, 21)];
    let mut options = options();
    assert!(!ModularOptions::default().source_constant_channels);
    assert!(!options.source_constant_channels);
    options.predictor = Predictor::Zero;
    options.sample_limit = Some(64);
    let policy = options.sampling.as_mut().unwrap();
    policy.nb_repeats = 1.0;
    policy.properties = vec![0];
    let ordinary = learn(&streams, &options).unwrap();
    let (_, predictor, offset, multiplier) = ordinary.lookup(&[0, 21]).unwrap();
    assert_eq!((predictor, offset, multiplier), (Predictor::Zero, 0, 1));
    options.sample_limit = Some(63);
    assert!(matches!(learn(&streams, &options), Err(Error::Limit(_))));
    options.source_constant_channels = true;
    options.sample_limit = Some(1);
    let bypassed = learn(&streams, &options).unwrap();
    let (_, predictor, offset, multiplier) = bypassed.lookup(&[0, 21]).unwrap();
    assert_eq!((predictor, offset, multiplier), (Predictor::Zero, 37, 1));
}

#[test]
fn source_zero_fraction_and_legacy_constant_bypass_remain_separate_policies() {
    let channels = [Channel {
        samples: vec![-7; 64],
        ..channel(8, 8, 0)
    }];
    let streams = [stream(&channels, 21)];
    let mut options = options();
    options.sampling.as_mut().unwrap().nb_repeats = 0.0;
    let ordinary = learn(&streams, &options).unwrap();
    let (_, predictor, offset, multiplier) = ordinary.lookup(&[0, 21]).unwrap();
    assert_eq!((predictor, offset, multiplier), (Predictor::Weighted, 0, 1));
    options.source_constant_channels = true;
    let bypassed = learn(&streams, &options).unwrap();
    assert_eq!(bypassed.lookup(&[0, 21]).unwrap().2, -7);
    options.sampling = None;
    for opt_in in [false, true] {
        options.source_constant_channels = opt_in;
        let legacy = learn(&streams, &options).unwrap();
        let (_, predictor, offset, multiplier) = legacy.lookup(&[0, 21]).unwrap();
        assert_eq!((predictor, offset, multiplier), (Predictor::Zero, -7, 1));
    }
}

#[test]
fn zero_fraction_uses_source_seed_predictor_and_verified_constant_offsets() {
    let channels = [
        channel(8, 8, 0),
        Channel {
            samples: vec![-23; 64],
            ..channel(8, 8, 0)
        },
    ];
    let mut options = options();
    options.source_constant_channels = true;
    let policy = options.sampling.as_mut().unwrap();
    policy.nb_repeats = 0.0;
    options.sample_limit = Some(1);
    let tree = learn(&[stream(&channels, 81)], &options).unwrap();
    assert_eq!(tree.leaves(), 2);
    assert_eq!(tree.lookup(&[0, 81]).unwrap().1, Predictor::Weighted);
    assert_eq!(tree.lookup(&[1, 81]).unwrap().2, -23);
    options.sampling = Some(
        SamplingPolicy::from_source(&SourcePolicy {
            decoding_speed_tier: 4,
            ..SourcePolicy::default()
        })
        .unwrap(),
    );
    let tree = learn(&[stream(&channels, 81)], &options).unwrap();
    assert_eq!(tree.lookup(&[0, 81]).unwrap().1, Predictor::Gradient);
}

#[test]
fn metadata_and_oversized_prefix_match_the_quantile_prepass() {
    let mut metadata = channel(8, 1, 0);
    metadata.hshift = -1;
    let channels = [
        metadata,
        channel(1, 4, 0),
        channel(8, 4, 0),
        channel(2, 2, 0),
    ];
    let stream = stream(&channels, 321);
    let mut policy = options().sampling.unwrap();
    policy.max_channel_size = 4;
    let (samples, stats) =
        gather_source_raw(&[&stream], &policy, &[Predictor::Gradient], 4, 100).unwrap();
    assert_eq!(stats.visited_pixels, 12);
    assert_eq!(samples.len(), 12);
    assert!(samples.iter().all(|sample| sample.properties[0] < 2));
    let views: Vec<_> = channels
        .iter()
        .enumerate()
        .map(|(c, p)| ChannelView {
            channel: c as u32,
            group: 321,
            width: p.width,
            height: p.height,
            stride: p.width,
            pixels: &p.samples,
            is_meta: p.hshift < 0,
        })
        .collect();
    let quantiles = sample::collect_pixel_samples(&views, &policy).unwrap();
    assert_eq!(quantiles.channel_pixel_count.get(&0), Some(&8));
    assert_eq!(quantiles.channel_pixel_count.len(), 1);
}

#[test]
fn legacy_mode_and_source_predictor_modes_remain_explicit() {
    let channels = [channel(16, 16, 0)];
    let options = ModularOptions {
        tree_samples: 8,
        sample_limit: Some(8),
        ..ModularOptions::default()
    };
    assert!(learn(&[stream(&channels, 0)], &options).is_ok());
    let mut policy = SamplingPolicy::from_source(&SourcePolicy::default()).unwrap();
    policy.tree_mode = TreeMode::NoWeighted;
    assert_eq!(
        source_predictors(Predictor::Best, &policy).unwrap(),
        [Predictor::Gradient]
    );
    assert!(source_predictors(Predictor::Weighted, &policy).is_err());
    policy.tree_mode = TreeMode::WeightedOnly;
    assert_eq!(
        source_predictors(Predictor::Zero, &policy).unwrap(),
        [Predictor::Weighted]
    );
    policy.tree_mode = TreeMode::GradientOnly;
    assert_eq!(
        source_predictors(Predictor::Best, &policy).unwrap(),
        [Predictor::Weighted, Predictor::Gradient]
    );
}

#[test]
fn legacy_preflight_rejects_before_collector_and_shared_debit() {
    let channels = [channel(16, 16, 0)];
    let stream = stream(&channels, 0);
    let options = ModularOptions {
        tree_samples: 8,
        sample_limit: Some(7),
        ..ModularOptions::default()
    };
    let budget = SampleBudget::new(32);
    assert!(matches!(
        gather_legacy(&[&stream], &options, Some(&budget), |_| {
            panic!("collector reached after local sample-budget rejection")
        }),
        Err(Error::Limit(_))
    ));
    assert_eq!(budget.remaining.load(Ordering::Relaxed), 32);
    let options = ModularOptions {
        sample_limit: Some(8),
        ..options
    };
    let budget = SampleBudget::new(7);
    assert!(matches!(
        gather_legacy(&[&stream], &options, Some(&budget), |_| {
            panic!("collector reached after shared sample-budget rejection")
        }),
        Err(Error::Limit(_))
    ));
    assert_eq!(budget.remaining.load(Ordering::Relaxed), 7);
}

#[test]
fn legacy_preflight_accounts_previous_streams_and_debits_only_once() {
    let channels = [channel(16, 16, 0)];
    let streams = [stream(&channels, 0), stream(&channels, 1)];
    let ordered = [&streams[0], &streams[1]];
    for (local_limit, shared_limit, calls, remaining, succeeds) in [
        (15, 32, 1, 24, false),
        (16, 15, 1, 7, false),
        (16, 16, 2, 0, true),
    ] {
        let options = ModularOptions {
            tree_samples: 8,
            sample_limit: Some(local_limit),
            ..ModularOptions::default()
        };
        let budget = SampleBudget::new(shared_limit);
        let mut collected = 0;
        let result = gather_legacy(&ordered, &options, Some(&budget), |stream| {
            collected += 1;
            assert_eq!(
                budget.remaining.load(Ordering::Relaxed),
                shared_limit - collected * 8
            );
            encode::collect_samples(stream.channels, &options, &stream.wp, stream.stream_id)
        });
        assert_eq!(collected, calls);
        assert_eq!(budget.remaining.load(Ordering::Relaxed), remaining);
        if succeeds {
            let samples = result.unwrap();
            assert_eq!(samples.len(), 16);
            assert!(samples[..8].iter().all(|s| s.properties[1] == 0));
            assert!(samples[8..].iter().all(|s| s.properties[1] == 1));
        } else {
            assert!(matches!(result, Err(Error::Limit(_))));
        }
    }
}

#[test]
fn rejects_malformed_planes_duplicate_ids_and_hard_caps() {
    let channels = [channel(16, 16, 0)];
    let mut options = options();
    options.sample_limit = Some(255);
    assert!(matches!(
        learn(&[stream(&channels, 5)], &options),
        Err(Error::Limit(_))
    ));
    options.sample_limit = Some(256);
    assert!(learn(&[stream(&channels, 5), stream(&channels, 5)], &options).is_err());
    assert!(learn(&[stream(&channels, i32::MAX as usize + 1)], &options).is_err());
    let malformed = [Channel {
        samples: vec![0; 1],
        ..channel(2, 2, 0)
    }];
    assert!(learn(&[stream(&malformed, 0)], &options).is_err());
    assert!(learn(&[], &options).is_err());
}

#[test]
fn all_constant_and_empty_domains_need_no_random_samples() {
    let channels = [
        Channel {
            samples: vec![7; 16],
            ..channel(4, 4, 0)
        },
        Channel {
            samples: vec![-9; 16],
            ..channel(4, 4, 0)
        },
    ];
    let mut options = options();
    options.source_constant_channels = true;
    options.sample_limit = Some(1);
    options.max_tree_leaves = 2;
    let tree = learn(&[stream(&channels, 17)], &options).unwrap();
    assert_eq!(tree.leaves(), 2);
    assert_eq!(tree.lookup(&[0, 17]).unwrap().2, 7);
    assert_eq!(tree.lookup(&[1, 17]).unwrap().2, -9);
    options.max_tree_leaves = 1;
    assert!(matches!(
        learn(&[stream(&channels, 17)], &options),
        Err(Error::Limit(_))
    ));
    let tree = learn(&[stream(&[], 0)], &options).unwrap();
    assert_eq!(tree.leaves(), 1);
    assert_eq!(tree.lookup(&[]).unwrap().1, Predictor::Weighted);
}

#[test]
fn active_batch_gather_preserves_population_and_cannot_evade_sample_limit() {
    let channels = [Channel {
        samples: (0..4096).map(|i| 1 + (i % 2)).collect(),
        ..channel(64, 64, 0)
    }];
    let first = stream(&channels, 1000);
    let second = stream(&channels, 2000);
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 1.0;
    policy.properties = vec![0, 1, 6];
    let views = [
        ChannelView {
            channel: 0,
            group: 1000,
            width: 64,
            height: 64,
            stride: 64,
            pixels: &channels[0].samples,
            is_meta: false,
        },
        ChannelView {
            channel: 0,
            group: 2000,
            width: 64,
            height: 64,
            stride: 64,
            pixels: &channels[0].samples,
            is_meta: false,
        },
    ];
    let population = sample::collect_pixel_samples(&views, &policy).unwrap();
    let quantization = population
        .quantization(&policy.properties, policy.max_property_values)
        .unwrap();
    let (raw, raw_stats) =
        gather_source_raw(&[&first, &second], &policy, &[Predictor::Zero], 1, 8192).unwrap();
    let (merged, stats) = gather_source(
        &[&first, &second],
        &policy,
        &[Predictor::Zero],
        1,
        8192,
        &quantization,
        None,
    )
    .unwrap();
    assert_eq!(stats, raw_stats);
    assert_eq!(stats.selected_samples, 8192);
    assert_eq!(stats.source_total_pixels, 12_288);
    assert_eq!(
        merged.iter().map(|s| u64::from(s.weight)).sum::<u64>(),
        8192
    );
    assert!(merged.len() < raw.len() / 100);
    assert!(merged.iter().any(|s| s.weight > 1));
    assert!(matches!(
        gather_source(
            &[&first, &second],
            &policy,
            &[Predictor::Zero],
            1,
            8191,
            &quantization,
            None
        ),
        Err(Error::Limit(_))
    ));
    let mut train = TrainOptions {
        predictors: vec![Predictor::Zero],
        require_complete_search: true,
        ..TrainOptions::default()
    };
    policy
        .apply_to_train(&population, stats, &mut train)
        .unwrap();
    let reference = ma_train::train(&raw, &train).unwrap();
    let compact = ma_train::train(&merged, &train).unwrap();
    let signature = |tree: &Tree| {
        tree.tokens()
            .iter()
            .map(|t| (t.context, t.value))
            .collect::<Vec<_>>()
    };
    assert_eq!(signature(&compact), signature(&reference));
    let mut options = options();
    options.predictor = Predictor::Zero;
    options.sampling = Some(policy);
    options.sample_limit = Some(8192);
    assert_eq!(
        signature(&learn(&[first, second], &options).unwrap()),
        signature(&reference)
    );
}

#[test]
fn source_requires_complete_learning_but_legacy_keeps_explicit_leaf_cap() {
    let channels = [Channel {
        samples: (0..32).map(|i| i32::from(i >= 16)).collect(),
        ..channel(32, 1, 0)
    }];
    let mut options = options();
    options.predictor = Predictor::Zero;
    options.max_tree_leaves = 1;
    options.tree_samples = 32;
    options.split_threshold = 0.0;
    let policy = options.sampling.as_mut().unwrap();
    policy.nb_repeats = 1.0;
    policy.properties = vec![3];
    policy.node_threshold = 0.0;
    assert!(matches!(
        learn(&[stream(&channels, 21)], &options),
        Err(Error::Limit(_))
    ));
    options.max_tree_leaves = 2;
    assert_eq!(
        learn(&[stream(&channels, 21)], &options).unwrap().leaves(),
        2
    );
    options.max_tree_leaves = 1;
    options.sampling.as_mut().unwrap().node_threshold = 1000.0;
    assert_eq!(
        learn(&[stream(&channels, 21)], &options).unwrap().leaves(),
        1
    );
    options.sampling = None;
    assert_eq!(
        learn(&[stream(&channels, 21)], &options).unwrap().leaves(),
        1
    );
}

#[test]
fn wide_residuals_are_narrowed_and_still_learn_adaptive_splits() {
    let channels = [Channel {
        samples: vec![i32::MIN, i32::MAX, i32::MIN],
        ..channel(3, 1, 0)
    }];
    let mut options = options();
    options.predictor = Predictor::Gradient;
    options.max_tree_leaves = 3;
    let policy = options.sampling.as_mut().unwrap();
    policy.properties = vec![3];
    policy.max_property_values = 256;
    policy.nb_repeats = 1.0;
    policy.node_threshold = 0.0;
    let input = stream(&channels, 21);
    let (samples, stats) =
        gather_source_raw(&[&input], policy, &[Predictor::Gradient], 1, 3).unwrap();
    assert_eq!(stats.selected_samples, 3);
    assert_eq!(
        samples.iter().map(|s| s.residuals[0]).collect::<Vec<_>>(),
        [i32::MIN, -1, 1]
    );
    let tree = learn(&[input], &options).unwrap();
    assert_eq!(tree.leaves(), 3);
    assert_eq!(tree.predictors(), [Predictor::Gradient]);
    for sample in &samples {
        let (_, predictor, offset, multiplier) = tree.lookup(&sample.properties).unwrap();
        assert_eq!((predictor, offset, multiplier), (Predictor::Gradient, 0, 1));
    }
    options.max_tree_leaves = 2;
    assert!(matches!(
        learn(&[stream(&channels, 21)], &options),
        Err(Error::Limit(_))
    ));
}

#[test]
fn variable_gather_narrows_all_predictors_after_wide_subtraction() {
    let extremes = [i32::MIN, i32::MAX, 0, -1, 1, i32::MAX, i32::MIN, 7];
    let channels = [Channel {
        samples: (0..32 * 33)
            .map(|i| extremes[(i * 73 + i / 32 * 13) % extremes.len()])
            .collect(),
        ..channel(32, 33, 0)
    }];
    let input = stream(&channels, 21);
    let predictors = encode::predictors(Predictor::Variable);
    let mut policy = options().sampling.unwrap();
    policy.nb_repeats = 1.0;
    let (samples, stats) = gather_source_raw(&[&input], &policy, &predictors, 1, 1056).unwrap();
    assert_eq!(stats.selected_samples, 1056);
    assert_eq!(predictors.len(), 14);
    let mut wide_positive = false;
    let mut wide_negative = false;
    let mut wide_prediction = false;
    encode::visit(
        &channels,
        &input.wp,
        true,
        0,
        input.stream_id,
        |_, position, neighbors, wp, properties, value| {
            let sample = &samples[position];
            assert_eq!(sample.properties, properties);
            for (index, &predictor) in predictors.iter().enumerate() {
                let guess = predict_one(predictor, neighbors, Some(i64::from(wp))).unwrap();
                let difference = i128::from(value) - i128::from(guess);
                wide_positive |= difference > i128::from(i32::MAX);
                wide_negative |= difference < i128::from(i32::MIN);
                wide_prediction |= i32::try_from(guess).is_err();
                let low_bits = difference.rem_euclid(1i128 << 32) as u32;
                assert_eq!(sample.residuals[index] as u32, low_bits);
            }
            Ok(())
        },
    )
    .unwrap();
    assert!(wide_positive && wide_negative && wide_prediction);
}
