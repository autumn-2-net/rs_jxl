use super::*;
use crate::decode::{DecodeOptions, decode};
use crate::entropy::HybridUintConfig;
use crate::modular::{
    encode::{FixedTree, ModularOptions},
    group::Channel,
    ma_train::{self, MaBudgetMode},
    predictor::{Neighbors, predict_one},
    training::{self, SampleBudget, TrainingStream},
    tree::{Node, Properties, pack_signed},
    tree_groups::TreeGroupOptions,
    weighted,
};
use crate::{EncodeOptions, Image, PixelData, PixelLayout};

fn experiment(tile: usize) -> Options {
    Options {
        tile_size: tile,
        receipt: Some(Arc::new(Receipt::default())),
        ..Default::default()
    }
}

fn options(tile: usize, leaves: usize) -> ModularOptions {
    ModularOptions {
        block_context: Some(experiment(tile)),
        ma_max_leaves: Some(leaves),
        max_merged_tree_leaves: Some(leaves),
        ..Default::default()
    }
}

fn view(channel: &Channel, index: u32, group: u32) -> ChannelView<'_> {
    ChannelView {
        channel: index,
        group,
        width: channel.width,
        height: channel.height,
        stride: channel.width,
        pixels: &channel.samples,
        is_meta: channel.hshift < 0 || channel.vshift < 0,
    }
}

fn stream(channels: &[Channel], id: usize) -> TrainingStream<'_> {
    TrainingStream {
        channels,
        stream_id: id,
        wp: weighted::Header::default(),
    }
}

fn hash(i: usize) -> i32 {
    let mut x = (i as u32).wrapping_add(0x9e37_79b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
    ((x ^ (x >> 16)) & 65535) as i32
}

fn plane(width: usize, height: usize, kind: usize) -> Channel {
    Channel {
        width,
        height,
        hshift: 0,
        vshift: 0,
        samples: (0..width * height)
            .map(|i| match kind {
                0 => 0,
                1 => hash(i / width),
                2 => hash(i % width),
                _ => hash(i),
            })
            .collect(),
    }
}

fn signature(tree: &Tree) -> Vec<(usize, u32)> {
    tree.tokens()
        .iter()
        .map(|token| (token.context, token.value))
        .collect()
}

#[test]
fn parallel_domains_and_spatial_plans_keep_ordered_histogram_decisions() {
    let channels = [plane(513, 257, 1), plane(513, 257, 2), plane(129, 83, 3)];
    let views = [
        view(&channels[0], 0, 1),
        view(&channels[1], 2, 1),
        view(&channels[2], 0, 7),
    ];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for limit in [1, 3, 512, 4096] {
        let serial_options = experiment(16);
        let parallel_options = experiment(16);
        let serial = crate::execution::serial(|| learn(&views, limit, &serial_options)).unwrap();
        let parallel =
            crate::execution::with_pool(&pool, || learn(&views, limit, &parallel_options)).unwrap();
        assert_eq!(signature(&serial), signature(&parallel));
        assert_eq!(
            serial_options.receipt.unwrap().snapshot(),
            parallel_options.receipt.unwrap().snapshot()
        );
    }
}

// Independent scalar scoring oracle, with explicit token counts/extra bits.
fn oracle(view: &ChannelView<'_>, rect: plan::Rect) -> Predictor {
    let mut winner = (Predictor::Zero, f64::INFINITY);
    for predictor in PREDICTORS {
        let mut counts = [0usize; 128];
        let mut extra = 0;
        let mut count = 0;
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let n = Neighbors::from_plane(view.pixels, view.width, view.stride, x, y).unwrap();
                let prediction = predict_one(predictor, n, None).unwrap();
                let r = (i64::from(view.pixels[y * view.stride + x]) - prediction) as i32;
                let token = HybridUintConfig::default().encode(pack_signed(r));
                counts[token.token as usize] += 1;
                extra += token.nbits as usize;
                count += 1;
            }
        }
        let n = count as f64;
        let entropy = if count == 0 {
            0.0
        } else {
            n * n.log2()
                - counts
                    .iter()
                    .filter(|&&c| c != 0)
                    .map(|&c| c as f64 * (c as f64).log2())
                    .sum::<f64>()
        };
        let bits =
            entropy + extra as f64 + 24.0 + 3.0 * counts.iter().filter(|&&c| c != 0).count() as f64;
        if bits < winner.1 {
            winner = (predictor, bits);
        }
    }
    winner.0
}

#[test]
fn single_tile_choices_match_pack_signed_hybrid_uint_oracle_and_ties() {
    for tile in [8, 16] {
        for (width, height) in [(1, 1), (1, 7), (7, 1), (7, 7), (8, 8)] {
            for kind in 0..4 {
                let channel = plane(width, height, kind);
                let v = view(&channel, 0, 7);
                let expected = oracle(
                    &v,
                    plan::Rect {
                        x0: 0,
                        y0: 0,
                        x1: width,
                        y1: height,
                    },
                );
                let tree = learn(&[v], 64, &experiment(tile)).unwrap();
                assert_eq!(tree.leaves(), 1);
                assert_eq!(tree.lookup(&[0, 7, 0, 0]).unwrap().1, expected);
                if kind == 0 {
                    assert_eq!(expected, Predictor::Zero);
                }
            }
        }
    }
    assert_eq!(PREDICTORS.map(|p| p as u32), [0, 1, 2, 5]);
}

#[test]
fn retained_tile_choices_have_exact_pixel_membership() {
    for tile in [8, 16] {
        let width = tile * 2;
        let channel = Channel {
            width,
            height: tile,
            hshift: 0,
            vshift: 0,
            samples: (0..width * tile)
                .map(|i| {
                    if i % width < tile {
                        hash(i / width)
                    } else {
                        hash(100 + i % width)
                    }
                })
                .collect(),
        };
        let v = view(&channel, 0, 23);
        let expected: Vec<_> = (0..2)
            .map(|b| {
                oracle(
                    &v,
                    plan::Rect {
                        x0: b * tile,
                        y0: 0,
                        x1: (b + 1) * tile,
                        y1: tile,
                    },
                )
            })
            .collect();
        assert_eq!(expected, [Predictor::Left, Predictor::Top]);
        let block = experiment(tile);
        let tree = learn(&[v], 64, &block).unwrap();
        assert_eq!(tree.leaves(), 2, "tile={tile}");
        for y in 0..tile {
            for x in 0..width {
                assert_eq!(
                    tree.lookup(&[0, 23, y as i32, x as i32]).unwrap().1,
                    expected[x / tile],
                    "tile={tile} x={x} y={y}"
                );
            }
        }
        let receipt = block.receipt.unwrap().snapshot();
        assert_eq!(receipt.spatial_splits, 1);
        assert_eq!(receipt.coarsened_views, 0);
    }
}

#[test]
fn scoring_uses_original_neighbors_at_tile_edges_and_full_i32_range() {
    for tile in [8, 16] {
        let width = tile + 3;
        let height = tile + 1;
        let stride = width + 5;
        let mut pixels = vec![123456; (height - 1) * stride + width];
        let values = [i32::MIN, i32::MAX, -1, 0, 65535, -65536, 7];
        for y in 0..height {
            for x in 0..width {
                pixels[y * stride + x] = values[(x + 3 * y) % values.len()];
            }
        }
        let v = ChannelView {
            channel: 4,
            group: 19,
            width,
            height,
            stride,
            pixels: &pixels,
            is_meta: true,
        };
        let rect = plan::Rect {
            x0: tile,
            y0: tile,
            x1: width,
            y1: height,
        };
        let mut actual = std::array::from_fn(|_| stats::Histogram::default());
        let mut expected = std::array::from_fn::<_, 4, _>(|_| stats::Histogram::default());
        plan::gather(&v, rect, &mut actual).unwrap();
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let n = Neighbors::from_plane(&pixels, width, stride, x, y).unwrap();
                for (p, histogram) in PREDICTORS.into_iter().zip(&mut expected) {
                    let prediction = predict_one(p, n, None).unwrap();
                    histogram.add(
                        (i64::from(pixels[y * stride + x]) - prediction) as i32,
                        false,
                    );
                }
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(actual[0].count, 3);
    }
}

#[test]
fn entropy_estimate_can_prefer_larger_sad_and_uniform_plans_coalesce() {
    let mut sparse = stats::Histogram::default();
    let mut dense = stats::Histogram::default();
    for i in 0..256 {
        sparse.add(if i == 0 { 1024 } else { 0 }, false);
        dense.add(if i % 2 == 0 { 1 } else { -1 }, false);
    }
    assert!(sparse.bits(0) < dense.bits(0)); // SAD would prefer 256 to 1024.
    let channels = [plane(65, 33, 0)];
    let opts = options(8, 1024);
    let tree = training::learn(&[stream(&channels, 3)], &opts).unwrap();
    assert_eq!(tree.leaves(), 1);
    assert_eq!(tree.lookup(&[0, 3, 32, 64]).unwrap().1, Predictor::Zero);
    let snapshot = opts.block_context.unwrap().receipt.unwrap().snapshot();
    assert_eq!(snapshot.requested_tiles, 9 * 5);
    assert_eq!(snapshot.coarsened_views, 0);
    assert_eq!(snapshot.predictor_leaves, [1, 0, 0, 0]);
}

#[test]
fn sparse_stream_and_metadata_channel_dispatch_preserves_original_indices() {
    let empty = plane(0, 0, 0);
    // A zero plane can legally share any predictor, so it cannot force an
    // original-channel dispatch. Noise and directional data need distinct costs.
    let mut meta_noise = plane(16, 16, 3);
    meta_noise.hshift = -1;
    let mut meta_left = plane(16, 16, 1);
    meta_left.hshift = -1;
    let a = [meta_noise, empty.clone(), plane(16, 16, 2)];
    let b = [meta_left, empty, plane(16, 16, 3)];
    let streams = [stream(&b, 91), stream(&a, 7)];
    let opts = options(16, 128);
    let tree = training::learn(&streams, &opts).unwrap();
    assert!(
        tree.nodes()
            .iter()
            .any(|n| matches!(n, Node::Split { property: 0, .. }))
    );
    assert!(
        tree.nodes()
            .iter()
            .any(|n| matches!(n, Node::Split { property: 1, .. }))
    );
    for s in &streams {
        for (c, channel) in s
            .channels
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.samples.is_empty())
        {
            let v = view(channel, c as u32, s.stream_id as u32);
            let expected = oracle(
                &v,
                plan::Rect {
                    x0: 0,
                    y0: 0,
                    x1: 16,
                    y1: 16,
                },
            );
            for y in 0..16 {
                for x in 0..16 {
                    let n = Neighbors::from_plane(&channel.samples, 16, 16, x, y).unwrap();
                    let props = Properties::default().calculate(n, x, y, c, s.stream_id, 0);
                    assert_eq!(
                        tree.lookup(&props).unwrap().1,
                        expected,
                        "stream={} c={c} x={x} y={y}",
                        s.stream_id
                    );
                }
            }
        }
    }
    let reverse = [stream(&a, 7), stream(&b, 91)];
    assert_eq!(
        signature(&tree),
        signature(&training::learn(&reverse, &opts).unwrap())
    );
    let snap = opts
        .block_context
        .as_ref()
        .unwrap()
        .receipt
        .as_ref()
        .unwrap()
        .snapshot();
    assert_eq!(snap.trees, 2);
    assert_eq!(snap.pixels, 2 * 4 * 256);
    assert_eq!(snap.views, 8);
    assert_eq!(snap.coarsened_views, 0);
}

#[test]
fn ragged_grid_compilation_maps_every_pixel_to_its_selected_block() {
    fn grid(tile: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> plan::Plan {
        if x1 - x0 == 1 && y1 - y0 == 1 {
            return plan::Plan::Leaf(PREDICTORS[(x0 + 2 * y0) % 4]);
        }
        if x1 - x0 >= y1 - y0 && x1 - x0 > 1 {
            let mid = (x0 + x1) / 2;
            plan::Plan::Split {
                property: 3,
                threshold: (mid * tile - 1) as i32,
                greater: Box::new(grid(tile, mid, x1, y0, y1)),
                less_equal: Box::new(grid(tile, x0, mid, y0, y1)),
            }
        } else {
            let mid = (y0 + y1) / 2;
            plan::Plan::Split {
                property: 2,
                threshold: (mid * tile - 1) as i32,
                greater: Box::new(grid(tile, x0, x1, mid, y1)),
                less_equal: Box::new(grid(tile, x0, x1, y0, mid)),
            }
        }
    }
    for tile in [8, 16] {
        for (width, height) in [(1usize, 41usize), (41, 1), (35, 19), (19, 35)] {
            let plan = grid(tile, 0, width.div_ceil(tile), 0, height.div_ceil(tile));
            let mut nodes = Vec::new();
            plan.append(&mut nodes);
            let tree = Tree::new(nodes).unwrap();
            for y in 0..height {
                for x in 0..width {
                    assert_eq!(
                        tree.lookup(&[13, 91, y as i32, x as i32]).unwrap().1,
                        PREDICTORS[(x / tile + 2 * (y / tile)) % 4]
                    );
                }
            }
        }
    }
}

#[test]
fn explicit_limits_coarsen_plans_even_in_strict_and_grouped_domains() {
    let a = [plane(65, 33, 3), plane(17, 1, 2), plane(1, 19, 1)];
    let b = [plane(19, 35, 1)];
    let streams = [stream(&b, 91), stream(&a, 7)];
    let total: usize = a.iter().chain(&b).map(|p| p.samples.len()).sum();
    for tile in [8, 16] {
        for limit in [1, 2, 3, 7, 64] {
            let mut opts = options(tile, limit);
            opts.ma_budget_mode = Some(MaBudgetMode::Strict);
            opts.tree_groups = Some(TreeGroupOptions {
                requested_shards: 8,
                min_work: 1,
                max_merged_leaves: limit,
                ..Default::default()
            });
            opts.tree_group_splits = vec![0, 8, 92];
            opts.sample_limit = Some(total);
            let budget = SampleBudget::new(total);
            let tree = training::learn_with_budget(&streams, &opts, Some(&budget), false).unwrap();
            assert!(tree.leaves() <= limit);
            assert!(training::learn_with_budget(&streams, &opts, Some(&budget), false).is_err());
            let from_grouped_entry = training::learn(&streams, &opts).unwrap();
            assert_eq!(signature(&tree), signature(&from_grouped_entry));
            for node in tree.nodes() {
                match node {
                    Node::Split { property, .. } => assert!(*property <= 3),
                    Node::Leaf {
                        predictor,
                        offset,
                        multiplier,
                    } => {
                        assert!(PREDICTORS.contains(predictor));
                        assert_eq!((*offset, *multiplier), (0, 1));
                    }
                }
            }
            let snap = opts
                .block_context
                .as_ref()
                .unwrap()
                .receipt
                .as_ref()
                .unwrap()
                .snapshot();
            assert_eq!(snap.pixels, 2 * total);
            assert_eq!(snap.leaves, snap.predictor_leaves.iter().sum::<usize>());
            assert_eq!(
                snap.spatial_splits + snap.channel_splits + snap.stream_splits,
                snap.leaves - snap.trees
            );
            if limit == 1 {
                assert_eq!(snap.coarsened_views, 8);
            }
            opts.sample_limit = Some(total - 1);
            assert!(matches!(
                training::learn(&streams, &opts),
                Err(Error::Limit(_))
            ));
        }
    }
    // A group ceiling smaller than the ordinary cap also coarsens dispatch.
    let mut opts = options(8, 128);
    opts.tree_groups = Some(TreeGroupOptions {
        max_merged_leaves: 1,
        ..Default::default()
    });
    assert_eq!(training::learn(&streams, &opts).unwrap().leaves(), 1);
}

#[test]
fn optional_work_bound_does_not_become_an_input_tile_cap() {
    let channels = [plane((DEFAULT_PLAN_LEAVES + 13) * 8, 1, 3)];
    let opts = options(8, ma_train::MAX_TREE_LEAVES);
    let tree = training::learn(&[stream(&channels, 19)], &opts).unwrap();
    assert!(tree.leaves() <= DEFAULT_PLAN_LEAVES);
    let snap = opts.block_context.unwrap().receipt.unwrap().snapshot();
    assert_eq!(snap.requested_tiles, DEFAULT_PLAN_LEAVES + 13);
    assert_eq!(snap.pixels, channels[0].samples.len());
    assert_eq!(snap.coarsened_views, 1);
    let mut expanded = options(8, ma_train::MAX_TREE_LEAVES);
    expanded.block_context.as_mut().unwrap().max_plan_leaves = DEFAULT_PLAN_LEAVES + 13;
    training::learn(&[stream(&channels, 19)], &expanded).unwrap();
    assert_eq!(
        expanded
            .block_context
            .unwrap()
            .receipt
            .unwrap()
            .snapshot()
            .coarsened_views,
        0
    );
}

#[test]
fn malformed_domains_fail_cleanly_and_empty_views_need_no_tiles() {
    let channels = [plane(9, 5, 3)];
    let opts = options(8, 16);
    assert!(training::learn(&[], &opts).is_err());
    assert!(training::learn(&[stream(&channels, 7), stream(&channels, 7)], &opts).is_err());
    assert!(training::learn(&[stream(&channels, i32::MAX as usize + 1)], &opts).is_err());
    let mut bad = channels[0].clone();
    bad.samples.pop();
    assert!(training::learn(&[stream(&[bad], 7)], &opts).is_err());
    for splits in [vec![0], vec![8, 20], vec![0, 7], vec![0, 20, 19]] {
        let mut grouped = opts.clone();
        grouped.tree_groups = Some(Default::default());
        grouped.tree_group_splits = splits;
        assert!(training::learn(&[stream(&channels, 7)], &grouped).is_err());
    }
    let empty = [plane(0, 9, 0), plane(17, 0, 0)];
    let empty_opts = options(8, 1);
    let tree = training::learn(&[stream(&empty, 3)], &empty_opts).unwrap();
    assert_eq!(tree.leaves(), 1);
    assert_eq!(tree.lookup(&[1, 3, 0, 0]).unwrap().1, Predictor::Zero);
    let snap = empty_opts
        .block_context
        .unwrap()
        .receipt
        .unwrap()
        .snapshot();
    assert_eq!((snap.pixels, snap.views, snap.requested_tiles), (0, 0, 0));
}

#[test]
fn configuration_is_opt_in_and_conflicts_are_explicit() {
    assert!(ModularOptions::default().block_context.is_none());
    assert_eq!(Options::default().tile_size, 16);
    for tile in [0, 1, 7, 9, 15, 17, usize::MAX] {
        assert!(experiment(tile).validate().is_err());
    }
    let opts = options(8, 64);
    for conflict in 0..8 {
        let mut bad = opts.clone();
        match conflict {
            0 => bad.fixed_tree = Some(FixedTree::GradientLeaf),
            1 => bad.custom_tree = Some(Tree::channel_predictors(&[Predictor::Zero]).unwrap()),
            2 => bad.fast_context = Some(Default::default()),
            3 => bad.source_ma_seed = Some(Default::default()),
            4 => bad.refinement.leaves = true,
            5 => bad.refinement.offsets = Some(Default::default()),
            6 => bad.source_channel_tree = ma_train::ChannelTreeMode::ChannelsAndRowsTwice,
            _ => bad.tree_learning = ma_train::TreeLearning::Cost,
        }
        assert!(
            matches!(bad.validate(), Err(Error::InvalidInput(_))),
            "conflict={conflict}"
        );
    }
    // Source sampling and entropy choices are supported, not replaced by MA
    // search. Post-tree entropy/context-map work may retain the frozen plan.
    let mut allowed = opts;
    allowed.refinement.entropy = true;
    allowed.validate().unwrap();
}

fn rgba(width: usize, height: usize, bits: u32, palette: bool) -> Image {
    let max = (1u32 << bits) - 1;
    let words: Vec<_> = (0..width * height * 4)
        .map(|i| {
            let pixel = i / 4;
            if i % 4 == 3 {
                if pixel % 3 == 0 { 0 } else { max }
            } else if palette {
                ((pixel % 7) as u32 * (37 + (i % 4) as u32 * 19)) & max
            } else {
                (hash(pixel * 3 + i % 4) as u32) & max
            }
        })
        .collect();
    let pixels = if bits == 8 {
        PixelData::U8(words.iter().map(|&v| v as u8).collect())
    } else {
        PixelData::U16(words.iter().map(|&v| v as u16).collect())
    };
    Image::new(width as u32, height as u32, PixelLayout::Rgba, pixels).unwrap()
}

fn assert_roundtrip(image: &Image, opts: &EncodeOptions) {
    let bytes = crate::encode(image, opts).unwrap();
    let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded.frames.len(), 1);
    assert_eq!(decoded.frames[0].pixels, *image.pixels());
    assert!(
        opts.modular
            .block_context
            .as_ref()
            .unwrap()
            .receipt
            .as_ref()
            .unwrap()
            .snapshot()
            .trees
            > 0
    );
}

#[test]
fn original_rgba8_rgba16_hidden_rgb_roundtrips_ragged_and_multistream_shapes() {
    for bits in [8, 16] {
        for (width, height) in [(1, 1), (1, 19), (19, 1), (35, 19), (145, 17)] {
            let image = rgba(width, height, bits, false);
            for tile in [8, 16] {
                for shared_tree in [false, true] {
                    let opts = EncodeOptions {
                        threads: 2,
                        shared_tree,
                        group_size_shift: 0,
                        modular: options(tile, 128),
                        ..Default::default()
                    };
                    assert_roundtrip(&image, &opts);
                }
            }
        }
    }
}

#[test]
fn transformed_metadata_palette_and_squeeze_roundtrip_all_original_pixels() {
    for bits in [8, 16] {
        let image = rgba(145, 19, bits, true);
        for tile in [8, 16] {
            for mode in 0..3 {
                let mut opts = EncodeOptions {
                    threads: 2,
                    shared_tree: true,
                    group_size_shift: 0,
                    modular: options(tile, 128),
                    ..Default::default()
                };
                if mode < 2 {
                    opts.modular.palette_colors = Some(32);
                    opts.global_palette = mode == 1;
                } else {
                    opts.modular.squeeze = vec![true, false];
                }
                assert_roundtrip(&image, &opts);
                let snap = opts
                    .modular
                    .block_context
                    .as_ref()
                    .unwrap()
                    .receipt
                    .as_ref()
                    .unwrap()
                    .snapshot();
                assert!(snap.pixels > 0 && snap.views > 0);
            }
        }
    }
}

#[test]
fn ordinary_effort9_native_projection_supports_block_options_and_roundtrips() {
    use crate::scheduler::{adapter, params::CandidateParams};
    let image = rgba(35, 19, 8, false);
    let params = CandidateParams::default().backend(9).unwrap();
    let mut opts = adapter::project_native(&params, &image, &EncodeOptions::default())
        .unwrap()
        .exact_options()
        .unwrap();
    // patch_lab installs the experiment after projection; explicit fixed trees
    // are conflicts, never a silently ignored block experiment.
    assert!(opts.modular.fixed_tree.is_none());
    opts.modular.block_context = Some(experiment(8));
    opts.modular.validate().unwrap();
    assert_roundtrip(&image, &opts);
}
