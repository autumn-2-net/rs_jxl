use super::*;
use crate::modular::predictor::predict_one;

fn full() -> Options {
    Options {
        predictors: PredictorSet::Full,
        ..Default::default()
    }
}

fn view(pixels: &[i32], width: usize, height: usize, stride: usize) -> ChannelView<'_> {
    ChannelView {
        channel: 0,
        group: 19,
        width,
        height,
        stride,
        pixels,
        is_meta: false,
    }
}

// Independent neighbor loads, unfused WP predict/update, and scalar equations.
fn oracle(view: &ChannelView<'_>, header: weighted::Header) -> Vec<Vec<i32>> {
    let mut state = weighted::State::new(header, view.width).unwrap();
    let mut samples = Vec::new();
    for y in 0..view.height {
        for x in 0..view.width {
            let n = Neighbors::from_plane(view.pixels, view.width, view.stride, x, y).unwrap();
            let (wp, _) = state
                .predict(x, y, n.top, n.left, n.topright, n.topleft, n.toptop)
                .unwrap();
            let value = view.pixels[y * view.stride + x];
            samples.push(
                FULL_PREDICTORS
                    .iter()
                    .map(|&predictor| {
                        (i64::from(value) - predict_one(predictor, n, Some(i64::from(wp))).unwrap())
                            as i32
                    })
                    .collect(),
            );
            state.update(value, x, y).unwrap();
        }
    }
    samples
}

fn patterned(width: usize, height: usize, stride: usize) -> Vec<i32> {
    let mut samples = vec![0; (height - 1) * stride + width];
    let mut random = 0x9371_05ab_u32;
    for y in 0..height {
        for x in 0..width {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            samples[y * stride + x] = match (x + y) % 7 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => -1,
                _ => random as i32,
            };
        }
    }
    samples
}

#[test]
fn basic_is_the_default_and_full_contains_exactly_the_standard_predictors() {
    assert_eq!(PredictorSet::default(), PredictorSet::Basic);
    assert_eq!(Options::default().predictors, PredictorSet::Basic);
    assert_eq!(PredictorSet::Basic.predictors(), &PREDICTORS);
    assert_eq!(Bin::new(PredictorSet::Basic).original.len(), 4);
    assert_eq!(Bin::new(PredictorSet::Full).original.len(), 14);
    assert_eq!(
        rows::Rows::new(1, PredictorSet::Basic)
            .residuals_at(0)
            .count(),
        4
    );
    assert_eq!(
        rows::Rows::new(1, PredictorSet::Full)
            .residuals_at(0)
            .count(),
        14
    );
    let mut ids: Vec<_> = FULL_PREDICTORS.iter().map(|&p| p as u32).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..14).collect::<Vec<_>>());
}

#[test]
fn full_rows_match_scalar_equations_and_exact_wp_for_all_presets_and_backends() {
    for width in [1, 2, 3, 17, 35] {
        let stride = width + 3;
        let pixels = patterned(width, 7, stride);
        let view = view(&pixels, width, 7, stride);
        let mut traces = Vec::new();
        for preset in 0..5 {
            let header = weighted::Header::preset(preset).unwrap();
            let expected = oracle(&view, header);
            traces.push(expected.iter().map(|sample| sample[13]).collect::<Vec<_>>());
            for backend in [
                scan::Backend::Scalar,
                scan::Backend::Auto,
                scan::Backend::Avx2,
                scan::Backend::Avx512,
            ] {
                if !backend.available() {
                    continue;
                }
                let mut rows = rows::Rows::new(width + 2, PredictorSet::Full);
                let mut state = weighted::State::new(header, width).unwrap();
                for y in 0..view.height {
                    rows.fill(&view, y, backend, Some(&mut state)).unwrap();
                    for x in 0..width {
                        assert_eq!(
                            rows.residuals_at(x).collect::<Vec<_>>(),
                            expected[y * width + x],
                            "{preset} {backend:?} {width} {x} {y}"
                        );
                    }
                }
            }
        }
        if width == 35 {
            assert!(
                traces[1..].iter().any(|trace| trace != &traces[0]),
                "fixture must distinguish non-default WP headers"
            );
        }
    }
}

#[test]
fn full_leaf_learning_can_select_true_weighted_for_every_preset() {
    let width = 35;
    let height = 17;
    for preset in 0..5 {
        let header = weighted::Header::preset(preset).unwrap();
        let mut state = weighted::State::new(header, width).unwrap();
        let mut pixels = vec![0; width * height];
        let mut random = 0x9371_05ab_u32;
        for y in 0..height {
            for x in 0..width {
                let n = Neighbors::from_plane(&pixels, width, width, x, y).unwrap();
                let (prediction, _) = state
                    .predict(x, y, n.top, n.left, n.topright, n.topleft, n.toptop)
                    .unwrap();
                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                let value = if y == 0 {
                    (random % 8192) as i32 - 4096
                } else {
                    prediction
                };
                pixels[y * width + x] = value;
                state.update(value, x, y).unwrap();
            }
        }
        let options = Options {
            mode: Mode::Leaf,
            ..full()
        };
        let tree = learn(
            &[view(&pixels, width, height, width)],
            1,
            64,
            false,
            &[(19, header)],
            &options,
        )
        .unwrap();
        assert!(
            matches!(
                tree.nodes(),
                [Node::Leaf {
                    predictor: Predictor::Weighted,
                    offset: 0,
                    multiplier: 1,
                }]
            ),
            "preset {preset}: {:?}",
            tree.nodes()
        );
    }
}

#[test]
fn full_scan_uses_stream_ids_and_resets_wp_per_view_and_bias_pass() {
    let pixels = patterned(35, 9, 39);
    let base = view(&pixels, 35, 9, 39);
    let groups = [19, 3, 997, 7, 90];
    let mut views: Vec<_> = groups
        .iter()
        .map(|&group| ChannelView { group, ..base })
        .collect();
    views.push(ChannelView {
        channel: 2,
        is_meta: true,
        ..base
    });
    views.push(ChannelView {
        group: 2031,
        ..base
    });
    let mut supplied: Vec<_> = groups
        .iter()
        .enumerate()
        .map(|(preset, &group)| (group, weighted::Header::preset(preset as u32).unwrap()))
        .rev()
        .collect();
    supplied.push((
        2031,
        weighted::Header {
            p1_c: 31,
            p2_c: 0,
            p3_ce: 29,
            w: [0, 15, 1, 14],
            ..weighted::Header::default()
        },
    ));
    let headers = resolve_headers(&views, &supplied, PredictorSet::Full).unwrap();
    let references: Vec<_> = views.iter().collect();
    let samples: Vec<_> = views
        .iter()
        .map(|view| oracle(view, headers[&view.group]))
        .collect();
    for bias in [Bias::Off, Bias::Static, Bias::Adaptive] {
        let options = Options { bias, ..full() };
        let mut actual = Bin::new(PredictorSet::Full);
        let mut expected = Bin::new(PredictorSet::Full);
        let mut axes: Vec<_> = [10, 11]
            .into_iter()
            .map(|property| Axis::new(property, vec![-100, 0, 100], PredictorSet::Full))
            .collect();
        let mut expected_axes: Vec<_> = [10, 11]
            .into_iter()
            .map(|property| Axis::new(property, vec![-100, 0, 100], PredictorSet::Full))
            .collect();
        for shifted in [false, true] {
            if shifted {
                if bias == Bias::Off {
                    break;
                }
                actual.prepare_bias(bias);
                expected.prepare_bias(bias);
                for axis in axes.iter_mut().chain(&mut expected_axes) {
                    for bin in &mut axis.bins {
                        bin.prepare_bias(bias);
                    }
                }
            }
            scan_views(
                &references,
                &headers,
                &mut axes,
                &mut actual,
                &options,
                shifted,
                &Previous::new(),
            )
            .unwrap();
            for (view, samples) in views.iter().zip(&samples) {
                for (index, sample) in samples.iter().enumerate() {
                    expected.add(sample.iter().copied(), bias, shifted);
                    let n = Neighbors::from_plane(
                        view.pixels,
                        view.width,
                        view.stride,
                        index % view.width,
                        index / view.width,
                    )
                    .unwrap();
                    for axis in &mut expected_axes {
                        let value = if axis.property == 10 {
                            n.left.wrapping_sub(n.topleft)
                        } else {
                            n.topleft.wrapping_sub(n.top)
                        };
                        let bin = axis.thresholds.partition_point(|&t| value > t);
                        axis.bins[bin].add(sample.iter().copied(), bias, shifted);
                    }
                }
            }
            assert_eq!(actual.original, expected.original);
            assert_eq!(actual.offsets, expected.offsets);
            assert_eq!(actual.shifted, expected.shifted);
            assert!(
                actual
                    .original
                    .iter()
                    .all(|h| h.count == (35 * 9 * views.len()) as u64)
            );
            for (actual, expected) in axes.iter().zip(&expected_axes) {
                for (actual, expected) in actual.bins.iter().zip(&expected.bins) {
                    assert_eq!(actual.original, expected.original);
                    assert_eq!(actual.offsets, expected.offsets);
                    assert_eq!(actual.shifted, expected.shifted);
                }
            }
        }
    }
}

#[test]
fn full_requires_unique_valid_headers_only_for_referenced_streams() {
    let pixels = [1, 2, 3, 4];
    let base = view(&pixels, 2, 2, 2);
    let views = [base, ChannelView { group: 3, ..base }];
    let header = weighted::Header::default();
    let invalid = weighted::Header { p1_c: 32, ..header };
    for headers in [
        vec![],
        vec![(19, header)],
        vec![(19, header), (19, header), (3, header)],
        vec![(19, header), (3, header), (3, header)],
        vec![(19, header), (3, invalid)],
    ] {
        assert!(matches!(
            learn(&views, 1, 64, false, &headers, &full()),
            Err(Error::InvalidInput(_))
        ));
        assert!(learn(&views, 1, 64, false, &headers, &Options::default()).is_ok());
    }
    let extra = [(88, invalid), (19, header), (88, invalid), (3, header)];
    assert!(learn(&views, 1, 64, false, &extra, &full()).is_ok());
    let empty = [ChannelView { height: 0, ..base }];
    assert!(learn(&empty, 1, 64, false, &[], &full()).is_err());
    assert!(learn(&empty, 1, 64, false, &[(19, header), (19, header)], &full()).is_err());
}

#[test]
fn zero_height_usize_max_width_never_sizes_scratch_or_weighted_history() {
    let empty = ChannelView {
        width: usize::MAX,
        height: 0,
        stride: usize::MAX,
        pixels: &[],
        ..view(&[], 0, 0, 0)
    };
    let pixels = [17, -19];
    let live = view(&pixels, 2, 1, 2);
    for predictors in [PredictorSet::Basic, PredictorSet::Full] {
        for bias in [Bias::Off, Bias::Static, Bias::Adaptive] {
            let receipt = Arc::new(Receipt::default());
            let options = Options {
                predictors,
                bias,
                receipt: Some(receipt.clone()),
                ..Default::default()
            };
            let headers = [(19, weighted::Header::default())];
            assert!(learn(&[empty], 1, 64, false, &headers, &options).is_ok());
            assert_eq!(receipt.snapshot().pixels, 0);
            assert!(learn(&[empty, live], 1, 64, false, &headers, &options).is_ok());
            assert_eq!(receipt.snapshot().pixels, 2);
        }
    }
}

#[test]
fn leaf_plans_preserve_shared_groups_metadata_and_sparse_channel_ids() {
    let a = [37; 12];
    let b = [-900; 12];
    let c = [123456; 12];
    let base = view(&a, 4, 3, 4);
    let views = [
        base,
        ChannelView { group: 3, ..base },
        ChannelView {
            channel: 2,
            pixels: &b,
            is_meta: true,
            ..base
        },
        ChannelView {
            channel: 4,
            group: 3,
            pixels: &c,
            ..base
        },
    ];
    let headers = [
        (3, weighted::Header::preset(4).unwrap()),
        (19, weighted::Header::preset(1).unwrap()),
    ];
    for predictors in [PredictorSet::Basic, PredictorSet::Full] {
        let receipt = Arc::new(Receipt::default());
        let options = Options {
            mode: Mode::Leaf,
            bias: Bias::Static,
            predictors,
            receipt: Some(receipt.clone()),
            ..Default::default()
        };
        let tree = learn(&views, 5, 64, false, &headers, &options).unwrap();
        for (channel, offset) in [(0, 37), (1, 0), (2, -900), (3, 0), (4, 123456)] {
            let mut properties = [0; 16];
            properties[0] = channel;
            let (_, predictor, actual, multiplier) = tree.lookup(&properties).unwrap();
            assert_eq!(
                (predictor, actual, multiplier),
                (Predictor::Zero, offset, 1)
            );
        }
        assert_eq!(receipt.snapshot().pixels, 48);
    }
}

#[test]
fn receipt_names_match_property_ids_and_negative_horizontal_semantics() {
    let pixels = [10, 30, 50, 7, -5, 100];
    let view = view(&pixels, 3, 2, 3);
    for mode in [Mode::Gradient12, Mode::Directional12] {
        let receipt = Arc::new(Receipt::default());
        let options = Options {
            mode,
            receipt: Some(receipt.clone()),
            ..Default::default()
        };
        let tree = learn(&[view], 1, 64, false, &[], &options).unwrap();
        let snapshot = receipt.snapshot();
        match tree.nodes()[0] {
            Node::Split { property: 10, .. } => {
                assert_eq!(snapshot.vertical, 1);
                assert_eq!(snapshot.horizontal, 0);
            }
            Node::Split { property: 11, .. } => {
                assert_eq!(snapshot.vertical, 0);
                assert_eq!(snapshot.horizontal, 1);
            }
            _ => panic!("fixture must produce a directional split"),
        }
    }
}
