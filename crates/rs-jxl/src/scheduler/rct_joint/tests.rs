use super::*;
use crate::{
    PixelLayout,
    decode::{self, DecodeOptions},
};

fn rgba16() -> Image {
    Image::new(
        4,
        3,
        PixelLayout::Rgba,
        PixelData::U16(
            (0..12u16)
                .flat_map(|i| {
                    [
                        i.wrapping_mul(6553),
                        65535 - i * 3071,
                        i * 4567,
                        if i % 3 == 0 { 0 } else { 65535 },
                    ]
                })
                .collect(),
        ),
    )
    .unwrap()
}

#[test]
fn all_42_use_exact_permutation_and_floor_arithmetic_preserving_alpha() {
    // Independent 16-bit-domain arithmetic oracle from enc_rct.cc. Widened
    // arithmetic is safe here; negative halves require floor, not truncation.
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 2, 1],
        [1, 0, 2],
        [2, 1, 0],
    ];
    let image = rgba16();
    let (source, _) = sample(&image, 8192);
    for rct in 0..42 {
        let mut actual = source.clone();
        transforms::rct_forward(&mut actual, 0, rct).unwrap();
        for i in 0..source[0].len() {
            let [a, b, c] = ORDERS[(rct / 7) as usize].map(|j| i64::from(source[j][i]));
            let out = match rct % 7 {
                0 => [a, b, c],
                1 => [a, b, c - a],
                2 => [a, b - a, c],
                3 => [a, b - a, c - a],
                4 => [a, b - (a + c).div_euclid(2), c],
                5 => [a, b - (a + c).div_euclid(2), c - a],
                6 => {
                    let co = a - c;
                    let tmp = c + co.div_euclid(2);
                    let cg = b - tmp;
                    [tmp + cg.div_euclid(2), co, cg]
                }
                _ => unreachable!(),
            };
            assert_eq!(
                [actual[0][i], actual[1][i], actual[2][i]].map(i64::from),
                out,
                "RCT {rct}"
            );
        }
        assert_eq!(actual[3], source[3]);
    }
}

#[test]
fn fitting_image_is_one_contiguous_tile_without_duplicate_samples() {
    let image = Image::new(
        40,
        40,
        PixelLayout::Rgba,
        PixelData::U16((0..6400u16).map(|i| i.wrapping_mul(113)).collect()),
    )
    .unwrap();
    for budget in [1600, 8192] {
        let (planes, tiles) = sample(&image, budget);
        assert_eq!(planes.len(), image.channel_count());
        assert_eq!(tiles.len(), 1);
        let tile = &tiles[0];
        assert_eq!(
            (tile.x, tile.y, tile.width, tile.height, tile.offset),
            (0, 0, 40, 40, 0)
        );
        for (c, plane) in planes.iter().enumerate() {
            assert_eq!(
                plane.len(),
                image.width() as usize * image.height() as usize
            );
            for (i, &value) in plane.iter().enumerate() {
                assert_eq!(value, image.modular_sample(c, i));
            }
        }
    }
    let (planes, _) = sample(&image, 1599);
    assert!(planes.iter().all(|plane| plane.len() <= 1599));
}

#[test]
fn screen_is_bounded_complete_and_deterministic() {
    let image = rgba16();
    let before = image.clone();
    let settings = RctJointOptions {
        shortlist: 3,
        max_sample_pixels: 81,
    };
    let a = screen_rcts(&image, &settings).unwrap();
    let b = screen_rcts(&image, &settings).unwrap();
    assert_eq!(image, before);
    assert!(a.sampled_pixels <= settings.max_sample_pixels);
    assert_eq!(a.sampled_channels, 4);
    assert_eq!(a.scores.len(), 42);
    assert_eq!(a.shortlisted_rcts, b.shortlisted_rcts);
    assert_eq!(a.shortlisted_rcts.len(), 3);
    for (index, (a, b)) in a.scores.iter().zip(&b.scores).enumerate() {
        assert_eq!(a.rct, index as u32);
        assert_eq!(a.proxy_bits, b.proxy_bits);
        assert_eq!(a.native_bits, a.proxy_bits[0]);
        assert!(a.proxy_bits[2] <= a.proxy_bits[1]);
        assert!(a.proxy_bits[1] <= a.proxy_bits[0]);
    }
}

#[test]
fn unsupported_domains_are_errors_not_sample_narrowing() {
    for (layout, pixels) in [
        (PixelLayout::Gray, PixelData::U8(vec![0])),
        (PixelLayout::Rgb, PixelData::U32(vec![0, 65536, 1])),
        (PixelLayout::Rgb, PixelData::F32(vec![0.0, 1.0, 0.5])),
    ] {
        let image = Image::new(1, 1, layout, pixels).unwrap();
        assert!(screen_rcts(&image, &RctJointOptions::default()).is_err());
    }
    for settings in [
        RctJointOptions {
            shortlist: 0,
            ..Default::default()
        },
        RctJointOptions {
            shortlist: 42,
            ..Default::default()
        },
        RctJointOptions {
            max_sample_pixels: 8,
            ..Default::default()
        },
    ] {
        assert!(screen_rcts(&rgba16(), &settings).is_err());
    }
}

#[test]
fn previous_channel_proxy_changes_scores_on_correlated_residuals() {
    let mut state = 1234567u32;
    let pixels = (0..64 * 32)
        .flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = if state & 1 == 0 { 0u16 } else { 60000 };
            [value, value, value]
        })
        .collect();
    let image = Image::new(64, 32, PixelLayout::Rgb, PixelData::U16(pixels)).unwrap();
    let screen = screen_rcts(&image, &RctJointOptions::default()).unwrap();
    let identity = &screen.scores[0];
    assert!(identity.proxy_bits[1] < identity.proxy_bits[0]);
    assert!(identity.proxy_bits[2] <= identity.proxy_bits[1]);
}

#[test]
fn hidden_rgba16_and_extra_plane_roundtrip_with_full_cost_accounting() {
    use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
    let image = rgba16();
    let mut metadata = image.metadata().clone();
    metadata
        .extra_channels
        .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
    let extra = PixelData::U16((0..12).map(|i| 65535 - i * 1234).collect());
    let image = image.with_metadata(metadata, vec![extra.clone()]).unwrap();
    let result = run_rct_joint(
        &image,
        &EncodeOptions::default(),
        &RctJointOptions {
            shortlist: 1,
            max_sample_pixels: 81,
        },
    )
    .unwrap();
    let decoded = decode::decode(&result.encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(&decoded.frames[0].pixels, image.pixels());
    assert_eq!(decoded.frames[0].extra_planes.last(), Some(&extra));
    assert_eq!(result.trials.len(), 7);
    assert_eq!(result.trials[1].reused_from, Some(0));
    assert_eq!(result.trials[1].seconds, 0.0);
    assert_eq!(
        result.trials[result.winner_trial].bytes,
        Some(result.encoded.len())
    );
    assert!(
        result
            .trials
            .iter()
            .filter_map(|t| t.bytes)
            .all(|n| n >= result.encoded.len())
    );
    assert_eq!(
        result.timing.encode_seconds,
        result.trials.iter().map(|t| t.seconds).sum::<f64>()
    );
    assert!(
        result.timing.total_seconds >= result.timing.screen_seconds + result.timing.encode_seconds
    );
    for trial in &result.trials {
        assert_eq!(trial.params.patches, Override::Off);
        assert!(trial.options.patches.is_none());
        assert!(trial.options.tile_patches.is_none());
        assert!(trial.options.source_patches.is_none());
        assert!(trial.options.source_patch_reference.is_none());
    }
    for previous in 0..=2 {
        let control = &result.trials[1 + previous];
        assert_eq!(control.options.modular.previous_channels, previous);
        let sampling = control.options.modular.sampling.as_ref().unwrap();
        assert_eq!(sampling.previous_channels, previous);
        for p in 16..16 + 4 * previous {
            assert!(sampling.properties.contains(&p));
        }
    }
}

#[test]
fn backend_consumes_first_and_second_previous_channel_values_not_just_a_flag() {
    use crate::modular::{
        predictor::Predictor,
        tree::{Node, Tree},
    };
    let image = Image::new(
        32,
        1,
        PixelLayout::Rgb,
        PixelData::U16(
            (0..32u16)
                .flat_map(|i| {
                    [
                        if i % 2 == 0 { 1000 } else { 60000 },
                        if i % 3 == 0 { 50000 } else { 2000 },
                        i * 1871,
                    ]
                })
                .collect(),
        ),
    )
    .unwrap();
    for previous in 1..=2u8 {
        let mut params = CandidateParams::default().backend(9).unwrap();
        params.max_properties = previous;
        params.rct = Some(0);
        let mut options = adapter::project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        assert!(
            options
                .modular
                .sampling
                .as_ref()
                .unwrap()
                .properties
                .contains(&(17 + 4 * (previous as usize - 1)))
        );
        options.source_patches = None;
        options.source_patch_reference = None;
        options.global_palette = false;
        options.modular.palette_policy = None;
        options.modular.palette_colors = None;
        options.modular.rct = None;
        // Force an observable property lookup independent of whether the
        // learner happens to choose this feature on such a tiny fixture.
        options.modular.custom_tree = Some(
            Tree::new(vec![
                Node::Split {
                    property: 17 + 4 * (previous as usize - 1),
                    threshold: 30000,
                    greater: 1,
                    less_equal: 2,
                },
                Node::Leaf {
                    predictor: Predictor::Zero,
                    offset: 65535,
                    multiplier: 1,
                },
                Node::Leaf {
                    predictor: Predictor::Zero,
                    offset: 0,
                    multiplier: 1,
                },
            ])
            .unwrap(),
        );
        let encoded = crate::encode::encode_serial(&image, &options).unwrap();
        let decoded = decode::decode(&encoded, &DecodeOptions::default()).unwrap();
        assert_eq!(
            &decoded.frames[0].pixels,
            image.pixels(),
            "previousChannels={previous}"
        );
    }
}
