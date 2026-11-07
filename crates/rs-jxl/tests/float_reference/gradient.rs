use super::*;
use rs_jxl::scheduler::{CandidateParams, adapter::project_native};

fn image(width: u32, height: u32, layout: PixelLayout) -> Image {
    let pattern = [0., 0., 0., 0., 0., 0., 0., 2., 1e-10f32];
    Image::new(
        width,
        height,
        layout,
        PixelData::F32(
            (0..width as usize * height as usize * layout.channels())
                .map(|i| pattern[i % pattern.len()])
                .collect(),
        ),
    )
    .unwrap()
}

fn gradient_leaf(offset: i32) -> Node {
    Node::Leaf {
        predictor: Predictor::Gradient,
        offset,
        multiplier: 1,
    }
}

#[test]
#[ignore = "requires official djxl; low-effort float32 gradient context overflow"]
fn finite_low_efforts_use_compatible_gradient_contexts() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        let image = image(16, 33, layout);
        for effort in [1, 2, 3, 7, 9, 11] {
            let params = CandidateParams::default().backend(effort).unwrap();
            let options = project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .unwrap();
            check_image_with_options(
                &image,
                &options,
                &format!("gradient/effort={effort}/{layout:?}"),
                &decoders,
                &scratch,
            );
        }
    }
}

#[test]
#[ignore = "requires official djxl; negative float32 gradient properties and signed zero"]
fn finite_negative_gradient_words_at_low_efforts() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let pattern = [-0., -2., 0., -2., 1e-10f32, -0.];
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        let image = Image::new(
            17,
            33,
            layout,
            PixelData::F32(
                (0..17 * 33 * layout.channels())
                    .map(|i| pattern[i % pattern.len()])
                    .collect(),
            ),
        )
        .unwrap();
        for effort in [1, 2] {
            let params = CandidateParams::default().backend(effort).unwrap();
            let options = project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .unwrap();
            check_image_with_options(
                &image,
                &options,
                &format!("gradient/negative/effort={effort}/{layout:?}"),
                &decoders,
                &scratch,
            );
        }
    }
}

#[test]
#[ignore = "requires official djxl; gradient LUT limits, generic paths and group specialization"]
fn finite_gradient_tree_dispatch_matches_official_contexts() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    let image = image(129, 11, PixelLayout::Rgb);
    for shared_tree in [false, true] {
        for threshold in [-8194, -8193, -8192, -1025, -1024, 0, 1023, 1024, 8190, 8191] {
            for offset in [0, 1] {
                let tree = Tree::new(vec![
                    Node::Split {
                        property: 0,
                        threshold: 0,
                        greater: 1,
                        less_equal: 2,
                    },
                    Node::Split {
                        property: 1,
                        threshold: 0,
                        greater: 3,
                        less_equal: 4,
                    },
                    Node::Leaf {
                        predictor: Predictor::Zero,
                        offset: 0,
                        multiplier: 1,
                    },
                    Node::Split {
                        property: 9,
                        threshold,
                        greater: 5,
                        less_equal: 6,
                    },
                    Node::Split {
                        property: 9,
                        threshold,
                        greater: 7,
                        less_equal: 8,
                    },
                    gradient_leaf(offset),
                    gradient_leaf(offset),
                    gradient_leaf(offset),
                    gradient_leaf(offset),
                ])
                .unwrap();
                for coding in [Coding::Ans, Coding::Prefix] {
                    let options = EncodeOptions {
                        threads: 4,
                        group_size_shift: 0,
                        shared_tree,
                        modular: ModularOptions {
                            rct: None,
                            lz77: false,
                            custom_tree: Some(tree.clone()),
                            entropy: coding,
                            parallel_tokens: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    check_image_with_options(
                        &image,
                        &options,
                        &format!(
                            "gradient/threshold={threshold}/offset={offset}/shared={shared_tree}/{coding:?}"
                        ),
                        &decoders,
                        &scratch,
                    );
                    let data = encode(&image, &options).unwrap();
                    let decoded = rs_jxl::decode::decode(&data, &Default::default()).unwrap();
                    assert_eq!(&decoded.frames[0].pixels, image.pixels());
                }
            }
        }
    }
}

#[test]
#[ignore = "requires official djxl; short-channel dispatch and parallel tokenization"]
fn finite_gradient_small_channels_and_parallel_tokens() {
    let decoders = decoder_paths();
    let scratch = ScratchDir::new();
    // 64 and 65 samples straddle the decoder's small-channel shortcut; the
    // final image has enough samples and channels for parallel tokenization.
    for (width, height) in [(8, 8), (13, 5), (256, 128)] {
        let image = image(width, height, PixelLayout::Rgb);
        for threshold in [0, 7000] {
            let tree = Tree::new(vec![
                Node::Split {
                    property: 9,
                    threshold,
                    greater: 1,
                    less_equal: 2,
                },
                gradient_leaf(0),
                gradient_leaf(0),
            ])
            .unwrap();
            let options = EncodeOptions {
                threads: 4,
                modular: ModularOptions {
                    rct: None,
                    lz77: false,
                    custom_tree: Some(tree),
                    parallel_tokens: true,
                    ..Default::default()
                },
                ..Default::default()
            };
            check_image_with_options(
                &image,
                &options,
                &format!("gradient/size={width}x{height}/threshold={threshold}"),
                &decoders,
                &scratch,
            );
            let data = encode(&image, &options).unwrap();
            let decoded = rs_jxl::decode::decode(&data, &Default::default()).unwrap();
            assert_eq!(&decoded.frames[0].pixels, image.pixels());
        }
    }
}
