// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;
use crate::headers::bit_depth::BitDepth;
use crate::headers::frame_header::BlendingMode;

fn headers(bits: u32, size: (usize, usize)) -> (FileHeader, FrameHeader) {
    let (mut file, mut header, _) = crate::tests::decode::read_headers_and_toc(include_bytes!(
        "../../resources/test/basic.jxl"
    ))
    .unwrap();
    file.size = crate::headers::Size::from_dimensions(size.0 as u32, size.1 as u32);
    file.image_metadata.bit_depth = BitDepth::integer_samples(bits);
    file.image_metadata.xyb_encoded = false;
    file.image_metadata.color_encoding.color_space = ColorSpace::RGB;
    file.image_metadata.extra_channel_info.clear();
    header.encoding = Encoding::Modular;
    header.do_ycbcr = false;
    header.width = size.0 as u32;
    header.height = size.1 as u32;
    header.x0 = 0;
    header.y0 = 0;
    header.upsampling = 1;
    header.ec_upsampling.clear();
    header.num_extra_channels = 0;
    header.ec_blending_info.clear();
    header.restoration_filter.gab = false;
    header.restoration_filter.epf_iters = 0;
    (file, header)
}

fn alpha(bits: u32, associated: bool) -> ExtraChannelInfo {
    ExtraChannelInfo::new(
        false,
        ExtraChannel::Alpha,
        BitDepth::integer_samples(bits),
        0,
        String::new(),
        associated,
        None,
        None,
    )
}

fn blend(mode: PatchBlendMode) -> PatchBlending {
    PatchBlending {
        mode,
        alpha_channel: 0,
        clamp: false,
    }
}

#[test]
fn native_wide_samples_never_round_through_f32() {
    for bits in 17..=31 {
        let (file, _) = headers(bits, (6, 1));
        let maximum = (1i64 << bits) - 1;
        let samples = vec![0, 1, maximum / 2, maximum / 2 + 1, maximum - 1, maximum];
        let image = IntegerImage {
            size: (6, 1),
            planes: vec![samples.clone(); 3],
        };
        let output = image.output(&file, Orientation::Identity).unwrap();
        assert_eq!(
            output.planes[0].samples,
            samples.iter().map(|&v| v as u32).collect::<Vec<_>>()
        );
        assert_eq!(output.planes[0].bits_per_sample, bits);
    }
}

#[test]
fn all_orientations_preserve_wide_words_and_extra_depth() {
    let (mut file, _) = headers(31, (2, 3));
    file.image_metadata
        .extra_channel_info
        .push(alpha(17, false));
    let original = vec![
        16_777_217,
        16_777_219,
        1_073_741_823,
        1_073_741_824,
        2_147_483_646,
        2_147_483_647,
    ];
    let mut image = IntegerImage {
        size: (2, 3),
        planes: vec![original.clone(); 3],
    };
    image.planes.push(vec![0, 1, 2, 3, 131_070, 131_071]);
    for orientation in [
        Orientation::Identity,
        Orientation::FlipHorizontal,
        Orientation::Rotate180,
        Orientation::FlipVertical,
        Orientation::Transpose,
        Orientation::Rotate90Cw,
        Orientation::AntiTranspose,
        Orientation::Rotate90Ccw,
    ] {
        let output = image.output(&file, orientation).unwrap();
        assert_eq!((output.width, output.height), orientation.map_size((2, 3)));
        assert_eq!(output.planes[3].bits_per_sample, 17);
        for (index, &sample) in original.iter().enumerate() {
            let (x, y) = orientation.display_pixel((index % 2, index / 2), (2, 3));
            assert_eq!(
                output.planes[0].samples[y * output.width + x],
                sample as u32
            );
            assert_eq!(
                output.planes[3].samples[y * output.width + x],
                image.planes[3][index] as u32
            );
        }
    }
}

#[test]
fn raw_output_rejects_out_of_range_instead_of_clamping() {
    let (file, _) = headers(31, (1, 1));
    for invalid_sample in [-1, 1i64 << 31] {
        let image = IntegerImage {
            size: (1, 1),
            planes: vec![vec![invalid_sample]; 3],
        };
        assert!(matches!(
            image.output(&file, Orientation::Identity),
            Err(Error::NativeIntegerUnsupported(_))
        ));
    }
}

#[test]
fn add_keeps_signed_residual_and_checked_overflow() {
    assert_eq!(
        combine(
            &[-1, 0, 0],
            &[16_777_218, 0, 0],
            0,
            &blend(PatchBlendMode::Add),
            &[]
        )
        .unwrap(),
        16_777_217
    );
    assert!(matches!(
        combine(&[i64::MAX], &[1], 0, &blend(PatchBlendMode::Add), &[]),
        Err(Error::NativeIntegerInvalid(_))
    ));
}

#[test]
fn integer_patches_use_exact_reference_words_and_stream_order() {
    use crate::features::patches::{PatchPosition, PatchReferencePosition};
    let source = Arc::new(IntegerImage {
        size: (2, 1),
        planes: vec![vec![16_777_218, 2_147_483_647]; 3],
    });
    let references = [
        None,
        None,
        None,
        Some(IntegerReference {
            image: source,
            before_color_transform: true,
        }),
    ];
    let mut image = IntegerImage {
        size: (2, 1),
        planes: vec![vec![-1, 0]; 3],
    };
    let mut patches = PatchesDictionary::new(0);
    patches.ref_positions.push(PatchReferencePosition {
        reference: 3,
        x0: 0,
        y0: 0,
        xsize: 2,
        ysize: 1,
    });
    patches.positions.push(PatchPosition {
        x: 0,
        y: 0,
        ref_pos_idx: 0,
    });
    patches.blendings.push(blend(PatchBlendMode::Add));
    apply_patches(&mut image, &patches, &references, &[]).unwrap();
    assert_eq!(image.planes[0], [16_777_217, 2_147_483_647]);

    // A subsequent overlapping replacement must read the immutable reference,
    // not the reconstructed destination of the preceding patch.
    patches.positions.push(PatchPosition {
        x: 0,
        y: 0,
        ref_pos_idx: 0,
    });
    patches.blendings.push(blend(PatchBlendMode::Replace));
    let mut image = IntegerImage {
        size: (2, 1),
        planes: vec![vec![-1, 0]; 3],
    };
    apply_patches(&mut image, &patches, &references, &[]).unwrap();
    assert_eq!(image.planes[0], [16_777_218, 2_147_483_647]);
}

#[test]
fn binary_mask_uses_alpha_depth_not_color_depth() {
    for alpha_bits in [1, 8, 17, 31] {
        let max = (1i64 << alpha_bits) - 1;
        let extras = [alpha(alpha_bits, false)];
        let under = [16_777_217, 16_777_219, 2_147_483_647, max];
        let over = [1_073_741_825, 1_073_741_827, 1, 0];
        for c in 0..4 {
            assert_eq!(
                combine(
                    &over,
                    &under,
                    c,
                    &blend(PatchBlendMode::BlendBelow),
                    &extras
                )
                .unwrap(),
                under[c]
            );
        }
        let opaque = [over[0], over[1], over[2], max];
        for c in 0..4 {
            assert_eq!(
                combine(
                    &opaque,
                    &under,
                    c,
                    &blend(PatchBlendMode::BlendBelow),
                    &extras
                )
                .unwrap(),
                opaque[c]
            );
        }
    }
}

#[test]
fn transparent_rgb_and_premultiplied_semantics_are_distinct() {
    let mask = blend(PatchBlendMode::BlendBelow);
    let top = [16_777_217, 0, 0, 0];
    let bottom = [19, 0, 0, 0];
    assert_eq!(
        combine(&top, &bottom, 0, &mask, &[alpha(8, false)]).unwrap(),
        0
    );
    assert_eq!(
        combine(&top, &bottom, 0, &mask, &[alpha(8, true)]).unwrap(),
        16_777_236
    );
    let fractional = [1, 2, 3, 127];
    assert!(matches!(
        combine(&fractional, &bottom, 0, &mask, &[alpha(8, false)]),
        Err(Error::NativeIntegerUnsupported(_))
    ));
}

#[test]
fn color_blend_updates_alpha_after_its_explicit_mode() {
    let mut result = [0; 4];
    blend_pixel(
        &[4, 5, 6, 0],
        &[7, 8, 9, 255],
        &[
            blend(PatchBlendMode::BlendBelow),
            blend(PatchBlendMode::None),
        ],
        &[alpha(8, false)],
        &mut result,
    )
    .unwrap();
    assert_eq!(result, [7, 8, 9, 255]);
}

#[test]
fn cropped_replace_preserves_off_canvas_background_and_extra_source() {
    let (mut file, mut header) = headers(31, (3, 2));
    file.image_metadata.extra_channel_info.push(alpha(8, false));
    header.blending_info.mode = BlendingMode::Replace;
    header.blending_info.source = 1;
    let mut ec_blend = header.blending_info.clone();
    ec_blend.source = 2;
    header.ec_blending_info = vec![ec_blend];
    header.x0 = -1;
    header.y0 = 1;
    let first = Arc::new(IntegerImage {
        size: (3, 2),
        planes: vec![vec![16_777_217; 6]; 4],
    });
    let second = Arc::new(IntegerImage {
        size: (3, 2),
        planes: vec![vec![127; 6]; 4],
    });
    let references = [
        None,
        Some(IntegerReference {
            image: first,
            before_color_transform: false,
        }),
        Some(IntegerReference {
            image: second,
            before_color_transform: false,
        }),
        None,
    ];
    let layer = IntegerImage {
        size: (2, 1),
        planes: vec![vec![0, 2_147_483_647], vec![1, 2], vec![3, 4], vec![255, 0]],
    };
    let output = coalesce(&layer, &header, &file, &references).unwrap();
    assert_eq!(
        output.planes[0],
        [
            16_777_217,
            16_777_217,
            16_777_217,
            2_147_483_647,
            16_777_217,
            16_777_217
        ]
    );
    assert_eq!(output.planes[3], [127, 127, 127, 0, 127, 127]);
}

#[test]
fn capture_requires_every_channel_group_and_is_one_shot() {
    let (file, header) = headers(31, (2, 1));
    let capture = NativeIntegerFrameBuffer::new(&header, &file, false).unwrap();
    assert!(capture.consume().is_err());
    for c in 0..3 {
        let mut input = Image::<i32>::new((2, 1)).unwrap();
        input
            .row_mut(0)
            .copy_from_slice(&[16_777_217, 2_147_483_647]);
        capture.write_group(&header, c, 0, input).unwrap();
    }
    assert_eq!(
        capture.consume().unwrap().planes[0],
        [16_777_217, 2_147_483_647]
    );
    assert!(capture.consume().is_err());
}

#[test]
fn unsupported_render_paths_are_explicit() {
    let (file, mut header) = headers(31, (1, 1));
    header.upsampling = 2;
    assert!(matches!(
        NativeIntegerFrameBuffer::new(&header, &file, false),
        Err(Error::NativeIntegerUnsupported(_))
    ));
    header.upsampling = 1;
    header.restoration_filter.gab = true;
    assert!(matches!(
        NativeIntegerFrameBuffer::new(&header, &file, false),
        Err(Error::NativeIntegerUnsupported(_))
    ));
    header.restoration_filter.gab = false;
    header.encoding = Encoding::VarDCT;
    assert!(matches!(
        NativeIntegerFrameBuffer::new(&header, &file, false),
        Err(Error::NativeIntegerUnsupported(_))
    ));
}
