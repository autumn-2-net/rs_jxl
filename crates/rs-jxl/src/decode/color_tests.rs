use super::*;
use jxl_decoder::api::{
    JxlBasicInfo, JxlBitDepth, JxlBlendInfo, JxlBlendMode, JxlExtraChannel, JxlFrameHeader,
    JxlLayerInfo, Orientation, ToneMapping,
};

fn profile(name: &str) -> JxlColorProfile {
    ColorProfile::parse(name).unwrap().as_jxl()
}

fn target(name: &str) -> ColorOptions {
    ColorOptions {
        output_profile: Some(name.parse().unwrap()),
        ..Default::default()
    }
}

fn extra(ec_type: ExtraChannel) -> JxlExtraChannel {
    JxlExtraChannel {
        ec_type,
        alpha_associated: false,
        bit_depth: JxlBitDepth::Int { bits_per_sample: 8 },
        dim_shift: 0,
        name: String::new(),
        spot_color: None,
        cfa_channel: None,
    }
}

fn image(layout: PixelLayout, values: &[f32], source: &str) -> DecodedImage {
    let count = values.len() / layout.channels();
    let extras = if layout.has_alpha() {
        vec![extra(ExtraChannel::Alpha)]
    } else {
        vec![]
    };
    let planes = if layout.has_alpha() {
        vec![PixelData::F32(
            values
                .chunks_exact(layout.channels())
                .map(|p| p[layout.channels() - 1])
                .collect(),
        )]
    } else {
        vec![]
    };
    DecodedImage {
        info: JxlBasicInfo {
            size: (count, 1),
            bit_depth: JxlBitDepth::Int { bits_per_sample: 8 },
            orientation: Orientation::Identity,
            extra_channels: extras,
            animation: None,
            uses_original_profile: true,
            tone_mapping: ToneMapping {
                intensity_target: 255.0,
                min_nits: 0.0,
                relative_to_max_display: false,
                linear_below: 0.0,
            },
            preview_size: None,
        },
        frames: vec![DecodedFrame {
            index: 0,
            width: count,
            height: 1,
            layout,
            pixels: PixelData::F32(values.to_vec()),
            extra_planes: planes,
            duration_ticks: 0,
            name: "frame".into(),
            complete: true,
            header: JxlFrameHeader {
                name: "frame".into(),
                duration: None,
                size: (count, 1),
                duration_ticks: 0,
                timecode: None,
                is_last: true,
                is_preview: false,
                layer_info: JxlLayerInfo {
                    have_crop: false,
                    crop_origin: (0, 0),
                    size: (count, 1),
                    blend_info: JxlBlendInfo {
                        mode: JxlBlendMode::Replace,
                        source: 0,
                        alpha_channel: 0,
                        clamp: false,
                    },
                    extra_channel_blend_info: vec![],
                    save_as_reference: None,
                    save_before_color_transform: false,
                },
            },
        }],
        preview: None,
        embedded_profile: Some(profile(source)),
        output_profile: Some(profile(source)),
        output_intensity_target: 255.0,
        boxes: vec![],
        complete: true,
        bytes_read: 0,
    }
}

fn values(image: &DecodedImage) -> &[f32] {
    f32_pixels(&image.frames[0].pixels).unwrap()
}

fn close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn descriptors_aliases_custom_primaries_and_gamma() {
    assert_eq!(profile("sRGB"), profile("RGB_D65_SRG_Rel_SRG"));
    assert_eq!(profile("linear"), profile("RGB_D65_SRG_Rel_Lin"));
    assert_eq!(profile("Rec2100PQ"), profile("RGB_D65_202_Rel_PeQ"));
    for name in [
        "DisplayP3",
        "Adobe98",
        "ProPhoto",
        "Rec2100HLG",
        "Gra_D65_Rel_SRG",
        "Gra_D50_Abs_g0.5",
        "RGB_0.3127;0.329_0.64;0.33;0.3;0.6;0.15;0.06_Sat_709",
    ] {
        let parsed = ColorProfile::parse(name).unwrap();
        Endpoint::new(&parsed.as_jxl(), &DecodeLimits::default(), true).unwrap();
    }
    let ColorProfile::Encoding(JxlColorEncoding::RgbColorSpace {
        transfer_function: JxlTransferFunction::Gamma(g),
        ..
    }) = ColorProfile::parse("Adobe98").unwrap()
    else {
        panic!("gamma expected");
    };
    close(g, 256.0 / 563.0, 1e-7);
}

#[test]
fn descriptors_reject_malformed_or_unsupported_values() {
    for invalid in [
        "",
        "sRGB ",
        "RGB_D65_SRG_Rel",
        "RGB_D65_SRG_Rel_SRG_extra",
        "RGB_D65_SRG_bad_SRG",
        "XYB_Per",
        "Gra_D65_SRG_Rel_SRG",
        "RGB_Cst_SRG_Rel_Lin",
        "RGB_NaN;0.3_SRG_Rel_Lin",
        "RGB_0.3;0_SRG_Rel_Lin",
        "RGB_D65_SRG_Rel_g0",
        "RGB_D65_SRG_Rel_g2",
        "RGB_D65_SRG_Rel_gNaN",
        "RGB_D65_SRG_Rel_ginf",
        "RGB_D65_SRG_Rel_g0.5junk",
        "RGB_D65_SRG_Rel_",
    ] {
        assert!(ColorProfile::parse(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn rendering_intents_are_not_forced_to_relative() {
    for (token, expected) in [
        ("Per", CmsIntent::Perceptual),
        ("Rel", CmsIntent::RelativeColorimetric),
        ("Sat", CmsIntent::Saturation),
        ("Abs", CmsIntent::AbsoluteColorimetric),
    ] {
        let destination = profile(&format!("RGB_D65_SRG_{token}_SRG"));
        let endpoint = Endpoint::new(&destination, &DecodeLimits::default(), true).unwrap();
        assert_eq!(endpoint.cms.rendering_intent, expected);
    }
}

#[test]
fn defaults_do_not_validate_allocate_or_touch_any_bits() {
    let nan = f32::from_bits(0x7fc12345);
    let mut image = image(PixelLayout::Rgba, &[nan, -0.0, 7.0, 0.0], "sRGB");
    image.output_profile = None;
    image.frames[0].width = usize::MAX;
    let original = image.frames[0].pixels.clone();
    let extras = image.frames[0].extra_planes.clone();
    let options = ColorOptions::default();
    assert!(options.is_noop());
    assert!(!options.needs_float());
    let limits = DecodeLimits {
        max_output_bytes: 0,
        max_samples: 0,
        ..Default::default()
    };
    process_image(&mut image, &options, &limits).unwrap();
    assert_eq!(image.frames[0].pixels, original);
    assert_eq!(image.frames[0].extra_planes, extras);
    image.frames[0].pixels = PixelData::U16(vec![1, 1023, 1024, 4095]);
    let original = image.frames[0].pixels.clone();
    process_image(&mut image, &options, &limits).unwrap();
    assert_eq!(image.frames[0].pixels, original);
}

#[test]
fn output_profile_is_source_and_embedded_profile_stays_unchanged() {
    let mut image = image(PixelLayout::Rgb, &[0.5; 3], "sRGB");
    image.embedded_profile = Some(profile("linear"));
    let embedded = image.embedded_profile.clone();
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    for &value in values(&image) {
        close(value, 0.21404114, 2e-6);
    }
    assert_eq!(image.output_profile, Some(profile("linear")));
    assert_eq!(image.embedded_profile, embedded);
    assert_eq!(
        image.info.bit_depth,
        JxlBitDepth::Int { bits_per_sample: 8 }
    );
}

#[test]
fn straight_alpha_and_extra_planes_are_not_color_managed() {
    let alpha_nan = f32::from_bits(0x7fc12345);
    let mut image = image(PixelLayout::Rgba, &[0.5, 0.25, 0.75, alpha_nan], "sRGB");
    image.info.extra_channels.push(extra(ExtraChannel::Depth));
    image.frames[0]
        .extra_planes
        .push(PixelData::U16(vec![1234]));
    let extras = image.frames[0].extra_planes.clone();
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    assert_eq!(values(&image)[3].to_bits(), alpha_nan.to_bits());
    assert_eq!(image.frames[0].extra_planes, extras);
    close(values(&image)[0], 0.21404114, 2e-6);
}

#[test]
fn associated_alpha_unassociates_reassociates_and_keeps_hidden_color() {
    let mut image = image(
        PixelLayout::Rgba,
        &[0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, -0.0],
        "sRGB",
    );
    image.info.extra_channels[0].alpha_associated = true;
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    close(values(&image)[0], 0.21404114 * 0.5, 2e-6);
    close(values(&image)[4], 0.21404114, 2e-6);
    assert_eq!(values(&image)[7].to_bits(), (-0.0f32).to_bits());
    assert!(image.info.extra_channels[0].alpha_associated);
}

#[test]
fn finite_extended_range_is_not_clamped() {
    let mut image = image(PixelLayout::Rgb, &[-0.5, 1.5, 2.0], "sRGB");
    process_image(
        &mut image,
        &target("RGB_D65_202_Rel_Lin"),
        &DecodeLimits::default(),
    )
    .unwrap();
    assert!(values(&image).iter().any(|v| *v > 1.0));
    // Same primaries exercise signed transfer behavior without a gamut change.
    let mut image = image_for_extended_sign();
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    close(values(&image)[0], -0.21404114, 2e-6);
    assert!(values(&image)[1] > 1.0);
}

#[test]
fn structured_wide_gamut_matrix_keeps_negative_and_above_one_components() {
    let mut image = image(PixelLayout::Rgb, &[1.0, 0.0, 0.0], "RGB_D65_202_Rel_Lin");
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    let rgb = values(&image);
    close(rgb[0], 1.6605, 0.002);
    close(rgb[1], -0.1246, 0.002);
    close(rgb[2], -0.0182, 0.002);
}

#[test]
fn structured_hdr_matrix_keeps_values_above_mastering_peak() {
    let mut image = image(PixelLayout::Rgb, &[pq_encode(10000.0); 3], "Rec2100PQ");
    image.info.tone_mapping.intensity_target = 1000.0;
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    for &sample in values(&image) {
        close(sample, 10.0, 0.003);
    }
}

#[test]
fn structured_matrix_absolute_intent_does_not_adapt_source_white() {
    let mut relative = image(PixelLayout::Rgb, &[1.0; 3], "RGB_D50_SRG_Rel_Lin");
    let mut absolute = image(PixelLayout::Rgb, &[1.0; 3], "RGB_D50_SRG_Rel_Lin");
    process_image(&mut relative, &target("linear"), &DecodeLimits::default()).unwrap();
    process_image(
        &mut absolute,
        &target("RGB_D65_SRG_Abs_Lin"),
        &DecodeLimits::default(),
    )
    .unwrap();
    for &sample in values(&relative) {
        close(sample, 1.0, 0.002);
    }
    assert!(
        values(&absolute)
            .iter()
            .zip(values(&relative))
            .any(|(a, b)| (a - b).abs() > 0.05)
    );
}

#[test]
fn opaque_icc_extended_input_returns_error_without_clipping_or_mutation() {
    let mut image = image(PixelLayout::Rgb, &[-0.25, 1.5, 0.5], "linear");
    let original = image.frames[0].pixels.clone();
    let options = ColorOptions {
        output_profile: Some(ColorProfile::Icc(CmsProfile::new_srgb().encode().unwrap())),
        ..Default::default()
    };
    let error = process_image(&mut image, &options, &DecodeLimits::default()).unwrap_err();
    assert!(error.to_string().contains("extended-range ICC input"));
    assert_eq!(image.frames[0].pixels, original);
    assert_eq!(image.output_profile, Some(profile("linear")));
}

#[test]
fn chunk_tail_retains_alpha_and_transforms_last_pixel() {
    let mut samples = vec![0.5; (CHUNK_PIXELS + 1) * 4];
    *samples.last_mut().unwrap() = -0.0;
    let mut image = image(PixelLayout::Rgba, &samples, "sRGB");
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    let tail = &values(&image)[CHUNK_PIXELS * 4..];
    for &sample in &tail[..3] {
        close(sample, 0.21404114, 2e-6);
    }
    assert_eq!(tail[3].to_bits(), (-0.0f32).to_bits());
}

fn image_for_extended_sign() -> DecodedImage {
    image(PixelLayout::Rgb, &[-0.5, 1.5, 2.0], "sRGB")
}

#[test]
fn background_parser_is_strict() {
    assert_eq!(parse_background("white").unwrap(), [1.0; 3]);
    assert_eq!(parse_background("black").unwrap(), [0.0; 3]);
    let rgb = parse_background("#fF8000").unwrap();
    assert_eq!(rgb[0], 1.0);
    close(rgb[1], 128.0 / 255.0, 1e-7);
    assert_eq!(rgb[2], 0.0);
    for value in [
        "#ff",
        "#12345z",
        "#1234567",
        "#123456junk",
        "white ",
        "red",
        "#12345\u{e9}",
    ] {
        assert!(parse_background(value).is_err(), "{value}");
    }
}

#[test]
fn background_blends_after_conversion_and_keeps_original_alpha_plane() {
    let mut image = image(PixelLayout::Rgba, &[0.5, 0.5, 0.5, 0.25], "sRGB");
    let extras = image.frames[0].extra_planes.clone();
    let options = ColorOptions {
        background: Some([1.0; 3]),
        ..target("linear")
    };
    process_image(&mut image, &options, &DecodeLimits::default()).unwrap();
    assert_eq!(image.frames[0].layout, PixelLayout::Rgb);
    for &value in values(&image) {
        close(value, 0.21404114 * 0.25 + 0.75, 2e-6);
    }
    assert_eq!(image.frames[0].extra_planes, extras);
}

#[test]
fn background_uses_encoded_space_and_gray_uses_first_component() {
    let mut image = image(PixelLayout::GrayAlpha, &[0.25, 0.5], "Gra_D65_Rel_SRG");
    let options = ColorOptions {
        background: Some([1.0, 0.0, 0.0]),
        ..Default::default()
    };
    process_image(&mut image, &options, &DecodeLimits::default()).unwrap();
    assert_eq!(image.frames[0].layout, PixelLayout::Gray);
    assert_eq!(values(&image), &[0.625]);
}

#[test]
fn associated_background_is_not_multiplied_twice() {
    let mut image = image(
        PixelLayout::Rgba,
        &[0.2, 0.3, 0.1, 0.5, 0.7, 0.6, 0.5, 0.0],
        "sRGB",
    );
    image.info.extra_channels[0].alpha_associated = true;
    let options = ColorOptions {
        background: Some([1.0; 3]),
        ..Default::default()
    };
    process_image(&mut image, &options, &DecodeLimits::default()).unwrap();
    for (actual, expected) in values(&image).iter().zip([0.7, 0.8, 0.6, 1.0, 1.0, 1.0]) {
        close(*actual, expected, 1e-6);
    }
}

#[test]
fn grayscale_can_expand_to_rgb_and_preserve_alpha() {
    let mut image = image(PixelLayout::GrayAlpha, &[0.5, 0.3], "Gra_D65_Rel_SRG");
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    assert_eq!(image.frames[0].layout, PixelLayout::Rgba);
    for &value in &values(&image)[..3] {
        close(value, 0.21404114, 1e-3);
    }
    assert_eq!(values(&image)[3].to_bits(), 0.3f32.to_bits());
}

#[test]
fn preview_is_processed_and_frame_headers_are_preserved() {
    let mut image = image(PixelLayout::Rgb, &[0.5; 3], "sRGB");
    image.preview = Some(image.frames[0].clone());
    image.preview.as_mut().unwrap().header.is_preview = true;
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    assert_eq!(
        image.preview.as_ref().unwrap().pixels,
        image.frames[0].pixels
    );
    assert!(image.preview.as_ref().unwrap().header.is_preview);
    assert_eq!(image.frames[0].header.name, "frame");
    assert!(image.frames[0].complete);
}

#[test]
fn errors_do_not_publish_partial_frame_or_profile_updates() {
    let mut image = image(PixelLayout::Rgb, &[0.5; 3], "sRGB");
    image.frames.push(image.frames[0].clone());
    image.frames[1].pixels = PixelData::F32(vec![f32::NAN, 0.0, 0.0]);
    let original = image.frames[0].pixels.clone();
    assert!(process_image(&mut image, &target("linear"), &DecodeLimits::default()).is_err());
    assert_eq!(image.frames[0].pixels, original);
    assert_eq!(image.output_profile, Some(profile("sRGB")));
}

#[test]
fn active_processing_checks_f32_dimensions_profiles_and_limits() {
    let mut image = image(PixelLayout::Rgb, &[0.5; 3], "sRGB");
    let limits = DecodeLimits::default();
    image.frames[0].pixels = PixelData::U8(vec![128; 3]);
    assert!(process_image(&mut image, &target("linear"), &limits).is_err());
    image.frames[0].pixels = PixelData::F32(vec![0.5; 3]);
    image.frames[0].width = usize::MAX;
    assert!(process_image(&mut image, &target("linear"), &limits).is_err());
    image.frames[0].width = 1;
    let small = DecodeLimits {
        max_output_bytes: 1,
        ..limits.clone()
    };
    assert!(matches!(
        process_image(&mut image, &target("linear"), &small),
        Err(Error::Limit(_))
    ));
    image.output_profile = None;
    assert!(process_image(&mut image, &target("linear"), &limits).is_err());
}

#[test]
fn invalid_options_and_icc_fail_without_mutation() {
    let mut image = image(PixelLayout::Rgb, &[0.5; 3], "sRGB");
    let original = image.frames[0].pixels.clone();
    for nits in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let options = ColorOptions {
            display_nits: Some(nits),
            ..Default::default()
        };
        assert!(process_image(&mut image, &options, &DecodeLimits::default()).is_err());
    }
    for background in [[f32::NAN, 0.0, 0.0], [1.1, 0.0, 0.0]] {
        let options = ColorOptions {
            background: Some(background),
            ..Default::default()
        };
        assert!(process_image(&mut image, &options, &DecodeLimits::default()).is_err());
    }
    let options = ColorOptions {
        output_profile: Some(ColorProfile::Icc(vec![0; 128])),
        ..Default::default()
    };
    assert!(process_image(&mut image, &options, &DecodeLimits::default()).is_err());
    assert_eq!(image.frames[0].pixels, original);
}

#[test]
fn opaque_absolute_conversion_applies_media_white_between_pcs_stages() {
    let mut cms = CmsProfile::new_srgb();
    cms.media_white_point = Some(Xyzd::new(0.4821, 0.5, 0.41245));
    let mut image = image(PixelLayout::Rgb, &[0.5; 3], "sRGB");
    image.output_profile = Some(JxlColorProfile::Icc(cms.encode().unwrap()));
    process_image(
        &mut image,
        &target("RGB_D65_SRG_Abs_SRG"),
        &DecodeLimits::default(),
    )
    .unwrap();
    for &value in values(&image) {
        close(value, 0.3607802, 1e-4);
    }
}

#[test]
fn opaque_absolute_pcs_overflow_does_not_clip_or_publish() {
    let mut cms = CmsProfile::new_srgb();
    cms.rendering_intent = CmsIntent::AbsoluteColorimetric;
    cms.media_white_point = Some(Xyzd::new(0.4821, 0.5, 0.41245));
    let options = ColorOptions {
        output_profile: Some(ColorProfile::Icc(cms.encode().unwrap())),
        ..Default::default()
    };
    let mut image = image(PixelLayout::Rgb, &[1.0; 3], "sRGB");
    let original = image.frames[0].pixels.clone();
    let profile = image.output_profile.clone();
    assert!(process_image(&mut image, &options, &DecodeLimits::default()).is_err());
    assert_eq!(image.frames[0].pixels, original);
    assert_eq!(image.output_profile, profile);
}

#[test]
fn explicit_icc_bytes_and_header_intent_are_preserved() {
    let mut cms = CmsProfile::new_srgb();
    cms.rendering_intent = CmsIntent::AbsoluteColorimetric;
    // Same media white as the source; this test isolates the transfer and intent.
    cms.media_white_point = Some(Xyzd::new(0.9642, 1.0, 0.8249));
    let bytes = cms.encode().unwrap();
    let endpoint = Endpoint::new(
        &JxlColorProfile::Icc(bytes.clone()),
        &DecodeLimits::default(),
        true,
    )
    .unwrap();
    assert_eq!(
        endpoint.cms.rendering_intent,
        CmsIntent::AbsoluteColorimetric
    );
    let mut image = image(PixelLayout::Rgb, &[0.21404114; 3], "linear");
    let options = ColorOptions {
        output_profile: Some(ColorProfile::Icc(bytes.clone())),
        ..Default::default()
    };
    process_image(&mut image, &options, &DecodeLimits::default()).unwrap();
    for &value in values(&image) {
        close(value, 0.5, 2e-3);
    }
    assert_eq!(image.output_profile, Some(JxlColorProfile::Icc(bytes)));
}

fn cmyk_icc() -> Vec<u8> {
    use moxcms::{LutDataType, LutStore, LutType, LutWarehouse, Matrix3d, ProfileClass};
    let lut = |peak: f32| {
        let mut clut = Vec::new();
        for index in 0..16 {
            let y = if index & 1 == 0 { peak } else { 0.0 };
            for value in [0.9642 * y, y, 0.8252 * y] {
                clut.push((value * 32768.0) as u16);
            }
        }
        LutWarehouse::Lut(LutDataType {
            num_input_channels: 4,
            num_output_channels: 3,
            num_clut_grid_points: 2,
            matrix: Matrix3d::IDENTITY,
            num_input_table_entries: 2,
            num_output_table_entries: 2,
            input_table: LutStore::Store16([0, 65535].repeat(4)),
            clut_table: LutStore::Store16(clut),
            output_table: LutStore::Store16([0, 65535].repeat(3)),
            lut_type: LutType::Lut16,
        })
    };
    let mut cms = CmsProfile::new_srgb();
    cms.color_space = DataColorSpace::Cmyk;
    cms.profile_class = ProfileClass::InputDevice;
    cms.cicp = None;
    cms.red_trc = None;
    cms.green_trc = None;
    cms.blue_trc = None;
    cms.lut_a_to_b_perceptual = Some(lut(0.2));
    cms.lut_a_to_b_colorimetric = Some(lut(0.8));
    cms.lut_a_to_b_saturation = Some(lut(0.5));
    cms.encode().unwrap()
}

fn cmyk_image() -> DecodedImage {
    let mut image = image(
        PixelLayout::Rgba,
        &[0.2, 0.3, 0.4, 0.75, 0.2, 0.3, 0.4, 0.75],
        "sRGB",
    );
    image.output_profile = Some(JxlColorProfile::Icc(cmyk_icc()));
    image.info.extra_channels.push(extra(ExtraChannel::Black));
    image.frames[0]
        .extra_planes
        .push(PixelData::F32(vec![0.0, 1.0]));
    image
}

#[test]
fn cmyk_routes_and_inverts_black_without_using_alpha_as_k() {
    let mut pixel = [0.0; 4];
    gather_color(&[0.2, 0.3, 0.4, 0.75], &mut pixel, Some(0.9), false, 0.75);
    for (actual, expected) in pixel.into_iter().zip([0.8, 0.7, 0.6, 0.1]) {
        close(actual, expected, 1e-6);
    }
    let mut image = cmyk_image();
    let extras = image.frames[0].extra_planes.clone();
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    assert!(values(&image)[4] > values(&image)[0] + 0.2);
    assert_eq!(values(&image)[3], 0.75);
    assert_eq!(values(&image)[7], 0.75);
    assert_eq!(image.frames[0].extra_planes, extras);
}

#[test]
fn destination_intent_selects_distinct_cmyk_luts() {
    let mut perceptual = cmyk_image();
    let mut relative = cmyk_image();
    process_image(
        &mut perceptual,
        &target("RGB_D65_SRG_Per_Lin"),
        &DecodeLimits::default(),
    )
    .unwrap();
    process_image(
        &mut relative,
        &target("RGB_D65_SRG_Rel_Lin"),
        &DecodeLimits::default(),
    )
    .unwrap();
    assert!(values(&relative)[4] > values(&perceptual)[4] + 0.2);
}

#[test]
fn cmyk_to_gray_uses_pcs_luminance_and_keeps_alpha() {
    let mut image = cmyk_image();
    let extras = image.frames[0].extra_planes.clone();
    process_image(
        &mut image,
        &target("Gra_D65_Rel_Lin"),
        &DecodeLimits::default(),
    )
    .unwrap();
    assert_eq!(image.frames[0].layout, PixelLayout::GrayAlpha);
    close(values(&image)[0], 0.0, 1e-6);
    close(values(&image)[2], 0.8, 1.0 / 32768.0);
    assert_eq!(values(&image)[1], 0.75);
    assert_eq!(values(&image)[3], 0.75);
    assert_eq!(image.frames[0].extra_planes, extras);
}

#[test]
fn cmyk_missing_black_and_cmyk_destination_are_errors() {
    let mut image = cmyk_image();
    image.frames[0].extra_planes.clear();
    assert!(process_image(&mut image, &target("sRGB"), &DecodeLimits::default()).is_err());
    let options = ColorOptions {
        output_profile: Some(ColorProfile::Icc(cmyk_icc())),
        ..Default::default()
    };
    assert!(process_image(&mut image, &options, &DecodeLimits::default()).is_err());
}

#[test]
fn pq_roundtrip_has_absolute_nits_scaling() {
    close(pq_encode(100.0), 0.5080784, 1e-6);
    close(pq_decode(0.5080784), 100.0, 0.001);
    let mut image = image(PixelLayout::Rgb, &[pq_encode(100.0); 3], "Rec2100PQ");
    image.info.tone_mapping.intensity_target = 1000.0;
    process_image(
        &mut image,
        &target("RGB_D65_202_Rel_Lin"),
        &DecodeLimits::default(),
    )
    .unwrap();
    for &value in values(&image) {
        close(value, 0.1, 2e-6);
    }
    process_image(&mut image, &target("Rec2100PQ"), &DecodeLimits::default()).unwrap();
    for &value in values(&image) {
        close(value, pq_encode(100.0), 2e-6);
    }
}

#[test]
fn pq_display_shoulder_is_monotonic_and_peak_limited() {
    let samples: Vec<f32> = [0.0, 1.0, 10.0, 100.0, 1000.0]
        .into_iter()
        .flat_map(|nits| [pq_encode(nits); 3])
        .collect();
    let mut image = image(PixelLayout::Rgb, &samples, "Rec2100PQ");
    image.info.tone_mapping.intensity_target = 1000.0;
    let options = ColorOptions {
        display_nits: Some(100.0),
        ..target("RGB_D65_202_Rel_Lin")
    };
    process_image(&mut image, &options, &DecodeLimits::default()).unwrap();
    let pixels: Vec<_> = values(&image).chunks_exact(3).map(|p| p[0]).collect();
    // libjxl's rational PQ EOTF has a small nonzero black offset.
    assert!(pixels[0] > 0.0 && pixels[0] < 0.001);
    assert!(pixels.windows(2).all(|pair| pair[0] <= pair[1]));
    close(pixels[4], 1.0, 2e-5);
    assert!(pixels.iter().all(|v| *v >= 0.0 && *v <= 1.0));
}

#[test]
fn hlg_display_adjustment_changes_midtone_but_not_peak() {
    let mut original = image(
        PixelLayout::Rgb,
        &[0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        "Rec2100HLG",
    );
    let mut dimmer = image(
        PixelLayout::Rgb,
        &[0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        "Rec2100HLG",
    );
    original.info.tone_mapping.intensity_target = 1000.0;
    dimmer.info.tone_mapping.intensity_target = 1000.0;
    process_image(
        &mut original,
        &target("RGB_D65_202_Rel_Lin"),
        &DecodeLimits::default(),
    )
    .unwrap();
    let options = ColorOptions {
        display_nits: Some(100.0),
        ..target("RGB_D65_202_Rel_Lin")
    };
    process_image(&mut dimmer, &options, &DecodeLimits::default()).unwrap();
    assert!(values(&dimmer)[0] > values(&original)[0]);
    close(values(&dimmer)[3], 1.0, 1e-5);
}

#[test]
fn opaque_icc_display_adjustment_is_not_guessed() {
    let mut image = image(PixelLayout::Rgb, &[0.5; 3], "sRGB");
    image.output_profile = Some(JxlColorProfile::Icc(
        CmsProfile::new_srgb().encode().unwrap(),
    ));
    let options = ColorOptions {
        display_nits: Some(100.0),
        ..target("linear")
    };
    assert!(process_image(&mut image, &options, &DecodeLimits::default()).is_err());
}

#[test]
fn resolved_intensity_changes_only_for_an_actual_display_map() {
    let options = ColorOptions {
        display_nits: Some(100.0),
        ..Default::default()
    };
    let mut sdr = image(PixelLayout::Rgb, &[0.5; 3], "sRGB");
    let nits = process_image(&mut sdr, &options, &DecodeLimits::default()).unwrap();
    assert_eq!(nits, 255.0);
    assert_eq!(values(&sdr), &[0.5; 3]);
    let mut hlg = image(PixelLayout::Rgb, &[0.5; 3], "Rec2100HLG");
    hlg.info.tone_mapping.intensity_target = 1000.0;
    assert_eq!(
        process_image(&mut hlg, &options, &DecodeLimits::default()).unwrap(),
        1000.0
    );
    let mut pq = image(PixelLayout::Rgb, &[0.5; 3], "Rec2100PQ");
    pq.info.tone_mapping.intensity_target = 1000.0;
    assert_eq!(
        process_image(&mut pq, &options, &DecodeLimits::default()).unwrap(),
        100.0
    );
    assert_eq!(pq.info.tone_mapping.intensity_target, 1000.0);
}

#[test]
fn linear_pixels_keep_their_coordinates_when_origin_is_pq() {
    let mut encoded = image(PixelLayout::Rgb, &[0.5; 3], "Rec2100PQ");
    encoded.info.tone_mapping.intensity_target = 1000.0;
    let mut linear = image(
        PixelLayout::Rgb,
        &[pq_display(0.5, 1000.0); 3],
        "RGB_D65_202_Rel_Lin",
    );
    linear.info.tone_mapping.intensity_target = 1000.0;
    linear.embedded_profile = encoded.embedded_profile.clone();
    let options = ColorOptions {
        display_nits: Some(100.0),
        ..target("RGB_D65_202_Rel_Lin")
    };
    assert_eq!(
        process_image(&mut encoded, &options, &DecodeLimits::default()).unwrap(),
        100.0
    );
    assert_eq!(
        process_image(&mut linear, &options, &DecodeLimits::default()).unwrap(),
        100.0
    );
    assert_eq!(values(&encoded), values(&linear));
}

#[test]
fn pq_display_matches_djxl_near_black_and_gamut_mapped_highlight() {
    let mut image = image(
        PixelLayout::Rgb,
        &[
            4.0 / 1023.0,
            4.0 / 1023.0,
            0.0,
            836.0 / 1023.0,
            836.0 / 1023.0,
            0.0,
        ],
        "Rec2100PQ",
    );
    image.info.tone_mapping.intensity_target = 10000.0;
    let options = ColorOptions {
        display_nits: Some(250.0),
        ..Default::default()
    };
    process_image(&mut image, &options, &DecodeLimits::default()).unwrap();
    for (&actual, expected) in values(&image).iter().zip([
        0.0211818, 0.0211818, 0.0216803, 0.6025591, 0.6025591, 0.19208696,
    ]) {
        close(actual, expected, 1e-5);
    }
}

#[test]
fn cmyk_pcs_route_preserves_out_of_gamut_rgb() {
    let mut cms = CmsProfile::new_from_slice(&cmyk_icc()).unwrap();
    let Some(moxcms::LutWarehouse::Lut(lut)) = &mut cms.lut_a_to_b_colorimetric else {
        panic!("test LUT")
    };
    // PCS XYZ with too much Z for sRGB produces a negative red component.
    let xyz = [0.1 * 32768.0, 0.2 * 32768.0, 0.7 * 32768.0].map(|x| x as u16);
    lut.clut_table = moxcms::LutStore::Store16(xyz.repeat(16));
    let mut image = cmyk_image();
    image.output_profile = Some(JxlColorProfile::Icc(cms.encode().unwrap()));
    let extras = image.frames[0].extra_planes.clone();
    process_image(&mut image, &target("linear"), &DecodeLimits::default()).unwrap();
    assert!(values(&image)[0] < -0.1);
    assert!(values(&image)[2] > 0.8);
    assert_eq!(image.frames[0].extra_planes, extras);
    assert_eq!(values(&image)[3], 0.75);
}
