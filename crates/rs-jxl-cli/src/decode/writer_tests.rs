use super::*;
use crate::decode::writers::{Output, Raster, serialize_raster, write_portable_depth};
use std::io::{Seek, SeekFrom, Write};

#[test]
fn npy_widens_mixed_integer_planes_without_float_rounding() {
    let mut f = frame(2, 1, PixelLayout::Gray, PixelData::U8(vec![1, 255]));
    f.extra_planes = vec![
        PixelData::U16(vec![32769, 65535]),
        PixelData::U32(vec![16_777_217, 2_147_483_647]),
    ];
    let mut bytes = Vec::new();
    write_npy(&mut bytes, &[f]).unwrap();
    let header_len = usize::from(u16::from_le_bytes(bytes[8..10].try_into().unwrap()));
    assert!(
        std::str::from_utf8(&bytes[10..10 + header_len])
            .unwrap()
            .contains("'<u4'")
    );
    let values: Vec<_> = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(values, [1, 32769, 16_777_217, 255, 65535, 2_147_483_647]);
}

#[test]
fn pnm_selects_gray_or_rgb_and_pgx_keeps_custom_range() {
    let gray = frame(2, 1, PixelLayout::Gray, PixelData::U16(vec![0x123, 0xfff]));
    for (format, prefix) in [
        (OutputFormat::Pnm, "P5\n2 1\n4095\n"),
        (OutputFormat::Pgx, "PG ML + 12 2 1\n"),
    ] {
        let mut bytes = Vec::new();
        write_portable_depth(&mut bytes, &gray, format, 12, false).unwrap();
        assert_eq!(&bytes[..prefix.len()], prefix.as_bytes());
        assert_eq!(&bytes[prefix.len()..], &[1, 0x23, 0xf, 0xff]);
    }
    let rgb = frame(1, 1, PixelLayout::Rgb, PixelData::U8(vec![1, 2, 3]));
    let mut bytes = Vec::new();
    write_portable_depth(&mut bytes, &rgb, OutputFormat::Pnm, 8, false).unwrap();
    assert_eq!(bytes, b"P6\n1 1\n255\n\x01\x02\x03");
    assert!(write_portable_depth(&mut Vec::new(), &rgb, OutputFormat::Pgx, 8, false).is_err());
    assert!(write_portable_depth(&mut Vec::new(), &gray, OutputFormat::Pgx, 8, false).is_err());
}

#[test]
fn pam_preserves_authoritative_alpha_once_and_every_extra() {
    let mut f = frame(1, 1, PixelLayout::GrayAlpha, PixelData::U16(vec![0x123, 0]));
    f.extra_planes = vec![PixelData::U16(vec![0xabc]), PixelData::U16(vec![0x456])];
    let mut image = fixture(f);
    image.info.bit_depth = JxlBitDepth::Int {
        bits_per_sample: 12,
    };
    image.info.extra_channels = vec![
        extra(ExtraChannel::Alpha, 12),
        extra(ExtraChannel::Depth, 12),
    ];
    let bytes = serialize(&image, OutputFormat::Pam, false, false).unwrap();
    let prefix = b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 3\nMAXVAL 4095\nTUPLTYPE JXL_CHANNELS\nENDHDR\n";
    assert_eq!(&bytes[..prefix.len()], prefix);
    assert_eq!(&bytes[prefix.len()..], &[1, 0x23, 0xa, 0xbc, 4, 0x56]);
    image.info.extra_channels[1].bit_depth = JxlBitDepth::Int { bits_per_sample: 8 };
    assert!(serialize(&image, OutputFormat::Pam, false, false).is_err());
}

#[test]
fn exr_keeps_f32_precision_and_native_extra_channels() {
    use exr::prelude::*;
    let mut f = frame(
        2,
        1,
        PixelLayout::Gray,
        PixelData::F32(vec![-0.0, 1.000001]),
    );
    f.extra_planes = vec![PixelData::F32(vec![1234.125, -7.25])];
    let mut image = fixture(f);
    image.info.extra_channels = vec![extra(ExtraChannel::Depth, 32)];
    image.output_profile = Some(decode::JxlColorProfile::Simple(
        decode::parse_color_description("Gra_D65_Rel_Lin").unwrap(),
    ));
    let bytes = serialize(&image, OutputFormat::Exr, false, false).unwrap();
    let decoded = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes()
        .from_buffered(Cursor::new(bytes))
        .unwrap();
    let layer = &decoded.layer_data[0];
    let channel = |name: &str| {
        layer
            .channel_data
            .list
            .iter()
            .find(|c| c.name.to_string() == name)
            .unwrap()
    };
    let FlatSamples::F32(values) = &channel("Y").sample_data else {
        panic!("not F32")
    };
    assert_eq!(
        values.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        vec![(-0.0f32).to_bits(), 1.000001f32.to_bits()]
    );
    let FlatSamples::F32(values) = &channel("ec0").sample_data else {
        panic!("not F32")
    };
    assert_eq!(values, &[1234.125, -7.25]);
    assert!(decoded.attributes.chromaticities.is_some());
}

#[test]
fn exr_white_luminance_uses_resolved_output_without_changing_source_metadata() {
    use exr::prelude::*;
    let mut image = fixture(frame(1, 1, PixelLayout::Gray, PixelData::F32(vec![0.5])));
    image.info.tone_mapping.intensity_target = 1000.0;
    image.output_profile = Some(decode::JxlColorProfile::Simple(
        decode::parse_color_description("Gra_D65_Rel_Lin").unwrap(),
    ));
    for output_intensity_target in [1000.0, 250.0] {
        image.output_intensity_target = output_intensity_target;
        let view = Raster::from(&image);
        let bytes = serialize_raster(
            view,
            OutputFormat::Exr,
            true,
            false,
            SampleFormat::F32,
            false,
            MAX_OUTPUT_BYTES,
        )
        .unwrap();
        let decoded = read()
            .no_deep_data()
            .largest_resolution_level()
            .all_channels()
            .all_layers()
            .all_attributes()
            .from_buffered(Cursor::new(bytes))
            .unwrap();
        assert_eq!(
            decoded.layer_data[0].attributes.white_luminance,
            Some(output_intensity_target)
        );
        assert_eq!(image.info.tone_mapping.intensity_target, 1000.0);
    }
}

#[test]
fn alpha_association_uses_output_mode_not_source_metadata() {
    let mut f = frame(1, 1, PixelLayout::GrayAlpha, PixelData::U8(vec![40, 128]));
    f.extra_planes = vec![PixelData::U8(vec![128])];
    let mut image = fixture(f);
    image.info.extra_channels = vec![extra(ExtraChannel::Alpha, 8)];
    image.info.extra_channels[0].alpha_associated = true;
    assert!(serialize(&image, OutputFormat::Png, false, false).is_err());
    let view = Raster {
        alpha_mode: decode::AlphaMode::Unpremultiplied,
        ..Raster::from(&image)
    };
    assert!(
        serialize_raster(
            view,
            OutputFormat::Png,
            true,
            false,
            SampleFormat::U8,
            false,
            MAX_OUTPUT_BYTES
        )
        .is_ok()
    );
}

#[test]
fn preview_writers_reject_mislabeled_native_ranges_and_nonlinear_exr() {
    let mut image = fixture(frame(1, 1, PixelLayout::Gray, PixelData::U16(vec![4095])));
    image.info.bit_depth = JxlBitDepth::Int {
        bits_per_sample: 12,
    };
    assert!(serialize(&image, OutputFormat::Png, true, false).is_err());
    let image = fixture(frame(1, 1, PixelLayout::Gray, PixelData::F32(vec![0.5])));
    assert!(serialize(&image, OutputFormat::Exr, true, false).is_err());
}

#[test]
fn metadata_extracts_tiff_offset_and_binary_payloads_without_modification() {
    let tiff = metadata::orientation_exif(6);
    let mut exif = vec![0, 0, 0, 3, 9, 8, 7];
    exif.extend_from_slice(&tiff);
    let boxes = vec![
        DecodedBox {
            kind: *b"Exif",
            data: exif,
        },
        DecodedBox {
            kind: *b"xml ",
            data: vec![0xff, 0, b'\n'],
        },
        DecodedBox {
            kind: *b"jumb",
            data: vec![1, 2, 3],
        },
    ];
    assert_eq!(
        metadata::extract(&boxes, OutputFormat::Exif, 100).unwrap(),
        tiff
    );
    assert_eq!(
        metadata::extract(&boxes, OutputFormat::Xmp, 100).unwrap(),
        [0xff, 0, b'\n']
    );
    assert_eq!(
        metadata::extract(&boxes, OutputFormat::Jumbf, 100).unwrap(),
        [1, 2, 3]
    );
    assert!(metadata::extract(&boxes, OutputFormat::Xmp, 2).is_err());
    assert!(metadata::extract(&[], OutputFormat::Xmp, 100).is_err());
    let duplicate = vec![boxes[1].clone(), boxes[1].clone()];
    assert!(metadata::extract(&duplicate, OutputFormat::Xmp, 100).is_err());
    assert!(
        metadata::extract(
            &[DecodedBox {
                kind: *b"Exif",
                data: vec![0, 0, 0, 20]
            }],
            OutputFormat::Exif,
            100
        )
        .is_err()
    );
}

#[test]
fn json_escapes_names_and_keeps_timing_crop_and_blend_metadata() {
    let mut image = fixture(frame(1, 1, PixelLayout::Gray, PixelData::U8(vec![7])));
    let header = &mut image.frames[0].header;
    header.name = "quote\" slash\\ newline\n".into();
    header.duration = Some(1500.0);
    header.duration_ticks = 3;
    header.timecode = Some(1234);
    header.layer_info.have_crop = true;
    header.layer_info.crop_origin = (-4, 17);
    header.layer_info.save_as_reference = Some(0);
    let bytes =
        metadata::json_metadata(&metadata::Metadata::from(&image), MAX_OUTPUT_BYTES).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["frames"][0]["name"], image.frames[0].header.name);
    assert_eq!(value["frames"][0]["duration"], 1.5);
    assert_eq!(value["frames"][0]["timecode"], 1234);
    assert_eq!(value["frames"][0]["layer_info"]["crop_origin"][0], -4);
    assert_eq!(value["frames"][0]["layer_info"]["save_as_reference"], 0);
    assert!(metadata::json_metadata(&metadata::Metadata::from(&image), 10).is_err());
}

#[test]
fn seekable_output_enforces_budget_on_holes_rewrites_and_growth() {
    let mut output = Output::with_limit(8);
    output.write_all(b"abcd").unwrap();
    output.seek(SeekFrom::Start(1)).unwrap();
    output.write_all(b"Z").unwrap();
    output.seek(SeekFrom::Start(7)).unwrap();
    assert!(output.write_all(b"too long").is_err());
    output.write_all(b"!").unwrap();
    assert!(output.seek(SeekFrom::Start(9)).is_err());
    assert!(output.seek(SeekFrom::End(-9)).is_err());
    assert_eq!(output.into_bytes(), b"aZcd\0\0\0!");
}
