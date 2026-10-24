use super::*;
use clap::Parser;
use std::io::Cursor;

fn frame(width: usize, height: usize, layout: PixelLayout, pixels: PixelData) -> DecodedFrame {
    DecodedFrame {
        width,
        height,
        layout,
        pixels,
        extra_planes: Vec::new(),
        duration_ticks: 0,
        name: String::new(),
        index: 0,
        header: header(width, height),
        complete: true,
    }
}

fn header(width: usize, height: usize) -> rs_jxl::decode::JxlFrameHeader {
    use rs_jxl::decode::{JxlBlendInfo, JxlBlendMode, JxlFrameHeader, JxlLayerInfo};
    JxlFrameHeader {
        name: String::new(),
        duration: None,
        size: (width, height),
        duration_ticks: 0,
        timecode: None,
        is_last: true,
        is_preview: false,
        layer_info: JxlLayerInfo {
            have_crop: false,
            crop_origin: (0, 0),
            size: (width, height),
            blend_info: JxlBlendInfo {
                mode: JxlBlendMode::Replace,
                source: 0,
                alpha_channel: 0,
                clamp: false,
            },
            extra_channel_blend_info: Vec::new(),
            save_as_reference: None,
            save_before_color_transform: false,
        },
    }
}

fn fixture(frame: DecodedFrame) -> DecodedImage {
    let source = rs_jxl::Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![17])).unwrap();
    let encoded = rs_jxl::encode(&source, &Default::default()).unwrap();
    let mut image = decode::decode(&encoded, &DecodeOptions::default()).unwrap();
    image.info.size = (frame.width, frame.height);
    image.info.bit_depth = match &frame.pixels {
        PixelData::F32(_) => JxlBitDepth::Float {
            bits_per_sample: 32,
            exponent_bits_per_sample: 8,
        },
        _ => JxlBitDepth::Int {
            bits_per_sample: u32::from(frame.pixels.bits()),
        },
    };
    image.frames = vec![frame];
    image
}

fn extra(kind: ExtraChannel, bits: u32) -> rs_jxl::decode::JxlExtraChannel {
    rs_jxl::decode::JxlExtraChannel {
        ec_type: kind,
        alpha_associated: false,
        bit_depth: JxlBitDepth::Int {
            bits_per_sample: bits,
        },
        dim_shift: 0,
        name: String::new(),
        spot_color: None,
        cfa_channel: None,
    }
}

#[path = "decode/control_tests.rs"]
mod control_tests;
#[path = "decode/writer_tests.rs"]
mod writer_tests;

#[test]
fn png_8_and_16_bit_samples_roundtrip_through_png_reader() {
    for (pixels, depth, expected) in [
        (
            PixelData::U8(vec![0, 255]),
            png::BitDepth::Eight,
            vec![0, 255],
        ),
        (
            PixelData::U16(vec![0x1234, 0xfedc]),
            png::BitDepth::Sixteen,
            vec![0x12, 0x34, 0xfe, 0xdc],
        ),
    ] {
        let source = rs_jxl::Image::new(2, 1, PixelLayout::Gray, pixels.clone()).unwrap();
        let jxl = rs_jxl::encode(&source, &Default::default()).unwrap();
        let mut image = decode::decode(&jxl, &DecodeOptions::default()).unwrap();
        // Test the writer with the requested samples independently of decoder precision.
        image.frames[0].pixels = pixels;
        let bytes = serialize(&image, OutputFormat::Png, false, false).unwrap();
        let mut reader = png::Decoder::new(Cursor::new(&bytes)).read_info().unwrap();
        assert_eq!(reader.info().bit_depth, depth);
        assert_eq!(reader.info().color_type, png::ColorType::Grayscale);
        assert_eq!(
            reader.info().icc_profile.as_deref().unwrap(),
            image
                .output_profile
                .as_ref()
                .unwrap()
                .try_as_icc()
                .unwrap()
                .as_slice()
        );
        let mut output = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut output).unwrap();
        assert_eq!((info.width, info.height), (2, 1));
        assert_eq!(&output[..info.buffer_size()], expected);
    }
}

#[test]
fn apng_roundtrip_preserves_frames_timing_and_selected_still() {
    let source = rs_jxl::Image::new(2, 1, PixelLayout::Gray, PixelData::U8(vec![0, 255])).unwrap();
    let jxl = rs_jxl::encode(&source, &Default::default()).unwrap();
    let mut image = decode::decode(&jxl, &DecodeOptions::default()).unwrap();
    let expected = [vec![0, 255], vec![17, 128]];
    image.frames = expected
        .iter()
        .enumerate()
        .map(|(index, pixels)| {
            let mut output = frame(2, 1, PixelLayout::Gray, PixelData::U8(pixels.clone()));
            output.index = index;
            output.duration_ticks = index as u32 + 1;
            output
        })
        .collect();
    image.info.animation = Some(rs_jxl::decode::JxlAnimation {
        tps_numerator: 24000,
        tps_denominator: 1001,
        num_loops: 3,
        have_timecodes: false,
    });
    let bytes = serialize(&image, OutputFormat::Png, false, false).unwrap();
    let mut reader = png::Decoder::new(Cursor::new(&bytes)).read_info().unwrap();
    let animation = reader.info().animation_control.as_ref().unwrap();
    assert_eq!(animation.num_frames, 2);
    assert_eq!(animation.num_plays, 3);
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    for (expected, delay) in expected.iter().zip([(1001, 24000), (1001, 12000)]) {
        let output = reader.next_frame(&mut pixels).unwrap();
        assert_eq!((output.width, output.height), (2, 1));
        assert_eq!(&pixels[..output.buffer_size()], expected.as_slice());
        let control = reader.info().frame_control.as_ref().unwrap();
        assert_eq!((control.delay_num, control.delay_den), delay);
        assert_eq!(control.blend_op, png::BlendOp::Source);
        assert_eq!(control.dispose_op, png::DisposeOp::None);
    }
    assert!(reader.next_frame(&mut pixels).is_err());

    for (numerator, denominator) in [(0, 1), (1, 0), (65537, 1), (1, 65536)] {
        let animation = image.info.animation.as_mut().unwrap();
        animation.tps_numerator = numerator;
        animation.tps_denominator = denominator;
        assert!(serialize(&image, OutputFormat::Png, false, false).is_err());
    }
    let animation = image.info.animation.as_mut().unwrap();
    animation.tps_numerator = 24000;
    animation.tps_denominator = 1001;
    image.frames = vec![image.frames[1].clone()];
    let bytes = serialize(&image, OutputFormat::Png, true, false).unwrap();
    let mut reader = png::Decoder::new(Cursor::new(&bytes)).read_info().unwrap();
    assert!(reader.info().animation_control.is_none());
    assert!(reader.info().frame_control.is_none());
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let output = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(&pixels[..output.buffer_size()], expected[1].as_slice());
    assert!(reader.next_frame(&mut pixels).is_err());
}

#[test]
fn npy_f32_stack_has_exact_schema_channel_order_and_ieee_words() {
    let words = [0x8000_0000, 0x7fc1_2345, 0x3f80_0000, 0x7f80_0000];
    let mut first = frame(
        2,
        1,
        PixelLayout::GrayAlpha,
        PixelData::F32(vec![
            f32::from_bits(words[0]),
            9.0,
            f32::from_bits(words[1]),
            8.0,
        ]),
    );
    first.extra_planes.push(PixelData::F32(vec![0.25, 0.75]));
    let mut second = first.clone();
    second.index = 1;
    second.pixels = PixelData::F32(vec![
        f32::from_bits(words[2]),
        7.0,
        f32::from_bits(words[3]),
        6.0,
    ]);
    let mut bytes = Vec::new();
    write_npy(&mut bytes, &[first.clone(), second]).unwrap();
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let start = 10 + usize::from(u16::from_le_bytes(bytes[8..10].try_into().unwrap()));
    assert_eq!(start % 64, 0);
    let header = std::str::from_utf8(&bytes[10..start]).unwrap();
    assert!(header.ends_with('\n'));
    assert_eq!(
        header.trim_end(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 1, 2, 2), }"
    );
    let actual: Vec<_> = bytes[start..]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(
        actual,
        [
            words[0],
            0.25f32.to_bits(),
            words[1],
            0.75f32.to_bits(),
            words[2],
            0.25f32.to_bits(),
            words[3],
            0.75f32.to_bits()
        ]
    );
    bytes.clear();
    write_npy(&mut bytes, &[first]).unwrap();
    let start = 10 + usize::from(u16::from_le_bytes(bytes[8..10].try_into().unwrap()));
    assert!(
        std::str::from_utf8(&bytes[10..start])
            .unwrap()
            .contains("'shape': (1, 2, 2)")
    );
}

#[test]
fn portable_integer_endian_and_float_bottom_up_rows_are_exact() {
    for (layout, format, magic, values) in [
        (
            PixelLayout::Gray,
            OutputFormat::Pgm,
            "P5",
            vec![0x1234, 0xfedc],
        ),
        (
            PixelLayout::Rgb,
            OutputFormat::Ppm,
            "P6",
            vec![0, 1, 0x1234, 0xabcd, 0xff00, 0xffff],
        ),
    ] {
        let mut bytes = Vec::new();
        let image = frame(2, 1, layout, PixelData::U16(values.clone()));
        write_portable(&mut bytes, &image, format).unwrap();
        let mut expected = format!("{magic}\n2 1\n65535\n").into_bytes();
        for value in values {
            expected.extend_from_slice(&value.to_be_bytes());
        }
        assert_eq!(bytes, expected);
    }
    let mut bytes = Vec::new();
    write_portable(
        &mut bytes,
        &frame(2, 1, PixelLayout::Gray, PixelData::U8(vec![0, 255])),
        OutputFormat::Pgm,
    )
    .unwrap();
    assert_eq!(bytes, b"P5\n2 1\n255\n\0\xff");
    for (layout, magic) in [(PixelLayout::Gray, "Pf"), (PixelLayout::Rgb, "PF")] {
        let row = 2 * layout.channels();
        let values: Vec<_> = (0..row * 2).map(|i| i as f32 + 0.25).collect();
        bytes.clear();
        write_portable(
            &mut bytes,
            &frame(2, 2, layout, PixelData::F32(values.clone())),
            OutputFormat::Pfm,
        )
        .unwrap();
        let mut expected = format!("{magic}\n2 2\n-1.0\n").into_bytes();
        for value in values[row..].iter().chain(&values[..row]) {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(bytes, expected);
    }
}

#[test]
fn cli_and_writers_reject_incompatible_formats_without_sample_loss() {
    for extra in [
        vec!["--threads", "0"],
        vec!["--threads", "257"],
        vec!["--bits", "17"],
        vec!["--bits", "abc"],
        vec!["--frame", "-1"],
        vec!["--format", "invalid"],
    ] {
        let mut argv = vec!["rs-jxl", "decode", "input.jxl", "output.png"];
        argv.extend(extra);
        assert!(crate::Cli::try_parse_from(argv).is_err());
    }
    for tail in [
        vec!["-"],
        vec!["out.png", "--bits", "32"],
        vec!["out.pfm", "--bits", "8"],
        vec!["out.jpg", "--frame", "0"],
        vec!["out.jpeg", "--bits", "16"],
    ] {
        let mut argv = vec!["rs-jxl", "decode", "input.jxl"];
        argv.extend(tail);
        let crate::Command::Decode(args) = crate::Cli::try_parse_from(argv).unwrap().command else {
            panic!("wrong command")
        };
        assert!(args.output_format().is_err());
    }
    let crate::Command::Decode(defaults) =
        crate::Cli::try_parse_from(["rs-jxl", "decode", "input.jxl", "output.png"])
            .unwrap()
            .command
    else {
        panic!("wrong command")
    };
    assert!(!defaults.allow_implicit_level10);
    assert!(!defaults.options(OutputFormat::Png).allow_implicit_main10);
    let crate::Command::Decode(args) = crate::Cli::try_parse_from([
        "rs-jxl",
        "decode",
        "-",
        "-",
        "--format",
        "npy",
        "--bits",
        "32",
        "--threads",
        "2",
        "--frame",
        "0",
        "--keep-orientation",
        "--allow-implicit-level10",
        "--force",
    ])
    .unwrap()
    .command
    else {
        panic!("wrong command")
    };
    assert_eq!(args.output_format().unwrap(), OutputFormat::Npy);
    let options = args.options(OutputFormat::Npy);
    assert_eq!(options.format, SampleFormat::F32);
    assert_eq!(options.threads, 2);
    assert!(args.allow_implicit_level10);
    assert!(options.allow_implicit_main10);
    assert!(!options.adjust_orientation);
    assert!(options.include_extra_channels && args.force);
    assert_eq!(args.frame, Some(0));
    let mut alpha = frame(1, 1, PixelLayout::GrayAlpha, PixelData::U8(vec![17, 255]));
    assert!(write_portable(&mut Vec::new(), &alpha, OutputFormat::Pgm).is_err());
    assert!(write_npy(&mut Vec::new(), &[alpha.clone()]).is_err());
    alpha.extra_planes.push(PixelData::U16(vec![65535]));
    assert!(write_npy(&mut Vec::new(), &[alpha]).is_err());
    let mut extra = frame(1, 1, PixelLayout::Gray, PixelData::U8(vec![17]));
    extra.extra_planes.push(PixelData::U8(vec![3]));
    assert!(write_portable(&mut Vec::new(), &extra, OutputFormat::Pgm).is_err());
    assert!(png_depth(&PixelData::F32(vec![1.0])).is_err());
    assert!(
        write_portable(
            &mut Vec::new(),
            &frame(1, 1, PixelLayout::Rgb, PixelData::U8(vec![1, 2, 3])),
            OutputFormat::Pgm
        )
        .is_err()
    );
}
