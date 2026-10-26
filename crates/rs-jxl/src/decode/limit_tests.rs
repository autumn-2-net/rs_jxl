use super::*;
use crate::color::{ColorEncoding, ColorSpace};
use crate::metadata::{ExtraChannelInfo, ExtraChannelType, ImageMetadata};

fn header(width: u32, height: u32) -> Vec<u8> {
    let mut metadata = ImageMetadata::new(8, ColorEncoding::srgb(ColorSpace::Rgb)).unwrap();
    metadata.modular_16_bit_buffer_sufficient = true;
    crate::codestream::image_header_fields(width, height, &metadata, 128).unwrap()
}

#[test]
fn defaults_do_not_impose_resource_policy_caps() {
    let options = DecodeOptions::default();
    let limits = &options.limits;
    assert_eq!(limits.max_input_bytes, usize::MAX);
    assert_eq!(limits.max_output_bytes, usize::MAX);
    assert_eq!(limits.max_metadata_bytes, usize::MAX);
    assert_eq!(limits.max_pixels, usize::MAX);
    assert_eq!(limits.max_samples, usize::MAX);
    assert_eq!(limits.max_frames, usize::MAX);
    let backend = backend_options_from_validated(&options);
    assert_eq!(backend.sample_limit, Some(usize::MAX));
    assert_eq!(backend.max_profile_level, ProfileLevel::Main10);
    assert!(!backend.allow_implicit_main10);
}

#[test]
fn headers_above_old_pixel_and_sample_caps_need_no_pixel_allocation() {
    let bytes = header(10_000, 10_000);
    assert!(bytes.len() < 128);
    let mut options = DecodeOptions::default();
    let mut info = read_basic_info(&bytes, &options).unwrap();
    assert_eq!(info.size, (10_000, 10_000));
    options.limits.max_pixels = 64_000_000;
    assert!(matches!(
        read_basic_info(&bytes, &options),
        Err(Error::Limit(_))
    ));
    options.limits.max_pixels = usize::MAX;
    options.limits.max_samples = 256_000_000;
    assert!(matches!(
        read_basic_info(&bytes, &options),
        Err(Error::Limit(_))
    ));

    options.limits.max_samples = usize::MAX;
    info.bit_depth = JxlBitDepth::Int {
        bits_per_sample: 31,
    };
    let bytes = native::byte_count(&info, 3, info.size, &options).unwrap();
    assert_eq!(bytes, 1_200_000_000);
    options.limits.max_output_bytes = bytes;
    assert_eq!(
        native::byte_count(&info, 3, info.size, &options).unwrap(),
        bytes
    );
    options.limits.max_output_bytes -= 1;
    assert!(matches!(
        native::byte_count(&info, 3, info.size, &options),
        Err(Error::Limit(_))
    ));
}

#[test]
fn explicit_input_pixel_and_sample_limits_are_inclusive() {
    let bytes = header(16, 1);
    let mut options = DecodeOptions::default();
    options.limits.max_input_bytes = bytes.len();
    options.limits.max_pixels = 16;
    options.limits.max_samples = 48;
    assert!(read_basic_info(&bytes, &options).is_ok());
    options.limits.max_input_bytes -= 1;
    assert!(matches!(
        read_basic_info(&bytes, &options),
        Err(Error::Limit(_))
    ));
    options.limits.max_input_bytes = bytes.len();
    options.limits.max_pixels -= 1;
    assert!(matches!(
        read_basic_info(&bytes, &options),
        Err(Error::Limit(_))
    ));
    options.limits.max_pixels = 16;
    options.limits.max_samples -= 1;
    assert!(matches!(
        read_basic_info(&bytes, &options),
        Err(Error::Limit(_))
    ));
    options.limits.max_samples = 0;
    assert!(matches!(
        read_basic_info(&bytes, &options),
        Err(Error::Limit(_))
    ));
}

#[test]
fn default_budgets_do_not_bypass_format_or_size_overflow_checks() {
    let options = DecodeOptions::default();
    // A bare codestream still implies Level 5, even without resource caps.
    let mut metadata = ImageMetadata::new(8, ColorEncoding::srgb(ColorSpace::Rgb)).unwrap();
    metadata.extra_channels =
        vec![ExtraChannelInfo::new(ExtraChannelType::Optional, 8).unwrap(); 5];
    let bytes = crate::codestream::image_header_fields(16, 1, &metadata, 128).unwrap();
    assert!(read_basic_info(&bytes, &options).is_err());
    let mut info = read_basic_info(&header(16, 1), &options).unwrap();
    info.size = (usize::MAX, 2);
    assert!(matches!(check_info(&info, &options), Err(Error::Limit(_))));
    assert!(matches!(
        native::byte_count(&info, 3, info.size, &options),
        Err(Error::Limit(_))
    ));
}

#[test]
fn default_frame_budget_does_not_reject_large_frame_indexes_before_parsing() {
    let mut decoder = Decoder::new(DecodeOptions::default()).unwrap();
    assert!(matches!(
        decoder.decode_frame(&[], 10_000),
        Err(Error::Decoding(_))
    ));
    let mut options = DecodeOptions::default();
    options.limits.max_frames = 1;
    let mut decoder = Decoder::new(options).unwrap();
    assert!(matches!(decoder.decode_frame(&[], 1), Err(Error::Limit(_))));
}

#[test]
fn metadata_expansion_obeys_explicit_inclusive_budgets() {
    use jxl_decoder::api::{Event, JxlDecoder};

    let mut writer = crate::container::ChunkedWriter::new(Vec::new(), 5, 256).unwrap();
    writer.write_metadata(*b"xml ", b"first").unwrap();
    writer.write_metadata(*b"xml ", b"second").unwrap();
    writer.write_chunk(&header(16, 1), true).unwrap();
    let bytes = writer.finish().unwrap();
    let mut backend = backend_options_from_validated(&DecodeOptions::default());
    backend.request_aux_boxes = boxes::REQUESTED.to_vec();
    let mut decoder = JxlDecoder::new(backend);
    decoder.close_input();
    let mut input = bytes.as_slice();
    assert!(matches!(
        decoder.process(&mut input, None, None).unwrap(),
        Event::BasicInfo
    ));
    for limit in [11, usize::MAX] {
        let metadata = boxes::extract(&decoder, &[], limit).unwrap();
        assert_eq!(
            metadata.iter().map(|entry| entry.data.len()).sum::<usize>(),
            11
        );
    }
    for limit in [0, 10] {
        assert!(matches!(
            boxes::extract(&decoder, &[], limit),
            Err(Error::Limit(_))
        ));
    }
}

#[test]
fn icc_stream_above_old_16_mib_cap_reaches_bitstream_validation() {
    // U64Coder length (2^24 + 1), followed by deliberately truncated histograms.
    // No buffer proportional to that declared length is allocated.
    let mut writer = crate::bit_writer::BitWriter::new(16);
    for (bits, value) in [(2, 3), (12, 1), (1, 1), (8, 0), (1, 1), (8, 16), (1, 0)] {
        writer.write(bits, value).unwrap();
    }
    writer.zero_pad_to_byte().unwrap();
    let bytes = writer.into_bytes().unwrap();
    assert!(matches!(
        JxlColorProfile::from_raw_bytes(&[], &bytes),
        Err(jxl_decoder::error::Error::OutOfBounds(_))
    ));
}
