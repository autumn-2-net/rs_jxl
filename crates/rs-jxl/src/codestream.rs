// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Image/frame/TOC and container fields from enc_fast_lossless.cc, encode.cc
//! and encode_internal.h at b866b3d. Modular tools serialize their own headers.
use crate::{Error, Image, Result, bit_writer::BitWriter};

pub(crate) const FRAME_SIZE_CHOICES: [(u32, u8); 4] = [(0, 8), (256, 11), (2304, 14), (18688, 30)];

fn check_level10_limits(extra_channels: usize, icc_size: usize) -> Result<()> {
    if extra_channels > 256 || icc_size > 1 << 28 {
        return Err(Error::Limit("Level 10 extra-channel/ICC limit"));
    }
    Ok(())
}

fn check_level10_geometry(width: u32, height: u32) -> Result<()> {
    if width == 0
        || height == 0
        || width > 1 << 30
        || height > 1 << 30
        || u64::from(width) * u64::from(height) > 1 << 40
    {
        return Err(Error::InvalidInput(
            "image dimensions exceed Level 10 limits".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_level10(image: &Image) -> Result<()> {
    validate_level10_geometry(image.into())
}

pub(crate) fn validate_level10_geometry(image: crate::frame::Geometry<'_>) -> Result<()> {
    check_level10_geometry(image.width(), image.height())?;
    check_level10_limits(
        image.metadata().extra_channels.len(),
        image
            .metadata()
            .color_encoding
            .icc_profile()
            .map_or(0, <[u8]>::len),
    )
}

/// encode.cc::VerifyLevelSettings and encode_internal.h::MakeContainerHeader.
/// A bare codestream implies Level 5; 16-bit Modular needs a Level 10 marker.
pub(crate) fn container_prefix(image: &Image, codestream_bytes: usize) -> Result<Vec<u8>> {
    container_prefix_geometry(image.into(), codestream_bytes)
}

pub(crate) fn container_prefix_geometry(
    image: crate::frame::Geometry<'_>,
    codestream_bytes: usize,
) -> Result<Vec<u8>> {
    let icc_size = image
        .metadata()
        .color_encoding
        .icc_profile()
        .map_or(0, <[u8]>::len);
    validate_level10_geometry(image)?;
    let has_black = image
        .metadata()
        .extra_channels
        .iter()
        .any(|e| e.channel_type == crate::metadata::ExtraChannelType::Black);
    if !has_black
        && image.metadata().modular_16_bit_buffer_sufficient
        && image.bits() <= 14
        && icc_size <= 1 << 22
        && image.metadata().extra_channels.len() <= 4
        && image
            .metadata()
            .extra_channels
            .iter()
            .all(|e| e.bit_depth.bits_per_sample <= 14)
        && image.width() <= 1 << 18
        && image.height() <= 1 << 18
        && u64::from(image.width()) * u64::from(image.height()) <= (1 << 28)
    {
        return Ok(Vec::new());
    }
    let mut prefix = Vec::with_capacity(57);
    prefix.extend_from_slice(b"\0\0\0\x0cJXL \x0d\x0a\x87\x0a");
    prefix.extend_from_slice(b"\0\0\0\x14ftypjxl \0\0\0\0jxl ");
    prefix.extend_from_slice(b"\0\0\0\x09jxll\x0a");
    let short_size = codestream_bytes
        .checked_add(8)
        .ok_or(Error::Limit("container size"))?;
    if let Ok(size) = u32::try_from(short_size) {
        prefix.extend_from_slice(&size.to_be_bytes());
        prefix.extend_from_slice(b"jxlc");
    } else {
        let size = u64::try_from(codestream_bytes)
            .ok()
            .and_then(|n| n.checked_add(16))
            .ok_or(Error::Limit("container size"))?;
        prefix.extend_from_slice(b"\0\0\0\x01jxlc");
        prefix.extend_from_slice(&size.to_be_bytes());
    }
    Ok(prefix)
}

pub(crate) fn write_choice(w: &mut BitWriter, value: u32, choices: [(u32, u8); 4]) -> Result<()> {
    for (i, (offset, bits)) in choices.into_iter().enumerate() {
        if value >= offset && u64::from(value - offset) < (1u64 << bits) {
            w.write(2, i as u64)?;
            w.write(bits, u64::from(value - offset))?;
            return Ok(());
        }
    }
    Err(Error::Encoding("header field exceeds coding range".into()))
}

pub(crate) fn image_header(image: &Image, max_bytes: usize) -> Result<Vec<u8>> {
    validate_level10(image)?;
    image_header_fields(image.width(), image.height(), image.metadata(), max_bytes)
}

/// Shared framing for pixel images and coefficient-only JPEG input. The latter
/// must not allocate dummy pixel planes just to serialize the image dimensions.
pub(crate) fn image_header_fields(
    width: u32,
    height: u32,
    metadata: &crate::metadata::ImageMetadata,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    check_level10_geometry(width, height)?;
    metadata.validate()?;
    check_level10_limits(
        metadata.extra_channels.len(),
        metadata.color_encoding.icc_profile().map_or(0, <[u8]>::len),
    )?;
    let mut w = BitWriter::new(max_bytes);
    w.write(16, 0xaff)?;
    w.write(1, 0)?;
    let size_choices = [(1, 9), (1, 13), (1, 18), (1, 30)];
    write_choice(&mut w, height, size_choices)?;
    w.write(3, 0)?;
    write_choice(&mut w, width, size_choices)?;
    metadata.write_header(&mut w)?;
    crate::metadata::write_default_transform_data(&mut w)?;
    if let Some(profile) = metadata.color_encoding.icc_profile() {
        crate::icc::write_icc(profile, &mut w)?;
    }
    w.zero_pad_to_byte()?;
    Ok(w.into_bytes()?)
}

// Geometry, metadata and section sizes are separate source framing inputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn frame_header(
    canvas: &Image,
    image: &Image,
    sizes: &[usize],
    group_size_shift: u8,
    passes: &crate::modular::layout::Passes,
    frame: &crate::frame::FrameOptions,
    last: bool,
    patches: bool,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    frame.validate(canvas, image, last)?;
    let mut w = frame_header_body(
        canvas.into(),
        image.into(),
        group_size_shift,
        passes,
        frame,
        last,
        patches,
        max_bytes,
    )?;
    w.write(1, 0)?;
    w.zero_pad_to_byte()?;
    for &size in sizes {
        let size = u32::try_from(size).map_err(|_| Error::Limit("section size"))?;
        write_choice(
            &mut w,
            size,
            [(0, 10), (1024, 14), (17408, 22), (4211712, 30)],
        )?;
    }
    w.zero_pad_to_byte()?;
    Ok(w.into_bytes()?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn frame_header_body(
    canvas: crate::frame::Geometry<'_>,
    image: crate::frame::Geometry<'_>,
    group_size_shift: u8,
    passes: &crate::modular::layout::Passes,
    frame: &crate::frame::FrameOptions,
    last: bool,
    patches: bool,
    max_bytes: usize,
) -> Result<BitWriter> {
    frame.validate_geometry(canvas, image, last)?;
    let reference_only = frame.reference_only;
    let mut w = BitWriter::new(max_bytes);
    let extras = image.metadata().extra_channels.len();
    passes.validate()?;
    // Original-color Modular frame, with explicit reference/blending fields.
    w.write(1, 0)?;
    w.write(2, if reference_only { 2 } else { 0 })?;
    w.write(1, 1)?;
    if patches {
        w.write(2, 1)?;
        w.write(4, 1)?;
    } else {
        w.write(2, 0)?;
    }
    w.write(1, 0)?;
    w.write(2, 0)?;
    for _ in 0..extras {
        w.write(2, 0)?;
    }
    w.write(2, u64::from(group_size_shift))?;
    if !reference_only {
        write_passes(&mut w, passes)?;
    }
    let custom = image.width() != canvas.width()
        || image.height() != canvas.height()
        || frame.origin != (0, 0);
    let partial = !reference_only && frame.partial_geometry(canvas, image);
    w.write(1, u64::from(custom))?;
    if custom {
        let choices = FRAME_SIZE_CHOICES;
        if !reference_only {
            write_choice(
                &mut w,
                crate::modular::tree::pack_signed(frame.origin.0),
                choices,
            )?;
            write_choice(
                &mut w,
                crate::modular::tree::pack_signed(frame.origin.1),
                choices,
            )?;
        }
        write_choice(&mut w, image.width(), choices)?;
        write_choice(&mut w, image.height(), choices)?;
    }
    if reference_only {
        w.write(2, u64::from(frame.save_as_reference))?;
        w.write(1, u64::from(frame.save_before_color_transform))?;
    } else {
        frame.blend.write(&mut w, extras, partial)?;
        for e in 0..extras {
            frame.extra(e).write(&mut w, extras, partial)?;
        }
        if let Some(animation) = canvas.metadata().animation {
            write_choice(&mut w, frame.duration, [(0, 0), (1, 0), (0, 8), (0, 32)])?;
            if animation.timecodes {
                w.write(32, u64::from(frame.timecode))?;
            }
        }
        w.write(1, u64::from(last))?;
        if !last {
            w.write(2, u64::from(frame.save_as_reference))?;
            if (frame.duration == 0 || frame.save_as_reference != 0)
                && !partial
                && frame.blend.mode == crate::frame::BlendMode::Replace
            {
                w.write(1, u64::from(frame.save_before_color_transform))?;
            }
        }
    }
    write_choice(
        &mut w,
        frame.name.len() as u32,
        [(0, 0), (0, 4), (16, 5), (48, 10)],
    )?;
    for byte in frame.name.bytes() {
        w.write(8, u64::from(byte))?;
    }
    // No gaborish/EPF or extensions.
    w.write(1, 0)?;
    w.write(1, 0)?;
    w.write(2, 0)?;
    w.write(2, 0)?;
    w.write(2, 0)?;
    Ok(w)
}

fn write_passes(w: &mut BitWriter, passes: &crate::modular::layout::Passes) -> Result<()> {
    passes.validate()?;
    write_choice(
        w,
        u32::from(passes.num_passes),
        [(1, 0), (2, 0), (3, 0), (4, 3)],
    )?;
    if passes.num_passes != 1 {
        write_choice(
            w,
            passes.downsample.len() as u32,
            [(0, 0), (1, 0), (2, 0), (3, 1)],
        )?;
        for _ in 1..passes.num_passes {
            w.write(2, 0)?;
        }
        for d in &passes.downsample {
            write_choice(w, u32::from(d.factor), [(1, 0), (2, 0), (4, 0), (8, 0)])?;
        }
        for d in &passes.downsample {
            write_choice(w, u32::from(d.last_pass), [(0, 0), (1, 0), (2, 0), (0, 3)])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PixelData, PixelLayout,
        color::{ColorEncoding, ColorSpace},
        metadata::{ExtraChannelInfo, ExtraChannelType},
    };

    #[test]
    fn level10_limit_checks_do_not_allocate_profile_data() {
        assert!(check_level10_limits(256, 1 << 28).is_ok());
        assert!(check_level10_limits(257, 0).is_err());
        assert!(check_level10_limits(0, (1 << 28) + 1).is_err());
        assert!(check_level10_limits(0, usize::MAX).is_err());
    }

    #[test]
    fn field_header_accepts_large_geometry_without_dummy_pixels() {
        let metadata =
            crate::metadata::ImageMetadata::new(8, ColorEncoding::srgb(ColorSpace::Rgb)).unwrap();
        let header = image_header_fields(65535, 65535, &metadata, 128).unwrap();
        assert_eq!(&header[..2], &[0xff, 0x0a]);
        assert!(header.len() < 128);
        for (width, height) in [(0, 1), (1, 0), (u32::MAX, 1), (1, u32::MAX)] {
            assert!(image_header_fields(width, height, &metadata, 128).is_err());
        }
        assert!(image_header_fields(1, 1, &metadata, 1).is_err());
        for (width, height) in [(262145, 1), (1, 262145), (1 << 30, 1024), (1024, 1 << 30)] {
            assert!(image_header_fields(width, height, &metadata, 128).is_ok());
        }
        for (width, height) in [((1 << 30) + 1, 1), (1 << 30, 1025), (1 << 21, 1 << 21)] {
            assert!(image_header_fields(width, height, &metadata, 128).is_err());
        }
    }

    #[test]
    fn black_channel_requires_level10_even_with_small_integer_buffers() {
        let base = Image::new(1, 1, PixelLayout::Rgb, PixelData::U8(vec![1, 2, 3])).unwrap();
        let mut metadata = base.metadata().clone();
        metadata.modular_16_bit_buffer_sufficient = true;
        let rgb = base
            .clone()
            .with_metadata(metadata.clone(), vec![])
            .unwrap();
        assert!(container_prefix(&rgb, 32).unwrap().is_empty());
        let mut profile = vec![0; 132];
        profile[..4].copy_from_slice(&132u32.to_be_bytes());
        profile[16..24].copy_from_slice(b"CMYKLab ");
        profile[36..40].copy_from_slice(b"acsp");
        metadata.color_encoding = ColorEncoding::icc(ColorSpace::Rgb, profile).unwrap();
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Black, 8).unwrap());
        let cmyk = base
            .with_metadata(metadata, vec![PixelData::U8(vec![4])])
            .unwrap();
        let prefix = container_prefix(&cmyk, 32).unwrap();
        assert!(prefix.windows(5).any(|w| w == b"jxll\x0a"));
    }
}
