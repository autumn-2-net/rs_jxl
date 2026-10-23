// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Metadata-aware patch frame assembly. Frozen b866b3d enc_patch_dictionary.cc
//! FindBestPatchDictionary/SubtractFrom affect color only, write None for every
//! extra, and RoundtripPatchFrame supplies zero placeholder extras. Native
//! planes keep their own depth and ceil(frame / 2^dim_shift) dimensions here;
//! neither the detector nor this adapter upsamples or resamples an extra.
//!
//! This is not a claim that the frozen reference writer handles dim_shift:
//! RoundtripPatchFrame forces ec_resampling=1, whereas frame_header.cc shifts
//! that factor by metadata dim_shift. The native frame writer must emit the
//! existing metadata-aware upsampling field. Decoder dec_cache.cc upsamples
//! extras before patches; None leaves that channel unchanged (blending.cc).
//!
//! Floating COLOR <=16 bits uses the typed float detector and source-format
//! atlas/residual conversion with an exact replay guard. Its stored planes are
//! declared-format words, not signed integer sample differences. Conversion or
//! replay failure is explicit; this adapter never silently substitutes integer
//! bit-pattern arithmetic. Floating extras remain opaque pass-through data.

use crate::modular::sample_convert::SampleFormat;
use crate::{Error, Image, Result, tools::patches};

pub(super) fn read_plane(image: &Image, channel: usize) -> Vec<i32> {
    let (width, height) = image.channel_dimensions(channel);
    (0..width * height)
        .map(|i| image.modular_sample(channel, i))
        .collect()
}

pub(super) fn prepare(
    image: &Image,
    config: &patches::SourcePatchConfig,
) -> Result<Option<(Image, Image, patches::PatchDictionary)>> {
    config.validate()?;
    let shifts: Vec<_> = image
        .metadata()
        .extra_channels
        .iter()
        .map(|e| u32::from(e.dim_shift))
        .collect();
    if !patches::source_supported(
        u32::from(image.bits()),
        image.metadata().bit_depth.is_float(),
        &shifts,
    ) {
        return Ok(None);
    }
    // Source FindTextLikePatches takes Image3F, not the extra-channel planes.
    // Their independent declared depths and sample types never enter detection.
    let channels: Vec<_> = (0..image.color_channels()).collect();
    let planes: Vec<_> = channels.iter().map(|&c| read_plane(image, c)).collect();
    let mut color_config = config.clone();
    color_config.floating_point = image.metadata().bit_depth.is_float();
    color_config.extra_channel_dim_shifts.clear();
    if color_config.floating_point {
        let depth = image.metadata().bit_depth;
        let Some(plan) = patches::discover_source_float(
            patches::FloatImageView {
                width: image.width() as usize,
                height: image.height() as usize,
                format: SampleFormat::Float {
                    bits: depth.bits_per_sample,
                    exponent_bits: depth.exponent_bits_per_sample,
                },
                planes: &planes,
            },
            &color_config,
        )?
        else {
            return Ok(None);
        };
        return Ok(Some(assemble(
            image,
            &channels,
            (plan.reference.width, plan.reference.height),
            plan.reference.planes,
            plan.residual.planes,
            plan.dictionary,
        )?));
    }
    let Some(plan) = patches::discover_source(
        patches::ImageView {
            width: image.width() as usize,
            height: image.height() as usize,
            bit_depth: u32::from(image.bits()),
            color_channels: image.color_channels(),
            planes: &planes,
        },
        &color_config,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(assemble(
        image,
        &channels,
        (plan.reference.width, plan.reference.height),
        plan.reference.planes,
        plan.residual.planes,
        plan.dictionary,
    )?))
}

/// Expand a detector's full-resolution plane subset into the original channel
/// order. Omitted channels use None, original residual words, and zero atlas
/// placeholders. Signed color differences never pass through unsigned storage.
pub(super) fn assemble(
    image: &Image,
    channels: &[usize],
    size: (usize, usize),
    reference: Vec<Vec<i32>>,
    residual: Vec<Vec<i32>>,
    mut dictionary: patches::PatchDictionary,
) -> Result<(Image, Image, patches::PatchDictionary)> {
    let colors = image.color_channels();
    if channels.len() < colors
        || channels[..colors].iter().copied().ne(0..colors)
        || channels.windows(2).any(|pair| pair[0] >= pair[1])
        || channels
            .iter()
            .any(|&c| c >= image.channel_count() || image.channel_shift(c) != 0)
        || reference.len() != channels.len()
        || residual.len() != channels.len()
        || dictionary.num_extra_channels != channels.len() - colors
    {
        return Err(Error::Encoding(
            "invalid patch detector channel mapping".into(),
        ));
    }
    let old_stride = dictionary.validate()?;
    let extras = image.metadata().extra_channels.len();
    let none = patches::PatchBlending {
        mode: patches::PatchBlendMode::None,
        alpha_channel: 0,
        clamp: false,
    };
    let count = dictionary
        .positions
        .len()
        .checked_mul(extras + 1)
        .ok_or(Error::Limit("patch blending count"))?;
    let mut blendings = Vec::with_capacity(count);
    for original in dictionary.blendings.chunks_exact(old_stride) {
        let start = blendings.len();
        blendings.resize(start + extras + 1, none);
        blendings[start] = original[0];
        for (selected, &channel) in channels.iter().enumerate().skip(colors) {
            blendings[start + 1 + channel - colors] = original[1 + selected - colors];
        }
    }
    dictionary.blendings = blendings;
    dictionary.num_extra_channels = extras;
    dictionary.validate()?;

    let mut atlas = Vec::with_capacity(image.channel_count());
    let mut remaining = Vec::with_capacity(image.channel_count());
    let mut selected = channels
        .iter()
        .copied()
        .zip(reference.into_iter().zip(residual))
        .peekable();
    for channel in 0..image.channel_count() {
        if selected.peek().is_some_and(|&(c, _)| c == channel) {
            let (_, (reference, residual)) = selected.next().expect("selected channel");
            atlas.push(reference);
            remaining.push(residual);
        } else {
            let scale = 1usize << image.channel_shift(channel);
            let count = size
                .0
                .div_ceil(scale)
                .checked_mul(size.1.div_ceil(scale))
                .ok_or(Error::Limit("patch extra plane size"))?;
            atlas.push(vec![0; count]);
            remaining.push(read_plane(image, channel));
        }
    }
    let reference = image.with_modular_planes(
        u32::try_from(size.0).map_err(|_| Error::Limit("patch width"))?,
        u32::try_from(size.1).map_err(|_| Error::Limit("patch height"))?,
        atlas,
    )?;
    let residual = image.with_modular_planes(image.width(), image.height(), remaining)?;
    Ok((reference, residual, dictionary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PixelData, PixelLayout,
        metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType},
    };

    fn glyph(x: usize, y: usize) -> bool {
        [(17, 17), (81, 17), (33, 53), (97, 53)]
            .iter()
            .any(|&(x0, y0)| (x0..x0 + 6).contains(&x) && (y0..y0 + 5).contains(&y))
    }

    fn fixture(bits: u8, layout: PixelLayout, with_extras: bool) -> Image {
        let (width, height) = (129usize, 83usize);
        let maximum = ((1u32 << bits) - 1) as u16;
        let colors = if layout.channels() < 3 { 1 } else { 3 };
        let mut samples = Vec::new();
        for y in 0..height {
            for x in 0..width {
                for _ in 0..colors {
                    samples.push(if glyph(x, y) { maximum / 4 } else { maximum });
                }
                if layout.has_alpha() {
                    samples.push(0);
                }
            }
        }
        let pixels = if bits <= 8 {
            PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
        } else {
            PixelData::U16(samples)
        };
        let image = Image::new(width as u32, height as u32, layout, pixels).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.bit_depth = BitDepth::new(bits).unwrap();
        for extra in &mut metadata.extra_channels {
            extra.bit_depth = metadata.bit_depth;
        }
        let mut planes = Vec::new();
        if with_extras {
            // Interleave participating and pass-through channels so dictionary
            // blend indices cannot accidentally follow the detector's subset.
            for (depth, shift) in [
                (BitDepth::new(12).unwrap(), 0),
                (BitDepth::floating(16, 5).unwrap(), 2),
                (BitDepth::new(8).unwrap(), 0),
                (BitDepth::new(31).unwrap(), 0),
                (BitDepth::floating(32, 8).unwrap(), 3),
                (BitDepth::new(16).unwrap(), 1),
            ] {
                let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
                info.bit_depth = depth;
                info.dim_shift = shift;
                info.name = format!("extra{}", planes.len());
                metadata.extra_channels.push(info);
                let count = width.div_ceil(1 << shift) * height.div_ceil(1 << shift);
                planes.push(match (depth.bits_per_sample, depth.is_float()) {
                    (16, true) => PixelData::F16(
                        (0..count)
                            .map(|i| [0x8000, 0x7e31, 0x7c00, 0xfc00, 1, 0x3c00][i % 6])
                            .collect(),
                    ),
                    (32, true) => PixelData::F32(
                        (0..count)
                            .map(|i| {
                                f32::from_bits(
                                    [
                                        0x8000_0000,
                                        0x7fc1_2345,
                                        0x7f80_0000,
                                        0xff80_0000,
                                        1,
                                        0x3f80_0000,
                                    ][i % 6],
                                )
                            })
                            .collect(),
                    ),
                    (31, false) => {
                        PixelData::U32((0..count).map(|i| 0x7fff_ffff - i as u32).collect())
                    }
                    (8, false) => PixelData::U8(
                        (0..count)
                            .map(|i| if glyph(i % width, i / width) { 21 } else { 185 })
                            .collect(),
                    ),
                    (12, false) => PixelData::U16(
                        (0..count)
                            .map(|i| {
                                if glyph(i % width, i / width) {
                                    333
                                } else {
                                    3210
                                }
                            })
                            .collect(),
                    ),
                    _ => {
                        PixelData::U16((0..count).map(|i| (i as u16).wrapping_mul(7919)).collect())
                    }
                });
            }
        }
        image.with_metadata(metadata, planes).unwrap()
    }

    fn replay(
        image: &Image,
        atlas: &Image,
        residual: &Image,
        dictionary: &patches::PatchDictionary,
    ) {
        assert!(
            !dictionary.positions.is_empty(),
            "fixture must activate patches"
        );
        assert_eq!(atlas.metadata(), image.metadata());
        assert_eq!(residual.metadata(), image.metadata());
        assert_eq!(
            dictionary.num_extra_channels,
            image.metadata().extra_channels.len()
        );
        let stride = dictionary.validate().unwrap();
        let mut output: Vec<_> = (0..image.channel_count())
            .map(|c| read_plane(residual, c))
            .collect();
        for (i, position) in dictionary.positions.iter().enumerate() {
            let crop = dictionary.references[position.ref_pos_idx];
            for (channel, plane) in output.iter_mut().enumerate() {
                let blend = if channel < image.color_channels() {
                    0
                } else {
                    1 + channel - image.color_channels()
                };
                let mode = dictionary.blendings[i * stride + blend].mode;
                if mode == patches::PatchBlendMode::None {
                    continue;
                }
                assert_eq!(image.channel_shift(channel), 0);
                assert!(!image.channel_depth(channel).is_float());
                for y in 0..crop.height as usize {
                    for x in 0..crop.width as usize {
                        let dst = (position.y as usize + y) * image.width() as usize
                            + position.x as usize
                            + x;
                        let src =
                            (crop.y0 as usize + y) * atlas.width() as usize + crop.x0 as usize + x;
                        let value = atlas.modular_sample(channel, src);
                        match mode {
                            patches::PatchBlendMode::Add => plane[dst] += value,
                            patches::PatchBlendMode::Replace => plane[dst] = value,
                            _ => panic!("unexpected blend"),
                        }
                    }
                }
            }
        }
        for (channel, plane) in output.iter().enumerate() {
            assert_eq!(*plane, read_plane(image, channel), "channel={channel}");
            assert_eq!(
                residual.channel_dimensions(channel),
                image.channel_dimensions(channel)
            );
            let scale = 1usize << image.channel_shift(channel);
            assert_eq!(
                atlas.channel_dimensions(channel),
                (
                    (atlas.width() as usize).div_ceil(scale),
                    (atlas.height() as usize).div_ceil(scale)
                )
            );
        }
        patches::tokenize(dictionary).unwrap();
    }

    #[test]
    fn source_add_keeps_native_extra_words_and_dimensions_including_hidden_rgb() {
        for bits in [8, 16] {
            for layout in [
                PixelLayout::Gray,
                PixelLayout::GrayAlpha,
                PixelLayout::Rgb,
                PixelLayout::Rgba,
            ] {
                let image = fixture(bits, layout, true);
                let original = image.clone();
                let (atlas, residual, dictionary) =
                    prepare(&image, &patches::SourcePatchConfig::default())
                        .unwrap()
                        .expect("active Add fixture");
                assert_ne!(
                    (atlas.width(), atlas.height()),
                    (image.width(), image.height())
                );
                let stride = dictionary.validate().unwrap();
                for blends in dictionary.blendings.chunks_exact(stride) {
                    assert_eq!(blends[0].mode, patches::PatchBlendMode::Add);
                    assert!(
                        blends[1..]
                            .iter()
                            .all(|b| b.mode == patches::PatchBlendMode::None)
                    );
                }
                for channel in image.color_channels()..image.channel_count() {
                    assert_eq!(read_plane(&residual, channel), read_plane(&image, channel));
                    assert!(read_plane(&atlas, channel).iter().all(|&word| word == 0));
                }
                assert!(
                    (0..image.color_channels())
                        .any(|c| read_plane(&atlas, c).iter().any(|&v| v < 0))
                );
                replay(&image, &atlas, &residual, &dictionary);
                assert_eq!(image, original);
            }
        }
    }

    #[test]
    fn replace_maps_additional_mixed_depth_planes_and_leaves_other_extras_untouched() {
        for bits in [8, 16] {
            for layout in [
                PixelLayout::Gray,
                PixelLayout::GrayAlpha,
                PixelLayout::Rgb,
                PixelLayout::Rgba,
            ] {
                let image = fixture(bits, layout, true);
                let original = image.clone();
                let (atlas, residual, dictionary) =
                    super::super::prepare(&image, &patches::PatchOptions::default())
                        .unwrap()
                        .expect("active Replace fixture");
                let stride = dictionary.validate().unwrap();
                for blends in dictionary.blendings.chunks_exact(stride) {
                    assert_eq!(blends[0].mode, patches::PatchBlendMode::Replace);
                    for (extra, info) in image.metadata().extra_channels.iter().enumerate() {
                        let replaced = info.dim_shift == 0
                            && !info.bit_depth.is_float()
                            && info.bit_depth.bits_per_sample <= 16;
                        assert_eq!(
                            blends[1 + extra].mode,
                            if replaced {
                                patches::PatchBlendMode::Replace
                            } else {
                                patches::PatchBlendMode::None
                            }
                        );
                    }
                }
                replay(&image, &atlas, &residual, &dictionary);
                assert_eq!(image, original);
            }
        }
    }

    #[test]
    fn source_detection_ignores_extra_types_and_preserves_all_nominal_color_depths() {
        for bits in 1..=16 {
            let plain = fixture(bits, PixelLayout::Rgb, false);
            let extras = fixture(bits, PixelLayout::Rgb, true);
            let config = patches::SourcePatchConfig {
                speed_tier: if bits <= 8 { -1 } else { 5 },
                ..Default::default()
            };
            let (plain_atlas, _, plain_dictionary) = prepare(&plain, &config)
                .unwrap()
                .expect("active color fixture");
            let (atlas, residual, dictionary) = prepare(&extras, &config)
                .unwrap()
                .expect("active extra fixture");
            assert_eq!(dictionary.positions, plain_dictionary.positions);
            assert_eq!(dictionary.references, plain_dictionary.references);
            for channel in 0..extras.color_channels() {
                assert_eq!(
                    read_plane(&atlas, channel),
                    read_plane(&plain_atlas, channel)
                );
            }
            replay(&extras, &atlas, &residual, &dictionary);
        }
    }

    #[test]
    fn float_add_assembles_declared_words_and_preserves_opaque_mixed_extras() {
        for layout in [
            PixelLayout::Gray,
            PixelLayout::GrayAlpha,
            PixelLayout::Rgb,
            PixelLayout::Rgba,
        ] {
            let base = fixture(16, layout, true);
            let mut words = Vec::new();
            for y in 0..base.height() as usize {
                for x in 0..base.width() as usize {
                    let right = x >= base.width() as usize / 2;
                    let value = if glyph(x, y) {
                        if right { 0x3c00 } else { 0x3800 } // 1 or 0.5
                    } else if right {
                        0x3e00
                    } else {
                        0x3c00
                    }; // 1.5 or 1
                    for _ in 0..base.color_channels() {
                        words.push(value);
                    }
                    if layout.has_alpha() {
                        // Negative zero alpha must not erase hidden color.
                        words.push(0x8000);
                    }
                }
            }
            let special = [0x8000, 0x7d31, 0xfe23, 0x7c00, 0xfc00, 1];
            for (x, &word) in special.iter().enumerate() {
                for channel in 0..base.color_channels() {
                    words[x * layout.channels() + channel] = word;
                }
            }
            let mut metadata = base.metadata().clone();
            metadata.bit_depth = BitDepth::floating(16, 5).unwrap();
            if layout.has_alpha() {
                metadata.extra_channels[0].bit_depth = metadata.bit_depth;
            }
            let image = Image::new(base.width(), base.height(), layout, PixelData::F16(words))
                .unwrap()
                .with_metadata(metadata, base.extra_planes().to_vec())
                .unwrap();
            let config = patches::SourcePatchConfig {
                speed_tier: -1,
                ..Default::default()
            };
            let (atlas, residual, dictionary) = prepare(&image, &config)
                .unwrap()
                .expect("active float Add fixture");
            assert!(!dictionary.positions.is_empty());
            assert_eq!(atlas.metadata(), image.metadata());
            assert_eq!(residual.metadata(), image.metadata());
            assert_ne!(
                (atlas.width(), atlas.height()),
                (image.width(), image.height())
            );
            let stride = dictionary.validate().unwrap();
            for blendings in dictionary.blendings.chunks_exact(stride) {
                assert_eq!(blendings[0].mode, patches::PatchBlendMode::Add);
                assert!(
                    blendings[1..]
                        .iter()
                        .all(|b| b.mode == patches::PatchBlendMode::None)
                );
            }
            let float_image = |image: &Image| patches::FloatPatchImage {
                width: image.width() as usize,
                height: image.height() as usize,
                format: SampleFormat::BINARY16,
                planes: (0..image.color_channels())
                    .map(|c| read_plane(image, c))
                    .collect(),
            };
            let replayed = patches::compose_float_add(
                &float_image(&residual),
                &float_image(&atlas),
                &dictionary,
            )
            .unwrap();
            for (channel, plane) in replayed.iter().enumerate() {
                assert_eq!(*plane, read_plane(&image, channel));
                assert_eq!(&plane[..special.len()], &special.map(i32::from));
            }
            for channel in image.color_channels()..image.channel_count() {
                assert_eq!(read_plane(&residual, channel), read_plane(&image, channel));
                assert_eq!(
                    residual.channel_dimensions(channel),
                    image.channel_dimensions(channel)
                );
                assert!(read_plane(&atlas, channel).iter().all(|&word| word == 0));
                let scale = 1usize << image.channel_shift(channel);
                assert_eq!(
                    atlas.channel_dimensions(channel),
                    (
                        (atlas.width() as usize).div_ceil(scale),
                        (atlas.height() as usize).div_ceil(scale),
                    )
                );
            }
            patches::tokenize(&dictionary).unwrap();
        }
    }

    #[test]
    fn large_formats_and_empty_float_detections_do_not_reinterpret_bits() {
        for pixels in [
            PixelData::F16(vec![0x8000, 0x7e31, 0x3c00]),
            PixelData::F32(vec![-0.0, f32::from_bits(0x7fc1_2345), 1.0]),
            PixelData::U32(vec![0, 1, 0x7fff_ffff]),
        ] {
            let image = Image::new(3, 1, PixelLayout::Gray, pixels).unwrap();
            let original = image.clone();
            assert!(
                prepare(&image, &patches::SourcePatchConfig::default())
                    .unwrap()
                    .is_none()
            );
            assert!(
                super::super::prepare(&image, &patches::PatchOptions::default())
                    .unwrap()
                    .is_none()
            );
            assert_eq!(image, original);
        }
        // F16 is supported, but the tiny input above contains no patch seeds.
        assert!(patches::source_enabled(16, 5, 0, true, false));
        assert!(patches::source_supported(16, true, &[]));
    }
}
