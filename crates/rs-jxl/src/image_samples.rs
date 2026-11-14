//! Region copies of immutable, constructor-validated Modular sample storage.

#![forbid(unsafe_code)]

use super::{BitDepth, Error, Image, PixelData, Result};

#[derive(Clone, Copy)]
struct Rect {
    source_width: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Image {
    /// Replace `out` with raster-order Modular samples, retaining its allocation.
    /// Coordinates use the selected channel's native (possibly subsampled)
    /// dimensions. Empty rectangles are allowed, including at the bottom/right
    /// boundary. Invalid regions or allocation errors leave `out` unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_modular_rect(
        &self,
        channel: usize,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
        out: &mut Vec<i32>,
    ) -> Result<()> {
        if channel >= self.channel_count() {
            return Err(Error::InvalidInput(
                "Modular copy channel out of range".into(),
            ));
        }
        let (source_width, source_height) = self.channel_dimensions(channel);
        let end_x = x0
            .checked_add(width)
            .ok_or(Error::Limit("Modular copy rectangle"))?;
        let end_y = y0
            .checked_add(height)
            .ok_or(Error::Limit("Modular copy rectangle"))?;
        if end_x > source_width || end_y > source_height {
            return Err(Error::InvalidInput(
                "Modular copy rectangle out of range".into(),
            ));
        }
        let count = width
            .checked_mul(height)
            .ok_or(Error::Limit("Modular copy sample count"))?;
        out.try_reserve(count.saturating_sub(out.len()))
            .map_err(|_| Error::Limit("Modular copy allocation"))?;
        out.clear();
        if count == 0 {
            return Ok(());
        }
        let rect = Rect {
            source_width,
            x: x0,
            y: y0,
            width,
            height,
        };
        if let Some(planes) = &self.modular_planes {
            for y in y0..end_y {
                let start = y * source_width + x0;
                out.extend_from_slice(&planes[channel][start..start + width]);
            }
            return Ok(());
        }
        let depth = self.channel_depth(channel);
        let interleaved = self.layout.channels();
        if channel >= interleaved {
            copy_pixels::<1>(
                &self.extra_planes[channel - interleaved],
                depth,
                0,
                rect,
                out,
            );
        } else {
            match interleaved {
                1 => copy_pixels::<1>(&self.pixels, depth, channel, rect, out),
                2 => copy_pixels::<2>(&self.pixels, depth, channel, rect, out),
                3 => copy_pixels::<3>(&self.pixels, depth, channel, rect, out),
                4 => copy_pixels::<4>(&self.pixels, depth, channel, rect, out),
                _ => unreachable!("validated pixel layout"),
            }
        }
        Ok(())
    }
}

fn copy_pixels<const CHANNELS: usize>(
    pixels: &PixelData,
    depth: BitDepth,
    channel: usize,
    rect: Rect,
    out: &mut Vec<i32>,
) {
    match pixels {
        PixelData::U8(values) => copy_typed::<_, CHANNELS>(values, channel, rect, out, i32::from),
        PixelData::U16(values) | PixelData::F16(values) => {
            copy_typed::<_, CHANNELS>(values, channel, rect, out, i32::from);
        }
        PixelData::U32(values) => {
            copy_typed::<_, CHANNELS>(values, channel, rect, out, |v| v as i32);
        }
        PixelData::F32(values) if depth == pixels.bit_depth() => {
            copy_typed::<_, CHANNELS>(values, channel, rect, out, |v| v.to_bits() as i32);
        }
        PixelData::F32(_) => {
            // Custom/reduced float words keep the same checked conversion as
            // modular_sample; channel metadata is still resolved only once.
            for y in rect.y..rect.y + rect.height {
                let start = (y * rect.source_width + rect.x) * CHANNELS + channel;
                for x in 0..rect.width {
                    out.push(pixels.encoded(start + x * CHANNELS, depth));
                }
            }
        }
    }
}

fn copy_typed<T: Copy, const CHANNELS: usize>(
    values: &[T],
    channel: usize,
    rect: Rect,
    out: &mut Vec<i32>,
    convert: impl Fn(T) -> i32,
) {
    // Image constructors bound complete rows, including channel interleaving;
    // the checked native rectangle bounds every slice and product below.
    let row_stride = rect.source_width * CHANNELS;
    for y in rect.y..rect.y + rect.height {
        let start = y * row_stride + rect.x * CHANNELS;
        let row = &values[start..start + rect.width * CHANNELS];
        out.extend(
            row.chunks_exact(CHANNELS)
                .map(|pixel| convert(pixel[channel])),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::PixelLayout;
    use crate::metadata::{ExtraChannelInfo, ExtraChannelType};

    const LAYOUTS: [PixelLayout; 4] = [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ];

    fn compare_regions(image: &Image) {
        let mut actual = vec![123, -456];
        for channel in 0..image.channel_count() {
            let (source_width, source_height) = image.channel_dimensions(channel);
            for y in 0..=source_height {
                for x in 0..=source_width {
                    for height in 0..=source_height - y {
                        for width in 0..=source_width - x {
                            let mut expected = Vec::new();
                            for row in y..y + height {
                                for column in x..x + width {
                                    expected.push(
                                        image.modular_sample(channel, row * source_width + column),
                                    );
                                }
                            }
                            image
                                .copy_modular_rect(channel, x, y, width, height, &mut actual)
                                .unwrap();
                            assert_eq!(
                                actual, expected,
                                "channel={channel} rect={x},{y},{width},{height}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn typed_integer_regions_match_samples_for_every_layout_and_depth() {
        for layout in LAYOUTS {
            for storage_bits in [8u8, 16, 32] {
                for bits in 1..=storage_bits.min(31) {
                    let alpha_bits = bits.div_ceil(2);
                    let values: Vec<u32> = (0..15 * layout.channels())
                        .map(|i| {
                            let depth = if layout.has_alpha()
                                && i % layout.channels() == layout.channels() - 1
                            {
                                alpha_bits
                            } else {
                                bits
                            };
                            let mask = (1u32 << depth) - 1;
                            [
                                0,
                                mask,
                                1,
                                mask >> 1,
                                (i as u32).wrapping_mul(0x5c12_3fa1) & mask,
                            ][i % 5]
                        })
                        .collect();
                    let pixels = match storage_bits {
                        8 => PixelData::U8(values.iter().map(|&v| v as u8).collect()),
                        16 => PixelData::U16(values.iter().map(|&v| v as u16).collect()),
                        _ => PixelData::U32(values),
                    };
                    let image = Image::new(5, 3, layout, pixels).unwrap();
                    let mut metadata = image.metadata().clone();
                    metadata.bit_depth = BitDepth::new(bits).unwrap();
                    if layout.has_alpha() {
                        metadata.extra_channels[0].bit_depth = BitDepth::new(alpha_bits).unwrap();
                    }
                    let image = image.with_metadata(metadata, vec![]).unwrap();
                    compare_regions(&image);
                }
            }
        }
    }

    fn with_extras(image: Image) -> Image {
        let mut metadata = image.metadata().clone();
        let mut planes = Vec::new();
        for (depth, shift, pixels) in [
            (
                BitDepth::new(3).unwrap(),
                1,
                PixelData::U8(vec![0, 7, 1, 2, 6, 3]),
            ),
            (BitDepth::new(12).unwrap(), 2, PixelData::U16(vec![4095, 0])),
            (
                BitDepth::new(31).unwrap(),
                0,
                PixelData::U32((0..15).map(|i| [0, i32::MAX as u32, 1][i % 3]).collect()),
            ),
            (
                BitDepth::floating(16, 5).unwrap(),
                1,
                PixelData::F16(vec![0, 0x8000, 1, 0x7c00, 0x7e01, 0xffff]),
            ),
            (
                BitDepth::floating(32, 8).unwrap(),
                2,
                PixelData::F32(vec![-0.0, f32::from_bits(0x7f80_0001)]),
            ),
            (
                BitDepth::floating(16, 5).unwrap(),
                3,
                PixelData::F32(vec![-0.0]),
            ),
        ] {
            let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
            info.bit_depth = depth;
            info.dim_shift = shift;
            metadata.extra_channels.push(info);
            planes.push(pixels);
        }
        image.with_metadata(metadata, planes).unwrap()
    }

    #[test]
    fn native_subsampled_extras_match_samples_for_every_layout() {
        for layout in LAYOUTS {
            let image =
                Image::new(5, 3, layout, PixelData::U8(vec![0; 15 * layout.channels()])).unwrap();
            compare_regions(&with_extras(image));
        }
    }

    #[test]
    fn native_float_words_preserve_signed_zero_subnormals_and_nan_payloads() {
        for layout in LAYOUTS {
            let count = 15 * layout.channels();
            for pixels in [
                PixelData::F16(
                    (0..count)
                        .map(|i| {
                            [
                                0, 0x8000, 1, 0x03ff, 0x0400, 0x7bff, 0x7c00, 0xfc00, 0x7c01,
                                0x7e01, 0xffff,
                            ][i % 11]
                        })
                        .collect(),
                ),
                PixelData::F32(
                    (0..count)
                        .map(|i| {
                            f32::from_bits(
                                [
                                    0,
                                    0x8000_0000,
                                    1,
                                    0x007f_ffff,
                                    0x0080_0000,
                                    0x7f7f_ffff,
                                    0x7f80_0000,
                                    0xff80_0000,
                                    0x7f80_0001,
                                    0x7fc1_2345,
                                    0xffff_ffff,
                                ][i % 11],
                            )
                        })
                        .collect(),
                ),
            ] {
                let image = with_extras(Image::new(5, 3, layout, pixels).unwrap());
                compare_regions(&image);
                let planes = (0..image.channel_count())
                    .map(|c| {
                        let (w, h) = image.channel_dimensions(c);
                        (0..w * h).map(|i| image.modular_sample(c, i)).collect()
                    })
                    .collect();
                compare_regions(&image.with_modular_planes(5, 3, planes).unwrap());
            }
        }
    }

    #[test]
    fn reduced_float_metadata_keeps_declared_words_including_distinct_alpha_depth() {
        use crate::modular::sample_convert::{SampleFormat, int_to_float};
        for layout in LAYOUTS {
            for (bits, exponent_bits) in [(16, 5), (24, 7), (16, 8), (5, 2)] {
                let mantissa = bits - exponent_bits - 1;
                let sign = 1u32 << (bits - 1);
                let exponent_max = (1u32 << exponent_bits) - 1;
                let bias = (1u32 << (exponent_bits - 1)) - 1;
                let words = [
                    0,
                    sign,
                    1 << mantissa,
                    bias << mantissa,
                    (bias << mantissa) | 1,
                    exponent_max << mantissa,
                    sign | (exponent_max << mantissa),
                    (exponent_max << mantissa) | 1,
                ];
                let input: Vec<i32> = (0..15 * layout.channels())
                    .map(|i| words[i % words.len()] as i32)
                    .collect();
                let mut floats = vec![0.0; input.len()];
                int_to_float(
                    &input,
                    &mut floats,
                    SampleFormat::Float {
                        bits,
                        exponent_bits,
                    },
                    1.0,
                )
                .unwrap();
                let image = Image::new(5, 3, layout, PixelData::F32(floats)).unwrap();
                let mut metadata = image.metadata().clone();
                metadata.bit_depth = BitDepth::floating(bits, exponent_bits).unwrap();
                // Interleaved alpha deliberately retains native binary32 depth.
                let image = with_extras(image.with_metadata(metadata, vec![]).unwrap());
                compare_regions(&image);
            }
        }
    }

    #[test]
    fn signed_working_planes_use_their_own_dimensions_and_keep_extremes() {
        for layout in LAYOUTS {
            let image = with_extras(
                Image::new(5, 3, layout, PixelData::U8(vec![0; 15 * layout.channels()])).unwrap(),
            );
            let planes = (0..image.channel_count())
                .map(|c| {
                    let scale = 1usize << image.channel_shift(c);
                    let count = 7usize.div_ceil(scale) * 5usize.div_ceil(scale);
                    (0..count)
                        .map(|i| {
                            if image.channel_depth(c).is_float() {
                                image.modular_sample(c, 0)
                            } else {
                                [i32::MIN, i32::MAX, -1, 0, 1, -65536, 65535][(i + c) % 7]
                            }
                        })
                        .collect()
                })
                .collect();
            compare_regions(&image.with_modular_planes(7, 5, planes).unwrap());
        }
    }

    #[test]
    fn invalid_regions_leave_destination_unchanged_and_valid_copies_reuse_capacity() {
        let image =
            with_extras(Image::new(5, 3, PixelLayout::Rgba, PixelData::U8(vec![0; 60])).unwrap());
        for (c, x, y, w, h) in [
            (image.channel_count(), 0, 0, 1, 1),
            (usize::MAX, 0, 0, 0, 0),
            (0, 5, 0, 1, 1),
            (0, 0, 3, 1, 1),
            (0, 6, 0, 0, 0),
            (0, 0, 4, 0, 0),
            (0, usize::MAX, 0, 1, 1),
            (0, 0, usize::MAX, 1, 1),
            (0, 1, 0, usize::MAX, 0),
            (0, 0, 1, 0, usize::MAX),
            (4, 3, 0, 1, 1),
            (4, 0, 2, 1, 1),
        ] {
            let mut out = vec![123, -456];
            assert!(image.copy_modular_rect(c, x, y, w, h, &mut out).is_err());
            assert_eq!(out, [123, -456]);
        }
        let mut out = Vec::with_capacity(64);
        image.copy_modular_rect(0, 0, 0, 5, 3, &mut out).unwrap();
        assert_eq!(out.len(), 15);
        image.copy_modular_rect(0, 4, 2, 1, 1, &mut out).unwrap();
        assert_eq!(out, [0]);
        assert_eq!(out.capacity(), 64);
        image.copy_modular_rect(0, 5, 3, 0, 0, &mut out).unwrap();
        assert!(out.is_empty());
        assert_eq!(out.capacity(), 64);
    }
}
