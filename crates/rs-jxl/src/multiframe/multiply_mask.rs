//! Two ordinary integer layers implement a mask without adding an alpha channel.
//! Multiply retains an unchanged sample with one or clears a changed sample with zero;
//! Add then supplies only changed target samples. Alpha is an independent sample here.

use super::{Rect, compatible, diff::DiffMap};
use crate::{Error, Image, PixelData, Result, metadata::ExtraChannelType};

fn supported(image: &Image) -> bool {
    let metadata = image.metadata();
    let depth = metadata.bit_depth;
    if image.has_modular_planes()
        || !matches!(image.pixels(), PixelData::U8(_) | PixelData::U16(_))
        || depth.is_float()
        || !(1..=16).contains(&depth.bits_per_sample)
        || !image.extra_planes().is_empty()
    {
        return false;
    }
    if !image.layout().has_alpha() {
        return metadata.extra_channels.is_empty();
    }
    let [alpha] = metadata.extra_channels.as_slice() else {
        return false;
    };
    alpha.channel_type == ExtraChannelType::Alpha
        && alpha.dim_shift == 0
        && !alpha.bit_depth.is_float()
        && (1..=16).contains(&alpha.bit_depth.bits_per_sample)
}

pub(super) fn eligible(previous: &Image, current: &Image) -> bool {
    supported(previous) && supported(current) && compatible(previous, current).is_ok()
}

/// `diff` must have been computed against this current input, including all its channels.
/// Return (Multiply keep mask, Add target update), with unchanged sample metadata.
///
/// These layers must bypass generated patch atlases. The ordinary integer-to-F32 stage maps
/// each declared-depth maximum to exactly 1.0, so repeated multiplication retains the old
/// normalized word. Source Add patches only promise recovery after integer quantization and
/// cannot establish that intermediate identity. The trial encoder uses patch-free options for these two layers.
pub(super) fn images(current: &Image, rect: Rect, diff: &DiffMap) -> Result<(Image, Image)> {
    if !supported(current) || diff.width() != current.width() || diff.height() != current.height() {
        return Err(Error::InvalidInput(
            "multiply mask requires matching integer inputs".into(),
        ));
    }
    if rect.width == 0
        || rect.height == 0
        || rect
            .x
            .checked_add(rect.width)
            .is_none_or(|x| x > current.width())
        || rect
            .y
            .checked_add(rect.height)
            .is_none_or(|y| y > current.height())
    {
        return Err(Error::InvalidInput(
            "multiply mask crop is outside image".into(),
        ));
    }
    let channels = current.layout().channels();
    let count = (rect.width as usize)
        .checked_mul(rect.height as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(Error::Limit("multiply mask sample count"))?;
    let mut maxima = [0u16; 4];
    for (c, maximum) in maxima[..channels].iter_mut().enumerate() {
        *maximum = ((1u32 << current.channel_depth(c).bits_per_sample) - 1) as u16;
    }
    macro_rules! construct {
        ($source:expr, $variant:ident, $word:ty) => {{
            let mut keep = Vec::<$word>::new();
            let mut update = Vec::<$word>::new();
            keep.try_reserve_exact(count)
                .map_err(|_| Error::Limit("multiply mask allocation"))?;
            update
                .try_reserve_exact(count)
                .map_err(|_| Error::Limit("multiply mask allocation"))?;
            let pixel = maxima.map(|v| v as $word);
            keep.resize(count, pixel[0]);
            if pixel[1..channels].iter().any(|&v| v != pixel[0]) {
                for sample in keep.chunks_exact_mut(channels) {
                    sample.copy_from_slice(&pixel[..channels]);
                }
            }
            update.resize(count, 0);
            diff.changed_spans(rect, |src, dst, len| {
                let range = dst * channels..(dst + len) * channels;
                keep[range.clone()].fill(0);
                let source = &$source[src * channels..(src + len) * channels];
                for (target, &value) in update[range].iter_mut().zip(source) {
                    *target = value as $word;
                }
            });
            (PixelData::$variant(keep), PixelData::$variant(update))
        }};
    }
    let (keep, update) = match current.pixels() {
        PixelData::U8(source) if maxima[..channels].iter().all(|&v| v <= u8::MAX as u16) => {
            construct!(source, U8, u8)
        }
        // A U8 input may declare a larger nominal depth; its keep maximum still needs U16.
        PixelData::U8(source) => construct!(source, U16, u16),
        PixelData::U16(source) => construct!(source, U16, u16),
        _ => unreachable!("integer storage checked above"),
    };
    let make = |pixels| {
        Image::new(rect.width, rect.height, current.layout(), pixels)?
            .with_metadata(current.metadata().clone(), vec![])
    };
    Ok((make(keep)?, make(update)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PixelLayout,
        metadata::{BitDepth, ExtraChannelInfo},
    };

    fn image(layout: PixelLayout, bits: u8, alpha_bits: u8) -> Image {
        let channels = layout.channels();
        let mut values = vec![3u16; 5 * 3 * channels];
        if layout.has_alpha() {
            for pixel in values.chunks_exact_mut(channels) {
                pixel[channels - 1] = 0;
            }
        }
        let image = Image::new(5, 3, layout, PixelData::U16(values)).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.bit_depth = BitDepth::new(bits).unwrap();
        if layout.has_alpha() {
            metadata.extra_channels[0].bit_depth = BitDepth::new(alpha_bits).unwrap();
            metadata.extra_channels[0].alpha_associated = true;
        }
        image.with_metadata(metadata, vec![]).unwrap()
    }

    #[test]
    fn declared_maxima_are_exact_normalized_multiplicative_identities() {
        // Same scalar arithmetic as ConvertModular{16,}ToF32Stage. No tolerance or final
        // integer rounding can hide an intermediate mask value different from zero/one.
        for bits in 1..=16 {
            let maximum = ((1u32 << bits) - 1) as f32;
            let scale = 1.0f32 / maximum;
            assert_eq!((maximum * scale).to_bits(), 1.0f32.to_bits(), "bits={bits}");
            assert_eq!((0.0f32 * scale).to_bits(), 0.0f32.to_bits(), "bits={bits}");
        }
    }

    #[test]
    fn all_layouts_use_independent_color_and_alpha_maxima() {
        for (layout, bits, alpha_bits) in [
            (PixelLayout::Gray, 8, 8),
            (PixelLayout::GrayAlpha, 12, 1),
            (PixelLayout::Rgb, 16, 16),
            (PixelLayout::Rgba, 10, 16),
        ] {
            let previous = image(layout, bits, alpha_bits);
            let mut values = match previous.pixels() {
                PixelData::U16(values) => values.clone(),
                _ => unreachable!(),
            };
            let channels = layout.channels();
            values[channels] ^= 1;
            // Include an alpha-only change when present, under otherwise unchanged color.
            values[(2 * 5 + 3) * channels + channels - 1] ^= 1;
            let current = Image::new(5, 3, layout, PixelData::U16(values))
                .unwrap()
                .with_metadata(previous.metadata().clone(), vec![])
                .unwrap();
            assert!(eligible(&previous, &current));
            let diff = DiffMap::between(&previous, &current).unwrap();
            let rect = Rect {
                x: 1,
                y: 0,
                width: 3,
                height: 3,
            };
            let (keep, update) = images(&current, rect, &diff).unwrap();
            assert_eq!(keep.metadata(), current.metadata());
            assert_eq!(update.metadata(), current.metadata());
            assert_eq!((keep.width(), keep.height()), (3, 3));
            assert_eq!((update.width(), update.height()), (3, 3));
            for y in 0..3usize {
                for x in 0..3usize {
                    for c in 0..channels {
                        let changed = diff.changed(x as u32 + 1, y as u32);
                        let index = y * 3 + x;
                        let maximum = (1i32 << current.channel_depth(c).bits_per_sample) - 1;
                        assert_eq!(
                            keep.modular_sample(c, index),
                            if changed { 0 } else { maximum }
                        );
                        assert_eq!(
                            update.modular_sample(c, index),
                            if changed {
                                current.modular_sample(c, y * 5 + x + 1)
                            } else {
                                0
                            }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unsupported_samples_extras_geometry_and_bad_crops_are_rejected() {
        let integer = image(PixelLayout::Rgb, 8, 8);
        let float = Image::new(5, 3, PixelLayout::Gray, PixelData::F32(vec![0.5; 15])).unwrap();
        let wide = Image::new(5, 3, PixelLayout::Gray, PixelData::U32(vec![1 << 20; 15])).unwrap();
        for image in [&float, &wide] {
            assert!(!eligible(image, image));
        }
        for channel_type in [ExtraChannelType::Depth, ExtraChannelType::Alpha] {
            let mut metadata = integer.metadata().clone();
            let mut extra = ExtraChannelInfo::new(channel_type, 8).unwrap();
            extra.dim_shift = 1;
            metadata.extra_channels.push(extra);
            let extra = integer
                .clone()
                .with_metadata(metadata, vec![PixelData::U8(vec![0; 6])])
                .unwrap();
            assert!(!eligible(&extra, &extra));
        }
        let small = Image::new(1, 1, PixelLayout::Rgb, PixelData::U8(vec![0; 3])).unwrap();
        assert!(!eligible(&integer, &small));
        let diff = DiffMap::between(&integer, &integer).unwrap();
        for rect in [
            Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 1,
            },
            Rect {
                x: u32::MAX,
                y: 0,
                width: 2,
                height: 1,
            },
            Rect {
                x: 4,
                y: 0,
                width: 2,
                height: 1,
            },
        ] {
            assert!(images(&integer, rect, &diff).is_err());
        }
        assert!(
            images(
                &small,
                Rect {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1
                },
                &diff
            )
            .is_err()
        );
    }
}
