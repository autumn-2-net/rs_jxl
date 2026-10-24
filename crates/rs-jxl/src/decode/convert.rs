use super::{AlphaMode, DecodeOptions, DecodedImage, ExtraChannel, JxlBitDepth, SampleFormat};
use crate::{Error, PixelData, Result};

/// Color conversion operates before quantization. Default decoding never takes
/// this path, preserving native integer values and floating-point NaN payloads.
pub(super) fn finish(image: &mut DecodedImage, options: &DecodeOptions) -> Result<()> {
    if !options.color.needs_float() {
        return Ok(());
    }
    let associated = image
        .info
        .extra_channels
        .iter()
        .find(|ec| ec.ec_type == ExtraChannel::Alpha)
        .is_some_and(|ec| ec.alpha_associated);
    for frame in image.preview.iter_mut().chain(&mut image.frames) {
        if frame.layout.has_alpha()
            && let PixelData::F32(pixels) = &mut frame.pixels
        {
            let premultiply = options.alpha_mode == AlphaMode::Premultiplied && !associated;
            let unpremultiply = options.alpha_mode == AlphaMode::Unpremultiplied && associated;
            if premultiply || unpremultiply {
                for pixel in pixels.chunks_exact_mut(frame.layout.channels()) {
                    let (colors, alpha) = pixel.split_at_mut(frame.layout.channels() - 1);
                    let scale = if premultiply {
                        alpha[0]
                    } else if alpha[0] == 0.0 {
                        1.0
                    } else {
                        alpha[0].recip()
                    };
                    for value in colors {
                        *value *= scale;
                    }
                }
            }
        }
        quantize(&mut frame.pixels, &image.info.bit_depth, options.format)?;
        if options.include_extra_channels {
            for (plane, ec) in frame
                .extra_planes
                .iter_mut()
                .zip(&image.info.extra_channels)
            {
                quantize(plane, &ec.bit_depth, options.format)?;
            }
        } else {
            frame.extra_planes.clear();
        }
    }
    Ok(())
}

fn quantize(data: &mut PixelData, depth: &JxlBitDepth, requested: SampleFormat) -> Result<()> {
    let PixelData::F32(samples) = data else {
        return Ok(());
    };
    let bits = match requested {
        SampleFormat::U8 => 8,
        SampleFormat::U16 => 16,
        SampleFormat::U8Bits(n) | SampleFormat::U16Bits(n) => n,
        SampleFormat::Native => match depth {
            JxlBitDepth::Int { bits_per_sample } if *bits_per_sample <= 16 => {
                *bits_per_sample as u8
            }
            _ => return Ok(()),
        },
        SampleFormat::F32 => return Ok(()),
    };
    let max = ((1u32 << bits) - 1) as f32;
    let wide = bits > 8 || matches!(requested, SampleFormat::U16 | SampleFormat::U16Bits(_));
    macro_rules! convert {
        ($variant:ident, $ty:ty) => {{
            let mut output = Vec::new();
            output
                .try_reserve_exact(samples.len())
                .map_err(|_| Error::Limit("quantized output"))?;
            output.extend(
                samples
                    .iter()
                    .map(|&x| (x.clamp(0.0, 1.0) * max).round() as $ty),
            );
            PixelData::$variant(output)
        }};
    }
    *data = if wide {
        convert!(U16, u16)
    } else {
        convert!(U8, u8)
    };
    Ok(())
}
