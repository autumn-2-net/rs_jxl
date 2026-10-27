// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! encode.cc::CanDoFastLossless typed-input adapter. The ordinary cumulative
//! integer search deliberately bypasses FJXL, including requested effort 1.

use super::{params::CandidateParams, policy::default_pixel_options};
use crate::{EncodeOptions, Error, Image, PixelData, Result, fast_lossless};

fn view(image: &Image) -> Option<fast_lossless::ImageView<'_>> {
    if image.has_modular_planes()
        || image.metadata().animation.is_some()
        || image.bits() > 16
        || !image.extra_planes().is_empty()
        || image.metadata().extra_channels.len() != usize::from(image.layout().has_alpha())
        || image.metadata().extra_channels.iter().any(|channel| {
            channel.bit_depth != image.metadata().bit_depth || channel.dim_shift != 0
        })
    {
        return None;
    }
    let (pixels, bytes) = match image.pixels() {
        PixelData::U8(values) if image.bits() <= 8 => (values.as_slice(), 1),
        PixelData::U16(values) | PixelData::F16(values) if image.bits() > 8 => {
            // SAFETY: u16 has no padding/uninitialized bytes. A shared byte view
            // is allowed to alias its immutable source, with exactly its length
            // and lifetime; no byte reference can escape the borrowed Image.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values.as_slice()),
                )
            };
            (bytes, 2)
        }
        _ => return None,
    };
    Some(fast_lossless::ImageView {
        pixels,
        width: image.width() as usize,
        height: image.height() as usize,
        row_stride: image.width() as usize * image.layout().channels() * bytes,
        channels: image.layout().channels() as u8,
        bit_depth: image.bits(),
        big_endian: cfg!(target_endian = "big"),
    })
}

pub(super) fn eligible(image: &Image, params: &CandidateParams, base: &EncodeOptions) -> bool {
    params.speed_tier == 9
        && (image.metadata().bit_depth.is_float() || !default_pixel_options(params))
        && !params.responsive
        && !base.responsive
        && base.passes.num_passes == 1
        && view(image).is_some()
}

/// Outer image metadata is not replaced by FJXL's standalone sRGB header.
pub(super) fn encode(image: &Image, base: &EncodeOptions) -> Result<Vec<u8>> {
    crate::encode::validate_frames(image, &[crate::frame::Frame::new(image)], base)?;
    if base.threads == 1 || crate::execution::is_serial() {
        return encode_in_pool(image, base, None);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(base.threads)
        .build()
        .map_err(|error| Error::Encoding(error.to_string()))?;
    encode_in_pool(image, base, Some(&pool))
}

pub(super) fn encode_in_pool(
    image: &Image,
    base: &EncodeOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<u8>> {
    let image_view =
        view(image).ok_or_else(|| Error::InvalidInput("ineligible FJXL input".into()))?;
    crate::encode::validate_frames(image, &[crate::frame::Frame::new(image)], base)?;
    let mut stream = crate::codestream::image_header(image, base.max_output_bytes)?;
    let maximum = base
        .max_output_bytes
        .checked_sub(stream.len())
        .filter(|&n| n > 0)
        .ok_or(Error::Limit("FJXL frame output bytes"))?;
    let options = fast_lossless::Options {
        limits: fast_lossless::Limits {
            max_pixels: base.max_pixels,
            max_output_bytes: maximum,
            ..Default::default()
        },
        ..Default::default()
    };
    let bytes = match pool {
        Some(pool) => fast_lossless::encode_frame_with_pool(image_view, &options, true, pool)?,
        None => {
            crate::execution::serial(|| fast_lossless::encode_frame(image_view, &options, true))?
        }
    };
    stream
        .try_reserve(bytes.len())
        .map_err(|_| Error::Limit("FJXL codestream allocation"))?;
    stream.extend(bytes);
    crate::encode::wrap_stream(image, stream, base.max_output_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelLayout, metadata::Animation, scheduler::params::Override};

    #[test]
    fn cumulative_levels_bypass_fjxl_and_unsupported_forms_keep_modular() {
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 48])).unwrap();
        let base = EncodeOptions::default();
        let mut params = CandidateParams {
            speed_tier: 9,
            ..Default::default()
        };
        assert!(!eligible(&image, &params, &base));
        params.patches = Override::Off;
        assert!(eligible(&image, &params, &base));
        params.responsive = true;
        assert!(!eligible(&image, &params, &base));
        params.responsive = false;
        params.speed_tier = 8;
        assert!(!eligible(&image, &params, &base));
        params.speed_tier = 9;
        let mut metadata = image.metadata().clone();
        metadata.animation = Some(Animation::default());
        assert!(!eligible(
            &image.clone().with_metadata(metadata, vec![]).unwrap(),
            &params,
            &base
        ));
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::F32(vec![0.; 48])).unwrap();
        assert!(!eligible(&image, &params, &base));
    }
}
