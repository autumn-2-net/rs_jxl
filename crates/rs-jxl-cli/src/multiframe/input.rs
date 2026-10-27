//! Still-image loading, compatibility checks and common animation metadata.

use rs_jxl::Image;
use rs_jxl::metadata::Animation;

use super::EncodeFramesArgs;
use crate::{Result, input, io};

pub(super) fn compatible(first: &Image, current: &Image) -> Result<()> {
    if first.width() != current.width() || first.height() != current.height() {
        return Err("all frames must have identical dimensions; no resizing is performed".into());
    }
    if first.layout() != current.layout()
        || std::mem::discriminant(first.pixels()) != std::mem::discriminant(current.pixels())
        || first.extra_planes().len() != current.extra_planes().len()
        || first
            .extra_planes()
            .iter()
            .zip(current.extra_planes())
            .any(|(a, b)| std::mem::discriminant(a) != std::mem::discriminant(b))
    {
        return Err("all frames must have identical sample storage and alpha layout; no conversion is performed".into());
    }
    if first.metadata() != current.metadata() {
        return Err("all frames must have identical metadata, including color and declared sample/alpha types; --strip-metadata does not remove supported color metadata or convert samples".into());
    }
    Ok(())
}

pub(super) fn with_animation(image: Image, animation: Animation) -> Result<Image> {
    if image.metadata().animation.is_some() {
        return Err(
            "encode-frames requires still inputs; existing animation is not discarded".into(),
        );
    }
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(animation);
    let extras = image.extra_planes().to_vec();
    Ok(image.with_metadata(metadata, extras)?)
}

pub(super) fn retained_bytes(image: &Image) -> Result<usize> {
    let icc_bytes = image
        .metadata()
        .color_encoding
        .icc_profile()
        .map_or(0, <[u8]>::len);
    std::iter::once(image.pixels())
        .chain(image.extra_planes())
        .try_fold(icc_bytes, |total, samples| {
            samples
                .len()
                .checked_mul(usize::from(samples.bits() / 8))
                .and_then(|n| total.checked_add(n))
                .ok_or_else(|| "retained sequence size overflow".into())
        })
}

pub(super) fn add_retained_bytes(total: usize, additional: usize) -> Result<usize> {
    total
        .checked_add(additional)
        .ok_or_else(|| "retained sequence size overflow".into())
}

pub(super) fn load_images(args: &EncodeFramesArgs) -> Result<Vec<Image>> {
    let mut images = Vec::new();
    images.try_reserve_exact(args.inputs.len())?;
    let mut retained = 0;
    let mut warned_jpeg = false;
    for (index, path) in args.inputs.iter().enumerate() {
        let read_frame = || -> Result<input::Decoded> {
            let bytes = io::read_input(path)?;
            input::decode(&bytes, args.strip_metadata)
        };
        let decoded = read_frame()
            .map_err(|error| format!("frame {} '{}': {error}", index + 1, path.display()))?;
        if decoded.jpeg && !warned_jpeg {
            eprintln!(
                "warning: JPEG frames are PIXEL LOSSLESS relative to zune-jpeg's decoded samples, NOT byte-reversible; original JPEG bytes cannot be reconstructed."
            );
            warned_jpeg = true;
        }
        // Compare original input metadata before attaching our common timebase.
        if let Some(first) = images.first() {
            compatible(first, &decoded.image)
                .map_err(|error| format!("frame {} '{}': {error}", index + 1, path.display()))?;
        }
        retained = add_retained_bytes(retained, retained_bytes(&decoded.image)?)?;
        images.push(decoded.image);
    }
    images
        .into_iter()
        .map(|image| with_animation(image, args.animation()))
        .collect()
}
