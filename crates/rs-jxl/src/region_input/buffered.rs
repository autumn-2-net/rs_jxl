//! Explicit source EncodeFrameOneShot fallback, limited to the current frame.

use super::{ImageDescriptor, RegionSource, load};
use crate::modular::{group::Channel, layout::Rect, sample_convert};
use crate::{Error, Image, PixelData, Result};

/// Covers the declared-format i32 planes and the native sample reconstruction,
/// each at four bytes/sample. Codec scratch and compressed output are separate.
pub(crate) fn buffered_sample_bytes(descriptor: &ImageDescriptor) -> Result<usize> {
    let float = descriptor.metadata().bit_depth.is_float();
    for channel in 0..descriptor.layout().channels() {
        if descriptor.channel_depth(channel)?.is_float() != float {
            return Err(Error::InvalidInput(
                "interleaved region channels require a common storage type".into(),
            ));
        }
    }
    (0..descriptor.channel_count()).try_fold(0usize, |total, channel| {
        let (width, height) = descriptor.channel_dimensions(channel)?;
        width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| total.checked_add(n))
            .ok_or(Error::Limit("buffered frame sample bytes"))
    })
}

pub(crate) fn read_frame<S: RegionSource + ?Sized>(
    descriptor: &ImageDescriptor,
    source: &mut S,
    maximum: usize,
) -> Result<Image> {
    if buffered_sample_bytes(descriptor)? > maximum {
        return Err(Error::Limit("buffered frame sample bytes"));
    }
    if source.descriptor() != descriptor {
        return Err(Error::InvalidInput("region descriptor changed".into()));
    }
    let channels = load::read_area(
        descriptor,
        source,
        Rect {
            x: 0,
            y: 0,
            width: descriptor.width() as usize,
            height: descriptor.height() as usize,
        },
    )?;
    let interleaved = descriptor.layout().channels();
    let pixels = pixels(descriptor, &channels[..interleaved], 0)?;
    let mut extras = Vec::new();
    extras
        .try_reserve_exact(channels.len() - interleaved)
        .map_err(|_| Error::Limit("buffered extra planes"))?;
    for (channel, plane) in channels.iter().enumerate().skip(interleaved) {
        extras.push(pixels_for_plane(descriptor, plane, channel)?);
    }
    Image::new(
        descriptor.width(),
        descriptor.height(),
        descriptor.layout(),
        pixels,
    )?
    .with_metadata(descriptor.metadata().clone(), extras)
}

fn pixels_for_plane(
    descriptor: &ImageDescriptor,
    channel: &Channel,
    index: usize,
) -> Result<PixelData> {
    pixels(descriptor, std::slice::from_ref(channel), index)
}

fn pixels(descriptor: &ImageDescriptor, channels: &[Channel], first: usize) -> Result<PixelData> {
    let count = channels[0]
        .samples
        .len()
        .checked_mul(channels.len())
        .ok_or(Error::Limit("buffered sample count"))?;
    if descriptor.channel_depth(first)?.is_float() {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("buffered float samples"))?;
        values.resize(count, 0.0);
        for (component, channel) in channels.iter().enumerate() {
            let depth = descriptor.channel_depth(first + component)?;
            let format = sample_convert::SampleFormat::Float {
                bits: depth.bits_per_sample,
                exponent_bits: depth.exponent_bits_per_sample,
            };
            for (i, &word) in channel.samples.iter().enumerate() {
                let output = i * channels.len() + component;
                sample_convert::int_to_float(
                    &[word],
                    &mut values[output..output + 1],
                    format,
                    1.0,
                )?;
            }
        }
        Ok(PixelData::F32(values))
    } else {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("buffered integer samples"))?;
        values.resize(count, 0);
        for (component, channel) in channels.iter().enumerate() {
            for (i, &value) in channel.samples.iter().enumerate() {
                values[i * channels.len() + component] = value as u32;
            }
        }
        Ok(PixelData::U32(values))
    }
}
