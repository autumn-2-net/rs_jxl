use super::{ImageDescriptor, RegionPlane, RegionRect, RegionRequest, RegionSource};
use crate::modular::{group::Channel, layout};
use crate::{Error, Result};

const INPUT_TILE: usize = 2048;

/// Allocate only this area's writable Modular planes. Color is deinterleaved
/// directly into them; no intermediate PixelData/Image/float staging buffer.
pub(super) fn read_area<S: RegionSource + ?Sized>(
    descriptor: &ImageDescriptor,
    source: &mut S,
    area: layout::Rect,
) -> Result<Vec<Channel>> {
    RegionRect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: area.height,
    }
    .validate_within(descriptor.width() as usize, descriptor.height() as usize)?;
    let mut channels = Vec::new();
    channels
        .try_reserve_exact(descriptor.channel_count())
        .map_err(|_| Error::Limit("region channels"))?;
    for channel in 0..descriptor.channel_count() {
        let shift = descriptor.channel_shift(channel)?;
        let scale = 1usize << shift;
        let width = (area.x + area.width).div_ceil(scale) - (area.x >> shift);
        let height = (area.y + area.height).div_ceil(scale) - (area.y >> shift);
        let count = width
            .checked_mul(height)
            .ok_or(Error::Limit("region channel samples"))?;
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("region channel allocation"))?;
        samples.resize(count, 0);
        channels.push(Channel {
            width,
            height,
            hshift: i32::from(shift),
            vshift: i32::from(shift),
            samples,
        });
    }
    let first_extra = usize::from(descriptor.layout().has_alpha());
    let planes = std::iter::once(RegionPlane::Color)
        .chain((first_extra..descriptor.metadata().extra_channels.len()).map(RegionPlane::Extra));
    for plane in planes {
        let (_, _, first_channel, components) = descriptor.plane_layout(plane)?;
        let shift = descriptor.channel_shift(first_channel)?;
        let origin = (area.x >> shift, area.y >> shift);
        let width = channels[first_channel].width;
        let height = channels[first_channel].height;
        for y in (0..height).step_by(INPUT_TILE) {
            for x in (0..width).step_by(INPUT_TILE) {
                let rect = RegionRect {
                    x: origin.0 + x,
                    y: origin.1 + y,
                    width: INPUT_TILE.min(width - x),
                    height: INPUT_TILE.min(height - y),
                };
                let request = RegionRequest { plane, rect };
                descriptor.validate_request(request)?;
                let lease = source.acquire(request)?;
                let view = lease.view();
                let row_samples = rect
                    .width
                    .checked_mul(components)
                    .ok_or(Error::Limit("region row samples"))?;
                let needed = (rect.height - 1)
                    .checked_mul(view.row_stride)
                    .and_then(|n| n.checked_add(row_samples))
                    .ok_or(Error::Limit("region storage extent"))?;
                if view.row_stride < row_samples || view.samples.len() < needed {
                    return Err(Error::InvalidInput(
                        "region row stride or storage is too short".into(),
                    ));
                }
                for (component, destination) in channels[first_channel..first_channel + components]
                    .iter_mut()
                    .enumerate()
                {
                    let depth = descriptor.channel_depth(first_channel + component)?;
                    view.samples.validate_depth(depth)?;
                    for row in 0..rect.height {
                        let start = (y + row) * destination.width + x;
                        let target = &mut destination.samples[start..start + rect.width];
                        for (column, value) in target.iter_mut().enumerate() {
                            *value = view.samples.encoded(
                                row * view.row_stride + column * components + component,
                                depth,
                            )?;
                        }
                    }
                }
                // Release before the next callback, before parallel work, and
                // before the first output byte from this area can be emitted.
                drop(lease);
            }
        }
    }
    Ok(channels)
}
