//! Exact full-frame difference planning. No alpha compositing or numeric residuals.
//! All color and extra samples participate, including RGB under zero alpha.
use crate::{
    Error, Image, PixelData, Result,
    frame::{BlendInfo, Frame, FrameOptions},
};
use std::borrow::Cow;

mod analysis;
mod auto;
mod diff;
mod lookahead;
mod partition;
mod payload;
mod references;
mod regions;
mod scales;
pub(crate) use analysis::analyze_in_pool;
pub use analysis::{
    AnalysisOptions, AnalysisReport, CropAnalysis, DiffStatistics, FrameAnalysis,
    GeometryCandidate, GeometryDiagnostics, GeometrySummary, HistoryAnalysis, RectangleWorkload,
    ReferenceSummary, Suitability, analyze,
};
pub(crate) use auto::encode_auto_in_pool;
pub use auto::{
    AutoCandidateReport, AutoDiffReport, AutoEncodeResult, AutoFrameReport, AutoOptions,
    AutoPrunedReport, Strategy, encode_auto,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reference {
    Full,
    #[default]
    Previous,
    First,
    SmallestCrop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct PlannedFrame<'a> {
    pub image: Cow<'a, Image>,
    pub options: FrameOptions,
    pub changed: Option<Rect>,
}
impl PlannedFrame<'_> {
    pub fn as_frame(&self) -> Frame<'_> {
        Frame {
            image: &self.image,
            options: self.options.clone(),
        }
    }
}

fn compatible(a: &Image, b: &Image) -> Result<()> {
    if a.width() != b.width()
        || a.height() != b.height()
        || a.layout() != b.layout()
        || a.metadata() != b.metadata()
    {
        return Err(Error::InvalidInput(
            "difference frames must have identical geometry and sample metadata".into(),
        ));
    }
    Ok(())
}

pub fn difference_bounds(previous: &Image, current: &Image) -> Result<Option<Rect>> {
    compatible(previous, current)?;
    let (w, h) = (current.width() as usize, current.height() as usize);
    let channels = current.layout().channels();
    let (mut left, mut top, mut right, mut bottom) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            let pixel = y * w + x;
            let changed = (0..channels)
                .any(|c| previous.modular_sample(c, pixel) != current.modular_sample(c, pixel));
            if changed {
                left = left.min(x);
                top = top.min(y);
                right = right.max(x + 1);
                bottom = bottom.max(y + 1);
            }
        }
    }
    for channel in channels..current.channel_count() {
        let (pw, ph) = current.channel_dimensions(channel);
        let scale = 1usize << current.channel_shift(channel);
        for y in 0..ph {
            for x in 0..pw {
                if previous.modular_sample(channel, y * pw + x)
                    != current.modular_sample(channel, y * pw + x)
                {
                    left = left.min(x * scale);
                    top = top.min(y * scale);
                    right = right.max(((x + 1) * scale).min(w));
                    bottom = bottom.max(((y + 1) * scale).min(h));
                }
            }
        }
    }
    if right <= left {
        return Ok(None);
    }
    let align = 1usize
        << current
            .metadata()
            .extra_channels
            .iter()
            .map(|e| e.dim_shift)
            .max()
            .unwrap_or(0);
    left = left / align * align;
    top = top / align * align;
    right = right.div_ceil(align).saturating_mul(align).min(w);
    bottom = bottom.div_ceil(align).saturating_mul(align).min(h);
    Ok(Some(Rect {
        x: left as u32,
        y: top as u32,
        width: (right - left) as u32,
        height: (bottom - top) as u32,
    }))
}

pub fn crop(image: &Image, rect: Rect) -> Result<Image> {
    if rect.width == 0
        || rect.height == 0
        || rect
            .x
            .checked_add(rect.width)
            .is_none_or(|x| x > image.width())
        || rect
            .y
            .checked_add(rect.height)
            .is_none_or(|y| y > image.height())
    {
        return Err(Error::InvalidInput("crop is outside image".into()));
    }
    let align = 1u32
        << image
            .metadata()
            .extra_channels
            .iter()
            .map(|e| e.dim_shift)
            .max()
            .unwrap_or(0);
    if !rect.x.is_multiple_of(align) || !rect.y.is_multiple_of(align) {
        return Err(Error::InvalidInput(
            "crop origin must align with subsampled extra-channel grid".into(),
        ));
    }
    let slice = |pixels: &PixelData, channels: usize, shift: u8| -> Result<PixelData> {
        let scale = 1usize << shift;
        let width = (rect.width as usize).div_ceil(scale);
        let height = (rect.height as usize).div_ceil(scale);
        let stride = (image.width() as usize).div_ceil(scale);
        let x = rect.x as usize / scale;
        let top = rect.y as usize / scale;
        let count = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(channels))
            .ok_or(Error::Limit("crop samples"))?;
        macro_rules! rows {
            ($values:expr,$variant:ident) => {{
                let mut result = Vec::new();
                result
                    .try_reserve_exact(count)
                    .map_err(|_| Error::Limit("crop allocation"))?;
                for y in top..top + height {
                    let start = (y * stride + x) * channels;
                    result.extend_from_slice(&$values[start..start + width * channels]);
                }
                PixelData::$variant(result)
            }};
        }
        Ok(match pixels {
            PixelData::U8(v) => rows!(v, U8),
            PixelData::U16(v) => rows!(v, U16),
            PixelData::U32(v) => rows!(v, U32),
            PixelData::F16(v) => rows!(v, F16),
            PixelData::F32(v) => rows!(v, F32),
        })
    };
    let pixels = slice(image.pixels(), image.layout().channels(), 0)?;
    let extras = image
        .extra_planes()
        .iter()
        .enumerate()
        .map(|(c, p)| slice(p, 1, image.channel_shift(image.layout().channels() + c)))
        .collect::<Result<Vec<_>>>()?;
    Image::new(rect.width, rect.height, image.layout(), pixels)?
        .with_metadata(image.metadata().clone(), extras)
}

fn select_reference(
    index: usize,
    reference: Reference,
    previous: Option<Rect>,
    first: Option<Rect>,
) -> (Option<Rect>, u8, usize) {
    let area = |rect: Option<Rect>| rect.map_or(0, |r| u64::from(r.width) * u64::from(r.height));
    if reference == Reference::First
        || (reference == Reference::SmallestCrop && area(first) < area(previous))
    {
        (first, 1, 0)
    } else {
        (previous, if index == 1 { 1 } else { 2 }, index - 1)
    }
}

/// Bbox is a geometric heuristic, not an encoded-byte estimator. SmallestCrop
/// compares previous/first areas; it never claims to choose the smallest JXL.
/// The immutable first reference uses slot 1; subsequent references use slot 2.
/// Subsampled extra planes retain full frames: decoder upsampling precedes
/// compositing, so even aligned crops can change the interpolation boundary.
pub fn plan<'a>(
    images: &'a [Image],
    durations: &[u32],
    reference: Reference,
) -> Result<Vec<PlannedFrame<'a>>> {
    if images.is_empty() || images.len() != durations.len() {
        return Err(Error::InvalidInput(
            "nonempty images and one duration per frame are required".into(),
        ));
    }
    for image in images {
        compatible(&images[0], image)?;
    }
    if images[0].metadata().animation.is_none() && durations.iter().any(|&d| d != 0) {
        return Err(Error::InvalidInput(
            "nonzero durations require animation metadata".into(),
        ));
    }
    let reference = if images[0]
        .metadata()
        .extra_channels
        .iter()
        .any(|e| e.dim_shift != 0)
    {
        Reference::Full
    } else {
        reference
    };
    let mut result = Vec::with_capacity(images.len());
    for (i, image) in images.iter().enumerate() {
        let last = i + 1 == images.len();
        let mut options = FrameOptions {
            duration: durations[i],
            save_as_reference: if last || reference == Reference::Full {
                0
            } else if i == 0 {
                1
            } else {
                2
            },
            ..Default::default()
        };
        if i == 0 || reference == Reference::Full {
            result.push(PlannedFrame {
                image: Cow::Borrowed(image),
                options,
                changed: Some(Rect {
                    x: 0,
                    y: 0,
                    width: image.width(),
                    height: image.height(),
                }),
            });
            continue;
        }
        let previous = difference_bounds(&images[i - 1], image)?;
        let first = match reference {
            Reference::First | Reference::SmallestCrop => difference_bounds(&images[0], image)?,
            _ => None,
        };
        let (changed, source, _) = select_reference(i, reference, previous, first);
        // JXL layers cannot have zero dimensions. An identical frame repeats one
        // exact sample while its duration/reference semantics remain unchanged.
        let rect = changed.unwrap_or(Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        });
        options.origin = (rect.x as i32, rect.y as i32);
        options.blend = BlendInfo {
            source,
            ..Default::default()
        };
        let pixels = if rect.width == image.width() && rect.height == image.height() {
            Cow::Borrowed(image)
        } else {
            Cow::Owned(crop(image, rect)?)
        };
        result.push(PlannedFrame {
            image: pixels,
            options,
            changed,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelLayout;
    #[test]
    fn hidden_rgb_is_changed_and_identical_is_empty() {
        let a = Image::new(
            2,
            1,
            PixelLayout::Rgba,
            PixelData::U8(vec![3, 5, 7, 0, 11, 13, 17, 128]),
        )
        .unwrap();
        let b = Image::new(
            2,
            1,
            PixelLayout::Rgba,
            PixelData::U8(vec![4, 5, 7, 0, 11, 13, 17, 128]),
        )
        .unwrap();
        assert_eq!(difference_bounds(&a, &a).unwrap(), None);
        assert_eq!(
            difference_bounds(&a, &b).unwrap(),
            Some(Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1
            })
        );
        let images = [a.clone(), b, a];
        let plan = plan(&images, &[0, 0, 0], Reference::SmallestCrop).unwrap();
        assert_eq!(plan[2].changed, None);
        assert_eq!(plan[2].options.blend.source, 1);
        assert_eq!(plan[2].options.duration, 0);
        assert_eq!(plan[2].options.save_as_reference, 0);
    }

    #[test]
    fn subsampled_diff_covers_the_full_sample_cell_and_crops_planar_data() {
        use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
        let image = Image::new(7, 5, PixelLayout::Rgb, PixelData::U8(vec![0; 105])).unwrap();
        let mut metadata = image.metadata().clone();
        let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        info.dim_shift = 1;
        metadata.extra_channels.push(info);
        let a = image
            .clone()
            .with_metadata(metadata.clone(), vec![PixelData::U8(vec![0; 12])])
            .unwrap();
        let mut extra = vec![0; 12];
        extra[6] = 9;
        let b = image
            .with_metadata(metadata, vec![PixelData::U8(extra)])
            .unwrap();
        let rect = difference_bounds(&a, &b).unwrap().unwrap();
        assert_eq!(
            rect,
            Rect {
                x: 4,
                y: 2,
                width: 2,
                height: 2
            }
        );
        let part = crop(&b, rect).unwrap();
        assert_eq!(part.extra_planes(), &[PixelData::U8(vec![9])]);
        assert!(
            crop(
                &b,
                Rect {
                    x: 1,
                    y: 0,
                    width: 2,
                    height: 2
                }
            )
            .is_err()
        );
    }
}
