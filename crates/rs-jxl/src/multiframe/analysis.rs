//! Exact difference statistics and geometry-only compression opportunity analysis.
//! No entropy coding, image crops, or guessed compressed sizes are produced.
use super::{
    AutoOptions, Rect, Reference, Strategy, auto, compatible, diff::DiffMap, select_reference,
};
use crate::{Error, Image, Result};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct AnalysisOptions {
    pub reference: Reference,
    pub allow_mask: bool,
    pub allow_multi_rect: bool,
    pub adaptive_rectangles: bool,
    pub tile_size: u32,
    pub max_rectangles: usize,
    /// Rectangle merge cost estimate only. Actual automatic encoding measures
    /// this value from its first full-frame payload; analysis does not encode.
    pub rectangle_bytes_per_pixel: f64,
    pub threads: usize,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            reference: Reference::SmallestCrop,
            allow_mask: true,
            allow_multi_rect: true,
            adaptive_rectangles: true,
            tile_size: 32,
            max_rectangles: 32,
            rectangle_bytes_per_pixel: 1.0,
            threads: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiffStatistics {
    pub bounds: Option<Rect>,
    /// Union of affected canvas pixels across all color and extra channels.
    pub changed_pixels: u64,
    pub changed_ratio: f64,
    pub bbox_pixels: u64,
    pub bbox_ratio: f64,
    /// Zero for identical frames. Small values suggest dispersed changes.
    pub bbox_fill_ratio: f64,
    /// Both images satisfy the encoder's exact binary-alpha mask requirements.
    pub mask_eligible: bool,
}

#[derive(Debug)]
pub struct CropAnalysis {
    pub reference_index: Option<usize>,
    pub source: u8,
    pub region: Rect,
    pub changed: Option<Rect>,
    /// Submitted rectangle area, including the 1x1 identical-frame placeholder.
    pub layer_pixels: u64,
    pub layer_area_ratio: f64,
}

#[derive(Debug)]
pub struct GeometryCandidate {
    pub strategy: Strategy,
    pub reference_index: Option<usize>,
    pub source: u8,
    pub rectangles: Vec<Rect>,
    /// Mask candidates still submit their entire bounding rectangle.
    pub layer_pixels: u64,
    pub layer_area_ratio: f64,
    pub layer_count: usize,
}

#[derive(Debug)]
pub struct FrameAnalysis {
    pub index: usize,
    /// Compatibility fields: the first frame uses the full canvas here.
    pub previous: Option<Rect>,
    pub first: Option<Rect>,
    /// The first frame has no reference comparison, so these are None.
    pub previous_stats: Option<DiffStatistics>,
    pub first_stats: Option<DiffStatistics>,
    pub plan: CropAnalysis,
    pub candidates: Vec<GeometryCandidate>,
}

/// Geometric reuse labels, not predictions of encoded bytes or encoding speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Suitability {
    SingleFrame,
    Strong,
    Moderate,
    Limited,
    FullFramesRequired,
}

#[derive(Debug)]
pub struct ReferenceSummary {
    pub reference: Reference,
    /// Includes the initial full frame and one rectangle per subsequent state.
    pub layer_pixels: u64,
    pub layer_area_ratio: f64,
    /// Excludes the initial frame. Full mode treats all subsequent pixels as new.
    pub average_changed_ratio: f64,
    pub crop_frames: usize,
    pub identical_frames: usize,
    pub suitability: Suitability,
}

#[derive(Debug)]
pub struct AnalysisReport {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub frame_count: usize,
    pub reference: Reference,
    /// False when subsampled extra channels require complete-frame replacement.
    pub crop_supported: bool,
    pub rectangle_bytes_per_pixel: f64,
    pub frames: Vec<FrameAnalysis>,
    pub reference_modes: Vec<ReferenceSummary>,
    /// Minimum total single-bbox area only, not an encoding recommendation.
    pub best_geometric_reference: Reference,
    /// Excludes input decoding, context creation, and transport serialization.
    pub analysis_elapsed: Duration,
}

/// Analyze matching full-image states without encoding or modifying any image.
/// Comparisons are exact, including hidden RGB, float words and extra channels.
/// `SmallestCrop` plans by bbox area; `encode_auto` selects by actual coded bytes.
pub fn analyze(images: &[Image], options: &AnalysisOptions) -> Result<AnalysisReport> {
    validate(images, options)?;
    if options.threads == 1 || crate::execution::is_serial() {
        return analyze_in_pool(images, options, None);
    }
    crate::codec::Encoder::new(options.threads)?.analyze_multiframe(images, options)
}

pub(crate) fn analyze_in_pool(
    images: &[Image],
    options: &AnalysisOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<AnalysisReport> {
    validate(images, options)?;
    let work = || analyze_inner(images, options);
    match pool {
        Some(pool) => crate::execution::with_pool(pool, work),
        None => crate::execution::serial(work),
    }
}

fn validate(images: &[Image], options: &AnalysisOptions) -> Result<()> {
    let Some(first) = images.first() else {
        return Err(Error::InvalidInput("nonempty images are required".into()));
    };
    if !(1..=256).contains(&options.threads)
        || options.tile_size == 0
        || options.max_rectangles == 0
        || !options.rectangle_bytes_per_pixel.is_finite()
        || options.rectangle_bytes_per_pixel < 0.0
    {
        return Err(Error::InvalidInput(
            "threads must be 1..=256, tile_size and max_rectangles positive, and rectangle_bytes_per_pixel finite and nonnegative".into(),
        ));
    }
    for image in images {
        compatible(first, image)?;
    }
    Ok(())
}

fn area(rect: Rect) -> u64 {
    u64::from(rect.width) * u64::from(rect.height)
}

fn statistics(map: &DiffMap, pixels: u64) -> DiffStatistics {
    let bbox_pixels = map.bounds().map_or(0, area);
    DiffStatistics {
        bounds: map.bounds(),
        changed_pixels: map.changed_pixels(),
        changed_ratio: map.changed_pixels() as f64 / pixels as f64,
        bbox_pixels,
        bbox_ratio: bbox_pixels as f64 / pixels as f64,
        bbox_fill_ratio: if bbox_pixels == 0 {
            0.0
        } else {
            map.changed_pixels() as f64 / bbox_pixels as f64
        },
        mask_eligible: map.mask_eligible(),
    }
}

fn crop_plan(
    index: usize,
    reference: Reference,
    full: Rect,
    crop_supported: bool,
    previous: Option<Rect>,
    first: Option<Rect>,
) -> CropAnalysis {
    if index == 0 || reference == Reference::Full || !crop_supported {
        return CropAnalysis {
            reference_index: None,
            source: 0,
            region: full,
            changed: Some(full),
            layer_pixels: area(full),
            layer_area_ratio: 1.0,
        };
    }
    let (changed, source, reference_index) = select_reference(index, reference, previous, first);
    let region = changed.unwrap_or(Rect {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    });
    CropAnalysis {
        reference_index: Some(reference_index),
        source,
        region,
        changed,
        layer_pixels: area(region),
        layer_area_ratio: area(region) as f64 / area(full) as f64,
    }
}

fn analyze_inner(images: &[Image], options: &AnalysisOptions) -> Result<AnalysisReport> {
    let started = Instant::now();
    let canvas = &images[0];
    let full = Rect {
        x: 0,
        y: 0,
        width: canvas.width(),
        height: canvas.height(),
    };
    let pixels = area(full);
    let full_pixels = pixels
        .checked_mul(images.len() as u64)
        .ok_or(Error::Limit("analysis total pixel area"))?;
    let crop_supported = !canvas
        .metadata()
        .extra_channels
        .iter()
        .any(|e| e.dim_shift != 0);
    let auto = AutoOptions {
        reference: options.reference,
        allow_mask: options.allow_mask,
        allow_multi_rect: options.allow_multi_rect,
        adaptive_rectangles: options.adaptive_rectangles,
        tile_size: options.tile_size,
        max_rectangles: options.max_rectangles,
        ..Default::default()
    };
    let mut summaries: Vec<_> = [
        Reference::Full,
        Reference::Previous,
        Reference::First,
        Reference::SmallestCrop,
    ]
    .into_iter()
    .map(|reference| ReferenceSummary {
        reference,
        layer_pixels: 0,
        layer_area_ratio: 0.0,
        average_changed_ratio: 0.0,
        crop_frames: 0,
        identical_frames: 0,
        suitability: Suitability::SingleFrame,
    })
    .collect();
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(images.len())
        .map_err(|_| Error::Limit("analysis frame reports"))?;
    for (index, image) in images.iter().enumerate() {
        // Only two reference maps live at once; frame one shares its map.
        let previous_map = if index == 0 {
            None
        } else {
            Some(Arc::new(DiffMap::between(&images[index - 1], image)?))
        };
        let first_map = if index <= 1 {
            previous_map.clone()
        } else {
            Some(Arc::new(DiffMap::between(canvas, image)?))
        };
        let previous_stats = previous_map.as_ref().map(|m| statistics(m, pixels));
        let first_stats = first_map.as_ref().map(|m| statistics(m, pixels));
        let previous = previous_stats.as_ref().map_or(Some(full), |s| s.bounds);
        let first = first_stats.as_ref().map_or(Some(full), |s| s.bounds);
        for summary in &mut summaries {
            let plan = crop_plan(
                index,
                summary.reference,
                full,
                crop_supported,
                previous,
                first,
            );
            summary.layer_pixels = summary
                .layer_pixels
                .checked_add(plan.layer_pixels)
                .ok_or(Error::Limit("analysis layer pixel area"))?;
            if index != 0 {
                let stats = match plan.reference_index {
                    Some(0) => first_stats.as_ref(),
                    Some(_) => previous_stats.as_ref(),
                    None => None,
                };
                summary.average_changed_ratio += stats.map_or(1.0, |s| s.changed_ratio);
                summary.identical_frames +=
                    usize::from(stats.is_some_and(|s| s.changed_pixels == 0));
                summary.crop_frames += usize::from(plan.region != full);
            }
        }
        let maps: Vec<_> = auto::reference_indices(index, options.reference)
            .into_iter()
            .map(|reference| {
                let map = if reference == 0 {
                    &first_map
                } else {
                    &previous_map
                };
                (
                    reference,
                    Arc::clone(map.as_ref().expect("reference map for noninitial frame")),
                )
            })
            .collect();
        let candidates = auto::candidates_from_maps(
            image,
            index,
            &auto,
            options.rectangle_bytes_per_pixel,
            &maps,
        )?
        .into_iter()
        .map(|candidate| {
            let layer_pixels = candidate.rects.iter().try_fold(0u64, |sum, &rect| {
                sum.checked_add(area(rect))
                    .ok_or(Error::Limit("analysis candidate pixel area"))
            })?;
            Ok(GeometryCandidate {
                strategy: candidate.strategy,
                reference_index: match candidate.source {
                    1 => Some(0),
                    2 => Some(index - 1),
                    _ => None,
                },
                source: candidate.source,
                layer_count: candidate.rects.len(),
                rectangles: candidate.rects,
                layer_pixels,
                layer_area_ratio: layer_pixels as f64 / pixels as f64,
            })
        })
        .collect::<Result<Vec<_>>>()?;
        frames.push(FrameAnalysis {
            index,
            previous,
            first,
            previous_stats,
            first_stats,
            plan: crop_plan(
                index,
                options.reference,
                full,
                crop_supported,
                previous,
                first,
            ),
            candidates,
        });
    }
    for summary in &mut summaries {
        summary.layer_area_ratio = summary.layer_pixels as f64 / full_pixels as f64;
        if images.len() > 1 {
            summary.average_changed_ratio /= (images.len() - 1) as f64;
            let subsequent_ratio =
                (summary.layer_pixels - pixels) as f64 / (full_pixels - pixels) as f64;
            summary.suitability = if !crop_supported {
                Suitability::FullFramesRequired
            } else if subsequent_ratio < 0.3 {
                Suitability::Strong
            } else if subsequent_ratio < 0.7 {
                Suitability::Moderate
            } else {
                Suitability::Limited
            };
        }
    }
    let best_geometric_reference = summaries
        .iter()
        .min_by_key(|s| s.layer_pixels)
        .expect("four reference modes")
        .reference;
    Ok(AnalysisReport {
        width: full.width,
        height: full.height,
        channels: canvas.channel_count(),
        frame_count: images.len(),
        reference: options.reference,
        crop_supported,
        rectangle_bytes_per_pixel: options.rectangle_bytes_per_pixel,
        frames,
        reference_modes: summaries,
        best_geometric_reference,
        analysis_elapsed: started.elapsed(),
    })
}
