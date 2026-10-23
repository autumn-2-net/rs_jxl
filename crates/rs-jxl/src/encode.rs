// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Frame/group orchestration. Modular tools and entropy live in their own modules.
use crate::execution;
use crate::{
    Error, Image, Result, codestream,
    modular::encode::{self as modular, Group, ModularOptions},
};
use std::io::Write;

pub(crate) mod budget;
use budget::{SectionBudget, TrialEncode, complete_or_return};

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Owned per-call pool; 1 runs on the caller without any pool or child tasks.
    pub threads: usize,
    pub max_pixels: u64,
    pub max_output_bytes: usize,
    pub modular: ModularOptions,
    /// 0..=3 gives 128, 256, 512 or 1024 pixel Modular groups.
    pub group_size_shift: u8,
    pub shared_tree: bool,
    /// Explicit lossless repeated-component patches; atlas overhead may exceed savings.
    pub patches: Option<crate::tools::patches::PatchOptions>,
    /// Background-relative source Add patches, retaining signed residuals.
    pub source_patches: Option<crate::tools::patches::SourcePatchConfig>,
    /// Resolve source atlas parameters after its dimensions are known.
    pub source_patch_reference: Option<crate::modular::source_options::SourcePatchReferencePolicy>,
    /// Whole-frame palette before group splitting; local palettes remain the fallback.
    pub global_palette: bool,
    /// Apply the source whole-frame squeeze plan before DC/AC partitioning.
    pub responsive: bool,
    /// Spatial pass layout; independent of the reversible squeeze transform.
    pub passes: crate::modular::layout::Passes,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            max_pixels: 64_000_000,
            max_output_bytes: 512 << 20,
            modular: ModularOptions::default(),
            group_size_shift: 1,
            shared_tree: false,
            patches: None,
            source_patches: None,
            source_patch_reference: None,
            global_palette: false,
            responsive: false,
            passes: Default::default(),
        }
    }
}

/// Encode original samples as a standard single-frame Modular JXL.
/// Full C++ tool/scheduler coverage is tracked separately; a valid stream alone
/// is not completion of pixel-lossless migration.
pub fn encode(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>> {
    encode_frames(image, &[crate::frame::Frame::new(image)], options)
}

/// Null-runner candidate encoding on the calling thread, even inside a Rayon
/// worker. Keeps all tool/search/resource settings; threads is still validated
/// but does not select execution here. Nested core work inherits this scope.
pub(crate) fn encode_serial(image: &Image, options: &EncodeOptions) -> Result<Vec<u8>> {
    execution::serial(|| {
        let frames = [crate::frame::Frame::new(image)];
        validate_frames(image, &frames, options)?;
        encode_validated_frames(image, &frames, options)
    })
}

/// Scheduler trials share one bounded pool instead of creating one per trial.
pub(crate) fn encode_with_pool(
    image: &Image,
    options: &EncodeOptions,
    pool: &rayon::ThreadPool,
) -> Result<Vec<u8>> {
    let frames = [crate::frame::Frame::new(image)];
    validate_frames(image, &frames, options)?;
    execution::with_pool(pool, || encode_validated_frames(image, &frames, options))
}

pub(crate) fn encode_trial_serial(
    image: &Image,
    options: &EncodeOptions,
    beat_bytes: Option<usize>,
) -> Result<TrialEncode<Vec<u8>>> {
    execution::serial(|| {
        let frames = [crate::frame::Frame::new(image)];
        validate_frames(image, &frames, options)?;
        let budget = SectionBudget::new(beat_bytes, options.max_output_bytes);
        encode_validated_frames_trial(image, &frames, options, budget.as_ref())
    })
}

pub(crate) fn encode_trial_with_pool(
    image: &Image,
    options: &EncodeOptions,
    beat_bytes: Option<usize>,
    pool: &rayon::ThreadPool,
) -> Result<TrialEncode<Vec<u8>>> {
    let frames = [crate::frame::Frame::new(image)];
    validate_frames(image, &frames, options)?;
    let budget = SectionBudget::new(beat_bytes, options.max_output_bytes);
    execution::with_pool(pool, || {
        encode_validated_frames_trial(image, &frames, options, budget.as_ref())
    })
}

fn validate(image: &Image, options: &EncodeOptions) -> Result<()> {
    validate_geometry(image.into(), options)
}

pub(crate) fn validate_geometry(
    image: crate::frame::Geometry<'_>,
    options: &EncodeOptions,
) -> Result<()> {
    codestream::validate_level10_geometry(image)?;
    if image.metadata().extra_channels.len() > 256 {
        return Err(Error::Limit("Level 10 extra-channel count"));
    }
    if image.metadata().modular_16_bit_buffer_sufficient
        && (image.metadata().bit_depth.is_float()
            || image.bits() > 12
            || image
                .metadata()
                .extra_channels
                .iter()
                .any(|e| e.bit_depth.is_float() || e.bit_depth.bits_per_sample > 12))
    {
        return Err(Error::InvalidInput(
            "16-bit Modular buffer promise requires at most 12-bit integer input channels".into(),
        ));
    }
    if !(1..=256).contains(&options.threads) {
        return Err(Error::InvalidInput("threads must be 1..=256".into()));
    }
    if u64::from(image.width()) * u64::from(image.height()) > options.max_pixels {
        return Err(Error::Limit("pixel count"));
    }
    if options.max_output_bytes == 0 {
        return Err(Error::Limit("encoded output bytes"));
    }
    options.modular.validate()?;
    if let Some(patches) = &options.patches {
        patches.validate()?;
    }
    if let Some(patches) = &options.source_patches {
        patches.validate()?;
    }
    if options.patches.is_some() && options.source_patches.is_some() {
        return Err(Error::InvalidInput(
            "select either Replace patches or source Add patches".into(),
        ));
    }
    options.passes.validate()?;
    if options.group_size_shift > 3 {
        return Err(Error::InvalidInput("group_size_shift must be 0..=3".into()));
    }
    Ok(())
}

/// Encode layers or animation frames. `canvas` provides dimensions/metadata;
/// its samples are not encoded unless it is included in `frames`. References
/// and zero-duration layers are explicit. No difference inference occurs here.
pub fn encode_frames(
    canvas: &Image,
    frames: &[crate::frame::Frame<'_>],
    options: &EncodeOptions,
) -> Result<Vec<u8>> {
    validate_frames(canvas, frames, options)?;
    if options.threads == 1 || execution::is_serial() {
        return execution::serial(|| encode_validated_frames(canvas, frames, options));
    }
    // MA property sweeps may run even when the image has only one group.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()
        .map_err(|e| Error::Encoding(e.to_string()))?;
    execution::with_pool(&pool, || encode_validated_frames(canvas, frames, options))
}

pub(crate) fn validate_frames(
    canvas: &Image,
    frames: &[crate::frame::Frame<'_>],
    options: &EncodeOptions,
) -> Result<()> {
    validate(canvas, options)?;
    if frames.is_empty() {
        return Err(Error::InvalidInput(
            "frame sequence must not be empty".into(),
        ));
    }
    // None is unsaved; Some(true) is a pre-transform patch-only reference.
    let mut available = [None; 4];
    for (i, frame) in frames.iter().enumerate() {
        validate(frame.image, options)?;
        frame
            .options
            .validate(canvas, frame.image, i + 1 == frames.len())?;
        let mut metadata = frame.image.metadata().clone();
        metadata.animation = canvas.metadata().animation;
        if metadata != *canvas.metadata() || frame.image.layout() != canvas.layout() {
            return Err(Error::InvalidInput(
                "all frames must use the canvas sample/color/extra metadata".into(),
            ));
        }
        available = validate_frame_references(
            canvas.into(),
            frame.image.into(),
            &frame.options,
            options,
            i + 1 == frames.len(),
            available,
        )?;
    }
    Ok(())
}

/// Validate against committed slots; callers commit the returned state only
/// after the frame has been completely encoded and its output finalized.
pub(crate) fn validate_frame_references(
    canvas: crate::frame::Geometry<'_>,
    image: crate::frame::Geometry<'_>,
    settings: &crate::frame::FrameOptions,
    options: &EncodeOptions,
    last: bool,
    mut available: [Option<bool>; 4],
) -> Result<[Option<bool>; 4]> {
    settings.validate_geometry(canvas, image, last)?;
    let patches = options.patches.is_some() || options.source_patches.is_some();
    if !settings.reference_only {
        for blend in std::iter::once(settings.blend)
            .chain((0..canvas.metadata.extra_channels.len()).map(|e| settings.extra(e)))
        {
            if settings.partial_geometry(canvas, image)
                || blend.mode != crate::frame::BlendMode::Replace
            {
                match available[usize::from(blend.source)] {
                    Some(false) => {}
                    Some(true) => {
                        return Err(Error::InvalidInput(
                            "frame blending cannot use a pre-transform reference".into(),
                        ));
                    }
                    None => {
                        return Err(Error::InvalidInput(
                            "frame uses an unsaved reference".into(),
                        ));
                    }
                }
            }
            if patches && blend.source == 3 {
                return Err(Error::InvalidInput(
                    "patches reserve reference slot 3".into(),
                ));
            }
        }
    }
    if patches && settings.save_as_reference == 3 {
        return Err(Error::InvalidInput(
            "patches reserve reference slot 3".into(),
        ));
    }
    if !last
        && (settings.reference_only || settings.duration == 0 || settings.save_as_reference != 0)
    {
        available[usize::from(settings.save_as_reference)] =
            Some(settings.save_before_color_transform);
    }
    Ok(available)
}

fn encode_validated_frames(
    canvas: &Image,
    frames: &[crate::frame::Frame<'_>],
    options: &EncodeOptions,
) -> Result<Vec<u8>> {
    encode_validated_frames_trial(canvas, frames, options, None)?.into_complete()
}

fn encode_validated_frames_trial(
    canvas: &Image,
    frames: &[crate::frame::Frame<'_>],
    options: &EncodeOptions,
    budget: Option<&SectionBudget>,
) -> Result<TrialEncode<Vec<u8>>> {
    let _encoding = crate::profile::scope("encode.frames");
    let mut stream = codestream::image_header(canvas, options.max_output_bytes)?;
    if let Some(budget) = budget {
        budget.retain_overhead(stream.len())?;
    }
    for (i, frame) in frames.iter().enumerate() {
        let last = i + 1 == frames.len();
        stream.extend(complete_or_return!(encode_logical_frame_trial(
            canvas, frame, options, last, budget,
        )));
        if stream.len() > options.max_output_bytes {
            return Err(Error::Limit("encoded output bytes"));
        }
    }
    Ok(TrialEncode::Complete(wrap_stream(
        canvas,
        stream,
        options.max_output_bytes,
    )?))
}

/// A logical frame may contain a preceding patch-reference frame. It does not
/// include an image header or container, and retains the caller's frame context.
pub(crate) fn encode_logical_frame_trial(
    canvas: &Image,
    frame: &crate::frame::Frame<'_>,
    options: &EncodeOptions,
    last: bool,
    budget: Option<&SectionBudget>,
) -> Result<TrialEncode<Vec<u8>>> {
    encode_logical_frame_impl(
        FrameCanvas::Image(canvas),
        frame,
        options,
        last,
        budget,
        None,
    )
}

/// The canvas contributes metadata and framing only, never samples. Region
/// sequence input may materialize the current frame without a canvas Image.
pub(crate) fn encode_logical_frame_geometry(
    canvas: crate::frame::Geometry<'_>,
    frame: &crate::frame::Frame<'_>,
    options: &EncodeOptions,
    last: bool,
) -> Result<(Vec<u8>, bool)> {
    let mut generated_reference = false;
    let bytes = encode_logical_frame_impl(
        FrameCanvas::Geometry(canvas),
        frame,
        options,
        last,
        None,
        Some(&mut generated_reference),
    )?
    .into_complete()?;
    Ok((bytes, generated_reference))
}

#[derive(Clone, Copy)]
enum FrameCanvas<'a> {
    Image(&'a Image),
    Geometry(crate::frame::Geometry<'a>),
}

impl<'a> FrameCanvas<'a> {
    fn geometry(self) -> crate::frame::Geometry<'a> {
        match self {
            Self::Image(image) => image.into(),
            Self::Geometry(geometry) => geometry,
        }
    }
}

fn encode_logical_frame_impl(
    canvas: FrameCanvas<'_>,
    frame: &crate::frame::Frame<'_>,
    options: &EncodeOptions,
    last: bool,
    budget: Option<&SectionBudget>,
    generated_reference: Option<&mut bool>,
) -> Result<TrialEncode<Vec<u8>>> {
    validate(frame.image, options)?;
    frame
        .options
        .validate_geometry(canvas.geometry(), frame.image.into(), last)?;
    if (options.patches.is_some() || options.source_patches.is_some())
        && (frame.options.save_as_reference == 3
            || std::iter::once(frame.options.blend)
                .chain(
                    (0..canvas.geometry().metadata.extra_channels.len())
                        .map(|e| frame.options.extra(e)),
                )
                .any(|blend| blend.source == 3))
    {
        return Err(Error::InvalidInput(
            "patches reserve reference slot 3".into(),
        ));
    }
    let plan = {
        let _patches = crate::profile::scope("encode.patch_analysis");
        if let Some(config) = &options.source_patches {
            crate::patch_frame::prepare_source(frame.image, config)?
        } else if let Some(patches) = &options.patches {
            crate::patch_frame::prepare(frame.image, patches)?
        } else {
            None
        }
    };
    let mut stream = Vec::new();
    if let Some(generated) = generated_reference {
        *generated = plan.is_some();
    }
    if let Some((reference, residual, dictionary)) = plan {
        let reference_options = crate::frame::FrameOptions {
            reference_only: true,
            save_as_reference: 3,
            save_before_color_transform: true,
            ..Default::default()
        };
        let mut atlas_options = options.clone();
        if options.source_patches.is_some() {
            atlas_options.modular.predictor = crate::modular::predictor::Predictor::Gradient;
            atlas_options.modular.squeeze.clear();
            atlas_options.responsive = false;
            atlas_options.passes = Default::default();
            if let Some(policy) = &options.source_patch_reference {
                let resolved = policy.resolve(&reference)?;
                atlas_options.modular = resolved.modular;
                atlas_options.group_size_shift = resolved.group_size_shift;
                atlas_options.shared_tree = true;
                atlas_options.global_palette = true;
            }
        }
        atlas_options.patches = None;
        atlas_options.source_patches = None;
        atlas_options.source_patch_reference = None;
        stream.extend(complete_or_return!(encode_frame_trial(
            canvas,
            &reference,
            &atlas_options,
            &reference_options,
            false,
            None,
            budget,
        )));
        stream.extend(complete_or_return!(encode_frame_trial(
            canvas,
            &residual,
            options,
            &frame.options,
            last,
            Some(&dictionary),
            budget,
        )));
    } else {
        stream.extend(complete_or_return!(encode_frame_trial(
            canvas,
            frame.image,
            options,
            &frame.options,
            last,
            None,
            budget,
        )));
    }
    if stream.len() > options.max_output_bytes {
        return Err(Error::Limit("encoded output bytes"));
    }
    Ok(TrialEncode::Complete(stream))
}

pub(crate) fn wrap_stream(canvas: &Image, stream: Vec<u8>, maximum: usize) -> Result<Vec<u8>> {
    let mut output = codestream::container_prefix(canvas, stream.len())?;
    let total = stream
        .len()
        .checked_add(output.len())
        .ok_or(Error::Limit("output size overflow"))?;
    if total > maximum {
        return Err(Error::Limit("encoded output bytes"));
    }
    output
        .try_reserve(total - output.len())
        .map_err(|_| Error::Limit("encoded output allocation"))?;
    output.extend(stream);
    Ok(output)
}

fn encode_frame_trial(
    canvas: FrameCanvas<'_>,
    image: &Image,
    options: &EncodeOptions,
    frame: &crate::frame::FrameOptions,
    last: bool,
    patches: Option<&crate::tools::patches::PatchDictionary>,
    budget: Option<&SectionBudget>,
) -> Result<TrialEncode<Vec<u8>>> {
    let _frame = crate::profile::scope("encode.frame");
    let modular_options = options.modular.for_pixel_metadata(image.metadata());
    let credited_before = budget.map_or(0, SectionBudget::section_bytes);
    let reference_passes = crate::modular::layout::Passes::default();
    let passes = if frame.reference_only {
        &reference_passes
    } else {
        &options.passes
    };
    let layout = options.responsive
        || passes.num_passes != 1
        || options.modular.palette_policy.is_some()
        || image
            .metadata()
            .extra_channels
            .iter()
            .any(|e| e.dim_shift != 0);
    let sections = if layout {
        complete_or_return!(crate::modular::orchestration::encode_layout_trial(
            image,
            &modular_options,
            options.group_size_shift,
            passes,
            options.responsive,
            options.shared_tree,
            options.max_output_bytes,
            patches,
            budget,
        ))
    } else {
        let group_dim = 128usize << options.group_size_shift;
        let nx = (image.width() as usize).div_ceil(group_dim);
        let ny = (image.height() as usize).div_ceil(group_dim);
        let dc_groups = (image.width() as usize).div_ceil(group_dim * 8)
            * (image.height() as usize).div_ceil(group_dim * 8);
        let mut groups = Vec::new();
        groups
            .try_reserve(nx * ny)
            .map_err(|_| Error::Limit("group allocation"))?;
        for gy in 0..ny {
            for gx in 0..nx {
                groups.push(Group {
                    x: gx * group_dim,
                    y: gy * group_dim,
                    width: (image.width() as usize - gx * group_dim).min(group_dim),
                    height: (image.height() as usize - gy * group_dim).min(group_dim),
                    stream_id: if nx * ny == 1 {
                        0
                    } else {
                        18 + 3 * dc_groups + gy * nx + gx
                    },
                });
            }
        }
        let single = groups.len() == 1;
        let job = |g: &Group| {
            modular::encode_group(
                image,
                *g,
                &modular_options,
                options.max_output_bytes,
                single,
                patches,
            )
        };
        // Scratch planes/tokens are group-local, so peak scratch scales with workers,
        // not the entire image. Results are collected in stable raster order.
        let palette = if options.modular.palette_policy.is_some() {
            Some(crate::modular::global::encode_policy(
                image,
                &groups,
                &modular_options,
                options.max_output_bytes,
                patches,
                options.shared_tree,
            )?)
        } else if options.global_palette {
            crate::modular::global::encode_palette(
                image,
                &groups,
                &modular_options,
                options.max_output_bytes,
                patches,
                options.shared_tree,
            )?
        } else {
            None
        };
        let (global, encoded) = if let Some(palette) = palette {
            palette
        } else if options.shared_tree {
            crate::modular::frame::encode_shared(
                image,
                &groups,
                &modular_options,
                options.max_output_bytes,
                patches,
            )?
        } else {
            (
                if single {
                    Vec::new()
                } else {
                    modular::empty_global(options.max_output_bytes, patches)?
                },
                execution::try_map(&groups, |_, group| job(group))?,
            )
        };
        let mut sections = Vec::new();
        if !single {
            sections.push(global);
            sections.extend((0..=dc_groups).map(|_| Vec::new()));
        }
        sections.extend(encoded);
        sections
    };
    let sizes: Vec<_> = sections.iter().map(Vec::len).collect();
    let header = match canvas {
        FrameCanvas::Image(canvas) => codestream::frame_header(
            canvas,
            image,
            &sizes,
            options.group_size_shift,
            passes,
            frame,
            last,
            patches.is_some(),
            options.max_output_bytes,
        )?,
        FrameCanvas::Geometry(canvas) => {
            let mut writer = codestream::frame_header_body(
                canvas,
                image.into(),
                options.group_size_shift,
                passes,
                frame,
                last,
                patches.is_some(),
                options.max_output_bytes,
            )?;
            writer.write(1, 0)?;
            writer.zero_pad_to_byte()?;
            for &size in &sizes {
                codestream::write_choice(
                    &mut writer,
                    u32::try_from(size).map_err(|_| Error::Limit("section size"))?,
                    [(0, 10), (1024, 14), (17408, 22), (4211712, 30)],
                )?;
            }
            writer.zero_pad_to_byte()?;
            writer.into_bytes()?
        }
    };
    let stream_size = sections
        .iter()
        .try_fold(header.len(), |n, s| n.checked_add(s.len()))
        .ok_or(Error::Limit("output size overflow"))?;
    if stream_size > options.max_output_bytes {
        return Err(Error::Limit("encoded output bytes"));
    }
    let mut output = Vec::new();
    output
        .try_reserve(stream_size)
        .map_err(|_| Error::Limit("encoded output allocation"))?;
    output.extend_from_slice(&header);
    for section in sections {
        output.extend_from_slice(&section);
    }
    if let Some(budget) = budget {
        // Final section bytes were already charged at their producing tasks.
        // Add only the frame header, or the whole uninstrumented small/legacy
        // frame, to the resource tally. Never count assembly as new evidence.
        let credited = budget.section_bytes() - credited_before;
        let overhead = output.len().checked_sub(credited).ok_or_else(|| {
            Error::Encoding("finalized sections exceed the assembled frame".into())
        })?;
        budget.retain_overhead(overhead)?;
    }
    Ok(TrialEncode::Complete(output))
}

/// Buffers a complete encoding, then writes it. No partial output on codec error.
pub fn encode_to(image: &Image, options: &EncodeOptions, mut output: impl Write) -> Result<()> {
    output.write_all(&encode(image, options)?)?;
    Ok(())
}

pub fn encode_frames_to(
    canvas: &Image,
    frames: &[crate::frame::Frame<'_>],
    options: &EncodeOptions,
    mut output: impl Write,
) -> Result<()> {
    output.write_all(&encode_frames(canvas, frames, options)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::{encode::FixedTree, palette::PaletteOptions, predictor::Predictor};
    use crate::{PixelData, PixelLayout};

    fn image() -> Image {
        Image::new(
            257,
            3,
            PixelLayout::Gray,
            PixelData::U8((0..257 * 3).map(|i| (i % 4) as u8).collect()),
        )
        .unwrap()
    }

    #[test]
    fn null_candidate_and_public_single_thread_cover_core_frame_routes() {
        let image = image();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let base = EncodeOptions {
            threads: 2,
            group_size_shift: 0,
            modular: ModularOptions {
                rct: None,
                predictor: Predictor::Gradient,
                fixed_tree: Some(FixedTree::GradientLeaf),
                lz77: false,
                ..Default::default()
            },
            ..Default::default()
        };
        for route in 0..5 {
            let mut options = base.clone();
            match route {
                0 => {}
                1 => options.shared_tree = true,
                2 => {
                    options.global_palette = true;
                    options.modular.palette_colors = Some(4);
                }
                3 => options.responsive = true,
                4 => options.modular.palette_policy = Some(PaletteOptions::default()),
                _ => unreachable!(),
            }
            pool.install(|| {
                let caller = std::thread::current().id();
                let bytes = encode_serial(&image, &options).unwrap();
                assert!(!bytes.is_empty());
                assert!(bytes.len() <= options.max_output_bytes);
                assert_eq!(std::thread::current().id(), caller);
                assert!(!execution::is_serial());
                assert_eq!(execution::worker_count(), 2);
            });
            // Full codestream equality is not a parallelism requirement. The
            // MA/gather tests check predictor, split, weight and budget semantics.
            assert!(
                !encode_with_pool(&image, &options, &pool)
                    .unwrap()
                    .is_empty()
            );
            options.threads = 1;
            assert!(!encode(&image, &options).unwrap().is_empty());
            assert!(!execution::is_serial());
        }
    }

    #[test]
    fn null_candidate_keeps_validation_and_resource_errors() {
        let image = image();
        let mut options = EncodeOptions {
            threads: 0,
            ..Default::default()
        };
        assert!(matches!(
            encode_serial(&image, &options),
            Err(Error::InvalidInput(_))
        ));
        assert!(!execution::is_serial());
        options.threads = 3;
        options.max_pixels = 1;
        assert!(matches!(
            encode_serial(&image, &options),
            Err(Error::Limit("pixel count"))
        ));
        assert!(!execution::is_serial());
        options.max_pixels = 64_000_000;
        options.max_output_bytes = 1;
        assert!(matches!(
            encode_serial(&image, &options),
            Err(Error::Limit(_))
        ));
        assert!(!execution::is_serial());
        assert_eq!(options.threads, 3);
    }
}
