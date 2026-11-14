// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! encode.cc::CanDoFastLossless typed-input adapter for direct effort-1
//! candidates. Cumulative search dispatch is owned by the caller.

use super::params::{CandidateParams, Override};
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
    // FJXL has its own fixed transform/tree/entropy policy. Do not silently
    // discard explicit Modular settings, including delta palettes (which the
    // cumulative-search default-options predicate intentionally permits).
    let expected = CandidateParams {
        speed_tier: 9,
        buffering: params.buffering,
        patches: params.patches,
        ..Default::default()
    };
    params.speed_tier == 9
        && *params == expected
        && matches!(
            params.buffering,
            super::params::SourceBuffering::Auto | super::params::SourceBuffering::FullFrame
        )
        && matches!(params.patches, Override::Default | Override::Off)
        && !params.responsive
        && !base.responsive
        && base.passes == Default::default()
        && compatible_options(image, base)
        && view(image).is_some()
}

// Callers currently pass projected options, not the original base. Accept the
// ordinary tier-8 Modular fallback policy, but retain surviving opt-in tools
// and diagnostic collectors on Modular rather than returning empty receipts.
// Settings cleared by projection cannot be recovered here: callers requiring
// exact original-base semantics must check those before preparing an artifact.
fn compatible_options(image: &Image, base: &EncodeOptions) -> bool {
    let modular = &base.modular;
    let workers = super::worker_pool::WorkerLimits::default();
    !base.has_patches()
        && base.source_patch_reference.is_none()
        && compatible_profile(image, base)
        && modular.custom_tree.is_none()
        && modular.tree_groups.is_none()
        && modular.tree_group_splits.is_empty()
        && modular.squeeze.is_empty()
        && modular.source_lz77.is_none()
        && modular.weighted_preset == 0
        && modular.weighted_candidates.is_none()
        && modular.rct_candidates.is_none()
        && modular.palette_colors.is_none()
        && modular.palette_predictor.is_none()
        && modular.previous_channels == 0
        && modular.candidate_cache.is_none()
        && !modular.refinement.enabled()
        && modular.refinement.receipt.is_none()
        && modular.sample_limit.is_none()
        && modular.ma_budget_mode.is_none()
        && modular.ma_max_leaves.is_none()
        && modular.fast_context.is_none()
        && modular.block_context.is_none()
        && modular.source_ma_seed.is_none()
        && modular.source_ma_lookahead.is_none()
        && !modular.tree_learning.enabled()
        && modular.tree_learning_receipt.is_none()
        && modular.source_gather.is_none()
        && !modular.source_constant_channels
        && !modular.source_predictor_dedup
        && modular.source_ma_histograms == Default::default()
        && modular.source_ma_scoring == Default::default()
        && modular.source_ma_search == Default::default()
        && !modular.source_ma_parallel.subtrees
        && !modular.source_ma_parallel.balanced
        && !modular.source_ma_parallel.small_subtrees
        && !modular.source_ma_parallel.dynamic_subtrees
        && !modular.source_ma_parallel.inherit_histograms
        && modular.source_ma_parallel.workers.tasks == workers.tasks
        && modular.source_ma_parallel.workers.scratch_bytes == workers.scratch_bytes
        && !modular.source_ma_parallel.quiet
        && modular.source_channel_tree == crate::modular::ma_train::ChannelTreeMode::Disabled
        && !modular.source_channel_tree_final_only
        && !modular.parallel_tokens
        && !modular.parallel_weighted
        && modular.palette_group_receipt.is_none()
        && modular.representation_receipt.is_none()
        && modular.model_cost_receipt.is_none()
        && modular.palette_policy.as_ref().is_none_or(|policy| {
            *policy
                == crate::modular::palette::PaletteOptions {
                    speed_tier: 8,
                    groupwise: policy.groupwise,
                    ..Default::default()
                }
        })
}

fn compatible_profile(image: &Image, base: &EncodeOptions) -> bool {
    use crate::entropy::{
        options::{EntropyOptions, ModularEntropyParams, SpeedTier},
        stream::Coding,
    };
    use crate::modular::{encode::FixedTree, predictor::Predictor};
    use crate::tools::lz77::Lz77Mode;

    let modular = &base.modular;
    if modular.sampling.is_none() {
        // Also permit the unprojected default used by adapter-level callers.
        return modular.predictor == Predictor::Best
            && modular.rct == Some(6)
            && modular.fixed_tree.is_none()
            && modular.entropy == Coding::Auto
            && modular.entropy_options == EntropyOptions::default()
            && modular.lz77
            && modular.lz77_mode == Lz77Mode::Lazy
            && !modular.source_tree_entropy
            && modular.tree_samples == 1024
            && modular.split_threshold == 48.0
            && modular.max_tree_leaves == 16
            && modular.max_merged_tree_leaves.is_none();
    }
    // Projection owns these settings. Do not accept a different fixed tree,
    // entropy profile or resource ceiling merely because the effort is one.
    let mut entropy = EntropyOptions::for_modular(ModularEntropyParams {
        speed_tier: SpeedTier::Thunder,
        ..Default::default()
    });
    entropy.streaming_mode = base
        .modular
        .palette_policy
        .as_ref()
        .is_some_and(|p| p.groupwise);
    modular.predictor == Predictor::Gradient
        && modular.rct == (image.color_channels() == 3).then_some(6)
        && modular.fixed_tree == Some(FixedTree::Gradient)
        && modular.entropy == Coding::Source
        && modular.entropy_options == entropy
        && !modular.lz77
        && modular.lz77_mode == Lz77Mode::Greedy
        && modular.source_tree_entropy
        && modular.tree_samples == 1024
        && modular.split_threshold == 187.0
        && modular.max_tree_leaves == crate::modular::ma_train::MAX_LEAVES
        && modular.max_merged_tree_leaves
            == Some(crate::modular::ma_train::global_tree_leaf_limit(
                image.width() as usize,
                image.height() as usize,
                image.channel_count(),
            ))
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
    use crate::{
        PixelLayout,
        metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType, IntrinsicSize},
        modular::predictor::Predictor,
        scheduler::{
            adapter::project_native,
            params::{DeltaPalette, SourceBuffering},
        },
    };

    fn effort_one() -> CandidateParams {
        CandidateParams::default().backend(1).unwrap()
    }

    #[test]
    fn default_integer_effort_one_accepts_actual_projected_options() {
        for layout in [
            PixelLayout::Gray,
            PixelLayout::GrayAlpha,
            PixelLayout::Rgb,
            PixelLayout::Rgba,
        ] {
            for wide in [false, true] {
                for side in [4, 1024] {
                    let count = side as usize * side as usize * layout.channels();
                    let pixels = if wide {
                        PixelData::U16(vec![0; count])
                    } else {
                        PixelData::U8(vec![0; count])
                    };
                    let image = Image::new(side, side, layout, pixels).unwrap();
                    for buffering in [SourceBuffering::Auto, SourceBuffering::FullFrame] {
                        let params = CandidateParams {
                            buffering,
                            ..effort_one()
                        };
                        let projected = project_native(&params, &image, &EncodeOptions::default())
                            .unwrap()
                            .exact_options()
                            .unwrap();
                        assert!(
                            eligible(&image, &params, &projected),
                            "{layout:?} wide={wide} side={side} {buffering:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn explicit_candidate_tools_are_not_silently_ignored() {
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 48])).unwrap();
        let overrides: &[fn(&mut CandidateParams)] = &[
            |p| p.speed_tier = 8,
            |p| p.responsive = true,
            |p| p.buffering = SourceBuffering::Streaming,
            |p| p.buffering = SourceBuffering::StreamingAbove2048,
            |p| p.group_size_shift = Some(1),
            |p| p.rct = Some(0),
            |p| p.palette_colors = 0,
            |p| p.channel_colors_percent = 0.0,
            |p| p.channel_colors_pre_transform_percent = 0.0,
            |p| p.patches = Override::On,
            |p| p.predictor = Some(Predictor::Zero),
            |p| p.tree_mode = super::super::params::TreeMode::GradientOnly,
            |p| p.learning_fraction = 0.1,
            |p| p.max_properties = Some(0),
            |p| p.previous_property_subset = 2,
            |p| p.lz77_careful_first_pass = true,
            |p| p.bucket_matcher = true,
            |p| p.precise_entropy = true,
            |p| p.probe_lz77 = 1,
            |p| p.delta_palette = DeltaPalette::ExactLeft,
            |p| p.squeeze.push(true),
        ];
        for (index, change) in overrides.iter().enumerate() {
            let mut params = effort_one();
            change(&mut params);
            assert!(
                !eligible(&image, &params, &EncodeOptions::default()),
                "override {index}: {params:?}"
            );
        }
    }

    #[test]
    fn surviving_base_tools_keep_the_projected_modular_route() {
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 48])).unwrap();
        let params = effort_one();
        let overrides: &[fn(&mut EncodeOptions)] = &[
            |o| o.modular.refinement.entropy = true,
            |o| o.modular.refinement.leaves = true,
            |o| o.modular.refinement.context_map = Some(Default::default()),
            |o| o.modular.sample_limit = Some(100),
            |o| o.modular.ma_max_leaves = Some(16),
            |o| o.modular.representation_receipt = Some(Default::default()),
            |o| o.modular.model_cost_receipt = Some(Default::default()),
            |o| {
                o.modular.palette_policy = Some(crate::modular::palette::PaletteOptions {
                    force_palette: true,
                    ..Default::default()
                })
            },
        ];
        for (index, change) in overrides.iter().enumerate() {
            let mut base = EncodeOptions::default();
            change(&mut base);
            let projected = project_native(&params, &image, &base)
                .unwrap()
                .exact_options()
                .unwrap();
            assert!(
                !eligible(&image, &params, &projected),
                "base override {index}"
            );
        }
        let mut base = EncodeOptions::default();
        base.responsive = true;
        assert!(!eligible(&image, &params, &base));
        base.responsive = false;
        base.passes = crate::modular::layout::Passes::progressive();
        assert!(!eligible(&image, &params, &base));
        // Projection rejects offsets with the low-effort fixed tree. The raw
        // eligibility predicate must not bypass that explicit tool either.
        let mut base = EncodeOptions::default();
        base.modular.refinement.offsets = Some(Default::default());
        assert!(!eligible(&image, &params, &base));
    }

    #[test]
    fn custom_profiles_and_unprojected_tools_do_not_qualify() {
        use crate::modular::{
            encode::FixedTree,
            tree::{Node, Tree},
        };

        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 48])).unwrap();
        let params = effort_one();
        let overrides: &[fn(&mut EncodeOptions)] = &[
            |o| {
                o.modular.custom_tree = Some(
                    Tree::new(vec![Node::Leaf {
                        predictor: Predictor::Zero,
                        offset: 0,
                        multiplier: 1,
                    }])
                    .unwrap(),
                )
            },
            |o| o.modular.tree_groups = Some(Default::default()),
            |o| o.modular.tree_group_splits = vec![0, 16],
            |o| o.modular.squeeze = vec![true],
            |o| o.modular.source_lz77 = Some(Default::default()),
            |o| o.modular.weighted_preset = 1,
            |o| o.modular.weighted_candidates = Some(2),
            |o| o.modular.fixed_tree = Some(FixedTree::Zero),
            |o| o.modular.rct_candidates = Some(2),
            |o| o.modular.tree_samples = 32,
            |o| o.modular.source_constant_channels = true,
            |o| o.modular.ma_budget_mode = Some(crate::modular::ma_train::MaBudgetMode::Bounded),
            |o| o.modular.max_tree_leaves = 8,
            |o| o.modular.max_merged_tree_leaves = Some(8),
            |o| o.modular.source_ma_parallel.workers.tasks = 1,
            |o| o.modular.source_ma_parallel.workers.scratch_bytes = 1024,
        ];
        let projected = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        for initial in [EncodeOptions::default(), projected] {
            assert!(eligible(&image, &params, &initial));
            for (index, change) in overrides.iter().enumerate() {
                let mut options = initial.clone();
                change(&mut options);
                assert!(
                    !eligible(&image, &params, &options),
                    "profile override {index}"
                );
            }
        }
    }

    #[test]
    fn projected_explicit_leaf_budget_keeps_modular() {
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 48])).unwrap();
        let params = effort_one();
        let options = crate::scheduler::adapter::project_native_with_resources(
            &params,
            &image,
            &EncodeOptions::default(),
            crate::scheduler::adapter::NativeResourcePolicy {
                max_tree_leaves: Some(8),
            },
        )
        .unwrap()
        .exact_options()
        .unwrap();
        assert!(!eligible(&image, &params, &options));
    }

    #[test]
    fn validation_and_transport_budgets_precede_fast_encoding() {
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 48])).unwrap();
        for options in [
            EncodeOptions {
                threads: 0,
                ..Default::default()
            },
            EncodeOptions {
                max_pixels: 15,
                ..Default::default()
            },
            EncodeOptions {
                max_output_bytes: 0,
                ..Default::default()
            },
        ] {
            assert!(encode(&image, &options).is_err());
            assert!(encode_in_pool(&image, &options, None).is_err());
        }
        // Invalid Modular controls must not be hidden by the FJXL branch.
        let mut options = EncodeOptions::default();
        options.modular.tree_samples = 0;
        assert!(matches!(
            encode(&image, &options),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            encode_in_pool(&image, &options, None),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn metadata_and_invisible_rgb_do_not_require_a_pixel_conversion() {
        let pixels = vec![23, 41, 67, 0, 89, 101, 127, 0];
        let image = Image::new(2, 1, PixelLayout::Rgba, PixelData::U8(pixels.clone())).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.intrinsic_size = Some(IntrinsicSize::new(8, 4).unwrap());
        metadata.extra_channels[0].name = "original alpha".into();
        metadata.extra_channels[0].alpha_associated = true;
        let image = image.with_metadata(metadata.clone(), vec![]).unwrap();
        assert!(eligible(&image, &effort_one(), &EncodeOptions::default()));
        let borrowed = view(&image).unwrap();
        assert_eq!(borrowed.pixels, pixels.as_slice());
        assert_eq!(borrowed.channels, 4);
        assert_eq!(image.metadata(), &metadata);
    }

    #[test]
    fn unsupported_forms_keep_modular() {
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 48])).unwrap();
        let base = EncodeOptions::default();
        let mut params = CandidateParams {
            speed_tier: 9,
            ..Default::default()
        };
        assert!(eligible(&image, &params, &base));
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
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U32(vec![0; 48])).unwrap();
        assert!(!eligible(&image, &params, &base));
        let image = Image::new(4, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 48])).unwrap();
        let mut metadata = image.metadata().clone();
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap());
        let image = image
            .with_metadata(metadata, vec![PixelData::U8(vec![0; 16])])
            .unwrap();
        assert!(!eligible(&image, &params, &base));
        let image = Image::new(4, 4, PixelLayout::Rgba, PixelData::U16(vec![0; 64])).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.extra_channels[0].bit_depth = BitDepth::new(8).unwrap();
        let image = image.with_metadata(metadata, vec![]).unwrap();
        assert!(!eligible(&image, &params, &base));
    }
}
