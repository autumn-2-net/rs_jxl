// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Coefficient-only JPEG frame assembly from enc_frame.cc::EncodeGroups at
//! b866b3d. Public byte input and resource contracts live in archival.rs.

use super::{
    ac_tokens::{self, AcTokenError, BlockContextModel, Dct8Input, Dct8Plane, TokenizeLimits},
    cfl::{CflMode, JpegCfl},
    coefficient_order::{Dct8Group, Dct8Image, Dct8Orders, OrderPolicy},
    data::{JpegData, JpegError},
    dc_streams::{self, CflMaps, DcGroup, DcStreamError, PrepareLimits},
    entropy::{self, CoefficientEntropy},
    entropy_writer::WriteError,
    frame_header::{self, FrameGeometry},
    quantization::{DcQuantization, QuantizerParams, RawDct8Quantization},
    reconstruction_metadata::MetadataError,
    scan_writer::ReconstructionError,
    transcode_layout::TranscodeLayout,
};
use crate::{
    Error, Result,
    bit_writer::{BitWriter, BitWriterError},
    entropy::options::SpeedTier,
    metadata::ImageMetadata,
    modular::auxiliary::{Domain, LearnerExecution, Shared, Stream},
    scheduler::WorkerLimits,
};

#[path = "search_cache.rs"]
mod search_cache;
pub(crate) use search_cache::SearchCache;

#[derive(Clone, Debug)]
pub struct Options {
    pub tier: SpeedTier,
    pub threads: usize,
    /// Default-off cooperative auxiliary learners; gathering remains serial.
    pub cooperative_ma: Option<WorkerLimits>,
    pub decoding_speed: u8,
    /// `usize::MAX` means no ceiling on samples selected by the learning policy.
    pub sample_limit: usize,
    pub reconstruction_quality: u8,
    pub cfl: CflMode,
    pub order: OrderPolicy,
    pub max_output_bytes: usize,
    pub max_coefficients: usize,
    pub max_tokens: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tier: SpeedTier::Falcon,
            threads: 1,
            cooperative_ma: None,
            decoding_speed: 0,
            sample_limit: usize::MAX,
            reconstruction_quality: 4,
            cfl: CflMode::Enabled,
            order: OrderPolicy::Source,
            max_output_bytes: usize::MAX,
            max_coefficients: usize::MAX,
            max_tokens: usize::MAX,
        }
    }
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::InvalidInput(error.to_string())
}

fn ac_token_error(error: AcTokenError) -> Error {
    match error {
        AcTokenError::LimitExceeded(resource) => Error::Limit(resource),
        AcTokenError::DimensionOverflow => Error::Limit("JPEG AC dimensions overflow"),
        AcTokenError::AllocationFailed => Error::Limit("JPEG AC token allocation"),
        other => invalid(other),
    }
}

fn dc_stream_error(error: DcStreamError) -> Error {
    match error {
        DcStreamError::Limit(resource) | DcStreamError::Quantization(Error::Limit(resource)) => {
            Error::Limit(resource)
        }
        other => invalid(other),
    }
}

fn metadata_error(error: MetadataError) -> Error {
    match error {
        MetadataError::Limit(resource)
        | MetadataError::Validation(ReconstructionError::Limit(resource))
        | MetadataError::Validation(ReconstructionError::Structural(JpegError::LimitExceeded {
            resource,
            ..
        })) => Error::Limit(resource),
        MetadataError::Bitstream(BitWriterError::SizeLimit | BitWriterError::Allocation) => {
            Error::Limit("JPEG reconstruction metadata output")
        }
        MetadataError::Validation(ReconstructionError::Entropy(WriteError::OutputLimit {
            ..
        })) => Error::Limit("JPEG reconstruction entropy output"),
        MetadataError::Validation(ReconstructionError::Entropy(WriteError::AllocationFailed)) => {
            Error::Limit("JPEG reconstruction entropy allocation")
        }
        other => invalid(other),
    }
}

// Full-size zero tails match the source ACImage allocation/order scan; the
// tokenizer only consumes each subsampled component's packed prefix.
fn coefficient_groups(
    layout: &TranscodeLayout<'_>,
    geometry: FrameGeometry,
    cfl: &JpegCfl<'_, '_>,
    maximum: usize,
) -> Result<Vec<[Vec<i32>; 3]>> {
    let (width, height) = geometry.dimensions_in_blocks();
    let count = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3 * 64))
        .ok_or(Error::Limit("JPEG AC coefficient storage"))?;
    if count > maximum {
        return Err(Error::Limit("JPEG AC coefficient storage"));
    }
    let hs = layout.subsampling().horizontal_shifts();
    let vs = layout.subsampling().vertical_shifts();
    let indices: Vec<_> = (0..geometry.group_count()).collect();
    crate::execution::try_map(&indices, |_, &index| {
        let rect = geometry.ac_group_rect(index).expect("validated group");
        let mut planes: [Vec<i32>; 3] = std::array::from_fn(|_| Vec::new());
        for c in 0..3 {
            let length = rect.width * rect.height * 64;
            planes[c]
                .try_reserve_exact(length)
                .map_err(|_| Error::Limit("JPEG AC group"))?;
            planes[c].resize(length, 0);
            let width = rect.width >> hs[c];
            let height = rect.height >> vs[c];
            for y in 0..height {
                for x in 0..width {
                    let offset = (y * width + x) * 64;
                    let block = (&mut planes[c][offset..offset + 64])
                        .try_into()
                        .expect("block size");
                    cfl.forward_block(
                        c,
                        ((rect.x >> hs[c]) + x) as u32,
                        ((rect.y >> vs[c]) + y) as u32,
                        block,
                    )
                    .map_err(invalid)?;
                    // DC is encoded in the separate Modular stream.
                    planes[c][offset] = 0;
                }
            }
        }
        Ok(planes)
    })
}

fn tokenize_groups(
    layout: &TranscodeLayout<'_>,
    geometry: FrameGeometry,
    groups: &[[Vec<i32>; 3]],
    dc: &[DcGroup],
    orders: &Dct8Orders,
    model: BlockContextModel<'_>,
    options: &Options,
) -> Result<CoefficientEntropy> {
    let hs = layout.subsampling().horizontal_shifts();
    let vs = layout.subsampling().vertical_shifts();
    let dc_columns = geometry.dimensions_in_blocks().0.div_ceil(256);
    // TokenizeCoefficients reserves at most 64 tokens per component block.
    // Check the aggregate reservation before dispatch, not one cap per worker.
    let mut reserved = 0usize;
    let mut bounds = Vec::with_capacity(groups.len());
    for index in 0..groups.len() {
        let rect = geometry.ac_group_rect(index).expect("validated group");
        let bound: usize = (0..3)
            .map(|c| (rect.width >> hs[c]) * (rect.height >> vs[c]) * 64)
            .sum();
        reserved = reserved
            .checked_add(bound)
            .filter(|&n| n <= options.max_tokens)
            .ok_or(Error::Limit("JPEG AC token reservations"))?;
        bounds.push(bound);
    }
    let streams = crate::execution::try_map(groups, |index, planes| {
        let rect = geometry.ac_group_rect(index).expect("validated group");
        let dc_group = &dc[(rect.y / 256) * dc_columns + rect.x / 256];
        let mut qdc = Vec::with_capacity(rect.width * rect.height);
        for y in 0..rect.height {
            let start =
                (rect.y - dc_group.rect.y + y) * dc_group.rect.width + rect.x - dc_group.rect.x;
            qdc.extend_from_slice(&dc_group.quant_dc[start..start + rect.width]);
        }
        let qf = vec![1; rect.width * rect.height];
        let input = Dct8Input {
            width_blocks: rect.width,
            height_blocks: rect.height,
            planes: std::array::from_fn(|c| {
                let width = rect.width >> hs[c];
                let height = rect.height >> vs[c];
                Dct8Plane {
                    coefficients: &planes[c][..width * height * 64],
                    width_blocks: width,
                    height_blocks: height,
                    horizontal_shift: hs[c],
                    vertical_shift: vs[c],
                }
            }),
            orders: orders.positions(),
            qdc: &qdc,
            qf: &qf,
        };
        ac_tokens::tokenize_dct8(
            &input,
            model,
            TokenizeLimits {
                max_tokens: bounds[index],
                max_scratch_bytes: 3 * 32,
            },
        )
        .map_err(ac_token_error)
    })?;
    CoefficientEntropy::source(
        streams,
        model.num_ac_contexts(),
        options.tier,
        options.decoding_speed,
        options.max_tokens,
    )
}

pub fn encode_coefficients(jpeg: &JpegData, options: &Options) -> Result<Vec<u8>> {
    with_threads(options, |pool| {
        Prepared::new(jpeg, options)?.encode(jpeg, options, pool)
    })
}

fn prepare_ac(
    layout: &TranscodeLayout<'_>,
    geometry: FrameGeometry,
    cfl: &JpegCfl<'_, '_>,
    dc: &[DcGroup],
    model: BlockContextModel<'_>,
    options: &Options,
) -> Result<(Dct8Orders, CoefficientEntropy)> {
    let groups = coefficient_groups(layout, geometry, cfl, options.max_coefficients)?;
    let views: Vec<_> = groups
        .iter()
        .map(|p| Dct8Group::I32(p.each_ref().map(Vec::as_slice)))
        .collect();
    let (width, height) = geometry.dimensions_in_blocks();
    let image = Dct8Image::new(width, height, &views).map_err(invalid)?;
    let orders = Dct8Orders::compute(&image, options.tier, options.order);
    let entropy = tokenize_groups(layout, geometry, &groups, dc, &orders, model, options)?;
    Ok((orders, entropy))
}

enum AcData<'a> {
    Direct(&'a Dct8Orders, &'a CoefficientEntropy),
    Cached(&'a search_cache::AcPayload),
}

impl AcData<'_> {
    fn write_global(&self, writer: &mut BitWriter, groups: usize) -> Result<()> {
        match self {
            Self::Direct(orders, entropy) => {
                entropy::write_single_histogram_group(writer, groups)?;
                entropy::write_orders(orders, writer)?;
                entropy.write_model(writer)
            }
            Self::Cached(payload) => payload.write_global(writer),
        }
    }

    fn write_group(&self, index: usize, writer: &mut BitWriter) -> Result<()> {
        match self {
            Self::Direct(_, entropy) => entropy.write_group(index, writer),
            Self::Cached(payload) => payload.write_group(index, writer),
        }
    }
}

pub(crate) fn with_threads<T: Send>(
    options: &Options,
    work: impl FnOnce(Option<&rayon::ThreadPool>) -> Result<T> + Send,
) -> Result<T> {
    if let Some(limits) = options.cooperative_ma {
        limits.validate()?;
    }
    if !(1..=256).contains(&options.threads)
        || options.decoding_speed > 4
        || options.reconstruction_quality > 11
    {
        return Err(Error::InvalidInput(
            "JPEG threads must be 1..=256, decoding_speed 0..=4, Brotli quality 0..=11".into(),
        ));
    }
    if options.sample_limit == 0 {
        return Err(Error::Limit("JPEG MA selected-sample budget"));
    }
    if options.max_output_bytes == 0 || options.max_coefficients == 0 || options.max_tokens == 0 {
        return Err(Error::Limit("JPEG encode budgets must be nonzero"));
    }
    if options.threads == 1 || crate::execution::is_serial() {
        return crate::execution::serial(|| work(None));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()
        .map_err(|e| Error::Encoding(e.to_string()))?;
    crate::execution::with_pool(&pool, || work(Some(&pool)))
}

/// Effort-independent data, prepared once for every whole-frame candidate.
pub(crate) struct Prepared {
    image_header: Vec<u8>,
    reconstruction: Vec<u8>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    level: u8,
}

impl Prepared {
    pub(crate) fn new(jpeg: &JpegData, options: &Options) -> Result<Self> {
        let layout = TranscodeLayout::new(jpeg).map_err(invalid)?;
        let geometry = FrameGeometry::new(jpeg.width, jpeg.height, layout.subsampling())?;
        let display = super::color_metadata::prepare_metadata(
            jpeg,
            options.max_output_bytes,
            crate::metadata::Orientation::Identity,
            true,
            options.max_output_bytes,
        )?;
        if display.color_transform != layout.color_transform() {
            return Err(Error::InvalidInput(
                "JPEG metadata/coefficient color transform mismatch".into(),
            ));
        }
        let mut metadata = ImageMetadata::new(8, display.color_encoding)?;
        metadata.orientation = display.orientation;
        let xmp = super::color_metadata::xmp_box_payload(jpeg, true, options.max_output_bytes)?;
        let (bw, bh) = geometry.dimensions_in_blocks();
        if bw
            .checked_mul(bh)
            .and_then(|n| n.checked_mul(3 * 64))
            .is_none_or(|n| n > options.max_coefficients)
        {
            return Err(Error::Limit("JPEG AC coefficient storage"));
        }
        // Validate reconstruction eligibility before the costly coefficient stage.
        let reconstruction =
            super::reconstruction_metadata::serialize_reconstruction_metadata_with_sources(
                jpeg,
                options.max_output_bytes,
                super::reconstruction_metadata::ReconstructionMetadataSources {
                    icc_profile: metadata.color_encoding.icc_profile(),
                    exif_box_payload: display.exif_box_payload.as_deref(),
                    xmp_box_payload: xmp.as_deref(),
                },
            )
            .map_err(metadata_error)?;
        let jbrd = super::reconstruction_brotli::compress_parts(
            &reconstruction.header,
            &reconstruction.payload,
            u32::from(options.reconstruction_quality),
            options.max_output_bytes,
        )?;
        let image_header = crate::codestream::image_header_fields(
            jpeg.width,
            jpeg.height,
            &metadata,
            options.max_output_bytes,
        )?;
        let level = if u64::from(jpeg.width) * u64::from(jpeg.height) > 1 << 28
            || metadata
                .color_encoding
                .icc_profile()
                .is_some_and(|p| p.len() > 1 << 22)
        {
            10
        } else {
            5
        };
        Ok(Self {
            image_header,
            reconstruction: jbrd,
            exif: display.exif_box_payload,
            xmp,
            level,
        })
    }

    pub(crate) fn encode(
        &self,
        jpeg: &JpegData,
        options: &Options,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Vec<u8>> {
        self.encode_inner(jpeg, options, pool, None)
    }

    pub(crate) fn encode_with_cache(
        &self,
        jpeg: &JpegData,
        options: &Options,
        pool: Option<&rayon::ThreadPool>,
        cache: &mut SearchCache<'_>,
    ) -> Result<Vec<u8>> {
        self.encode_inner(jpeg, options, pool, Some(cache))
    }

    fn encode_inner(
        &self,
        jpeg: &JpegData,
        options: &Options,
        pool: Option<&rayon::ThreadPool>,
        cache: Option<&mut SearchCache<'_>>,
    ) -> Result<Vec<u8>> {
        let layout = TranscodeLayout::new(jpeg).map_err(invalid)?;
        let geometry = FrameGeometry::new(jpeg.width, jpeg.height, layout.subsampling())?;
        let cfl = JpegCfl::compute(&layout, options.cfl);
        let quants = layout.planes().each_ref().map(|p| p.quantization());
        let dc_quant = DcQuantization::from_layout(quants)?;
        let ac_quant = RawDct8Quantization::from_layout(quants)?;
        let contexts = layout.dc_contexts();
        let model = BlockContextModel::new(
            [&[], contexts.luma_thresholds(), &[]],
            &[],
            contexts.context_map(),
            usize::from(contexts.num_contexts()),
        )
        .map_err(ac_token_error)?;
        let dc_indices: Vec<_> = (0..geometry.dc_group_count()).collect();
        let dc = crate::execution::try_map(&dc_indices, |_, &i| {
            dc_streams::prepare_dc_group(
                &layout,
                &dc_quant,
                &contexts,
                geometry,
                i,
                &PrepareLimits::default(),
            )
            .map_err(dc_stream_error)
        })?;
        let build_ac = || prepare_ac(&layout, geometry, &cfl, &dc, model, options);
        let direct;
        let ac = if let Some(cache) = cache {
            AcData::Cached(cache.get_or_build(jpeg, geometry, options, build_ac)?)
        } else {
            direct = build_ac()?;
            AcData::Direct(&direct.0, &direct.1)
        };
        let maps = cfl.dimensions_in_tiles();
        let maps = CflMaps {
            width: maps.0 as usize,
            height: maps.1 as usize,
            ytox: cfl.ytox_map(),
            ytob: cfl.ytob_map(),
        };
        let context = super::modular_policy::StreamContext {
            num_streams: 1 + 3 * geometry.dc_group_count() + 17 + geometry.group_count(),
            force_cfl_jpeg_recompression: options.cfl == CflMode::Enabled,
            dc_tree_pixels: Some(
                dc.iter()
                    .flat_map(|g| &g.channels)
                    .map(|c| c.samples.len())
                    .sum(),
            ),
            ..Default::default()
        };
        let mut dc_options = context.options_for_dc(options.tier as i8, options.decoding_speed)?;
        let mut metadata_options =
            context.options_for_ac_metadata(options.tier as i8, options.decoding_speed)?;
        let mut quant_options =
            context.options_for_quant(options.tier as i8, options.decoding_speed)?;
        let mut global_options = super::modular_policy::options_for_global_entropy(
            options.tier as i8,
            options.decoding_speed,
        )?;
        for policy in [
            &mut dc_options,
            &mut metadata_options,
            &mut quant_options,
            &mut global_options,
        ] {
            policy.sample_limit =
                (options.sample_limit != usize::MAX).then_some(options.sample_limit);
        }
        let ac_metadata = (0..geometry.dc_group_count())
            .map(|index| {
                dc_streams::prepare_ac_metadata_group(
                    geometry,
                    index,
                    maps,
                    &PrepareLimits::default(),
                )
                .map_err(dc_stream_error)
            })
            .collect::<Result<Vec<_>>>()?;
        let quant_channels = ac_quant.prepare_modular_channels()?;
        let dc_streams: Vec<_> = dc
            .iter()
            .enumerate()
            .map(|(i, g)| Stream {
                id: 1 + i,
                channels: &g.channels,
                transforms: &[],
            })
            .collect();
        let metadata_streams: Vec<_> = ac_metadata
            .iter()
            .enumerate()
            .map(|(i, g)| Stream {
                id: 1 + 2 * geometry.dc_group_count() + i,
                channels: &g.channels,
                transforms: &[],
            })
            .collect();
        let quant_id = 1 + 3 * geometry.dc_group_count();
        let quant_streams = [Stream {
            id: quant_id,
            channels: &quant_channels,
            transforms: &[],
        }];
        // No Global/ModularDC/ModularAC pixels exist, but their ID gaps must
        // remain in the learned tree's property-1 routing.
        let auxiliary = Shared::build_with_execution(
            &[
                Domain {
                    start: 1,
                    streams: &dc_streams,
                    options: &dc_options,
                },
                Domain {
                    start: 1 + 2 * geometry.dc_group_count(),
                    streams: &metadata_streams,
                    options: &metadata_options,
                },
                Domain {
                    start: quant_id,
                    streams: &quant_streams,
                    options: &quant_options,
                },
            ],
            context.num_streams,
            &global_options,
            8,
            options.max_tokens,
            LearnerExecution {
                pool,
                cooperative: options.cooperative_ma,
            },
        )?;
        let mut sections: Vec<_> = (0..geometry.section_count())
            .map(|_| BitWriter::new(options.max_output_bytes))
            .collect();
        let section = |index| {
            if geometry.section_count() == 1 {
                0
            } else {
                index
            }
        };
        let global = &mut sections[0];
        dc_quant.write(global)?;
        QuantizerParams::jpeg().write(global)?;
        ac_tokens::write_block_context_map(model, global)?;
        entropy::write_jpeg_color_correlation(global)?;
        auxiliary.write_global_info(global)?;
        for (index, dc_group) in dc.iter().enumerate() {
            let writer = &mut sections[section(index + 1)];
            writer.write(2, u64::from(dc_group.extra_dc_precision))?;
            auxiliary.write_stream(index + 1, writer)?;
            let metadata = &ac_metadata[index];
            let area = metadata.rect.width * metadata.rect.height;
            let bits = (usize::BITS - (area - 1).leading_zeros()) as u8;
            writer.write(bits, (area - 1) as u64)?;
            auxiliary.write_stream(1 + 2 * geometry.dc_group_count() + index, writer)?;
        }
        let global_ac = &mut sections[section(1 + geometry.dc_group_count())];
        ac_quant.write_header(global_ac, |w, _| auxiliary.write_stream(quant_id, w))?;
        ac.write_global(global_ac, geometry.group_count())?;
        for index in 0..geometry.group_count() {
            ac.write_group(
                index,
                &mut sections[section(2 + geometry.dc_group_count() + index)],
            )?;
        }
        let sections = sections
            .into_iter()
            .map(|mut w| {
                w.zero_pad_to_byte()?;
                w.into_bytes().map_err(Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        let sizes: Vec<_> = sections.iter().map(Vec::len).collect();
        let frame_header = frame_header::write_header(
            geometry,
            layout.color_transform(),
            &sizes,
            options.max_output_bytes,
        )?;
        let mut output = BitWriter::new(options.max_output_bytes);
        output.append_byte_aligned(&self.image_header)?;
        output.append_byte_aligned(&frame_header)?;
        for bytes in sections {
            output.append_byte_aligned(&bytes)?;
        }
        let codestream = output.into_bytes()?;
        let mut boxes: Vec<(&[u8; 4], &[u8])> = self
            .exif
            .as_deref()
            .map(|payload| (b"Exif", payload))
            .into_iter()
            .collect();
        if let Some(payload) = self.xmp.as_deref() {
            boxes.push((b"xml ", payload));
        }
        super::container::wrap_with_metadata(
            &codestream,
            &self.reconstruction,
            self.level,
            &boxes,
            options.max_output_bytes,
        )
    }
}

#[cfg(test)]
#[path = "transcode_tests.rs"]
mod tests;

#[cfg(test)]
mod resource_error_tests {
    use super::*;

    #[test]
    fn metadata_resource_errors_preserve_limit_classification() {
        for error in [
            MetadataError::Limit("metadata payload"),
            MetadataError::Validation(ReconstructionError::Limit("scan work")),
            MetadataError::Validation(ReconstructionError::Structural(JpegError::LimitExceeded {
                offset: 17,
                resource: "marker count",
            })),
            MetadataError::Bitstream(BitWriterError::SizeLimit),
            MetadataError::Bitstream(BitWriterError::Allocation),
            MetadataError::Validation(ReconstructionError::Entropy(WriteError::OutputLimit {
                limit: 1,
            })),
            MetadataError::Validation(ReconstructionError::Entropy(WriteError::AllocationFailed)),
        ] {
            let actual = metadata_error(error);
            assert!(matches!(actual, Error::Limit(_)), "{actual:?}");
        }
    }

    #[test]
    fn malformed_and_unsupported_metadata_remain_invalid_input() {
        for error in [
            MetadataError::Invalid("marker order"),
            MetadataError::Unsupported("restart fill"),
            MetadataError::Validation(ReconstructionError::Invalid("coefficient storage")),
            MetadataError::Validation(ReconstructionError::Structural(JpegError::Unsupported {
                offset: 17,
                reason: "arithmetic coding",
            })),
            MetadataError::Validation(ReconstructionError::Entropy(
                WriteError::MissingHuffmanSymbol(0),
            )),
            MetadataError::Bitstream(BitWriterError::ValueOutOfRange),
        ] {
            let actual = metadata_error(error);
            assert!(matches!(actual, Error::InvalidInput(_)), "{actual:?}");
        }
    }

    #[test]
    fn ac_token_errors_distinguish_resources_from_invalid_input() {
        for error in [
            AcTokenError::LimitExceeded("token reservation"),
            AcTokenError::DimensionOverflow,
            AcTokenError::AllocationFailed,
        ] {
            let actual = ac_token_error(error);
            assert!(matches!(actual, Error::Limit(_)), "{actual:?}");
        }
        assert!(matches!(
            ac_token_error(AcTokenError::Invalid("coefficient order")),
            Error::InvalidInput(_)
        ));
    }

    #[test]
    fn dc_errors_distinguish_nested_resources_from_invalid_input() {
        for error in [
            DcStreamError::Limit("channel allocation"),
            DcStreamError::Quantization(Error::Limit("quantization allocation")),
        ] {
            let actual = dc_stream_error(error);
            assert!(matches!(actual, Error::Limit(_)), "{actual:?}");
        }
        for error in [
            DcStreamError::Invalid("frame geometry"),
            DcStreamError::Quantization(Error::InvalidInput("quantization range".into())),
            DcStreamError::DcOutOfRange {
                channel: 1,
                x: 0,
                y: 0,
                coefficient: 2048,
            },
            DcStreamError::DcNotReversible {
                channel: 1,
                x: 0,
                y: 0,
                adjusted: 1,
                encoded: 2,
            },
        ] {
            let actual = dc_stream_error(error);
            assert!(matches!(actual, Error::InvalidInput(_)), "{actual:?}");
        }
    }

    #[test]
    fn zero_backend_budgets_are_limits_before_frame_work() {
        for options in [
            Options {
                max_output_bytes: 0,
                ..Default::default()
            },
            Options {
                max_coefficients: 0,
                ..Default::default()
            },
            Options {
                max_tokens: 0,
                ..Default::default()
            },
            Options {
                sample_limit: 0,
                ..Default::default()
            },
        ] {
            let error = encode_coefficients(&JpegData::default(), &options).unwrap_err();
            assert!(matches!(error, Error::Limit(_)), "{error:?}");
        }
    }
}
