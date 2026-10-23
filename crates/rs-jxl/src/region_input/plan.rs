use super::{ImageDescriptor, RegionSource, load};
use crate::modular::{layout, orchestration, source_options::one_pass_geometry};
use crate::{EncodeOptions, Error, Result, execution};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SectionKind {
    DcGlobal,
    DcGroup(usize),
    AcGroup(usize),
    AcGlobal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectionInfo {
    /// Canonical TOC position, not emission position and not an MA stream ID.
    pub canonical_index: usize,
    pub kind: SectionKind,
    /// Absolute root-frame Modular stream ID. ACGlobal has no Modular stream.
    pub stream_id: Option<usize>,
}

/// Validated source mode-1 Modular topology. This plan never silently changes
/// a full-frame request to independent streaming or reduces sample fractions.
/// Construct it BEFORE writing any image/container/frame headers to a sink.
pub struct FrameAreaPlan {
    descriptor: ImageDescriptor,
    options: crate::ModularOptions,
    group_size_shift: u8,
    num_groups: usize,
    num_dc_groups: usize,
    num_streams: usize,
    areas: Vec<layout::Rect>,
    sections: Vec<SectionInfo>,
    limit: usize,
    serial: bool,
}

impl FrameAreaPlan {
    /// `options` must already contain the source policy resolved for the ROOT
    /// frame, including palette_policy.groupwise=true. Buffering auto/speed
    /// resolution belongs to the caller; this is not a source search frontend.
    pub fn new(descriptor: ImageDescriptor, options: &EncodeOptions) -> Result<Self> {
        descriptor.validate()?;
        crate::encode::validate_geometry(descriptor.geometry(), options)?;
        let decision = orchestration::groupwise_decision_geometry(
            descriptor.width(),
            descriptor.height(),
            &options.modular,
            options.group_size_shift,
            &options.passes,
            options.responsive,
            options.patches.is_some() || options.source_patches.is_some(),
        )?;
        if decision != orchestration::GroupwiseDecision::FrameAreas {
            return Err(Error::InvalidInput(format!(
                "region encoding requires source frame-area eligibility: {decision:?}"
            )));
        }
        let geometry = one_pass_geometry(
            descriptor.width(),
            descriptor.height(),
            Some(options.group_size_shift),
        )?;
        let areas = layout::frame_areas(
            descriptor.width() as usize,
            descriptor.height() as usize,
            options.group_size_shift,
        )?;
        let count = geometry
            .num_groups
            .checked_add(geometry.num_dc_groups)
            .and_then(|n| n.checked_add(2))
            .ok_or(Error::Limit("region section count"))?;
        let mut sections = Vec::new();
        sections
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("region section topology"))?;
        sections.push(SectionInfo {
            canonical_index: 0,
            kind: SectionKind::DcGlobal,
            stream_id: Some(0),
        });
        let columns = (descriptor.width() as usize).div_ceil(geometry.group_dim);
        for (dc_group, area) in areas.iter().enumerate() {
            sections.push(SectionInfo {
                canonical_index: 1 + dc_group,
                kind: SectionKind::DcGroup(dc_group),
                stream_id: Some(1 + geometry.num_dc_groups + dc_group),
            });
            for y in 0..area.height.div_ceil(geometry.group_dim) {
                for x in 0..area.width.div_ceil(geometry.group_dim) {
                    let group = (area.y / geometry.group_dim + y) * columns
                        + area.x / geometry.group_dim
                        + x;
                    sections.push(SectionInfo {
                        canonical_index: 2 + geometry.num_dc_groups + group,
                        kind: SectionKind::AcGroup(group),
                        stream_id: Some(18 + 3 * geometry.num_dc_groups + group),
                    });
                }
            }
        }
        sections.push(SectionInfo {
            canonical_index: 1 + geometry.num_dc_groups,
            kind: SectionKind::AcGlobal,
            stream_id: None,
        });
        debug_assert_eq!(sections.len(), count);
        let mut local = options
            .modular
            .for_pixel_metadata(descriptor.metadata())
            .into_owned();
        local.entropy_options.streaming_mode = true;
        local.tree_groups = None;
        local.tree_group_splits.clear();
        Ok(Self {
            descriptor,
            options: local,
            group_size_shift: options.group_size_shift,
            num_groups: geometry.num_groups,
            num_dc_groups: geometry.num_dc_groups,
            num_streams: geometry.num_streams,
            areas,
            sections,
            limit: options.max_output_bytes,
            serial: options.threads == 1,
        })
    }

    pub fn descriptor(&self) -> &ImageDescriptor {
        &self.descriptor
    }
    pub fn group_size_shift(&self) -> u8 {
        self.group_size_shift
    }
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }
    pub fn num_dc_groups(&self) -> usize {
        self.num_dc_groups
    }
    pub fn num_streams(&self) -> usize {
        self.num_streams
    }
    pub fn global_section_count(&self) -> usize {
        2
    }
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }
    pub fn area_count(&self) -> usize {
        self.areas.len()
    }

    /// Source output order: DCGlobal, then each raster DC group followed by its
    /// raster AC groups, then empty ACGlobal. Every section, even empty, occurs
    /// once. Parent writes permutation[canonical_index] = emission_position.
    pub fn sections(&self) -> &[SectionInfo] {
        &self.sections
    }

    /// Emits completed byte-aligned sections, never framing or TOC bytes.
    /// Input/sink callbacks stay on this thread and need not be Send. Only owned
    /// area planes and codec work enter the supplied pool. None is the source
    /// null runner, not the global Rayon pool. Samples, trees and encoded bytes
    /// from an area are released before requesting the next area.
    ///
    /// The result counts section bytes only. Parent must also budget framing,
    /// padding and container overhead. A callback/input/codec error terminates
    /// immediately and can leave partial output; no buffered transactional or
    /// resumption guarantee is provided. No whole-frame fallback exists here.
    pub fn emit<S: RegionSource + ?Sized>(
        &self,
        source: &mut S,
        pool: Option<&rayon::ThreadPool>,
        mut emit: impl FnMut(SectionInfo, &[u8]) -> Result<()>,
    ) -> Result<usize> {
        if source.descriptor() != &self.descriptor {
            return Err(Error::InvalidInput(
                "region source descriptor differs from its plan".into(),
            ));
        }
        let mut next = self.sections.iter().copied();
        let mut script = None;
        let mut total = 0usize;
        for (index, &area) in self.areas.iter().enumerate() {
            let channels = load::read_area(&self.descriptor, source, area)?;
            let work = || {
                orchestration::encode_region_area(
                    &self.descriptor,
                    channels,
                    &self.options,
                    self.group_size_shift,
                    area,
                    index == 0,
                    self.limit,
                )
            };
            let encoded = if self.serial {
                execution::serial(work)
            } else if let Some(pool) = pool {
                execution::with_pool(pool, work)
            } else {
                execution::serial(work)
            }?;
            if let Some(script) = &script {
                if script != &encoded.transforms {
                    return Err(Error::Encoding(
                        "streaming global transform script changed between areas".into(),
                    ));
                }
            } else {
                script = Some(encoded.transforms);
            }
            for (canonical_index, bytes) in encoded.sections {
                let section = next
                    .next()
                    .ok_or_else(|| Error::Encoding("unexpected region section".into()))?;
                if canonical_index != section.canonical_index {
                    return Err(Error::Encoding(
                        "region section order differs from its plan".into(),
                    ));
                }
                total = total
                    .checked_add(bytes.len())
                    .filter(|&n| n <= self.limit)
                    .ok_or(Error::Limit("frame section bytes"))?;
                emit(section, &bytes)?;
            }
        }
        let last = next
            .next()
            .ok_or_else(|| Error::Encoding("missing region ACGlobal".into()))?;
        if last.kind != SectionKind::AcGlobal || next.next().is_some() {
            return Err(Error::Encoding("incomplete region section topology".into()));
        }
        emit(last, &[])?;
        Ok(total)
    }
}
