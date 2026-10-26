// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Resumable enc_frame.cc::EncodeFrameStreaming / enc_modular.cc group work.
//! Full-resolution channels permit independent original AC rectangles: global
//! RCT is pointwise, DC streams are empty, and palettes/WP/MA remain local.
//! The caller owns metadata validation, option resolution, framing and the pool.

#![forbid(unsafe_code)]

use super::*;
use layout::Rect;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Borrows the original image; retains only encoded sections and their global
/// transform script, never prepared image planes or a shared tree/model.
pub(crate) struct GroupSession<'a> {
    image: &'a Image,
    local: ModularOptions,
    policy: PaletteOptions,
    group_dim: usize,
    columns: usize,
    num_dc_groups: usize,
    stream_id_base: usize,
    groups: Vec<Option<Vec<u8>>>,
    global: Option<Vec<u8>>,
    script: Option<Vec<Transform>>,
    completed: usize,
    retained: usize,
    limit: usize,
    poisoned: bool,
}

impl<'a> GroupSession<'a> {
    /// `None` is only an unsupported original-image layout. Invalid options and
    /// resource failures are errors. Pass metadata-resolved options and no patches.
    pub(crate) fn new(
        image: &'a Image,
        options: &ModularOptions,
        group_size_shift: u8,
        passes: &Passes,
        responsive: bool,
        limit: usize,
    ) -> Result<Option<Self>> {
        if groupwise_decision(image, options, group_size_shift, passes, responsive, false)?
            != GroupwiseDecision::FrameAreas
            || image
                .metadata()
                .extra_channels
                .iter()
                .any(|extra| extra.dim_shift != 0)
        {
            return Ok(None);
        }
        let width = image.width() as usize;
        let height = image.height() as usize;
        // Reuse the original layout's dimension and total-stream resource gates.
        let num_dc_groups = layout::frame_areas(width, height, group_size_shift)?.len();
        let group_dim = 128usize << group_size_shift;
        let columns = width.div_ceil(group_dim);
        let num_groups = columns
            .checked_mul(height.div_ceil(group_dim))
            .ok_or(Error::Limit("AC group count"))?;
        let stream_id_base = num_dc_groups
            .checked_mul(3)
            .and_then(|n| n.checked_add(18))
            .ok_or(Error::Limit("Modular stream ID"))?;
        stream_id_base
            .checked_add(num_groups - 1)
            .ok_or(Error::Limit("Modular stream ID"))?;
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(num_groups)
            .map_err(|_| Error::Limit("group session sections"))?;
        groups.resize_with(num_groups, || None);
        let policy = source_policy(options, false)?;
        let mut local = options.clone();
        local.entropy_options.streaming_mode = true;
        local.tree_groups = None;
        local.tree_group_splits.clear();
        // Do not keep another candidate's cached whole-frame planes alive.
        local.candidate_cache = None;
        if let Some(collector) = &local.representation_receipt {
            collector.begin()?;
        }
        Ok(Some(Self {
            image,
            local,
            policy,
            group_dim,
            columns,
            num_dc_groups,
            stream_id_base,
            groups,
            global: None,
            script: None,
            completed: 0,
            retained: 0,
            limit,
            poisoned: false,
        }))
    }

    pub(crate) fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Original, aligned AC group rectangle, clipped at the image edges.
    pub(crate) fn group_rect(&self, index: usize) -> Result<Rect> {
        if index >= self.group_count() {
            return Err(Error::InvalidInput(
                "group session index out of range".into(),
            ));
        }
        let x = index % self.columns * self.group_dim;
        let y = index / self.columns * self.group_dim;
        Ok(Rect {
            x,
            y,
            width: self.group_dim.min(self.image.width() as usize - x),
            height: self.group_dim.min(self.image.height() as usize - y),
        })
    }

    pub(crate) fn completed_groups(&self) -> usize {
        self.completed
    }

    /// Encoded payload bytes, including DCGlobal once the first batch completes.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained
    }

    /// None for either an unencoded group or an out-of-range index.
    pub(crate) fn group_bytes(&self, index: usize) -> Option<usize> {
        self.groups.get(index)?.as_ref().map(Vec::len)
    }

    fn prepare_group(&self, index: usize) -> Result<(Vec<Transform>, Prepared)> {
        let _scope = crate::profile::scope("modular.prepare_session_group");
        let mut channels = read_image_area(self.image, &self.local, self.group_rect(index)?)?;
        let mut state = effective_depth_metadata(self.image.metadata())?;
        let max_depth = if self.image.metadata().modular_16_bit_buffer_sufficient {
            16
        } else {
            32
        };
        if state.max_bitdepth > max_depth {
            return Err(Error::InvalidInput(
                "sample depth exceeds promised Modular buffer".into(),
            ));
        }
        group::validate_layout(&channels, 0)?;
        let transforms = apply_global_rct(
            &mut channels,
            &mut state,
            &self.local,
            &self.policy,
            max_depth,
        )?;
        let prepared = finish_stream(
            channels,
            Vec::new(),
            PaletteState {
                meta_channels: 0,
                max_bitdepth: 0,
                max_value: 0,
            },
            self.stream_id_base + index,
            true,
            state.max_bitdepth < max_depth,
            true,
            self.image.bits(),
            &self.local,
            &self.policy,
        )?;
        Ok((transforms, prepared))
    }

    /// Uses the caller's execution scope. Duplicate and completed indices are
    /// free; every error permanently prevents further encoding or finalization.
    pub(crate) fn encode_groups(&mut self, indices: &[usize]) -> Result<()> {
        if self.poisoned {
            return Err(Error::Encoding("group session is poisoned".into()));
        }
        // Set before entering workers so an unwinding caller cannot resume a
        // partially encoded invocation whose diagnostics may already be updated.
        self.poisoned = true;
        let result = self.encode_new_groups(indices);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn encode_new_groups(&mut self, indices: &[usize]) -> Result<()> {
        let capacity = indices.len().min(self.group_count());
        let mut seen = HashSet::new();
        seen.try_reserve(capacity)
            .map_err(|_| Error::Limit("group session work list"))?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Limit("group session work list"))?;
        for &index in indices {
            let slot = self
                .groups
                .get(index)
                .ok_or_else(|| Error::InvalidInput("group session index out of range".into()))?;
            if slot.is_none() && seen.insert(index) {
                pending.push(index);
            }
        }
        if pending.is_empty() {
            return Ok(());
        }
        // Include previous stages while bounding concurrently completed output.
        // Worker-local planes/tokens are released before try_map returns.
        let size = AtomicUsize::new(self.retained);
        let remaining = self.limit - self.retained;
        let encoded = execution::try_map(&pending, |_, &index| {
            let (transforms, prepared) = self.prepare_group(index)?;
            let bytes = frame::encode_prepared_independent(
                &prepared,
                &self.local,
                remaining,
                false,
                None,
                self.image.bits(),
            )?;
            size.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                n.checked_add(bytes.len()).filter(|&n| n <= self.limit)
            })
            .map_err(|_| Error::Limit("frame section bytes"))?;
            Ok((transforms, bytes))
        })?;
        let script = self.script.as_ref().unwrap_or(&encoded[0].0);
        if encoded.iter().any(|(transforms, _)| transforms != script) {
            return Err(Error::Encoding(
                "streaming global transform script changed between groups".into(),
            ));
        }
        let mut size = size.into_inner();
        if self.script.is_none() {
            if let Some(collector) = &self.local.representation_receipt {
                collector.publish(RepresentationReceipt::groupwise(
                    self.image.width(),
                    self.image.height(),
                    self.image.channel_count(),
                ))?;
            }
            let global = Prepared {
                planes: Vec::new(),
                transforms: script.clone(),
                wp: weighted::Header::preset(u32::from(self.local.weighted_preset))?,
                stream_id: 0,
            };
            let bytes = frame::encode_prepared_independent(
                &global,
                &self.local,
                self.limit - size,
                true,
                None,
                self.image.bits(),
            )?;
            size = size
                .checked_add(bytes.len())
                .filter(|&n| n <= self.limit)
                .ok_or(Error::Limit("frame section bytes"))?;
            self.script = Some(global.transforms);
            self.global = Some(bytes);
        }
        // Commit only a completely successful batch. Model diagnostics may be
        // partial on errors, just as for the ordinary non-resumable encoder.
        for (index, (_, bytes)) in pending.into_iter().zip(encoded) {
            self.groups[index] = Some(bytes);
            self.completed += 1;
        }
        self.retained = size;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn complete_remaining(&mut self) -> Result<()> {
        if self.poisoned {
            return Err(Error::Encoding("group session is poisoned".into()));
        }
        let mut remaining = Vec::new();
        if remaining
            .try_reserve_exact(self.group_count() - self.completed)
            .is_err()
        {
            self.poisoned = true;
            return Err(Error::Limit("group session work list"));
        }
        remaining.extend(
            self.groups
                .iter()
                .enumerate()
                .filter_map(|(i, bytes)| bytes.is_none().then_some(i)),
        );
        self.encode_groups(&remaining)
    }

    /// Canonical TOC: DCGlobal, empty DC groups, empty ACGlobal, then AC groups.
    pub(crate) fn finish(self) -> Result<Vec<Vec<u8>>> {
        if self.poisoned {
            return Err(Error::Encoding("group session is poisoned".into()));
        }
        if self.completed != self.group_count() {
            return Err(Error::Encoding("group session is incomplete".into()));
        }
        let count = self
            .num_dc_groups
            .checked_add(self.group_count())
            .and_then(|n| n.checked_add(2))
            .ok_or(Error::Limit("TOC sections"))?;
        let mut sections = Vec::new();
        sections
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("group session TOC"))?;
        sections.push(
            self.global
                .ok_or_else(|| Error::Encoding("group session DCGlobal missing".into()))?,
        );
        sections.resize_with(self.num_dc_groups + 2, Vec::new);
        for bytes in self.groups {
            sections.push(
                bytes.ok_or_else(|| Error::Encoding("group session AC group missing".into()))?,
            );
        }
        Ok(sections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::stream::Coding;
    use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
    use crate::modular::encode::FixedTree;
    use crate::modular::encode::receipt::{ReceiptCollector, ReceiptRoute};
    use crate::modular::encode::receipt_cost::ModelCostReceipt;
    use crate::modular::predictor::Predictor;
    use crate::modular::tree::{Node, Tree};
    use crate::{PixelData, PixelLayout};

    const LIMIT: usize = 1 << 24;

    fn options() -> ModularOptions {
        ModularOptions {
            predictor: Predictor::Gradient,
            fixed_tree: Some(FixedTree::GradientLeaf),
            rct: None,
            lz77: false,
            entropy: Coding::Prefix,
            palette_policy: Some(PaletteOptions {
                groupwise: true,
                palette_colors: 0,
                channel_colors_percent: 0.0,
                channel_colors_pre_transform_percent: 0.0,
                ..PaletteOptions::default()
            }),
            ..ModularOptions::default()
        }
    }

    fn gray(width: usize, height: usize) -> Image {
        Image::new(
            width as u32,
            height as u32,
            PixelLayout::Gray,
            PixelData::U8((0..width * height).map(|i| (i * 17 + 9) as u8).collect()),
        )
        .unwrap()
    }

    fn rgba(bits: u8) -> Image {
        let mask = (1u32 << bits) - 1;
        let samples: Vec<u16> = (0..1025 * 3)
            .flat_map(|i| {
                [
                    ((i * 257 + 19) & mask) as u16,
                    ((i * 131 + 41) & mask) as u16,
                    ((i * 31 + 97) & mask) as u16,
                    if i % 3 == 0 { 0 } else { mask as u16 },
                ]
            })
            .collect();
        Image::new(
            1025,
            3,
            PixelLayout::Rgba,
            if bits == 8 {
                PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
            } else {
                PixelData::U16(samples)
            },
        )
        .unwrap()
    }

    // Byte identity here is primitive equivalence, not codec compatibility.
    // The parent exercises finished sections with the official pixel decoder.
    fn reference(image: &Image, options: &ModularOptions, shift: u8) -> Vec<Vec<u8>> {
        encode_layout_trial(
            image,
            options,
            shift,
            &Passes::default(),
            false,
            true,
            LIMIT,
            None,
            None,
        )
        .unwrap()
        .into_complete()
        .unwrap()
    }

    fn session<'a>(image: &'a Image, options: &ModularOptions, shift: u8) -> GroupSession<'a> {
        GroupSession::new(image, options, shift, &Passes::default(), false, LIMIT)
            .unwrap()
            .unwrap()
    }

    #[test]
    fn resumed_reverse_rgba8_rgba16_preserves_hidden_rgb_and_counts_once() {
        for bits in [8, 16] {
            let image = rgba(bits);
            let mut options = options();
            options.rct = Some(6);
            let expected_cost = Arc::new(ModelCostReceipt::default());
            options.model_cost_receipt = Some(Arc::clone(&expected_cost));
            let expected = reference(&image, &options, 0);
            let receipt = Arc::new(ReceiptCollector::default());
            let cost = Arc::new(ModelCostReceipt::default());
            options.representation_receipt = Some(Arc::clone(&receipt));
            options.model_cost_receipt = Some(Arc::clone(&cost));
            let mut session = session(&image, &options, 0);
            options.rct = None; // The session owns its resolved option snapshot.
            assert_eq!(session.local.rct, Some(6));
            assert_eq!(session.group_count(), 9);
            assert_eq!(session.completed_groups(), 0);
            assert_eq!(session.retained_bytes(), 0);
            assert_eq!(session.group_bytes(8), None);
            assert_eq!(session.group_bytes(usize::MAX), None);
            assert!(receipt.get().is_none());
            session.encode_groups(&[]).unwrap();
            session.encode_groups(&[8, 8, 6, 8]).unwrap();
            assert_eq!(session.completed_groups(), 2);
            assert_eq!(session.script.as_deref(), Some(&[Transform::Rct(6)][..]));
            assert_eq!(session.group_bytes(8), Some(expected[12].len()));
            assert_eq!(
                session.retained_bytes(),
                expected[0].len() + expected[12].len() + expected[10].len()
            );
            let first_receipt = receipt.get().unwrap();
            assert_eq!(first_receipt.route, ReceiptRoute::Groupwise);
            assert_eq!(
                (first_receipt.input_width, first_receipt.input_height),
                (1025, 3)
            );
            assert_eq!(first_receipt.input_channels, 4);
            let first_cost = cost.bits();
            let first_bytes = session.retained_bytes();
            session.encode_groups(&[6, 8, 6]).unwrap();
            assert_eq!(cost.bits(), first_cost);
            assert_eq!(session.retained_bytes(), first_bytes);
            assert_eq!(session.completed_groups(), 2);
            session.encode_groups(&[7, 6, 5, 4, 3]).unwrap();
            assert_eq!(session.completed_groups(), 6);
            session.encode_groups(&[2, 1, 0, 8, 0]).unwrap();
            assert_eq!(session.completed_groups(), 9);
            assert_eq!(
                session.retained_bytes(),
                expected.iter().map(Vec::len).sum::<usize>()
            );
            assert_eq!(cost.tree_bits(), expected_cost.tree_bits());
            assert_eq!(
                cost.entropy_header_bits(),
                expected_cost.entropy_header_bits()
            );
            session.complete_remaining().unwrap();
            assert_eq!(session.finish().unwrap(), expected);
            assert_eq!(cost.bits(), expected_cost.bits());
            // A second session must not reset or reuse this single-use receipt.
            assert!(
                GroupSession::new(&image, &options, 0, &Passes::default(), false, LIMIT).is_err()
            );
        }
    }

    #[test]
    fn shifts_zero_through_three_use_original_rectangles_and_wire_stream_ids() {
        for shift in 0..=3 {
            let dim = 128usize << shift;
            let mut shapes = vec![(8 * dim + 3, 3), (3, 8 * dim + 5)];
            if shift == 0 {
                shapes.push((2 * dim + 3, 2 * dim + 5));
            }
            for (width, height) in shapes {
                let image = gray(width, height);
                let mut options = options();
                let dc = width.div_ceil(8 * dim) * height.div_ceil(8 * dim);
                options.fixed_tree = None;
                options.custom_tree = Some(
                    Tree::new(vec![
                        Node::Split {
                            property: 1,
                            threshold: (18 + 3 * dc + 4) as i32,
                            greater: 1,
                            less_equal: 2,
                        },
                        Node::Leaf {
                            predictor: Predictor::Zero,
                            offset: 17,
                            multiplier: 1,
                        },
                        Node::Leaf {
                            predictor: Predictor::Gradient,
                            offset: 0,
                            multiplier: 1,
                        },
                    ])
                    .unwrap(),
                );
                // Frame-area policy must clear both global learner routing fields.
                options.tree_groups = Some(Default::default());
                options.tree_group_splits = vec![0, 1];
                let expected = reference(&image, &options, shift);
                let mut session = session(&image, &options, shift);
                assert!(session.local.entropy_options.streaming_mode);
                assert!(session.local.tree_groups.is_none());
                assert!(session.local.tree_group_splits.is_empty());
                assert!(options.tree_groups.is_some());
                assert_eq!(options.tree_group_splits, [0, 1]);
                assert_eq!(session.group_count(), 9);
                let mut pixels = 0;
                for index in 0..session.group_count() {
                    let rect = session.group_rect(index).unwrap();
                    assert_eq!(rect.x, index % width.div_ceil(dim) * dim);
                    assert_eq!(rect.y, index / width.div_ceil(dim) * dim);
                    assert_eq!(rect.width, dim.min(width - rect.x));
                    assert_eq!(rect.height, dim.min(height - rect.y));
                    pixels += rect.width * rect.height;
                    let (_, mut prepared) = session.prepare_group(index).unwrap();
                    assert_eq!(prepared.stream_id, 18 + 3 * dc + index);
                    for y in 0..rect.height {
                        for x in 0..rect.width {
                            assert_eq!(
                                prepared.planes[0].samples[y * rect.width + x],
                                image.modular_sample(0, (rect.y + y) * width + rect.x + x)
                            );
                        }
                    }
                    if index == 8 {
                        // The fixed tree deliberately distinguishes an erroneous
                        // crop-local ID from the real original-frame stream ID.
                        prepared.stream_id = 0;
                        let wrong = frame::encode_prepared_independent(
                            &prepared,
                            &session.local,
                            LIMIT,
                            false,
                            None,
                            image.bits(),
                        )
                        .unwrap();
                        assert_ne!(wrong, expected[2 + dc + index]);
                    }
                }
                assert_eq!(pixels, width * height);
                assert!(session.group_rect(9).is_err());
                let indices: Vec<_> = (0..session.group_count()).rev().collect();
                execution::serial(|| session.encode_groups(&indices[..4])).unwrap();
                session.encode_groups(&indices[4..]).unwrap();
                assert_eq!(session.finish().unwrap(), expected);
                assert!(expected[1..2 + dc].iter().all(Vec::is_empty));
            }
        }
    }

    #[test]
    fn full_resolution_extra_depth_controls_global_and_local_rct_headroom() {
        let image = rgba(8);
        let mut metadata = image.metadata().clone();
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 31).unwrap());
        let image = image
            .with_metadata(
                metadata,
                vec![PixelData::U32((0..1025 * 3).map(|i| i * 65537).collect())],
            )
            .unwrap();
        for candidates in [None, Some(3)] {
            let mut options = options();
            options.rct = Some(6);
            options.rct_candidates = candidates;
            let expected = reference(&image, &options, 0);
            let mut session = session(&image, &options, 0);
            let (global, prepared) = session.prepare_group(8).unwrap();
            assert!(global.is_empty());
            assert!(prepared.transforms.is_empty());
            assert_eq!(prepared.planes.len(), 5);
            session.encode_groups(&[8, 0]).unwrap();
            session.complete_remaining().unwrap();
            assert_eq!(session.finish().unwrap(), expected);
        }
    }

    #[test]
    fn local_palette_wp_and_rct_search_match_frame_area_preparation() {
        let image = Image::new(
            1025,
            3,
            PixelLayout::Rgba,
            PixelData::U8(
                (0..1025 * 3)
                    .flat_map(|i| {
                        if i % 2 == 0 {
                            [9, 41, 73, 0]
                        } else {
                            [83, 5, 101, 255]
                        }
                    })
                    .collect(),
            ),
        )
        .unwrap();
        for (speed_tier, palettes) in [(3, false), (3, true), (6, false), (6, true)] {
            let mut options = options();
            options.predictor = Predictor::Weighted;
            options.fixed_tree = Some(FixedTree::Weighted);
            options.weighted_candidates = Some(2);
            options.rct_candidates = Some(3);
            let policy = options.palette_policy.as_mut().unwrap();
            policy.speed_tier = speed_tier;
            if palettes {
                policy.palette_colors = 16;
                policy.force_palette = true;
                policy.channel_colors_percent = 100.0;
            }
            let expected = reference(&image, &options, 0);
            let mut session = session(&image, &options, 0);
            session.encode_groups(&[8, 2, 6]).unwrap();
            session.complete_remaining().unwrap();
            assert_eq!(session.finish().unwrap(), expected);
        }
    }

    #[test]
    fn unsupported_original_layouts_do_not_start_receipts_or_become_crop_encodes() {
        let image = gray(1025, 1);
        let small = gray(1024, 1);
        let mut options = options();
        let receipt = Arc::new(ReceiptCollector::default());
        options.representation_receipt = Some(Arc::clone(&receipt));
        assert!(
            GroupSession::new(&small, &options, 0, &Passes::default(), false, LIMIT)
                .unwrap()
                .is_none()
        );
        assert!(
            GroupSession::new(&image, &options, 1, &Passes::default(), false, LIMIT)
                .unwrap()
                .is_none()
        );
        assert!(
            GroupSession::new(&image, &options, 0, &Passes::progressive(), false, LIMIT)
                .unwrap()
                .is_none()
        );
        assert!(
            GroupSession::new(&image, &options, 0, &Passes::default(), true, LIMIT)
                .unwrap()
                .is_none()
        );
        options.squeeze.push(true);
        assert!(
            GroupSession::new(&image, &options, 0, &Passes::default(), false, LIMIT)
                .unwrap()
                .is_none()
        );
        options.squeeze.clear();
        options.palette_policy.as_mut().unwrap().responsive = true;
        assert!(
            GroupSession::new(&image, &options, 0, &Passes::default(), false, LIMIT)
                .unwrap()
                .is_none()
        );
        options.palette_policy.as_mut().unwrap().responsive = false;
        options.palette_policy.as_mut().unwrap().groupwise = false;
        assert!(
            GroupSession::new(&image, &options, 0, &Passes::default(), false, LIMIT)
                .unwrap()
                .is_none()
        );
        options.palette_policy.as_mut().unwrap().groupwise = true;
        for shift in [1, 3] {
            let mut metadata = image.metadata().clone();
            let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
            extra.dim_shift = shift;
            metadata.extra_channels.push(extra);
            let shifted = image
                .clone()
                .with_metadata(
                    metadata,
                    vec![PixelData::U8(vec![7; 1025usize.div_ceil(1usize << shift)])],
                )
                .unwrap();
            assert!(
                GroupSession::new(&shifted, &options, 0, &Passes::default(), false, LIMIT)
                    .unwrap()
                    .is_none()
            );
        }
        assert!(receipt.get().is_none());
        receipt.begin().unwrap();
        options.representation_receipt = None;
        assert!(GroupSession::new(&image, &options, 4, &Passes::default(), false, LIMIT).is_err());
        options.weighted_preset = 5;
        assert!(GroupSession::new(&image, &options, 0, &Passes::default(), false, LIMIT).is_err());
    }

    #[test]
    fn limits_include_global_and_prior_stages_and_errors_poison_the_session() {
        let image = gray(1025, 1);
        let options = options();
        let expected = reference(&image, &options, 0);
        let total: usize = expected.iter().map(Vec::len).sum();
        let ac_bytes: usize = expected[4..].iter().map(Vec::len).sum();
        assert!(!expected[0].is_empty());
        for limit in [0, 1, ac_bytes, total - 1] {
            let mut session =
                GroupSession::new(&image, &options, 0, &Passes::default(), false, limit)
                    .unwrap()
                    .unwrap();
            assert!(session.complete_remaining().is_err());
            assert_eq!(session.completed_groups(), 0);
            assert_eq!(session.retained_bytes(), 0);
            assert!(session.encode_groups(&[]).is_err());
            assert!(session.finish().is_err());
        }
        let mut exact = GroupSession::new(&image, &options, 0, &Passes::default(), false, total)
            .unwrap()
            .unwrap();
        exact.encode_groups(&[8, 0, 8]).unwrap();
        exact.complete_remaining().unwrap();
        assert_eq!(exact.retained_bytes(), total);
        assert_eq!(exact.finish().unwrap(), expected);

        let mut short =
            GroupSession::new(&image, &options, 0, &Passes::default(), false, total - 1)
                .unwrap()
                .unwrap();
        short.encode_groups(&[0]).unwrap();
        let retained = short.retained_bytes();
        assert!(short.complete_remaining().is_err());
        assert_eq!(short.completed_groups(), 1);
        assert_eq!(short.retained_bytes(), retained);
        assert!(short.finish().is_err());
    }

    #[test]
    fn incomplete_invalid_and_changed_script_sessions_cannot_finish() {
        let image = gray(1025, 1);
        let options = options();
        assert!(session(&image, &options, 0).finish().is_err());
        let mut partial = session(&image, &options, 0);
        partial.encode_groups(&[8]).unwrap();
        assert!(partial.finish().is_err());

        let mut invalid = session(&image, &options, 0);
        invalid.encode_groups(&[8]).unwrap();
        let retained = invalid.retained_bytes();
        assert!(invalid.encode_groups(&[0, usize::MAX]).is_err());
        assert_eq!(invalid.completed_groups(), 1);
        assert_eq!(invalid.retained_bytes(), retained);
        assert!(invalid.complete_remaining().is_err());
        assert!(invalid.finish().is_err());

        let mut completed = session(&image, &options, 0);
        completed.complete_remaining().unwrap();
        assert!(completed.encode_groups(&[9]).is_err());
        assert!(completed.finish().is_err());

        let mut changed = session(&image, &options, 0);
        changed.encode_groups(&[0]).unwrap();
        changed.script = Some(vec![Transform::Rct(6)]);
        assert!(changed.encode_groups(&[1]).is_err());
        assert_eq!(changed.completed_groups(), 1);
        assert!(changed.finish().is_err());
    }
}
