//! Opt-in representation diagnostics for one native Image encode invocation.
//!
//! This describes accepted transforms, not palette eligibility or encoded size.
//! The Image may itself be a crop: dimensions and input channel numbers always
//! refer to that Image, never an unseen parent image. No samples or palette
//! tables are retained, so this receipt is not a replay recipe or an identity
//! test for palette index assignments.

use std::ops::Range;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

use crate::modular::{group::Transform, predictor::Predictor};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptRoute {
    /// Actual full-Image preparation, including a declined groupwise request.
    FullImage,
    /// DC-area streaming; palettes are local, not global tuple palettes.
    Groupwise,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalTuplePalette {
    /// Half-open range in the input Image, before any metadata insertion.
    pub input_channels: Range<usize>,
    /// Accepted explicit absolute entries, not a unique-color estimate.
    pub colors: usize,
    /// Accepted explicit delta entries; implicit colors are not counted here.
    pub deltas: usize,
    pub predictor: Predictor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationReceipt {
    pub route: ReceiptRoute,
    pub input_width: u32,
    pub input_height: u32,
    pub input_channels: usize,
    /// Non-meta channels after global transforms, BEFORE partition/local work.
    /// Includes squeeze outputs and constant channels; not MA variable channels.
    pub spatial_channels: usize,
    /// Global metadata channels at the same boundary, including single palettes.
    pub meta_channels: usize,
    /// Only accepted pre-RCT global transforms with more than one input channel.
    /// Single-channel compaction and all local palettes are deliberately absent.
    pub global_tuple_palettes: Vec<GlobalTuplePalette>,
}

/// Single-use, caller-owned collector. Install a new Arc in each invocation's
/// ModularOptions, AFTER cloning shared base options for concurrent probes.
///
/// get() borrows the immutable snapshot without copying. None means no snapshot
/// was published: preparation is incomplete/failed or the entry point does not
/// support receipts. Publication follows layout preparation (the first area in
/// groupwise mode), before entropy encoding; bounded/failed outputs may have a
/// snapshot. Direct Modular group calls and region-input callbacks do not publish
/// receipts. Reuse by another Image layout invocation is an error, including
/// after failed preparation. Do not share across animation/reference frames.
#[derive(Debug, Default)]
pub struct ReceiptCollector {
    started: AtomicBool,
    receipt: OnceLock<RepresentationReceipt>,
}

impl ReceiptCollector {
    pub fn get(&self) -> Option<&RepresentationReceipt> {
        self.receipt.get()
    }

    pub(crate) fn begin(&self) -> Result<()> {
        self.started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                Error::InvalidInput(
                    "representation receipt requires a fresh collector per invocation".into(),
                )
            })?;
        Ok(())
    }

    pub(crate) fn publish(&self, receipt: RepresentationReceipt) -> Result<()> {
        debug_assert!(self.started.load(Ordering::Acquire));
        self.receipt
            .set(receipt)
            .map_err(|_| Error::InvalidInput("representation receipt published twice".into()))
    }
}

impl RepresentationReceipt {
    pub(crate) fn groupwise(width: u32, height: u32, input_channels: usize) -> Self {
        // The groupwise global stage only applies RCT, which preserves count.
        Self {
            route: ReceiptRoute::Groupwise,
            input_width: width,
            input_height: height,
            input_channels,
            spatial_channels: input_channels,
            meta_channels: 0,
            global_tuple_palettes: Vec::new(),
        }
    }

    pub(crate) fn full_image(
        width: u32,
        height: u32,
        input_channels: usize,
        palette_script: &[Transform],
        spatial_channels: usize,
        meta_channels: usize,
    ) -> Result<Self> {
        let mut receipt = Self {
            route: ReceiptRoute::FullImage,
            spatial_channels,
            meta_channels,
            ..Self::groupwise(width, height, input_channels)
        };
        if palette_script.is_empty() {
            return Ok(receipt);
        }
        // A palette inserts its table at physical channel zero and collapses
        // an adjacent spatial range. Track original channel coverage separately.
        let mut coverage: Vec<_> = (0..input_channels).map(|c| c..c + 1).collect();
        for (meta, transform) in palette_script.iter().enumerate() {
            let Transform::Palette {
                begin,
                channels,
                colors,
                deltas,
                predictor,
            } = transform
            else {
                return Err(Error::Encoding(
                    "non-palette in global palette receipt stage".into(),
                ));
            };
            let start = begin
                .checked_sub(meta)
                .ok_or_else(|| Error::Encoding("global palette receipt selects metadata".into()))?;
            let end = start
                .checked_add(*channels)
                .filter(|&end| *channels != 0 && end <= coverage.len())
                .ok_or_else(|| Error::Encoding("global palette receipt channel range".into()))?;
            let input_range = coverage[start].start..coverage[end - 1].end;
            if *channels > 1 {
                receipt.global_tuple_palettes.push(GlobalTuplePalette {
                    input_channels: input_range.clone(),
                    colors: *colors,
                    deltas: *deltas,
                    predictor: *predictor,
                });
            }
            coverage[start] = input_range;
            coverage.drain(start + 1..end);
        }
        Ok(receipt)
    }
}
