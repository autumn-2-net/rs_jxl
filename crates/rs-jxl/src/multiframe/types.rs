//! Public automatic-search controls and reports; no encoding execution.
use super::{Rect, Reference};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct AutoOptions {
    /// SmallestCrop tests both first/previous references using actual bytes.
    pub reference: Reference,
    pub allow_mask: bool,
    pub allow_multi_rect: bool,
    pub tile_size: u32,
    /// Additional rectangle grids, tried after the original candidates. Empty
    /// by default. Equivalent rectangle plans are tried only once per source.
    pub extra_tile_sizes: Vec<u32>,
    pub max_rectangles: usize,
    /// Stop only when finalized output proves a trial cannot beat the winner.
    pub prune: bool,
    /// Add a cost-estimated rectangle proposal; retain all legacy candidates.
    pub adaptive_rectangles: bool,
    /// Retain a winning delta's background and rotate the other saved canvas.
    /// Opt-in, requires SmallestCrop. May evict frame zero; not a global optimum.
    pub retained_reference: bool,
    /// Use exact next-input geometry to preserve a useful old canvas. Opt-in;
    /// requires retained_reference and does not guarantee a global minimum.
    pub reference_lookahead: bool,
    /// Logical input frames, independent of inserted zero-duration layers.
    pub max_frames: usize,
    pub max_total_trials: usize,
    /// Sum of submitted rectangle areas, including payload-cache hits.
    pub max_trial_pixels: u64,
}
impl Default for AutoOptions {
    fn default() -> Self {
        Self {
            reference: Reference::SmallestCrop,
            allow_mask: true,
            allow_multi_rect: true,
            tile_size: 32,
            extra_tile_sizes: Vec::new(),
            max_rectangles: 32,
            prune: true,
            adaptive_rectangles: true,
            retained_reference: false,
            reference_lookahead: false,
            max_frames: usize::MAX,
            max_total_trials: usize::MAX,
            max_trial_pixels: u64::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Full,
    CropPrevious,
    CropFirst,
    MaskPrevious,
    MaskFirst,
    TilesPrevious,
    TilesFirst,
    MergedPrevious,
    MergedFirst,
    CropRetained,
    MaskRetained,
    TilesRetained,
    MergedRetained,
}

#[derive(Debug)]
pub struct AutoCandidateReport {
    pub strategy: Strategy,
    /// Grid that proposed a rectangle partition; None for full/crop/mask.
    pub tile_size: Option<u32>,
    pub reference_index: Option<usize>,
    pub source: u8,
    /// Final layer's save slot (0 on the final stream frame).
    pub save_as_reference: u8,
    /// Actual layer bytes, including frame headers and any private patch atlas.
    pub encoded_bytes: usize,
    /// Regular layers, excluding private reference-only patch frames.
    pub encoded_frames: usize,
    /// Submitted rectangle area including payload-cache hits, not analysis scans.
    pub trial_pixels: u64,
    pub elapsed: Duration,
}
#[derive(Debug)]
pub struct AutoPrunedReport {
    pub strategy: Strategy,
    pub tile_size: Option<u32>,
    pub reference_index: Option<usize>,
    pub source: u8,
    /// Intended final layer's save slot; pruned trials never change live slots.
    pub save_as_reference: u8,
    /// Proven size lower bound, not an estimated or complete encoded byte count.
    pub lower_bound: usize,
    pub encoded_frames: usize,
    pub planned_frames: usize,
    pub trial_pixels: u64,
    pub elapsed: Duration,
}
#[derive(Debug)]
pub struct AutoDiffReport {
    pub reference: usize,
    pub source: u8,
    pub changed_pixels: u64,
    pub bounds: Option<Rect>,
}
#[derive(Debug)]
pub struct AutoFrameReport {
    pub retained_reference: bool,
    pub reference_lookahead: bool,
    pub lookahead_comparisons: usize,
    pub reused_payloads: usize,
    pub reused_payload_pixels: u64,
    /// Complete input states in each slot; None also covers partial scratch state.
    pub references_before: [Option<usize>; 4],
    pub references_after: [Option<usize>; 4],
    pub winner_index: usize,
    pub candidates: Vec<AutoCandidateReport>,
    pub pruned: Vec<AutoPrunedReport>,
    pub differences: Vec<AutoDiffReport>,
    pub analysis_elapsed: Duration,
}
#[derive(Debug)]
pub struct AutoEncodeResult {
    pub bytes: Vec<u8>,
    /// One report per input, also when its duration is zero.
    pub frames: Vec<AutoFrameReport>,
}
