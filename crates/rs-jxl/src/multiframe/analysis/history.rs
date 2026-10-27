//! Older logical states are observations, not automatically retained canvases.
use super::{DiffStatistics, statistics};
use crate::{Error, Image, Result, multiframe::diff::DiffMap};

#[derive(Clone, Debug)]
pub struct HistoryAnalysis {
    /// Logical input index. No codestream slot is claimed for this state.
    pub reference_index: usize,
    pub stats: DiffStatistics,
    /// Strictly smaller bbox than both first/previous. Always false if crop is
    /// unsupported, regardless of observed similarity. Not a byte-size promise.
    pub improves_static_bbox: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn analyze(
    images: &[Image],
    index: usize,
    history_window: usize,
    crop_supported: bool,
    pixels: u64,
    previous: Option<&DiffStatistics>,
    first: Option<&DiffStatistics>,
) -> Result<Vec<HistoryAnalysis>> {
    let mut history = Vec::new();
    if history_window == 0 || index < 3 {
        return Ok(history);
    }
    // Exclude previous (index - 1) and first (0), even for an unlimited window.
    let end = index - 1;
    let start = index.saturating_sub(history_window).max(1);
    if start >= end {
        return Ok(history);
    }
    history
        .try_reserve_exact(end - start)
        .map_err(|_| Error::Limit("analysis history reports"))?;
    let static_bbox = previous
        .into_iter()
        .chain(first)
        .map(|stats| stats.bbox_pixels)
        .min()
        .expect("noninitial frames have first and previous statistics");
    for reference_index in (start..end).rev() {
        // Only one additional bitmap lives at a time, regardless of window size.
        let stats = statistics(
            &DiffMap::between(&images[reference_index], &images[index])?,
            pixels,
        );
        let improves_static_bbox = crop_supported && stats.bbox_pixels < static_bbox;
        history.push(HistoryAnalysis {
            reference_index,
            stats,
            improves_static_bbox,
        });
    }
    Ok(history)
}
