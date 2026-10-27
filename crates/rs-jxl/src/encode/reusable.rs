//! Candidate-local reuse of complete Modular sections, never reference state.
use super::{
    EncodeOptions, FrameCanvas, assemble_frame_sections,
    budget::{self, SectionBudget, TrialEncode, complete_or_return},
    encode_frame_sections_trial, encode_logical_frame_from_plan, prepare_patch_plan,
    validate_logical_frame,
};
use crate::{Error, Image, Result, frame::Frame};

/// One payload, scoped to one logical input and fixed sample/encoding metadata.
/// A key must identify identical samples and dimensions in every color and extra
/// channel. Frame source/save slots, origin, timing and `last` may vary. Encoding
/// options must stay fixed; resource limits may change and are checked per call.
/// The caller still validates references and commits only the winning state.
pub(crate) struct LogicalFrameCache<K> {
    entry: Option<(K, Vec<Vec<u8>>)>,
}

impl<K> Default for LogicalFrameCache<K> {
    fn default() -> Self {
        Self { entry: None }
    }
}

impl<K: Eq> LogicalFrameCache<K> {
    pub(crate) fn contains(&self, key: &K) -> bool {
        self.entry.as_ref().is_some_and(|(cached, _)| cached == key)
    }
}

/// Reuse regular payloads when patch preparation produces no atlas. A cache hit
/// also reuses that no-plan result under the fixed-samples/options contract.
/// Actual atlases and reference-only frames clear the cache and are not reused.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_logical_frame_cached_trial<K: Eq>(
    canvas: &Image,
    frame: &Frame<'_>,
    options: &EncodeOptions,
    last: bool,
    budget: Option<&SectionBudget>,
    cache: &mut LogicalFrameCache<K>,
    key: K,
) -> Result<TrialEncode<Vec<u8>>> {
    validate_logical_frame(FrameCanvas::Image(canvas), frame, options, last)?;
    if frame.options.reference_only {
        cache.entry = None;
        let plan = prepare_patch_plan(frame.image, options)?;
        return encode_logical_frame_from_plan(
            FrameCanvas::Image(canvas),
            frame,
            options,
            last,
            budget,
            plan,
        );
    }
    if let Some((cached, sections)) = &cache.entry
        && *cached == key
    {
        let _frame = crate::profile::scope("encode.frame");
        let _hit = crate::profile::scope("encode.frame_cache_hit");
        // Rebuild the variable-length header/TOC. This is the sole payload copy,
        // into the returned frame; the cached sections themselves are borrowed.
        let output = assemble_frame_sections(
            FrameCanvas::Image(canvas),
            frame.image,
            options,
            &frame.options,
            last,
            None,
            sections,
        )?;
        // A complete cached frame is exact finalized evidence, including its
        // new header. A fresh candidate budget must not inherit the miss tally.
        return budget::section(budget, || Ok(output));
    }

    // Drop the previous entry before encoding to keep retained payloads bounded.
    cache.entry = None;
    let plan = prepare_patch_plan(frame.image, options)?;
    if plan.is_some() {
        return encode_logical_frame_from_plan(
            FrameCanvas::Image(canvas),
            frame,
            options,
            last,
            budget,
            plan,
        );
    }
    let _frame = crate::profile::scope("encode.frame");
    let credited_before = budget.map_or(0, SectionBudget::section_bytes);
    let sections = complete_or_return!(encode_frame_sections_trial(
        frame.image,
        options,
        &frame.options,
        None,
        budget,
    ));
    let output = assemble_frame_sections(
        FrameCanvas::Image(canvas),
        frame.image,
        options,
        &frame.options,
        last,
        None,
        &sections,
    )?;
    if let Some(budget) = budget {
        let credited = budget.section_bytes() - credited_before;
        let overhead = output.len().checked_sub(credited).ok_or_else(|| {
            Error::Encoding("finalized sections exceed the assembled frame".into())
        })?;
        budget.retain_overhead(overhead)?;
    }
    cache.entry = Some((key, sections));
    Ok(TrialEncode::Complete(output))
}

#[cfg(test)]
#[path = "reusable_tests.rs"]
mod tests;
