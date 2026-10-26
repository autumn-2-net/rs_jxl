//! Explicit experimental tool routing and probe estimates, never output bounds.
use super::*;
use crate::tools::lz77::{
    SourceBlockMode, SourceBucketSelection, SourceIndexMode, SourceIndexTags, SourceMatchCost,
};

pub(super) fn apply(options: &mut EncodeOptions, params: &CandidateParams) {
    options.modular.parallel_weighted = true;
    let parallel = &mut options.modular.source_ma_parallel;
    parallel.subtrees = true;
    parallel.balanced = true;
    parallel.small_subtrees = true;
    let lz = options
        .modular
        .source_lz77
        .get_or_insert_with(Default::default);
    let variable = params.predictor == Some(crate::modular::predictor::Predictor::Variable);
    if variable {
        options.modular.source_ma_histograms.events = true;
    }
    if variable
        || (params.predictor == Some(crate::modular::predictor::Predictor::Zero)
            && params.palette_colors > 0)
    {
        lz.bucket_selection = Some(SourceBucketSelection::ParallelWide);
        lz.match_cost = SourceMatchCost::Frontier;
        lz.bucket_dominance = true;
        lz.bucket_distances = true;
        lz.index_mode = SourceIndexMode::All;
        lz.index_tags = SourceIndexTags::Simd;
        lz.parallel_index_build = true;
        lz.index_compact_values = true;
        lz.index_word_match = true;
        lz.block_mode = SourceBlockMode::Intervals;
        lz.block_sparse_cache = true;
        lz.block_shared_frontier = true;
        lz.block_distance_bound = true;
        lz.block_distance_classes = true;
        lz.block_zero_tail = true;
        lz.block_literal_bound = true;
    }
}

fn adjusted_size(
    bytes: usize,
    model_bits: usize,
    sample_pixels: u64,
    full_pixels: u64,
) -> Result<usize> {
    let bits = bytes as u128 * 8;
    if model_bits as u128 > bits || full_pixels == 0 || sample_pixels > full_pixels {
        return Err(Error::Encoding("invalid probe model cost receipt".into()));
    }
    let adjusted = bits - model_bits as u128
        + (model_bits as u128 * u128::from(sample_pixels)).div_ceil(u128::from(full_pixels));
    usize::try_from(adjusted.div_ceil(8).max(1)).map_err(|_| Error::Limit("probe cost estimate"))
}

pub(super) fn probe_score(
    bytes: usize,
    sample: &Image,
    full_pixels: u64,
    options: &EncodeOptions,
    enabled: bool,
) -> Result<usize> {
    if !enabled {
        return Ok(bytes);
    }
    let model_bits = options
        .modular
        .model_cost_receipt
        .as_ref()
        .map_or(0, |c| c.bits());
    let sample_pixels = u64::from(sample.width()) * u64::from(sample.height());
    let score = adjusted_size(bytes, model_bits, sample_pixels, full_pixels)?;
    crate::profile::note(
        "scheduler.probe_model",
        format_args!("bytes={bytes} model_bits={model_bits} score={score} pixels={sample_pixels}"),
    );
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indexed_routes_cover_cross_channel_and_palette_but_not_auto() {
        use crate::modular::predictor::Predictor;
        for (predictor, palette_colors, indexed) in [
            (None, 1024, false),
            (Some(Predictor::Variable), 0, true),
            (Some(Predictor::Zero), 10000, true),
            (Some(Predictor::Zero), 0, false),
        ] {
            let mut options = EncodeOptions {
                threads: 3,
                max_pixels: 12345,
                ..Default::default()
            };
            let params = CandidateParams {
                predictor,
                palette_colors,
                ..Default::default()
            };
            apply(&mut options, &params);
            let lz = options.modular.source_lz77.as_ref().unwrap();
            assert_eq!(lz.index_mode == SourceIndexMode::All, indexed);
            assert_eq!(lz.parallel_index_build, indexed);
            assert_eq!(
                options.modular.source_ma_histograms.events,
                predictor == Some(Predictor::Variable)
            );
            assert_eq!(options.threads, 3);
            assert_eq!(options.max_pixels, 12345);
        }
    }

    #[test]
    fn amortization_only_changes_estimates_and_rounds_up() {
        assert_eq!(adjusted_size(100, 80, 10, 100).unwrap(), 91);
        assert_eq!(adjusted_size(100, 81, 10, 100).unwrap(), 91);
        assert_eq!(adjusted_size(100, 800, 10, 100).unwrap(), 10);
        assert_eq!(adjusted_size(100, 0, 10, 100).unwrap(), 100);
        assert_eq!(adjusted_size(100, 80, 100, 100).unwrap(), 100);
        assert!(adjusted_size(100, 801, 10, 100).is_err());
        assert!(adjusted_size(100, 80, 101, 100).is_err());
        assert!(adjusted_size(100, 80, 10, 0).is_err());
        assert_eq!(adjusted_size(usize::MAX, 0, 1, 1).unwrap(), usize::MAX);
    }
}
