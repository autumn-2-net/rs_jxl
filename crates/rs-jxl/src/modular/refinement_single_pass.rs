//! Direct offset fitting: no baseline entropy model, LZ parse or serialization.
use super::{LiteralCache, Stats, Tree, encode, training};
use crate::Result;
use crate::entropy::options::UintConfigStrategy;
use std::time::Instant;

pub(super) fn fit(
    raw_cache: Option<&LiteralCache>,
    tree: &Tree,
    views: &[training::TrainingStream<'_>],
    options: &encode::ModularOptions,
    stats: &mut Stats,
) -> Result<Option<Tree>> {
    let _scope = crate::profile::scope("modular.offset_single_pass");
    let start = Instant::now();
    let offsets = options
        .refinement
        .offsets
        .as_ref()
        .expect("single-pass offsets");
    // Fixed mappings remain authoritative. Adaptive models have not been built
    // yet: use the source 420 proxy, and let the final model choose its mapping.
    let configs = match options.entropy_options.uint_config {
        UintConfigStrategy::Fixed(config) => Some(vec![config; tree.leaves()]),
        _ => None,
    };
    let cached = if let Some(raw) = raw_cache
        .and_then(|cache| std::cell::Ref::filter_map(cache.borrow(), |raw| raw.as_ref()).ok())
    {
        super::super::leaf_offset::fit_literals(
            tree,
            views,
            options.previous_channels,
            &raw,
            offsets,
            configs.as_deref(),
        )?
    } else {
        None
    };
    let (refined, fitted) = if let Some(fit) = cached {
        stats.offset_token_fits += 1;
        fit
    } else {
        super::super::leaf_offset::fit_with_configs(
            tree,
            views,
            options.previous_channels,
            offsets,
            configs.as_deref(),
        )?
    };
    stats.offset_single_pass_domains += 1;
    stats.offset_applied += usize::from(refined.is_some());
    stats.offset_proposals += usize::from(refined.is_some());
    stats.offset_seconds += start.elapsed().as_secs_f64();
    stats.offset_leaves += fitted.leaves;
    stats.offset_samples += fitted.samples;
    stats.offset_population += fitted.population;
    stats.offset_candidates += fitted.candidates;
    Ok(refined)
}
