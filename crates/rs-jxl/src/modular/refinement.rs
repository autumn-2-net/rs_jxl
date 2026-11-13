//! Opt-in post-training trials. Native implementation of fixed-topology leaf
//! refitting and entropy-mode comparison; no external codec dependency.
//!
//! Each entropy alternative reuses the same tokens, LZ parse, transforms and
//! tree. Distribution trials can fit and remap cached pre-LZ residual tokens;
//! changed residuals still run the LZ parser and entropy model builder again.
//! Other leaf proposals retokenize after learning. Training/preparation are
//! not repeated. Only actual serialized costs select a winner.

use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{encode, training, tree::Tree, weighted};
use crate::bit_writer::BitWriter;
use crate::entropy::stream::{Coding, Model, Token};
use crate::{Error, Result};

pub use super::leaf_offset::{OffsetMode, OffsetOptions};
pub use crate::entropy::cluster::neighborhood::ContextMapOptions;

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub entropy: bool,
    pub leaves: bool,
    /// Static offset fitting with frozen topology/predictors; default disabled.
    pub offsets: Option<OffsetOptions>,
    /// ANS merge proposals including context-map cost; default disabled.
    pub context_map: Option<ContextMapOptions>,
    /// Invocation-local diagnostics, aggregated outside pixel/token hot loops.
    pub receipt: Option<Arc<Receipt>>,
}

impl Options {
    pub(crate) fn enabled(&self) -> bool {
        self.entropy || self.leaves || self.offsets.is_some() || self.context_map.is_some()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub domains: usize,
    pub models: usize,
    pub refit_proposals: usize,
    pub refit_wins: usize,
    pub offset_proposals: usize,
    pub offset_wins: usize,
    pub offset_leaves: usize,
    pub offset_samples: usize,
    pub offset_population: usize,
    pub offset_candidates: usize,
    pub offset_seconds: f64,
    /// Distribution fits that avoided the predictor/property pixel walk.
    pub offset_token_fits: usize,
    /// Literal tokens reused for a proposed offset tree, without retokenizing.
    pub offset_reused_tokens: usize,
    /// Pre-LZ literals reused; candidates still run the original LZ parser.
    pub offset_raw_reused_tokens: usize,
    pub context_map_proposals: usize,
    pub context_map_wins: usize,
    pub context_map_proposed_merges: usize,
    pub context_map_selected_merges: usize,
    pub context_map_pairs: usize,
    pub context_map_map_trials: usize,
    pub context_map_skipped: usize,
    /// Proposal search (including map trials) and candidate model construction;
    /// excludes final candidate-output serialization, counted in write_seconds.
    pub context_map_seconds: f64,
    pub entropy_wins: usize,
    pub saved_bits: usize,
    pub refit_seconds: f64,
    pub model_seconds: f64,
    pub write_seconds: f64,
}

#[derive(Debug, Default)]
pub struct Receipt(Mutex<Stats>);

impl Receipt {
    pub fn snapshot(&self) -> Stats {
        self.0.lock().expect("refinement receipt poisoned").clone()
    }

    pub(super) fn record_raw_reuse(&self, tokens: usize) {
        self.0
            .lock()
            .expect("refinement receipt poisoned")
            .offset_raw_reused_tokens += tokens;
    }

    fn record(&self, stats: &Stats) {
        let mut total = self.0.lock().expect("refinement receipt poisoned");
        total.domains += stats.domains;
        total.models += stats.models;
        total.refit_proposals += stats.refit_proposals;
        total.refit_wins += stats.refit_wins;
        total.offset_proposals += stats.offset_proposals;
        total.offset_wins += stats.offset_wins;
        total.offset_leaves += stats.offset_leaves;
        total.offset_samples += stats.offset_samples;
        total.offset_population += stats.offset_population;
        total.offset_candidates += stats.offset_candidates;
        total.offset_seconds += stats.offset_seconds;
        total.offset_token_fits += stats.offset_token_fits;
        total.offset_reused_tokens += stats.offset_reused_tokens;
        total.offset_raw_reused_tokens += stats.offset_raw_reused_tokens;
        total.context_map_proposals += stats.context_map_proposals;
        total.context_map_wins += stats.context_map_wins;
        total.context_map_proposed_merges += stats.context_map_proposed_merges;
        total.context_map_selected_merges += stats.context_map_selected_merges;
        total.context_map_pairs += stats.context_map_pairs;
        total.context_map_map_trials += stats.context_map_map_trials;
        total.context_map_skipped += stats.context_map_skipped;
        total.context_map_seconds += stats.context_map_seconds;
        total.entropy_wins += stats.entropy_wins;
        total.saved_bits += stats.saved_bits;
        total.refit_seconds += stats.refit_seconds;
        total.model_seconds += stats.model_seconds;
        total.write_seconds += stats.write_seconds;
    }
}

struct Trial<T> {
    value: T,
    bits: usize,
    costs: Option<Arc<encode::receipt_cost::ModelCostReceipt>>,
    origin: Origin,
    alternate: bool,
    map_merges: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    Original,
    Predictors,
    Offsets,
}

pub(super) fn reuse_literals(tree: &Tree, options: &encode::ModularOptions) -> bool {
    options
        .refinement
        .offsets
        .as_ref()
        .is_some_and(|o| o.mode == OffsetMode::Distribution)
        && tree.nodes().iter().all(
            |node| !matches!(node, super::tree::Node::Leaf { multiplier, .. } if *multiplier != 1),
        )
}

pub(super) type LiteralCache = RefCell<Option<Vec<Vec<crate::tools::lz77::Literal>>>>;

pub(super) fn choose<T>(
    raw_cache: Option<&LiteralCache>,
    tree: &Tree,
    views: &[training::TrainingStream<'_>],
    options: &encode::ModularOptions,
    tokenize: impl Fn(&Tree) -> Result<(Vec<Vec<Token>>, bool)>,
    serialize: impl Fn(&Tree, &Model, &[Vec<Token>], &encode::ModularOptions) -> Result<(usize, T)>,
) -> Result<T> {
    let _scope = crate::profile::scope("modular.refinement");
    let mut stats = Stats {
        domains: 1,
        ..Default::default()
    };
    let mut best: Option<Trial<T>> = None;
    let mut original_bits = 0;
    let mut evaluate = |tree: &Tree,
                        origin: Origin,
                        reused: Option<Vec<Vec<Token>>>|
     -> Result<(
        Option<Vec<crate::entropy::HybridUintConfig>>,
        Option<Vec<Vec<Token>>>,
    )> {
        let (streams, lz77) = if let Some(streams) = reused {
            (streams, false)
        } else {
            tokenize(tree)?
        };
        let start = Instant::now();
        let base = Model::build_with_options(
            &streams,
            tree.leaves() + usize::from(lz77),
            lz77,
            options.entropy,
            &options.entropy_options,
        )?;
        stats.model_seconds += start.elapsed().as_secs_f64();
        let offset_configs = if origin == Origin::Original
            && options
                .refinement
                .offsets
                .as_ref()
                .is_some_and(|o| o.mode == OffsetMode::Distribution)
        {
            let mut configs = Vec::new();
            configs
                .try_reserve_exact(tree.leaves())
                .map_err(|_| Error::Limit("offset model configurations"))?;
            for context in 0..tree.leaves() {
                configs.push(base.context_uint_config(context).ok_or_else(|| {
                    Error::Encoding("missing offset context configuration".into())
                })?);
            }
            Some(configs)
        } else {
            None
        };
        let other_coding = if base.uses_ans() {
            Coding::Prefix
        } else {
            Coding::Ans
        };
        let mut write = |model: &Model, alternate: bool, map_merges: usize| -> Result<()> {
            let mut scratch = options.clone();
            // Scratch output must not charge the final model-cost receipt.
            scratch.model_cost_receipt = options
                .model_cost_receipt
                .as_ref()
                .map(|_| Arc::new(encode::receipt_cost::ModelCostReceipt::default()));
            let start = Instant::now();
            let result = serialize(tree, model, &streams, &scratch);
            stats.write_seconds += start.elapsed().as_secs_f64();
            let (bits, value) = match result {
                Ok(output) => output,
                // A successful baseline remains usable when a larger optional
                // output reaches the writer ceiling. Allocation/coding errors
                // are distinct and must not be hidden by this fallback.
                Err(Error::Limit("encoded output size")) if best.is_some() => return Ok(()),
                Err(error) => return Err(error),
            };
            stats.models += 1;
            if best.is_none() {
                original_bits = bits;
            }
            if best.as_ref().is_none_or(|old| bits < old.bits) {
                best = Some(Trial {
                    value,
                    bits,
                    costs: scratch.model_cost_receipt,
                    origin,
                    alternate,
                    map_merges,
                });
            }
            Ok(())
        };
        write(&base, false, 0)?;
        if let Some(search) = &options.refinement.context_map {
            let start = Instant::now();
            let candidates =
                base.context_map_candidates(&streams, &options.entropy_options, search)?;
            stats.context_map_seconds += start.elapsed().as_secs_f64();
            if let Some((models, work)) = candidates {
                stats.context_map_proposals += models.len();
                stats.context_map_proposed_merges += work.merges;
                stats.context_map_pairs += work.pairs;
                stats.context_map_map_trials += work.map_trials;
                for (model, merges) in models {
                    write(&model, false, merges)?;
                }
            } else {
                stats.context_map_skipped += 1;
            }
        }
        if options.refinement.entropy {
            let start = Instant::now();
            let alternate = Model::build_with_options(
                &streams,
                tree.leaves() + usize::from(lz77),
                lz77,
                other_coding,
                &options.entropy_options,
            );
            stats.model_seconds += start.elapsed().as_secs_f64();
            match alternate {
                Ok(model) => write(&model, true, 0)?,
                // An explicitly wide prefix alphabet need not fit ANS.
                Err(Error::InvalidInput(_)) if matches!(other_coding, Coding::Ans) => {}
                Err(error) => return Err(error),
            }
        }
        let reusable = offset_configs.is_some()
                && !options.lz77
                && !lz77
                && tree.nodes().iter().all(|node| {
                    !matches!(node, super::tree::Node::Leaf { multiplier, .. } if *multiplier != 1)
                })
                && streams.iter().flatten().all(|t| !t.is_length && t.context < tree.leaves());
        Ok((offset_configs, reusable.then_some(streams)))
    };
    let (offset_configs, mut offset_tokens) = evaluate(tree, Origin::Original, None)?;
    // Explicit fixed trees/predictor plans remain exact user instructions.
    if options.refinement.leaves
        && options.fixed_tree.is_none()
        && options.custom_tree.is_none()
        && options.fast_context.is_none()
    {
        let start = Instant::now();
        let allowed = if let Some(policy) = &options.sampling {
            training::source_predictors(options.predictor, policy)?
        } else {
            encode::predictors(options.predictor)
        };
        let refined = super::leaf_refit::refit(tree, views, &allowed, options.previous_channels)?;
        stats.refit_seconds += start.elapsed().as_secs_f64();
        if let Some(refined) = refined {
            stats.refit_proposals += 1;
            evaluate(&refined, Origin::Predictors, None)?;
        }
    }
    if let Some(offsets) = &options.refinement.offsets {
        let start = Instant::now();
        let cached_fit = if let Some(tokens) = offset_tokens.as_deref() {
            super::leaf_offset::fit_tokens(
                tree,
                views,
                options.previous_channels,
                tokens,
                offsets,
                offset_configs.as_deref(),
            )?
        } else if let Some(raw) = raw_cache
            .and_then(|cache| std::cell::Ref::filter_map(cache.borrow(), |raw| raw.as_ref()).ok())
        {
            super::leaf_offset::fit_literals(
                tree,
                views,
                options.previous_channels,
                &raw,
                offsets,
                offset_configs.as_deref(),
            )?
        } else {
            None
        };
        let (refined, fitted) = if let Some(fitted) = cached_fit {
            stats.offset_token_fits += 1;
            fitted
        } else if let Some(configs) = offset_configs.as_deref() {
            super::leaf_offset::fit_with_configs(
                tree,
                views,
                options.previous_channels,
                offsets,
                Some(configs),
            )?
        } else {
            super::leaf_offset::fit(tree, views, options.previous_channels, offsets)?
        };
        stats.offset_seconds += start.elapsed().as_secs_f64();
        stats.offset_leaves += fitted.leaves;
        stats.offset_samples += fitted.samples;
        stats.offset_population += fitted.population;
        stats.offset_candidates += fitted.candidates;
        if let Some(refined) = refined {
            stats.offset_proposals += 1;
            if let Some(tokens) = &mut offset_tokens {
                if let Some(count) = super::leaf_offset::remap_tokens(tree, &refined, tokens)? {
                    stats.offset_reused_tokens += count;
                } else {
                    offset_tokens = None;
                }
            }
            evaluate(&refined, Origin::Offsets, offset_tokens.take())?;
        }
    }
    let best = best.expect("baseline was encoded");
    stats.refit_wins = usize::from(best.origin == Origin::Predictors);
    stats.offset_wins = usize::from(best.origin == Origin::Offsets);
    stats.entropy_wins = usize::from(best.alternate);
    stats.context_map_wins = usize::from(best.map_merges != 0);
    stats.context_map_selected_merges = best.map_merges;
    stats.saved_bits = original_bits - best.bits;
    if let (Some(final_costs), Some(costs)) = (&options.model_cost_receipt, best.costs) {
        final_costs.record(costs.tree_bits(), costs.entropy_header_bits());
    }
    if let Some(receipt) = &options.refinement.receipt {
        receipt.record(&stats);
    }
    Ok(best.value)
}

pub(super) fn write_local(
    tree: &Tree,
    planes: &[super::group::Channel],
    wp: &weighted::Header,
    stream_id: usize,
    options: &encode::ModularOptions,
    writer: &mut BitWriter,
) -> Result<()> {
    let views = [training::TrainingStream {
        channels: planes,
        wp: *wp,
        stream_id,
    }];
    let cache_raw = options.lz77 && reuse_literals(tree, options);
    let raw_cache = LiteralCache::default();
    let raw_reused = Cell::new(0);
    let best = choose(
        Some(&raw_cache),
        tree,
        &views,
        options,
        |candidate| {
            let width = planes.iter().map(|c| c.width).max().unwrap_or(1);
            if cache_raw && std::ptr::eq(candidate, tree) {
                let raw = super::tokens::tokenize(
                    planes,
                    candidate,
                    wp,
                    options.previous_channels,
                    stream_id,
                    false,
                )?;
                let (tokens, lz77) =
                    encode::parse_group_tokens(&raw, candidate.leaves(), width, options)?;
                *raw_cache.borrow_mut() = Some(vec![raw]);
                return Ok((vec![tokens], lz77));
            }
            if let Some(raw) = raw_cache.borrow_mut().as_mut()
                && let Some(count) = super::leaf_offset::remap_literals(tree, candidate, raw)?
            {
                let (tokens, lz77) =
                    encode::parse_group_tokens(&raw[0], candidate.leaves(), width, options)?;
                raw_reused.set(raw_reused.get() + count);
                return Ok((vec![tokens], lz77));
            }
            let (tokens, lz77) = encode::group_stream(planes, candidate, options, wp, stream_id)?;
            Ok((vec![tokens], lz77))
        },
        |tree, model, streams, scratch| {
            let mut output = BitWriter::new(writer.max_bytes());
            encode::write_tree_and_model(tree, model, &mut output, scratch)?;
            model.write_stream(&streams[0], &mut output)?;
            Ok((output.bits_written(), output))
        },
    )?;
    if let Some(receipt) = &options.refinement.receipt {
        receipt.record_raw_reuse(raw_reused.get());
    }
    writer.append_unaligned(&best)?;
    Ok(())
}

#[cfg(test)]
#[path = "refinement_tests.rs"]
mod tests;
