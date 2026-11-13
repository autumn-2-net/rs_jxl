//! Bounded reuse of literal-only streams. LZ commands are never rewritten.

use crate::entropy::{HybridUintConfig, stream::Token};
use crate::tools::lz77::Literal;
use crate::{Error, Result};

use super::{FitStats, OffsetMode, OffsetOptions, collect_leaves, fit_distribution};
use crate::modular::{
    leaf_refit::ordered_streams,
    training::TrainingStream,
    tree::{Node, Tree, pack_signed},
};

fn unpack(value: u32) -> i32 {
    ((value >> 1) as i32) ^ -((value & 1) as i32)
}

/// None requests the original pixel fitter, not a different proposal model.
pub(in crate::modular) fn fit_tokens(
    tree: &Tree,
    views: &[TrainingStream<'_>],
    previous_channels: usize,
    streams: &[Vec<Token>],
    options: &OffsetOptions,
    configs: Option<&[HybridUintConfig]>,
) -> Result<Option<(Option<Tree>, FitStats)>> {
    options.validate()?;
    if configs.is_some_and(|configs| configs.len() != tree.leaves()) {
        return Err(Error::InvalidInput(
            "offset context configuration count".into(),
        ));
    }
    ordered_streams(tree, views, previous_channels)?;
    if options.mode != OffsetMode::Distribution
        || tree.nodes().iter().any(|node| {
            matches!(node, Node::Leaf { predictor, multiplier, .. }
                if *multiplier != 1 || predictor.has_weighted())
        })
    {
        return Ok(None);
    }
    // All stateless predictions are bounded by 20*B/16 + 1 (Average4).
    // With |sample| <= B, the original pre-offset residual fits i32. Without
    // this proof a packed literal could alias a wide residual by 2^32, changing
    // the full-population mean and reservoir median. Keep the pixel fitter then.
    const B: i32 = (1 << 28) - 1;
    if views.iter().flat_map(|s| s.channels).any(|channel| {
        channel
            .samples
            .iter()
            .any(|&value| !(-B..=B).contains(&value))
    }) {
        return Ok(None);
    }
    let nonempty_views: Vec<_> = views
        .iter()
        .filter(|s| s.channels.iter().any(|c| !c.samples.is_empty()))
        .collect();
    let nonempty_tokens: Vec<_> = streams.iter().filter(|s| !s.is_empty()).collect();
    if nonempty_views.len() != nonempty_tokens.len() {
        return Ok(None);
    }
    // Frame streams can include an empty global entry. Associate before sorting
    // so reservoir updates still follow the pixel fitter's stream-ID order.
    let mut ordered: Vec<_> = nonempty_views.into_iter().zip(nonempty_tokens).collect();
    for (view, tokens) in &ordered {
        let count = view.channels.iter().try_fold(0usize, |sum, channel| {
            sum.checked_add(channel.samples.len())
                .ok_or(Error::Limit("offset population overflow"))
        })?;
        if tokens.len() != count
            || tokens
                .iter()
                .any(|t| t.is_length || t.context >= tree.leaves())
        {
            return Ok(None);
        }
    }
    ordered.sort_unstable_by_key(|(view, _)| view.stream_id);
    let (mut leaves, counts) = collect_leaves(tree, options)?;
    for (_, tokens) in ordered {
        for token in tokens {
            let leaf = leaves[token.context]
                .as_mut()
                .expect("unit multiplier checked");
            let residual = unpack(token.value).wrapping_add(leaf.baseline);
            leaf.add(i64::from(residual), options.samples_per_leaf)?;
        }
    }
    fit_distribution(tree, &mut leaves, counts, configs).map(Some)
}

/// Only an offset-only, unit-multiplier tree replacement can use this mapping.
/// Wide source residuals are safe here: subtraction commutes with narrowing.
pub(in crate::modular) fn remap_tokens(
    before: &Tree,
    after: &Tree,
    streams: &mut [Vec<Token>],
) -> Result<usize> {
    let invalid = || Error::Encoding("invalid offset token reuse".into());
    let deltas = offset_deltas(before, after)?.ok_or_else(invalid)?;
    let mut count = 0usize;
    for token in streams.iter_mut().flatten() {
        if token.is_length {
            return Err(invalid());
        }
        let delta = *deltas.get(token.context).ok_or_else(invalid)?;
        token.value = pack_signed(unpack(token.value).wrapping_add(delta));
        count = count
            .checked_add(1)
            .ok_or(Error::Limit("offset population overflow"))?;
    }
    Ok(count)
}

pub(in crate::modular) fn remap_literals(
    before: &Tree,
    after: &Tree,
    streams: &mut [Vec<Literal>],
) -> Result<Option<usize>> {
    let Some(deltas) = offset_deltas(before, after)? else {
        return Ok(None);
    };
    let mut count = 0usize;
    for token in streams.iter_mut().flatten() {
        let delta = *deltas
            .get(token.context as usize)
            .ok_or_else(|| Error::Encoding("invalid offset literal context".into()))?;
        token.value = pack_signed(unpack(token.value).wrapping_add(delta));
        count = count
            .checked_add(1)
            .ok_or(Error::Limit("offset population overflow"))?;
    }
    Ok(Some(count))
}

fn offset_deltas(before: &Tree, after: &Tree) -> Result<Option<Vec<i32>>> {
    if before.nodes().len() != after.nodes().len() {
        return Ok(None);
    }
    for (old, new) in before.nodes().iter().zip(after.nodes()) {
        match (old, new) {
            (
                Node::Split {
                    property: a,
                    threshold: b,
                    greater: c,
                    less_equal: d,
                },
                Node::Split {
                    property: e,
                    threshold: f,
                    greater: g,
                    less_equal: h,
                },
            ) if (a, b, c, d) == (e, f, g, h) => {}
            (
                Node::Leaf {
                    predictor: a,
                    multiplier: 1,
                    ..
                },
                Node::Leaf {
                    predictor: b,
                    multiplier: 1,
                    ..
                },
            ) if a == b => {}
            _ => return Ok(None),
        }
    }
    let mut deltas = Vec::new();
    deltas
        .try_reserve_exact(before.leaves())
        .map_err(|_| Error::Limit("offset token reuse allocation"))?;
    let mut queue = std::collections::VecDeque::from([0]);
    while let Some(node) = queue.pop_front() {
        match before.nodes()[node] {
            Node::Split {
                greater,
                less_equal,
                ..
            } => {
                queue.push_back(greater);
                queue.push_back(less_equal);
            }
            Node::Leaf { offset: old, .. } => {
                let Node::Leaf { offset: new, .. } = after.nodes()[node] else {
                    return Ok(None);
                };
                deltas.push(old.wrapping_sub(new));
            }
        }
    }
    Ok(Some(deltas))
}
