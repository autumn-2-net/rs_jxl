//! Bounded reuse of literal streams, including cached input to LZ parsing.
//! Parsed LZ commands are never used to fit or rewrite residuals.

use crate::entropy::{HybridUintConfig, stream::Token};
use crate::tools::lz77::Literal;
use crate::{Error, Result};

use super::{FitStats, OffsetOptions, collect_leaves, fit_distribution};
use crate::modular::{
    leaf_refit::ordered_streams,
    predictor::Predictor,
    training::TrainingStream,
    tree::{ChannelTree, Node, Tree, pack_signed},
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
    fit_values(
        tree,
        views,
        previous_channels,
        streams,
        options,
        configs,
        |t| (!t.is_length).then_some((t.context, t.value)),
    )
}

pub(in crate::modular) fn fit_literals(
    tree: &Tree,
    views: &[TrainingStream<'_>],
    previous_channels: usize,
    streams: &[Vec<Literal>],
    options: &OffsetOptions,
    configs: Option<&[HybridUintConfig]>,
) -> Result<Option<(Option<Tree>, FitStats)>> {
    fit_values(
        tree,
        views,
        previous_channels,
        streams,
        options,
        configs,
        |t| Some((t.context as usize, t.value)),
    )
}

fn fit_values<T>(
    tree: &Tree,
    views: &[TrainingStream<'_>],
    previous_channels: usize,
    streams: &[Vec<T>],
    options: &OffsetOptions,
    configs: Option<&[HybridUintConfig]>,
    literal: impl Fn(&T) -> Option<(usize, u32)>,
) -> Result<Option<(Option<Tree>, FitStats)>> {
    options.validate()?;
    if configs.is_some_and(|configs| configs.len() != tree.leaves()) {
        return Err(Error::InvalidInput(
            "offset context configuration count".into(),
        ));
    }
    ordered_streams(tree, views, previous_channels)?;
    if !options.mode.is_distribution()
        || ChannelTree::offset_sensitive_routing(tree)
        || tree
            .nodes()
            .iter()
            .any(|node| matches!(node, Node::Leaf { multiplier, .. } if *multiplier != 1))
    {
        return Ok(None);
    }
    // All predictors except Average4 return an i32 value (including rounded
    // Weighted). Average4 is bounded by (20*B + 8)/16; prove its range before
    // recovering the signed predictor from the original sample and packed token.
    const B: i32 = (1 << 28) - 1;
    let uses_average4 = tree.nodes().iter().any(|node| {
        matches!(
            node,
            Node::Leaf {
                predictor: Predictor::Average4,
                ..
            }
        )
    });
    if uses_average4
        && views.iter().flat_map(|s| s.channels).any(|channel| {
            channel
                .samples
                .iter()
                .any(|&value| !(-B..=B).contains(&value))
        })
    {
        return Ok(None);
    }
    let nonempty_views = views
        .iter()
        .filter(|s| s.channels.iter().any(|c| !c.samples.is_empty()));
    let mut nonempty_tokens = streams.iter().filter(|s| !s.is_empty());
    // Frame streams can include an empty global entry. Associate before sorting
    // so reservoir updates still follow the pixel fitter's stream-ID order.
    let mut ordered = Vec::new();
    for view in nonempty_views {
        let Some(tokens) = nonempty_tokens.next() else {
            return Ok(None);
        };
        ordered.push((view, tokens));
    }
    if nonempty_tokens.next().is_some() {
        return Ok(None);
    }
    for (view, tokens) in &ordered {
        let count = view.channels.iter().try_fold(0usize, |sum, channel| {
            sum.checked_add(channel.samples.len())
                .ok_or(Error::Limit("offset population overflow"))
        })?;
        if tokens.len() != count {
            return Ok(None);
        }
    }
    ordered.sort_unstable_by_key(|(view, _)| view.stream_id);
    let (mut leaves, counts) = collect_leaves(tree, options)?;
    for (view, tokens) in ordered {
        let pixels = view
            .channels
            .iter()
            .flat_map(|channel| channel.samples.iter());
        for (token, &pixel) in tokens.iter().zip(pixels) {
            // Validate while collecting, without another full literal scan.
            // Partial samples are local and discarded if reuse is unsafe.
            let Some((context, value)) = literal(token) else {
                return Ok(None);
            };
            let Some(leaf) = leaves.get_mut(context).and_then(Option::as_mut) else {
                return Ok(None);
            };
            let residual = recover_residual(value, leaf.baseline, pixel);
            leaf.add(residual, options.samples_per_leaf)?;
        }
    }
    fit_distribution(tree, &mut leaves, counts, configs).map(Some)
}

fn recover_residual(packed: u32, offset: i32, pixel: i32) -> i64 {
    // The token stores (pixel - prediction - offset) modulo 2^32. Prediction
    // is known to fit i32, so recover it modulo 2^32, then subtract in i64.
    // Directly sign-extending the unpacked residual loses +/-2^32 at extremes.
    let prediction = pixel.wrapping_sub(unpack(packed).wrapping_add(offset));
    i64::from(pixel) - i64::from(prediction)
}

/// Only an offset-only, unit-multiplier tree replacement can use this mapping.
/// Wide source residuals are safe here: subtraction commutes with narrowing.
pub(in crate::modular) fn remap_tokens(
    before: &Tree,
    after: &Tree,
    streams: &mut [Vec<Token>],
) -> Result<Option<usize>> {
    let invalid = || Error::Encoding("invalid offset token reuse".into());
    let Some(deltas) = offset_deltas(before, after)? else {
        return Ok(None);
    };
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
    Ok(Some(count))
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
    if ChannelTree::offset_sensitive_routing(before) || ChannelTree::offset_sensitive_routing(after)
    {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_recovery_preserves_wide_difference_at_every_signed_extreme() {
        let values = [
            i32::MIN,
            i32::MIN + 1,
            -37,
            -1,
            0,
            1,
            37,
            i32::MAX - 1,
            i32::MAX,
        ];
        for pixel in values {
            for prediction in values {
                let expected = i64::from(pixel) - i64::from(prediction);
                for offset in values {
                    let packed = pack_signed((expected - i64::from(offset)) as i32);
                    assert_eq!(recover_residual(packed, offset, pixel), expected);
                }
            }
        }
    }
}
