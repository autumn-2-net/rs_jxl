//! Cached low-effort trees need three routing properties, not a full MA record.

use super::{Channel, ChannelTree, Literal, Predictor, Result, pack_signed};
use crate::modular::predictor::clamped_gradient;

pub(super) fn emit(
    plane: &Channel,
    plan: &ChannelTree,
    history: Option<&[[i32; 2]]>,
    mut emit: impl FnMut(Literal),
) -> Result<bool> {
    let Some(history) = history else {
        return Ok(false);
    };
    let Some(contexts) = plan.pair_contexts() else {
        return Ok(false);
    };
    let _scope = crate::profile::scope("modular.tokenize_pair");
    let width = plane.width;
    if width == 0
        || width.checked_mul(plane.height) != Some(plane.samples.len())
        || history.len() != plane.samples.len()
    {
        return Err(crate::Error::Encoding(
            "invalid pair-token plane/history dimensions".into(),
        ));
    }
    for (y, (current, predictions)) in plane
        .samples
        .chunks_exact(width)
        .zip(history.chunks_exact(width))
        .enumerate()
    {
        let above = if y == 0 {
            current
        } else {
            &plane.samples[(y - 1) * width..y * width]
        };
        for x in 0..width {
            let left = if x > 0 {
                current[x - 1]
            } else if y > 0 {
                above[0]
            } else {
                0
            };
            let top = if y == 0 { left } else { above[x] };
            let northwest = if x > 0 && y > 0 { above[x - 1] } else { left };
            let [weighted, error] = predictions[x];
            let gradient = if contexts.gradient_lut {
                (i64::from(left) + i64::from(top) - i64::from(northwest)).clamp(-8192, 8191) as i32
            } else {
                left.wrapping_add(top).wrapping_sub(northwest)
            };
            let leaf = contexts.lookup(gradient, left.wrapping_sub(northwest), error);
            let prediction = if leaf.predictor == Predictor::Weighted {
                weighted
            } else {
                clamped_gradient(left, top, northwest)
            };
            emit(Literal {
                context: leaf.context as u32,
                value: pack_signed(current[x].wrapping_sub(prediction)),
            });
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "tokens_pair_tests.rs"]
mod tests;
