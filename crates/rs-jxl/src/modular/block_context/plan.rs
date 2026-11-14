use super::{ChannelView, PREDICTORS, stats::Histogram};
use crate::modular::{
    predictor::{NeighborRow, Predictor, predict_one},
    tree::{Node, pack_signed},
};
use crate::{Error, Result};

type Histograms = [Histogram; 4];

pub(super) enum Plan {
    Leaf(Predictor),
    Split {
        property: usize,
        threshold: i32,
        greater: Box<Plan>,
        less_equal: Box<Plan>,
    },
}

impl Plan {
    pub(super) fn append(&self, nodes: &mut Vec<Node>) -> usize {
        let index = nodes.len();
        nodes.push(Node::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
        });
        nodes[index] = match self {
            Self::Leaf(predictor) => Node::Leaf {
                predictor: *predictor,
                offset: 0,
                multiplier: 1,
            },
            Self::Split {
                property,
                threshold,
                greater,
                less_equal,
            } => {
                let greater = greater.append(nodes);
                let less_equal = less_equal.append(nodes);
                Node::Split {
                    property: *property,
                    threshold: *threshold,
                    greater,
                    less_equal,
                }
            }
        };
        index
    }
}

pub(super) struct Candidate {
    histograms: Box<Histograms>,
    pub(super) plan: Plan,
    bits: f64,
    pub(super) coarsened_views: usize,
}

// Keep array initialization and its debug temporaries off recursive frames.
#[inline(never)]
fn empty_histograms() -> Box<Histograms> {
    Box::new(std::array::from_fn(|_| Histogram::default()))
}

fn best(histograms: &Histograms) -> (Predictor, f64) {
    let mut best = (PREDICTORS[0], f64::INFINITY);
    for (&predictor, histogram) in PREDICTORS.iter().zip(histograms) {
        let bits = histogram.bits(0);
        if bits < best.1 {
            best = (predictor, bits);
        }
    }
    best
}

fn leaf(histograms: Box<Histograms>, coarsened_views: usize) -> Candidate {
    let (predictor, bits) = best(&histograms);
    Candidate {
        histograms,
        plan: Plan::Leaf(predictor),
        bits,
        coarsened_views,
    }
}

fn combine(property: usize, threshold: i32, lower: Candidate, upper: Candidate) -> Candidate {
    let mut histograms = lower.histograms;
    for (to, from) in histograms.iter_mut().zip(upper.histograms.iter()) {
        to.merge(from);
    }
    let (predictor, single_bits) = best(&histograms);
    // Histograms include a per-leaf charge. This additional split-description
    // estimate includes threshold magnitude, but is not an actual wire size.
    let split_bits =
        lower.bits + upper.bits + 24.0 + f64::from(32 - pack_signed(threshold).leading_zeros());
    // The same predictor can still need different entropy contexts. Coalesce
    // only on rate, never merely because the two predictor IDs match.
    let (plan, bits) = if split_bits < single_bits {
        (
            Plan::Split {
                property,
                threshold,
                greater: Box::new(upper.plan),
                less_equal: Box::new(lower.plan),
            },
            split_bits,
        )
    } else {
        (Plan::Leaf(predictor), single_bits)
    };
    Candidate {
        histograms,
        plan,
        bits,
        coarsened_views: lower.coarsened_views + upper.coarsened_views,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_predictors_keep_profitable_separate_contexts() {
        let mut low = empty_histograms();
        let mut high = empty_histograms();
        for _ in 0..2048 {
            for histogram in low.iter_mut() {
                histogram.add(0, false);
            }
            for histogram in high.iter_mut() {
                histogram.add(128, false);
            }
        }
        let result = combine(3, 15, leaf(low.clone(), 0), leaf(high, 0));
        assert!(matches!(result.plan, Plan::Split { .. }));
        let same = combine(3, 15, leaf(low.clone(), 0), leaf(low, 0));
        assert!(matches!(same.plan, Plan::Leaf(Predictor::Zero)));
    }
}

#[derive(Clone, Copy)]
pub(super) struct Rect {
    pub(super) x0: usize,
    pub(super) y0: usize,
    pub(super) x1: usize,
    pub(super) y1: usize,
}

impl Rect {
    fn whole(view: &ChannelView<'_>) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: view.width,
            y1: view.height,
        }
    }

    fn tiles(self, tile: usize) -> (usize, usize) {
        (
            self.x1.div_ceil(tile) - self.x0 / tile,
            self.y1.div_ceil(tile) - self.y0 / tile,
        )
    }
}

/// Rectangles restrict scored samples, never the causal neighbor plane.
pub(super) fn gather(view: &ChannelView<'_>, rect: Rect, out: &mut Histograms) -> Result<()> {
    for y in rect.y0..rect.y1 {
        let row = NeighborRow::new(view.pixels, view.width, view.stride, y)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        for x in rect.x0..rect.x1 {
            let n = row.at(x).map_err(|e| Error::InvalidInput(e.to_string()))?;
            let value = i64::from(view.pixels[y * view.stride + x]);
            for (&predictor, histogram) in PREDICTORS.iter().zip(out.iter_mut()) {
                let prediction = predict_one(predictor, n, None)
                    .map_err(|e| Error::InvalidInput(e.to_string()))?;
                // Multiplier-one tokenization subtracts wide then narrows the
                // quotient for PackSigned, with no saturation or SAD proxy.
                histogram.add((value - prediction) as i32, false);
            }
        }
    }
    Ok(())
}

/// Divide candidate budget by population, retaining one leaf per side.
fn divide(limit: usize, lower: usize, total: usize) -> usize {
    ((limit as u128 * lower as u128 / total as u128) as usize).clamp(1, limit - 1)
}

#[inline(never)]
fn spatial_leaf(view: &ChannelView<'_>, rect: Rect, coarsened: bool) -> Result<Candidate> {
    let mut histograms = empty_histograms();
    gather(view, rect, &mut histograms)?;
    Ok(leaf(histograms, usize::from(coarsened)))
}

#[inline(never)]
fn domain_leaf(views: &[&ChannelView<'_>]) -> Result<Candidate> {
    let mut histograms = empty_histograms();
    for view in views {
        gather(view, Rect::whole(view), &mut histograms)?;
    }
    Ok(leaf(histograms, views.len()))
}

fn spatial(view: &ChannelView<'_>, rect: Rect, limit: usize, tile: usize) -> Result<Candidate> {
    let (nx, ny) = rect.tiles(tile);
    // x/y properties are i32 casts. Disable a wrapped axis to coarsen the plan
    // without rejecting legal input or emitting incorrect coordinate ranges.
    let split_x = nx > 1 && view.width <= i32::MAX as usize + 1;
    let split_y = ny > 1 && view.height <= i32::MAX as usize + 1;
    if limit == 1 || !(split_x || split_y) {
        return spatial_leaf(view, rect, nx > 1 || ny > 1);
    }
    let horizontal = split_x && (!split_y || nx >= ny);
    let mut lower = rect;
    let mut upper = rect;
    let (property, cut, before, total) = if horizontal {
        let before = nx / 2;
        let cut = (rect.x0 / tile + before) * tile;
        lower.x1 = cut;
        upper.x0 = cut;
        (3, cut, before, nx)
    } else {
        let before = ny / 2;
        let cut = (rect.y0 / tile + before) * tile;
        lower.y1 = cut;
        upper.y0 = cut;
        (2, cut, before, ny)
    };
    let lower_limit = divide(limit, before, total);
    let work = (rect.x1 - rect.x0) * (rect.y1 - rect.y0);
    let (lower, upper) = fork(
        work,
        || spatial(view, lower, lower_limit, tile),
        || spatial(view, upper, limit - lower_limit, tile),
    );
    let (lower, upper) = (lower?, upper?);
    let coarsened = usize::from(lower.coarsened_views != 0 || upper.coarsened_views != 0);
    let mut result = combine(property, (cut - 1) as i32, lower, upper);
    result.coarsened_views = coarsened;
    Ok(result)
}

/// Views are ordered by original (stream, channel), never dense relabelings.
/// Boxed histograms live with active candidates, not once per input tile.
pub(super) fn domain(views: &[&ChannelView<'_>], limit: usize, tile: usize) -> Result<Candidate> {
    if views.len() == 1 {
        return spatial(views[0], Rect::whole(views[0]), limit, tile);
    }
    if views.is_empty() || limit == 1 {
        return domain_leaf(views);
    }
    let groups = 1 + views
        .windows(2)
        .filter(|v| v[0].group != v[1].group)
        .count();
    let (property, mid, threshold) = if groups > 1 {
        let target = groups / 2;
        let mut found = 0;
        let mid = (1..views.len())
            .find(|&i| {
                found += usize::from(views[i - 1].group != views[i].group);
                found == target
            })
            .expect("nonempty stream partition");
        (1, mid, (views[mid].group - 1) as i32)
    } else {
        let mid = views.len() / 2;
        (0, mid, (views[mid].channel - 1) as i32)
    };
    let lower_limit = divide(limit, mid, views.len());
    // Population was checked at the public learning boundary. Keep large
    // independent subplans in the caller's bounded pool, not one task per tile.
    let work = views.iter().map(|v| v.width * v.height).sum();
    let (lower, upper) = fork(
        work,
        || domain(&views[..mid], lower_limit, tile),
        || domain(&views[mid..], limit - lower_limit, tile),
    );
    Ok(combine(property, threshold, lower?, upper?))
}

fn fork(
    pixels: usize,
    lower: impl FnOnce() -> Result<Candidate> + Send,
    upper: impl FnOnce() -> Result<Candidate> + Send,
) -> (Result<Candidate>, Result<Candidate>) {
    if pixels >= 65_536 && crate::execution::worker_count() > 1 {
        crate::execution::join(lower, upper)
    } else {
        (lower(), upper())
    }
}

#[cfg(test)]
#[path = "plan/stack_tests.rs"]
mod stack_tests;
