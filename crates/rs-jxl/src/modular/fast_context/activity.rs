//! Independently implemented static joint-gradient contexts. C29 motivates
//! activity-conditioned statistics, not this partition or any decoder change.
//! A single SIMD-assisted scan feeds a fixed grid. Bottom-up rectangle costs
//! choose axis order and merge partitions; pixels are never sorted or rescanned.

use super::{ChannelView, Histogram, Leaf, Mode, Node, Options, Ordering, PREDICTORS, Tree, rows};
use crate::{Error, Result};

type Histograms = [Histogram; 4];
const COARSE: &[i32] = &[-5, -1, 0, 4];
const FINE: &[i32] = &[-65, -17, -5, -1, 0, 4, 16, 64];
// Same conservative estimate used by the existing fast context learner.
const SPLIT_BITS: f64 = 24.0;

fn best_leaf(hist: &Histograms) -> Leaf {
    let mut best = Leaf {
        predictor: PREDICTORS[0],
        offset: 0,
        bits: f64::INFINITY,
    };
    for (hist, &predictor) in hist.iter().zip(&PREDICTORS) {
        let bits = hist.bits(0);
        if bits < best.bits {
            best = Leaf {
                predictor,
                offset: 0,
                bits,
            };
        }
    }
    best
}

fn merged(a: &Histograms, b: &Histograms) -> Histograms {
    let mut result = a.clone();
    for (to, from) in result.iter_mut().zip(b) {
        to.merge(from);
    }
    result
}

#[derive(Clone, Copy)]
struct Range {
    start: usize,
    children: Option<(usize, usize)>,
}

fn ranges(n: usize) -> Vec<Range> {
    fn append(out: &mut Vec<Range>, start: usize, end: usize) -> usize {
        let id = out.len();
        out.push(Range {
            start,
            children: None,
        });
        if end - start > 1 {
            let mid = (start + end) / 2;
            let lo = append(out, start, mid);
            let hi = append(out, mid, end);
            out[id].children = Some((lo, hi));
        }
        id
    }
    let mut out = Vec::new();
    append(&mut out, 0, n);
    out
}

#[derive(Clone, Copy)]
enum Choice {
    Leaf,
    Vertical,
    Horizontal,
}

struct State {
    hist: Histograms,
    leaf: Leaf,
    bits: f64,
    choice: Choice,
}

struct Grid {
    cells: Vec<Histograms>,
    ranges: Vec<Range>,
    states: Vec<Option<State>>,
    bins: usize,
}

impl Grid {
    fn new(cells: Vec<Histograms>, bins: usize) -> Self {
        let ranges = ranges(bins);
        let states = (0..ranges.len() * ranges.len()).map(|_| None).collect();
        Self {
            cells,
            ranges,
            states,
            bins,
        }
    }

    fn state(&self, v: usize, h: usize) -> &State {
        self.states[v * self.ranges.len() + h]
            .as_ref()
            .expect("solved rectangle")
    }

    fn solve(&mut self, v: usize, h: usize) -> f64 {
        let id = v * self.ranges.len() + h;
        if let Some(state) = &self.states[id] {
            return state.bits;
        }
        let vertical = self.ranges[v].children;
        let horizontal = self.ranges[h].children;
        let vb = vertical.map(|(lo, hi)| self.solve(lo, h) + self.solve(hi, h) + SPLIT_BITS);
        let hb = horizontal.map(|(lo, hi)| self.solve(v, lo) + self.solve(v, hi) + SPLIT_BITS);
        // Both orders cover the same rectangle. Aggregate its raw histograms
        // once; no candidate's chosen predictor can contaminate another cost.
        let hist = if let Some((lo, hi)) = vertical {
            merged(&self.state(lo, h).hist, &self.state(hi, h).hist)
        } else if let Some((lo, hi)) = horizontal {
            merged(&self.state(v, lo).hist, &self.state(v, hi).hist)
        } else {
            self.cells[self.ranges[v].start * self.bins + self.ranges[h].start].clone()
        };
        let leaf = best_leaf(&hist);
        let mut bits = leaf.bits;
        let mut choice = Choice::Leaf;
        if let Some(cost) = vb.filter(|&b| b < bits) {
            bits = cost;
            choice = Choice::Vertical;
        }
        if let Some(cost) = hb.filter(|&b| b < bits) {
            bits = cost;
            choice = Choice::Horizontal;
        }
        self.states[id] = Some(State {
            hist,
            leaf,
            bits,
            choice,
        });
        bits
    }

    fn append(&self, nodes: &mut Vec<Node>, thresholds: &[i32], v: usize, h: usize) -> usize {
        let state = self.state(v, h);
        let id = nodes.len();
        nodes.push(state.leaf.node());
        let (property, range) = match state.choice {
            Choice::Leaf => return id,
            Choice::Vertical => (10, v),
            Choice::Horizontal => (11, h),
        };
        let (lo, hi) = self.ranges[range].children.expect("split range");
        let threshold = thresholds[self.ranges[hi].start - 1];
        let (greater, less_equal) = if property == 10 {
            (
                self.append(nodes, thresholds, hi, h),
                self.append(nodes, thresholds, lo, h),
            )
        } else {
            (
                self.append(nodes, thresholds, v, hi),
                self.append(nodes, thresholds, v, lo),
            )
        };
        nodes[id] = Node::Split {
            property,
            threshold,
            greater,
            less_equal,
        };
        id
    }
}

struct Plan {
    nodes: Vec<Node>,
    bits: f64,
    single: Leaf,
    hist: Histograms,
}

impl Plan {
    fn leaves(&self) -> usize {
        (self.nodes.len() + 1) / 2
    }
    fn collapse(&mut self) {
        self.nodes = vec![self.single.node()];
        self.bits = self.single.bits;
    }
}

fn scan(views: &[&ChannelView<'_>], thresholds: &[i32], options: &Options) -> Result<Plan> {
    let scan_scope = crate::profile::scope("ma.fast_context.activity_scan");
    let bins = thresholds.len() + 1;
    let mut cells: Vec<Histograms> = (0..bins * bins)
        .map(|_| std::array::from_fn(|_| Histogram::default()))
        .collect();
    let width = views
        .iter()
        .filter(|v| v.height != 0)
        .map(|v| v.width)
        .max()
        .unwrap_or(0);
    let mut row = rows::Rows::new(width, options.predictors);
    for view in views {
        if view.width == 0 || view.height == 0 {
            continue;
        }
        for y in 0..view.height {
            row.fill(view, y, options.scan.backend(), None)?;
            for x in 0..view.width {
                let v = thresholds.partition_point(|&t| row.basic.vertical[x] > t);
                let h = thresholds.partition_point(|&t| row.basic.horizontal[x] > t);
                for (hist, residual) in cells[v * bins + h].iter_mut().zip(row.residuals_at(x)) {
                    hist.add(residual, false);
                }
            }
        }
    }
    drop(scan_scope);
    let _scope = crate::profile::scope("ma.fast_context.activity_prune");
    let mut grid = Grid::new(cells, bins);
    let bits = grid.solve(0, 0);
    let mut nodes = Vec::new();
    grid.append(&mut nodes, thresholds, 0, 0);
    let root = grid.state(0, 0);
    Ok(Plan {
        nodes,
        bits,
        single: root.leaf,
        hist: root.hist.clone(),
    })
}

fn append_channels(nodes: &mut Vec<Node>, plans: &[Plan], first: usize) -> usize {
    let id = nodes.len();
    if plans.len() == 1 {
        nodes.extend(plans[0].nodes.iter().map(|node| match *node {
            Node::Leaf { .. } => node.clone(),
            Node::Split {
                property,
                threshold,
                greater,
                less_equal,
            } => Node::Split {
                property,
                threshold,
                greater: greater + id,
                less_equal: less_equal + id,
            },
        }));
    } else {
        nodes.push(plans[0].single.node());
        let mid = plans.len() / 2;
        let greater = append_channels(nodes, &plans[mid..], first + mid);
        let less_equal = append_channels(nodes, &plans[..mid], first);
        nodes[id] = Node::Split {
            property: 0,
            threshold: (first + mid - 1) as i32,
            greater,
            less_equal,
        };
    }
    id
}

pub(super) fn learn(
    views: &[ChannelView<'_>],
    channels: usize,
    limit: usize,
    strict: bool,
    options: &Options,
) -> Result<Tree> {
    if limit == 0 {
        return Err(Error::Limit(
            "activity grid requires a positive leaf budget",
        ));
    }
    let thresholds = match options.mode {
        Mode::Activity25 => COARSE,
        Mode::Activity81 => FINE,
        _ => unreachable!(),
    };
    let by_channel: Vec<Vec<_>> = (0..channels.max(1))
        .map(|channel| {
            views
                .iter()
                .filter(|v| v.channel as usize == channel)
                .collect()
        })
        .collect();
    let mut plans =
        crate::execution::try_map(&by_channel, |_, views| scan(views, thresholds, options))?;
    let mut leaves = plans.iter().map(Plan::leaves).sum::<usize>();
    if strict && leaves > limit {
        return Err(Error::Limit(
            "activity grid exceeds explicit strict leaf budget",
        ));
    }
    // Only user/decoder limits trigger this fallback, never an implicit new cap.
    while leaves > limit {
        let choice = plans
            .iter()
            .enumerate()
            .filter(|(_, p)| p.leaves() > 1)
            .min_by(|(_, a), (_, b)| {
                let a = (a.single.bits - a.bits) / (a.leaves() - 1) as f64;
                let b = (b.single.bits - b.bits) / (b.leaves() - 1) as f64;
                a.total_cmp(&b)
            })
            .map(|(i, _)| i);
        if let Some(i) = choice {
            leaves -= plans[i].leaves() - 1;
            plans[i].collapse();
        } else {
            let mut hist: Histograms = std::array::from_fn(|_| Histogram::default());
            for p in &plans {
                for (to, from) in hist.iter_mut().zip(&p.hist) {
                    to.merge(from);
                }
            }
            let single = best_leaf(&hist);
            plans = vec![Plan {
                nodes: vec![single.node()],
                bits: single.bits,
                single,
                hist,
            }];
            break;
        }
    }
    let mut nodes = Vec::new();
    append_channels(&mut nodes, &plans, 0);
    let tree = Tree::new(nodes)?;
    if let Some(receipt) = &options.receipt {
        receipt.trees.fetch_add(1, Ordering::Relaxed);
        receipt.leaves.fetch_add(tree.leaves(), Ordering::Relaxed);
        receipt.pixels.fetch_add(
            views.iter().map(|v| v.width * v.height).sum::<usize>(),
            Ordering::Relaxed,
        );
        for plan in plans {
            let vertical = plan
                .nodes
                .iter()
                .any(|n| matches!(n, Node::Split { property: 10, .. }));
            let horizontal = plan
                .nodes
                .iter()
                .any(|n| matches!(n, Node::Split { property: 11, .. }));
            let counter = match (vertical, horizontal) {
                (true, true) => &receipt.joint,
                (true, false) => &receipt.vertical,
                (false, true) => &receipt.horizontal,
                _ => &receipt.single,
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(tree)
}

#[cfg(test)]
mod tests;
