//! Opt-in complete shallow MA plans, using only standard JXL properties and
//! predictors. This is an alternative learner, not a prefix followed by the
//! original exhaustive search. No fNBLI predictor or entropy coder is emitted.

mod rows;
mod scan;
mod stats;

use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::{
    ma_train::ChannelView,
    predictor::{Neighbors, Predictor},
    tree::{Node, Tree},
    weighted,
};
use crate::{Error, Result};
use stats::Histogram;

const PREDICTORS: [Predictor; 4] = [
    Predictor::Zero,
    Predictor::Left,
    Predictor::Top,
    Predictor::Gradient,
];

// Keep Basic's order first so extending the set preserves existing tie breaks.
const FULL_PREDICTORS: [Predictor; 14] = [
    Predictor::Zero,
    Predictor::Left,
    Predictor::Top,
    Predictor::Gradient,
    Predictor::Average0,
    Predictor::Select,
    Predictor::TopRight,
    Predictor::TopLeft,
    Predictor::LeftLeft,
    Predictor::Average1,
    Predictor::Average2,
    Predictor::Average3,
    Predictor::Average4,
    Predictor::Weighted,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PredictorSet {
    /// Zero, Left, Top, and Gradient, using the fused feature scan.
    #[default]
    Basic,
    /// All 14 standard bitstream predictors, including native Weighted state.
    Full,
}

impl PredictorSet {
    fn predictors(self) -> &'static [Predictor] {
        match self {
            Self::Basic => &PREDICTORS,
            Self::Full => &FULL_PREDICTORS,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    Leaf,
    Gradient12,
    Directional12,
    #[default]
    Auto12,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Bias {
    #[default]
    Off,
    Static,
    Adaptive,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scan {
    #[default]
    Auto,
    Scalar,
    Avx2,
    Avx512,
}

impl Scan {
    fn backend(self) -> scan::Backend {
        match self {
            Self::Auto => scan::Backend::Auto,
            Self::Scalar => scan::Backend::Scalar,
            Self::Avx2 => scan::Backend::Avx2,
            Self::Avx512 => scan::Backend::Avx512,
        }
    }
}

/// Explicit experimental tool. The normal encoder never enables it implicitly.
/// `Adaptive` uses a causal residual-bias estimate as an additional *static*
/// leaf-offset proposal. Both proposals are rescored over the complete domain;
/// the emitted decoder prediction has no custom adaptive state.
#[derive(Clone, Debug, Default)]
pub struct Options {
    pub mode: Mode,
    pub bias: Bias,
    pub scan: Scan,
    /// Full is opt-in and requires each view's exact stream WP header.
    pub predictors: PredictorSet,
    pub receipt: Option<Arc<Receipt>>,
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        if !self.scan.backend().available() {
            return Err(Error::InvalidInput(
                "requested fast-context SIMD backend unavailable".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct Receipt {
    trees: AtomicUsize,
    leaves: AtomicUsize,
    pixels: AtomicUsize,
    offset_leaves: AtomicUsize,
    horizontal: AtomicUsize,
    vertical: AtomicUsize,
    single: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub trees: usize,
    pub leaves: usize,
    pub pixels: usize,
    pub offset_leaves: usize,
    /// Plans split on property 11 (TopLeft - Top, negative horizontal).
    pub horizontal: usize,
    /// Plans split on property 10 (Left - TopLeft, vertical).
    pub vertical: usize,
    pub single: usize,
}

impl Receipt {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            trees: self.trees.load(Ordering::Relaxed),
            leaves: self.leaves.load(Ordering::Relaxed),
            pixels: self.pixels.load(Ordering::Relaxed),
            offset_leaves: self.offset_leaves.load(Ordering::Relaxed),
            horizontal: self.horizontal.load(Ordering::Relaxed),
            vertical: self.vertical.load(Ordering::Relaxed),
            single: self.single.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy)]
struct Leaf {
    predictor: Predictor,
    offset: i32,
    bits: f64,
}

impl Leaf {
    fn node(self) -> Node {
        Node::Leaf {
            predictor: self.predictor,
            offset: self.offset,
            multiplier: 1,
        }
    }
}

struct Bin {
    original: Vec<Histogram>,
    offsets: Vec<[i32; 2]>,
    shifted: Vec<[Histogram; 2]>,
}

impl Bin {
    fn new(predictors: PredictorSet) -> Self {
        let count = predictors.predictors().len();
        Self {
            original: vec![Histogram::default(); count],
            offsets: vec![[0; 2]; count],
            shifted: (0..count)
                .map(|_| std::array::from_fn(|_| Histogram::default()))
                .collect(),
        }
    }

    fn prepare_bias(&mut self, bias: Bias) {
        for p in 0..self.original.len() {
            self.offsets[p] = [
                self.original[p].mean(),
                if bias == Bias::Adaptive {
                    self.original[p].recent_bias()
                } else {
                    0
                },
            ];
        }
    }

    fn best(&self, bias: Bias, predictors: PredictorSet) -> Leaf {
        let mut best = Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            bits: f64::INFINITY,
        };
        for (p, &predictor) in predictors.predictors().iter().enumerate() {
            let mut choices = [
                (0, self.original[p].bits(0)),
                (0, f64::INFINITY),
                (0, f64::INFINITY),
            ];
            if bias != Bias::Off {
                for i in 0..2 {
                    if self.offsets[p][i] != 0 {
                        choices[i + 1] = (
                            self.offsets[p][i],
                            self.shifted[p][i].bits(self.offsets[p][i]),
                        );
                    }
                }
            }
            for (offset, bits) in choices {
                if bits < best.bits {
                    best = Leaf {
                        predictor,
                        offset,
                        bits,
                    };
                }
            }
        }
        best
    }

    #[inline]
    fn add(&mut self, values: impl IntoIterator<Item = i32>, bias: Bias, shifted: bool) {
        for (p, value) in values.into_iter().enumerate() {
            if shifted {
                for i in 0..2 {
                    if self.offsets[p][i] != 0 {
                        self.shifted[p][i].add(value.wrapping_sub(self.offsets[p][i]), false);
                    }
                }
            } else {
                self.original[p].add(value, bias == Bias::Adaptive);
            }
        }
    }
}

struct Axis {
    property: usize,
    thresholds: Vec<i32>,
    bins: Vec<Bin>,
}

impl Axis {
    fn new(property: usize, thresholds: Vec<i32>, predictors: PredictorSet) -> Self {
        Self {
            property,
            bins: (0..=thresholds.len())
                .map(|_| Bin::new(predictors))
                .collect(),
            thresholds,
        }
    }

    fn plan(&self, bias: Bias, predictors: PredictorSet) -> Plan {
        let leaves: Vec<_> = self.bins.iter().map(|b| b.best(bias, predictors)).collect();
        let bits = leaves.iter().map(|l| l.bits).sum::<f64>() + self.thresholds.len() as f64 * 24.0;
        Plan {
            property: self.property,
            thresholds: self.thresholds.clone(),
            leaves,
            bits,
        }
    }
}

struct Plan {
    property: usize,
    thresholds: Vec<i32>,
    leaves: Vec<Leaf>,
    bits: f64,
}

fn leaf_plan(leaf: Leaf) -> Plan {
    Plan {
        property: 0,
        thresholds: Vec::new(),
        leaves: vec![leaf],
        bits: leaf.bits,
    }
}

// The small stratified pass only chooses legal split coordinates. Every pixel
// participates in subsequent rate statistics, including rare/transparent data.
fn thresholds(views: &[&ChannelView<'_>], property: usize) -> Result<Vec<i32>> {
    let mut samples = Vec::new();
    for view in views {
        let pixels = view.width * view.height;
        let count = pixels.min(256);
        for i in 0..count {
            let lo = (i as u128 * pixels as u128 / count as u128) as usize;
            let hi = ((i + 1) as u128 * pixels as u128 / count as u128) as usize;
            let hash = (i as u64)
                .wrapping_add(u64::from(view.group))
                .wrapping_mul(0x9e3779b97f4a7c15);
            let index = lo + (hash as usize % (hi - lo));
            let n = Neighbors::from_plane(
                view.pixels,
                view.width,
                view.stride,
                index % view.width,
                index / view.width,
            )
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
            samples.push(if property == 10 {
                n.left.wrapping_sub(n.topleft)
            } else {
                n.topleft.wrapping_sub(n.top)
            });
        }
    }
    samples.sort_unstable();
    let mut result = Vec::new();
    if !samples.is_empty() {
        for i in 1..12 {
            result.push(samples[(samples.len() * i / 12).min(samples.len() - 1)]);
        }
        result.dedup();
        result.retain(|&v| v != i32::MAX);
    }
    Ok(result)
}

fn scan_views(
    views: &[&ChannelView<'_>],
    headers: &BTreeMap<u32, weighted::Header>,
    axes: &mut [Axis],
    total: &mut Bin,
    options: &Options,
    shifted: bool,
) -> Result<()> {
    let _scope = crate::profile::scope(if shifted {
        "ma.fast_context.bias_scan"
    } else {
        "ma.fast_context.scan"
    });
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
        // Every independently coded view (including metadata) starts fresh,
        // also on the static-offset rescore pass.
        let mut state = if options.predictors == PredictorSet::Full {
            Some(weighted::State::new(headers[&view.group], view.width)?)
        } else {
            None
        };
        for y in 0..view.height {
            row.fill(view, y, options.scan.backend(), state.as_mut())?;
            for x in 0..view.width {
                total.add(row.residuals_at(x), options.bias, shifted);
                for axis in axes.iter_mut() {
                    let value = if axis.property == 10 {
                        row.basic.vertical[x]
                    } else {
                        row.basic.horizontal[x]
                    };
                    let bin = axis
                        .thresholds
                        .partition_point(|&threshold| value > threshold);
                    axis.bins[bin].add(row.residuals_at(x), options.bias, shifted);
                }
            }
        }
    }
    Ok(())
}

fn resolve_headers(
    views: &[ChannelView<'_>],
    headers: &[(u32, weighted::Header)],
    predictors: PredictorSet,
) -> Result<BTreeMap<u32, weighted::Header>> {
    if predictors == PredictorSet::Basic {
        return Ok(BTreeMap::new());
    }
    let mut relevant: BTreeMap<_, Option<weighted::Header>> =
        views.iter().map(|view| (view.group, None)).collect();
    for &(group, header) in headers {
        if let Some(slot) = relevant.get_mut(&group) {
            if slot.is_some() {
                return Err(Error::InvalidInput(format!(
                    "duplicate fast-context weighted header for stream {group}"
                )));
            }
            header.validate()?;
            *slot = Some(header);
        }
    }
    relevant
        .into_iter()
        .map(|(group, header)| {
            header.map(|header| (group, header)).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "missing fast-context weighted header for stream {group}"
                ))
            })
        })
        .collect()
}

fn append_plan(nodes: &mut Vec<Node>, plan: &Plan, from: usize, to: usize) -> usize {
    let index = nodes.len();
    nodes.push(plan.leaves[from].node());
    if to - from > 1 {
        let mid = (from + to) / 2;
        let greater = append_plan(nodes, plan, mid, to);
        let less_equal = append_plan(nodes, plan, from, mid);
        nodes[index] = Node::Split {
            property: plan.property,
            threshold: plan.thresholds[mid - 1],
            greater,
            less_equal,
        };
    }
    index
}

fn append_channels(nodes: &mut Vec<Node>, plans: &[Plan], first: usize) -> usize {
    if plans.len() == 1 {
        return append_plan(nodes, &plans[0], 0, plans[0].leaves.len());
    }
    let index = nodes.len();
    nodes.push(plans[0].leaves[0].node());
    let mid = plans.len() / 2;
    let greater = append_channels(nodes, &plans[mid..], first + mid);
    let less_equal = append_channels(nodes, &plans[..mid], first);
    nodes[index] = Node::Split {
        property: 0,
        threshold: (first + mid - 1) as i32,
        greater,
        less_equal,
    };
    index
}

/// Views are validated by the shared training entry point before this call.
/// Full requires exactly one header for every view's stream ID (`group`),
/// including empty views. Unreferenced headers are ignored; Basic needs none.
pub(super) fn learn(
    views: &[ChannelView<'_>],
    channels: usize,
    limit: usize,
    strict: bool,
    headers: &[(u32, weighted::Header)],
    options: &Options,
) -> Result<Tree> {
    let _scope = crate::profile::scope("ma.fast_context");
    options.validate()?;
    let headers = resolve_headers(views, headers, options.predictors)?;
    let mut plans = Vec::new();
    let mut singles = Vec::new();
    let mut global = Bin::new(options.predictors);
    let by_channel: Vec<Vec<_>> = (0..channels.max(1))
        .map(|channel| {
            views
                .iter()
                .filter(|v| v.channel as usize == channel)
                .collect()
        })
        .collect();
    let results = crate::execution::try_map(&by_channel, |_, views| {
        let mut axes = Vec::new();
        if options.mode != Mode::Leaf {
            axes.push(Axis::new(10, thresholds(&views, 10)?, options.predictors));
        }
        if matches!(options.mode, Mode::Directional12 | Mode::Auto12) {
            axes.push(Axis::new(11, thresholds(&views, 11)?, options.predictors));
        }
        let mut total = Bin::new(options.predictors);
        scan_views(&views, &headers, &mut axes, &mut total, options, false)?;
        if options.bias != Bias::Off {
            total.prepare_bias(options.bias);
            for axis in &mut axes {
                for bin in &mut axis.bins {
                    bin.prepare_bias(options.bias);
                }
            }
            scan_views(&views, &headers, &mut axes, &mut total, options, true)?;
        }
        let single = total.best(options.bias, options.predictors);
        let mut candidates: Vec<_> = axes
            .iter()
            .map(|a| a.plan(options.bias, options.predictors))
            .collect();
        if matches!(options.mode, Mode::Leaf | Mode::Auto12) {
            candidates.insert(0, leaf_plan(single));
        }
        let plan = candidates
            .into_iter()
            .min_by(|a, b| a.bits.total_cmp(&b.bits))
            .expect("at least one candidate");
        Ok((plan, single, total.original))
    })?;
    for (plan, single, originals) in results {
        for (to, from) in global.original.iter_mut().zip(originals) {
            to.merge(&from);
        }
        singles.push(single);
        plans.push(plan);
    }
    // Honor explicit and decoder limits without turning a shallow-plan choice
    // into a new encoding failure. Remove the least valuable partitions first.
    let mut leaves = plans.iter().map(|p| p.leaves.len()).sum::<usize>();
    if strict && leaves > limit {
        return Err(Error::Limit(
            "fast-context plan exceeds explicit strict leaf budget",
        ));
    }
    while leaves > limit {
        let choice = plans
            .iter()
            .enumerate()
            .filter(|(_, p)| p.leaves.len() > 1)
            .min_by(|(a, pa), (b, pb)| {
                let ca = (singles[*a].bits - pa.bits) / (pa.leaves.len() - 1) as f64;
                let cb = (singles[*b].bits - pb.bits) / (pb.leaves.len() - 1) as f64;
                ca.total_cmp(&cb)
            })
            .map(|(i, _)| i);
        if let Some(i) = choice {
            leaves -= plans[i].leaves.len() - 1;
            plans[i] = leaf_plan(singles[i]);
        } else {
            plans = vec![leaf_plan(global.best(Bias::Off, options.predictors))];
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
        receipt.offset_leaves.fetch_add(
            tree.nodes()
                .iter()
                .filter(|n| matches!(n, Node::Leaf { offset, .. } if *offset != 0))
                .count(),
            Ordering::Relaxed,
        );
        for plan in plans {
            let counter = if plan.leaves.len() == 1 {
                &receipt.single
            } else if plan.property == 10 {
                &receipt.vertical
            } else {
                &receipt.horizontal
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(tree)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod full_tests;
