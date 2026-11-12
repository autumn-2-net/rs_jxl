//! Independently implemented activity-grid warm start over prepared MA records.
//! C29 motivates activity-conditioned statistics, not this partition or scorer.
//! Quantizers, residual tokens, and source scoring stay owned by the MA learner.

use super::{ParentHistogram, PropertyColumn, SampleStorage, Samples, Split, TrainOptions, bits};
use crate::{Error, Result};

const COARSE: &[i32] = &[-5, -1, 0, 4];
const FINE: &[i32] = &[-65, -17, -5, -1, 0, 4, 16, 64];

pub(super) struct Plan {
    pub(super) predictor: usize,
    pub(super) bits: f32,
    pub(super) parents: Vec<ParentHistogram>,
    pub(super) records: usize,
    pub(super) children: Option<Box<(Split, Plan, Plan)>>,
}

struct Axis {
    property: Option<usize>,
    bins: Vec<usize>,
}

impl Axis {
    fn new(
        samples: &Samples<impl SampleStorage>,
        options: &TrainOptions,
        property: usize,
        desired: &[i32],
    ) -> Self {
        let property = options.properties.iter().position(|&id| id == property);
        let mut bins = Vec::new();
        if let Some(property) = property {
            let thresholds = &samples.thresholds[property];
            for &value in desired {
                if let Some(bin) = thresholds.partition_point(|&t| t <= value).checked_sub(1)
                    && thresholds[bin] != i32::MAX
                    && bins.last() != Some(&bin)
                {
                    bins.push(bin);
                }
            }
        }
        Self { property, bins }
    }

    fn cells(&self, samples: &Samples<impl SampleStorage>, indices: &[usize]) -> Vec<usize> {
        let Some(property) = self.property.filter(|_| !self.bins.is_empty()) else {
            return vec![0; indices.len()];
        };
        with_property_values!(&samples.properties[property], |values| {
            indices
                .iter()
                .map(|&i| self.bins.partition_point(|&bin| values[i] as usize > bin))
                .collect()
        })
    }
}

#[derive(Clone, Copy)]
struct Range {
    start: usize,
    children: Option<(usize, usize)>,
}

fn ranges(bins: usize) -> Vec<Range> {
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
    append(&mut out, 0, bins);
    out
}

fn empty_parents(predictors: usize) -> Vec<ParentHistogram> {
    (0..predictors)
        .map(|_| ParentHistogram {
            counts: vec![0],
            extra_bits: 0,
        })
        .collect()
}

fn merge(a: &[ParentHistogram], b: &[ParentHistogram]) -> Vec<ParentHistogram> {
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let mut counts = a.counts.clone();
            counts.resize(counts.len().max(b.counts.len()), 0);
            for (to, &from) in counts.iter_mut().zip(&b.counts) {
                *to += from;
            }
            ParentHistogram {
                counts,
                extra_bits: a.extra_bits + b.extra_bits,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Choice {
    Leaf,
    Split { vertical: bool },
    Skip { v: usize, h: usize },
}

struct State {
    parents: Vec<ParentHistogram>,
    records: usize,
    predictor: usize,
    bits: f32,
    choice: Choice,
}

struct Grid<'a> {
    axes: [Axis; 2],
    ranges: [Vec<Range>; 2],
    cells: Vec<(Vec<ParentHistogram>, usize)>,
    states: Vec<Option<State>>,
    options: &'a TrainOptions,
    split_bits: f32,
}

impl Grid<'_> {
    fn id(&self, v: usize, h: usize) -> usize {
        v * self.ranges[1].len() + h
    }

    fn state(&self, v: usize, h: usize) -> &State {
        self.states[self.id(v, h)]
            .as_ref()
            .expect("solved activity rectangle")
    }

    fn candidate(&self, lo: (usize, usize), hi: (usize, usize), vertical: bool) -> (f32, Choice) {
        let below = self.state(lo.0, lo.1);
        let above = self.state(hi.0, hi.1);
        if below.records == 0 {
            (above.bits, Choice::Skip { v: hi.0, h: hi.1 })
        } else if above.records == 0 {
            (below.bits, Choice::Skip { v: lo.0, h: lo.1 })
        } else {
            (
                above.bits + below.bits + self.split_bits,
                Choice::Split { vertical },
            )
        }
    }

    fn solve(&mut self, v: usize, h: usize) {
        let id = self.id(v, h);
        if self.states[id].is_some() {
            return;
        }
        let vertical = self.ranges[0][v].children;
        let horizontal = self.ranges[1][h].children;
        let vc = vertical.map(|(lo, hi)| {
            self.solve(lo, h);
            self.solve(hi, h);
            self.candidate((lo, h), (hi, h), true)
        });
        let hc = horizontal.map(|(lo, hi)| {
            self.solve(v, lo);
            self.solve(v, hi);
            self.candidate((v, lo), (v, hi), false)
        });
        // Aggregate raw counts, never the children's selected predictors.
        let (parents, records) = if let Some((lo, hi)) = vertical {
            let a = self.state(lo, h);
            let b = self.state(hi, h);
            (merge(&a.parents, &b.parents), a.records + b.records)
        } else if let Some((lo, hi)) = horizontal {
            let a = self.state(v, lo);
            let b = self.state(v, hi);
            (merge(&a.parents, &b.parents), a.records + b.records)
        } else {
            let cell =
                self.ranges[0][v].start * (self.axes[1].bins.len() + 1) + self.ranges[1][h].start;
            (self.cells[cell].0.clone(), self.cells[cell].1)
        };
        let mut predictor = 0;
        let mut cost = bits(&parents[0], self.options);
        for (p, parent) in parents.iter().enumerate().skip(1) {
            let candidate = bits(parent, self.options);
            if candidate < cost {
                predictor = p;
                cost = candidate;
            }
        }
        let mut choice = Choice::Leaf;
        for (candidate, candidate_choice) in vc.into_iter().chain(hc) {
            if candidate < cost {
                cost = candidate;
                choice = candidate_choice;
            }
        }
        self.states[id] = Some(State {
            parents,
            records,
            predictor,
            bits: cost,
            choice,
        });
    }

    fn take_plan(&mut self, v: usize, h: usize) -> Plan {
        let id = self.id(v, h);
        let state = self.states[id].take().expect("selected activity rectangle");
        let children = match state.choice {
            Choice::Leaf => None,
            Choice::Skip { v, h } => return self.take_plan(v, h),
            Choice::Split { vertical } => {
                let axis = usize::from(!vertical);
                let range = if vertical { v } else { h };
                let (lo, hi) = self.ranges[axis][range].children.expect("split range");
                let bin = self.axes[axis].bins[self.ranges[axis][hi].start - 1];
                let above = if vertical {
                    self.take_plan(hi, h)
                } else {
                    self.take_plan(v, hi)
                };
                let below = if vertical {
                    self.take_plan(lo, h)
                } else {
                    self.take_plan(v, lo)
                };
                let split = Split {
                    property: self.axes[axis].property.expect("split property"),
                    bin,
                    below_count: below.records,
                    below_predictor: below.predictor,
                    above_predictor: above.predictor,
                    bits: above.bits + below.bits,
                    split_cost: self.split_bits,
                    symmetric: None,
                };
                Some(Box::new((split, above, below)))
            }
        };
        Plan {
            predictor: state.predictor,
            bits: state.bits,
            parents: state.parents,
            records: state.records,
            children,
        }
    }
}

pub(super) fn plan(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    fine: bool,
    split_bits: f32,
) -> Result<Plan> {
    if options.predictors.is_empty()
        || samples.residuals.len() != options.predictors.len()
        || !split_bits.is_finite()
        || split_bits < 0.0
    {
        return Err(Error::InvalidInput(
            "invalid activity seed predictors/model cost".into(),
        ));
    }
    let desired = if fine { FINE } else { COARSE };
    let axes = [
        Axis::new(samples, options, 10, desired),
        Axis::new(samples, options, 11, desired),
    ];
    let coordinates = [
        axes[0].cells(samples, indices),
        axes[1].cells(samples, indices),
    ];
    let bins = [axes[0].bins.len() + 1, axes[1].bins.len() + 1];
    let mut cells: Vec<_> = (0..bins[0] * bins[1])
        .map(|_| (empty_parents(options.predictors.len()), 0usize))
        .collect();
    for (record, &i) in indices.iter().enumerate() {
        let (parents, records) =
            &mut cells[coordinates[0][record] * bins[1] + coordinates[1][record]];
        *records += 1;
        let weight = u32::from(samples.counts[i]);
        if weight == 0 {
            continue;
        }
        for (parent, residuals) in parents.iter_mut().zip(&samples.residuals) {
            let token = residuals[i];
            let symbol = usize::from(token.token);
            if parent.counts.len() <= symbol {
                parent.counts.resize(symbol + 1, 0);
            }
            parent.counts[symbol] += weight;
            parent.extra_bits += u64::from(token.nbits) * u64::from(weight);
        }
    }
    let ranges = [ranges(bins[0]), ranges(bins[1])];
    let states = (0..ranges[0].len() * ranges[1].len())
        .map(|_| None)
        .collect();
    let mut grid = Grid {
        axes,
        ranges,
        cells,
        states,
        options,
        split_bits,
    };
    grid.solve(0, 0);
    Ok(grid.take_plan(0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::{
        ma_train::{ResidualToken, TreeSamples, learner, parent_histograms},
        predictor::Predictor,
        tree::pack_signed,
    };

    const QUANTIZER: &[i32] = &[-100, -20, -6, -2, 3, 19, 90, i32::MAX];

    fn token(value: i32) -> ResidualToken {
        let encoded = crate::entropy::HybridUintConfig::new(4, 1, 2)
            .unwrap()
            .encode(pack_signed(value));
        ResidualToken {
            token: encoded.token as u8,
            nbits: encoded.nbits as u8,
        }
    }

    fn fixture(raw: &[[i32; 2]], wide: bool) -> (TreeSamples, TrainOptions) {
        let options = TrainOptions {
            predictors: (0..14).map(|id| Predictor::try_from(id).unwrap()).collect(),
            // Reversed property order ensures Split.property is a column index.
            properties: vec![11, 10],
            ..Default::default()
        };
        let properties = [1, 0]
            .into_iter()
            .map(|axis| {
                let values: Vec<_> = raw
                    .iter()
                    .map(|v| QUANTIZER.partition_point(|&t| v[axis] > t) as u32)
                    .collect();
                if wide {
                    PropertyColumn::Wide(values)
                } else {
                    PropertyColumn::Packed(values.into_iter().map(|v| v as u8).collect())
                }
            })
            .collect();
        let samples = TreeSamples {
            properties,
            residuals: (0..14)
                .map(|p| {
                    (0..raw.len())
                        .map(|i| token(((i + p) as i32 * 71) - 900))
                        .collect()
                })
                .collect(),
            counts: (0..raw.len()).map(|i| (i % 7 + 1) as u16).collect(),
            thresholds: vec![QUANTIZER.to_vec(); 2],
            global_max_symbols: 256,
        };
        (samples, options)
    }

    fn verify(plan: &Plan, samples: &mut TreeSamples, indices: &mut [usize], split_bits: f32) {
        assert_eq!(plan.records, indices.len());
        assert_eq!(plan.parents, parent_histograms(samples, indices));
        if let Some(children) = &plan.children {
            let (split, above, below) = &**children;
            assert!(above.records > 0 && below.records > 0);
            assert_eq!(split.below_count, below.records);
            assert_eq!(split.above_predictor, above.predictor);
            assert_eq!(split.below_predictor, below.predictor);
            assert_eq!(split.bits, above.bits + below.bits);
            assert_eq!(split.split_cost, split_bits);
            assert!(split.symmetric.is_none());
            assert_eq!(plan.bits, split.bits + split_bits);
            assert_ne!(samples.thresholds[split.property][split.bin], i32::MAX);
            assert_eq!(
                samples.properties[split.property].count_le(indices, split.bin),
                below.records
            );
            learner::partition_node::<false>(
                samples,
                indices,
                split.property,
                split.bin,
                below.records,
            );
            let (below_indices, above_indices) = indices.split_at_mut(below.records);
            verify(above, samples, above_indices, split_bits);
            verify(below, samples, below_indices, split_bits);
        }
    }

    fn lookup<'a>(
        plan: &'a Plan,
        samples: &TreeSamples,
        raw: &[i32; 2],
        options: &TrainOptions,
    ) -> &'a Plan {
        match &plan.children {
            None => plan,
            Some(children) => {
                let (split, above, below) = &**children;
                let value = raw[options.properties[split.property] - 10];
                let threshold = samples.thresholds[split.property][split.bin];
                lookup(
                    if value > threshold { above } else { below },
                    samples,
                    raw,
                    options,
                )
            }
        }
    }

    #[test]
    fn quantizer_floor_mapping_keeps_original_bins_and_deduplicates() {
        let (samples, options) = fixture(&[], false);
        let coarse = Axis::new(&samples, &options, 10, COARSE);
        let fine = Axis::new(&samples, &options, 11, FINE);
        assert_eq!(coarse.property, Some(1));
        assert_eq!(coarse.bins, [2, 3, 4]);
        assert_eq!(fine.property, Some(0));
        assert_eq!(fine.bins, [0, 1, 2, 3, 4, 5]);
        let mut samples = samples;
        samples.thresholds = vec![vec![100, i32::MAX]; 2];
        assert!(Axis::new(&samples, &options, 10, FINE).bins.is_empty());
    }

    #[test]
    fn packed_wide_and_subset_histograms_preserve_weighted_mass_and_extra_bits() {
        let raw: Vec<_> = (-12..=12).map(|i| [i * 9, 19 - i * 7]).collect();
        for wide in [false, true] {
            let (mut samples, options) = fixture(&raw, wide);
            for fine in [false, true] {
                let mut indices: Vec<_> = (0..raw.len()).rev().step_by(2).collect();
                let result = plan(&samples, &indices, &options, fine, 3.0).unwrap();
                verify(&result, &mut samples, &mut indices, 3.0);
                let weight: u32 = indices.iter().map(|&i| u32::from(samples.counts[i])).sum();
                for parent in &result.parents {
                    assert_eq!(parent.counts.iter().sum::<u32>(), weight);
                }
            }
        }
    }

    #[test]
    fn boundaries_match_emitted_thresholds_and_real_partition_counts() {
        let values: Vec<_> = [i32::MIN, i32::MAX]
            .into_iter()
            .chain(
                QUANTIZER[..QUANTIZER.len() - 1]
                    .iter()
                    .flat_map(|&v| [v - 1, v, v + 1]),
            )
            .collect();
        let raw: Vec<_> = values
            .iter()
            .flat_map(|&v| values.iter().map(move |&h| [v, h]))
            .collect();
        for wide in [false, true] {
            for fine in [false, true] {
                let (mut samples, options) = fixture(&raw, wide);
                let desired = if fine { FINE } else { COARSE };
                let axes = [
                    Axis::new(&samples, &options, 10, desired),
                    Axis::new(&samples, &options, 11, desired),
                ];
                let indices: Vec<_> = (0..raw.len()).collect();
                let v = axes[0].cells(&samples, &indices);
                let h = axes[1].cells(&samples, &indices);
                let winners: Vec<_> = v
                    .iter()
                    .zip(&h)
                    .map(|(&v, &h)| (v * (axes[1].bins.len() + 1) + h) % options.predictors.len())
                    .collect();
                samples.counts.fill(200);
                for (p, residuals) in samples.residuals.iter_mut().enumerate() {
                    for (i, residual) in residuals.iter_mut().enumerate() {
                        *residual = token(if p == winners[i] { 0 } else { 1 << 20 });
                    }
                }
                let result = plan(&samples, &indices, &options, fine, 1.0).unwrap();
                assert!(result.children.is_some());
                for (i, value) in raw.iter().enumerate() {
                    assert_eq!(
                        lookup(&result, &samples, value, &options).predictor,
                        winners[i]
                    );
                }
                verify(&result, &mut samples, &mut indices.clone(), 1.0);
            }
        }
    }

    #[test]
    fn every_configured_standard_predictor_can_win_the_leaf() {
        let raw = [[-3, 7], [4, -8]];
        for winner in 0..14 {
            let (mut samples, options) = fixture(&raw, false);
            for (p, residuals) in samples.residuals.iter_mut().enumerate() {
                residuals.fill(token(if p == winner { 0 } else { 1 << 20 }));
            }
            for fine in [false, true] {
                let result = plan(&samples, &[0, 1], &options, fine, 1e9).unwrap();
                assert!(result.children.is_none());
                assert_eq!(result.predictor, winner);
                assert_eq!(result.bits, bits(&result.parents[winner], &options));
            }
        }
    }

    #[test]
    fn wide_columns_retain_quantizer_indices_above_255() {
        let raw = [[-4, -4], [0, 0]];
        let (mut samples, options) = fixture(&raw, true);
        samples.thresholds = vec![(-300..=300).collect(); 2];
        samples.properties = [1, 0]
            .into_iter()
            .map(|axis| {
                PropertyColumn::Wide(
                    raw.iter()
                        .map(|value| {
                            samples.thresholds[0].partition_point(|&t| value[axis] > t) as u32
                        })
                        .collect(),
                )
            })
            .collect();
        samples.counts.fill(1000);
        for (p, residuals) in samples.residuals.iter_mut().enumerate() {
            for (i, residual) in residuals.iter_mut().enumerate() {
                *residual = token(if p == i { 0 } else { 1 << 20 });
            }
        }
        let result = plan(&samples, &[0, 1], &options, true, 1.0).unwrap();
        assert!(result.children.as_ref().unwrap().0.bin > 255);
        verify(&result, &mut samples, &mut [0, 1], 1.0);
    }

    #[test]
    fn borrowed_subtree_columns_preserve_local_indices_and_weighted_parents() {
        let raw: Vec<_> = (-12..=12).map(|i| [i * 9, 19 - i * 7]).collect();
        let (mut samples, options) = fixture(&raw, false);
        let expected = parent_histograms(&samples, &(2..raw.len() - 2).collect::<Vec<_>>());
        let views = samples.ranges_mut(&[2..raw.len() - 2]).unwrap();
        let indices: Vec<_> = (0..views[0].counts.len()).collect();
        let result = plan(&views[0], &indices, &options, true, 1.0).unwrap();
        assert_eq!(result.records, indices.len());
        assert_eq!(result.parents, expected);
        assert_eq!(result.parents, parent_histograms(&views[0], &indices));
    }

    #[test]
    fn missing_axes_empty_records_and_empty_halves_are_pruned_gracefully() {
        for properties in [vec![], vec![10], vec![11], vec![10, 11]] {
            let raw = [[-4, -4], [0, 0]];
            let (mut samples, mut options) = fixture(&raw, false);
            let original = options.properties.clone();
            samples.properties = properties
                .iter()
                .map(|id| {
                    samples.properties[original.iter().position(|p| p == id).unwrap()].clone()
                })
                .collect();
            samples.thresholds = vec![QUANTIZER.to_vec(); properties.len()];
            options.properties = properties;
            samples.counts.fill(1000);
            for (p, residuals) in samples.residuals.iter_mut().enumerate() {
                for (i, residual) in residuals.iter_mut().enumerate() {
                    *residual = token(if p == i { 0 } else { 1 << 20 });
                }
            }
            let result = plan(&samples, &[0, 1], &options, true, 1.0).unwrap();
            assert_eq!(result.children.is_some(), !options.properties.is_empty());
            verify(&result, &mut samples, &mut [0, 1], 1.0);
            let empty = plan(&samples, &[], &options, true, 1.0).unwrap();
            assert_eq!(empty.records, 0);
            assert!(empty.children.is_none());
            verify(&empty, &mut samples, &mut [], 1.0);
        }
    }
}
