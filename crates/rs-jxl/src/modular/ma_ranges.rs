// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! Safe disjoint column loans for enc_ma.cc channel and whole-subtree learners.

use super::{PropertyColumn, ResidualToken};
use crate::{Error, Result};
use std::ops::{Deref, DerefMut, Range};

pub(super) trait SampleStorage {
    type Bytes: DerefMut<Target = [u8]> + Send + Sync;
    type Words: DerefMut<Target = [u32]> + Send + Sync;
    type Residuals: DerefMut<Target = [ResidualToken]> + Send + Sync;
    type Counts: DerefMut<Target = [u16]> + Send + Sync;
    type Thresholds: Deref<Target = [Vec<i32>]> + Send + Sync;
    fn base(counts: &Self::Counts) -> usize;
}

pub(super) struct Owned;
impl SampleStorage for Owned {
    type Bytes = Vec<u8>;
    type Words = Vec<u32>;
    type Residuals = Vec<ResidualToken>;
    type Counts = Vec<u16>;
    type Thresholds = Vec<Vec<i32>>;
    fn base(_: &Self::Counts) -> usize {
        0
    }
}

pub(super) struct Borrowed<'a>(std::marker::PhantomData<&'a mut ()>);
pub(super) struct Counts<'a> {
    values: &'a mut [u16],
    base: usize,
}
impl Deref for Counts<'_> {
    type Target = [u16];
    fn deref(&self) -> &Self::Target {
        self.values
    }
}
impl DerefMut for Counts<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values
    }
}
impl<'a> SampleStorage for Borrowed<'a> {
    type Bytes = &'a mut [u8];
    type Words = &'a mut [u32];
    type Residuals = &'a mut [ResidualToken];
    type Counts = Counts<'a>;
    type Thresholds = &'a [Vec<i32>];
    fn base(counts: &Self::Counts) -> usize {
        counts.base
    }
}

pub(super) struct Samples<S: SampleStorage> {
    pub(super) properties: Vec<PropertyColumn<S::Bytes, S::Words>>,
    pub(super) residuals: Vec<S::Residuals>,
    pub(super) counts: S::Counts,
    pub(super) thresholds: S::Thresholds,
}
pub(super) type TreeSamples = Samples<Owned>;
pub(super) type SampleView<'a> = Samples<Borrowed<'a>>;

#[cfg(test)]
impl Clone for TreeSamples {
    fn clone(&self) -> Self {
        Self {
            properties: self.properties.clone(),
            residuals: self.residuals.clone(),
            counts: self.counts.clone(),
            thresholds: self.thresholds.clone(),
        }
    }
}

fn reserve<T>(items: &mut Vec<T>, additional: usize) -> Result<()> {
    items
        .try_reserve(additional)
        .map_err(|_| Error::Limit("MA range descriptor allocation"))
}

fn slices<'a, T>(mut values: &'a mut [T], ranges: &[Range<usize>]) -> Result<Vec<&'a mut [T]>> {
    let mut result = Vec::new();
    reserve(&mut result, ranges.len())?;
    let mut previous = 0;
    for range in ranges {
        let (_, tail) = values.split_at_mut(range.start - previous);
        let (part, tail) = tail.split_at_mut(range.len());
        result.push(part);
        values = tail;
        previous = range.end;
    }
    Ok(result)
}

impl<S: SampleStorage> Samples<S> {
    pub(super) fn base(&self) -> usize {
        S::base(&self.counts)
    }

    /// Sorted, nonoverlapping local ranges; gaps remain owned by the caller.
    /// Every returned column is a split_at_mut loan, never a copied sample.
    pub(super) fn ranges_mut(&mut self, ranges: &[Range<usize>]) -> Result<Vec<SampleView<'_>>> {
        if ranges
            .iter()
            .any(|r| r.start >= r.end || r.end > self.counts.len())
            || ranges.windows(2).any(|pair| pair[0].end > pair[1].start)
        {
            return Err(Error::InvalidInput(
                "invalid MA disjoint sample ranges".into(),
            ));
        }
        let base = self.base();
        let mut views: Vec<SampleView<'_>> = Vec::new();
        reserve(&mut views, ranges.len())?;
        for (range, counts) in ranges.iter().zip(slices(&mut self.counts, ranges)?) {
            let mut properties = Vec::new();
            let mut residuals = Vec::new();
            reserve(&mut properties, self.properties.len())?;
            reserve(&mut residuals, self.residuals.len())?;
            views.push(Samples {
                properties,
                residuals,
                counts: Counts {
                    values: counts,
                    base: base + range.start,
                },
                thresholds: &self.thresholds[..],
            });
        }
        for column in &mut self.properties {
            match column {
                PropertyColumn::Packed(values) => {
                    for (view, part) in views.iter_mut().zip(slices(values, ranges)?) {
                        view.properties.push(PropertyColumn::Packed(part));
                    }
                }
                PropertyColumn::Wide(values) => {
                    for (view, part) in views.iter_mut().zip(slices(values, ranges)?) {
                        view.properties.push(PropertyColumn::Wide(part));
                    }
                }
            }
        }
        for column in &mut self.residuals {
            for (view, part) in views.iter_mut().zip(slices(column, ranges)?) {
                view.residuals.push(part);
            }
        }
        Ok(views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> TreeSamples {
        TreeSamples {
            properties: vec![
                PropertyColumn::Packed((0..12).collect()),
                PropertyColumn::Wide((0..12).collect()),
            ],
            residuals: vec![
                (0..12)
                    .map(|token| ResidualToken {
                        token,
                        nbits: token % 3,
                    })
                    .collect(),
            ],
            counts: (1..=12).collect(),
            thresholds: vec![(0..11).collect(), (0..11).collect()],
        }
    }

    #[test]
    fn ma_ranges_disjoint_loans_share_quantizers_and_move_all_columns_without_copying() {
        let mut samples = fixture();
        let count_base = samples.counts.as_ptr() as usize;
        let residual_base = samples.residuals[0].as_ptr() as usize;
        let threshold_base = samples.thresholds.as_ptr();
        {
            let mut views = samples.ranges_mut(&[1..5, 7..12]).unwrap();
            assert_eq!(views[0].counts.as_ptr() as usize, count_base + 2);
            assert_eq!(views[1].counts.as_ptr() as usize, count_base + 14);
            assert_eq!(
                views[1].residuals[0].as_ptr() as usize,
                residual_base + 7 * std::mem::size_of::<ResidualToken>()
            );
            assert!(
                views
                    .iter()
                    .all(|view| view.thresholds.as_ptr() == threshold_base)
            );
            assert_eq!((views[0].base(), views[1].base()), (1, 7));
            for view in &mut views {
                let last = view.counts.len() - 1;
                view.counts.swap(0, last);
                view.residuals[0].swap(0, last);
                for column in &mut view.properties {
                    column.swap_pairs(&[0], &[last]);
                }
            }
            let nested = views[1].ranges_mut(&[1..3, 3..5]).unwrap();
            assert_eq!((nested[0].base(), nested[1].base()), (8, 10));
        }
        for i in 0..12 {
            let original = match i {
                1 => 4,
                4 => 1,
                7 => 11,
                11 => 7,
                _ => i,
            };
            assert_eq!(samples.counts[i], original as u16 + 1);
            assert_eq!(samples.residuals[0][i].token, original as u8);
            assert_eq!(samples.properties[0].value(i), original as u32);
            assert_eq!(samples.properties[1].value(i), original as u32);
        }
    }

    #[test]
    fn ma_ranges_reject_overlap_reverse_empty_and_out_of_bounds_before_mutation() {
        let mut samples = fixture();
        let original = samples.counts.clone();
        for ranges in [
            vec![1..4, 3..7],
            vec![7..9, 1..3],
            std::iter::once(5..5).collect(),
            std::iter::once(2..13).collect(),
        ] {
            assert!(matches!(
                samples.ranges_mut(&ranges),
                Err(Error::InvalidInput(_))
            ));
            assert_eq!(samples.counts, original);
        }
        assert!(samples.ranges_mut(&[]).unwrap().is_empty());
        let range = 0..12;
        assert_eq!(
            samples.ranges_mut(std::slice::from_ref(&range)).unwrap()[0]
                .counts
                .len(),
            12
        );
    }
}
