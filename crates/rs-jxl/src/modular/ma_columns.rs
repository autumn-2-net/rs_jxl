// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! enc_ma.h::TreeSamples property storage and enc_ma.cc::SwapBatch, frozen
//! at b866b3defec13494bfbaf705db93581cda118e37. Static bins stay u32; dynamic
//! bins use u8 when their complete quantizer fits, with a wide legacy fallback.

#[cfg_attr(test, derive(Clone, Debug, Eq, PartialEq))]
pub(super) enum PropertyColumn<B = Vec<u8>, W = Vec<u32>> {
    Packed(B),
    Wide(W),
}

pub(super) trait PropertyValue: Copy {
    fn index(self) -> usize;
}

impl PropertyValue for u8 {
    #[inline]
    fn index(self) -> usize {
        usize::from(self)
    }
}

impl PropertyValue for u32 {
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

pub(super) fn swap_column<T>(column: &mut [T], left: &[usize], right: &[usize]) {
    for (&a, &b) in left.iter().zip(right) {
        column.swap(a, b);
    }
}

impl PropertyColumn {
    pub(super) fn new(property: usize, thresholds: usize) -> Self {
        if property >= 2 && thresholds < 256 {
            Self::Packed(Vec::new())
        } else {
            Self::Wide(Vec::new())
        }
    }

    // Only sample preparation inserts values. Search loops dispatch to typed
    // slices once per property, never through a per-record enum accessor.
    pub(super) fn push(&mut self, value: u32) {
        match self {
            Self::Packed(values) => {
                values.push(u8::try_from(value).expect("MA packed quantizer invariant"));
            }
            Self::Wide(values) => values.push(value),
        }
    }
}

impl<B, W> PropertyColumn<B, W>
where
    B: std::ops::DerefMut<Target = [u8]>,
    W: std::ops::DerefMut<Target = [u32]>,
{
    pub(super) fn swap_pairs(&mut self, left: &[usize], right: &[usize]) {
        match self {
            Self::Packed(values) => swap_column(values, left, right),
            Self::Wide(values) => swap_column(values, left, right),
        }
    }

    pub(super) fn count_le(&self, indices: &[usize], bin: usize) -> usize {
        fn count<T: PropertyValue>(values: &[T], indices: &[usize], bin: usize) -> usize {
            indices
                .iter()
                .filter(|&&i| values[i].index() <= bin)
                .count()
        }
        match self {
            Self::Packed(values) => count(values, indices, bin),
            Self::Wide(values) => count(values, indices, bin),
        }
    }
}

impl PropertyColumn {
    #[cfg(test)]
    pub(super) fn value(&self, index: usize) -> u32 {
        match self {
            Self::Packed(values) => u32::from(values[index]),
            Self::Wide(values) => values[index],
        }
    }

    #[cfg(test)]
    pub(super) fn to_wide(&self) -> Vec<u32> {
        match self {
            Self::Packed(values) => values.iter().map(|&v| u32::from(v)).collect(),
            Self::Wide(values) => values.clone(),
        }
    }

    #[cfg(test)]
    pub(super) fn as_ptr(&self) -> *const () {
        match self {
            Self::Packed(values) => values.as_ptr().cast(),
            Self::Wide(values) => values.as_ptr().cast(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_uses_full_quantizer_cardinality_and_keeps_static_bins_wide() {
        for bins in [1, 255, 256, 257, 513] {
            for property in [0, 1, 2, 6, 15, 17, 255] {
                let mut column = PropertyColumn::new(property, bins - 1);
                let expected: Vec<_> = (0..bins as u32).collect();
                for &value in &expected {
                    column.push(value);
                }
                assert_eq!(column.to_wide(), expected);
                match &column {
                    PropertyColumn::Packed(values) => {
                        assert!(property >= 2 && bins <= 256);
                        assert_eq!(std::mem::size_of_val(values.as_slice()), bins);
                    }
                    PropertyColumn::Wide(values) => {
                        assert!(property < 2 || bins > 256);
                        assert_eq!(std::mem::size_of_val(values.as_slice()), bins * 4);
                    }
                }
                let indices: Vec<_> = (0..bins).rev().collect();
                assert_eq!(column.count_le(&indices, bins / 2), bins / 2 + 1);
            }
        }
        // Representation follows the complete quantizer, not the observed max.
        let mut sparse = PropertyColumn::new(6, 256);
        sparse.push(0);
        assert!(matches!(sparse, PropertyColumn::Wide(_)));
    }
}
