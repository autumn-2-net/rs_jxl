//! Encoder-only rate estimates. These do not introduce decoder state.

use crate::entropy::HybridUintConfig;
use crate::modular::tree::pack_signed;

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(super) struct Histogram {
    counts: [u64; 128],
    extra: u64,
    pub count: u64,
    sum: i128,
    // A causal estimate, used only to propose a STATIC standard MA offset.
    bias16: i64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            counts: [0; 128],
            extra: 0,
            count: 0,
            sum: 0,
            bias16: 0,
        }
    }
}

impl Histogram {
    #[inline]
    pub fn add(&mut self, residual: i32, adaptive: bool) {
        let token = HybridUintConfig::default().encode(pack_signed(residual));
        self.counts[token.token as usize] += 1;
        self.extra += u64::from(token.nbits);
        self.count += 1;
        self.sum += i128::from(residual);
        if adaptive {
            self.bias16 += i64::from(residual) - self.bias16 / 16;
        }
    }

    pub fn merge(&mut self, other: &Self) {
        for (to, from) in self.counts.iter_mut().zip(other.counts) {
            *to += from;
        }
        self.extra += other.extra;
        self.count += other.count;
        self.sum += other.sum;
    }

    pub fn mean(&self) -> i32 {
        if self.count == 0 {
            0
        } else {
            (self.sum / i128::from(self.count)) as i32
        }
    }

    pub fn recent_bias(&self) -> i32 {
        (self.bias16 / 16) as i32
    }

    pub fn bits(&self, offset: i32) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let n = self.count as f64;
        let entropy = n * n.log2()
            - self
                .counts
                .iter()
                .filter(|&&n| n != 0)
                .map(|&n| n as f64 * (n as f64).log2())
                .sum::<f64>();
        let occupied = self.counts.iter().filter(|&&n| n != 0).count();
        // Deliberately an estimate: final clustering/LZ77/UintConfig selection
        // can change these costs. The benchmark measures the real codestream.
        let offset_bits = if offset == 0 {
            0
        } else {
            32 - pack_signed(offset).leading_zeros()
        };
        entropy + self.extra as f64 + 24.0 + occupied as f64 * 3.0 + f64::from(offset_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_extremes_and_merge_preserve_mass() {
        let values = [i32::MIN, i32::MAX, -17, 0, 42];
        let mut whole = Histogram::default();
        let mut merged = Histogram::default();
        for value in values {
            whole.add(value, true);
            let mut part = Histogram::default();
            part.add(value, false);
            merged.merge(&part);
        }
        assert_eq!(whole.counts, merged.counts);
        assert_eq!(whole.extra, merged.extra);
        assert_eq!(whole.mean(), merged.mean());
        assert!(whole.bits(0).is_finite());
    }

    #[test]
    fn static_offset_can_remove_extra_bits() {
        let mut original = Histogram::default();
        let mut shifted = Histogram::default();
        for _ in 0..1000 {
            original.add(5000, true);
            shifted.add(0, false);
        }
        assert_eq!(original.mean(), 5000);
        assert!(shifted.bits(5000) < original.bits(0));
        assert!((original.recent_bias() - 5000).abs() <= 1);
    }
}
