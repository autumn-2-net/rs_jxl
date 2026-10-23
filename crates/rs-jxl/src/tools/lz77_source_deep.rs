// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77_deep.h::LZ77DeepPrefixChain at
//! b866b3defec13494bfbaf705db93581cda118e37. Hash collisions consume visits;
//! these are candidate proposals, never certificates of matching symbols.

use super::super::matchers::filled;
use crate::Result;

const BASE: u64 = 0x9e37_79b1_85eb_ca87;
const MIX: u64 = 0xd6e8_feb8_6659_fd93;

#[derive(Default)]
pub(super) struct DeepChain {
    previous: Vec<u32>,
}

impl DeepChain {
    pub(super) fn build(values: &[u32], prefix: usize) -> Result<Self> {
        let mut previous = filled(values.len(), u32::MAX)?;
        if prefix == 0 || values.len() < prefix {
            return Ok(Self { previous });
        }
        let mut heads = filled(1 << 16, u32::MAX)?;
        let (mut hash, mut power) = (0u64, 1u64);
        for (i, &value) in values[..prefix].iter().enumerate() {
            hash = hash.wrapping_mul(BASE).wrapping_add(u64::from(value) + 1);
            if i + 1 < prefix {
                power = power.wrapping_mul(BASE);
            }
        }
        for (pos, entry) in previous
            .iter_mut()
            .take(values.len() - prefix + 1)
            .enumerate()
        {
            let bucket = ((hash ^ (hash >> 32)).wrapping_mul(MIX) >> 48) as usize;
            *entry = heads[bucket];
            heads[bucket] = pos as u32;
            if pos + prefix < values.len() {
                hash = hash
                    .wrapping_sub((u64::from(values[pos]) + 1).wrapping_mul(power))
                    .wrapping_mul(BASE)
                    .wrapping_add(u64::from(values[pos + prefix]) + 1);
            }
        }
        Ok(Self { previous })
    }
    pub(super) fn find(
        &self,
        position: usize,
        window: usize,
        count: usize,
        mut visit: impl FnMut(usize),
    ) {
        let Some(&first) = self.previous.get(position) else {
            return;
        };
        let mut previous = first;
        for _ in 0..count {
            if previous == u32::MAX || position - previous as usize >= window {
                break;
            }
            visit(previous as usize);
            previous = self.previous[previous as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_deep_hash_equals_independent_rehash_with_full_u32_values_and_tails() {
        let values: Vec<_> = (0..391).map(|i| u32::MAX - (i % 71)).collect();
        for prefix in [0, 1, 16, 64, 392] {
            let chain = DeepChain::build(&values, prefix).unwrap();
            let mut expected = vec![u32::MAX; values.len()];
            if prefix != 0 && prefix <= values.len() {
                let mut heads = vec![u32::MAX; 1 << 16];
                for (pos, entry) in expected
                    .iter_mut()
                    .take(values.len() - prefix + 1)
                    .enumerate()
                {
                    let hash = values[pos..pos + prefix].iter().fold(0u64, |h, v| {
                        h.wrapping_mul(BASE).wrapping_add(u64::from(*v) + 1)
                    });
                    let bucket = ((hash ^ (hash >> 32)).wrapping_mul(MIX) >> 48) as usize;
                    *entry = heads[bucket];
                    heads[bucket] = pos as u32;
                }
            }
            assert_eq!(chain.previous, expected, "prefix={prefix}");
        }
    }

    #[test]
    fn deep_visits_are_exactly_four_or_eight_and_window_end_is_excluded() {
        let values = vec![u32::MAX; 129];
        for prefix in [16, 64] {
            let chain = DeepChain::build(&values, prefix).unwrap();
            for count in [4, 8] {
                for window in [1, 5, 64] {
                    let mut actual = Vec::new();
                    chain.find(64, window, count, |p| actual.push(p));
                    let expected: Vec<_> = (1..window).take(count).map(|d| 64 - d).collect();
                    assert_eq!(actual, expected);
                }
            }
            let mut tail = Vec::new();
            chain.find(129 - prefix + 1, 64, 8, |p| tail.push(p));
            assert!(tail.is_empty());
        }
    }
}
