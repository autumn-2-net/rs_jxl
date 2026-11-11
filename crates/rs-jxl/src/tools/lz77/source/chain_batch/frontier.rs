// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Compact SymbolId specialization of source index::Dominance. Only entries
//! below size are live; overflow requires the caller's complete demand requery.

use super::SPECULATIVE_MATCH;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub(super) length: u32,
    pub(super) symbol: u32,
}

#[derive(Clone)]
pub(super) struct Frontier {
    pub(super) entries: [Entry; 16],
    pub(super) size: usize,
    pub(super) overflowed: bool,
}

impl Frontier {
    pub(super) fn new() -> Self {
        Self {
            entries: [Entry::default(); 16],
            size: 0,
            overflowed: false,
        }
    }

    pub(super) fn clear(&mut self) {
        self.size = 0;
        self.overflowed = false;
    }

    pub(super) fn add(&mut self, length: usize, symbol: u32) {
        if length > SPECULATIVE_MATCH {
            self.overflowed = true;
            return;
        }
        let length = length as u32;
        let mut begin = 0;
        // f32(symbol) is monotonic; its equal-cost ties use the integer symbol.
        while begin < self.size && self.entries[begin].symbol < symbol {
            if self.entries[begin].length >= length {
                return;
            }
            begin += 1;
        }
        if begin < self.size
            && self.entries[begin].symbol == symbol
            && self.entries[begin].length >= length
        {
            return;
        }
        let mut end = begin;
        while end < self.size && self.entries[end].length <= length {
            end += 1;
        }
        let remaining = self.size - (end - begin);
        if remaining == self.entries.len() {
            self.overflowed = true;
            return;
        }
        self.entries.copy_within(end..self.size, begin);
        self.entries.copy_within(begin..remaining, begin + 1);
        self.entries[begin] = Entry { length, symbol };
        self.size = remaining + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::index::Dominance;
    use super::*;

    fn equivalent(actual: &Frontier, expected: &Dominance) {
        assert_eq!(actual.size, expected.size);
        assert_eq!(actual.overflowed, expected.overflowed);
        for (actual, expected) in actual.entries[..actual.size]
            .iter()
            .zip(&expected.entries[..expected.size])
        {
            assert_eq!(actual.length as usize, expected.length);
            assert_eq!(actual.symbol, expected.symbol);
        }
    }

    fn add_both(actual: &mut Frontier, expected: &mut Dominance, length: usize, symbol: u32) {
        actual.add(length, symbol);
        expected.add(length, symbol, symbol as f32);
        equivalent(actual, expected);
    }

    #[test]
    fn random_symbol_frontiers_equal_general_dominance_after_every_update() {
        let mut actual = Frontier::new();
        let mut expected = Dominance::new();
        let mut state = 0x163277bdu32;
        for cycle in 0..64 {
            actual.clear();
            expected.clear();
            equivalent(&actual, &expected);
            for step in 0..257 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let symbol = match step % 7 {
                    0 => 0,
                    1 => u32::MAX,
                    2 => (1 << 24) + (state & 7),
                    3 => state & 15,
                    _ => state,
                };
                let length = (state as usize).wrapping_add(cycle) % (SPECULATIVE_MATCH + 1);
                add_both(&mut actual, &mut expected, length, symbol);
            }
        }
    }

    #[test]
    fn adversarial_ties_float_rounding_and_replacements_preserve_order() {
        let symbols = [
            0,
            1,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            (1 << 24) + 2,
            u32::MAX - 2,
            u32::MAX - 1,
            u32::MAX,
        ];
        assert_eq!((1u32 << 24) as f32, ((1u32 << 24) + 1) as f32);
        assert_eq!((u32::MAX - 1) as f32, u32::MAX as f32);
        for reverse in [false, true] {
            let mut actual = Frontier::new();
            let mut expected = Dominance::new();
            for length in [0, 3, 3, 8, 7, 127, 128, 128, 1] {
                for i in 0..symbols.len() {
                    let symbol = symbols[if reverse { symbols.len() - 1 - i } else { i }];
                    add_both(&mut actual, &mut expected, length, symbol);
                }
            }
        }
        for (low, high) in [(1 << 24, (1 << 24) + 1), (u32::MAX - 1, u32::MAX)] {
            for reverse in [false, true] {
                let mut actual = Frontier::new();
                let mut expected = Dominance::new();
                let pair = if reverse {
                    [(4, high), (3, low)]
                } else {
                    [(3, low), (4, high)]
                };
                for (length, symbol) in pair {
                    add_both(&mut actual, &mut expected, length, symbol);
                }
                assert_eq!(actual.size, 2);
                add_both(&mut actual, &mut expected, 4, low);
                assert_eq!(actual.size, 1);
            }
        }
    }

    #[test]
    fn capacity_overflow_stays_set_after_later_dominating_insertions() {
        let mut actual = Frontier::new();
        let mut expected = Dominance::new();
        for i in 0..16 {
            add_both(&mut actual, &mut expected, i + 3, i as u32);
        }
        assert_eq!(actual.size, 16);
        assert!(!actual.overflowed);
        add_both(&mut actual, &mut expected, 19, 16);
        assert_eq!(actual.size, 16);
        assert!(actual.overflowed);
        add_both(&mut actual, &mut expected, 128, 0);
        assert_eq!(actual.size, 1);
        assert!(actual.overflowed);
    }

    #[test]
    fn clear_resets_live_state_without_zeroing_the_backing_entries() {
        let mut actual = Frontier::new();
        let mut expected = Dominance::new();
        for count in [7, 0, 17, 2, 0, 16, 1] {
            let backing = actual.entries;
            actual.clear();
            expected.clear();
            assert_eq!(actual.entries, backing);
            equivalent(&actual, &expected);
            for i in 0..count {
                add_both(&mut actual, &mut expected, i + 3, i as u32);
            }
            equivalent(&actual.clone(), &expected);
        }
    }

    #[test]
    fn oversized_lengths_require_fallback_without_truncation_or_insertion() {
        let mut actual = Frontier::new();
        actual.add(128, 7);
        assert!(!actual.overflowed);
        let backing = actual.entries;
        for length in [SPECULATIVE_MATCH + 1, u32::MAX as usize, usize::MAX] {
            actual.add(length, 0);
            assert!(actual.overflowed);
            assert_eq!(actual.size, 1);
            assert_eq!(actual.entries, backing);
        }
        actual.clear();
        actual.add(128, u32::MAX);
        assert!(!actual.overflowed);
        assert_eq!(actual.entries[0].length, 128);
        assert_eq!(actual.entries[0].symbol, u32::MAX);
    }

    #[test]
    fn compact_frontier_uses_substantially_less_storage_than_general_dominance() {
        assert!(size_of::<Entry>() <= size_of::<super::super::super::index::Entry>());
        assert!(size_of::<Frontier>() <= size_of::<Dominance>() * 3 / 4);
    }
}
