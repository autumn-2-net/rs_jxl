// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Match eligibility/tie rules derived from BSD-3-Clause libjxl;
// see docs/PORTING_KERNELS.md. The SIMD experiment is native Rust.

//! Exact experimental extensions for independent fast-matcher candidates.

use super::super::super::extension::Extension;
use super::super::extension_receipt::LocalReceipt;
use super::{MIN_MATCH, Match, SourceConfig};
use crate::Result;

pub(super) struct FastExtension {
    kernel: Extension,
    batch: bool,
    receipt: LocalReceipt,
}

impl FastExtension {
    pub(super) fn new(config: &SourceConfig) -> Result<Option<Self>> {
        Ok(Extension::new(config.match_extension)?.map(|kernel| Self {
            kernel,
            batch: config.batch_fast_matches,
            receipt: LocalReceipt::new(config.match_extension_receipt.clone(), false),
        }))
    }

    pub(super) fn find<F: FnOnce() -> u32>(
        &mut self,
        values: &[u32],
        position: usize,
        limit: usize,
        best: &mut Match,
        candidates: impl Iterator<Item = (usize, F)>,
    ) {
        if limit < MIN_MATCH {
            return;
        }
        let current = &values[position + MIN_MATCH..position + limit];
        let mut pending = [(0usize, 0u32); 4];
        let mut count = 0;
        for (distance, symbol) in candidates {
            let previous = position - distance;
            if values[position] != values[previous]
                || values[position + 2] != values[previous + 2]
                || values[position + 1] != values[previous + 1]
            {
                continue;
            }
            let symbol = symbol();
            let required = required_length(best, symbol);
            if required > limit
                || (required > MIN_MATCH
                    && values[position + required - 1] != values[previous + required - 1])
            {
                continue;
            }
            if !self.batch {
                let length = MIN_MATCH
                    + self
                        .kernel
                        .length_u32(current, &values[previous + MIN_MATCH..previous + limit]);
                self.receipt.extend(1, false);
                update_best(best, distance, symbol, length);
                continue;
            }
            pending[count] = (distance, symbol);
            count += 1;
            if count == 4 {
                let histories = pending.map(|(distance, _)| {
                    &values[position - distance + MIN_MATCH..position - distance + limit]
                });
                let lengths = self.kernel.lengths4_u32(current, histories);
                self.receipt.extend(4, true);
                // Filtering against the old best may do extra work, but cannot
                // discard a winner. Reduce in original order to preserve ties.
                for ((distance, symbol), length) in pending.into_iter().zip(lengths) {
                    update_best(best, distance, symbol, MIN_MATCH + length);
                }
                count = 0;
            }
        }
        for &(distance, symbol) in &pending[..count] {
            let previous = position - distance;
            let length = MIN_MATCH
                + self
                    .kernel
                    .length_u32(current, &values[previous + MIN_MATCH..previous + limit]);
            self.receipt.extend(1, false);
            update_best(best, distance, symbol, length);
        }
    }
}

fn required_length(best: &Match, symbol: u32) -> usize {
    if best.length == 0 {
        MIN_MATCH
    } else {
        best.length + usize::from(symbol >= best.symbol)
    }
}

fn update_best(best: &mut Match, distance: usize, symbol: u32, length: usize) {
    if length >= required_length(best, symbol) {
        *best = Match {
            length,
            distance,
            symbol,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::lz77::MatchExtension;

    #[test]
    fn batch_preserves_first_winner_and_distance_ties() {
        for width in [1, 2, 3, 7, 31] {
            let values: Vec<u32> = (0..500)
                .map(|i| ((i % width) + (i / 117) % 2) as u32)
                .collect();
            for position in 1..values.len() {
                for limit in [2, 3, 4, 8, 17, 64] {
                    let limit = limit.min(values.len() - position);
                    let candidates: Vec<_> = (1..=position.min(31))
                        .map(|distance| (distance, (distance % 5) as u32))
                        .collect();
                    let mut expected = Match::default();
                    for &(distance, symbol) in &candidates {
                        super::super::consider_fast_match(
                            &values,
                            position,
                            limit,
                            distance,
                            super::super::SourceFastRanges::default(),
                            &mut expected,
                            || symbol,
                        );
                    }
                    for batch in [false, true] {
                        let config = SourceConfig {
                            match_extension: MatchExtension::Scalar,
                            batch_fast_matches: batch,
                            ..Default::default()
                        };
                        let mut actual = Match::default();
                        FastExtension::new(&config).unwrap().unwrap().find(
                            &values,
                            position,
                            limit,
                            &mut actual,
                            candidates
                                .iter()
                                .map(|&(distance, symbol)| (distance, move || symbol)),
                        );
                        assert_eq!(
                            (actual.length, actual.distance, actual.symbol),
                            (expected.length, expected.distance, expected.symbol)
                        );
                    }
                }
            }
        }
    }
}
