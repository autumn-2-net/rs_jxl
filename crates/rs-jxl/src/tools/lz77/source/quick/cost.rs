//! Candidate ranking under the existing source seed model, not final ANS cost.

use super::{DoubleHash, MIN_MATCH, Match, static_cost_with_penalty};

impl DoubleHash<'_> {
    pub(super) fn find_priced(
        &mut self,
        position: usize,
        sums: &[f32],
        penalty: f32,
        require_gain: bool,
    ) -> (Match, bool) {
        let mut best = Match::default();
        let mut rejected = Match::default();
        let mut legal = false;
        let mut best_gain = if require_gain { 0.0 } else { f32::NEG_INFINITY };
        // A zero-priced suffix cannot yield positive savings. Avoid extending
        // many identical zero runs merely to rediscover that fact.
        if !require_gain || sums[self.values.len()] > sums[position] {
            self.origins(position, |previous| {
                let mut matched = Match::default();
                self.consider(position, previous, &mut matched);
                if matched.length >= MIN_MATCH {
                    legal = true;
                    if matched.length > rejected.length
                        || (matched.length == rejected.length && matched.symbol < rejected.symbol)
                    {
                        rejected = matched;
                    }
                    rank_prefixes(
                        matched,
                        position,
                        sums,
                        penalty,
                        require_gain,
                        &mut best,
                        &mut best_gain,
                    );
                }
            });
        }
        // Keep the source rejection skip for long, cheap-to-code runs. Returning
        // no match here would repeatedly extend the same unprofitable region.
        // For ungated methods too, prefer consuming the long rejected run over
        // repeatedly rediscovering it to emit the least-negative short prefix.
        // The caller still checks the original cost gate, including equality.
        if best_gain <= 0.0 && legal {
            best = rejected;
        }
        self.update_range(position, position + 1);
        (best, legal)
    }
}

fn rank_prefixes(
    matched: Match,
    position: usize,
    sums: &[f32],
    penalty: f32,
    require_gain: bool,
    best: &mut Match,
    best_gain: &mut f32,
) {
    let mut score = |length| {
        let candidate = Match { length, ..matched };
        let gain =
            sums[position + length] - sums[position] - static_cost_with_penalty(candidate, penalty);
        if gain.is_finite()
            && (!require_gain || gain > 0.0)
            && (gain > *best_gain
                || (gain == *best_gain
                    && (length > best.length
                        || (length == best.length && candidate.symbol < best.symbol))))
        {
            *best = candidate;
            *best_gain = gain;
        }
    };
    score(matched.length);
    // The source seed uses HybridUint(1,0,0) on length - 3. Its length cost
    // is constant within each interval. Nonnegative literal prices mean only
    // interval endpoints can maximize local gain; this is not global DP.
    let mut power = 1usize;
    while let Some(length) = power.checked_add(MIN_MATCH - 1) {
        if length >= matched.length {
            break;
        }
        score(length);
        let Some(next) = power.checked_mul(2) else {
            break;
        };
        power = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_endpoints_equal_exhaustive_source_seed_gain() {
        for length in [3, 4, 5, 7, 9, 10, 11, 17, 18, 19, 129, 1027] {
            let matched = Match {
                length,
                distance: 4096,
                symbol: 4215,
            };
            for offset in [0, 13] {
                for price_kind in 0..4 {
                    let mut sums = vec![0.0f32; offset + length + 1];
                    for i in 0..offset + length {
                        let price = match price_kind {
                            0 => 0.0,
                            1 => 16.0,
                            2 => {
                                if i % 7 == 0 {
                                    23.25
                                } else {
                                    0.0
                                }
                            }
                            _ => (i % 5) as f32 * 0.125,
                        };
                        sums[i + 1] = sums[i] + price;
                    }
                    for require_gain in [false, true] {
                        let mut best = Match::default();
                        let mut gain = if require_gain { 0.0 } else { f32::NEG_INFINITY };
                        rank_prefixes(
                            matched,
                            offset,
                            &sums,
                            6.0,
                            require_gain,
                            &mut best,
                            &mut gain,
                        );
                        let mut expected = Match::default();
                        let mut expected_gain = if require_gain { 0.0 } else { f32::NEG_INFINITY };
                        for n in MIN_MATCH..=length {
                            let candidate = Match {
                                length: n,
                                ..matched
                            };
                            let candidate_gain = sums[offset + n]
                                - sums[offset]
                                - static_cost_with_penalty(candidate, 6.0);
                            if (!require_gain || candidate_gain > 0.0)
                                && candidate_gain >= expected_gain
                            {
                                expected = candidate;
                                expected_gain = candidate_gain;
                            }
                        }
                        assert_eq!(best, expected, "length={length} price={price_kind}");
                        assert_eq!(gain, expected_gain);
                    }
                }
            }
        }
    }
}
