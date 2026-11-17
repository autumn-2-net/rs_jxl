//! Encoder-side AC histogram-group selection; no token rewriting or wire syntax.
//!
//! Context layout comes from `ac_tokens` (libjxl's enc_entropy_coder.cc and
//! ac_context.h at b866b3defec13494bfbaf705db93581cda118e37). Clustering is a
//! local heuristic, not a port of upstream histogram selection. Block IDs retain
//! the tokenizer's DC/channel conditioning; only prediction/density subcontexts
//! are pooled. Conditional entropy is a proxy, not an exact serialized cost.

use super::ac_tokens::{NONZERO_BUCKETS, ZERO_DENSITY_CONTEXTS};
use crate::entropy::{hybrid_uint::HybridUintConfig, stream::Token};
use crate::{Error, Result};

// HybridUint(4, 2, 0) maps the entire u32 domain to 0..128.
const BUCKETS: usize = 128;
const MAX_PASSES: usize = 6;
const DISTANCE_EPSILON: f64 = 1e-12;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Assignment {
    pub ids: Vec<usize>,
    pub count: usize,
}

struct Histogram {
    // Sorted (block ID * 2 + count/coefficient family, HybridUint bucket).
    bins: Vec<(usize, u64)>,
    total: f64,
}

impl Histogram {
    fn from_counts(counts: &[u64]) -> Self {
        let bins: Vec<_> = counts
            .iter()
            .enumerate()
            .filter_map(|(key, &count)| (count != 0).then_some((key, count)))
            .collect();
        let total = bins.iter().map(|&(_, count)| count as f64).sum();
        Self { bins, total }
    }

    fn pool(members: &[usize], features: &[Self], scratch: &mut [u64]) -> Result<Self> {
        scratch.fill(0);
        for &member in members {
            for &(key, count) in &features[member].bins {
                scratch[key] = scratch[key]
                    .checked_add(count)
                    .ok_or(Error::Limit("JPEG AC histogram-group counts"))?;
            }
        }
        Ok(Self::from_counts(scratch))
    }

    // Squared Hellinger distance (up to a constant), independent of stream size.
    fn distance(&self, other: &Self) -> f64 {
        let mut affinity = 0.0;
        let (mut a, mut b) = (0, 0);
        while a < self.bins.len() && b < other.bins.len() {
            let (ka, ca) = self.bins[a];
            let (kb, cb) = other.bins[b];
            match ka.cmp(&kb) {
                std::cmp::Ordering::Less => a += 1,
                std::cmp::Ordering::Greater => b += 1,
                std::cmp::Ordering::Equal => {
                    affinity += ((ca as f64) * (cb as f64)).sqrt();
                    a += 1;
                    b += 1;
                }
            }
        }
        (1.0 - affinity / (self.total * other.total).sqrt()).clamp(0.0, 1.0)
    }

    // Do not charge for block/family mixtures: those are already conditioned by
    // the original contexts. This also rejects splits solely by channel mix.
    fn entropy(&self) -> f64 {
        let mut cost = 0.0;
        let mut stratum = usize::MAX;
        let mut total = 0.0;
        for &(key, count) in &self.bins {
            if key / BUCKETS != stratum {
                cost += xlogx(total);
                total = 0.0;
                stratum = key / BUCKETS;
            }
            let count = count as f64;
            cost -= xlogx(count);
            total += count;
        }
        cost + xlogx(total)
    }

    fn split_overhead(&self) -> f64 {
        // A conservative admission margin for another model/map, plus a proxy
        // for occupied-symbol table costs. Actual contexts remain unpooled in
        // the encoder, so this cannot promise a smaller final bitstream.
        256.0 + 8.0 * self.bins.len() as f64
    }
}

fn xlogx(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value * value.log2()
    }
}

struct Group {
    members: Vec<usize>,
    histogram: Histogram,
}

struct Split {
    left: Group,
    right: Group,
    savings: f64,
}

fn propose_split(
    parent: &Group,
    features: &[Histogram],
    scratch: &mut [u64],
) -> Result<Option<Split>> {
    if parent.members.len() < 2 {
        return Ok(None);
    }
    // Largest stream first; ties and farthest-seed ties follow input order.
    let mut first = parent.members[0];
    for &member in &parent.members[1..] {
        if features[member].total > features[first].total {
            first = member;
        }
    }
    let mut second = first;
    let mut farthest = DISTANCE_EPSILON;
    for &member in &parent.members {
        let distance = features[first].distance(&features[member]);
        if distance > farthest {
            farthest = distance;
            second = member;
        }
    }
    if second == first {
        return Ok(None);
    }

    let mut left = Group {
        members: vec![first],
        histogram: Histogram::pool(&[first], features, scratch)?,
    };
    let mut right = Group {
        members: vec![second],
        histogram: Histogram::pool(&[second], features, scratch)?,
    };
    let mut sides = vec![usize::MAX; parent.members.len()];
    for _ in 0..MAX_PASSES {
        left.members.clear();
        right.members.clear();
        let mut changed = false;
        for (position, &member) in parent.members.iter().enumerate() {
            let dl = features[member].distance(&left.histogram);
            let dr = features[member].distance(&right.histogram);
            let side = if (dl - dr).abs() <= DISTANCE_EPSILON {
                // Keep existing ties stable; initially prefer the first seed.
                usize::from(sides[position] == 1)
            } else {
                usize::from(dr < dl)
            };
            changed |= side != sides[position];
            sides[position] = side;
            if side == 0 {
                left.members.push(member);
            } else {
                right.members.push(member);
            }
        }
        if left.members.is_empty() || right.members.is_empty() {
            return Ok(None);
        }
        if !changed {
            break;
        }
        left.histogram = Histogram::pool(&left.members, features, scratch)?;
        right.histogram = Histogram::pool(&right.members, features, scratch)?;
    }
    let savings = parent.histogram.entropy()
        - left.histogram.entropy()
        - right.histogram.entropy()
        - parent.histogram.split_overhead();
    Ok((savings > 0.0).then_some(Split {
        left,
        right,
        savings,
    }))
}

fn validate_token(token: &Token, contexts: usize) -> Result<()> {
    if token.is_length || token.context >= contexts {
        return Err(Error::InvalidInput(
            "JPEG histogram assignment requires raw AC literals in range".into(),
        ));
    }
    Ok(())
}

/// Assign raw AC streams to at most `requested` useful histogram groups.
///
/// IDs are dense in first-nonempty-stream order. Empty streams use ID zero;
/// empty input still reports one legal group. Inputs are never rewritten. The
/// caller offsets contexts by `id * block_contexts * (37 + 458)` afterwards.
///
/// At most six candidate bisections are tried for four groups, each with six
/// refinement passes. Per-stream features are sparse. The one reused dense
/// scratch array has only `block_contexts * 2 * 128` counters (at most 32 KiB),
/// rather than `original_contexts * 256` counters per image group.
pub(super) fn partition(
    streams: &[Vec<Token>],
    block_contexts: usize,
    requested: usize,
) -> Result<Assignment> {
    if !(1..=4).contains(&requested) {
        return Err(Error::InvalidInput(
            "JPEG AC histogram-group request must be in 1..=4".into(),
        ));
    }
    // Match BlockContextModel's validated wire domain before any arithmetic.
    if !(1..=16).contains(&block_contexts) {
        return Err(Error::InvalidInput(
            "JPEG AC block-context count must be in 1..=16".into(),
        ));
    }
    let contexts = block_contexts * (NONZERO_BUCKETS + ZERO_DENSITY_CONTEXTS);
    let mut ids = vec![0; streams.len()];
    if requested == 1 || streams.len() <= 1 {
        for token in streams.iter().flatten() {
            validate_token(token, contexts)?;
        }
        return Ok(Assignment { ids, count: 1 });
    }

    let config = HybridUintConfig::default();
    let nonzero_contexts = block_contexts * NONZERO_BUCKETS;
    let mut scratch = vec![0u64; block_contexts * 2 * BUCKETS];
    let mut features = Vec::with_capacity(streams.len());
    let mut members = Vec::new();
    for (index, stream) in streams.iter().enumerate() {
        scratch.fill(0);
        for token in stream {
            validate_token(token, contexts)?;
            let stratum = if token.context < nonzero_contexts {
                2 * (token.context % block_contexts)
            } else {
                2 * ((token.context - nonzero_contexts) / ZERO_DENSITY_CONTEXTS) + 1
            };
            let bucket = config.encode(token.value).token as usize;
            scratch[stratum * BUCKETS + bucket] += 1;
        }
        features.push(Histogram::from_counts(&scratch));
        if !stream.is_empty() {
            members.push(index);
        }
    }
    if members.len() <= 1 {
        return Ok(Assignment { ids, count: 1 });
    }
    let histogram = Histogram::pool(&members, &features, &mut scratch)?;
    let mut groups = vec![Group { members, histogram }];
    while groups.len() < requested {
        let mut best: Option<(usize, Split)> = None;
        for (index, group) in groups.iter().enumerate() {
            if let Some(split) = propose_split(group, &features, &mut scratch)? {
                if best
                    .as_ref()
                    .is_none_or(|(_, old)| split.savings > old.savings)
                {
                    best = Some((index, split));
                }
            }
        }
        let Some((index, split)) = best else {
            break;
        };
        groups[index] = split.left;
        groups.push(split.right);
    }
    groups.sort_by_key(|group| group.members[0]);
    for (id, group) in groups.iter().enumerate() {
        for &member in &group.members {
            ids[member] = id;
        }
    }
    Ok(Assignment {
        ids,
        count: groups.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated(context: usize, value: u32, count: usize) -> Vec<Token> {
        vec![Token::literal(context, value); count]
    }

    #[test]
    fn deterministic_dense_ids_and_requested_cap() {
        let streams: Vec<_> = [3, 0, 2, 1, 3, 2, 0, 1]
            .into_iter()
            .map(|value| repeated(NONZERO_BUCKETS, value, 4096))
            .collect();
        let before = streams.clone();
        for requested in 1..=4 {
            let assignment = partition(&streams, 1, requested).unwrap();
            assert_eq!(assignment, partition(&streams, 1, requested).unwrap());
            assert_eq!(assignment.count, requested);
            assert_eq!(assignment.ids.len(), streams.len());
            let mut seen = vec![false; assignment.count];
            for &id in &assignment.ids {
                assert!(id < assignment.count);
                seen[id] = true;
            }
            assert!(seen.into_iter().all(|used| used));
            assert_eq!(assignment.ids[0], 0);
            assert_eq!(assignment.ids[0], assignment.ids[4]);
            assert_eq!(assignment.ids[2], assignment.ids[5]);
        }
        assert_eq!(streams, before);
    }

    #[test]
    fn one_requested_even_for_separated_distributions() {
        let streams = vec![repeated(0, 0, 4096), repeated(0, u32::MAX, 4096)];
        assert_eq!(
            partition(&streams, 1, 1).unwrap(),
            Assignment {
                ids: vec![0, 0],
                count: 1
            }
        );
    }

    #[test]
    fn identical_distributions_ignore_token_order_and_stream_size() {
        let mut first = repeated(0, 1, 4096);
        first.extend(repeated(NONZERO_BUCKETS, 8, 4096));
        let mut second = first.repeat(2);
        second.reverse();
        let assignment = partition(&[first.clone(), second, first], 1, 4).unwrap();
        assert_eq!(
            assignment,
            Assignment {
                ids: vec![0; 3],
                count: 1
            }
        );
    }

    #[test]
    fn prediction_subcontexts_and_hybrid_extra_bits_are_pooled() {
        let coefficient_start = 2 * NONZERO_BUCKETS;
        let mut first = repeated(0, 16, 4096);
        first.extend(repeated(coefficient_start, 16, 4096));
        // Last count-prediction bucket for block zero, and last coefficient
        // subcontext for that same block. Values 16 and 19 share a HybridUint
        // token and differ only in extra bits, which histograms do not code.
        let mut second = repeated(2 * (NONZERO_BUCKETS - 1), 19, 4096);
        second.extend(repeated(
            coefficient_start + ZERO_DENSITY_CONTEXTS - 1,
            19,
            4096,
        ));
        assert_eq!(partition(&[first, second], 2, 4).unwrap().count, 1);
    }

    #[test]
    fn separated_distributions_cap_at_two_distinct_groups() {
        let a = repeated(NONZERO_BUCKETS, 0, 4096);
        let b = repeated(NONZERO_BUCKETS + ZERO_DENSITY_CONTEXTS - 1, u32::MAX, 4096);
        let assignment = partition(&[a.clone(), b.clone(), a, b], 1, 4).unwrap();
        assert_eq!(
            assignment,
            Assignment {
                ids: vec![0, 1, 0, 1],
                count: 2
            }
        );
    }

    #[test]
    fn empty_streams_and_no_streams_keep_legal_zero_id() {
        for requested in 1..=4 {
            assert_eq!(
                partition(&[], 1, requested).unwrap(),
                Assignment {
                    ids: vec![],
                    count: 1
                }
            );
            assert_eq!(
                partition(&[vec![], vec![]], 1, requested).unwrap(),
                Assignment {
                    ids: vec![0, 0],
                    count: 1
                }
            );
        }
        let a = repeated(0, 0, 4096);
        let b = repeated(0, 15, 4096);
        assert_eq!(
            partition(&[vec![], a, vec![], b, vec![]], 1, 4).unwrap(),
            Assignment {
                ids: vec![0, 0, 0, 1, 0],
                count: 2
            }
        );
        assert_eq!(
            partition(&[vec![], repeated(0, 1, 4)], 1, 4).unwrap().count,
            1
        );
    }

    #[test]
    fn small_splits_do_not_pay_for_another_model() {
        let streams = vec![repeated(0, 0, 16), repeated(0, 15, 16)];
        assert_eq!(partition(&streams, 1, 4).unwrap().count, 1);
    }

    #[test]
    fn block_and_family_conditioning_survive_pooling() {
        // Swapping values across block IDs must not disappear into a global
        // histogram. Cover both modulo count contexts and divided coefficients.
        for (a, b) in [
            (0, 1),
            (
                2 * NONZERO_BUCKETS,
                2 * NONZERO_BUCKETS + ZERO_DENSITY_CONTEXTS,
            ),
        ] {
            let mut first = repeated(a, 0, 4096);
            first.extend(repeated(b, 15, 4096));
            let mut second = repeated(a, 15, 4096);
            second.extend(repeated(b, 0, 4096));
            assert_eq!(partition(&[first, second], 2, 4).unwrap().count, 2);
        }
        let mut first = repeated(0, 0, 4096);
        first.extend(repeated(NONZERO_BUCKETS, 15, 4096));
        let mut second = repeated(0, 15, 4096);
        second.extend(repeated(NONZERO_BUCKETS, 0, 4096));
        assert_eq!(partition(&[first, second], 1, 4).unwrap().count, 2);
    }

    #[test]
    fn channel_mix_alone_has_no_conditional_coding_gain() {
        let mut first = repeated(0, 0, 4096);
        first.extend(repeated(1, 15, 1024));
        let mut second = repeated(0, 0, 1024);
        second.extend(repeated(1, 15, 4096));
        assert_eq!(partition(&[first, second], 2, 4).unwrap().count, 1);
    }

    #[test]
    fn rejects_invalid_bounds_contexts_and_length_tokens() {
        for requested in [0, 5, usize::MAX] {
            assert!(partition(&[], 1, requested).is_err());
        }
        for blocks in [0, 17, usize::MAX] {
            assert!(partition(&[], blocks, 1).is_err());
        }
        for requested in [1, 4] {
            for blocks in [1, 2, 16] {
                let contexts = blocks * (NONZERO_BUCKETS + ZERO_DENSITY_CONTEXTS);
                assert!(
                    partition(
                        &[vec![Token::literal(contexts - 1, u32::MAX)]],
                        blocks,
                        requested
                    )
                    .is_ok()
                );
                for context in [contexts, usize::MAX] {
                    // Two streams exercise feature extraction as well as the
                    // requested-one/single-stream validation fast paths.
                    let invalid = vec![Token::literal(context, 0)];
                    assert!(partition(&[invalid.clone()], blocks, requested).is_err());
                    assert!(partition(&[vec![], invalid], blocks, requested).is_err());
                }
                let invalid = vec![Token {
                    context: 0,
                    value: 1,
                    is_length: true,
                }];
                assert!(partition(&[invalid.clone()], blocks, requested).is_err());
                assert!(partition(&[vec![], invalid], blocks, requested).is_err());
            }
        }
    }
}
