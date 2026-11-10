//! Opt-in, bounded JXL-native profitability scout. This is an experimental
//! routing heuristic, not an optimal parser or a safe pruning certificate.
//! Sparse samples can miss repeats; negative sparse evidence keeps `Full`.
//! Prices use the existing raw Costs model, not an external compressor.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::super::{SPECIAL_DISTANCES, special_distance};
use super::{Costs, DistanceCodes, Literal, MIN_MATCH, SourceConfig, WINDOW_SIZE, reserved};
use crate::Result;

const MAX_STREAMS: usize = 8;
const WINDOWS: usize = 4;
const WINDOW_TOKENS: usize = 512;
const DENSE_TOKENS: usize = WINDOWS * WINDOW_TOKENS;
const HASH_SLOTS: usize = 1024;
const MAX_ANCHORS: usize = 2048;
const MAX_COMPARISONS: usize = 1 << 20;
const BIN_ENDS: [usize; 8] = [1, 8, 64, 512, 4096, 32768, 262144, WINDOW_SIZE];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceScoutMode {
    /// No sampling, timing, allocation, or receipt updates.
    #[default]
    Off,
    /// Compute a recommendation; the caller must not apply it to routing.
    Observe,
    /// Permit the caller to apply the experimental recommendation.
    Route,
}

#[derive(Clone, Debug, Default)]
pub struct SourceScout {
    pub mode: SourceScoutMode,
    pub receipt: Option<Arc<SourceScoutReceipt>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceScoutDecision {
    #[default]
    Full,
    Literal,
    Fast,
}

/// Strength of sampled evidence, NOT a statistical confidence probability.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceScoutConfidence {
    #[default]
    Insufficient,
    Low,
    Strong,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceScoutReason {
    #[default]
    Off,
    SmallSample,
    UnsampledStreams,
    WorkLimit,
    SparseNegative,
    MixedEvidence,
    DenseNoRepeats,
    LocalSavings,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SourceScoutDistanceBin {
    /// Inclusive upper physical distance; the preceding bin's end is exclusive.
    pub max_distance: usize,
    pub matches: u64,
    pub tokens: u64,
    pub savings_bits: f64,
}

/// Calls, *_calls and total_elapsed_ns are cumulative; other fields describe
/// the most recent probe. Route counters count recommendations in BOTH modes,
/// not applied dispatches: Observe never authorizes changing the parse route.
/// Savings are positive, disjoint SAMPLE gains under the supplied raw model;
/// they are neither extrapolated whole-input savings nor a bitstream size.
#[derive(Clone, Debug)]
pub struct SourceScoutSnapshot {
    pub calls: u64,
    pub full_calls: u64,
    pub literal_calls: u64,
    pub fast_calls: u64,
    pub total_elapsed_ns: u64,
    pub mode: SourceScoutMode,
    pub decision: SourceScoutDecision,
    pub confidence: SourceScoutConfidence,
    pub reason: SourceScoutReason,
    pub input_tokens: u64,
    pub input_streams: usize,
    pub sampled_streams: usize,
    pub sampled_windows: usize,
    pub sampled_tokens: u64,
    /// Examined destination tokens / input tokens, not dictionary coverage.
    pub coverage: f64,
    pub useful_matches: u64,
    pub matched_tokens: u64,
    pub spatial_match_tokens: u64,
    pub match_coverage: f64,
    pub literal_bits: f64,
    pub savings_bits: f64,
    /// Number of candidates whose checked exact prefix reaches MIN_MATCH.
    pub exact_match_candidates: u64,
    pub sparse_anchors: u64,
    pub compared_values: u64,
    pub all_streams_sampled: bool,
    pub all_tokens_sampled: bool,
    pub work_limited: bool,
    /// Excludes Costs::raw and caller dispatch; no timer runs in Off mode.
    pub elapsed_ns: u64,
    pub distance_bins: [SourceScoutDistanceBin; 8],
}

impl Default for SourceScoutSnapshot {
    fn default() -> Self {
        Self {
            calls: 0,
            full_calls: 0,
            literal_calls: 0,
            fast_calls: 0,
            total_elapsed_ns: 0,
            mode: SourceScoutMode::Off,
            decision: SourceScoutDecision::Full,
            confidence: SourceScoutConfidence::Insufficient,
            reason: SourceScoutReason::Off,
            input_tokens: 0,
            input_streams: 0,
            sampled_streams: 0,
            sampled_windows: 0,
            sampled_tokens: 0,
            coverage: 0.0,
            useful_matches: 0,
            matched_tokens: 0,
            spatial_match_tokens: 0,
            match_coverage: 0.0,
            literal_bits: 0.0,
            savings_bits: 0.0,
            exact_match_candidates: 0,
            sparse_anchors: 0,
            compared_values: 0,
            all_streams_sampled: false,
            all_tokens_sampled: false,
            work_limited: false,
            elapsed_ns: 0,
            distance_bins: BIN_ENDS.map(|max_distance| SourceScoutDistanceBin {
                max_distance,
                ..SourceScoutDistanceBin::default()
            }),
        }
    }
}

/// One lock per completed probe. Cloned configs can share a receipt safely;
/// snapshots are coherent and the latest completed probe wins.
#[derive(Debug, Default)]
pub struct SourceScoutReceipt {
    snapshot: Mutex<SourceScoutSnapshot>,
}

impl SourceScoutReceipt {
    pub fn snapshot(&self) -> SourceScoutSnapshot {
        self.snapshot
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn record(&self, mut sample: SourceScoutSnapshot) {
        let mut snapshot = self.snapshot.lock().unwrap_or_else(|p| p.into_inner());
        sample.calls = snapshot.calls.saturating_add(1);
        sample.full_calls = snapshot
            .full_calls
            .saturating_add(u64::from(sample.decision == SourceScoutDecision::Full));
        sample.literal_calls = snapshot
            .literal_calls
            .saturating_add(u64::from(sample.decision == SourceScoutDecision::Literal));
        sample.fast_calls = snapshot
            .fast_calls
            .saturating_add(u64::from(sample.decision == SourceScoutDecision::Fast));
        sample.total_elapsed_ns = snapshot.total_elapsed_ns.saturating_add(sample.elapsed_ns);
        *snapshot = sample;
    }
}

#[derive(Clone, Copy)]
struct Anchor {
    position: usize,
    key: [u32; MIN_MATCH],
}

impl Anchor {
    const EMPTY: Self = Self {
        position: usize::MAX,
        key: [0; MIN_MATCH],
    };
}

struct Sparse {
    slots: Vec<[Anchor; 2]>,
}

impl Sparse {
    fn new() -> Result<Self> {
        let mut slots = reserved(HASH_SLOTS)?;
        slots.resize(HASH_SLOTS, [Anchor::EMPTY; 2]);
        Ok(Self { slots })
    }

    fn slot(key: [u32; MIN_MATCH]) -> usize {
        let hash = key.into_iter().fold(0x9e37_79b9u32, |hash, value| {
            (hash ^ value).rotate_left(7).wrapping_mul(0x85eb_ca6b)
        });
        (hash as usize) & (HASH_SLOTS - 1)
    }

    fn insert(&mut self, position: usize, key: [u32; MIN_MATCH]) {
        let slot = &mut self.slots[Self::slot(key)];
        // Retain an early anchor as well as a recent one. These nominations
        // are deliberately lossy; key equality never establishes a full match.
        if slot[0].position == usize::MAX {
            slot[0] = Anchor { position, key };
        }
        slot[1] = Anchor { position, key };
    }

    fn get(&self, key: [u32; MIN_MATCH]) -> [Anchor; 2] {
        self.slots[Self::slot(key)]
    }
}

fn prefix(input: &[Literal], position: usize) -> Option<[u32; MIN_MATCH]> {
    let values = input.get(position..position.checked_add(MIN_MATCH)?)?;
    Some(std::array::from_fn(|i| values[i].value))
}

struct Spatial {
    distances: [usize; 25],
    len: usize,
}

impl Spatial {
    fn new(width: usize, input_len: usize) -> Self {
        let mut result = Self {
            distances: [0; 25],
            len: 0,
        };
        result.push(1, input_len);
        if width != 0 {
            // Native first-ring offsets include W, W+-1, 2W, 2W+-1 and
            // narrow-width aliases. Signed/clamped arithmetic is shared with
            // DistanceCodes, including widths near usize::MAX.
            for (symbol, _) in SPECIAL_DISTANCES.iter().take(24).enumerate() {
                let distance = special_distance(symbol, width);
                if distance <= WINDOW_SIZE as i128 {
                    result.push(distance as usize, input_len);
                }
            }
        } else {
            for distance in [2, 3, 4, 8] {
                result.push(distance, input_len);
            }
        }
        result
    }

    fn push(&mut self, distance: usize, input_len: usize) {
        if distance > 0
            && distance < input_len
            && distance <= WINDOW_SIZE
            && !self.distances[..self.len].contains(&distance)
        {
            self.distances[self.len] = distance;
            self.len += 1;
        }
    }

    fn values(&self) -> &[usize] {
        &self.distances[..self.len]
    }
}

#[derive(Default)]
struct Budget {
    used: usize,
    exhausted: bool,
}

impl Budget {
    fn length(
        &mut self,
        input: &[Literal],
        position: usize,
        distance: usize,
        limit: usize,
    ) -> usize {
        debug_assert!((1..=position.min(WINDOW_SIZE)).contains(&distance));
        let mut length = 0;
        while length < limit {
            if self.used == MAX_COMPARISONS {
                self.exhausted = true;
                break;
            }
            self.used += 1;
            // Only exact u32 values match. Contexts do not participate in
            // equality, and distance < length is legal overlapping LZ77 copy.
            if input[position + length].value != input[position - distance + length].value {
                break;
            }
            length += 1;
        }
        length
    }
}

fn gain(
    sums: &[f32],
    offset: usize,
    length: usize,
    context: u32,
    symbol: u32,
    costs: &Costs,
    config: &SourceConfig,
) -> f32 {
    let literal_bits = sums[offset + length] - sums[offset];
    // Keep the source f32 length+distance addition. Distance uses default420
    // even with a different literal uint; Costs::distance owns that rule.
    let match_bits = costs.length(context, length, config) + costs.distance(symbol);
    literal_bits - match_bits
}

#[derive(Clone, Copy, Default)]
struct Offer {
    length: usize,
    distance: usize,
    gain: f32,
    spatial: bool,
}

#[derive(Default)]
struct Evidence {
    dense_no_repeats: bool,
    strong_local: bool,
}

fn sample_stream(
    input: &[Literal],
    width: usize,
    costs: &Costs,
    config: &SourceConfig,
    budget: &mut Budget,
    sample: &mut SourceScoutSnapshot,
) -> Result<Evidence> {
    let codes = DistanceCodes::new(width, WINDOW_SIZE);
    let spatial = Spatial::new(width, input.len());
    let mut sparse = Sparse::new()?;
    let prefixes = input.len().saturating_sub(MIN_MATCH - 1);
    let stride = prefixes.div_ceil(MAX_ANCHORS).max(1);
    for position in (0..prefixes).step_by(stride) {
        sparse.insert(position, prefix(input, position).expect("bounded prefix"));
        sample.sparse_anchors += 1;
    }

    let complete = input.len() <= DENSE_TOKENS;
    // This small exact dictionary is a negative-evidence veto only. Sparse
    // hash misses must not manufacture a Literal recommendation. Include
    // triples crossing adjacent sample windows in the veto.
    let mut dense = BTreeMap::new();
    let mut repeated_prefix = false;
    let before_tokens = sample.sampled_tokens;
    let mut strong_windows = 0;
    let count = if complete {
        input.len().div_ceil(WINDOW_TOKENS)
    } else {
        WINDOWS
    };
    for window in 0..count {
        if budget.exhausted {
            break;
        }
        let begin = if complete {
            window * WINDOW_TOKENS
        } else {
            let span = input.len() - WINDOW_TOKENS;
            span / (WINDOWS - 1) * window + span % (WINDOWS - 1) * window / (WINDOWS - 1)
        };
        let end = (begin + WINDOW_TOKENS).min(input.len());
        let sums = costs.cumulative(&input[begin..end], config)?;
        let mut position = begin;
        let mut local_tokens = 0;
        let mut window_gain = 0.0f64;
        while position < end && !budget.exhausted {
            let key = prefix(input, position);
            if complete && let Some(key) = key {
                repeated_prefix |= dense.insert(key, position).is_some();
            }
            let limit = (end - position).min(config.max_match_length);
            let mut best = Offer::default();
            if limit >= MIN_MATCH {
                let mut consider = |distance: usize, is_spatial: bool| {
                    if budget.exhausted || distance > position || distance > WINDOW_SIZE {
                        return;
                    }
                    let length = budget.length(input, position, distance, limit);
                    if length < MIN_MATCH {
                        return;
                    }
                    sample.exact_match_candidates += 1;
                    // Longest checked prefix only, not an optimal length search.
                    let saving = gain(
                        &sums,
                        position - begin,
                        length,
                        input[position].context,
                        codes.symbol(distance),
                        costs,
                        config,
                    );
                    if saving.is_finite() && saving > best.gain {
                        best = Offer {
                            length,
                            distance,
                            gain: saving,
                            spatial: is_spatial,
                        };
                    }
                };
                for &distance in spatial.values() {
                    consider(distance, true);
                }
                if let Some(key) = key {
                    let anchors = sparse.get(key);
                    for (index, anchor) in anchors.into_iter().enumerate() {
                        if index != 0 && anchor.position == anchors[0].position {
                            continue;
                        }
                        if anchor.position < position && anchor.key == key {
                            let distance = position - anchor.position;
                            if !spatial.values().contains(&distance) {
                                consider(distance, false);
                            }
                        }
                    }
                }
            }
            if let Some(key) = key {
                sparse.insert(position, key);
            }
            if best.length >= MIN_MATCH {
                let saving = f64::from(best.gain);
                sample.useful_matches += 1;
                sample.matched_tokens += best.length as u64;
                sample.spatial_match_tokens += if best.spatial { best.length as u64 } else { 0 };
                sample.savings_bits += saving;
                window_gain += saving;
                local_tokens += if best.spatial { best.length } else { 0 };
                let bin = BIN_ENDS
                    .iter()
                    .position(|&end| best.distance <= end)
                    .unwrap();
                sample.distance_bins[bin].matches += 1;
                sample.distance_bins[bin].tokens += best.length as u64;
                sample.distance_bins[bin].savings_bits += saving;
                // Samples are disjoint, and accepted destination spans are
                // consumed once. Competing overlapping gains are not summed.
                position += best.length;
            } else {
                position += 1;
            }
        }
        let examined = position - begin;
        sample.sampled_tokens += examined as u64;
        sample.literal_bits += f64::from(sums[examined]);
        sample.sampled_windows += usize::from(examined != 0);
        if examined == end - begin
            && local_tokens * 5 >= examined * 3
            && window_gain > (f64::from(sums[examined]) * 0.20).max(32.0)
        {
            strong_windows += 1;
        }
    }
    Ok(Evidence {
        dense_no_repeats: complete
            && !repeated_prefix
            && sample.sampled_tokens - before_tokens == input.len() as u64,
        strong_local: count != 0 && strong_windows * 4 >= count * 3,
    })
}

/// Call with the raw, unmerged input streams and their actual widths, after
/// validation and Costs::raw, before seed/index work. Missing widths mean zero.
/// Both Observe and Route return the recommendation; only Route may apply it.
pub(super) fn probe(
    inputs: &[&[Literal]],
    widths: &[usize],
    costs: &Costs,
    config: &SourceConfig,
) -> Result<SourceScoutDecision> {
    if config.scout.mode == SourceScoutMode::Off {
        return Ok(SourceScoutDecision::Full);
    }
    let started = Instant::now();
    let mut sample = SourceScoutSnapshot {
        mode: config.scout.mode,
        input_tokens: inputs
            .iter()
            .fold(0u64, |n, input| n.saturating_add(input.len() as u64)),
        input_streams: inputs.iter().filter(|input| !input.is_empty()).count(),
        reason: SourceScoutReason::SmallSample,
        ..SourceScoutSnapshot::default()
    };
    let mut budget = Budget::default();
    let mut dense_no_repeats = true;
    let mut strong_local = true;
    for (index, input) in inputs
        .iter()
        .enumerate()
        .filter(|(_, input)| !input.is_empty())
    {
        if sample.sampled_streams == MAX_STREAMS || budget.exhausted {
            break;
        }
        let evidence = sample_stream(
            input,
            widths.get(index).copied().unwrap_or(0),
            costs,
            config,
            &mut budget,
            &mut sample,
        )?;
        sample.sampled_streams += 1;
        dense_no_repeats &= evidence.dense_no_repeats;
        // Streams too short for any length-three backward copy are neutral.
        strong_local &= input.len() <= MIN_MATCH || evidence.strong_local;
    }
    sample.compared_values = budget.used as u64;
    sample.work_limited = budget.exhausted;
    sample.all_streams_sampled = sample.sampled_streams == sample.input_streams;
    sample.all_tokens_sampled = sample.sampled_tokens == sample.input_tokens;
    if sample.input_tokens != 0 {
        sample.coverage = sample.sampled_tokens as f64 / sample.input_tokens as f64;
    }
    if sample.sampled_tokens != 0 {
        sample.match_coverage = sample.matched_tokens as f64 / sample.sampled_tokens as f64;
    }
    if sample.work_limited {
        sample.reason = SourceScoutReason::WorkLimit;
    } else if !sample.all_streams_sampled {
        sample.reason = SourceScoutReason::UnsampledStreams;
    } else if sample.sampled_tokens < DENSE_TOKENS as u64 {
        sample.reason = SourceScoutReason::SmallSample;
    } else if dense_no_repeats && sample.all_tokens_sampled && sample.exact_match_candidates == 0 {
        sample.decision = SourceScoutDecision::Literal;
        sample.confidence = SourceScoutConfidence::Strong;
        sample.reason = SourceScoutReason::DenseNoRepeats;
    } else if strong_local
        && sample.spatial_match_tokens as f64 >= sample.sampled_tokens as f64 * 0.60
        && sample.savings_bits > (sample.literal_bits * 0.20).max(256.0)
    {
        sample.decision = SourceScoutDecision::Fast;
        sample.confidence = SourceScoutConfidence::Strong;
        sample.reason = SourceScoutReason::LocalSavings;
    } else {
        sample.confidence = SourceScoutConfidence::Low;
        sample.reason = if sample.useful_matches == 0 && !sample.all_tokens_sampled {
            SourceScoutReason::SparseNegative
        } else {
            SourceScoutReason::MixedEvidence
        };
    }
    sample.elapsed_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let decision = sample.decision;
    if let Some(receipt) = &config.scout.receipt {
        receipt.record(sample);
    }
    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::super::{Command, HybridUintConfig};
    use super::*;

    fn tokens(values: impl IntoIterator<Item = u32>) -> Vec<Literal> {
        values
            .into_iter()
            .enumerate()
            .map(|(i, value)| Literal {
                context: if i % 2 == 0 { u32::MAX } else { 0x8000_0001 },
                value,
            })
            .collect()
    }

    fn config(mode: SourceScoutMode) -> SourceConfig {
        SourceConfig {
            scout: SourceScout {
                mode,
                receipt: Some(Arc::new(SourceScoutReceipt::default())),
            },
            ..SourceConfig::default()
        }
    }

    fn observe(
        inputs: &[&[Literal]],
        widths: &[usize],
        config: &SourceConfig,
    ) -> SourceScoutSnapshot {
        let costs = Costs::raw(inputs, config);
        let decision = probe(inputs, widths, &costs, config).unwrap();
        let sample = config.scout.receipt.as_ref().unwrap().snapshot();
        assert_eq!(decision, sample.decision);
        assert_eq!(
            sample.calls,
            sample.full_calls + sample.literal_calls + sample.fast_calls
        );
        assert!(sample.sampled_tokens <= sample.input_tokens);
        assert!(sample.matched_tokens <= sample.sampled_tokens);
        assert!(sample.compared_values <= MAX_COMPARISONS as u64);
        assert!(sample.savings_bits <= sample.literal_bits + 0.01);
        assert_eq!(
            sample
                .distance_bins
                .iter()
                .map(|bin| bin.tokens)
                .sum::<u64>(),
            sample.matched_tokens
        );
        sample
    }

    #[test]
    fn off_does_not_touch_receipt() {
        let config = config(SourceScoutMode::Off);
        let input = tokens([1, 2, 3]);
        let costs = Costs::raw(&[&input], &config);
        assert_eq!(
            probe(&[&input], &[], &costs, &config).unwrap(),
            SourceScoutDecision::Full
        );
        let sample = config.scout.receipt.as_ref().unwrap().snapshot();
        assert_eq!(sample.calls, 0);
        assert_eq!(sample.elapsed_ns, 0);
        assert_eq!(sample.sampled_tokens, 0);
        assert_eq!(SourceScout::default().mode, SourceScoutMode::Off);
    }

    #[test]
    fn dense_random_and_sparse_random_have_different_negative_confidence() {
        let config = config(SourceScoutMode::Observe);
        // Deterministic full-width xorshift noise and sparse contexts must not
        // be truncated into fake repeats.
        let mut state = 0x1234_5678u32;
        let input = tokens((0..16_384).map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        }));
        let dense = observe(&[&input[..DENSE_TOKENS]], &[37], &config);
        assert_eq!(dense.decision, SourceScoutDecision::Literal);
        assert_eq!(dense.coverage, 1.0);
        assert_eq!(dense.savings_bits, 0.0);
        let sparse = observe(&[&input], &[37], &config);
        assert_eq!(sparse.decision, SourceScoutDecision::Full);
        assert_eq!(sparse.reason, SourceScoutReason::SparseNegative);
        assert!(sparse.coverage < 1.0);
        assert_eq!(sparse.calls, 2);
        assert_eq!(sparse.full_calls, 1);
        assert_eq!(sparse.literal_calls, 1);
        assert_eq!(sparse.fast_calls, 0);
        assert!(sparse.total_elapsed_ns >= sparse.elapsed_ns);
    }

    #[test]
    fn repetitive_and_wide_rle_recommend_fast_in_both_modes() {
        let periodic = tokens((0..8193).map(|i| 0x1234_0000 + (i % 31) as u32));
        let rle = tokens(std::iter::repeat_n(u32::MAX, 8193));
        for mode in [SourceScoutMode::Observe, SourceScoutMode::Route] {
            let config = config(mode);
            for (input, width) in [(&periodic, 31), (&rle, 1)] {
                let sample = observe(&[input], &[width], &config);
                assert_eq!(sample.decision, SourceScoutDecision::Fast);
                assert!(sample.match_coverage > 0.9);
                assert!(sample.savings_bits > 0.0);
                assert!(sample.spatial_match_tokens > 0);
                assert!(!sample.work_limited);
            }
            assert_eq!(
                config.scout.receipt.as_ref().unwrap().snapshot().fast_calls,
                2
            );
        }
    }

    #[test]
    fn cross_window_triples_veto_dense_negative_recommendations() {
        let mut input = tokens((0..DENSE_TOKENS).map(|i| 0xf000_0000 + i as u32));
        for offset in 0..MIN_MATCH {
            input[WINDOW_TOKENS - 2 + offset].value = input[40 + offset].value;
        }
        let sample = observe(&[&input], &[37], &config(SourceScoutMode::Observe));
        assert_eq!(sample.coverage, 1.0);
        assert_eq!(sample.exact_match_candidates, 0);
        assert_eq!(sample.decision, SourceScoutDecision::Full);
        assert_eq!(sample.reason, SourceScoutReason::MixedEvidence);
    }

    #[test]
    fn zero_price_repetition_exhausts_work_without_inventing_savings() {
        let input = tokens(std::iter::repeat_n(0, 16_384));
        let sample = observe(&[&input], &[37], &config(SourceScoutMode::Observe));
        assert_eq!(sample.literal_bits, 0.0);
        assert_eq!(sample.savings_bits, 0.0);
        assert!(sample.work_limited);
        assert_eq!(sample.reason, SourceScoutReason::WorkLimit);
        assert_eq!(sample.decision, SourceScoutDecision::Full);
    }

    #[test]
    fn widths_follow_original_stream_indices_including_empty_streams() {
        let input = tokens((0..8193).map(|i| 0x1234_0000 + (i % 31) as u32));
        let sample = observe(&[&[], &input], &[1, 31], &config(SourceScoutMode::Observe));
        assert_eq!(sample.decision, SourceScoutDecision::Fast);
        assert_eq!(sample.input_streams, 1);
        assert_eq!(sample.sampled_streams, 1);
        assert!(sample.spatial_match_tokens as f64 > sample.sampled_tokens as f64 * 0.9);
    }

    #[test]
    fn configured_length_limit_caps_every_counted_offer() {
        let config = SourceConfig {
            max_match_length: MIN_MATCH,
            ..config(SourceScoutMode::Observe)
        };
        let input = tokens(std::iter::repeat_n(u32::MAX, 8193));
        let sample = observe(&[&input], &[1], &config);
        assert_eq!(sample.decision, SourceScoutDecision::Fast);
        assert_eq!(
            sample.matched_tokens,
            sample.useful_matches * MIN_MATCH as u64
        );
    }

    #[test]
    fn odd_short_empty_and_stream_limits_are_uncertain() {
        let config = config(SourceScoutMode::Route);
        for len in [0, 1, 2, 3, 5, 17, 511, 513, 1025] {
            let input = tokens((0..len).map(|i| 0xf000_0000 + i as u32));
            let sample = observe(&[&[], &input], &[], &config);
            assert_eq!(sample.decision, SourceScoutDecision::Full);
        }
        let input = tokens(std::iter::repeat_n(u32::MAX, 2048));
        let inputs = vec![input.as_slice(); MAX_STREAMS + 1];
        let sample = observe(&inputs, &[], &config);
        assert_eq!(sample.decision, SourceScoutDecision::Full);
        assert_eq!(sample.reason, SourceScoutReason::UnsampledStreams);
        assert!(!sample.all_streams_sampled);
    }

    #[test]
    fn native_spatial_offsets_are_positive_bounded_and_canonical() {
        for width in [0, 1, 2, 37, WINDOW_SIZE, usize::MAX] {
            let spatial = Spatial::new(width, WINDOW_SIZE + 1);
            let codes = DistanceCodes::new(width, WINDOW_SIZE);
            for &distance in spatial.values() {
                assert!((1..=WINDOW_SIZE).contains(&distance));
                assert_eq!(
                    codes.symbol(distance),
                    super::super::super::distance_symbol(distance, width).unwrap()
                );
            }
            if width == 37 {
                for distance in [1, 36, 37, 38, 73, 74, 75] {
                    assert!(spatial.values().contains(&distance));
                }
            }
        }
        assert!(Spatial::new(usize::MAX, 0).values().is_empty());
        assert_eq!(Spatial::new(1, 2).values(), &[1]);
    }

    #[test]
    fn scoring_uses_extra_bits_first_context_and_default_distance_uint() {
        let config = SourceConfig {
            literal_uint_config: HybridUintConfig::new(1, 0, 0).unwrap(),
            length_uint_config: HybridUintConfig::new(4, 0, 0).unwrap(),
            min_symbol: Some(512),
            ..config(SourceScoutMode::Observe)
        };
        let input = tokens((0..200).map(|i| 0x1234_0000 + (i % 7) as u32));
        let seed = vec![vec![Command::Match {
            context: u32::MAX,
            length: 35,
            distance: 4096,
        }]];
        let costs = Costs::fitted(&seed, &[37], &config);
        let sums = costs.cumulative(&input, &config).unwrap();
        let context = input[0].context;
        let symbol = DistanceCodes::new(37, WINDOW_SIZE).symbol(4096);
        let length = config.length_uint_config.encode(35 - MIN_MATCH as u32);
        let distance = HybridUintConfig::default().encode(symbol);
        assert_ne!(
            distance.token,
            config.literal_uint_config.encode(symbol).token
        );
        let expected_length =
            length.nbits as f32 + costs.bits(context, config.min_symbol() + length.token);
        let expected_distance = distance.nbits as f32
            + costs
                .distances
                .bits
                .get(&distance.token)
                .copied()
                .unwrap_or(12.0);
        let expected = (sums[35] - sums[0]) - (expected_length + expected_distance);
        assert_eq!(
            gain(&sums, 0, 35, context, symbol, &costs, &config).to_bits(),
            expected.to_bits()
        );
        let encoded = config.literal_uint_config.encode(input[1].value);
        let expected_literal =
            costs.bits(input[1].context, encoded.token) + encoded.nbits as f32 + sums[1];
        assert_eq!(sums[2].to_bits(), expected_literal.to_bits());
    }

    #[test]
    fn exact_u32_comparison_allows_overlap_but_not_low_bit_aliases() {
        let input = tokens([
            0x1234_0001,
            0x5678_0001,
            0x1234_0001,
            0x5678_0001,
            0x1234_0001,
        ]);
        let mut budget = Budget::default();
        assert_eq!(budget.length(&input, 1, 1, 4), 0);
        assert_eq!(budget.length(&input, 2, 2, 3), 3);
        let input = tokens(std::iter::repeat_n(u32::MAX, 19));
        assert_eq!(budget.length(&input, 1, 1, 18), 18);
        budget.used = MAX_COMPARISONS;
        assert_eq!(budget.length(&input, 1, 1, 18), 0);
        assert!(budget.exhausted);
    }

    #[test]
    fn sparse_hash_finds_distant_repeats_without_calling_them_spatial() {
        let period = 4096u32;
        // Every window starts at a period boundary; the retained first sparse
        // anchor is therefore a deterministic distant-repeat nomination.
        let input = tokens(
            (0..period * 3 + WINDOW_TOKENS as u32).map(|i| (i % period).wrapping_mul(0x9e37_79b9)),
        );
        let sample = observe(&[&input], &[0], &config(SourceScoutMode::Observe));
        assert!(sample.useful_matches > 0);
        assert!(
            sample
                .distance_bins
                .iter()
                .filter(|bin| bin.max_distance >= 4096)
                .any(|bin| bin.matches > 0)
        );
        assert_eq!(sample.spatial_match_tokens, 0);
        assert_eq!(sample.decision, SourceScoutDecision::Full);
    }

    #[test]
    fn destination_savings_are_disjoint_and_histories_reset_per_stream() {
        let config = config(SourceScoutMode::Observe);
        let input = tokens(std::iter::repeat_n(u32::MAX, DENSE_TOKENS));
        let sample = observe(&[&input, &input], &[1, 37], &config);
        assert_eq!(sample.decision, SourceScoutDecision::Fast);
        assert_eq!(sample.matched_tokens, (2 * (DENSE_TOKENS - 1)) as u64);
        assert_eq!(sample.distance_bins[0].tokens, sample.matched_tokens);
        assert_eq!(sample.useful_matches, (2 * WINDOWS) as u64);
        assert_eq!(sample.coverage, 1.0);
    }
}
