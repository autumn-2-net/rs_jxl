// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from the BSD-3-Clause sources identified in docs/PORTING_KERNELS.md.
// The complete upstream notice and license are retained in that document.

//! Mandatory-three-token rejection filters, not a match finder or encoder.
//!
//! Used by the legacy and source LZ77 parsers. Native prefix SIMD is active
//! when their filter policy enables it; the source default disables filtering.
//! Equal tags are only candidates: always compare the actual tokens afterwards.
//! Optional long-prefix tokens must never contribute to a mandatory-prefix tag.

use std::fmt;

pub const MANDATORY_PREFIX: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixError {
    PositionOutOfBounds,
    CandidateNotPrevious,
    DistancesNotStrictlyIncreasing,
    InvalidBucketCapacity,
    TooManyTags,
}

impl fmt::Display for PrefixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PositionOutOfBounds => "token position is out of bounds",
            Self::CandidateNotPrevious => "candidate must precede the query",
            Self::DistancesNotStrictlyIncreasing => {
                "distances must be positive and strictly increasing"
            }
            Self::InvalidBucketCapacity => "bucket capacity must be a power of two in 1..=64",
            Self::TooManyTags => "a fingerprint mask holds at most 64 tags",
        })
    }
}

impl std::error::Error for PrefixError {}

/// The frozen fork's `GetHash<3>`, with explicit modulo-2^32 arithmetic.
/// Missing prefixes return `None`, not a zero tag that could reject a short match.
pub fn prefix_hash(tokens: &[u32], position: usize) -> Option<u32> {
    let end = position.checked_add(MANDATORY_PREFIX)?;
    let prefix = tokens.get(position..end)?;
    let mut hash = 0u32;
    for &token in prefix {
        hash ^= token
            .wrapping_add(0x9e37_79b9)
            .wrapping_add(hash << 6)
            .wrapping_add(hash >> 2);
    }
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2_ae35);
    Some(hash ^ (hash >> 16))
}

/// Bit i corresponds to tags[i]. Empty slices are valid; no padding is read.
/// Runtime-detected x86_64 AVX2/SSE2 with a scalar fallback on other targets.
pub fn fingerprint_mask(tags: &[u8], wanted: u8) -> Result<u64, PrefixError> {
    if tags.len() > 64 {
        return Err(PrefixError::TooManyTags);
    }
    Ok(super::prefix_simd::fingerprint_mask(tags, wanted))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    Disabled,
    Pending,
    Active,
    /// Enough work was observed, but less than half the stream remains.
    SkippedLate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilterPolicy {
    Disabled,
    Eager,
    Lazy,
}

/// One 16-distance group. Bit i describes distance `maximum_distance - i`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecialPrefixGroup {
    maximum_distance: usize,
    valid: u16,
}

impl SpecialPrefixGroup {
    pub fn maximum_distance(&self) -> usize {
        self.maximum_distance
    }

    pub fn valid_mask(&self) -> u16 {
        self.valid
    }

    fn eligible_mask(&self, position: usize, safe_distance: usize) -> u16 {
        let mut mask = 0;
        for bit in 0..16 {
            if self.valid & (1 << bit) == 0 {
                continue;
            }
            let distance = self.maximum_distance - bit;
            if distance > safe_distance && distance <= position {
                mask |= 1 << bit;
            }
        }
        mask
    }
}

/// Groups positive, unique, ascending physical distances without reordering.
/// Callers must canonicalize width-dependent duplicates to their smallest
/// distance symbol first. Empty input is valid. No image-width arithmetic occurs
/// here, and even distances near `usize::MAX` cannot overflow group construction.
pub fn group_distances(distances: &[usize]) -> Result<Vec<SpecialPrefixGroup>, PrefixError> {
    let mut groups: Vec<SpecialPrefixGroup> = Vec::new();
    let mut previous = 0;
    for &distance in distances {
        if distance <= previous {
            return Err(PrefixError::DistancesNotStrictlyIncreasing);
        }
        previous = distance;
        if groups.last().is_none_or(|g| distance > g.maximum_distance) {
            groups.push(SpecialPrefixGroup {
                maximum_distance: distance.saturating_add(15),
                valid: 0,
            });
        }
        let group = groups.last_mut().expect("group was just created");
        group.valid |= 1 << (group.maximum_distance - distance);
    }
    Ok(groups)
}

/// Cache and feedback are tied to one immutable token stream. Lazy activation
/// uses actual candidate work, independently of sampled profiling or query count.
pub struct PrefixFilter<'a> {
    tokens: &'a [u32],
    tags: Vec<u8>,
    state: Activation,
    candidate_work: u64,
}

impl<'a> PrefixFilter<'a> {
    pub fn new(tokens: &'a [u32], policy: FilterPolicy) -> Self {
        let mut filter = Self {
            tokens,
            tags: Vec::new(),
            state: match policy {
                FilterPolicy::Disabled => Activation::Disabled,
                FilterPolicy::Eager => Activation::Active,
                FilterPolicy::Lazy => Activation::Pending,
            },
            candidate_work: 0,
        };
        if policy == FilterPolicy::Eager {
            filter.build();
        }
        filter
    }

    pub fn activation(&self) -> Activation {
        self.state
    }

    pub fn cached_bytes(&self) -> usize {
        self.tags.len()
    }

    pub fn candidate_work(&self) -> u64 {
        self.candidate_work
    }

    pub fn activation_threshold(&self) -> u64 {
        4096.max((self.tokens.len() / 2) as u64)
    }

    /// Record eligible candidate work while pending, not the number of matches.
    /// Saturation prevents wrapping feedback from postponing activation.
    pub fn record_candidate_work(&mut self, count: u64) {
        if self.state == Activation::Pending {
            self.candidate_work = self.candidate_work.saturating_add(count);
        }
    }

    /// Call before recording the current query's work, as in the frozen fork.
    /// Passing the end position is valid; it cannot activate a late cache.
    pub fn begin_query(&mut self, position: usize) -> Result<Activation, PrefixError> {
        if position > self.tokens.len() {
            return Err(PrefixError::PositionOutOfBounds);
        }
        if self.state == Activation::Pending && self.candidate_work >= self.activation_threshold() {
            if position < self.tokens.len() / 2 {
                self.build();
            } else {
                self.state = Activation::SkippedLate;
            }
        }
        Ok(self.state)
    }

    /// False proves unequal mandatory prefixes; true does not prove a match.
    /// Short minimum lengths and incomplete tail prefixes bypass rejection.
    pub fn may_match(
        &self,
        position: usize,
        candidate: usize,
        minimum_match: usize,
    ) -> Result<bool, PrefixError> {
        if position >= self.tokens.len() || candidate >= self.tokens.len() {
            return Err(PrefixError::PositionOutOfBounds);
        }
        if candidate >= position {
            return Err(PrefixError::CandidateNotPrevious);
        }
        Ok(
            !self.can_filter(position, minimum_match)
                || self.tags[position] == self.tags[candidate],
        )
    }

    /// Visit eligible physical distances in ascending order. Group boundaries
    /// and filtering never change survivor order. The caller still checks exact
    /// tokens, match lengths, window limits, and distance-symbol tie breaks.
    /// Groups should be the unmodified output of `group_distances`.
    pub fn visit_special(
        &mut self,
        position: usize,
        safe_distance: usize,
        minimum_match: usize,
        groups: &[SpecialPrefixGroup],
        mut visit: impl FnMut(usize),
    ) -> Result<(), PrefixError> {
        if position >= self.tokens.len() {
            return Err(PrefixError::PositionOutOfBounds);
        }
        self.begin_query(position)?;
        let filter = self.can_filter(position, minimum_match);
        for group in groups {
            let mut mask = group.eligible_mask(position, safe_distance);
            self.record_candidate_work(u64::from(mask.count_ones()));
            if filter && mask != 0 && position >= group.maximum_distance {
                // As in enc_lz77_bucket.cc::SpecialPrefixMasks, lane i is
                // candidate start+i. Groups start at a positive distance + 15,
                // so start+16 <= position: no C++-style cache padding is needed.
                let start = position - group.maximum_distance;
                mask &= super::prefix_simd::fingerprint_mask(
                    &self.tags[start..start + 16],
                    self.tags[position],
                ) as u16;
            } else if filter {
                let mut matches = 0;
                for bit in 0..16 {
                    if mask & (1 << bit) != 0 {
                        let candidate = position - (group.maximum_distance - bit);
                        if self.tags[candidate] == self.tags[position] {
                            matches |= 1 << bit;
                        }
                    }
                }
                mask = matches;
            }
            while mask != 0 {
                let bit = 15 - mask.leading_zeros() as usize;
                visit(group.maximum_distance - bit);
                mask &= !(1 << bit);
            }
        }
        Ok(())
    }

    fn can_filter(&self, position: usize, minimum_match: usize) -> bool {
        self.state == Activation::Active
            && minimum_match >= MANDATORY_PREFIX
            && self.tokens.len() - position >= MANDATORY_PREFIX
    }

    fn build(&mut self) {
        self.tags = vec![0; self.tokens.len()];
        for position in 0..self.tokens.len().saturating_sub(2) {
            self.tags[position] = (prefix_hash(self.tokens, position)
                .expect("complete three-token prefix")
                >> 24) as u8;
        }
        self.state = Activation::Active;
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BucketStats {
    pub scanned: u64,
    pub retained: u64,
    pub uniform_queries: u64,
    pub mixed_queries: u64,
}

/// One bounded recent-position bucket. Its caller owns bucket-key selection.
/// Tags are retained from the first insert, even when queries bypass filtering.
/// A long-prefix bucket must insert/query the separate three-token hash, never
/// the optional long-prefix key. Eviction policy is unchanged by filtering.
pub struct RecentBucket<const N: usize> {
    positions: [usize; N],
    tags: [u8; N],
    next: usize,
    size: usize,
    common: u8,
    mixed: bool,
    stats: BucketStats,
}

impl<const N: usize> RecentBucket<N> {
    pub fn new() -> Result<Self, PrefixError> {
        if !N.is_power_of_two() || N > 64 {
            return Err(PrefixError::InvalidBucketCapacity);
        }
        Ok(Self {
            positions: [0; N],
            tags: [0; N],
            next: 0,
            size: 0,
            common: 0,
            mixed: false,
            stats: BucketStats::default(),
        })
    }

    pub fn insert(&mut self, position: usize, mandatory_prefix_hash: u32) {
        let tag = (mandatory_prefix_hash >> 24) as u8;
        if self.size == 0 {
            self.common = tag;
        }
        self.mixed |= tag != self.common;
        self.tags[self.next] = tag;
        self.positions[self.next] = position;
        self.next = (self.next + 1) & (N - 1);
        self.size = (self.size + 1).min(N);
    }

    pub fn oldest_if_full(&self) -> Option<usize> {
        (self.size == N).then(|| self.positions[self.next])
    }

    pub fn stats(&self) -> BucketStats {
        self.stats
    }

    /// Visit newest first, including across ring wrap. `None` bypasses tags.
    /// The caller must bypass filtering when no complete prefix is available.
    pub fn visit(
        &mut self,
        mandatory_prefix_hash: Option<u32>,
        minimum_match: usize,
        mut visit: impl FnMut(usize),
    ) {
        let valid = if self.size == 64 {
            u64::MAX
        } else {
            (1u64 << self.size) - 1
        };
        let mut mask = valid;
        if let Some(hash) = mandatory_prefix_hash.filter(|_| minimum_match >= 3 && self.size != 0) {
            let wanted = (hash >> 24) as u8;
            if self.mixed {
                mask = fingerprint_mask(&self.tags, wanted).expect("capacity checked") & valid;
                self.stats.mixed_queries = self.stats.mixed_queries.saturating_add(1);
            } else {
                mask = if wanted == self.common { valid } else { 0 };
                self.stats.uniform_queries = self.stats.uniform_queries.saturating_add(1);
            }
            self.stats.scanned = self.stats.scanned.saturating_add(self.size as u64);
            self.stats.retained = self
                .stats
                .retained
                .saturating_add(u64::from(mask.count_ones()));
            if mask == valid {
                self.common = wanted;
                self.mixed = false;
            }
        }
        // enc_lz77_bucket.h::ForEach bypasses the bit walk for unfiltered/all-hit
        // queries, including partial buckets and a full 64-entry ring.
        if mask == valid {
            for i in 0..self.size {
                visit(self.positions[self.next.wrapping_sub(1 + i) & (N - 1)]);
            }
            return;
        }
        let before = (1u64 << self.next) - 1;
        for mut bits in [mask & before, mask & !before] {
            while bits != 0 {
                let index = 63 - bits.leading_zeros() as usize;
                visit(self.positions[index]);
                bits &= !(1u64 << index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_hash(prefix: &[u32]) -> u32 {
        let mut h = 0u64;
        for &token in prefix {
            h ^= (u64::from(token) + 0x9e37_79b9 + (h << 6) + (h >> 2)) & 0xffff_ffff;
        }
        h ^= h >> 16;
        h = (h * 0x85eb_ca6b) & 0xffff_ffff;
        h ^= h >> 13;
        h = (h * 0xc2b2_ae35) & 0xffff_ffff;
        (h ^ (h >> 16)) as u32
    }

    #[test]
    fn hashes_wrap_and_tails_are_not_prefixes() {
        for triple in [[0; 3], [u32::MAX; 3], [1, u32::MAX, 7], [0x8000_0000, 4, 9]] {
            assert_eq!(prefix_hash(&triple, 0), Some(reference_hash(&triple)));
        }
        for len in 0..3 {
            assert_eq!(prefix_hash(&vec![0; len], 0), None);
        }
        assert_eq!(prefix_hash(&[1, 2, 3], 1), None);
        assert_eq!(prefix_hash(&[1, 2, 3], usize::MAX), None);
    }

    #[test]
    fn mask_boundaries() {
        assert_eq!(fingerprint_mask(&[], 0), Ok(0));
        assert_eq!(fingerprint_mask(&[7; 64], 7), Ok(u64::MAX));
        assert_eq!(fingerprint_mask(&[7; 64], 8), Ok(0));
        assert_eq!(fingerprint_mask(&[7; 65], 7), Err(PrefixError::TooManyTags));
        let mut tags = [0; 64];
        tags[63] = 1;
        assert_eq!(fingerprint_mask(&tags, 1), Ok(1u64 << 63));
    }

    #[test]
    fn public_mask_matches_scalar_for_every_length_and_byte() {
        let tags: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(73));
        for len in 0..=64 {
            for wanted in 0..=u8::MAX {
                let mut expected = 0;
                for (i, &tag) in tags[..len].iter().enumerate() {
                    if tag == wanted {
                        expected |= 1u64 << i;
                    }
                }
                assert_eq!(fingerprint_mask(&tags[..len], wanted), Ok(expected));
            }
        }
    }

    #[test]
    fn collisions_survive_and_exact_comparison_rejects() {
        let mut seen = [None; 256];
        let (a, b) = (0..=256u32)
            .find_map(|value| {
                let tag = (prefix_hash(&[value, 8, 9], 0).unwrap() >> 24) as usize;
                seen[tag].replace(value).map(|previous| (previous, value))
            })
            .expect("257 prefixes must collide in 256 tags");
        let tokens = [a, 8, 9, b, 8, 9];
        let filter = PrefixFilter::new(&tokens, FilterPolicy::Eager);
        assert!(filter.may_match(3, 0, 3).unwrap());
        assert_ne!(tokens[..3], tokens[3..]);
    }

    #[test]
    fn minimum_and_tail_bypass_and_invalid_bounds() {
        let tokens = [10, 11, 12, 13, 14];
        let filter = PrefixFilter::new(&tokens, FilterPolicy::Eager);
        for minimum in 0..3 {
            assert!(filter.may_match(1, 0, minimum).unwrap());
        }
        assert!(filter.may_match(4, 0, 3).unwrap());
        assert_eq!(
            filter.may_match(5, 0, 3),
            Err(PrefixError::PositionOutOfBounds)
        );
        assert_eq!(
            filter.may_match(1, usize::MAX, 3),
            Err(PrefixError::PositionOutOfBounds)
        );
        assert_eq!(
            filter.may_match(1, 1, 3),
            Err(PrefixError::CandidateNotPrevious)
        );
        let mut empty = PrefixFilter::new(&[], FilterPolicy::Lazy);
        assert_eq!(empty.begin_query(0), Ok(Activation::Pending));
        assert_eq!(empty.begin_query(1), Err(PrefixError::PositionOutOfBounds));
    }

    #[test]
    fn activation_depends_on_prior_work_and_remaining_input() {
        let data = vec![0; 20_000];
        let mut filter = PrefixFilter::new(&data, FilterPolicy::Lazy);
        assert_eq!(filter.activation_threshold(), 10_000);
        filter.record_candidate_work(9999);
        assert_eq!(filter.begin_query(5), Ok(Activation::Pending));
        assert_eq!(filter.cached_bytes(), 0);
        filter.record_candidate_work(1);
        assert_eq!(filter.begin_query(9999), Ok(Activation::Active));
        assert_eq!(filter.cached_bytes(), data.len());
        let mut late = PrefixFilter::new(&data, FilterPolicy::Lazy);
        late.record_candidate_work(u64::MAX);
        late.record_candidate_work(1);
        assert_eq!(late.candidate_work(), u64::MAX);
        assert_eq!(late.begin_query(10_000), Ok(Activation::SkippedLate));
        assert_eq!(late.begin_query(0), Ok(Activation::SkippedLate));
        assert_eq!(late.cached_bytes(), 0);
        let mut disabled = PrefixFilter::new(&data, FilterPolicy::Disabled);
        disabled.record_candidate_work(u64::MAX);
        assert_eq!(disabled.begin_query(0), Ok(Activation::Disabled));
        let small = PrefixFilter::new(&[0; 12], FilterPolicy::Lazy);
        assert_eq!(small.activation_threshold(), 4096);
    }

    #[test]
    fn grouped_scan_matches_scalar_at_narrow_widths_and_tails() {
        let tokens: Vec<_> = (0..160).map(|i| (i % 7) as u32).collect();
        for width in [0usize, 1, 2, 3, 15, 16, 17, 32] {
            let mut distances: Vec<_> = (0..8)
                .flat_map(|y| (0..8).map(move |x| y * width + x + 1))
                .collect();
            distances.sort_unstable();
            distances.dedup();
            let groups = group_distances(&distances).unwrap();
            let mut filter = PrefixFilter::new(&tokens, FilterPolicy::Eager);
            for position in 0..tokens.len() {
                for safe in [0, 1, 15, 16, position, usize::MAX] {
                    for minimum in [1, 3] {
                        let expected: Vec<_> = distances
                            .iter()
                            .copied()
                            .filter(|&d| {
                                d > safe
                                    && d <= position
                                    && filter.may_match(position, position - d, minimum).unwrap()
                            })
                            .collect();
                        let mut actual = Vec::new();
                        filter
                            .visit_special(position, safe, minimum, &groups, |d| actual.push(d))
                            .unwrap();
                        assert_eq!(actual, expected);
                        for d in distances
                            .iter()
                            .copied()
                            .filter(|&d| d > safe && d <= position)
                        {
                            if position + 3 <= tokens.len()
                                && tokens[position..position + 3]
                                    == tokens[position - d..position - d + 3]
                            {
                                assert!(actual.contains(&d));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn group_bounds_and_work_feedback() {
        assert!(group_distances(&[]).unwrap().is_empty());
        for distances in [&[0][..], &[1, 1], &[2, 1]] {
            assert_eq!(
                group_distances(distances),
                Err(PrefixError::DistancesNotStrictlyIncreasing)
            );
        }
        let huge = group_distances(&[usize::MAX - 3, usize::MAX]).unwrap();
        assert_eq!(huge[0].maximum_distance(), usize::MAX);
        assert_eq!(huge[0].valid_mask(), 9);
        let groups = group_distances(&[1, 2, 16, 17, 32]).unwrap();
        let mut filter = PrefixFilter::new(&[0; 100], FilterPolicy::Lazy);
        let mut seen = Vec::new();
        filter
            .visit_special(20, 2, 3, &groups, |d| seen.push(d))
            .unwrap();
        assert_eq!(seen, [16, 17]);
        assert_eq!(filter.candidate_work(), 2);
        filter.record_candidate_work(4094);
        filter.visit_special(21, 2, 3, &groups, |_| {}).unwrap();
        assert_eq!(filter.activation(), Activation::Active);
        assert_eq!(
            filter.visit_special(100, 0, 3, &groups, |_| {}),
            Err(PrefixError::PositionOutOfBounds)
        );
    }

    #[test]
    fn cached_groups_preserve_scalar_masks_and_order_for_all_bytes() {
        let tokens = [0; 66];
        let distances = [
            1,
            2,
            3,
            8,
            15,
            16,
            17,
            18,
            23,
            31,
            32,
            33,
            47,
            48,
            usize::MAX - 3,
            usize::MAX,
        ];
        let groups = group_distances(&distances).unwrap();
        let mut filter = PrefixFilter::new(&tokens, FilterPolicy::Eager);
        for wanted in 0..=u8::MAX {
            // Synthetic cache bytes isolate mask integration from prefix hashing.
            for (i, tag) in filter.tags.iter_mut().enumerate() {
                *tag = if i % 16 == usize::from(wanted % 16) || i % 5 == 0 {
                    wanted
                } else {
                    wanted ^ 0xff
                };
            }
            for position in [0, 1, 14, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65] {
                let original = filter.tags[position];
                filter.tags[position] = wanted;
                for safe in [0, 1, 15, 16, 17, 31, 32, 47, 48, usize::MAX] {
                    for minimum in [2, 3] {
                        let expected: Vec<_> = distances
                            .iter()
                            .copied()
                            .filter(|&distance| {
                                distance > safe
                                    && distance <= position
                                    && (minimum < MANDATORY_PREFIX
                                        || tokens.len() - position < MANDATORY_PREFIX
                                        || filter.tags[position - distance] == wanted)
                            })
                            .collect();
                        let mut actual = Vec::new();
                        filter
                            .visit_special(position, safe, minimum, &groups, |d| actual.push(d))
                            .unwrap();
                        assert_eq!(
                            actual, expected,
                            "wanted={wanted}, position={position}, safe={safe}, minimum={minimum}"
                        );
                    }
                }
                filter.tags[position] = original;
            }
        }
    }

    #[test]
    fn exact_match_sequences_are_unchanged_across_activation() {
        let mut random = 9u32;
        let tokens: Vec<_> = (0..1024)
            .map(|i| {
                random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if i % 11 < 5 { i % 11 } else { random >> 29 }
            })
            .collect();
        let distances: Vec<_> = (1..=120).collect();
        let groups = group_distances(&distances).unwrap();
        for minimum in [1, 2, 3, 4, 8] {
            for policy in [
                FilterPolicy::Disabled,
                FilterPolicy::Eager,
                FilterPolicy::Lazy,
            ] {
                let mut filter = PrefixFilter::new(&tokens, policy);
                for position in 0..tokens.len() {
                    let safe = position % 8;
                    let exact = |distance| {
                        tokens[position..]
                            .iter()
                            .zip(&tokens[position - distance..])
                            .take_while(|(a, b)| a == b)
                            .count()
                    };
                    let expected: Vec<_> = distances
                        .iter()
                        .copied()
                        .filter(|&d| d > safe && d <= position)
                        .map(|d| (d, exact(d)))
                        .filter(|&(_, length)| length >= minimum)
                        .collect();
                    let mut actual = Vec::new();
                    filter
                        .visit_special(position, safe, minimum, &groups, |d| {
                            let length = exact(d);
                            if length >= minimum {
                                actual.push((d, length));
                            }
                        })
                        .unwrap();
                    assert_eq!(actual, expected);
                }
                if policy == FilterPolicy::Lazy {
                    assert_eq!(filter.activation(), Activation::Active);
                }
            }
        }
    }

    fn check_ring<const N: usize>() {
        let mut bucket = RecentBucket::<N>::new().unwrap();
        let mut history = Vec::new();
        bucket.visit(Some(0), 3, |_| panic!("empty bucket"));
        assert_eq!(bucket.oldest_if_full(), None);
        for position in 0..(N * 5 + 3) {
            let hash = ((position % 3) as u32) << 24;
            bucket.insert(position, hash);
            history.push((position, hash));
            if history.len() > N {
                history.remove(0);
            }
            for query in [None, Some(0), Some(1 << 24), Some(3 << 24)] {
                for minimum in [2, 3] {
                    let mut actual = Vec::new();
                    bucket.visit(query, minimum, |p| actual.push(p));
                    let expected: Vec<_> = history
                        .iter()
                        .rev()
                        .filter(|(_, h)| minimum < 3 || query.is_none_or(|q| q >> 24 == h >> 24))
                        .map(|(p, _)| *p)
                        .collect();
                    assert_eq!(actual, expected);
                }
            }
            assert_eq!(
                bucket.oldest_if_full(),
                (history.len() == N).then_some(history[0].0)
            );
        }
    }

    #[test]
    fn rings_preserve_newest_first_order_and_mask_64_lanes() {
        assert!(RecentBucket::<0>::new().is_err());
        assert!(RecentBucket::<3>::new().is_err());
        assert!(RecentBucket::<128>::new().is_err());
        check_ring::<1>();
        check_ring::<2>();
        check_ring::<4>();
        check_ring::<8>();
        check_ring::<16>();
        check_ring::<32>();
        check_ring::<64>();
    }

    #[test]
    fn bucket_masks_cover_all_bytes_live_lengths_and_ring_wrap() {
        let mut bucket = RecentBucket::<64>::new().unwrap();
        for position in 0usize..320 {
            bucket.insert(position, u32::from(position as u8) << 24);
            for wanted in 0..=u8::MAX {
                let expected: Vec<_> = (position.saturating_sub(63)..=position)
                    .rev()
                    .filter(|&candidate| candidate as u8 == wanted)
                    .collect();
                let mut actual = Vec::new();
                bucket.visit(Some(u32::from(wanted) << 24), 3, |p| actual.push(p));
                assert_eq!(actual, expected, "position={position}, wanted={wanted}");
            }
        }
    }

    #[test]
    fn uniform_cache_recovers_after_mixed_entries_are_evicted() {
        let mut bucket = RecentBucket::<2>::new().unwrap();
        bucket.insert(0, 1 << 24);
        bucket.insert(1, 2 << 24);
        bucket.insert(2, 2 << 24);
        let mut actual = Vec::new();
        bucket.visit(Some(2 << 24), 3, |p| actual.push(p));
        assert_eq!(actual, [2, 1]);
        assert_eq!(bucket.stats().mixed_queries, 1);
        bucket.visit(Some(2 << 24), 3, |_| {});
        assert_eq!(bucket.stats().uniform_queries, 1);
        assert_eq!(bucket.stats().retained, 4);
    }

    #[test]
    fn all_hit_ring_fast_path_preserves_wrap_and_filter_stats() {
        let mut bucket = RecentBucket::<64>::new().unwrap();
        for position in 0usize..193 {
            bucket.insert(position, 7 << 24);
            let expected: Vec<_> = (position.saturating_sub(63)..=position).rev().collect();
            let mut actual = Vec::new();
            let before = bucket.stats();
            bucket.visit(None, 3, |p| actual.push(p));
            assert_eq!(actual, expected);
            assert_eq!(bucket.stats(), before);
            actual.clear();
            bucket.visit(Some(7 << 24), 3, |p| actual.push(p));
            assert_eq!(actual, expected);
            let after = bucket.stats();
            assert_eq!(after.scanned - before.scanned, expected.len() as u64);
            assert_eq!(after.retained - before.retained, expected.len() as u64);
            assert_eq!(after.uniform_queries - before.uniform_queries, 1);
        }
    }

    #[test]
    fn long_bucket_tags_depend_only_on_mandatory_prefix() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 2, 3, 80, 70, 60, 50, 40];
        let mut bucket = RecentBucket::<2>::new().unwrap();
        bucket.insert(0, prefix_hash(&a, 0).unwrap());
        let mut kept = Vec::new();
        bucket.visit(prefix_hash(&b, 0), 3, |p| kept.push(p));
        assert_eq!(kept, [0]);
    }
}
