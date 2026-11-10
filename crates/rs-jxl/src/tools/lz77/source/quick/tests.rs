use std::sync::Arc;

use super::super::{Costs, SourceLz77Method};
use super::*;

const MODES: [SourceQuickMode; 3] = [
    SourceQuickMode::Sparse,
    SourceQuickMode::DoubleFast,
    SourceQuickMode::DoubleFastSparse,
];

fn tokens(values: &[u32]) -> Vec<Literal> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| Literal {
            context: if i % 7 == 0 { u32::MAX } else { (i % 3) as u32 },
            value,
        })
        .collect()
}

fn noise(count: usize) -> Vec<u32> {
    let mut state = 0x1234_abcd_u32;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect()
}

fn replay(input: &[Literal], commands: &[Command], maximum: usize) {
    let mut decoded = Vec::new();
    for &command in commands {
        let origin = decoded.len();
        assert!(origin < input.len());
        match command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[origin]);
                decoded.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[origin].context);
                assert!((MIN_MATCH..=maximum).contains(&(length as usize)));
                let distance = distance as usize;
                assert!((1..=origin.min(WINDOW_SIZE)).contains(&distance));
                assert!(origin + length as usize <= input.len());
                for _ in 0..length {
                    decoded.push(decoded[decoded.len() - distance]);
                }
            }
        }
    }
    assert_eq!(
        decoded,
        input
            .iter()
            .map(|literal| literal.value)
            .collect::<Vec<_>>()
    );
}

fn run(
    values: &[u32],
    width: usize,
    mode: SourceQuickMode,
    maximum: usize,
) -> (SourceQuickStats, f32) {
    let input = tokens(values);
    let receipt = Arc::new(SourceQuickReceipt::default());
    let config = SourceConfig {
        method: SourceLz77Method::Lz77b7w3t,
        quick: mode,
        quick_receipt: Some(receipt.clone()),
        max_match_length: maximum,
        ..Default::default()
    };
    let costs = Costs::raw(&[&input], &config);
    let penalties = prices::Penalties::new(&costs).unwrap();
    let sums = prices::Prices::new(&costs)
        .unwrap()
        .cumulative(&input, &config)
        .unwrap();
    let mut saving = 0.0;
    let commands = parse(
        &input,
        values,
        width,
        &config,
        &penalties,
        &sums,
        &mut saving,
    )
    .unwrap();
    replay(&input, &commands, maximum);
    (receipt.snapshot(), saving)
}

#[test]
fn exact_replay_on_empty_short_tails_wide_values_and_contexts() {
    let edges = [0, 1, u16::MAX as u32, 65536, u32::MAX - 1, u32::MAX];
    for count in 0..=19 {
        let values: Vec<_> = (0..count).map(|i| edges[i % edges.len()]).collect();
        for mode in MODES {
            for width in [0, 1, 31] {
                let (stats, _) = run(&values, width, mode, 11);
                assert_eq!(stats.calls, 1);
                assert_eq!(stats.sparse_calls, u64::from(mode.sparse()));
                assert_eq!(stats.double_fast_calls, u64::from(mode.double()));
                assert!(stats.parse_seconds.is_finite() && stats.parse_seconds >= 0.0);
                if count < MIN_MATCH {
                    assert_eq!(stats.queries, 0);
                }
            }
        }
    }
    assert_eq!(SourceQuickMode::default(), SourceQuickMode::Off);
}

#[test]
fn repeated_noisy_and_mixed_streams_replay_with_length_caps() {
    let noisy = noise(4096);
    let repeated = vec![u32::MAX; 4096];
    let mut mixed = noisy[..1024].to_vec();
    mixed.extend(std::iter::repeat_n(65536, 513));
    mixed.extend_from_within(0..1024);
    mixed.extend_from_slice(&noisy[1024..]);
    for values in [&noisy, &repeated, &mixed] {
        for mode in MODES {
            for width in [0, 2, 37, usize::MAX] {
                let (stats, saving) = run(values, width, mode, 129);
                assert!(saving.is_finite());
                if values == &repeated {
                    assert!(stats.useful_matches > 0);
                    assert!(stats.long_matches > 0);
                    assert!(stats.useful_saving_bits > 0.0);
                }
            }
        }
    }
}

#[test]
fn sparse_schedule_escalates_is_bounded_and_resets() {
    let mut probes = SparseProbes::default();
    for _ in 0..31 {
        assert_eq!(probes.miss(), 1);
    }
    assert_eq!(probes.miss(), 2);
    for _ in 0..8192 {
        assert!((1..=MAX_PROBE_STEP).contains(&probes.miss()));
    }
    assert_eq!(probes.miss(), MAX_PROBE_STEP);
    probes.misses = usize::MAX;
    assert_eq!(probes.miss(), MAX_PROBE_STEP);
    probes.useful_long_match();
    assert_eq!(probes.miss(), 1);
}

#[test]
fn sparse_queries_decrease_on_noise() {
    let noisy = noise(8192);
    for mode in [SourceQuickMode::Sparse, SourceQuickMode::DoubleFastSparse] {
        let (stats, saving) = run(&noisy, 0, mode, 256);
        assert_eq!(stats.matches, 0);
        assert_eq!(stats.query_misses, stats.queries);
        assert_eq!(saving, 0.0);
        assert!(stats.queries < noisy.len() as u64 / 2);
        assert!(stats.skipped_positions > 0);
    }
    let (stats, _) = run(&noisy, 0, SourceQuickMode::DoubleFast, 256);
    assert_eq!(stats.skipped_positions, 0);
    assert_eq!(stats.queries, (noisy.len() - MIN_MATCH + 1) as u64);
}

#[test]
fn skipped_insertions_retain_the_same_history_as_dense_queries() {
    let mut values = noise(513);
    values.extend_from_within(450..480);
    let config = SourceConfig::default();
    let mut dense = DoubleHash::new(&values, 0, &config).unwrap();
    let mut skipped = DoubleHash::new(&values, 0, &config).unwrap();
    for position in 0..513 {
        dense.find(position);
    }
    skipped.find(0);
    skipped.update_range(1, 513);
    assert_eq!(dense.short, skipped.short);
    assert_eq!(dense.long, skipped.long);
    assert_eq!(dense.find(513), skipped.find(513));

    let mut dense = Fast::<7>::new(&values, 0, &config).unwrap();
    let mut skipped = Fast::<7>::new(&values, 0, &config).unwrap();
    for position in 0..513 {
        dense.find(position);
    }
    skipped.find(0);
    skipped.update_range(1, 513);
    assert_eq!(dense.find(513), skipped.find(513));
}

fn collision<const N: usize>() -> ([u32; N], [u32; N]) {
    let mut seen = vec![None; BUCKETS];
    for i in 0..=BUCKETS as u32 {
        let prefix = std::array::from_fn(|j| {
            if j == 0 {
                i
            } else {
                i | (0x8000_0000 >> (j % 8))
            }
        });
        let hash = key::<N>(&prefix, 0).unwrap();
        if let Some(previous) = seen[hash] {
            assert_ne!(previous, prefix);
            return (previous, prefix);
        }
        seen[hash] = Some(prefix);
    }
    unreachable!("pigeonhole collision in fixed table")
}

#[test]
fn short_and_long_hash_collisions_never_become_matches() {
    let config = SourceConfig {
        max_match_length: 8,
        ..Default::default()
    };
    let (left, right) = collision::<3>();
    let values: Vec<_> = left.into_iter().chain(right).collect();
    let mut matcher = DoubleHash::new(&values, 0, &config).unwrap();
    matcher.update_range(0, 1);
    assert_eq!(matcher.find(3), Match::default());

    let (left, right) = collision::<8>();
    let values: Vec<_> = left.into_iter().chain(right).collect();
    let mut matcher = DoubleHash::new(&values, 0, &config).unwrap();
    matcher.update_range(0, 1);
    assert_eq!(matcher.find(8), Match::default());
}

#[test]
fn long_hash_recovers_a_short_bucket_eviction() {
    let a = [1, 2, 3, 10, 11, 12, 13, 14];
    let b = [1, 2, 3, 20, 21, 22, 23, 24];
    let c = [1, 2, 3, 30, 31, 32, 33, 34];
    assert_ne!(key::<8>(&a, 0), key::<8>(&b, 0));
    assert_ne!(key::<8>(&a, 0), key::<8>(&c, 0));
    let values: Vec<_> = a.into_iter().chain(b).chain(c).chain(a).collect();
    let config = SourceConfig {
        max_match_length: 8,
        ..Default::default()
    };
    let mut matcher = DoubleHash::new(&values, 0, &config).unwrap();
    for position in [0, 8, 16] {
        matcher.update_range(position, position + 1);
    }
    assert_eq!(matcher.short[key::<3>(&values, 24).unwrap()], [16, 8]);
    let matched = matcher.find(24);
    assert_eq!(matched.length, 8);
    assert_eq!(matched.distance, 24);
    assert_eq!(matcher.short.len(), BUCKETS);
    assert_eq!(matcher.long.len(), BUCKETS);
}

#[test]
fn spatial_probes_are_tiny_deduplicated_and_checked_at_wide_widths() {
    let config = SourceConfig {
        max_match_length: 8,
        ..Default::default()
    };
    for (width, expected) in [
        (0, vec![1]),
        (1, vec![1, 2]),
        (2, vec![1, 2, 3, 4]),
        (17, vec![1, 16, 17, 18, 34]),
        (usize::MAX, vec![1]),
        (WINDOW_SIZE + 1, vec![1, WINDOW_SIZE]),
    ] {
        let values: Vec<u32> = (0..128).collect();
        let matcher = DoubleHash::new(&values, width, &config).unwrap();
        assert_eq!(
            &matcher.spatial[..matcher.spatial_count],
            expected.as_slice()
        );
        for distance in expected.into_iter().filter(|&distance| distance <= 80) {
            let mut values: Vec<u32> = (0..128).collect();
            for i in 0..8 {
                values[80 + i] = values[80 + i - distance];
            }
            let mut matcher = DoubleHash::new(&values, width, &config).unwrap();
            let matched = matcher.find(80);
            assert_eq!(matched.length, 8);
            assert_eq!(matched.distance, distance);
            assert_eq!(
                matched.symbol,
                super::super::super::distance_symbol(distance, width).unwrap()
            );
        }
    }
}

#[test]
fn double_hash_honors_window_boundary_and_overlapping_copy_limit() {
    let values = vec![u32::MAX; WINDOW_SIZE + 9];
    let config = SourceConfig {
        max_match_length: 8,
        ..Default::default()
    };
    let mut matcher = DoubleHash::new(&values, 0, &config).unwrap();
    matcher.update_range(0, 1);
    // Isolate bucket legality from the independent distance-1 spatial probe.
    let mut matched = Match::default();
    matcher.consider(WINDOW_SIZE, 0, &mut matched);
    assert_eq!(matched.length, 8);
    assert_eq!(matched.distance, WINDOW_SIZE);
    matched = Match::default();
    matcher.consider(WINDOW_SIZE + 1, 0, &mut matched);
    assert_eq!(matched, Match::default());
    let matched = matcher.find(1);
    assert_eq!(matched.length, 8);
    assert_eq!(matched.distance, 1);
}

#[test]
fn source_cost_gate_rejection_skip_and_ungated_savings_remain_unchanged() {
    let values = vec![65536; 128];
    let input = tokens(&values);
    let sums = vec![0.0; values.len() + 1];
    for mode in MODES {
        for method in [SourceLz77Method::Lz77b7w3t, SourceLz77Method::Lz77b7w3f] {
            let receipt = Arc::new(SourceQuickReceipt::default());
            let config = SourceConfig {
                method,
                quick: mode,
                quick_receipt: Some(receipt.clone()),
                ..Default::default()
            };
            let costs = Costs::raw(&[&input], &config);
            let penalties = prices::Penalties::new(&costs).unwrap();
            let (mut saving, mut baseline_saving) = (0.0, 0.0);
            let commands =
                parse(&input, &values, 0, &config, &penalties, &sums, &mut saving).unwrap();
            let baseline = fast_stream::<7>(
                &input,
                &values,
                0,
                &config,
                &penalties,
                &sums,
                &mut baseline_saving,
            )
            .unwrap();
            assert_eq!(commands, baseline);
            assert_eq!(saving.to_bits(), baseline_saving.to_bits());
            replay(&input, &commands, config.max_match_length);
            let stats = receipt.snapshot();
            assert!(stats.unprofitable_queries > 0);
            assert_eq!(stats.useful_matches, 0);
            assert_eq!(stats.long_matches, 0);
            assert_eq!(stats.useful_saving_bits, 0.0);
            assert_eq!(stats.matches == 0, method.runtime_cost_comparison());
        }
    }
}

#[test]
fn off_delegates_to_original_fast_capacities_without_a_receipt() {
    let mut values = noise(128);
    values.extend_from_within(..);
    let input = tokens(&values);
    for method in [
        SourceLz77Method::Lz77b1w3t,
        SourceLz77Method::Lz77b3w3f,
        SourceLz77Method::Lz77b7w3t,
        SourceLz77Method::Lz77b15w3f,
        SourceLz77Method::Lz77b31w3t,
        SourceLz77Method::Optc256,
    ] {
        let receipt = Arc::new(SourceQuickReceipt::default());
        let config = SourceConfig {
            method,
            quick_receipt: Some(receipt.clone()),
            ..Default::default()
        };
        let costs = Costs::raw(&[&input], &config);
        let penalties = prices::Penalties::new(&costs).unwrap();
        let sums = prices::Prices::new(&costs)
            .unwrap()
            .cumulative(&input, &config)
            .unwrap();
        let (mut saving, mut expected_saving) = (0.0, 0.0);
        let commands = parse(&input, &values, 17, &config, &penalties, &sums, &mut saving).unwrap();
        let expected = match method.fast_capacity() {
            Some(1) => fast_stream::<1>(
                &input,
                &values,
                17,
                &config,
                &penalties,
                &sums,
                &mut expected_saving,
            ),
            Some(3) => fast_stream::<3>(
                &input,
                &values,
                17,
                &config,
                &penalties,
                &sums,
                &mut expected_saving,
            ),
            Some(7) | None => fast_stream::<7>(
                &input,
                &values,
                17,
                &config,
                &penalties,
                &sums,
                &mut expected_saving,
            ),
            Some(15) => fast_stream::<15>(
                &input,
                &values,
                17,
                &config,
                &penalties,
                &sums,
                &mut expected_saving,
            ),
            Some(31) => fast_stream::<31>(
                &input,
                &values,
                17,
                &config,
                &penalties,
                &sums,
                &mut expected_saving,
            ),
            _ => unreachable!(),
        }
        .unwrap();
        assert_eq!(commands, expected);
        assert_eq!(saving.to_bits(), expected_saving.to_bits());
        assert_eq!(receipt.snapshot(), SourceQuickStats::default());
    }
}

#[test]
fn receipt_merges_once_per_completed_call() {
    let receipt = SourceQuickReceipt::default();
    let stats = SourceQuickStats {
        calls: 1,
        sparse_calls: 1,
        double_fast_calls: 1,
        queries: 50,
        query_misses: 47,
        unprofitable_queries: 2,
        matches: 3,
        useful_matches: 3,
        long_matches: 2,
        matched_values: 40,
        skipped_positions: 100,
        parse_seconds: 0.125,
        useful_saving_bits: 12.5,
        ..Default::default()
    };
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| receipt.record(stats));
        }
    });
    let total = receipt.snapshot();
    assert_eq!(total.calls, 4);
    assert_eq!(total.sparse_calls, 4);
    assert_eq!(total.double_fast_calls, 4);
    assert_eq!(total.queries, 200);
    assert_eq!(total.query_misses, 188);
    assert_eq!(total.unprofitable_queries, 8);
    assert_eq!(total.matches, 12);
    assert_eq!(total.useful_matches, 12);
    assert_eq!(total.long_matches, 8);
    assert_eq!(total.matched_values, 160);
    assert_eq!(total.skipped_positions, 400);
    assert_eq!(total.parse_seconds, 0.5);
    assert_eq!(total.useful_saving_bits, 50.0);
}

#[test]
fn cost_ranked_sparse_mode_replays_wide_contexts_overlap_noise_and_tails() {
    let mut values = noise(4097);
    values.extend(std::iter::repeat_n(u32::MAX, 71));
    values.extend_from_within(302..709);
    for width in [0, 1, 2, 37, usize::MAX] {
        for cap in [3, 4, 7, 8, 9, 511] {
            let (stats, gain) = run(&values, width, SourceQuickMode::DoubleFastCost, cap);
            assert_eq!(stats.cost_ranked_calls, 1);
            assert_eq!(stats.calls, 1);
            assert!(gain.is_finite());
        }
        for length in 0..10 {
            run(
                &values[..length],
                width,
                SourceQuickMode::DoubleFastCost,
                11,
            );
        }
    }
}

#[test]
fn cost_ranked_matching_can_prefer_shorter_and_more_expensive_numbered_symbols() {
    let mut values: Vec<_> = (0..96u32).map(|n| n + 100).collect();
    values[43..47].copy_from_slice(&[7, 7, 7, 8]);
    values[79..84].copy_from_slice(&[7, 7, 7, 7, 8]);
    let config = SourceConfig {
        quick: SourceQuickMode::DoubleFastCost,
        max_match_length: 4,
        ..Default::default()
    };
    let mut sums = vec![0.0; values.len() + 1];
    sums[81] = 3.0;
    sums[82] = 6.0;
    sums[83..].fill(9.0);
    let mut matcher = DoubleHash::new(&values, 37, &config).unwrap();
    let (chosen, legal) = matcher.find_priced(80, &sums, 0.0, true);
    assert!(legal);
    assert_eq!((chosen.length, chosen.distance, chosen.symbol), (3, 1, 1));
    let mut longest = DoubleHash::new(&values, 37, &config).unwrap();
    let chosen = longest.find(80);
    assert_eq!((chosen.length, chosen.distance), (4, 37));

    let values = vec![7; 96];
    let config = SourceConfig {
        max_match_length: 3,
        ..config
    };
    let mut matcher = DoubleHash::new(&values, 37, &config).unwrap();
    let (chosen, _) = matcher.find_priced(80, &sums, 0.0, true);
    assert_eq!((chosen.length, chosen.distance, chosen.symbol), (3, 1, 1));
}

#[test]
fn priced_matching_filters_zero_gain_but_preserves_ungated_methods_and_history() {
    let values = vec![65536; 48];
    let config = SourceConfig {
        quick: SourceQuickMode::DoubleFastCost,
        ..Default::default()
    };
    let sums = vec![0.0; values.len() + 1];
    let mut priced = DoubleHash::new(&values, 1, &config).unwrap();
    assert_eq!(priced.find_priced(1, &sums, 6.0, true).0, Match::default());
    let mut dense = DoubleHash::new(&values, 1, &config).unwrap();
    dense.update_range(1, 2);
    assert_eq!(priced.short, dense.short);
    assert_eq!(priced.long, dense.long);
    assert!(priced.find_priced(2, &sums, 6.0, false).0.length >= MIN_MATCH);
    for width in [0, 1, 2, 37, usize::MAX] {
        let mut matcher = DoubleHash::new(&values, width, &config).unwrap();
        let sums: Vec<_> = (0..=values.len()).map(|n| n as f32 * 32.0).collect();
        let (chosen, _) = matcher.find_priced(20, &sums, 0.0, true);
        assert_eq!(
            chosen.symbol,
            super::super::super::distance_symbol(chosen.distance, width).unwrap()
        );
        assert!(matcher.spatial_count <= 29);
    }
}

#[test]
fn priced_rejected_runs_keep_source_skip_and_zero_gain_acceptance() {
    let values = vec![65536; 128];
    let config = SourceConfig {
        quick: SourceQuickMode::DoubleFastCost,
        max_match_length: 16,
        ..Default::default()
    };
    let mut sums = vec![0.0; values.len() + 1];
    *sums.last_mut().unwrap() = 0.25;
    let mut matcher = DoubleHash::new(&values, 0, &config).unwrap();
    let (matched, legal) = matcher.find_priced(1, &sums, 6.0, true);
    assert!(legal);
    assert_eq!(matched.length, 16);
    assert!(static_cost_with_penalty(matched, 6.0) > sums[17] - sums[1]);

    let values = vec![7; 4];
    let config = SourceConfig {
        max_match_length: 3,
        ..config
    };
    let mut matcher = DoubleHash::new(&values, 0, &config).unwrap();
    let expected = Match {
        length: 3,
        distance: 1,
        symbol: matcher.codes.symbol(1),
    };
    let cost = static_cost_with_penalty(expected, 6.0);
    let (matched, legal) = matcher.find_priced(1, &[0.0, 0.0, 0.0, 0.0, cost], 6.0, true);
    assert!(legal);
    assert_eq!(matched, expected);

    let values = vec![0; 32768];
    let config = SourceConfig {
        method: SourceLz77Method::Lz77b7w3f,
        quick: SourceQuickMode::DoubleFastCost,
        ..Default::default()
    };
    let mut matcher = DoubleHash::new(&values, 0, &config).unwrap();
    let (matched, legal) = matcher.find_priced(1, &vec![0.0; values.len() + 1], 6.0, false);
    assert!(legal);
    assert_eq!(matched.length, values.len() - 1);
}
