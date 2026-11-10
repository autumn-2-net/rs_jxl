use super::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn index(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut order: Vec<_> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, self.index(i + 1));
        }
        order
    }

    fn graph(&mut self, n: usize) -> Vec<u64> {
        let mut weights = vec![0; n * n];
        for a in 0..n {
            // Nonzero diagonal entries must not affect any graph operation.
            weights[a * n + a] = u64::from(u32::MAX);
            for b in a + 1..n {
                let weight = match self.index(8) {
                    0..=2 => 0,
                    3..=4 => 1,
                    5 => 7,
                    6 => self.next() % 1000,
                    _ => u64::from(u32::MAX),
                };
                weights[a * n + b] = weight;
                weights[b * n + a] = weight;
            }
        }
        weights
    }
}

fn graph(n: usize, edges: &[(usize, usize, u64)]) -> Vec<u64> {
    let mut weights = vec![0; n * n];
    for &(a, b, weight) in edges {
        weights[a * n + b] = weight;
        weights[b * n + a] = weight;
    }
    weights
}

// Independent oracle: iterate edges by vertex ID, not by arrangement position.
// Also supports partial arrangements used during EZeng growth.
fn brute_cost(weights: &[u64], n: usize, order: &[usize]) -> i128 {
    let mut positions = vec![None; n];
    for (i, &v) in order.iter().enumerate() {
        assert!(positions[v].replace(i).is_none());
    }
    let mut total = 0;
    for a in 0..n {
        for b in a + 1..n {
            if let (Some(i), Some(j)) = (positions[a], positions[b]) {
                total += i128::from(weights[a * n + b]) * i.abs_diff(j) as i128;
            }
        }
    }
    total
}

fn relocated(order: &[usize], from: usize, to: usize) -> Vec<usize> {
    let mut moved = order.to_vec();
    let vertex = moved.remove(from);
    moved.insert(to, vertex);
    moved
}

fn assert_permutation(order: &[usize], n: usize) {
    let mut sorted = order.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..n).collect::<Vec<_>>());
}

fn assert_cut_profile(weights: &[u64], n: usize, order: &[usize], balance: &[i128]) {
    let mut prefix = 0;
    for k in 0..=order.len() {
        let mut expected = 0;
        for &a in &order[..k] {
            for &b in &order[k..] {
                expected += i128::from(weights[a * n + b]);
            }
        }
        assert_eq!(prefix, expected, "cut {k}, order {order:?}");
        if k < order.len() {
            prefix += balance[order[k]];
        }
    }
}

fn brute_ezeng(weights: &[u64], n: usize) -> Vec<usize> {
    if n < 2 {
        return (0..n).collect();
    }
    let mut edge = (0, 1);
    for a in 0..n {
        for b in a + 1..n {
            if weights[a * n + b] > weights[edge.0 * n + edge.1] {
                edge = (a, b);
            }
        }
    }
    if weights[edge.0 * n + edge.1] == 0 {
        return (0..n).collect();
    }
    let mut order = vec![edge.0, edge.1];
    while order.len() < n {
        let mut next = None;
        let mut strongest = -1;
        for v in 0..n {
            if !order.contains(&v) {
                let connection: i128 = order.iter().map(|&u| i128::from(weights[v * n + u])).sum();
                if connection > strongest {
                    next = Some(v);
                    strongest = connection;
                }
            }
        }
        let vertex = next.unwrap();
        let mut best: Option<(i128, Vec<usize>)> = None;
        for slot in 0..=order.len() {
            let mut trial = order.clone();
            trial.insert(slot, vertex);
            let trial_cost = brute_cost(weights, n, &trial);
            if best.as_ref().is_none_or(|(value, _)| trial_cost < *value) {
                best = Some((trial_cost, trial));
            }
        }
        order = best.unwrap().1;
    }
    order
}

fn brute_refine(weights: &[u64], order: &mut Vec<usize>, rounds: u8) -> Stats {
    let n = order.len();
    let mut current = brute_cost(weights, n, order);
    let mut stats = Stats {
        before: u64::try_from(current).unwrap(),
        after: u64::try_from(current).unwrap(),
        ..Stats::default()
    };
    if n < 2 {
        return stats;
    }
    for _ in 0..rounds {
        let old_moves = stats.moves;
        for vertex in 0..n {
            let from = order.iter().position(|&v| v == vertex).unwrap();
            let mut best_cost = current;
            let mut best = order.clone();
            for to in 0..n {
                let trial = relocated(order, from, to);
                let trial_cost = brute_cost(weights, n, &trial);
                if trial_cost < best_cost {
                    best_cost = trial_cost;
                    best = trial;
                }
            }
            if best_cost < current {
                stats.moves += 1;
                current = best_cost;
                *order = best;
            }
        }
        stats.rounds += 1;
        if stats.moves == old_moves {
            break;
        }
    }
    stats.after = u64::try_from(current).unwrap();
    stats
}

#[test]
fn cost_is_undirected_distance_weighted_and_ignores_diagonal() {
    let mut weights = graph(4, &[(0, 1, 3), (0, 2, 5), (1, 3, 7), (2, 3, 11)]);
    for v in 0..4 {
        weights[v * 4 + v] = u64::from(u32::MAX);
    }
    let order = [2, 0, 3, 1];
    assert_eq!(cost(&weights, &order), 3 * 2 + 5 + 7 + 11 * 2);
    assert_eq!(
        i128::from(cost(&weights, &order)),
        brute_cost(&weights, 4, &order)
    );
}

#[test]
fn ezeng_insertion_must_pay_for_crossed_old_edges() {
    let weights = graph(3, &[(0, 1, 10), (0, 2, 6), (1, 2, 6)]);
    let order = [0, 1];
    let balance = balances(&weights, 3, &order);
    let mut scores = [0; 3];
    insertion_scores(&weights, 3, &order, &balance, 2, &mut scores);
    // Incident edges alone prefer the middle (12 < 18), but that stretches 0-1.
    assert_eq!(scores, [18, 22, 18]);
    assert_eq!(ezeng(&weights, 3), [2, 0, 1]);
}

#[test]
fn every_partial_insertion_score_matches_full_cost() {
    let mut rng = Rng(0x28e9_4a76_30c5_128b);
    for n in 1..=11 {
        for _ in 0..24 {
            let weights = rng.graph(n);
            let permutation = rng.permutation(n);
            for len in 0..n {
                let order = &permutation[..len];
                let vertex = permutation[len];
                let balance = balances(&weights, n, order);
                assert_cut_profile(&weights, n, order, &balance);
                let mut scores = vec![0; len + 1];
                insertion_scores(&weights, n, order, &balance, vertex, &mut scores);
                let old_cost = brute_cost(&weights, n, order);
                for (slot, &score) in scores.iter().enumerate() {
                    let mut trial = order.to_vec();
                    trial.insert(slot, vertex);
                    assert_eq!(score, brute_cost(&weights, n, &trial) - old_cost);
                }
            }
        }
    }
}

#[test]
fn every_move_delta_choice_and_balance_update_matches_brute_force() {
    let mut rng = Rng(0xa63c_54e8_b192_70df);
    for n in 1..=11 {
        for _ in 0..32 {
            let weights = rng.graph(n);
            let order = rng.permutation(n);
            let balance = balances(&weights, n, &order);
            assert_cut_profile(&weights, n, &order, &balance);
            let old_cost = brute_cost(&weights, n, &order);
            for from in 0..n {
                let mut deltas = vec![i128::MIN; n];
                move_deltas(&weights, n, &order, &balance, from, &mut deltas);
                let mut expected_best = (from, 0);
                for (to, &delta) in deltas.iter().enumerate() {
                    let expected = relocated(&order, from, to);
                    let expected_delta = brute_cost(&weights, n, &expected) - old_cost;
                    assert_eq!(delta, expected_delta, "n={n}, from={from}, to={to}");
                    if expected_delta < expected_best.1 {
                        expected_best = (to, expected_delta);
                    }
                    let mut actual = order.clone();
                    let mut updated_balance = balance.clone();
                    apply_move(&weights, n, &mut actual, &mut updated_balance, from, to);
                    assert_eq!(actual, expected);
                    assert_eq!(updated_balance, balances(&weights, n, &actual));
                }
                assert_eq!(best_move(&deltas, from), expected_best);
            }
        }
    }
}

#[test]
fn incremental_balances_survive_long_move_sequences() {
    let mut rng = Rng(0x598c_470b_2d1a_ef36);
    for n in 2..=13 {
        let weights = rng.graph(n);
        let mut order = rng.permutation(n);
        let mut balance = balances(&weights, n, &order);
        for _ in 0..100 {
            let from = rng.index(n);
            let to = rng.index(n);
            let mut deltas = vec![0; n];
            move_deltas(&weights, n, &order, &balance, from, &mut deltas);
            let before = brute_cost(&weights, n, &order);
            apply_move(&weights, n, &mut order, &mut balance, from, to);
            assert_eq!(brute_cost(&weights, n, &order) - before, deltas[to]);
            assert_eq!(balance, balances(&weights, n, &order));
            assert_cut_profile(&weights, n, &order, &balance);
            assert_permutation(&order, n);
        }
    }
}

#[test]
fn ezeng_matches_brute_growth_with_strongest_connection_and_deterministic_ties() {
    let mut rng = Rng(0x0b8e_9dc2_457a_63f1);
    for n in 0..=13 {
        for _ in 0..32 {
            let weights = rng.graph(n);
            let actual = ezeng(&weights, n);
            assert_eq!(actual, brute_ezeng(&weights, n));
            assert_eq!(ezeng(&weights, n), actual);
            assert_permutation(&actual, n);
        }
    }
    let weights = graph(5, &[(0, 1, 10), (0, 2, 6), (1, 2, 6), (0, 3, 9), (3, 4, 9)]);
    assert_eq!(ezeng(&weights, 5), brute_ezeng(&weights, 5));
}

#[test]
fn refinement_matches_full_cost_oracle_including_stats_and_round_zero() {
    let mut rng = Rng(0x9c14_ba78_53d6_e02f);
    for n in 0..=11 {
        for _ in 0..20 {
            let weights = rng.graph(n);
            let initial = rng.permutation(n);
            let mut previous_cost = cost(&weights, &initial);
            for rounds in [0, 1, 2, 7, 40] {
                let mut actual = initial.clone();
                let mut expected = initial.clone();
                let stats = refine(&weights, &mut actual, rounds);
                let expected_stats = brute_refine(&weights, &mut expected, rounds);
                assert_eq!(actual, expected, "n={n}, rounds={rounds}");
                assert_eq!(stats, expected_stats);
                assert_eq!(stats.after, cost(&weights, &actual));
                assert!(stats.after <= stats.before);
                assert!(stats.after <= previous_cost);
                assert!(stats.rounds <= usize::from(rounds));
                assert!(stats.moves <= stats.rounds * n);
                assert_eq!(stats.moves == 0, stats.before == stats.after);
                assert_permutation(&actual, n);
                if rounds == 0 {
                    assert_eq!(actual, initial);
                    assert_eq!(stats.rounds, 0);
                    assert_eq!(stats.moves, 0);
                }
                if rounds > 0 && stats.rounds < usize::from(rounds) {
                    let balance = balances(&weights, n, &actual);
                    let mut deltas = vec![0; n];
                    for from in 0..n {
                        move_deltas(&weights, n, &actual, &balance, from, &mut deltas);
                        assert_eq!(best_move(&deltas, from), (from, 0));
                    }
                }
                previous_cost = stats.after;
            }
        }
    }
}

#[test]
fn zero_isolated_and_tied_graphs_do_not_accept_plateau_moves() {
    assert_eq!(ezeng(&[], 0), []);
    assert_eq!(refine(&[], &mut [], 40), Stats::default());
    let mut singleton = [0];
    assert_eq!(ezeng(&[u64::from(u32::MAX)], 1), singleton);
    assert_eq!(
        refine(&[u64::from(u32::MAX)], &mut singleton, 40),
        Stats::default()
    );

    let zero = vec![0; 25];
    assert_eq!(ezeng(&zero, 5), [0, 1, 2, 3, 4]);
    let mut order = [4, 1, 3, 0, 2];
    let stats = refine(&zero, &mut order, 40);
    assert_eq!(order, [4, 1, 3, 0, 2]);
    assert_eq!(
        stats,
        Stats {
            rounds: 1,
            ..Stats::default()
        }
    );

    let isolated = graph(5, &[(1, 3, 9)]);
    assert_eq!(ezeng(&isolated, 5), [4, 2, 0, 1, 3]);
    let mut order = [1, 3, 0, 2, 4];
    let stats = refine(&isolated, &mut order, 40);
    assert_eq!(order, [1, 3, 0, 2, 4]);
    assert_eq!(
        (stats.rounds, stats.moves, stats.before, stats.after),
        (1, 0, 9, 9)
    );

    let tied = vec![7; 16];
    assert_eq!(ezeng(&tied, 4), [3, 2, 0, 1]);
    let mut order = [2, 0, 3, 1];
    let stats = refine(&tied, &mut order, 40);
    assert_eq!(order, [2, 0, 3, 1]);
    assert_eq!(
        (stats.rounds, stats.moves, stats.before, stats.after),
        (1, 0, 70, 70)
    );
}

#[test]
fn bounded_large_weights_and_256_vertices_remain_exact() {
    let n = 256;
    let weight = u64::from(u32::MAX);
    let complete = vec![weight; n * n];
    let mut order: Vec<_> = (0..n).rev().collect();
    let expected = weight * (n as u64 * (n as u64 * n as u64 - 1) / 6);
    assert!(expected > (1 << 53));
    assert_eq!(cost(&complete, &order), expected);
    let before = order.clone();
    let stats = refine(&complete, &mut order, 40);
    assert_eq!(order, before);
    assert_eq!(
        (stats.rounds, stats.moves, stats.before, stats.after),
        (1, 0, expected, expected)
    );
    assert_permutation(&ezeng(&complete, n), n);

    let edges: Vec<_> = (1..n).map(|v| (v - 1, v, weight - v as u64)).collect();
    let path = graph(n, &edges);
    let mut order: Vec<_> = (0..n).map(|v| (73 * v) % n).collect();
    let stats = refine(&path, &mut order, 4);
    assert!(stats.moves > 0);
    assert!(stats.after < stats.before);
    assert_eq!(stats.after, cost(&path, &order));
    assert_eq!(i128::from(stats.after), brute_cost(&path, n, &order));
    assert_permutation(&order, n);
    let grown = ezeng(&path, n);
    assert_permutation(&grown, n);
    assert_eq!(
        cost(&path, &grown),
        edges.iter().map(|edge| edge.2).sum::<u64>()
    );
}

#[test]
#[should_panic(expected = "duplicate")]
fn duplicate_order_is_rejected() {
    refine(&[0; 4], &mut [0, 0], 0);
}

#[test]
#[should_panic(expected = "0..=40")]
fn out_of_range_round_budget_is_rejected() {
    refine(&[], &mut [], 41);
}
