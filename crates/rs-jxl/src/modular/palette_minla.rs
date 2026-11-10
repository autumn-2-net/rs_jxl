//! Independent implementations of EZeng-style graph growth and bounded MinLA
//! insertion refinement, derived from weighted-cut identities, not source ports.
//!
//! Inputs are symmetric row-major n-by-n weights with n <= 256; diagonal entries
//! are ignored. Orders contain each vertex exactly once. The parent validates
//! rounds (0..=40) and bounds sampled weights so reported costs fit in u64.
//! Internal i128 arithmetic is exact even for full u64 edge weights at this n.
//! EZeng's entire growth takes O(n^2) work, refinement O((rounds + 1) * n^2).
//! Both use O(n) scratch space, apart from the supplied dense graph.

#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Stats {
    /// Completed sweeps, including the final sweep with no improvements.
    /// Empty/singleton orders and a zero budget complete no sweeps.
    pub rounds: usize,
    /// Strictly improving insertions, at most n per completed sweep.
    pub moves: usize,
    pub before: u64,
    pub after: u64,
}

fn validate_graph(weights: &[u64], n: usize) {
    assert!(n <= 256, "palette graph exceeds 256 vertices");
    assert_eq!(weights.len(), n * n, "palette graph must be square");
}

fn validate_order(order: &[usize]) {
    let mut seen = [false; 256];
    for &vertex in order {
        assert!(vertex < order.len(), "palette vertex out of range");
        assert!(!seen[vertex], "palette order contains duplicate vertices");
        seen[vertex] = true;
    }
}

fn wide_cost(weights: &[u64], order: &[usize]) -> i128 {
    let n = order.len();
    let mut total = 0;
    for (i, &a) in order.iter().enumerate() {
        for (j, &b) in order.iter().enumerate().skip(i + 1) {
            total += i128::from(weights[a * n + b]) * (j - i) as i128;
        }
    }
    total
}

fn reported_cost(total: i128) -> u64 {
    u64::try_from(total).expect("palette arrangement cost does not fit in u64")
}

/// Sum w[a,b] * |position[a] - position[b]|, counting each undirected edge once.
/// Panics if the exact result exceeds u64 (excluded by the parent's edge bound).
pub(super) fn cost(weights: &[u64], order: &[usize]) -> u64 {
    validate_graph(weights, order.len());
    validate_order(order);
    reported_cost(wide_cost(weights, order))
}

/// Balance[v] = weight to vertices right of v minus weight to those left of v.
/// The prefix sum of balances is the weight crossing that prefix's cut.
fn balances(weights: &[u64], n: usize, order: &[usize]) -> Vec<i128> {
    let mut balance = vec![0; n];
    for (i, &a) in order.iter().enumerate() {
        for &b in &order[i + 1..] {
            let weight = i128::from(weights[a * n + b]);
            balance[a] += weight;
            balance[b] -= weight;
        }
    }
    balance
}

/// Full cost added by inserting vertex at each slot in the partial order.
/// In addition to its incident edges, insertion lengthens every old edge that
/// straddles the slot. Advancing the slot changes that cut by balance[v].
fn insertion_scores(
    weights: &[u64],
    n: usize,
    order: &[usize],
    balance: &[i128],
    vertex: usize,
    scores: &mut [i128],
) {
    let row = &weights[vertex * n..(vertex + 1) * n];
    let mut total = 0;
    let mut score = 0;
    for (i, &v) in order.iter().enumerate() {
        let weight = i128::from(row[v]);
        total += weight;
        score += (i + 1) as i128 * weight;
    }
    scores[0] = score;
    let mut left = 0;
    for (i, &v) in order.iter().enumerate() {
        let weight = i128::from(row[v]);
        score += 2 * left + weight - total + balance[v];
        left += weight;
        scores[i + 1] = score;
    }
}

/// Start with the strongest edge, then add the vertex most strongly connected
/// to the placed set at its exact cheapest slot. Ties use the lexicographically
/// first edge, lowest vertex ID, and leftmost insertion slot, respectively.
pub(super) fn ezeng(weights: &[u64], n: usize) -> Vec<usize> {
    validate_graph(weights, n);
    if n < 2 {
        return (0..n).collect();
    }

    let mut edge = (0, 1);
    let mut strongest = 0;
    for a in 0..n {
        for b in a + 1..n {
            let weight = weights[a * n + b];
            if weight > strongest {
                strongest = weight;
                edge = (a, b);
            }
        }
    }
    if strongest == 0 {
        return (0..n).collect();
    }

    let mut order = Vec::with_capacity(n);
    order.extend([edge.0, edge.1]);
    let mut placed = vec![false; n];
    placed[edge.0] = true;
    placed[edge.1] = true;
    let mut connection: Vec<i128> = (0..n)
        .map(|v| i128::from(weights[v * n + edge.0]) + i128::from(weights[v * n + edge.1]))
        .collect();
    let mut balance = balances(weights, n, &order);
    let mut scores = vec![0; n];

    while order.len() < n {
        let mut next = None;
        for v in 0..n {
            if !placed[v] && next.is_none_or(|best| connection[v] > connection[best]) {
                next = Some(v);
            }
        }
        let vertex = next.expect("an unplaced palette vertex remains");
        let slots = order.len() + 1;
        insertion_scores(weights, n, &order, &balance, vertex, &mut scores[..slots]);
        let mut slot = 0;
        for k in 1..slots {
            if scores[k] < scores[slot] {
                slot = k;
            }
        }

        // Only relations involving the new vertex change any balance.
        for (i, &v) in order.iter().enumerate() {
            let weight = i128::from(weights[vertex * n + v]);
            if i < slot {
                balance[v] += weight;
                balance[vertex] -= weight;
            } else {
                balance[v] -= weight;
                balance[vertex] += weight;
            }
        }
        order.insert(slot, vertex);
        placed[vertex] = true;
        for v in 0..n {
            if !placed[v] {
                connection[v] += i128::from(weights[v * n + vertex]);
            }
        }
    }
    order
}

/// Exact cost deltas for relocating order[from] to every final index.
fn move_deltas(
    weights: &[u64],
    n: usize,
    order: &[usize],
    balance: &[i128],
    from: usize,
    deltas: &mut [i128],
) {
    let vertex = order[from];
    let row = &weights[vertex * n..(vertex + 1) * n];
    deltas[from] = 0;

    // Swapping adjacent a,b changes only their intervening cut, by
    // balance[b] - balance[a] + 2*w[a,b]. After crossing a neighbor, update
    // the moved vertex's balance; all not-yet-crossed balances stay unchanged.
    let mut moving_balance = balance[vertex];
    let mut delta = 0;
    for to in from + 1..n {
        let neighbor = order[to];
        let twice_weight = 2 * i128::from(row[neighbor]);
        delta += balance[neighbor] - moving_balance + twice_weight;
        moving_balance -= twice_weight;
        deltas[to] = delta;
    }
    let mut moving_balance = balance[vertex];
    let mut delta = 0;
    for to in (0..from).rev() {
        let neighbor = order[to];
        let twice_weight = 2 * i128::from(row[neighbor]);
        delta += moving_balance - balance[neighbor] + twice_weight;
        moving_balance += twice_weight;
        deltas[to] = delta;
    }
}

fn best_move(deltas: &[i128], from: usize) -> (usize, i128) {
    let mut best = (from, 0);
    for (to, &delta) in deltas.iter().enumerate() {
        if delta < best.1 {
            best = (to, delta);
        }
    }
    best
}

fn apply_move(
    weights: &[u64],
    n: usize,
    order: &mut [usize],
    balance: &mut [i128],
    from: usize,
    to: usize,
) {
    let vertex = order[from];
    if to > from {
        for &neighbor in &order[from + 1..=to] {
            let twice_weight = 2 * i128::from(weights[vertex * n + neighbor]);
            balance[vertex] -= twice_weight;
            balance[neighbor] += twice_weight;
        }
        order[from..=to].rotate_left(1);
    } else if to < from {
        for &neighbor in &order[to..from] {
            let twice_weight = 2 * i128::from(weights[vertex * n + neighbor]);
            balance[vertex] += twice_weight;
            balance[neighbor] -= twice_weight;
        }
        order[to..=from].rotate_right(1);
    }
}

/// Refine the supplied permutation in place, accepting only strict improvements.
/// Each sweep visits vertex IDs in ascending order exactly once, irrespective
/// of earlier moves. Equal improving slots use the leftmost index; equal-cost
/// moves never displace the current slot. Stops after an unproductive sweep.
pub(super) fn refine(weights: &[u64], order: &mut [usize], rounds: u8) -> Stats {
    let n = order.len();
    assert!(rounds <= 40, "palette refinement rounds must be 0..=40");
    let before = cost(weights, order);
    let mut total = i128::from(before);
    let mut stats = Stats {
        before,
        after: before,
        ..Stats::default()
    };
    if rounds == 0 || n < 2 {
        return stats;
    }

    let mut balance = balances(weights, n, order);
    let mut positions = vec![0; n];
    for (i, &v) in order.iter().enumerate() {
        positions[v] = i;
    }
    let mut deltas = vec![0; n];
    for _ in 0..rounds {
        let previous_moves = stats.moves;
        for vertex in 0..n {
            let from = positions[vertex];
            move_deltas(weights, n, order, &balance, from, &mut deltas);
            let (to, delta) = best_move(&deltas, from);
            if delta < 0 {
                apply_move(weights, n, order, &mut balance, from, to);
                for i in from.min(to)..=from.max(to) {
                    positions[order[i]] = i;
                }
                total += delta;
                stats.moves += 1;
            }
        }
        stats.rounds += 1;
        if stats.moves == previous_moves {
            break;
        }
    }
    stats.after = reported_cost(total);
    stats
}

#[cfg(test)]
#[path = "palette_minla_tests.rs"]
mod tests;
