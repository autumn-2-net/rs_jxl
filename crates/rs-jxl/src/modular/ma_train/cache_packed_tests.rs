use super::super::super::{PropertyColumn, ResidualToken, TreeSamples, parent_histograms};
use super::*;

fn samples(records: usize, bins: usize, pattern: usize) -> (TreeSamples, Vec<u32>) {
    let values: Vec<_> = (0..records)
        .map(|i| match pattern {
            0 => (i * 37 % bins) as u32,
            1 => (i / 128 % bins) as u32,
            _ => 0,
        })
        .collect();
    let residuals = (0..2)
        .map(|p| {
            (0..records)
                .map(|i| ResidualToken {
                    token: ((i * (p + 7) + i / 37) % (173 + p * 83)) as u8,
                    // Storage permits arbitrary bit counts, not just canonical HybridUint.
                    nbits: ((i * (p + 3)) % 256) as u8,
                })
                .collect()
        })
        .collect();
    (
        TreeSamples {
            properties: vec![PropertyColumn::Wide(values.clone())],
            residuals,
            counts: (0..records)
                .map(|i| [1, 2, 17, 257, u16::MAX][i % 5])
                .collect(),
            thresholds: vec![(0..bins - 1).map(|i| i as i32).collect()],
            global_max_symbols: 0,
        },
        values,
    )
}

fn empty(bins: usize, stride: usize) -> BinHistogram {
    BinHistogram {
        symbols: stride,
        counts: vec![0; bins * stride],
        extra: vec![0; bins],
    }
}

fn check<T: PropertyValue>(samples: &TreeSamples, values: &[T], indices: &[usize], bins: usize) {
    let parents = parent_histograms(samples, indices);
    let mut used = vec![0; bins];
    let mut weights = vec![0; bins];
    let (mut a, mut b) = (empty(bins, 256), empty(bins, 264));
    let accepted = collect_pair(
        samples,
        indices,
        values,
        &parents,
        &mut used,
        &mut weights,
        &mut a,
        &mut b,
    );
    assert_eq!(accepted, usize::BITS >= 64 && indices.len() >= 512);
    if !accepted {
        return;
    }
    let mut expected_used = vec![0; bins];
    let mut expected_weights = vec![0; bins];
    let (mut ea, mut eb) = (empty(bins, 256), empty(bins, 264));
    for &i in indices {
        let bin = values[i].index();
        let weight = u32::from(samples.counts[i]);
        expected_used[bin] += 1;
        expected_weights[bin] += weight;
        for (p, histogram) in [&mut ea, &mut eb].into_iter().enumerate() {
            let t = samples.residuals[p][i];
            histogram.counts[bin * histogram.symbols + usize::from(t.token)] += weight;
            histogram.extra[bin] += u64::from(weight) * u64::from(t.nbits);
        }
    }
    assert_eq!(used, expected_used);
    assert_eq!(weights, expected_weights);
    assert_eq!(a.counts, ea.counts);
    assert_eq!(b.counts, eb.counts);
    assert_eq!(a.extra, ea.extra);
    assert_eq!(b.extra, eb.extra);
}

#[test]
fn fused_packed_pairs_match_wide_independent_counters() {
    for bins in [1, 17, 256, 513] {
        for records in [0, 1, 511, 512, 1025] {
            for pattern in 0..3 {
                let (samples, values) = samples(records, bins, pattern);
                let indices: Vec<_> = (0..records).rev().collect();
                check(&samples, &values, &indices, bins);
                if bins <= 256 {
                    let packed: Vec<_> = values.iter().map(|&v| v as u8).collect();
                    check(&samples, &packed, &indices, bins);
                }
                let removed: Vec<_> = indices.iter().copied().filter(|i| i % 7 != 0).collect();
                check(&samples, &values, &removed, bins);
            }
        }
    }
}

#[test]
fn fused_packed_pairs_accept_full_halves_and_reject_overflow_before_writes() {
    let (mut samples, values) = samples(512, 1, 2);
    samples.counts.fill(u16::MAX);
    for tokens in &mut samples.residuals {
        for (i, token) in tokens.iter_mut().enumerate() {
            token.nbits = if i < 257 {
                255
            } else if i == 257 {
                2
            } else {
                0
            };
        }
    }
    let indices: Vec<_> = (0..512).collect();
    let mut parents = parent_histograms(&samples, &indices);
    assert_eq!(parents[0].extra_bits, u64::from(u32::MAX));
    check(&samples, &values, &indices, 1);
    for overflow in 0..3 {
        let original = parents.clone();
        match overflow {
            0 | 1 => parents[overflow].extra_bits += 1,
            _ => parents[0].counts = vec![u32::MAX, 1],
        }
        let (mut a, mut b) = (empty(1, 256), empty(1, 264));
        let (mut used, mut weights) = ([0], [0]);
        assert!(!collect_pair(
            &samples,
            &indices,
            &values,
            &parents,
            &mut used,
            &mut weights,
            &mut a,
            &mut b
        ));
        assert_eq!(used, [0]);
        assert_eq!(weights, [0]);
        assert!(a.counts.iter().chain(&b.counts).all(|&v| v == 0));
        assert_eq!(a.extra, [0]);
        assert_eq!(b.extra, [0]);
        parents = original;
    }
}
