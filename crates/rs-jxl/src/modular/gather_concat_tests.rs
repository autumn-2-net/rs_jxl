use std::collections::BTreeMap;

use super::*;

fn collector(cache: bool, rows: std::ops::Range<usize>) -> SourceSampleCollector {
    let mut collector = SourceSampleCollector::new(
        &[Predictor::Zero, Predictor::Gradient],
        &PropertyQuantization {
            properties: vec![2, 0, 1],
            thresholds: vec![(0..99).collect(), vec![0], vec![0]],
        },
        cache,
    )
    .unwrap();
    collector.prepare(rows.len().max(1)).unwrap();
    for row in rows {
        collector
            .push(&[0, 0, row as i32], &[0, 16 + row as i32 % 3], 1)
            .unwrap();
    }
    collector
}

fn keys(samples: &PreparedSamples) -> BTreeMap<Vec<u32>, u64> {
    let mut keys = BTreeMap::new();
    for (row, &count) in samples.counts.iter().enumerate() {
        let mut key: Vec<_> = samples.properties.iter().map(|p| p.value(row)).collect();
        for residuals in &samples.residuals {
            key.extend(residuals[row].map(u32::from));
        }
        *keys.entry(key).or_default() += u64::from(count);
    }
    keys
}

fn split_statistics(samples: &PreparedSamples) -> Vec<(Vec<u32>, u64, u64)> {
    let mut result = Vec::new();
    for (property, thresholds) in samples.quantization.thresholds.iter().enumerate() {
        for split in 0..=thresholds.len() {
            for residuals in &samples.residuals {
                for above in [false, true] {
                    let mut counts = vec![0u32; 256];
                    let mut extra = 0u64;
                    let mut total = 0u64;
                    for (row, &weight) in samples.counts.iter().enumerate() {
                        if (samples.properties[property].value(row) as usize > split) != above {
                            continue;
                        }
                        let [token, nbits] = residuals[row];
                        counts[usize::from(token)] += u32::from(weight);
                        extra += u64::from(nbits) * u64::from(weight);
                        total += u64::from(weight);
                    }
                    result.push((counts, extra, total));
                }
            }
        }
    }
    result
}

fn assert_objective(actual: &PreparedSamples, expected: &PreparedSamples) {
    assert_eq!(actual.predictors, expected.predictors);
    assert_eq!(actual.quantization, expected.quantization);
    assert_eq!(actual.total, expected.total);
    assert_eq!(keys(actual), keys(expected));
    assert_eq!(split_statistics(actual), split_statistics(expected));
    assert_eq!(
        actual.counts.iter().map(|&n| u64::from(n)).sum::<u64>(),
        actual.total
    );
}

fn old_append(
    mut master: SourceSampleCollector,
    batches: Vec<SourceSampleCollector>,
) -> PreparedSamples {
    master.append(batches).unwrap();
    master.finish()
}

#[test]
fn concat_keeps_cross_batch_duplicates_and_exact_weighted_splits() {
    for master_cache in [false, true] {
        for batch_cache in [false, true] {
            let batches = || {
                let mut weighted = collector(batch_cache, 0..100);
                weighted.push(&[0, 0, 0], &[0, 16], 1).unwrap();
                vec![weighted, collector(!batch_cache, 0..100)]
            };
            let expected = old_append(collector(master_cache, 0..0), batches());
            let actual = collector(master_cache, 0..0)
                .finish_appended(batches())
                .unwrap();
            assert_eq!(actual.len(), 200);
            assert_eq!(actual.total, 201);
            assert_eq!(actual.counts[0], 2);
            assert!(expected.len() < actual.len());
            assert_objective(&actual, &expected);
            for row in 0..200 {
                assert_eq!(actual.properties[0].value(row), (row % 100) as u32);
            }
        }
    }
}

#[test]
fn concat_empty_batches_and_existing_master_preserve_all_columns() {
    for cache in [false, true] {
        let batches = || vec![collector(cache, 0..0), collector(!cache, 0..100)];
        let expected = old_append(collector(cache, 0..10), batches());
        let actual = collector(cache, 0..10).finish_appended(batches()).unwrap();
        assert_eq!(actual.len(), 110);
        assert_objective(&actual, &expected);
        let empty = collector(cache, 0..0)
            .finish_appended(vec![collector(!cache, 0..0)])
            .unwrap();
        assert_eq!((empty.len(), empty.total), (0, 0));
        let original = collector(cache, 0..10).finish();
        let unchanged = collector(cache, 0..10).finish_appended(vec![]).unwrap();
        assert_objective(&unchanged, &original);
    }
}

#[test]
fn concat_saturated_weights_and_high_merge_rate_use_original_append() {
    for cache in [false, true] {
        let batch = || {
            let mut batch = collector(cache, 0..0);
            batch
                .push(&[0, 0, 0], &[0, 16], u32::from(u16::MAX) + 7)
                .unwrap();
            batch
        };
        let expected = old_append(collector(!cache, 0..0), vec![batch(), batch()]);
        let actual = collector(!cache, 0..0)
            .finish_appended(vec![batch(), batch()])
            .unwrap();
        assert_eq!(actual.counts, expected.counts);
        assert_eq!(actual.counts.iter().filter(|&&n| n == u16::MAX).count(), 2);
        assert_objective(&actual, &expected);
    }
}

#[test]
fn concat_density_and_payload_limits_are_inclusive_and_checked() {
    assert_eq!(MAX_CONCAT_PAYLOAD_BYTES, 64 * 1024 * 1024);
    assert!(concat_allowed(99, 100, 15, MAX_CONCAT_PAYLOAD_BYTES));
    assert!(!concat_allowed(98, 100, 15, MAX_CONCAT_PAYLOAD_BYTES));
    assert!(!concat_allowed(98, 99, 15, MAX_CONCAT_PAYLOAD_BYTES));
    assert!(concat_allowed(0, 0, 15, MAX_CONCAT_PAYLOAD_BYTES));
    let rows = MAX_CONCAT_PAYLOAD_BYTES / 16;
    assert!(concat_allowed(rows, rows as u64, 16, usize::MAX));
    assert!(!concat_allowed(rows + 1, (rows + 1) as u64, 16, usize::MAX));
    assert!(!concat_allowed(usize::MAX, 1, 16, usize::MAX));

    for cache in [false, true] {
        let batches = || vec![collector(cache, 0..100), collector(!cache, 0..100)];
        let master = || collector(cache, 0..0);
        let row_bytes = master().prepared.validate_concat().unwrap();
        assert_eq!(row_bytes, 15);
        let bytes = 200 * row_bytes;
        let expected = old_append(master(), batches());
        let fallback = master()
            .finish_appended_with_limit(batches(), bytes - 1)
            .unwrap();
        assert_eq!(fallback.counts, expected.counts);
        assert_objective(&fallback, &expected);
        let exact = master()
            .finish_appended_with_limit(batches(), bytes)
            .unwrap();
        assert_eq!(exact.len(), 200);
        assert_objective(&exact, &expected);
    }
}

#[test]
fn concat_rejects_schema_lengths_and_weight_mismatches_before_fallback() {
    let corruptions: &[fn(&mut PreparedSamples)] = &[
        |s| s.predictors.swap(0, 1),
        |s| s.quantization.thresholds[0].push(100),
        |s| s.quantization.properties.swap(0, 1),
        |s| {
            s.properties.pop();
        },
        |s| s.properties[0] = GatherColumn::Static(vec![0; s.len()]),
        |s| {
            s.residuals.pop();
        },
        |s| {
            s.residuals[0].pop();
        },
        |s| s.counts[0] = 0,
        |s| s.counts[0] += 1,
        |s| s.total += 1,
        |s| {
            s.static_ids.pop();
        },
    ];
    for &corrupt in corruptions {
        for payload_limit in [0, MAX_CONCAT_PAYLOAD_BYTES] {
            let mut bad = collector(true, 0..10);
            corrupt(&mut bad.prepared);
            assert!(matches!(
                collector(false, 0..0)
                    .finish_appended_with_limit(vec![collector(false, 0..10), bad], payload_limit),
                Err(Error::InvalidInput(_))
            ));
        }
    }
    let mut bad_master = collector(false, 0..10);
    bad_master.prepared.counts[0] = 0;
    assert!(matches!(
        bad_master.finish_appended(vec![]),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn concat_retains_i32_total_limit_even_when_density_requires_fallback() {
    let mut largest = collector(false, 0..0);
    largest.push(&[0, 0, 0], &[0, 16], i32::MAX as u32).unwrap();
    assert_eq!(largest.prepared.total, i32::MAX as u64);
    assert!(matches!(
        collector(false, 0..0).finish_appended(vec![largest, collector(true, 0..1)]),
        Err(Error::Limit("MA sample weight exceeds source i32::MAX"))
    ));
    let mut invalid = collector(false, 0..0);
    invalid.prepared.total = i32::MAX as u64 + 1;
    assert!(matches!(
        invalid.finish_appended(vec![]),
        Err(Error::Limit("MA sample weight exceeds source i32::MAX"))
    ));
}
