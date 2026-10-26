use super::*;

// The pre-specialization push path: scan dimensions, search all quantizers,
// and use the generic hybrid encoder. Hashing/insertion remain shared.
fn reference_push(
    collector: &mut SourceSampleCollector,
    properties: &[i32],
    residuals: &[i32],
    weight: u32,
) -> Result<()> {
    if properties.len() < 2
        || residuals.len() != collector.prepared.predictors.len()
        || collector
            .prepared
            .quantization
            .properties
            .iter()
            .any(|&id| id >= properties.len())
    {
        return Err(Error::InvalidInput(
            "invalid MA gather sample dimensions".into(),
        ));
    }
    if weight == 0 {
        return Ok(());
    }
    if collector.table.is_empty() {
        return Err(Error::InvalidInput(
            "MA gather requires prepare before samples".into(),
        ));
    }
    let total = collector
        .prepared
        .total
        .checked_add(u64::from(weight))
        .filter(|&n| n <= i32::MAX as u64)
        .ok_or(Error::Limit("MA sample weight exceeds source i32::MAX"))?;
    let mut bins = [0; 64];
    for (p, &id) in collector
        .prepared
        .quantization
        .properties
        .iter()
        .enumerate()
    {
        let value = if id < 2 {
            properties[id]
        } else {
            properties[id].clamp(-511, 511)
        };
        bins[p] =
            collector.prepared.quantization.thresholds[p].partition_point(|&t| value > t) as u32;
    }
    let config = HybridUintConfig::new(4, 1, 2).unwrap();
    let mut tokens = [[0; 2]; 14];
    for (token, &value) in tokens.iter_mut().zip(residuals) {
        let encoded = config.encode(pack_signed(value));
        *token = [encoded.token as u8, encoded.nbits as u8];
    }
    collector.insert(
        &bins[..collector.prepared.properties.len()],
        &tokens[..residuals.len()],
        [properties[0], properties[1]],
        weight,
    )?;
    collector.prepared.total = total;
    Ok(())
}

fn assert_same(actual: &SourceSampleCollector, expected: &SourceSampleCollector) {
    assert_eq!(actual.table, expected.table);
    assert_eq!(actual.cached_hashes, expected.cached_hashes);
    assert_eq!(actual.prepared.total, expected.prepared.total);
    assert_eq!(actual.prepared.counts, expected.prepared.counts);
    assert_eq!(actual.prepared.static_ids, expected.prepared.static_ids);
    assert_eq!(actual.prepared.residuals, expected.prepared.residuals);
    assert_eq!(actual.prepared.predictors, expected.prepared.predictors);
    assert_eq!(actual.prepared.quantization, expected.prepared.quantization);
    assert_eq!(
        actual.prepared.properties.len(),
        expected.prepared.properties.len()
    );
    for (a, b) in actual
        .prepared
        .properties
        .iter()
        .zip(&expected.prepared.properties)
    {
        match (a, b) {
            (GatherColumn::Packed(a), GatherColumn::Packed(b)) => assert_eq!(a, b),
            (GatherColumn::Static(a), GatherColumn::Static(b)) => assert_eq!(a, b),
            _ => panic!("property column kind changed"),
        }
    }
}

#[test]
fn specialized_residual_token_matches_generic_full_width_shapes() {
    let config = HybridUintConfig::new(4, 1, 2).unwrap();
    let check = |packed: u32| {
        let residual = ((packed >> 1) as i32) ^ -((packed & 1) as i32);
        let expected = config.encode(packed);
        assert_eq!(
            residual_token(residual),
            [expected.token as u8, expected.nbits as u8],
            "packed={packed:#010x}, residual={residual}",
        );
    };
    for packed in 0..=u32::from(u16::MAX) {
        check(packed);
    }
    // Every non-direct token, including both extremes and interior points of
    // its extra-bit payload, through exponent 31 and both i32 endpoints.
    for exponent in 4..32 {
        let nbits = exponent - 3;
        let mask = (1u32 << nbits) - 1;
        for high in 0..2 {
            for low in 0..4 {
                for extra in [0, 1, mask / 2, mask - 1, mask] {
                    check((1u32 << exponent) | (high << (exponent - 1)) | (extra << 2) | low);
                }
            }
        }
    }
    let mut state = 0x718a_b4d3u32;
    for _ in 0..4096 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        check(state);
    }
}

fn quantization(ids: Vec<usize>) -> PropertyQuantization {
    let thresholds = ids
        .iter()
        .map(|&id| {
            if id < 2 {
                (-1024..1024).step_by(2).collect()
            } else if id % 3 == 0 {
                vec![i32::MIN, -512, -511, -1, 0, 1, 511, 512, i32::MAX]
            } else if id % 3 == 1 {
                (0..255).map(|i| -508 + i * 4).collect()
            } else {
                vec![]
            }
        })
        .collect();
    PropertyQuantization {
        properties: ids,
        thresholds,
    }
}

#[test]
fn specialized_push_preserves_columns_hashes_collisions_and_append() {
    for ids in [
        vec![0],
        vec![1, 0],
        vec![9, 1, 0, 2, 255],
        (193..256).rev().collect(),
    ] {
        let quantization = quantization(ids);
        for predictor_count in [1, 2, 14] {
            let predictors: Vec<_> = (0..predictor_count)
                .map(|id| Predictor::try_from(id as u32).unwrap())
                .collect();
            for cache in [false, true] {
                let make = || {
                    let mut c =
                        SourceSampleCollector::new(&predictors, &quantization, cache).unwrap();
                    // Exercise both the cached prefix and its uncached tail.
                    c.hash_limit = if cache { 7 } else { 0 };
                    c
                };
                let mut actual = make();
                let mut expected = make();
                let values = [
                    i32::MIN,
                    i32::MAX,
                    -513,
                    -512,
                    -511,
                    -1,
                    0,
                    1,
                    510,
                    511,
                    512,
                    513,
                ];
                for extra in [1, 0, 256, 0] {
                    actual.prepare(extra).unwrap();
                    expected.prepare(extra).unwrap();
                    assert_same(&actual, &expected);
                    let slots = actual.table.len();
                    for i in 0..42 {
                        let mut properties = [0; 256];
                        for (id, value) in properties.iter_mut().enumerate().skip(2) {
                            *value = values[(i / 3 + id) % values.len()];
                        }
                        [properties[0], properties[1]] = [
                            [i32::MIN, i32::MAX],
                            [-1024, 1022],
                            [0, 0],
                            [i32::MAX, i32::MIN],
                            [i32::MIN, i32::MAX],
                        ][i / 9];
                        let residuals: Vec<_> = (0..predictor_count)
                            .map(|p| values[(i / 3 + p) % values.len()])
                            .collect();
                        let properties = &properties[..actual.required_properties];
                        for weight in [0, 1, 2, 65_534, 65_535, 65_536, 65_537] {
                            actual.push(properties, &residuals, weight).unwrap();
                            reference_push(&mut expected, properties, &residuals, weight).unwrap();
                        }
                    }
                    assert_eq!(actual.table.len(), slots, "overshoot resized slots");
                    assert_same(&actual, &expected);
                }
                assert!(actual.prepared.counts.contains(&u16::MAX));
                assert!(actual.prepared.len() > actual.hash_limit);
                let mut batch = make();
                let mut reference_batch = make();
                batch.prepare(1).unwrap();
                reference_batch.prepare(1).unwrap();
                let properties = [i32::MAX; 256];
                let residuals = vec![i32::MIN; predictor_count];
                batch.push(&properties, &residuals, 65_537).unwrap();
                reference_push(&mut reference_batch, &properties, &residuals, 65_537).unwrap();
                actual.append(vec![batch]).unwrap();
                expected.append(vec![reference_batch]).unwrap();
                assert_same(&actual, &expected);
                actual.prepare(0).unwrap();
                expected.prepare(0).unwrap();
                assert_same(&actual, &expected);
                // Append does not replace the destination's last-ID cache.
                actual.push(&properties, &residuals, 1).unwrap();
                reference_push(&mut expected, &properties, &residuals, 1).unwrap();
                assert_same(&actual, &expected);
            }
        }
    }
}

#[test]
fn static_bin_cache_keeps_wide_bins_and_exact_id_changes() {
    let mut c =
        SourceSampleCollector::new(&[Predictor::Zero], &quantization(vec![1, 0]), false).unwrap();
    c.prepare(1).unwrap();
    let ids = [i32::MIN, -1024, -1023, 0, 1022, 1023, i32::MAX, 0, i32::MIN];
    for value in ids {
        let properties = [value, value.wrapping_neg()];
        for _ in 0..2 {
            c.push(&properties, &[0], 1).unwrap();
            for (id, &property) in properties.iter().enumerate() {
                let p = 1 - id;
                let bin =
                    c.prepared.quantization.thresholds[p].partition_point(|&t| property > t) as u32;
                assert_eq!(c.static_bins[id], Some((property, bin)));
            }
        }
    }
    assert!(c.prepared.properties.iter().any(|column| match column {
        GatherColumn::Static(values) => values.iter().any(|&bin| bin > 255),
        GatherColumn::Packed(_) => false,
    }));
}

#[test]
fn specialized_push_keeps_dimension_and_weight_error_precedence() {
    for ids in [vec![0], vec![1], vec![255, 0], vec![9, 1, 0]] {
        let quantization = quantization(ids);
        let mut actual =
            SourceSampleCollector::new(&[Predictor::Zero], &quantization, false).unwrap();
        let mut expected =
            SourceSampleCollector::new(&[Predictor::Zero], &quantization, false).unwrap();
        let required = actual.required_properties;
        let properties = [i32::MAX; 256];
        for prepared in [false, true] {
            if prepared {
                actual.prepare(1).unwrap();
                expected.prepare(1).unwrap();
                actual.push(&properties, &[0], 1).unwrap();
                reference_push(&mut expected, &properties, &[0], 1).unwrap();
            }
            for width in [0, 1, required - 1, required, 256] {
                for residuals in [&[][..], &[i32::MIN][..], &[0, 0][..]] {
                    for weight in [0, 1, u32::MAX] {
                        let cache = actual.static_bins;
                        let a = actual.push(&properties[..width], residuals, weight);
                        let b =
                            reference_push(&mut expected, &properties[..width], residuals, weight);
                        if a.is_err() || weight == 0 {
                            assert_eq!(actual.static_bins, cache);
                        }
                        assert_eq!(a.map_err(|e| e.to_string()), b.map_err(|e| e.to_string()));
                        assert_same(&actual, &expected);
                    }
                }
            }
        }
    }
}
