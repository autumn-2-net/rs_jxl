use super::*;

#[test]
fn fresh_pair_matches_scalar_with_full_width_extra_counts() {
    fn check<T: PropertyValue>(
        samples: &TreeSamples,
        indices: &[usize],
        values: &[T],
        bins: usize,
    ) {
        let tokens = [&samples.residuals[0][..], &samples.residuals[1][..]];
        let bounds = tokens.map(|tokens| {
            indices
                .iter()
                .map(|&i| u64::from(samples.counts[i]) * u64::from(tokens[i].nbits))
                .sum()
        });
        for strides in [[256, 264], [264, 256]] {
            let [sa, sb] = strides;
            // Sentinel cells also check the borrowed-buffer boundaries.
            let (mut a, mut b) = (vec![0; bins * sa + 2], vec![0; bins * sb + 2]);
            let (mut ea, mut eb) = (vec![0; bins + 2], vec![0; bins + 2]);
            for counts in [&mut a, &mut b] {
                counts[0] = u32::MAX;
                *counts.last_mut().unwrap() = u32::MAX;
            }
            for extra in [&mut ea, &mut eb] {
                extra[0] = u64::MAX;
                *extra.last_mut().unwrap() = u64::MAX;
            }
            let (mut expected_a, mut expected_b) = (a.clone(), b.clone());
            let (mut expected_ea, mut expected_eb) = (ea.clone(), eb.clone());
            collect_fresh_pair(
                samples,
                indices,
                values,
                tokens,
                strides,
                bounds,
                [&mut a[1..bins * sa + 1], &mut b[1..bins * sb + 1]],
                [&mut ea[1..bins + 1], &mut eb[1..bins + 1]],
            );
            for (token, stride, counts, extra) in [
                (tokens[0], sa, &mut expected_a, &mut expected_ea),
                (tokens[1], sb, &mut expected_b, &mut expected_eb),
            ] {
                collect_mapped::<_, false, false>(
                    samples,
                    indices,
                    values,
                    token,
                    stride,
                    &[],
                    None,
                    &mut counts[1..bins * stride + 1],
                    &mut extra[1..bins + 1],
                );
            }
            assert_eq!(
                (a, b, ea, eb),
                (expected_a, expected_b, expected_ea, expected_eb)
            );
        }
    }
    for bins in [1, 17, 513] {
        for records in [0, 1, 511, 512, 1025] {
            for mode in 0..4 {
                let samples = TreeSamples {
                    properties: Vec::new(),
                    thresholds: Vec::new(),
                    global_max_symbols: 0,
                    counts: (0..records)
                        .map(|i| {
                            if mode == 0 {
                                1
                            } else if mode == 1 {
                                [1, 17, 65535][i % 3]
                            } else {
                                65535
                            }
                        })
                        .collect(),
                    residuals: (0..2)
                        .map(|p| {
                            (0..records)
                                .map(|i| ResidualToken {
                                    token: ((i * 31 + p * 17) % 256) as u8,
                                    nbits: if mode == 3 {
                                        // Exactly u32::MAX in each half for 512+ records.
                                        if i < 257 {
                                            255
                                        } else if i == 257 {
                                            2
                                        } else {
                                            0
                                        }
                                    } else {
                                        ((i * 7 + p * 13) % 256) as u8
                                    },
                                })
                                .collect()
                        })
                        .collect(),
                };
                let indices: Vec<_> = (0..records).rev().collect();
                let values: Vec<_> = (0..records).map(|i| (i * 37 % bins) as u32).collect();
                check(&samples, &indices, &values, bins);
                if bins < 256 {
                    let bytes: Vec<_> = values.iter().map(|&n| n as u8).collect();
                    check(&samples, &indices, &bytes, bins);
                }
                let filtered: Vec<_> = indices.into_iter().filter(|i| i % 7 != 0).collect();
                check(&samples, &filtered, &values, bins);
            }
        }
    }
}
