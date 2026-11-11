use super::*;

// Retain the old u32 tag/sentinel comparisons and use scalar exact extension,
// independent of the narrowed tags and the production prefix/cache filters.
fn old_metadata_matches(
    chain: &Chain<'_>,
    tags: &[u32],
    position: usize,
    keep_short: bool,
) -> (Vec<(usize, usize, u32)>, usize) {
    let slot = position & (chain.window - 1);
    let hash = tags[slot];
    let mut previous_slot = chain.chain[slot] as usize;
    let (mut previous_distance, mut best, mut visits) = (0, 0usize, 0);
    let limit = chain.maximum.min(chain.values.len() - position);
    let mut matches = Vec::new();
    loop {
        let distance = slot.wrapping_sub(previous_slot) & (chain.window - 1);
        if distance < previous_distance || distance > position.min(WINDOW_SIZE) {
            break;
        }
        previous_distance = distance;
        let length = if distance == 0 {
            0
        } else {
            (0..limit)
                .take_while(|&i| {
                    chain.values[position + i] == chain.values[position - distance + i]
                })
                .count()
        };
        if length >= MIN_MATCH && (keep_short || length >= best.saturating_sub(2)) {
            matches.push((length, distance, chain.codes.symbol(distance)));
            best = best.max(length);
        }
        visits += 1;
        if visits >= chain.limit {
            break;
        }
        if chain.numzeros >= 3 && length > chain.numzeros {
            let next = chain.chainz[previous_slot] as usize;
            if previous_slot == next {
                break;
            }
            previous_slot = next;
            if chain.zeros[previous_slot] as usize != chain.numzeros {
                break;
            }
        } else {
            let next = chain.chain[previous_slot] as usize;
            if previous_slot == next {
                break;
            }
            previous_slot = next;
            if tags[previous_slot] != hash {
                break;
            }
        }
    }
    (matches, visits)
}

#[test]
fn narrowed_chain_tags_preserve_old_u32_traversal_at_collisions_wrap_and_tails() {
    // Pigeonhole search guarantees distinct mandatory prefixes in one bucket.
    let mut seen = vec![None; 1 << 15];
    let (first, second) = (0..=1 << 15)
        .find_map(|value| {
            let prefix = [value, u32::MAX, 17];
            let key = hash(&prefix, 0) as usize & ((1 << 15) - 1);
            if let Some(previous) = seen[key] {
                Some((previous, prefix))
            } else {
                seen[key] = Some(prefix);
                None
            }
        })
        .expect("32769 distinct prefixes must collide in 32768 buckets");
    assert_ne!(first, second);
    assert_eq!(hash(&first, 0) & 32767, hash(&second, 0) & 32767);
    let mut values = Vec::new();
    for block in 0..16 {
        values.extend(first);
        values.extend([11, 12, 13, block]);
        values.extend(second);
        values.extend([11, 12, 13, block]);
        values.extend(std::iter::repeat_n(0, 3 + block as usize % 7));
        values.extend([u32::MAX, 0, 0, 0, 0, 19]);
    }
    for window in [1, 2, 4, 8, 32, 64] {
        for cap in [1, 3, 256] {
            for maximum in [3, 8, 97] {
                for keep_short in [false, true] {
                    for cached in [false, true] {
                        let mut chain =
                            Chain::with_window(&values, 37, cap, maximum, window).unwrap();
                        let mut old_tags = vec![u32::MAX; window];
                        assert!(chain.val.iter().all(|&tag| tag == u16::MAX));
                        assert_eq!(std::mem::size_of_val(chain.val.as_slice()), 2 * window);
                        if cached {
                            chain.cache = MatchCache::new(window).unwrap();
                        }
                        for position in 0..values.len() {
                            let key = hash(&values, position) & 32767;
                            assert!(key < u32::from(u16::MAX));
                            chain.update_range(position, position + 1);
                            old_tags[position & (window - 1)] = key;
                            for (&narrow, &old) in chain.val.iter().zip(&old_tags) {
                                assert_eq!(
                                    narrow,
                                    if old == u32::MAX {
                                        u16::MAX
                                    } else {
                                        old as u16
                                    }
                                );
                            }
                            let (expected, visits) =
                                old_metadata_matches(&chain, &old_tags, position, keep_short);
                            let mut actual = Vec::new();
                            chain.find_matches(position, keep_short, |m| {
                                actual.push((m.length, m.distance, m.symbol));
                            });
                            assert_eq!(
                                actual, expected,
                                "p={position} window={window} cap={cap} max={maximum} short={keep_short} cached={cached}"
                            );
                            assert_eq!(chain.last_visits, visits);
                        }
                    }
                }
            }
        }
    }
}
