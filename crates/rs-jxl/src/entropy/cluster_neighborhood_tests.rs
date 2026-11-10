use super::*;

fn fixture() -> (Vec<Vec<u64>>, Vec<u8>) {
    let mut state = 19u32;
    let map: Vec<_> = (0..1024)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 31) as u8
        })
        .collect();
    let raw = map
        .iter()
        .map(|&id| if id == 0 { vec![2, 1] } else { vec![1, 2] })
        .collect();
    (raw, map)
}

#[test]
fn positive_population_loss_can_be_paid_by_map_savings() {
    let (raw, map) = fixture();
    let mut totals = [vec![0; 2], vec![0; 2]];
    for (counts, &id) in raw.iter().zip(&map) {
        for (to, n) in totals[id as usize].iter_mut().zip(counts) {
            *to += n;
        }
    }
    let a = Histogram::new(&totals[0]).unwrap();
    let b = Histogram::new(&totals[1]).unwrap();
    assert!(
        pair_population_cost(&a, &b).unwrap()
            > a.population_cost().unwrap() + b.population_cost().unwrap()
    );
    let original = raw.clone();
    let (plans, work) = propose(&raw, &map, 2, &ContextMapOptions::default()).unwrap();
    assert_eq!(raw, original);
    assert_eq!(plans.len(), 1);
    assert_eq!(work.merges, 1);
    assert_eq!(plans[0].map, vec![0; map.len()]);
    assert_eq!(
        plans[0].counts[0],
        vec![totals[0][0] + totals[1][0], totals[0][1] + totals[1][1]]
    );
    assert_eq!(map_bits(&plans[0].map, 1).unwrap(), 3);
}

#[test]
fn map_renumbering_preserves_context_positions_and_table_order() {
    let (map, order) = merged_map(&[2, 0, 1, 2, 3, 1], 1, 2);
    assert_eq!(map, [0, 1, 0, 0, 2, 0]);
    assert_eq!(order, [1, 0, 3]);
}

#[test]
fn bounded_chain_returns_only_first_and_last_and_is_deterministic() {
    let raw = vec![vec![1, 1]; 16];
    let map: Vec<_> = (0..16).collect();
    let search = ContextMapOptions {
        shortlist: 3,
        max_merges: 4,
    };
    let (plans, work) = propose(&raw, &map, 16, &search).unwrap();
    let (again, _) = propose(&raw, &map, 16, &search).unwrap();
    assert_eq!(work.merges, 4);
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].merges, 1);
    assert_eq!(plans[1].merges, 4);
    assert!(work.map_trials <= search.shortlist * search.max_merges);
    for (plan, repeated) in plans.iter().zip(again) {
        assert_eq!(plan.map, repeated.map);
        assert_eq!(plan.counts, repeated.counts);
        assert_eq!(plan.counts.len(), 16 - plan.merges);
        assert_eq!(plan.counts.iter().flatten().sum::<u64>(), 32);
    }
}

#[test]
fn syntax_boundaries_and_invalid_options() {
    assert_eq!(map_bits(&[0], 1).unwrap(), 0);
    assert_eq!(map_bits(&[0; 20], 1).unwrap(), 3);
    for n in [2, 4, 8, 9, 255, 256] {
        let map: Vec<_> = (0..n).map(|i| i as u8).collect();
        assert!(map_bits(&map, n).unwrap() > 0);
    }
    for (raw, map, n) in [
        (vec![vec![1]], vec![2], 1),
        (vec![vec![1]; 2], vec![0, 0], 2),
    ] {
        assert!(propose(&raw, &map, n, &ContextMapOptions::default()).is_err());
    }
    for search in [
        ContextMapOptions {
            shortlist: 0,
            max_merges: 1,
        },
        ContextMapOptions {
            shortlist: 1,
            max_merges: 0,
        },
    ] {
        assert!(propose(&[vec![1]], &[0], 1, &search).is_err());
    }
    assert!(
        propose(&[vec![1]], &[0], 1, &ContextMapOptions::default())
            .unwrap()
            .0
            .is_empty()
    );
}
