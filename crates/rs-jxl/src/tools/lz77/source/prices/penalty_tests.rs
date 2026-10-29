use super::super::{
    ContextCost, Costs, HybridUintConfig, LENGTH_COST, Literal, MIN_MATCH, Match, SourceConfig,
    SourceFastRanges, SourceLz77Method, WINDOW_SIZE, fast_stream, static_cost_with_penalty,
    static_distance, wire_token_count,
};
use super::{CONTEXTS, Penalties};
use std::collections::BTreeMap;

fn costs_with_penalties(rows: impl IntoIterator<Item = (u32, f32)>) -> Costs {
    Costs {
        literals: rows
            .into_iter()
            .map(|(context, penalty)| {
                (
                    context,
                    ContextCost {
                        bits: BTreeMap::new(),
                        penalty,
                    },
                )
            })
            .collect(),
        distances: ContextCost {
            bits: BTreeMap::new(),
            penalty: 0.0,
        },
    }
}

#[test]
fn bounded_penalties_preserve_bits_holes_and_wide_contexts() {
    let costs = costs_with_penalties([
        (0, -0.0),
        (3, f32::from_bits(0x3f80_0001)),
        (16383, 2.375),
        (65535, 6.0),
        (65536, 0.0),
        (u32::MAX, 1.125),
    ]);
    let penalties = Penalties::new(&costs).unwrap();
    assert_eq!(penalties.dense.len(), CONTEXTS);
    assert_eq!(
        penalties.dense.len() * std::mem::size_of::<f32>(),
        256 << 10
    );
    for context in (0..65538).chain([u32::MAX - 1, u32::MAX]) {
        assert_eq!(
            penalties.get(context).to_bits(),
            costs.penalty(context).to_bits(),
            "context={context}"
        );
    }
    for rows in [vec![], vec![(65536, 1.5), (u32::MAX, 2.5)]] {
        let costs = costs_with_penalties(rows);
        let penalties = Penalties::new(&costs).unwrap();
        assert!(penalties.dense.is_empty());
        for context in [0, 65535, 65536, u32::MAX] {
            assert_eq!(penalties.get(context), costs.penalty(context));
        }
    }
}

#[test]
fn penalty_domain_is_independent_of_the_dp_price_row_budget() {
    let costs =
        costs_with_penalties((0..13278).map(|context| (context, (context % 49) as f32 / 8.0)));
    let penalties = Penalties::new(&costs).unwrap();
    assert_eq!(penalties.dense.len(), 13278);
    for context in 0..13280 {
        assert_eq!(
            penalties.get(context).to_bits(),
            costs.penalty(context).to_bits()
        );
    }
}

#[test]
fn static_seed_prices_keep_original_f32_addition_order() {
    let costs = costs_with_penalties([
        (0, -0.0),
        (7, f32::from_bits(0x3f80_0001)),
        (65535, 5.999_999_5),
        (65536, 0.0),
        (u32::MAX, 2.375),
    ]);
    let penalties = Penalties::new(&costs).unwrap();
    for context in [0, 1, 7, 65535, 65536, u32::MAX] {
        for length in [3, 4, 5, 6, 7, 10, 127, 128, 129, 4096, i32::MAX as usize] {
            for symbol in [
                0,
                1,
                119,
                120,
                127,
                128,
                129,
                255,
                256,
                WINDOW_SIZE as u32,
                u32::MAX,
            ] {
                let matched = Match {
                    length,
                    distance: 1,
                    symbol,
                };
                let encoded = HybridUintConfig::new(1, 0, 0)
                    .unwrap()
                    .encode((length - MIN_MATCH) as u32);
                let length_cost = LENGTH_COST[(encoded.token as usize).min(LENGTH_COST.len() - 1)]
                    as f32
                    + encoded.nbits as f32;
                let expected = length_cost + static_distance(symbol) + costs.penalty(context);
                assert_eq!(
                    static_cost_with_penalty(matched, penalties.get(context)).to_bits(),
                    expected.to_bits(),
                    "context={context} length={length} symbol={symbol}"
                );
            }
        }
    }
}

fn compare_seed<const N: usize>(inputs: &[&[Literal]], config: &SourceConfig) {
    let costs = Costs::raw(inputs, config);
    let compiled = Penalties::new(&costs).unwrap();
    let sparse = Penalties {
        costs: &costs,
        dense: Vec::new(),
    };
    let mut compiled_saving = 0.123_456f32;
    let mut sparse_saving = compiled_saving;
    for (index, input) in inputs.iter().enumerate() {
        let values: Vec<_> = input.iter().map(|literal| literal.value).collect();
        let sums = costs.cumulative(input, config).unwrap();
        let width = [0, 1, 37][index % 3];
        let actual = fast_stream::<N>(
            input,
            &values,
            width,
            config,
            &compiled,
            &sums,
            &mut compiled_saving,
        )
        .unwrap();
        let expected = fast_stream::<N>(
            input,
            &values,
            width,
            config,
            &sparse,
            &sums,
            &mut sparse_saving,
        )
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(wire_token_count(&actual), wire_token_count(&expected));
        assert_eq!(compiled_saving.to_bits(), sparse_saving.to_bits());
    }
}

#[test]
fn shared_penalties_keep_seed_commands_and_accumulated_savings_bitwise() {
    let mut input: Vec<_> = (0..8192u32)
        .map(|i| Literal {
            context: [0, 7, 13277, 65535, 65536, u32::MAX][i as usize % 6],
            value: if i % 149 < 31 { 0 } else { i % 97 },
        })
        .collect();
    for bounded in [false, true] {
        if bounded {
            for literal in &mut input {
                literal.context %= 16384;
            }
        }
        let inputs = [&input[..4096], &[][..], &input[4096..], &input[..2]];
        for method in [SourceLz77Method::Optc256, SourceLz77Method::Lz77b7w3f] {
            for fast_ranges in [SourceFastRanges::Disabled, SourceFastRanges::Both] {
                let config = SourceConfig {
                    method,
                    fast_ranges,
                    max_match_length: 129,
                    ..SourceConfig::default()
                };
                compare_seed::<1>(&inputs, &config);
                compare_seed::<3>(&inputs, &config);
                compare_seed::<7>(&inputs, &config);
                compare_seed::<15>(&inputs, &config);
                compare_seed::<31>(&inputs, &config);
            }
        }
    }
}
