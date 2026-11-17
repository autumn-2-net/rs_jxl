//! Bounded, PJA-inspired coefficient-order experiments, not a source-code port.
//!
//! These six policies only permute the existing DCT8 wire order. They introduce
//! no predictors, coefficient changes, decoder syntax, or automatic search, and
//! do not guarantee smaller output. Compare complete encoded sizes, including
//! order signaling, before choosing an experimental policy.
//!
//! Positions address already-transposed JXL blocks: `(u, v) = (p % 8, p / 8)`.
//! Do not transpose them again. DC is fixed; the 63 AC positions comprise 14
//! axis-edge positions and 49 interior positions. Channels are handled separately.

use super::coefficient_order::{DEFAULT_DCT8_ORDER, Dct8Orders, OrderError};

/// A fixed candidate transformation of the current, potentially adaptive order.
#[derive(Default, Clone, Copy, Debug, Eq, PartialEq)]
pub enum Policy {
    /// Preserve all source state, including an explicitly retained transmit bit.
    #[default]
    Source,
    /// Use the default DCT8 zigzag on every channel, without order signaling.
    Zigzag,
    /// Stable-partition AC positions, placing axis edges before the interior.
    EdgeFirst,
    /// Stable-partition AC positions, placing the interior before axis edges.
    EdgeLast,
    /// Increasing `(u + v) / 3` band, then edges, then source-order rank.
    BandEdgeFirst,
    /// Increasing `(u + v) / 3` band, then interior, then source-order rank.
    BandEdgeLast,
}

/// Apply one bounded policy without modifying the source or its coefficients.
///
/// Except for `Source`, signaling is enabled exactly when at least one channel
/// differs from the default. Within each category, the current source order is
/// retained, not replaced by numeric-position or default-zigzag ordering. The
/// result is validated by the existing wire-order constructor.
pub fn apply(source: &Dct8Orders, policy: Policy) -> Result<Dct8Orders, OrderError> {
    let (banded, edge_first) = match policy {
        Policy::Source => return Ok(source.clone()),
        Policy::Zigzag => return Ok(Dct8Orders::default()),
        Policy::EdgeFirst => (false, true),
        Policy::EdgeLast => (false, false),
        Policy::BandEdgeFirst => (true, true),
        Policy::BandEdgeLast => (true, false),
    };
    let mut positions = *source.positions();
    for order in &mut positions {
        // Stable sorting makes the channel's source rank the final tie-breaker.
        order[1..].sort_by_key(|&position| {
            let (u, v) = (position % 8, position / 8);
            let edge = u == 0 || v == 0;
            let band = if banded { (u + v) / 3 } else { 0 };
            (band, edge != edge_first)
        });
    }
    let transmit = positions.iter().any(|order| order != &DEFAULT_DCT8_ORDER);
    Dct8Orders::from_positions(positions, transmit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::ac_tokens::{
        BlockContextModel, Dct8Input, Dct8Plane, TokenizeLimits, tokenize_dct8,
    };
    use crate::jpeg::coefficient_order::{DEFAULT_DCT8_ORDER_LUT, tokenize_permutation};

    const POLICIES: [Policy; 6] = [
        Policy::Source,
        Policy::Zigzag,
        Policy::EdgeFirst,
        Policy::EdgeLast,
        Policy::BandEdgeFirst,
        Policy::BandEdgeLast,
    ];

    fn source() -> Dct8Orders {
        let positions = std::array::from_fn(|channel| {
            std::array::from_fn(|rank| {
                if rank == 0 {
                    return 0;
                }
                let ac_rank = rank - 1;
                let index = match channel {
                    0 => 62 - ac_rank,
                    1 => (ac_rank + 17) % 63,
                    _ => (ac_rank * 5 + 11) % 63,
                };
                DEFAULT_DCT8_ORDER[index + 1]
            })
        });
        Dct8Orders::from_positions(positions, true).unwrap()
    }

    fn edge(position: u32) -> bool {
        (1..=7).contains(&position) || (position != 0 && position % 8 == 0)
    }

    fn band(position: u32) -> u32 {
        (position % 8 + position / 8) / 3
    }

    // Independent bucket enumeration rather than the implementation's sort key.
    fn expected(order: &[u32; 64], policy: Policy) -> Vec<u32> {
        match policy {
            Policy::Source => return order.to_vec(),
            Policy::Zigzag => return DEFAULT_DCT8_ORDER.to_vec(),
            _ => {}
        }
        let banded = matches!(policy, Policy::BandEdgeFirst | Policy::BandEdgeLast);
        let first = matches!(policy, Policy::EdgeFirst | Policy::BandEdgeFirst);
        let mut output = vec![0];
        for bucket in 0..if banded { 5 } else { 1 } {
            for category in [first, !first] {
                output.extend(order[1..].iter().copied().filter(|&position| {
                    (!banded || band(position) == bucket) && edge(position) == category
                }));
            }
        }
        output
    }

    #[test]
    fn every_policy_is_a_dc_first_permutation_and_preserves_input() {
        for source in [Dct8Orders::default(), source()] {
            let original = source.clone();
            for policy in POLICIES {
                let result = apply(&source, policy).unwrap();
                for (input, output) in source.positions().iter().zip(result.positions()) {
                    assert_eq!(output[0], 0, "{policy:?}");
                    let mut sorted = *output;
                    sorted.sort_unstable();
                    assert_eq!(sorted, std::array::from_fn(|i| i as u32));
                    assert_eq!(output.as_slice(), expected(input, policy), "{policy:?}");
                    assert_eq!(output[1..].iter().filter(|&&p| edge(p)).count(), 14);
                    assert_eq!(output[1..].iter().filter(|&&p| !edge(p)).count(), 49);
                }
                assert_eq!(result, apply(&source, policy).unwrap());
                assert_eq!(result, apply(&result, policy).unwrap());
                assert_eq!(source, original);
                if policy != Policy::Source {
                    assert_eq!(
                        result.used_orders(),
                        u16::from(result.positions().iter().any(|p| p != &DEFAULT_DCT8_ORDER))
                    );
                }
            }
        }
    }

    #[test]
    fn whole_order_policies_stably_partition_exactly_14_edges_and_49_interior() {
        let source = source();
        for policy in [Policy::EdgeFirst, Policy::EdgeLast] {
            let result = apply(&source, policy).unwrap();
            for (input, output) in source.positions().iter().zip(result.positions()) {
                let source_edges: Vec<_> =
                    input[1..].iter().copied().filter(|&p| edge(p)).collect();
                let source_interior: Vec<_> =
                    input[1..].iter().copied().filter(|&p| !edge(p)).collect();
                if policy == Policy::EdgeFirst {
                    assert_eq!(&output[1..15], source_edges);
                    assert_eq!(&output[15..], source_interior);
                } else {
                    assert_eq!(&output[1..50], source_interior);
                    assert_eq!(&output[50..], source_edges);
                }
            }
        }
    }

    #[test]
    fn banded_policies_prioritize_frequency_then_category_then_source_rank() {
        let source = source();
        for policy in [Policy::BandEdgeFirst, Policy::BandEdgeLast] {
            let result = apply(&source, policy).unwrap();
            for (input, output) in source.positions().iter().zip(result.positions()) {
                assert!(output[1..].windows(2).all(|p| band(p[0]) <= band(p[1])));
                for bucket in 0..5 {
                    let within_band: Vec<_> = output[1..]
                        .iter()
                        .copied()
                        .filter(|&p| band(p) == bucket)
                        .collect();
                    let first = policy == Policy::BandEdgeFirst;
                    assert!(within_band.windows(2).all(|p| {
                        usize::from(edge(p[0]) != first) <= usize::from(edge(p[1]) != first)
                    }));
                    for category in [true, false] {
                        let select = |p: &u32| band(*p) == bucket && edge(*p) == category;
                        assert_eq!(
                            output[1..]
                                .iter()
                                .copied()
                                .filter(select)
                                .collect::<Vec<_>>(),
                            input[1..]
                                .iter()
                                .copied()
                                .filter(select)
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn channel_orders_are_independent_and_one_changed_channel_signals_all() {
        let source = source();
        for policy in POLICIES {
            let result = apply(&source, policy).unwrap();
            for channel in 0..3 {
                let isolated =
                    Dct8Orders::from_positions([source.positions()[channel]; 3], true).unwrap();
                let isolated = apply(&isolated, policy).unwrap();
                assert_eq!(result.positions()[channel], isolated.positions()[0]);
                if policy != Policy::Zigzag {
                    assert_ne!(
                        result.positions()[channel],
                        result.positions()[(channel + 1) % 3]
                    );
                }
            }
        }
        for channel in 0..3 {
            let mut positions = [DEFAULT_DCT8_ORDER; 3];
            positions[channel].swap(1, 2);
            let source = Dct8Orders::from_positions(positions, true).unwrap();
            assert_eq!(apply(&source, Policy::Source).unwrap(), source);
            assert_eq!(apply(&source, Policy::Source).unwrap().used_orders(), 1);
            assert_eq!(apply(&source, Policy::Zigzag).unwrap().used_orders(), 0);
        }
    }

    #[test]
    fn source_is_an_exact_no_op_even_with_an_explicit_default_transmit_bit() {
        assert_eq!(Policy::default(), Policy::Source);
        for transmit in [false, true] {
            let source = Dct8Orders::from_positions([DEFAULT_DCT8_ORDER; 3], transmit).unwrap();
            let result = apply(&source, Policy::default()).unwrap();
            assert_eq!(result, source);
            assert_eq!(result.used_orders(), u16::from(transmit));
            assert_eq!(result.tokens(), source.tokens());
            assert_eq!(
                apply(&source, Policy::Zigzag).unwrap(),
                Dct8Orders::default()
            );
        }
        let source = source();
        assert_eq!(apply(&source, Policy::Source).unwrap(), source);
        assert_eq!(
            apply(&source, Policy::Zigzag).unwrap(),
            Dct8Orders::default()
        );
    }

    #[test]
    fn all_policies_support_existing_permutation_and_ac_tokenization() {
        let coefficients: [[i32; 64]; 3] = std::array::from_fn(|channel| {
            std::array::from_fn(|p| {
                if p == 0 {
                    101
                } else if channel == 0 || (edge(p as u32) == (channel == 1)) {
                    if p % 2 == 0 { 7 } else { -3 }
                } else {
                    0
                }
            })
        });
        let original = coefficients;
        let limits = TokenizeLimits {
            max_tokens: 3 * 64,
            max_scratch_bytes: 3,
        };
        for source in [Dct8Orders::default(), source()] {
            for policy in POLICIES {
                let result = apply(&source, policy).unwrap();
                let mut expected_tokens = Vec::new();
                for order in result.positions() {
                    let ranks = order.map(|p| DEFAULT_DCT8_ORDER_LUT[p as usize]);
                    let tokens = tokenize_permutation(&ranks, 1).unwrap();
                    if result.used_orders() != 0 {
                        expected_tokens.extend(tokens);
                    }
                }
                assert_eq!(result.tokens(), expected_tokens);
                let input = Dct8Input {
                    width_blocks: 1,
                    height_blocks: 1,
                    planes: coefficients.each_ref().map(|plane| Dct8Plane {
                        coefficients: plane,
                        width_blocks: 1,
                        height_blocks: 1,
                        horizontal_shift: 0,
                        vertical_shift: 0,
                    }),
                    orders: result.positions(),
                    qdc: &[0],
                    qf: &[1],
                };
                let model = BlockContextModel::default();
                let tokens = tokenize_dct8(&input, model, limits).unwrap();
                assert!(!tokens.is_empty());
                assert!(tokens.len() <= limits.max_tokens);
                assert!(
                    tokens
                        .iter()
                        .all(|t| !t.is_length && t.context < model.num_ac_contexts())
                );
                assert_eq!(tokens, tokenize_dct8(&input, model, limits).unwrap());
                assert_eq!(coefficients, original);
            }
        }
    }
}
