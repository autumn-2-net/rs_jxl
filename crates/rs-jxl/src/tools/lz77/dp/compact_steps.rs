// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77.cc::ApplyLZ77_Optimal::CompactStep at b866b3d. This replaces
//! endpoint NODE storage only, not the existing exact relaxation engine.

use super::super::{WINDOW_SIZE, special_distance};
use super::{DistanceCodes, Step, reserved};
use crate::{Error, Result};

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct CompactStep {
    length: u32,
    // Zero is a literal; matches store the source coded distance plus one.
    distance: u32,
    cost: f32,
}

impl Default for CompactStep {
    fn default() -> Self {
        Self {
            length: 0,
            distance: 0,
            cost: f32::MAX,
        }
    }
}

pub(super) enum Steps {
    Wide(Vec<Step>),
    Compact {
        nodes: Vec<CompactStep>,
        codes: DistanceCodes,
        width: usize,
    },
}

impl Steps {
    pub(super) fn new(size: usize, compact_width: Option<usize>) -> Result<Self> {
        // Source input indices are signed. Check before allocating or casting.
        if compact_width.is_some() && size > i32::MAX as usize {
            return Err(Error::Limit("compact LZ77 DP input length"));
        }
        let count = size
            .checked_add(1)
            .ok_or(Error::Limit("LZ77 DP input length"))?;
        if let Some(width) = compact_width {
            let mut nodes = reserved(count)?;
            nodes.resize(count, CompactStep::default());
            nodes[0].cost = 0.0;
            Ok(Self::Compact {
                nodes,
                codes: DistanceCodes::new(width, WINDOW_SIZE),
                width,
            })
        } else {
            let mut nodes = reserved(count)?;
            nodes.resize(count, Step::default());
            nodes[0].cost = 0.0;
            Ok(Self::Wide(nodes))
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Wide(nodes) => nodes.len(),
            Self::Compact { nodes, .. } => nodes.len(),
        }
    }

    pub(super) fn get(&self, position: usize) -> Step {
        match self {
            Self::Wide(nodes) => nodes[position],
            Self::Compact { nodes, .. } => {
                let node = nodes[position];
                Step {
                    length: node.length,
                    distance: node.distance,
                    cost: f64::from(node.cost),
                }
            }
        }
    }

    pub(super) fn set(&mut self, position: usize, step: Step) {
        match self {
            Self::Wide(nodes) => nodes[position] = step,
            Self::Compact { nodes, .. } => {
                // The source float route rounds at the caller's additions,
                // never here. Precise-ring must not select this representation.
                debug_assert_eq!(step.cost, f64::from(step.cost as f32));
                nodes[position] = CompactStep {
                    length: step.length,
                    distance: step.distance,
                    cost: step.cost as f32,
                };
            }
        }
    }

    pub(super) fn encode_distance(&self, physical: usize) -> u32 {
        match self {
            Self::Wide(_) => physical as u32,
            Self::Compact { codes, .. } => {
                debug_assert!((1..=WINDOW_SIZE).contains(&physical));
                // At most WINDOW_SIZE + 120, so the literal bias cannot wrap.
                codes.symbol(physical) + 1
            }
        }
    }

    pub(super) fn physical_distance(&self, stored: u32) -> Result<u32> {
        let Self::Compact { width, .. } = self else {
            return Ok(stored);
        };
        if stored == 0 {
            return Ok(0);
        }
        let symbol = stored - 1;
        let distance = if *width != 0 && symbol < 120 {
            special_distance(symbol as usize, *width)
        } else {
            i128::from(symbol) + 1 - if *width == 0 { 0 } else { 120 }
        };
        if !(1..=WINDOW_SIZE as i128).contains(&distance) {
            return Err(Error::Encoding("invalid compact LZ77 distance".into()));
        }
        Ok(distance as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{Command, Literal, distance_symbol};
    use super::super::RangeParser;
    use super::*;

    #[test]
    fn source_compact_node_layout_limits_and_unreachable_sentinel_are_exact() {
        assert_eq!(size_of::<CompactStep>(), 12);
        assert_eq!(align_of::<CompactStep>(), 4);
        assert_eq!(std::mem::offset_of!(CompactStep, length), 0);
        assert_eq!(std::mem::offset_of!(CompactStep, distance), 4);
        assert_eq!(std::mem::offset_of!(CompactStep, cost), 8);
        assert!(Steps::new(i32::MAX as usize + 1, Some(0)).is_err());
        assert!(Steps::new(usize::MAX, Some(0)).is_err());
        assert!(Steps::new(usize::MAX, None).is_err());
        let mut compact = Steps::new(17, Some(37)).unwrap();
        assert_eq!(compact.len(), 18);
        assert_eq!(compact.get(0).cost, 0.0);
        let empty = compact.get(17);
        assert_eq!(
            (empty.length, empty.distance, empty.cost),
            (0, 0, f64::from(f32::MAX))
        );
        compact.set(
            17,
            Step {
                length: i32::MAX as u32,
                distance: WINDOW_SIZE as u32 + 120,
                cost: 1.5,
            },
        );
        let node = compact.get(17);
        assert_eq!(
            (node.length, node.distance, node.cost),
            (i32::MAX as u32, WINDOW_SIZE as u32 + 120, 1.5)
        );
        let Steps::Compact { nodes, .. } = compact else {
            panic!("compact node buffer")
        };
        assert_eq!(nodes.len() * size_of::<CompactStep>(), 12 * 18);
        let Steps::Wide(nodes) = Steps::new(17, None).unwrap() else {
            panic!("ordinary node buffer")
        };
        assert_eq!(nodes.len() * size_of::<Step>(), 16 * 18);
        assert!(nodes[17].cost.is_infinite());
        assert!(
            RangeParser::new_source_compact(0, 0)
                .unwrap()
                .backtrack(&[])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn compact_distance_codes_keep_literal_zero_bias_aliases_and_physical_window() {
        let nodes = Steps::new(0, Some(37)).unwrap();
        assert_eq!(nodes.encode_distance(37), 1); // source symbol 0, not a literal
        assert_eq!(nodes.encode_distance(1), 2);
        assert_eq!(nodes.physical_distance(0).unwrap(), 0);
        assert_eq!(nodes.physical_distance(1).unwrap(), 37);
        assert_eq!(nodes.physical_distance(2).unwrap(), 1);
        for width in [0, 1, 2, 37, 257, WINDOW_SIZE, usize::MAX] {
            let nodes = Steps::new(0, Some(width)).unwrap();
            let mut distances = vec![1, 2, 3, 127, 128, WINDOW_SIZE - 1, WINDOW_SIZE];
            for symbol in 0..120 {
                let distance = special_distance(symbol, width);
                if distance <= WINDOW_SIZE as i128 {
                    distances.push(distance as usize);
                }
            }
            for distance in distances {
                let stored = nodes.encode_distance(distance);
                assert_eq!(stored, distance_symbol(distance, width).unwrap() + 1);
                assert_eq!(nodes.physical_distance(stored).unwrap(), distance as u32);
            }
            assert!(nodes.physical_distance(u32::MAX).is_err());
        }
        assert!(
            Steps::new(0, Some(usize::MAX))
                .unwrap()
                .physical_distance(1)
                .is_err()
        );
    }

    fn decode(input: &[Literal], commands: &[Command]) {
        let mut output = Vec::new();
        for command in commands {
            let origin = output.len();
            match *command {
                Command::Literal(literal) => {
                    assert_eq!(literal, input[origin]);
                    output.push(literal.value);
                }
                Command::Match {
                    context,
                    length,
                    distance,
                } => {
                    assert_eq!(context, input[origin].context);
                    assert!(length >= 3 && origin + length as usize <= input.len());
                    assert!(distance != 0 && distance as usize <= origin);
                    for _ in 0..length {
                        output.push(output[output.len() - distance as usize]);
                    }
                }
            }
        }
        assert_eq!(
            output,
            input
                .iter()
                .map(|literal| literal.value)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn compact_and_wide_endpoints_equal_scalar_strict_dp_across_tree_and_float_boundaries() {
        for size in [3usize, 15, 16, 17, 31, 32, 33, 63, 65, 129, 257] {
            for width in [0, 1, 37, usize::MAX] {
                for expensive in [false, true] {
                    let input: Vec<_> = (0..size)
                        .map(|p| Literal {
                            value: u32::MAX,
                            context: if p.is_multiple_of(3) {
                                u32::MAX
                            } else {
                                p as u32
                            },
                        })
                        .collect();
                    let mut compact = RangeParser::new_source_compact(size, width).unwrap();
                    let mut wide = RangeParser::new_source(size).unwrap();
                    // Independent explicit endpoint oracle, in source origin
                    // order. Strict '<' retains the first origin/offer on ties.
                    let mut reference = vec![(f32::MAX, 0usize, 0usize); size + 1];
                    reference[0].0 = 0.0;
                    for origin in 0..size {
                        compact.advance(origin);
                        wide.advance(origin);
                        assert_eq!(compact.cost(origin), f64::from(reference[origin].0));
                        assert_eq!(wide.cost(origin), compact.cost(origin));
                        let base = reference[origin].0;
                        let literal_cost = base + if expensive { 16_777_216.0 } else { 1.25 };
                        compact.literal(origin, f64::from(literal_cost));
                        wide.literal(origin, f64::from(literal_cost));
                        if literal_cost < reference[origin + 1].0 {
                            reference[origin + 1] = (literal_cost, 1, 0);
                        }
                        if origin == 0 || size - origin < 3 {
                            continue;
                        }
                        for (offer, limit) in [4, 40, 113].into_iter().enumerate() {
                            let end = (size - origin + 1).min(limit);
                            let distance = (1 + offer).min(origin);
                            let cost = base + (origin % 7) as f32 * 0.25;
                            compact.interval(origin, 3, end, distance, f64::from(cost));
                            wide.interval(origin, 3, end, distance, f64::from(cost));
                            for (offset, step) in
                                reference[origin + 3..origin + end].iter_mut().enumerate()
                            {
                                if cost < step.0 {
                                    *step = (cost, 3 + offset, distance);
                                }
                            }
                        }
                    }
                    compact.advance(size);
                    wide.advance(size);
                    assert_eq!(compact.cost(size), f64::from(reference[size].0));
                    assert_eq!(wide.cost(size), compact.cost(size));
                    let mut end = size;
                    let mut expected = Vec::new();
                    while end != 0 {
                        let (_, length, distance) = reference[end];
                        assert!(length != 0 && length <= end);
                        end -= length;
                        expected.push(if distance == 0 {
                            Command::Literal(input[end])
                        } else {
                            Command::Match {
                                context: input[end].context,
                                length: length as u32,
                                distance: distance as u32,
                            }
                        });
                    }
                    expected.reverse();
                    let commands = compact.backtrack(&input).unwrap();
                    assert_eq!(commands, expected);
                    assert_eq!(commands, wide.backtrack(&input).unwrap());
                    decode(&input, &commands);
                    assert_eq!(compact.work.scalar_endpoints, wide.work.scalar_endpoints);
                    assert_eq!(compact.work.tree_updates, wide.work.tree_updates);
                }
            }
        }
    }

    #[test]
    fn compact_costs_are_rounded_by_caller_and_precise_storage_stays_double() {
        let input = vec![
            Literal {
                value: 7,
                context: u32::MAX
            };
            4
        ];
        let mut compact = RangeParser::new_source_compact(4, 0).unwrap();
        let mut precise = RangeParser::new_source(4).unwrap();
        compact.literal(0, f64::from(16_777_217.0f64 as f32));
        precise.literal(0, 16_777_217.0);
        compact.advance(1);
        precise.advance(1);
        assert_eq!(compact.cost(1), 16_777_216.0);
        assert_eq!(precise.cost(1), 16_777_217.0);
        compact.interval(1, 3, 4, 1, 16_777_218.0);
        compact.advance(4);
        let commands = compact.backtrack(&input).unwrap();
        assert!(matches!(
            commands[1],
            Command::Match {
                context: u32::MAX,
                length: 3,
                distance: 1
            }
        ));
        decode(&input, &commands);
    }
}
