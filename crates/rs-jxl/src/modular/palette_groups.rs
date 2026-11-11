// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Explicit partial-channel palette experiments, independent of source policy.
//!
//! RCT equations/permutations follow libjxl b866b3d enc_rct.cc; palette
//! application delegates to the existing enc_palette.cc ports. The caller owns
//! RGB/RGBA interpretation, matching alpha depth, routing and final-byte trials.
//! No metadata, sample depth, alpha association or hidden colors are rewritten.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::group::{self, Channel, Transform};
use super::palette::{PaletteMode, PaletteState, PaletteTools, apply_palette_range_with_options};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Grouping {
    Rg,
    YCo,
    YCg,
    CoCg,
    YaCoCg,
}

/// Invocation-owned diagnostics; independent of PaletteOptions equality.
#[derive(Debug, Default)]
pub struct Receipt {
    attempts: AtomicUsize,
    applied: AtomicUsize,
    tables: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Snapshot {
    pub attempts: usize,
    pub applied: usize,
    pub tables: usize,
}

impl Receipt {
    /// Counters are individually atomic, not a coherent live transaction.
    /// Read after preparation completes for final invocation totals.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            attempts: self.attempts.load(Ordering::Relaxed),
            applied: self.applied.load(Ordering::Relaxed),
            tables: self.tables.load(Ordering::Relaxed),
        }
    }

    /// Count one completed attempt; declined attempts contribute no tables.
    pub(crate) fn record(&self, applied: bool, tables: usize) {
        self.attempts.fetch_add(1, Ordering::Relaxed);
        if applied {
            self.applied.fetch_add(1, Ordering::Relaxed);
            self.tables.fetch_add(tables, Ordering::Relaxed);
        }
    }
}

// Wrapping before arithmetic shifts is part of the native RCT6 contract.
fn ycocg(r: i32, g: i32, b: i32) -> [i32; 3] {
    let co = r.wrapping_sub(b);
    let t = b.wrapping_add(co >> 1);
    let cg = g.wrapping_sub(t);
    [t.wrapping_add(cg >> 1), co, cg]
}

fn admit(seen: &mut HashSet<u64>, a: i32, b: i32, cap: u32) -> Result<bool> {
    let key = (u64::from(a as u32) << 32) | u64::from(b as u32);
    if seen.contains(&key) {
        return Ok(true);
    }
    if seen.len() as u64 >= u64::from(cap) {
        return Ok(false);
    }
    seen.try_reserve(1)
        .map_err(|_| Error::Limit("partial palette preflight allocation"))?;
    seen.insert(key);
    Ok(true)
}

fn preflight(channels: &[Channel], grouping: Grouping, cap: u32) -> Result<bool> {
    let mut first = HashSet::new();
    let mut second = HashSet::new();
    for i in 0..channels[0].samples.len() {
        let r = channels[0].samples[i];
        let g = channels[1].samples[i];
        let b = channels[2].samples[i];
        let (a, b) = if grouping == Grouping::Rg {
            (r, g)
        } else {
            let [y, co, cg] = ycocg(r, g, b);
            match grouping {
                Grouping::YCo => (y, co),
                Grouping::YCg => (y, cg),
                Grouping::CoCg => (co, cg),
                Grouping::YaCoCg => {
                    if !admit(&mut second, y, channels[3].samples[i], cap)? {
                        return Ok(false);
                    }
                    (co, cg)
                }
                Grouping::Rg => unreachable!(),
            }
        };
        if !admit(&mut first, a, b, cap)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn pair(
    channels: &mut Vec<Channel>,
    state: &mut PaletteState,
    bits: u8,
    begin: usize,
    cap: u32,
    script: &mut Vec<Transform>,
) -> Result<bool> {
    let Some(transform) = apply_palette_range_with_options(
        channels,
        &mut state.meta_channels,
        bits,
        begin,
        2,
        cap,
        true,
        PaletteMode::Absolute,
        PaletteTools::default(),
    )?
    else {
        return Ok(false);
    };
    // Inserting the table shifts the new index plane by one. Keep conservative
    // bounds: compaction must not erase the depth required by existing tables.
    let max_index = channels[begin + 1]
        .samples
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    state.max_value = state.max_value.max(max_index);
    state.max_bitdepth = state
        .max_bitdepth
        .max((32 - (max_index as u32).leading_zeros()) as i32 + 1);
    script.push(transform);
    Ok(true)
}

/// Applies one explicit grouping atomically, returning transforms in wire order.
///
/// `cap` bounds exact joint cardinality for admission, not a format/data limit
/// or a prediction of the explicit dictionary size (implicit colors can alias
/// or be promoted). All pixels, including invisible RGB, participate. Original
/// `bits` is passed unchanged to palettes, even after RCT6. Ineligible layouts,
/// insufficient headroom and kernel declines return None without mutation;
/// invalid layout/depth and kernel errors likewise leave both inputs unchanged.
pub(crate) fn apply(
    channels: &mut Vec<Channel>,
    state: &mut PaletteState,
    bits: u8,
    grouping: Grouping,
    cap: u32,
    max_depth: i32,
) -> Result<Option<Vec<Transform>>> {
    group::validate_layout(channels, state.meta_channels)?;
    if !(1..=32).contains(&bits) || !(0..=32).contains(&state.max_bitdepth) {
        return Err(Error::InvalidInput(
            "invalid partial palette bit depth".into(),
        ));
    }
    let input = &channels[state.meta_channels..];
    if !matches!(input.len(), 3 | 4)
        || (grouping == Grouping::YaCoCg && input.len() != 4)
        || bits > 24
        || cap < 2
        || state.max_bitdepth > max_depth
    {
        return Ok(None);
    }
    let shape = &input[0];
    if shape.width == 0
        || shape.height == 0
        || input.iter().any(|c| {
            (c.width, c.height, c.hshift, c.vshift)
                != (shape.width, shape.height, shape.hshift, shape.vshift)
        })
    {
        return Ok(None);
    }
    let needs_rct = grouping != Grouping::Rg;
    if needs_rct && state.max_bitdepth + 1 >= max_depth {
        return Ok(None);
    }
    if !preflight(input, grouping, cap)? {
        return Ok(None);
    }

    let mut candidate = channels.clone();
    let mut next = *state;
    let mut script = Vec::new();
    let begin = next.meta_channels;
    if needs_rct {
        let Some(t) = group::apply_rct(&mut candidate, next.meta_channels, begin, 6)? else {
            return Ok(None);
        };
        script.push(t);
        next.max_bitdepth += 1;
    }
    let permutation = match grouping {
        Grouping::YCg => Some((begin, 21)), // [Y, Co, Cg] -> [Y, Cg, Co].
        Grouping::YaCoCg => Some((begin + 1, 14)), // [Co, Cg, A] -> [A, Co, Cg].
        _ => None,
    };
    if let Some((offset, kind)) = permutation {
        let Some(t) = group::apply_rct(&mut candidate, next.meta_channels, offset, kind)? else {
            return Ok(None);
        };
        script.push(t);
    }
    let offset = match grouping {
        Grouping::CoCg => begin + 1,
        Grouping::YaCoCg => begin + 2,
        _ => begin,
    };
    if !pair(&mut candidate, &mut next, bits, offset, cap, &mut script)? {
        return Ok(None);
    }
    if grouping == Grouping::YaCoCg {
        let offset = next.meta_channels;
        if !pair(&mut candidate, &mut next, bits, offset, cap, &mut script)? {
            return Ok(None);
        }
    }
    if next.max_bitdepth > max_depth {
        return Ok(None);
    }
    *channels = candidate;
    *state = next;
    Ok(Some(script))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUPINGS: [Grouping; 5] = [
        Grouping::Rg,
        Grouping::YCo,
        Grouping::YCg,
        Grouping::CoCg,
        Grouping::YaCoCg,
    ];

    fn fixture(bits: u8, count: usize, meta: usize) -> Vec<Channel> {
        let top = (1i32 << bits) - 1;
        let pixels = [
            [1, 7, top - 2, 0],
            [top - 3, 19, 5, 0],
            [100, 31, 53, top / 2],
            [100, 31, 53, top],
        ];
        let mut channels: Vec<_> = (0..meta)
            .map(|i| Channel {
                width: 2,
                height: 1,
                hshift: -1,
                vshift: -1,
                samples: vec![-7 - i as i32, 13 + i as i32],
            })
            .collect();
        channels.extend((0..count).map(|c| Channel {
            width: 4,
            height: 3,
            hshift: 0,
            vshift: 0,
            samples: (0..12).map(|i| pixels[i % pixels.len()][c]).collect(),
        }));
        channels
    }

    fn state(bits: u8, meta: usize) -> PaletteState {
        PaletteState {
            meta_channels: meta,
            ..PaletteState::new(bits).unwrap()
        }
    }

    // Independent inverse from rct.cc::InvRCTRow. Only the three emitted types
    // are needed; explicit inverse permutations catch swapped wire directions.
    fn undo_rct(channels: &mut [Channel], begin: usize, kind: u32) {
        for i in 0..channels[begin].samples.len() {
            let [a, b, c] = std::array::from_fn(|c| channels[begin + c].samples[i]);
            let values = match kind {
                6 => {
                    let t = a.wrapping_sub(c >> 1);
                    let blue = t.wrapping_sub(b >> 1);
                    [blue.wrapping_add(b), c.wrapping_add(t), blue]
                }
                21 => [a, c, b],
                14 => [b, c, a],
                _ => panic!("unexpected RCT {kind}"),
            };
            for c in 0..3 {
                channels[begin + c].samples[i] = values[c];
            }
        }
    }

    fn undo(channels: &mut Vec<Channel>, script: &[Transform], bits: u8) {
        for t in script.iter().rev() {
            match *t {
                Transform::Palette { .. } => {
                    group::undo_palette_for_test(channels, t.clone(), bits)
                }
                Transform::Rct(kind) => undo_rct(channels, 0, kind),
                Transform::RctAt { begin, kind } => undo_rct(channels, begin, kind),
                _ => panic!("unexpected transform"),
            }
        }
    }

    fn declined(
        channels: Vec<Channel>,
        initial: PaletteState,
        bits: u8,
        g: Grouping,
        cap: u32,
        depth: i32,
    ) {
        let mut actual = channels.clone();
        let mut next = initial;
        assert!(
            apply(&mut actual, &mut next, bits, g, cap, depth)
                .unwrap()
                .is_none()
        );
        assert_eq!(actual, channels);
        assert_eq!(next, initial);
    }

    #[test]
    fn all_scripts_reverse_exactly_with_meta_prefix_signed_chroma_and_hidden_alpha() {
        for bits in [8, 16] {
            for meta in [0, 2] {
                for count in [3, 4] {
                    for g in GROUPINGS {
                        if g == Grouping::YaCoCg && count != 4 {
                            continue;
                        }
                        let original = fixture(bits, count, meta);
                        assert!(
                            ycocg(
                                original[meta].samples[0],
                                original[meta + 1].samples[0],
                                original[meta + 2].samples[0]
                            )[1] < 0
                        );
                        let mut channels = original.clone();
                        let initial = state(bits, meta);
                        let mut next = initial;
                        let script = apply(&mut channels, &mut next, bits, g, 4, 32)
                            .unwrap()
                            .unwrap();
                        let palettes = if g == Grouping::YaCoCg { 2 } else { 1 };
                        assert_eq!(next.meta_channels, meta + palettes);
                        assert_eq!(channels.len() - next.meta_channels, count - palettes);
                        assert_eq!(
                            next.max_bitdepth,
                            initial.max_bitdepth + i32::from(g != Grouping::Rg)
                        );
                        assert_eq!(next.max_value, initial.max_value);
                        assert_eq!(&channels[palettes..palettes + meta], &original[..meta]);
                        let actual: Vec<_> = script
                            .iter()
                            .filter_map(|t| match t {
                                Transform::Palette {
                                    begin,
                                    channels,
                                    deltas,
                                    ..
                                } => {
                                    assert_eq!((*channels, *deltas), (2, 0));
                                    Some(*begin)
                                }
                                _ => None,
                            })
                            .collect();
                        let expected = match g {
                            Grouping::CoCg => vec![meta + 1],
                            Grouping::YaCoCg => vec![meta + 2, meta + 1],
                            _ => vec![meta],
                        };
                        assert_eq!(actual, expected);
                        if g != Grouping::Rg {
                            assert_eq!(
                                script[0],
                                if meta == 0 {
                                    Transform::Rct(6)
                                } else {
                                    Transform::RctAt {
                                        begin: meta,
                                        kind: 6,
                                    }
                                }
                            );
                        }
                        if g == Grouping::YCg {
                            assert_eq!(
                                script[1],
                                if meta == 0 {
                                    Transform::Rct(21)
                                } else {
                                    Transform::RctAt {
                                        begin: meta,
                                        kind: 21,
                                    }
                                }
                            );
                        }
                        if g == Grouping::YaCoCg {
                            assert_eq!(
                                script[1],
                                Transform::RctAt {
                                    begin: meta + 1,
                                    kind: 14
                                }
                            );
                        }
                        undo(&mut channels, &script, bits);
                        assert_eq!(channels, original, "{bits} {meta} {count} {g:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn exact_joint_cardinality_not_marginal_counts_controls_admission() {
        let mut channels = fixture(8, 3, 0);
        for (c, values) in [vec![1, 1, 2, 2], vec![7, 9, 7, 9], vec![13; 4]]
            .into_iter()
            .enumerate()
        {
            channels[c].width = 4;
            channels[c].height = 1;
            channels[c].samples = values;
        }
        declined(channels.clone(), state(8, 0), 8, Grouping::Rg, 3, 32);
        assert!(
            apply(&mut channels, &mut state(8, 0), 8, Grouping::Rg, 4, 32)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn both_compound_pairs_are_preflighted_including_transparent_pixels() {
        let mut channels = fixture(16, 4, 0);
        // Constant chroma, three distinct Y/A tuples, all under zero alpha.
        for c in 0..3 {
            channels[c].samples = (0..12).map(|i| 1001 + (i % 3) as i32).collect();
        }
        channels[3].samples.fill(0);
        assert!(!preflight(&channels, Grouping::YaCoCg, 2).unwrap());
        declined(channels.clone(), state(16, 0), 16, Grouping::YaCoCg, 2, 32);
        // Constant Y/A but three distinct Co/Cg tuples.
        for i in 0..12 {
            let co = (i % 3) as i32 * 2;
            channels[0].samples[i] = 1000 + co / 2;
            channels[1].samples[i] = 1000;
            channels[2].samples[i] = 1000 - co / 2;
        }
        assert!(!preflight(&channels, Grouping::YaCoCg, 2).unwrap());
        declined(channels, state(16, 0), 16, Grouping::YaCoCg, 2, 32);
    }

    #[test]
    fn second_kernel_decline_rolls_back_first_palette_and_permutations() {
        let mut channels = fixture(8, 4, 2);
        // Y=100,A=0 is constant: the YA multichannel kernel declines, although
        // chroma is nonconstant and its palette has already succeeded.
        for i in 0..12 {
            let co = (i % 3) as i32 * 2;
            channels[2].samples[i] = 100 + co / 2;
            channels[3].samples[i] = 100;
            channels[4].samples[i] = 100 - co / 2;
            channels[5].samples[i] = 0;
        }
        assert!(preflight(&channels[2..], Grouping::YaCoCg, 3).unwrap());
        declined(channels, state(8, 2), 8, Grouping::YaCoCg, 3, 32);
    }

    #[test]
    fn first_kernel_decline_is_atomic_for_every_grouping() {
        for g in GROUPINGS {
            let mut channels = fixture(16, 4, 2);
            for (c, channel) in channels[2..].iter_mut().enumerate() {
                channel
                    .samples
                    .fill(if c == 3 { 0 } else { 1001 + c as i32 });
            }
            assert!(preflight(&channels[2..], g, 4).unwrap());
            declined(channels, state(16, 2), 16, g, 4, 32);
        }
    }

    #[test]
    fn incompatible_layouts_and_kernel_depth_support_decline_atomically() {
        for g in GROUPINGS {
            let base = fixture(8, 4, 1);
            let mut different = base.clone();
            different[2].width = 3;
            different[2].height = 4;
            declined(different, state(8, 1), 8, g, 4, 32);
            let mut shifted = base.clone();
            shifted[4].hshift = 1;
            declined(shifted, state(8, 1), 8, g, 4, 32);
            for cap in [0, 1, 2] {
                declined(base.clone(), state(8, 1), 8, g, cap, 32);
            }
            for bits in [25, 31, 32] {
                declined(base.clone(), state(8, 1), bits, g, 4, 32);
            }
            for count in [0, 1, 2, 5] {
                let mut wrong_count = fixture(8, 3, 1);
                while wrong_count.len() < 1 + count {
                    wrong_count.push(wrong_count[1].clone());
                }
                wrong_count.truncate(1 + count);
                declined(wrong_count, state(8, 1), 8, g, 4, 32);
            }
        }
        declined(fixture(8, 3, 0), state(8, 0), 8, Grouping::YaCoCg, 4, 32);
    }

    #[test]
    fn headroom_is_strict_for_rct_but_permutations_add_no_depth() {
        for g in GROUPINGS {
            let input = fixture(16, 4, 0);
            if g != Grouping::Rg {
                declined(input.clone(), state(16, 0), 16, g, 4, 18);
            }
            let mut channels = input;
            let mut next = state(16, 0);
            assert!(
                apply(
                    &mut channels,
                    &mut next,
                    16,
                    g,
                    4,
                    if g == Grouping::Rg { 17 } else { 19 }
                )
                .unwrap()
                .is_some()
            );
            assert_eq!(next.max_bitdepth, if g == Grouping::Rg { 17 } else { 18 });
        }
    }

    #[test]
    fn virtual_rct_matches_native_wrapping_and_signed_tuple_keys_are_distinct() {
        let values = [i32::MIN, i32::MAX, -1, 0, 1, 65535];
        let mut planes: Vec<Vec<i32>> = (0..3)
            .map(|c| {
                (0..values.len())
                    .map(|i| values[(i + c) % values.len()])
                    .collect()
            })
            .collect();
        let expected: Vec<_> = (0..values.len())
            .map(|i| ycocg(planes[0][i], planes[1][i], planes[2][i]))
            .collect();
        super::super::transforms::rct_forward(&mut planes, 0, 6).unwrap();
        for (i, expected) in expected.into_iter().enumerate() {
            assert_eq!([planes[0][i], planes[1][i], planes[2][i]], expected);
        }
        let mut seen = HashSet::new();
        assert!(admit(&mut seen, -1, 0, 2).unwrap());
        assert!(admit(&mut seen, 0, -1, 2).unwrap());
        assert!(admit(&mut seen, -1, 0, 2).unwrap());
        assert!(!admit(&mut seen, i32::MIN, i32::MAX, 2).unwrap());
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn supported_depth_and_large_admission_bound_are_not_artificially_capped() {
        for bits in [17, 24] {
            for g in GROUPINGS {
                let mut channels = fixture(bits, 4, 0);
                let original = channels.clone();
                let mut next = state(bits, 0);
                let script = apply(&mut channels, &mut next, bits, g, u32::MAX, 32)
                    .unwrap()
                    .unwrap();
                undo(&mut channels, &script, bits);
                assert_eq!(channels, original);
            }
        }
    }

    #[test]
    fn palette_index_bounds_are_conservative_and_enforce_the_buffer_promise() {
        let mut channels: Vec<_> = (0..3)
            .map(|c| Channel {
                width: 512,
                height: 1,
                hshift: 0,
                vshift: 0,
                samples: (0..512)
                    .map(|i| match c {
                        0 => i % 256,
                        1 => i / 256,
                        _ => 13,
                    })
                    .collect(),
            })
            .collect();
        let original = channels.clone();
        declined(channels.clone(), state(8, 0), 8, Grouping::Rg, 512, 9);
        let mut next = state(8, 0);
        let script = apply(&mut channels, &mut next, 8, Grouping::Rg, 512, 32)
            .unwrap()
            .unwrap();
        assert!(next.max_bitdepth >= 10);
        assert!(next.max_value >= 511);
        undo(&mut channels, &script, 8);
        assert_eq!(channels, original);
    }

    #[test]
    fn malformed_inputs_error_atomically_and_empty_shapes_decline() {
        let mut channels = fixture(8, 3, 1);
        channels[1].samples.pop();
        let original = channels.clone();
        let mut next = state(8, 1);
        let initial = next;
        assert!(apply(&mut channels, &mut next, 8, Grouping::Rg, 4, 32).is_err());
        assert_eq!(channels, original);
        assert_eq!(next, initial);
        for bits in [0, 33] {
            let mut channels = fixture(8, 3, 1);
            let original = channels.clone();
            assert!(apply(&mut channels, &mut next, bits, Grouping::Rg, 4, 32).is_err());
            assert_eq!(channels, original);
            assert_eq!(next, initial);
        }
        let mut empty = fixture(8, 4, 1);
        for c in &mut empty[1..] {
            c.width = 0;
            c.samples.clear();
        }
        declined(empty, initial, 8, Grouping::YaCoCg, 4, 32);
    }

    #[test]
    fn receipt_counts_attempts_and_only_committed_tables() {
        let receipt = Receipt::default();
        assert_eq!(receipt.snapshot(), Snapshot::default());
        receipt.record(false, 7);
        receipt.record(true, 1);
        receipt.record(true, 2);
        assert_eq!(
            receipt.snapshot(),
            Snapshot {
                attempts: 3,
                applied: 2,
                tables: 3
            }
        );
    }

    #[test]
    fn receipt_can_be_shared_between_preparation_workers() {
        let receipt = Receipt::default();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let receipt = &receipt;
                scope.spawn(move || {
                    for _ in 0..16 {
                        receipt.record(true, 2);
                        receipt.record(false, 0);
                    }
                });
            }
        });
        assert_eq!(
            receipt.snapshot(),
            Snapshot {
                attempts: 128,
                applied: 64,
                tables: 128
            }
        );
    }
}
