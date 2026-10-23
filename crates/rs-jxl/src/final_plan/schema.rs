// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Record layouts/validation from the frozen FinalPlanParameters and its
//! enc_frame, enc_modular, enc_ans, enc_context_map and enc_patch_dictionary
//! hooks. These schemas do not apply transforms, rebuild histograms, find
//! matches, or generate pixels/coefficients. See FINAL_PLAN_PORT_STATUS.md.

use super::{MAX_TREE_NODES, MAX_VALUES, invalid};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tag {
    Frame,
    Input,
    Transforms,
    GlobalState,
    Wp,
    Tree,
    Lz,
    Matches,
    Prefix,
    Clusters,
    MapMode,
    Uint,
    Ans,
    Patches,
    Cfl,
    DcThresholds,
    CoeffOrder,
}

impl Tag {
    pub const ALL: [Self; 17] = [
        Self::Frame,
        Self::Input,
        Self::Transforms,
        Self::GlobalState,
        Self::Wp,
        Self::Tree,
        Self::Lz,
        Self::Matches,
        Self::Prefix,
        Self::Clusters,
        Self::MapMode,
        Self::Uint,
        Self::Ans,
        Self::Patches,
        Self::Cfl,
        Self::DcThresholds,
        Self::CoeffOrder,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Frame => "frame",
            Self::Input => "input",
            Self::Transforms => "transforms",
            Self::GlobalState => "global_state",
            Self::Wp => "wp",
            Self::Tree => "tree",
            Self::Lz => "lz",
            Self::Matches => "matches",
            Self::Prefix => "prefix",
            Self::Clusters => "clusters",
            Self::MapMode => "map_mode",
            Self::Uint => "uint",
            Self::Ans => "ans",
            Self::Patches => "patches",
            Self::Cfl => "cfl",
            Self::DcThresholds => "dc_thresholds",
            Self::CoeffOrder => "coeff_order",
        }
    }

    pub fn from_name(name: &[u8]) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|tag| tag.name().as_bytes() == name)
    }
}

fn ensure(valid: bool, tag: Tag) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(invalid(format!("invalid final-plan {} record", tag.name())))
    }
}
fn range(value: i64, min: i64, max: i64) -> bool {
    (min..=max).contains(&value)
}
fn size_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::Limit("final-plan schema size"))
}

/// Exact field order of FinalPlanParameters. Enum representations are integers
/// so this module does not impose a second encoder option/effort policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameParameters {
    pub speed_tier: i32,
    pub modular_group_size_shift: i32,
    pub colorspace: i32,
    pub responsive: i32,
    pub patches: i32,
    pub palette_colors: i32,
    pub predictor: i32,
    pub max_properties: i32,
    pub force_cfl_jpeg_recompression: bool,
    pub force_lfs_jpeg_recompression: i32,
    pub lz77_careful_first_pass: bool,
    pub keep_invisible: i32,
}

impl FrameParameters {
    pub fn to_values(self) -> [i64; 12] {
        [
            self.speed_tier.into(),
            self.modular_group_size_shift.into(),
            self.colorspace.into(),
            self.responsive.into(),
            self.patches.into(),
            self.palette_colors.into(),
            self.predictor.into(),
            self.max_properties.into(),
            self.force_cfl_jpeg_recompression.into(),
            self.force_lfs_jpeg_recompression.into(),
            self.lz77_careful_first_pass.into(),
            self.keep_invisible.into(),
        ]
    }

    pub fn from_values(values: &[i64], expected_keep_invisible: i32) -> Result<Self> {
        ensure(values.len() == 12, Tag::Frame)?;
        let bounds = [
            (0, 9),
            (-1, 3),
            (-1, 41),
            (-1, 1),
            (-1, 1),
            (-65536, 70000),
            (-1, 15),
            (-1, 7),
            (0, 1),
            (-1, 0),
            (0, 1),
        ];
        ensure(
            values
                .iter()
                .zip(bounds)
                .all(|(&v, (lo, hi))| range(v, lo, hi))
                && values[11] == i64::from(expected_keep_invisible),
            Tag::Frame,
        )?;
        Ok(Self {
            speed_tier: values[0] as i32,
            modular_group_size_shift: values[1] as i32,
            colorspace: values[2] as i32,
            responsive: values[3] as i32,
            patches: values[4] as i32,
            palette_colors: values[5] as i32,
            predictor: values[6] as i32,
            max_properties: values[7] as i32,
            force_cfl_jpeg_recompression: values[8] != 0,
            force_lfs_jpeg_recompression: values[9] as i32,
            lz77_careful_first_pass: values[10] != 0,
            keep_invisible: expected_keep_invisible,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputKind {
    Pixels,
    Jpeg,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputShape {
    pub kind: InputKind,
    pub width: u64,
    pub height: u64,
    pub channels: u64,
}

impl InputShape {
    pub fn to_values(self) -> Result<[i64; 4]> {
        let convert = |v| i64::try_from(v).map_err(|_| Error::Limit("final-plan input dimensions"));
        Ok([
            if self.kind == InputKind::Jpeg { 2 } else { 1 },
            convert(self.width)?,
            convert(self.height)?,
            convert(self.channels)?,
        ])
    }

    pub fn validate_record(self, values: &[i64]) -> Result<()> {
        ensure(values == self.to_values()?, Tag::Input)
    }

    /// Source min(2^22, 1024 + width * height * channels / 16). Saturation
    /// avoids unsigned product wrap for Rust-only oversized metadata; the cap
    /// is exact even there. Input eligibility remains the codec's obligation.
    pub fn tree_limit(self) -> usize {
        let samples = self
            .width
            .saturating_mul(self.height)
            .saturating_mul(self.channels);
        (1024 + samples / 16).min(MAX_TREE_NODES as u64) as usize
    }
}

pub fn global_state(values: &[i64]) -> Result<[i32; 3]> {
    ensure(values.len() == 3, Tag::GlobalState)?;
    ensure(
        range(values[0], 0, 32)
            && range(values[1], i32::MIN.into(), i32::MAX.into())
            && range(values[2], -1, 41),
        Tag::GlobalState,
    )?;
    Ok([values[0] as i32, values[1] as i32, values[2] as i32])
}

pub fn wp(values: &[i64], stream_id: usize) -> Result<u8> {
    ensure(values.len() == 2, Tag::Wp)?;
    ensure(
        values[0] == size_i64(stream_id)? && range(values[1], 0, 4),
        Tag::Wp,
    )?;
    Ok(values[1] as u8)
}

pub fn prefix(values: &[i64]) -> Result<bool> {
    ensure(values.len() == 1 && range(values[0], 0, 1), Tag::Prefix)?;
    Ok(values[0] != 0)
}

pub fn ans(values: &[i64]) -> Result<u8> {
    ensure(values.len() == 1 && range(values[0], 0, 12), Tag::Ans)?;
    Ok(values[0] as u8)
}

pub fn patches(values: &[i64]) -> Result<()> {
    ensure(values == [0], Tag::Patches)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MapMode {
    pub simple: bool,
    pub use_mtf: bool,
}

impl MapMode {
    pub fn to_values(self) -> [i64; 2] {
        [self.simple.into(), self.use_mtf.into()]
    }
    pub fn from_values(values: &[i64], entry_bits: usize) -> Result<Self> {
        ensure(values.len() == 2, Tag::MapMode)?;
        ensure(
            range(values[0], 0, 1) && range(values[1], 0, 1) && (values[0] == 0 || entry_bits < 4),
            Tag::MapMode,
        )?;
        Ok(Self {
            simple: values[0] != 0,
            use_mtf: values[1] != 0,
        })
    }
}

/// Returns max cluster id + 1. Source accepts gaps; do not silently compact,
/// reorder or recluster. Histogram accumulation remains a parent hook.
pub fn clusters(
    values: &[i64],
    contexts: usize,
    previous_histograms: usize,
    context_offset: usize,
) -> Result<usize> {
    ensure(
        previous_histograms == 0
            && context_offset == 0
            && !values.is_empty()
            && values.len() == contexts
            && values.len() <= MAX_VALUES,
        Tag::Clusters,
    )?;
    ensure(values.iter().all(|&v| range(v, 0, 255)), Tag::Clusters)?;
    Ok(*values.iter().max().expect("validated nonempty map") as usize + 1)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UintConfig {
    pub split: u8,
    pub msb: u8,
    pub lsb: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct UintConfigs<'a> {
    pub log_alpha_size: u8,
    triples: &'a [i64],
}

impl<'a> UintConfigs<'a> {
    pub fn from_values(values: &'a [i64], histograms: usize, prefix: bool) -> Result<Self> {
        let length = histograms
            .checked_mul(3)
            .and_then(|v| v.checked_add(1))
            .ok_or(Error::Limit("final-plan uint schema size"))?;
        ensure(values.len() == length && length <= MAX_VALUES, Tag::Uint)?;
        let log = values[0];
        ensure(
            log >= 5 && if prefix { log == 15 } else { log <= 8 },
            Tag::Uint,
        )?;
        for c in values[1..].chunks_exact(3) {
            ensure(
                range(c[0], 0, log)
                    && range(c[1], 0, c[0])
                    && range(c[2], 0, c[0])
                    && c[1] + c[2] <= c[0]
                    && (c[0] != log || (c[1] == 0 && c[2] == 0)),
                Tag::Uint,
            )?;
        }
        Ok(Self {
            log_alpha_size: log as u8,
            triples: &values[1..],
        })
    }

    pub fn configs(self) -> impl ExactSizeIterator<Item = UintConfig> + 'a {
        self.triples.chunks_exact(3).map(|c| UintConfig {
            split: c[0] as u8,
            msb: c[1] as u8,
            lsb: c[2] as u8,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformKind {
    Rct,
    Palette,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformDecision {
    pub kind: TransformKind,
    pub begin_channel: usize,
    pub rct_type: u32,
    pub num_channels: usize,
    pub colors: u32,
    pub predictor: u8,
    pub ordered_palette: bool,
}

impl TransformDecision {
    /// Call immediately before applying EACH transform, with the then-current
    /// channel count. Palette metadata can change the count for the next row.
    pub fn validate_channels(self, channels: usize) -> Result<()> {
        let affected = if self.kind == TransformKind::Rct {
            3
        } else {
            self.num_channels
        };
        ensure(
            self.begin_channel < channels
                && self.num_channels <= channels
                && self
                    .begin_channel
                    .checked_add(affected)
                    .is_some_and(|n| n <= channels)
                && (self.kind != TransformKind::Palette || self.num_channels > 0),
            Tag::Transforms,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Transforms<'a> {
    rows: &'a [i64],
}

impl<'a> Transforms<'a> {
    pub fn from_values(values: &'a [i64]) -> Result<Self> {
        ensure(
            values.len().is_multiple_of(7) && values.len() <= 7 * 512,
            Tag::Transforms,
        )?;
        for v in values.chunks_exact(7) {
            ensure(
                range(v[0], 0, 1)
                    && usize::try_from(v[1]).is_ok()
                    && range(v[2], 0, 41)
                    && usize::try_from(v[3]).is_ok()
                    && range(v[4], 0, 65536)
                    && range(v[5], 0, 13)
                    && range(v[6], 0, 1),
                Tag::Transforms,
            )?;
        }
        Ok(Self { rows: values })
    }

    pub fn decisions(self) -> impl ExactSizeIterator<Item = TransformDecision> + 'a {
        self.rows.chunks_exact(7).map(|v| TransformDecision {
            kind: if v[0] == 0 {
                TransformKind::Rct
            } else {
                TransformKind::Palette
            },
            begin_channel: v[1] as usize,
            rct_type: v[2] as u32,
            num_channels: v[3] as usize,
            colors: v[4] as u32,
            predictor: v[5] as u8,
            ordered_palette: v[6] != 0,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TreeNode {
    Leaf {
        offset: i32,
        predictor: u8,
        multiplier: u32,
    },
    Split {
        property: u8,
        threshold: i32,
        left: usize,
        right: usize,
    },
}

/// Reconstruct and validate source breadth-first topology, property ranges and
/// DFS depth. This is a decision schema, not the encoder's MA tree integration.
pub fn tree(values: &[i64], tree_limit: usize) -> Result<Vec<TreeNode>> {
    let count = values.len() / 4;
    ensure(
        values.len().is_multiple_of(4)
            && values.len() <= MAX_VALUES
            && count > 0
            && count % 2 == 1
            && count <= tree_limit
            && tree_limit <= MAX_TREE_NODES,
        Tag::Tree,
    )?;
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("final-plan tree allocation"))?;
    let mut next = 1;
    for (i, v) in values.chunks_exact(4).enumerate() {
        ensure(
            i < next && range(v[0], -1, 255) && range(v[1], i32::MIN.into(), i32::MAX.into()),
            Tag::Tree,
        )?;
        if v[0] == -1 {
            ensure(
                range(v[2], 0, 13) && range(v[3], 1, i32::MAX.into()),
                Tag::Tree,
            )?;
            nodes.push(TreeNode::Leaf {
                offset: v[1] as i32,
                predictor: v[2] as u8,
                multiplier: v[3] as u32,
            });
        } else {
            ensure(v[2] == 0 && v[3] == 1 && next + 2 <= count, Tag::Tree)?;
            nodes.push(TreeNode::Split {
                property: v[0] as u8,
                threshold: v[1] as i32,
                left: next,
                right: next + 1,
            });
            next += 2;
        }
    }
    ensure(next == count, Tag::Tree)?;
    #[derive(Clone, Copy)]
    struct Visit {
        index: usize,
        stage: u8,
        lo: i32,
        hi: i32,
    }
    let mut ranges = [(i32::MIN, i32::MAX); 256];
    let mut stack = Vec::new();
    stack
        .try_reserve_exact(2048)
        .map_err(|_| Error::Limit("final-plan tree stack"))?;
    stack.push(Visit {
        index: 0,
        stage: 0,
        lo: 0,
        hi: 0,
    });
    while let Some(visit) = stack.last().copied() {
        ensure(stack.len() < 2048, Tag::Tree)?;
        let TreeNode::Split {
            property,
            threshold,
            left,
            right,
        } = nodes[visit.index]
        else {
            stack.pop();
            continue;
        };
        let range = &mut ranges[property as usize];
        let last = stack.len() - 1;
        match visit.stage {
            0 => {
                ensure(range.0 <= threshold && threshold < range.1, Tag::Tree)?;
                stack[last] = Visit {
                    stage: 1,
                    lo: range.0,
                    hi: range.1,
                    ..visit
                };
                range.0 = threshold + 1;
                stack.push(Visit {
                    index: left,
                    stage: 0,
                    lo: 0,
                    hi: 0,
                });
            }
            1 => {
                *range = (visit.lo, threshold);
                stack[last].stage = 2;
                stack.push(Visit {
                    index: right,
                    stage: 0,
                    lo: 0,
                    hi: 0,
                });
            }
            _ => {
                *range = (visit.lo, visit.hi);
                stack.pop();
            }
        }
    }
    Ok(nodes)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LzHeader {
    pub enabled: bool,
    pub contexts: usize,
    pub streams: usize,
    pub min_symbol: u32,
    pub min_length: u32,
    pub length_config: UintConfig,
}

impl LzHeader {
    pub fn to_values(self) -> Result<[i64; 8]> {
        Ok([
            self.enabled.into(),
            size_i64(self.contexts)?,
            size_i64(self.streams)?,
            self.min_symbol.into(),
            self.min_length.into(),
            self.length_config.split.into(),
            self.length_config.msb.into(),
            self.length_config.lsb.into(),
        ])
    }

    /// Only enabled is replayed. All seven scaffolding fields must match the
    /// caller's current LZ parameters and stream/context counts exactly.
    pub fn from_values(values: &[i64], expected: Self) -> Result<Self> {
        ensure(values.len() == 8, Tag::Lz)?;
        ensure(
            range(values[0], 0, 1) && values[1..] == expected.to_values()?[1..],
            Tag::Lz,
        )?;
        Ok(Self {
            enabled: values[0] != 0,
            ..expected
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchSpan {
    pub position: usize,
    pub length: usize,
    pub distance_symbol: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Matches<'a> {
    rows: &'a [i64],
}

impl<'a> Matches<'a> {
    /// Structural source validation only. Parent must decode/clamp the source
    /// special distance and compare every referenced token VALUE (not context)
    /// before replacing a span. Never search for a substitute match.
    pub fn from_values(values: &'a [i64], input_len: usize, min_length: u32) -> Result<Self> {
        ensure(
            values.len().is_multiple_of(3)
                && values.len() <= MAX_VALUES
                && values.len() / 3 <= input_len,
            Tag::Matches,
        )?;
        let mut cursor = 0;
        for v in values.chunks_exact(3) {
            ensure(
                v[0] >= 0 && v[1] >= i64::from(min_length) && range(v[2], 0, u32::MAX.into()),
                Tag::Matches,
            )?;
            let position =
                usize::try_from(v[0]).map_err(|_| Error::Limit("final-plan match position"))?;
            let length =
                usize::try_from(v[1]).map_err(|_| Error::Limit("final-plan match length"))?;
            ensure(
                position >= cursor && position < input_len && length <= input_len - position,
                Tag::Matches,
            )?;
            cursor = position + length;
        }
        Ok(Self { rows: values })
    }

    pub fn spans(self) -> impl ExactSizeIterator<Item = MatchSpan> + 'a {
        self.rows.chunks_exact(3).map(|v| MatchSpan {
            position: v[0] as usize,
            length: v[1] as usize,
            distance_symbol: v[2] as u32,
        })
    }
}

pub fn cfl(values: &[i64], width: usize, height: usize) -> Result<()> {
    let length = width
        .checked_mul(height)
        .ok_or(Error::Limit("final-plan CFL shape"))?;
    ensure(
        values.len() == length
            && length <= MAX_VALUES
            && values.iter().all(|&v| range(v, -128, 127)),
        Tag::Cfl,
    )
}

pub fn dc_thresholds(values: &[i64]) -> Result<()> {
    ensure(
        values.len() <= 7
            && values.iter().all(|&v| range(v, -1025, 1023))
            && values.windows(2).all(|pair| pair[0] < pair[1]),
        Tag::DcThresholds,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoeffOrder {
    pub used_order: bool,
    pub channels: [[u8; 64]; 3],
}

impl CoeffOrder {
    /// Supply the frozen DCT natural order, not a newly sorted permutation.
    /// Parent capture also checks used_acs == 1 and used_orders.len() == 1.
    pub fn from_values(values: &[i64], passes: usize, natural: &[u32; 64]) -> Result<Self> {
        ensure(passes == 1 && values.len() == 193, Tag::CoeffOrder)?;
        ensure(range(values[0], 0, 1), Tag::CoeffOrder)?;
        let mut channels = [[0u8; 64]; 3];
        for (channel, row) in channels.iter_mut().zip(values[1..].chunks_exact(64)) {
            let mut seen = [false; 64];
            for (i, &value) in row.iter().enumerate() {
                ensure(range(value, 0, 63), Tag::CoeffOrder)?;
                let v = value as usize;
                ensure(
                    !seen[v]
                        && (i != 0 || v == 0)
                        && (values[0] != 0 || value == i64::from(natural[i])),
                    Tag::CoeffOrder,
                )?;
                seen[v] = true;
                channel[i] = v as u8;
            }
        }
        Ok(Self {
            used_order: values[0] != 0,
            channels,
        })
    }
}
