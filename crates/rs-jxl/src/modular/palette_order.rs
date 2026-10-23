// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Lossless palette tools from frozen libjxl b866b3d: enc_palette_order.h,
//! enc_palette_map.h and enc_palette.cc::FwdPaletteIteration. No environment
//! reads. These only permute exact tuples; even invisible RGB stays intact.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PaletteOrder {
    #[default]
    Default,
    FrequencyLuma,
    FrequencyBandsLuma,
    FrequencyBandsMorton,
    FrequentMorton,
    Morton,
    FrequencyBandsHilbert,
    Hilbert,
    Adjacency,
    FrequencyBandsAdjacency,
}

impl PaletteOrder {
    /// JXL_LOSSLESS_PALETTE_ORDER, with zero denoting the ordinary luma order.
    pub fn from_source_mode(mode: u8) -> Result<Self> {
        match mode {
            0 => Ok(Self::Default),
            1 => Ok(Self::FrequencyLuma),
            2 => Ok(Self::FrequencyBandsLuma),
            3 => Ok(Self::FrequencyBandsMorton),
            4 => Ok(Self::FrequentMorton),
            5 => Ok(Self::Morton),
            6 => Ok(Self::FrequencyBandsHilbert),
            7 => Ok(Self::Hilbert),
            8 => Ok(Self::Adjacency),
            9 => Ok(Self::FrequencyBandsAdjacency),
            _ => Err(Error::InvalidInput(
                "palette order mode must be 0..9".into(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PaletteGeometry {
    #[default]
    None,
    Principal,
    FrequencyBandsPrincipal,
}

impl PaletteGeometry {
    /// JXL_LOSSLESS_PALETTE_GEOMETRY; geometry takes precedence over order.
    pub fn from_source_mode(mode: u8) -> Result<Self> {
        match mode {
            0 => Ok(Self::None),
            1 => Ok(Self::Principal),
            2 => Ok(Self::FrequencyBandsPrincipal),
            _ => Err(Error::InvalidInput(
                "palette geometry mode must be 0..2".into(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PaletteMapping {
    #[default]
    Tree,
    /// JXL_LOSSLESS_PALETTE_MAP=1: full-key linear probing for <=4 channels.
    Flat,
}

/// Optional source experiments, all disabled by default. Single-channel
/// palettes stay sorted. `ordered=false` bypasses order/geometry, not mapping.
/// Geometry/Hilbert/adjacency require 3..=4 channels; at >4 channels source
/// modes 6..9 fall through to unbanded signed Morton, while geometry is ignored.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PaletteTools {
    pub order: PaletteOrder,
    pub geometry: PaletteGeometry,
    pub mapping: PaletteMapping,
}

#[derive(Clone, Debug, Default)]
pub(super) struct MapEntry {
    key: [i32; 4],
    pub count: usize,
    pub index: usize,
    pub implicit: bool,
    pub seen: bool,
}

/// Exact enc_palette_map.h table: slots are entry index + 1, entries keep
/// insertion order. Cache an index, never a reference invalidated by Vec growth.
pub(super) struct LosslessPaletteMap {
    channels: usize,
    entries: Vec<MapEntry>,
    slots: Vec<u32>,
    last: Option<([i32; 4], usize)>,
}

impl LosslessPaletteMap {
    fn new(channels: usize) -> Self {
        Self {
            channels,
            entries: Vec::new(),
            slots: vec![0; 1024],
            last: None,
        }
    }

    fn hash(&self, key: &[i32; 4]) -> u64 {
        let mut hash = 0x9e3779b97f4a7c15u64;
        for &value in &key[..self.channels] {
            hash = (hash ^ u64::from(value as u32)).wrapping_mul(0xbf58476d1ce4e5b9);
        }
        hash ^ (hash >> 32)
    }

    fn rehash(&mut self) -> Result<()> {
        let len = self
            .slots
            .len()
            .checked_mul(2)
            .ok_or(Error::Limit("palette map size"))?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(len)
            .map_err(|_| Error::Limit("palette map allocation"))?;
        slots.resize(len, 0);
        for (i, entry) in self.entries.iter().enumerate() {
            let mut slot = self.hash(&entry.key) as usize & (len - 1);
            while slots[slot] != 0 {
                slot = (slot + 1) & (len - 1);
            }
            slots[slot] = (i + 1) as u32;
        }
        self.slots = slots;
        Ok(())
    }

    fn get(&mut self, color: &[i32]) -> Result<&mut MapEntry> {
        let mut key = [0; 4];
        key[..self.channels].copy_from_slice(color);
        if let Some((last, index)) = self.last
            && last == key
        {
            return Ok(&mut self.entries[index]);
        }
        if self.entries.len() >= self.slots.len() / 2 {
            self.rehash()?;
        }
        let mut slot = self.hash(&key) as usize & (self.slots.len() - 1);
        while self.slots[slot] != 0 {
            let index = self.slots[slot] as usize - 1;
            if self.entries[index].key == key {
                self.last = Some((key, index));
                return Ok(&mut self.entries[index]);
            }
            slot = (slot + 1) & (self.slots.len() - 1);
        }
        let index = self.entries.len();
        let stored = u32::try_from(index + 1).map_err(|_| Error::Limit("palette map entries"))?;
        self.entries
            .try_reserve(1)
            .map_err(|_| Error::Limit("palette map allocation"))?;
        self.entries.push(MapEntry {
            key,
            ..MapEntry::default()
        });
        self.slots[slot] = stored;
        self.last = Some((key, index));
        Ok(&mut self.entries[index])
    }
}

pub(super) enum PaletteMap {
    Tree(BTreeMap<Vec<i32>, MapEntry>),
    Flat(LosslessPaletteMap),
}

impl PaletteMap {
    pub fn new(channels: usize, mapping: PaletteMapping) -> Self {
        if mapping == PaletteMapping::Flat && channels <= 4 {
            Self::Flat(LosslessPaletteMap::new(channels))
        } else {
            Self::Tree(BTreeMap::new())
        }
    }

    pub fn get(&mut self, color: &[i32]) -> Result<&mut MapEntry> {
        match self {
            Self::Flat(map) => map.get(color),
            Self::Tree(map) => Ok(map.entry(color.to_vec()).or_default()),
        }
    }
}

fn band(frequency: usize) -> usize {
    (usize::BITS - 1 - frequency.max(1).leading_zeros()) as usize / 2
}

// Preserve the source's indexed covariance/power-iteration evaluation order.
#[allow(clippy::needless_range_loop)]
fn principal_order(
    colors: &[[i32; 4]],
    frequencies: &[usize],
    channels: usize,
    banded: bool,
) -> Vec<usize> {
    let n = colors.len();
    let mut order: Vec<_> = (0..n).collect();
    if n < 2 || channels == 0 || channels > 4 || frequencies.len() != n {
        return order;
    }
    let mut mean = [0.0; 4];
    let mut axis = [0.0; 4];
    let mut covariance = [[0.0; 4]; 4];
    for color in colors {
        for c in 0..channels {
            mean[c] += f64::from(color[c]);
        }
    }
    for value in &mut mean[..channels] {
        *value /= n as f64;
    }
    for color in colors {
        for c in 0..channels {
            for d in 0..channels {
                covariance[c][d] +=
                    (f64::from(color[c]) - mean[c]) * (f64::from(color[d]) - mean[d]);
            }
        }
    }
    let mut widest = 0;
    for c in 1..channels {
        if covariance[c][c] > covariance[widest][widest] {
            widest = c;
        }
    }
    axis[widest] = 1.0;
    for _ in 0..12 {
        let mut next = [0.0; 4];
        let mut norm = 0.0;
        for c in 0..channels {
            for d in 0..channels {
                next[c] += covariance[c][d] * axis[d];
            }
            norm += next[c] * next[c];
        }
        if norm == 0.0 {
            break;
        }
        let scale = 1.0 / norm.sqrt();
        for c in 0..channels {
            axis[c] = next[c] * scale;
        }
    }
    let mut projection = vec![0.0; n];
    let mut bands = vec![0; n];
    for i in 0..n {
        for c in 0..channels {
            projection[i] += (f64::from(colors[i][c]) - mean[c]) * axis[c];
        }
        if banded {
            bands[i] = band(frequencies[i]);
        }
    }
    order.sort_by(|&a, &b| {
        bands[b]
            .cmp(&bands[a])
            .then_with(|| projection[a].partial_cmp(&projection[b]).unwrap())
            .then_with(|| colors[a].cmp(&colors[b]))
            .then_with(|| a.cmp(&b))
    });
    order
}

fn root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn adjacency_order(frequencies: &[usize], mut pairs: Vec<u64>) -> Vec<usize> {
    let n = frequencies.len();
    pairs.sort_unstable();
    let mut edges = Vec::new();
    let mut begin = 0;
    while begin < pairs.len() {
        let mut end = begin + 1;
        while end < pairs.len() && pairs[end] == pairs[begin] {
            end += 1;
        }
        let a = (pairs[begin] >> 32) as usize;
        let b = (pairs[begin] as u32) as usize;
        if a < n && b < n && a != b {
            edges.push((a, b, end - begin));
        }
        begin = end;
    }
    edges.sort_unstable_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let mut parent: Vec<_> = (0..n).collect();
    let mut sizes = vec![1usize; n];
    let mut neighbors = vec![[n; 2]; n];
    let mut degree = vec![0; n];
    let mut visited = vec![false; n];
    for (ea, eb, _) in edges {
        if degree[ea] == 2 || degree[eb] == 2 {
            continue;
        }
        let mut a = root(&mut parent, ea);
        let mut b = root(&mut parent, eb);
        if a == b {
            continue;
        }
        neighbors[ea][degree[ea]] = eb;
        degree[ea] += 1;
        neighbors[eb][degree[eb]] = ea;
        degree[eb] += 1;
        if sizes[a] < sizes[b] {
            std::mem::swap(&mut a, &mut b);
        }
        parent[b] = a;
        sizes[a] += sizes[b];
    }
    let mut endpoints: Vec<_> = (0..n).filter(|&i| degree[i] < 2).collect();
    endpoints.sort_by(|&a, &b| frequencies[b].cmp(&frequencies[a]));
    let mut order = Vec::with_capacity(n);
    for endpoint in endpoints {
        let mut node = endpoint;
        while node < n && !visited[node] {
            order.push(node);
            visited[node] = true;
            let next = neighbors[node][0];
            node = if next < n && !visited[next] {
                next
            } else {
                neighbors[node][1]
            };
        }
    }
    order
}

fn hilbert_key(mut axes: [u32; 4], dimensions: usize, bits: u32) -> [u32; 4] {
    if !(2..=4).contains(&dimensions) || !(1..=32).contains(&bits) {
        return axes;
    }
    let highest = 1u32 << (bits - 1);
    let mut mask = highest;
    while mask > 1 {
        let low = mask - 1;
        for c in 0..dimensions {
            if axes[c] & mask != 0 {
                axes[0] ^= low;
            } else {
                let exchange = (axes[0] ^ axes[c]) & low;
                axes[0] ^= exchange;
                axes[c] ^= exchange;
            }
        }
        mask >>= 1;
    }
    for c in 1..dimensions {
        axes[c] ^= axes[c - 1];
    }
    let mut correction = 0;
    mask = highest;
    while mask > 1 {
        if axes[dimensions - 1] & mask != 0 {
            correction ^= mask - 1;
        }
        mask >>= 1;
    }
    for value in &mut axes[..dimensions] {
        *value ^= correction;
    }
    axes
}

fn interleaved_cmp(a: &[u32], b: &[u32]) -> Ordering {
    let mut highest = 0;
    let mut channel = 0;
    for c in 0..a.len() {
        let difference = a[c] ^ b[c];
        if difference > highest && (difference ^ highest) > highest {
            highest = difference;
            channel = c;
        }
    }
    if highest == 0 {
        Ordering::Equal
    } else {
        a[channel].cmp(&b[channel])
    }
}

fn morton_cmp(a: &[i32], b: &[i32]) -> Ordering {
    let mut highest_bit = 0;
    let mut channel = 0;
    for c in 0..a.len() {
        let bit = u32::BITS - ((a[c] as u32) ^ (b[c] as u32)).leading_zeros();
        if bit > highest_bit {
            highest_bit = bit;
            channel = c;
        }
    }
    if highest_bit == 0 {
        Ordering::Equal
    } else {
        a[channel].cmp(&b[channel])
    }
}

fn luma(color: &[i32], frequency: usize) -> f32 {
    let mut y = 0.299f32 * color[0] as f32
        + 0.587f32 * color[1] as f32
        + 0.114f32 * color[2] as f32
        + 0.1f32;
    if color.len() > 3 {
        y *= 1.0f32 + color[3] as f32;
    }
    if frequency > 4 { -y } else { y }
}

// Inputs have already passed the sample transform's dimension/range checks.
pub(super) fn reorder(
    colors: &mut Vec<Vec<i32>>,
    map: &mut PaletteMap,
    planes: &[Vec<i32>],
    width: usize,
    height: usize,
    tools: PaletteTools,
) -> Result<()> {
    let channels = planes.len();
    if channels < 3 {
        return Ok(());
    }
    let frequencies: Vec<_> = colors
        .iter()
        .map(|c| map.get(c).map(|e| e.count))
        .collect::<Result<_>>()?;
    let n = colors.len();
    let order = if channels <= 4 && tools.geometry != PaletteGeometry::None {
        let tuples: Vec<_> = colors
            .iter()
            .map(|color| {
                let mut tuple = [0; 4];
                tuple[..channels].copy_from_slice(color);
                tuple
            })
            .collect();
        principal_order(
            &tuples,
            &frequencies,
            channels,
            tools.geometry == PaletteGeometry::FrequencyBandsPrincipal,
        )
    } else if channels <= 4
        && matches!(
            tools.order,
            PaletteOrder::Adjacency | PaletteOrder::FrequencyBandsAdjacency
        )
    {
        for (i, color) in colors.iter().enumerate() {
            map.get(color)?.index = i;
        }
        let len = width * height;
        let step = len.div_ceil(262144).max(1);
        let mut pairs = Vec::new();
        let mut color = vec![0; channels];
        let mut index = |pos: usize| -> Result<usize> {
            for c in 0..channels {
                color[c] = planes[c][pos];
            }
            Ok(map.get(&color)?.index)
        };
        let mut add = |a: usize, b: usize| {
            if a >= n || b >= n || a == b {
                return;
            }
            if tools.order == PaletteOrder::FrequencyBandsAdjacency
                && band(frequencies[a]) != band(frequencies[b])
            {
                return;
            }
            pairs.push(((a.min(b) as u64) << 32) | a.max(b) as u64);
        };
        for pos in (0..len).step_by(step) {
            let current = index(pos)?;
            if pos % width + 1 < width {
                add(current, index(pos + 1)?);
            }
            if pos / width + 1 < height {
                add(current, index(pos + width)?);
            }
        }
        adjacency_order(&frequencies, pairs)
    } else if channels <= 4
        && matches!(
            tools.order,
            PaletteOrder::FrequencyBandsHilbert | PaletteOrder::Hilbert
        )
    {
        let keys: Vec<_> = colors
            .iter()
            .map(|color| {
                let mut axes = [0; 4];
                for c in 0..channels {
                    axes[c] = color[c] as u32 ^ 0x80000000;
                }
                hilbert_key(axes, channels, 32)
            })
            .collect();
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let bands = if tools.order == PaletteOrder::FrequencyBandsHilbert {
                band(frequencies[b]).cmp(&band(frequencies[a]))
            } else {
                Ordering::Equal
            };
            bands.then_with(|| interleaved_cmp(&keys[a][..channels], &keys[b][..channels]))
        });
        order
    } else {
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let (af, bf) = (frequencies[a], frequencies[b]);
            let primary = match tools.order {
                PaletteOrder::FrequencyLuma => bf.cmp(&af),
                PaletteOrder::FrequencyBandsLuma | PaletteOrder::FrequencyBandsMorton => {
                    band(bf).cmp(&band(af))
                }
                PaletteOrder::FrequentMorton => (bf > 4).cmp(&(af > 4)),
                _ => Ordering::Equal,
            };
            primary.then_with(|| match tools.order {
                PaletteOrder::Default
                | PaletteOrder::FrequencyLuma
                | PaletteOrder::FrequencyBandsLuma => luma(&colors[a], af)
                    .partial_cmp(&luma(&colors[b], bf))
                    .unwrap(),
                _ => morton_cmp(&colors[a], &colors[b]),
            })
        });
        order
    };
    // Stable sorting preserves the existing Rust tie order where C++ std::sort
    // leaves equivalent keys unspecified. Principal ties explicitly include i.
    debug_assert_eq!(order.len(), n);
    let mut old = std::mem::take(colors);
    *colors = order
        .into_iter()
        .map(|i| std::mem::take(&mut old[i]))
        .collect();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_modes_and_defaults_are_explicit() {
        let modes = [
            PaletteOrder::Default,
            PaletteOrder::FrequencyLuma,
            PaletteOrder::FrequencyBandsLuma,
            PaletteOrder::FrequencyBandsMorton,
            PaletteOrder::FrequentMorton,
            PaletteOrder::Morton,
            PaletteOrder::FrequencyBandsHilbert,
            PaletteOrder::Hilbert,
            PaletteOrder::Adjacency,
            PaletteOrder::FrequencyBandsAdjacency,
        ];
        for (i, mode) in modes.into_iter().enumerate() {
            assert_eq!(PaletteOrder::from_source_mode(i as u8).unwrap(), mode);
        }
        for (i, mode) in [
            PaletteGeometry::None,
            PaletteGeometry::Principal,
            PaletteGeometry::FrequencyBandsPrincipal,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(PaletteGeometry::from_source_mode(i as u8).unwrap(), mode);
        }
        assert!(PaletteOrder::from_source_mode(10).is_err());
        assert!(PaletteGeometry::from_source_mode(3).is_err());
        assert_eq!(
            PaletteTools::default(),
            PaletteTools {
                order: PaletteOrder::Default,
                geometry: PaletteGeometry::None,
                mapping: PaletteMapping::Tree,
            }
        );
        assert_eq!(
            [0, 1, 3, 4, 15, 16, 63, 64].map(band),
            [0, 0, 0, 1, 1, 2, 2, 3]
        );
    }

    #[test]
    fn flat_map_collisions_rehash_and_cached_entries_keep_full_signed_keys() {
        let mut map = LosslessPaletteMap::new(4);
        let mut buckets = vec![None; 1024];
        let mut collision = None;
        for i in 0..1025i32 {
            let key = [i, -i, i32::MIN + i, 0];
            let slot = map.hash(&key) as usize & 1023;
            if let Some(previous) = buckets[slot] {
                collision = Some((previous, key));
                break;
            }
            buckets[slot] = Some(key);
        }
        let (a, b) = collision.unwrap();
        map.get(&a).unwrap().count = 17;
        map.get(&b).unwrap().count = 31;
        assert_eq!(map.get(&a).unwrap().count, 17);
        assert_eq!(map.get(&b).unwrap().count, 31);
        assert_eq!(map.entries.len(), 2);
        assert_eq!(map.entries[0].key, a);
        assert_eq!(map.entries[1].key, b);
        for i in 0..3000i32 {
            let entry = map.get(&[i, -i, i32::MAX - i, 0]).unwrap();
            entry.count = i as usize;
            entry.index = 3000 - i as usize;
            entry.implicit = i % 2 == 0;
            entry.seen = true;
            // Consecutive repeats take the cache and must not insert again.
            assert_eq!(
                map.get(&[i, -i, i32::MAX - i, 0]).unwrap().count,
                i as usize
            );
        }
        assert!(map.slots.len() >= 8192);
        assert_eq!(map.entries.len(), 3002);
        for i in (0..3000i32).rev() {
            let entry = map.get(&[i, -i, i32::MAX - i, 0]).unwrap();
            assert_eq!(
                (entry.count, entry.index, entry.implicit, entry.seen),
                (i as usize, 3000 - i as usize, i % 2 == 0, true)
            );
        }
        assert_eq!(map.get(&a).unwrap().count, 17);
        assert_eq!(map.get(&b).unwrap().count, 31);
        // Alpha is part of equality, including when RGB is identical.
        map.get(&[a[0], a[1], a[2], 1]).unwrap().count = 99;
        assert_eq!(map.get(&a).unwrap().count, 17);
        for channels in 1..=4 {
            let mut map = LosslessPaletteMap::new(channels);
            map.get(&a[..channels]).unwrap().count = 8;
            assert_eq!(map.get(&a[..channels]).unwrap().count, 8);
            assert_eq!(map.entries[0].key[channels..], [0; 4][channels..]);
        }
        let mut wide = PaletteMap::new(5, PaletteMapping::Flat);
        assert!(matches!(&wide, PaletteMap::Tree(_)));
        wide.get(&[1, 2, 3, 0, 9]).unwrap().count = 1;
        wide.get(&[1, 2, 3, 0, 10]).unwrap().count = 2;
        assert_eq!(wide.get(&[1, 2, 3, 0, 9]).unwrap().count, 1);
    }

    #[test]
    fn principal_axis_bands_and_complete_tie_rules() {
        let colors = [[30, 0, 0, 0], [10, 0, 0, 0], [20, 0, 0, 0]];
        assert_eq!(principal_order(&colors, &[16, 1, 4], 3, false), [1, 2, 0]);
        assert_eq!(principal_order(&colors, &[16, 1, 4], 3, true), [0, 2, 1]);
        let tied = [[1, 1, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0], [-1, -1, 0, 0]];
        // Equal diagonal variance chooses the first axis; projections tie by
        // complete tuple, then original candidate index for identical tuples.
        assert_eq!(principal_order(&tied, &[1; 4], 2, false), [3, 2, 1, 0]);
        assert_eq!(
            principal_order(&[[7; 4]; 3], &[4, 16, 4], 4, false),
            [0, 1, 2]
        );
        assert_eq!(
            principal_order(&[[7; 4]; 3], &[4, 16, 4], 4, true),
            [1, 0, 2]
        );
        assert_eq!(principal_order(&colors, &[1], 3, true), [0, 1, 2]);
        assert_eq!(principal_order(&colors, &[1; 3], 0, false), [0, 1, 2]);
        assert!(principal_order(&[], &[], 3, false).is_empty());
    }

    #[test]
    fn adjacency_forest_cycle_degree_endpoint_and_edge_ties() {
        let pair = |a: u64, b: u64| (a << 32) | b;
        assert_eq!(
            adjacency_order(
                &[2, 9, 2, 9, 0],
                vec![
                    pair(0, 1),
                    pair(0, 1),
                    pair(0, 1),
                    pair(1, 2),
                    pair(1, 2),
                    pair(0, 2),
                    pair(2, 3),
                    pair(0, 0),
                    pair(0, 99),
                ]
            ),
            [3, 2, 1, 0, 4]
        );
        assert_eq!(
            adjacency_order(
                &[1; 4],
                vec![pair(2, 3), pair(1, 2), pair(0, 2), pair(0, 1)]
            ),
            [1, 0, 2, 3]
        );
        // All edges incident to 0: degree-two cap leaves vertex 3 isolated.
        assert_eq!(
            adjacency_order(&[1; 4], vec![pair(0, 1), pair(0, 2), pair(0, 3)]),
            [1, 0, 2, 3]
        );
        assert_eq!(adjacency_order(&[1, 4, 4, 0], vec![]), [1, 2, 0, 3]);
        assert!(adjacency_order(&[], vec![0]).is_empty());
        assert_eq!(adjacency_order(&[8], vec![0]), [0]);
    }

    fn interleaved_reference(a: &[u32], b: &[u32]) -> Ordering {
        for bit in (0..32).rev() {
            for c in 0..a.len() {
                let cmp = ((a[c] >> bit) & 1).cmp(&((b[c] >> bit) & 1));
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
        }
        Ordering::Equal
    }

    #[test]
    fn curve_comparators_keep_all_bits_and_first_channel_ties() {
        let tuples = [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [i32::MIN, i32::MAX, -1, 0],
            [i32::MAX, i32::MIN, 0, -1],
            [7, 8, 9, 10],
            [7, 8, 9, 11],
        ];
        for dimensions in 2..=4 {
            for a in tuples {
                for b in tuples {
                    let ua = a.map(|x| x as u32 ^ 0x80000000);
                    let ub = b.map(|x| x as u32 ^ 0x80000000);
                    assert_eq!(
                        morton_cmp(&a[..dimensions], &b[..dimensions]),
                        interleaved_reference(&ua[..dimensions], &ub[..dimensions])
                    );
                    let ha = hilbert_key(ua, dimensions, 32);
                    let hb = hilbert_key(ub, dimensions, 32);
                    assert_eq!(
                        interleaved_cmp(&ha[..dimensions], &hb[..dimensions]),
                        interleaved_reference(&ha[..dimensions], &hb[..dimensions])
                    );
                }
            }
        }
        assert_eq!(morton_cmp(&[0, 1], &[1, 0]), Ordering::Less);
        assert_eq!(interleaved_cmp(&[4, 7], &[7, 4]), Ordering::Less);
        let axes = [1, 2, 3, 4];
        for (dimensions, bits) in [(1, 32), (5, 32), (3, 0), (3, 33)] {
            assert_eq!(hilbert_key(axes, dimensions, bits), axes);
        }
    }

    #[test]
    fn hilbert_small_cube_is_bijective_and_visits_neighboring_cells() {
        let mut square = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]];
        square.sort_by_key(|&p| {
            let k = hilbert_key(p, 2, 1);
            (k[0] << 1) | k[1]
        });
        assert_eq!(
            square,
            [[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]]
        );
        let mut cube: Vec<_> = (0..64u32)
            .map(|i| [i & 3, (i >> 2) & 3, i >> 4, 0])
            .collect();
        cube.sort_by(|&a, &b| {
            interleaved_cmp(&hilbert_key(a, 3, 2)[..3], &hilbert_key(b, 3, 2)[..3])
        });
        let mut keys: Vec<_> = cube.iter().map(|&p| hilbert_key(p, 3, 2)).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), 64);
        for cells in cube.windows(2) {
            assert_eq!(
                (0..3)
                    .map(|c| cells[0][c].abs_diff(cells[1][c]))
                    .sum::<u32>(),
                1
            );
        }
    }

    #[test]
    fn luma_and_morton_modes_keep_source_thresholds_and_directions() {
        let gray = [90, 10, 120, 50, 70, 30];
        let frequencies = [5, 5, 1, 4, 16, 1];
        let colors: Vec<_> = gray.iter().map(|&x| vec![x; 3]).collect();
        let expected = [
            [0, 4, 1, 5, 3, 2],
            [4, 0, 1, 3, 5, 2],
            [4, 0, 1, 3, 5, 2],
            [4, 1, 3, 0, 5, 2],
            [1, 4, 0, 5, 3, 2],
            [1, 5, 3, 4, 0, 2],
        ];
        for mode in 0..=5 {
            let mut map = PaletteMap::new(3, PaletteMapping::Tree);
            for (color, count) in colors.iter().zip(frequencies) {
                map.get(color).unwrap().count = count;
            }
            let mut sorted = colors.clone();
            reorder(
                &mut sorted,
                &mut map,
                &[Vec::new(), Vec::new(), Vec::new()],
                0,
                0,
                PaletteTools {
                    order: PaletteOrder::from_source_mode(mode).unwrap(),
                    ..PaletteTools::default()
                },
            )
            .unwrap();
            assert_eq!(
                sorted.iter().map(|c| c[0]).collect::<Vec<_>>(),
                expected[mode as usize].map(|i| gray[i])
            );
        }
    }
}
