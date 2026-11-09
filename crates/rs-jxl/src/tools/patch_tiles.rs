//! Exact fixed-grid repetition, inspired by Cygnus' encodeWithPatches idea.
//! This is an independent Rust detector, not a Java encoder/writer translation.
//! Hashes are only an index: every admitted occurrence is compared in ALL
//! supplied planes, including alpha and hidden RGB. The existing standard
//! Replace dictionary and reference-frame writer provide the wire format.

use std::{
    cmp::Ordering,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex},
    time::Instant,
};

use super::{AtlasPacking, ImageView, PatchInfo, PatchKey, PatchPlan, build_plan, invalid};
use crate::{Error, Result};

#[path = "patch_tile_admission.rs"]
mod admission;

#[derive(Clone, Debug)]
pub struct TilePatchOptions {
    /// Diagnostic bypass of cheap admission. Never enabled by effort alone.
    pub force: bool,
    /// Estimated net saving as a fraction of a cheap sampled pixel-cost proxy.
    pub min_estimated_savings: f64,
    /// Stronger hurdle when a dictionary would leave the frame-area path.
    pub full_frame_min_estimated_savings: f64,
    /// Complete, disjoint tiles anchored at (0, 0). Partial edges are untouched.
    pub tile_size: usize,
    pub min_occurrences: usize,
    /// Retained dictionary limits, not a prefix limit on input discovery.
    pub max_unique_tiles: usize,
    pub max_positions: usize,
    pub max_reference_pixels: usize,
    /// Optional per-invocation evidence. Reused collectors accumulate scans.
    pub receipt: Option<Arc<TilePatchReceipt>>,
}

impl Default for TilePatchOptions {
    fn default() -> Self {
        Self {
            force: false,
            min_estimated_savings: 0.02,
            full_frame_min_estimated_savings: 0.10,
            tile_size: 16,
            min_occurrences: 3,
            max_unique_tiles: 4096,
            max_positions: 65_536,
            max_reference_pixels: 1 << 22,
            receipt: None,
        }
    }
}

impl TilePatchOptions {
    pub(crate) fn position_budget(&self, pixels: usize, extras: usize) -> Result<usize> {
        // Both decoder implementations bound blending-info capacity. Reserve
        // headroom for geometric growth, not just the final position count.
        let maximum = (pixels / 4)
            .checked_add(1024)
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::Limit("tile patch decoder position budget"))?;
        let blendings = maximum
            .checked_mul(4)
            .ok_or(Error::Limit("tile patch decoder blending budget"))?;
        let stride = extras
            .checked_add(1)
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::Limit("tile patch blending stride"))?;
        Ok(self.max_positions.min(maximum).min(blendings / stride))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.tile_size == 0
            || self.tile_size > i32::MAX as usize
            || self.tile_size.checked_mul(self.tile_size).is_none()
            || self.min_occurrences < 2
            || self.max_unique_tiles == 0
            || self.max_positions == 0
            || self.max_reference_pixels == 0
            || !self.min_estimated_savings.is_finite()
            || !(0.0..=1.0).contains(&self.min_estimated_savings)
            || !self.full_frame_min_estimated_savings.is_finite()
            || !(0.0..=1.0).contains(&self.full_frame_min_estimated_savings)
        {
            return Err(invalid("invalid exact tile patch options"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TilePatchStats {
    pub calls: usize,
    pub tiles_examined: usize,
    pub constant_tiles: usize,
    pub repeated_groups: usize,
    pub selected_groups: usize,
    pub positions: usize,
    pub covered_pixels: usize,
    pub atlas_pixels: usize,
    pub admission_rejected: usize,
    pub estimated_saved_bits: f64,
    pub estimated_baseline_bits: f64,
    /// Detector + atlas/residual preparation; excludes frame encoding and
    /// conversion of Image storage to the integer plane view.
    pub seconds: f64,
}

#[derive(Debug, Default)]
pub struct TilePatchReceipt(Mutex<TilePatchStats>);

impl TilePatchReceipt {
    pub fn snapshot(&self) -> TilePatchStats {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn record(&self, other: TilePatchStats) {
        let mut total = self.0.lock().unwrap_or_else(|e| e.into_inner());
        total.calls += other.calls;
        total.tiles_examined += other.tiles_examined;
        total.constant_tiles += other.constant_tiles;
        total.repeated_groups += other.repeated_groups;
        total.selected_groups += other.selected_groups;
        total.positions += other.positions;
        total.covered_pixels += other.covered_pixels;
        total.atlas_pixels += other.atlas_pixels;
        total.admission_rejected += other.admission_rejected;
        total.estimated_saved_bits += other.estimated_saved_bits;
        total.estimated_baseline_bits += other.estimated_baseline_bits;
        total.seconds += other.seconds;
    }
}

#[derive(Clone, Copy)]
struct Tile {
    hash: u64,
    origin: usize,
}

fn fingerprint(image: ImageView<'_>, origin: usize, side: usize) -> Option<u64> {
    let mut hash = DefaultHasher::new();
    let mut flat = true;
    for plane in image.planes {
        let first = plane[origin];
        for y in 0..side {
            let row = &plane[origin + y * image.width..origin + y * image.width + side];
            row.hash(&mut hash);
            flat &= row.iter().all(|&v| v == first);
        }
    }
    // A constant tile is already handled cheaply by prediction/RLE. Do not
    // turn plain backgrounds into thousands of dictionary entries.
    (!flat).then(|| hash.finish())
}

fn compare(image: ImageView<'_>, a: usize, b: usize, side: usize) -> Ordering {
    for plane in image.planes {
        for y in 0..side {
            let a = a + y * image.width;
            let b = b + y * image.width;
            let order = plane[a..a + side].cmp(&plane[b..b + side]);
            if order != Ordering::Equal {
                return order;
            }
        }
    }
    Ordering::Equal
}

/// Return ranges into the index, never copied tile payloads. A rare hash
/// collision uses exact ordering rather than a potentially quadratic chain.
fn repeated_ranges(
    image: ImageView<'_>,
    index: &mut [Tile],
    side: usize,
    minimum: usize,
) -> Vec<std::ops::Range<usize>> {
    index.sort_unstable_by_key(|tile| (tile.hash, tile.origin));
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < index.len() {
        let mut end = start + 1;
        while end < index.len() && index[end].hash == index[start].hash {
            end += 1;
        }
        if end - start >= minimum {
            let bucket = &mut index[start..end];
            let first = bucket[0].origin;
            if bucket
                .iter()
                .skip(1)
                .all(|tile| compare(image, first, tile.origin, side).is_eq())
            {
                ranges.push(start..end);
            } else {
                bucket.sort_unstable_by(|a, b| {
                    compare(image, a.origin, b.origin, side).then(a.origin.cmp(&b.origin))
                });
                let mut group = start;
                for next in start + 1..=end {
                    if next == end
                        || !compare(image, index[group].origin, index[next].origin, side).is_eq()
                    {
                        if next - group >= minimum {
                            ranges.push(group..next);
                        }
                        group = next;
                    }
                }
            }
        }
        start = end;
    }
    ranges
}

fn tile_atlas(
    count: usize,
    side: usize,
    max_columns: usize,
    budget: usize,
) -> Option<AtlasPacking> {
    let columns = count.isqrt().max(1).min(max_columns);
    let rows = count.div_ceil(columns);
    let width = columns.checked_mul(side)?;
    let height = rows.checked_mul(side)?;
    if width > i32::MAX as usize
        || height > i32::MAX as usize
        || width.checked_mul(height)? > budget
    {
        return None;
    }
    Some((
        width,
        height,
        (0..count)
            .map(|i| ((i % columns) * side, (i / columns) * side))
            .collect(),
    ))
}

/// Candidate only: reference-frame and dictionary costs can exceed savings.
/// All original planes are immutable. Unsupported/partial tiles stay in the
/// ordinary residual; a returned plan copies exact samples with Replace.
pub fn discover_tiles(
    image: ImageView<'_>,
    options: &TilePatchOptions,
) -> Result<Option<PatchPlan>> {
    options.validate()?;
    image.validate()?;
    let start = Instant::now();
    let mut stats = TilePatchStats {
        calls: 1,
        ..Default::default()
    };
    let result = discover_inner(image, options, &mut stats);
    stats.seconds = start.elapsed().as_secs_f64();
    if let Some(receipt) = &options.receipt {
        receipt.record(stats);
    }
    result
}

fn discover_inner(
    image: ImageView<'_>,
    options: &TilePatchOptions,
    stats: &mut TilePatchStats,
) -> Result<Option<PatchPlan>> {
    let side = options.tile_size;
    let (cols, rows) = (image.width / side, image.height / side);
    let area = side * side;
    let count = cols * rows;
    if count < options.min_occurrences || area > options.max_reference_pixels {
        return Ok(None);
    }
    let mut index = Vec::new();
    index
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("tile patch index allocation"))?;
    for y in 0..rows {
        for x in 0..cols {
            let origin = y * side * image.width + x * side;
            stats.tiles_examined += 1;
            if let Some(hash) = fingerprint(image, origin, side) {
                index.push(Tile { hash, origin });
            } else {
                stats.constant_tiles += 1;
            }
        }
    }
    let mut ranges = repeated_ranges(image, &mut index, side, options.min_occurrences);
    stats.repeated_groups = ranges.len();
    // This is a bounded raw-coverage priority, not an entropy or savings claim.
    ranges.sort_unstable_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then(index[a.start].origin.cmp(&index[b.start].origin))
    });
    ranges.truncate(
        options
            .max_unique_tiles
            .min(options.max_reference_pixels / area),
    );
    let mut remaining = options.position_budget(
        image.width * image.height,
        image.planes.len() - image.color_channels,
    )?;
    let mut selected = Vec::new();
    for mut range in ranges {
        let occurrences = range.len().min(remaining);
        if occurrences < options.min_occurrences {
            break;
        }
        range.end = range.start + occurrences;
        remaining -= occurrences;
        selected.push(range);
    }
    if selected.is_empty() {
        return Ok(None);
    }
    // Padding counts towards the explicit atlas budget too.
    let packing = loop {
        if let Some(packing) = tile_atlas(selected.len(), side, cols, options.max_reference_pixels)
        {
            break packing;
        }
        selected.pop();
        if selected.is_empty() {
            return Ok(None);
        }
    };
    let payload = area
        .checked_mul(image.planes.len())
        .ok_or(Error::Limit("tile patch samples"))?;
    let mut info = Vec::with_capacity(selected.len());
    for range in selected {
        let tiles = &index[range];
        let origin = tiles[0].origin;
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(payload)
            .map_err(|_| Error::Limit("tile patch payload allocation"))?;
        for plane in image.planes {
            for y in 0..side {
                let first = origin + y * image.width;
                samples.extend_from_slice(&plane[first..first + side]);
            }
        }
        info.push(PatchInfo {
            key: PatchKey {
                width: side,
                height: side,
                samples,
            },
            positions: tiles
                .iter()
                .map(|tile| {
                    (
                        (tile.origin % image.width) as u32,
                        (tile.origin / image.width) as u32,
                    )
                })
                .collect(),
        });
    }
    if !options.force {
        let estimate = admission::estimate(image, &info, &packing, options.tile_size);
        stats.estimated_saved_bits = estimate.saved_bits;
        stats.estimated_baseline_bits = estimate.baseline_bits;
        if estimate.saved_bits <= 0.0
            || estimate.saved_bits < options.min_estimated_savings * estimate.baseline_bits
        {
            stats.admission_rejected = 1;
            return Ok(None);
        }
    }
    let plan = build_plan(image, &info, packing)?;
    stats.selected_groups = plan.dictionary.references.len();
    stats.positions = plan.dictionary.positions.len();
    stats.covered_pixels = stats.positions * area;
    stats.atlas_pixels = plan.reference.width * plan.reference.height;
    Ok(Some(plan))
}

#[cfg(test)]
#[path = "patch_tiles_tests.rs"]
mod tests;
