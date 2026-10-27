//! One-input lookahead for the two retained canvases, without trial encoding.
use super::{diff::DiffMap, references::ReferenceSlots};
use crate::{Image, Result};
use std::sync::Arc;

#[derive(Default)]
pub(super) struct Lookahead {
    target: usize,
    maps: Vec<(usize, Arc<DiffMap>)>,
    pub(super) keep: Option<u8>,
    pub(super) comparisons: usize,
}

impl Lookahead {
    pub(super) fn map(&self, reference: usize, target: usize) -> Option<Arc<DiffMap>> {
        (self.target == target)
            .then(|| self.maps.iter().find(|(index, _)| *index == reference))
            .flatten()
            .map(|(_, map)| Arc::clone(map))
    }

    pub(super) fn prepare(images: &[Image], index: usize, slots: &ReferenceSlots) -> Result<Self> {
        if index + 1 == images.len()
            || images[index]
                .metadata()
                .extra_channels
                .iter()
                .any(|e| e.dim_shift != 0)
        {
            return Ok(Self::default());
        }
        let target = index + 1;
        let mut maps = Vec::with_capacity(3);
        let current = Arc::new(DiffMap::between(&images[index], &images[target])?);
        let current_score = score(&current);
        maps.push((index, current));
        let mut ranked = Vec::with_capacity(2);
        for (reference, slot) in slots.sources(index) {
            let map = Arc::new(DiffMap::between(&images[reference], &images[target])?);
            ranked.push((score(&map), slot));
            maps.push((reference, map));
        }
        ranked.sort_unstable();
        // Override the candidate's normal eviction only for a strictly useful
        // old canvas, and never on a tie between the available old canvases.
        let keep = ranked.first().and_then(|&(best, slot)| {
            (best < current_score && ranked.get(1).is_none_or(|&(other, _)| best < other))
                .then_some(slot)
        });
        Ok(Self {
            target,
            comparisons: maps.len(),
            maps,
            keep,
        })
    }

    pub(super) fn retain(&mut self, slots: &ReferenceSlots) {
        self.maps
            .retain(|(index, _)| slots.frames.contains(&Some(*index)));
    }
}

fn score(map: &DiffMap) -> (u64, u64) {
    let area = map
        .bounds()
        .map_or(0, |r| u64::from(r.width) * u64::from(r.height));
    (area, map.changed_pixels())
}
