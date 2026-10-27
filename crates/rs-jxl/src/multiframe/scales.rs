//! Optional extra rectangle grids. Reuse exact difference maps and retain the
//! original candidates; geometry ranks proposals but never predicts a winner.
use super::{AutoOptions, Reference, Strategy, auto::Candidate, diff::DiffMap, partition, regions};
use crate::Result;
use std::sync::Arc;

pub(super) fn extend(
    index: usize,
    options: &AutoOptions,
    bytes_per_pixel: f64,
    maps: &[(usize, u8, Arc<DiffMap>)],
    result: &mut Vec<Candidate>,
) -> Result<()> {
    if !options.allow_multi_rect {
        return Ok(());
    }
    for (scale_index, &tile_size) in options.extra_tile_sizes.iter().enumerate() {
        if tile_size == options.tile_size
            || options.extra_tile_sizes[..scale_index].contains(&tile_size)
        {
            continue;
        }
        for &(reference, source, ref map) in maps {
            if map.bounds().is_none() {
                continue;
            }
            let first = reference == 0 && (index != 1 || options.reference == Reference::First);
            let previous = reference + 1 == index;
            let tiles = regions::tiles_from_map(map, tile_size, options.max_rectangles)?;
            let merged = if options.adaptive_rectangles {
                partition::propose(map, tile_size, options.max_rectangles, bytes_per_pixel)?
            } else {
                None
            };
            for (adaptive, rects) in [(false, tiles), (true, merged)] {
                let Some(rects) = rects else { continue };
                if rects.len() <= 1
                    || result.iter().any(|candidate| {
                        !candidate.mask && candidate.source == source && candidate.rects == rects
                    })
                {
                    continue;
                }
                let strategy = match (adaptive, first, previous) {
                    (false, true, _) => Strategy::TilesFirst,
                    (false, false, true) => Strategy::TilesPrevious,
                    (false, false, false) => Strategy::TilesRetained,
                    (true, true, _) => Strategy::MergedFirst,
                    (true, false, true) => Strategy::MergedPrevious,
                    (true, false, false) => Strategy::MergedRetained,
                };
                result.push(Candidate {
                    strategy,
                    tile_size: Some(tile_size),
                    reference_index: Some(reference),
                    diff: None,
                    source,
                    rects,
                    mask: false,
                });
            }
        }
    }
    Ok(())
}
