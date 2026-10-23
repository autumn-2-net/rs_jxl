// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77.cc fast special_dist_table_ / bucket special_lookup_,
//! enc_lz77_index.h special_lookup_, and enc_lz77_suffix.h symbols_ at
//! b866b3d. These are bounded symbol tables, not distance-price caches.
//! Holes retain the generic distance code.

use super::super::{DistanceCodes, SPECIAL_DISTANCES, WINDOW_SIZE, reserved, special_distance};
use crate::Result;

pub(super) struct FastCodes {
    special_count: usize,
    dense: Vec<i8>,
}

impl FastCodes {
    pub(super) fn new(width: usize) -> Result<Self> {
        let mut dense = Vec::new();
        if width != 0 {
            // Source allocates width * 8 + 8. No fast query exceeds the
            // dictionary window, so its unreachable suffix needs no storage.
            let size = width
                .saturating_mul(8)
                .saturating_add(8)
                .min(WINDOW_SIZE + 1);
            dense = reserved(size)?;
            dense.resize(size, -1);
            for symbol in (0..SPECIAL_DISTANCES.len()).rev() {
                let distance = special_distance(symbol, width);
                if distance < size as i128 {
                    dense[distance as usize] = symbol as i8;
                }
            }
        }
        Ok(Self {
            special_count: if width == 0 {
                0
            } else {
                SPECIAL_DISTANCES.len()
            },
            dense,
        })
    }

    pub(super) fn symbol(&self, distance: usize) -> u32 {
        debug_assert!((1..=WINDOW_SIZE).contains(&distance));
        match self.dense.get(distance).copied() {
            Some(symbol) if symbol >= 0 => symbol as u32,
            _ => (self.special_count + distance - 1) as u32,
        }
    }
}

pub(super) struct Codes {
    sparse: DistanceCodes,
    dense: Vec<i16>,
}

impl Codes {
    pub(super) fn new(width: usize, enabled: bool) -> Result<Self> {
        let sparse = DistanceCodes::new(width, WINDOW_SIZE);
        let mut dense = Vec::new();
        if enabled && width != 0 {
            let largest = (0..SPECIAL_DISTANCES.len())
                .map(|symbol| special_distance(symbol, width))
                .max()
                .unwrap_or(0);
            if largest > 0 && largest < 65536 {
                dense = reserved(largest as usize + 1)?;
                dense.resize(largest as usize + 1, -1);
                // Reverse insertion preserves the lowest code for narrow-row aliases.
                for symbol in (0..SPECIAL_DISTANCES.len()).rev() {
                    dense[special_distance(symbol, width) as usize] = symbol as i16;
                }
            }
        }
        Ok(Self { sparse, dense })
    }

    pub(super) fn symbol(&self, distance: usize) -> u32 {
        if self.dense.is_empty() {
            return self.sparse.symbol(distance);
        }
        match self.dense.get(distance).copied() {
            Some(symbol) if symbol >= 0 => symbol as u32,
            _ => (self.sparse.special_count + distance - 1) as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_i8_symbols_preserve_source_aliases_and_window_clipped_tables() {
        for width in [
            0,
            1,
            2,
            37,
            8192,
            WINDOW_SIZE / 8 - 1,
            WINDOW_SIZE / 8,
            WINDOW_SIZE,
            usize::MAX,
        ] {
            let fast = FastCodes::new(width).unwrap();
            let scalar = DistanceCodes::new(width, WINDOW_SIZE);
            assert_eq!(
                fast.dense.len(),
                if width == 0 {
                    0
                } else {
                    width
                        .saturating_mul(8)
                        .saturating_add(8)
                        .min(WINDOW_SIZE + 1)
                }
            );
            for distance in 1..=65536 {
                assert_eq!(
                    fast.symbol(distance),
                    scalar.symbol(distance),
                    "{width}/{distance}"
                );
            }
            for symbol in 0..SPECIAL_DISTANCES.len() {
                let distance = special_distance(symbol, width);
                if distance <= WINDOW_SIZE as i128 {
                    let distance = distance as usize;
                    assert_eq!(fast.symbol(distance), scalar.symbol(distance));
                }
            }
            for distance in [WINDOW_SIZE - 1, WINDOW_SIZE] {
                assert_eq!(fast.symbol(distance), scalar.symbol(distance));
            }
            if !fast.dense.is_empty() {
                assert_eq!(fast.dense[0], -1);
            }
        }
    }

    #[test]
    fn dense_symbols_match_scalar_including_aliases_holes_and_last_distance() {
        for width in [0, 1, 2, 37, 256, 8191, 8192, 9361, 9362, usize::MAX] {
            let dense = Codes::new(width, true).unwrap();
            let sparse = Codes::new(width, false).unwrap();
            for distance in 1..=65536 {
                assert_eq!(
                    dense.symbol(distance),
                    sparse.symbol(distance),
                    "{width}/{distance}"
                );
            }
            assert_eq!(dense.symbol(WINDOW_SIZE), sparse.symbol(WINDOW_SIZE));
            assert!(dense.dense.len() <= 65536);
            if !dense.dense.is_empty() {
                assert_eq!(dense.dense[0], -1);
            }
        }
    }

    #[test]
    fn source_strict_65536_boundary_and_default_gate() {
        // The table's largest offset is 7 * width + 8 for these row widths.
        assert_eq!(Codes::new(9361, true).unwrap().dense.len(), 65536);
        assert!(Codes::new(9362, true).unwrap().dense.is_empty());
        assert!(Codes::new(37, false).unwrap().dense.is_empty());
        assert!(Codes::new(0, true).unwrap().dense.is_empty());
    }
}
