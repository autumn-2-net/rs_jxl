// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::LazyLock;

use crate::api::DitherMode;

// 32x32 blue noise dithering pattern from
// PBBN_32T.png by DZgas (CC BY)
// https://gate-dzgas.com/Perfect-best-blue-noise.html
// Stored as 8-bit rank values p in 0..255 (1,024 bytes).
// Scaled to zero-centered floats in (-0.5, +0.5) via (2.0 * p - 255.0) / 512.0.
// Rows are expanded to 64 (32 + 32) at runtime to allow wrap-around SIMD loads.
const DITHER_TABLE_U8: &[u8; 1024] = include_bytes!("dither_32x32.bin");

#[path = "dither_djxl.rs"]
mod reference;

pub type DitherTable = [[f32; 64]; 32];

// Also used by smooth_squeeze: this must remain the upstream table regardless
// of the decoder's output dithering mode.
pub static DITHER_TABLE: LazyLock<Box<DitherTable>> = LazyLock::new(|| {
    let mut table = crate::util::box_array([0.0f32; 64]);
    for y in 0..32 {
        for x in 0..32 {
            let p = DITHER_TABLE_U8[y * 32 + x] as f32;
            let v = (2.0 * p - 255.0) / 512.0;
            table[y][x] = v;
            table[y][x + 32] = v;
        }
    }
    table
});

pub static LIBJXL_DITHER_TABLE: LazyLock<Box<DitherTable>> = LazyLock::new(|| {
    let mut table = crate::util::box_array([0.0f32; 64]);
    for y in 0..32 {
        for x in 0..32 {
            let v = reference::REFERENCE_DITHER[y][x];
            table[y][x] = v;
            table[y][x + 32] = v;
        }
    }
    table
});

pub fn dither_table(mode: DitherMode) -> &'static DitherTable {
    match mode {
        DitherMode::JxlRs => &DITHER_TABLE,
        DitherMode::Libjxl => &LIBJXL_DITHER_TABLE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_constants_and_simd_padding() {
        let table = dither_table(DitherMode::Libjxl);
        assert_eq!(table[0][0].to_bits(), (-0.26057f32).to_bits());
        assert_eq!(table[0][31].to_bits(), (-0.48833f32).to_bits());
        let mut sum = 0.0f64;
        for (y, row) in table.iter().enumerate() {
            for x in 0..32 {
                let v = row[x];
                assert!(v.is_finite() && v.abs() < 0.5);
                assert_eq!(v.to_bits(), reference::REFERENCE_DITHER[y][x].to_bits());
                assert_eq!(v.to_bits(), row[x + 32].to_bits());
                sum += f64::from(v);
            }
        }
        assert!(sum.abs() < 0.00001, "nonzero noise mean: {sum}");
    }

    #[test]
    fn default_preserves_upstream_ranks_and_simd_padding() {
        assert_eq!(DitherMode::default(), DitherMode::JxlRs);
        let table = dither_table(DitherMode::default());
        assert!(std::ptr::eq(table, &**DITHER_TABLE));
        for (y, row) in table.iter().enumerate() {
            for x in 0..32 {
                let expected = (2.0 * f32::from(DITHER_TABLE_U8[y * 32 + x]) - 255.0) / 512.0;
                assert_eq!(row[x].to_bits(), expected.to_bits());
                assert_eq!(row[x].to_bits(), row[x + 32].to_bits());
            }
        }
        assert_ne!(table, dither_table(DitherMode::Libjxl));
    }

    #[test]
    fn integer_samples_survive_dithering_for_all_positions() {
        for mode in [DitherMode::JxlRs, DitherMode::Libjxl] {
            for sample in 0..=255u16 {
                let normalized = f32::from(sample) / 255.0;
                for row in dither_table(mode).iter() {
                    for noise in &row[..32] {
                        let rounded = (normalized * 255.0 + noise)
                            .clamp(0.0, 255.0)
                            .round_ties_even() as u16;
                        assert_eq!(rounded, sample);
                    }
                }
            }
        }
    }
}
