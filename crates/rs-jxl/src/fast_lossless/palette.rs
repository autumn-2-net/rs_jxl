// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! b866b3d enc_fast_lossless.cc: hash palette detection and alpha*luma order.

use super::ImageView;
#[cfg(test)]
use super::Options;

fn hash(pixel: u32) -> usize {
    (pixel.wrapping_mul(2_654_435_761) >> 16) as usize
}

pub(super) struct Palette {
    pub colors: Vec<u32>,
    lookup: Vec<u16>,
}

impl Palette {
    pub fn index(&self, pixel: u32) -> u16 {
        self.lookup[hash(pixel)]
    }

    #[cfg(test)]
    pub fn detect(image: ImageView<'_>, options: &Options) -> Option<Self> {
        if options.effort < 2 || image.bit_depth != 8 || !options.oneshot {
            return None;
        }
        let mut table = vec![0u32; 65536];
        if !Self::add_region(&mut table, image) {
            return None;
        }
        Self::from_table(table, image.channels)
    }

    pub(super) fn add_region(table: &mut [u32], image: ImageView<'_>) -> bool {
        for y in 0..image.height {
            for x in 0..image.width {
                let pixel = image.pixel8(x, y);
                let h = hash(pixel);
                // The source zero sentinel cannot distinguish a nonzero
                // hash-zero color from the mandatory zero palette entry.
                if (h == 0 && pixel != 0) || (table[h] != 0 && table[h] != pixel) {
                    return false;
                }
                table[h] = pixel;
            }
        }
        true
    }

    pub(super) fn from_table(table: Vec<u32>, channels: u8) -> Option<Self> {
        let mut colors = Vec::new();
        let mut have_color = false;
        let mut min_g = 255i32;
        let mut max_g = 0i32;
        for pixel in table.into_iter().filter(|&p| p != 0) {
            let [r, g, b, _] = pixel.to_le_bytes();
            have_color |= r != g || r != b;
            min_g = min_g.min(i32::from(g));
            max_g = max_g.max(i32::from(g));
            colors.push(pixel);
            if colors.len() + 1 > 512 {
                return None;
            }
        }
        if !have_color && ((max_g - min_g) as f32) < colors.len() as f32 * 1.4 {
            return None;
        }
        let score = |pixel: u32| {
            let [r, g, b, a] = pixel.to_le_bytes();
            let y = 0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b) + 0.01;
            if channels == 4 { y * f32::from(a) } else { y }
        };
        // Source comparator has no secondary tie key (std::sort ties unspecified).
        colors.sort_by(|&a, &b| score(a).partial_cmp(&score(b)).unwrap());
        colors.insert(0, 0);
        let mut lookup = vec![0u16; 65536];
        for (i, &pixel) in colors.iter().enumerate() {
            lookup[hash(pixel)] = i as u16;
        }
        Some(Self { colors, lookup })
    }
}
