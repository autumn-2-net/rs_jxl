// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md.
//! Port of ProcessSOF in lib/jxl/jpeg/enc_jpeg_data_reader.cc, b866b3d.
//! This parses ONE SOF segment, not a JPEG file or its entropy-coded data.
use super::data::{Component, JpegData, MAX_COMPONENTS, MAX_DIM_PIXELS};

/// `segment` starts at the two-byte big-endian marker length, excluding FF/Cn.
/// Returns consumed bytes. Error leaves the destination unchanged.
pub fn process_sof(segment: &[u8], jpeg: &mut JpegData) -> Result<usize, &'static str> {
    if jpeg.width != 0 {
        return Err("duplicate SOF");
    }
    if segment.len() < 8 {
        return Err("truncated SOF header");
    }
    let len = usize::from(u16::from_be_bytes([segment[0], segment[1]]));
    let precision = segment[2];
    let height = u32::from(u16::from_be_bytes([segment[3], segment[4]]));
    let width = u32::from(u16::from_be_bytes([segment[5], segment[6]]));
    let count = usize::from(segment[7]);
    if precision != 8 {
        return Err("JPEG reconstruction supports 8-bit precision only");
    }
    if !(1..=MAX_DIM_PIXELS).contains(&width) || !(1..=MAX_DIM_PIXELS).contains(&height) {
        return Err("invalid SOF dimensions");
    }
    if !(1..=MAX_COMPONENTS).contains(&count) {
        return Err("invalid SOF component count");
    }
    if len != 8 + 3 * count || segment.len() < len {
        return Err("invalid SOF length");
    }
    let mut components = Vec::with_capacity(count);
    let mut seen = [false; 256];
    let mut max_h = 1;
    let mut max_v = 1;
    for component in segment[8..len].chunks_exact(3) {
        let id = usize::from(component[0]);
        if seen[id] {
            return Err("duplicate SOF component ID");
        }
        seen[id] = true;
        let h = u32::from(component[1] >> 4);
        let v = u32::from(component[1] & 15);
        if h == 0 || v == 0 {
            return Err("invalid SOF sampling factor");
        }
        max_h = max_h.max(h);
        max_v = max_v.max(v);
        components.push(Component {
            id: id as u32,
            h_samp_factor: h,
            v_samp_factor: v,
            quant_index: u32::from(component[2]),
            ..Default::default()
        });
    }
    let cols = width.div_ceil(max_h * 8);
    let rows = height.div_ceil(max_v * 8);
    for c in &mut components {
        if max_h % c.h_samp_factor != 0 || max_v % c.v_samp_factor != 0 {
            return Err("non-integral SOF subsampling");
        }
        c.width_in_blocks = cols * c.h_samp_factor;
        c.height_in_blocks = rows * c.v_samp_factor;
    }
    jpeg.width = width;
    jpeg.height = height;
    jpeg.components = components;
    Ok(len)
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOF: [u8; 17] = [0, 17, 8, 0, 9, 0, 17, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1];
    #[test]
    fn parses_subsampled_frame_without_pixel_allocation() {
        let mut jpeg = JpegData::default();
        assert_eq!(process_sof(&SOF, &mut jpeg), Ok(17));
        assert_eq!((jpeg.width, jpeg.height), (17, 9));
        assert_eq!(
            (
                jpeg.components[0].width_in_blocks,
                jpeg.components[0].height_in_blocks
            ),
            (4, 2)
        );
        assert!(jpeg.components.iter().all(|c| c.coefficients.is_empty()));
        assert!(process_sof(&SOF, &mut jpeg).is_err());
    }
    #[test]
    fn failures_are_checked_and_atomic() {
        for n in 0..SOF.len() {
            let mut jpeg = JpegData::default();
            assert!(process_sof(&SOF[..n], &mut jpeg).is_err());
            assert_eq!(jpeg.width, 0);
            assert!(jpeg.components.is_empty());
        }
        for (index, value) in [
            (1, 16),
            (2, 12),
            (5, 0),
            (7, 5),
            (11, 1),
            (9, 0x20),
            (12, 0x31),
        ] {
            let mut bytes = SOF;
            bytes[index] = value;
            if index == 5 {
                bytes[6] = 0;
            }
            let mut jpeg = JpegData::default();
            assert!(
                process_sof(&bytes, &mut jpeg).is_err(),
                "case {index}/{value}"
            );
            assert!(jpeg.components.is_empty());
        }
    }
}
