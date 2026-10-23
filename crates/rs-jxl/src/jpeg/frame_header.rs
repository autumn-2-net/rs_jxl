// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! JPEG VarDCT geometry/framing from frame_dimensions.h, frame_header.{h,cc},
//! toc.h and enc_frame.cc::MakeFrameHeader at frozen b866b3d.
//! One full-image final frame/pass, original-color data and no extra channels.
//! Coefficient/DC/control sections are supplied separately; this is not encoding.

use super::data::MAX_DIM_PIXELS;
use super::transcode_layout::{ChromaSubsampling, JpegColorTransform};
use crate::{Error, Result, bit_writer::BitWriter, codestream::write_choice};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockRect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameGeometry {
    width: u32,
    height: u32,
    subsampling: ChromaSubsampling,
    blocks: (usize, usize),
    groups: (usize, usize),
    dc_groups: (usize, usize),
}

impl FrameGeometry {
    pub fn new(width: u32, height: u32, subsampling: ChromaSubsampling) -> Result<Self> {
        if !(1..=MAX_DIM_PIXELS).contains(&width) || !(1..=MAX_DIM_PIXELS).contains(&height) {
            return Err(Error::InvalidInput(
                "JPEG frame dimensions must be 1..=65535".into(),
            ));
        }
        let (h, v) = subsampling.max_shifts();
        let blocks = (
            (width as usize).div_ceil(8 << h) << h,
            (height as usize).div_ceil(8 << v) << v,
        );
        Ok(Self {
            width,
            height,
            subsampling,
            blocks,
            groups: (
                (width as usize).div_ceil(256),
                (height as usize).div_ceil(256),
            ),
            dc_groups: (blocks.0.div_ceil(256), blocks.1.div_ceil(256)),
        })
    }
    pub fn dimensions(self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn subsampling(self) -> ChromaSubsampling {
        self.subsampling
    }
    pub fn dimensions_in_blocks(self) -> (usize, usize) {
        self.blocks
    }
    pub fn group_count(self) -> usize {
        self.groups.0 * self.groups.1
    }
    pub fn dc_group_count(self) -> usize {
        self.dc_groups.0 * self.dc_groups.1
    }
    pub fn section_count(self) -> usize {
        if self.group_count() == 1 {
            1
        } else {
            2 + self.dc_group_count() + self.group_count()
        }
    }
    pub fn ac_group_rect(self, group: usize) -> Option<BlockRect> {
        self.rect(group, self.groups, 32)
    }
    pub fn dc_group_rect(self, group: usize) -> Option<BlockRect> {
        self.rect(group, self.dc_groups, 256)
    }
    fn rect(self, group: usize, grid: (usize, usize), span: usize) -> Option<BlockRect> {
        if group >= grid.0 * grid.1 {
            return None;
        }
        let x = group % grid.0 * span;
        let y = group / grid.0 * span;
        Some(BlockRect {
            x,
            y,
            width: span.min(self.blocks.0 - x),
            height: span.min(self.blocks.1 - y),
        })
    }
}

/// Serialize one final JPEG coefficient-frame header and its nonpermuted TOC.
/// LF smoothing is explicitly disabled: pre-0.13 decoders otherwise change
/// reconstructed JPEG DC coefficients. Gaborish/EPF are disabled as well.
pub fn write_header(
    geometry: FrameGeometry,
    color: JpegColorTransform,
    section_sizes: &[usize],
    max_bytes: usize,
) -> Result<Vec<u8>> {
    if section_sizes.len() != geometry.section_count() {
        return Err(Error::InvalidInput(
            "JPEG frame section count does not match geometry".into(),
        ));
    }
    if color == JpegColorTransform::None && geometry.subsampling.max_shifts() != (0, 0) {
        return Err(Error::InvalidInput(
            "RGB JPEG frame cannot have chroma subsampling".into(),
        ));
    }
    let mut w = BitWriter::new(max_bytes);
    w.write(1, 0)?; // Nondefault FrameHeader.
    w.write(2, 0)?; // Regular frame.
    w.write(1, 0)?; // VarDCT.
    w.write(2, 2)?; // U64 flag encoding for 17..272.
    w.write(8, 128 - 17)?; // kSkipAdaptiveDCSmoothing.
    w.write(1, u64::from(color == JpegColorTransform::YCbCr))?;
    if color == JpegColorTransform::YCbCr {
        for mode in geometry.subsampling.channel_modes() {
            w.write(2, u64::from(mode))?;
        }
    }
    w.write(2, 0)?; // Upsampling 1.
    w.write(2, 0)?; // One pass; no XYB scales or Modular group-size field.
    w.write(1, 0)?; // Full canvas.
    w.write(2, 0)?; // Replace blending.
    w.write(1, 1)?; // Last frame; no reference slot or save-before flag.
    w.write(2, 0)?; // Empty frame name.
    w.write(1, 0)?; // Nondefault LoopFilter.
    w.write(1, 0)?; // No Gaborish.
    w.write(2, 0)?; // No EPF.
    w.write(2, 0)?; // LoopFilter extensions.
    w.write(2, 0)?; // FrameHeader extensions.
    w.write(1, 0)?; // No TOC permutation.
    w.zero_pad_to_byte()?;
    for &size in section_sizes {
        write_choice(
            &mut w,
            u32::try_from(size).map_err(|_| Error::Limit("JPEG frame section bytes"))?,
            [(0, 10), (1024, 14), (17408, 22), (4211712, 30)],
        )?;
    }
    w.zero_pad_to_byte()?;
    Ok(w.into_bytes()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling(h: [u32; 3], v: [u32; 3]) -> ChromaSubsampling {
        ChromaSubsampling::from_sampling(h, v).unwrap()
    }

    #[test]
    fn jpeg_group_geometry_covers_padded_mcus_once() {
        for (width, height) in [
            (1, 1),
            (15, 17),
            (256, 256),
            (257, 259),
            (2049, 2051),
            (65535, 65535),
        ] {
            for subsampling in [
                sampling([1; 3], [1; 3]),
                sampling([2, 1, 1], [2, 1, 1]),
                sampling([2, 1, 1], [1; 3]),
            ] {
                let geometry = FrameGeometry::new(width, height, subsampling).unwrap();
                let (h, v) = subsampling.max_shifts();
                let bx = (width as usize).div_ceil(8 << h) * (1 << h);
                let by = (height as usize).div_ceil(8 << v) * (1 << v);
                assert_eq!(geometry.dimensions_in_blocks(), (bx, by));
                for dc in [false, true] {
                    let count = if dc {
                        geometry.dc_group_count()
                    } else {
                        geometry.group_count()
                    };
                    let mut area = 0;
                    for i in 0..count {
                        let r = if dc {
                            geometry.dc_group_rect(i)
                        } else {
                            geometry.ac_group_rect(i)
                        }
                        .unwrap();
                        assert!(
                            r.width > 0
                                && r.height > 0
                                && r.x + r.width <= bx
                                && r.y + r.height <= by
                        );
                        area += r.width * r.height;
                    }
                    assert_eq!(area, bx * by);
                }
                assert!(geometry.ac_group_rect(geometry.group_count()).is_none());
                assert!(geometry.dc_group_rect(geometry.dc_group_count()).is_none());
            }
        }
    }

    #[test]
    fn frame_fields_have_explicit_smoothing_off_and_expected_toc() {
        fn push(bits: &mut Vec<u8>, n: usize, value: u64) {
            bits.extend((0..n).map(|i| ((value >> i) & 1) as u8));
        }
        for (color, sub) in [
            (JpegColorTransform::None, sampling([1; 3], [1; 3])),
            (JpegColorTransform::YCbCr, sampling([2, 1, 1], [2, 1, 1])),
        ] {
            let geometry = FrameGeometry::new(17, 19, sub).unwrap();
            let mut expected = Vec::new();
            for (n, v) in [
                (1, 0),
                (2, 0),
                (1, 0),
                (2, 2),
                (8, 111),
                (1, u64::from(color == JpegColorTransform::YCbCr)),
            ] {
                push(&mut expected, n, v);
            }
            if color == JpegColorTransform::YCbCr {
                for mode in [0, 1, 0] {
                    push(&mut expected, 2, mode);
                }
            }
            for (n, v) in [
                (2, 0),
                (2, 0),
                (1, 0),
                (2, 0),
                (1, 1),
                (2, 0),
                (1, 0),
                (1, 0),
                (2, 0),
                (2, 0),
                (2, 0),
                (1, 0),
            ] {
                push(&mut expected, n, v);
            }
            expected.resize(expected.len().div_ceil(8) * 8, 0);
            push(&mut expected, 2, 0);
            push(&mut expected, 10, 37);
            expected.resize(expected.len().div_ceil(8) * 8, 0);
            let bytes = write_header(geometry, color, &[37], 100).unwrap();
            let actual: Vec<_> = bytes
                .iter()
                .flat_map(|b| (0..8).map(move |i| (b >> i) & 1))
                .collect();
            assert_eq!(actual, expected);
            assert!(write_header(geometry, color, &[37], bytes.len() - 1).is_err());
        }
    }

    #[test]
    fn invalid_geometry_and_section_contracts_fail() {
        let sub = sampling([1; 3], [1; 3]);
        assert!(FrameGeometry::new(0, 1, sub).is_err());
        assert!(FrameGeometry::new(1, 65536, sub).is_err());
        let geometry = FrameGeometry::new(257, 1, sub).unwrap();
        assert_eq!(geometry.section_count(), 5);
        assert!(write_header(geometry, JpegColorTransform::None, &[1], 100).is_err());
        assert!(write_header(geometry, JpegColorTransform::None, &[usize::MAX; 5], 100).is_err());
        let sub = FrameGeometry::new(17, 19, sampling([2, 1, 1], [2, 1, 1])).unwrap();
        assert!(write_header(sub, JpegColorTransform::None, &[1], 100).is_err());
    }
}
