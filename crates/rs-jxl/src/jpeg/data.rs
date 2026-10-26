// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Semantic port of lib/jxl/jpeg/jpeg_data.h at b866b3d, with archival records.
//! Internal Rust structures, not a C ABI or serialized memory layout.

use std::fmt;

pub const MAX_COMPONENTS: usize = 4;
pub const BLOCK_SIZE: usize = 64;
pub const MAX_QUANT_TABLES: usize = 4;
pub const MAX_HUFFMAN_TABLES: usize = 4;
/// No cumulative DHT definition ceiling; table slot IDs remain format-bounded.
pub const MAX_DHT_MARKERS: usize = usize::MAX;
pub const MAX_DIM_PIXELS: u32 = 65535;

/// Structural errors include the byte offset in the input JPEG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JpegError {
    Truncated {
        offset: usize,
        context: &'static str,
    },
    Invalid {
        offset: usize,
        reason: &'static str,
    },
    Unsupported {
        offset: usize,
        reason: &'static str,
    },
    LimitExceeded {
        offset: usize,
        resource: &'static str,
    },
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { offset, context } => {
                write!(f, "truncated JPEG at byte {offset}: {context}")
            }
            Self::Invalid { offset, reason } => {
                write!(f, "invalid JPEG at byte {offset}: {reason}")
            }
            Self::Unsupported { offset, reason } => {
                write!(f, "unsupported JPEG at byte {offset}: {reason}")
            }
            Self::LimitExceeded { offset, resource } => {
                write!(f, "JPEG limit exceeded at byte {offset}: {resource}")
            }
        }
    }
}

impl std::error::Error for JpegError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodingProcess {
    Baseline,
    ExtendedSequential,
    Progressive,
}

/// An exact marker segment, excluding preceding fill bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerSegment {
    /// Offset of the marker's FF prefix in the input.
    pub offset: usize,
    pub marker: u8,
    /// Original length bytes and payload; empty for SOI and EOI.
    pub data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestartMarker {
    /// Offset of FF/Dn relative to `EntropyData::bytes`, after any FF fill.
    pub offset: usize,
    pub marker: u8,
    /// Additional 0xff bytes preceding the marker prefix, not entropy bytes.
    pub fill_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntropyData {
    pub offset: usize,
    pub scan_index: usize,
    /// Uninterpreted scan bytes, including FF/00 stuffing, FF/Dn and restart fill.
    /// Fill before the terminating non-restart marker belongs to a Fill record.
    pub bytes: Vec<u8>,
    pub restart_markers: Vec<RestartMarker>,
}

/// Ordered archival records, not a cached copy of the complete input file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JpegRecord {
    Marker(MarkerSegment),
    Fill {
        offset: usize,
        count: usize,
    },
    /// Index into `JpegData::inter_marker_data`.
    InterMarker {
        offset: usize,
        index: usize,
    },
    Entropy(EntropyData),
    /// The bytes following EOI live in `JpegData::tail_data`.
    Tail {
        offset: usize,
    },
}

pub const NATURAL_ORDER: [usize; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63,
];
pub const ZIGZAG_ORDER: [usize; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantTable {
    pub values: [i32; BLOCK_SIZE],
    pub precision: u32,
    pub index: u32,
    pub is_last: bool,
}
impl Default for QuantTable {
    fn default() -> Self {
        Self {
            values: [0; BLOCK_SIZE],
            precision: 0,
            index: 0,
            is_last: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HuffmanCode {
    /// Original on-wire histogram. Index zero is unused; no sentinel is added.
    pub counts: [u32; 17],
    /// Original symbols in code-length order; unused entries are zero.
    pub values: [u32; 257],
    pub slot_id: u32,
    pub is_last: bool,
}
impl Default for HuffmanCode {
    fn default() -> Self {
        Self {
            counts: [0; 17],
            values: [0; 257],
            slot_id: 0,
            is_last: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComponentScanInfo {
    pub component_index: usize,
    pub dc_table_index: u32,
    pub ac_table_index: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ExtraZeroRun {
    pub block_index: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ScanInfo {
    pub ss: u32,
    pub se: u32,
    pub ah: u32,
    pub al: u32,
    // Vec length replaces C++ num_components + fixed array; validate <= 4.
    pub components: Vec<ComponentScanInfo>,
    /// Active DRI at this SOS, in scan MCUs (not blocks).
    pub restart_interval: u32,
    pub mcu_cols: u32,
    pub mcu_rows: u32,
    /// Snapshots of all active table slots at this SOS. Indices refer to the
    /// append-only quant / huffman_code vectors, not on-wire destination IDs.
    pub quant_table_indices: [Option<usize>; MAX_QUANT_TABLES],
    pub dc_huffman_indices: [Option<usize>; MAX_HUFFMAN_TABLES],
    pub ac_huffman_indices: [Option<usize>; MAX_HUFFMAN_TABLES],
    pub last_needed_pass: u32,
    pub reset_points: Vec<u32>,
    pub extra_zero_runs: Vec<ExtraZeroRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub id: u32,
    pub h_samp_factor: u32,
    pub v_samp_factor: u32,
    /// On-wire DQT destination ID, resolved using the scan's table snapshot.
    pub quant_index: u32,
    pub width_in_blocks: u32,
    pub height_in_blocks: u32,
    /// Quantized coefficients in block-major natural order. Empty before entropy decoding.
    pub coefficients: Vec<i16>,
}
impl Default for Component {
    fn default() -> Self {
        Self {
            id: 0,
            h_samp_factor: 1,
            v_samp_factor: 1,
            quant_index: 0,
            width_in_blocks: 0,
            height_in_blocks: 0,
            coefficients: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum AppMarkerType {
    #[default]
    Unknown = 0,
    Icc = 1,
    Exif = 2,
    Xmp = 3,
}

#[derive(Clone, Debug, Default)]
pub struct JpegData {
    pub width: u32,
    pub height: u32,
    pub precision: u8,
    pub coding_process: Option<CodingProcess>,
    pub restart_interval: u32,
    pub app_data: Vec<Vec<u8>>,
    pub app_marker_type: Vec<AppMarkerType>,
    pub com_data: Vec<Vec<u8>>,
    pub quant: Vec<QuantTable>,
    pub huffman_code: Vec<HuffmanCode>,
    pub components: Vec<Component>,
    pub scan_info: Vec<ScanInfo>,
    /// Top-level marker codes including SOI/EOI, excluding restart markers.
    pub marker_order: Vec<u8>,
    pub inter_marker_data: Vec<Vec<u8>>,
    pub tail_data: Vec<u8>,
    pub has_zero_padding_bit: bool,
    pub padding_bits: Vec<u8>,
    pub records: Vec<JpegRecord>,
}

impl JpegData {
    pub fn entropy_data(&self, scan_index: usize) -> Option<&EntropyData> {
        self.records.iter().find_map(|record| match record {
            JpegRecord::Entropy(data) if data.scan_index == scan_index => Some(data),
            _ => None,
        })
    }

    /// Replays archival records. Parsed metadata is descriptive: changing a
    /// table or frame field does not rewrite its original marker segment.
    /// This is JPEG reconstruction, not JPEG-to-JXL encoding or validation of
    /// user-edited record order / entropy data.
    pub fn reconstruct(&self) -> Result<Vec<u8>, JpegError> {
        let inter_marker = |index: usize, offset: usize| {
            self.inter_marker_data
                .get(index)
                .map(Vec::as_slice)
                .ok_or(JpegError::Invalid {
                    offset,
                    reason: "invalid inter-marker record index",
                })
        };
        let overflow = || JpegError::LimitExceeded {
            offset: 0,
            resource: "reconstructed byte length",
        };
        let mut len = 0usize;
        for record in &self.records {
            let n = match record {
                JpegRecord::Marker(segment) => {
                    segment.data.len().checked_add(2).ok_or_else(overflow)?
                }
                JpegRecord::Fill { count, .. } => *count,
                JpegRecord::InterMarker { offset, index } => inter_marker(*index, *offset)?.len(),
                JpegRecord::Entropy(data) => data.bytes.len(),
                JpegRecord::Tail { .. } => self.tail_data.len(),
            };
            len = len.checked_add(n).ok_or_else(overflow)?;
        }
        let mut output = Vec::new();
        output.try_reserve_exact(len).map_err(|_| overflow())?;
        for record in &self.records {
            match record {
                JpegRecord::Marker(segment) => {
                    output.extend_from_slice(&[0xff, segment.marker]);
                    output.extend_from_slice(&segment.data);
                }
                JpegRecord::Fill { count, .. } => output.resize(output.len() + *count, 0xff),
                JpegRecord::InterMarker { offset, index } => {
                    output.extend_from_slice(inter_marker(*index, *offset)?);
                }
                JpegRecord::Entropy(data) => output.extend_from_slice(&data.bytes),
                JpegRecord::Tail { .. } => output.extend_from_slice(&self.tail_data),
            }
        }
        Ok(output)
    }

    /// Port of JPEGData::CalculateMcuSize; validates mutable/public structures.
    pub fn mcu_size(&self, scan: &ScanInfo) -> Result<(u32, u32), &'static str> {
        if self.width == 0
            || self.height == 0
            || self.width > MAX_DIM_PIXELS
            || self.height > MAX_DIM_PIXELS
        {
            return Err("invalid JPEG dimensions");
        }
        if self.components.is_empty()
            || self.components.len() > MAX_COMPONENTS
            || scan.components.is_empty()
            || scan.components.len() > MAX_COMPONENTS
        {
            return Err("invalid JPEG component count");
        }
        let mut seen = [false; MAX_COMPONENTS];
        for sc in &scan.components {
            if sc.component_index >= self.components.len() || seen[sc.component_index] {
                return Err("invalid scan component index");
            }
            seen[sc.component_index] = true;
        }
        let mut max_h = 1;
        let mut max_v = 1;
        for c in &self.components {
            if !(1..=15).contains(&c.h_samp_factor) || !(1..=15).contains(&c.v_samp_factor) {
                return Err("invalid sampling factor");
            }
            max_h = max_h.max(c.h_samp_factor);
            max_v = max_v.max(c.v_samp_factor);
        }
        if self
            .components
            .iter()
            .any(|c| max_h % c.h_samp_factor != 0 || max_v % c.v_samp_factor != 0)
        {
            return Err("non-integral subsampling ratio");
        }
        let c = &self.components[scan.components[0].component_index];
        let (h, v) = if scan.components.len() > 1 {
            (1, 1)
        } else {
            (c.h_samp_factor, c.v_samp_factor)
        };
        Ok((
            (self.width * h).div_ceil(8 * max_h),
            (self.height * v).div_ceil(8 * max_v),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_includes_offset_and_reason() {
        let error: Box<dyn std::error::Error> = Box::new(JpegError::Invalid {
            offset: 42,
            reason: "bad marker",
        });
        assert_eq!(error.to_string(), "invalid JPEG at byte 42: bad marker");
        assert!(error.source().is_none());
    }

    #[test]
    fn reconstruction_checks_public_record_indices_and_length_overflow() {
        let mut jpeg = JpegData {
            records: vec![JpegRecord::InterMarker {
                offset: 2,
                index: 0,
            }],
            ..Default::default()
        };
        assert!(matches!(jpeg.reconstruct(), Err(JpegError::Invalid { .. })));
        jpeg.records = vec![
            JpegRecord::Fill {
                offset: 0,
                count: usize::MAX,
            },
            JpegRecord::Marker(MarkerSegment {
                offset: 0,
                marker: 0xd9,
                data: vec![],
            }),
        ];
        assert!(matches!(
            jpeg.reconstruct(),
            Err(JpegError::LimitExceeded { .. })
        ));
        jpeg.records.pop();
        assert!(matches!(
            jpeg.reconstruct(),
            Err(JpegError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn zigzag_and_natural_order_are_inverse() {
        for (i, &natural) in NATURAL_ORDER[..64].iter().enumerate() {
            assert_eq!(ZIGZAG_ORDER[natural], i);
        }
        assert!(NATURAL_ORDER[64..].iter().all(|&v| v == 63));
    }
    #[test]
    fn interleaved_and_noninterleaved_mcu_geometry() {
        let mut jpeg = JpegData {
            width: 17,
            height: 9,
            components: vec![
                Component {
                    h_samp_factor: 2,
                    v_samp_factor: 2,
                    ..Default::default()
                },
                Component::default(),
                Component::default(),
            ],
            ..Default::default()
        };
        let scan = ScanInfo {
            components: (0..3)
                .map(|i| ComponentScanInfo {
                    component_index: i,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        assert_eq!(jpeg.mcu_size(&scan), Ok((2, 1)));
        for (i, expected) in [(0, (3, 2)), (1, (2, 1)), (2, (2, 1))] {
            let scan = ScanInfo {
                components: vec![ComponentScanInfo {
                    component_index: i,
                    ..Default::default()
                }],
                ..Default::default()
            };
            assert_eq!(jpeg.mcu_size(&scan), Ok(expected));
        }
        jpeg.components[0].h_samp_factor = 0;
        assert!(jpeg.mcu_size(&scan).is_err());
        assert!(JpegData::default().mcu_size(&ScanInfo::default()).is_err());
    }
}
