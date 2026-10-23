// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! JPEG-to-VarDCT layout prerequisites, not a JPEG-to-JXL encoder.
//!
//! Source: jpeg/enc_jpeg_data.cc, frame_header.{h,cc}, frame_dimensions.h,
//! enc_frame.cc::ComputeJPEGTranscodingData and chroma_from_luma.h at
//! b866b3defec13494bfbaf705db93581cda118e37. Coefficients remain borrowed,
//! quantized integers. No pixel decode, requantization or frame output occurs.

use std::fmt;

use super::data::{BLOCK_SIZE, JpegData, MAX_DIM_PIXELS, MAX_QUANT_TABLES};

pub const CFL_FIXED_POINT_PRECISION: u8 = 11;
pub const CFL_FIXED_POINT_RATIO_MAX: i32 = (256 << CFL_FIXED_POINT_PRECISION) - 1;
/// coeff_order_fwd.h; only the DCT8 order is populated by this preparation.
pub const NUM_COEFFICIENT_ORDERS: usize = 13;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    Invalid(&'static str),
    Unsupported(&'static str),
    QuantizationRedefined {
        destination_id: u32,
        first_table_index: usize,
        table_index: usize,
    },
    CflRatioTooLarge {
        channel: usize,
        natural_coefficient: usize,
        ratio: i32,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid JPEG transcode layout: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported JPEG transcode layout: {reason}"),
            Self::QuantizationRedefined {
                destination_id,
                first_table_index,
                table_index,
            } => write!(
                f,
                "JPEG quantization slot {destination_id} changes between tables {first_table_index} and {table_index}"
            ),
            Self::CflRatioTooLarge {
                channel,
                natural_coefficient,
                ratio,
            } => write!(
                f,
                "JPEG CfL ratio {ratio} exceeds the source limit at channel {channel}, coefficient {natural_coefficient}"
            ),
        }
    }
}
impl std::error::Error for LayoutError {}

/// The only two source transforms that can describe a recompressed JPEG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum JpegColorTransform {
    None = 1,
    YCbCr = 2,
}

/// Source JpegOrder: JXL channel -> JPEG component, not the inverse map.
pub fn jpeg_order(transform: JpegColorTransform, grayscale: bool) -> [usize; 3] {
    if grayscale {
        [0, 0, 0]
    } else if transform == JpegColorTransform::YCbCr {
        [1, 0, 2]
    } else {
        [0, 1, 2]
    }
}

fn component_count(jpeg: &JpegData) -> Result<usize, LayoutError> {
    match jpeg.components.len() {
        n @ (1 | 3) => Ok(n),
        _ => Err(LayoutError::Unsupported("requires one or three components")),
    }
}

/// Source marker precedence: ANY APP0 means YCbCr; otherwise the first exact
/// 15-byte Adobe APP14 wins; only without that marker are RGB IDs considered.
/// app_data includes the marker byte and the two length bytes, as in the reader.
pub fn color_transform_from_jpeg(jpeg: &JpegData) -> Result<JpegColorTransform, LayoutError> {
    let count = component_count(jpeg)?;
    let mut apps = jpeg.app_data.iter();
    let mut adobe = None;
    for &marker in &jpeg.marker_order {
        if marker & 0xf0 != 0xe0 {
            continue;
        }
        let data = apps
            .next()
            .ok_or(LayoutError::Invalid("missing APP marker data"))?;
        if data.len() < 3
            || data[0] != marker
            || usize::from(u16::from_be_bytes([data[1], data[2]])) != data.len() - 1
        {
            return Err(LayoutError::Invalid("inconsistent APP marker data"));
        }
        if adobe.is_none() && marker == 0xee && data.len() == 15 && &data[3..8] == b"Adobe" {
            adobe = Some(data[14] == 0);
        }
    }
    if apps.next().is_some() {
        return Err(LayoutError::Invalid("unmatched APP marker data"));
    }
    let rgb_ids = count == 3
        && jpeg
            .components
            .iter()
            .zip(b"RGB")
            .all(|(c, &id)| c.id == u32::from(id));
    let rgb = !jpeg.marker_order.contains(&0xe0) && adobe.unwrap_or(rgb_ids);
    Ok(if count == 3 && rgb {
        JpegColorTransform::None
    } else {
        JpegColorTransform::YCbCr
    })
}

/// Validated source channel modes (0=1x1, 1=2x2, 2=2x1, 3=1x2).
/// Mode indices always swap JPEG components 0/1, independently of JpegOrder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChromaSubsampling {
    modes: [u8; 3],
    horizontal: [u8; 3],
    vertical: [u8; 3],
    max_horizontal: u8,
    max_vertical: u8,
}

impl ChromaSubsampling {
    pub fn from_sampling(horizontal: [u32; 3], vertical: [u32; 3]) -> Result<Self, LayoutError> {
        const H: [u8; 4] = [0, 1, 1, 0];
        const V: [u8; 4] = [0, 1, 0, 1];
        let mut modes = [0; 3];
        let mut raw_h = [0; 3];
        let mut raw_v = [0; 3];
        for (c, jpeg_c) in [1, 0, 2].into_iter().enumerate() {
            let mode = (0..4)
                .find(|&i| horizontal[jpeg_c] == 1 << H[i] && vertical[jpeg_c] == 1 << V[i])
                .ok_or(LayoutError::Unsupported(
                    "sampling factors must each be one or two",
                ))?;
            modes[c] = mode as u8;
            raw_h[c] = H[mode];
            raw_v[c] = V[mode];
        }
        let max_horizontal = *raw_h.iter().max().expect("three channels");
        let max_vertical = *raw_v.iter().max().expect("three channels");
        Ok(Self {
            modes,
            horizontal: raw_h.map(|n| max_horizontal - n),
            vertical: raw_v.map(|n| max_vertical - n),
            max_horizontal,
            max_vertical,
        })
    }

    pub fn from_jpeg(jpeg: &JpegData) -> Result<Self, LayoutError> {
        let count = component_count(jpeg)?;
        let index = if count == 1 { [0; 3] } else { [0, 1, 2] };
        Self::from_sampling(
            index.map(|i| jpeg.components[i].h_samp_factor),
            index.map(|i| jpeg.components[i].v_samp_factor),
        )
    }

    pub fn channel_modes(self) -> [u8; 3] {
        self.modes
    }
    pub fn horizontal_shifts(self) -> [u8; 3] {
        self.horizontal
    }
    pub fn vertical_shifts(self) -> [u8; 3] {
        self.vertical
    }
    pub fn max_shifts(self) -> (u8, u8) {
        (self.max_horizontal, self.max_vertical)
    }
    pub fn is_444(self) -> bool {
        self.horizontal == [0; 3] && self.vertical == [0; 3]
    }
    pub fn is_420(self) -> bool {
        self.horizontal == [1, 0, 1] && self.vertical == [1, 0, 1]
    }
    pub fn is_422(self) -> bool {
        self.horizontal == [1, 0, 1] && self.vertical == [0; 3]
    }
    pub fn is_440(self) -> bool {
        self.horizontal == [0; 3] && self.vertical == [1, 0, 1]
    }
}

/// Per-plane quantizer INPUT preparation. These DC factors precede
/// DequantMatricesSetCustomDC's F16 serialization round-trip, not its output.
#[derive(Clone, Debug)]
pub struct Quantization {
    table_index: usize,
    destination_id: u32,
    transposed: [i32; BLOCK_SIZE],
    cfl_ratios: [i32; BLOCK_SIZE],
    dc_quantization: f32,
    inverse_dc_quantization: f32,
    dc_offset: i32,
}

impl Quantization {
    /// Append-only Rust quant-vector index, resolved from participating SOS snapshots.
    pub fn table_index(&self) -> usize {
        self.table_index
    }
    pub fn destination_id(&self) -> u32 {
        self.destination_id
    }
    pub fn transposed(&self) -> &[i32; BLOCK_SIZE] {
        &self.transposed
    }
    /// Natural JPEG order, unlike transposed(). Ratios truncate positive integer division.
    pub fn cfl_ratios(&self) -> &[i32; BLOCK_SIZE] {
        &self.cfl_ratios
    }
    pub fn dc_quantization(&self) -> f32 {
        self.dc_quantization
    }
    pub fn inverse_dc_quantization(&self) -> f32 {
        self.inverse_dc_quantization
    }
    /// Zero for YCbCr/gray; integer 1024 / quant[0] for RGB.
    pub fn dc_offset(&self) -> i32 {
        self.dc_offset
    }
}

/// Borrowed natural-order input, or the source's implicit grayscale chroma zero.
#[derive(Clone, Copy, Debug)]
pub enum BlockView<'a> {
    Jpeg(&'a [i16; BLOCK_SIZE]),
    Zero,
}

impl BlockView<'_> {
    pub fn transposed_coefficient(self, index: usize) -> Option<i32> {
        if index >= BLOCK_SIZE {
            return None;
        }
        Some(match self {
            Self::Jpeg(block) => i32::from(block[8 * (index % 8) + index / 8]),
            Self::Zero => 0,
        })
    }

    /// Writes one caller-owned block scratch only; DC stays the original integer.
    pub fn write_transposed(self, output: &mut [i32; BLOCK_SIZE]) {
        for (index, coefficient) in output.iter_mut().enumerate() {
            *coefficient = self.transposed_coefficient(index).expect("block index");
        }
    }
}

#[derive(Debug)]
pub struct PlaneLayout<'a> {
    jpeg_component: usize,
    dimensions: (u32, u32),
    coefficients: Option<&'a [i16]>,
    quantization: Quantization,
}

impl<'a> PlaneLayout<'a> {
    pub fn jpeg_component(&self) -> usize {
        self.jpeg_component
    }
    pub fn dimensions_in_blocks(&self) -> (u32, u32) {
        self.dimensions
    }
    pub fn is_synthetic_zero(&self) -> bool {
        self.coefficients.is_none()
    }
    pub fn quantization(&self) -> &Quantization {
        &self.quantization
    }

    pub fn block(&self, x: u32, y: u32) -> Option<BlockView<'a>> {
        if x >= self.dimensions.0 || y >= self.dimensions.1 {
            return None;
        }
        let Some(coefficients) = self.coefficients else {
            return Some(BlockView::Zero);
        };
        let offset = (y as usize * self.dimensions.0 as usize + x as usize) * BLOCK_SIZE;
        Some(BlockView::Jpeg(
            coefficients[offset..offset + BLOCK_SIZE]
                .try_into()
                .expect("validated block geometry"),
        ))
    }

    pub fn adjusted_dc(&self, x: u32, y: u32) -> Option<i32> {
        Some(match self.block(x, y)? {
            BlockView::Zero => 0,
            BlockView::Jpeg(block) => i32::from(block[0]) + self.quantization.dc_offset,
        })
    }

    /// The source's DC image input; no whole-image allocation or pixel decoding.
    pub fn scaled_dc(&self, x: u32, y: u32) -> Option<f32> {
        Some(self.adjusted_dc(x, y)? as f32 * self.quantization.inverse_dc_quantization)
    }
}

/// Resolve on-wire IDs, never index `jpeg.quant` with Component::quant_index.
/// Repeated identical tables may differ in precision/marker grouping. A used
/// slot changing values is rejected even across disjoint component scans.
fn quant_tables(jpeg: &JpegData) -> Result<[usize; 3], LayoutError> {
    let mut slots: [Option<usize>; MAX_QUANT_TABLES] = [None; MAX_QUANT_TABLES];
    let mut components = [None; 3];
    for scan in &jpeg.scan_info {
        if scan.components.is_empty() || scan.components.len() > jpeg.components.len() {
            return Err(LayoutError::Invalid("invalid scan component count"));
        }
        let mut seen = [false; 3];
        for sc in &scan.components {
            let c = sc.component_index;
            if c >= jpeg.components.len() || seen[c] {
                return Err(LayoutError::Invalid("invalid or duplicate scan component"));
            }
            seen[c] = true;
            let slot = jpeg.components[c].quant_index as usize;
            if slot >= MAX_QUANT_TABLES {
                return Err(LayoutError::Invalid("invalid quantization destination ID"));
            }
            let index = scan.quant_table_indices[slot]
                .ok_or(LayoutError::Invalid("missing scan quantization table"))?;
            let table = jpeg.quant.get(index).ok_or(LayoutError::Invalid(
                "quantization snapshot index out of range",
            ))?;
            if table.index != slot as u32 || table.precision > 1 {
                return Err(LayoutError::Invalid(
                    "quantization table slot or precision mismatch",
                ));
            }
            let maximum = if table.precision == 0 { 255 } else { 65535 };
            if table.values.iter().any(|&q| !(1..=maximum).contains(&q)) {
                return Err(LayoutError::Invalid(
                    "quantization entry outside its precision range",
                ));
            }
            if let Some(first) = slots[slot] {
                if jpeg.quant[first].values != table.values {
                    return Err(LayoutError::QuantizationRedefined {
                        destination_id: slot as u32,
                        first_table_index: first,
                        table_index: index,
                    });
                }
            } else {
                slots[slot] = Some(index);
            }
            components[c].get_or_insert(index);
        }
    }
    let mut result = [0; 3];
    for c in 0..jpeg.components.len() {
        result[c] = components[c].ok_or(LayoutError::Invalid(
            "component has no scan quantization snapshot",
        ))?;
    }
    Ok(result)
}

#[derive(Debug)]
pub struct TranscodeLayout<'a> {
    color_transform: JpegColorTransform,
    order: [usize; 3],
    subsampling: ChromaSubsampling,
    frame_blocks: (u32, u32),
    grayscale: bool,
    planes: [PlaneLayout<'a>; 3],
}

impl<'a> TranscodeLayout<'a> {
    /// Requires decoded block-major coefficients and each component's SOS table
    /// snapshots. Validates the public data model without cloning coefficients.
    /// This is not a scan-progression, JPEG reconstruction, or JXL-frame validator.
    pub fn new(jpeg: &'a JpegData) -> Result<Self, LayoutError> {
        let count = component_count(jpeg)?;
        if jpeg.precision != 8 {
            return Err(LayoutError::Unsupported("only 8-bit JPEG precision"));
        }
        if !(1..=MAX_DIM_PIXELS).contains(&jpeg.width)
            || !(1..=MAX_DIM_PIXELS).contains(&jpeg.height)
        {
            return Err(LayoutError::Invalid("invalid frame dimensions"));
        }
        let mut ids = [false; 256];
        for component in &jpeg.components {
            if component.id > 255 || ids[component.id as usize] {
                return Err(LayoutError::Invalid("invalid or duplicate component ID"));
            }
            ids[component.id as usize] = true;
        }
        let color_transform = color_transform_from_jpeg(jpeg)?;
        let subsampling = ChromaSubsampling::from_jpeg(jpeg)?;
        let (max_h, max_v) = subsampling.max_shifts();
        // MakeFrameHeader checks RAW maxima, not Is444 (all-2x2 RGB also fails).
        if color_transform != JpegColorTransform::YCbCr && (max_h != 0 || max_v != 0) {
            return Err(LayoutError::Unsupported(
                "RGB JPEG sampling factors must all be one",
            ));
        }
        let mcu_cols = jpeg.width.div_ceil(8 << max_h);
        let mcu_rows = jpeg.height.div_ceil(8 << max_v);
        let frame_blocks = (mcu_cols << max_h, mcu_rows << max_v);
        for component in &jpeg.components {
            let width = mcu_cols * component.h_samp_factor;
            let height = mcu_rows * component.v_samp_factor;
            let length = (width as usize)
                .checked_mul(height as usize)
                .and_then(|n| n.checked_mul(BLOCK_SIZE));
            if component.width_in_blocks != width
                || component.height_in_blocks != height
                || length != Some(component.coefficients.len())
            {
                return Err(LayoutError::Invalid(
                    "coefficient storage does not match MCU-padded geometry",
                ));
            }
        }
        let tables = quant_tables(jpeg)?;
        let grayscale = count == 1;
        let order = jpeg_order(color_transform, grayscale);
        let luma_quant = &jpeg.quant[tables[order[1]]].values;
        let plane = |channel: usize| -> Result<PlaneLayout<'a>, LayoutError> {
            let jpeg_component = order[channel];
            let component = &jpeg.components[jpeg_component];
            let table_index = tables[jpeg_component];
            let table = &jpeg.quant[table_index];
            let mut transposed = [0; BLOCK_SIZE];
            let mut cfl_ratios = [0; BLOCK_SIZE];
            for natural in 0..BLOCK_SIZE {
                transposed[8 * (natural % 8) + natural / 8] = table.values[natural];
                // The source reads transposed qt and transposes the ratio back.
                let ratio =
                    (1 << CFL_FIXED_POINT_PRECISION) * luma_quant[natural] / table.values[natural];
                if ratio > CFL_FIXED_POINT_RATIO_MAX {
                    return Err(LayoutError::CflRatioTooLarge {
                        channel,
                        natural_coefficient: natural,
                        ratio,
                    });
                }
                cfl_ratios[natural] = ratio;
            }
            let dc_quantization = 255.0f32 * 8.0f32 / table.values[0] as f32;
            let inverse_dc_quantization = 1.0f32 / dc_quantization;
            Ok(PlaneLayout {
                jpeg_component,
                dimensions: (component.width_in_blocks, component.height_in_blocks),
                coefficients: if grayscale && channel != 1 {
                    None
                } else {
                    Some(&component.coefficients)
                },
                quantization: Quantization {
                    table_index,
                    destination_id: component.quant_index,
                    transposed,
                    cfl_ratios,
                    dc_quantization,
                    inverse_dc_quantization,
                    dc_offset: if color_transform == JpegColorTransform::YCbCr {
                        0
                    } else {
                        1024 / table.values[0]
                    },
                },
            })
        };
        Ok(Self {
            color_transform,
            order,
            subsampling,
            frame_blocks,
            grayscale,
            planes: [plane(0)?, plane(1)?, plane(2)?],
        })
    }

    pub fn color_transform(&self) -> JpegColorTransform {
        self.color_transform
    }
    pub fn jpeg_order(&self) -> [usize; 3] {
        self.order
    }
    pub fn subsampling(&self) -> ChromaSubsampling {
        self.subsampling
    }
    pub fn dimensions_in_blocks(&self) -> (u32, u32) {
        self.frame_blocks
    }
    pub fn is_grayscale(&self) -> bool {
        self.grayscale
    }
    pub fn planes(&self) -> &[PlaneLayout<'a>; 3] {
        &self.planes
    }
    /// Caller still needs the source force_cfl_jpeg_recompression option.
    pub fn can_use_ac_cfl(&self) -> bool {
        !self.grayscale && self.subsampling.is_444()
    }

    /// Source histogram/threshold preparation, scanning only borrowed DCs.
    /// Padding participates exactly as in FrameDimensions/ComputeJPEGTranscodingData.
    pub fn dc_contexts(&self) -> DcContexts {
        let luma = &self.planes[1];
        let (width, height) = luma.dimensions;
        let mut counts = [0u64; 2048];
        for y in 0..height {
            for x in 0..width {
                let dc = luma.adjusted_dc(x, y).expect("validated geometry");
                // Preserve std::min<uint32_t>: negative sums wrap before clamping.
                let bucket = ((dc + 1024) as u32).min(2047) as usize;
                counts[bucket] += 1;
            }
        }
        let total = u64::from(width) * u64::from(height);
        // This is deliberately qt[1..6] of CHANNEL ZERO, not the luma table.
        let quality_sum = self.planes[0].quantization.transposed[1..6]
            .iter()
            .map(|&q| q as u64)
            .sum();
        DcContexts::from_counts(&counts, total, quality_sum, self.grayscale)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DcContexts {
    thresholds: Vec<i32>,
    requested_thresholds: u8,
    context_map: Vec<u8>,
    num_contexts: u8,
}

impl DcContexts {
    fn from_counts(counts: &[u64; 2048], total: u64, quality_sum: u64, grayscale: bool) -> Self {
        let ceil_log2 = |n: u64| (u64::BITS - (n - 1).leading_zeros()) as i32;
        let requested_thresholds =
            (ceil_log2(total) - ceil_log2(quality_sum) - 7).clamp(1, 7) as u8;
        let mut thresholds = Vec::with_capacity(usize::from(requested_thresholds));
        let mut cumsum = 0;
        let buckets = u64::from(requested_thresholds) + 1;
        let mut cut = total / buckets;
        for (j, &count) in counts.iter().enumerate() {
            cumsum += count;
            // One threshold per bin, including empty bins after a large jump.
            if cumsum > cut {
                thresholds.push(j as i32 - 1025);
                cut = total * (thresholds.len() as u64 + 1) / buckets;
            }
        }
        let num_dc_contexts = thresholds.len() + 1;
        let mut context_map = vec![0; 3 * NUM_COEFFICIENT_ORDERS * num_dc_contexts];
        for i in 0..num_dc_contexts {
            context_map[i] = i as u8;
            if grayscale {
                context_map[NUM_COEFFICIENT_ORDERS * num_dc_contexts + i] = num_dc_contexts as u8;
                context_map[2 * NUM_COEFFICIENT_ORDERS * num_dc_contexts + i] =
                    num_dc_contexts as u8;
            } else {
                context_map[NUM_COEFFICIENT_ORDERS * num_dc_contexts + i] =
                    (num_dc_contexts + i / 2) as u8;
                context_map[2 * NUM_COEFFICIENT_ORDERS * num_dc_contexts + i] =
                    (num_dc_contexts + num_dc_contexts.div_ceil(2) + i / 2) as u8;
            }
        }
        let num_contexts = context_map
            .iter()
            .copied()
            .max()
            .expect("nonempty context map")
            + 1;
        Self {
            thresholds,
            requested_thresholds,
            context_map,
            num_contexts,
        }
    }

    /// Only JXL channel 1 has thresholds; channels 0/2 have empty vectors.
    pub fn luma_thresholds(&self) -> &[i32] {
        &self.thresholds
    }
    pub fn requested_thresholds(&self) -> u8 {
        self.requested_thresholds
    }
    pub fn num_dc_contexts(&self) -> usize {
        self.thresholds.len() + 1
    }
    /// Source block-context map order is Y, X, B; non-DCT8 orders remain zero.
    pub fn context_map(&self) -> &[u8] {
        &self.context_map
    }
    pub fn num_contexts(&self) -> u8 {
        self.num_contexts
    }
}

#[cfg(test)]
mod tests {
    use super::super::data::{CodingProcess, Component, ComponentScanInfo, QuantTable, ScanInfo};
    use super::*;

    fn fixture(width: u32, height: u32, sampling: &[(u32, u32)]) -> JpegData {
        let max_h = sampling.iter().map(|s| s.0).max().unwrap();
        let max_v = sampling.iter().map(|s| s.1).max().unwrap();
        let components = sampling
            .iter()
            .enumerate()
            .map(|(c, &(h, v))| {
                let width_in_blocks = width.div_ceil(max_h * 8) * h;
                let height_in_blocks = height.div_ceil(max_v * 8) * v;
                Component {
                    id: c as u32 + 1,
                    h_samp_factor: h,
                    v_samp_factor: v,
                    quant_index: [3, 1, 2][c],
                    width_in_blocks,
                    height_in_blocks,
                    coefficients: vec![
                        0;
                        width_in_blocks as usize
                            * height_in_blocks as usize
                            * BLOCK_SIZE
                    ],
                }
            })
            .collect();
        JpegData {
            width,
            height,
            precision: 8,
            coding_process: Some(CodingProcess::Baseline),
            components,
            // Deliberately unrelated wire-ID and append-only-vector order.
            quant: [2, 3, 1]
                .map(|index| QuantTable {
                    index,
                    values: std::array::from_fn(|i| 1 + 8 * index as i32 + i as i32),
                    ..QuantTable::default()
                })
                .to_vec(),
            scan_info: vec![ScanInfo {
                components: (0..sampling.len())
                    .map(|component_index| ComponentScanInfo {
                        component_index,
                        ..ComponentScanInfo::default()
                    })
                    .collect(),
                ss: 0,
                se: 63,
                quant_table_indices: [None, Some(2), Some(0), Some(1)],
                ..ScanInfo::default()
            }],
            ..JpegData::default()
        }
    }

    fn app(jpeg: &mut JpegData, marker: u8, payload: &[u8]) {
        let mut data = vec![marker];
        data.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        data.extend_from_slice(payload);
        jpeg.marker_order.push(marker);
        jpeg.app_data.push(data);
    }

    fn adobe(jpeg: &mut JpegData, transform: u8) {
        app(
            jpeg,
            0xee,
            &[b'A', b'd', b'o', b'b', b'e', 0, 100, 0, 0, 0, 0, transform],
        );
    }

    #[test]
    fn color_transform_preserves_source_marker_precedence() {
        let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
        for (c, id) in jpeg.components.iter_mut().zip(b"RGB") {
            c.id = u32::from(*id);
        }
        assert_eq!(
            color_transform_from_jpeg(&jpeg).unwrap(),
            JpegColorTransform::None
        );
        adobe(&mut jpeg, 1);
        assert_eq!(
            color_transform_from_jpeg(&jpeg).unwrap(),
            JpegColorTransform::YCbCr
        );
        adobe(&mut jpeg, 0); // First matching Adobe marker still wins.
        assert_eq!(
            color_transform_from_jpeg(&jpeg).unwrap(),
            JpegColorTransform::YCbCr
        );
        jpeg.app_data[0][14] = 0;
        assert_eq!(
            color_transform_from_jpeg(&jpeg).unwrap(),
            JpegColorTransform::None
        );
        app(&mut jpeg, 0xe0, b"not JFIF"); // Source checks the marker, not its payload.
        assert_eq!(
            color_transform_from_jpeg(&jpeg).unwrap(),
            JpegColorTransform::YCbCr
        );
        jpeg.app_data.pop();
        jpeg.marker_order.pop();
        jpeg.app_data[0][3] = b'a';
        jpeg.app_data[1][3] = b'a';
        assert_eq!(
            color_transform_from_jpeg(&jpeg).unwrap(),
            JpegColorTransform::None
        ); // ID fallback.
        jpeg.app_data[0][0] = 0xe1;
        assert!(color_transform_from_jpeg(&jpeg).is_err());
        let mut gray = fixture(8, 8, &[(1, 1)]);
        adobe(&mut gray, 0);
        assert_eq!(
            color_transform_from_jpeg(&gray).unwrap(),
            JpegColorTransform::YCbCr
        );
        assert_eq!(jpeg_order(JpegColorTransform::YCbCr, false), [1, 0, 2]);
        assert_eq!(jpeg_order(JpegColorTransform::None, false), [0, 1, 2]);
        assert_eq!(jpeg_order(JpegColorTransform::None, true), [0; 3]);
        gray.app_data.clear();
        assert!(color_transform_from_jpeg(&gray).is_err());
        gray.components.push(Component::default());
        assert!(matches!(
            color_transform_from_jpeg(&gray),
            Err(LayoutError::Unsupported(_))
        ));
    }

    #[test]
    fn all_source_sampling_modes_match_independent_shift_math() {
        let modes = [(1, 1), (2, 2), (2, 1), (1, 2)];
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    let samples = [modes[a], modes[b], modes[c]];
                    let cs = ChromaSubsampling::from_sampling(
                        samples.map(|s| s.0),
                        samples.map(|s| s.1),
                    )
                    .unwrap();
                    assert_eq!(cs.channel_modes(), [b as u8, a as u8, c as u8]);
                    let max_h = samples.iter().map(|s| s.0).max().unwrap();
                    let max_v = samples.iter().map(|s| s.1).max().unwrap();
                    assert_eq!(
                        cs.horizontal_shifts(),
                        [1, 0, 2].map(|i| u8::from(samples[i].0 != max_h))
                    );
                    assert_eq!(
                        cs.vertical_shifts(),
                        [1, 0, 2].map(|i| u8::from(samples[i].1 != max_v))
                    );
                }
            }
        }
        for bad in [0, 3, 4, 15, 256, u32::MAX] {
            assert!(ChromaSubsampling::from_sampling([bad, 1, 1], [1; 3]).is_err());
            assert!(ChromaSubsampling::from_sampling([1; 3], [1, bad, 1]).is_err());
        }
        let c420 = ChromaSubsampling::from_sampling([2, 1, 1], [2, 1, 1]).unwrap();
        assert!(c420.is_420());
        assert!(!c420.is_422());
        assert!(!c420.is_444());
        assert!(
            ChromaSubsampling::from_sampling([2, 1, 1], [1; 3])
                .unwrap()
                .is_422()
        );
        assert!(
            ChromaSubsampling::from_sampling([1; 3], [2, 1, 1])
                .unwrap()
                .is_440()
        );
        assert!(
            ChromaSubsampling::from_sampling([2; 3], [2; 3])
                .unwrap()
                .is_444()
        );
    }

    #[test]
    fn quantization_transpose_ratios_and_borrowed_block_golden() {
        let mut jpeg = fixture(17, 9, &[(2, 2), (1, 1), (1, 1)]);
        for (i, coefficient) in jpeg.components[1].coefficients.iter_mut().enumerate() {
            *coefficient = i as i16 - 70;
        }
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        assert_eq!(layout.jpeg_order(), [1, 0, 2]);
        assert_eq!(layout.dimensions_in_blocks(), (4, 2));
        assert!(!layout.can_use_ac_cfl());
        let cb = &layout.planes()[0];
        assert_eq!(cb.dimensions_in_blocks(), (2, 1));
        assert_eq!(cb.jpeg_component(), 1);
        let q = cb.quantization();
        assert_eq!((q.destination_id(), q.table_index()), (1, 2));
        assert_eq!(&q.transposed()[..8], &[9, 17, 25, 33, 41, 49, 57, 65]);
        assert_eq!(q.cfl_ratios()[1], 5324);
        for i in 0..64 {
            assert_eq!(q.transposed()[i], 9 + (i % 8 * 8 + i / 8) as i32);
            let rational = 2048u64 * (25 + i as u64) / (9 + i as u64);
            assert_eq!(q.cfl_ratios()[i], rational as i32);
        }
        let expected = (2040.0f64 / 9.0) as f32;
        assert_eq!(q.dc_quantization().to_bits(), expected.to_bits());
        assert_eq!(
            q.inverse_dc_quantization().to_bits(),
            ((1.0f64 / f64::from(expected)) as f32).to_bits()
        );
        let block = cb.block(1, 0).unwrap();
        let BlockView::Jpeg(borrowed) = block else {
            panic!("expected borrowed block")
        };
        assert_eq!(
            borrowed.as_ptr(),
            jpeg.components[1].coefficients[64..].as_ptr()
        );
        let mut output = [0; 64];
        block.write_transposed(&mut output);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(output[x * 8 + y], (64 + y * 8 + x) as i32 - 70);
            }
        }
        assert_eq!(block.transposed_coefficient(64), None);
        assert!(cb.block(2, 0).is_none());
        assert!(cb.block(0, 1).is_none());
    }

    #[test]
    fn rgb_offsets_and_grayscale_synthetic_planes_do_not_change_coefficients() {
        let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
        adobe(&mut jpeg, 0);
        jpeg.components[0].coefficients[0] = -10;
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        assert_eq!(layout.color_transform(), JpegColorTransform::None);
        assert_eq!(layout.jpeg_order(), [0, 1, 2]);
        assert!(layout.can_use_ac_cfl());
        let plane = &layout.planes()[0];
        assert_eq!(plane.quantization().dc_offset(), 40); // floor(1024/25).
        assert_eq!(plane.adjusted_dc(0, 0), Some(30));
        assert_eq!(
            plane.scaled_dc(0, 0),
            Some(30.0f32 * (1.0f32 / (2040.0f32 / 25.0)))
        );
        assert_eq!(
            plane.block(0, 0).unwrap().transposed_coefficient(0),
            Some(-10)
        );
        let mut gray = fixture(17, 9, &[(2, 2)]);
        gray.components[0].coefficients[0] = -71;
        let gray_layout = TranscodeLayout::new(&gray).unwrap();
        assert_eq!(gray_layout.jpeg_order(), [0; 3]);
        assert!(gray_layout.is_grayscale());
        assert!(!gray_layout.can_use_ac_cfl());
        for c in [0, 2] {
            assert!(gray_layout.planes()[c].is_synthetic_zero());
            assert_eq!(gray_layout.planes()[c].scaled_dc(0, 0), Some(0.0));
            let mut scratch = [123; 64];
            gray_layout.planes()[c]
                .block(0, 0)
                .unwrap()
                .write_transposed(&mut scratch);
            assert_eq!(scratch, [0; 64]);
        }
        assert_eq!(gray_layout.planes()[1].adjusted_dc(0, 0), Some(-71));
        let mut sampled_rgb = fixture(8, 8, &[(2, 2); 3]);
        adobe(&mut sampled_rgb, 0);
        assert!(ChromaSubsampling::from_jpeg(&sampled_rgb).unwrap().is_444());
        assert!(matches!(
            TranscodeLayout::new(&sampled_rgb),
            Err(LayoutError::Unsupported(_))
        ));
    }

    #[test]
    fn scan_quantization_ids_are_resolved_and_conflicting_redefinitions_rejected() {
        let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
        assert_eq!(
            TranscodeLayout::new(&jpeg).unwrap().planes()[1]
                .quantization()
                .table_index(),
            1
        );
        let mut repeated = jpeg.quant[1].clone();
        repeated.precision = 1; // Same values, different on-wire precision/grouping.
        repeated.is_last = false;
        jpeg.quant.push(repeated);
        let mut next = jpeg.scan_info[0].clone();
        next.quant_table_indices[3] = Some(3);
        jpeg.scan_info.push(next);
        assert!(TranscodeLayout::new(&jpeg).is_ok());
        jpeg.quant[3].values[63] += 1;
        assert_eq!(
            TranscodeLayout::new(&jpeg).unwrap_err(),
            LayoutError::QuantizationRedefined {
                destination_id: 3,
                first_table_index: 1,
                table_index: 3,
            }
        );
        // Even different components cannot give one used destination ID two meanings.
        jpeg.components[1].quant_index = 3;
        jpeg.scan_info[0].components = vec![ComponentScanInfo {
            component_index: 0,
            ..Default::default()
        }];
        jpeg.scan_info[1].components = vec![ComponentScanInfo {
            component_index: 1,
            ..Default::default()
        }];
        assert!(matches!(
            TranscodeLayout::new(&jpeg),
            Err(LayoutError::QuantizationRedefined { .. })
        ));
    }

    #[test]
    fn invalid_snapshot_values_and_geometry_fail_without_fallback() {
        let original = fixture(8, 8, &[(1, 1); 3]);
        for case in 0..13 {
            let mut jpeg = original.clone();
            match case {
                0 => jpeg.scan_info[0].quant_table_indices[3] = None,
                1 => jpeg.scan_info[0].quant_table_indices[3] = Some(99),
                2 => jpeg.quant[1].index = 0,
                3 => jpeg.quant[1].precision = 2,
                4 => jpeg.quant[1].values[0] = 0,
                5 => jpeg.quant[1].values[5] = -1,
                6 => jpeg.quant[1].values[5] = 256,
                7 => {
                    jpeg.components[0].coefficients.pop().unwrap();
                }
                8 => jpeg.components[0].width_in_blocks += 1,
                9 => jpeg.scan_info.clear(),
                10 => jpeg.scan_info[0].components[1].component_index = 0,
                11 => jpeg.width = 0,
                12 => jpeg.precision = 12,
                _ => unreachable!(),
            }
            assert!(TranscodeLayout::new(&jpeg).is_err(), "case {case}");
        }
    }

    #[test]
    fn fixed_point_ratio_limit_includes_16_bit_tables_and_dc_entry() {
        let mut jpeg = fixture(8, 8, &[(1, 1); 3]);
        for table in &mut jpeg.quant {
            table.precision = 1;
            table.values = [256; 64];
        }
        jpeg.quant[1].values = [65535; 64];
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        assert_eq!(layout.planes()[0].quantization().cfl_ratios()[0], 524280);
        assert_eq!(layout.planes()[1].quantization().cfl_ratios(), &[2048; 64]);
        jpeg.quant[1].values = [256; 64];
        jpeg.quant[2].values[0] = 1;
        assert_eq!(
            TranscodeLayout::new(&jpeg).unwrap_err(),
            LayoutError::CflRatioTooLarge {
                channel: 0,
                natural_coefficient: 0,
                ratio: 524288,
            }
        );
        jpeg.quant[1].values[0] = 255;
        assert!(TranscodeLayout::new(&jpeg).is_ok());
    }

    #[test]
    fn dc_threshold_and_context_map_goldens_keep_strict_cuts_and_empty_bins() {
        let uniform = [1024; 2048];
        let total = 2048 * 1024;
        for grayscale in [false, true] {
            let contexts = DcContexts::from_counts(&uniform, total, 5, grayscale);
            assert_eq!(contexts.requested_thresholds(), 7);
            assert_eq!(
                contexts.luma_thresholds(),
                [-769, -513, -257, -1, 255, 511, 767]
            );
            assert_eq!(contexts.num_dc_contexts(), 8);
            assert_eq!(contexts.context_map().len(), 3 * 13 * 8);
            assert_eq!(&contexts.context_map()[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(
                &contexts.context_map()[13 * 8..13 * 8 + 8],
                if grayscale {
                    &[8; 8]
                } else {
                    &[8, 8, 9, 9, 10, 10, 11, 11]
                }
            );
            assert_eq!(
                &contexts.context_map()[26 * 8..26 * 8 + 8],
                if grayscale {
                    &[8; 8]
                } else {
                    &[12, 12, 13, 13, 14, 14, 15, 15]
                }
            );
            assert_eq!(contexts.num_contexts(), if grayscale { 9 } else { 16 });
            for plane in contexts.context_map().chunks_exact(13 * 8) {
                assert!(plane[8..].iter().all(|&v| v == 0));
            }
        }
        let mut concentrated = [0; 2048];
        concentrated[1024] = total;
        let contexts = DcContexts::from_counts(&concentrated, total, 5, false);
        assert_eq!(contexts.luma_thresholds(), [-1, 0, 1, 2, 3, 4, 5]);
        concentrated[1024] = 0;
        concentrated[2047] = total;
        let contexts = DcContexts::from_counts(&concentrated, total, 5, false);
        assert_eq!(contexts.requested_thresholds(), 7);
        assert_eq!(contexts.luma_thresholds(), [1022]);
    }

    #[test]
    fn dc_preparation_uses_channel_zero_transposed_quant_and_padded_blocks() {
        let mut jpeg = fixture(512, 512, &[(1, 1); 3]);
        for table in &mut jpeg.quant {
            table.values = [64; 64];
        }
        for i in [8, 16, 24, 32, 40] {
            jpeg.quant[2].values[i] = 1;
        }
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let contexts = layout.dc_contexts();
        assert_eq!(contexts.requested_thresholds(), 2);
        assert_eq!(contexts.luma_thresholds(), [-1, 0]);
        let mut odd = fixture(1, 1, &[(2, 2)]);
        // Four MCU-padded blocks participate, not only the visible top-left block.
        for (block, dc) in odd.components[0]
            .coefficients
            .chunks_exact_mut(64)
            .zip([-1024, -1024, 0, 1023])
        {
            block[0] = dc;
        }
        assert_eq!(
            TranscodeLayout::new(&odd)
                .unwrap()
                .dc_contexts()
                .luma_thresholds(),
            [-1]
        );
        odd.components[0]
            .coefficients
            .chunks_exact_mut(64)
            .for_each(|b| b[0] = -1025);
        // Source unsigned conversion before upper clamping, not a lower clamp.
        assert_eq!(
            TranscodeLayout::new(&odd)
                .unwrap()
                .dc_contexts()
                .luma_thresholds(),
            [1022]
        );
    }
}
