// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! JPEG quantization prerequisites from libjxl b866b3defec13494bfbaf705db93581cda118e37.
//! Sources: enc_quant_weights.cc::{DequantMatricesEncodeDC, DequantMatricesSetCustomDC,
//! DequantMatricesEncode}, quant_weights.{h,cc}::{SetDCQuant, DecodeDC, QuantEncoding::RAW},
//! quantizer.cc::QuantizerParams::VisitFields, and enc_modular.cc::{AddQuantTable,
//! EncodeQuantTable}. enc_frame.cc supplies the JPEG-specific scale and channel order.
//!
//! This is not a full JPEG transcoder, VarDCT encoder, or lossy weight generator.
//! Callers resolve JPEG tables into JXL channel order and supply the Modular stream
//! writer. No padding is inserted at the header/Modular boundary.

use super::data::{BLOCK_SIZE, QuantTable};
use super::transcode_layout::Quantization;
use crate::bit_writer::BitWriter;
use crate::metadata::f16_bits_truncating as f16_bits;
use crate::modular::group::Channel;
use crate::{Error, Result};

pub const NUM_QUANT_TABLES: usize = 17;
pub const DCT8_QUANT_TABLE_INDEX: usize = 0;
pub const QUANT_TABLE_BITS: u8 = 8;
const MODE_LIBRARY: u64 = 0;
const MODE_RAW: u64 = 7;
const DEFAULT_INVERSE_DC: [f32; 3] = [4096.0, 512.0, 256.0];
const DEFAULT_DC: [f32; 3] = [1.0 / 4096.0, 1.0 / 512.0, 1.0 / 256.0];
const ALMOST_ZERO: f32 = 1e-8;

// fields.cc::F16Coder::Read; deliberately not an IEEE rounding conversion.
fn read_f16(bits: u16) -> Result<f32> {
    let sign = u32::from(bits >> 15);
    let exponent = u32::from((bits >> 10) & 31);
    let mantissa = u32::from(bits & 1023);
    if exponent == 31 {
        return Err(Error::InvalidInput(
            "quantization F16 must be finite".into(),
        ));
    }
    if exponent == 0 {
        let value = (1.0f32 / 16384.0) * (mantissa as f32 * (1.0f32 / 1024.0));
        return Ok(if sign != 0 { -value } else { value });
    }
    Ok(f32::from_bits(
        (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13),
    ))
}

fn dc_fields(dc: [f32; 3]) -> Result<Option<[u16; 3]>> {
    if dc == DEFAULT_DC {
        return Ok(None);
    }
    let mut fields = [0; 3];
    for (field, value) in fields.iter_mut().zip(dc) {
        *field = f16_bits(value * 128.0f32)?;
        if read_f16(*field)? * (1.0f32 / 128.0) < ALMOST_ZERO {
            return Err(Error::InvalidInput("DC quantization is too small".into()));
        }
    }
    Ok(Some(fields))
}

/// DequantMatricesEncodeDC; values are DCQuants, NOT the inverse SetCustomDC input.
/// Invalid values fail before writing. Writer allocation errors may leave a prefix.
pub fn write_dc_quantization(writer: &mut BitWriter, dc_quants: [f32; 3]) -> Result<()> {
    let fields = dc_fields(dc_quants)?;
    writer.write(1, u64::from(fields.is_none()))?;
    if let Some(fields) = fields {
        for field in fields {
            writer.write(16, u64::from(field))?;
        }
    }
    Ok(())
}

/// Decoder-visible DC factors after SetCustomDC's encode/decode round-trip.
/// They do not replace the original layout factors used to prepare JPEG DCs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DcQuantization {
    dc: [f32; 3],
    inverse_dc: [f32; 3],
}

impl Default for DcQuantization {
    fn default() -> Self {
        Self {
            dc: DEFAULT_DC,
            inverse_dc: DEFAULT_INVERSE_DC,
        }
    }
}

impl DcQuantization {
    /// `dc` is SetCustomDC's inverse-quant input, e.g. `2040.0 / jpeg_quant[0]`.
    pub fn from_inverse(dc: [f32; 3]) -> Result<Self> {
        if dc.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Error::InvalidInput(
                "inverse DC quantization must be positive and finite".into(),
            ));
        }
        let mut result = Self {
            dc: dc.map(|v| 1.0f32 / v),
            inverse_dc: dc,
        };
        // DecodeDC leaves the existing state unchanged on the all-default bit.
        if let Some(fields) = dc_fields(result.dc)? {
            for (c, field) in fields.into_iter().enumerate() {
                result.dc[c] = read_f16(field)? * (1.0f32 / 128.0);
                result.inverse_dc[c] = 1.0f32 / result.dc[c];
            }
        }
        Ok(result)
    }

    pub fn from_layout(channels: [&Quantization; 3]) -> Result<Self> {
        Self::from_inverse(channels.map(Quantization::dc_quantization))
    }

    /// Tables must already be in internal JXL channel order, not JPEG slot order.
    pub fn from_jpeg_tables(tables: [&QuantTable; 3]) -> Result<Self> {
        validate_jpeg_tables(tables)?;
        Self::from_inverse(tables.map(|table| 255.0f32 * 8.0f32 / table.values[0] as f32))
    }

    pub fn dc_quants(&self) -> &[f32; 3] {
        &self.dc
    }
    pub fn inverse_dc_quants(&self) -> &[f32; 3] {
        &self.inverse_dc
    }

    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        write_dc_quantization(writer, self.dc)
    }
}

/// Source wire fields. Default is VisitFields' `(1, 1)`, not the JPEG preset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantizerParams {
    pub global_scale: u32,
    pub quant_dc: u32,
}

impl Default for QuantizerParams {
    fn default() -> Self {
        Self {
            global_scale: 1,
            quant_dc: 1,
        }
    }
}

impl QuantizerParams {
    /// JPEG uses unit inverse global scale and unit per-block raw quantization.
    pub const fn jpeg() -> Self {
        Self {
            global_scale: 65536,
            quant_dc: 1,
        }
    }

    /// QuantizerParams::VisitFields has no all-default prefix or byte padding.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        let global = choose_u32(
            self.global_scale,
            [(11, 1), (11, 2049), (12, 4097), (16, 8193)],
        )?;
        let dc = choose_u32(self.quant_dc, [(0, 16), (5, 1), (8, 1), (16, 1)])?;
        for (selector, bits, value) in [global, dc] {
            writer.write(2, u64::from(selector))?;
            writer.write(bits, u64::from(value))?;
        }
        Ok(())
    }
}

// fields.cc::U32Coder::ChooseSelector: least extra bits, then first occurrence.
fn choose_u32(value: u32, choices: [(u8, u32); 4]) -> Result<(u8, u8, u32)> {
    let mut best: Option<(u8, u8, u32)> = None;
    for (selector, (bits, offset)) in choices.into_iter().enumerate() {
        if let Some(extra) = value.checked_sub(offset)
            && u64::from(extra) < (1u64 << bits)
            && best.is_none_or(|(_, previous, _)| bits < previous)
        {
            best = Some((selector as u8, bits, extra));
        }
    }
    best.ok_or_else(|| Error::InvalidInput("quantizer field has no source U32 encoding".into()))
}

fn validate_jpeg_tables(tables: [&QuantTable; 3]) -> Result<()> {
    for table in tables {
        let maximum = match table.precision {
            0 => 255,
            1 => 65535,
            _ => {
                return Err(Error::InvalidInput(
                    "invalid JPEG quantization precision".into(),
                ));
            }
        };
        if table.values.iter().any(|&v| !(1..=maximum).contains(&v)) {
            return Err(Error::InvalidInput(
                "JPEG quantization value exceeds its precision".into(),
            ));
        }
    }
    Ok(())
}

/// DCT8 RAW table in channel-major, already-transposed JXL order.
/// Samples stay signed integers even though AddQuantTable declares depth eight.
/// The scalar denominator is decoder-rounded; it does not scale stored integers.
#[derive(Clone, Debug, PartialEq)]
pub struct RawDct8Quantization {
    samples: [[i32; BLOCK_SIZE]; 3],
    denominator_bits: u16,
}

impl RawDct8Quantization {
    /// Checks RAW stream constraints, not the unported complete weight generator.
    /// All positive i32 symbols are preserved; JPEG factories enforce JPEG precision.
    pub fn new(samples: [[i32; BLOCK_SIZE]; 3], denominator: f32) -> Result<Self> {
        if samples.iter().flatten().any(|&value| value <= 0) {
            return Err(Error::InvalidInput(
                "raw quantization values must be positive".into(),
            ));
        }
        let denominator_bits = f16_bits(denominator)?;
        if read_f16(denominator_bits)? < ALMOST_ZERO {
            return Err(Error::InvalidInput(
                "raw quantization denominator is too small".into(),
            ));
        }
        Ok(Self {
            samples,
            denominator_bits,
        })
    }

    /// QuantEncoding::RAW: `(1 << shift) * (1.f / (8 * 255))`, followed by F16.
    /// Reject shifts outside source's defined positive signed-int domain.
    pub fn from_source_shift(samples: [[i32; BLOCK_SIZE]; 3], shift: u32) -> Result<Self> {
        if shift > 30 {
            return Err(Error::InvalidInput(
                "raw quantization shift exceeds source int domain".into(),
            ));
        }
        Self::new(samples, (1u32 << shift) as f32 * (1.0f32 / (8.0 * 255.0)))
    }

    pub fn from_layout(channels: [&Quantization; 3]) -> Result<Self> {
        Self::from_source_shift(channels.map(|q| *q.transposed()), 0)
    }

    /// Caller resolves slots/channel order. This transposes natural JPEG entries once.
    pub fn from_jpeg_tables(tables: [&QuantTable; 3]) -> Result<Self> {
        validate_jpeg_tables(tables)?;
        Self::from_source_shift(
            tables.map(|table| std::array::from_fn(|i| table.values[8 * (i % 8) + i / 8])),
            0,
        )
    }

    pub fn samples(&self) -> &[[i32; BLOCK_SIZE]; 3] {
        &self.samples
    }
    pub fn denominator_bits(&self) -> u16 {
        self.denominator_bits
    }
    pub fn denominator(&self) -> f32 {
        read_f16(self.denominator_bits).expect("validated F16")
    }

    /// AddQuantTable's three integer planes: 8x8, hshift/vshift zero, depth eight.
    /// Suitable for the parent's unpadded `modular::encode::write_prepared_stream`.
    pub fn prepare_modular_channels(&self) -> Result<[Channel; 3]> {
        let mut channels = std::array::from_fn(|_| Channel {
            width: 8,
            height: 8,
            hshift: 0,
            vshift: 0,
            samples: Vec::new(),
        });
        for (channel, samples) in channels.iter_mut().zip(&self.samples) {
            channel
                .samples
                .try_reserve_exact(BLOCK_SIZE)
                .map_err(|_| Error::Limit("JPEG quantization channel allocation"))?;
            channel.samples.extend_from_slice(samples);
        }
        Ok(channels)
    }

    /// DequantMatricesEncode for JPEG: custom flag, RAW DCT8, F16 denominator,
    /// caller-written Modular stream, then 16 Library(0) modes (zero index bits).
    /// The closure runs once, immediately after the denominator, without padding.
    /// Callback/writer errors propagate and may leave a partial header; no rollback.
    pub fn write_header(
        &self,
        writer: &mut BitWriter,
        write_modular: impl FnOnce(&mut BitWriter, &Self) -> Result<()>,
    ) -> Result<()> {
        writer.write(1, 0)?;
        writer.write(3, MODE_RAW)?;
        writer.write(16, u64::from(self.denominator_bits))?;
        write_modular(writer, self)?;
        for _ in 1..NUM_QUANT_TABLES {
            writer.write(3, MODE_LIBRARY)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(mut writer: BitWriter) -> (usize, Vec<u8>) {
        let bits = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        (bits, writer.into_bytes().unwrap())
    }

    fn read(bytes: &[u8], position: &mut usize, width: usize) -> u32 {
        let mut value = 0;
        for shift in 0..width {
            value |= u32::from((bytes[*position / 8] >> (*position % 8)) & 1) << shift;
            *position += 1;
        }
        value
    }

    #[test]
    fn dc_default_custom_and_truncation_have_known_wire_vectors() {
        let defaults = DcQuantization::from_inverse(DEFAULT_INVERSE_DC).unwrap();
        assert_eq!(defaults, DcQuantization::default());
        let mut writer = BitWriter::new(1);
        defaults.write(&mut writer).unwrap();
        assert_eq!(packed(writer), (1, vec![1]));

        let dc = DcQuantization::from_inverse([2040.0, 8.0, 2040.0 / 65535.0]).unwrap();
        let mut writer = BitWriter::new(7);
        dc.write(&mut writer).unwrap();
        assert_eq!(
            packed(writer),
            (49, vec![0x08, 0x58, 0x00, 0x98, 0x08, 0xd8, 0x00])
        );
        for (c, half) in [0x2c04, 0x4c00, 0x6c04].into_iter().enumerate() {
            assert_eq!(dc.dc[c], read_f16(half).unwrap() / 128.0);
            assert_eq!(dc.inverse_dc[c], 1.0f32 / dc.dc[c]);
        }
        assert_ne!(dc.inverse_dc[0], 2040.0);

        let truncated = DcQuantization::from_inverse([128.0 / 1.0009; 3]).unwrap();
        assert_eq!(truncated.dc, [1.0 / 128.0; 3]);
        let raw = RawDct8Quantization::new([[1; 64]; 3], 1.0009).unwrap();
        assert_eq!(raw.denominator_bits(), 0x3c00); // nearest would be 0x3c01
    }

    #[test]
    fn every_finite_half_matches_source_read_and_truncating_write() {
        for bits in 0..=u16::MAX {
            let exponent = (bits >> 10) & 31;
            if exponent == 31 {
                assert!(read_f16(bits).is_err());
                continue;
            }
            let mantissa = bits & 1023;
            let magnitude = if exponent == 0 {
                f32::from(mantissa) * 2.0f32.powi(-24)
            } else {
                (1.0 + f32::from(mantissa) / 1024.0) * 2.0f32.powi(i32::from(exponent) - 15)
            };
            let expected = if bits & 0x8000 != 0 {
                -magnitude
            } else {
                magnitude
            };
            let decoded = read_f16(bits).unwrap();
            assert_eq!(decoded.to_bits(), expected.to_bits());
            // F16Coder::Write deliberately erases the sign of tiny/zero values.
            let canonical = if bits == 0x8000 { 0 } else { bits };
            assert_eq!(f16_bits(decoded).unwrap(), canonical);
            if bits < 0x7c00 {
                let dc = decoded / 128.0;
                assert_eq!(dc_fields([dc; 3]).is_ok(), dc >= ALMOST_ZERO);
            }
        }
    }

    #[test]
    fn source_f16_write_upper_tail_and_post_roundtrip_lower_limits() {
        let upper = f32::from_bits(65536.0f32.to_bits() - 1);
        for value in [65504.0, 65505.0, upper] {
            assert_eq!(f16_bits(value).unwrap(), 0x7bff);
            assert_eq!(f16_bits(-value).unwrap(), 0xfbff);
            let raw = RawDct8Quantization::new([[1; 64]; 3], value).unwrap();
            assert_eq!(raw.denominator(), 65504.0);
            assert_eq!(dc_fields([value / 128.0; 3]).unwrap(), Some([0x7bff; 3]));
        }
        let smallest = 2.0f32.powi(-24);
        assert_eq!(
            RawDct8Quantization::new([[1; 64]; 3], smallest)
                .unwrap()
                .denominator_bits(),
            1
        );
        assert!(
            RawDct8Quantization::new([[1; 64]; 3], f32::from_bits(smallest.to_bits() - 1)).is_err()
        );
        assert!(dc_fields([read_f16(21).unwrap() / 128.0; 3]).is_err());
        assert!(dc_fields([read_f16(22).unwrap() / 128.0; 3]).is_ok());
        // Valid before rounding is not sufficient: this truncates to half 21.
        assert!(dc_fields([1e-8; 3]).is_err());
        for bad in [
            0.0,
            -0.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            65536.0,
            f32::MAX,
        ] {
            assert!(RawDct8Quantization::new([[1; 64]; 3], bad).is_err());
        }
        for bad in [
            0.0,
            -0.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::MIN_POSITIVE,
            f32::MAX,
        ] {
            assert!(DcQuantization::from_inverse([1.0, 1.0, bad]).is_err());
        }
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY, 512.0, 1e-8] {
            let mut writer = BitWriter::new(16);
            writer.write(3, 5).unwrap();
            assert!(
                write_dc_quantization(&mut writer, [DEFAULT_DC[0], DEFAULT_DC[1], bad]).is_err()
            );
            assert_eq!(writer.bits_written(), 3);
        }
    }

    #[test]
    fn quantizer_params_known_vectors_and_all_selector_edges() {
        let mut writer = BitWriter::new(4);
        QuantizerParams::jpeg().write(&mut writer).unwrap();
        assert_eq!(packed(writer), (25, vec![0xff, 0x7f, 0x07, 0x00]));
        let mut writer = BitWriter::new(3);
        QuantizerParams::default().write(&mut writer).unwrap();
        assert_eq!(packed(writer), (20, vec![0x00, 0x20, 0x00]));
        for (global, gs, gb, go) in [
            (1, 0, 11, 1),
            (2048, 0, 11, 1),
            (2049, 1, 11, 2049),
            (4096, 1, 11, 2049),
            (4097, 2, 12, 4097),
            (8192, 2, 12, 4097),
            (8193, 3, 16, 8193),
            (65536, 3, 16, 8193),
            (73728, 3, 16, 8193),
        ] {
            for (dc, ds, db, offset) in [
                (1, 1, 5, 1),
                (16, 0, 0, 16),
                (32, 1, 5, 1),
                (33, 2, 8, 1),
                (256, 2, 8, 1),
                (257, 3, 16, 1),
                (65536, 3, 16, 1),
            ] {
                let mut writer = BitWriter::new(16);
                writer.write(3, 5).unwrap();
                QuantizerParams {
                    global_scale: global,
                    quant_dc: dc,
                }
                .write(&mut writer)
                .unwrap();
                let (length, bytes) = packed(writer);
                let mut p = 0;
                assert_eq!(read(&bytes, &mut p, 3), 5);
                assert_eq!(read(&bytes, &mut p, 2), gs);
                assert_eq!(read(&bytes, &mut p, gb), global - go);
                assert_eq!(read(&bytes, &mut p, 2), ds);
                assert_eq!(read(&bytes, &mut p, db), dc - offset);
                assert_eq!(p, length);
            }
        }
        for (global_scale, quant_dc) in [(0, 1), (73729, 1), (u32::MAX, 1), (1, 0), (1, 65537)] {
            let mut writer = BitWriter::new(16);
            writer.write(3, 5).unwrap();
            assert!(
                QuantizerParams {
                    global_scale,
                    quant_dc
                }
                .write(&mut writer)
                .is_err()
            );
            assert_eq!(writer.bits_written(), 3);
        }
    }

    #[test]
    fn raw_header_has_one_raw_seventeen_modes_and_unpadded_callback() {
        let raw = RawDct8Quantization::from_source_shift([[1; 64]; 3], 0).unwrap();
        assert_eq!(raw.denominator_bits(), 0x1004);
        assert_ne!(raw.denominator(), 1.0f32 / 2040.0);
        let mut writer = BitWriter::new(10);
        let mut calls = 0;
        raw.write_header(&mut writer, |writer, table| {
            calls += 1;
            assert!(std::ptr::eq(table, &raw));
            assert_eq!(writer.bits_written(), 20);
            writer.write(5, 21)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(
            packed(writer),
            (73, vec![0x4e, 0x00, 0x51, 0x01, 0, 0, 0, 0, 0, 0])
        );
        for prefix in 0..8u8 {
            let mut writer = BitWriter::new(16);
            writer.write(prefix, (1u64 << prefix) - 1).unwrap();
            raw.write_header(&mut writer, |writer, _| {
                assert_eq!(writer.bits_written(), usize::from(prefix) + 20);
                writer.write(5, 21)?;
                Ok(())
            })
            .unwrap();
            let (length, bytes) = packed(writer);
            let mut p = usize::from(prefix);
            assert_eq!(read(&bytes, &mut p, 1), 0);
            assert_eq!(read(&bytes, &mut p, 3), 7);
            assert_eq!(read(&bytes, &mut p, 16), 0x1004);
            assert_eq!(read(&bytes, &mut p, 5), 21);
            for _ in 1..17 {
                assert_eq!(read(&bytes, &mut p, 3), 0);
            }
            assert_eq!(p, length); // no hidden Library(0) index bits
        }
    }

    #[test]
    fn raw_integer_channels_and_source_denominator_shift_are_preserved() {
        let values = [1, 255, 256, 65535, 65536, 16_777_217, i32::MAX];
        let samples =
            std::array::from_fn(|c| std::array::from_fn(|i| values[(i + c) % values.len()]));
        for shift in 0..=26 {
            let raw = RawDct8Quantization::from_source_shift(samples, shift).unwrap();
            assert_eq!(raw.samples(), &samples);
            assert_eq!(
                raw.denominator_bits(),
                f16_bits((1u32 << shift) as f32 * (1.0f32 / 2040.0)).unwrap()
            );
            let channels = raw.prepare_modular_channels().unwrap();
            for (channel, expected) in channels.iter().zip(samples) {
                assert_eq!(
                    (
                        channel.width,
                        channel.height,
                        channel.hshift,
                        channel.vshift
                    ),
                    (8, 8, 0, 0)
                );
                assert_eq!(channel.samples, expected);
            }
        }
        assert_eq!(QUANT_TABLE_BITS, 8);
        assert_eq!(DCT8_QUANT_TABLE_INDEX, 0);
        for shift in [27, 30, 31, 32, u32::MAX] {
            assert!(RawDct8Quantization::from_source_shift(samples, shift).is_err());
        }
        for bad in [0, -1, i32::MIN] {
            let mut invalid = samples;
            invalid[2][63] = bad;
            assert!(RawDct8Quantization::from_source_shift(invalid, 0).is_err());
        }
    }

    #[test]
    fn jpeg_table_precision_transpose_and_layout_factories_agree() {
        use crate::jpeg::data::{CodingProcess, Component, ComponentScanInfo, JpegData, ScanInfo};
        use crate::jpeg::transcode_layout::TranscodeLayout;
        let table = QuantTable {
            index: 3,
            values: std::array::from_fn(|i| i as i32 + 1),
            ..QuantTable::default()
        };
        let raw = RawDct8Quantization::from_jpeg_tables([&table; 3]).unwrap();
        for channel in raw.samples() {
            for (i, &value) in channel.iter().enumerate() {
                assert_eq!(value, table.values[8 * (i % 8) + i / 8]);
            }
        }
        let jpeg = JpegData {
            width: 8,
            height: 8,
            precision: 8,
            coding_process: Some(CodingProcess::Baseline),
            components: vec![Component {
                id: 1,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_index: 3,
                width_in_blocks: 1,
                height_in_blocks: 1,
                coefficients: vec![0; 64],
            }],
            quant: vec![table.clone()],
            scan_info: vec![ScanInfo {
                components: vec![ComponentScanInfo {
                    component_index: 0,
                    ..ComponentScanInfo::default()
                }],
                ss: 0,
                se: 63,
                quant_table_indices: [None, None, None, Some(0)],
                ..ScanInfo::default()
            }],
            ..JpegData::default()
        };
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        let channels = layout.planes().each_ref().map(|p| p.quantization());
        assert_eq!(RawDct8Quantization::from_layout(channels).unwrap(), raw);
        assert_eq!(
            DcQuantization::from_layout(channels).unwrap(),
            DcQuantization::from_jpeg_tables([&table; 3]).unwrap()
        );
        assert_ne!(
            DcQuantization::from_layout(channels).unwrap().dc[0],
            channels[0].inverse_dc_quantization()
        );
        let wide = QuantTable {
            precision: 1,
            values: [65535; 64],
            ..table.clone()
        };
        assert_eq!(
            RawDct8Quantization::from_jpeg_tables([&wide; 3])
                .unwrap()
                .samples(),
            &[[65535; 64]; 3]
        );
        for (precision, value) in [(2, 1), (0, 256), (1, 65536), (1, 0), (1, -1)] {
            let bad = QuantTable {
                precision,
                values: [value; 64],
                ..table.clone()
            };
            assert!(RawDct8Quantization::from_jpeg_tables([&bad; 3]).is_err());
            assert!(DcQuantization::from_jpeg_tables([&bad; 3]).is_err());
        }
    }

    #[test]
    fn raw_callback_and_writer_errors_do_not_emit_remaining_modes() {
        let raw = RawDct8Quantization::from_source_shift([[1; 64]; 3], 0).unwrap();
        let mut writer = BitWriter::new(16);
        let error = raw
            .write_header(&mut writer, |writer, _| {
                writer.write(3, 5)?;
                Err(Error::Encoding("caller stream failed".into()))
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "caller stream failed");
        assert_eq!(writer.bits_written(), 23);
        let mut writer = BitWriter::new(2);
        let mut called = false;
        assert!(
            raw.write_header(&mut writer, |_, _| {
                called = true;
                Ok(())
            })
            .is_err()
        );
        assert!(!called);
        assert_eq!(writer.bits_written(), 4);
    }

    #[test]
    fn caller_prepared_modular_stream_is_embedded_without_padding() {
        use crate::entropy::stream::Coding;
        use crate::modular::encode::{FixedTree, ModularOptions, write_prepared_stream};
        let raw = RawDct8Quantization::from_source_shift(
            std::array::from_fn(|c| {
                std::array::from_fn(|i| 1 + ((i * 997 + c * 101) % 65535) as i32)
            }),
            0,
        )
        .unwrap();
        let channels = raw.prepare_modular_channels().unwrap();
        for entropy in [Coding::Prefix, Coding::Ans] {
            let options = ModularOptions {
                entropy,
                fixed_tree: Some(FixedTree::GradientLeaf),
                rct: None,
                lz77: false,
                ..ModularOptions::default()
            };
            let mut standalone = BitWriter::new(65536);
            write_prepared_stream(
                &channels,
                &[],
                &options,
                &mut standalone,
                0,
                QUANT_TABLE_BITS,
            )
            .unwrap();
            let (stream_bits, stream_bytes) = packed(standalone);
            let mut writer = BitWriter::new(65536);
            writer.write(3, 5).unwrap();
            raw.write_header(&mut writer, |writer, table| {
                assert_eq!(writer.bits_written(), 23);
                let planes = table.prepare_modular_channels()?;
                write_prepared_stream(&planes, &[], &options, writer, 0, QUANT_TABLE_BITS)
            })
            .unwrap();
            let (bits, bytes) = packed(writer);
            assert_eq!(bits, 3 + 68 + stream_bits);
            let mut p = 0;
            assert_eq!(read(&bytes, &mut p, 3), 5);
            assert_eq!(read(&bytes, &mut p, 1), 0);
            assert_eq!(read(&bytes, &mut p, 3), 7);
            assert_eq!(read(&bytes, &mut p, 16), 0x1004);
            for i in 0..stream_bits {
                assert_eq!(
                    read(&bytes, &mut p, 1),
                    u32::from((stream_bytes[i / 8] >> (i % 8)) & 1)
                );
            }
            for _ in 1..17 {
                assert_eq!(read(&bytes, &mut p, 3), 0);
            }
            assert_eq!(p, bits);
        }
    }
}
