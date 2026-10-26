// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! JPEGData::VisitFields from lib/jxl/jpeg/jpeg_data.cc and payload ordering
//! from enc_jpeg_data.cc, frozen at b866b3defec13494bfbaf705db93581cda118e37.
//!
//! This is reconstruction metadata, not JXL coefficient/frame encoding. The
//! default serializer keeps complete Unknown APP bytes. Explicit matching output
//! profile/box sources enable source ICC/Exif/XMP payload deduplication.
//! The caller supplies Brotli compression of `payload` after `header`.

use std::fmt;

use crate::bit_writer::{BitWriter, BitWriterError};

use super::data::{AppMarkerType, HuffmanCode, JpegData, JpegRecord, MarkerSegment};
use super::reconstruction_validation::validate;
use super::scan_writer::{ReconstructionError, WriteLimits};

const MAX_MARKERS: usize = 16_384;
const MAX_HUFFMAN_TABLES: usize = 89;
const MAX_INTERMARKER: usize = 65_535;
const MAX_TAIL: usize = 4_260_096;
const MAX_BLOCK_INDEX: u32 = 3 << 26;
const ICC_TAG: &[u8; 12] = b"ICC_PROFILE\0";
const EXIF_TAG: &[u8; 6] = b"Exif\0\0";
const XMP_TAG: &[u8; 29] = b"http://ns.adobe.com/xap/1.0/\0";

/// Borrowed metadata that the caller will actually emit in the same JXL output.
/// Missing or byte-mismatched sources retain the corresponding Unknown APPs.
/// This is not permission to change JPEG records or their existing annotations.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReconstructionMetadataSources<'a> {
    /// The explicit original-profile ICC bytes, e.g. ColorEncoding::icc_profile().
    /// Do not supply a synthesized profile for an implicit color encoding.
    pub icc_profile: Option<&'a [u8]>,
    /// Complete emitted Exif box payload: four zero offset bytes, then TIFF.
    pub exif_box_payload: Option<&'a [u8]>,
    /// Complete emitted `xml ` box payload, without the JPEG XMP signature.
    pub xmp_box_payload: Option<&'a [u8]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedMetadata {
    /// VisitFields bitstream, LSB-first and zero-padded to a byte boundary.
    pub header: Vec<u8>,
    /// Uncompressed Unknown APP, COM, inter-marker and tail data, in that order.
    pub payload: Vec<u8>,
}

#[derive(Debug)]
pub enum MetadataError {
    Invalid(&'static str),
    Unsupported(&'static str),
    Limit(&'static str),
    Validation(ReconstructionError),
    Bitstream(BitWriterError),
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid JPEG reconstruction metadata: {reason}"),
            Self::Unsupported(reason) => {
                write!(f, "unsupported JPEG reconstruction metadata: {reason}")
            }
            Self::Limit(resource) => write!(f, "JPEG reconstruction metadata limit: {resource}"),
            Self::Validation(error) => error.fmt(f),
            Self::Bitstream(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for MetadataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            Self::Bitstream(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitWriterError> for MetadataError {
    fn from(error: BitWriterError) -> Self {
        Self::Bitstream(error)
    }
}

type Result<T> = std::result::Result<T, MetadataError>;

// A zero-bit choice is Val(offset); all others are BitsOffset(bits, offset).
#[derive(Clone, Copy)]
struct Choice {
    bits: u8,
    offset: u32,
}

const fn val(offset: u32) -> Choice {
    Choice { bits: 0, offset }
}
const fn bits(bits: u8, offset: u32) -> Choice {
    Choice { bits, offset }
}

const ONE_TO_FOUR: [Choice; 4] = [val(1), val(2), val(3), val(4)];
const APP_TYPE: [Choice; 4] = [val(0), val(1), bits(1, 2), bits(2, 4)];
const HUFFMAN_COUNT: [Choice; 4] = [val(4), bits(3, 2), bits(4, 10), bits(6, 26)];
const CODE_COUNT: [Choice; 4] = [val(0), val(1), bits(3, 2), bits(8, 0)];
const CODE_VALUE: [Choice; 4] = [bits(2, 0), bits(2, 4), bits(4, 8), bits(8, 1)];
const LAST_PASS: [Choice; 4] = [val(0), val(1), val(2), bits(3, 3)];
const HINT_COUNT: [Choice; 4] = [val(0), bits(2, 1), bits(4, 4), bits(16, 20)];
const BLOCK_DELTA: [Choice; 4] = [val(0), bits(3, 1), bits(5, 9), bits(28, 41)];
const ZERO_RUNS: [Choice; 4] = [val(1), bits(2, 2), bits(4, 5), bits(8, 20)];
const TAIL_LENGTH: [Choice; 4] = [val(0), bits(8, 1), bits(16, 257), bits(22, 65_793)];

// U32Coder::ChooseSelector uses the shortest choice, then its first occurrence.
fn choose(choices: [Choice; 4], value: u32) -> Result<(usize, Choice)> {
    let mut best: Option<(usize, Choice)> = None;
    for (index, choice) in choices.into_iter().enumerate() {
        if value < choice.offset || u64::from(value - choice.offset) >= (1u64 << choice.bits) {
            continue;
        }
        if choice.bits == 0 {
            return Ok((index, choice));
        }
        if best.is_none_or(|(_, previous)| choice.bits < previous.bits) {
            best = Some((index, choice));
        }
    }
    best.ok_or(MetadataError::Unsupported(
        "value has no source U32 encoding",
    ))
}

fn u32_field(writer: &mut BitWriter, choices: [Choice; 4], value: u32) -> Result<()> {
    let (selector, choice) = choose(choices, value)?;
    writer.write(2, selector as u64)?;
    writer.write(choice.bits, u64::from(value - choice.offset))?;
    Ok(())
}

fn length(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| MetadataError::Limit("32-bit field length"))
}

fn push<T>(values: &mut Vec<T>, value: T, limit: usize) -> Result<()> {
    if values.len() >= limit {
        return Err(MetadataError::Limit("wire model entries"));
    }
    values
        .try_reserve(1)
        .map_err(|_| MetadataError::Limit("wire model allocation"))?;
    values.push(value);
    Ok(())
}

struct WireModel<'a> {
    markers: Vec<u8>,
    app_types: Vec<AppMarkerType>,
    inter_lengths: Vec<usize>,
    // None is the source's synthetic all-zero table for an empty DHT marker.
    huffman: Vec<Option<&'a HuffmanCode>>,
    component_quant: [usize; 3],
    restart_interval: Option<u32>,
    payload_len: usize,
}

impl WireModel<'_> {
    fn add_payload(&mut self, count: usize, max_bytes: usize) -> Result<()> {
        self.payload_len = self
            .payload_len
            .checked_add(count)
            .filter(|&n| n <= max_bytes)
            .ok_or(MetadataError::Limit("combined header and payload bytes"))?;
        Ok(())
    }

    fn intermarker(&mut self, count: usize, max_bytes: usize) -> Result<()> {
        if count == 0 {
            return Err(MetadataError::Invalid("empty fill or inter-marker record"));
        }
        // FindNextMarker groups all consecutive skipped bytes, including FF
        // fill. Preserve that grouping and reject rather than truncate 16 bits.
        if self.markers.last() == Some(&0xff) {
            let previous = self
                .inter_lengths
                .last_mut()
                .expect("fake marker has a length");
            *previous = previous
                .checked_add(count)
                .filter(|&n| n <= MAX_INTERMARKER)
                .ok_or(MetadataError::Limit(
                    "consecutive inter-marker bytes exceed 65535",
                ))?;
        } else {
            if count > MAX_INTERMARKER {
                return Err(MetadataError::Limit("inter-marker bytes exceed 65535"));
            }
            push(&mut self.markers, 0xff, MAX_MARKERS)?;
            push(&mut self.inter_lengths, count, MAX_MARKERS)?;
        }
        self.add_payload(count, max_bytes)
    }
}

fn same_saved_marker(saved: Option<&Vec<u8>>, segment: &MarkerSegment) -> bool {
    saved.is_some_and(|bytes| bytes.first() == Some(&segment.marker) && bytes[1..] == segment.data)
}

fn classify_apps(
    jpeg: &JpegData,
    sources: ReconstructionMetadataSources<'_>,
) -> Result<Vec<AppMarkerType>> {
    if jpeg.app_data.len() > MAX_MARKERS {
        return Err(MetadataError::Limit("APP marker count"));
    }
    let mut kinds = Vec::new();
    kinds
        .try_reserve_exact(jpeg.app_data.len())
        .map_err(|_| MetadataError::Limit("APP classification allocation"))?;
    kinds.resize(jpeg.app_data.len(), AppMarkerType::Unknown);
    // Source DetectIccProfile selects an ordered subsequence, not all APP2s
    // with an ICC signature. DecodeJPEGData regenerates only its ordinal/count.
    if let Some(profile) = sources.icc_profile.filter(|p| !p.is_empty()) {
        let mut count = 0usize;
        let mut expected = 0usize;
        let mut position = 0usize;
        let mut matches_profile = true;
        for (app, kind) in jpeg.app_data.iter().zip(&mut kinds) {
            if app.first() != Some(&0xe2) {
                continue;
            }
            let Some(payload) = app
                .get(3..)
                .and_then(|p| p.strip_prefix(ICC_TAG.as_slice()))
            else {
                continue;
            };
            let Some((&ordinal, payload)) = payload.split_first() else {
                continue;
            };
            let Some((&total, chunk)) = payload.split_first() else {
                continue;
            };
            if usize::from(ordinal) != count + 1 {
                continue;
            }
            if expected == 0 {
                expected = usize::from(total);
            }
            if expected != usize::from(total) {
                continue;
            }
            count += 1;
            *kind = AppMarkerType::Icc;
            // A zero total can slip through the source's sentinel handling,
            // but the decoder would replace it with the final nonzero count.
            matches_profile &= total != 0;
            let end = position
                .checked_add(chunk.len())
                .ok_or(MetadataError::Limit("ICC chunk lengths"))?;
            matches_profile &= profile.get(position..end) == Some(chunk);
            position = end;
        }
        // Source fails incomplete selections. The archival adapter instead
        // rolls them back atomically, retaining every original payload byte.
        if count != expected || !matches_profile || position != profile.len() {
            kinds.fill(AppMarkerType::Unknown);
        }
    }
    let mut have_exif = false;
    let mut have_xmp = false;
    for (app, kind) in jpeg.app_data.iter().zip(&mut kinds) {
        if app.first() != Some(&0xe1) {
            continue;
        }
        let Some(payload) = app.get(3..) else {
            continue;
        };
        if !have_exif
            && let Some(tiff) = payload
                .strip_prefix(EXIF_TAG.as_slice())
                .filter(|p| !p.is_empty())
        {
            have_exif = true;
            if sources
                .exif_box_payload
                .is_some_and(|p| p.get(..4) == Some(&[0; 4]) && p.get(4..) == Some(tiff))
            {
                *kind = AppMarkerType::Exif;
            }
        }
        if !have_xmp && let Some(xmp) = payload.strip_prefix(XMP_TAG.as_slice()) {
            // Unlike SetBlobsFromJpegData, DetectBlobs accepts empty XMP. Do
            // not skip that first marker to classify a later nonempty blob.
            have_xmp = true;
            if sources.xmp_box_payload == Some(xmp) {
                *kind = AppMarkerType::Xmp;
            }
        }
    }
    Ok(kinds)
}

fn adapt<'a>(
    jpeg: &'a JpegData,
    max_bytes: usize,
    sources: ReconstructionMetadataSources<'_>,
) -> Result<WireModel<'a>> {
    if max_bytes == 0 {
        return Err(MetadataError::Limit("combined header and payload bytes"));
    }
    if !matches!(jpeg.components.len(), 1 | 3) {
        return Err(MetadataError::Unsupported(
            "source supports only one or three components",
        ));
    }
    if !(1..=3).contains(&jpeg.quant.len()) {
        return Err(MetadataError::Unsupported(
            "source requires one to three quantization tables",
        ));
    }
    let limits = WriteLimits::default();
    if jpeg.records.len() > limits.structural.max_records
        || jpeg.scan_info.len() > limits.structural.max_scans
        || jpeg.huffman_code.len() > MAX_HUFFMAN_TABLES
        || jpeg.padding_bits.len() > limits.max_padding_bits
        || jpeg.padding_bits.len() >= 1 << 24
        || jpeg.tail_data.len() > MAX_TAIL
    {
        return Err(MetadataError::Limit("source metadata fields"));
    }
    if jpeg.has_zero_padding_bit != jpeg.padding_bits.contains(&0) {
        return Err(MetadataError::Invalid(
            "zero-padding flag disagrees with padding bits",
        ));
    }
    let mut model = WireModel {
        markers: Vec::new(),
        app_types: classify_apps(jpeg, sources)?,
        inter_lengths: Vec::new(),
        huffman: Vec::new(),
        component_quant: [usize::MAX; 3],
        restart_interval: None,
        payload_len: 0,
    };
    let mut real_markers = 0;
    let mut app = 0;
    let mut com = 0;
    let mut inter = 0;
    let mut huff = 0;
    let mut scan = 0;
    let mut pending_scan = false;
    let mut eoi = false;
    let mut tail = false;
    for (record_index, record) in jpeg.records.iter().enumerate() {
        if record_index == 0
            && !matches!(record, JpegRecord::Marker(s) if s.marker == 0xd8 && s.data.is_empty())
        {
            return Err(MetadataError::Invalid("SOI must be the first record"));
        }
        if pending_scan && !matches!(record, JpegRecord::Entropy(_)) {
            return Err(MetadataError::Unsupported(
                "SOS must be immediately followed by its entropy descriptor",
            ));
        }
        if eoi && !matches!(record, JpegRecord::Tail { .. }) {
            return Err(MetadataError::Unsupported("only tail data may follow EOI"));
        }
        match record {
            JpegRecord::Marker(segment) => {
                if jpeg.marker_order.get(real_markers) != Some(&segment.marker) {
                    return Err(MetadataError::Invalid(
                        "marker order disagrees with records",
                    ));
                }
                real_markers += 1;
                if segment.marker == 0xd8 {
                    if record_index != 0 {
                        return Err(MetadataError::Unsupported("duplicate SOI"));
                    }
                    continue;
                }
                if !matches!(segment.marker, 0xc0..=0xc2 | 0xc4 | 0xd9 | 0xda | 0xdb | 0xdd | 0xe0..=0xef | 0xfe)
                {
                    return Err(MetadataError::Unsupported(
                        "marker cannot be reconstructed by the source writer",
                    ));
                }
                push(&mut model.markers, segment.marker, MAX_MARKERS)?;
                match segment.marker {
                    0xe0..=0xef => {
                        if !same_saved_marker(jpeg.app_data.get(app), segment) {
                            return Err(MetadataError::Invalid(
                                "APP data disagrees with marker record",
                            ));
                        }
                        if model.app_types[app] == AppMarkerType::Unknown {
                            model.add_payload(jpeg.app_data[app].len(), max_bytes)?;
                        }
                        app += 1;
                    }
                    0xfe => {
                        if !same_saved_marker(jpeg.com_data.get(com), segment) {
                            return Err(MetadataError::Invalid(
                                "COM data disagrees with marker record",
                            ));
                        }
                        model.add_payload(jpeg.com_data[com].len(), max_bytes)?;
                        com += 1;
                    }
                    0xc4 => {
                        if segment.data == [0, 2] {
                            push(&mut model.huffman, None, MAX_HUFFMAN_TABLES)?;
                        } else {
                            loop {
                                let table = jpeg
                                    .huffman_code
                                    .get(huff)
                                    .ok_or(MetadataError::Invalid("DHT table count"))?;
                                push(&mut model.huffman, Some(table), MAX_HUFFMAN_TABLES)?;
                                huff += 1;
                                if table.is_last {
                                    break;
                                }
                            }
                        }
                    }
                    0xdd => {
                        if segment.data.len() != 4 {
                            return Err(MetadataError::Invalid("DRI length"));
                        }
                        let interval =
                            u32::from(u16::from_be_bytes([segment.data[2], segment.data[3]]));
                        if model
                            .restart_interval
                            .is_some_and(|previous| previous != interval)
                        {
                            return Err(MetadataError::Unsupported(
                                "DRI redefinition requires more than one global interval",
                            ));
                        }
                        model.restart_interval = Some(interval);
                    }
                    0xda => pending_scan = true,
                    0xd9 => eoi = true,
                    _ => {}
                }
            }
            JpegRecord::Entropy(entropy) => {
                if !pending_scan || entropy.scan_index != scan {
                    return Err(MetadataError::Invalid(
                        "entropy descriptor does not follow its SOS",
                    ));
                }
                if entropy
                    .restart_markers
                    .iter()
                    .any(|restart| restart.fill_count != 0)
                {
                    return Err(MetadataError::Unsupported(
                        "restart marker fill has no standard jbrd representation",
                    ));
                }
                scan += 1;
                pending_scan = false;
            }
            JpegRecord::Fill { count, .. } => model.intermarker(*count, max_bytes)?,
            JpegRecord::InterMarker { index, .. } => {
                if *index != inter {
                    return Err(MetadataError::Invalid("inter-marker record order"));
                }
                let bytes = jpeg
                    .inter_marker_data
                    .get(*index)
                    .ok_or(MetadataError::Invalid("inter-marker index"))?;
                model.intermarker(bytes.len(), max_bytes)?;
                inter += 1;
            }
            JpegRecord::Tail { .. } => {
                if !eoi || tail || jpeg.tail_data.is_empty() {
                    return Err(MetadataError::Invalid("tail record order or length"));
                }
                tail = true;
                model.add_payload(jpeg.tail_data.len(), max_bytes)?;
            }
        }
    }
    if !eoi
        || pending_scan
        || scan == 0
        || scan != jpeg.scan_info.len()
        || real_markers != jpeg.marker_order.len()
        || app != jpeg.app_data.len()
        || com != jpeg.com_data.len()
        || inter != jpeg.inter_marker_data.len()
        || huff != jpeg.huffman_code.len()
        || tail == jpeg.tail_data.is_empty()
        || jpeg.app_marker_type.len() != app
    {
        return Err(MetadataError::Invalid(
            "incomplete or unmatched metadata records",
        ));
    }
    if jpeg.restart_interval != model.restart_interval.unwrap_or(0) {
        return Err(MetadataError::Invalid(
            "global restart interval disagrees with DRI",
        ));
    }
    choose(HUFFMAN_COUNT, length(model.huffman.len())?)?;
    // This reparse is bounded by WriteLimits, independently of the encoded-byte
    // cap: raw DQT/DHT records can be larger than their compact metadata header.
    validate(jpeg, &limits).map_err(MetadataError::Validation)?;
    adapt_quantization(jpeg, &mut model)?;
    // Progressive DC refinement carries raw correction bits, not Huffman codes.
    // Preserve its original selector even when that unused table is undefined.
    for scan in &jpeg.scan_info {
        for component in &scan.components {
            if scan.ss == 0
                && scan.ah == 0
                && scan.dc_huffman_indices[component.dc_table_index as usize].is_none()
            {
                return Err(MetadataError::Unsupported(
                    "initial DC scan requires a Huffman table",
                ));
            }
        }
    }
    Ok(model)
}

fn adapt_quantization(jpeg: &JpegData, model: &mut WireModel<'_>) -> Result<()> {
    // One component quant_idx is global on the wire. Identical redefinitions
    // can share its first active vector index; differing values cannot.
    for (component_index, component) in jpeg.components.iter().enumerate() {
        for scan in &jpeg.scan_info {
            if !scan
                .components
                .iter()
                .any(|sc| sc.component_index == component_index)
            {
                continue;
            }
            let active = scan.quant_table_indices[component.quant_index as usize]
                .ok_or(MetadataError::Invalid("missing active quantization table"))?;
            let chosen = &mut model.component_quant[component_index];
            if *chosen == usize::MAX {
                *chosen = active;
            }
            if jpeg.quant[*chosen].values != jpeg.quant[active].values {
                return Err(MetadataError::Unsupported(
                    "component quantization changes between scans",
                ));
            }
        }
        if model.component_quant[component_index] == usize::MAX {
            return Err(MetadataError::Invalid(
                "component has no quantization mapping",
            ));
        }
    }
    for (index, table) in jpeg.quant.iter().enumerate() {
        if model.component_quant[..jpeg.components.len()].contains(&index) {
            continue;
        }
        if index == 0 || table.values != jpeg.quant[index - 1].values {
            return Err(MetadataError::Unsupported(
                "unused quantization table must copy its predecessor",
            ));
        }
    }
    Ok(())
}

fn write_huffman(writer: &mut BitWriter, table: Option<&HuffmanCode>) -> Result<()> {
    let empty = HuffmanCode::default();
    let source = table.unwrap_or(&empty);
    writer.write(1, u64::from(source.slot_id >> 4))?;
    writer.write(2, u64::from(source.slot_id & 15))?;
    writer.write(1, u64::from(source.is_last))?;
    let last_length = source.counts.iter().rposition(|&count| count != 0);
    let mut total = 0;
    for (index, &count) in source.counts.iter().enumerate() {
        total += count as usize;
        let count = count + u32::from(table.is_some() && Some(index) == last_length);
        u32_field(writer, CODE_COUNT, count)?;
    }
    if table.is_some() {
        for &value in &source.values[..total] {
            u32_field(writer, CODE_VALUE, value)?;
        }
        u32_field(writer, CODE_VALUE, 256)?;
    }
    Ok(())
}

fn write_block_delta(writer: &mut BitWriter, previous: &mut Option<u32>, index: u32) -> Result<()> {
    if index >= MAX_BLOCK_INDEX {
        return Err(MetadataError::Unsupported("source block index limit"));
    }
    let base = previous.map_or(0, |value| value + 1);
    let delta = index
        .checked_sub(base)
        .ok_or(MetadataError::Invalid("non-increasing block hints"))?;
    u32_field(writer, BLOCK_DELTA, delta)?;
    *previous = Some(index);
    Ok(())
}

fn write_header(jpeg: &JpegData, model: &WireModel<'_>, max_bytes: usize) -> Result<Vec<u8>> {
    let mut writer = BitWriter::new(max_bytes);
    writer.write(1, u64::from(jpeg.components.len() == 1))?;
    for &marker in &model.markers {
        writer.write(6, u64::from(marker - 0xc0))?;
    }
    for (app, &kind) in jpeg.app_data.iter().zip(&model.app_types) {
        u32_field(&mut writer, APP_TYPE, kind as u32)?;
        writer.write(16, u64::from(length(app.len() - 1)?))?;
    }
    for com in &jpeg.com_data {
        writer.write(16, u64::from(length(com.len() - 1)?))?;
    }
    u32_field(&mut writer, ONE_TO_FOUR, length(jpeg.quant.len())?)?;
    for table in &jpeg.quant {
        writer.write(1, u64::from(table.precision))?;
        writer.write(2, u64::from(table.index))?;
        writer.write(1, u64::from(table.is_last))?;
    }
    let component_type = if jpeg.components.len() == 1 && jpeg.components[0].id == 1 {
        0
    } else if jpeg.components.len() == 3 && jpeg.components.iter().map(|c| c.id).eq([1, 2, 3]) {
        1
    } else if jpeg.components.len() == 3 && jpeg.components.iter().map(|c| c.id).eq([82, 71, 66]) {
        2
    } else {
        3
    };
    writer.write(2, component_type)?;
    if component_type == 3 {
        u32_field(&mut writer, ONE_TO_FOUR, length(jpeg.components.len())?)?;
        for component in &jpeg.components {
            writer.write(8, u64::from(component.id))?;
        }
    }
    for &index in &model.component_quant[..jpeg.components.len()] {
        writer.write(2, index as u64)?;
    }
    u32_field(&mut writer, HUFFMAN_COUNT, length(model.huffman.len())?)?;
    for &table in &model.huffman {
        write_huffman(&mut writer, table)?;
    }
    for scan in &jpeg.scan_info {
        u32_field(&mut writer, ONE_TO_FOUR, length(scan.components.len())?)?;
        writer.write(6, u64::from(scan.ss))?;
        writer.write(6, u64::from(scan.se))?;
        writer.write(4, u64::from(scan.al))?;
        writer.write(4, u64::from(scan.ah))?;
        for component in &scan.components {
            writer.write(2, component.component_index as u64)?;
            writer.write(2, u64::from(component.ac_table_index))?;
            writer.write(2, u64::from(component.dc_table_index))?;
        }
        u32_field(&mut writer, LAST_PASS, scan.last_needed_pass)?;
    }
    if let Some(interval) = model.restart_interval {
        writer.write(16, u64::from(interval))?;
    }
    for scan in &jpeg.scan_info {
        u32_field(&mut writer, HINT_COUNT, length(scan.reset_points.len())?)?;
        let mut previous = None;
        for &index in &scan.reset_points {
            write_block_delta(&mut writer, &mut previous, index)?;
        }
        u32_field(&mut writer, HINT_COUNT, length(scan.extra_zero_runs.len())?)?;
        previous = None;
        for hint in &scan.extra_zero_runs {
            u32_field(&mut writer, ZERO_RUNS, hint.count)?;
            write_block_delta(&mut writer, &mut previous, hint.block_index)?;
        }
    }
    for &count in &model.inter_lengths {
        writer.write(16, count as u64)?;
    }
    u32_field(&mut writer, TAIL_LENGTH, length(jpeg.tail_data.len())?)?;
    writer.write(1, u64::from(jpeg.has_zero_padding_bit))?;
    if jpeg.has_zero_padding_bit {
        writer.write(24, jpeg.padding_bits.len() as u64)?;
        for &bit in &jpeg.padding_bits {
            writer.write(1, u64::from(bit))?;
        }
    }
    writer.zero_pad_to_byte()?;
    Ok(writer.into_bytes()?)
}

/// Serialize only the reconstruction metadata prerequisite. `max_bytes` bounds
/// the combined header and raw payload lengths; internal validation also uses
/// the fixed default WriteLimits. Requires decoded coefficient buffers, but
/// never reads or clones coefficients or archived entropy payloads. All output
/// is local until success. No Brotli, JXL frames, container boxes or I/O occur.
pub fn serialize_reconstruction_metadata(
    jpeg: &JpegData,
    max_bytes: usize,
) -> Result<EncodedMetadata> {
    serialize_reconstruction_metadata_with_sources(
        jpeg,
        max_bytes,
        ReconstructionMetadataSources::default(),
    )
}

/// Deduplicate source-recognized ICC/Exif/XMP APP payloads against metadata that
/// will actually be emitted. The old serializer remains the all-Unknown form.
/// Selected ICC chunks must reconstruct exactly from the supplied profile;
/// Exif/XMP must match the source's first eligible blob. Incomplete or mismatched
/// selections fall back to Unknown, not an error or partial ICC deduplication.
/// Existing JpegData::app_marker_type annotations are ignored and never changed.
///
/// `max_bytes` bounds header plus retained raw payload, excluding the supplied
/// sources and their container framing. Validation and allocation bounds are
/// unchanged; invalid APP envelopes/record mirrors still fail even if selected.
/// The caller must retain these exact sources in the final output and enforce
/// its total output limit. No JUMBF extraction or metadata rewriting is performed.
pub fn serialize_reconstruction_metadata_with_sources(
    jpeg: &JpegData,
    max_bytes: usize,
    sources: ReconstructionMetadataSources<'_>,
) -> Result<EncodedMetadata> {
    let model = adapt(jpeg, max_bytes, sources)?;
    let header = write_header(jpeg, &model, max_bytes - model.payload_len)?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(model.payload_len)
        .map_err(|_| MetadataError::Limit("payload allocation"))?;
    for (app, &kind) in jpeg.app_data.iter().zip(&model.app_types) {
        if kind == AppMarkerType::Unknown {
            payload.extend_from_slice(app);
        }
    }
    for com in &jpeg.com_data {
        payload.extend_from_slice(com);
    }
    for record in &jpeg.records {
        match record {
            JpegRecord::Fill { count, .. } => payload.resize(payload.len() + count, 0xff),
            JpegRecord::InterMarker { index, .. } => {
                payload.extend_from_slice(&jpeg.inter_marker_data[*index])
            }
            _ => {}
        }
    }
    payload.extend_from_slice(&jpeg.tail_data);
    debug_assert_eq!(payload.len(), model.payload_len);
    Ok(EncodedMetadata { header, payload })
}

#[cfg(test)]
#[path = "reconstruction_metadata_tests.rs"]
mod tests;
