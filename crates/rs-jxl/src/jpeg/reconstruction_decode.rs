// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Bounded inverse of JPEGData::VisitFields and DecodeJPEGData in
//! lib/jxl/jpeg/{jpeg_data.cc,dec_jpeg_data.cc}, frozen at
//! b866b3defec13494bfbaf705db93581cda118e37. Marker generation follows
//! dec_jpeg_data_writer.cc. No pixel conversion or archived JPEG input is used.

use std::fmt;

use super::data::{
    AppMarkerType, ComponentScanInfo, ExtraZeroRun, HuffmanCode, JpegData, JpegError, JpegRecord,
    NATURAL_ORDER, QuantTable, RestartMarker, ScanInfo,
};
use super::huffman::HuffmanTable;
use super::reader::read_jpeg_with_limits;
use super::reconstruction_metadata::ReconstructionMetadataSources;
use super::reconstruction_validation::validate;
use super::scan_progression::ScanProgression;
use super::scan_writer::{ReconstructionError, WriteLimits, reconstruct_from_coefficients};

/// Independent limits for compressed input, expanded metadata and DCT storage.
/// Brotli uses a standard window (at most 24 bits); its utility-crate workspace
/// is additional to these limits. Writer/structural limits apply independently.
#[derive(Clone, Debug)]
pub struct ReconstructionDecodeLimits {
    pub max_jbrd_bytes: usize,
    /// Header plus expanded APP/COM/inter-marker/tail bytes, including deduped APPs.
    pub max_metadata_bytes: usize,
    /// Total i16 values across all components, including MCU edge padding.
    pub max_coefficient_values: usize,
    pub write: WriteLimits,
}

impl Default for ReconstructionDecodeLimits {
    fn default() -> Self {
        Self {
            max_jbrd_bytes: 64 * 1024 * 1024,
            max_metadata_bytes: 64 * 1024 * 1024,
            max_coefficient_values: 128 * 1024 * 1024,
            write: WriteLimits::default(),
        }
    }
}

/// Decoder handoff, in original JPEG component order, not JXL internal
/// channel order. Supply the original JPEG sampling factors, not upsampled pixels.
#[derive(Debug)]
pub struct CapturedJpegCoefficients {
    pub width: u32,
    pub height: u32,
    pub components: Vec<CapturedJpegComponent>,
}

#[derive(Debug)]
pub struct CapturedJpegComponent {
    pub h_samp_factor: u32,
    pub v_samp_factor: u32,
    /// MCU-padded dimensions, NOT merely ceil(component pixel dimensions / 8).
    pub width_in_blocks: u32,
    pub height_in_blocks: u32,
    /// Final quantized coefficients, block-major, row-major blocks, natural 8x8
    /// coefficient order. Include DC and edge blocks; undo JXL channel ordering,
    /// coefficient transposition, prediction and DC offsets before handing off.
    pub coefficients: Vec<i16>,
    /// Original JPEG quantization values in natural 8x8 order, without scaling.
    pub quantization: [i32; 64],
}

#[derive(Debug)]
pub enum ReconstructionDecodeError {
    Truncated(&'static str),
    Invalid(&'static str),
    Unsupported(&'static str),
    MissingSource(&'static str),
    Limit(&'static str),
    Structural(JpegError),
    Reconstruction(ReconstructionError),
}

impl fmt::Display for ReconstructionDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated(s) => write!(f, "truncated jbrd: {s}"),
            Self::Invalid(s) => write!(f, "invalid jbrd: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported jbrd: {s}"),
            Self::MissingSource(s) => write!(f, "missing JPEG reconstruction source: {s}"),
            Self::Limit(s) => write!(f, "JPEG reconstruction decode limit: {s}"),
            Self::Structural(e) => e.fmt(f),
            Self::Reconstruction(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for ReconstructionDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Structural(e) => Some(e),
            Self::Reconstruction(e) => Some(e),
            _ => None,
        }
    }
}

type Error = ReconstructionDecodeError;
type Result<T> = std::result::Result<T, Error>;
type Choices = [(u8, u32); 4];
const ONE_TO_FOUR: Choices = [(0, 1), (0, 2), (0, 3), (0, 4)];
const APP_TYPE: Choices = [(0, 0), (0, 1), (1, 2), (2, 4)];
const HUFFMAN_COUNT: Choices = [(0, 4), (3, 2), (4, 10), (6, 26)];
const CODE_COUNT: Choices = [(0, 0), (0, 1), (3, 2), (8, 0)];
const CODE_VALUE: Choices = [(2, 0), (2, 4), (4, 8), (8, 1)];
const LAST_PASS: Choices = [(0, 0), (0, 1), (0, 2), (3, 3)];
const HINT_COUNT: Choices = [(0, 0), (2, 1), (4, 4), (16, 20)];
const BLOCK_DELTA: Choices = [(0, 0), (3, 1), (5, 9), (28, 41)];
const ZERO_RUNS: Choices = [(0, 1), (2, 2), (4, 5), (8, 20)];
const TAIL_LENGTH: Choices = [(0, 0), (8, 1), (16, 257), (22, 65_793)];

fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<()> {
    v.try_reserve_exact(n)
        .map_err(|_| Error::Limit("allocation"))
}

fn push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    v.try_reserve(1).map_err(|_| Error::Limit("allocation"))?;
    v.push(value);
    Ok(())
}

fn add(total: &mut usize, n: usize, limit: usize, resource: &'static str) -> Result<()> {
    *total = total
        .checked_add(n)
        .filter(|&v| v <= limit)
        .ok_or(Error::Limit(resource))?;
    Ok(())
}

fn copy(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    reserve(&mut out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(out)
}

struct Bits<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Bits<'_> {
    fn bits(&mut self, count: u8) -> Result<u32> {
        let end = self
            .position
            .checked_add(usize::from(count))
            .ok_or(Error::Limit("bit offset"))?;
        if end.div_ceil(8) > self.bytes.len() {
            return Err(Error::Truncated("metadata header"));
        }
        let mut value = 0;
        for bit in 0..count {
            let index = self.position + usize::from(bit);
            value |= u32::from((self.bytes[index / 8] >> (index % 8)) & 1) << bit;
        }
        self.position = end;
        Ok(value)
    }

    fn field(&mut self, choices: Choices) -> Result<u32> {
        let (count, offset) = choices[self.bits(2)? as usize];
        Ok(self.bits(count)? + offset)
    }

    fn block(&mut self, next: &mut u32) -> Result<u32> {
        let index = next
            .checked_add(self.field(BLOCK_DELTA)?)
            .filter(|&v| v < (3 << 26))
            .ok_or(Error::Invalid("block hint index"))?;
        *next = index + 1;
        Ok(index)
    }

    fn finish(&mut self) -> Result<usize> {
        let count = ((8 - self.position % 8) % 8) as u8;
        if self.bits(count)? != 0 {
            return Err(Error::Invalid("nonzero metadata byte alignment"));
        }
        Ok(self.position / 8)
    }
}

#[derive(Debug)]
struct App {
    kind: AppMarkerType,
    length: usize,
}

/// Parsed metadata is opaque until matched to a captured coefficient frame.
/// It contains no original JPEG entropy bytes and cannot reconstruct from pixels.
#[derive(Debug)]
pub struct DecodedReconstructionMetadata {
    markers: Vec<u8>,
    apps: Vec<App>,
    com_lengths: Vec<usize>,
    quant: Vec<QuantTable>,
    ids: Vec<u32>,
    component_quant: Vec<usize>,
    // Empty histograms represent empty DHT markers, not actual table definitions.
    huffman: Vec<HuffmanCode>,
    scans: Vec<ScanInfo>,
    restart_interval: u32,
    inter_lengths: Vec<usize>,
    tail_length: usize,
    has_zero_padding_bit: bool,
    padding_bits: Vec<u8>,
    payload: Vec<u8>,
    metadata_bytes: usize,
}

impl DecodedReconstructionMetadata {
    pub fn component_count(&self) -> usize {
        self.ids.len()
    }
    pub fn scan_count(&self) -> usize {
        self.scans.len()
    }
    pub fn needs_icc(&self) -> bool {
        self.apps.iter().any(|a| a.kind == AppMarkerType::Icc)
    }
    pub fn needs_exif(&self) -> bool {
        self.apps.iter().any(|a| a.kind == AppMarkerType::Exif)
    }
    pub fn needs_xmp(&self) -> bool {
        self.apps.iter().any(|a| a.kind == AppMarkerType::Xmp)
    }
}

fn read_huffman(r: &mut Bits<'_>) -> Result<HuffmanCode> {
    let class = r.bits(1)?;
    let mut table = HuffmanCode {
        slot_id: (class << 4) | r.bits(2)?,
        is_last: r.bits(1)? != 0,
        ..Default::default()
    };
    let mut count = 0usize;
    for n in &mut table.counts {
        *n = r.field(CODE_COUNT)?;
        count += *n as usize;
    }
    if table.counts[0] != 0 || count > 257 {
        return Err(Error::Invalid("Huffman histogram"));
    }
    if count == 0 {
        return Ok(table);
    }
    if count < 2 {
        return Err(Error::Invalid("Huffman table has no real symbols"));
    }
    let mut seen = [false; 257];
    for i in 0..count {
        let value = r.field(CODE_VALUE)?;
        if seen[value as usize]
            || (i + 1 == count) != (value == 256)
            || (class == 0 && value != 256 && value >= 12)
        {
            return Err(Error::Invalid("Huffman symbol or sentinel"));
        }
        seen[value as usize] = true;
        table.values[i] = value;
    }
    let last = table
        .counts
        .iter()
        .rposition(|&n| n != 0)
        .ok_or(Error::Invalid("Huffman histogram"))?;
    table.counts[last] -= 1;
    table.values[count - 1] = 0;
    let counts = std::array::from_fn(|i| table.counts[i + 1] as u8);
    let mut symbols = [0u8; 256];
    for (symbol, &value) in symbols.iter_mut().zip(&table.values[..count - 1]) {
        *symbol = value as u8;
    }
    HuffmanTable::new(&counts, &symbols[..count - 1])
        .map_err(|_| Error::Invalid("Huffman code tree"))?;
    Ok(table)
}

/// Parse a complete `jbrd` BOX PAYLOAD (no box header). A standard Brotli stream
/// must follow the zero-aligned VisitFields header, even for an empty payload.
/// Declared expansion, marker/scan/hint counts and padding are bounded before
/// allocation. Trailing compressed bytes and extra decoded output are errors.
pub fn decode_reconstruction_metadata(
    encoded: &[u8],
    limits: &ReconstructionDecodeLimits,
) -> Result<DecodedReconstructionMetadata> {
    if encoded.len() > limits.max_jbrd_bytes {
        return Err(Error::Limit("jbrd bytes"));
    }
    let mut r = Bits {
        bytes: encoded,
        position: 0,
    };
    let gray = r.bits(1)? != 0;
    let mut m = DecodedReconstructionMetadata {
        markers: Vec::new(),
        apps: Vec::new(),
        com_lengths: Vec::new(),
        quant: Vec::new(),
        ids: Vec::new(),
        component_quant: Vec::new(),
        huffman: Vec::new(),
        scans: Vec::new(),
        restart_interval: 0,
        inter_lengths: Vec::new(),
        tail_length: 0,
        has_zero_padding_bit: false,
        padding_bits: Vec::new(),
        payload: Vec::new(),
        metadata_bytes: 0,
    };
    let mut scan_count = 0usize;
    let mut record_count = 1usize;
    loop {
        let marker = r.bits(6)? as u8 + 0xc0;
        if !matches!(marker, 0xc0..=0xc2 | 0xc4 | 0xd9 | 0xda | 0xdb | 0xdd | 0xe0..=0xef | 0xfe | 0xff)
        {
            return Err(Error::Unsupported("JPEG marker"));
        }
        if m.markers.len() == 16_384 {
            return Err(Error::Limit("marker count"));
        }
        add(
            &mut record_count,
            if marker == 0xda { 2 } else { 1 },
            limits.write.structural.max_records,
            "record count",
        )?;
        if marker == 0xda {
            add(
                &mut scan_count,
                1,
                limits.write.structural.max_scans,
                "scan count",
            )?;
        }
        push(&mut m.markers, marker)?;
        if marker == 0xd9 {
            break;
        }
    }
    if scan_count == 0 {
        return Err(Error::Invalid("no scans"));
    }
    let mut expanded = 0usize;
    let mut payload_length = 0usize;
    let cap = limits.max_metadata_bytes;
    for _ in m.markers.iter().filter(|&&v| (0xe0..=0xef).contains(&v)) {
        let kind = match r.field(APP_TYPE)? {
            0 => AppMarkerType::Unknown,
            1 => AppMarkerType::Icc,
            2 => AppMarkerType::Exif,
            3 => AppMarkerType::Xmp,
            _ => return Err(Error::Unsupported("APP type")),
        };
        let length = r.bits(16)? as usize + 1;
        let minimum = match kind {
            AppMarkerType::Unknown => 3,
            AppMarkerType::Icc => 17,
            AppMarkerType::Exif => 9,
            AppMarkerType::Xmp => 32,
        };
        if length < minimum {
            return Err(Error::Invalid("APP length"));
        }
        add(&mut expanded, length, cap, "expanded metadata")?;
        if kind == AppMarkerType::Unknown {
            add(&mut payload_length, length, cap, "payload length")?;
        }
        push(&mut m.apps, App { kind, length })?;
    }
    for _ in m.markers.iter().filter(|&&v| v == 0xfe) {
        let length = r.bits(16)? as usize + 1;
        if length < 3 {
            return Err(Error::Invalid("COM length"));
        }
        add(&mut expanded, length, cap, "expanded metadata")?;
        add(&mut payload_length, length, cap, "payload length")?;
        push(&mut m.com_lengths, length)?;
    }
    let count = r.field(ONE_TO_FOUR)? as usize;
    if count == 4 {
        return Err(Error::Invalid("four quantization tables"));
    }
    if count > limits.write.structural.max_quant_tables {
        return Err(Error::Limit("quantization tables"));
    }
    for _ in 0..count {
        push(
            &mut m.quant,
            QuantTable {
                precision: r.bits(1)?,
                index: r.bits(2)?,
                is_last: r.bits(1)? != 0,
                ..Default::default()
            },
        )?;
    }
    let component_type = r.bits(2)?;
    let count = match component_type {
        0 => 1,
        1 | 2 => 3,
        _ => r.field(ONE_TO_FOUR)? as usize,
    };
    if !matches!(count, 1 | 3) || gray != (count == 1) {
        return Err(Error::Invalid("component count"));
    }
    for i in 0..count {
        let id = match component_type {
            0 => 1,
            1 => i as u32 + 1,
            2 => u32::from(b"RGB"[i]),
            _ => r.bits(8)?,
        };
        if m.ids.contains(&id) {
            return Err(Error::Invalid("duplicate component ID"));
        }
        push(&mut m.ids, id)?;
    }
    for _ in 0..count {
        let index = r.bits(2)? as usize;
        if index >= m.quant.len() {
            return Err(Error::Invalid("component quantization index"));
        }
        push(&mut m.component_quant, index)?;
    }
    if !m.component_quant.contains(&0) {
        return Err(Error::Invalid("unused first quantization table"));
    }
    let count = r.field(HUFFMAN_COUNT)? as usize;
    if count > limits.write.structural.max_huffman_tables {
        return Err(Error::Limit("Huffman tables"));
    }
    for _ in 0..count {
        push(&mut m.huffman, read_huffman(&mut r)?)?;
    }
    for _ in 0..scan_count {
        let count = r.field(ONE_TO_FOUR)? as usize;
        if count > m.ids.len() {
            return Err(Error::Invalid("scan component count"));
        }
        let mut scan = ScanInfo {
            ss: r.bits(6)?,
            se: r.bits(6)?,
            al: r.bits(4)?,
            ah: r.bits(4)?,
            ..Default::default()
        };
        let mut seen = [false; 3];
        for _ in 0..count {
            let index = r.bits(2)? as usize;
            if index >= m.ids.len() || seen[index] {
                return Err(Error::Invalid("scan component index"));
            }
            seen[index] = true;
            push(
                &mut scan.components,
                ComponentScanInfo {
                    component_index: index,
                    ac_table_index: r.bits(2)?,
                    dc_table_index: r.bits(2)?,
                },
            )?;
        }
        scan.last_needed_pass = r.field(LAST_PASS)?;
        push(&mut m.scans, scan)?;
    }
    if m.markers.contains(&0xdd) {
        m.restart_interval = r.bits(16)?;
    }
    let mut hints = 0usize;
    for scan in &mut m.scans {
        let count = r.field(HINT_COUNT)? as usize;
        add(
            &mut hints,
            count,
            limits.write.max_hint_entries,
            "reconstruction hints",
        )?;
        let mut next = 0;
        for _ in 0..count {
            push(&mut scan.reset_points, r.block(&mut next)?)?;
        }
        let count = r.field(HINT_COUNT)? as usize;
        add(
            &mut hints,
            count,
            limits.write.max_hint_entries,
            "reconstruction hints",
        )?;
        next = 0;
        for _ in 0..count {
            let count = r.field(ZERO_RUNS)?;
            if count > 4 || scan.ah != 0 {
                return Err(Error::Invalid("extra zero run"));
            }
            push(
                &mut scan.extra_zero_runs,
                ExtraZeroRun {
                    count,
                    block_index: r.block(&mut next)?,
                },
            )?;
        }
    }
    for _ in m.markers.iter().filter(|&&v| v == 0xff) {
        let length = r.bits(16)? as usize;
        add(&mut expanded, length, cap, "expanded metadata")?;
        add(&mut payload_length, length, cap, "payload length")?;
        push(&mut m.inter_lengths, length)?;
    }
    m.tail_length = r.field(TAIL_LENGTH)? as usize;
    add(&mut expanded, m.tail_length, cap, "expanded metadata")?;
    add(&mut payload_length, m.tail_length, cap, "payload length")?;
    if m.tail_length != 0 {
        add(
            &mut record_count,
            1,
            limits.write.structural.max_records,
            "record count",
        )?;
    }
    m.has_zero_padding_bit = r.bits(1)? != 0;
    if m.has_zero_padding_bit {
        let count = r.bits(24)? as usize;
        if count > limits.write.max_padding_bits {
            return Err(Error::Limit("padding bits"));
        }
        if count > encoded.len().saturating_mul(8).saturating_sub(r.position) {
            return Err(Error::Truncated("padding bits"));
        }
        reserve(&mut m.padding_bits, count)?;
        for _ in 0..count {
            m.padding_bits.push(r.bits(1)? as u8);
        }
    }
    let header_length = r.finish()?;
    add(&mut expanded, header_length, cap, "expanded metadata")?;
    m.metadata_bytes = expanded;
    validate_layout(&m)?;
    m.payload = decompress_exact(&encoded[header_length..], payload_length)?;
    validate_payload(&m)?;
    Ok(m)
}

fn decompress_exact(input: &[u8], length: usize) -> Result<Vec<u8>> {
    let capacity = length.checked_add(1).ok_or(Error::Limit("Brotli output"))?;
    let mut output = Vec::new();
    reserve(&mut output, capacity)?;
    output.resize(capacity, 0);
    let mut state = brotli::BrotliState::new_strict(
        brotli::HeapAlloc::<u8>::default(),
        brotli::HeapAlloc::<u32>::default(),
        brotli::HeapAlloc::<brotli::HuffmanCode>::default(),
    );
    let mut available_in = input.len();
    let mut input_offset = 0;
    let mut available_out = capacity;
    let mut output_offset = 0;
    let mut total_out = 0;
    let result = brotli::BrotliDecompressStream(
        &mut available_in,
        &mut input_offset,
        input,
        &mut available_out,
        &mut output_offset,
        &mut output,
        &mut total_out,
        &mut state,
    );
    match result {
        brotli::BrotliResult::ResultSuccess => {}
        brotli::BrotliResult::NeedsMoreInput => return Err(Error::Truncated("Brotli stream")),
        brotli::BrotliResult::NeedsMoreOutput => {
            return Err(Error::Invalid("excess Brotli output"));
        }
        brotli::BrotliResult::ResultFailure => return Err(Error::Invalid("Brotli stream")),
    }
    if available_in != 0 || input_offset != input.len() {
        return Err(Error::Invalid("trailing Brotli bytes"));
    }
    if output_offset != length {
        return Err(Error::Invalid("Brotli output length"));
    }
    output.truncate(length);
    Ok(output)
}

fn validate_layout(m: &DecodedReconstructionMetadata) -> Result<()> {
    let mut quant = 0;
    let mut huffman = 0;
    let mut scan_index = 0;
    let mut frame = None;
    let mut dc = [false; 4];
    let mut ac = [false; 4];
    let mut q = [false; 4];
    let mut progression = ScanProgression::new(m.ids.len()).map_err(Error::Invalid)?;
    for &marker in &m.markers {
        match marker {
            0xc0..=0xc2 => {
                if frame.replace(marker).is_some() {
                    return Err(Error::Invalid("duplicate SOF"));
                }
            }
            0xdb => loop {
                let t = m.quant.get(quant).ok_or(Error::Invalid("DQT grouping"))?;
                quant += 1;
                q[t.index as usize] = true;
                if t.is_last {
                    break;
                }
            },
            0xc4 => {
                let first = huffman;
                loop {
                    let t = m
                        .huffman
                        .get(huffman)
                        .ok_or(Error::Invalid("DHT grouping"))?;
                    huffman += 1;
                    if t.counts.iter().all(|&n| n == 0) {
                        if !t.is_last || huffman != first + 1 {
                            return Err(Error::Invalid("empty DHT grouping"));
                        }
                    } else if t.slot_id & 16 == 0 {
                        dc[t.slot_id as usize] = true;
                    } else {
                        ac[(t.slot_id & 15) as usize] = true;
                    }
                    if t.is_last {
                        break;
                    }
                }
            }
            0xda => {
                let progressive = frame.ok_or(Error::Invalid("SOS before SOF"))? == 0xc2;
                let scan = &m.scans[scan_index];
                scan_index += 1;
                if scan.ss > scan.se || scan.al > 10 {
                    return Err(Error::Invalid("scan spectral band or approximation"));
                }
                if progressive {
                    if (scan.ss == 0 && scan.se != 0)
                        || (scan.ss != 0 && scan.components.len() != 1)
                        || (scan.ah != 0 && scan.ah != scan.al + 1)
                    {
                        return Err(Error::Invalid("progressive scan parameters"));
                    }
                } else if scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0 {
                    return Err(Error::Invalid("sequential scan parameters"));
                }
                for c in &scan.components {
                    if !q[m.quant[m.component_quant[c.component_index]].index as usize]
                        || ((!progressive || (scan.ss == 0 && scan.ah == 0))
                            && !dc[c.dc_table_index as usize])
                        || ((!progressive || scan.se != 0) && !ac[c.ac_table_index as usize])
                    {
                        return Err(Error::Invalid("table used before definition"));
                    }
                    let coverage = progression
                        .coverage(c.component_index)
                        .ok_or(Error::Invalid("scan component"))?;
                    if (scan.ss != 0 && coverage[0] == 0)
                        || (scan.ah != 0
                            && coverage[scan.ss as usize..=scan.se as usize].contains(&0))
                    {
                        return Err(Error::Invalid("scan precedes its initial coefficients"));
                    }
                }
                progression
                    .apply(scan, progressive)
                    .map_err(Error::Invalid)?;
            }
            _ => {}
        }
    }
    if quant != m.quant.len() || huffman != m.huffman.len() {
        return Err(Error::Invalid("unused table definitions"));
    }
    for i in 0..m.ids.len() {
        if progression.coverage(i).ok_or(Error::Invalid("component"))?[0] == 0 {
            return Err(Error::Invalid("component has no DC scan"));
        }
    }
    Ok(())
}

struct Payload<'a>(&'a [u8]);
impl<'a> Payload<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let bytes = self
            .0
            .get(..n)
            .ok_or(Error::Truncated("expanded metadata"))?;
        self.0 = &self.0[n..];
        Ok(bytes)
    }
}

fn validate_envelope(bytes: &[u8], marker: u8) -> Result<()> {
    if bytes.len() < 3
        || bytes[0] != marker
        || usize::from(u16::from_be_bytes([bytes[1], bytes[2]])) + 1 != bytes.len()
    {
        return Err(Error::Invalid("APP/COM marker envelope"));
    }
    Ok(())
}

fn validate_payload(m: &DecodedReconstructionMetadata) -> Result<()> {
    let mut payload = Payload(&m.payload);
    let markers = m
        .markers
        .iter()
        .copied()
        .filter(|v| (0xe0..=0xef).contains(v));
    let mut icc = 0;
    let mut exif = 0;
    let mut xmp = 0;
    for (app, marker) in m.apps.iter().zip(markers) {
        match app.kind {
            AppMarkerType::Unknown => validate_envelope(payload.take(app.length)?, marker)?,
            AppMarkerType::Icc => {
                icc += 1;
                if marker != 0xe2 || icc > 255 {
                    return Err(Error::Invalid("ICC marker count or code"));
                }
            }
            AppMarkerType::Exif => {
                exif += 1;
                if marker != 0xe1 || exif > 1 {
                    return Err(Error::Invalid("Exif marker count or code"));
                }
            }
            AppMarkerType::Xmp => {
                xmp += 1;
                if marker != 0xe1 || xmp > 1 {
                    return Err(Error::Invalid("XMP marker count or code"));
                }
            }
        }
    }
    for &length in &m.com_lengths {
        validate_envelope(payload.take(length)?, 0xfe)?;
    }
    Ok(())
}

struct Hydrated {
    apps: Vec<Vec<u8>>,
    comments: Vec<Vec<u8>>,
    inter: Vec<Vec<u8>>,
    tail: Vec<u8>,
}

fn hydrate(
    m: &DecodedReconstructionMetadata,
    sources: ReconstructionMetadataSources<'_>,
) -> Result<Hydrated> {
    let mut result = Hydrated {
        apps: Vec::new(),
        comments: Vec::new(),
        inter: Vec::new(),
        tail: Vec::new(),
    };
    let mut payload = Payload(&m.payload);
    let icc_count = m
        .apps
        .iter()
        .filter(|a| a.kind == AppMarkerType::Icc)
        .count();
    let mut icc_ordinal = 0;
    let mut icc_position = 0usize;
    for app in &m.apps {
        if app.kind == AppMarkerType::Unknown {
            push(&mut result.apps, copy(payload.take(app.length)?)?)?;
            continue;
        }
        let mut bytes = Vec::new();
        reserve(&mut bytes, app.length)?;
        bytes.resize(app.length, 0);
        bytes[1..3].copy_from_slice(&((app.length - 1) as u16).to_be_bytes());
        match app.kind {
            AppMarkerType::Icc => {
                let profile = sources
                    .icc_profile
                    .ok_or(Error::MissingSource("original ICC profile"))?;
                let count = app.length - 17;
                let end = icc_position
                    .checked_add(count)
                    .ok_or(Error::Limit("ICC length"))?;
                let chunk = profile
                    .get(icc_position..end)
                    .ok_or(Error::Invalid("ICC source length"))?;
                icc_position = end;
                icc_ordinal += 1;
                bytes[0] = 0xe2;
                bytes[3..15].copy_from_slice(b"ICC_PROFILE\0");
                bytes[15] = icc_ordinal;
                bytes[16] = icc_count as u8;
                bytes[17..].copy_from_slice(chunk);
            }
            AppMarkerType::Exif => {
                let source = sources
                    .exif_box_payload
                    .ok_or(Error::MissingSource("Exif box"))?;
                // libjxl's reconstruction contract preserves the TIFF payload
                // only; a nonzero TIFF offset is not a byte-reversible source.
                if source.get(..4) != Some(&[0; 4]) || source.len() - 4 != app.length - 9 {
                    return Err(Error::Invalid("Exif source length or TIFF offset"));
                }
                bytes[0] = 0xe1;
                bytes[3..9].copy_from_slice(b"Exif\0\0");
                bytes[9..].copy_from_slice(&source[4..]);
            }
            AppMarkerType::Xmp => {
                let source = sources
                    .xmp_box_payload
                    .ok_or(Error::MissingSource("xml box"))?;
                if source.len() != app.length - 32 {
                    return Err(Error::Invalid("XMP source length"));
                }
                bytes[0] = 0xe1;
                bytes[3..32].copy_from_slice(b"http://ns.adobe.com/xap/1.0/\0");
                bytes[32..].copy_from_slice(source);
            }
            AppMarkerType::Unknown => unreachable!(),
        }
        push(&mut result.apps, bytes)?;
    }
    if icc_count != 0 && sources.icc_profile.map(<[u8]>::len) != Some(icc_position) {
        return Err(Error::Invalid("unused ICC source bytes"));
    }
    for &length in &m.com_lengths {
        push(&mut result.comments, copy(payload.take(length)?)?)?;
    }
    for &length in &m.inter_lengths {
        push(&mut result.inter, copy(payload.take(length)?)?)?;
    }
    result.tail = copy(payload.take(m.tail_length)?)?;
    if !payload.0.is_empty() {
        return Err(Error::Invalid("unused metadata payload"));
    }
    Ok(result)
}

fn append(output: &mut Vec<u8>, bytes: &[u8], cap: usize) -> Result<()> {
    if bytes.len() > cap.saturating_sub(output.len()) {
        return Err(Error::Limit("marker template bytes"));
    }
    output
        .try_reserve(bytes.len())
        .map_err(|_| Error::Limit("marker template allocation"))?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn segment(output: &mut Vec<u8>, marker: u8, payload: &[u8], cap: usize) -> Result<()> {
    let length = payload
        .len()
        .checked_add(2)
        .and_then(|n| u16::try_from(n).ok())
        .ok_or(Error::Invalid("marker length"))?;
    append(output, &[0xff, marker], cap)?;
    append(output, &length.to_be_bytes(), cap)?;
    append(output, payload, cap)
}

fn match_frame(
    m: &mut DecodedReconstructionMetadata,
    frame: &CapturedJpegCoefficients,
    limits: &ReconstructionDecodeLimits,
) -> Result<()> {
    if !(1..=65535).contains(&frame.width)
        || !(1..=65535).contains(&frame.height)
        || frame.components.len() != m.ids.len()
    {
        return Err(Error::Invalid(
            "captured frame dimensions or component count",
        ));
    }
    let mut max_h = 1;
    let mut max_v = 1;
    for c in &frame.components {
        if !(1..=15).contains(&c.h_samp_factor) || !(1..=15).contains(&c.v_samp_factor) {
            return Err(Error::Invalid("captured sampling factors"));
        }
        max_h = max_h.max(c.h_samp_factor);
        max_v = max_v.max(c.v_samp_factor);
    }
    let cols = frame.width.div_ceil(8 * max_h);
    let rows = frame.height.div_ceil(8 * max_v);
    let mut values = 0usize;
    let mut assigned = [false; 3];
    for (index, c) in frame.components.iter().enumerate() {
        if max_h % c.h_samp_factor != 0
            || max_v % c.v_samp_factor != 0
            || c.width_in_blocks != cols * c.h_samp_factor
            || c.height_in_blocks != rows * c.v_samp_factor
        {
            return Err(Error::Invalid("captured MCU-padded block geometry"));
        }
        let count = u64::from(c.width_in_blocks)
            .checked_mul(u64::from(c.height_in_blocks))
            .and_then(|n| n.checked_mul(64))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Error::Limit("coefficient geometry"))?;
        add(
            &mut values,
            count,
            limits.max_coefficient_values,
            "coefficient values",
        )?;
        if count != c.coefficients.len() {
            return Err(Error::Invalid("captured coefficient storage length"));
        }
        let q = m.component_quant[index];
        if assigned[q] && m.quant[q].values != c.quantization {
            return Err(Error::Invalid("shared quantization table disagrees"));
        }
        assigned[q] = true;
        m.quant[q].values = c.quantization;
    }
    for (q, &used) in assigned.iter().enumerate().take(m.quant.len()) {
        if !used {
            // VisitFields requires table zero to be used. Remaining unused
            // definitions reproduce the immediately preceding table values.
            if q == 0 {
                return Err(Error::Invalid("unused first quantization table"));
            }
            m.quant[q].values = m.quant[q - 1].values;
        }
        let max = if m.quant[q].precision == 0 {
            255
        } else {
            65535
        };
        if m.quant[q].values.iter().any(|&v| v <= 0 || v > max) {
            return Err(Error::Invalid("quantization values or precision"));
        }
    }
    Ok(())
}

fn marker_template(
    m: &DecodedReconstructionMetadata,
    frame: &CapturedJpegCoefficients,
    data: &Hydrated,
    cap: usize,
) -> Result<Vec<u8>> {
    let mut template = Vec::new();
    append(&mut template, &[0xff, 0xd8], cap)?;
    let mut q = 0;
    let mut h = 0;
    let mut scan = 0;
    let mut app = 0;
    let mut com = 0;
    for &marker in &m.markers {
        match marker {
            0xc0..=0xc2 => {
                let mut sof = Vec::new();
                reserve(&mut sof, 6 + 3 * m.ids.len())?;
                sof.push(8);
                sof.extend_from_slice(&(frame.height as u16).to_be_bytes());
                sof.extend_from_slice(&(frame.width as u16).to_be_bytes());
                sof.push(m.ids.len() as u8);
                for (i, c) in frame.components.iter().enumerate() {
                    sof.extend_from_slice(&[
                        m.ids[i] as u8,
                        ((c.h_samp_factor << 4) | c.v_samp_factor) as u8,
                        m.quant[m.component_quant[i]].index as u8,
                    ]);
                }
                segment(&mut template, marker, &sof, cap)?;
            }
            0xdb => {
                let mut dqt = Vec::new();
                loop {
                    let t = &m.quant[q];
                    q += 1;
                    reserve(&mut dqt, 1 + 64 * (t.precision as usize + 1))?;
                    dqt.push(((t.precision << 4) | t.index) as u8);
                    for &natural in &NATURAL_ORDER[..64] {
                        let value = t.values[natural] as u16;
                        if t.precision == 0 {
                            dqt.push(value as u8);
                        } else {
                            dqt.extend_from_slice(&value.to_be_bytes());
                        }
                    }
                    if t.is_last {
                        break;
                    }
                }
                segment(&mut template, marker, &dqt, cap)?;
            }
            0xc4 => {
                let mut dht = Vec::new();
                loop {
                    let t = &m.huffman[h];
                    h += 1;
                    let count: usize = t.counts.iter().map(|&n| n as usize).sum();
                    if count != 0 {
                        reserve(&mut dht, 17 + count)?;
                        dht.push(t.slot_id as u8);
                        dht.extend(t.counts[1..].iter().map(|&v| v as u8));
                        dht.extend(t.values[..count].iter().map(|&v| v as u8));
                    }
                    if t.is_last {
                        break;
                    }
                }
                segment(&mut template, marker, &dht, cap)?;
            }
            0xdd => segment(
                &mut template,
                marker,
                &(m.restart_interval as u16).to_be_bytes(),
                cap,
            )?,
            0xda => {
                let s = &m.scans[scan];
                scan += 1;
                let mut sos = Vec::new();
                reserve(&mut sos, 4 + 2 * s.components.len())?;
                sos.push(s.components.len() as u8);
                for c in &s.components {
                    sos.extend_from_slice(&[
                        m.ids[c.component_index] as u8,
                        ((c.dc_table_index << 4) | c.ac_table_index) as u8,
                    ]);
                }
                sos.extend_from_slice(&[s.ss as u8, s.se as u8, ((s.ah << 4) | s.al) as u8]);
                segment(&mut template, marker, &sos, cap)?;
                // There is deliberately no entropy payload in this template.
            }
            0xe0..=0xef => {
                append(&mut template, &[0xff], cap)?;
                append(&mut template, &data.apps[app], cap)?;
                app += 1;
            }
            0xfe => {
                append(&mut template, &[0xff], cap)?;
                append(&mut template, &data.comments[com], cap)?;
                com += 1;
            }
            0xd9 => append(&mut template, &[0xff, marker], cap)?,
            // Insert opaque inter-marker bytes after structural parsing; bytes
            // after an empty SOS would otherwise be mistaken for entropy.
            0xff => {}
            _ => return Err(Error::Unsupported("JPEG marker")),
        }
    }
    Ok(template)
}

impl DecodedReconstructionMetadata {
    /// Attach exact captured coefficients, rehydrate deduplicated APPs and
    /// construct a validated writer model. Coefficient buffers are moved, not
    /// cloned. All EntropyData::bytes remain empty; restart descriptors are
    /// derived from each scan's active DRI and MCU geometry.
    ///
    /// The returned model uses an empty padding vector when VisitFields omits
    /// all-one padding. The coefficient writer must support that wire default.
    pub fn into_jpeg(
        mut self,
        frame: CapturedJpegCoefficients,
        sources: ReconstructionMetadataSources<'_>,
        limits: &ReconstructionDecodeLimits,
    ) -> Result<JpegData> {
        if self.metadata_bytes > limits.max_metadata_bytes {
            return Err(Error::Limit("expanded metadata"));
        }
        match_frame(&mut self, &frame, limits)?;
        let data = hydrate(&self, sources)?;
        // DQT/DHT marker bytes can exceed their compact VisitFields header.
        // Bound this independent expansion by the existing writer budgets.
        let cap = limits
            .write
            .max_output_bytes
            .min(limits.write.structural.max_input_bytes);
        let template = marker_template(&self, &frame, &data, cap)?;
        let mut jpeg = read_jpeg_with_limits(&template, &limits.write.structural)
            .map_err(Error::Structural)?;
        drop(template);
        if jpeg.scan_info.len() != self.scans.len() || jpeg.quant != self.quant {
            return Err(Error::Invalid("generated marker layout"));
        }
        // Table destination IDs can be redefined. A component's captured table
        // must match every active definition used by that component's scans.
        for scan in &jpeg.scan_info {
            for sc in &scan.components {
                let c = &jpeg.components[sc.component_index];
                let q = scan.quant_table_indices[c.quant_index as usize]
                    .ok_or(Error::Invalid("missing active quantization table"))?;
                if jpeg.quant[q].values != frame.components[sc.component_index].quantization {
                    return Err(Error::Invalid(
                        "captured quantization disagrees with scan table",
                    ));
                }
            }
        }
        for (c, captured) in jpeg.components.iter_mut().zip(frame.components) {
            if c.width_in_blocks != captured.width_in_blocks
                || c.height_in_blocks != captured.height_in_blocks
            {
                return Err(Error::Invalid("captured block geometry disagrees with SOF"));
            }
            c.coefficients = captured.coefficients;
        }
        for (s, wire) in jpeg.scan_info.iter_mut().zip(self.scans) {
            s.last_needed_pass = wire.last_needed_pass;
            s.reset_points = wire.reset_points;
            s.extra_zero_runs = wire.extra_zero_runs;
        }
        jpeg.app_marker_type = self.apps.into_iter().map(|a| a.kind).collect();
        jpeg.inter_marker_data = data.inter;
        jpeg.tail_data = data.tail;
        jpeg.has_zero_padding_bit = self.has_zero_padding_bit;
        jpeg.padding_bits = self.padding_bits;
        let mut records = std::mem::take(&mut jpeg.records).into_iter();
        let mut output = Vec::new();
        push(
            &mut output,
            records
                .next()
                .ok_or(Error::Invalid("missing generated SOI"))?,
        )?;
        let mut inter = 0;
        let mut restarts = 0usize;
        let mut visits = 0usize;
        for marker in self.markers {
            if marker == 0xff {
                push(
                    &mut output,
                    JpegRecord::InterMarker {
                        offset: 0,
                        index: inter,
                    },
                )?;
                inter += 1;
                continue;
            }
            let record = records
                .next()
                .ok_or(Error::Invalid("missing generated marker"))?;
            if !matches!(&record, JpegRecord::Marker(s) if s.marker == marker) {
                return Err(Error::Invalid("generated marker order"));
            }
            push(&mut output, record)?;
            if marker == 0xda {
                let Some(JpegRecord::Entropy(mut entropy)) = records.next() else {
                    return Err(Error::Invalid("missing generated scan descriptor"));
                };
                let scan = &jpeg.scan_info[entropy.scan_index];
                let mcus = u64::from(scan.mcu_cols) * u64::from(scan.mcu_rows);
                let blocks: u64 = if scan.components.len() == 1 {
                    1
                } else {
                    scan.components
                        .iter()
                        .map(|sc| {
                            let c = &jpeg.components[sc.component_index];
                            u64::from(c.h_samp_factor) * u64::from(c.v_samp_factor)
                        })
                        .sum()
                };
                let blocks = mcus
                    .checked_mul(blocks)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or(Error::Limit("scan block visits"))?;
                add(
                    &mut visits,
                    blocks,
                    limits.write.max_block_visits,
                    "scan block visits",
                )?;
                let count = if scan.restart_interval == 0 {
                    0
                } else {
                    (mcus - 1) / u64::from(scan.restart_interval)
                };
                let count = usize::try_from(count).map_err(|_| Error::Limit("restart markers"))?;
                add(
                    &mut restarts,
                    count,
                    limits.write.structural.max_restart_markers,
                    "restart markers",
                )?;
                reserve(&mut entropy.restart_markers, count)?;
                for i in 0..count {
                    entropy.restart_markers.push(RestartMarker {
                        offset: 0,
                        marker: 0xd0 + (i % 8) as u8,
                        fill_count: 0,
                    });
                }
                push(&mut output, JpegRecord::Entropy(entropy))?;
            }
        }
        if records.next().is_some() {
            return Err(Error::Invalid("unused generated records"));
        }
        if !jpeg.tail_data.is_empty() {
            push(&mut output, JpegRecord::Tail { offset: 0 })?;
        }
        jpeg.records = output;
        validate(&jpeg, &limits.write).map_err(Error::Reconstruction)?;
        Ok(jpeg)
    }
}

/// Reconstruct original JPEG bytes from one complete jbrd payload, captured
/// quantized coefficients and the original metadata sources. Neither pixels nor
/// an original JPEG are accepted. Container parsing and coefficient extraction
/// belong to the caller; no subprocess, native runtime or I/O is used here.
pub fn reconstruct_jpeg(
    jbrd_payload: &[u8],
    frame: CapturedJpegCoefficients,
    sources: ReconstructionMetadataSources<'_>,
    limits: &ReconstructionDecodeLimits,
) -> Result<Vec<u8>> {
    let metadata = decode_reconstruction_metadata(jbrd_payload, limits)?;
    let jpeg = metadata.into_jpeg(frame, sources, limits)?;
    reconstruct_from_coefficients(&jpeg, &limits.write).map_err(Error::Reconstruction)
}

#[cfg(test)]
#[path = "reconstruction_decode_tests.rs"]
mod tests;
