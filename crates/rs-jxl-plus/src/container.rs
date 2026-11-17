//! Experimental hashless version-1 container. Syntax is specific to jxl_plus;
//! it is not the planned JXL-derived extended frame syntax. Version 0 is rejected.
//! JPEG reconstruction metadata is delegated to rs-jxl's attributed libjxl port.

use rs_jxl::jpeg::{
    coefficient_reader::{CoefficientReadLimits, read_coefficients_with_limits},
    data::JpegData,
    reader::{ReadLimits, read_jpeg_with_limits},
    reconstruction_brotli::compress_parts,
    reconstruction_decode::{
        CapturedJpegCoefficients, CapturedJpegComponent, ReconstructionDecodeLimits,
        reconstruct_jpeg,
    },
    reconstruction_metadata::{ReconstructionMetadataSources, serialize_reconstruction_metadata},
    scan_writer::WriteLimits,
};

use crate::{
    Error, Result,
    model::{self, Component},
};

pub const MAGIC: &[u8; 8] = b"JXLPLUS\n";
pub const VERSION: u16 = 1;

/// Per-buffer and work limits, not a process-wide allocator limit.
#[derive(Clone, Debug)]
pub struct DecodeLimits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_coefficient_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_pixels: u64,
    pub max_scans: usize,
    pub max_block_visits: usize,
    pub max_hint_entries: usize,
}
impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 512 << 20,
            max_output_bytes: 512 << 20,
            max_coefficient_bytes: 256 << 20,
            max_metadata_bytes: 64 << 20,
            max_pixels: 100_000_000,
            max_scans: 1024,
            max_block_visits: 100_000_000,
            max_hint_entries: 1_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Bounds original JPEG input and all reconstructed/encoded buffers.
    pub limits: DecodeLimits,
    /// Brotli quality for rs-jxl's reconstruction metadata only, 0..=11.
    pub metadata_quality: u32,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            limits: DecodeLimits::default(),
            metadata_quality: 6,
        }
    }
}

fn jpeg_error(error: impl std::fmt::Display) -> Error {
    Error::Jpeg(error.to_string())
}

fn structural(limits: &DecodeLimits) -> ReadLimits {
    ReadLimits {
        max_input_bytes: limits.max_input_bytes.max(limits.max_output_bytes),
        max_records: limits.max_hint_entries,
        max_scans: limits.max_scans,
        max_quant_tables: 1024,
        max_huffman_tables: 1024,
        max_restart_markers: limits.max_hint_entries,
    }
}

pub(crate) fn reconstruction_limits(limits: &DecodeLimits) -> ReconstructionDecodeLimits {
    ReconstructionDecodeLimits {
        max_jbrd_bytes: limits.max_metadata_bytes,
        max_metadata_bytes: limits.max_metadata_bytes,
        max_coefficient_values: limits.max_coefficient_bytes / 2,
        write: WriteLimits {
            structural: structural(limits),
            max_output_bytes: limits.max_output_bytes,
            max_block_visits: limits.max_block_visits,
            max_hint_entries: limits.max_hint_entries,
            max_padding_bits: limits.max_metadata_bytes.saturating_mul(8),
            max_refinement_bits: limits.max_metadata_bytes.saturating_mul(8),
        },
    }
}

fn dimensions(width: u32, height: u32, limits: &DecodeLimits) -> Result<()> {
    if width == 0 || height == 0 || width > 65535 || height > 65535 {
        return Err(Error::Invalid("JPEG dimensions"));
    }
    if u64::from(width) * u64::from(height) > limits.max_pixels {
        return Err(Error::Limit("pixels"));
    }
    Ok(())
}

fn check_components(
    width: u32,
    height: u32,
    components: &[Component],
    limits: &DecodeLimits,
) -> Result<usize> {
    if !matches!(components.len(), 1 | 3) {
        return Err(Error::Unsupported(
            "version 1 requires one or three JPEG components",
        ));
    }
    let max_h = components.iter().map(|c| c.h_samp).max().unwrap();
    let max_v = components.iter().map(|c| c.v_samp).max().unwrap();
    if max_h == 0 || max_h > 4 || max_v == 0 || max_v > 4 {
        return Err(Error::Unsupported("JPEG sampling factors"));
    }
    let mcu_cols = (width as usize).div_ceil(max_h * 8);
    let mcu_rows = (height as usize).div_ceil(max_v * 8);
    let mut blocks = 0usize;
    for c in components {
        if c.h_samp == 0 || c.v_samp == 0 || max_h % c.h_samp != 0 || max_v % c.v_samp != 0 {
            return Err(Error::Unsupported("non-integral JPEG sampling"));
        }
        if c.width != mcu_cols * c.h_samp || c.height != mcu_rows * c.v_samp {
            return Err(Error::Invalid("component padded dimensions"));
        }
        if c.quant.contains(&0) {
            return Err(Error::Invalid("zero quantization value"));
        }
        blocks = blocks
            .checked_add(
                c.width
                    .checked_mul(c.height)
                    .ok_or(Error::Limit("coefficient dimensions"))?,
            )
            .ok_or(Error::Limit("coefficient dimensions"))?;
    }
    let bytes = blocks
        .checked_mul(128)
        .ok_or(Error::Limit("coefficient dimensions"))?;
    if bytes > limits.max_coefficient_bytes {
        return Err(Error::Limit("coefficient bytes"));
    }
    if blocks > limits.max_block_visits {
        return Err(Error::Limit("coefficient block visits"));
    }
    Ok((blocks / 65536).clamp(1, 16).min(mcu_rows))
}

fn components_from_jpeg(jpeg: &mut JpegData) -> Result<Vec<Component>> {
    let mut result = Vec::new();
    for (i, c) in jpeg.components.iter_mut().enumerate() {
        let scan = jpeg
            .scan_info
            .iter()
            .find(|s| s.components.iter().any(|sc| sc.component_index == i))
            .ok_or(Error::Invalid("missing JPEG component scan"))?;
        let q = scan
            .quant_table_indices
            .get(c.quant_index as usize)
            .and_then(|q| *q)
            .ok_or(Error::Invalid("missing quantization table"))?;
        let table = jpeg
            .quant
            .get(q)
            .ok_or(Error::Invalid("quantization table index"))?;
        let mut quant = [0u16; 64];
        for (out, &value) in quant.iter_mut().zip(&table.values) {
            *out = u16::try_from(value).map_err(|_| Error::Unsupported("quantization value"))?;
        }
        result.push(Component {
            width: c.width_in_blocks as usize,
            height: c.height_in_blocks as usize,
            h_samp: c.h_samp_factor as usize,
            v_samp: c.v_samp_factor as usize,
            quant,
            coefficients: std::mem::take(&mut c.coefficients),
        });
    }
    Ok(result)
}

/// Encode a supported 8-bit Huffman DCT JPEG. Verification reconstructs and
/// compares every original byte before this function returns an encoded file.
/// There is no opaque-input or pixel fallback. The format remains experimental.
pub fn encode_jpeg(input: &[u8], options: &EncodeOptions) -> Result<Vec<u8>> {
    let limits = &options.limits;
    if input.len() > limits.max_input_bytes || input.len() > limits.max_output_bytes {
        return Err(Error::Limit("original JPEG bytes"));
    }
    if options.metadata_quality > 11 {
        return Err(Error::Invalid("metadata quality must be 0..=11"));
    }
    // Structural parsing allocates no coefficient planes. Reject the pixel
    // ceiling before the coefficient reader allocates or scans the image.
    let headers = read_jpeg_with_limits(input, &structural(limits)).map_err(jpeg_error)?;
    dimensions(headers.width, headers.height, limits)?;
    drop(headers);
    let read_limits = CoefficientReadLimits {
        structural: structural(limits),
        max_coefficient_bytes: limits.max_coefficient_bytes,
        max_block_visits: limits.max_block_visits,
        max_hint_entries: limits.max_hint_entries,
        max_padding_bits: limits.max_metadata_bytes.saturating_mul(8),
        ..CoefficientReadLimits::default()
    };
    let mut jpeg = read_coefficients_with_limits(input, &read_limits).map_err(jpeg_error)?;
    dimensions(jpeg.width, jpeg.height, limits)?;
    let parts =
        serialize_reconstruction_metadata(&jpeg, limits.max_metadata_bytes).map_err(jpeg_error)?;
    let metadata = compress_parts(
        &parts.header,
        &parts.payload,
        options.metadata_quality,
        limits.max_metadata_bytes,
    )
    .map_err(jpeg_error)?;
    let width = jpeg.width;
    let height = jpeg.height;
    let components = components_from_jpeg(&mut jpeg)?;
    drop(jpeg);
    check_components(width, height, &components, limits)?;
    let bands = model::band_count(&components)?;
    let streams = model::encode(&components, limits.max_output_bytes)?;
    if streams.len() != bands {
        return Err(Error::Internal("model band count"));
    }
    let mut out = Output {
        bytes: Vec::new(),
        limit: limits.max_output_bytes,
    };
    out.bytes(MAGIC)?;
    out.u16(VERSION)?;
    out.u16(0)?;
    out.u32(width)?;
    out.u32(height)?;
    out.u64(input.len() as u64)?;
    out.bytes(&[components.len() as u8, bands as u8, 0, 0])?;
    out.u64(metadata.len() as u64)?;
    for c in &components {
        out.bytes(&[c.h_samp as u8, c.v_samp as u8, 0, 0])?;
        out.u32(c.width as u32)?;
        out.u32(c.height as u32)?;
        for q in c.quant {
            out.u16(q)?;
        }
    }
    for s in &streams {
        out.u64(s.len() as u64)?;
    }
    out.bytes(&metadata)?;
    for s in &streams {
        out.bytes(s)?;
    }
    drop(components);
    drop(streams);
    // Encoded input and original output have independent limits in the decoder.
    let verify_limits = DecodeLimits {
        max_input_bytes: limits.max_output_bytes,
        ..limits.clone()
    };
    if decode_jpeg(&out.bytes, &verify_limits)? != input {
        return Err(Error::Internal("original JPEG verification failed"));
    }
    Ok(out.bytes)
}

/// Decode one experimental version-1 file. Length, resource and syntax checks
/// apply; no content hash is stored or checked. A structurally valid alteration
/// can produce different original bytes. Encoding still compares its own
/// decoded output against every input byte before returning.
pub fn decode_jpeg(input: &[u8], limits: &DecodeLimits) -> Result<Vec<u8>> {
    if input.len() > limits.max_input_bytes {
        return Err(Error::Limit("encoded bytes"));
    }
    let mut r = Reader {
        bytes: input,
        offset: 0,
    };
    if r.take(8)? != MAGIC {
        return Err(Error::Invalid("magic signature"));
    }
    if r.u16()? != VERSION {
        return Err(Error::Unsupported("format version"));
    }
    if r.u16()? != 0 {
        return Err(Error::Unsupported("format flags"));
    }
    let width = r.u32()?;
    let height = r.u32()?;
    dimensions(width, height, limits)?;
    let original_len = r.length()?;
    if original_len > limits.max_output_bytes {
        return Err(Error::Limit("original JPEG bytes"));
    }
    let n = r.take(1)?[0] as usize;
    let bands = r.take(1)?[0] as usize;
    if r.u16()? != 0 {
        return Err(Error::Unsupported("header reserved bits"));
    }
    if !matches!(n, 1 | 3) || !(1..=16).contains(&bands) {
        return Err(Error::Invalid("component or band count"));
    }
    let metadata_len = r.length()?;
    if metadata_len > limits.max_metadata_bytes {
        return Err(Error::Limit("metadata bytes"));
    }
    let mut components = Vec::with_capacity(n);
    for _ in 0..n {
        let h_samp = r.take(1)?[0] as usize;
        let v_samp = r.take(1)?[0] as usize;
        if r.u16()? != 0 {
            return Err(Error::Unsupported("component reserved bits"));
        }
        let cw = r.u32()? as usize;
        let ch = r.u32()? as usize;
        let mut quant = [0; 64];
        for q in &mut quant {
            *q = r.u16()?;
        }
        components.push(Component {
            width: cw,
            height: ch,
            h_samp,
            v_samp,
            quant,
            coefficients: Vec::new(),
        });
    }
    if check_components(width, height, &components, limits)? != bands {
        return Err(Error::Invalid("model band count"));
    }
    let mut lengths = Vec::with_capacity(bands);
    for _ in 0..bands {
        let length = r.length()?;
        if length < 4 {
            return Err(Error::Invalid("short arithmetic stream"));
        }
        lengths.push(length);
    }
    let metadata = r.take(metadata_len)?;
    let mut streams = Vec::with_capacity(bands);
    for length in lengths {
        streams.push(r.take(length)?);
    }
    if r.offset != input.len() {
        return Err(Error::Invalid("trailing container bytes"));
    }
    // No coefficient allocation until all lengths, geometry and budget checks pass.
    for c in &mut components {
        let count = c.width * c.height * 64;
        c.coefficients
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("coefficient allocation"))?;
        c.coefficients.resize(count, 0);
    }
    model::decode_strict(&mut components, &streams)?;
    let captured = CapturedJpegCoefficients {
        width,
        height,
        components: components
            .into_iter()
            .map(|c| CapturedJpegComponent {
                h_samp_factor: c.h_samp as u32,
                v_samp_factor: c.v_samp as u32,
                width_in_blocks: c.width as u32,
                height_in_blocks: c.height as u32,
                coefficients: c.coefficients,
                quantization: c.quant.map(i32::from),
            })
            .collect(),
    };
    let jpeg = reconstruct_jpeg(
        metadata,
        captured,
        ReconstructionMetadataSources::default(),
        &reconstruction_limits(limits),
    )
    .map_err(jpeg_error)?;
    if jpeg.len() != original_len {
        return Err(Error::Invalid("original JPEG size"));
    }
    Ok(jpeg)
}

struct Output {
    bytes: Vec<u8>,
    limit: usize,
}
impl Output {
    fn bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(Error::Limit("encoded bytes"));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|_| Error::Limit("container allocation"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn u16(&mut self, n: u16) -> Result<()> {
        self.bytes(&n.to_le_bytes())
    }
    fn u32(&mut self, n: u32) -> Result<()> {
        self.bytes(&n.to_le_bytes())
    }
    fn u64(&mut self, n: u64) -> Result<()> {
        self.bytes(&n.to_le_bytes())
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(n)
            .ok_or(Error::Invalid("length overflow"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(Error::Invalid("truncated container"))?;
        self.offset = end;
        Ok(bytes)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn length(&mut self) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| Error::Limit("length exceeds address space"))
    }
}
