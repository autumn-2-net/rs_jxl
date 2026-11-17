//! Complete internal pixel experiment, NOT the final JXL extension format.
//! JXL RCT/MA/WP/HybridUint/ANS tools plus the PJA adaptive binary counter.
//! Every group resets predictors and gates. Channels share gate state within
//! a group; gate context is MA leaf x previous-zero for that leaf. Profile 2
//! fixes the counter recurrence, 32768/0 initialization and 256-square tiles.
use crate::arith::{AdaptiveBit, BitCoder, Decoder, Encoder};
use rs_jxl::entropy::{
    HybridUintConfig,
    experimental_decode::{EntropyDecoder, decode_tree},
    options::{EntropyOptions, UintConfigStrategy},
    stream::{Coding, Model},
};
use rs_jxl::modular::{
    encode::ModularOptions,
    group::Transform,
    plus_bridge::{self, PlaneLayout, Rect},
    weighted::Header,
};
use rs_jxl::{Error, Image, PixelData, Result, bit_writer::BitWriter};

const MAGIC: &[u8; 8] = b"JXLPPX02";
const TILE: usize = 256;
const MAX_BYTES: usize = 256 << 20;
const MAX_SAMPLES: usize = 32 << 20;
const MAX_GROUP_BYTES: usize = 8 << 20;
const MAX_LEAVES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    Static,
    Fitted,
    Adaptive,
    StaticLz,
    FittedLz,
    AdaptiveLz,
    /// Encoder policy only. No selection algorithm is needed by the decoder.
    Selected,
}
impl Mode {
    fn gate(self) -> Self {
        match self {
            Self::Static | Self::StaticLz => Self::Static,
            Self::Fitted | Self::FittedLz => Self::Fitted,
            Self::Adaptive | Self::AdaptiveLz => Self::Adaptive,
            Self::Selected => unreachable!("selection is resolved before group encoding"),
        }
    }
    fn lz(self) -> bool {
        matches!(self, Self::StaticLz | Self::FittedLz | Self::AdaptiveLz)
    }
}
#[derive(Default, Debug)]
pub struct Stats {
    pub groups: usize,
    pub lz_groups: usize,
    pub selected_groups: [usize; 6],
    pub tree_bytes: usize,
    pub wp_bytes: usize,
    pub gate_bytes: usize,
    pub table_bytes: usize,
    pub entropy_header_bytes: usize,
    pub entropy_data_bytes: usize,
}
impl Stats {
    fn add(&mut self, other: &Self) {
        self.groups += other.groups;
        self.lz_groups += other.lz_groups;
        for (a, b) in self.selected_groups.iter_mut().zip(other.selected_groups) {
            *a += b;
        }
        self.tree_bytes += other.tree_bytes;
        self.wp_bytes += other.wp_bytes;
        self.gate_bytes += other.gate_bytes;
        self.table_bytes += other.table_bytes;
        self.entropy_header_bytes += other.entropy_header_bytes;
        self.entropy_data_bytes += other.entropy_data_bytes;
    }
}
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub stats: Stats,
}
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub bits: u8,
    pub channels: usize,
    pub pixels: Vec<u32>,
}
fn invalid(s: &str) -> Error {
    Error::InvalidInput(format!("pixel experiment: {s}"))
}
fn codec_error(e: crate::Error) -> Error {
    invalid(&e.to_string())
}
fn put32(out: &mut Vec<u8>, n: usize) -> Result<()> {
    let n = u32::try_from(n).map_err(|_| invalid("length exceeds u32"))?;
    out.extend_from_slice(&n.to_le_bytes());
    Ok(())
}
fn section(out: &mut Vec<u8>, write: impl FnOnce(&mut BitWriter) -> Result<()>) -> Result<usize> {
    let mut w = BitWriter::new(MAX_GROUP_BYTES);
    write(&mut w)?;
    let bits = w.bits_written();
    w.zero_pad_to_byte()?;
    let bytes = w.into_bytes()?;
    put32(out, bits)?;
    out.extend_from_slice(&bytes);
    Ok(4 + bytes.len())
}
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&v| v <= self.bytes.len())
            .ok_or_else(|| invalid("truncated section"))?;
        let bytes = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn section(&mut self) -> Result<(&'a [u8], usize)> {
        let bits = self.u32()?;
        if bits > MAX_GROUP_BYTES * 8 {
            return Err(invalid("section limit"));
        }
        let bytes = self.take(bits.div_ceil(8))?;
        if bits % 8 != 0 && bytes.last().is_some_and(|b| b >> (bits % 8) != 0) {
            return Err(invalid("nonzero bit padding"));
        }
        Ok((bytes, bits))
    }
    fn finish(&self) -> Result<()> {
        if self.pos != self.bytes.len() {
            Err(invalid("trailing bytes"))
        } else {
            Ok(())
        }
    }
}
fn read_wp(bytes: &[u8], bits: usize) -> Result<Header> {
    if bits == 1 && bytes == [1] {
        return Ok(Header::default());
    }
    if bits != 52 || bytes[0] & 1 != 0 {
        return Err(invalid("WP header"));
    }
    let mut pos = 1;
    let mut read = |count: usize| {
        let mut n = 0;
        for i in 0..count {
            n |= u32::from((bytes[pos / 8] >> (pos % 8)) & 1) << i;
            pos += 1;
        }
        n
    };
    let h = Header {
        p1_c: read(5) as i32,
        p2_c: read(5) as i32,
        p3_ca: read(5) as i32,
        p3_cb: read(5) as i32,
        p3_cc: read(5) as i32,
        p3_cd: read(5) as i32,
        p3_ce: read(5) as i32,
        w: [read(4), read(4), read(4), read(4)],
    };
    h.validate()?;
    Ok(h)
}
fn geometry(w: usize, h: usize, c: usize, max_samples: usize) -> Result<usize> {
    if w == 0 || h == 0 || !(1..=4).contains(&c) {
        return Err(invalid("geometry"));
    }
    w.checked_mul(h)
        .and_then(|n| n.checked_mul(c))
        .filter(|&n| n <= max_samples)
        .ok_or_else(|| invalid("sample budget"))
}

fn encode_group(
    prepared: &plus_bridge::PreparedGroup,
    requested: Mode,
    width: usize,
    entropy_options: &EntropyOptions,
) -> Result<Encoded> {
    let mode = requested.gate();
    let leaves = prepared.tree.leaves();
    if leaves == 0 || leaves > MAX_LEAVES {
        return Err(invalid("tree leaf limit"));
    }
    let mut stats = Stats {
        groups: 1,
        ..Stats::default()
    };
    let mut group = vec![mode as u8]; // bit 2 is set only when the entropy header enables LZ.
    group.push(prepared.transforms.len() as u8);
    for t in &prepared.transforms {
        let (begin, kind) = match t {
            Transform::Rct(k) => (0, *k),
            Transform::RctAt { begin, kind } => (*begin, *kind),
            _ => return Err(invalid("transform profile")),
        };
        group.extend_from_slice(&[begin as u8, kind as u8]);
    }
    stats.wp_bytes = section(&mut group, |w| prepared.wp.write_header(w))?;
    stats.tree_bytes = section(&mut group, |w| prepared.tree.write(w))?;
    let mut previous = vec![false; leaves];
    let mut fitted = vec![32768u16; leaves * 2];
    if mode == Mode::Fitted {
        let mut counts = vec![[0u64; 2]; leaves * 2];
        for t in &prepared.tokens {
            let zero = t.value == 0;
            counts[2 * t.context + usize::from(previous[t.context])][usize::from(zero)] += 1;
            previous[t.context] = zero;
        }
        for (p, c) in fitted.iter_mut().zip(counts) {
            let total = c[0] + c[1];
            if total != 0 {
                *p = ((c[1] * 65536 + total / 2) / total).clamp(1, 65535) as u16;
            }
        }
        previous.fill(false);
    }
    let mut adaptive = vec![AdaptiveBit::default(); leaves * 2];
    let mut coder = Encoder::new(MAX_GROUP_BYTES);
    let mut tokens = Vec::with_capacity(prepared.tokens.len());
    for t in &prepared.tokens {
        if mode != Mode::Static {
            let key = 2 * t.context + usize::from(previous[t.context]);
            let zero = t.value == 0;
            let p = if mode == Mode::Adaptive {
                adaptive[key].p
            } else {
                fitted[key]
            };
            coder.bit(u32::from(p), i32::from(zero));
            if mode == Mode::Adaptive {
                adaptive[key].update(i32::from(zero));
            }
            previous[t.context] = zero;
            if zero {
                continue;
            }
        }
        tokens.push(*t);
    }
    let gate = if mode == Mode::Static {
        Vec::new()
    } else {
        coder.finish().map_err(codec_error)?
    };
    // Expanded output count, not the encoded LZ command/token count.
    put32(&mut group, tokens.len())?;
    put32(&mut group, gate.len())?;
    group.extend_from_slice(&gate);
    stats.gate_bytes = gate.len();
    let table_count = if mode == Mode::Fitted {
        fitted.len()
    } else {
        0
    };
    put32(&mut group, table_count)?;
    if mode == Mode::Fitted {
        for p in fitted {
            group.extend_from_slice(&p.to_le_bytes());
        }
    }
    stats.table_bytes = table_count * 2;
    let mut has_lz = false;
    if requested.lz() {
        let options = ModularOptions {
            lz77: true,
            source_ma_defaults: false,
            source_tool_defaults: false,
            entropy_options: *entropy_options,
            ..ModularOptions::default()
        };
        // A filtered stream has no row geometry; JXL's multiplier zero uses
        // direct backward distances. Full streams keep the original tile width.
        let multiplier = if mode == Mode::Static {
            width as u32
        } else {
            0
        };
        (tokens, has_lz) =
            plus_bridge::parse_residual_tokens(&tokens, leaves, multiplier, &options)?;
    }
    group[0] |= u8::from(has_lz) << 2;
    stats.lz_groups = usize::from(has_lz);
    stats.selected_groups[mode as usize + 3 * usize::from(has_lz)] = 1;
    let model = Model::build_with_options(
        std::slice::from_ref(&tokens),
        leaves + usize::from(has_lz),
        has_lz,
        Coding::Ans,
        entropy_options,
    )?;
    stats.entropy_header_bytes = section(&mut group, |w| model.write_header(w))?;
    stats.entropy_data_bytes = section(&mut group, |w| model.write_stream(&tokens, w))?;
    if group.len() > MAX_GROUP_BYTES {
        return Err(invalid("group output budget"));
    }
    Ok(Encoded {
        bytes: group,
        stats,
    })
}

pub fn encode(image: &Image, mode: Mode) -> Result<Encoded> {
    let (width, height, channels) = (
        image.width() as usize,
        image.height() as usize,
        image.channel_count(),
    );
    geometry(width, height, channels, MAX_SAMPLES)?;
    let bits = image.bits();
    if !matches!(
        image.pixels(),
        PixelData::U8(_) | PixelData::U16(_) | PixelData::U32(_)
    ) || !(1..=31).contains(&bits)
        || !image.extra_planes().is_empty()
        || channels != image.layout().channels()
        || (0..channels).any(|c| {
            image.channel_depth(c).bits_per_sample != bits
                || image.channel_depth(c).exponent_bits_per_sample != 0
                || image.channel_shift(c) != 0
        })
    {
        return Err(invalid("requires equal-depth interleaved integer channels"));
    }
    let options = ModularOptions {
        lz77: false,
        source_ma_defaults: false,
        source_tool_defaults: false,
        rct: if image.color_channels() == 3 {
            Some(6)
        } else {
            None
        },
        max_tree_leaves: MAX_LEAVES,
        ..ModularOptions::default()
    };
    let entropy_options = EntropyOptions {
        uint_config: UintConfigStrategy::Fixed(HybridUintConfig::default()),
        ..EntropyOptions::default()
    };
    let groups = width.div_ceil(TILE) * height.div_ceil(TILE);
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&[
        0, // reserved: encoder selection policy is not decoder state
        bits,
        image.color_channels() as u8,
        channels as u8,
    ]);
    put32(&mut out, width)?;
    put32(&mut out, height)?;
    out.extend_from_slice(&(TILE as u16).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // deterministic model profile
    put32(&mut out, groups)?;
    let mut stats = Stats::default();
    for y in (0..height).step_by(TILE) {
        for x in (0..width).step_by(TILE) {
            let rect = Rect {
                x,
                y,
                width: (width - x).min(TILE),
                height: (height - y).min(TILE),
                stream_id: stats.groups + 1,
            };
            let prepared = plus_bridge::prepare(image, rect, &options)?;
            let encoded = if mode == Mode::Selected {
                let mut best = encode_group(&prepared, Mode::Static, rect.width, &entropy_options)?;
                for candidate in [Mode::Adaptive, Mode::StaticLz, Mode::AdaptiveLz] {
                    let encoded = encode_group(&prepared, candidate, rect.width, &entropy_options)?;
                    // Equal lengths retain the earlier, simpler candidate.
                    if encoded.bytes.len() < best.bytes.len() {
                        best = encoded;
                    }
                }
                best
            } else {
                encode_group(&prepared, mode, rect.width, &entropy_options)?
            };
            if out.len() + encoded.bytes.len() + 4 > MAX_BYTES {
                return Err(invalid("output budget"));
            }
            put32(&mut out, encoded.bytes.len())?;
            out.extend_from_slice(&encoded.bytes);
            stats.add(&encoded.stats);
        }
    }
    Ok(Encoded { bytes: out, stats })
}

pub fn decode(bytes: &[u8]) -> Result<Decoded> {
    decode_with_limits(bytes, MAX_BYTES, MAX_SAMPLES)
}
pub fn decode_with_limits(bytes: &[u8], max_bytes: usize, max_samples: usize) -> Result<Decoded> {
    if bytes.len() > max_bytes.min(MAX_BYTES) {
        return Err(invalid("input budget"));
    }
    let mut r = Reader::new(bytes);
    if r.take(8)? != MAGIC {
        return Err(invalid("magic"));
    }
    if r.byte()? != 0 {
        return Err(invalid("reserved global flags"));
    }
    let bits = r.byte()?;
    let colors = r.byte()? as usize;
    let channels = r.byte()? as usize;
    if !(1..=31).contains(&bits)
        || ![1, 3].contains(&colors)
        || !(colors..=colors + 1).contains(&channels)
    {
        return Err(invalid("sample descriptor"));
    }
    let width = r.u32()?;
    let height = r.u32()?;
    let count = geometry(width, height, channels, max_samples.min(MAX_SAMPLES))?;
    if usize::from(r.u16()?) != TILE || r.u16()? != 2 {
        return Err(invalid("model profile"));
    }
    let groups = r.u32()?;
    if groups != width.div_ceil(TILE) * height.div_ceil(TILE) {
        return Err(invalid("group count"));
    }
    // At least length + entropy flags + transform count must be present.
    if groups > (bytes.len() - r.pos) / 6 {
        return Err(invalid("missing groups"));
    }
    // Validate all group boundaries before allocating the full image, even
    // though individual models are parsed later in raster order.
    let mut framing = Reader::new(&bytes[r.pos..]);
    for _ in 0..groups {
        let len = framing.u32()?;
        if len > MAX_GROUP_BYTES {
            return Err(invalid("group bytes"));
        }
        framing.take(len)?;
    }
    framing.finish()?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(count)
        .map_err(|_| invalid("sample allocation"))?;
    pixels.resize(count, 0u32);
    let maximum = (1u32 << bits) - 1;
    let mut index = 0;
    for y in (0..height).step_by(TILE) {
        for x in (0..width).step_by(TILE) {
            let len = r.u32()?;
            if len > MAX_GROUP_BYTES {
                return Err(invalid("group bytes"));
            }
            let mut g = Reader::new(r.take(len)?);
            let flags = g.byte()?;
            if flags & !7 != 0 || flags & 3 == 3 {
                return Err(invalid("group entropy flags"));
            }
            let mode = match flags & 3 {
                0 => Mode::Static,
                1 => Mode::Fitted,
                2 => Mode::Adaptive,
                _ => unreachable!(),
            };
            let has_lz = flags & 4 != 0;
            let transforms = g.byte()? as usize;
            if transforms > 1 {
                return Err(invalid("transform count"));
            }
            let mut transform_list = Vec::new();
            for _ in 0..transforms {
                let begin = usize::from(g.byte()?);
                let kind = u32::from(g.byte()?);
                if begin != 0 || kind != 6 || colors != 3 {
                    return Err(invalid("RCT profile"));
                }
                transform_list.push(Transform::RctAt { begin, kind });
            }
            let (wp, bits_wp) = g.section()?;
            let wp = read_wp(wp, bits_wp)?;
            let (tree, bits_tree) = g.section()?;
            let tree = decode_tree(tree, bits_tree, MAX_LEAVES * 2 - 1)?;
            let leaves = tree.leaves();
            let tw = (width - x).min(TILE);
            let th = (height - y).min(TILE);
            let samples = tw * th * channels;
            let symbols = g.u32()?;
            if symbols > samples || (mode == Mode::Static && symbols != samples) {
                return Err(invalid("symbol count"));
            }
            let gate_len = g.u32()?;
            let gate = g.take(gate_len)?;
            if (mode == Mode::Static && !gate.is_empty())
                || (mode != Mode::Static && gate.len() < 4)
            {
                return Err(invalid("gate section"));
            }
            let table_count = g.u32()?;
            if table_count != if mode == Mode::Fitted { leaves * 2 } else { 0 } {
                return Err(invalid("gate table count"));
            }
            let mut fitted = Vec::with_capacity(table_count);
            for _ in 0..table_count {
                let p = g.u16()?;
                if p == 0 {
                    return Err(invalid("zero probability"));
                }
                fitted.push(p);
            }
            let (header, header_bits) = g.section()?;
            let (data, data_bits) = g.section()?;
            g.finish()?;
            let mut entropy = if has_lz {
                EntropyDecoder::new_with_lz(
                    header,
                    header_bits,
                    data,
                    data_bits,
                    leaves,
                    symbols,
                    if mode == Mode::Static { tw as u32 } else { 0 },
                )?
            } else {
                EntropyDecoder::new(header, header_bits, data, data_bits, leaves, symbols)?
            };
            if entropy.has_lz77() != has_lz {
                return Err(invalid("LZ header/descriptor mismatch"));
            }
            let mut gate = Decoder::new(gate);
            let mut adaptive = vec![AdaptiveBit::default(); leaves * 2];
            let mut previous = vec![false; leaves];
            let layout = vec![
                PlaneLayout {
                    width: tw,
                    height: th,
                    hshift: 0,
                    vshift: 0
                };
                channels
            ];
            let mut planes = plus_bridge::decode_planes(
                &layout,
                &tree,
                &wp,
                index + 1,
                0,
                samples,
                |context| {
                    if context >= leaves {
                        return Err(invalid("gate context"));
                    }
                    if mode != Mode::Static {
                        let key = 2 * context + usize::from(previous[context]);
                        let p = if mode == Mode::Adaptive {
                            adaptive[key].p
                        } else {
                            fitted[key]
                        };
                        let bit = gate.bit(u32::from(p), 0);
                        if mode == Mode::Adaptive {
                            adaptive[key].update(bit);
                        }
                        previous[context] = bit != 0;
                        if bit != 0 {
                            return Ok(0);
                        }
                    }
                    let value = entropy.read(context)?;
                    if mode != Mode::Static && value == 0 {
                        return Err(invalid("zero in nonzero stream"));
                    }
                    Ok(value)
                },
            )?;
            entropy.finish()?;
            if mode != Mode::Static {
                gate.finish_check().map_err(codec_error)?;
            }
            plus_bridge::inverse_transforms(&mut planes, &transform_list)?;
            for (c, plane) in planes.iter().enumerate() {
                for row in 0..th {
                    for col in 0..tw {
                        let sample = plane.samples[row * tw + col];
                        if sample < 0 || sample as u32 > maximum {
                            return Err(invalid("sample out of range"));
                        }
                        pixels[((y + row) * width + x + col) * channels + c] = sample as u32;
                    }
                }
            }
            index += 1;
        }
    }
    r.finish()?;
    Ok(Decoded {
        width: width as u32,
        height: height as u32,
        bits,
        channels,
        pixels,
    })
}
