// PJA MIT translation, including IJG-derived arithmetic JPEG algorithms.
// See ../LICENSE-PJA and ../NOTICE-PJA.
//! Native scalar port of pas/pjajpeg.pas at
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
use crate::{Error, Result};
mod arithmetic;
mod bits;
mod huffman;
mod lossless;
mod parser;
use bits::*;
pub(crate) use huffman::Huff;

pub(crate) const LIBJPEG_MAX_BE: usize = 937;
pub(crate) const MAX_BE_LIMIT: usize = 4032;
pub(crate) const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_blocks: usize,
    pub max_samples: usize,
    pub max_scans: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_input_bytes: 1 << 30,
            max_output_bytes: 1 << 30,
            max_blocks: 1 << 22,
            max_samples: 1 << 27,
            max_scans: 1024,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub(crate) struct Component {
    pub id: u8,
    pub h: usize,
    pub v: usize,
    pub tq: usize,
    pub bw: usize,
    pub bh: usize,
    pub bw_real: usize,
    pub bh_real: usize,
    pub coef: Vec<i16>,
    pub samples: Vec<u16>,
}
#[derive(Clone, Debug)]
pub(crate) struct Scan {
    pub comp: Vec<usize>,
    pub dc: [Huff; 4],
    pub ac: [Huff; 4],
    pub td: [usize; 4],
    pub ta: [usize; 4],
    pub dc_l: [usize; 4],
    pub dc_u: [usize; 4],
    pub ac_k: [usize; 4],
    pub ss: usize,
    pub se: usize,
    pub ah: usize,
    pub al: usize,
    pub ri: usize,
    pub max_eob: usize,
    pub padbit: i32,
    pub max_be: usize,
    pub ignore_ri: bool,
    pub wrap_pred: bool,
    pub cut: usize,
    pub data_start: usize,
    pub data_end: usize,
}
impl Default for Scan {
    fn default() -> Self {
        Self {
            comp: Vec::new(),
            dc: std::array::from_fn(|_| Huff::default()),
            ac: std::array::from_fn(|_| Huff::default()),
            td: [0; 4],
            ta: [0; 4],
            dc_l: [0; 4],
            dc_u: [1; 4],
            ac_k: [5; 4],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
            ri: 0,
            max_eob: 1,
            padbit: 1,
            max_be: LIBJPEG_MAX_BE,
            ignore_ri: false,
            wrap_pred: false,
            cut: 0,
            data_start: 0,
            data_end: 0,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct Jpeg {
    pub width: usize,
    pub height: usize,
    pub precision: usize,
    pub lossless: bool,
    pub dnl: bool,
    pub progressive: bool,
    pub arithmetic: bool,
    pub components: Vec<Component>,
    pub hmax: usize,
    pub vmax: usize,
    pub mcusx: usize,
    pub mcusy: usize,
    pub qt: [[u16; 64]; 4],
    pub qpresent: [bool; 4],
    pub scans: Vec<Scan>,
    pub raw: Vec<u8>,
    pub eoi: usize,
    pub scans_exact: bool,
}
impl Default for Jpeg {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            precision: 0,
            lossless: false,
            dnl: false,
            progressive: false,
            arithmetic: false,
            components: Vec::new(),
            hmax: 0,
            vmax: 0,
            mcusx: 0,
            mcusy: 0,
            qt: [[0; 64]; 4],
            qpresent: [false; 4],
            scans: Vec::new(),
            raw: Vec::new(),
            eoi: 0,
            scans_exact: false,
        }
    }
}
fn fail(what: &str, pos: usize) -> Error {
    Error::Jpeg(format!("{what}@{pos}"))
}
fn zeros<T: Default + Clone>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Limit("JPEG allocation"))?;
    v.resize(n, T::default());
    Ok(v)
}
fn category(v: i32) -> usize {
    (32 - v.unsigned_abs().leading_zeros()) as usize
}
fn extend(v: i32, n: usize) -> i32 {
    if n > 0 && v < (1 << (n - 1)) {
        v - (1 << n) + 1
    } else {
        v
    }
}
fn coefficient(v: i32) -> Result<i16> {
    if !(-32767..=32767).contains(&v) {
        Err(Error::Invalid("JPEG coefficient out of range"))
    } else {
        Ok(v as i16)
    }
}
fn units(j: &Jpeg, s: &Scan) -> (usize, usize) {
    if s.comp.len() == 1 {
        let c = &j.components[s.comp[0]];
        (c.bw_real, c.bw_real * c.bh_real)
    } else {
        (j.mcusx, j.mcusx * j.mcusy)
    }
}
fn blocks(
    j: &Jpeg,
    s: &Scan,
    u: usize,
) -> std::iter::Take<std::array::IntoIter<(usize, usize, usize), 10>> {
    let (w, _) = units(j, s);
    let (x, y) = (u % w, u / w);
    let mut b = [(0, 0, 0); 10];
    let mut n = 0;
    for (i, &ci) in s.comp.iter().enumerate() {
        let c = &j.components[ci];
        if s.comp.len() == 1 {
            b[n] = (i, ci, y * c.bw + x);
            n += 1;
        } else {
            for yy in 0..c.v {
                for xx in 0..c.h {
                    b[n] = (i, ci, (y * c.v + yy) * c.bw + x * c.h + xx);
                    n += 1;
                }
            }
        }
    }
    b.into_iter().take(n)
}
fn segment_end(d: &[u8], mut p: usize) -> usize {
    while p < d.len() {
        if d[p] == 255 {
            let mut r = p + 1;
            while r < d.len() && d[r] == 255 {
                r += 1;
            }
            if r < d.len() && d[r] == 0 {
                p = r + 1;
                continue;
            }
            return p;
        }
        p += 1;
    }
    p
}
fn decode_scan(data: &[u8], first: usize, j: &mut Jpeg, s: &mut Scan) -> Result<usize> {
    if j.lossless {
        lossless::decode(data, first, j, s)
    } else if j.arithmetic {
        arithmetic::decode(data, first, j, s)
    } else {
        huffman::decode(data, first, j, s)
    }
}
fn encode_scan(w: &mut Writer, j: &Jpeg, s: &Scan) -> Result<()> {
    if j.lossless {
        lossless::encode(w, j, s)
    } else if j.arithmetic {
        arithmetic::encode(w, j, s)
    } else {
        huffman::encode(w, j, s)
    }
}
pub(crate) fn decompose(data: &[u8], limits: &Limits) -> Result<Jpeg> {
    parser::parse(data, true, limits)
}
pub(crate) fn parse_headers(raw: &[u8], limits: &Limits) -> Result<Jpeg> {
    parser::parse(raw, false, limits)
}
pub(crate) fn recompose(j: &Jpeg, limits: &Limits) -> Result<Vec<u8>> {
    Ok(recompose_at(j, limits)?.0)
}
fn recompose_at(j: &Jpeg, limits: &Limits) -> Result<(Vec<u8>, Vec<(usize, usize)>)> {
    validate(j, limits)?;
    let mut w = Writer::new(limits.max_output_bytes);
    let mut from = 0;
    let mut ends = Vec::new();
    for s in &j.scans {
        if s.cut < from || s.cut > j.raw.len() {
            return Err(Error::Invalid("JPEG scan cut"));
        }
        w.copy(&j.raw[from..s.cut])?;
        from = s.cut;
        let first = w.data.len();
        encode_scan(&mut w, j, s)?;
        ends.push((first, w.data.len()));
    }
    w.copy(&j.raw[from..])?;
    Ok((w.data, ends))
}
pub(crate) fn verify_round_trip(data: &[u8], j: &Jpeg, limits: &Limits) -> Result<()> {
    let other = recompose(j, limits)?;
    if other != data {
        let p = other
            .iter()
            .zip(data)
            .position(|(a, b)| a != b)
            .unwrap_or(other.len().min(data.len()));
        return Err(fail("NoRoundTrip", p));
    }
    Ok(())
}
pub(crate) fn decompose_verified(data: &[u8], limits: &Limits) -> Result<Jpeg> {
    let mut j = decompose(data, limits)?;
    let (other, ends) = recompose_at(&j, limits)?;
    if other != data {
        return Err(Error::Invalid("JPEG byte roundtrip mismatch"));
    }
    j.scans_exact = j
        .scans
        .iter()
        .zip(ends)
        .all(|(s, (a, b))| s.data_start == a && s.data_end == b);
    Ok(j)
}
pub(crate) fn may_wrap_pred(j: &Jpeg, s: &Scan) -> bool {
    j.lossless && (s.ss == 5 || s.ss == 6) && j.precision - s.al == 16
}
pub(crate) fn has_ignored_restarts(j: &Jpeg) -> bool {
    j.scans.iter().any(|s| s.ignore_ri)
}
pub(crate) fn has_wrapped_predictor(j: &Jpeg) -> bool {
    j.scans.iter().any(|s| s.wrap_pred)
}
pub(crate) fn has_correction_limits(j: &Jpeg) -> bool {
    j.scans.iter().any(|s| s.max_be != LIBJPEG_MAX_BE)
}

fn validate(j: &Jpeg, limits: &Limits) -> Result<()> {
    if j.width == 0
        || j.height == 0
        || j.width > 65535
        || j.height > 65535
        || j.components.is_empty()
        || j.components.len() > 4
        || j.hmax == 0
        || j.vmax == 0
        || j.hmax > 4
        || j.vmax > 4
    {
        return Err(Error::Invalid("JPEG frame dimensions"));
    }
    if if j.lossless {
        !(2..=16).contains(&j.precision)
    } else {
        j.precision != 8 && j.precision != 12
    } {
        return Err(Error::Invalid("JPEG precision"));
    }
    let scale = if j.lossless { 1 } else { 8 };
    if j.mcusx != j.width.div_ceil(scale * j.hmax) || j.mcusy != j.height.div_ceil(scale * j.vmax) {
        return Err(Error::Invalid("JPEG MCU dimensions"));
    }
    let mut count = 0usize;
    for c in &j.components {
        if c.h == 0
            || c.h > j.hmax
            || c.v == 0
            || c.v > j.vmax
            || c.tq > 3
            || c.bw != j.mcusx * c.h
            || c.bh != j.mcusy * c.v
            || c.bw_real != (j.width * c.h).div_ceil(j.hmax).div_ceil(scale)
            || c.bh_real != (j.height * c.v).div_ceil(j.vmax).div_ceil(scale)
        {
            return Err(Error::Invalid("JPEG component grid"));
        }
        let n = c.bw * c.bh;
        count = count
            .checked_add(n)
            .ok_or(Error::Limit("JPEG component storage"))?;
        if if j.lossless {
            c.samples.len() != n
        } else {
            c.coef.len() != n * 64 || c.coef.contains(&i16::MIN)
        } {
            return Err(Error::Invalid("JPEG component storage"));
        }
    }
    if count
        > if j.lossless {
            limits.max_samples
        } else {
            limits.max_blocks
        }
        || j.scans.len() > limits.max_scans
    {
        return Err(Error::Limit("JPEG reconstruction storage"));
    }
    for s in &j.scans {
        if s.comp.is_empty()
            || s.comp.len() > 4
            || s.comp.iter().any(|&c| c >= j.components.len())
            || s.comp.windows(2).any(|w| w[0] >= w[1])
            || s.al > 15
            || s.ah > 15
            || s.max_be > MAX_BE_LIMIT
            || s.max_eob == 0
            || s.max_eob > 32767
        {
            return Err(Error::Invalid("JPEG reconstruction scan"));
        }
        if s.comp.len() > 1
            && s.comp
                .iter()
                .map(|&c| j.components[c].h * j.components[c].v)
                .sum::<usize>()
                > 10
        {
            return Err(Error::Invalid("JPEG MCU block count"));
        }
        if j.lossless {
            let row = if s.comp.len() == 1 {
                j.components[s.comp[0]].bw_real
            } else {
                j.mcusx
            };
            if !(1..=7).contains(&s.ss)
                || s.se != 0
                || s.ah != 0
                || s.al >= j.precision
                || (s.ri > 0 && s.ri % row != 0)
            {
                return Err(Error::Invalid("JPEG lossless scan"));
            }
        } else if j.progressive {
            if (s.ss == 0 && s.se != 0)
                || (s.ss > 0 && (s.se < s.ss || s.se > 63 || s.comp.len() != 1))
                || (s.ah > 0 && s.al != s.ah - 1)
                || s.al > 13
            {
                return Err(Error::Invalid("JPEG progressive scan"));
            }
        } else if s.al != 0 || s.ah != 0 {
            return Err(Error::Invalid("JPEG sequential scan"));
        }
        for i in 0..s.comp.len() {
            if s.td[i] > 3
                || s.ta[i] > 3
                || s.dc_l[i] > s.dc_u[i]
                || s.dc_u[i] > 15
                || s.ac_k[i] == 0
                || s.ac_k[i] > 63
            {
                return Err(Error::Invalid("JPEG arithmetic conditioning"));
            }
        }
    }
    Ok(())
}

/// Same semantics as VerifyRoundTripFrom. The source's identical-scan reuse
/// optimization is deliberately omitted: the full decoder verifies the input.
pub(crate) fn verify_round_trip_from(
    data: &[u8],
    _reference: &Jpeg,
    _reference_data: &[u8],
    limits: &Limits,
) -> Result<Jpeg> {
    decompose_verified(data, limits)
}

/// APP2 MPF entry types 0x02xxxx identify multi-frame MPO, as IsMpo in PJA.
pub(crate) fn is_mpo(data: &[u8]) -> bool {
    fn get16(d: &[u8], p: usize, le: bool) -> Option<u16> {
        let a: [u8; 2] = d.get(p..p.checked_add(2)?)?.try_into().ok()?;
        Some(if le {
            u16::from_le_bytes(a)
        } else {
            u16::from_be_bytes(a)
        })
    }
    fn get32(d: &[u8], p: usize, le: bool) -> Option<u32> {
        let a: [u8; 4] = d.get(p..p.checked_add(4)?)?.try_into().ok()?;
        Some(if le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        })
    }
    fn inner(d: &[u8]) -> Option<bool> {
        if d.get(..2) != Some(&[255, 216]) {
            return Some(false);
        }
        let mut p = 2usize;
        while p + 4 <= d.len() {
            if d[p] != 255 {
                return Some(false);
            }
            let m = d[p + 1];
            if m == 218 || m == 217 {
                return Some(false);
            }
            let n = get16(d, p + 2, false)? as usize;
            let end = p.checked_add(2 + n)?;
            if n < 2 || end > d.len() {
                return Some(false);
            }
            if m == 226 && n >= 16 && d.get(p + 4..p + 8) == Some(b"MPF\0") {
                let t = p + 8;
                let le = match d.get(t..t + 2)? {
                    b"II" => true,
                    b"MM" => false,
                    _ => return Some(false),
                };
                let ifd = t.checked_add(get32(d, t + 4, le)? as usize)?;
                if ifd + 2 > end {
                    return Some(false);
                }
                let count = get16(d, ifd, le)? as usize;
                for k in 0..count {
                    let e = ifd + 2 + 12 * k;
                    if e + 12 > end {
                        return Some(false);
                    }
                    if get16(d, e, le)? == 0xb002 {
                        let off = t.checked_add(get32(d, e + 8, le)? as usize)?;
                        let images = get32(d, e + 4, le)? as usize / 16;
                        for i in 0..images {
                            let pos = off.checked_add(16 * i)?;
                            if pos.checked_add(16)? > d.len() {
                                return Some(false);
                            }
                            if (get32(d, pos, le)? >> 16) & 255 == 2 {
                                return Some(true);
                            }
                        }
                        return Some(false);
                    }
                }
                return Some(false);
            }
            p = end;
        }
        Some(false)
    }
    inner(data).unwrap_or(false)
}

#[cfg(test)]
mod tests;
