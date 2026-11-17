// PJA MIT translation; see ../../LICENSE-PJA and ../../NOTICE-PJA.
//! JPEG 2000 source port. Integration is tracked in docs/PJA_PORT_STATUS.md.
pub(crate) mod codestream;
pub(crate) mod ht;
pub(crate) mod ht_model;
pub(crate) mod model;
pub(crate) mod modes;
pub(crate) mod pipeline;
pub(crate) mod pixel_model;
#[cfg(test)]
mod pixel_oracle_tests;
pub(crate) mod pixel_tier1;
pub(crate) mod pixels;
pub(crate) mod tier1;
pub(crate) mod transform;

/// Locations use the original byte order, including trailing/opaque bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Codestream {
    pub off: usize,
    pub len: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    None,
    Codestream,
    Boxes,
}
const SIGNATURE: &[u8; 12] = b"\0\0\0\x0cjP  \r\n\x87\n";
fn be16(d: &[u8], p: usize) -> usize {
    u16::from_be_bytes(d[p..p + 2].try_into().unwrap()) as usize
}
fn be32(d: &[u8], p: usize) -> usize {
    u32::from_be_bytes(d[p..p + 4].try_into().unwrap()) as usize
}
pub(crate) fn kind(d: &[u8]) -> Kind {
    if d.starts_with(b"\xff\x4f\xff\x51") {
        Kind::Codestream
    } else if d.starts_with(SIGNATURE) {
        Kind::Boxes
    } else {
        Kind::None
    }
}
pub(crate) fn is_superbox(t: &[u8]) -> bool {
    [
        b"jp2h", b"res ", b"uinf", b"asoc", b"jpch", b"jplh", b"cgrp", b"comp", b"drep", b"page",
        b"pcol", b"lobj", b"objc", b"moov", b"trak", b"mdia", b"minf", b"stbl", b"dinf", b"edts",
        b"udta", b"mvex", b"moof", b"traf", b"mdat",
    ]
    .iter()
    .any(|v| v.as_slice() == t)
}
pub(crate) fn codestream_end(d: &[u8], off: usize, limit: usize) -> Option<usize> {
    if limit > d.len() || limit.saturating_sub(off) < 4 || d.get(off..off + 2)? != b"\xff\x4f" {
        return None;
    }
    let mut p = off + 2;
    loop {
        if limit.saturating_sub(p) < 4 || d[p] != 255 {
            return None;
        }
        let m = d[p + 1];
        if m == 0x90 {
            break;
        }
        if m < 0x40 || matches!(m, 0x4f | 0x93 | 0xd9) {
            return None;
        }
        let len = be16(d, p + 2);
        if len < 2 {
            return None;
        }
        p = p.checked_add(2 + len)?;
    }
    loop {
        if limit.saturating_sub(p) < 2 {
            return None;
        }
        if d[p..p + 2] == *b"\xff\xd9" {
            return Some(p + 2);
        }
        if limit - p < 12 || d[p..p + 2] != *b"\xff\x90" || be16(d, p + 2) != 10 {
            return None;
        }
        let len = be32(d, p + 6);
        if len == 0 {
            return (limit - p >= 14 && d[limit - 2..limit] == *b"\xff\xd9").then_some(limit);
        }
        if len < 14 || len > limit - p {
            return None;
        }
        p += len;
    }
}
fn walk(d: &[u8], a: usize, b: usize, depth: usize, count: &mut usize, cs: &mut Vec<Codestream>) {
    let mut p = a;
    while b - p >= 8 && *count < 1 << 20 {
        let l = be32(d, p);
        let mut head = 8;
        let len = if l == 1 {
            if b - p < 16 {
                return;
            }
            head = 16;
            let n = u64::from_be_bytes(d[p + 8..p + 16].try_into().unwrap());
            if n > (b - p) as u64 {
                return;
            }
            n as usize
        } else if l == 0 {
            b - p
        } else {
            l
        };
        if len < head || len > b - p {
            return;
        }
        *count += 1;
        let t = &d[p + 4..p + 8];
        if t == b"jp2c" {
            if len - head >= 4 && cs.len() < 16384 && d[p + head..].starts_with(b"\xff\x4f\xff\x51")
            {
                cs.push(Codestream {
                    off: p + head,
                    len: len - head,
                });
            }
        } else if depth < 16 && is_superbox(t) {
            walk(d, p + head, p + len, depth + 1, count, cs);
        }
        p += len;
    }
}
pub(crate) fn codestreams(d: &[u8]) -> Option<Vec<Codestream>> {
    match kind(d) {
        Kind::None => None,
        Kind::Codestream => Some(vec![Codestream {
            off: 0,
            len: codestream_end(d, 0, d.len()).unwrap_or(d.len()),
        }]),
        Kind::Boxes => {
            let mut cs = Vec::new();
            walk(d, 0, d.len(), 0, &mut 0, &mut cs);
            Some(cs)
        }
    }
}
pub(crate) fn is_ht(d: &[u8]) -> bool {
    if !d.starts_with(b"\xff\x4f") {
        return false;
    }
    let e = d.len().min(1 << 20);
    let (mut p, mut nc) = (2, 0);
    while e.saturating_sub(p) >= 4 {
        if d[p] != 255 {
            return false;
        }
        let m = be16(d, p);
        if matches!(m, 0xff90 | 0xff93 | 0xffd9) {
            return false;
        }
        if (0xff30..=0xff3f).contains(&m) {
            p += 2;
            continue;
        }
        let len = be16(d, p + 2);
        if len < 2 || len + 2 > e - p {
            return false;
        }
        match m {
            0xff51 if len >= 38 => nc = be16(d, p + 38),
            0xff50 if len >= 6 && be32(d, p + 4) & (1 << 17) != 0 => return true,
            0xff52 if len >= 12 && d[p + 12] & 64 != 0 => return true,
            0xff53 => {
                let q = if nc < 257 { 9 } else { 10 };
                if len >= q && d[p + q] & 64 != 0 {
                    return true;
                }
            }
            _ => {}
        }
        p += 2 + len;
    }
    false
}
pub(crate) fn extension(d: &[u8]) -> &'static str {
    match kind(d) {
        Kind::None => "",
        Kind::Codestream => "j2k",
        Kind::Boxes => {
            if d.len() >= 24 && &d[16..20] == b"ftyp" {
                match &d[20..24] {
                    b"jpx " => "jpx",
                    b"jpm " => "jpm",
                    b"mjp2" | b"mj2s" => "mj2",
                    b"jph " => "jph",
                    _ => "jp2",
                }
            } else {
                "jp2"
            }
        }
    }
}
pub(crate) fn siz_samples(s: &[u8]) -> Option<u64> {
    if s.len() < 40 || !s.starts_with(b"\xff\x51") {
        return None;
    }
    let a = be32(s, 6).saturating_sub(be32(s, 14)) as u64;
    let b = be32(s, 10).saturating_sub(be32(s, 18)) as u64;
    Some(
        a.saturating_mul(b)
            .saturating_mul(be16(s, 38) as u64)
            .min(i64::MAX as u64),
    )
}
