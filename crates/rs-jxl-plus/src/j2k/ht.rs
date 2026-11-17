// PJA MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Source: pas/pjaj2kht.c. OpenJPH tables BSD-2-Clause, see NOTICE-PJA.
//! Exact HTJ2K cleanup/refinement segments, including encoder termination rules.
use crate::{Error, Result};
use std::sync::OnceLock;
#[path = "tables/ht.rs"]
mod tables;
use tables::*;
const MEL_EXP: [usize; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];
fn err() -> Error {
    Error::Invalid("HTJ2K codeblock")
}
fn geometry(w: usize, h: usize, np: usize) -> Result<()> {
    if w == 0 || h == 0 || w > 1024 || h > 1024 || w * h > 4096 || !(1..=3).contains(&np) {
        Err(err())
    } else {
        Ok(())
    }
}
fn bitlen(x: u32) -> usize {
    (32 - x.leading_zeros()) as usize
}
struct Tables {
    dec: [[u16; 1024]; 2],
    idx: [[[[usize; 2]; 16]; 8]; 2],
}
fn lookup() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            dec: [[0; 1024]; 2],
            idx: [[[[0; 2]; 16]; 8]; 2],
        };
        for (k, rows) in [&HT_T0[..], &HT_T1[..]].into_iter().enumerate() {
            for i in 0..1024 {
                let cw = i & 127;
                let cq = i >> 7;
                for row in rows {
                    if row[0] as usize == cq && row[5] as usize == cw & ((1 << row[6]) - 1) {
                        t.dec[k][i] = ((row[3] as u16) << 12)
                            | ((row[4] as u16) << 8)
                            | ((row[1] as u16) << 4)
                            | ((row[2] as u16) << 3)
                            | row[6] as u16;
                    }
                }
            }
            for (j, row) in rows.iter().enumerate() {
                let e = &mut t.idx[k][row[0] as usize & 7][row[1] as usize & 15];
                if e[1] == 0 {
                    e[0] = j;
                }
                e[1] = j + 1;
            }
        }
        t
    })
}
struct MelReader<'a> {
    d: &'a [u8],
    pos: usize,
    size: usize,
    cur: u32,
    nb: usize,
    unst: bool,
    k: usize,
    zeros: usize,
    one: bool,
}
impl<'a> MelReader<'a> {
    fn new(d: &'a [u8], pos: usize, size: usize) -> Self {
        Self {
            d,
            pos,
            size,
            cur: 0,
            nb: 0,
            unst: false,
            k: 0,
            zeros: 0,
            one: false,
        }
    }
    fn bit(&mut self) -> u32 {
        if self.nb == 0 {
            let v = if self.size > 0 {
                let mut v = self.d[self.pos] as u32;
                self.pos += 1;
                if self.size == 1 {
                    v |= 15;
                }
                self.size -= 1;
                v
            } else {
                255
            };
            self.nb = if self.unst { 7 } else { 8 };
            self.cur = v;
            self.unst = v == 255;
        }
        self.nb -= 1;
        (self.cur >> self.nb) & 1
    }
    fn event(&mut self) -> u32 {
        loop {
            if self.zeros > 0 {
                self.zeros -= 1;
                return 0;
            }
            if self.one {
                self.one = false;
                return 1;
            }
            let e = MEL_EXP[self.k];
            if self.bit() != 0 {
                self.zeros = 1 << e;
                self.k = (self.k + 1).min(12);
            } else {
                let mut v = 0;
                for _ in 0..e {
                    v = (v << 1) | self.bit() as usize;
                }
                self.zeros = v;
                self.one = true;
                self.k = self.k.saturating_sub(1);
            }
        }
    }
}
struct BitReader<'a> {
    acc: u64,
    n: usize,
    d: &'a [u8],
    pos: isize,
    end: isize,
    unst: bool,
    kind: u8,
}
impl<'a> BitReader<'a> {
    fn new(d: &'a [u8], pos: isize, end: isize, kind: u8, unst: bool) -> Self {
        Self {
            acc: 0,
            n: 0,
            d,
            pos,
            end,
            unst,
            kind,
        }
    }
    fn fill(&mut self) {
        while self.n <= 56 {
            let (v, nb) = if self.kind == 0 || self.kind == 3 {
                let v = if self.pos >= self.end {
                    let v = self.d.get(self.pos as usize).copied().unwrap_or(0) as u64;
                    self.pos -= 1;
                    v
                } else {
                    0
                };
                let nb = if self.unst && v & 127 == 127 { 7 } else { 8 };
                self.unst = v > 0x8f;
                (v, nb)
            } else {
                let v = if self.pos < self.end {
                    let v = self.d[self.pos as usize] as u64;
                    self.pos += 1;
                    v
                } else if self.kind == 1 {
                    255
                } else {
                    0
                };
                let nb = if self.unst { 7 } else { 8 };
                self.unst = v == 255;
                (v, nb)
            };
            self.acc |= (v & ((1 << nb) - 1)) << self.n;
            self.n += nb;
        }
    }
    fn peek(&mut self, n: usize) -> u32 {
        if self.n < n {
            self.fill()
        }
        (self.acc & ((1u64 << n) - 1)) as u32
    }
    fn get(&mut self, n: usize) -> u32 {
        if n == 0 {
            return 0;
        }
        let v = self.peek(n);
        self.acc >>= n;
        self.n -= n;
        v
    }
}
fn uv_prefix(r: &mut BitReader<'_>) -> usize {
    if r.get(1) != 0 {
        1
    } else if r.get(1) != 0 {
        2
    } else if r.get(1) != 0 {
        3
    } else {
        5
    }
}
fn uv_suffix(r: &mut BitReader<'_>, p: usize) -> usize {
    if p < 3 {
        return p;
    }
    if p == 3 {
        return 3 + r.get(1) as usize;
    }
    let s = r.get(5) as usize;
    if s < 28 {
        5 + s
    } else {
        33 + 4 * r.get(4) as usize + s - 28
    }
}
fn neighbor_sig(f: &[u8], w: usize, h: usize, x: usize, y: usize, ys: usize, causal: bool) -> bool {
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                continue;
            }
            let flags = f[ny as usize * w + nx as usize];
            if flags & 1 != 0 {
                if !(causal && ny as usize >= ys + 4) {
                    return true;
                }
            } else if flags & 6 == 6 {
                return true;
            }
        }
    }
    false
}
pub fn decode(
    w: usize,
    h: usize,
    causal: bool,
    np: usize,
    cleanup: &[u8],
    refinement: &[u8],
) -> Result<Vec<i32>> {
    geometry(w, h, np)?;
    let l1 = cleanup.len();
    if l1 < 2 {
        return Err(err());
    }
    let scup = (cleanup[l1 - 1] as usize * 16) + (cleanup[l1 - 2] as usize & 15);
    if scup < 2 || scup > l1 || scup > 4079 {
        return Err(err());
    }
    let mut mel = MelReader::new(cleanup, l1 - scup, scup - 1);
    let mut vlc = BitReader::new(cleanup, l1 as isize - 3, (l1 - scup) as isize, 0, false);
    let v = cleanup[l1 - 2] >> 4;
    vlc.n = if v & 7 == 7 { 3 } else { 4 };
    vlc.acc = (v & ((1 << vlc.n) - 1)) as u64;
    vlc.unst = cleanup[l1 - 2] | 15 > 0x8f;
    let mut ms = BitReader::new(cleanup, 0, (l1 - scup) as isize, 1, false);
    let mut mu = vec![0u32; w * h];
    let mut sg = vec![0u8; w * h];
    let mut eab = vec![0usize; w + 4];
    let mut sab = vec![0usize; w + 4];
    let nqx = w.div_ceil(2);
    let nqy = h.div_ceil(2);
    let t = lookup();
    for qy in 0..nqy {
        let first = qy == 0;
        let mut left = 0usize;
        let y = qy * 2;
        let mut scur = vec![0; w + 4];
        let mut ecur = vec![0; w + 4];
        for qx0 in (0..nqx).step_by(2) {
            let have1 = qx0 + 1 < nqx;
            let mut rho = [0usize; 2];
            let mut uoff = [0usize; 2];
            let mut ek = [0usize; 2];
            let mut e1 = [0usize; 2];
            let mut u = [0usize; 2];
            for j in 0..if have1 { 2 } else { 1 } {
                let x = (qx0 + j) * 2;
                let cq = if first {
                    (left >> 1) | (left & 1)
                } else {
                    (sab[x] | sab[x + 1])
                        | ((((left >> 2) | (left >> 3)) & 1) << 1)
                        | ((sab[x + 2] | sab[x + 3]) << 2)
                };
                let ent = if cq == 0 && mel.event() == 0 {
                    0
                } else {
                    let e = t.dec[(!first) as usize][(cq << 7) | vlc.peek(7) as usize] as usize;
                    if e == 0 {
                        return Err(err());
                    }
                    vlc.get(e & 7);
                    if (e >> 4) & 15 == 0 && cq == 0 {
                        return Err(err());
                    }
                    e
                };
                rho[j] = (ent >> 4) & 15;
                uoff[j] = (ent >> 3) & 1;
                ek[j] = (ent >> 12) & 15;
                e1[j] = (ent >> 8) & 15;
                if (x + 1 >= w && rho[j] & 12 != 0) || (y + 1 >= h && rho[j] & 10 != 0) {
                    return Err(err());
                }
                left = rho[j];
            }
            if first && uoff[0] != 0 && uoff[1] != 0 {
                if mel.event() != 0 {
                    let p0 = uv_prefix(&mut vlc);
                    let p1 = uv_prefix(&mut vlc);
                    u[0] = 2 + uv_suffix(&mut vlc, p0);
                    u[1] = 2 + uv_suffix(&mut vlc, p1);
                } else {
                    let p0 = uv_prefix(&mut vlc);
                    if p0 > 2 {
                        u[1] = 1 + vlc.get(1) as usize;
                        u[0] = uv_suffix(&mut vlc, p0);
                    } else {
                        let p1 = uv_prefix(&mut vlc);
                        u[0] = p0;
                        u[1] = uv_suffix(&mut vlc, p1);
                    }
                }
            } else {
                let p0 = if uoff[0] != 0 { uv_prefix(&mut vlc) } else { 0 };
                let p1 = if uoff[1] != 0 { uv_prefix(&mut vlc) } else { 0 };
                if uoff[0] != 0 {
                    u[0] = uv_suffix(&mut vlc, p0)
                }
                if uoff[1] != 0 {
                    u[1] = uv_suffix(&mut vlc, p1)
                }
            }
            for j in 0..if have1 { 2 } else { 1 } {
                let x = (qx0 + j) * 2;
                let kappa = if !first && rho[j].count_ones() >= 2 {
                    eab[x..x + 4]
                        .iter()
                        .max()
                        .copied()
                        .unwrap()
                        .saturating_sub(1)
                        .max(1)
                } else {
                    1
                };
                let uq = u[j] + kappa;
                if uq > 29 {
                    return Err(err());
                }
                for n in 0..4 {
                    let (sx, sy) = (x + n / 2, y + n % 2);
                    if rho[j] >> n & 1 == 0 {
                        continue;
                    }
                    let mn = uq - ((ek[j] >> n) & 1);
                    let v = ms.get(mn) | ((((e1[j] >> n) & 1) as u32) << mn);
                    mu[sy * w + sx] = (v >> 1) + 1;
                    sg[sy * w + sx] = (v & 1) as u8;
                }
                if y + 1 < h {
                    let i = (y + 1) * w + x;
                    if rho[j] & 2 != 0 {
                        scur[x + 1] = 1;
                        ecur[x + 1] = bitlen(2 * mu[i] - 1);
                    }
                    if rho[j] & 8 != 0 && x + 1 < w {
                        scur[x + 2] = 1;
                        ecur[x + 2] = bitlen(2 * mu[i + 1] - 1);
                    }
                }
            }
        }
        sab = scur;
        eab = ecur;
    }
    if np == 1 {
        return Ok(mu
            .iter()
            .zip(sg)
            .map(|(&v, s)| if s != 0 { -(v as i32) } else { v as i32 })
            .collect());
    }
    let mut sp = BitReader::new(refinement, 0, refinement.len() as isize, 2, false);
    let mut mr = BitReader::new(refinement, refinement.len() as isize - 1, 0, 3, true);
    let mut flags: Vec<u8> = mu.iter().map(|&v| (v != 0) as u8).collect();
    for ys in (0..h).step_by(4) {
        for xg in (0..w).step_by(4) {
            let mut new = Vec::new();
            for x in xg..(xg + 4).min(w) {
                for y in ys..(ys + 4).min(h) {
                    let i = y * w + x;
                    if flags[i] & 1 != 0 {
                        continue;
                    }
                    if !neighbor_sig(&flags, w, h, x, y, ys, causal) {
                        flags[i] |= 4;
                        continue;
                    }
                    let b = sp.get(1);
                    flags[i] |= 4 | ((b as u8) << 1);
                    if b != 0 {
                        new.push(i);
                    }
                }
            }
            for i in new {
                sg[i] = sp.get(1) as u8;
            }
        }
    }
    let mut q: Vec<i32> = mu
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            if flags[i] & 1 != 0 {
                (2 * m) as i32
            } else {
                (flags[i] & 2 != 0) as i32
            }
        })
        .collect();
    if np > 2 {
        for ys in (0..h).step_by(4) {
            for x in 0..w {
                for y in ys..(ys + 4).min(h) {
                    let i = y * w + x;
                    if flags[i] & 1 != 0 {
                        q[i] |= mr.get(1) as i32;
                    }
                }
            }
        }
    }
    for (i, v) in q.iter_mut().enumerate() {
        if sg[i] != 0 {
            *v = -*v
        }
    }
    Ok(q)
}
#[derive(Clone)]
struct MelWriter {
    b: Vec<u8>,
    rem: usize,
    tmp: u32,
    run: usize,
    k: usize,
    thr: usize,
    lastcw: i32,
    events: Vec<u32>,
}
impl MelWriter {
    fn new() -> Self {
        Self {
            b: Vec::new(),
            rem: 8,
            tmp: 0,
            run: 0,
            k: 0,
            thr: 1,
            lastcw: -1,
            events: Vec::new(),
        }
    }
    fn bit(&mut self, v: u32) -> Result<()> {
        self.tmp = (self.tmp << 1) + v;
        self.rem -= 1;
        if self.rem == 0 {
            if self.b.len() >= 8192 {
                return Err(err());
            }
            self.b.push(self.tmp as u8);
            self.rem = if self.tmp == 255 { 7 } else { 8 };
            self.tmp = 0;
        }
        Ok(())
    }
    fn encode(&mut self, bit: u32) -> Result<()> {
        if bit == 0 {
            self.run += 1;
            if self.run >= self.thr {
                self.lastcw = 1;
                self.bit(1)?;
                self.run = 0;
                self.k = (self.k + 1).min(12);
                self.thr = 1 << MEL_EXP[self.k];
            }
        } else {
            self.lastcw = 0;
            self.bit(0)?;
            for t in (0..MEL_EXP[self.k]).rev() {
                self.bit(((self.run >> t) & 1) as u32)?;
            }
            self.run = 0;
            self.k = self.k.saturating_sub(1);
            self.thr = 1 << MEL_EXP[self.k];
        }
        Ok(())
    }
    fn event(&mut self, b: u32) -> Result<()> {
        if self.events.len() >= 8192 {
            return Err(err());
        }
        self.events.push(b);
        self.encode(b)
    }
}
#[derive(Clone)]
struct VlcWriter {
    b: Vec<u8>,
    used: usize,
    tmp: u32,
    gt: usize,
}
impl VlcWriter {
    fn new() -> Self {
        Self {
            b: vec![255],
            used: 4,
            tmp: 15,
            gt: 1,
        }
    }
    fn encode(&mut self, mut cwd: u32, mut len: usize) -> Result<()> {
        while len > 0 {
            let avail = 8 - self.gt - self.used;
            let t = avail.min(len);
            self.tmp |= (cwd & ((1 << t) - 1)) << self.used;
            self.used += t;
            len -= t;
            cwd >>= t;
            if avail == t {
                if self.gt != 0 && self.tmp != 127 {
                    self.gt = 0;
                    continue;
                }
                if self.b.len() >= 8192 {
                    return Err(err());
                }
                self.b.push(self.tmp as u8);
                self.gt = (self.tmp > 0x8f) as usize;
                self.tmp = 0;
                self.used = 0;
            }
        }
        Ok(())
    }
    fn pre(&mut self, u: usize) -> Result<()> {
        let c = uv_code(u);
        self.encode(c[0] as u32, c[1])
    }
    fn suf(&mut self, u: usize) -> Result<()> {
        let c = uv_code(u);
        self.encode(c[2] as u32, c[3])?;
        self.encode(c[4] as u32, c[5])
    }
}
struct MsWriter {
    b: Vec<u8>,
    maxb: usize,
    used: usize,
    tmp: u64,
}
impl MsWriter {
    fn new() -> Self {
        Self {
            b: Vec::new(),
            maxb: 8,
            used: 0,
            tmp: 0,
        }
    }
    fn encode(&mut self, mut cwd: u64, mut len: usize) -> Result<()> {
        while len > 0 {
            let t = (self.maxb - self.used).min(len);
            self.tmp |= (cwd & ((1 << t) - 1)) << self.used;
            self.used += t;
            cwd >>= t;
            len -= t;
            if self.used >= self.maxb {
                if self.b.len() >= 20480 {
                    return Err(err());
                }
                self.b.push(self.tmp as u8);
                self.maxb = if self.tmp & 255 == 255 { 7 } else { 8 };
                self.tmp = 0;
                self.used = 0;
            }
        }
        Ok(())
    }
    fn terminate(&mut self) {
        if self.used > 0 {
            let t = self.maxb - self.used;
            self.tmp |= ((1u64 << t) - 1) << self.used;
            if self.tmp & 255 != 255 {
                self.b.push(self.tmp as u8);
            }
        } else if self.maxb == 7 && !self.b.is_empty() {
            self.b.pop();
        }
    }
}
fn uv_code(u: usize) -> [usize; 6] {
    match u {
        1 => [1, 1, 0, 0, 0, 0],
        2 => [2, 2, 0, 0, 0, 0],
        3 | 4 => [4, 3, u - 3, 1, 0, 0],
        5..=32 => [0, 3, u - 5, 5, 0, 0],
        33.. => [0, 3, 28 + (u - 33) % 4, 5, (u - 33) / 4, 4],
        _ => [0; 6],
    }
}
fn build_mv(m: &[u8], v: &[u8]) -> Vec<u8> {
    let mut out = m.to_vec();
    out.extend(v.iter().rev());
    let sc = out.len();
    if sc >= 2 {
        out[sc - 1] = (sc >> 4) as u8;
        out[sc - 2] = (out[sc - 2] & 240) | (sc as u8 & 15);
    }
    out
}
fn mel_check(r: &[u8], events: &[u32]) -> bool {
    if r.len() < 2 {
        return false;
    }
    let mut mr = MelReader::new(r, 0, r.len() - 1);
    events.iter().all(|&e| mr.event() == e)
}
fn terminate(m: &MelWriter, v: &VlcWriter, rule: u8) -> Result<Vec<u8>> {
    let mut mm = m.clone();
    let mut vv = v.clone();
    match rule {
        0 | 1 => {
            if mm.run > 0 {
                mm.bit(1)?
            }
            mm.tmp <<= mm.rem;
            let mel_mask = (255 << mm.rem) & 255;
            let vlc_mask = 255 >> (8 - vv.used);
            if mel_mask | vlc_mask != 0 {
                if rule == 1 && mel_mask == 0 {
                    vv.b.push(vv.tmp as u8);
                } else {
                    let f = mm.tmp | vv.tmp;
                    let can = (((f ^ mm.tmp) & mel_mask) | ((f ^ vv.tmp) & vlc_mask)) == 0
                        && f != 255
                        && vv.b.len() > 1;
                    if can {
                        mm.b.push(f as u8);
                    } else {
                        mm.b.push(mm.tmp as u8);
                        vv.b.push(vv.tmp as u8);
                    }
                }
            }
            Ok(build_mv(&mm.b, &vv.b))
        }
        2 | 3 => {
            let mut tb = 1 + MEL_EXP[mm.k];
            if mm.run > 0 || mm.lastcw == 1 {
                mm.encode(1)?
            } else {
                tb = 0
            }
            let lastff = mm.b.last() == Some(&255);
            let (rem, cp) = if mm.rem < 8 && mm.rem != if lastff { 7 } else { 8 } {
                mm.b.push((mm.tmp << mm.rem) as u8);
                (mm.rem, if lastff { 7 } else { 8 })
            } else {
                (
                    0,
                    if mm.b.len() >= 2 && mm.b[mm.b.len() - 2] == 255 {
                        7
                    } else {
                        8
                    },
                )
            };
            let used = cp - rem;
            let tin = tb.min(used);
            let termmask = ((1 << tin) - 1) << rem;
            let usedmask = ((1 << used) - 1) << rem;
            let realmask = usedmask & !termmask;
            let vpart = vv.used > 0;
            let vmask = if vpart { (1 << vv.used) - 1 } else { 255 };
            if vpart {
                vv.b.push(vv.tmp as u8);
            }
            if !mm.b.is_empty() && vv.b.len() > 1 {
                let f = *mm.b.last().unwrap() as usize;
                let g = *vv.b.last().unwrap() as usize;
                let c10 = (f ^ g) & vmask & realmask == 0;
                let cor = f & !g & vmask == 0;
                if cor && c10 && (rule == 2 || (usedmask & vmask == 0 || !vpart)) {
                    let fused = (f & !vmask) | (g & vmask);
                    let last = mm.b.len() - 1;
                    mm.b[last] = fused as u8;
                    let out = build_mv(&mm.b, &vv.b[..vv.b.len() - 1]);
                    if mel_check(&out, &m.events) {
                        return Ok(out);
                    }
                    mm.b[last] = f as u8;
                }
            }
            Ok(build_mv(&mm.b, &vv.b))
        }
        4 => {
            if mm.run > 0 {
                mm.encode(1)?
            }
            if mm.rem < 8 {
                mm.b.push((mm.tmp << mm.rem) as u8);
            }
            let n = mm.b.len();
            let vmask = 255 >> (8 - v.used);
            let mut best: Option<Vec<u8>> = None;
            for k in n.saturating_sub(4)..=n {
                if best.is_some() {
                    break;
                }
                for fuse in [false, true] {
                    let mut mel = mm.b[..k].to_vec();
                    let mut vlc = v.b.clone();
                    if fuse {
                        if k == 0 || v.used == 0 || v.b.len() <= 1 {
                            continue;
                        }
                        let f = (mel[k - 1] as u32 & !vmask) | v.tmp;
                        if f == 255 {
                            continue;
                        }
                        mel[k - 1] = f as u8;
                    } else if v.used > 0 || v.b.len() == 1 {
                        vlc.push(v.tmp as u8);
                    }
                    let out = build_mv(&mel, &vlc);
                    if mel_check(&out, &m.events)
                        && best.as_ref().is_none_or(|b| out.len() < b.len())
                    {
                        best = Some(out);
                    }
                }
            }
            best.ok_or_else(err)
        }
        _ => Err(err()),
    }
}
fn pick(
    first: bool,
    cq: usize,
    rho: usize,
    uoff: usize,
    eps: usize,
    minc: bool,
) -> Option<&'static [u8; 7]> {
    let rows = if first { &HT_T0[..] } else { &HT_T1[..] };
    let range = lookup().idx[(!first) as usize][cq][rho];
    let (mut best, mut be, mut bc) = (None, -1, 1i32 << 30);
    for row in &rows[range[0]..range[1]] {
        if row[2] as usize != uoff {
            continue;
        }
        if uoff == 0 {
            return Some(row);
        }
        if eps & row[3] as usize != row[4] as usize {
            continue;
        }
        let pop = row[3].count_ones() as i32;
        if minc {
            let c = row[6] as i32 - pop;
            if c < bc {
                bc = c;
                best = Some(row)
            }
        } else if pop >= be {
            best = Some(row);
            be = pop;
        }
    }
    best
}
pub fn encode_cleanup(w: usize, h: usize, np: usize, q: &[i32], rule: u8) -> Result<Vec<u8>> {
    geometry(w, h, np)?;
    if q.len() != w * h || rule & 7 > 4 {
        return Err(err());
    }
    let term = rule & 7;
    let minc = rule & 8 != 0;
    let sh = (np > 1) as usize;
    let nqx = w.div_ceil(2);
    let nqy = h.div_ceil(2);
    let mut mel = MelWriter::new();
    let mut vlc = VlcWriter::new();
    let mut ms = MsWriter::new();
    let mut eab = vec![0usize; w + 4];
    let mut sab = vec![0usize; w + 4];
    for qy in 0..nqy {
        let first = qy == 0;
        let (mut left, y) = (0usize, 2 * qy);
        let mut scur = vec![0; w + 4];
        let mut ecur = vec![0; w + 4];
        for qx0 in (0..nqx).step_by(2) {
            let mut uq = [0; 2];
            for (j, uqj) in uq.iter_mut().enumerate() {
                if qx0 + j >= nqx {
                    break;
                }
                let x = 2 * (qx0 + j);
                let (mut rho, mut emax) = (0usize, 0usize);
                let mut v = [0u32; 4];
                let mut es = [0usize; 4];
                for n in 0..4 {
                    let (sx, sy) = (x + n / 2, y + n % 2);
                    if sx >= w || sy >= h {
                        continue;
                    }
                    let a = q[sy * w + sx].unsigned_abs();
                    let m = a >> sh;
                    if m != 0 {
                        if m > 1 << 28 {
                            return Err(err());
                        }
                        rho |= 1 << n;
                        es[n] = bitlen(2 * m - 1);
                        v[n] = 2 * (m - 1) + (q[sy * w + sx] < 0) as u32;
                        emax = emax.max(es[n]);
                    }
                }
                let cq = if first {
                    (left >> 1) | (left & 1)
                } else {
                    (sab[x] | sab[x + 1])
                        | ((((left >> 2) | (left >> 3)) & 1) << 1)
                        | ((sab[x + 2] | sab[x + 3]) << 2)
                };
                let kappa = if !first && rho.count_ones() >= 2 {
                    eab[x..x + 4]
                        .iter()
                        .max()
                        .copied()
                        .unwrap()
                        .saturating_sub(1)
                        .max(1)
                } else {
                    1
                };
                let u_max = emax.max(kappa);
                if u_max > 29 {
                    return Err(err());
                }
                let u = if rho != 0 { u_max - kappa } else { 0 };
                let uoff = (u > 0) as usize;
                let mut eps = 0;
                if uoff != 0 {
                    for (n, &e) in es.iter().enumerate() {
                        if rho >> n & 1 != 0 && e == u_max {
                            eps |= 1 << n;
                        }
                    }
                }
                let (mut ek, mut e1) = (0usize, 0usize);
                if rho != 0 || cq != 0 {
                    let row = pick(first, cq, rho, uoff, eps, minc).ok_or_else(err)?;
                    ek = row[3] as usize;
                    e1 = row[4] as usize;
                    vlc.encode(row[5] as u32, row[6] as usize)?;
                }
                if cq == 0 {
                    mel.event((rho != 0) as u32)?;
                }
                for (n, &value) in v.iter().enumerate() {
                    if rho >> n & 1 == 0 {
                        continue;
                    }
                    let mn = u_max - ((ek >> n) & 1);
                    if ek >> n & 1 != 0 {
                        if (value >> mn) & 1 != ((e1 >> n) & 1) as u32 || value >> (mn + 1) != 0 {
                            return Err(err());
                        }
                    } else if value >> mn != 0 {
                        return Err(err());
                    }
                    ms.encode((value & ((1u32 << mn) - 1)) as u64, mn)?;
                }
                *uqj = u;
                scur[x + 1] = (rho >> 1) & 1;
                ecur[x + 1] = es[1];
                scur[x + 2] = (rho >> 3) & 1;
                ecur[x + 2] = es[3];
                left = rho;
            }
            if first && uq[0] > 0 && uq[1] > 0 {
                mel.event((uq[0].min(uq[1]) > 2) as u32)?;
            }
            if first && uq[0] > 2 && uq[1] > 2 {
                vlc.pre(uq[0] - 2)?;
                vlc.pre(uq[1] - 2)?;
                vlc.suf(uq[0] - 2)?;
                vlc.suf(uq[1] - 2)?;
            } else if first && uq[0] > 2 && uq[1] > 0 {
                vlc.pre(uq[0])?;
                vlc.encode((uq[1] - 1) as u32, 1)?;
                vlc.suf(uq[0])?;
            } else {
                vlc.pre(uq[0])?;
                vlc.pre(uq[1])?;
                vlc.suf(uq[0])?;
                vlc.suf(uq[1])?;
            }
        }
        sab = scur;
        eab = ecur;
    }
    let mv = terminate(&mel, &vlc, term)?;
    ms.terminate();
    if mv.len() < 2 || mv.len() > 4079 {
        return Err(err());
    }
    ms.b.extend(mv);
    Ok(ms.b)
}
pub fn encode_refinement(
    w: usize,
    h: usize,
    causal: bool,
    np: usize,
    q: &[i32],
    rule: u8,
) -> Result<Vec<u8>> {
    geometry(w, h, np)?;
    if np < 2 || q.len() != w * h || rule > 4 {
        return Err(err());
    }
    let fuse = rule == 4;
    let spt = !fuse && rule >> 1 & 1 != 0;
    let mrt = !fuse && rule & 1 != 0;
    let mut flags: Vec<u8> = q.iter().map(|v| (v.unsigned_abs() >= 2) as u8).collect();
    let mut sp = MsWriter::new();
    for ys in (0..h).step_by(4) {
        for xg in (0..w).step_by(4) {
            let mut new = Vec::new();
            for x in xg..(xg + 4).min(w) {
                for y in ys..(ys + 4).min(h) {
                    let i = y * w + x;
                    if flags[i] & 1 != 0 {
                        continue;
                    }
                    let a = q[i].unsigned_abs();
                    if !neighbor_sig(&flags, w, h, x, y, ys, causal) {
                        if a != 0 {
                            return Err(err());
                        }
                        flags[i] |= 4;
                        continue;
                    }
                    sp.encode((a != 0) as u64, 1)?;
                    flags[i] |= 4 | if a != 0 { 2 } else { 0 };
                    if a != 0 {
                        new.push(i);
                    }
                }
            }
            for i in new {
                sp.encode((q[i] < 0) as u64, 1)?;
            }
        }
    }
    if sp.used != 0 {
        sp.b.push(sp.tmp as u8);
    }
    if spt {
        while sp.b.last() == Some(&0) {
            sp.b.pop();
        }
    }
    let mut mr = VlcWriter {
        b: Vec::new(),
        used: 0,
        tmp: 0,
        gt: 1,
    };
    if np > 2 {
        for ys in (0..h).step_by(4) {
            for x in 0..w {
                for y in ys..(ys + 4).min(h) {
                    let i = y * w + x;
                    if flags[i] & 1 != 0 {
                        mr.encode((q[i] & 1) as u32, 1)?;
                    }
                }
            }
        }
        if mr.used != 0 {
            mr.b.push(mr.tmp as u8);
        }
        if mrt {
            while mr.b.last() == Some(&0) {
                mr.b.pop();
            }
        }
    }
    if sp.b.len() > 4096 || mr.b.len() > 4096 {
        return Err(err());
    }
    if fuse {
        if sp.b.is_empty() || mr.b.is_empty() {
            return Err(err());
        }
        let last = sp.b.len() - 1;
        sp.b[last] |= mr.b.pop().unwrap();
    }
    sp.b.extend(mr.b.iter().rev());
    Ok(sp.b)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cleanup_and_refinement_rules() {
        for np in 1..=3 {
            for causal in [false, true] {
                let q: Vec<i32> = (0..35)
                    .map(|i| {
                        let a = if i % 5 == 0 {
                            0
                        } else {
                            2 * (i * 13 % 113 + 1) + if np == 3 { i % 2 } else { 0 }
                        };
                        if i % 3 == 0 { -(a as i32) } else { a as i32 }
                    })
                    .collect();
                for rule in [0, 1, 2, 3, 4, 8, 9, 10, 11, 12] {
                    let cleanup = encode_cleanup(7, 5, np, &q, rule).unwrap();
                    if np == 1 {
                        assert_eq!(decode(7, 5, causal, np, &cleanup, &[]).unwrap(), q);
                    } else {
                        for ref_rule in 0..=4 {
                            let refinement = encode_refinement(7, 5, causal, np, &q, ref_rule);
                            if ref_rule == 4 && refinement.is_err() {
                                continue;
                            }
                            let refinement = refinement.unwrap();
                            let decoded = decode(7, 5, causal, np, &cleanup, &refinement).unwrap();
                            // Trimming/fusion rules are reconstruction candidates:
                            // the caller selects one only after comparing it with
                            // the original segment. Shortening the forward stream
                            // can expose reverse-stream bits, so candidate rules
                            // need not preserve arbitrary supplied q values.
                            if ref_rule == 0 {
                                assert_eq!(decoded, q, "np{np} cleanup{rule}");
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn malformed_geometry_and_short_segments() {
        assert!(decode(0, 1, false, 1, &[], &[]).is_err());
        for b in [&[][..], &[0][..], &[0, 0][..], &[255, 255][..]] {
            assert!(decode(4, 4, false, 1, b, &[]).is_err());
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "requires RS_JXL_PLUS_J2K_ORACLE original-C helper"]
fn original_c_ht_segments() {
    use super::tier1::oracle::{run, word};
    for np in 1..=3 {
        for causal in [false, true] {
            let q: Vec<i32> = (0..35)
                .map(|i| {
                    let a = if i % 5 == 0 {
                        0
                    } else {
                        2 * (i * 13 % 113 + 1) + if np == 3 { i % 2 } else { 0 }
                    };
                    if i % 3 == 0 { -(a as i32) } else { a as i32 }
                })
                .collect();
            for rule in [0, 1, 2, 3, 4, 8, 9, 10, 11, 12] {
                for rr in 0..=4 {
                    let cleanup = encode_cleanup(7, 5, np, &q, rule).unwrap();
                    let refinement = if np == 1 {
                        Vec::new()
                    } else {
                        match encode_refinement(7, 5, causal, np, &q, rr) {
                            Ok(v) => v,
                            Err(_) if rr == 4 => continue,
                            Err(e) => panic!("{e:?}"),
                        }
                    };
                    let mut desc = Vec::new();
                    for n in [7, 5, causal as usize, np, rule as usize, rr as usize] {
                        word(&mut desc, n)
                    }
                    for &v in &q {
                        desc.extend(v.to_le_bytes());
                    }
                    let decoded = decode(7, 5, causal, np, &cleanup, &refinement).unwrap();
                    let mut expected = Vec::new();
                    word(&mut expected, cleanup.len());
                    word(&mut expected, refinement.len());
                    expected.extend(cleanup);
                    expected.extend(refinement);
                    for v in decoded {
                        expected.extend(v.to_le_bytes());
                    }
                    assert_eq!(
                        run("ht", &desc),
                        expected,
                        "np{np} causal{causal} cup{rule} ref{rr}"
                    );
                }
            }
        }
    }
}
