// PJA MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Source: pas/pjaj2kmc.c, pjaj2kt1.h, pjaj2km_tab.h. See NOTICE-PJA.
//! Persistent context-mixing model for classic JPEG2000 tier-1 decisions.
use super::tier1::{self, PIV, Passes, REF, SIG};
use crate::{Error, Result, arith::BitCoder};
#[path = "tables/model.rs"]
mod tables;
use tables::*;
pub(crate) fn shr(x: i32, n: u32) -> i32 {
    x.wrapping_add((x >> 31) & ((1 << n) - 1)) >> n
}
pub(crate) fn qlog(v: u64) -> i32 {
    let l = 63 - v.leading_zeros();
    (2 * l + ((v.wrapping_shl(1) >> l) & 1) as u32) as i32
}
fn qlog_or(v: u64, d: i32, lo: i32, hi: i32, z: i32) -> i32 {
    if v == 0 {
        z
    } else {
        (qlog(v) - d).clamp(lo, hi)
    }
}
#[derive(Clone, Copy, Default)]
pub(crate) struct Counter {
    p: u16,
    f: u16,
    n: u8,
}
impl Counter {
    fn learn(&mut self, b: i32) {
        let p = (self.p ^ 32768) as u32;
        let d = if b != 0 { 65535 - p } else { p - 1 };
        let step = d * J2KM_RC[self.n as usize] as u32 >> 16;
        self.p = ((if b != 0 { p + step } else { p - step }) as u16) ^ 32768;
        self.n = self.n.saturating_add(1);
        let f = (self.f ^ 32768) as i32;
        self.f = (f + ((if b != 0 { 65535 } else { 0 }) - f >> 3)) as u16 ^ 32768;
    }
}
pub(crate) struct Mixer {
    pk: [u32; 8],
    nk: usize,
    hshift: u32,
    t: [Vec<Counter>; 8],
    weights: [Vec<[i32; 16]>; 3],
    counts: [Vec<u16>; 3],
    apm: [Vec<[u16; 33]>; 3],
    misc: [Counter; 64],
}
impl Mixer {
    pub(crate) fn new(
        hb: u32,
        nk: usize,
        pk: [u32; 8],
        sets: [usize; 3],
        maps: [usize; 3],
    ) -> Result<Self> {
        if !(10..=20).contains(&hb) {
            return Err(Error::Invalid("J2K model hash bits"));
        }
        let mut weights = std::array::from_fn(|k| vec![[0; 16]; sets[k]]);
        for group in &mut weights {
            for w in group {
                for j in 0..8 {
                    w[2 * j] = 8192;
                }
            }
        }
        let mut initial = [0u16; 33];
        for (j, p) in initial.iter_mut().enumerate() {
            *p = J2KM_SQ[((j as i32 - 16) * 128 + 3072) as usize];
        }
        Ok(Self {
            pk,
            nk,
            hshift: 32 - hb,
            t: std::array::from_fn(|i| {
                vec![Counter::default(); (pk[i] as usize * nk).min(1usize << hb)]
            }),
            weights,
            counts: std::array::from_fn(|i| vec![0; sets[i]]),
            apm: std::array::from_fn(|i| vec![initial; maps[i]]),
            misc: [Counter::default(); 64],
        })
    }
    pub(crate) fn code<C: BitCoder>(
        &mut self,
        cd: &mut C,
        bit: i32,
        v: [u32; 8],
        sets: [usize; 3],
        maps: [usize; 3],
    ) -> Result<i32> {
        if sets
            .iter()
            .enumerate()
            .any(|(i, &s)| s >= self.weights[i].len())
            || maps
                .iter()
                .enumerate()
                .any(|(i, &s)| s >= self.apm[i].len())
        {
            return Err(Error::Invalid("J2K model selector"));
        }
        let mut idx = [0usize; 8];
        let mut s = [0i32; 16];
        for i in 0..8 {
            if v[i] as usize >= self.pk[i] as usize * self.nk {
                return Err(Error::Invalid("J2K context range"));
            }
            idx[i] = if self.pk[i] as usize * self.nk > self.t[i].len() {
                v[i].wrapping_add(1).wrapping_mul(0x9e3779b1) >> self.hshift
            } else {
                v[i]
            } as usize;
            let c = self.t[i][idx[i]];
            s[2 * i] = J2KM_ST[((c.p ^ 32768) >> 4) as usize] as i32;
            s[2 * i + 1] = J2KM_ST[((c.f ^ 32768) >> 4) as usize] as i32;
        }
        let mut d = [0i32; 3];
        for k in 0..3 {
            d[k] = crate::mix_simd::dot(&s, &self.weights[k][sets[k]]);
            d[k] = shr(d[k], 16).clamp(-3071, 3071);
        }
        let x = shr((d[0] + d[1] + d[2]) * 21846, 16);
        let pm = J2KM_SQ[(x + 3072) as usize] as i32;
        let pos = (x.clamp(-2047, 2047) + 2048) * 32;
        let lo = (pos >> 12) as usize;
        let fr = pos & 4095;
        let mut pa = [0i32; 3];
        for k in 0..3 {
            let t = &self.apm[k][maps[k]];
            pa[k] = (t[lo] as i32 * (4096 - fr) + t[lo + 1] as i32 * fr) >> 12;
        }
        let b = cd.bit(((pm + pa[0] + pa[1] + pa[2]) >> 2) as u32, bit);
        for k in 0..3 {
            let count = &mut self.counts[k][sets[k]];
            let lr = 6 + 31457 / (1024 + 64 * (*count as i32 >> 6));
            *count = count.saturating_add(1);
            let e = (((b << 16) - J2KM_SQ[(d[k] + 3072) as usize] as i32) >> 4) * lr;
            let w = &mut self.weights[k][sets[k]];
            crate::mix_simd::update(w, &s, e, 16, 1 << 22);
            let t = &mut self.apm[k][maps[k]];
            let tg = if b != 0 { 65535 } else { 0 };
            t[lo] = (t[lo] as i32 + shr(((tg - t[lo] as i32) * (4096 - fr)) >> 6, 12)) as u16;
            t[lo + 1] = (t[lo + 1] as i32 + shr(((tg - t[lo + 1] as i32) * fr) >> 6, 12)) as u16;
        }
        for (i, &j) in idx.iter().enumerate() {
            self.t[i][j].learn(b);
        }
        Ok(b)
    }
    pub(crate) fn misc<C: BitCoder>(&mut self, cd: &mut C, bit: i32, j: usize) -> i32 {
        let c = &mut self.misc[j];
        let b = cd.bit((c.p ^ 32768) as u32, bit);
        c.learn(b);
        b
    }
}
#[derive(Clone, Debug)]
pub struct Band {
    pub comp: i32,
    pub res: i32,
    pub orient: i32,
    pub nres: i32,
    pub width: usize,
    pub height: usize,
    pub mb: i32,
    pub parent: Option<usize>,
    pub cross0: Option<usize>,
    pub cross1: Option<usize>,
    pub map: Option<usize>,
}
pub struct CoefficientModel {
    mix: Mixer,
    bands: Vec<Band>,
    maps: Vec<u8>,
    zcomp: i32,
}
const PK: [u32; 8] = [
    12 * 3 * 8,
    12 * 20 * 3 * 81,
    20 * 18 * 4 * 4 * 9,
    12 * 18 * 18 * 21 * 4,
    9 * 14 * 20 * 3 * 4 * 17,
    20 * 21 * 64 * 4 * 8,
    18 * 18 * 20 * 14 * 4,
    20 * 18 * 18 * 11 * 9 * 4,
];
pub(crate) fn mapcode(v: i32) -> u8 {
    if v == 0 {
        0
    } else {
        (qlog(v.unsigned_abs() as u64) + 1) as u8 | if v < 0 { 128 } else { 0 }
    }
}
pub(crate) fn mapval(c: u8) -> i32 {
    let v = c & 63;
    if v == 0 {
        return 0;
    }
    let q = v - 1;
    let l = q >> 1;
    let m = (1u32 << l.min(30))
        | if q & 1 != 0 && l > 0 && l <= 30 {
            1u32 << (l - 1)
        } else {
            0
        };
    if c & 128 != 0 { -(m as i32) } else { m as i32 }
}
impl CoefficientModel {
    pub fn new(hash_bits: u32) -> Result<Self> {
        Ok(Self {
            mix: Mixer::new(
                hash_bits,
                5,
                PK,
                [5 * 20 * 24, 5 * 12 * 18, 5 * 64 * 4],
                [5 * 20, 5 * 18 * 18, 5 * 12 * 21],
            )?,
            bands: Vec::new(),
            maps: Vec::new(),
            zcomp: -1,
        })
    }
    pub fn tile(&mut self, bands: Vec<Band>, map_bytes: usize) -> Result<()> {
        if map_bytes > 1 << 27 {
            return Err(Error::Limit("J2K band maps"));
        }
        for g in &bands {
            if g.width.checked_mul(g.height).is_none()
                || g.width > i32::MAX as usize
                || g.height > i32::MAX as usize
                || !(-1024..=1024).contains(&g.mb)
            {
                return Err(Error::Invalid("J2K band geometry"));
            }
        }
        self.bands = bands;
        self.maps.resize(map_bytes, 0);
        self.zcomp = -1;
        for g in &self.bands {
            if g.comp <= 1 {
                if let Some(range) = map_range(g, self.maps.len()) {
                    self.maps[range].fill(0)
                }
            }
        }
        Ok(())
    }
    fn band_map(&self, b: Option<usize>) -> Option<(&Band, &[u8])> {
        let g = self.bands.get(b?)?;
        let r = map_range(g, self.maps.len())?;
        Some((g, &self.maps[r]))
    }
}
fn map_range(g: &Band, n: usize) -> Option<std::ops::Range<usize>> {
    if g.width == 0 || g.height == 0 {
        return None;
    }
    let s = g.map?;
    let e = s.checked_add(g.width.checked_mul(g.height)?)?;
    if e > n { None } else { Some(s..e) }
}
struct BlockState {
    w: usize,
    h: usize,
    rw: usize,
    mbc: i32,
    orient: i32,
    lvl: i32,
    comp: i32,
    r: Vec<i32>,
    pq: Vec<i32>,
    xq: Vec<i32>,
    ps: Vec<i32>,
    xs: Vec<i32>,
    pqoff: i32,
    xqoff: i32,
    have_p: bool,
    have_x: bool,
    symbols: Vec<u8>,
    source: Vec<u8>,
    cap: usize,
}
impl BlockState {
    fn ri(&self, x: usize, y: usize) -> usize {
        (y + 2) * self.rw + x + 2
    }
    fn take(&mut self, cx: usize, encoding: bool) -> Result<i32> {
        if !encoding {
            return Ok(0);
        }
        let c = *self
            .source
            .get(self.symbols.len())
            .ok_or(Error::Invalid("J2K decisions exhausted"))?;
        if c >> 1 != 127 && c as usize >> 1 != cx {
            return Err(Error::Invalid("J2K decision context"));
        }
        self.symbols.push(c);
        Ok((c & 1) as i32)
    }
    fn give(&mut self, cx: usize, b: i32, encoding: bool) -> Result<()> {
        if !encoding {
            if self.symbols.len() >= self.cap {
                return Err(Error::Limit("J2K decisions"));
            }
            self.symbols.push(((cx << 1) | b as usize) as u8);
        }
        Ok(())
    }
}
impl BlockState {
    fn event<C: BitCoder>(
        &self,
        m: &mut Mixer,
        cd: &mut C,
        bit: i32,
        k: i32,
        x: usize,
        y: usize,
        b: i32,
        sd: i32,
        rc: i32,
    ) -> Result<i32> {
        let i = self.ri(x, y);
        let rw = self.rw;
        let pi = y * self.w + x;
        let r = &self.r;
        let abs = |i: usize| r[i].unsigned_abs() as u64;
        let n4 = abs(i - 1) + abs(i + 1) + abs(i - rw) + abs(i + rw);
        let nd = abs(i - rw - 1) + abs(i - rw + 1) + abs(i + rw - 1) + abs(i + rw + 1);
        let n2 = abs(i - 2) + abs(i + 2) + abs(i - 2 * rw) + abs(i + 2 * rw);
        let nsum = 2 * n4 + nd;
        let ln = qlog_or(nsum, 2 * b + 2, -6, 12, -7);
        let ln2 = qlog_or(n2, 2 * b, -4, 8, -5);
        let (mut lp, mut lps, mut lx, mut lxs) = (-9, 0, -9, 0);
        let depth = (self.mbc - b).clamp(0, 20);
        let orl = self.orient * 16 + self.lvl;
        if self.have_p {
            let q = self.pq[pi];
            lps = self.ps[pi];
            if q > -100 {
                lp = (q + self.pqoff - 2 * b).clamp(-8, 8)
            }
        }
        if self.have_x {
            let q = self.xq[pi];
            lxs = self.xs[pi];
            if q > -100 {
                lx = (q + self.xqoff - 2 * b).clamp(-8, 8)
            }
        }
        let (lni, lpi, lxi) = (ln + 7, lp + 9, lx + 9);
        let mut v = [0i32; 8];
        let ext;
        if k == 0 || k == 1 {
            let nsig = [
                i - 1,
                i + 1,
                i - rw,
                i + rw,
                i - rw - 1,
                i - rw + 1,
                i + rw - 1,
                i + rw + 1,
            ]
            .iter()
            .filter(|&&j| r[j] != 0)
            .count() as i32;
            ext = 0;
            v[0] = (sd * 3 + rc) * 8;
            v[1] = (sd * 20 + lni) * 3 + rc;
            v[2] = ((lni * 18 + lxi) * 4 + self.comp) * 4 + self.orient;
            v[3] = ((sd * 18 + lpi) * 18 + lxi) * 21 * 4 + depth * 4 + self.comp;
            v[4] = (((nsig * 14 + ln2 + 5) * 20 + lni) * 3 + rc) * 4 + self.orient;
            v[5] = ((lni * 21 + depth) * 64 + orl) * 4 + self.comp;
            v[6] = (((lpi * 18 + lxi) * 20 + lni) * 14 + ln2 + 5) * 4 + self.orient;
        } else if k == 2 {
            let sign = |j: usize| r[j].signum() + 1;
            let (sl, su, sr, sbl, sul, sur) = (
                sign(i - 1),
                sign(i - rw),
                sign(i + 1),
                sign(i + rw),
                sign(i - rw - 1),
                sign(i - rw + 1),
            );
            ext = sl * 3 + su;
            v[0] = sd;
            v[1] = ((sl * 27 + su * 9 + sr * 3 + sbl) * 3 + sul) * 4 + self.orient;
            v[2] = ((lxs + 1) * 4 + self.comp) * 9 * 4 + ext * 4 + self.orient;
            v[3] = (((lps + 1) * 9 + sul * 3 + sur) * 9 + ext) * 64 + orl;
            v[4] = (sd * 20 + lni) * 64 + orl;
            v[5] = (((ext * 3 + lps + 1) * 3 + lxs + 1) * 9 + sr * 3 + sbl) * 4 + self.orient;
            v[6] = (sd * 3 + lps + 1) * 3 + lxs + 1;
        } else {
            let me = r[i].unsigned_abs();
            let rr = ((31 - me.max(1).leading_zeros()) as i32 - b).clamp(1, 8) - 1;
            let rel = if me != 0 {
                (qlog(nsum + 1) - qlog(me.max(1) as u64)).clamp(-8, 8)
            } else {
                0
            };
            let mb = (me >> if b > 0 { b - 1 } else { 0 }) as i32 & 7;
            ext = rr;
            v[0] = sd * 3 * 8 + rr;
            v[1] = (rr * 20 + lni) * 4 + self.orient;
            v[2] = (rr * 17 + rel + 8) * 64 + orl;
            v[3] = ((sd * 8 + rr) * 21 + depth) * 4 + self.comp;
            v[4] = ((rr * 17 + rel + 8) * 18 + lpi) * 14 + ln2 + 5;
            v[5] = ((rr * 8 + mb) * 20 + lni) * 64 + orl;
            v[6] = ((rr * 17 + rel + 8) * 18 + lxi) * 4 + self.comp;
        }
        v[7] = ((((lni * 18 + lpi) * 18 + lxi) * 11 + (depth >> 1)) * 9 + ext) * 4 + self.orient;
        let v = std::array::from_fn(|j| (v[j] as u32).wrapping_add(k as u32 * PK[j]));
        m.code(
            cd,
            bit,
            v,
            [
                ((k * 20 + lni) * 24 + depth.clamp(0, 23)) as usize,
                ((k * 12 + sd) * 18 + lpi.clamp(0, 17)) as usize,
                ((k * 64 + orl) * 4 + self.comp) as usize,
            ],
            [
                (k * 20 + lni) as usize,
                ((k * 18 + lpi) * 18 + lxi) as usize,
                ((k * 12 + sd) * 21 + depth) as usize,
            ],
        )
    }
    fn event_run<C: BitCoder>(
        &self,
        m: &mut Mixer,
        cd: &mut C,
        bit: i32,
        x: usize,
        y: usize,
        b: i32,
        rl: i32,
    ) -> Result<i32> {
        let i = self.ri(x, y);
        let mut sw = 0u64;
        for k in -2isize..6 {
            for dx in -2isize..=2 {
                sw +=
                    self.r[(i as isize + k * self.rw as isize + dx) as usize].unsigned_abs() as u64;
            }
        }
        let lw = qlog_or(sw, 2 * b + 2, -6, 12, -7);
        let (mut lp, mut lx) = (-9, -9);
        let depth = (self.mbc - b).clamp(0, 20);
        let orl = self.orient * 16 + self.lvl;
        for k in 0..4 {
            let pi = (y + k) * self.w + x;
            if self.have_p && self.pq[pi] > -100 {
                lp = lp.max((self.pq[pi] + self.pqoff - 2 * b).clamp(-8, 8));
            }
            if self.have_x && self.xq[pi] > -100 {
                lx = lx.max((self.xq[pi] + self.xqoff - 2 * b).clamp(-8, 8));
            }
        }
        let (lwi, lpi, lxi) = (lw + 7, lp + 9, lx + 9);
        let pos = (x == 0) as i32
            | (((x == self.w - 1) as i32) << 1)
            | (((y == 0) as i32) << 2)
            | (((y + 4 >= self.h) as i32) << 3);
        let v = [
            rl * 20 + lwi,
            (lwi * 18 + lpi) * 64 + orl,
            (lpi * 18 + lxi) * 21 + depth,
            ((lwi * 21 + depth) * 8 + rl) * 4 + self.comp,
            (pos * 20 + lwi) * 64 + orl,
            ((lwi * 21 + depth) * 64 + orl) * 4 + self.comp,
            (((lpi * 18 + lxi) * 20 + lwi) * 8 + rl) * 4 + self.orient,
            ((((lwi * 18 + lpi) * 18 + lxi) * 11 + (depth >> 1)) * 9 + rl) * 4 + self.orient,
        ];
        m.code(
            cd,
            bit,
            std::array::from_fn(|j| v[j] as u32 + 4 * PK[j]),
            [
                ((4 * 20 + lwi) * 24 + depth.clamp(0, 23)) as usize,
                ((4 * 12 + rl) * 18 + lpi.clamp(0, 17)) as usize,
                ((4 * 64 + orl) * 4 + self.comp) as usize,
            ],
            [
                (4 * 20 + lwi) as usize,
                ((4 * 18 + lpi) * 18 + lxi) as usize,
                ((4 * 12 + rl) * 21 + depth) as usize,
            ],
        )
    }
    fn sign<C: BitCoder>(
        &mut self,
        m: &mut Mixer,
        cd: &mut C,
        st: &mut [u8],
        nb: &mut [u8],
        i: usize,
        s: usize,
        excl: bool,
        x: usize,
        y: usize,
        b: i32,
        raw: bool,
    ) -> Result<()> {
        let (cx, mut xr) = tier1::sign_context(st, i, s, excl);
        let sd = if raw {
            10
        } else {
            (cx as i32 - 9) * 2 + xr as i32
        };
        if raw {
            xr = 0
        }
        let enc = cd.encoding();
        let neg = self.take(if raw { 19 } else { cx }, enc)? ^ xr as i32;
        let neg = self.event(m, cd, neg, 2, x, y, b, sd, 0)?;
        self.give(if raw { 19 } else { cx }, neg ^ xr as i32, enc)?;
        tier1::make_significant(st, nb, i, s, neg as u8);
        let v = ((1u32 << b) | if b > 0 { 1u32 << (b - 1) } else { 0 }) as i32;
        let ri = self.ri(x, y);
        self.r[ri] = if neg != 0 { -v } else { v };
        Ok(())
    }
    fn run<C: BitCoder>(
        &mut self,
        m: &mut Mixer,
        cd: &mut C,
        style: u32,
        passes: usize,
        top: i32,
    ) -> Result<Vec<usize>> {
        let enc = cd.encoding();
        let s = self.w + 2;
        let mut st = vec![0u8; s * (self.h + 2)];
        let mut nb = st.clone();
        let causal = style & 8 != 0;
        let mut ends = Vec::new();
        for pass in 0..passes {
            let kind = if pass == 0 { 2 } else { (pass - 1) % 3 };
            let raw = style & 1 != 0 && pass >= 10 && kind < 2;
            let b = (top - (pass as i32 + 2) / 3).clamp(0, 30);
            for y0 in (0..self.h).step_by(4) {
                let y1 = (y0 + 4).min(self.h);
                let mut rl = 0i32;
                for x in 0..self.w {
                    let i0 = (y0 + 1) * s + x + 1;
                    let mut start = y0;
                    let mut rc = 0;
                    if kind == 2
                        && y1 - y0 == 4
                        && (st[i0] | st[i0 + s] | st[i0 + 2 * s] | (st[i0 + 3 * s] & (SIG | PIV)))
                            == 0
                        && (nb[i0]
                            | nb[i0 + s]
                            | nb[i0 + 2 * s]
                            | (nb[i0 + 3 * s] & if causal { 31 } else { 255 }))
                            == 0
                    {
                        let rq = if rl == 0 {
                            0
                        } else {
                            ((31 - (rl as u32 | 1).leading_zeros()) as i32 + 1).clamp(0, 7)
                        };
                        let mut rr = 4usize;
                        if enc {
                            if self.take(17, true)? != 0 {
                                rr = ((self.take(18, true)? << 1) | self.take(18, true)?) as usize;
                            }
                            self.event_run(m, cd, (rr < 4) as i32, x, y0, b, rq)?;
                            if rr < 4 {
                                for k in 0..3.min(rr + 1) {
                                    self.event(m, cd, (k == rr) as i32, 1, x, y0 + k, b, 0, 1)?;
                                }
                            }
                        } else {
                            if self.event_run(m, cd, 0, x, y0, b, rq)? != 0 {
                                rr = 3;
                                for k in 0..3 {
                                    if self.event(m, cd, 0, 1, x, y0 + k, b, 0, 1)? != 0 {
                                        rr = k;
                                        break;
                                    }
                                }
                            }
                            self.give(17, (rr < 4) as i32, false)?;
                            if rr < 4 {
                                self.give(18, (rr >> 1) as i32, false)?;
                                self.give(18, (rr & 1) as i32, false)?;
                            }
                        }
                        if rr == 4 {
                            rl += 1;
                            continue;
                        }
                        rl = 0;
                        self.sign(
                            m,
                            cd,
                            &mut st,
                            &mut nb,
                            i0 + rr * s,
                            s,
                            causal && (y0 + rr) % 4 == 3,
                            x,
                            y0 + rr,
                            b,
                            false,
                        )?;
                        start = y0 + rr + 1;
                        rc = 2;
                    }
                    if rc == 0 {
                        rl = 0
                    }
                    for y in start..y1 {
                        let i = i0 + (y - y0) * s;
                        let mut mk = nb[i];
                        if causal && y % 4 == 3 {
                            mk &= 31
                        }
                        if kind == 1 {
                            if st[i] & (SIG | PIV) != SIG {
                                continue;
                            }
                            let cx = if st[i] & REF != 0 {
                                16
                            } else if mk != 0 {
                                15
                            } else {
                                14
                            };
                            let value = self.take(if raw { 19 } else { cx }, enc)?;
                            let value = self.event(
                                m,
                                cd,
                                value,
                                3,
                                x,
                                y,
                                b,
                                if raw { 3 } else { cx as i32 - 14 },
                                0,
                            )?;
                            self.give(if raw { 19 } else { cx }, value, enc)?;
                            st[i] |= REF;
                            let half = if b > 0 { 1 << (b - 1) } else { 0 };
                            let ri = self.ri(x, y);
                            let delta = if value != 0 { half } else { -half };
                            self.r[ri] = self.r[ri].wrapping_add(if self.r[ri] < 0 {
                                -delta
                            } else {
                                delta
                            });
                        } else {
                            if kind == 0 {
                                if st[i] & SIG != 0 || mk == 0 {
                                    continue;
                                }
                                st[i] |= PIV;
                            } else if st[i] & (SIG | PIV) != 0 {
                                continue;
                            }
                            let zc = tier1::zero_context(self.orient as usize, mk);
                            let cx = if raw { 19 } else { zc };
                            let value = self.take(cx, enc)?;
                            let value = self.event(
                                m,
                                cd,
                                value,
                                if kind == 0 { 0 } else { 1 },
                                x,
                                y,
                                b,
                                if raw { 9 } else { zc as i32 },
                                rc,
                            )?;
                            self.give(cx, value, enc)?;
                            if value != 0 {
                                self.sign(
                                    m,
                                    cd,
                                    &mut st,
                                    &mut nb,
                                    i,
                                    s,
                                    causal && y % 4 == 3,
                                    x,
                                    y,
                                    b,
                                    raw,
                                )?;
                            }
                        }
                    }
                }
            }
            if kind == 2 {
                for y in 0..self.h {
                    for x in 0..self.w {
                        st[(y + 1) * s + x + 1] &= !PIV;
                    }
                }
                if style & 32 != 0 {
                    for j in 0..4 {
                        let value = self.take(18, enc)?;
                        let value = m.misc(cd, value, j);
                        self.give(18, value, enc)?;
                    }
                }
            }
            ends.push(self.symbols.len());
        }
        Ok(ends)
    }
}
impl CoefficientModel {
    #[allow(clippy::too_many_arguments)]
    pub fn code_block<C: BitCoder>(
        &mut self,
        cd: &mut C,
        band: usize,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        style: u32,
        passes: usize,
        zbp: i32,
        symbols: &[u8],
        cap: usize,
    ) -> Result<Passes> {
        let g = self
            .bands
            .get(band)
            .ok_or(Error::Invalid("J2K band index"))?
            .clone();
        if w == 0
            || h == 0
            || w > 1024
            || h > 1024
            || w * h > 4096
            || passes > 255
            || !(0..=3).contains(&g.orient)
            || !(-1024..=1024).contains(&zbp)
        {
            return Err(Error::Invalid("J2K codeblock arguments"));
        }
        if cd.encoding() && symbols.len() > cap {
            return Err(Error::Limit("J2K decisions"));
        }
        let mapped = map_range(&g, self.maps.len()).is_some()
            && x0.checked_add(w).is_some_and(|e| e <= g.width)
            && y0.checked_add(h).is_some_and(|e| e <= g.height);
        if mapped && g.comp > 1 && g.comp != self.zcomp {
            self.zcomp = g.comp;
            for b in &self.bands {
                if b.comp == g.comp {
                    if let Some(r) = map_range(b, self.maps.len()) {
                        self.maps[r].fill(0)
                    }
                }
            }
        }
        let mut state = BlockState {
            w,
            h,
            rw: w + 4,
            mbc: g.mb,
            orient: g.orient,
            lvl: g.nres.saturating_sub(g.res).clamp(0, 15),
            comp: g.comp.clamp(0, 3),
            r: vec![0; (w + 4) * (h + 4)],
            pq: vec![-100; w * h],
            xq: vec![-100; w * h],
            ps: vec![0; w * h],
            xs: vec![0; w * h],
            pqoff: 0,
            xqoff: 0,
            have_p: false,
            have_x: false,
            symbols: Vec::new(),
            source: symbols.to_vec(),
            cap,
        };
        if mapped {
            let (_, map) = self.band_map(Some(band)).unwrap();
            for y in -2isize..h as isize + 2 {
                let gy = y0 as isize + y;
                if gy < 0 || gy >= g.height as isize {
                    continue;
                }
                for x in -2isize..w as isize + 2 {
                    if y >= 0 && y < h as isize && x >= 0 && x < w as isize {
                        continue;
                    }
                    let gx = x0 as isize + x;
                    if gx >= 0 && gx < g.width as isize {
                        state.r[((y + 2) as usize) * state.rw + (x + 2) as usize] =
                            mapval(map[gy as usize * g.width + gx as usize]);
                    }
                }
            }
            if let Some((pg, pmap)) = self.band_map(g.parent) {
                state.have_p = true;
                state.pqoff = -2 * pg.mb + 2 * g.mb - 4;
                for y in 0..h {
                    let py = (y0 + y) * pg.height / g.height;
                    for x in 0..w {
                        let px = (x0 + x) * pg.width / g.width;
                        let pv = mapval(pmap[py * pg.width + px]);
                        let sum = 2 * pv.unsigned_abs() as u64
                            + mapval(pmap[py * pg.width + px.saturating_sub(1)]).unsigned_abs()
                                as u64
                            + mapval(pmap[py * pg.width + (px + 1).min(pg.width - 1)])
                                .unsigned_abs() as u64
                            + mapval(pmap[py.saturating_sub(1) * pg.width + px]).unsigned_abs()
                                as u64
                            + mapval(pmap[(py + 1).min(pg.height - 1) * pg.width + px])
                                .unsigned_abs() as u64;
                        state.ps[y * w + x] = pv.signum();
                        state.pq[y * w + x] = if sum != 0 { qlog(sum) } else { -100 };
                    }
                }
            }
            if g.comp > 0 {
                if let Some((g0, xm0)) = self.band_map(g.cross0) {
                    state.have_x = true;
                    state.xqoff = -2 * g0.mb + 2 * g.mb - 2;
                    let other = if g.comp > 1 {
                        self.band_map(g.cross1)
                    } else {
                        None
                    };
                    for y in 0..h {
                        let py = (y0 + y) * g0.height / g.height;
                        for x in 0..w {
                            let px = (x0 + x) * g0.width / g.width;
                            let v0 = mapval(xm0[py * g0.width + px]);
                            let mut sum = 2 * v0.unsigned_abs() as u64;
                            if let Some((g1, xm1)) = other {
                                let cx = (x0 + x) * g1.width / g.width;
                                let cy = (y0 + y) * g1.height / g.height;
                                sum += 2 * mapval(xm1[cy * g1.width + cx]).unsigned_abs() as u64;
                            }
                            state.xs[y * w + x] = v0.signum();
                            state.xq[y * w + x] = if sum != 0 { qlog(sum) } else { -100 };
                        }
                    }
                }
            }
        }
        let pass_ends = state.run(&mut self.mix, cd, style, passes, g.mb - zbp)?;
        if cd.encoding() && state.symbols.len() != symbols.len() {
            return Err(Error::Invalid("J2K unused decisions"));
        }
        if mapped {
            let start = g.map.unwrap();
            for y in 0..h {
                for x in 0..w {
                    self.maps[start + (y0 + y) * g.width + x0 + x] =
                        mapcode(state.r[state.ri(x, y)]);
                }
            }
        }
        Ok(Passes {
            symbols: state.symbols,
            pass_ends,
            used_bits: 0,
        })
    }
}

impl Counter {
    pub(crate) fn code<C: BitCoder>(&mut self, cd: &mut C, bit: i32) -> i32 {
        let b = cd.bit((self.p ^ 32768) as u32, bit);
        self.learn(b);
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::{Decoder, Encoder};
    pub(crate) fn fixture_bands() -> Vec<Band> {
        let mut bands = Vec::new();
        for comp in 0..3 {
            for res in 0..2 {
                let i = bands.len();
                bands.push(Band {
                    comp,
                    res,
                    orient: if res == 0 { 0 } else { 1 },
                    nres: 2,
                    width: 8,
                    height: 8,
                    mb: 12,
                    parent: if res == 1 { Some(i - 1) } else { None },
                    cross0: if comp > 0 { Some(res as usize) } else { None },
                    cross1: if comp > 1 {
                        Some(2 + res as usize)
                    } else {
                        None
                    },
                    map: Some(i * 64),
                });
            }
        }
        bands
    }
    #[test]
    fn classic_model_roundtrip_with_maps() {
        let bands = fixture_bands();
        let bits: Vec<u8> = (0..1024).map(|i| (i * 31 + i / 7) as u8).collect();
        let mut model = CoefficientModel::new(10).unwrap();
        model.tile(bands.clone(), 384).unwrap();
        let mut encoder = Encoder::new(1 << 20);
        let mut fixtures = Vec::new();
        for band in 0..6 {
            for x0 in [0, 4] {
                let block = tier1::Block {
                    width: 4,
                    height: 8,
                    orientation: bands[band].orient as usize,
                    style: 9,
                    passes: 15,
                };
                let passes = tier1::replay(block, &bits, band * 13, bits.len() * 8, 8192).unwrap();
                let encoded = model
                    .code_block(
                        &mut encoder,
                        band,
                        x0,
                        0,
                        4,
                        8,
                        9,
                        15,
                        2,
                        &passes.symbols,
                        8192,
                    )
                    .unwrap();
                assert_eq!(encoded.symbols, passes.symbols);
                fixtures.push((band, x0, passes));
            }
        }
        let bytes = encoder.finish().unwrap();
        let mut model = CoefficientModel::new(10).unwrap();
        model.tile(bands, 384).unwrap();
        let mut decoder = Decoder::new(&bytes);
        for (band, x0, original) in fixtures {
            let decoded = model
                .code_block(&mut decoder, band, x0, 0, 4, 8, 9, 15, 2, &[], 8192)
                .unwrap();
            assert_eq!(decoded.symbols, original.symbols);
            assert_eq!(decoded.pass_ends, original.pass_ends);
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "requires RS_JXL_PLUS_J2K_MODEL_ORACLE original-C helper built with -DPJA_CLASSIC_MODEL"]
fn original_c_classic_model_stream() {
    use super::tier1::oracle::{bands_descriptor, run_env, word};
    use crate::arith::Encoder;
    let bands = tests::fixture_bands();
    let bits: Vec<u8> = (0..1024).map(|i| (i * 31 + i / 7) as u8).collect();
    let mut model = CoefficientModel::new(10).unwrap();
    model.tile(bands.clone(), 384).unwrap();
    let mut encoder = Encoder::new(1 << 20);
    let mut desc = bands_descriptor(&bands, 384, 12);
    for band in 0..6 {
        for x0 in [0, 4] {
            let block = tier1::Block {
                width: 4,
                height: 8,
                orientation: bands[band].orient as usize,
                style: 9,
                passes: 15,
            };
            let passes = tier1::replay(block, &bits, band * 13, bits.len() * 8, 8192).unwrap();
            for n in [band, x0, 0, 4, 8, 9, 15, 2, passes.symbols.len()] {
                word(&mut desc, n)
            }
            desc.extend(&passes.symbols);
            model
                .code_block(
                    &mut encoder,
                    band,
                    x0,
                    0,
                    4,
                    8,
                    9,
                    15,
                    2,
                    &passes.symbols,
                    8192,
                )
                .unwrap();
        }
    }
    let rust = encoder.finish().unwrap();
    assert_eq!(
        run_env("RS_JXL_PLUS_J2K_MODEL_ORACLE", "model", &desc),
        rust
    );
}
