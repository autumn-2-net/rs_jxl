// PJA MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Source: pas/pjaj2khm.c. Shared exact mixer: model.rs. See NOTICE-PJA.
//! HTJ2K coefficient-value model (stream-v8 mode3), separate from bitstream coding.
use super::model::{Band, Counter, Mixer, qlog};
use crate::{Error, Result, arith::BitCoder};
const PK: [u32; 8] = [
    32 * 42 * 5,
    32 * 25 * 25 * 4,
    32 * 42 * 18 * 18,
    32 * 42 * 34 * 4,
    32 * 42 * 24 * 64 * 4,
    32 * 5 * 18 * 24,
    32 * 128 * 11 * 4,
    32 * 42 * 18 * 24,
];
fn range(g: &Band, n: usize) -> Option<std::ops::Range<usize>> {
    if g.width == 0 || g.height == 0 {
        return None;
    }
    let start = g.map?;
    let end = start.checked_add(g.width.checked_mul(g.height)?)?;
    if end > n { None } else { Some(start..end) }
}
fn mapcode(v: i32, bq: i32) -> u8 {
    if v == 0 {
        0
    } else {
        (qlog(v.unsigned_abs() as u64) + 2 * bq + 1).clamp(1, 127) as u8
            | if v < 0 { 128 } else { 0 }
    }
}
fn mapval(c: u8, bq: i32) -> i32 {
    if c & 127 == 0 {
        return 0;
    }
    let l = (c & 127) as i32 - 1;
    let e = (l >> 1) - 1 - bq;
    let mant = if l & 1 != 0 { 3i32 } else { 2 };
    let v = if e >= 0 {
        mant << e.min(28)
    } else if -e > 31 {
        0
    } else {
        mant >> -e
    };
    if c & 128 != 0 { -v } else { v }
}
fn lg(v: u64, top: i32) -> i32 {
    if v == 0 {
        0
    } else {
        qlog(v).clamp(0, top - 2) + 1
    }
}
#[derive(Default)]
struct Features {
    ln: i32,
    ln2: i32,
    nsig: i32,
    wi: i32,
    ni: i32,
    lp: i32,
    lx: i32,
    mx: i32,
    pat: i32,
    sp: i32,
    sx: i32,
}
struct State {
    w: usize,
    h: usize,
    rw: usize,
    orient: i32,
    lvl: i32,
    comp: i32,
    dep: i32,
    bq: i32,
    r: Vec<i32>,
    flags: Vec<u8>,
    pq: Vec<i32>,
    xq: Vec<i32>,
    ps: Vec<i32>,
    xs: Vec<i32>,
    pqoff: i32,
    xqoff: i32,
    have_p: bool,
    have_x: bool,
}
impl State {
    fn ri(&self, x: usize, y: usize) -> usize {
        (y + 2) * self.rw + x + 2
    }
    fn features(&self, x: usize, y: usize) -> Features {
        let i = self.ri(x, y);
        let rw = self.rw;
        let pi = y * self.w + x;
        let ab = |j: usize| self.r[j].unsigned_abs() as u64;
        let (w, n, nw, ne, ww, nn, e, s) = (
            ab(i - 1),
            ab(i - rw),
            ab(i - rw - 1),
            ab(i - rw + 1),
            ab(i - 2),
            ab(i - 2 * rw),
            ab(i + 1),
            ab(i + rw),
        );
        let sum = 2 * (w + n) + nw + ne + e + s + (ww + nn) / 2;
        let mut f = Features {
            ln: lg(sum, 42),
            ln2: lg(ww + nn + ab(i - rw + 2) + ab(i - rw - 2), 34),
            nsig: (w != 0) as i32 + (n != 0) as i32 + (nw != 0) as i32 + (ne != 0) as i32,
            wi: lg(w, 25),
            ni: lg(n, 25),
            mx: lg(w.max(n).max(nw).max(ne), 42),
            pat: (w != 0) as i32
                | (((n != 0) as i32) << 1)
                | (((nw != 0) as i32) << 2)
                | (((ne != 0) as i32) << 3)
                | (((ww != 0) as i32) << 4)
                | (((nn != 0) as i32) << 5)
                | (((ab(i - rw + 2) != 0) as i32) << 6),
            sp: 1,
            sx: 1,
            ..Features::default()
        };
        if self.have_p {
            f.sp = self.ps[pi] + 1;
            if self.pq[pi] > -100 {
                f.lp = (self.pq[pi] + self.pqoff - 2 * self.bq).clamp(-8, 8) + 9;
            }
        }
        if self.have_x {
            f.sx = self.xs[pi] + 1;
            if self.xq[pi] > -100 {
                f.lx = (self.xq[pi] + self.xqoff - 2 * self.bq).clamp(-8, 8) + 9;
            }
        }
        f
    }
    fn event<C: BitCoder>(
        &self,
        m: &mut Mixer,
        cd: &mut C,
        bit: i32,
        k: i32,
        a: i32,
        f: &Features,
    ) -> Result<i32> {
        let orl = self.orient * 16 + self.lvl;
        let mut v = [
            (a * 42 + f.ln) * 5 + f.nsig,
            ((a * 25 + f.wi.clamp(0, 24)) * 25 + f.ni.clamp(0, 24)) * 4 + self.orient,
            ((a * 42 + f.ln) * 18 + f.lp) * 18 + f.lx,
            ((a * 42 + f.ln) * 34 + f.ln2) * 4 + self.orient,
            (((a * 42 + f.ln) * 24 + self.dep) * 64 + orl) * 4 + self.comp,
            ((a * 5 + f.nsig) * 18 + f.lp) * 24 + self.dep,
            0,
            ((a * 42 + f.mx) * 18 + f.lp) * 24 + self.dep,
        ];
        v[6] = if k == 1 || k == 2 {
            ((2 * if k == 1 { a } else { a & 15 } - f.ln + 30).clamp(0, 63) * 32 + a) * 4
                + self.orient
        } else {
            ((a * 128 + f.pat) * 11 + (f.mx >> 2)) * 4 + self.orient
        };
        m.code(
            cd,
            bit,
            std::array::from_fn(|j| v[j] as u32 + k as u32 * PK[j]),
            [
                ((k * 42 + f.ln) * 24 + self.dep) as usize,
                ((k * 32 + a) * 18 + f.lp) as usize,
                ((k * 64 + orl) * 4 + self.comp) as usize,
            ],
            [
                (k * 42 + f.ln) as usize,
                ((k * 18 + f.lp) * 18 + f.lx) as usize,
                ((k * 32 + a) * 24 + self.dep) as usize,
            ],
        )
    }
    fn sign<C: BitCoder>(
        &self,
        m: &mut Mixer,
        cd: &mut C,
        bit: i32,
        x: usize,
        y: usize,
        f: &Features,
    ) -> Result<i32> {
        let i = self.ri(x, y);
        let rw = self.rw;
        let sign = |j: usize| self.r[j].signum() + 1;
        let (sw, sn, snw, sne, se, ss) = (
            sign(i - 1),
            sign(i - rw),
            sign(i - rw - 1),
            sign(i - rw + 1),
            sign(i + 1),
            sign(i + rw),
        );
        let a = (sw * 3 + sn) * 3 + se;
        let b = (snw * 3 + sne) * 3 + ss;
        let orl = self.orient * 16 + self.lvl;
        let v = [
            (a * 27 + b) * 4 + self.orient,
            (a * 3 + f.sp) * 3 + f.sx,
            ((sw * 3 + sn) * 3 + f.sp) * 64 + orl,
            (b * 3 + f.sp) * 4 + self.orient,
            (a * 3 + f.sx) * 4 + self.comp,
            ((sw * 3 + sn) * 42 + f.ln) * 4 + self.orient,
            (f.sp * 3 + f.sx) * 18 + f.lp,
            (a * 27 + b) * 9 + f.sp * 3 + f.sx,
        ];
        m.code(
            cd,
            bit,
            std::array::from_fn(|j| v[j] as u32 + 3 * PK[j]),
            [
                ((3 * 42 + a.clamp(0, 41)) * 24 + self.dep) as usize,
                ((3 * 32 + sw * 3 + sn) * 18 + f.sp * 3 + f.sx) as usize,
                ((3 * 64 + orl) * 4 + self.comp) as usize,
            ],
            [
                (3 * 42 + b.clamp(0, 41)) as usize,
                ((3 * 18 + f.sp * 3 + f.sx) * 18 + sw * 3 + sn) as usize,
                ((3 * 32 + sw * 3 + sn) * 24 + self.dep) as usize,
            ],
        )
    }
    fn neighbor_sig(&self, x: usize, y: usize, ys: usize, causal: bool) -> bool {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx < 0 || ny < 0 || nx >= self.w as isize || ny >= self.h as isize {
                    continue;
                }
                let f = self.flags[ny as usize * self.w + nx as usize];
                if f & 1 != 0 {
                    if !(causal && ny as usize >= ys + 4) {
                        return true;
                    }
                } else if f & 6 == 6 {
                    return true;
                }
            }
        }
        false
    }
    fn run<C: BitCoder>(
        &mut self,
        m: &mut Mixer,
        low: &mut [Counter],
        cd: &mut C,
        causal: bool,
        np: usize,
        q: &mut [i32],
    ) -> Result<()> {
        let enc = cd.encoding();
        let sh = (np > 1) as u32;
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let ri = self.ri(x, y);
                let mut mu = if enc { q[i].unsigned_abs() >> sh } else { 0 };
                let mut neg = if enc { (q[i] < 0) as i32 } else { 0 };
                if mu > 1 << 28 {
                    return Err(Error::Invalid("HT model magnitude"));
                }
                let mut len = (32 - mu.leading_zeros()) as i32;
                let f = self.features(x, y);
                let nz = self.event(m, cd, (mu != 0) as i32, 0, 0, &f)?;
                if nz == 0 {
                    self.r[ri] = 0;
                    self.flags[i] = 0;
                    if !enc {
                        q[i] = 0
                    }
                    continue;
                }
                if enc {
                    for k in 1..29.min(len + 1) {
                        self.event(m, cd, (len > k) as i32, 1, k.clamp(0, 31), &f)?;
                    }
                } else {
                    len = 1;
                    while len < 29 && self.event(m, cd, 0, 1, len.clamp(0, 31), &f)? != 0 {
                        len += 1;
                    }
                }
                let mut v = 1u32;
                for j in (0..len - 1).rev() {
                    let mut b = if enc { ((mu >> j) & 1) as i32 } else { 0 };
                    if j >= len - 3 {
                        let aux = if j == len - 2 {
                            len.clamp(0, 15)
                        } else {
                            16 + len.clamp(0, 7) * 2 + (v & 1) as i32
                        };
                        b = self.event(m, cd, b, 2, aux, &f)?;
                    } else {
                        let idx = ((len.clamp(0, 31) * 32 + (len - 2 - j).clamp(0, 31)) * 2
                            + (v & 1) as i32) as usize;
                        b = low[idx].code(cd, b);
                    }
                    v = (v << 1) | b as u32;
                }
                mu = v;
                neg = self.sign(m, cd, neg, x, y, &f)?;
                let r = (((2 * mu + 1) << sh).min(1 << 30)) as i32;
                self.r[ri] = if neg != 0 { -r } else { r };
                self.flags[i] = 1;
                if !enc {
                    q[i] = (mu << sh) as i32 * if neg != 0 { -1 } else { 1 };
                }
            }
        }
        if np == 1 {
            return Ok(());
        }
        for ys in (0..self.h).step_by(4) {
            for xg in (0..self.w).step_by(4) {
                let mut new = Vec::new();
                for x in xg..(xg + 4).min(self.w) {
                    for y in ys..(ys + 4).min(self.h) {
                        let i = y * self.w + x;
                        if self.flags[i] & 1 != 0 {
                            continue;
                        }
                        if !self.neighbor_sig(x, y, ys, causal) {
                            self.flags[i] |= 4;
                            continue;
                        }
                        let f = self.features(x, y);
                        let b = self.event(
                            m,
                            cd,
                            if enc {
                                (q[i].unsigned_abs() == 1) as i32
                            } else {
                                0
                            },
                            4,
                            0,
                            &f,
                        )?;
                        self.flags[i] |= 4 | ((b as u8) << 1);
                        if b != 0 {
                            new.push(i);
                            let ri = self.ri(x, y);
                            self.r[ri] = 3;
                        }
                    }
                }
                for i in new {
                    let (x, y) = (i % self.w, i / self.w);
                    let f = self.features(x, y);
                    let neg =
                        self.sign(m, cd, if enc { (q[i] < 0) as i32 } else { 0 }, x, y, &f)?;
                    let ri = self.ri(x, y);
                    if neg != 0 {
                        self.r[ri] = -3
                    }
                    if !enc {
                        q[i] = if neg != 0 { -1 } else { 1 };
                    }
                }
            }
        }
        if np > 2 {
            for y in 0..self.h {
                for x in 0..self.w {
                    let i = y * self.w + x;
                    if self.flags[i] & 1 == 0 {
                        continue;
                    }
                    let ri = self.ri(x, y);
                    let a = self.r[ri].unsigned_abs();
                    let f = self.features(x, y);
                    let b = self.event(
                        m,
                        cd,
                        if enc {
                            (q[i].unsigned_abs() & 1) as i32
                        } else {
                            0
                        },
                        5,
                        (32 - (a >> 2).leading_zeros()).min(31) as i32,
                        &f,
                    )?;
                    let a = if b != 0 { a + 1 } else { a - 1 };
                    self.r[ri] = if self.r[ri] < 0 {
                        -(a as i32)
                    } else {
                        a as i32
                    };
                    if !enc {
                        q[i] += if q[i] < 0 { -b } else { b };
                    }
                }
            }
        }
        Ok(())
    }
}
pub struct HtModel {
    mix: Mixer,
    low: Vec<Counter>,
    bands: Vec<Band>,
    maps: Vec<u8>,
    zcomp: i32,
}
impl HtModel {
    pub fn new(hash_bits: u32) -> Result<Self> {
        Ok(Self {
            mix: Mixer::new(
                hash_bits,
                7,
                PK,
                [7 * 42 * 24, 7 * 32 * 18, 7 * 64 * 4],
                [7 * 42, 7 * 18 * 18, 7 * 32 * 24],
            )?,
            low: vec![Counter::default(); 2048],
            bands: Vec::new(),
            maps: Vec::new(),
            zcomp: -1,
        })
    }
    pub fn tile(&mut self, bands: Vec<Band>, map_bytes: usize) -> Result<()> {
        if map_bytes > 1 << 27 {
            return Err(Error::Limit("HT model maps"));
        }
        for g in &bands {
            if g.width.checked_mul(g.height).is_none()
                || g.width > i32::MAX as usize
                || g.height > i32::MAX as usize
                || !(-1024..=1024).contains(&g.mb)
            {
                return Err(Error::Invalid("HT model band geometry"));
            }
        }
        self.bands = bands;
        self.maps.resize(map_bytes, 0);
        self.zcomp = -1;
        for g in &self.bands {
            if g.comp <= 1 {
                if let Some(r) = range(g, self.maps.len()) {
                    self.maps[r].fill(0)
                }
            }
        }
        Ok(())
    }
    fn band_map(&self, b: Option<usize>) -> Option<(&Band, &[u8])> {
        let g = self.bands.get(b?)?;
        Some((g, &self.maps[range(g, self.maps.len())?]))
    }
}
impl HtModel {
    #[allow(clippy::too_many_arguments)]
    pub fn code_block<C: BitCoder>(
        &mut self,
        cd: &mut C,
        band: usize,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        causal: bool,
        np: usize,
        b: i32,
        q: &mut [i32],
    ) -> Result<()> {
        let g = self
            .bands
            .get(band)
            .ok_or(Error::Invalid("HT model band"))?
            .clone();
        if w == 0
            || h == 0
            || w > 1024
            || h > 1024
            || w * h > 4096
            || !(1..=3).contains(&np)
            || !(-64..=64).contains(&b)
            || q.len() != w * h
            || !(0..=3).contains(&g.orient)
        {
            return Err(Error::Invalid("HT model codeblock arguments"));
        }
        let bq = (b - (np > 1) as i32).max(0);
        let mapped = range(&g, self.maps.len()).is_some()
            && x0.checked_add(w).is_some_and(|e| e <= g.width)
            && y0.checked_add(h).is_some_and(|e| e <= g.height);
        if mapped && g.comp > 1 && g.comp != self.zcomp {
            self.zcomp = g.comp;
            for band in &self.bands {
                if band.comp == g.comp {
                    if let Some(r) = range(band, self.maps.len()) {
                        self.maps[r].fill(0)
                    }
                }
            }
        }
        let mut state = State {
            w,
            h,
            rw: w + 4,
            orient: g.orient,
            lvl: g.nres.saturating_sub(g.res).clamp(0, 15),
            comp: g.comp.clamp(0, 3),
            dep: (g.mb - b).clamp(0, 23),
            bq,
            r: vec![0; (w + 4) * (h + 4)],
            flags: vec![0; w * h],
            pq: vec![-100; w * h],
            xq: vec![-100; w * h],
            ps: vec![0; w * h],
            xs: vec![0; w * h],
            pqoff: 0,
            xqoff: 0,
            have_p: false,
            have_x: false,
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
                        state.r[(y + 2) as usize * state.rw + (x + 2) as usize] =
                            mapval(map[gy as usize * g.width + gx as usize], bq);
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
                        let pc = pmap[py * pg.width + px];
                        let sum = 2 * mapval(pc, -1).unsigned_abs() as u64
                            + mapval(pmap[py * pg.width + px.saturating_sub(1)], -1).unsigned_abs()
                                as u64
                            + mapval(pmap[py * pg.width + (px + 1).min(pg.width - 1)], -1)
                                .unsigned_abs() as u64
                            + mapval(pmap[py.saturating_sub(1) * pg.width + px], -1).unsigned_abs()
                                as u64
                            + mapval(pmap[(py + 1).min(pg.height - 1) * pg.width + px], -1)
                                .unsigned_abs() as u64;
                        state.ps[y * w + x] = if pc == 0 {
                            0
                        } else if pc & 128 != 0 {
                            -1
                        } else {
                            1
                        };
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
                            let c0 = xm0[py * g0.width + px];
                            let mut sum = 2 * mapval(c0, -1).unsigned_abs() as u64;
                            if let Some((g1, xm1)) = other {
                                let cx = (x0 + x) * g1.width / g.width;
                                let cy = (y0 + y) * g1.height / g.height;
                                sum +=
                                    2 * mapval(xm1[cy * g1.width + cx], -1).unsigned_abs() as u64;
                            }
                            state.xs[y * w + x] = if c0 == 0 {
                                0
                            } else if c0 & 128 != 0 {
                                -1
                            } else {
                                1
                            };
                            state.xq[y * w + x] = if sum != 0 { qlog(sum) } else { -100 };
                        }
                    }
                }
            }
        }
        state.run(&mut self.mix, &mut self.low, cd, causal, np, q)?;
        if mapped {
            let start = g.map.unwrap();
            for y in 0..h {
                for x in 0..w {
                    self.maps[start + (y0 + y) * g.width + x0 + x] =
                        mapcode(state.r[state.ri(x, y)], bq);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::{Decoder, Encoder};
    #[test]
    fn ht_model_roundtrip_with_maps() {
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
        for np in 1..=3 {
            let mut model = HtModel::new(10).unwrap();
            model.tile(bands.clone(), 384).unwrap();
            let mut encoder = Encoder::new(1 << 20);
            let mut fixtures = Vec::new();
            for band in 0..6 {
                for x0 in [0, 4] {
                    let mut q: Vec<i32> = (0..32)
                        .map(|i| {
                            let a = if i % 7 == 0 {
                                0
                            } else {
                                2 * (i * 13 % 113 + 1) + if np == 3 { i % 2 } else { 0 }
                            };
                            if i % 3 == 0 { -(a as i32) } else { a as i32 }
                        })
                        .collect();
                    let original = q.clone();
                    model
                        .code_block(&mut encoder, band, x0, 0, 4, 8, true, np, 7, &mut q)
                        .unwrap();
                    fixtures.push((band, x0, original));
                }
            }
            let bytes = encoder.finish().unwrap();
            let mut model = HtModel::new(10).unwrap();
            model.tile(bands.clone(), 384).unwrap();
            let mut decoder = Decoder::new(&bytes);
            for (band, x0, original) in fixtures {
                let mut q = vec![0; 32];
                model
                    .code_block(&mut decoder, band, x0, 0, 4, 8, true, np, 7, &mut q)
                    .unwrap();
                assert_eq!(q, original, "np{np} band{band} x{x0}");
            }
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "requires RS_JXL_PLUS_J2K_HT_MODEL_ORACLE helper built with -DPJA_HT_MODEL"]
fn original_c_ht_model_stream() {
    use super::tier1::oracle::{bands_descriptor, run_env, word};
    use crate::arith::Encoder;
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
    for np in 1..=3 {
        let mut model = HtModel::new(10).unwrap();
        model.tile(bands.clone(), 384).unwrap();
        let mut encoder = Encoder::new(1 << 20);
        let mut desc = bands_descriptor(&bands, 384, 12);
        for band in 0..6 {
            for x0 in [0, 4] {
                let mut q: Vec<i32> = (0..32)
                    .map(|i| {
                        let a = if i % 7 == 0 {
                            0
                        } else {
                            2 * (i * 13 % 113 + 1) + if np == 3 { i % 2 } else { 0 }
                        };
                        if i % 3 == 0 { -(a as i32) } else { a as i32 }
                    })
                    .collect();
                for n in [band, x0, 0, 4, 8, 1, np, 7] {
                    word(&mut desc, n)
                }
                for &v in &q {
                    desc.extend(v.to_le_bytes())
                }
                model
                    .code_block(&mut encoder, band, x0, 0, 4, 8, true, np, 7, &mut q)
                    .unwrap();
            }
        }
        assert_eq!(
            run_env("RS_JXL_PLUS_J2K_HT_MODEL_ORACLE", "model", &desc),
            encoder.finish().unwrap(),
            "np{np}"
        );
    }
}
