// PJA MIT: pas/pjaj2kpm.c pixel models; see ../../NOTICE-PJA.
use crate::{Error, Result, arith::BitCoder};
#[path = "tables/model.rs"]
mod tables;
use super::model::shr;
use tables::{J2KM_RC, J2KM_SQ, J2KM_ST};
#[derive(Clone, Debug)]
pub(crate) struct Plane {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub samples: Vec<u16>,
}
#[derive(Clone, Copy, Default)]
struct Counter {
    p: u16,
    f: u16,
    n: u8,
}
impl Counter {
    fn learn(&mut self, bit: i32) {
        let p = (self.p ^ 32768) as u32;
        let d = if bit != 0 { 65535 - p } else { p - 1 };
        let step = d * J2KM_RC[self.n as usize] as u32 >> 16;
        self.p = (if bit != 0 { p + step } else { p - step }) as u16 ^ 32768;
        self.n = self.n.saturating_add(1);
        let f = (self.f ^ 32768) as i32;
        self.f = (f + ((if bit != 0 { 65535 } else { 0 }) - f >> 4)) as u16 ^ 32768;
    }
}
#[derive(Clone)]
struct Weights {
    w: [i32; 14],
    count: u16,
}
#[derive(Clone, Copy)]
struct Bias {
    a: i32,
    b: i32,
    c: i32,
    n: i32,
}
impl Default for Bias {
    fn default() -> Self {
        Self {
            a: 4,
            b: 0,
            c: 0,
            n: 1,
        }
    }
}
impl Bias {
    fn learn(&mut self, e: i32) {
        self.a += e.abs();
        self.b += e;
        if self.n == 64 {
            self.a >>= 1;
            self.b >>= 1;
            self.n >>= 1;
        }
        self.n += 1;
        if self.b + self.n <= 0 {
            self.b += self.n;
            if self.b <= -self.n {
                self.b = -self.n + 1;
            }
            self.c = (self.c - 1).max(-128);
        } else if self.b > 0 {
            self.b -= self.n;
            if self.b > 0 {
                self.b = 0;
            }
            self.c = (self.c + 1).min(127);
        }
    }
}
fn allocated<T: Clone>(n: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Limit("J2K pixel model allocation"))?;
    v.resize(n, value);
    Ok(v)
}
struct Model {
    t: Vec<Vec<Counter>>,
    w: [Vec<Weights>; 4],
    apm: [Vec<[u16; 33]>; 4],
    bias: Vec<Bias>,
    lms: [i32; 8],
    hb: u32,
}
impl Model {
    fn new(hb: u32) -> Result<Self> {
        if !(12..=22).contains(&hb) {
            return Err(Error::Invalid("J2K pixel hash bits"));
        }
        let h = 1usize << hb;
        let sizes = [
            1024,
            65536,
            h,
            h,
            h / 4,
            h / 4,
            h / 4,
            65536,
            65536,
            65536,
            65536,
            65536,
            65536,
            65536,
            65536,
            65536,
            65536,
            4096,
            4096,
            4096,
            1024,
            1024,
        ];
        let mut t = Vec::new();
        for n in sizes {
            t.push(allocated(n, Counter::default())?);
        }
        let w = [
            allocated(
                256,
                Weights {
                    w: [65536 / 7; 14],
                    count: 0,
                },
            )?,
            allocated(
                512,
                Weights {
                    w: [65536 / 5; 14],
                    count: 0,
                },
            )?,
            allocated(
                512,
                Weights {
                    w: [65536 / 5; 14],
                    count: 0,
                },
            )?,
            allocated(
                512,
                Weights {
                    w: [65536 / 5; 14],
                    count: 0,
                },
            )?,
        ];
        let init = std::array::from_fn(|i| J2KM_SQ[((i as i32 - 16) * 128 + 3072) as usize]);
        Ok(Self {
            t,
            w,
            apm: [
                allocated(1024, init)?,
                allocated(1024, init)?,
                allocated(1024, init)?,
                allocated(1024, init)?,
            ],
            bias: allocated(730, Bias::default())?,
            lms: [0; 8],
            hb,
        })
    }
    fn mix<C: BitCoder>(
        &mut self,
        cd: &mut C,
        bit: i32,
        keys: &[(usize, usize)],
        group: usize,
        sel: usize,
        map: usize,
    ) -> i32 {
        let mut st = [0i32; 14];
        for (i, &(t, k)) in keys.iter().enumerate() {
            let c = self.t[t][k];
            st[2 * i] = J2KM_ST[((c.p ^ 32768) >> 4) as usize] as i32;
            st[2 * i + 1] = J2KM_ST[((c.f ^ 32768) >> 4) as usize] as i32;
        }
        let ws = &mut self.w[group][sel];
        let dot = crate::mix_simd::dot(&st[..2 * keys.len()], &ws.w[..2 * keys.len()]);
        let x = shr(dot, 16).clamp(-3071, 3071);
        let pm = J2KM_SQ[(x + 3072) as usize] as i32;
        let pos = (x.clamp(-2047, 2047) + 2048) * 32;
        let lo = (pos >> 12) as usize;
        let fr = pos & 4095;
        let t = &mut self.apm[group][map];
        let pa = (t[lo] as i32 * (4096 - fr) + t[lo + 1] as i32 * fr) >> 12;
        let bit = cd.bit(((pm + pa) >> 1) as u32, bit);
        let mut lr = 6 + 31457 / (1024 + 64 * (ws.count as i32 >> 6));
        if group == 0 {
            lr = lr * 3 / 2;
        }
        ws.count = ws.count.saturating_add(1);
        let e = (((bit << 16) - pm) >> 4) * lr;
        crate::mix_simd::update(
            &mut ws.w[..2 * keys.len()],
            &st[..2 * keys.len()],
            e,
            16,
            1 << 22,
        );
        let tg = if bit != 0 { 65535 } else { 0 };
        t[lo] = (t[lo] as i32 + shr(((tg - t[lo] as i32) * (4096 - fr)) >> 6, 12)) as u16;
        t[lo + 1] = (t[lo + 1] as i32 + shr(((tg - t[lo + 1] as i32) * fr) >> 6, 12)) as u16;
        for &(t, k) in keys {
            self.t[t][k].learn(bit);
        }
        bit
    }
    fn single<C: BitCoder>(&mut self, cd: &mut C, bit: i32, t: usize, k: usize) -> i32 {
        let c = &mut self.t[t][k];
        let b = cd.bit((c.p ^ 32768) as u32, bit);
        c.learn(b);
        b
    }
    fn bitonal<C: BitCoder>(&mut self, cd: &mut C, p: &mut Plane) -> Result<()> {
        let (w, h) = (p.width, p.height);
        let hb = self.hb;
        let mask = (1u32 << hb) - 1;
        let sh = 32 - hb;
        let xs = 64 - (hb - 2);
        for y in 0..h {
            let fetch = |data: &[u16], row: usize, x: isize| -> u64 {
                if y < row || x < 0 || x >= w as isize {
                    0
                } else {
                    (data[(y - row) * w + x as usize] & 1) as u64
                }
            };
            let (mut w2, mut w1, mut w3, mut w4, mut w6, mut w8, mut wc, mut wl, mut w0) =
                (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
            for x in -4..=4 {
                w2 = w2 << 1 | fetch(&p.samples, 2, x);
                w6 = w6 << 1 | fetch(&p.samples, 6, x);
                w8 = w8 << 1 | fetch(&p.samples, 8, x);
                wc = wc << 1 | fetch(&p.samples, 12, x);
            }
            for x in -6..=6 {
                w1 = w1 << 1 | fetch(&p.samples, 1, x);
            }
            for x in -8..=8 {
                w3 = w3 << 1 | fetch(&p.samples, 3, x);
                w4 = w4 << 1 | fetch(&p.samples, 4, x);
            }
            for x in -24..=24 {
                wl = wl << 1 | fetch(&p.samples, 1, x);
            }
            for x in 0..w {
                let t1 = (((w2 >> 3) & 7) << 7 | ((w1 >> 4) & 31) << 2 | (w0 & 3)) as u32;
                let t2 = (((w2 >> 2) & 31) << 11 | ((w1 >> 3) & 127) << 4 | (w0 & 15)) as u32;
                let t3 = (((w2 >> 1) & 127) << 15 | ((w1 >> 2) & 511) << 6 | (w0 & 63)) as u32;
                let t4 = ((w2 & 511) << 21 | (w1 & 8191) << 8 | (w0 & 255)) as u32;
                let h3 = if hb >= 22 {
                    t3
                } else {
                    t3.wrapping_add(1).wrapping_mul(0x9e3779b1) >> sh
                };
                let h4 = t4
                    .wrapping_add(1)
                    .wrapping_mul(0x9e3779b1)
                    .wrapping_add(0x632be5ab)
                    >> sh;
                let mut keys = [
                    (0, t1 as usize),
                    (1, t2 as usize),
                    (2, (h3 & mask) as usize),
                    (3, (h4 & mask) as usize),
                    (4, 0),
                    (5, 0),
                    (6, 0),
                ];
                let mut k = ((w4 >> 4) & 511) << 40 ^ w3 << 22 ^ t3 as u64;
                keys[4].1 = (k.wrapping_mul(0x9e3779b97f4a7c15) >> xs) as usize;
                k = ((wl >> 8) & 0x1ffffffff) << 16 ^ (w0 & 65535) ^ ((w2 >> 2) & 31) << 50;
                keys[5].1 = (k.wrapping_add(7).wrapping_mul(0xd6e8feb86659fd93) >> xs) as usize;
                k = t1 as u64;
                for window in [w6, w8, wc] {
                    k = k << 3 | ((window >> 8) & 1) << 2 | ((window >> 4) & 1) << 1 | (window & 1);
                }
                k = k << 4
                    | ((wl >> 44) & 1) << 3
                    | ((wl >> 36) & 1) << 2
                    | ((wl >> 12) & 1) << 1
                    | ((wl >> 4) & 1);
                k = k << 2 | ((w0 >> 11) & 1) << 1 | ((w0 >> 19) & 1);
                keys[6].1 = (k.wrapping_add(3).wrapping_mul(0x9e3779b97f4a7c15) >> xs) as usize;
                let sel = (((w2 >> 3) & 7) << 5 | ((w1 >> 5) & 7) << 2 | (w0 & 3)) as usize;
                let bit = self.mix(
                    cd,
                    (p.samples[y * w + x] & 1) as i32,
                    &keys,
                    0,
                    sel,
                    t1 as usize,
                );
                if !cd.encoding() {
                    p.samples[y * w + x] = bit as u16;
                }
                let x = x as isize;
                w2 = (w2 << 1 | fetch(&p.samples, 2, x + 5)) & 511;
                w1 = (w1 << 1 | fetch(&p.samples, 1, x + 7)) & 8191;
                w0 = (w0 << 1 | bit as u64) & 0xffffffff;
                w3 = (w3 << 1 | fetch(&p.samples, 3, x + 9)) & 0x1ffff;
                w4 = (w4 << 1 | fetch(&p.samples, 4, x + 9)) & 0x1ffff;
                w6 = (w6 << 1 | fetch(&p.samples, 6, x + 5)) & 511;
                w8 = (w8 << 1 | fetch(&p.samples, 8, x + 5)) & 511;
                wc = (wc << 1 | fetch(&p.samples, 12, x + 5)) & 511;
                wl = (wl << 1 | fetch(&p.samples, 1, x + 25)) & 0x1ffffffffffff;
            }
        }
        Ok(())
    }
    fn samples<C: BitCoder>(
        &mut self,
        cd: &mut C,
        p: &mut Plane,
        po: Option<&Plane>,
        cls: usize,
    ) -> Result<()> {
        let (w, h, depth) = (p.width, p.height, p.depth);
        let range = 1i32 << depth;
        let maxv = range - 1;
        let half = range >> 1;
        let (t1, t2, t3) = crate::jls::default_thresholds(maxv, 0);
        let mut res = allocated(18 * w, 0i32)?;
        for y in 0..h {
            for x in 0..w {
                let at = y * w + x;
                let e0 = (y & 1) * w;
                let e1 = ((y.wrapping_sub(1)) & 1) * w;
                let pe0 = 2 * w + (y & 1) * 8 * w;
                let pe1 = 2 * w + ((y.wrapping_sub(1)) & 1) * 8 * w;
                let b = if y > 0 {
                    p.samples[at - w] as i32
                } else if x > 0 {
                    p.samples[at - 1] as i32
                } else {
                    half
                };
                let a = if x > 0 { p.samples[at - 1] as i32 } else { b };
                let c = if y > 0 && x > 0 {
                    p.samples[at - w - 1] as i32
                } else {
                    b
                };
                let d = if y > 0 && x + 1 < w {
                    p.samples[at - w + 1] as i32
                } else {
                    b
                };
                let eo = if let Some(o) = po {
                    let ob = if y > 0 {
                        o.samples[at - w] as i32
                    } else if x > 0 {
                        o.samples[at - 1] as i32
                    } else {
                        half
                    };
                    let oa = if x > 0 { o.samples[at - 1] as i32 } else { ob };
                    let oc = if y > 0 && x > 0 {
                        o.samples[at - w - 1] as i32
                    } else {
                        ob
                    };
                    o.samples[at] as i32 - crate::jls::med(oa, ob, oc)
                } else {
                    0
                };
                let ctx = (quant(d - b, t1, t2, t3) * 9 + quant(b - c, t1, t2, t3)) * 9
                    + quant(c - a, t1, t2, t3);
                let sgn = if ctx < 0 { -1 } else { 1 };
                let ctx = ctx.unsigned_abs() as usize;
                let med = crate::jls::med(a, b, c);
                let mut pred = [
                    med,
                    a + b - c,
                    b + ((d - c) >> 1),
                    a + ((d - b) >> 1),
                    if po.is_some() {
                        med + eo
                    } else {
                        (a + d + 1) >> 1
                    },
                    b,
                    a,
                    0,
                ];
                let nn = if y > 1 {
                    p.samples[at - 2 * w] as i32
                } else {
                    b
                };
                let ww = if x > 1 { p.samples[at - 2] as i32 } else { a };
                let nw2 = if y > 1 && x > 0 {
                    p.samples[at - 2 * w - 1] as i32
                } else {
                    c
                };
                let ne2 = if y > 1 && x + 1 < w {
                    p.samples[at - 2 * w + 1] as i32
                } else {
                    d
                };
                let q = [0, b - a, c - a, d - a, nn - a, ww - a, nw2 - a, ne2 - a];
                let acc: i64 = self
                    .lms
                    .iter()
                    .zip(q)
                    .map(|(&l, q)| l as i64 * q as i64)
                    .sum();
                pred[7] = a + (acc >> 12) as i32;
                let mut logs = [0; 8];
                let mut lmin = 99;
                for i in 0..8 {
                    pred[i] = pred[i].clamp(0, maxv);
                    let mut e = if x > 0 {
                        res[pe0 + i * w + x - 1] as u32
                    } else {
                        0
                    };
                    if y > 0 {
                        e += res[pe1 + i * w + x] as u32;
                        if x > 0 {
                            e += (res[pe1 + i * w + x - 1] as u32) >> 1;
                        }
                        if x + 1 < w {
                            e += (res[pe1 + i * w + x + 1] as u32) >> 1;
                        }
                    }
                    logs[i] = flog(e);
                    lmin = lmin.min(logs[i]);
                }
                let (mut sw, mut sp) = (0u32, 0u32);
                for i in 0..8 {
                    let wt = 4096u32 >> (((logs[i] - lmin) * 3 / 4).min(12));
                    sw += wt;
                    sp += wt * pred[i] as u32;
                }
                let px = ((sp + (sw >> 1)) / sw) as i32;
                let spr = flog((pred.iter().max().unwrap() - pred.iter().min().unwrap()) as u32)
                    .min(31) as usize;
                let lmin = lmin.min(39) as usize;
                let mut e = if cd.encoding() {
                    (p.samples[at] as i32 - px) * sgn
                } else {
                    0
                };
                if e < -half {
                    e += range;
                } else if e >= half {
                    e -= range;
                }
                let bias = self.bias[cls * 365 + ctx];
                let act = flog((bias.a / bias.n) as u32).min(31) as usize;
                let mut nbb = if x > 0 {
                    res[e0 + x - 1].unsigned_abs()
                } else {
                    0
                };
                if y > 0 {
                    nbb += res[e1 + x].unsigned_abs();
                    if x + 1 < w {
                        nbb += res[e1 + x + 1].unsigned_abs() >> 1;
                    }
                    if x > 0 {
                        nbb += res[e1 + x - 1].unsigned_abs() >> 1;
                    }
                }
                let nbb = flog(nbb).min(31) as usize;
                let oth = if po.is_some() {
                    flog(eo.unsigned_abs()).min(30) as usize + 1
                } else {
                    0
                };
                let bi = cls * 32 + act;
                let keys = [
                    (7, bi),
                    (8, cls * 365 + ctx),
                    (9, (cls * 32 + nbb) * 32 + act),
                    (10, (oth * 32 + act) * 2 + cls),
                    (11, (cls * 40 + lmin) * 32 + spr),
                ];
                let bit = self.mix(
                    cd,
                    i32::from(e != 0),
                    &keys,
                    1,
                    cls * 256 + (lmin >> 2) * 16 + (spr >> 1),
                    (cls * 32 + act) * 16 + (nbb >> 1),
                );
                if bit != 0 {
                    let mut av = e.unsigned_abs();
                    let mut ex = if cd.encoding() {
                        (32 - av.leading_zeros()) as usize
                    } else {
                        0
                    };
                    let mut k = 1;
                    while k <= depth {
                        let ci = k - 1;
                        let keys = [
                            (12, bi * 18 + ci),
                            (13, (cls * 365 + ctx) * 18 + ci),
                            (14, ((cls * 32 + nbb) * 32 + act) * 18 + ci),
                            (15, ((oth * 32 + act) * 18 + ci) * 2 + cls),
                            (16, (lmin * 32 + spr) * 18 + ci),
                        ];
                        let g = self.mix(
                            cd,
                            i32::from(ex > k),
                            &keys,
                            2,
                            (cls * 16 + ci.min(15)) * 16 + (act >> 1),
                            ((cls * 18 + ci) * 32 + act) & 1023,
                        );
                        if g == 0 {
                            break;
                        }
                        k += 1;
                    }
                    if !cd.encoding() {
                        ex = k;
                        av = 1;
                    }
                    for ci in (0..ex - 1).rev() {
                        let top = ex - 2 - ci;
                        let pb = if cd.encoding() {
                            (av >> (ci + 1)) & 1
                        } else {
                            av & 1
                        } as usize;
                        let val = if cd.encoding() {
                            ((av >> ci) & 1) as i32
                        } else {
                            0
                        };
                        let mb = if top < 2 {
                            let base =
                                ((cls * 18 + ex) * 2 + top) * 2 + if top != 0 { pb } else { 0 };
                            self.mix(
                                cd,
                                val,
                                &[
                                    (17, base),
                                    (18, (base * 32 + act) & 4095),
                                    (19, (base * 32 + nbb) & 4095),
                                ],
                                3,
                                (cls * 18 + ex) * 2 + top,
                                ((cls * 18 + ex) * 2 + top) * 8 + (act >> 2),
                            )
                        } else {
                            self.single(cd, val, 17, 2048 + ((cls * 18 + ex) * 18 + ci) % 2048)
                        };
                        if !cd.encoding() {
                            av = av << 1 | mb as u32;
                        }
                    }
                    let sign = |v: i32| {
                        if v < 0 {
                            1
                        } else if v > 0 {
                            2
                        } else {
                            0
                        }
                    };
                    let mut sc = if x > 0 { sign(res[e0 + x - 1]) } else { 0 };
                    if y > 0 {
                        sc += 3 * sign(res[e1 + x]);
                    }
                    if po.is_some() {
                        sc += 9 * sign(eo);
                    }
                    let neg = self.single(cd, i32::from(e < 0), 20 + cls, (sc * 32 + act) & 1023);
                    if !cd.encoding() {
                        e = if neg != 0 { -(av as i32) } else { av as i32 };
                    }
                }
                if !cd.encoding() {
                    let mut v = px + e * sgn;
                    if v < 0 {
                        v += range;
                    } else if v > maxv {
                        v -= range;
                    }
                    if !(0..=maxv).contains(&v) {
                        return Err(Error::Invalid("J2K pixel residual range"));
                    }
                    p.samples[at] = v as u16;
                }
                res[e0 + x] = e;
                for i in 0..8 {
                    res[pe0 + i * w + x] = (p.samples[at] as i32 - pred[i]).abs();
                }
                let er = p.samples[at] as i32 - pred[7];
                if er != 0 {
                    for (i, &q) in q.iter().enumerate().skip(1) {
                        self.lms[i] =
                            (self.lms[i] + er.signum() * q.signum() * 16).clamp(-8192, 8192);
                    }
                }
                self.bias[cls * 365 + ctx].learn(e);
            }
        }
        Ok(())
    }
}
fn flog(v: u32) -> i32 {
    if v == 0 {
        0
    } else {
        let l = 31 - v.leading_zeros();
        (2 * l + if l > 0 { (v >> (l - 1)) & 1 } else { 0 } + 1) as i32
    }
}
fn quant(d: i32, t1: i32, t2: i32, t3: i32) -> i32 {
    if d <= -t3 {
        -4
    } else if d <= -t2 {
        -3
    } else if d <= -t1 {
        -2
    } else if d < 0 {
        -1
    } else if d == 0 {
        0
    } else if d < t1 {
        1
    } else if d < t2 {
        2
    } else if d < t3 {
        3
    } else {
        4
    }
}
pub(crate) fn code<C: BitCoder>(
    cd: &mut C,
    planes: &mut [Plane],
    hb: u32,
    max_samples: usize,
) -> Result<()> {
    let mut count = 0usize;
    for p in planes.iter() {
        let size = p
            .width
            .checked_mul(p.height)
            .ok_or(Error::Limit("J2K plane size"))?;
        count = count
            .checked_add(size)
            .filter(|&n| n <= max_samples)
            .ok_or(Error::Limit("J2K plane samples"))?;
        if p.samples.len() != size
            || !(1..=16).contains(&p.depth)
            || p.depth > 1 && p.width > 1 << 20
        {
            return Err(Error::Invalid("J2K pixel plane shape"));
        }
        if cd.encoding() && p.samples.iter().any(|&v| u32::from(v) >= 1u32 << p.depth) {
            return Err(Error::Invalid("J2K input sample range"));
        }
    }
    let mut model = Model::new(hb)?;
    for i in 0..planes.len() {
        let (prev, after) = planes.split_at_mut(i);
        let p = &mut after[0];
        if p.samples.is_empty() {
            continue;
        }
        if p.depth == 1 {
            model.bitonal(cd, p)?;
        } else {
            let po = prev
                .last()
                .filter(|v| v.width == p.width && v.height == p.height && v.depth > 1);
            model.samples(cd, p, po, usize::from(i > 0))?;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::{Decoder, Encoder};
    #[test]
    fn bitonal_and_high_depth_multicomponent() {
        for depths in [vec![1, 1], vec![2], vec![8, 8, 8], vec![12, 16]] {
            let original: Vec<_> = depths
                .iter()
                .enumerate()
                .map(|(c, &depth)| Plane {
                    width: 33,
                    height: 19,
                    depth,
                    samples: (0..33 * 19)
                        .map(|i| ((i * 31 + (i / 17) * 19 + c * 37) % (1usize << depth)) as u16)
                        .collect(),
                })
                .collect();
            let mut planes = original.clone();
            let mut enc = Encoder::new(1 << 20);
            code(&mut enc, &mut planes, 12, 1 << 20).unwrap();
            let bytes = enc.finish().unwrap();
            for p in &mut planes {
                p.samples.fill(0);
            }
            code(&mut Decoder::new(&bytes), &mut planes, 12, 1 << 20).unwrap();
            for (a, b) in planes.iter().zip(original) {
                assert_eq!(a.samples, b.samples);
            }
        }
    }
}
