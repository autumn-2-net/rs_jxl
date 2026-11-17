//! LOCO-I scan state translated from PJA pas/pjajls.pas (MIT), pinned
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; CharLS notice in NOTICE-PJA.
use super::{
    entropy::{Io, RUN_J},
    *,
};
#[derive(Clone, Copy, Default)]
struct Regular {
    a: i32,
    b: i32,
    c: i32,
    n: i32,
}
#[derive(Clone, Copy, Default)]
struct Run {
    a: i32,
    n: i32,
    nn: i32,
    typ: i32,
}
struct Coder {
    near: i32,
    maxval: i32,
    range: i32,
    qbpp: i32,
    limit: i32,
    reset: i32,
    t1: i32,
    t2: i32,
    t3: i32,
    ctx: [Regular; 365],
    run: [Run; 2],
    index: usize,
}
impl Coder {
    fn new(s: &Scan) -> Self {
        let range = (s.maxval + 2 * s.near) / (2 * s.near + 1) + 1;
        let log = |n: i32| {
            let mut k = 0;
            while n > (1 << k) {
                k += 1;
            }
            k
        };
        let bpp = log(s.maxval);
        let mut c = Self {
            near: s.near,
            maxval: s.maxval,
            range,
            qbpp: log(range),
            limit: 2 * (bpp + 8.max(bpp)),
            reset: s.reset,
            t1: s.t1,
            t2: s.t2,
            t3: s.t3,
            ctx: [Regular::default(); 365],
            run: [Run::default(); 2],
            index: 0,
        };
        c.reset_state();
        c
    }
    fn reset_state(&mut self) {
        let a = 2.max((self.range + 32) / 64);
        self.ctx.fill(Regular {
            a,
            b: 0,
            c: 0,
            n: 1,
        });
        self.run = [
            Run {
                a,
                n: 1,
                nn: 0,
                typ: 0,
            },
            Run {
                a,
                n: 1,
                nn: 0,
                typ: 1,
            },
        ];
        self.index = 0;
    }
    fn quant(&self, d: i32) -> i32 {
        if d <= -self.t3 {
            -4
        } else if d <= -self.t2 {
            -3
        } else if d <= -self.t1 {
            -2
        } else if d < -self.near {
            -1
        } else if d <= self.near {
            0
        } else if d < self.t1 {
            1
        } else if d < self.t2 {
            2
        } else if d < self.t3 {
            3
        } else {
            4
        }
    }
    fn context(&self, d: i32, b: i32, c: i32, a: i32) -> i32 {
        (self.quant(d - b) * 9 + self.quant(b - c)) * 9 + self.quant(c - a)
    }
    fn reconstruct(&self, px: i32, e: i32) -> i32 {
        let q = 2 * self.near + 1;
        let mut v = px + e * q;
        if v < -self.near {
            v += self.range * q;
        } else if v > self.maxval + self.near {
            v -= self.range * q;
        }
        v.clamp(0, self.maxval)
    }
    fn error(&self, e: i32) -> i32 {
        let mut e = if e > 0 {
            (e + self.near) / (2 * self.near + 1)
        } else {
            -((self.near - e) / (2 * self.near + 1))
        };
        if e < 0 {
            e += self.range;
        }
        if e >= (self.range + 1) / 2 {
            e -= self.range;
        }
        e
    }
    fn regular(
        &mut self,
        io: &mut Io<'_>,
        qs: i32,
        x: i32,
        pred: i32,
        c: usize,
        i: usize,
    ) -> Result<i32> {
        let sign = if qs < 0 { -1 } else { 1 };
        let q = qs.unsigned_abs() as usize;
        if q >= 365 {
            return Err(Error::Invalid("JPEG-LS gradient context"));
        }
        let mut ct = self.ctx[q];
        let mut k = 0;
        while k < 16 && ((ct.n as i64) << k) < ct.a as i64 {
            k += 1;
        }
        if k >= 16 {
            return Err(Error::Invalid("JPEG-LS Golomb parameter"));
        }
        let px = (pred + sign * ct.c).clamp(0, self.maxval);
        let inv = k == 0 && self.near == 0 && 2 * ct.b + ct.n - 1 < 0;
        let mut err = if io.encoding() {
            self.error((x - px) * sign)
        } else {
            0
        };
        let mut m = if inv { err ^ -1 } else { err };
        m = if m >= 0 { 2 * m } else { -2 * m - 1 };
        m = io.mapped(m, c, i, k, q as i32, px, sign, inv, self.limit, self.qbpp)?;
        if !io.encoding() {
            err = if m & 1 != 0 { -((m + 1) >> 1) } else { m >> 1 };
            if err.abs() > 65535 {
                return Err(Error::Invalid("JPEG-LS error magnitude"));
            }
            if inv {
                err ^= -1;
            }
        }
        ct.a += err.abs();
        ct.b += err * (2 * self.near + 1);
        if ct.a >= 65536 * 256 || ct.b.abs() >= 65536 * 256 {
            return Err(Error::Invalid("JPEG-LS context overflow"));
        }
        if ct.n == self.reset {
            ct.a >>= 1;
            ct.b >>= 1;
            ct.n >>= 1;
        }
        ct.n += 1;
        if ct.b + ct.n <= 0 {
            ct.b += ct.n;
            if ct.b <= -ct.n {
                ct.b = -ct.n + 1;
            }
            if ct.c > -128 {
                ct.c -= 1;
            }
        } else if ct.b > 0 {
            ct.b -= ct.n;
            if ct.b > 0 {
                ct.b = 0;
            }
            if ct.c < 127 {
                ct.c += 1;
            }
        }
        self.ctx[q] = ct;
        Ok(self.reconstruct(px, err * sign))
    }
    #[allow(clippy::too_many_arguments)]
    fn interruption(
        &mut self,
        io: &mut Io<'_>,
        typ: usize,
        mut err: i32,
        px: i32,
        sgn: i32,
        c: usize,
        i: usize,
    ) -> Result<i32> {
        let mut r = self.run[typ];
        let temp = r.a + (r.n >> 1) * r.typ;
        let mut nt = r.n as i64;
        let mut k = 0;
        while nt < temp as i64 {
            nt <<= 1;
            k += 1;
            if k > 16 {
                return Err(Error::Invalid("JPEG-LS run parameter"));
            }
        }
        let m;
        if io.encoding() {
            let map = (k == 0 && err > 0 && 2 * r.nn < r.n)
                || (err < 0 && 2 * r.nn >= r.n)
                || (err < 0 && k != 0);
            let v = 2 * err.abs() - r.typ - map as i32;
            if v < 0 {
                return Err(Error::Invalid("JPEG-LS run interruption"));
            }
            m = io.mapped(
                v,
                c,
                i,
                k,
                365 + r.typ,
                px,
                sgn,
                false,
                self.limit - RUN_J[self.index] - 1,
                self.qbpp,
            )?;
        } else {
            m = io.mapped(
                0,
                c,
                i,
                k,
                365 + r.typ,
                px,
                sgn,
                false,
                self.limit - RUN_J[self.index] - 1,
                self.qbpp,
            )?;
            let t = m + r.typ;
            let map = t & 1 != 0;
            let a = (t + map as i32) / 2;
            err = if (k != 0 || 2 * r.nn >= r.n) == map {
                -a
            } else {
                a
            };
            if err.abs() > 65535 {
                return Err(Error::Invalid("JPEG-LS interruption error"));
            }
        }
        if err < 0 {
            r.nn += 1;
        }
        r.a += (m + 1 - r.typ) >> 1;
        if r.n == self.reset {
            r.a >>= 1;
            r.n >>= 1;
            r.nn >>= 1;
        }
        r.n += 1;
        self.run[typ] = r;
        Ok(err)
    }
}
pub(super) struct State {
    cd: Coder,
    width: usize,
    cw: [usize; 4],
    prev: [Vec<i32>; 4],
    cur: [Vec<i32>; 4],
    run_idx: [usize; 4],
    pub runs: Vec<i32>,
    pub run_pos: usize,
}
impl State {
    pub fn new(j: &Image, s: &Scan) -> Self {
        let cw = std::array::from_fn(|c| s.components.get(c).map_or(0, |&k| j.components[k].width));
        Self {
            cd: Coder::new(s),
            width: j.width,
            cw,
            prev: std::array::from_fn(|c| vec![0; cw[c] + 2]),
            cur: std::array::from_fn(|c| vec![0; cw[c] + 2]),
            run_idx: [0; 4],
            runs: s.runs.clone(),
            run_pos: 0,
        }
    }
    pub fn reset(&mut self) {
        self.cd.reset_state();
        for c in 0..4 {
            self.prev[c].fill(0);
            self.cur[c].fill(0);
            self.run_idx[c] = 0;
        }
    }
    fn swap(&mut self, c: usize) {
        std::mem::swap(&mut self.prev[c], &mut self.cur[c]);
    }
    fn run(
        &mut self,
        io: &mut Io<'_>,
        c: usize,
        i: usize,
        left: usize,
        proposed: usize,
        limits: &Limits,
    ) -> Result<usize> {
        let mut run = 0usize;
        if io.encoding() {
            run = if self.cd.near > 0 {
                let v = *self
                    .runs
                    .get(self.run_pos)
                    .ok_or(Error::Invalid("missing JPEG-LS recorded run"))?;
                self.run_pos += 1;
                if v < 0 {
                    return Err(Error::Invalid("negative JPEG-LS run"));
                }
                v as usize
            } else {
                proposed
            };
            if run > left {
                return Err(Error::Invalid("JPEG-LS run exceeds line"));
            }
            let mut cnt = run;
            while cnt >= 1usize << RUN_J[self.cd.index] {
                io.run_bit(1, c, i + run - cnt, self.cd.index)?;
                cnt -= 1usize << RUN_J[self.cd.index];
                self.cd.index = (self.cd.index + 1).min(31);
            }
            if run == left {
                if cnt != 0 {
                    io.run_bit(1, c, i + run - cnt, self.cd.index)?;
                }
            } else {
                io.run_bit(0, c, i + run - cnt, self.cd.index)?;
                if RUN_J[self.cd.index] > 0 {
                    io.run_rest(cnt as i32, RUN_J[self.cd.index], c, self.cd.index)?;
                }
            }
        } else {
            loop {
                if io.run_bit(0, c, i + run, self.cd.index)? != 1 {
                    break;
                }
                let cnt = (1usize << RUN_J[self.cd.index]).min(left - run);
                run += cnt;
                if cnt == 1usize << RUN_J[self.cd.index] {
                    self.cd.index = (self.cd.index + 1).min(31);
                }
                if run == left {
                    break;
                }
            }
            if run != left && RUN_J[self.cd.index] > 0 {
                let v = io.run_rest(0, RUN_J[self.cd.index], c, self.cd.index)?;
                if v < 0 {
                    return Err(Error::Invalid("negative JPEG-LS run rest"));
                }
                run = run
                    .checked_add(v as usize)
                    .ok_or(Error::Invalid("JPEG-LS run overflow"))?;
            }
            if run > left {
                return Err(Error::Invalid("JPEG-LS run exceeds line"));
            }
            if self.cd.near > 0 {
                if self.runs.len() >= limits.max_runs {
                    return Err(Error::Limit("JPEG-LS recorded runs"));
                }
                self.runs.push(run as i32);
                self.run_pos += 1;
            }
        }
        Ok(run)
    }
    fn line(
        &mut self,
        j: &mut Image,
        s: &Scan,
        io: &mut Io<'_>,
        c: usize,
        row: usize,
        limits: &Limits,
    ) -> Result<()> {
        let w = self.cw[c];
        let ci = s.components[c];
        let base = row * w;
        self.prev[c][w + 1] = self.prev[c][w];
        self.cur[c][0] = self.prev[c][1];
        let mut i = 0;
        while i < w {
            let a = self.cur[c][i];
            let b = self.prev[c][i + 1];
            let cc = self.prev[c][i];
            let d = self.prev[c][i + 2];
            let qs = self.cd.context(d, b, cc, a);
            if qs != 0 {
                let x = if io.encoding() {
                    j.components[ci].samples[base + i] as i32
                } else {
                    0
                };
                let x = self.cd.regular(io, qs, x, med(a, b, cc), c, i)?;
                self.cur[c][i + 1] = x;
                io.sample(c, i, x)?;
                i += 1;
                continue;
            }
            let left = w - i;
            let mut proposed = 0;
            if io.encoding() && self.cd.near == 0 {
                while proposed < left && j.components[ci].samples[base + i + proposed] as i32 == a {
                    proposed += 1;
                }
            }
            let run = self.run(io, c, i, left, proposed, limits)?;
            for x in 0..run {
                self.cur[c][i + 1 + x] = a;
                io.sample(c, i + x, a)?;
            }
            io.run_end(c, i, run)?;
            i += run;
            if i == w {
                break;
            }
            let rb = self.prev[c][i + 1];
            let x = if io.encoding() {
                j.components[ci].samples[base + i] as i32
            } else {
                0
            };
            let result = if (a - rb).abs() <= self.cd.near {
                let err = if io.encoding() {
                    self.cd.error(x - a)
                } else {
                    0
                };
                let err = self.cd.interruption(io, 1, err, a, 1, c, i)?;
                self.cd.reconstruct(a, err)
            } else {
                let sg = if rb - a < 0 { -1 } else { 1 };
                let err = if io.encoding() {
                    self.cd.error((x - rb) * sg)
                } else {
                    0
                };
                let err = self.cd.interruption(io, 0, err, rb, sg, c, i)?;
                self.cd.reconstruct(rb, err * sg)
            };
            self.cur[c][i + 1] = result;
            io.sample(c, i, result)?;
            self.cd.index = self.cd.index.saturating_sub(1);
            i += 1;
        }
        if !io.encoding() {
            for i in 0..w {
                j.components[ci].samples[base + i] = self.cur[c][i + 1] as u16;
            }
        }
        Ok(())
    }
    fn sample_line(
        &mut self,
        j: &mut Image,
        s: &Scan,
        io: &mut Io<'_>,
        row: usize,
        limits: &Limits,
    ) -> Result<()> {
        let n = s.components.len();
        let w = self.width;
        let base = row * w;
        for c in 0..n {
            self.prev[c][w + 1] = self.prev[c][w];
            self.cur[c][0] = self.prev[c][1];
        }
        let mut i = 0;
        while i < w {
            let a: [i32; 4] = std::array::from_fn(|c| if c < n { self.cur[c][i] } else { 0 });
            let b: [i32; 4] = std::array::from_fn(|c| if c < n { self.prev[c][i + 1] } else { 0 });
            let cc: [i32; 4] = std::array::from_fn(|c| if c < n { self.prev[c][i] } else { 0 });
            let d: [i32; 4] = std::array::from_fn(|c| if c < n { self.prev[c][i + 2] } else { 0 });
            let qs: [i32; 4] = std::array::from_fn(|c| self.cd.context(d[c], b[c], cc[c], a[c]));
            if qs[..n].iter().any(|&q| q != 0) {
                for c in 0..n {
                    let ci = s.components[c];
                    let x = if io.encoding() {
                        j.components[ci].samples[base + i] as i32
                    } else {
                        0
                    };
                    let x = self
                        .cd
                        .regular(io, qs[c], x, med(a[c], b[c], cc[c]), c, i)?;
                    self.cur[c][i + 1] = x;
                    io.sample(c, i, x)?;
                }
                i += 1;
                continue;
            }
            let left = w - i;
            let mut proposed = 0;
            if io.encoding() && self.cd.near == 0 {
                while proposed < left
                    && (0..n).all(|c| {
                        j.components[s.components[c]].samples[base + i + proposed] as i32 == a[c]
                    })
                {
                    proposed += 1;
                }
            }
            let run = self.run(io, 0, i, left, proposed, limits)?;
            for x in 0..run {
                for (c, &a) in a.iter().enumerate().take(n) {
                    self.cur[c][i + 1 + x] = a;
                    io.sample(c, i + x, a)?;
                }
            }
            for c in 0..n {
                io.run_end(c, i, run)?;
            }
            i += run;
            if i == w {
                break;
            }
            for (c, &a) in a.iter().enumerate().take(n) {
                let rb = self.prev[c][i + 1];
                let sg = if rb - a < 0 { -1 } else { 1 };
                let ci = s.components[c];
                let err = if io.encoding() {
                    self.cd
                        .error(sg * (j.components[ci].samples[base + i] as i32 - rb))
                } else {
                    0
                };
                let err = self.cd.interruption(io, 0, err, rb, sg, c, i)?;
                let x = self.cd.reconstruct(rb, err * sg);
                self.cur[c][i + 1] = x;
                io.sample(c, i, x)?;
            }
            self.cd.index = self.cd.index.saturating_sub(1);
            i += 1;
        }
        if !io.encoding() {
            for c in 0..n {
                for i in 0..w {
                    j.components[s.components[c]].samples[base + i] = self.cur[c][i + 1] as u16;
                }
            }
        }
        Ok(())
    }
    pub fn lines(
        &mut self,
        j: &mut Image,
        s: &Scan,
        io: &mut Io<'_>,
        y0: usize,
        y1: usize,
        limits: &Limits,
    ) -> Result<()> {
        let n = s.components.len();
        if s.ilv == 1 && grouped(j, s) {
            for y in y0..y1 {
                for c in 0..n {
                    let comp = &j.components[s.components[c]];
                    let v = comp.v;
                    let height = comp.height;
                    for k in 0..v {
                        let line = y * v + k;
                        if line >= height {
                            break;
                        }
                        self.swap(c);
                        self.cd.index = self.run_idx[c];
                        self.line(j, s, io, c, line, limits)?;
                        self.run_idx[c] = self.cd.index;
                    }
                }
            }
            return Ok(());
        }
        for y in y0..y1 {
            for c in 0..n {
                self.swap(c);
            }
            if s.ilv == 2 {
                self.sample_line(j, s, io, y, limits)?;
            } else {
                for c in 0..n {
                    self.cd.index = self.run_idx[c];
                    self.line(j, s, io, c, y, limits)?;
                    self.run_idx[c] = self.cd.index;
                }
            }
        }
        Ok(())
    }
}
