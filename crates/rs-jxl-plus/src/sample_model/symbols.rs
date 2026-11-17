//! TJlsSymModel and CodeJlsSymbolStream from PJA pas/pjamodel.pas, MIT,
//! revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA/NOTICE-PJA.
use super::*;
struct SymModel<'a, C: BitCoder> {
    cd: &'a mut C,
    m: Bank,
    bias_sum: [i32; 2048],
    bias_n: [i32; 2048],
    vals: [Vec<i32>; 4],
    errs: [Vec<i32>; 4],
    lines: [[Vec<i32>; 3]; 4],
    pend_tex: [i32; 4],
    pend_p: [i32; 4],
    pend_p2: [i32; 4],
    near: i32,
    maxval: i32,
    scale: i32,
    wc: [usize; 4],
    same: [bool; 4],
}
impl<'a, C: BitCoder> SymModel<'a, C> {
    fn new(cd: &'a mut C) -> Self {
        Self {
            cd,
            m: Bank::new(
                &[
                    3 * 17 * 19,
                    367 * 19,
                    3 * 24 * 19,
                    20 * 17 * 19,
                    16 * 17 * 19,
                    24 * 17 * 19,
                    2048 * 19,
                    20 * 16 * 17,
                    20 * 16 * 19,
                    3 * 15 * 20 * 3,
                    2 * 32 * 8,
                    32 * 80,
                ],
                &[(8, 3 * 17 * 19)],
            ),
            bias_sum: [0; 2048],
            bias_n: [1; 2048],
            vals: std::array::from_fn(|_| Vec::new()),
            errs: std::array::from_fn(|_| Vec::new()),
            lines: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            pend_tex: [-1; 4],
            pend_p: [0; 4],
            pend_p2: [0; 4],
            near: 0,
            maxval: 0,
            scale: 1,
            wc: [0; 4],
            same: [false; 4],
        }
    }
}
impl<C: BitCoder> jls::Symbols for SymModel<'_, C> {
    fn begin_scan(&mut self, s: &jls::Scan, j: &Image) -> Result<()> {
        self.near = s.near;
        self.maxval = s.maxval;
        self.scale = 1.max((s.maxval + 1) / 256);
        for c in 0..4 {
            let comp = s.components.get(c).map(|&k| &j.components[k]);
            let width = comp.map_or(0, |c| c.width);
            self.wc[c] = width;
            self.same[c] = comp.is_some_and(|c| {
                c.width == j.components[s.components[0]].width
                    && c.height == j.components[s.components[0]].height
            });
            self.vals[c] = vec![0; if comp.is_some() { width + 2 } else { 0 }];
            self.errs[c] = self.vals[c].clone();
            self.lines[c] =
                std::array::from_fn(|_| vec![0; if comp.is_some() { width + 4 } else { 0 }]);
            self.pend_tex[c] = -1;
        }
        Ok(())
    }
    fn sample(&mut self, c: usize, i: usize, x: i32) -> Result<()> {
        if c >= 4 || i >= self.wc[c] {
            return Err(Error::Invalid("JPEG-LS symbol sample position"));
        }
        if i == 0 {
            self.lines[c].rotate_right(1);
        }
        self.lines[c][0][i + 2] = x;
        let k = self.pend_tex[c];
        if k >= 0 {
            let k = k as usize;
            self.bias_sum[k] += x - self.pend_p2[c];
            self.bias_n[k] += 1;
            if self.bias_n[k] >= 256 {
                self.bias_sum[k] /= 2;
                self.bias_n[k] /= 2;
            }
            self.errs[c][i + 1] = (x - self.pend_p[c]).abs();
            self.pend_tex[c] = -1;
        } else {
            self.errs[c][i + 1] = 0;
        }
        Ok(())
    }
    fn mapped(
        &mut self,
        m: i32,
        c: usize,
        i: usize,
        k: i32,
        ctx: i32,
        px: i32,
        sgn: i32,
        inv: bool,
    ) -> Result<i32> {
        if c >= 4 || i >= self.wc[c] || !(0..367).contains(&ctx) || k < 0 {
            return Err(Error::Invalid("JPEG-LS symbol position/context"));
        }
        let kind = if ctx >= 365 {
            1 + ctx as usize - 365
        } else {
            0
        };
        let kk = (k as usize).min(16);
        let row = &self.vals[c];
        let nb = finelog((row[i] + row[i + 1] + row[i + 2] / 2) as i64).min(23) as usize;
        let first = if c > 0 && self.same[c] {
            finelog(self.vals[0][i + 1] as i64).min(18) as usize
        } else {
            19
        };
        let q = 2 * self.near + 1;
        let l0 = &self.lines[c][0];
        let l1 = &self.lines[c][1];
        let l2 = &self.lines[c][2];
        let a = l0[i + 1];
        let aa = l0[i];
        let n = l1[i + 2];
        let nw = l1[i + 1];
        let ne = l1[i + 3];
        let nn = l2[i + 2];
        let nne = l2[i + 3];
        let dh = (a - aa).abs() + (n - nw).abs() + (n - ne).abs();
        let dv = (a - nw).abs() + (n - nn).abs() + (ne - nne).abs();
        let p2 = if dv - dh > 80 * self.scale {
            a
        } else if dh - dv > 80 * self.scale {
            n
        } else {
            let mut p = (a + n) / 2 + (ne - nw) / 4;
            if dv - dh > 32 * self.scale {
                p = (p + a) / 2;
            } else if dv - dh > 8 * self.scale {
                p = (3 * p + a) / 4;
            } else if dh - dv > 32 * self.scale {
                p = (p + n) / 2;
            } else if dh - dv > 8 * self.scale {
                p = (3 * p + n) / 4;
            }
            p
        };
        let delta = (dh + dv + 2 * self.errs[c][i]) / q;
        let tex = (n < p2) as usize
            | (((a < p2) as usize) << 1)
            | (((nw < p2) as usize) << 2)
            | (((ne < p2) as usize) << 3)
            | (((aa < p2) as usize) << 4)
            | (((nn < p2) as usize) << 5)
            | (((2 * n - nn < p2) as usize) << 6)
            | (((2 * a - aa < p2) as usize) << 7);
        let tex = (c.min(1) * 256 + tex) * 4 + (finelog(delta as i64) / 3).min(3) as usize;
        let p3 = (p2 + divroundr(self.bias_sum[tex] as i64, self.bias_n[tex] as i64, 0.0) as i32)
            .clamp(0, self.maxval);
        if kind == 0 {
            self.pend_tex[c] = tex as i32;
            self.pend_p[c] = p3;
            self.pend_p2[c] = p2;
        }
        let mut ex = divroundr((p3 - px) as i64 * sgn as i64, q as i64, 0.0) as i32;
        if inv {
            ex = -ex - 1;
        }
        let me = if kind == 0 {
            if ex >= 0 { 2 * ex } else { -2 * ex - 1 }
        } else {
            2 * ex.abs()
        };
        let mb = finelog(me as i64).min(15) as usize;
        let sl = finelog(ex as i64).min(7) * if ex < 0 { -1 } else { 1 };
        let eb = (kind as i32 * 15 + sl + 7) as usize;
        let act = finelog(delta as i64).min(23) as usize;
        let e = if self.cd.encoding() {
            if m < 0 {
                return Err(Error::Invalid("negative JPEG-LS mapped symbol"));
            }
            let e = bitlen64(m as i64);
            if e >= 19 {
                return Err(Error::Invalid("JPEG-LS mapped symbol magnitude"));
            }
            e
        } else {
            0
        };
        let mut t = 0usize;
        while t < 18 {
            let sel = (kind * 17 + kk) * 19 + t;
            let bit = self.m.mix(
                self.cd,
                0,
                sel,
                &[
                    (0, sel),
                    (1, ctx as usize * 19 + t),
                    (2, (kind * 24 + nb) * 19 + t),
                    (3, (first * 17 + kk) * 19 + t),
                    (4, (mb * 17 + kk) * 19 + t),
                    (5, (act * 17 + kk) * 19 + t),
                    (6, tex * 19 + t),
                ],
                i32::from(e > t as i32),
            );
            if bit == 0 {
                break;
            }
            t += 1;
        }
        let mut v = if t == 0 { 0 } else { 1i32 << (t - 1) };
        for b in (0..t.saturating_sub(1)).rev() {
            let bit = if b == 0 {
                self.m
                    .one(self.cd, 9, (eb * 20 + t) * 3 + (t - 1).min(2), m & 1)
            } else if b == t - 2 {
                self.m
                    .one(self.cd, 7, (t * 16 + mb) * 17 + kk, (m >> b) & 1)
            } else {
                self.m.one(self.cd, 8, (t * 16 + mb) * 19 + b, (m >> b) & 1)
            };
            v |= bit << b;
        }
        self.vals[c][i + 1] = v;
        Ok(v)
    }
    fn run_bit(&mut self, bit: i32, c: usize, i: usize, index: usize) -> Result<i32> {
        if c >= 4 || index >= 32 {
            return Err(Error::Invalid("JPEG-LS symbol run context"));
        }
        let above = if i < self.wc[c] {
            finelog(self.vals[c][i + 1] as i64).min(7) as usize
        } else {
            0
        };
        Ok(self
            .m
            .one(self.cd, 10, (c.min(1) * 32 + index) * 8 + above, bit))
    }
    fn run_rest(&mut self, v: i32, bits: i32, _c: usize, index: usize) -> Result<i32> {
        if index >= 32 || !(0..=15).contains(&bits) {
            return Err(Error::Invalid("JPEG-LS symbol run rest"));
        }
        let mut result = 0;
        let mut node = 1usize;
        for b in (0..bits).rev() {
            let idx = if node < 64 { node } else { 64 + b as usize };
            let bit = self.m.one(self.cd, 11, index * 80 + idx, (v >> b) & 1);
            result = (result << 1) | bit;
            node = if node < 64 {
                2 * node + bit as usize
            } else {
                64
            };
        }
        Ok(result)
    }
    fn run_end(&mut self, c: usize, i: usize, len: usize) -> Result<()> {
        if c >= 4 || i > self.wc[c] || len > self.wc[c] - i {
            return Err(Error::Invalid("JPEG-LS symbol run extent"));
        }
        self.vals[c][i + 1..i + len + 1].fill(0);
        Ok(())
    }
}
pub(crate) fn code_symbol_stream<C: BitCoder>(
    cd: &mut C,
    j: &mut Image,
    limits: &Limits,
) -> Result<()> {
    jls::validate(j, limits)?;
    let encoding = cd.encoding();
    let mut model = SymModel::new(cd);
    jls::code_symbols(j, &mut model, encoding, limits)
}
