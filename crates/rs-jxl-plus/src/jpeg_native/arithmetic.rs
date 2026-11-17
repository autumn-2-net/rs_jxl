// PJA MIT / IJG arithmetic JPEG translation; see ../../NOTICE-PJA.
use super::*;
const QE: [u32; 114] = [
    0x5a1d0181, 0x2586020e, 0x11140310, 0x080b0412, 0x03d80514, 0x01da0617, 0x00e50719, 0x006f081c,
    0x0036091e, 0x001a0a21, 0x000d0b23, 0x00060c09, 0x00030d0a, 0x00010d0c, 0x5a7f0f8f, 0x3f251024,
    0x2cf21126, 0x207c1227, 0x17b91328, 0x1182142a, 0x0cef152b, 0x09a1162d, 0x072f172e, 0x055c1830,
    0x04061931, 0x03031a33, 0x02401b34, 0x01b11c36, 0x01441d38, 0x00f51e39, 0x00b71f3b, 0x008a203c,
    0x0068213e, 0x004e223f, 0x003b2320, 0x002c0921, 0x5ae125a5, 0x484c2640, 0x3a0d2741, 0x2ef12843,
    0x261f2944, 0x1f332a45, 0x19a82b46, 0x15182c48, 0x11772d49, 0x0e742e4a, 0x0bfb2f4b, 0x09f8304d,
    0x0861314e, 0x0706324f, 0x05cd3330, 0x04de3432, 0x040f3532, 0x03633633, 0x02d43734, 0x025c3835,
    0x01f83936, 0x01a43a37, 0x01603b38, 0x01253c39, 0x00f63d3a, 0x00cb3e3b, 0x00ab3f3d, 0x008f203d,
    0x5b1241c1, 0x4d044250, 0x412c4351, 0x37d84452, 0x2fe84553, 0x293c4654, 0x23794756, 0x1edf4857,
    0x1aa94957, 0x174e4a48, 0x14244b48, 0x119c4c4a, 0x0f6b4d4a, 0x0d514e4b, 0x0bb64f4d, 0x0a40304d,
    0x583251d0, 0x4d1c5258, 0x438e5359, 0x3bdd545a, 0x34ee555b, 0x2eae565c, 0x299a575d, 0x25164756,
    0x557059d8, 0x4ca95a5f, 0x44d95b60, 0x3e225c61, 0x38245d63, 0x32b45e63, 0x2e17565d, 0x56a860df,
    0x4f466165, 0x47e56266, 0x41cf6367, 0x3c3d6468, 0x375e5d63, 0x52316669, 0x4c0f676a, 0x4639686b,
    0x415e6367, 0x56276ae9, 0x50e76b6c, 0x4b85676d, 0x55976d6e, 0x504f6b6f, 0x5a106fee, 0x55226d70,
    0x59eb6ff0, 0x5a1d7171,
];
trait QCoder {
    fn bit(&mut self, w: &mut Writer, st: &mut u8, value: i32) -> Result<i32>;
}
struct QDecoder<'a> {
    d: &'a [u8],
    p: usize,
    last: usize,
    c: i64,
    a: i64,
    ct: i32,
    unread: bool,
}
impl<'a> QDecoder<'a> {
    fn new(d: &'a [u8], p: usize, last: usize) -> Self {
        Self {
            d,
            p,
            last,
            c: 0,
            a: 0,
            ct: -16,
            unread: false,
        }
    }
    fn byte(&mut self) -> i64 {
        if self.p < self.last {
            let b = self.d[self.p];
            self.p += 1;
            b as i64
        } else {
            0
        }
    }
}
impl QCoder for QDecoder<'_> {
    fn bit(&mut self, _w: &mut Writer, st: &mut u8, _value: i32) -> Result<i32> {
        while self.a < 0x8000 {
            self.ct -= 1;
            if self.ct < 0 {
                let mut data = 0;
                if !self.unread {
                    data = self.byte();
                    if data == 255 {
                        loop {
                            data = self.byte();
                            if data != 255 {
                                break;
                            }
                        }
                        if data == 0 {
                            data = 255;
                        } else {
                            self.unread = true;
                            data = 0;
                        }
                    }
                }
                self.c = (self.c << 8) | data;
                self.ct += 8;
                if self.ct < 0 {
                    self.ct += 1;
                    if self.ct == 0 {
                        self.a = 0x8000;
                    }
                }
            }
            self.a <<= 1;
        }
        let mut sv = *st;
        let entry = *QE
            .get((sv & 127) as usize)
            .ok_or(Error::Invalid("JPEG QM state"))?;
        let nl = (entry & 255) as u8;
        let nm = ((entry >> 8) & 255) as u8;
        let qe = (entry >> 16) as i64;
        self.a -= qe;
        let temp = self.a << self.ct;
        if self.c >= temp {
            self.c -= temp;
            if self.a < qe {
                self.a = qe;
                *st = (sv & 128) ^ nm;
            } else {
                self.a = qe;
                *st = (sv & 128) ^ nl;
                sv ^= 128;
            }
        } else if self.a < 0x8000 {
            if self.a < qe {
                *st = (sv & 128) ^ nl;
                sv ^= 128;
            } else {
                *st = (sv & 128) ^ nm;
            }
        }
        Ok((sv >> 7) as i32)
    }
}
struct QEncoder {
    c: i64,
    a: i64,
    ct: i32,
    buffer: i32,
    sc: usize,
    zc: usize,
}
impl QEncoder {
    fn new() -> Self {
        Self {
            c: 0,
            a: 0x10000,
            ct: 11,
            buffer: -1,
            sc: 0,
            zc: 0,
        }
    }
    fn zeros(&mut self, w: &mut Writer) -> Result<()> {
        while self.zc > 0 {
            w.byte(0)?;
            self.zc -= 1;
        }
        Ok(())
    }
    fn pending(&mut self, w: &mut Writer) -> Result<()> {
        if self.buffer == 0 {
            self.zc += 1;
        } else if self.buffer >= 0 {
            self.zeros(w)?;
            w.byte(self.buffer as u8)?;
        }
        if self.sc > 0 {
            self.zeros(w)?;
            while self.sc > 0 {
                w.copy(&[255, 0])?;
                self.sc -= 1;
            }
        }
        Ok(())
    }
    fn carry(&mut self, w: &mut Writer) -> Result<()> {
        if self.buffer >= 0 {
            self.zeros(w)?;
            w.byte((self.buffer + 1) as u8)?;
            if self.buffer + 1 == 255 {
                w.byte(0)?;
            }
        }
        self.zc += self.sc;
        self.sc = 0;
        Ok(())
    }
    fn finish(&mut self, w: &mut Writer) -> Result<()> {
        let temp = (self.a - 1 + self.c) & 0xffff0000;
        self.c = if temp < self.c { temp + 0x8000 } else { temp };
        self.c <<= self.ct;
        if self.c & 0xf8000000 != 0 {
            self.carry(w)?;
        } else {
            self.pending(w)?;
        }
        if self.c & 0x7fff800 != 0 {
            self.zeros(w)?;
            let b = ((self.c >> 19) & 255) as u8;
            w.byte(b)?;
            if b == 255 {
                w.byte(0)?;
            }
            if self.c & 0x7f800 != 0 {
                let b = ((self.c >> 11) & 255) as u8;
                w.byte(b)?;
                if b == 255 {
                    w.byte(0)?;
                }
            }
        }
        Ok(())
    }
}
impl QCoder for QEncoder {
    fn bit(&mut self, w: &mut Writer, st: &mut u8, value: i32) -> Result<i32> {
        let sv = *st;
        let entry = QE[(sv & 127) as usize];
        let nl = (entry & 255) as u8;
        let nm = ((entry >> 8) & 255) as u8;
        let qe = (entry >> 16) as i64;
        self.a -= qe;
        if value != (sv >> 7) as i32 {
            if self.a >= qe {
                self.c += self.a;
                self.a = qe;
            }
            *st = (sv & 128) ^ nl;
        } else {
            if self.a >= 0x8000 {
                return Ok(value);
            }
            if self.a < qe {
                self.c += self.a;
                self.a = qe;
            }
            *st = (sv & 128) ^ nm;
        }
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                let temp = (self.c >> 19) as i32;
                if temp > 255 {
                    self.carry(w)?;
                    self.buffer = temp & 255;
                } else if temp == 255 {
                    self.sc += 1;
                } else {
                    self.pending(w)?;
                    self.buffer = temp & 255;
                }
                self.c &= 0x7ffff;
                self.ct += 8;
            }
            if self.a >= 0x8000 {
                break;
            }
        }
        Ok(value)
    }
}
struct Stats {
    dc: [[u8; 64]; 4],
    ac: [[u8; 256]; 4],
    fixed: u8,
    context: [usize; 4],
    last: [i32; 4],
}
impl Stats {
    fn new() -> Self {
        Self {
            dc: [[0; 64]; 4],
            ac: [[0; 256]; 4],
            fixed: 113,
            context: [0; 4],
            last: [0; 4],
        }
    }
    fn reset(&mut self, j: &Jpeg, s: &Scan) {
        for i in 0..s.comp.len() {
            if !j.progressive || (s.ss == 0 && s.ah == 0) {
                self.dc[s.td[i]].fill(0);
                self.last[i] = 0;
                self.context[i] = 0;
            }
            if !j.progressive || s.se > 0 {
                self.ac[s.ta[i]].fill(0);
            }
        }
        self.fixed = 113;
    }
}
fn dc_diff<Q: QCoder>(
    q: &mut Q,
    w: &mut Writer,
    st: &mut Stats,
    s: &Scan,
    i: usize,
    value: i32,
) -> Result<i32> {
    let t = s.td[i];
    let mut x = st.context[i];
    if q.bit(w, &mut st.dc[t][x], i32::from(value != 0))? == 0 {
        st.context[i] = 0;
        return Ok(0);
    }
    let sign = q.bit(w, &mut st.dc[t][x + 1], i32::from(value < 0))?;
    x += 2 + sign as usize;
    let mag = value.abs() - 1;
    let mut m = q.bit(w, &mut st.dc[t][x], i32::from(mag > 0))?;
    if m != 0 {
        x = 20;
        while q.bit(w, &mut st.dc[t][x], i32::from(mag >= 2 * m))? != 0 {
            m <<= 1;
            if m == 32768 {
                return Err(Error::Invalid("JPEG arithmetic DC magnitude"));
            }
            x += 1;
        }
    }
    st.context[i] = if m < ((1 << s.dc_l[i]) >> 1) {
        0
    } else if m > ((1 << s.dc_u[i]) >> 1) {
        12 + sign as usize * 4
    } else {
        4 + sign as usize * 4
    };
    let mut v = m;
    x += 14;
    m >>= 1;
    while m > 0 {
        if q.bit(w, &mut st.dc[t][x], i32::from(m & mag != 0))? != 0 {
            v |= m;
        }
        m >>= 1;
    }
    v += 1;
    Ok(if sign != 0 { -v } else { v })
}
fn ac_mag<Q: QCoder>(
    q: &mut Q,
    w: &mut Writer,
    st: &mut Stats,
    t: usize,
    k: usize,
    kx: usize,
    mut x: usize,
    value: i32,
) -> Result<i32> {
    let mag = value - 1;
    let mut m = q.bit(w, &mut st.ac[t][x], i32::from(mag > 0))?;
    if m != 0 && q.bit(w, &mut st.ac[t][x], i32::from(mag > 1))? != 0 {
        m <<= 1;
        x = if k <= kx { 189 } else { 217 };
        while q.bit(w, &mut st.ac[t][x], i32::from(mag >= 2 * m))? != 0 {
            m <<= 1;
            if m == 32768 {
                return Err(Error::Invalid("JPEG arithmetic AC magnitude"));
            }
            x += 1;
        }
    }
    let mut v = m;
    x += 14;
    m >>= 1;
    while m > 0 {
        if q.bit(w, &mut st.ac[t][x], i32::from(m & mag != 0))? != 0 {
            v |= m;
        }
        m >>= 1;
    }
    Ok(v + 1)
}
fn decode_block(
    q: &mut QDecoder,
    w: &mut Writer,
    st: &mut Stats,
    cf: &mut [i16],
    progressive: bool,
    s: &Scan,
    i: usize,
) -> Result<()> {
    if !progressive || (s.ss == 0 && s.ah == 0) {
        let v = dc_diff(q, w, st, s, i, 0)?;
        st.last[i] = (st.last[i] + v) as i16 as i32;
        cf[0] = coefficient(st.last[i] * (1 << s.al))?;
        if progressive {
            return Ok(());
        }
    } else if s.ss == 0 {
        if q.bit(w, &mut st.fixed, 0)? != 0 {
            cf[0] |= (1 << s.al) as i16;
        }
        return Ok(());
    }
    let (ss, se) = if progressive { (s.ss, s.se) } else { (1, 63) };
    let t = s.ta[i];
    let mut k = ss;
    if !progressive || s.ah == 0 {
        while k <= se {
            let mut x = 3 * (k - 1);
            if q.bit(w, &mut st.ac[t][x], 0)? != 0 {
                break;
            }
            while q.bit(w, &mut st.ac[t][x + 1], 0)? == 0 {
                k += 1;
                if k > se {
                    return Err(Error::Invalid("JPEG arithmetic AC run"));
                }
                x += 3;
            }
            let sign = q.bit(w, &mut st.fixed, 0)?;
            let mut v = ac_mag(q, w, st, t, k, s.ac_k[i], x + 2, 0)?;
            if sign != 0 {
                v = -v;
            }
            cf[k] = coefficient(v * (1 << s.al))?;
            k += 1;
        }
        return Ok(());
    }
    let p = 1 << s.al;
    let mut kex = se;
    while kex > 0 && cf[kex] == 0 {
        kex -= 1;
    }
    while k <= se {
        let mut x = 3 * (k - 1);
        if k > kex && q.bit(w, &mut st.ac[t][x], 0)? != 0 {
            break;
        }
        loop {
            let c = cf[k] as i32;
            if c != 0 {
                if q.bit(w, &mut st.ac[t][x + 2], 0)? != 0 {
                    cf[k] = coefficient(if c < 0 { c - p } else { c + p })?;
                }
                break;
            }
            if q.bit(w, &mut st.ac[t][x + 1], 0)? != 0 {
                cf[k] = if q.bit(w, &mut st.fixed, 0)? != 0 {
                    -p
                } else {
                    p
                } as i16;
                break;
            }
            x += 3;
            k += 1;
            if k > se {
                return Err(Error::Invalid("JPEG arithmetic refinement run"));
            }
        }
        k += 1;
    }
    Ok(())
}
fn encode_block(
    q: &mut QEncoder,
    w: &mut Writer,
    st: &mut Stats,
    cf: &[i16],
    progressive: bool,
    s: &Scan,
    i: usize,
) -> Result<()> {
    if !progressive || (s.ss == 0 && s.ah == 0) {
        let v = (cf[0] as i32) >> s.al;
        dc_diff(q, w, st, s, i, v - st.last[i])?;
        st.last[i] = v;
        if progressive {
            return Ok(());
        }
    } else if s.ss == 0 {
        q.bit(w, &mut st.fixed, ((cf[0] as i32) >> s.al) & 1)?;
        return Ok(());
    }
    let (ss, se, al) = if progressive {
        (s.ss, s.se, s.al)
    } else {
        (1, 63, 0)
    };
    let t = s.ta[i];
    let mag = |k: usize, al: usize| (cf[k] as i32).abs() >> al;
    let mut ke = se;
    while ke > 0 && mag(ke, al) == 0 {
        ke -= 1;
    }
    let mut k = ss;
    if !progressive || s.ah == 0 {
        while k <= ke {
            let mut x = 3 * (k - 1);
            q.bit(w, &mut st.ac[t][x], 0)?;
            while mag(k, al) == 0 {
                q.bit(w, &mut st.ac[t][x + 1], 0)?;
                x += 3;
                k += 1;
            }
            q.bit(w, &mut st.ac[t][x + 1], 1)?;
            q.bit(w, &mut st.fixed, i32::from(cf[k] < 0))?;
            ac_mag(q, w, st, t, k, s.ac_k[i], x + 2, mag(k, al))?;
            k += 1;
        }
    } else {
        let mut kex = ke;
        while kex > 0 && mag(kex, s.ah) == 0 {
            kex -= 1;
        }
        while k <= ke {
            let mut x = 3 * (k - 1);
            if k > kex {
                q.bit(w, &mut st.ac[t][x], 0)?;
            }
            while mag(k, al) == 0 {
                q.bit(w, &mut st.ac[t][x + 1], 0)?;
                x += 3;
                k += 1;
            }
            let v = mag(k, al);
            if v >> 1 != 0 {
                q.bit(w, &mut st.ac[t][x + 2], v & 1)?;
            } else {
                q.bit(w, &mut st.ac[t][x + 1], 1)?;
                q.bit(w, &mut st.fixed, i32::from(cf[k] < 0))?;
            }
            k += 1;
        }
    }
    if k <= se {
        q.bit(w, &mut st.ac[t][3 * (k - 1)], 1)?;
    }
    Ok(())
}
pub(super) fn decode(data: &[u8], first: usize, j: &mut Jpeg, s: &mut Scan) -> Result<usize> {
    s.padbit = 1;
    let mut st = Stats::new();
    let (_, total) = units(j, s);
    let (mut p, mut rst, mut u0) = (first, 0, 0);
    let mut sink = Writer::new(0);
    while u0 < total {
        let stop = segment_end(data, p);
        let mut q = QDecoder::new(data, p, stop);
        st.reset(j, s);
        let end = if s.ri == 0 {
            total
        } else {
            total.min(u0 + s.ri)
        };
        for u in u0..end {
            for (i, c, b) in blocks(j, s, u) {
                decode_block(
                    &mut q,
                    &mut sink,
                    &mut st,
                    &mut j.components[c].coef[b * 64..b * 64 + 64],
                    j.progressive,
                    s,
                    i,
                )?;
            }
        }
        if end < total {
            if data.get(stop..stop + 2) != Some(&[255, 208 + (rst & 7) as u8]) {
                return Err(fail("BadRst", stop));
            }
            p = stop + 2;
            rst += 1;
        } else {
            p = stop;
        }
        u0 = end;
    }
    if p < data.len() && p + 1 >= data.len() {
        return Err(fail("DataAfterScan", p));
    }
    Ok(p)
}
pub(super) fn encode(w: &mut Writer, j: &Jpeg, s: &Scan) -> Result<()> {
    let mut st = Stats::new();
    let mut q = QEncoder::new();
    st.reset(j, s);
    let (_, total) = units(j, s);
    let mut rst = 0;
    for u in 0..total {
        if s.ri > 0 && u > 0 && u % s.ri == 0 {
            q.finish(w)?;
            w.restart(rst)?;
            rst += 1;
            q = QEncoder::new();
            st.reset(j, s);
        }
        for (i, c, b) in blocks(j, s, u) {
            encode_block(
                &mut q,
                w,
                &mut st,
                &j.components[c].coef[b * 64..b * 64 + 64],
                j.progressive,
                s,
                i,
            )?;
        }
    }
    q.finish(w)
}
