// Scalar translation of pjajpeg.pas BuildHuff and sequential/progressive scans.
use super::*;
#[derive(Clone, Debug)]
pub(crate) struct Huff {
    pub counts: [u8; 16],
    pub values: Vec<u8>,
    pub code: [u16; 256],
    pub len: [u8; 256],
    min: [i32; 17],
    max: [i32; 17],
    ptr: [usize; 17],
}
impl Default for Huff {
    fn default() -> Self {
        Self {
            counts: [0; 16],
            values: Vec::new(),
            code: [0; 256],
            len: [0; 256],
            min: [0; 17],
            max: [-1; 17],
            ptr: [0; 17],
        }
    }
}
impl Huff {
    pub(super) fn build(&mut self) -> Result<()> {
        let (mut cd, mut k) = (0i32, 0usize);
        self.code.fill(0);
        self.len.fill(0);
        for l in 1..=16 {
            self.ptr[l] = k;
            self.min[l] = cd;
            self.max[l] = -1;
            for _ in 0..self.counts[l - 1] {
                if cd >= (1 << l) || k >= self.values.len() {
                    return Err(Error::Invalid("JPEG bad Huffman table"));
                }
                let v = self.values[k] as usize;
                if self.len[v] == 0 {
                    self.code[v] = cd as u16;
                    self.len[v] = l as u8;
                }
                cd += 1;
                k += 1;
                self.max[l] = cd - 1;
            }
            cd <<= 1;
        }
        if k != self.values.len() {
            return Err(Error::Invalid("JPEG Huffman value count"));
        }
        Ok(())
    }
    pub(super) fn decode(&self, r: &mut Reader) -> Result<i32> {
        let mut cd = 0;
        for l in 1..=16 {
            cd = (cd << 1) | r.bit()?;
            if cd <= self.max[l] {
                let i = self.ptr[l] + (cd - self.min[l]) as usize;
                return self
                    .values
                    .get(i)
                    .copied()
                    .map(i32::from)
                    .ok_or_else(|| fail("BadCode", r.p));
            }
        }
        Err(fail("BadCode", r.p))
    }
    pub(super) fn emit(&self, w: &mut Writer, sym: usize) -> Result<()> {
        if sym > 255 || self.len[sym] == 0 {
            return Err(Error::Invalid("JPEG missing encoding symbol"));
        }
        w.bits(self.code[sym] as i32, self.len[sym] as usize)
    }
}
#[derive(Default)]
struct Prog {
    eob: usize,
    bits: Vec<i32>,
    run_be: usize,
    run_len: usize,
    prev_ended: bool,
    prev_be: usize,
    prev_len: usize,
    be_low: usize,
    be_high: i32,
}
fn flush(w: &mut Writer, s: &Scan, st: &mut Prog) -> Result<()> {
    if st.eob == 0 {
        return Ok(());
    }
    let n = category(st.eob as i32) - 1;
    if n > 14 {
        return Err(Error::Invalid("JPEG EOB run"));
    }
    s.ac[0].emit(w, n << 4)?;
    w.bits(st.eob as i32, n)?;
    st.eob = 0;
    for b in st.bits.drain(..) {
        w.bits(b, 1)?;
    }
    Ok(())
}
fn refine(c: i16, p: i32) -> i16 {
    if c > 0 {
        (c as i32 | p) as i16
    } else {
        -((-(c as i32)) | p) as i16
    }
}
fn decode_block(
    r: &mut Reader,
    cf: &mut [i16],
    precision: usize,
    progressive: bool,
    s: &Scan,
    i: usize,
    pred: &mut i32,
    st: &mut Prog,
) -> Result<()> {
    if !progressive || s.ss == 0 {
        if !progressive || s.ah == 0 {
            let t = s.dc[i].decode(r)? as usize;
            if t > precision + 3 {
                return Err(fail("BadCode", r.p));
            }
            *pred += extend(r.bits(t)?, t);
            cf[0] = coefficient(*pred * (1 << s.al))?;
        } else if r.bit()? != 0 {
            cf[0] |= (1 << s.al) as i16;
        }
        if progressive {
            return Ok(());
        }
    }
    let (mut k, se) = if progressive { (s.ss, s.se) } else { (1, 63) };
    if !progressive || s.ah == 0 {
        if progressive && st.eob > 0 {
            st.eob -= 1;
            return Ok(());
        }
        while k <= se {
            let rs = s.ac[i].decode(r)?;
            let run = (rs >> 4) as usize;
            let t = (rs & 15) as usize;
            if t == 0 {
                if run == 15 {
                    k += 16;
                    continue;
                }
                if progressive {
                    st.eob = (1 << run) + r.bits(run)? as usize - 1;
                }
                break;
            }
            if t > precision + 2 {
                return Err(fail("BadCode", r.p));
            }
            k += run;
            if k > se {
                return Err(fail("BadCode", r.p));
            }
            cf[k] = coefficient(extend(r.bits(t)?, t) * (1 << s.al))?;
            k += 1;
        }
        return Ok(());
    }
    let p = 1 << s.al;
    if p > 16383 {
        return Err(fail("CoefOutOfRange", r.p));
    }
    let mut first = true;
    if st.eob == 0 {
        while k <= se {
            let rs = s.ac[0].decode(r)?;
            let mut run = rs >> 4;
            let mut t = rs & 15;
            if t != 0 {
                if t != 1 {
                    return Err(fail("BadCode", r.p));
                }
                t = if r.bit()? != 0 { p } else { -p };
            } else if run != 15 {
                st.eob = (1 << run) + r.bits(run as usize)? as usize;
                if first && st.prev_ended && st.prev_len < s.max_eob {
                    st.be_high = st.be_high.min(st.prev_be as i32 - 1);
                }
                st.run_be = 0;
                st.run_len = st.eob;
                break;
            }
            first = false;
            while k <= se {
                if cf[k] != 0 {
                    if r.bit()? != 0 {
                        cf[k] = refine(cf[k], p);
                    }
                } else {
                    run -= 1;
                    if run < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if t != 0 {
                if k > se {
                    return Err(fail("BadCode", r.p));
                }
                cf[k] = t as i16;
            }
            k += 1;
        }
    }
    if st.eob > 0 {
        let mut bits = 0;
        while k <= se {
            if cf[k] != 0 {
                if r.bit()? != 0 {
                    cf[k] = refine(cf[k], p);
                }
                bits += 1;
            }
            k += 1;
        }
        st.eob -= 1;
        st.run_be += bits;
        if st.eob > 0 {
            st.be_low = st.be_low.max(st.run_be);
        } else {
            st.prev_ended = true;
            st.prev_be = st.run_be;
            st.prev_len = st.run_len;
        }
    } else {
        st.prev_ended = false;
    }
    Ok(())
}
fn encode_block(
    w: &mut Writer,
    cf: &[i16],
    progressive: bool,
    s: &Scan,
    i: usize,
    pred: &mut i32,
    st: &mut Prog,
) -> Result<()> {
    if !progressive || s.ss == 0 {
        if !progressive || s.ah == 0 {
            let v = (cf[0] as i32) >> s.al;
            let d = v - *pred;
            *pred = v;
            let t = category(d);
            s.dc[i].emit(w, t)?;
            w.bits(d - i32::from(d < 0), t)?;
        } else {
            w.bits(((cf[0] as i32) >> s.al) & 1, 1)?;
        }
        if progressive {
            return Ok(());
        }
    }
    if !progressive {
        let mut run = 0;
        for &c in &cf[1..64] {
            let v = c as i32;
            if v == 0 {
                run += 1;
                continue;
            }
            while run > 15 {
                s.ac[i].emit(w, 240)?;
                run -= 16;
            }
            let t = category(v);
            s.ac[i].emit(w, (run << 4) | t)?;
            w.bits(v - i32::from(v < 0), t)?;
            run = 0;
        }
        if run > 0 {
            s.ac[i].emit(w, 0)?;
        }
        return Ok(());
    }
    if s.ah == 0 {
        let mut run = 0;
        for &c in &cf[s.ss..=s.se] {
            let v = (c as i32).abs() >> s.al;
            if v == 0 {
                run += 1;
                continue;
            }
            flush(w, s, st)?;
            while run > 15 {
                s.ac[0].emit(w, 240)?;
                run -= 16;
            }
            let t = category(v);
            s.ac[0].emit(w, (run << 4) | t)?;
            w.bits(if c < 0 { -v - 1 } else { v }, t)?;
            run = 0;
        }
        if run > 0 {
            st.eob += 1;
            if st.eob == s.max_eob {
                flush(w, s, st)?;
            }
        }
        return Ok(());
    }
    let mut abs = [0; 64];
    let mut eob = 0;
    for k in s.ss..=s.se {
        abs[k] = (cf[k] as i32).abs() >> s.al;
        if abs[k] == 1 {
            eob = k;
        }
    }
    let (mut run, mut br) = (0, Vec::new());
    for k in s.ss..=s.se {
        let v = abs[k];
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 && k <= eob {
            flush(w, s, st)?;
            s.ac[0].emit(w, 240)?;
            run -= 16;
            for b in br.drain(..) {
                w.bits(b, 1)?;
            }
        }
        if v > 1 {
            br.push(v & 1);
            continue;
        }
        flush(w, s, st)?;
        s.ac[0].emit(w, (run << 4) | 1)?;
        w.bits(i32::from(cf[k] > 0), 1)?;
        for b in br.drain(..) {
            w.bits(b, 1)?;
        }
        run = 0;
    }
    if run > 0 || !br.is_empty() {
        st.eob += 1;
        st.bits.extend(br);
        if st.eob == s.max_eob || st.bits.len() > s.max_be {
            flush(w, s, st)?;
        }
    }
    Ok(())
}
pub(super) fn decode(data: &[u8], first: usize, j: &mut Jpeg, s: &mut Scan) -> Result<usize> {
    let mut r = Reader::new(data, first);
    s.padbit = -1;
    let mut pred = [0; 4];
    let mut st = Prog {
        be_high: i32::MAX,
        ..Prog::default()
    };
    let (_, total) = units(j, s);
    let mut rst = 0;
    for u in 0..total {
        if s.ri > 0 && u > 0 && u % s.ri == 0 {
            r.padding(s)?;
            r.restart(rst)?;
            rst += 1;
            pred.fill(0);
            st.eob = 0;
            st.prev_ended = false;
        }
        for (i, c, b) in blocks(j, s, u) {
            let cf = &mut j.components[c].coef[b * 64..b * 64 + 64];
            decode_block(
                &mut r,
                cf,
                j.precision,
                j.progressive,
                s,
                i,
                &mut pred[i],
                &mut st,
            )?;
        }
    }
    r.padding(s)?;
    if s.padbit < 0 {
        s.padbit = 1;
    }
    if j.progressive
        && s.ah > 0
        && s.ss > 0
        && (st.be_low > LIBJPEG_MAX_BE || st.be_high < LIBJPEG_MAX_BE as i32)
        && st.be_low as i32 <= st.be_high
        && st.be_low <= MAX_BE_LIMIT
    {
        s.max_be = st.be_low;
    }
    r.end()
}
pub(super) fn encode(w: &mut Writer, j: &Jpeg, s: &Scan) -> Result<()> {
    let mut pred = [0; 4];
    let mut st = Prog::default();
    let (_, total) = units(j, s);
    let mut rst = 0;
    for u in 0..total {
        if s.ri > 0 && u > 0 && u % s.ri == 0 {
            flush(w, s, &mut st)?;
            w.align(s.padbit)?;
            w.restart(rst)?;
            rst += 1;
            pred.fill(0);
        }
        for (i, c, b) in blocks(j, s, u) {
            encode_block(
                w,
                &j.components[c].coef[b * 64..b * 64 + 64],
                j.progressive,
                s,
                i,
                &mut pred[i],
                &mut st,
            )?;
        }
    }
    flush(w, s, &mut st)?;
    w.align(s.padbit)
}
