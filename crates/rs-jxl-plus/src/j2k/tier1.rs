// PJA MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Sources: pas/pjaj2kt1.c and pjaj2kt1.h. See LICENSE-PJA and NOTICE-PJA.
//! JPEG 2000 classic tier-1 decisions and exact MQ segment termination.
use crate::{Error, Result};
#[path = "tables/tier1.rs"]
mod tables;
use tables::*;
pub const SIG: u8 = 1;
pub const NEG: u8 = 2;
pub const PIV: u8 = 4;
pub const REF: u8 = 8;
const SOUTH: u8 = 224;
pub type Contexts = [u8; 38];
pub fn reset_contexts() -> Contexts {
    let mut x = [0; 38];
    x[18] = 46;
    x[17] = 3;
    x[0] = 4;
    x
}
fn valid_contexts(x: &Contexts) -> bool {
    x[..19].iter().all(|&v| v < 47) && x[19..].iter().all(|&v| v < 2)
}
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub width: usize,
    pub height: usize,
    pub orientation: usize,
    pub style: u32,
    pub passes: usize,
}
impl Block {
    fn validate(self) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.width > 1024
            || self.height > 1024
            || self.width * self.height > 4096
            || self.orientation > 3
            || self.passes > 255
        {
            Err(Error::Invalid("tier1 block arguments"))
        } else {
            Ok(())
        }
    }
}
pub struct Segment<'a> {
    pub bytes: &'a [u8],
    pub passes: usize,
}
#[derive(Debug)]
pub struct Passes {
    pub symbols: Vec<u8>,
    pub pass_ends: Vec<usize>,
    pub used_bits: usize,
}
#[derive(Clone)]
struct MqDecoder<'a> {
    data: &'a [u8],
    p: usize,
    a: u32,
    c: u32,
    ct: i32,
}
impl<'a> MqDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut q = Self {
            data,
            p: 0,
            a: 0x8000,
            c: u32::from(data.first().copied().unwrap_or(255)) << 16,
            ct: 0,
        };
        q.bytein();
        q.c = q.c.wrapping_shl(7);
        q.ct -= 7;
        q
    }
    fn byte(&self, p: usize) -> u32 {
        self.data.get(p).copied().unwrap_or(255) as u32
    }
    fn bytein(&mut self) {
        if self.byte(self.p) == 255 {
            if self.byte(self.p + 1) > 0x8f {
                self.c = self.c.wrapping_add(0xff00);
                self.ct = 8;
            } else {
                self.p += 1;
                self.c = self.c.wrapping_add(self.byte(self.p) << 9);
                self.ct = 7;
            }
        } else {
            self.p += 1;
            self.c = self.c.wrapping_add(self.byte(self.p) << 8);
            self.ct = 8;
        }
    }
    fn bit(&mut self, x: &mut Contexts, cx: usize) -> u8 {
        let i = x[cx] as usize;
        let qe = QE[i] as u32;
        self.a -= qe;
        let d;
        if self.c >> 16 < qe {
            if self.a < qe {
                self.a = qe;
                d = x[19 + cx];
                x[cx] = NMPS[i];
            } else {
                self.a = qe;
                d = 1 - x[19 + cx];
                x[19 + cx] ^= SWITCH[i];
                x[cx] = NLPS[i];
            }
        } else {
            self.c = self.c.wrapping_sub(qe << 16);
            if self.a & 0x8000 == 0 {
                if self.a < qe {
                    d = 1 - x[19 + cx];
                    x[19 + cx] ^= SWITCH[i];
                    x[cx] = NLPS[i];
                } else {
                    d = x[19 + cx];
                    x[cx] = NMPS[i];
                }
            } else {
                return x[19 + cx];
            }
        }
        loop {
            if self.ct == 0 {
                self.bytein()
            }
            self.a <<= 1;
            self.c = self.c.wrapping_shl(1);
            self.ct -= 1;
            if self.a & 0x8000 != 0 {
                break;
            }
        }
        d
    }
}
pub(crate) fn sign_context(st: &[u8], i: usize, stride: usize, excl: bool) -> (usize, u8) {
    fn sign(f: u8) -> i32 {
        if f & SIG == 0 {
            0
        } else if f & NEG != 0 {
            -1
        } else {
            1
        }
    }
    let mut hc = (sign(st[i - 1]) + sign(st[i + 1])).clamp(-1, 1);
    let mut vc = (sign(st[i - stride]) + if excl { 0 } else { sign(st[i + stride]) }).clamp(-1, 1);
    let xr = (hc < 0 || (hc == 0 && vc < 0)) as u8;
    if xr != 0 {
        hc = -hc;
        vc = -vc
    }
    (
        if hc == 0 {
            9 + (vc != 0) as usize
        } else {
            (12 + vc) as usize
        },
        xr,
    )
}
pub(crate) fn make_significant(st: &mut [u8], nb: &mut [u8], i: usize, s: usize, neg: u8) {
    st[i] |= SIG | if neg != 0 { NEG } else { 0 };
    for (p, flag) in [
        (i - s - 1, 128),
        (i - s, 64),
        (i - s + 1, 32),
        (i - 1, 16),
        (i + 1, 8),
        (i + s - 1, 4),
        (i + s, 2),
        (i + s + 1, 1),
    ] {
        nb[p] |= flag
    }
}
struct Walker<'a> {
    raw: bool,
    q: MqDecoder<'a>,
    contexts: Contexts,
    rp: usize,
    rc: u32,
    rct: i32,
    bits: Option<&'a [u8]>,
    bp: usize,
    nbits: usize,
    cap: usize,
    symbols: Vec<u8>,
}
impl Walker<'_> {
    fn decide(&mut self, cx: usize) -> Result<u8> {
        if self.symbols.len() >= self.cap {
            return Err(Error::Limit("tier1 decisions"));
        }
        let b = if let Some(bits) = self.bits {
            if self.bp >= self.nbits {
                return Err(Error::Invalid("tier1 replay bits exhausted"));
            }
            let b = (bits[self.bp / 8] >> (7 - self.bp % 8)) & 1;
            self.bp += 1;
            b
        } else if self.raw {
            if self.rct == 0 {
                if self.rc == 255 {
                    let b = self.q.byte(self.rp);
                    if b > 0x8f {
                        self.rc = 255;
                        self.rct = 8;
                    } else {
                        self.rc = b;
                        self.rp += 1;
                        self.rct = 7;
                    }
                } else {
                    self.rc = self.q.byte(self.rp);
                    self.rp += 1;
                    self.rct = 8;
                }
            }
            self.rct -= 1;
            ((self.rc >> self.rct) & 1) as u8
        } else {
            self.q.bit(&mut self.contexts, cx)
        };
        self.symbols
            .push(((if self.raw { 19 } else { cx }) << 1) as u8 | b);
        Ok(b)
    }
    fn sign(&mut self, st: &mut [u8], nb: &mut [u8], i: usize, s: usize, excl: bool) -> Result<()> {
        let (cx, xr) = sign_context(st, i, s, excl);
        let b = self.decide(cx)?;
        make_significant(st, nb, i, s, if self.raw { b } else { b ^ xr });
        Ok(())
    }
    fn pass(&mut self, b: Block, kind: usize, st: &mut [u8], nb: &mut [u8]) -> Result<()> {
        let s = b.width + 2;
        let causal = b.style & 8 != 0;
        for y0 in (0..b.height).step_by(4) {
            let y1 = (y0 + 4).min(b.height);
            for x in 0..b.width {
                let i0 = (y0 + 1) * s + x + 1;
                let mut start = y0;
                if kind == 2
                    && y1 - y0 == 4
                    && (st[i0] | st[i0 + s] | st[i0 + 2 * s] | (st[i0 + 3 * s] & (SIG | PIV))) == 0
                    && (nb[i0]
                        | nb[i0 + s]
                        | nb[i0 + 2 * s]
                        | (nb[i0 + 3 * s] & if causal { !SOUTH } else { 255 }))
                        == 0
                {
                    if self.decide(17)? == 0 {
                        continue;
                    }
                    let r = ((self.decide(18)? << 1) | self.decide(18)?) as usize;
                    self.sign(st, nb, i0 + r * s, s, causal && (y0 + r) % 4 == 3)?;
                    start = y0 + r + 1;
                }
                for y in start..y1 {
                    let i = i0 + (y - y0) * s;
                    let mut m = nb[i];
                    if causal && y % 4 == 3 {
                        m &= !SOUTH
                    }
                    match kind {
                        0 => {
                            if st[i] & SIG != 0 || m == 0 {
                                continue;
                            }
                            st[i] |= PIV;
                            if self.decide(ZC[b.orientation][m as usize] as usize)? != 0 {
                                self.sign(st, nb, i, s, causal && y % 4 == 3)?;
                            }
                        }
                        1 => {
                            if st[i] & (SIG | PIV) != SIG {
                                continue;
                            }
                            self.decide(if st[i] & REF != 0 {
                                16
                            } else if m != 0 {
                                15
                            } else {
                                14
                            })?;
                            st[i] |= REF;
                        }
                        _ => {
                            if st[i] & (SIG | PIV) != 0 {
                                continue;
                            }
                            if self.decide(ZC[b.orientation][m as usize] as usize)? != 0 {
                                self.sign(st, nb, i, s, causal && y % 4 == 3)?;
                            }
                        }
                    }
                }
            }
        }
        if kind == 2 {
            for y in 0..b.height {
                for x in 0..b.width {
                    st[(y + 1) * s + x + 1] &= !PIV;
                }
            }
            if b.style & 32 != 0 {
                for _ in 0..4 {
                    self.decide(18)?;
                }
            }
        }
        Ok(())
    }
}
fn walk<'a>(
    b: Block,
    segments: &[Segment<'a>],
    bits: Option<(&'a [u8], usize, usize)>,
    cap: usize,
) -> Result<Passes> {
    b.validate()?;
    let replay = bits.is_some();
    let (packed, bp, nbits) = bits.unwrap_or((&[], 0, 0));
    if nbits > packed.len().saturating_mul(8) || bp > nbits {
        return Err(Error::Invalid("tier1 replay range"));
    }
    let mut w = Walker {
        raw: false,
        q: MqDecoder::new(&[]),
        contexts: reset_contexts(),
        rp: 0,
        rc: 0,
        rct: 0,
        bits: if replay { Some(packed) } else { None },
        bp,
        nbits,
        cap,
        symbols: Vec::new(),
    };
    let n = (b.width + 2) * (b.height + 2);
    let mut st = vec![0; n];
    let mut nb = vec![0; n];
    let mut ends = Vec::new();
    let (mut seg, mut left) = (0, 0);
    for k in 0..b.passes {
        let kind = if k == 0 { 2 } else { (k - 1) % 3 };
        w.raw = b.style & 1 != 0 && k >= 10 && kind < 2;
        if !replay {
            if left == 0 {
                let sg = segments
                    .get(seg)
                    .ok_or(Error::Invalid("tier1 missing segment"))?;
                if sg.passes == 0 {
                    return Err(Error::Invalid("tier1 zero segment passes"));
                }
                left = sg.passes;
                seg += 1;
                w.q = MqDecoder::new(sg.bytes);
                w.rp = 0;
                w.rc = 0;
                w.rct = 0;
            }
            left -= 1;
        }
        w.pass(b, kind, &mut st, &mut nb)?;
        ends.push(w.symbols.len());
        if b.style & 2 != 0 && !w.raw && !replay {
            w.contexts = reset_contexts();
        }
    }
    if !replay && (seg != segments.len() || left != 0) {
        return Err(Error::Invalid("tier1 segment passes mismatch"));
    }
    Ok(Passes {
        symbols: w.symbols,
        pass_ends: ends,
        used_bits: w.bp - bp,
    })
}
pub fn decode(b: Block, segments: &[Segment<'_>], cap: usize) -> Result<Passes> {
    walk(b, segments, None, cap)
}
pub fn replay(b: Block, bits: &[u8], bitpos: usize, nbits: usize, cap: usize) -> Result<Passes> {
    walk(b, &[], Some((bits, bitpos, nbits)), cap)
}

#[derive(Clone)]
struct MqEncoder {
    a: u32,
    c: u32,
    ct: u32,
    bp: usize,
    buf: Vec<u8>,
}
impl MqEncoder {
    fn new() -> Self {
        Self {
            a: 0x8000,
            c: 0,
            ct: 12,
            bp: 0,
            buf: vec![0],
        }
    }
    fn next(&mut self, b: u8) {
        self.bp += 1;
        if self.bp == self.buf.len() {
            self.buf.push(b)
        } else {
            self.buf[self.bp] = b
        }
    }
    fn byteout(&mut self) {
        if self.buf[self.bp] == 255 {
            self.next((self.c >> 20) as u8);
            self.c &= 0xfffff;
            self.ct = 7;
        } else if self.c & 0x8000000 == 0 {
            self.next((self.c >> 19) as u8);
            self.c &= 0x7ffff;
            self.ct = 8;
        } else {
            self.buf[self.bp] = self.buf[self.bp].wrapping_add(1);
            if self.buf[self.bp] == 255 {
                self.c &= 0x7ffffff;
                self.next((self.c >> 20) as u8);
                self.c &= 0xfffff;
                self.ct = 7;
            } else {
                self.next((self.c >> 19) as u8);
                self.c &= 0x7ffff;
                self.ct = 8;
            }
        }
    }
    fn renorm(&mut self) {
        loop {
            self.a <<= 1;
            self.c = self.c.wrapping_shl(1);
            self.ct -= 1;
            if self.ct == 0 {
                self.byteout()
            }
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }
    fn bit(&mut self, x: &mut Contexts, cx: usize, d: u8) {
        let i = x[cx] as usize;
        let q = QE[i] as u32;
        self.a -= q;
        if d == x[19 + cx] {
            if self.a & 0x8000 == 0 {
                if self.a < q {
                    self.a = q
                } else {
                    self.c = self.c.wrapping_add(q)
                }
                x[cx] = NMPS[i];
                self.renorm();
            } else {
                self.c = self.c.wrapping_add(q)
            }
        } else {
            if self.a < q {
                self.c = self.c.wrapping_add(q)
            } else {
                self.a = q
            }
            x[19 + cx] ^= SWITCH[i];
            x[cx] = NLPS[i];
            self.renorm();
        }
    }
    fn flush(&mut self) -> usize {
        let t = self.c.wrapping_add(self.a);
        self.c |= 0xffff;
        if self.c >= t {
            self.c = self.c.wrapping_sub(0x8000)
        }
        self.c = self.c.wrapping_shl(self.ct);
        self.byteout();
        self.c = self.c.wrapping_shl(self.ct);
        self.byteout();
        if self.buf[self.bp] == 255 {
            self.bp - 1
        } else {
            self.bp
        }
    }
    fn erterm(&mut self) -> usize {
        let mut k = 12 - self.ct as i32;
        while k > 0 {
            self.c = self.c.wrapping_shl(self.ct);
            self.ct = 0;
            self.byteout();
            k -= self.ct as i32;
        }
        if self.buf[self.bp] != 255 {
            self.byteout()
        }
        self.bp - 1
    }
    fn lengths(&self) -> [usize; 3] {
        let mut f = self.clone();
        let mut e = self.clone();
        [self.bp.saturating_sub(1), f.flush(), e.erterm()]
    }
}
#[derive(Debug)]
pub struct MqEncoded {
    pub flush: Vec<u8>,
    pub erterm: Vec<u8>,
    pub committed: usize,
    pub pass_lengths: Vec<[usize; 3]>,
    pub contexts: Contexts,
}
pub fn mq_encode(
    symbols: &[u8],
    pass_ends: &[usize],
    reset: bool,
    mut contexts: Contexts,
) -> Result<MqEncoded> {
    if !valid_contexts(&contexts) || symbols.iter().any(|&v| v >= 38) {
        return Err(Error::Invalid("MQ context or decision"));
    }
    let mut m = MqEncoder::new();
    let mut i = 0;
    let mut lengths = Vec::new();
    for end in pass_ends
        .iter()
        .copied()
        .chain(std::iter::once(symbols.len()))
    {
        if end < i || end > symbols.len() {
            return Err(Error::Invalid("MQ pass boundary"));
        }
        for &v in &symbols[i..end] {
            m.bit(&mut contexts, (v >> 1) as usize, v & 1)
        }
        i = end;
        if lengths.len() < pass_ends.len() {
            lengths.push(m.lengths());
            if reset {
                contexts = reset_contexts()
            }
        }
    }
    let committed = m.bp.saturating_sub(1);
    let mut e = m.clone();
    let nf = m.flush();
    let ne = e.erterm();
    Ok(MqEncoded {
        flush: m.buf[1..1 + nf].to_vec(),
        erterm: e.buf[1..1 + ne].to_vec(),
        committed,
        pass_lengths: lengths,
        contexts,
    })
}
fn mq_check(
    data: &[u8],
    contexts: &Contexts,
    symbols: &[u8],
    pass_ends: &[usize],
    reset: bool,
) -> Option<usize> {
    if !valid_contexts(contexts) {
        return None;
    }
    let mut q = MqDecoder::new(data);
    let mut x = *contexts;
    let mut j = 0;
    for (i, &v) in symbols.iter().enumerate() {
        while j < pass_ends.len() && pass_ends[j] == i {
            if reset {
                x = reset_contexts()
            }
            j += 1;
        }
        if v >= 38 || q.bit(&mut x, (v >> 1) as usize) != (v & 1) {
            return None;
        }
    }
    Some(q.p + 1 + (q.ct <= 1) as usize)
}
pub fn mq_valid(
    data: &[u8],
    contexts: &Contexts,
    symbols: &[u8],
    pass_ends: &[usize],
    reset: bool,
) -> bool {
    mq_check(data, contexts, symbols, pass_ends, reset).is_some()
}
pub fn mq_reach(
    data: &[u8],
    contexts: &Contexts,
    symbols: &[u8],
    pass_ends: &[usize],
    reset: bool,
) -> Option<usize> {
    mq_check(data, contexts, symbols, pass_ends, reset)
}
pub fn mq_minprefix(
    data: &[u8],
    contexts: &Contexts,
    symbols: &[u8],
    pass_ends: &[usize],
    reset: bool,
    window: usize,
) -> usize {
    let mut n = data.len();
    let min = n.saturating_sub(window);
    while n > min && mq_valid(&data[..n - 1], contexts, symbols, pass_ends, reset) {
        n -= 1
    }
    n
}
pub fn raw_pack(symbols: &[u8], pad: u8) -> Result<Vec<u8>> {
    if pad > 2 {
        return Err(Error::Invalid("raw padding"));
    }
    let mut out = Vec::new();
    let (mut ct, mut c, mut prev) = (8u32, 0u32, 0u32);
    for &v in symbols {
        if ct == 0 {
            out.push(c as u8);
            prev = c;
            ct = if prev == 255 { 7 } else { 8 };
            c = 0;
        }
        ct -= 1;
        c |= u32::from(v & 1) << ct;
    }
    if ct < if prev == 255 { 7 } else { 8 } {
        let mut bv = 0;
        while ct > 0 {
            ct -= 1;
            if pad == 0 {
                c |= bv << ct;
                bv ^= 1;
            } else if pad == 2 {
                c |= 1 << ct
            }
        }
        out.push(c as u8);
    }
    while out.last() == Some(&255) {
        out.pop();
    }
    Ok(out)
}
pub(crate) fn zero_context(orientation: usize, mask: u8) -> usize {
    ZC[orientation][mask as usize] as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mq_termination_and_prefixes() {
        let sym: Vec<u8> = (0..300)
            .map(|i| (((i * 7 % 19) << 1) | ((i * 17 + i / 5) % 2)) as u8)
            .collect();
        for reset in [false, true] {
            let ends = [0, 17, 100, 299];
            let ctx = reset_contexts();
            let e = mq_encode(&sym, &ends, reset, ctx).unwrap();
            assert!(mq_valid(&e.flush, &ctx, &sym, &ends, reset));
            assert!(mq_valid(&e.erterm, &ctx, &sym, &ends, reset));
            let min = mq_minprefix(&e.flush, &ctx, &sym, &ends, reset, 16);
            assert!(mq_valid(&e.flush[..min], &ctx, &sym, &ends, reset));
            assert!(mq_reach(&e.flush, &ctx, &sym, &ends, reset).is_some());
        }
    }
    #[test]
    fn replay_and_decode_passes() {
        let bits: Vec<u8> = (0..4096).map(|i| (i * 37 + i / 3) as u8).collect();
        for style in [0, 2, 8, 32, 1, 3, 9, 35] {
            let block = Block {
                width: 5,
                height: 4,
                orientation: 3,
                style,
                passes: 12,
            };
            let replayed = replay(block, &bits, 3, bits.len() * 8, 4096).unwrap();
            if style & 1 == 0 {
                let encoded = mq_encode(
                    &replayed.symbols,
                    &replayed.pass_ends[..11],
                    style & 2 != 0,
                    reset_contexts(),
                )
                .unwrap();
                let decoded = decode(
                    block,
                    &[Segment {
                        bytes: &encoded.flush,
                        passes: 12,
                    }],
                    4096,
                )
                .unwrap();
                assert_eq!(decoded.symbols, replayed.symbols);
                assert_eq!(decoded.pass_ends, replayed.pass_ends);
            }
        }
    }
    #[test]
    fn raw_padding() {
        assert!(raw_pack(&[1; 8], 2).unwrap().is_empty());
        assert_eq!(raw_pack(&[0, 1, 0], 0).unwrap(), [0x4a]);
        assert!(raw_pack(&[], 3).is_err());
    }
}
#[cfg(test)]
pub(crate) mod oracle;
