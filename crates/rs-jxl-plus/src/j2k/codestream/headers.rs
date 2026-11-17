use super::*;
pub(super) fn bare(m: u16) -> bool {
    (0xff30..=0xff3f).contains(&m)
}
fn with_length(m: u16) -> bool {
    m >= 0xff40 && m != 0xff4f && m != 0xff93 && m != 0xffd9 && m != 0xffff
}
fn component(d: &[u8], p: &mut usize, n: usize) -> Result<usize> {
    let v = if n < 257 {
        *d.get(*p).ok_or(refuse("J2K component index"))? as usize
    } else {
        be16(d, *p)? as usize
    };
    *p += if n < 257 { 1 } else { 2 };
    if v >= n {
        return Err(refuse("J2K component index"));
    }
    Ok(v)
}
fn cod(d: &[u8], mut p: usize, coc: bool, ht: bool) -> Result<(CodC, CodG)> {
    let mut c = CodC::default();
    let mut g = CodG::default();
    let s;
    if coc {
        s = *d.get(p).ok_or(refuse("J2K COC"))?;
        p += 1;
        if s & 254 != 0 {
            return Err(refuse("J2K Part2 COC style"));
        }
    } else {
        let b = d.get(p..p + 5).ok_or(refuse("J2K COD"))?;
        s = b[0];
        g.scod = s;
        g.prog = b[1];
        g.layers = u16::from_be_bytes([b[2], b[3]]) as usize;
        g.mct = b[4];
        p += 5;
        if s & 248 != 0 || g.prog > 4 || g.layers == 0 {
            return Err(refuse("J2K COD parameters"));
        }
        g.have = true;
    }
    let b = d.get(p..p + 5).ok_or(refuse("J2K COD"))?;
    c.nl = b[0] as usize;
    c.xcb = b[1] as usize + 2;
    c.ycb = b[2] as usize + 2;
    c.sty = b[3] as u32;
    c.tr = b[4];
    p += 5;
    if c.nl > MAXNL || c.xcb > 10 || c.ycb > 10 || c.xcb + c.ycb > 12 {
        return Err(refuse("J2K COD geometry"));
    }
    if ht {
        if c.sty & 64 == 0 || c.sty & 128 != 0 {
            return Err(refuse("J2K mixed HT codeblocks"));
        }
    } else if c.sty & 192 != 0 {
        return Err(refuse("J2K HT codeblock in classic mode"));
    }
    c.prec = s & 1 != 0;
    for r in 0..=c.nl {
        let pp = if c.prec {
            let b = *d.get(p).ok_or(refuse("J2K COD precincts"))?;
            p += 1;
            b
        } else {
            255
        };
        if r > 0 && (pp & 15 == 0 || pp >> 4 == 0) {
            return Err(refuse("J2K COD precinct size"));
        }
        c.pp.push(pp);
    }
    g.cod = c.clone();
    Ok((c, g))
}
fn qnt(d: &[u8], p: usize) -> Result<QntC> {
    let s = *d.get(p).ok_or(refuse("J2K QCD"))?;
    let mut q = QntC {
        style: s & 31,
        guard: (s >> 5) as i32,
        ex: Vec::new(),
    };
    let stride = match q.style {
        0 => 1,
        1 | 2 => 2,
        _ => return Err(refuse("J2K QCD style")),
    };
    for b in d[p + 1..].chunks_exact(stride).take(3 * MAXNL + 1) {
        q.ex.push((b[0] >> 3) as i32);
    }
    Ok(q)
}
fn pocs(d: &[u8], mut p: usize, n: usize) -> Result<Vec<Poc>> {
    let size = if n < 257 { 7 } else { 9 };
    let mut out = Vec::new();
    while p + size <= d.len() {
        let rs = d[p] as usize;
        p += 1;
        let cs = if n < 257 {
            let v = d[p] as usize;
            p += 1;
            v
        } else {
            let v = be16(d, p)? as usize;
            p += 2;
            v
        };
        let lye = be16(d, p)? as usize;
        p += 2;
        let re = d[p] as usize;
        p += 1;
        let ce = if n < 257 {
            let v = d[p] as usize;
            p += 1;
            v
        } else {
            let v = be16(d, p)? as usize;
            p += 2;
            v
        };
        let prg = d[p];
        p += 1;
        if prg > 4 {
            return Err(refuse("J2K POC progression"));
        }
        out.push(Poc {
            rs,
            cs,
            lye,
            re,
            ce,
            prg,
        });
    }
    Ok(out)
}
impl State<'_> {
    pub(super) fn main_item(&mut self, it: &Item) -> Result<()> {
        let d = &it.raw;
        let mut p = 4;
        match it.code {
            0xff52 => {
                self.mainG = cod(d, p, false, self.ht)?.1;
            }
            0xff53 => {
                let c = component(d, &mut p, self.ncomp)?;
                self.mainCoc[c] = Some(cod(d, p, true, self.ht)?.0);
            }
            0xff5c => self.mainQcd = Some(qnt(d, p)?),
            0xff5d => {
                let c = component(d, &mut p, self.ncomp)?;
                self.mainQcc[c] = Some(qnt(d, p)?);
            }
            0xff5f => self.mainPoc.extend(pocs(d, p, self.ncomp)?),
            0xff50 => {
                if d.len() >= 8 && be32(d, 4)? & (1 << 17) != 0 && !self.ht {
                    return Err(refuse("J2K HT CAP in classic mode"));
                }
            }
            0xff60 => self.hasPpm = true,
            0xff5e => self.hasRgn = true,
            0xff72 | 0xff73 => return Err(refuse("J2K Part2 decomposition")),
            _ => {}
        }
        if it.code == 0xff51 {
            if d.len() < 42 {
                return Err(refuse("J2K SIZ"));
            }
            self.xsiz = be32(d, 6)? as i64;
            self.ysiz = be32(d, 10)? as i64;
            self.xosiz = be32(d, 14)? as i64;
            self.yosiz = be32(d, 18)? as i64;
            self.xtsiz = be32(d, 22)? as i64;
            self.ytsiz = be32(d, 26)? as i64;
            self.xtosiz = be32(d, 30)? as i64;
            self.ytosiz = be32(d, 34)? as i64;
            self.ncomp = be16(d, 38)? as usize;
            let n = self.ncomp;
            if n == 0 || n > MAXCOMPS || d.len() < 40 + 3 * n {
                return Err(refuse("J2K SIZ components"));
            }
            if self.xtsiz == 0
                || self.ytsiz == 0
                || self.xsiz <= self.xosiz
                || self.ysiz <= self.yosiz
                || self.xtosiz > self.xosiz
                || self.ytosiz > self.yosiz
                || self.xtosiz + self.xtsiz <= self.xosiz
                || self.ytosiz + self.ytsiz <= self.yosiz
            {
                return Err(refuse("J2K SIZ grid"));
            }
            let nx = ceil(self.xsiz - self.xtosiz, self.xtsiz);
            let ny = ceil(self.ysiz - self.ytosiz, self.ytsiz);
            if nx * ny > MAXTILES as i64 {
                return Err(Error::Limit("J2K tiles"));
            }
            self.nxt = nx as usize;
            self.nyt = ny as usize;
            self.cdx.clear();
            self.cdy.clear();
            self.csiz.clear();
            for i in 0..n {
                let (dx, dy) = (d[41 + 3 * i] as usize, d[42 + 3 * i] as usize);
                if dx == 0 || dy == 0 {
                    return Err(refuse("J2K SIZ subsampling"));
                }
                self.cdx.push(dx);
                self.cdy.push(dy);
                self.csiz.push(d[40 + 3 * i]);
            }
            self.mainCoc.resize(n, None);
            self.mainQcc.resize(n, None);
        }
        Ok(())
    }
    pub(super) fn structure(&mut self) -> Result<()> {
        let d = &self.cs;
        let e = d.len();
        if e < 4 || d[..2] != [255, 79] {
            return Err(refuse("J2K no SOC"));
        }
        let mut p = 2;
        let mut items = Vec::new();
        loop {
            if p + 4 > e || d[p] != 255 {
                return Err(refuse("J2K main header cut short"));
            }
            let m = be16(d, p)?;
            if m == 0xff90 {
                break;
            }
            if items.is_empty() && m != 0xff51 {
                return Err(refuse("J2K no SIZ"));
            }
            let len = if bare(m) {
                2
            } else {
                if !with_length(m) {
                    return Err(refuse("J2K main header marker"));
                }
                let n = be16(d, p + 2)? as usize;
                if n < 2 || p + 2 + n > e {
                    return Err(refuse("J2K main header length"));
                }
                2 + n
            };
            items.push(Item {
                code: m,
                raw: d[p..p + len].to_vec(),
                ..Item::default()
            });
            p += len;
        }
        for it in &items {
            self.main_item(it)?;
        }
        self.main = items;
        if !self.mainG.have || self.mainQcd.is_none() {
            return Err(refuse("J2K missing COD/QCD"));
        }
        let d = &self.cs;
        while p + 2 <= e {
            if be16(d, p)? != 0xff90 || p + 12 > e || be16(d, p + 2)? != 10 {
                break;
            }
            let mut tp = Tp {
                hstart: p,
                isot: be16(d, p + 4)? as usize,
                psot: be32(d, p + 6)? as usize,
                tpsot: d[p + 10],
                tnsot: d[p + 11],
                ..Tp::default()
            };
            if tp.isot >= self.nxt * self.nyt {
                return Err(refuse("J2K SOT tile"));
            }
            let mut q = p + 12;
            let mut ok = false;
            while q + 2 <= e {
                let m = be16(d, q)?;
                if m == 0xff93 {
                    ok = true;
                    break;
                }
                let len = if bare(m) {
                    2
                } else {
                    if d[q] != 255 || !with_length(m) || q + 4 > e {
                        break;
                    }
                    let n = be16(d, q + 2)? as usize;
                    if n < 2 || q + 2 + n > e {
                        break;
                    }
                    2 + n
                };
                tp.items.push(Item {
                    code: m,
                    raw: d[q..q + len].to_vec(),
                    ..Item::default()
                });
                q += len;
            }
            if !ok {
                break;
            }
            tp.dstart = q + 2;
            if tp.psot == 0 {
                tp.psotMode = 1;
                tp.dend = e;
                if e >= tp.dstart + 2 && d[e - 2..e] == [255, 217] {
                    tp.dend = e - 2;
                }
            } else if p + tp.psot > e {
                tp.psotMode = 2;
                tp.dend = e;
            } else {
                tp.dend = p + tp.psot;
                if tp.dend < tp.dstart {
                    return Err(refuse("J2K SOT Psot"));
                }
            }
            p = tp.dend;
            let mode = tp.psotMode;
            self.tps.push(tp);
            if self.tps.len() > MAXTILES * 256 {
                return Err(Error::Limit("J2K tile parts"));
            }
            if mode == 1 {
                break;
            }
        }
        if self.tps.is_empty() {
            return Err(refuse("J2K no tile-part"));
        }
        self.tail = self.cs[p..].to_vec();
        self.ieCap = side_bound(e);
        self.split_ppm()
    }
    pub(super) fn split_ppm(&mut self) -> Result<()> {
        if !self.hasPpm {
            return Ok(());
        }
        self.ppm.clear();
        for z in 0..=255 {
            for it in &self.main {
                if it.code == 0xff60 && it.raw.len() >= 5 && it.raw[4] == z {
                    self.ppm.extend_from_slice(&it.raw[5..]);
                }
            }
        }
        let mut q = 0;
        for tp in &mut self.tps {
            let n = be32(&self.ppm, q)? as usize;
            if n > self.ppm.len().saturating_sub(q + 4) {
                return Err(refuse("J2K PPM cut short"));
            }
            tp.ppmOff = q + 4;
            tp.ppmLen = n;
            q += 4 + n;
        }
        Ok(())
    }
    pub(super) fn tile_params(&mut self) -> Result<()> {
        let mut tg = None;
        let mut tcoc = vec![None; self.ncomp];
        let mut tqcc = vec![None; self.ncomp];
        let mut tqcd = None;
        let mut tpoc = Vec::new();
        self.hdrPacked = self.hasPpm;
        self.pkd.clear();
        self.tileRgn = self.hasRgn;
        for &ti in &self.tileTps {
            let tp = &self.tps[ti];
            if self.hasPpm && tp.ppmLen > 0 {
                self.pkd
                    .extend_from_slice(&self.ppm[tp.ppmOff..tp.ppmOff + tp.ppmLen]);
            }
            for it in &tp.items {
                let d = &it.raw;
                let mut p = 4;
                match it.code {
                    0xff52 => tg = Some(cod(d, p, false, self.ht)?.1),
                    0xff53 => {
                        let c = component(d, &mut p, self.ncomp)?;
                        tcoc[c] = Some(cod(d, p, true, self.ht)?.0);
                    }
                    0xff5c => tqcd = Some(qnt(d, p)?),
                    0xff5d => {
                        let c = component(d, &mut p, self.ncomp)?;
                        tqcc[c] = Some(qnt(d, p)?);
                    }
                    0xff5f => tpoc.extend(pocs(d, p, self.ncomp)?),
                    0xff5e => self.tileRgn = true,
                    0xff61 => {
                        if self.hasPpm || d.len() < 5 {
                            return Err(refuse("J2K PPM/PPT"));
                        }
                        self.hdrPacked = true;
                    }
                    0xff72 | 0xff73 => return Err(refuse("J2K Part2 decomposition")),
                    _ => {}
                }
            }
        }
        if self.hdrPacked && !self.hasPpm {
            for z in 0..=255 {
                for &ti in &self.tileTps {
                    for it in &self.tps[ti].items {
                        if it.code == 0xff61 && it.raw.len() > 5 && it.raw[4] == z {
                            self.pkd.extend_from_slice(&it.raw[5..]);
                        }
                    }
                }
            }
        }
        self.par = Params {
            g: tg.clone().unwrap_or_else(|| self.mainG.clone()),
            cod: Vec::new(),
            qnt: Vec::new(),
            pocs: if tpoc.is_empty() {
                self.mainPoc.clone()
            } else {
                tpoc
            },
        };
        for c in 0..self.ncomp {
            self.par.cod.push(
                tcoc[c]
                    .clone()
                    .or_else(|| tg.as_ref().map(|g| g.cod.clone()))
                    .or_else(|| self.mainCoc[c].clone())
                    .unwrap_or_else(|| self.mainG.cod.clone()),
            );
            self.par.qnt.push(
                tqcc[c]
                    .clone()
                    .or_else(|| tqcd.clone())
                    .or_else(|| self.mainQcc[c].clone())
                    .or_else(|| self.mainQcd.clone())
                    .ok_or(refuse("J2K no QCD"))?,
            );
        }
        Ok(())
    }
}
