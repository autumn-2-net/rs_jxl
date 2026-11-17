// PJA MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Sources: pas/pjaj2kcs.pas PltCount/Bytes, TlmCount/Bytes,
// WriteHeaders, ReadHeaders, Assemble. See NOTICE-PJA.
use super::*;
pub(super) fn plt_count(raw: &[u8]) -> usize {
    raw.get(5..)
        .unwrap_or(&[])
        .iter()
        .filter(|&&b| b & 128 == 0)
        .count()
}
pub(super) fn plt_bytes(z: u8, lens: &[usize], from: usize, count: usize) -> Result<Vec<u8>> {
    let values = lens
        .get(from..from.checked_add(count).ok_or(refuse("J2K PLT range"))?)
        .ok_or(refuse("J2K PLT range"))?;
    let mut w = BW::default();
    w.put_n(&[255, 88, 0, 0, z], 0, 5)?;
    for &value in values {
        let mut v = value as u64;
        let mut temp = [0u8; 10];
        let mut n = 0;
        loop {
            temp[n] = (v & 127) as u8;
            v >>= 7;
            n += 1;
            if v == 0 || n == 10 {
                break;
            }
        }
        for k in (0..n).rev() {
            w.put(temp[k] | if k > 0 { 128 } else { 0 })?;
        }
    }
    if w.d.len() - 2 > 65535 {
        return Err(refuse("J2K PLT too long"));
    }
    let len = (w.d.len() - 2) as u16;
    w.d[2..4].copy_from_slice(&len.to_be_bytes());
    Ok(w.d)
}
pub(super) fn tlm_count(raw: &[u8]) -> usize {
    if raw.len() < 6 {
        return 0;
    }
    let st = raw[5];
    let tsz = (st >> 4) & 3;
    if tsz == 3 {
        return 0;
    }
    let size = tsz as usize + 2 + 2 * ((st >> 6) & 1) as usize;
    if !(raw.len() - 6).is_multiple_of(size) {
        return 0;
    }
    (raw.len() - 6) / size
}
pub(super) fn tlm_bytes(
    j: &State<'_>,
    z: u8,
    st: u8,
    from: usize,
    count: usize,
    lens: &[usize],
) -> Result<Vec<u8>> {
    let end = from.checked_add(count).ok_or(refuse("J2K TLM range"))?;
    if end > lens.len() || end > j.tps.len() {
        return Err(refuse("J2K TLM range"));
    }
    let tsz = (st >> 4) & 3;
    let psz = (st >> 6) & 1;
    let mut w = BW::default();
    w.put_n(&[255, 85, 0, 0, z, st], 0, 6)?;
    for (i, &v) in lens.iter().enumerate().take(end).skip(from) {
        if tsz == 1 {
            w.put(j.tps[i].isot as u8)?;
        } else if tsz == 2 {
            let bytes = (j.tps[i].isot as u16).to_be_bytes();
            w.put_n(&bytes, 0, 2)?;
        }
        if psz == 1 {
            let v = u32::try_from(v).map_err(|_| refuse("J2K TLM length"))?;
            w.put_n(&v.to_be_bytes(), 0, 4)?;
        } else {
            let v = u16::try_from(v).map_err(|_| refuse("J2K TLM length"))?;
            w.put_n(&v.to_be_bytes(), 0, 2)?;
        }
    }
    if w.d.len() - 2 > 65535 {
        return Err(refuse("J2K TLM too long"));
    }
    let len = (w.d.len() - 2) as u16;
    w.d[2..4].copy_from_slice(&len.to_be_bytes());
    Ok(w.d)
}
impl State<'_> {
    pub(super) fn write_headers(&mut self) -> Result<()> {
        let lens: Vec<usize> = self.tps.iter().map(|t| t.dend - t.hstart).collect();
        let mut out = BW::default();
        out.v(self.main.len() as u64)?;
        let mut tli = 0usize;
        for it in &self.main {
            if it.code == 0xff55 {
                let q = tlm_count(&it.raw);
                if q > 0 && tli.checked_add(q).is_some_and(|n| n <= self.tps.len()) {
                    let bytes = tlm_bytes(self, it.raw[4], it.raw[5], tli, q, &lens)?;
                    if bytes == it.raw {
                        out.put(1)?;
                        out.put(it.raw[4])?;
                        out.put(it.raw[5])?;
                        out.v(q as u64)?;
                        tli += q;
                        continue;
                    }
                }
                tli = tli.checked_add(q).ok_or(refuse("J2K TLM index"))?;
            }
            out.put(0)?;
            out.v(it.raw.len() as u64)?;
            out.put_n(&it.raw, 0, it.raw.len())?;
        }
        out.v(self.tps.len() as u64)?;
        for tp in &self.tps {
            out.v(tp.isot as u64)?;
            out.put(tp.tpsot)?;
            out.put(tp.tnsot)?;
            out.put(tp.psotMode)?;
            if tp.psotMode == 2 {
                out.v(tp.psot as u64)?;
            }
            out.v(tp.items.len() as u64)?;
            let mut pi = 0usize;
            for it in &tp.items {
                if it.code == 0xff58 {
                    let count = plt_count(&it.raw);
                    if it.raw.len() > 5 && pi.checked_add(count).is_some_and(|n| n <= tp.npk) {
                        let bytes = plt_bytes(it.raw[4], &tp.plens, pi, count)?;
                        if bytes == it.raw {
                            out.put(2)?;
                            out.put(it.raw[4])?;
                            out.v(count as u64)?;
                            pi += count;
                            continue;
                        }
                    }
                    pi = pi.checked_add(count).ok_or(refuse("J2K PLT index"))?;
                }
                out.put(0)?;
                out.v(it.raw.len() as u64)?;
                out.put_n(&it.raw, 0, it.raw.len())?;
            }
        }
        out.v(self.tail.len() as u64)?;
        out.put_n(&self.tail, 0, self.tail.len())?;
        self.sw[SH_HDR] = out;
        Ok(())
    }
    pub(super) fn read_headers(&mut self) -> Result<()> {
        let n = self.sr[SH_HDR].vi((1 << 20).min(self.sr[SH_HDR].remaining()))?;
        self.main.clear();
        for i in 0..n {
            let kind = self.sr[SH_HDR].get()?;
            let mut it = Item::default();
            if kind == 1 {
                it.code = 0xff55;
                it.kind = 1;
                it.z = self.sr[SH_HDR].get()?;
                it.st = self.sr[SH_HDR].get()?;
                it.cnt = self.sr[SH_HDR].vi(MAXTILES * 256)?;
            } else if kind == 0 {
                let n = self.sr[SH_HDR].vi(65537)?;
                it.raw = self.sr[SH_HDR].take(n)?;
                if it.raw.len() < 2 || it.raw[0] != 255 {
                    return Err(refuse("J2K bad header"));
                }
                it.code = be16(&it.raw, 0)?;
                if it.raw.len() < 4 && !headers::bare(it.code) {
                    return Err(refuse("J2K bad header"));
                }
                if i == 0 && it.code != 0xff51 {
                    return Err(refuse("J2K missing first SIZ"));
                }
                self.main_item(&it)?;
            } else {
                return Err(refuse("J2K bad header kind"));
            }
            self.main.push(it);
        }
        if n == 0 || !self.mainG.have || self.mainQcd.is_none() {
            return Err(refuse("J2K missing main header"));
        }
        let n = self.sr[SH_HDR].vi((MAXTILES * 256).min(self.sr[SH_HDR].remaining() / 5))?;
        self.tps.clear();
        for _ in 0..n {
            let mut tp = Tp::default();
            let tiles = self
                .nxt
                .checked_mul(self.nyt)
                .filter(|&n| n > 0)
                .ok_or(refuse("J2K tile count"))?;
            tp.isot = self.sr[SH_HDR].vi(tiles - 1)?;
            tp.tpsot = self.sr[SH_HDR].get()?;
            tp.tnsot = self.sr[SH_HDR].get()?;
            tp.psotMode = self.sr[SH_HDR].get()?;
            if tp.psotMode > 2 {
                return Err(refuse("J2K PSOT mode"));
            }
            if tp.psotMode == 2 {
                tp.psot = self.sr[SH_HDR].vi(u32::MAX as usize)?;
            }
            let count = self.sr[SH_HDR].vi((1 << 20).min(self.sr[SH_HDR].remaining()))?;
            for _ in 0..count {
                let mut it = Item::default();
                let kind = self.sr[SH_HDR].get()?;
                if kind == 2 {
                    it.code = 0xff58;
                    it.kind = 2;
                    it.z = self.sr[SH_HDR].get()?;
                    it.cnt = self.sr[SH_HDR].vi(MAXENTRIES)?;
                } else if kind == 0 {
                    let n = self.sr[SH_HDR].vi(65537)?;
                    it.raw = self.sr[SH_HDR].take(n)?;
                    if it.raw.len() < 2 || it.raw[0] != 255 {
                        return Err(refuse("J2K bad tile header"));
                    }
                    it.code = be16(&it.raw, 0)?;
                    if it.raw.len() < 4 && !headers::bare(it.code) {
                        return Err(refuse("J2K bad tile header"));
                    }
                } else {
                    return Err(refuse("J2K bad tile header kind"));
                }
                tp.items.push(it);
            }
            self.tps.push(tp);
        }
        if n == 0 {
            return Err(refuse("J2K no tile parts"));
        }
        let n = self.sr[SH_HDR].vi(i32::MAX as usize)?;
        self.tail = self.sr[SH_HDR].take(n)?;
        self.split_ppm()?;
        Ok(())
    }
    pub(super) fn assemble(&self) -> Result<Vec<u8>> {
        let lens: Vec<usize> = self.tps.iter().map(|t| t.bytes.len()).collect();
        let mut w = BW::default();
        w.put_n(&[255, 79], 0, 2)?;
        let mut tli = 0usize;
        for it in &self.main {
            if it.kind == 1 {
                if tli.checked_add(it.cnt).is_none_or(|n| n > self.tps.len()) {
                    return Err(refuse("J2K bad TLM"));
                }
                let bytes = tlm_bytes(self, it.z, it.st, tli, it.cnt, &lens)?;
                w.put_n(&bytes, 0, bytes.len())?;
                tli += it.cnt;
            } else {
                w.put_n(&it.raw, 0, it.raw.len())?;
                if it.code == 0xff55 {
                    tli = tli
                        .checked_add(tlm_count(&it.raw))
                        .ok_or(refuse("J2K TLM index"))?;
                }
            }
        }
        for tp in &self.tps {
            w.put_n(&tp.bytes, 0, tp.bytes.len())?;
        }
        w.put_n(&self.tail, 0, self.tail.len())?;
        if w.d.len() > self.limits.max_output_bytes {
            return Err(Error::Limit("J2K codestream bytes"));
        }
        Ok(w.d)
    }
}
