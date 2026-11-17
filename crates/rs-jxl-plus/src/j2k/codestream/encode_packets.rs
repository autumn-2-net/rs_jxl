use super::*;
impl State<'_> {
    fn bit_out(&mut self, b: i32) -> Result<()> {
        if self.bwCt == 0 {
            self.bwBuf = (self.bwBuf << 8) & 65535;
            self.bwCt = if self.bwBuf == 0xff00 { 7 } else { 8 };
            self.bw.put((self.bwBuf >> 8) as u8)?;
        }
        self.bwCt -= 1;
        self.bwBuf |= (b as u32) << self.bwCt;
        Ok(())
    }
    fn bits_out(&mut self, v: u64, n: usize) -> Result<()> {
        if n > 63 {
            return Err(refuse("J2K packet length width"));
        }
        for i in (0..n).rev() {
            self.bit_out(((v >> i) & 1) as i32)?;
        }
        Ok(())
    }
    fn tt_encode(&mut self, base: usize, leaf: usize, thr: i32) -> Result<()> {
        let mut nd = base + leaf;
        let mut stack = Vec::new();
        while self.ndParent[nd] >= 0 {
            if stack.len() >= 64 {
                return Err(refuse("J2K tag tree depth"));
            }
            stack.push(nd);
            nd = self.ndParent[nd] as usize;
        }
        let mut low = 0;
        loop {
            if low > self.eLow[nd] {
                self.eLow[nd] = low;
            } else {
                low = self.eLow[nd];
            }
            while low < thr {
                if low >= self.eVal[nd] {
                    if !self.eKnown[nd] {
                        self.bit_out(1)?;
                        self.eKnown[nd] = true;
                    }
                    break;
                }
                self.bit_out(0)?;
                low += 1;
            }
            self.eLow[nd] = low;
            if let Some(n) = stack.pop() {
                nd = n;
            } else {
                break;
            }
        }
        Ok(())
    }
    pub(super) fn pass3(&mut self) -> Result<()> {
        self.cbELb.fill(3);
        self.cbCur.clone_from(&self.cbDOff);
        self.tileOut.d.clear();
        self.tilePk.clear();
        for k in 0..self.npk {
            self.tilePk.push(self.tileOut.d.len());
            if !self.an && self.pkFlags[k] & 2 != 0 {
                let nsop = self.pkNsop[k];
                self.tileOut
                    .put_n(&[255, 145, 0, 4, (nsop >> 8) as u8, nsop as u8], 0, 6)?;
            }
            if !self.hdrPacked {
                self.bw.d.clear();
                self.bwBuf = 0;
                self.bwCt = 8;
                self.bit_out(self.pkFlags[k] & 1)?;
                let e0 = self.pkIe0[k] as usize;
                let en = e0 + self.pkIeN[k] as usize;
                for e in e0..en {
                    let cb = self.ieCb[e] as usize;
                    let pb = self.cbPb[cb] as usize;
                    if self.ieFlags[e] & 2 == 0 {
                        self.tt_encode(
                            self.pbNode[pb] as usize,
                            self.cbLeaf[cb] as usize,
                            self.pkL[k] + 1,
                        )?;
                    } else {
                        self.bit_out(self.ieFlags[e] & 1)?;
                    }
                    if self.ieFlags[e] & 1 == 0 {
                        continue;
                    }
                    if self.ieFlags[e] & 2 == 0 {
                        self.tt_encode(
                            (self.pbNode[pb] + self.pbNn[pb]) as usize,
                            self.cbLeaf[cb] as usize,
                            ZBPLIMIT,
                        )?;
                        self.cbELb[cb] = 3;
                    }
                    let np = self.ieNp[e] as u64;
                    match np {
                        1 => self.bit_out(0)?,
                        2 => self.bits_out(2, 2)?,
                        3..=5 => self.bits_out(12 | (np - 3), 4)?,
                        6..=36 => self.bits_out(0x1e0 | (np - 6), 9)?,
                        37..=164 => self.bits_out(0xff80 | (np - 37), 16)?,
                        _ => return Err(refuse("J2K packet passes")),
                    };
                    let mut lb = self.cbELb[cb];
                    let mut need = 0;
                    let p0 = self.iePc0[e] as usize;
                    let pn = p0 + self.iePcN[e] as usize;
                    for q in p0..pn {
                        let n = flog(self.pcLen[q]) as i32 + 1
                            - (lb + flog(self.pcNp[q] as usize) as i32);
                        need = need.max(n);
                    }
                    let inc = if self.an {
                        let inc = self.ieInc[e];
                        if inc < need {
                            return Err(refuse("J2K internal Lblock"));
                        }
                        self.sw[SH_LB].v((inc - need) as u64)?;
                        if inc != need {
                            self.st.lb_non_minimal += 1;
                        }
                        inc
                    } else {
                        need + self.sr[SH_LB].vi(32)? as i32
                    };
                    if lb + inc > 40 {
                        return Err(refuse("J2K Lblock"));
                    }
                    for _ in 0..inc {
                        self.bit_out(1)?;
                    }
                    self.bit_out(0)?;
                    lb += inc;
                    self.cbELb[cb] = lb;
                    for q in p0..pn {
                        self.bits_out(
                            self.pcLen[q] as u64,
                            lb as usize + flog(self.pcNp[q] as usize),
                        )?;
                    }
                }
                self.bwBuf = (self.bwBuf << 8) & 65535;
                self.bwCt = if self.bwBuf == 0xff00 { 7 } else { 8 };
                self.bw.put((self.bwBuf >> 8) as u8)?;
                if self.bwCt == 7 {
                    self.bwBuf = (self.bwBuf << 8) & 65535;
                    self.bwCt = 8;
                    self.bw.put((self.bwBuf >> 8) as u8)?;
                }
                if self.an {
                    let hl = self.pkHdrLen[k];
                    let same = self.cs.get(self.pkHdrPos[k]..self.pkHdrPos[k] + hl)
                        == Some(self.bw.d.as_slice());
                    self.sw[SH_HDRX].put(u8::from(!same))?;
                    if !same {
                        self.st.hdr_exceptions += 1;
                        self.sw[SH_HDRX].v(hl as u64)?;
                        self.sw[SH_HDRX].put_n(&self.cs, self.pkHdrPos[k], hl)?;
                    }
                } else {
                    if self.sr[SH_HDRX].get()? != 0 {
                        let n = self.sr[SH_HDRX].vi(i32::MAX as usize)?;
                        let data = self.sr[SH_HDRX].take(n)?;
                        self.tileOut.put_n(&data, 0, n)?;
                    } else {
                        self.tileOut.put_n(&self.bw.d, 0, self.bw.d.len())?;
                    }
                    if self.pkFlags[k] & 4 != 0 {
                        self.tileOut.put(255)?;
                        self.tileOut.put(146)?;
                    }
                }
            }
            if !self.an {
                let e0 = self.pkIe0[k] as usize;
                let en = e0 + self.pkIeN[k] as usize;
                for e in e0..en {
                    if self.ieFlags[e] & 1 == 0 {
                        continue;
                    }
                    let cb = self.ieCb[e] as usize;
                    let p0 = self.iePc0[e] as usize;
                    let pn = p0 + self.iePcN[e] as usize;
                    for q in p0..pn {
                        self.tileOut
                            .put_n(&self.cbData.d, self.cbCur[cb], self.pcLen[q])?;
                        self.cbCur[cb] = self.cbCur[cb]
                            .checked_add(self.pcLen[q])
                            .ok_or(refuse("J2K codeblock lengths"))?;
                    }
                }
            }
        }
        self.tilePk.push(self.tileOut.d.len());
        Ok(())
    }
}
