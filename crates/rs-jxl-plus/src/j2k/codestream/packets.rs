//! Precinct tag trees, packet header parsing and reconstruction (pjaj2kcs).
use super::*;
#[derive(Default)]
struct Entry {
    cb: usize,
    incl: i32,
    first: bool,
    zbp: i32,
    np: usize,
    inc: i32,
    l0: usize,
    ln: usize,
    found: bool,
}
impl State<'_> {
    fn new_nodes(&mut self, cw: usize, ch: usize) -> Result<(usize, usize)> {
        let mut levels = Vec::new();
        let (mut w, mut h, mut nn) = (cw, ch, 0);
        loop {
            levels.push((w, h, nn));
            nn += w * h;
            if w * h <= 1 {
                break;
            }
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        if self.nnd + 2 * nn > 8 * MAXCBS {
            return Err(Error::Limit("J2K tag trees"));
        }
        let base = self.nnd;
        for tree in 0..2 {
            let b = base + tree * nn;
            for (l, &(w, h, off)) in levels.iter().enumerate() {
                for y in 0..h {
                    for x in 0..w {
                        let parent = if l + 1 < levels.len() {
                            let (pw, _, poff) = levels[l + 1];
                            (b + poff + (y / 2) * pw + x / 2) as i32
                        } else {
                            -1
                        };
                        let _ = off;
                        self.ndParent.push(parent);
                        self.ndLow.push(0);
                        self.ndVal.push(INF);
                        self.eLow.push(0);
                        self.eVal.push(INF);
                        self.eKnown.push(false);
                    }
                }
            }
        }
        self.nnd += 2 * nn;
        Ok((base, nn))
    }
    fn make_pb(&mut self, c: usize, r: usize, p: usize) -> Result<()> {
        let rg = self.ress[self.comps[c].res0 + r].clone();
        self.prPb[rg.prBase + p] = self.npb as i32;
        let (pi, pj) = (p % rg.pw, p / rg.pw);
        let sh = usize::from(r > 0);
        for b in 0..rg.nb {
            let bi = rg.band[b];
            let bg = self.bands[bi].clone();
            let sx = ((fp2(rg.x0, rg.ppx) + pi as i64) << rg.ppx) >> sh;
            let sy = ((fp2(rg.y0, rg.ppy) + pj as i64) << rg.ppy) >> sh;
            let ex = sx + (1i64 << (rg.ppx - sh));
            let ey = sy + (1i64 << (rg.ppy - sh));
            let (px0, px1, py0, py1) = (bg.x0.max(sx), bg.x1.min(ex), bg.y0.max(sy), bg.y1.min(ey));
            let (cw, ch, lx, ly) = if px0 >= px1 || py0 >= py1 || bg.ncbx == 0 {
                (0, 0, 0, 0)
            } else {
                let cx = fp2(px0, bg.cbw);
                let cy = fp2(py0, bg.cbh);
                let cw = cp2(px1, bg.cbw) - cx;
                let ch = cp2(py1, bg.cbh) - cy;
                let lx = cx - bg.gx0;
                let ly = cy - bg.gy0;
                if cw > 65536
                    || ch > 65536
                    || lx < 0
                    || ly < 0
                    || lx + cw > bg.ncbx as i64
                    || ly + ch > bg.ncby as i64
                {
                    return Err(refuse("J2K precinct geometry"));
                }
                (cw as usize, ch as usize, lx as usize, ly as usize)
            };
            self.pbBand.push(bi as i32);
            self.pbCw.push(cw as i32);
            self.pbCh.push(ch as i32);
            self.pbLx.push(lx as i32);
            self.pbLy.push(ly as i32);
            let (node, nn) = if cw * ch > 0 {
                let (a, n) = self.new_nodes(cw, ch)?;
                for y in 0..ch {
                    for x in 0..cw {
                        let cb = bg.base + (ly + y) * bg.ncbx + lx + x;
                        self.cbPb[cb] = self.npb as i32;
                        self.cbLeaf[cb] = (y * cw + x) as i32;
                    }
                }
                (a as i32, n as i32)
            } else {
                (-1, 0)
            };
            self.pbNode.push(node);
            self.pbNn.push(nn);
            self.npb += 1;
        }
        Ok(())
    }
    fn pb_of(&mut self, c: usize, r: usize, p: usize) -> Result<usize> {
        let idx = self.ress[self.comps[c].res0 + r].prBase + p;
        if self.prPb[idx] < 0 {
            self.make_pb(c, r, p)?;
        }
        Ok(self.prPb[idx] as usize)
    }
    fn log_node(&mut self, nd: usize) {
        self.lgNode.push(nd as i32);
        self.lgLow.push(self.ndLow[nd]);
        self.lgVal.push(self.ndVal[nd]);
        self.nlg += 1;
    }
    fn undo(&mut self, n: usize) {
        while self.nlg > n {
            self.nlg -= 1;
            let nd = self.lgNode.pop().unwrap() as usize;
            self.ndLow[nd] = self.lgLow.pop().unwrap();
            self.ndVal[nd] = self.lgVal.pop().unwrap();
        }
    }
    fn bit_in(&mut self) -> i32 {
        if self.brCt == 0 {
            self.brBuf = (self.brBuf << 8) & 65535;
            self.brCt = if self.brBuf == 0xff00 { 7 } else { 8 };
            if self.brP < self.brE {
                let b = if self.hdrPacked {
                    self.pkd[self.brP]
                } else {
                    self.cs[self.brP]
                };
                self.brBuf |= b as u32;
                self.brP += 1;
            } else {
                self.brBad = true;
                self.brCt = 0;
                return 0;
            }
        }
        self.brCt -= 1;
        ((self.brBuf >> self.brCt) & 1) as i32
    }
    fn bits_in(&mut self, n: usize) -> u64 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit_in() as u64;
        }
        v
    }
    fn tt_decode(&mut self, base: usize, leaf: usize, thr: i32) -> Result<bool> {
        let mut stack = Vec::new();
        let mut nd = base + leaf;
        while self.ndParent[nd] >= 0 {
            if stack.len() > 63 {
                return Err(refuse("J2K tag tree depth"));
            }
            stack.push(nd);
            nd = self.ndParent[nd] as usize;
        }
        let mut low = 0;
        loop {
            if low > self.ndLow[nd] {
                self.log_node(nd);
                self.ndLow[nd] = low;
            } else {
                low = self.ndLow[nd];
            }
            while low < thr && low < self.ndVal[nd] {
                if self.bit_in() != 0 {
                    self.log_node(nd);
                    self.ndVal[nd] = low;
                } else {
                    low += 1;
                }
                if self.brBad {
                    return Ok(false);
                }
            }
            if self.ndLow[nd] != low {
                self.log_node(nd);
                self.ndLow[nd] = low;
            }
            if let Some(child) = stack.pop() {
                nd = child;
            } else {
                break;
            }
        }
        Ok(self.ndVal[base + leaf] < thr)
    }
    fn next_seg(sty: u32, n: &mut i32, have: &mut i32, mx: &mut i32) {
        *mx = if sty & 4 != 0 {
            1
        } else if sty & 1 != 0 {
            if *n == 0 {
                10
            } else if *mx == 1 || *mx == 10 {
                2
            } else {
                1
            }
        } else {
            109
        };
        *have = 0;
        *n += 1;
    }
    fn pass_pieces(&mut self, cb: usize, mut np: usize) -> Vec<usize> {
        let sty = self.comps[self.bands[self.cbBand[cb] as usize].comp].sty;
        let (mut have, mut mx, mut n) = (self.cbSegHave[cb], self.cbSegMax[cb], self.cbNseg[cb]);
        let mut out = Vec::new();
        while np > 0 {
            if n == 0 || have == mx {
                Self::next_seg(sty, &mut n, &mut have, &mut mx);
            }
            let take = (mx - have).min(np as i32);
            out.push(take as usize);
            have += take;
            np -= take as usize;
        }
        out
    }
    fn piece(
        &mut self,
        cb: usize,
        take: usize,
        pos: &mut usize,
        l0: usize,
        k: usize,
        use_lens: bool,
    ) -> Result<()> {
        if self.npc >= MAXENTRIES * 4 {
            return Err(Error::Limit("J2K segment pieces"));
        }
        self.pcSeg.push(self.cbNseg[cb] - 1);
        self.pcNp.push(take as i32);
        self.pcNext.push(-1);
        let len = if use_lens { self.tLens[l0 + k] } else { 0 };
        self.pcLen.push(len);
        self.pcPos.push(if use_lens { *pos } else { 0 });
        *pos += len;
        if self.cbTail[cb] >= 0 {
            self.pcNext[self.cbTail[cb] as usize] = self.npc as i32;
        } else {
            self.cbHead[cb] = self.npc as i32;
        }
        self.cbTail[cb] = self.npc as i32;
        self.npc += 1;
        self.cbSegHave[cb] += take as i32;
        self.cbPasses[cb] += take as i32;
        Ok(())
    }
    fn derive(
        &mut self,
        cb: usize,
        mut np: usize,
        mut pos: usize,
        l0: usize,
        use_lens: bool,
    ) -> Result<()> {
        let sty = self.comps[self.bands[self.cbBand[cb] as usize].comp].sty;
        let mut k = 0;
        while np > 0 {
            if self.cbNseg[cb] == 0 || self.cbSegHave[cb] == self.cbSegMax[cb] {
                Self::next_seg(
                    sty,
                    &mut self.cbNseg[cb],
                    &mut self.cbSegHave[cb],
                    &mut self.cbSegMax[cb],
                );
            }
            let take = (self.cbSegMax[cb] - self.cbSegHave[cb]).min(np as i32) as usize;
            self.piece(cb, take, &mut pos, l0, k, use_lens)?;
            k += 1;
            np -= take;
        }
        Ok(())
    }
    fn ht_asks(&self, cb: usize, np: usize) -> bool {
        self.cbHt[cb] == 0 && np as i32 - (self.cbPasses[cb] + np as i32 - 1) % 3 >= 1
    }
    fn ht_read_lens(
        &mut self,
        cb: usize,
        np: usize,
        lb: usize,
        end: usize,
    ) -> Option<(Vec<usize>, Vec<usize>, bool)> {
        let mut tot = self.cbPasses[cb] as usize;
        let mut rem = np;
        let mut passes = Vec::new();
        let mut lens = Vec::new();
        let mut found = false;
        if self.cbHt[cb] == 0 {
            let sp = np as i32 - ((tot + np - 1) % 3) as i32;
            let l = if sp >= 1 {
                self.bits_in(lb + flog(sp as usize)) as usize
            } else {
                0
            };
            if sp >= 1 && l != 0 {
                found = true;
                passes.push(sp as usize);
                lens.push(l);
                rem = np - sp as usize;
                tot += sp as usize;
            } else {
                let l = if sp >= 1 {
                    self.bits_in(flog(np) - flog(sp as usize))
                } else {
                    self.bits_in(lb + flog(np))
                };
                if l != 0 || self.brBad {
                    return None;
                }
                passes.push(np);
                lens.push(0);
                return Some((passes, lens, false));
            }
        }
        while rem > 0 {
            let take = if tot % 3 == 0 { 1 } else { 3 - tot % 3 }.min(rem);
            let l = self.bits_in(lb + flog(take));
            if l > end as u64 {
                return None;
            }
            passes.push(take);
            lens.push(l as usize);
            rem -= take;
            tot += take;
        }
        if self.brBad {
            None
        } else {
            Some((passes, lens, found))
        }
    }
    fn derive_ht(
        &mut self,
        cb: usize,
        mut np: usize,
        found: bool,
        mut pos: usize,
        l0: usize,
        use_lens: bool,
    ) -> Result<()> {
        let mut k = 0;
        if self.cbHt[cb] == 0 {
            if self.cbNseg[cb] == 0 {
                self.cbNseg[cb] += 1;
                self.cbSegHave[cb] = 0;
                self.cbSegMax[cb] = (MAXPASSES * 4) as i32;
            }
            let sp = np as i32 - (self.cbPasses[cb] + np as i32 - 1) % 3;
            if sp < 1 || !found {
                self.piece(cb, np, &mut pos, l0, k, use_lens)?;
                return Ok(());
            }
            self.piece(cb, sp as usize, &mut pos, l0, k, use_lens)?;
            k += 1;
            self.cbSegMax[cb] = self.cbSegHave[cb];
            self.cbHt[cb] = 1;
            np -= sp as usize;
        }
        while np > 0 {
            let take;
            if self.cbPasses[cb] % 3 == 0 {
                self.cbNseg[cb] += 1;
                self.cbSegHave[cb] = 0;
                self.cbSegMax[cb] = 1;
                take = 1;
            } else {
                if self.cbPasses[cb] % 3 == 1 || self.cbSegMax[cb] != 2 || self.cbSegHave[cb] >= 2 {
                    self.cbNseg[cb] += 1;
                    self.cbSegHave[cb] = 0;
                    self.cbSegMax[cb] = 2;
                }
                take = (3 - self.cbPasses[cb] % 3) as usize;
            }
            let take = take.min(np);
            self.piece(cb, take, &mut pos, l0, k, use_lens)?;
            k += 1;
            np -= take;
        }
        Ok(())
    }
    fn add_ie(&mut self, cb: usize, flags: i32, np: usize, inc: i32) -> Result<()> {
        if self.nie >= MAXENTRIES {
            return Err(Error::Limit("J2K packet entries"));
        }
        if self.an {
            self.ieTotal += 1;
            if self.ieTotal > self.ieCap {
                return Err(Error::Limit("J2K packet entries per byte"));
            }
        }
        self.ieCb.push(cb as i32);
        self.ieFlags.push(flags);
        self.ieNp.push(np as i32);
        self.ieInc.push(inc);
        self.iePc0.push(self.npc as i32);
        self.iePcN.push(0);
        self.nie += 1;
        Ok(())
    }
    fn pass_pred(&self, cb: usize) -> i64 {
        if self.ht {
            1
        } else {
            (3 * (self.bands[self.cbBand[cb] as usize].mb - self.cbZbp[cb]) - 2 - self.cbPasses[cb])
                .max(1) as i64
        }
    }
    fn packet_record(&mut self, c: usize, r: usize, p: usize, l: usize, flags: i32, nsop: i32) {
        self.pkC.push(c as i32);
        self.pkR.push(r as i32);
        self.pkP.push(p as i32);
        self.pkL.push(l as i32);
        self.pkFlags.push(flags);
        self.pkNsop.push(nsop);
        self.pkIe0.push(self.nie as i32);
        self.pkIeN.push(0);
    }
    fn try_packet(&mut self, c: usize, r: usize, p: usize, l: usize, end: usize) -> Result<bool> {
        let (mut pos, lg0) = (self.curPos, self.nlg);
        let (mut sop, mut eph, mut nsop) = (0, 0, 0);
        if self.par.g.scod & 2 != 0 && pos + 6 <= end && self.cs[pos..pos + 4] == [255, 145, 0, 4] {
            sop = 1;
            nsop = be16(&self.cs, pos + 4)? as i32;
            pos += 6;
        }
        self.brP = if self.hdrPacked { self.pkdPos } else { pos };
        self.brE = if self.hdrPacked { self.pkd.len() } else { end };
        let hstart = self.brP;
        self.brBuf = 0;
        self.brCt = 0;
        self.brBad = false;
        let present = self.bit_in();
        if self.brBad {
            return Ok(false);
        }
        let mut entries = Vec::new();
        self.tLens.clear();
        if present != 0 {
            let nb = self.ress[self.comps[c].res0 + r].nb;
            for b in 0..nb {
                let pb = self.pb_of(c, r, p)? + b;
                let (cw, ch) = (self.pbCw[pb] as usize, self.pbCh[pb] as usize);
                if cw * ch == 0 {
                    continue;
                }
                let bg = self.bands[self.pbBand[pb] as usize].clone();
                for y in 0..ch {
                    for x in 0..cw {
                        let cb = bg.base
                            + (self.pbLy[pb] as usize + y) * bg.ncbx
                            + self.pbLx[pb] as usize
                            + x;
                        let mut e = Entry {
                            cb,
                            l0: self.tLens.len(),
                            ..Entry::default()
                        };
                        if self.cbFirst[cb] < 0 {
                            e.incl = i32::from(self.tt_decode(
                                self.pbNode[pb] as usize,
                                y * cw + x,
                                l as i32 + 1,
                            )?);
                            e.first = e.incl != 0;
                        } else {
                            e.incl = self.bit_in();
                        }
                        if self.brBad {
                            self.undo(lg0);
                            return Ok(false);
                        }
                        if e.incl != 0 {
                            let mut lb;
                            if e.first {
                                if !self.tt_decode(
                                    (self.pbNode[pb] + self.pbNn[pb]) as usize,
                                    y * cw + x,
                                    ZBPLIMIT,
                                )? {
                                    self.undo(lg0);
                                    return Ok(false);
                                }
                                e.zbp = self.ndVal
                                    [(self.pbNode[pb] + self.pbNn[pb]) as usize + y * cw + x];
                                lb = 3;
                            } else {
                                lb = self.cbLb[cb];
                            }
                            let np = if self.bit_in() == 0 {
                                1
                            } else if self.bit_in() == 0 {
                                2
                            } else {
                                let a = self.bits_in(2) as usize;
                                if a != 3 {
                                    a + 3
                                } else {
                                    let a = self.bits_in(5) as usize;
                                    if a != 31 {
                                        a + 6
                                    } else {
                                        self.bits_in(7) as usize + 37
                                    }
                                }
                            };
                            let mut inc = 0;
                            while self.bit_in() == 1 {
                                inc += 1;
                                if inc > 32 || self.brBad {
                                    self.undo(lg0);
                                    return Ok(false);
                                }
                            }
                            lb += inc;
                            if self.brBad
                                || lb > 32
                                || self.cbPasses[cb] + np as i32 > (4 * MAXPASSES) as i32
                            {
                                self.undo(lg0);
                                return Ok(false);
                            }
                            e.np = np;
                            e.inc = inc;
                            let lens = if self.ht {
                                if let Some((_passes, lens, found)) =
                                    self.ht_read_lens(cb, np, lb as usize, end)
                                {
                                    e.found = found;
                                    lens
                                } else {
                                    self.undo(lg0);
                                    return Ok(false);
                                }
                            } else {
                                let pieces = self.pass_pieces(cb, np);
                                let mut lens = Vec::new();
                                for np in pieces {
                                    let len = self.bits_in(lb as usize + flog(np));
                                    if len > end as u64 {
                                        self.undo(lg0);
                                        return Ok(false);
                                    }
                                    lens.push(len as usize);
                                }
                                lens
                            };
                            if self.brBad {
                                self.undo(lg0);
                                return Ok(false);
                            }
                            e.ln = lens.len();
                            self.tLens.extend(lens);
                        }
                        entries.push(e);
                    }
                }
            }
        }
        if self.brBuf & 255 == 255 {
            if self.brP >= self.brE {
                self.undo(lg0);
                return Ok(false);
            }
            self.brP += 1;
        }
        self.brCt = 0;
        let hdr = if self.hdrPacked { &self.pkd } else { &self.cs };
        if self.par.g.scod & 4 != 0
            && self.brP + 2 <= self.brE
            && hdr[self.brP..self.brP + 2] == [255, 146]
        {
            eph = 1;
            self.brP += 2;
        }
        let mut body = if self.hdrPacked { pos } else { self.brP };
        let total = self
            .tLens
            .iter()
            .try_fold(0usize, |a, &n| a.checked_add(n))
            .ok_or(Error::Limit("J2K packet length"))?;
        if total > end.saturating_sub(body) {
            self.undo(lg0);
            return Ok(false);
        }
        let flags = present | (sop << 1) | (eph << 2);
        self.packet_record(c, r, p, l, flags, nsop);
        self.pkHdrPos.push(hstart);
        self.pkHdrLen.push(self.brP - hstart - 2 * eph as usize);
        self.pkStart.push(self.curPos);
        self.pkEnd.push(body + total);
        self.sw[SH_T2].put(flags as u8)?;
        let pred = (self.nsopLast + 1) & 65535;
        if sop != 0 {
            self.sw[SH_T2].s((nsop - pred) as i64)?;
            self.nsopLast = nsop;
        } else {
            self.nsopLast = pred;
        }
        for e in entries {
            let cb = e.cb;
            self.add_ie(
                cb,
                e.incl | (i32::from(self.cbFirst[cb] >= 0) << 1),
                e.np,
                e.inc,
            )?;
            self.sw[SH_T2].put(e.incl as u8)?;
            if e.incl == 0 {
                continue;
            }
            if e.first {
                let b = self.cbBand[cb] as usize;
                self.sw[SH_ZBP].s((e.zbp - self.bandZ[b]) as i64)?;
                self.bandZ[b] = e.zbp;
                self.cbFirst[cb] = l as i32;
                self.cbZbp[cb] = e.zbp;
                self.cbLb[cb] = 3;
            }
            self.sw[SH_T2].s(self.pass_pred(cb) - e.np as i64)?;
            self.cbLb[cb] += e.inc;
            if self.ht {
                if self.ht_asks(cb, e.np) {
                    self.sw[SH_T2].put(u8::from(e.found))?;
                }
                self.derive_ht(cb, e.np, e.found, body, e.l0, true)?;
            } else {
                self.derive(cb, e.np, body, e.l0, true)?;
            }
            self.iePcN[self.nie - 1] = self.npc as i32 - self.iePc0[self.nie - 1];
            body += self.tLens[e.l0..e.l0 + e.ln].iter().sum::<usize>();
        }
        self.pkIeN[self.npk] = self.nie as i32 - self.pkIe0[self.npk];
        self.npk += 1;
        self.nlg = 0;
        self.lgNode.clear();
        self.lgLow.clear();
        self.lgVal.clear();
        if self.hdrPacked {
            self.pkdPos = self.brP;
        }
        self.curPos = body;
        Ok(true)
    }
    pub(super) fn emit_analyse(&mut self, c: usize, r: usize, p: usize, l: usize) -> Result<bool> {
        while self.curTp < self.tileTps.len() {
            let ti = self.tileTps[self.curTp];
            if self.try_packet(c, r, p, l, self.tps[ti].dend)? {
                self.tps[ti].npk += 1;
                return Ok(true);
            }
            self.tps[ti].gap = self.cs[self.curPos..self.tps[ti].dend].to_vec();
            self.curTp += 1;
            if self.curTp < self.tileTps.len() {
                self.curPos = self.tps[self.tileTps[self.curTp]].dstart;
            }
        }
        Ok(false)
    }
    pub(super) fn emit_rebuild(&mut self, c: usize, r: usize, p: usize, l: usize) -> Result<bool> {
        if self.npk >= self.npkTarget {
            return Ok(false);
        }
        let flags = self.sr[SH_T2].get()? as i32;
        if flags > 7 {
            return Err(refuse("J2K packet flags"));
        }
        let pred = (self.nsopLast + 1) & 65535;
        let nsop = if flags & 2 != 0 {
            pred + self.sr[SH_T2].si(-pred as i64, (65535 - pred) as i64)? as i32
        } else {
            pred
        };
        self.nsopLast = nsop;
        self.packet_record(c, r, p, l, flags, nsop);
        if flags & 1 != 0 {
            for b in 0..self.ress[self.comps[c].res0 + r].nb {
                let pb = self.pb_of(c, r, p)? + b;
                let (cw, ch) = (self.pbCw[pb] as usize, self.pbCh[pb] as usize);
                if cw * ch == 0 {
                    continue;
                }
                let bg = self.bands[self.pbBand[pb] as usize].clone();
                for y in 0..ch {
                    for x in 0..cw {
                        let cb = bg.base
                            + (self.pbLy[pb] as usize + y) * bg.ncbx
                            + self.pbLx[pb] as usize
                            + x;
                        let ex = self.sr[SH_T2].get()? as i32;
                        if ex > 1 {
                            return Err(refuse("J2K inclusion"));
                        }
                        self.add_ie(cb, ex | (i32::from(self.cbFirst[cb] >= 0) << 1), 0, 0)?;
                        if ex == 0 {
                            continue;
                        }
                        if self.cbFirst[cb] < 0 {
                            let bb = self.cbBand[cb] as usize;
                            let zp = self.bandZ[bb] as i64
                                + self.sr[SH_ZBP].si(-ZBPLIMIT as i64, ZBPLIMIT as i64)?;
                            if zp < 0 || zp >= ZBPLIMIT as i64 {
                                return Err(refuse("J2K zero bitplanes"));
                            }
                            self.bandZ[bb] = zp as i32;
                            self.cbFirst[cb] = l as i32;
                            self.cbZbp[cb] = zp as i32;
                        }
                        let np = self.pass_pred(cb) - self.sr[SH_T2].si(-1000, 1000)?;
                        if np < 1
                            || np > 164
                            || self.cbPasses[cb] + np as i32 > (4 * MAXPASSES) as i32
                        {
                            return Err(refuse("J2K passes"));
                        }
                        self.ieNp[self.nie - 1] = np as i32;
                        if self.ht {
                            let fd = if self.ht_asks(cb, np as usize) {
                                self.sr[SH_T2].get()?
                            } else {
                                1
                            };
                            if fd > 1 {
                                return Err(refuse("J2K HT passes"));
                            }
                            self.derive_ht(cb, np as usize, fd == 1, 0, 0, false)?;
                        } else {
                            self.derive(cb, np as usize, 0, 0, false)?;
                        }
                        self.iePcN[self.nie - 1] = self.npc as i32 - self.iePc0[self.nie - 1];
                    }
                }
            }
        }
        self.pkIeN[self.npk] = self.nie as i32 - self.pkIe0[self.npk];
        self.npk += 1;
        Ok(true)
    }
    pub(super) fn trees(&mut self) -> Result<()> {
        let mut overrides = Vec::new();
        for pb in 0..self.npb {
            let nn = self.pbNn[pb] as usize;
            if nn == 0 {
                continue;
            }
            let bg = self.bands[self.pbBand[pb] as usize].clone();
            for t in 0..2 {
                let base = self.pbNode[pb] as usize + t * nn;
                self.eVal[base..base + nn].fill(INF);
                for y in 0..self.pbCh[pb] as usize {
                    for x in 0..self.pbCw[pb] as usize {
                        let cb = bg.base
                            + (self.pbLy[pb] as usize + y) * bg.ncbx
                            + self.pbLx[pb] as usize
                            + x;
                        if self.cbFirst[cb] >= 0 {
                            self.eVal[base + y * self.pbCw[pb] as usize + x] = if t == 0 {
                                self.cbFirst[cb]
                            } else {
                                self.cbZbp[cb]
                            };
                        }
                    }
                }
                for i in 0..nn {
                    let k = self.ndParent[base + i];
                    if k >= 0 {
                        let k = k as usize;
                        self.eVal[k] = self.eVal[k].min(self.eVal[base + i]);
                    }
                }
                if self.an {
                    for i in 0..nn {
                        if self.ndVal[base + i] != self.eVal[base + i] {
                            overrides.push((pb, t, i, self.ndVal[base + i]));
                        }
                    }
                }
            }
        }
        if self.an {
            self.sw[SH_TREE].v(overrides.len() as u64)?;
            for (pb, t, i, v) in overrides {
                self.sw[SH_TREE].v(pb as u64)?;
                self.sw[SH_TREE].put(t as u8)?;
                self.sw[SH_TREE].v(i as u64)?;
                self.sw[SH_TREE].v(if v == INF { 0 } else { v as u64 + 1 })?;
                self.eVal[self.pbNode[pb] as usize + t * self.pbNn[pb] as usize + i] = v;
                self.st.tree_overrides += 1;
            }
        } else {
            let n = self.sr[SH_TREE].vi(MAXENTRIES)?;
            if n > 0 && self.npb == 0 {
                return Err(refuse("J2K bad tree"));
            }
            for _ in 0..n {
                let pb = self.sr[SH_TREE].vi(self.npb - 1)?;
                let t = self.sr[SH_TREE].get()? as usize;
                let nn = self.pbNn[pb] as usize;
                if t > 1 || nn == 0 {
                    return Err(refuse("J2K bad tree"));
                }
                let i = self.sr[SH_TREE].vi(nn - 1)?;
                let v = self.sr[SH_TREE].vi(1 << 24)?;
                self.eVal[self.pbNode[pb] as usize + t * nn + i] =
                    if v == 0 { INF } else { v as i32 - 1 };
            }
        }
        self.eLow.fill(0);
        self.eKnown.fill(false);
        Ok(())
    }
}
