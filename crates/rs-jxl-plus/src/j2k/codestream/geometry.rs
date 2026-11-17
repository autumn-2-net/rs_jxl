use super::*;
impl State<'_> {
    pub(super) fn geometry(&mut self) -> Result<()> {
        macro_rules! clear {($($name:ident),*$(,)?)=>{$(self.$name.clear();)*};}
        clear!(
            pkC, pkR, pkP, pkL, pkFlags, pkNsop, pkIe0, pkIeN, pkHdrPos, pkHdrLen, pkStart, pkEnd,
            ieCb, ieFlags, ieNp, ieInc, iePc0, iePcN, pcSeg, pcNp, pcNext, pcLen, pcPos, lgNode,
            lgLow, lgVal
        );
        let (p, q) = (self.tileNo % self.nxt, self.tileNo / self.nxt);
        self.tx0 = (self.xtosiz + p as i64 * self.xtsiz).max(self.xosiz);
        self.ty0 = (self.ytosiz + q as i64 * self.ytsiz).max(self.yosiz);
        self.tx1 = (self.xtosiz + (p + 1) as i64 * self.xtsiz).min(self.xsiz);
        self.ty1 = (self.ytosiz + (q + 1) as i64 * self.ytsiz).min(self.ysiz);
        if self.tx0 >= self.tx1 || self.ty0 >= self.ty1 {
            return Err(refuse("J2K empty tile"));
        }
        self.comps.clear();
        self.ress.clear();
        self.bands.clear();
        let (mut cbs, mut precs) = (0usize, 0usize);
        for c in 0..self.ncomp {
            let cod = &self.par.cod[c];
            let qnt = &self.par.qnt[c];
            let cg = CompG {
                dx: self.cdx[c],
                dy: self.cdy[c],
                x0: ceil(self.tx0, self.cdx[c] as i64),
                y0: ceil(self.ty0, self.cdy[c] as i64),
                x1: ceil(self.tx1, self.cdx[c] as i64),
                y1: ceil(self.ty1, self.cdy[c] as i64),
                nres: cod.nl + 1,
                res0: self.ress.len(),
                sty: cod.sty,
            };
            for r in 0..=cod.nl {
                let lv = cod.nl - r;
                let mut rg = ResG {
                    x0: cp2(cg.x0, lv),
                    y0: cp2(cg.y0, lv),
                    x1: cp2(cg.x1, lv),
                    y1: cp2(cg.y1, lv),
                    ppx: (cod.pp[r] & 15) as usize,
                    ppy: (cod.pp[r] >> 4) as usize,
                    ..ResG::default()
                };
                rg.pw = if rg.x0 == rg.x1 {
                    0
                } else {
                    (cp2(rg.x1, rg.ppx) - fp2(rg.x0, rg.ppx)) as usize
                };
                rg.ph = if rg.y0 == rg.y1 {
                    0
                } else {
                    (cp2(rg.y1, rg.ppy) - fp2(rg.y0, rg.ppy)) as usize
                };
                let np = rg
                    .pw
                    .checked_mul(rg.ph)
                    .ok_or(Error::Limit("J2K precincts"))?;
                if np > MAXPRECINCTS.saturating_sub(precs) {
                    return Err(Error::Limit("J2K precincts"));
                }
                rg.prBase = precs;
                precs += np;
                rg.nb = if r == 0 { 1 } else { 3 };
                let cbw = cod.xcb.min(rg.ppx - usize::from(r > 0));
                let cbh = cod.ycb.min(rg.ppy - usize::from(r > 0));
                for b in 0..rg.nb {
                    let mut bg = BandG {
                        comp: c,
                        res: r,
                        orient: if r == 0 { 0 } else { b + 1 },
                        cbw,
                        cbh,
                        base: cbs,
                        ..BandG::default()
                    };
                    let ebi;
                    if r == 0 {
                        bg.x0 = rg.x0;
                        bg.y0 = rg.y0;
                        bg.x1 = rg.x1;
                        bg.y1 = rg.y1;
                        ebi = 0;
                    } else {
                        let n = cod.nl - r + 1;
                        let xo = [1, 0, 1][b];
                        let yo = [0, 1, 1][b];
                        bg.x0 = cp2(cg.x0 - (xo << n - 1), n);
                        bg.y0 = cp2(cg.y0 - (yo << n - 1), n);
                        bg.x1 = cp2(cg.x1 - (xo << n - 1), n);
                        bg.y1 = cp2(cg.y1 - (yo << n - 1), n);
                        ebi = 1 + 3 * (r - 1) + b;
                    }
                    if bg.x0 < bg.x1 && bg.y0 < bg.y1 {
                        bg.gx0 = fp2(bg.x0, cbw);
                        bg.gy0 = fp2(bg.y0, cbh);
                        bg.ncbx = (cp2(bg.x1, cbw) - bg.gx0) as usize;
                        bg.ncby = (cp2(bg.y1, cbh) - bg.gy0) as usize;
                        let n = bg
                            .ncbx
                            .checked_mul(bg.ncby)
                            .ok_or(Error::Limit("J2K codeblocks"))?;
                        if n > MAXCBS.saturating_sub(cbs) {
                            return Err(Error::Limit("J2K codeblocks"));
                        }
                        cbs += n;
                    }
                    bg.mb = if qnt.style == 1 {
                        qnt.guard
                            + if ebi == 0 {
                                *qnt.ex.first().unwrap_or(&0)
                            } else {
                                (qnt.ex.first().copied().unwrap_or(0) - ((ebi - 1) / 3) as i32)
                                    .max(0)
                            }
                            - 1
                    } else if ebi < qnt.ex.len() {
                        qnt.guard + qnt.ex[ebi] - 1
                    } else {
                        qnt.guard + 8
                    };
                    rg.band[b] = self.bands.len();
                    self.bands.push(bg);
                }
                self.ress.push(rg);
            }
            self.comps.push(cg);
        }
        self.ncb = cbs;
        self.nprec = precs;
        macro_rules! cb{($($name:ident=$value:expr),*$(,)?)=>{$(self.$name=filled(cbs,$value)?;)*};}
        cb!(
            cbBand = 0,
            cbFirst = -1,
            cbZbp = 0,
            cbPasses = 0,
            cbNseg = 0,
            cbSegHave = 0,
            cbSegMax = 0,
            cbLb = 3,
            cbHead = -1,
            cbTail = -1,
            cbPb = -1,
            cbLeaf = 0,
            cbELb = 0,
            cbHt = 0,
            cbDOff = 0usize,
            cbCur = 0usize
        );
        for (b, bg) in self.bands.iter().enumerate() {
            self.cbBand[bg.base..bg.base + bg.ncbx * bg.ncby].fill(b as i32);
        }
        self.prNext = filled(precs, 0)?;
        self.prPb = filled(precs, -1)?;
        self.npb = 0;
        self.nnd = 0;
        self.npk = 0;
        self.nie = 0;
        self.npc = 0;
        self.nlg = 0;
        self.pbCw.clear();
        self.pbCh.clear();
        self.pbLx.clear();
        self.pbLy.clear();
        self.pbBand.clear();
        self.pbNode.clear();
        self.pbNn.clear();
        self.ndParent.clear();
        self.ndLow.clear();
        self.ndVal.clear();
        self.eLow.clear();
        self.eVal.clear();
        self.eKnown.clear();
        Ok(())
    }
    fn step(&mut self) -> Result<()> {
        self.iterations += 1;
        if self.iterations > MAXITER {
            return Err(Error::Limit("J2K progression steps"));
        }
        Ok(())
    }
    fn prec_at(&self, c: usize, r: usize, x: i64, y: i64) -> Option<usize> {
        let cg = &self.comps[c];
        let rg = &self.ress[cg.res0 + r];
        let lv = cg.nres - 1 - r;
        let rpx = rg.ppx + lv;
        let rpy = rg.ppy + lv;
        if rpx >= 31 || rpy >= 31 {
            return None;
        }
        let dx = (cg.dx as i64) << rpx;
        let dy = (cg.dy as i64) << rpy;
        if !(y % dy == 0 || (y == self.ty0 && ((rg.y0 << lv) % (1i64 << rpy)) != 0))
            || !(x % dx == 0 || (x == self.tx0 && ((rg.x0 << lv) % (1i64 << rpx)) != 0))
            || rg.pw == 0
            || rg.ph == 0
            || rg.x0 == rg.x1
            || rg.y0 == rg.y1
        {
            return None;
        }
        let px = fp2(ceil(x, (cg.dx as i64) << lv), rg.ppx) - fp2(rg.x0, rg.ppx);
        let py = fp2(ceil(y, (cg.dy as i64) << lv), rg.ppy) - fp2(rg.y0, rg.ppy);
        if px < 0 || py < 0 || px >= rg.pw as i64 || py >= rg.ph as i64 {
            return None;
        }
        Some(px as usize + py as usize * rg.pw)
    }
    fn min_step(&self, start: usize, end: usize) -> (i64, i64) {
        let (mut x, mut y) = (0, 0);
        for c in start..end {
            let cg = &self.comps[c];
            for r in 0..cg.nres {
                let rg = &self.ress[cg.res0 + r];
                let lv = cg.nres - 1 - r;
                if rg.ppx + lv < 40 {
                    let v = (cg.dx as i64) << (rg.ppx + lv);
                    if x == 0 || v < x {
                        x = v;
                    }
                }
                if rg.ppy + lv < 40 {
                    let v = (cg.dy as i64) << (rg.ppy + lv);
                    if y == 0 || v < y {
                        y = v;
                    }
                }
            }
        }
        (x, y)
    }
    fn layers(
        &mut self,
        c: usize,
        r: usize,
        p: usize,
        ly: usize,
        emit: fn(&mut Self, usize, usize, usize, usize) -> Result<bool>,
    ) -> Result<bool> {
        let idx = self.ress[self.comps[c].res0 + r].prBase + p;
        for l in self.prNext[idx] as usize..ly {
            self.step()?;
            self.prNext[idx] = l as i32 + 1;
            if !emit(self, c, r, p, l)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(super) fn iterate(
        &mut self,
        emit: fn(&mut Self, usize, usize, usize, usize) -> Result<bool>,
    ) -> Result<()> {
        let maxres = self.comps.iter().map(|c| c.nres).max().unwrap_or(0);
        let pocs = if self.par.pocs.is_empty() {
            vec![Poc {
                rs: 0,
                cs: 0,
                lye: self.par.g.layers,
                re: maxres,
                ce: self.ncomp,
                prg: self.par.g.prog,
            }]
        } else {
            self.par.pocs.clone()
        };
        let (dx, dy) = self.min_step(0, self.ncomp);
        for poc in pocs {
            let (rs, re, cs, ce, ly) = (
                poc.rs,
                poc.re.min(maxres),
                poc.cs,
                poc.ce.min(self.ncomp),
                poc.lye.min(self.par.g.layers),
            );
            match poc.prg {
                0 | 1 => {
                    let outer = if poc.prg == 0 {
                        ly
                    } else {
                        re.saturating_sub(rs)
                    };
                    let inner = if poc.prg == 0 {
                        re.saturating_sub(rs)
                    } else {
                        ly
                    };
                    for a in 0..outer {
                        for b in 0..inner {
                            let (l, r) = if poc.prg == 0 {
                                (a, rs + b)
                            } else {
                                (b, rs + a)
                            };
                            for c in cs..ce {
                                self.step()?;
                                if r >= self.comps[c].nres {
                                    continue;
                                }
                                let rg = &self.ress[self.comps[c].res0 + r];
                                let (np, base) = (rg.pw * rg.ph, rg.prBase);
                                for p in 0..np {
                                    self.step()?;
                                    let idx = base + p;
                                    if self.prNext[idx] != l as i32 {
                                        continue;
                                    }
                                    self.prNext[idx] = l as i32 + 1;
                                    if !emit(self, c, r, p, l)? {
                                        return Ok(());
                                    }
                                }
                            }
                        }
                    }
                }
                2 => {
                    if dx <= 0 || dy <= 0 {
                        return Err(refuse("J2K progression step"));
                    }
                    for r in rs..re {
                        let mut y = self.ty0;
                        while y < self.ty1 {
                            let mut x = self.tx0;
                            while x < self.tx1 {
                                for c in cs..ce {
                                    self.step()?;
                                    if r >= self.comps[c].nres {
                                        continue;
                                    }
                                    if let Some(p) = self.prec_at(c, r, x, y) {
                                        if !self.layers(c, r, p, ly, emit)? {
                                            return Ok(());
                                        }
                                    }
                                }
                                x += dx - x % dx;
                            }
                            y += dy - y % dy;
                        }
                    }
                }
                3 => {
                    if dx <= 0 || dy <= 0 {
                        return Err(refuse("J2K progression step"));
                    }
                    let mut y = self.ty0;
                    while y < self.ty1 {
                        let mut x = self.tx0;
                        while x < self.tx1 {
                            for c in cs..ce {
                                for r in rs..re.min(self.comps[c].nres) {
                                    self.step()?;
                                    if let Some(p) = self.prec_at(c, r, x, y) {
                                        if !self.layers(c, r, p, ly, emit)? {
                                            return Ok(());
                                        }
                                    }
                                }
                            }
                            x += dx - x % dx;
                        }
                        y += dy - y % dy;
                    }
                }
                4 => {
                    for c in cs..ce {
                        let (dx, dy) = self.min_step(c, c + 1);
                        if dx <= 0 || dy <= 0 {
                            return Err(refuse("J2K progression step"));
                        }
                        let mut y = self.ty0;
                        while y < self.ty1 {
                            let mut x = self.tx0;
                            while x < self.tx1 {
                                for r in rs..re.min(self.comps[c].nres) {
                                    self.step()?;
                                    if let Some(p) = self.prec_at(c, r, x, y) {
                                        if !self.layers(c, r, p, ly, emit)? {
                                            return Ok(());
                                        }
                                    }
                                }
                                x += dx - x % dx;
                            }
                            y += dy - y % dy;
                        }
                    }
                }
                _ => return Err(refuse("J2K progression")),
            }
        }
        Ok(())
    }
}
