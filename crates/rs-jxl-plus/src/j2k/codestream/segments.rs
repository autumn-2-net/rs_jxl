// PJA MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Translation of pas/pjaj2kcs.pas DoCb/DoCbHt and segment boundary rules.
use super::super::{ht, tier1};
use super::*;
const HT_CUPS: [u8; 10] = [0, 2, 3, 1, 4, 8, 10, 11, 9, 12];
const HT_REFS: u8 = 5;
const HT_OUTCAP: usize = 65536;
impl State<'_> {
    fn check_cb_bytes(&self, additional: usize) -> Result<()> {
        if additional
            > self
                .limits
                .max_output_bytes
                .saturating_sub(self.cbData.d.len())
        {
            return Err(Error::Limit("J2K reconstructed codeblock bytes"));
        }
        Ok(())
    }
    fn pieces(&self, cb: usize) -> Result<Vec<usize>> {
        let mut next = *self.cbHead.get(cb).ok_or(refuse("J2K codeblock index"))?;
        let mut pieces = Vec::new();
        while next >= 0 {
            let pc = next as usize;
            if pc >= self.pcNext.len()
                || pc >= self.pcSeg.len()
                || pc >= self.pcNp.len()
                || pc >= self.pcLen.len()
                || pieces.len() >= self.pcNext.len()
            {
                return Err(refuse("J2K piece chain"));
            }
            pieces.push(pc);
            next = self.pcNext[pc];
        }
        Ok(pieces)
    }
    fn block_geometry(&self, cb: usize) -> Result<(BandG, usize, usize, usize, usize, u32)> {
        let bi = *self.cbBand.get(cb).ok_or(refuse("J2K codeblock index"))?;
        let bg = self
            .bands
            .get(usize::try_from(bi).map_err(|_| refuse("J2K band index"))?)
            .ok_or(refuse("J2K band index"))?
            .clone();
        if cb < bg.base || bg.ncbx == 0 {
            return Err(refuse("J2K block geometry"));
        }
        let gx = ((cb - bg.base) % bg.ncbx) as i64;
        let gy = ((cb - bg.base) / bg.ncbx) as i64;
        let x0 = bg.x0.max((bg.gx0 + gx) << bg.cbw);
        let x1 = bg.x1.min((bg.gx0 + gx + 1) << bg.cbw);
        let y0 = bg.y0.max((bg.gy0 + gy) << bg.cbh);
        let y1 = bg.y1.min((bg.gy0 + gy + 1) << bg.cbh);
        if x1 <= x0 || y1 <= y0 {
            return Err(refuse("J2K block geometry"));
        }
        let sty = self
            .comps
            .get(bg.comp)
            .ok_or(refuse("J2K component index"))?
            .sty;
        Ok((
            bg.clone(),
            (x0 - bg.x0) as usize,
            (y0 - bg.y0) as usize,
            (x1 - x0) as usize,
            (y1 - y0) as usize,
            sty,
        ))
    }
    fn original_segment_bytes(&self, pieces: &[usize]) -> Result<Vec<u8>> {
        let mut out = BW::default();
        for &pc in pieces {
            let pos = *self.pcPos.get(pc).ok_or(refuse("J2K piece position"))?;
            out.put_n(&self.cs, pos, self.pcLen[pc])?;
        }
        Ok(out.d)
    }
    fn keep_block_bytes(
        &mut self,
        pieces: &[usize],
        bytes: &[u8],
        ht_reason: Option<usize>,
    ) -> Result<()> {
        if self.kind == Kind::PixEncode && ht_reason.is_some() {
            return Err(refuse("pixels: codeblock kept as bytes"));
        }
        self.sw[SH_T1R].put(if ht_reason.is_some() { 255 } else { 1 })?;
        self.st.cb_raw += 1;
        if let Some(reason) = ht_reason {
            self.st.ht_raw[reason] += 1;
        }
        for &pc in pieces {
            self.sw[SH_T1X].v(self.pcLen[pc] as u64)?;
        }
        self.sw[SH_T1X].put_n(bytes, 0, bytes.len())
    }
    fn restore_block_bytes(&mut self, pieces: &[usize]) -> Result<()> {
        let mut total = 0usize;
        for &pc in pieces {
            let n = self.sr[SH_T1X].vi(i32::MAX as usize)?;
            self.pcLen[pc] = n;
            total = total.checked_add(n).ok_or(refuse("J2K codeblock size"))?;
        }
        self.check_cb_bytes(total)?;
        let data = self.sr[SH_T1X].take(total)?;
        self.cbData.put_n(&data, 0, data.len())
    }
    pub(super) fn do_cb(&mut self, cb: usize) -> Result<()> {
        let (bg, x0, y0, w, h, sty) = self.block_geometry(cb)?;
        let npass = usize::try_from(self.cbPasses[cb]).map_err(|_| refuse("J2K passes"))?;
        let nseg = usize::try_from(self.cbNseg[cb]).map_err(|_| refuse("J2K segments"))?;
        if nseg > self.pcSeg.len() + 1 {
            return Err(refuse("J2K segments"));
        }
        let pieces = self.pieces(cb)?;
        let mut segpass = vec![0usize; nseg];
        let mut seglen = vec![0usize; nseg];
        for &pc in &pieces {
            let sg = usize::try_from(self.pcSeg[pc]).map_err(|_| refuse("J2K segment index"))?;
            if sg >= nseg || self.pcNp[pc] < 0 {
                return Err(refuse("J2K segment index"));
            }
            segpass[sg] = segpass[sg]
                .checked_add(self.pcNp[pc] as usize)
                .ok_or(refuse("J2K segment passes"))?;
            if self.an {
                seglen[sg] = seglen[sg]
                    .checked_add(self.pcLen[pc])
                    .filter(|&n| n <= i32::MAX as usize)
                    .ok_or(refuse("J2K segment size"))?;
            }
        }
        let data = if self.an {
            self.original_segment_bytes(&pieces)?
        } else {
            Vec::new()
        };
        let cap = npass
            .checked_mul(2 * w * h + 3 * w * h.div_ceil(4) + 8)
            .and_then(|n| n.checked_add(64))
            .ok_or(refuse("J2K decision capacity"))?;
        let block = tier1::Block {
            width: w,
            height: h,
            orientation: bg.orient,
            style: sty,
            passes: npass,
        };
        let saved_rule = self.sw[SH_T1R].d.len();
        let saved_extra = self.sw[SH_T1X].d.len();
        let saved_stats = self.st.clone();
        let decoded;
        if self.an {
            if npass > MAXPASSES || w > 1024 || h > 1024 || w * h > 4096 {
                if self.kind == Kind::PixEncode {
                    return Err(refuse("pixels: codeblock does not decode"));
                }
                self.keep_block_bytes(&pieces, &data, None)?;
                return Ok(());
            }
            self.tileMaxCap = self.tileMaxCap.max(cap);
            let mut offset = 0;
            let mut segments = Vec::new();
            for k in 0..nseg {
                let end = offset + seglen[k];
                segments.push(tier1::Segment {
                    bytes: data.get(offset..end).ok_or(refuse("J2K segment size"))?,
                    passes: segpass[k],
                });
                offset = end;
            }
            match tier1::decode(block, &segments, cap) {
                Ok(v) => decoded = v,
                Err(_) => {
                    if self.kind == Kind::PixEncode {
                        return Err(refuse("pixels: codeblock does not decode"));
                    }
                    self.keep_block_bytes(&pieces, &data, None)?;
                    return Ok(());
                }
            }
            self.sw[SH_T1R].put(0)?;
        } else {
            self.cbDOff[cb] = self.cbData.d.len();
            if self.sr[SH_T1R].get()? != 0 {
                return self.restore_block_bytes(&pieces);
            }
            if npass > MAXPASSES || w > 1024 || h > 1024 || w * h > 4096 {
                return Err(refuse("J2K bad passes"));
            }
            decoded = if matches!(self.kind, Kind::Decode | Kind::PixDecode) {
                self.classic_values(cb, x0, y0, w, h, sty, npass, cap, None)?
            } else {
                let result = tier1::replay(
                    block,
                    &self.sr[SH_DEC].d,
                    self.dbits,
                    self.sr[SH_DEC].d.len() * 8,
                    cap,
                )?;
                self.dbits += result.used_bits;
                result
            };
        }
        self.st.cbs += 1;
        let mut any_raw = false;
        let mut contexts = tier1::reset_contexts();
        let reset = sty & 2 != 0;
        let (mut pass0, mut data0, mut piece_at) = (0usize, 0usize, 0usize);
        for k in 0..nseg {
            let pass1 = pass0 + segpass[k];
            if pass1 == 0 || pass1 > decoded.pass_ends.len() {
                return Err(refuse("J2K segment passes"));
            }
            let a = if pass0 == 0 {
                0
            } else {
                decoded.pass_ends[pass0 - 1]
            };
            let end = decoded.pass_ends[pass1 - 1];
            let symbols = decoded
                .symbols
                .get(a..end)
                .ok_or(refuse("J2K decision range"))?;
            let pe: Vec<usize> = (pass0..pass1 - 1)
                .map(|p| decoded.pass_ends[p] - a)
                .collect();
            let raw = sty & 1 != 0 && pass0 >= 10 && (pass0 - 1) % 3 < 2;
            let original = if self.an {
                data.get(data0..data0 + seglen[k])
                    .ok_or(refuse("J2K segment bytes"))?
            } else {
                &[]
            };
            self.st.segs += 1;
            let ctx0 = contexts;
            let mut kp = Vec::new();
            let out;
            if raw {
                self.st.raw_segs += 1;
                if self.an {
                    let mut rule = R_RAWB;
                    for pad in 0..=2 {
                        if tier1::raw_pack(symbols, pad)? == original {
                            rule = R_PAD0 + pad;
                            break;
                        }
                    }
                    self.sw[SH_T1R].put(rule)?;
                    if rule == R_RAWB {
                        self.sw[SH_T1X].v(original.len() as u64)?;
                        self.sw[SH_T1X].put_n(original, 0, original.len())?;
                        any_raw = true;
                    }
                    out = original.to_vec();
                    self.st.rule[rule as usize] += 1;
                } else {
                    let rule = self.sr[SH_T1R].get()?;
                    out = if (R_PAD0..=R_PAD2).contains(&rule) {
                        tier1::raw_pack(symbols, rule - R_PAD0)?
                    } else if rule == R_RAWB {
                        let n = self.sr[SH_T1X].vi(i32::MAX as usize)?;
                        self.sr[SH_T1X].take(n)?
                    } else {
                        return Err(refuse("J2K bad raw rule"));
                    };
                    self.st.rule[rule as usize] += 1;
                }
            } else {
                let encoded = tier1::mq_encode(symbols, &pe, reset, contexts)?;
                contexts = encoded.contexts;
                if reset {
                    contexts = tier1::reset_contexts()
                }
                let f = encoded.flush;
                let e = encoded.erterm;
                let committed = encoded.committed;
                kp = encoded.pass_lengths;
                let minf = || tier1::mq_minprefix(&f, &ctx0, symbols, &pe, reset, MINWINDOW);
                let mine = || tier1::mq_minprefix(&e, &ctx0, symbols, &pe, reset, MINWINDOW);
                if self.an {
                    let on = original.len();
                    let okf = f.starts_with(original);
                    let oke = e.starts_with(original);
                    let mut rule = if on == 0 {
                        R_EMPTY
                    } else if okf && on == f.len() {
                        R_FLUSH
                    } else if oke && on == e.len() {
                        R_ERTERM
                    } else if okf && f.len() - on <= MINWINDOW && minf() == on {
                        R_MINF
                    } else if oke && e.len() - on <= MINWINDOW && mine() == on {
                        R_MINE
                    } else if committed <= on
                        && committed <= f.len()
                        && on - committed <= TAILMAX
                        && original[..committed] == f[..committed]
                    {
                        R_TAILF
                    } else {
                        R_RAW
                    };
                    if rule == R_TAILF {
                        let bits = 8 * (on - committed);
                        let same = |reference: &[u8]| {
                            let mut d = 0;
                            while d < bits
                                && bit_at(original, committed * 8 + d)
                                    == bit_at(reference, committed * 8 + d)
                            {
                                d += 1
                            }
                            d
                        };
                        let mut d = same(&f);
                        let dd = same(&e);
                        if dd > d {
                            rule = R_TAILE;
                            d = dd
                        }
                        self.sw[SH_T1R].put(rule)?;
                        self.sw[SH_T1X].v((on - committed) as u64)?;
                        self.sw[SH_T1X].v(d as u64)?;
                        let (mut acc, mut count) = (0u8, 0usize);
                        for bit in d + 1..bits {
                            acc = (acc << 1) | bit_at(original, committed * 8 + bit);
                            count += 1;
                            if count == 8 {
                                self.sw[SH_T1X].put(acc)?;
                                acc = 0;
                                count = 0;
                            }
                        }
                        if count > 0 {
                            self.sw[SH_T1X].put(acc << (8 - count))?;
                        }
                        self.st.tails += 1;
                        if bits > d {
                            self.st.tail_bits += (bits - d - 1) as u64;
                        }
                    } else {
                        self.sw[SH_T1R].put(rule)?;
                        if rule == R_RAW {
                            self.sw[SH_T1X].v(on as u64)?;
                            self.sw[SH_T1X].put_n(original, 0, on)?;
                            any_raw = true;
                        }
                    }
                    out = original.to_vec();
                    self.st.rule[rule as usize] += 1;
                } else {
                    let rule = self.sr[SH_T1R].get()?;
                    out = match rule {
                        R_EMPTY => Vec::new(),
                        R_FLUSH => f.clone(),
                        R_ERTERM => e.clone(),
                        R_MINF => f[..minf()].to_vec(),
                        R_MINE => e[..mine()].to_vec(),
                        R_RAW => {
                            let n = self.sr[SH_T1X].vi(i32::MAX as usize)?;
                            self.sr[SH_T1X].take(n)?
                        }
                        R_TAILF | R_TAILE => {
                            if committed > f.len() {
                                return Err(refuse("J2K bad tail"));
                            }
                            let tail = self.sr[SH_T1X].vi(TAILMAX)?;
                            let n = committed.checked_add(tail).ok_or(refuse("J2K tail size"))?;
                            let bits = tail * 8;
                            let d = self.sr[SH_T1X].vi(bits)?;
                            let reference = if rule == R_TAILF { &f } else { &e };
                            let mut bytes = f[..committed].to_vec();
                            self.check_cb_bytes(n)?;
                            bytes.resize(n, 0);
                            for i in committed..n {
                                bytes[i] = reference.get(i).copied().unwrap_or(0);
                            }
                            if d < bits {
                                bytes[committed + d / 8] ^= 128 >> (d % 8);
                                let (mut current, mut count) = (0u8, 0usize);
                                for bit in d + 1..bits {
                                    if count == 0 {
                                        current = self.sr[SH_T1X].get()?;
                                        count = 8;
                                    }
                                    count -= 1;
                                    let at = committed + bit / 8;
                                    let mask = 128 >> (bit % 8);
                                    bytes[at] = (bytes[at] & !mask)
                                        | if current >> count & 1 != 0 { mask } else { 0 };
                                }
                            }
                            bytes
                        }
                        _ => return Err(refuse("J2K bad MQ rule")),
                    };
                    self.st.rule[rule as usize] += 1;
                }
            }
            // Where packet layers cut this segment, before the final piece.
            let mut prev = 0usize;
            let mut sp = 0usize;
            while piece_at < pieces.len() && self.pcSeg[pieces[piece_at]] == k as i32 {
                let pc = pieces[piece_at];
                sp += self.pcNp[pc] as usize;
                let last =
                    piece_at + 1 == pieces.len() || self.pcSeg[pieces[piece_at + 1]] != k as i32;
                let length;
                if last {
                    length = out.len();
                } else {
                    if sp == 0 || sp > segpass[k] {
                        return Err(refuse("J2K boundary pass"));
                    }
                    let lo = if raw {
                        tier1::raw_pack(&decoded.symbols[a..decoded.pass_ends[pass0 + sp - 1]], 1)?
                            .len()
                            .min(out.len())
                    } else {
                        let committed = kp.get(sp - 1).ok_or(refuse("J2K boundary pass"))?[0];
                        let mut l = (committed + 3).min(out.len());
                        if l > 0 && out[l - 1] == 255 {
                            l -= 1
                        }
                        l
                    };
                    let shortest = || -> Option<usize> {
                        if raw {
                            return None;
                        }
                        let committed = kp.get(sp - 1)?[0];
                        let limit = *pe.get(sp - 1)?;
                        let prefix = symbols.get(..limit)?;
                        ((committed.saturating_sub(3))..=(committed + 8).min(out.len()))
                            .find(|&l| tier1::mq_valid(&out[..l], &ctx0, prefix, &pe, reset))
                    };
                    let reach = || -> Option<usize> {
                        if raw {
                            return None;
                        }
                        let limit = *pe.get(sp - 1)?;
                        tier1::mq_reach(&out, &ctx0, symbols.get(..limit)?, &pe, reset)
                    };
                    let br;
                    if self.an {
                        let actual = prev
                            .checked_add(self.pcLen[pc])
                            .ok_or(refuse("J2K boundary overflow"))?;
                        br = if lo == actual {
                            0
                        } else if !raw && shortest() == Some(actual) {
                            1
                        } else if !raw && reach() == Some(actual) {
                            3
                        } else {
                            2
                        };
                        self.sw[SH_T1R].put(br)?;
                        if br == 2 {
                            let delta = actual as i64 - lo as i64;
                            if delta < -(out.len() as i64) - 1 || delta > out.len() as i64 + 1 {
                                return Err(refuse("J2K boundary out of range"));
                            }
                            self.sw[SH_T1X].s(delta)?;
                        }
                        length = actual;
                    } else {
                        br = self.sr[SH_T1R].get()?;
                        length = match br {
                            0 => lo,
                            1 if !raw => shortest().ok_or(refuse("J2K shortest boundary"))?,
                            3 if !raw => reach().ok_or(refuse("J2K decoder boundary"))?,
                            2 => {
                                let delta = self.sr[SH_T1X]
                                    .si(-(out.len() as i64) - 1, out.len() as i64 + 1)?;
                                usize::try_from(lo as i64 + delta)
                                    .map_err(|_| refuse("J2K boundary"))?
                            }
                            _ => return Err(refuse("J2K bad boundary")),
                        };
                        if length < prev || length > out.len() {
                            return Err(refuse("J2K bad boundary"));
                        }
                    }
                    self.st.bounds += 1;
                    self.st.brule[br as usize] += 1;
                }
                if length < prev {
                    return Err(refuse("J2K decreasing boundary"));
                }
                if self.an {
                    if length - prev != self.pcLen[pc] {
                        return Err(refuse("J2K piece length mismatch"));
                    }
                } else {
                    self.pcLen[pc] = length - prev;
                }
                prev = length;
                piece_at += 1;
            }
            if self.an {
                if out != original {
                    return Err(refuse("J2K segment mismatch"));
                }
            } else {
                self.check_cb_bytes(out.len())?;
                self.cbData.put_n(&out, 0, out.len())?;
            }
            data0 += out.len();
            pass0 = pass1;
        }
        if piece_at != pieces.len() {
            return Err(refuse("J2K unused segment pieces"));
        }
        if !self.an {
            return Ok(());
        }
        if self.kind == Kind::PixEncode {
            self.classic_values(
                cb,
                x0,
                y0,
                w,
                h,
                sty,
                npass,
                decoded.symbols.len(),
                Some(&decoded),
            )?;
        }
        if any_raw {
            self.sw[SH_T1R].d.truncate(saved_rule);
            self.sw[SH_T1X].d.truncate(saved_extra);
            self.st = saved_stats;
            return self.keep_block_bytes(&pieces, &data, None);
        }
        self.st.decisions += decoded.symbols.len() as u64;
        if matches!(self.kind, Kind::Encode | Kind::Dump) {
            self.classic_values(
                cb,
                x0,
                y0,
                w,
                h,
                sty,
                npass,
                decoded.symbols.len(),
                Some(&decoded),
            )?;
        } else if self.kind != Kind::PixEncode {
            let total = self
                .dbits
                .checked_add(decoded.symbols.len())
                .ok_or(refuse("J2K decision bits"))?;
            self.sw[SH_DEC].d.resize(total.div_ceil(8), 0);
            for &symbol in &decoded.symbols {
                if symbol & 1 != 0 {
                    self.sw[SH_DEC].d[self.dbits / 8] |= 128 >> (self.dbits % 8);
                }
                self.dbits += 1;
            }
        }
        Ok(())
    }
    pub(super) fn do_cb_ht(&mut self, cb: usize) -> Result<()> {
        let (bg, x0, y0, w, h, sty) = self.block_geometry(cb)?;
        let causal = sty & 8 != 0;
        let pieces = self.pieces(cb)?;
        if !self.an {
            self.cbDOff[cb] = self.cbData.d.len();
        }
        if self.cbHt[cb] == 0 {
            for &pc in &pieces {
                if self.an && self.pcLen[pc] != 0 {
                    return Err(refuse("J2K HT placeholder bytes"));
                }
                self.pcLen[pc] = 0;
            }
            return Ok(());
        }
        let nseg = usize::try_from(self.cbNseg[cb]).map_err(|_| refuse("J2K HT segments"))?;
        let (mut s0, mut np) = (0usize, 0usize);
        let mut counts = [0usize; 2];
        let mut lens = [0usize; 2];
        let mut last_pc0 = None;
        let mut first_nz = false;
        for &pc in &pieces {
            let seg =
                usize::try_from(self.pcSeg[pc]).map_err(|_| refuse("J2K HT segment index"))?;
            let passes = usize::try_from(self.pcNp[pc]).map_err(|_| refuse("J2K HT passes"))?;
            if seg == 0 {
                s0 = s0.checked_add(passes).ok_or(refuse("J2K HT passes"))?;
                if self.an && last_pc0.is_some_and(|p| self.pcLen[p] != 0) {
                    first_nz = true
                }
                last_pc0 = Some(pc);
            } else if seg == 1 {
                np = np.checked_add(passes).ok_or(refuse("J2K HT passes"))?;
            }
            if seg <= 1 {
                counts[seg] += 1;
                if self.an {
                    lens[seg] = lens[seg]
                        .checked_add(self.pcLen[pc])
                        .filter(|&v| v <= i32::MAX as usize)
                        .ok_or(refuse("J2K HT segment size"))?;
                }
            }
        }
        if s0 < 1 || (s0 - 1) % 3 != 0 {
            return Err(refuse("J2K HT cleanup passes"));
        }
        np += 1;
        let plane = bg.mb - 1 - self.cbZbp[cb] - ((s0 - 1) / 3) as i32;
        let mut q;
        let cup;
        let refinement_rule;
        if self.an {
            let data = self.original_segment_bytes(&pieces)?;
            self.st.cbs += 1;
            let reason = if nseg > 2 {
                Some(2)
            } else if first_nz || counts[1] > 1 {
                Some(3)
            } else if lens[0] > HT_OUTCAP
                || lens[1] > HT_OUTCAP
                || w > 1024
                || h > 1024
                || w * h > 4096
            {
                Some(4)
            } else {
                None
            };
            if let Some(r) = reason {
                return self.keep_block_bytes(&pieces, &data, Some(r));
            }
            let cleanup = data.get(..lens[0]).ok_or(refuse("J2K HT cleanup range"))?;
            let refinement = data
                .get(lens[0]..lens[0] + lens[1])
                .ok_or(refuse("J2K HT refinement range"))?;
            q = match ht::decode(w, h, causal, np, cleanup, refinement) {
                Ok(v) => v,
                Err(_) => return self.keep_block_bytes(&pieces, &data, Some(0)),
            };
            let mut found = None;
            for rule in std::iter::once(self.htLastCup).chain(HT_CUPS) {
                if found.is_some() {
                    break;
                }
                if let Ok(bytes) = ht::encode_cleanup(w, h, np, &q, rule) {
                    if bytes == cleanup {
                        found = Some(rule);
                    }
                }
            }
            cup = match found {
                Some(v) => v,
                None => return self.keep_block_bytes(&pieces, &data, Some(1)),
            };
            self.htLastCup = cup;
            let mut rr = 0;
            if np > 1 {
                rr = HT_REFS;
                for rule in std::iter::once(self.htLastRef).chain(0..HT_REFS) {
                    if let Ok(bytes) = ht::encode_refinement(w, h, causal, np, &q, rule) {
                        if bytes == refinement {
                            rr = rule;
                            break;
                        }
                    }
                }
                if rr < HT_REFS {
                    self.htLastRef = rr;
                }
            }
            refinement_rule = rr;
            let index = HT_CUPS
                .iter()
                .position(|&r| r == cup)
                .ok_or(refuse("J2K cleanup rule index"))?;
            self.sw[SH_T1R].put((index * 8 + rr as usize) as u8)?;
            self.st.ht_cup[cup as usize] += 1;
            self.st.ht_cbs += 1;
            self.st.ht_samples += (w * h) as u64;
            if np > 1 {
                self.st.ht_ref[rr as usize] += 1;
                if rr == HT_REFS {
                    self.sw[SH_T1X].v(refinement.len() as u64)?;
                    self.sw[SH_T1X].put_n(refinement, 0, refinement.len())?;
                }
            }
        } else {
            let rule = self.sr[SH_T1R].get()?;
            if rule == 255 {
                return self.restore_block_bytes(&pieces);
            }
            if nseg > 2 || counts[1] > 1 || w > 1024 || h > 1024 || w * h > 4096 {
                return Err(refuse("J2K bad HT codeblock"));
            }
            if rule as usize >> 3 >= HT_CUPS.len()
                || rule & 7 > HT_REFS
                || (np == 1 && rule & 7 != 0)
            {
                return Err(refuse("J2K bad HT rule"));
            }
            cup = HT_CUPS[rule as usize >> 3];
            refinement_rule = rule & 7;
            q = vec![0; w * h];
        }
        let npm;
        if matches!(self.kind, Kind::PixEncode | Kind::PixDecode) {
            self.ht_values(cb, x0, y0, w, h, causal, np, plane, &mut q)?;
            if self.an {
                return Ok(());
            }
            npm = np;
        } else {
            npm = if np > 1 && refinement_rule == HT_REFS {
                1
            } else {
                np
            };
            if self.an && npm != np {
                for v in &mut q {
                    *v = if *v < 0 { -((-*v) >> 1) } else { *v >> 1 };
                }
            }
            if matches!(self.kind, Kind::Encode | Kind::Decode) {
                self.ht_values(cb, x0, y0, w, h, causal, npm, plane, &mut q)?;
            } else if self.kind == Kind::None {
                if self.an {
                    for &v in &q {
                        self.sw[SH_DEC].s(v as i64)?;
                    }
                } else {
                    for v in &mut q {
                        *v = self.sr[SH_DEC].si(-(1 << 30), 1 << 30)? as i32;
                    }
                }
            } else {
                return Err(refuse("J2K HT link kind"));
            }
            if self.an {
                return Ok(());
            }
        }
        let cleanup = ht::encode_cleanup(w, h, npm, &q, cup)?;
        if cleanup.len() < 2 || cleanup.len() > HT_OUTCAP {
            return Err(refuse("J2K HT cleanup"));
        }
        self.check_cb_bytes(cleanup.len())?;
        self.cbData.put_n(&cleanup, 0, cleanup.len())?;
        let refinement = if np <= 1 {
            Vec::new()
        } else if refinement_rule == HT_REFS {
            let n = self.sr[SH_T1X].vi(HT_OUTCAP)?;
            self.sr[SH_T1X].take(n)?
        } else {
            ht::encode_refinement(w, h, causal, np, &q, refinement_rule)?
        };
        if refinement.len() > HT_OUTCAP {
            return Err(refuse("J2K HT refinement"));
        }
        self.check_cb_bytes(refinement.len())?;
        self.cbData.put_n(&refinement, 0, refinement.len())?;
        for &pc in &pieces {
            self.pcLen[pc] = if self.pcSeg[pc] == 0 {
                if Some(pc) == last_pc0 {
                    cleanup.len()
                } else {
                    0
                }
            } else {
                refinement.len()
            };
        }
        self.st.cbs += 1;
        Ok(())
    }
}
fn bit_at(bytes: &[u8], bit: usize) -> u8 {
    bytes.get(bit / 8).map_or(0, |b| (b >> (7 - bit % 8)) & 1)
}
