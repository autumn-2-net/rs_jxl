// PJA MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Source: pas/pjaj2kpx.pas. See NOTICE-PJA.
//! JPEG2000 pixel mode2/HT pixel mode4: reversible coefficient/plane bridge.
use super::{
    codestream::{self, BlockInfo, Kind, Limits, Link, PixTile},
    model::Band,
    pixel_model::{self, Plane},
    pixel_tier1,
    tier1::{Block, Passes},
    transform,
};
use crate::{
    Error, Result,
    arith::{Decoder, Encoder},
    raw_model,
};
const MAX_SAMPLES: usize = 1 << 27;
const TILE_CAP: usize = 1 << 26;
const MAX_WIDTH: usize = 1 << 20;
const FREE: usize = 1 << 20;
const PER_BYTE: usize = 1024;
fn invalid() -> Error {
    Error::Invalid("J2K pixel reconstruction")
}
fn be16(d: &[u8], p: usize) -> Result<usize> {
    Ok(u16::from_be_bytes(d.get(p..p + 2).ok_or_else(invalid)?.try_into().unwrap()) as usize)
}
fn be32(d: &[u8], p: usize) -> Result<i64> {
    Ok(u32::from_be_bytes(d.get(p..p + 4).ok_or_else(invalid)?.try_into().unwrap()) as i64)
}
fn ceil(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}
fn cs(a: i64, n: usize) -> i64 {
    (a + (1i64 << n) - 1) >> n
}
#[derive(Clone)]
struct Siz {
    ix0: i64,
    iy0: i64,
    ix1: i64,
    iy1: i64,
    dx: Vec<usize>,
    dy: Vec<usize>,
    depth: Vec<usize>,
    signed: Vec<bool>,
    pw: Vec<usize>,
    ph: Vec<usize>,
    ox: Vec<i64>,
    oy: Vec<i64>,
    total: usize,
}
fn siz(raw: &[u8], len: usize, limit: &Limits) -> Result<Siz> {
    if raw.len() < 42 || raw.get(..2) != Some(&[255, 81]) {
        return Err(invalid());
    }
    let (ix1, iy1, ix0, iy0, tw, th) = (
        be32(raw, 6)?,
        be32(raw, 10)?,
        be32(raw, 14)?,
        be32(raw, 18)?,
        be32(raw, 22)?,
        be32(raw, 26)?,
    );
    let n = be16(raw, 38)?;
    if n == 0
        || n > 16384
        || raw.len() < 40 + 3 * n
        || ix1 <= ix0
        || iy1 <= iy0
        || tw == 0
        || th == 0
    {
        return Err(invalid());
    }
    let mut z = Siz {
        ix0,
        iy0,
        ix1,
        iy1,
        dx: Vec::new(),
        dy: Vec::new(),
        depth: Vec::new(),
        signed: Vec::new(),
        pw: Vec::new(),
        ph: Vec::new(),
        ox: Vec::new(),
        oy: Vec::new(),
        total: 0,
    };
    for c in 0..n {
        let depth = (raw[40 + 3 * c] & 127) as usize + 1;
        let (dx, dy) = (raw[41 + 3 * c] as usize, raw[42 + 3 * c] as usize);
        if dx == 0 || dy == 0 || depth > 16 {
            return Err(Error::Unsupported("J2K pixel depth or sampling"));
        }
        let (ox, oy) = (ceil(ix0, dx as i64), ceil(iy0, dy as i64));
        let pw = (ceil(ix1, dx as i64) - ox) as usize;
        let ph = (ceil(iy1, dy as i64) - oy) as usize;
        if depth > 1 && pw > MAX_WIDTH {
            return Err(Error::Limit("J2K pixel row width"));
        }
        let samples = pw.checked_mul(ph).ok_or(Error::Limit("J2K pixels"))?;
        z.total = z
            .total
            .checked_add(samples)
            .filter(|&v| v <= MAX_SAMPLES)
            .ok_or(Error::Limit("J2K pixels"))?;
        z.dx.push(dx);
        z.dy.push(dy);
        z.depth.push(depth);
        z.signed.push(raw[40 + 3 * c] & 128 != 0);
        z.pw.push(pw);
        z.ph.push(ph);
        z.ox.push(ox);
        z.oy.push(oy);
    }
    if z.total > FREE && z.total > len.saturating_mul(PER_BYTE) {
        return Err(Error::Limit("J2K pixels per encoded byte"));
    }
    if limit.max_samples > 0 && super::siz_samples(raw).is_some_and(|n| n > limit.max_samples) {
        return Err(Error::Limit("J2K sample cap"));
    }
    Ok(z)
}
pub(crate) fn siz_supported(raw: &[u8], len: usize, limits: &Limits) -> Result<()> {
    siz(raw, len, limits).map(|_| ())
}
fn allocated<T: Clone>(n: usize, value: T) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| Error::Limit("J2K pixel allocation"))?;
    out.resize(n, value);
    Ok(out)
}
struct Worker {
    tile: PixTile,
    coef: Vec<Vec<i32>>,
    boff: Vec<(usize, usize)>,
    bw: Vec<usize>,
    bh: Vec<usize>,
}
struct Pixels {
    encoding: bool,
    siz: Siz,
    planes: Vec<Plane>,
    worker: Option<Worker>,
    no_mct: bool,
    truncated: bool,
}
fn color_i32(coef: &mut [Vec<i32>], fwd: bool) -> Result<()> {
    if coef.len() < 3 {
        return Err(invalid());
    }
    let (a, rest) = coef.split_at_mut(1);
    let (b, c) = rest.split_at_mut(1);
    transform::rct(&mut a[0], &mut b[0], &mut c[0], fwd)
}
fn color_planes(planes: &mut [Plane], depth: usize, fwd: bool) -> Result<()> {
    if planes.len() < 3 {
        return Err(invalid());
    }
    let (a, rest) = planes.split_at_mut(1);
    let (b, c) = rest.split_at_mut(1);
    transform::planes_rct(
        &mut a[0].samples,
        &mut b[0].samples,
        &mut c[0].samples,
        depth,
        fwd,
    )
}
impl Pixels {
    fn new(encoding: bool, siz: Siz) -> Result<Self> {
        let mut planes = Vec::new();
        for c in 0..siz.depth.len() {
            planes.push(Plane {
                width: siz.pw[c],
                height: siz.ph[c],
                depth: siz.depth[c],
                samples: allocated(siz.pw[c] * siz.ph[c], 0u16)?,
            });
        }
        Ok(Self {
            encoding,
            siz,
            planes,
            worker: None,
            no_mct: false,
            truncated: false,
        })
    }
    fn color_ok(&self) -> bool {
        let z = &self.siz;
        z.depth.len() >= 3
            && z.depth[0] <= 15
            && z.depth[1] == z.depth[0]
            && z.depth[2] == z.depth[0]
            && z.pw[1] == z.pw[0]
            && z.pw[2] == z.pw[0]
            && z.ph[1] == z.ph[0]
            && z.ph[2] == z.ph[0]
            && z.pw[0] * z.ph[0] > 0
    }
    fn hb(&self) -> u32 {
        let n: usize = self
            .siz
            .depth
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == 1)
            .map(|(c, _)| self.siz.pw[c] * self.siz.ph[c])
            .sum();
        let mut hb = 12;
        while hb < 22 && (1usize << hb) < n {
            hb += 1;
        }
        hb
    }
    fn code_planes<C: crate::arith::BitCoder>(
        &mut self,
        cd: &mut C,
        hb: u32,
        color: bool,
    ) -> Result<()> {
        if color {
            if !self.color_ok() {
                return Err(invalid());
            }
            self.planes[1].depth = self.siz.depth[0] + 1;
            self.planes[2].depth = self.siz.depth[0] + 1;
            if self.encoding {
                color_planes(&mut self.planes, self.siz.depth[0], true)?;
            }
        }
        if self
            .planes
            .iter()
            .any(|p| p.depth > 1 && p.width > MAX_WIDTH)
        {
            return Err(Error::Limit("J2K pixel row width"));
        }
        pixel_model::code(cd, &mut self.planes, hb, MAX_SAMPLES)?;
        if color && !self.encoding {
            color_planes(&mut self.planes, self.siz.depth[0], false)?;
            self.planes[1].depth = self.siz.depth[1];
            self.planes[2].depth = self.siz.depth[2];
        }
        Ok(())
    }
    fn block_location(&self, info: &BlockInfo) -> Result<(usize, usize, usize, usize)> {
        let wk = self.worker.as_ref().ok_or_else(invalid)?;
        let band = wk.tile.bands.get(info.band).ok_or_else(invalid)?;
        if info.width == 0
            || info.height == 0
            || info
                .x0
                .checked_add(info.width)
                .is_none_or(|v| v > wk.bw[info.band])
            || info
                .y0
                .checked_add(info.height)
                .is_none_or(|v| v > wk.bh[info.band])
        {
            return Err(invalid());
        }
        let (ox, oy) = wk.boff[info.band];
        Ok((
            band.comp,
            ox + info.x0,
            oy + info.y0,
            (wk.tile.comps[band.comp].x1 - wk.tile.comps[band.comp].x0) as usize,
        ))
    }
}
impl Link for Pixels {
    fn kind(&self) -> Kind {
        if self.encoding {
            Kind::PixEncode
        } else {
            Kind::PixDecode
        }
    }
    fn start_slice(&mut self, _k: usize, _hb: usize) -> Result<()> {
        Ok(())
    }
    fn end_slice(&mut self, _k: usize) -> Result<()> {
        Ok(())
    }
    fn tile(&mut self, t: &PixTile, _bands: Vec<Band>, _maps: usize) -> Result<()> {
        let z = &self.siz;
        if t.rgn {
            return Err(Error::Unsupported("J2K pixel RGN"));
        }
        if !t.mct {
            self.no_mct = true;
        }
        if t.comps.len() != z.depth.len()
            || (t.ix0, t.iy0, t.ix1, t.iy1) != (z.ix0, z.iy0, z.ix1, z.iy1)
        {
            return Err(invalid());
        }
        if t.mct {
            if t.comps.len() < 3 {
                return Err(invalid());
            }
            let a = &t.comps[0];
            if t.comps[1..3]
                .iter()
                .any(|b| (b.x0, b.y0, b.x1, b.y1) != (a.x0, a.y0, a.x1, a.y1))
            {
                return Err(Error::Unsupported("J2K pixel color transform layout"));
            }
        }
        let mut coef = Vec::new();
        let mut total = 0usize;
        for (c, comp) in t.comps.iter().enumerate() {
            if !comp.ok {
                return Err(Error::Unsupported("J2K pixel irreversible component"));
            }
            if comp.depth != z.depth[c]
                || comp.signed != z.signed[c]
                || comp.dx != z.dx[c]
                || comp.dy != z.dy[c]
                || !(1..=33).contains(&comp.nres)
                || comp.x0 < z.ox[c]
                || comp.y0 < z.oy[c]
                || comp.x1 > z.ox[c] + z.pw[c] as i64
                || comp.y1 > z.oy[c] + z.ph[c] as i64
                || comp.x1 < comp.x0
                || comp.y1 < comp.y0
            {
                return Err(invalid());
            }
            let w = (comp.x1 - comp.x0) as usize;
            let h = (comp.y1 - comp.y0) as usize;
            total = total
                .checked_add(w.checked_mul(h).ok_or(Error::Limit("J2K pixel tile"))?)
                .filter(|&n| n <= TILE_CAP)
                .ok_or(Error::Limit("J2K pixel tile"))?;
            let mut buffer = allocated(w * h, 0i32)?;
            if !self.encoding {
                let off = 1i32 << (comp.depth - 1);
                for y in 0..h {
                    let row = (comp.y0 - z.oy[c]) as usize + y;
                    let col = (comp.x0 - z.ox[c]) as usize;
                    for x in 0..w {
                        let v = self.planes[c].samples[row * z.pw[c] + col + x] as i32;
                        if v >= 1 << comp.depth {
                            return Err(invalid());
                        }
                        buffer[y * w + x] = v - off;
                    }
                }
            }
            coef.push(buffer);
        }
        let (mut boff, mut bw, mut bh) = (Vec::new(), Vec::new(), Vec::new());
        for b in &t.bands {
            let comp = t.comps.get(b.comp).ok_or_else(invalid)?;
            let nl = comp.nres - 1;
            if b.res > nl || ((b.res == 0) != (b.orient == 0)) || b.orient > 3 {
                return Err(invalid());
            }
            let lv = nl - b.res;
            let rw = cs(comp.x1, lv) - cs(comp.x0, lv);
            let rh = cs(comp.y1, lv) - cs(comp.y0, lv);
            let (sn, sv) = if b.res == 0 {
                (rw, rh)
            } else {
                (
                    cs(comp.x1, lv + 1) - cs(comp.x0, lv + 1),
                    cs(comp.y1, lv + 1) - cs(comp.y0, lv + 1),
                )
            };
            let (ox, ww) = if matches!(b.orient, 1 | 3) {
                (sn, rw - sn)
            } else {
                (0, sn)
            };
            let (oy, hh) = if matches!(b.orient, 2 | 3) {
                (sv, rh - sv)
            } else {
                (0, sv)
            };
            if ww != b.x1 - b.x0 || hh != b.y1 - b.y0 || ww < 0 || hh < 0 {
                return Err(invalid());
            }
            boff.push((ox as usize, oy as usize));
            bw.push(ww as usize);
            bh.push(hh as usize);
        }
        if !self.encoding {
            if t.mct && !coef[0].is_empty() {
                color_i32(&mut coef, true)?;
            }
            for (c, comp) in t.comps.iter().enumerate() {
                let (w, h) = ((comp.x1 - comp.x0) as usize, (comp.y1 - comp.y0) as usize);
                if w > 0 && h > 0 {
                    transform::dwt(&mut coef[c], w, w, h, comp.x0, comp.y0, comp.nres - 1, true)?;
                }
            }
        }
        self.worker = Some(Worker {
            tile: t.clone(),
            coef,
            boff,
            bw,
            bh,
        });
        Ok(())
    }
    fn block(
        &mut self,
        info: &BlockInfo,
        symbols: &mut Vec<u8>,
        pass_ends: &mut Vec<usize>,
        cap: usize,
    ) -> Result<usize> {
        let (c, ox, oy, stride) = self.block_location(info)?;
        let wk = self.worker.as_mut().unwrap();
        let p = wk.tile.bands[info.band].mb - info.zbp;
        if !(1..=30).contains(&p) {
            return Err(invalid());
        }
        let block = Block {
            width: info.width,
            height: info.height,
            orientation: info.orientation,
            style: info.style,
            passes: info.passes,
        };
        let mut values = Vec::new();
        for y in 0..info.height {
            let start = (oy + y) * stride + ox;
            values.extend_from_slice(&wk.coef[c][start..start + info.width]);
        }
        if self.encoding {
            let input = Passes {
                symbols: symbols.clone(),
                pass_ends: pass_ends.clone(),
                used_bits: 0,
            };
            values = pixel_tier1::coefficients(block, p as u32, &input)?;
            let back =
                pixel_tier1::symbols(block, p as u32, &values, symbols.len().saturating_add(16))?;
            if back.symbols != *symbols {
                return Err(invalid());
            }
            if info.passes < pixel_tier1::full_passes(block, p as u32, &values)? {
                self.truncated = true;
                return Err(Error::Unsupported("J2K pixel rate-truncated codeblock"));
            }
            for y in 0..info.height {
                let start = (oy + y) * stride + ox;
                wk.coef[c][start..start + info.width]
                    .copy_from_slice(&values[y * info.width..(y + 1) * info.width]);
            }
            Ok(symbols.len())
        } else {
            let passes = pixel_tier1::symbols(block, p as u32, &values, cap)?;
            *symbols = passes.symbols;
            *pass_ends = passes.pass_ends;
            Ok(symbols.len())
        }
    }
    fn ht_block(
        &mut self,
        info: &BlockInfo,
        np: usize,
        plane: i32,
        q: &mut Vec<i32>,
    ) -> Result<()> {
        let (c, ox, oy, stride) = self.block_location(info)?;
        if !(1..=3).contains(&np) || np == 2 || plane - (np > 1) as i32 != 0 {
            if self.encoding {
                self.truncated = true;
            }
            return Err(Error::Unsupported("J2K HT pixel incomplete bitplanes"));
        }
        if q.len() != info.width * info.height {
            return Err(invalid());
        }
        let wk = self.worker.as_mut().unwrap();
        for y in 0..info.height {
            for x in 0..info.width {
                let at = (oy + y) * stride + ox + x;
                if self.encoding {
                    wk.coef[c][at] = q[y * info.width + x];
                } else {
                    let v = wk.coef[c][at];
                    if !(-(1 << 29)..=1 << 29).contains(&v) {
                        return Err(invalid());
                    }
                    q[y * info.width + x] = v;
                }
            }
        }
        Ok(())
    }
    fn tile_end(&mut self) -> Result<()> {
        if !self.encoding {
            return Ok(());
        }
        let wk = self.worker.as_mut().ok_or_else(invalid)?;
        for (c, comp) in wk.tile.comps.iter().enumerate() {
            let (w, h) = ((comp.x1 - comp.x0) as usize, (comp.y1 - comp.y0) as usize);
            if w > 0 && h > 0 {
                transform::dwt(
                    &mut wk.coef[c],
                    w,
                    w,
                    h,
                    comp.x0,
                    comp.y0,
                    comp.nres - 1,
                    false,
                )?;
            }
        }
        if wk.tile.mct && !wk.coef[0].is_empty() {
            color_i32(&mut wk.coef, false)?;
        }
        for (c, comp) in wk.tile.comps.iter().enumerate() {
            let (w, h) = ((comp.x1 - comp.x0) as usize, (comp.y1 - comp.y0) as usize);
            let off = 1i64 << (comp.depth - 1);
            for y in 0..h {
                let row = (comp.y0 - self.siz.oy[c]) as usize + y;
                let col = (comp.x0 - self.siz.ox[c]) as usize;
                for x in 0..w {
                    let v = wk.coef[c][y * w + x] as i64 + off;
                    if !(0..1i64 << comp.depth).contains(&v) {
                        return Err(Error::Invalid("J2K pixel sample range"));
                    }
                    self.planes[c].samples[row * self.siz.pw[c] + col + x] = v as u16;
                }
            }
        }
        Ok(())
    }
}
fn side_bound(len: usize, limits: &Limits) -> usize {
    len.saturating_add(len / 4)
        .saturating_add(65536)
        .min(limits.max_side_bytes)
}
const INTERNAL_LAYOUT: u8 = 1;
fn encoded_len(side: usize, pixels: usize, limit: usize) -> Result<usize> {
    7usize
        .checked_add(side)
        .and_then(|n| n.checked_add(pixels))
        .filter(|&v| v < i32::MAX as usize && v <= limit)
        .ok_or(Error::Limit("J2K pixel encoded bytes"))
}
fn le32(input: &[u8], p: usize) -> Result<usize> {
    Ok(u32::from_le_bytes(input.get(p..p + 4).ok_or_else(invalid)?.try_into().unwrap()) as usize)
}
/// Internal experimental pixel layout 1, not the original PJA mode2/4 body.
/// layout:u8, side_size:u32le, side, context_bits:u8, color:u8, pixel_data.
/// The returned stream is always independently reconstructed and compared.
pub(crate) fn encode(input: &[u8], ht: bool, limits: &Limits) -> Result<Vec<u8>> {
    if input.len() > limits.max_input_bytes || input.len() > limits.max_output_bytes {
        return Err(Error::Limit("J2K pixel input bytes"));
    }
    if input.len() < 46 || !input.starts_with(&[255, 79, 255, 81]) {
        return Err(invalid());
    }
    let siz_len = be16(input, 4)?.checked_add(2).ok_or_else(invalid)?;
    let raw = input.get(2..2 + siz_len).ok_or_else(invalid)?;
    let siz = siz(raw, input.len(), limits)?;
    let mut pixels = Pixels::new(true, siz)?;
    let (blob, _stats) = codestream::analyze_with(input, ht, limits, &mut pixels)?;
    if blob.len() > side_bound(input.len(), limits) {
        return Err(Error::Limit("J2K pixel side bytes"));
    }
    let mut side_coder = Encoder::new(limits.max_output_bytes);
    let mut side = blob.clone();
    raw_model::code(&mut side_coder, &mut side, side_bound(input.len(), limits))?;
    let side = side_coder.finish()?;
    let hb = pixels.hb();
    let color = !pixels.no_mct
        && pixels.color_ok()
        && (pixels.siz.depth[0] > 1 || pixels.siz.pw[0] <= MAX_WIDTH);
    let mut coder = Encoder::new(limits.max_output_bytes);
    pixels.code_planes(&mut coder, hb, color)?;
    let coded = coder.finish()?;
    let total = encoded_len(side.len(), coded.len(), limits.max_output_bytes)?;
    let mut stream = Vec::new();
    stream
        .try_reserve_exact(total)
        .map_err(|_| Error::Limit("J2K pixel allocation"))?;
    stream.push(INTERNAL_LAYOUT);
    stream.extend(
        u32::try_from(side.len())
            .map_err(|_| invalid())?
            .to_le_bytes(),
    );
    stream.extend(&side);
    stream.push(hb as u8);
    stream.push(color as u8);
    stream.extend(coded);
    let verify_limits = Limits {
        max_input_bytes: limits.max_output_bytes,
        ..limits.clone()
    };
    let reconstructed = decode(&stream, input.len(), ht, &verify_limits)?;
    if reconstructed != input {
        return Err(Error::Invalid("J2K pixel verification"));
    }
    Ok(stream)
}
pub(crate) fn decode(stream: &[u8], len: usize, ht: bool, limits: &Limits) -> Result<Vec<u8>> {
    if stream.len() > limits.max_input_bytes || len > limits.max_output_bytes {
        return Err(Error::Limit("J2K pixel stream bytes"));
    }
    // Five fixed header bytes, two plane flags and two arithmetic tails.
    if stream.len() < 15 {
        return Err(invalid());
    }
    if stream[0] != INTERNAL_LAYOUT {
        return Err(Error::Unsupported("J2K pixel layout"));
    }
    let side = le32(stream, 1)?;
    if side < 4 {
        return Err(invalid());
    }
    let pos = 5usize
        .checked_add(side)
        .filter(|&p| p <= stream.len() - 6)
        .ok_or_else(invalid)?;
    let hb = stream[pos] as u32;
    let color = stream[pos + 1];
    if !(12..=22).contains(&hb) || color > 1 {
        return Err(invalid());
    }
    let mut cd = Decoder::new(&stream[5..pos]);
    let mut blob = Vec::new();
    raw_model::code(&mut cd, &mut blob, side_bound(len, limits))?;
    cd.finish_check()?;
    let raw = codestream::blob_siz(&blob)?;
    let siz = siz(&raw, len, limits)?;
    let mut pixels = Pixels::new(false, siz)?;
    let mut cd = Decoder::new(&stream[pos + 2..]);
    pixels.code_planes(&mut cd, hb, color != 0)?;
    cd.finish_check()?;
    let cs = codestream::rebuild_with(&blob, ht, limits, &mut pixels)?;
    if cs.len() != len {
        return Err(Error::Invalid("J2K pixel reconstructed length"));
    }
    Ok(cs)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hash_free_layout_header_and_arithmetic_bounds() {
        // Five header bytes + four side bytes + two flags + four data bytes.
        let mut d = vec![INTERNAL_LAYOUT];
        d.extend(4u32.to_le_bytes());
        d.extend([0; 4]);
        d.extend([12, 0]);
        d.extend([0; 4]);
        assert_eq!(d.len(), 15);
        let limits = Limits::default();
        for n in 0..d.len() {
            assert!(
                decode(&d[..n], 0, false, &limits).is_err(),
                "accepted truncated layout at {n}"
            );
        }
        let mut old = d.clone();
        old[0] = 0;
        assert!(matches!(
            decode(&old, 0, false, &limits),
            Err(Error::Unsupported("J2K pixel layout"))
        ));
        let mut short = d;
        short[1..5].copy_from_slice(&3u32.to_le_bytes());
        assert!(decode(&short, 0, false, &limits).is_err());
    }
    fn siz_marker(depth: u8, width: u32, height: u32) -> Vec<u8> {
        let mut raw = vec![0; 43];
        raw[..4].copy_from_slice(&[255, 81, 0, 41]);
        raw[6..10].copy_from_slice(&width.to_be_bytes());
        raw[10..14].copy_from_slice(&height.to_be_bytes());
        raw[22..26].copy_from_slice(&width.to_be_bytes());
        raw[26..30].copy_from_slice(&height.to_be_bytes());
        raw[38..40].copy_from_slice(&1u16.to_be_bytes());
        raw[40] = depth - 1;
        raw[41] = 1;
        raw[42] = 1;
        raw
    }
    #[test]
    fn siz_budgets_before_plane_allocation() {
        let limits = Limits::default();
        assert!(siz_supported(&siz_marker(8, 16, 16), 100, &limits).is_ok());
        assert!(siz_supported(&siz_marker(17, 16, 16), 100, &limits).is_err());
        assert!(siz_supported(&siz_marker(8, MAX_WIDTH as u32 + 1, 1), 1 << 20, &limits).is_err());
        assert!(siz_supported(&siz_marker(1, 65536, 65536), 100, &limits).is_err());
        let mut cap = limits;
        cap.max_samples = 255;
        assert!(siz_supported(&siz_marker(8, 16, 16), 100, &cap).is_err());
    }
    #[test]
    fn malformed_mode_headers() {
        for bytes in [&[][..], &[0; 10][..], &[255; 11][..], &[0; 11][..]] {
            assert!(decode(bytes, 10, false, &Limits::default()).is_err());
        }
    }
}

#[cfg(test)]
mod linked_tests {
    use super::super::codestream::{PixBand, PixComp};
    use super::*;
    fn linked_case(ht: bool, mct: bool) -> Result<()> {
        let n = if mct { 3 } else { 1 };
        let (w, h) = (7usize, 5usize);
        let z = Siz {
            ix0: 1,
            iy0: 3,
            ix1: 8,
            iy1: 8,
            dx: vec![1; n],
            dy: vec![1; n],
            depth: vec![8; n],
            signed: vec![false; n],
            pw: vec![w; n],
            ph: vec![h; n],
            ox: vec![1; n],
            oy: vec![3; n],
            total: w * h * n,
        };
        let original: Vec<Vec<u16>> = (0..n)
            .map(|c| {
                (0..w * h)
                    .map(|i| ((i * 31 + i / 3 * 17 + c * 41) % 256) as u16)
                    .collect()
            })
            .collect();
        let mut coeff: Vec<Vec<i32>> = original
            .iter()
            .map(|p| p.iter().map(|&v| v as i32 - 128).collect())
            .collect();
        if mct {
            color_i32(&mut coeff, true)?;
        }
        for a in &mut coeff {
            transform::dwt(a, w, w, h, 1, 3, 1, true)?;
        }
        let comps = (0..n)
            .map(|_| PixComp {
                dx: 1,
                dy: 1,
                nres: 2,
                depth: 8,
                signed: false,
                ok: true,
                x0: 1,
                y0: 3,
                x1: 8,
                y1: 8,
            })
            .collect();
        let mut bands = Vec::new();
        for c in 0..n {
            for (res, orient, bw, bh) in [(0, 0, 3, 2), (1, 1, 4, 2), (1, 2, 3, 3), (1, 3, 4, 3)] {
                bands.push(PixBand {
                    comp: c,
                    res,
                    orient,
                    mb: 12,
                    x0: 0,
                    y0: 0,
                    x1: bw,
                    y1: bh,
                });
            }
        }
        let tile = PixTile {
            tile: 0,
            mct,
            rgn: false,
            ix0: 1,
            iy0: 3,
            ix1: 8,
            iy1: 8,
            comps,
            bands,
        };
        let mut px = Pixels::new(true, z.clone())?;
        px.tile(&tile, Vec::new(), 0)?;
        let mut fixtures = Vec::new();
        for (b, band) in tile.bands.iter().enumerate() {
            let (ox, oy) = px.worker.as_ref().unwrap().boff[b];
            let ww = px.worker.as_ref().unwrap().bw[b];
            let hh = px.worker.as_ref().unwrap().bh[b];
            let mut values = Vec::new();
            for y in 0..hh {
                values.extend_from_slice(
                    &coeff[band.comp][(oy + y) * w + ox..(oy + y) * w + ox + ww],
                );
            }
            let info = BlockInfo {
                band: b,
                x0: 0,
                y0: 0,
                width: ww,
                height: hh,
                style: 9,
                passes: 34,
                zbp: 0,
                orientation: band.orient,
            };
            if ht {
                let expected = values.clone();
                px.ht_block(&info, 1, 0, &mut values)?;
                fixtures.push((info, Vec::new(), Vec::new(), expected));
            } else {
                let block = Block {
                    width: ww,
                    height: hh,
                    orientation: band.orient,
                    style: 9,
                    passes: 34,
                };
                let generated = pixel_tier1::symbols(block, 12, &values, 20000)?;
                let mut symbols = generated.symbols;
                let mut ends = generated.pass_ends;
                px.block(&info, &mut symbols, &mut ends, 20000)?;
                fixtures.push((info, symbols, ends, values));
            }
        }
        px.tile_end()?;
        if !ht && !mct {
            // Reuse this successful linked fixture's actual plane encoding to
            // exercise the production aggregate output-budget calculation.
            let mut coder = Encoder::new(1 << 20);
            px.code_planes(&mut coder, px.hb(), false)?;
            let coded = coder.finish()?;
            let side = Encoder::new(4).finish()?;
            let exact = 7 + side.len() + coded.len();
            assert_eq!(encoded_len(side.len(), coded.len(), exact)?, exact);
            assert!(matches!(
                encoded_len(side.len(), coded.len(), exact - 1),
                Err(Error::Limit("J2K pixel encoded bytes"))
            ));
        }
        for (c, p) in px.planes.iter().enumerate() {
            assert_eq!(p.samples, original[c], "ht{ht} mct{mct} component{c}");
        }
        let mut back = Pixels::new(false, z)?;
        back.planes = px.planes;
        back.tile(&tile, Vec::new(), 0)?;
        for (info, expected_symbols, expected_ends, expected_values) in fixtures {
            if ht {
                let mut values = vec![0; info.width * info.height];
                back.ht_block(&info, 1, 0, &mut values)?;
                assert_eq!(values, expected_values);
            } else {
                let (mut symbols, mut ends) = (Vec::new(), Vec::new());
                back.block(&info, &mut symbols, &mut ends, 20000)?;
                assert_eq!(symbols, expected_symbols);
                assert_eq!(ends, expected_ends);
            }
        }
        Ok(())
    }
    #[test]
    fn coefficient_pixel_bridge_odd_origins_and_rct() {
        for ht in [false, true] {
            for mct in [false, true] {
                linked_case(ht, mct).unwrap();
            }
        }
    }
}
