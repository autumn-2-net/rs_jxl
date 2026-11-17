// Marker/header parser and decomposition from pjajpeg.pas Read* and ParseWith.
use super::*;
fn u16be(d: &[u8], p: usize) -> usize {
    u16::from_be_bytes([d[p], d[p + 1]]) as usize
}
struct Tables {
    h: [[Option<Huff>; 4]; 2],
    l: [usize; 4],
    u: [usize; 4],
    k: [usize; 4],
}
impl Default for Tables {
    fn default() -> Self {
        Self {
            h: std::array::from_fn(|_| std::array::from_fn(|_| None)),
            l: [0; 4],
            u: [1; 4],
            k: [5; 4],
        }
    }
}
fn sof(d: &[u8], p: usize, last: usize, j: &mut Jpeg, limits: &Limits) -> Result<()> {
    if last - p < 6 {
        return Err(fail("BadSof", p));
    }
    j.precision = d[p] as usize;
    if if j.lossless {
        !(2..=16).contains(&j.precision)
    } else {
        j.precision != 8 && j.precision != 12
    } {
        return Err(Error::Unsupported("JPEG sample precision"));
    }
    j.height = u16be(d, p + 1);
    j.width = u16be(d, p + 3);
    let nf = d[p + 5] as usize;
    if j.height == 0 {
        let mut q = last;
        while q + 6 <= d.len() && !(d[q] == 255 && d[q + 1] == 220 && u16be(d, q + 2) == 4) {
            q += 1;
        }
        if q + 6 > d.len() {
            return Err(fail("BadSof", p));
        }
        j.height = u16be(d, q + 4);
        j.dnl = true;
    }
    if j.width == 0 || j.height == 0 || nf == 0 || nf > 4 || last - p != 6 + 3 * nf {
        return Err(fail("BadSof", p));
    }
    j.hmax = 1;
    j.vmax = 1;
    for i in 0..nf {
        let id = d[p + 6 + 3 * i];
        let h = (d[p + 7 + 3 * i] >> 4) as usize;
        let v = (d[p + 7 + 3 * i] & 15) as usize;
        let tq = d[p + 8 + 3 * i] as usize;
        if h == 0 || h > 4 || v == 0 || v > 4 || tq > 3 || j.components.iter().any(|c| c.id == id) {
            return Err(fail("BadSof", p));
        }
        j.components.push(Component {
            id,
            h,
            v,
            tq,
            ..Component::default()
        });
        j.hmax = j.hmax.max(h);
        j.vmax = j.vmax.max(v);
    }
    let scale = if j.lossless { 1 } else { 8 };
    j.mcusx = j.width.div_ceil(scale * j.hmax);
    j.mcusy = j.height.div_ceil(scale * j.vmax);
    let mut count = 0usize;
    for c in &mut j.components {
        c.bw = j.mcusx * c.h;
        c.bh = j.mcusy * c.v;
        c.bw_real = (j.width * c.h).div_ceil(j.hmax).div_ceil(scale);
        c.bh_real = (j.height * c.v).div_ceil(j.vmax).div_ceil(scale);
        count = count
            .checked_add(c.bw * c.bh)
            .ok_or(Error::Limit("JPEG dimensions"))?;
    }
    if count
        > if j.lossless {
            limits.max_samples
        } else {
            limits.max_blocks
        }
    {
        return Err(Error::Limit("JPEG frame storage"));
    }
    for c in &mut j.components {
        if j.lossless {
            c.samples = zeros(c.bw * c.bh)?;
        } else {
            c.coef = zeros(c.bw * c.bh * 64)?;
        }
    }
    Ok(())
}
fn sos(d: &[u8], p: usize, last: usize, j: &Jpeg, t: &Tables, ri: usize) -> Result<Scan> {
    if last <= p {
        return Err(fail("BadSos", p));
    }
    let n = d[p] as usize;
    if n == 0 || n > 4 || last - p != 4 + 2 * n {
        return Err(fail("BadSos", p));
    }
    let mut s = Scan::default();
    let mut prev = None;
    let mut sum = 0;
    for i in 0..n {
        let ci = j
            .components
            .iter()
            .position(|c| c.id == d[p + 1 + 2 * i])
            .ok_or_else(|| fail("BadSos", p))?;
        if prev.is_some_and(|v| ci <= v) {
            return Err(fail("BadSos", p));
        }
        prev = Some(ci);
        s.comp.push(ci);
        sum += j.components[ci].h * j.components[ci].v;
        s.td[i] = (d[p + 2 + 2 * i] >> 4) as usize;
        s.ta[i] = (d[p + 2 + 2 * i] & 15) as usize;
        if s.td[i] > 3 || s.ta[i] > 3 {
            return Err(fail("BadSos", p));
        }
    }
    if n > 1 && sum > 10 {
        return Err(fail("BadSos", p));
    }
    let q = p + 1 + 2 * n;
    s.ss = d[q] as usize;
    s.se = d[q + 1] as usize;
    s.ah = (d[q + 2] >> 4) as usize;
    s.al = (d[q + 2] & 15) as usize;
    s.ri = ri;
    if j.lossless {
        if s.ss < 1 || s.ss > 7 || s.se != 0 || s.ah != 0 || s.al >= j.precision {
            return Err(fail("BadSos", p));
        }
        let row = if n == 1 {
            j.components[s.comp[0]].bw_real
        } else {
            j.mcusx
        };
        if ri > 0 && ri % row != 0 {
            return Err(Error::Unsupported("JPEG lossless partial-row restart"));
        }
        for i in 0..n {
            s.dc[i] = t.h[0][s.td[i]]
                .clone()
                .ok_or_else(|| fail("MissingTable", p))?;
        }
        return Ok(s);
    }
    if !j.progressive {
        if s.ah != 0 || s.al != 0 {
            return Err(fail("BadSos", p));
        }
        s.ss = 0;
        s.se = 63;
    } else if (s.ss == 0 && s.se != 0)
        || (s.ss > 0 && (s.se < s.ss || s.se > 63 || n != 1))
        || (s.ah != 0 && s.al != s.ah - 1)
        || s.al > 13
    {
        return Err(fail("BadSos", p));
    }
    for i in 0..n {
        if j.arithmetic {
            s.dc_l[i] = t.l[s.td[i]];
            s.dc_u[i] = t.u[s.td[i]];
            s.ac_k[i] = t.k[s.ta[i]];
        } else {
            if s.ss == 0 && s.ah == 0 {
                s.dc[i] = t.h[0][s.td[i]]
                    .clone()
                    .ok_or_else(|| fail("MissingTable", p))?;
            }
            if s.se > 0 {
                s.ac[i] = t.h[1][s.ta[i]]
                    .clone()
                    .ok_or_else(|| fail("MissingTable", p))?;
            }
        }
    }
    if j.progressive && !j.arithmetic && s.se > 0 {
        for k in (0..=14).rev() {
            if s.ac[0].len[k << 4] > 0 {
                s.max_eob = (2 << k) - 1;
                break;
            }
        }
    }
    Ok(s)
}
pub(super) fn parse(d: &[u8], decode: bool, limits: &Limits) -> Result<Jpeg> {
    if d.len() > limits.max_input_bytes {
        return Err(Error::Limit("JPEG input bytes"));
    }
    if d.len() < 4 || d[..2] != [255, 216] {
        return Err(fail("NotJpeg", 0));
    }
    let mut j = Jpeg::default();
    let mut tables = Tables::default();
    let (mut p, mut copied, mut ri) = (2, 0, 0);
    let mut have_frame = false;
    let mut raw = Writer::new(limits.max_input_bytes);
    loop {
        if p >= d.len() {
            return Err(fail("Truncated", p));
        }
        if d[p] != 255 {
            return Err(fail("UnexpectedMarker", p));
        }
        while p < d.len() && d[p] == 255 {
            p += 1;
        }
        let m = *d.get(p).ok_or_else(|| fail("Truncated", p))?;
        p += 1;
        if m == 217 {
            j.eoi = p;
            break;
        }
        if m == 0 || m == 216 || m == 1 {
            return Err(fail("UnexpectedMarker", p - 1));
        }
        if (208..=215).contains(&m) {
            continue;
        }
        if p + 2 > d.len() {
            return Err(fail("Truncated", p));
        }
        let len = u16be(d, p);
        if len < 2 {
            return Err(fail("BadSegment", p));
        }
        let last = p.checked_add(len).ok_or(Error::Limit("JPEG segment"))?;
        if last > d.len() {
            return Err(fail("Truncated", p));
        }
        let first = p + 2;
        match m {
            219 => {
                let mut q = first;
                while q < last {
                    let pq = (d[q] >> 4) as usize;
                    let tq = (d[q] & 15) as usize;
                    q += 1;
                    if pq > 1 || tq > 3 || q + 64 * (pq + 1) > last {
                        return Err(fail("BadDqt", q));
                    }
                    for k in 0..64 {
                        j.qt[tq][k] = if pq == 0 {
                            let v = d[q] as u16;
                            q += 1;
                            v
                        } else {
                            let v = u16be(d, q) as u16;
                            q += 2;
                            v
                        };
                    }
                    j.qpresent[tq] = true;
                }
            }
            196 => {
                let mut q = first;
                while q < last {
                    if q + 17 > last {
                        return Err(fail("BadDht", q));
                    }
                    let tc = (d[q] >> 4) as usize;
                    let th = (d[q] & 15) as usize;
                    q += 1;
                    if tc > 1 || th > 3 {
                        return Err(fail("BadDht", q));
                    }
                    let mut h = Huff::default();
                    h.counts.copy_from_slice(&d[q..q + 16]);
                    q += 16;
                    let n = h.counts.iter().map(|&v| v as usize).sum::<usize>();
                    if n > 256 || q + n > last {
                        return Err(fail("BadDht", q));
                    }
                    h.values = d[q..q + n].to_vec();
                    q += n;
                    h.build()?;
                    tables.h[tc][th] = Some(h);
                }
            }
            221 => {
                if len != 4 {
                    return Err(fail("BadDri", first));
                }
                ri = u16be(d, first);
            }
            220 => {
                if !j.dnl || j.scans.len() != 1 || len != 4 || u16be(d, first) != j.height {
                    return Err(fail("UnexpectedMarker", first));
                }
            }
            192 | 193 | 194 | 195 | 201 | 202 => {
                if have_frame {
                    return Err(fail("RepeatedSof", first));
                }
                j.lossless = m == 195;
                j.progressive = m == 194 || m == 202;
                j.arithmetic = m >= 201;
                sof(d, first, last, &mut j, limits)?;
                have_frame = true;
            }
            204 => {
                if (last - first) % 2 != 0 {
                    return Err(fail("BadDac", first));
                }
                for q in (first..last).step_by(2) {
                    let tc = (d[q] >> 4) as usize;
                    let tb = (d[q] & 15) as usize;
                    let cs = d[q + 1] as usize;
                    if tc > 1 || tb > 3 {
                        return Err(fail("BadDac", q));
                    }
                    if tc == 0 {
                        tables.l[tb] = cs & 15;
                        tables.u[tb] = cs >> 4;
                        if tables.l[tb] > tables.u[tb] {
                            return Err(fail("BadDac", q));
                        }
                    } else {
                        if cs < 1 || cs > 63 {
                            return Err(fail("BadDac", q));
                        }
                        tables.k[tb] = cs;
                    }
                }
            }
            197..=199 | 203 | 205..=207 | 222 | 223 | 247 => {
                return Err(Error::Unsupported("JPEG frame coding process"));
            }
            218 => {
                if !have_frame {
                    return Err(fail("UnexpectedMarker", first));
                }
                if j.scans.len() >= limits.max_scans {
                    return Err(Error::Limit("JPEG scan count"));
                }
                let mut s = sos(d, first, last, &j, &tables, ri)?;
                if j.lossless && !j.scans.is_empty() && s.al != j.scans[0].al {
                    return Err(Error::Unsupported("JPEG per-scan point transforms"));
                }
                raw.copy(&d[copied..last])?;
                copied = last;
                s.cut = raw.data.len();
                if decode {
                    let result = decode_scan(d, last, &mut j, &mut s);
                    copied = match result {
                        Ok(end) => end,
                        Err(error) => {
                            if s.ri == 0 || j.progressive || j.arithmetic {
                                return Err(error);
                            }
                            for &ci in &s.comp {
                                j.components[ci].coef.fill(0);
                                j.components[ci].samples.fill(0);
                            }
                            s.ri = 0;
                            // ParseWith only replaces the original diagnostic
                            // when the no-restart retry succeeds. Preserve the
                            // initial BadRst position if the retry also fails.
                            let end = match decode_scan(d, last, &mut j, &mut s) {
                                Ok(end) => end,
                                Err(_) => return Err(error),
                            };
                            s.ignore_ri = true;
                            end
                        }
                    };
                    if may_wrap_pred(&j, &s) {
                        let rough = lossless::roughness(&j, &s);
                        let keep: Vec<_> = s
                            .comp
                            .iter()
                            .map(|&ci| j.components[ci].samples.clone())
                            .collect();
                        let mut s2 = s.clone();
                        s2.wrap_pred = true;
                        let retry = lossless::decode(d, last, &mut j, &mut s2);
                        if retry.as_ref().is_ok_and(|&end| end == copied)
                            && lossless::roughness(&j, &s2) < rough / 4
                        {
                            s = s2;
                        } else {
                            for (&ci, samples) in s.comp.iter().zip(keep) {
                                j.components[ci].samples = samples;
                            }
                        }
                    }
                    s.data_start = last;
                    s.data_end = copied;
                }
                j.scans.push(s);
                p = copied;
                if p >= d.len() {
                    j.eoi = d.len();
                    break;
                }
                continue;
            }
            _ => {}
        }
        p = last;
    }
    if j.scans.is_empty() {
        return Err(fail("UnexpectedMarker", p));
    }
    raw.copy(&d[copied..])?;
    j.raw = raw.data;
    Ok(j)
}
