//! Marker/cut preservation translated from PJA pjajls.pas, MIT,
//! revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA/NOTICE-PJA.
use super::{entropy::Io, state::State, *};
fn bad(p: usize, msg: &str) -> Error {
    Error::Jpeg(format!("JPEG-LS at {p}: {msg}"))
}
fn u16be(d: &[u8], i: usize) -> i32 {
    u16::from_be_bytes([d[i], d[i + 1]]) as i32
}
fn rst(m: u8) -> bool {
    (0xd0..=0xd7).contains(&m)
}
pub(crate) fn is_jls(d: &[u8]) -> bool {
    if d.len() < 4 || d[..2] != [255, 216] {
        return false;
    }
    let mut p = 2;
    while p + 4 <= d.len() {
        if d[p] != 255 {
            return false;
        }
        while p + 1 < d.len() && d[p + 1] == 255 {
            p += 1;
        }
        if p + 4 > d.len() {
            return false;
        }
        let m = d[p + 1];
        if m == 0xf7 {
            return true;
        }
        if m == 0xda || m == 0xd9 || ((0xc0..=0xcf).contains(&m) && m != 0xc4 && m != 0xcc) {
            return false;
        }
        if rst(m) {
            p += 2;
        } else {
            let len = u16be(d, p + 2) as usize;
            if len < 2 {
                return false;
            }
            p += 2 + len;
        }
    }
    false
}
fn segment_end(data: &[u8], from: usize) -> usize {
    for p in from..data.len().saturating_sub(1) {
        if data[p] == 255 && data[p + 1] >= 128 {
            return p;
        }
    }
    data.len()
}
fn decode_scan(
    data: &[u8],
    first: usize,
    j: &mut Image,
    s: &mut Scan,
    limits: &Limits,
) -> Result<usize> {
    let mut st = State::new(j, s);
    st.runs.clear();
    let rows = scan_rows(j, s);
    let mut p = first;
    let mut y = 0;
    let mut r = 0;
    while y < rows {
        let end = if s.ri > 0 {
            rows.min(y + s.ri as usize)
        } else {
            rows
        };
        let stop = segment_end(data, p);
        let mut io = Io::reader(&data[p..stop]);
        if y > 0 {
            st.reset();
        }
        st.lines(j, s, &mut io, y, end, limits)?;
        io.check()?;
        if end < rows {
            if stop + 1 >= data.len() || data[stop + 1] != 0xd0 + (r & 7) {
                return Err(bad(stop, "restart marker"));
            }
            p = stop + 2;
            r += 1;
        } else {
            p = stop;
        }
        y = end;
    }
    s.runs = st.runs;
    Ok(p)
}
fn append(out: &mut Vec<u8>, bytes: &[u8], limit: usize) -> Result<()> {
    if bytes.len() > limit.saturating_sub(out.len()) {
        return Err(Error::Limit("JPEG-LS output bytes"));
    }
    out.try_reserve(bytes.len())
        .map_err(|_| Error::Limit("JPEG-LS allocation"))?;
    out.extend_from_slice(bytes);
    Ok(())
}
fn encode_scan(j: &mut Image, s: &Scan, limits: &Limits) -> Result<Vec<u8>> {
    let mut st = State::new(j, s);
    let rows = scan_rows(j, s);
    let mut y = 0;
    let mut r = 0u8;
    let mut out = Vec::new();
    while y < rows {
        let end = if s.ri > 0 {
            rows.min(y + s.ri as usize)
        } else {
            rows
        };
        if y > 0 {
            append(&mut out, &[255, 0xd0 + (r & 7)], limits.max_output_bytes)?;
            r = r.wrapping_add(1);
            st.reset();
        }
        let mut io = Io::writer(limits.max_output_bytes.saturating_sub(out.len()));
        st.lines(j, s, &mut io, y, end, limits)?;
        let bytes = io.finish()?;
        append(&mut out, &bytes, limits.max_output_bytes)?;
        y = end;
    }
    if s.near > 0 && st.run_pos != s.runs.len() {
        return Err(Error::Invalid("JPEG-LS unused recorded runs"));
    }
    Ok(out)
}

pub(crate) fn decompose(data: &[u8], limits: &Limits) -> Result<Image> {
    parse(data, true, limits)
}
pub(crate) fn parse_headers(raw: &[u8], limits: &Limits) -> Result<Image> {
    parse(raw, false, limits)
}
fn parse(data: &[u8], decode: bool, limits: &Limits) -> Result<Image> {
    if data.len() > limits.max_input_bytes {
        return Err(Error::Limit("JPEG-LS input bytes"));
    }
    if data.len() < 4 || data[..2] != [255, 216] {
        return Err(bad(0, "SOI"));
    }
    let mut j = Image::default();
    let mut p = 2usize;
    let mut copied = 0;
    let mut ri = 0i32;
    let mut frame = false;
    let mut preset = [0i32; 5];
    let mut coded = [false; 4];
    loop {
        if p >= data.len() {
            return Err(bad(p, "truncated marker"));
        }
        if data[p] != 255 {
            return Err(bad(p, "expected marker"));
        }
        while p < data.len() && data[p] == 255 {
            p += 1;
        }
        if p >= data.len() {
            return Err(bad(p, "truncated marker"));
        }
        let m = data[p];
        p += 1;
        if m == 0xd9 {
            break;
        }
        if rst(m) {
            continue;
        }
        if matches!(m, 0 | 0xd8 | 1) {
            return Err(bad(p - 1, "unexpected marker"));
        }
        if p + 2 > data.len() {
            return Err(bad(p, "truncated segment"));
        }
        let len = u16be(data, p) as usize;
        if len < 2 {
            return Err(bad(p, "segment length"));
        }
        let last = p
            .checked_add(len)
            .filter(|&n| n <= data.len())
            .ok_or_else(|| bad(p, "truncated segment"))?;
        let first = p + 2;
        match m {
            0xf7 => {
                if frame || last - first < 6 {
                    return Err(bad(first, "repeated or short SOF55"));
                }
                j.bits = data[first] as usize;
                j.height = u16be(data, first + 1) as usize;
                j.width = u16be(data, first + 3) as usize;
                let n = data[first + 5] as usize;
                if !(2..=16).contains(&j.bits) || !(1..=4).contains(&n) || last - first != 6 + 3 * n
                {
                    return Err(bad(first, "SOF55 parameters"));
                }
                if j.width == 0 || j.height == 0 {
                    return Err(Error::Unsupported("JPEG-LS zero/oversize dimensions"));
                }
                let mut hm = 1;
                let mut vm = 1;
                for i in 0..n {
                    let id = data[first + 6 + 3 * i];
                    if j.components.iter().any(|c| c.id == id) {
                        return Err(bad(first, "duplicate component"));
                    }
                    let h = (data[first + 7 + 3 * i] >> 4) as usize;
                    let v = (data[first + 7 + 3 * i] & 15) as usize;
                    if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                        return Err(bad(first, "sampling factors"));
                    }
                    hm = hm.max(h);
                    vm = vm.max(v);
                    j.components.push(Component {
                        id,
                        h,
                        v,
                        width: 0,
                        height: 0,
                        samples: Vec::new(),
                    });
                }
                let mut total = 0usize;
                for c in &mut j.components {
                    c.width = (j.width * c.h).div_ceil(hm);
                    c.height = (j.height * c.v).div_ceil(vm);
                    let n = c
                        .width
                        .checked_mul(c.height)
                        .ok_or(Error::Limit("JPEG-LS sample count"))?;
                    total = total
                        .checked_add(n)
                        .ok_or(Error::Limit("JPEG-LS sample count"))?;
                    if total > limits.max_samples {
                        return Err(Error::Limit("JPEG-LS sample count"));
                    }
                    c.samples
                        .try_reserve_exact(n)
                        .map_err(|_| Error::Limit("JPEG-LS samples allocation"))?;
                    c.samples.resize(n, 0);
                }
                frame = true;
            }
            0xf8 => {
                if first == last {
                    return Err(bad(first, "LSE length"));
                }
                match data[first] {
                    1 => {
                        if last - first != 11 {
                            return Err(bad(first, "preset length"));
                        }
                        for (i, v) in preset.iter_mut().enumerate() {
                            *v = u16be(data, first + 1 + 2 * i);
                        }
                    }
                    2 | 3 => {}
                    _ => return Err(Error::Unsupported("JPEG-LS LSE type")),
                }
            }
            0xdd => {
                if !(4..=6).contains(&len) {
                    return Err(bad(first, "DRI length"));
                }
                let mut v = 0u32;
                for &b in &data[first..last] {
                    v = (v << 8) | b as u32;
                }
                ri = i32::try_from(v).map_err(|_| {
                    Error::Unsupported("JPEG-LS restart interval exceeds signed range")
                })?;
            }
            0xc0..=0xc3 | 0xc5..=0xcb | 0xcd..=0xcf => {
                return Err(Error::Unsupported("non JPEG-LS frame"));
            }
            0xda => {
                if !frame || first == last {
                    return Err(bad(first, "SOS without frame"));
                }
                let ns = data[first] as usize;
                if ns < 1 || ns > j.components.len() || last - first != 4 + 2 * ns {
                    return Err(bad(first, "SOS length/components"));
                }
                if j.scans.len() >= limits.max_scans {
                    return Err(Error::Limit("JPEG-LS scans"));
                }
                let mut s = Scan::default();
                for i in 0..ns {
                    let id = data[first + 1 + 2 * i];
                    let c = j
                        .components
                        .iter()
                        .position(|c| c.id == id)
                        .ok_or_else(|| bad(first, "unknown scan component"))?;
                    if s.components.contains(&c) || (decode && coded[c]) {
                        return Err(bad(first, "repeated scan component"));
                    }
                    s.components.push(c);
                }
                for &c in &s.components {
                    coded[c] = true;
                }
                let k = first + 1 + 2 * ns;
                s.near = data[k] as i32;
                s.ilv = data[k + 1] as i32;
                if data[k + 2] != 0 {
                    return Err(Error::Unsupported("JPEG-LS point transform"));
                }
                if s.ilv > 2 || (s.ilv == 0 && ns != 1) || (s.ilv > 0 && ns < 2) {
                    return Err(bad(first, "interleave mode"));
                }
                if s.ilv == 2 && ns < 3 {
                    return Err(Error::Unsupported(
                        "JPEG-LS sample interleave component count",
                    ));
                }
                let group = grouped(&j, &s);
                if group && (s.ilv == 2 || (s.ilv == 1 && ri > 0)) {
                    return Err(Error::Unsupported(
                        "JPEG-LS grouped sample/restart combination",
                    ));
                }
                if s.ilv > 0
                    && !group
                    && s.components.iter().any(|&c| {
                        j.components[c].height != j.height
                            || (s.ilv == 2 && j.components[c].width != j.width)
                    })
                {
                    return Err(Error::Unsupported("JPEG-LS interleave component geometry"));
                }
                s.maxval = if preset[0] != 0 {
                    preset[0]
                } else {
                    (1i32 << j.bits) - 1
                };
                if s.maxval < 1 || s.maxval > (1i32 << j.bits) - 1 {
                    return Err(bad(first, "MAXVAL"));
                }
                if s.near > 255.min(s.maxval / 2) {
                    return Err(bad(first, "NEAR"));
                }
                let (a, b, c) = default_thresholds(s.maxval, s.near);
                s.t1 = if preset[1] != 0 { preset[1] } else { a };
                s.t2 = if preset[2] != 0 { preset[2] } else { b };
                s.t3 = if preset[3] != 0 { preset[3] } else { c };
                s.reset = if preset[4] != 0 { preset[4] } else { 64 };
                if s.t1 < s.near + 1
                    || s.t1 > s.maxval
                    || s.t2 < s.t1
                    || s.t2 > s.maxval
                    || s.t3 < s.t2
                    || s.t3 > s.maxval
                    || s.reset < 3
                    || s.reset > 255.max(s.maxval)
                {
                    return Err(bad(first, "preset values"));
                }
                s.ri = ri;
                append(&mut j.raw, &data[copied..last], limits.max_input_bytes)?;
                s.cut = j.raw.len();
                p = if decode {
                    decode_scan(data, last, &mut j, &mut s, limits)?
                } else {
                    last
                };
                copied = p;
                if decode && p >= data.len() {
                    return Err(bad(p, "missing EOI"));
                }
                j.scans.push(s);
                continue;
            }
            _ => {}
        }
        p = last;
    }
    if j.scans.is_empty() {
        return Err(bad(p, "no scans"));
    }
    append(&mut j.raw, &data[copied..], limits.max_input_bytes)?;
    Ok(j)
}
pub(crate) fn validate(j: &Image, limits: &Limits) -> Result<()> {
    if j.width == 0
        || j.height == 0
        || !(2..=16).contains(&j.bits)
        || j.components.is_empty()
        || j.components.len() > 4
    {
        return Err(Error::Invalid("JPEG-LS image shape"));
    }
    let mut count = 0usize;
    for c in &j.components {
        let n = c
            .width
            .checked_mul(c.height)
            .ok_or(Error::Limit("JPEG-LS samples"))?;
        count = count
            .checked_add(n)
            .ok_or(Error::Limit("JPEG-LS samples"))?;
        if c.width == 0
            || c.height == 0
            || !(1..=4).contains(&c.h)
            || !(1..=4).contains(&c.v)
            || c.samples.len() != n
        {
            return Err(Error::Invalid("JPEG-LS component shape"));
        }
        if c.samples.iter().any(|&x| x as usize >= (1usize << j.bits)) {
            return Err(Error::Invalid("JPEG-LS sample precision"));
        }
    }
    if count > limits.max_samples {
        return Err(Error::Limit("JPEG-LS samples"));
    }
    if j.scans.len() > limits.max_scans {
        return Err(Error::Limit("JPEG-LS scans"));
    }
    for s in &j.scans {
        if s.components.is_empty()
            || s.components.len() > 4
            || s.components.iter().any(|&c| c >= j.components.len())
            || !(0..=2).contains(&s.ilv)
            || s.maxval < 1
            || s.maxval > 65535
            || s.near < 0
            || s.near > 255.min(s.maxval / 2)
            || s.reset < 3
            || s.t1 < s.near + 1
            || s.t2 < s.t1
            || s.t3 < s.t2
            || s.t3 > s.maxval
            || s.ri < 0
        {
            return Err(Error::Invalid("JPEG-LS scan shape"));
        }
        if s.runs.len() > limits.max_runs {
            return Err(Error::Limit("JPEG-LS recorded runs"));
        }
        if (s.ilv == 0 && s.components.len() != 1)
            || (s.ilv == 1 && s.components.len() < 2)
            || (s.ilv == 2 && s.components.len() < 3)
        {
            return Err(Error::Invalid("JPEG-LS scan interleave count"));
        }
        let group = grouped(j, s);
        if group && (s.ilv == 2 || (s.ilv == 1 && s.ri > 0)) {
            return Err(Error::Unsupported(
                "JPEG-LS grouped sample/restart combination",
            ));
        }
        if s.ilv > 0
            && !group
            && s.components.iter().any(|&c| {
                j.components[c].height != j.height
                    || (s.ilv == 2 && j.components[c].width != j.width)
            })
        {
            return Err(Error::Invalid("JPEG-LS scan geometry"));
        }
        if s.components
            .iter()
            .enumerate()
            .any(|(i, c)| s.components[..i].contains(c))
        {
            return Err(Error::Invalid("JPEG-LS repeated scan component"));
        }
    }
    Ok(())
}
pub(crate) fn recompose(j: &Image, limits: &Limits) -> Result<Vec<u8>> {
    validate(j, limits)?;
    let mut work = j.clone();
    let mut out = Vec::new();
    let mut from = 0;
    for s in &j.scans {
        if s.cut < from || s.cut > j.raw.len() {
            return Err(Error::Invalid("JPEG-LS scan cuts"));
        }
        append(&mut out, &j.raw[from..s.cut], limits.max_output_bytes)?;
        from = s.cut;
        let bytes = encode_scan(&mut work, s, limits)?;
        append(&mut out, &bytes, limits.max_output_bytes)?;
    }
    append(&mut out, &j.raw[from..], limits.max_output_bytes)?;
    Ok(out)
}
pub(crate) fn verify(data: &[u8], limits: &Limits) -> Result<Image> {
    let j = decompose(data, limits)?;
    if recompose(&j, limits)? != data {
        return Err(Error::Invalid("JPEG-LS byte reconstruction mismatch"));
    }
    Ok(j)
}
