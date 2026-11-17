// PJA MIT: pas/pjaj2kpm.c reversible 5/3 DWT, RCT and sample level shifts.
use crate::{Error, Result};
fn bounded(v: i64) -> Result<i32> {
    if !(-(1 << 30)..=1 << 30).contains(&v) {
        Err(Error::Invalid("J2K transform range"))
    } else {
        Ok(v as i32)
    }
}
fn lift(v: &mut [i64], odd: usize, fwd: bool) -> Result<()> {
    let n = v.len();
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        if odd != 0 {
            if !fwd && v[0] & 1 != 0 {
                return Err(Error::Invalid("J2K non-reversible odd singleton"));
            }
            v[0] = if fwd { v[0] * 2 } else { v[0] >> 1 };
        }
    } else {
        let steps = if fwd {
            [(1 - odd, -1, 0, 1), (odd, 1, 2, 2)]
        } else {
            [(odd, -1, 2, 2), (1 - odd, 1, 0, 1)]
        };
        for (start, sgn, rnd, sh) in steps {
            for i in (start..n).step_by(2) {
                let a = if i == 0 { 1 } else { i - 1 };
                let b = if i + 1 == n { n - 2 } else { i + 1 };
                v[i] += sgn * ((v[a] + v[b] + rnd) >> sh);
            }
        }
    }
    for &x in v.iter() {
        bounded(x)?;
    }
    Ok(())
}
fn line(
    a: &mut [i32],
    step: usize,
    n: usize,
    odd: usize,
    fwd: bool,
    scratch: &mut [i64],
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let v = &mut scratch[..n];
    let (mut li, mut hi) = (0, (n + 1 - odd) / 2);
    if fwd {
        for (i, v) in v.iter_mut().enumerate() {
            *v = a[i * step] as i64;
        }
        lift(v, odd, true)?;
        for (i, &v) in v.iter().enumerate() {
            if (i + odd) & 1 != 0 {
                a[hi * step] = v as i32;
                hi += 1;
            } else {
                a[li * step] = v as i32;
                li += 1;
            }
        }
    } else {
        for (i, v) in v.iter_mut().enumerate() {
            *v = if (i + odd) & 1 != 0 {
                let x = a[hi * step];
                hi += 1;
                x
            } else {
                let x = a[li * step];
                li += 1;
                x
            } as i64;
        }
        lift(v, odd, false)?;
        for (i, &v) in v.iter().enumerate() {
            a[i * step] = v as i32;
        }
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn dwt(
    a: &mut [i32],
    stride: usize,
    w: usize,
    h: usize,
    x0: i64,
    y0: i64,
    levels: usize,
    fwd: bool,
) -> Result<()> {
    if levels > 32 || stride < w {
        return Err(Error::Invalid("J2K DWT layout"));
    }
    if w == 0 || h == 0 {
        return Ok(());
    }
    let span = (h - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(w))
        .ok_or(Error::Limit("J2K DWT span"))?;
    if span > a.len() {
        return Err(Error::Invalid("J2K DWT storage"));
    }
    let x1 = x0
        .checked_add(i64::try_from(w).map_err(|_| Error::Limit("J2K DWT width"))?)
        .ok_or(Error::Limit("J2K DWT endpoint"))?;
    let y1 = y0
        .checked_add(i64::try_from(h).map_err(|_| Error::Limit("J2K DWT height"))?)
        .ok_or(Error::Limit("J2K DWT endpoint"))?;
    let mut scratch = vec![0i64; w.max(h)];
    let ceil = |a: i64, n: usize| {
        let d = 1i64 << n;
        a.div_euclid(d) + i64::from(a.rem_euclid(d) != 0)
    };
    for k in 0..levels {
        let lv = if fwd { k } else { levels - k - 1 };
        let rx0 = ceil(x0, lv);
        let ry0 = ceil(y0, lv);
        let rw = (ceil(x1, lv) - rx0) as usize;
        let rh = (ceil(y1, lv) - ry0) as usize;
        if fwd {
            for x in 0..rw {
                line(
                    &mut a[x..],
                    stride,
                    rh,
                    (ry0 & 1) as usize,
                    true,
                    &mut scratch,
                )?;
            }
            for y in 0..rh {
                line(
                    &mut a[y * stride..],
                    1,
                    rw,
                    (rx0 & 1) as usize,
                    true,
                    &mut scratch,
                )?;
            }
        } else {
            for y in 0..rh {
                line(
                    &mut a[y * stride..],
                    1,
                    rw,
                    (rx0 & 1) as usize,
                    false,
                    &mut scratch,
                )?;
            }
            for x in 0..rw {
                line(
                    &mut a[x..],
                    stride,
                    rh,
                    (ry0 & 1) as usize,
                    false,
                    &mut scratch,
                )?;
            }
        }
    }
    Ok(())
}
pub(crate) fn rct(a: &mut [i32], b: &mut [i32], c: &mut [i32], fwd: bool) -> Result<()> {
    if a.len() != b.len() || b.len() != c.len() {
        return Err(Error::Invalid("J2K RCT lengths"));
    }
    for ((a, b), c) in a.iter_mut().zip(b).zip(c) {
        let (x, y, z) = (*a as i64, *b as i64, *c as i64);
        let (v0, v1, v2) = if fwd {
            ((x + 2 * y + z) >> 2, z - y, x - y)
        } else {
            let g = x - ((y + z) >> 2);
            (z + g, g, y + g)
        };
        *a = bounded(v0)?;
        *b = bounded(v1)?;
        *c = bounded(v2)?;
    }
    Ok(())
}
pub(crate) fn planes_rct(
    a: &mut [u16],
    b: &mut [u16],
    c: &mut [u16],
    depth: usize,
    fwd: bool,
) -> Result<()> {
    if !(1..=15).contains(&depth) || a.len() != b.len() || b.len() != c.len() {
        return Err(Error::Invalid("J2K plane RCT layout"));
    }
    let off = 1i32 << depth;
    for ((a, b), c) in a.iter_mut().zip(b).zip(c) {
        let (x, y, z) = (*a as i32, *b as i32, *c as i32);
        let (v0, v1, v2) = if fwd {
            ((x + 2 * y + z) >> 2, z - y + off, x - y + off)
        } else {
            let u = y - off;
            let v = z - off;
            let g = x - ((u + v) >> 2);
            (v + g, g, u + g)
        };
        if fwd {
            if !(0..off).contains(&x) || !(0..off).contains(&y) || !(0..off).contains(&z) {
                return Err(Error::Invalid("J2K plane RGB range"));
            }
        } else if !(0..off).contains(&v0) || !(0..off).contains(&v1) || !(0..off).contains(&v2) {
            return Err(Error::Invalid("J2K inverse plane RGB range"));
        }
        *a = v0 as u16;
        *b = v1 as u16;
        *c = v2 as u16;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn odd_origins_singletons_and_roundtrip() {
        for x in 0..3 {
            for y in 0..3 {
                for (w, h) in [(1, 1), (1, 9), (8, 1), (9, 7), (8, 8)] {
                    let mut a: Vec<i32> = (0..w * h).map(|i| (i * 31 % 91) as i32 - 45).collect();
                    let original = a.clone();
                    dwt(&mut a, w, w, h, x, y, 3, true).unwrap();
                    dwt(&mut a, w, w, h, x, y, 3, false).unwrap();
                    assert_eq!(a, original);
                }
            }
        }
        assert!(lift(&mut [3], 1, false).is_err());
    }
}
