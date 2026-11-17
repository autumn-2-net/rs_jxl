// pjajpeg.pas Annex H predictor, sample traversal and Huffman differences.
use super::*;
pub(super) fn predict(psv: usize, ra: i32, rb: i32, rc: i32, wrap: bool) -> i32 {
    match psv {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => {
            ra + (if wrap {
                (rb - rc) as i16 as i32
            } else {
                rb - rc
            } >> 1)
        }
        6 => {
            rb + (if wrap {
                (ra - rc) as i16 as i32
            } else {
                ra - rc
            } >> 1)
        }
        _ => (ra + rb) >> 1,
    }
}
fn prediction(
    precision: usize,
    s: &Scan,
    sm: &[u16],
    x: usize,
    y: usize,
    y0: usize,
    w: usize,
) -> i32 {
    if y == y0 {
        if x == 0 {
            1 << (precision - s.al - 1)
        } else {
            sm[y * w + x - 1] as i32
        }
    } else if x == 0 {
        sm[(y - 1) * w] as i32
    } else {
        predict(
            s.ss,
            sm[y * w + x - 1] as i32,
            sm[(y - 1) * w + x] as i32,
            sm[(y - 1) * w + x - 1] as i32,
            s.wrap_pred,
        )
    }
}
fn grid(j: &Jpeg, s: &Scan) -> (usize, usize, usize) {
    let (x, y) = if s.comp.len() == 1 {
        let c = &j.components[s.comp[0]];
        (c.bw_real, c.bh_real)
    } else {
        (j.mcusx, j.mcusy)
    };
    (x, y, if s.ri > 0 { s.ri / x } else { 0 })
}
pub(super) fn roughness(j: &Jpeg, s: &Scan) -> u64 {
    let mut rough = 0;
    for &ci in &s.comp {
        let c = &j.components[ci];
        for k in 1..c.samples.len() {
            if k % c.bw != 0 {
                let d = (c.samples[k] as i32 - c.samples[k - 1] as i32) & 65535;
                rough += if d > 32768 { 65536 - d } else { d } as u64;
            }
        }
    }
    rough
}
pub(super) fn decode(data: &[u8], first: usize, j: &mut Jpeg, s: &mut Scan) -> Result<usize> {
    let mut rd = Reader::new(data, first);
    s.padbit = -1;
    let (mx, my, rows) = grid(j, s);
    let top = (1i32 << (j.precision - s.al)) - 1;
    let (mut rst, mut y0) = (0, 0);
    for yy in 0..my {
        if rows > 0 && yy > 0 && yy % rows == 0 {
            rd.padding(s)?;
            rd.restart(rst)?;
            rst += 1;
            y0 = yy;
        }
        for xx in 0..mx {
            for (i, &ci) in s.comp.iter().enumerate() {
                let c = &mut j.components[ci];
                let (h, v) = if s.comp.len() == 1 {
                    (1, 1)
                } else {
                    (c.h, c.v)
                };
                for r in 0..v {
                    for q in 0..h {
                        let t = s.dc[i].decode(&mut rd)? as usize;
                        if t > 16 {
                            return Err(fail("BadCode", rd.p));
                        }
                        let diff = if t == 16 {
                            32768
                        } else {
                            extend(rd.bits(t)?, t)
                        };
                        let (x, y) = (xx * h + q, yy * v + r);
                        let value = (diff
                            + prediction(j.precision, s, &c.samples, x, y, y0 * v, c.bw))
                            & 65535;
                        if value > top {
                            return Err(fail("CoefOutOfRange", rd.p));
                        }
                        c.samples[y * c.bw + x] = value as u16;
                    }
                }
            }
        }
    }
    rd.padding(s)?;
    if s.padbit < 0 {
        s.padbit = 1;
    }
    rd.end()
}
pub(super) fn encode(w: &mut Writer, j: &Jpeg, s: &Scan) -> Result<()> {
    let (mx, my, rows) = grid(j, s);
    let (mut rst, mut y0) = (0, 0);
    for yy in 0..my {
        if rows > 0 && yy > 0 && yy % rows == 0 {
            w.align(s.padbit)?;
            w.restart(rst)?;
            rst += 1;
            y0 = yy;
        }
        for xx in 0..mx {
            for (i, &ci) in s.comp.iter().enumerate() {
                let c = &j.components[ci];
                let (h, v) = if s.comp.len() == 1 {
                    (1, 1)
                } else {
                    (c.h, c.v)
                };
                for r in 0..v {
                    for q in 0..h {
                        let (x, y) = (xx * h + q, yy * v + r);
                        let d = (c.samples[y * c.bw + x] as i32
                            - prediction(j.precision, s, &c.samples, x, y, y0 * v, c.bw))
                            & 65535;
                        let (mag, bits) = if d & 32768 != 0 {
                            let m = (-d) & 32767;
                            if m == 0 { (32768, 0) } else { (m, !m) }
                        } else {
                            (d & 32767, d & 32767)
                        };
                        let n = category(mag);
                        s.dc[i].emit(w, n)?;
                        if n > 0 && n < 16 {
                            w.bits(bits, n)?;
                        }
                    }
                }
            }
        }
    }
    w.align(s.padbit)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wrapped_half_difference() {
        assert_eq!(predict(5, 7, 65535, 0, false), 32774);
        assert_eq!(predict(5, 7, 65535, 0, true), 6);
        assert_eq!(predict(6, 65535, 7, 0, true), 6);
        assert_eq!(predict(4, 2, 3, 4, false), 1);
    }
}
