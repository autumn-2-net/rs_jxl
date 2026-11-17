// PJA MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Source: pas/pjaj2kpm.c px_dec through pjaj2kp_cb_full. See NOTICE-PJA.
//! Exact bidirectional classic tier-1 coefficient/decision walk for pixel mode.
use super::tier1::{self, Block, NEG, PIV, Passes, REF, SIG};
use crate::{Error, Result};
fn invalid() -> Error {
    Error::Invalid("J2K pixel tier1 decisions")
}
fn validate(b: Block, p: u32) -> Result<()> {
    if b.width == 0
        || b.height == 0
        || b.width > 1024
        || b.height > 1024
        || b.width * b.height > 4096
        || b.orientation > 3
        || b.passes > 255
        || !(1..=30).contains(&p)
    {
        Err(invalid())
    } else {
        Ok(())
    }
}
struct Walk<'a> {
    input: Option<&'a [u8]>,
    count_only: bool,
    output: Vec<u8>,
    cap: usize,
    ns: usize,
    nsig: usize,
    raw: bool,
    mag: Vec<u32>,
}
impl Walk<'_> {
    fn decision(&mut self, cx: usize, bit: u8) -> Result<u8> {
        let ctx = if self.raw { 19 } else { cx };
        let b;
        if let Some(input) = self.input {
            let v = *input.get(self.ns).ok_or_else(invalid)?;
            if (v >> 1) as usize != ctx {
                return Err(invalid());
            }
            b = v & 1;
        } else {
            b = bit;
            if !self.count_only {
                if self.ns >= self.cap {
                    return Err(Error::Limit("pixel tier1 decisions"));
                }
                self.output.push(((ctx << 1) as u8) | b);
            }
        }
        self.ns += 1;
        Ok(b)
    }
    fn sign(
        &mut self,
        st: &mut [u8],
        nb: &mut [u8],
        i: usize,
        s: usize,
        excl: bool,
        ci: usize,
    ) -> Result<()> {
        let (cx, xr) = tier1::sign_context(st, i, s, excl);
        let neg = (self.mag[ci] & 0x80000000 != 0) as u8;
        let value = if self.input.is_some() {
            0
        } else if self.raw {
            neg
        } else {
            neg ^ xr
        };
        let value = self.decision(cx, value)?;
        let neg = if self.input.is_some() {
            if self.raw { value } else { value ^ xr }
        } else {
            neg
        };
        self.nsig += 1;
        tier1::make_significant(st, nb, i, s, neg);
        Ok(())
    }
    fn pass(
        &mut self,
        b: Block,
        kind: usize,
        plane: u32,
        st: &mut [u8],
        nb: &mut [u8],
        segmark: bool,
    ) -> Result<()> {
        let s = b.width + 2;
        let causal = b.style & 8 != 0;
        for y0 in (0..b.height).step_by(4) {
            let y1 = (y0 + 4).min(b.height);
            for x in 0..b.width {
                let i0 = (y0 + 1) * s + x + 1;
                let mut start = y0;
                if kind == 2
                    && y1 - y0 == 4
                    && (st[i0] | st[i0 + s] | st[i0 + 2 * s] | (st[i0 + 3 * s] & (SIG | PIV))) == 0
                    && (nb[i0]
                        | nb[i0 + s]
                        | nb[i0 + 2 * s]
                        | (nb[i0 + 3 * s] & if causal { 31 } else { 255 }))
                        == 0
                {
                    let mut r = 0usize;
                    let mut any = 0;
                    if self.input.is_none() {
                        for j in 0..4 {
                            if self.mag[(y0 + j) * b.width + x] >> plane & 1 != 0 {
                                r = j;
                                any = 1;
                                break;
                            }
                        }
                    }
                    if self.decision(17, any)? == 0 {
                        continue;
                    }
                    if self.input.is_none() {
                        self.decision(18, (r >> 1) as u8)?;
                        self.decision(18, (r & 1) as u8)?;
                    } else {
                        r = ((self.decision(18, 0)? << 1) | self.decision(18, 0)?) as usize;
                    }
                    let y = y0 + r;
                    let ci = y * b.width + x;
                    if self.input.is_some() {
                        self.mag[ci] |= 1 << plane;
                    }
                    self.sign(st, nb, i0 + r * s, s, causal && y % 4 == 3, ci)?;
                    start = y + 1;
                }
                for y in start..y1 {
                    let i = i0 + (y - y0) * s;
                    let ci = y * b.width + x;
                    let mut m = nb[i];
                    if causal && y % 4 == 3 {
                        m &= 31
                    }
                    if kind == 1 {
                        if st[i] & (SIG | PIV) != SIG {
                            continue;
                        }
                        let cx = if st[i] & REF != 0 {
                            16
                        } else if m != 0 {
                            15
                        } else {
                            14
                        };
                        let bit = if self.input.is_none() {
                            ((self.mag[ci] >> plane) & 1) as u8
                        } else {
                            0
                        };
                        let bit = self.decision(cx, bit)?;
                        if self.input.is_some() && bit != 0 {
                            self.mag[ci] |= 1 << plane;
                        }
                        st[i] |= REF;
                    } else {
                        if kind == 0 {
                            if st[i] & SIG != 0 || m == 0 {
                                continue;
                            }
                            st[i] |= PIV;
                        } else if st[i] & (SIG | PIV) != 0 {
                            continue;
                        }
                        let bit = if self.input.is_none() {
                            ((self.mag[ci] >> plane) & 1) as u8
                        } else {
                            0
                        };
                        let bit = self.decision(tier1::zero_context(b.orientation, m), bit)?;
                        if bit != 0 {
                            if self.input.is_some() {
                                self.mag[ci] |= 1 << plane;
                            }
                            self.sign(st, nb, i, s, causal && y % 4 == 3, ci)?;
                        }
                    }
                }
            }
        }
        if kind == 2 {
            for y in 0..b.height {
                for x in 0..b.width {
                    st[(y + 1) * s + x + 1] &= !PIV;
                }
            }
            if segmark {
                for bit in [1, 0, 1, 0] {
                    self.decision(18, bit)?;
                }
            }
        }
        Ok(())
    }
}
fn magnitudes(coef: &[i32], b: Block, p: u32) -> Result<Vec<u32>> {
    if coef.len() != b.width * b.height {
        return Err(invalid());
    }
    coef.iter()
        .map(|&v| {
            let a = v.unsigned_abs();
            if a >> p != 0 {
                Err(invalid())
            } else {
                Ok(a | if v < 0 { 0x80000000 } else { 0 })
            }
        })
        .collect()
}
fn run(
    b: Block,
    p: u32,
    coef: Option<&[i32]>,
    input: Option<&Passes>,
    cap: usize,
    count_only: bool,
) -> Result<(Passes, Vec<i32>, usize)> {
    validate(b, p)?;
    let mag = if let Some(coef) = coef {
        magnitudes(coef, b, p)?
    } else {
        vec![0; b.width * b.height]
    };
    if let Some(input) = input {
        if input.pass_ends.len() != b.passes {
            return Err(invalid());
        }
    }
    let mut w = Walk {
        input: input.map(|i| i.symbols.as_slice()),
        count_only,
        output: Vec::new(),
        cap,
        ns: 0,
        nsig: 0,
        raw: false,
        mag,
    };
    let mut st = vec![0u8; (b.width + 2) * (b.height + 2)];
    let mut nb = st.clone();
    let mut ends = Vec::new();
    let mut last = 0;
    let count = if count_only {
        (3 * p - 2) as usize
    } else {
        b.passes
    };
    for k in 0..count {
        let kind = if k == 0 { 2 } else { (k - 1) % 3 };
        let plane = p as i32 - 1 - (k as i32 + 2) / 3;
        if plane < 0 {
            return Err(invalid());
        }
        w.raw = b.style & 1 != 0 && k >= 10 && kind < 2;
        let (ns, nsig) = (w.ns, w.nsig);
        w.pass(
            b,
            kind,
            plane as u32,
            &mut st,
            &mut nb,
            !count_only && b.style & 32 != 0,
        )?;
        if let Some(input) = input {
            if input.pass_ends[k] != w.ns {
                return Err(invalid());
            }
        }
        ends.push(w.ns);
        if if kind == 1 { w.ns > ns } else { w.nsig > nsig } {
            last = k + 1;
        }
    }
    if let Some(input) = input {
        if input.symbols.len() != w.ns {
            return Err(invalid());
        }
    }
    let mut coefficients = vec![0; b.width * b.height];
    for y in 0..b.height {
        for x in 0..b.width {
            let v = (w.mag[y * b.width + x] & 0x7fffffff) as i32;
            coefficients[y * b.width + x] = if st[(y + 1) * (b.width + 2) + x + 1] & NEG != 0 {
                -v
            } else {
                v
            };
        }
    }
    Ok((
        Passes {
            symbols: w.output,
            pass_ends: ends,
            used_bits: 0,
        },
        coefficients,
        last,
    ))
}
pub fn coefficients(block: Block, planes: u32, passes: &Passes) -> Result<Vec<i32>> {
    Ok(run(block, planes, None, Some(passes), 0, false)?.1)
}
pub fn symbols(block: Block, planes: u32, coef: &[i32], cap: usize) -> Result<Passes> {
    Ok(run(block, planes, Some(coef), None, cap, false)?.0)
}
pub fn full_passes(block: Block, planes: u32, coef: &[i32]) -> Result<usize> {
    Ok(run(
        Block { passes: 0, ..block },
        planes,
        Some(coef),
        None,
        0,
        true,
    )?
    .2)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_coefficients_and_context_checks() {
        for style in [0, 1, 2, 8, 9, 32, 35] {
            for orientation in 0..4 {
                let block = Block {
                    width: 5,
                    height: 7,
                    orientation,
                    style,
                    passes: 22,
                };
                let coef: Vec<i32> = (0..35)
                    .map(|i| {
                        if i % 5 == 0 {
                            0
                        } else {
                            ((i * 37 % 255) as i32 - 127) * if i % 3 == 0 { -1 } else { 1 }
                        }
                    })
                    .collect();
                let passes = symbols(block, 8, &coef, 8192).unwrap();
                assert_eq!(coefficients(block, 8, &passes).unwrap(), coef);
                assert!(full_passes(block, 8, &coef).unwrap() <= 22);
                let mut bad = passes;
                bad.symbols[0] ^= 2;
                assert!(coefficients(block, 8, &bad).is_err());
            }
        }
    }
    #[test]
    fn plane_and_storage_bounds() {
        let b = Block {
            width: 1,
            height: 1,
            orientation: 0,
            style: 0,
            passes: 2,
        };
        assert!(symbols(b, 1, &[1], 20).is_err());
        assert!(symbols(b, 31, &[1], 20).is_err());
        assert!(symbols(b, 2, &[4], 20).is_err());
    }
}

#[cfg(test)]
#[test]
#[ignore = "requires RS_JXL_PLUS_PIXEL_TIER1_ORACLE helper built with -DPJA_PIXEL_T1"]
fn original_c_pixel_tier1() {
    use super::tier1::oracle::{run_env, word};
    for style in [0, 1, 2, 8, 9, 32, 35] {
        for orientation in 0..4 {
            let block = Block {
                width: 5,
                height: 7,
                orientation,
                style,
                passes: 22,
            };
            let coef: Vec<i32> = (0..35)
                .map(|i| {
                    if i % 5 == 0 {
                        0
                    } else {
                        ((i * 37 % 255) as i32 - 127) * if i % 3 == 0 { -1 } else { 1 }
                    }
                })
                .collect();
            let passes = symbols(block, 8, &coef, 8192).unwrap();
            let full = full_passes(block, 8, &coef).unwrap();
            let decoded = coefficients(block, 8, &passes).unwrap();
            let mut desc = Vec::new();
            for n in [5, 7, orientation, style as usize, 22, 8] {
                word(&mut desc, n)
            }
            for v in &coef {
                desc.extend(v.to_le_bytes());
            }
            let mut expected = Vec::new();
            word(&mut expected, passes.symbols.len());
            word(&mut expected, full);
            expected.extend(passes.symbols);
            for n in passes.pass_ends {
                word(&mut expected, n)
            }
            for v in decoded {
                expected.extend(v.to_le_bytes());
            }
            assert_eq!(
                run_env("RS_JXL_PLUS_PIXEL_TIER1_ORACLE", "pixel_tier1", &desc),
                expected,
                "style{style} orientation{orientation}"
            );
        }
    }
}
