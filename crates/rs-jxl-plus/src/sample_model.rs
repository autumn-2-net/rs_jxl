//! JPEG-LS sample and symbol models from PJA pas/pjamodel.pas (MIT),
//! revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA/NOTICE-PJA.
use crate::{
    Error, Result,
    arith::{AdaptiveBit, BitCoder, Decoder, Encoder},
    jls::{self, Image, Limits},
    predict::{bitlen64, divroundr, finelog},
    probability,
};
mod symbols;
pub(crate) use symbols::code_symbol_stream;

pub(super) struct Bank {
    pub bits: Vec<Vec<AdaptiveBit>>,
    weights: Vec<Vec<[i32; 8]>>,
}
impl Bank {
    pub fn new(sizes: &[usize], mixers: &[(usize, usize)]) -> Self {
        let bits = sizes
            .iter()
            .map(|&s| vec![AdaptiveBit::default(); s])
            .collect();
        let weights = mixers
            .iter()
            .map(|&(n, s)| {
                let mut w = [0; 8];
                w[..n].fill(65536 / n as i32);
                vec![w; s]
            })
            .collect();
        Self { bits, weights }
    }
    pub fn one<C: BitCoder>(&mut self, cd: &mut C, t: usize, i: usize, v: i32) -> i32 {
        probability::one(cd, &mut self.bits[t][i], v)
    }
    pub fn mix<C: BitCoder>(
        &mut self,
        cd: &mut C,
        which: usize,
        sel: usize,
        refs: &[(usize, usize)],
        v: i32,
    ) -> i32 {
        let t = probability::tables();
        let n = refs.len() + 1;
        let mut inputs = [0i32; 8];
        for (p, &(a, i)) in inputs.iter_mut().zip(refs) {
            *p = t.stretch[self.bits[a][i].p as usize >> 4] as i32;
        }
        inputs[n - 1] = t.stretch[40960 >> 4] as i32;
        let w = &mut self.weights[which][sel];
        let dot = crate::mix_simd::dot(&inputs[..n], &w[..n]);
        let pr = t.squash[((dot / 65536).clamp(-2048, 2047) + 2048) as usize] as i32;
        let bit = cd.bit((pr as u32) << 4, v);
        let err = ((bit << 12) - pr) * 6;
        crate::mix_simd::update(&mut w[..n], &inputs[..n], err, 14, 131072);
        for &(a, i) in refs {
            self.bits[a][i].update(bit);
        }
        bit
    }
}
#[derive(Clone, Copy)]
struct Bias {
    a: i32,
    b: i32,
    c: i32,
    n: i32,
}
impl Bias {
    fn update(&mut self, e: i32) {
        self.a += e.abs();
        self.b += e;
        if self.n == 64 {
            self.a >>= 1;
            self.b >>= 1;
            self.n >>= 1;
        }
        self.n += 1;
        if self.b + self.n <= 0 {
            self.b += self.n;
            if self.b <= -self.n {
                self.b = -self.n + 1;
            }
            if self.c > -128 {
                self.c -= 1;
            }
        } else if self.b > 0 {
            self.b -= self.n;
            if self.b > 0 {
                self.b = 0;
            }
            if self.c < 127 {
                self.c += 1;
            }
        }
    }
}
fn quant(d: i32, t1: i32, t2: i32, t3: i32) -> i32 {
    if d <= -t3 {
        -4
    } else if d <= -t2 {
        -3
    } else if d <= -t1 {
        -2
    } else if d < 0 {
        -1
    } else if d == 0 {
        0
    } else if d < t1 {
        1
    } else if d < t2 {
        2
    } else if d < t3 {
        3
    } else {
        4
    }
}
pub(crate) fn band_count(j: &Image) -> usize {
    let total = j
        .components
        .iter()
        .map(|c| c.width.saturating_mul(c.height))
        .fold(0usize, usize::saturating_add);
    (total / (1 << 20)).min(16).min(j.height).max(1)
}
fn validate_samples(j: &Image, limits: &Limits) -> Result<()> {
    // pjacodec.LosslessView may have one effective bit after a JPEG point
    // transform. Such a view has no JPEG-LS scans; SOF55 itself still requires
    // 2..16 bits and goes through the stricter JPEG-LS validation.
    if j.bits == 1 && j.scans.is_empty() {
        if j.width == 0 || j.height == 0 || j.components.is_empty() || j.components.len() > 4 {
            return Err(Error::Invalid("lossless JPEG sample view"));
        }
        let mut count = 0usize;
        for c in &j.components {
            let n = c
                .width
                .checked_mul(c.height)
                .ok_or(Error::Limit("sample view count"))?;
            count = count
                .checked_add(n)
                .ok_or(Error::Limit("sample view count"))?;
            if c.width == 0
                || c.height == 0
                || c.samples.len() != n
                || c.samples.iter().any(|&s| s > 1)
            {
                return Err(Error::Invalid("lossless JPEG sample view"));
            }
        }
        if count > limits.max_samples {
            return Err(Error::Limit("sample view count"));
        }
        Ok(())
    } else {
        jls::validate(j, limits)
    }
}
pub(crate) fn code_band<C: BitCoder>(
    cd: &mut C,
    j: &mut Image,
    band: usize,
    blend: bool,
    limits: &Limits,
) -> Result<()> {
    validate_samples(j, limits)?;
    let nb = band_count(j);
    if band >= nb {
        return Err(Error::Invalid("JPEG-LS sample band"));
    }
    let maxv = (1i32 << j.bits) - 1;
    let range = maxv + 1;
    let half = range / 2;
    let (t1, t2, t3) = jls::default_thresholds(maxv, 0);
    let mut bias = [[Bias {
        a: 4,
        b: 0,
        c: 0,
        n: 1,
    }; 365]; 2];
    let mut m = Bank::new(
        &[
            48,
            730,
            192,
            25 * 24,
            48 * 18,
            730 * 18,
            48 * 18,
            25 * 18,
            19 * 24,
            19 * 18,
            48 * 9,
        ],
        &[(5, 48), (5, 36)],
    );
    let sub = j
        .components
        .iter()
        .any(|c| c.width != j.width || c.height != j.height);
    let mut res: Vec<Vec<i32>> = j.components.iter().map(|c| vec![0; c.width]).collect();
    let mut up = res.clone();
    let mut pe: Vec<Vec<i32>> = j
        .components
        .iter()
        .map(|c| vec![0; if blend { 8 * c.width } else { 0 }])
        .collect();
    let mut pe_up = pe.clone();
    let first: Vec<_> = j.components.iter().map(|c| c.height * band / nb).collect();
    let mut rows = Vec::new();
    if sub {
        for (ci, c) in j.components.iter().enumerate() {
            for y in first[ci]..c.height * (band + 1) / nb {
                rows.push((ci, y));
            }
        }
    } else {
        for y in j.height * band / nb..j.height * (band + 1) / nb {
            for ci in 0..j.components.len() {
                rows.push((ci, y));
            }
        }
    }
    for (ci, y) in rows {
        let w = j.components[ci].width;
        let have_up = y > first[ci];
        std::mem::swap(&mut res[ci], &mut up[ci]);
        if blend {
            std::mem::swap(&mut pe[ci], &mut pe_up[ci]);
        }
        let cls = usize::from(ci != 0);
        let base = y * w;
        for i in 0..w {
            let b = if have_up {
                j.components[ci].samples[base - w + i] as i32
            } else if i > 0 {
                j.components[ci].samples[base + i - 1] as i32
            } else {
                half
            };
            let a = if i > 0 {
                j.components[ci].samples[base + i - 1] as i32
            } else {
                b
            };
            let c = if have_up && i > 0 {
                j.components[ci].samples[base - w + i - 1] as i32
            } else {
                b
            };
            let d = if have_up && i + 1 < w {
                j.components[ci].samples[base - w + i + 1] as i32
            } else {
                b
            };
            let ctx = (quant(d - b, t1, t2, t3) * 9 + quant(b - c, t1, t2, t3)) * 9
                + quant(c - a, t1, t2, t3);
            let sgn = if ctx < 0 { -1 } else { 1 };
            let ctx = ctx.unsigned_abs() as usize;
            let bs = &mut bias[cls][ctx];
            let mut guesses = [0i32; 8];
            let mut px = if !blend {
                jls::med(a, b, c)
            } else {
                guesses = [
                    jls::med(a, b, c),
                    a,
                    b,
                    (a + b - c).clamp(0, maxv),
                    (a + b + 1) >> 1,
                    (a + d + 1) >> 1,
                    (b + (d - c) / 2).clamp(0, maxv),
                    (a + (d - c) / 2).clamp(0, maxv),
                ];
                let mut sw = 0i64;
                let mut sp = 0i64;
                for k in 0..8 {
                    let mut ee = if i > 0 { pe[ci][(i - 1) * 8 + k] } else { 0 };
                    if have_up {
                        ee += pe_up[ci][i * 8 + k];
                        if i > 0 {
                            ee += pe_up[ci][(i - 1) * 8 + k];
                        }
                        if i + 1 < w {
                            ee += pe_up[ci][(i + 1) * 8 + k];
                        }
                    }
                    let denom = (ee as i64 + 1) * (ee as i64 + 1);
                    let weight = ((1i64 << 40) / denom).max(1);
                    sw += weight;
                    sp += weight * guesses[k] as i64;
                }
                ((sp + sw / 2) / sw) as i32
            };
            px = (px + sgn * bs.c).clamp(0, maxv);
            let mut e = if cd.encoding() {
                (j.components[ci].samples[base + i] as i32 - px) * sgn
            } else {
                0
            };
            if e < -(range / 2) {
                e += range;
            } else if e >= (range + 1) / 2 {
                e -= range;
            }
            let act = finelog((bs.a / bs.n) as i64).min(23) as usize;
            let mut nbb = if i > 0 { res[ci][i - 1].abs() } else { 0 };
            if have_up {
                nbb += up[ci][i].abs();
            }
            let nbb = finelog(nbb as i64).min(23) as usize;
            let first = if ci > 0 && !sub {
                finelog(res[0][i] as i64).min(23) as usize
            } else {
                24
            };
            let bi = cls * 24 + act;
            if m.mix(
                cd,
                0,
                bi,
                &[
                    (0, bi),
                    (1, cls * 365 + ctx),
                    (2, (cls * 24 + nbb) * 4 + act.min(3)),
                    (3, first * 24 + act),
                ],
                i32::from(e != 0),
            ) != 0
            {
                let ex = bitlen64(e as i64);
                let mut k = 1usize;
                while k < 18 {
                    if m.mix(
                        cd,
                        1,
                        cls * 18 + k - 1,
                        &[
                            (4, bi * 18 + k - 1),
                            (5, (cls * 365 + ctx) * 18 + k - 1),
                            (6, (cls * 24 + nbb) * 18 + k - 1),
                            (7, first * 18 + k - 1),
                        ],
                        i32::from(ex > k as i32),
                    ) == 0
                    {
                        break;
                    }
                    k += 1;
                }
                let original = e.abs();
                let mut v = 1i32 << (k - 1);
                for b in (0..k - 1).rev() {
                    let bit = if b == k - 2 {
                        m.one(cd, 8, k * 24 + act, (original >> b) & 1)
                    } else {
                        m.one(cd, 9, k * 18 + b, (original >> b) & 1)
                    };
                    v |= bit << b;
                }
                let mut signs = if i > 0 {
                    if res[ci][i - 1] < 0 {
                        1
                    } else if res[ci][i - 1] > 0 {
                        2
                    } else {
                        0
                    }
                } else {
                    0
                };
                if have_up {
                    signs += if up[ci][i] < 0 {
                        3
                    } else if up[ci][i] > 0 {
                        6
                    } else {
                        0
                    };
                }
                e = if m.one(cd, 10, bi * 9 + signs, i32::from(e < 0)) != 0 {
                    -v
                } else {
                    v
                };
            }
            if !cd.encoding() {
                let mut x = px + e * sgn;
                if x < 0 {
                    x += range;
                } else if x > maxv {
                    x -= range;
                }
                if !(0..=maxv).contains(&x) {
                    return Err(Error::Invalid("JPEG-LS sample model range"));
                }
                j.components[ci].samples[base + i] = x as u16;
            }
            res[ci][i] = e;
            if blend {
                let x = j.components[ci].samples[base + i] as i32;
                for (k, &p) in guesses.iter().enumerate() {
                    pe[ci][i * 8 + k] = (x - p).abs();
                }
            }
            bs.update(e);
        }
    }
    Ok(())
}
pub(crate) fn code_runs<C: BitCoder>(cd: &mut C, j: &mut Image, max_runs: usize) -> Result<()> {
    let mut ex = [AdaptiveBit::default(); 32];
    let mut mn = vec![AdaptiveBit::default(); 1024];
    for s in &mut j.scans {
        if s.near == 0 {
            continue;
        }
        let n = cd.direct(s.runs.len() as i32, 32);
        if n < 0 || n as usize > max_runs {
            return Err(Error::Limit("JPEG-LS run count"));
        }
        if !cd.encoding() {
            s.runs.resize(n as usize, 0);
        }
        for i in 0..n as usize {
            let x = if cd.encoding() {
                let v = s.runs[i];
                if v < 0 {
                    return Err(Error::Invalid("negative JPEG-LS run"));
                }
                v as u32 + 1
            } else {
                0
            };
            let e = bitlen64(x as i64) - 1;
            let mut k = 0usize;
            while k < 31 {
                if probability::one(cd, &mut ex[k], i32::from(e > k as i32)) == 0 {
                    break;
                }
                k += 1;
            }
            let mut v = 1u64;
            for b in (0..k).rev() {
                v = (v << 1)
                    | probability::one(cd, &mut mn[k * 32 + b], ((x >> b) & 1) as i32) as u64;
            }
            if v - 1 > j.width as u64 {
                return Err(Error::Invalid("JPEG-LS recorded run exceeds width"));
            }
            if !cd.encoding() {
                s.runs[i] = (v - 1) as i32;
            }
        }
    }
    Ok(())
}
pub(crate) fn encode_samples(
    j: &Image,
    blend: bool,
    max_output: usize,
    limits: &Limits,
) -> Result<Vec<Vec<u8>>> {
    let mut j = j.clone();
    let mut remaining = max_output;
    let mut out = Vec::new();
    for band in 0..band_count(&j) {
        let mut cd = Encoder::new(remaining);
        code_band(&mut cd, &mut j, band, blend, limits)?;
        let bytes = cd.finish()?;
        remaining = remaining
            .checked_sub(bytes.len())
            .ok_or(Error::Limit("JPEG-LS sample streams"))?;
        out.push(bytes);
    }
    Ok(out)
}
pub(crate) fn decode_samples(
    j: &mut Image,
    streams: &[&[u8]],
    blend: bool,
    limits: &Limits,
) -> Result<()> {
    if streams.len() != band_count(j) {
        return Err(Error::Invalid("JPEG-LS sample band count"));
    }
    for (band, stream) in streams.iter().enumerate() {
        let mut cd = Decoder::new(stream);
        code_band(&mut cd, j, band, blend, limits)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn near_run_side_information_and_limit() {
        let limits = Limits::default();
        let mut j = jls::tests::fixture(8, 3, 1, 2, 0);
        let original = j.scans.iter().map(|s| s.runs.clone()).collect::<Vec<_>>();
        let mut cd = Encoder::new(1 << 20);
        code_runs(&mut cd, &mut j, limits.max_runs).unwrap();
        let bytes = cd.finish().unwrap();
        for s in &mut j.scans {
            s.runs.clear();
        }
        code_runs(&mut Decoder::new(&bytes), &mut j, limits.max_runs).unwrap();
        assert_eq!(
            j.scans.iter().map(|s| s.runs.clone()).collect::<Vec<_>>(),
            original
        );
        assert!(code_runs(&mut Decoder::new(&bytes), &mut j, 0).is_err());
    }
    #[test]
    fn point_transformed_one_bit_lossless_view() {
        let mut j = Image {
            width: 7,
            height: 3,
            bits: 1,
            components: vec![jls::Component {
                id: 0,
                h: 1,
                v: 1,
                width: 7,
                height: 3,
                samples: (0..21).map(|i| ((i * 3 + i / 7) & 1) as u16).collect(),
            }],
            scans: Vec::new(),
            raw: Vec::new(),
        };
        let original = j.components.clone();
        let limits = Limits::default();
        for blend in [false, true] {
            let streams = encode_samples(&j, blend, 1 << 20, &limits).unwrap();
            for c in &mut j.components {
                c.samples.fill(0);
            }
            decode_samples(
                &mut j,
                &streams.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                blend,
                &limits,
            )
            .unwrap();
            assert_eq!(j.components, original);
        }
    }
}
