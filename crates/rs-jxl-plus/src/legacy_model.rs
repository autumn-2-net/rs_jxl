//! PJA stream-v5/v6 full and fast DCT models, translated from pas/pjacoef.inc
//! and pas/pjamodel.pas Init/CodeBand, MIT, pinned revision
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA/NOTICE-PJA.
use crate::{Error, Result, arith::BitCoder, model::Component, predict::*, sample_model::Bank};
mod state;
use state::*;

pub(crate) fn band_count(c: &[Component]) -> Result<usize> {
    crate::model::band_count(c)
}
pub(crate) fn code_band<C: BitCoder>(
    cd: &mut C,
    components: &mut [Component],
    band: usize,
    fast: bool,
) -> Result<()> {
    let nb = band_count(components)?;
    if band >= nb {
        return Err(Error::Invalid("legacy DCT band"));
    }
    let rows = components[0].height / components[0].v_samp;
    let m0 = rows * band / nb;
    let m1 = rows * (band + 1) / nb;
    let t = tables();
    let mut coef: Vec<Vec<i16>> = components
        .iter()
        .map(|c| {
            let mut v = vec![0; c.coefficients.len()];
            for (a, b) in c.coefficients.chunks_exact(64).zip(v.chunks_exact_mut(64)) {
                for k in 0..64 {
                    b[k] = a[t.zz_table[k] as usize];
                }
            }
            v
        })
        .collect();
    let mut luma = Model::new(false, fast);
    let mut chroma = Model::new(true, fast);
    let mut luma_nz = Vec::new();
    for ci in 0..components.len() {
        let c = &components[ci];
        let mut q = [0i32; 64];
        for (k, v) in q.iter_mut().enumerate() {
            *v = c.quant[t.zz_table[k] as usize] as i32;
        }
        let sx = components[0].h_samp / c.h_samp;
        let sy = components[0].v_samp / c.v_samp;
        let use_luma = ci > 0
            && sx > 0
            && sy > 0
            && c.width * sx <= components[0].width
            && c.height * sy <= components[0].height;
        let nz = component(
            cd,
            if ci == 0 { &mut luma } else { &mut chroma },
            &mut coef[ci],
            c.width,
            c.height,
            &q,
            if use_luma { Some(&luma_nz) } else { None },
            components[0].width,
            sx,
            sy,
            m0 * c.v_samp,
            m1 * c.v_samp,
        )?;
        if ci == 0 {
            luma_nz = nz;
        }
    }
    if !cd.encoding() {
        for (c, src) in components.iter_mut().zip(coef) {
            for (a, b) in src
                .chunks_exact(64)
                .zip(c.coefficients.chunks_exact_mut(64))
            {
                for k in 0..64 {
                    b[t.zz_table[k] as usize] = a[k];
                }
            }
        }
    }
    Ok(())
}
#[derive(Clone, Default)]
struct Cache {
    eg: [u32; 8],
    px: [u32; 8],
}
#[allow(clippy::too_many_arguments)]
fn component<C: BitCoder>(
    cd: &mut C,
    m: &mut Model,
    coef: &mut [i16],
    bw: usize,
    bh: usize,
    q: &[i32; 64],
    luma_nz: Option<&Vec<u8>>,
    luma_w: usize,
    sx: usize,
    sy: usize,
    y0: usize,
    y1: usize,
) -> Result<Vec<u8>> {
    let t = tables();
    let enc = cd.encoding();
    let dq0 = 2896 * q[0] as i64;
    let mut fb = [0i64; 64];
    let mut rows = vec![Cache::default(); bw];
    let mut left = Cache::default();
    let mut nzc = vec![0u8; bw * bh];
    let mut dcr = nzc.clone();
    let nzb = |v: i32| t.nz_table[v as usize] as i32;
    macro_rules! one {
        ($a:ident[$i:expr],$v:expr) => {
            m.one(cd, B::$a, $i as usize, $v as i32)
        };
    }
    macro_rules! avg {
        ($a:ident[$i:expr],$b:ident[$j:expr],$v:expr) => {
            m.avg(cd, (B::$a, $i as usize), (B::$b, $j as usize), $v as i32)
        };
    }
    macro_rules! mix{($w:ident,$sel:expr,$v:expr; $($a:ident[$i:expr]),+$(,)?)=>{m.mix(cd,W::$w,$sel as usize,&[$((B::$a,$i as usize)),+],$v as i32)}}
    for by in y0..y1 {
        for bx in 0..bw {
            let base = (by * bw + bx) * 64;
            let ha = by > y0;
            let hl = bx > 0;
            let ab = if ha { base - bw * 64 } else { base };
            let lb = if hl { base - 64 } else { base };
            let mut b: [i16; 64] = coef[base..base + 64].try_into().unwrap();
            let na = if ha {
                nzc[(by - 1) * bw + bx] as i32
            } else {
                0
            };
            let nl = if hl { nzc[by * bw + bx - 1] as i32 } else { 0 };
            let mut ctx = if ha && hl {
                nzb((na + nl + 1) / 2)
            } else if ha {
                nzb(na)
            } else if hl {
                nzb(nl)
            } else {
                0
            };
            let za = if ha { nzb(na) } else { 0 };
            let zb = if hl { nzb(nl) } else { 0 };
            let lz = luma_nz.map_or(11, |v| nzb(v[(by * sy) * luma_w + bx * sx] as i32));
            let nt = if enc {
                (nzmask(&b) & t.interior_zz).count_ones() as i32
            } else {
                0
            };
            let mut node = 1;
            for k in (0..6).rev() {
                let bit = if m.fast {
                    if m.chroma {
                        one!(Nz3[(lz * 11 + ctx) * 64 + node], (nt >> k) & 1)
                    } else {
                        one!(Nz2[(za * 11 + zb) * 64 + node], (nt >> k) & 1)
                    }
                } else {
                    mix!(Nz,5-k,(nt>>k)&1;Nz1[ctx*64+node],Nz2[(za*11+zb)*64+node],Nz3[(lz*11+ctx)*64+node])
                };
                node = node * 2 + bit;
            }
            let n = node - 64;
            if n > 49 {
                return Err(Error::Invalid("legacy interior count"));
            }
            nzc[by * bw + bx] = n as u8;
            let nz_b = nzb(n);
            let mut rem = n;
            for idx in 0..49 {
                if rem == 0 {
                    break;
                }
                let k = t.interior[idx] as usize;
                let al = ha && hl;
                let ar = ha && bx + 1 < bw;
                let ws = 2 * ha as u32 + al as u32 + ar as u32 + 2 * hl as u32;
                let nb = if ws == 0 {
                    -1
                } else {
                    nbrmag(
                        if ha { coef[ab + k] } else { 0 },
                        if al { coef[ab - 64 + k] } else { 0 },
                        if ar { coef[ab + 64 + k] } else { 0 },
                        if hl { coef[lb + k] } else { 0 },
                        ws,
                    )
                };
                let ib = t.ib_wa[idx] as i32 * iabs(b[t.ib_na[idx] as usize] as i32)
                    + t.ib_wb[idx] as i32 * iabs(b[t.ib_nb[idx] as usize] as i32);
                let mbi = if t.ib_none[idx] != 0 {
                    19
                } else {
                    finelog(ib as i64).min(18)
                };
                let (mbn, mbs) = if nb < 0 {
                    (19, finelog(2 * ib as i64).min(18))
                } else {
                    (
                        finelog(nb as i64).min(18),
                        finelog((nb + ib) as i64).min(18),
                    )
                };
                let kb = t.k_bucket[k] as i32;
                let rb = t.rem_table[rem as usize] as i32;
                let v = if enc { b[k] as i32 } else { 0 };
                let left = t.interior_left[idx] as usize;
                let bit = if rem >= left as i32 {
                    1
                } else if m.fast {
                    if m.chroma {
                        avg!(
                            Fz[(k as i32 * 9 + rb) * 20 + mbn],
                            Zero4[(kb * 20 + mbs) * 9 + rb],
                            v != 0
                        )
                    } else {
                        avg!(
                            Fz[(k as i32 * 9 + rb) * 20 + mbs],
                            Zero1[(k as i32 * 9 + rb) * 20 + mbn],
                            v != 0
                        )
                    }
                } else {
                    mix!(Zero,k,v!=0;Zero5[(k as i32*12+lz)*9+rb],Zero1[(k as i32*9+rb)*20+mbn],Zero2[(k as i32*20+mbi)*4+mbn.min(3)],Zero3[k*17+t.ratio_of[left][rem as usize] as usize],Zero4[(kb*20+mbs)*9+rb])
                };
                if bit != 0 {
                    let e = bitlen64(v as i64);
                    let mut i = 1;
                    while i < 15 {
                        let bit = if m.fast {
                            avg!(
                                Fx[((kb * 9 + rb) * 20 + mbs) * 15 + i - 1],
                                Expo1[(kb * 20 + mbn) * 15 + i - 1],
                                e > i
                            )
                        } else {
                            mix!(Expo,(i-1)*11+kb,e>i;Expo1[(kb*20+mbn)*15+i-1],Expo2[(kb*20+mbi)*15+i-1],Expo3[(kb*9+rb)*15+i-1],Expo4[mbs*15+i-1])
                        };
                        if bit == 0 {
                            break;
                        }
                        i += 1;
                    }
                    let x = m.mantissa(
                        cd,
                        B::Mant1,
                        kb as usize,
                        11,
                        B::Mant,
                        15,
                        i as usize,
                        v.abs(),
                    );
                    ctx = if ha { sgn3(coef[ab + k] as i64) } else { 1 };
                    ctx = ctx * 3 + if hl { sgn3(coef[lb + k] as i64) } else { 1 };
                    let bit = one!(Sign[kb * 9 + ctx], v < 0);
                    if !enc {
                        b[k] = if bit != 0 { -x as i16 } else { x as i16 };
                    }
                    rem -= 1;
                }
            }
            if rem > 0 {
                return Err(Error::Invalid("legacy unresolved interior count"));
            }
            let mut fm = 0u64;
            let mut zm = nzmask(&b) & t.interior_zz;
            while zm != 0 {
                let k = zm.trailing_zeros() as usize;
                zm &= zm - 1;
                let p = t.zz_table[k] as usize;
                fb[p] = dequant(b[k] as i32, q[k]);
                fm |= 1u64 << p;
            }
            for ep in 0..14usize {
                let (u, w) = if ep & 1 == 0 {
                    (0, ep / 2 + 1)
                } else {
                    (ep / 2 + 1, 0)
                };
                let k = t.unzig[u * 8 + w] as usize;
                let mut pb = 21;
                let mut pred = 0i64;
                if u == 0 && ha {
                    let mut sum = 0i64;
                    let mut mm = fm & (0x0101010101010100u64 << w);
                    while mm != 0 {
                        let p = mm.trailing_zeros() as usize;
                        mm &= mm - 1;
                        sum += t.w3[p >> 3] * fb[p];
                    }
                    pred = divroundr(w32(rows[bx].eg[w] as i64 - sum), 724 * q[k] as i64, 0.0);
                    pb = finelog(div8round(w32(5 * pred))).min(20);
                } else if w == 0 && hl {
                    let mut sum = 0i64;
                    let mut mm = fm & (0xfeu64 << (u * 8));
                    while mm != 0 {
                        let p = mm.trailing_zeros() as usize;
                        mm &= mm - 1;
                        sum += t.w3[p & 7] * fb[p];
                    }
                    pred = divroundr(w32(left.eg[u] as i64 - sum), 724 * q[k] as i64, 0.0);
                    pb = finelog(div8round(w32(5 * pred))).min(20);
                }
                let mbn = if m.fast {
                    19
                } else {
                    let a = if ha { iabs(coef[ab + k] as i32) } else { 0 };
                    let l = if hl { iabs(coef[lb + k] as i32) } else { 0 };
                    if ha && hl {
                        bitlen64((a + l) as i64).min(18)
                    } else if ha || hl {
                        bitlen64(2 * (a + l) as i64).min(18)
                    } else {
                        19
                    }
                };
                let prev_b = if ep >= 2 {
                    bitlen64(b[t.unzig[if u == 0 { w - 1 } else { (u - 1) * 8 }] as usize] as i64)
                        .min(7)
                } else {
                    7
                };
                let ib =
                    finelog(b[t.unzig[if u == 0 { 8 + w } else { u * 8 + 1 }] as usize] as i64)
                        .min(11);
                let v = if enc { b[k] as i32 } else { 0 };
                let ep = ep as i32;
                let bit = if m.fast {
                    avg!(
                        Fez[((ep * 11 + nz_b) * 8 + prev_b) * 22 + pb],
                        Ezero5[(ep * 12 + ib) * 4 + pb.min(3)],
                        v != 0
                    )
                } else {
                    mix!(EZero,ep,v!=0;Ezero5[(ep*12+ib)*4+pb.min(3)],Ezero1[ep*22+pb],Ezero2[ep*11+nz_b],Ezero3[(ep*8+prev_b)*22+pb],Ezero4[ep*20+mbn])
                };
                if bit != 0 {
                    let e = bitlen64(v as i64);
                    let mut i = 1;
                    while i < 15 {
                        let bit = if m.fast {
                            ctx = if m.chroma {
                                ((ep * 22 + pb) * 11 + nz_b) * 8 * 15 + i - 1
                            } else {
                                (((ep * 11 + nz_b) * 22 + pb) * 8 + prev_b) * 15 + i - 1
                            };
                            avg!(Fex[ctx], Eexpo5[(ep * 12 + ib) * 15 + i - 1], e > i)
                        } else {
                            mix!(EExpo,(i-1)*14+ep,e>i;Eexpo5[(ep*12+ib)*15+i-1],Eexpo1[(ep*22+pb)*15+i-1],Eexpo2[(ep*20+mbn)*15+i-1],Eexpo3[(pb*8+prev_b)*15+i-1],Eexpo4[(ep*11+nz_b)*15+i-1])
                        };
                        if bit == 0 {
                            break;
                        }
                        i += 1;
                    }
                    let x = m.mantissa(
                        cd,
                        B::Emant1,
                        pb as usize,
                        22,
                        B::Mant,
                        15,
                        i as usize,
                        v.abs(),
                    );
                    let sg = if pb == 21 { 1 } else { sgn3(pred) };
                    let bit = one!(Esign[((ep & 1) * 3 + sg) * 22 + pb], v < 0);
                    if !enc {
                        b[k] = if bit != 0 { -x as i16 } else { x as i16 };
                    }
                }
            }
            zm = nzmask(&b) & t.edge_zz;
            while zm != 0 {
                let k = zm.trailing_zeros() as usize;
                zm &= zm - 1;
                let p = t.zz_table[k] as usize;
                fb[p] = dequant(b[k] as i32, q[k]);
                fm |= 1u64 << p;
            }
            let mut ka = [0u32; 32];
            let mut eg = [0u32; 16];
            kernels(&fb, fm, &mut ka, &mut eg);
            let mut ests = Vec::with_capacity(16);
            if hl {
                for (i, &v) in ka[..8].iter().enumerate() {
                    ests.push(left.px[i].wrapping_sub(v) as i32 as i64 / 4);
                }
            }
            if ha {
                for (i, &v) in ka[8..16].iter().enumerate() {
                    ests.push(rows[bx].px[i].wrapping_sub(v) as i32 as i64 / 4);
                }
            }
            ests.sort_unstable();
            let n = ests.len();
            let (mut pred, ctx) = if n > 0 {
                let lo = ests[0];
                let hi = ests[n - 1];
                let total = w32(ests[n / 4..n - n / 4].iter().sum());
                (
                    divroundr(total, (n - 2 * (n / 4)) as i64 * dq0, 0.0),
                    finelog(divroundr(w32(hi - lo), dq0, 0.0)).min(22),
                )
            } else {
                (0, 23)
            };
            pred = pred.clamp(-32767, 32767);
            let mut res = if enc { b[0] as i32 - pred as i32 } else { 0 };
            let mut rb2 = 0;
            let mut nbs = 0;
            if ha {
                rb2 += dcr[(by - 1) * bw + bx] as i32;
                nbs += 1;
            }
            if hl {
                rb2 += dcr[by * bw + bx - 1] as i32;
                nbs += 1;
            }
            rb2 = rb2.min(19);
            let bit = if m.fast {
                one!(Fdz[(rb2 * 24 + ctx) * 3 + nbs], res != 0)
            } else {
                mix!(DcZero,ctx,res!=0;DcZero[ctx],Dz2[rb2],Dz3[nz_b*8+ctx/3],Dz4[ctx*3+nbs])
            };
            if bit != 0 {
                let e = bitlen64(res as i64);
                let mut i = 1;
                while i < 17 {
                    let bit = if m.fast {
                        one!(Fdx[(ctx * 3 + nbs) * 17 + i - 1], e > i)
                    } else {
                        mix!(DcExpo,i-1,e>i;DcExpo[ctx*17+i-1],Dx2[rb2*17+i-1],Dx3[(nz_b*8+ctx/3)*17+i-1],Dx4[(ctx*3+nbs)*17+i-1])
                    };
                    if bit == 0 {
                        break;
                    }
                    i += 1;
                }
                let x = m.mantissa(
                    cd,
                    B::DcMant1,
                    ctx as usize,
                    24,
                    B::DcMant,
                    17,
                    i as usize,
                    res.abs(),
                );
                res = if one!(DcSign[ctx], res < 0) != 0 {
                    -x
                } else {
                    x
                };
            }
            dcr[by * bw + bx] = finelog(res as i64).min(15) as u8;
            if !enc {
                let x = pred + res as i64;
                if !(-32767..=32767).contains(&x) {
                    return Err(Error::Invalid("legacy DC range"));
                }
                b[0] = x as i16;
            }
            fb[0] = dequant(b[0] as i32, q[0]);
            let f = fb[0] as u32;
            for y in 0..8 {
                left.px[y] = ka[16 + y].wrapping_add(t.k4[0][16 + y].wrapping_mul(f));
                rows[bx].px[y] = ka[24 + y].wrapping_add(t.k4[0][24 + y].wrapping_mul(f));
                left.eg[y] = eg[y];
                rows[bx].eg[y] = eg[8 + y];
            }
            left.eg[0] = left.eg[0].wrapping_add((t.e3[0] as u32).wrapping_mul(f));
            rows[bx].eg[0] = rows[bx].eg[0].wrapping_add((t.e3[0] as u32).wrapping_mul(f));
            let mut mm = fm | 1;
            while mm != 0 {
                fb[mm.trailing_zeros() as usize] = 0;
                mm &= mm - 1;
            }
            coef[base..base + 64].copy_from_slice(&b);
        }
    }
    Ok(nzc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::{Decoder, Encoder};
    fn component(w: usize, h: usize, hs: usize, vs: usize, seed: usize) -> Component {
        let mut coefficients = vec![0; w * h * 64];
        for (block, b) in coefficients.chunks_exact_mut(64).enumerate() {
            b[0] = ((block * 113 + seed * 31) % 1600) as i16 - 800;
            for (k, v) in b.iter_mut().enumerate().skip(1) {
                if block.is_multiple_of(3) || (block + k + seed).is_multiple_of(5) {
                    let m = (1 + (block * 37 + k * 17 + seed * 29) % 1600) as i16;
                    *v = if (block + k).is_multiple_of(2) { m } else { -m };
                }
            }
        }
        Component {
            width: w,
            height: h,
            h_samp: hs,
            v_samp: vs,
            quant: std::array::from_fn(|k| 1 + ((k * 13 + seed * 29) % 1000) as u16),
            coefficients,
        }
    }
    #[test]
    fn full_fast_roundtrip_luma_chroma_multiline() {
        for fast in [false, true] {
            for original in [
                vec![component(3, 4, 1, 1, 0)],
                vec![
                    component(4, 4, 2, 2, 1),
                    component(2, 2, 1, 1, 2),
                    component(2, 2, 1, 1, 3),
                ],
            ] {
                let mut encoded = original.clone();
                let mut coder = Encoder::new(1 << 24);
                code_band(&mut coder, &mut encoded, 0, fast).unwrap();
                let bytes = coder.finish().unwrap();
                let mut back = original.clone();
                for c in &mut back {
                    c.coefficients.fill(0);
                }
                let mut coder = Decoder::new(&bytes);
                code_band(&mut coder, &mut back, 0, fast).unwrap();
                assert_eq!(back, original, "fast={fast}");
            }
        }
    }
    #[test]
    fn extremes_and_invalid_band() {
        for fast in [false, true] {
            let mut original = vec![component(2, 2, 1, 1, 0)];
            original[0].quant.fill(65535);
            original[0].coefficients.fill(0);
            for k in [0, 1, 8, 9, 63, 64] {
                original[0].coefficients[k] = if k.is_multiple_of(2) { 32767 } else { -32767 };
            }
            let mut encoded = original.clone();
            let mut coder = Encoder::new(1 << 20);
            code_band(&mut coder, &mut encoded, 0, fast).unwrap();
            let bytes = coder.finish().unwrap();
            let mut back = original.clone();
            back[0].coefficients.fill(0);
            code_band(&mut Decoder::new(&bytes), &mut back, 0, fast).unwrap();
            assert_eq!(back, original);
            assert!(code_band(&mut Decoder::new(&bytes), &mut back, 1, fast).is_err());
        }
    }
}
