//! PJA stream-v7 coefficient model with runtime SIMD kernels.
//!
//! Translated from PJA `pas/pjav7c_loop.h` and `pas/pjamodel.pas`
//! (`TCoefModel.InitV7`, `V7Component`, `CodeBand`), commit
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd. MIT licensed; see
//! LICENSE-PJA and NOTICE-PJA. This is a model for the independent jxl_plus
//! format, not a standard-JXL entropy mode or a PJA archive reader.

use crate::arith::{AdaptiveBit, BitCoder, Decoder, Encoder};
use crate::predict::*;
use crate::probability;
use crate::{Error, Result};

#[cfg(test)]
pub(crate) mod band_tools;
#[cfg(test)]
pub(crate) mod context_refine;
#[cfg(test)]
pub(crate) mod ma_context;
#[cfg(test)]
mod ma_context_tree;

// Retain the tested parallel entry while the public experiment remains serial.
#[allow(dead_code)]
mod parallel;
mod state;
#[cfg(test)]
pub(crate) mod static_model;
#[allow(unused_imports)]
pub(crate) use parallel::{decode_parallel, encode_parallel};
use state::*;
#[cfg(test)]
pub(crate) use static_model::{decode_static, encode_static, encode_static_profile};

type Kernel = fn(&[i64; 64], u64, &mut [u32; 32], &mut [u32; 16]);
#[derive(Clone, Copy)]
struct BandConfig {
    mcu_origin: usize,
    kernel: Kernel,
}
impl Default for BandConfig {
    fn default() -> Self {
        Self {
            mcu_origin: 0,
            kernel: kernels,
        }
    }
}

/// Block dimensions include MCU padding. Coefficients and quantizers use
/// natural (row-major frequency) order, unlike the internal PJA zigzag order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Component {
    pub width: usize,
    pub height: usize,
    pub h_samp: usize,
    pub v_samp: usize,
    pub quant: [u16; 64],
    pub coefficients: Vec<i16>,
}

fn validate(components: &[Component]) -> Result<(usize, usize)> {
    if components.is_empty() || components.len() > 4 {
        return Err(Error::Invalid("coefficient component count"));
    }
    let mut total = 0usize;
    let mut rows = None;
    for c in components {
        if c.width == 0
            || c.height == 0
            || !(1..=4).contains(&c.h_samp)
            || !(1..=4).contains(&c.v_samp)
            || c.height % c.v_samp != 0
            || c.width % c.h_samp != 0
        {
            return Err(Error::Invalid("coefficient component dimensions"));
        }
        let blocks = c
            .width
            .checked_mul(c.height)
            .ok_or(Error::Limit("coefficient blocks"))?;
        if blocks.checked_mul(64) != Some(c.coefficients.len()) {
            return Err(Error::Invalid("coefficient storage length"));
        }
        if c.coefficients.contains(&i16::MIN) {
            return Err(Error::Unsupported("PJA v7 coefficient -32768"));
        }
        if c.quant.contains(&0) {
            return Err(Error::Invalid("zero quantization entry"));
        }
        let shape = (c.width / c.h_samp, c.height / c.v_samp);
        if rows.is_some_and(|r| r != shape) {
            return Err(Error::Invalid("inconsistent MCU dimensions"));
        }
        rows = Some(shape);
        total = total
            .checked_add(blocks)
            .ok_or(Error::Limit("coefficient blocks"))?;
    }
    let mcu_rows = rows.unwrap().1;
    Ok((mcu_rows, source_band_count(total, mcu_rows)))
}

fn source_band_count(total_blocks: usize, mcu_rows: usize) -> usize {
    (total_blocks / 65536).min(16).min(mcu_rows).max(1)
}

pub(crate) fn band_count(components: &[Component]) -> Result<usize> {
    Ok(validate(components)?.1)
}

fn zigzag(components: &[Component], encoding: bool) -> Vec<Vec<i16>> {
    let t = tables();
    components
        .iter()
        .map(|c| {
            let mut out = vec![0; c.coefficients.len()];
            if encoding {
                for (src, dst) in c
                    .coefficients
                    .chunks_exact(64)
                    .zip(out.chunks_exact_mut(64))
                {
                    for k in 0..64 {
                        dst[k] = src[t.zz_table[k] as usize];
                    }
                }
            }
            out
        })
        .collect()
}

pub(crate) fn encode(components: &[Component], max_output: usize) -> Result<Vec<Vec<u8>>> {
    let (rows, bands) = validate(components)?;
    let mut coef = zigzag(components, true);
    let mut streams = Vec::with_capacity(bands);
    let mut remaining = max_output;
    for band in 0..bands {
        let mut cd = Encoder::new(remaining);
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        let stream = cd.finish()?;
        remaining = remaining
            .checked_sub(stream.len())
            .ok_or(Error::Limit("coefficient output"))?;
        streams.push(stream);
    }
    Ok(streams)
}

// Source-compatible zero padding is retained only for original-code oracles.
#[cfg(test)]
pub(crate) fn decode(components: &mut [Component], streams: &[&[u8]]) -> Result<()> {
    decode_inner(components, streams, false)
}
pub(crate) fn decode_strict(components: &mut [Component], streams: &[&[u8]]) -> Result<()> {
    decode_inner(components, streams, true)
}
fn decode_inner(components: &mut [Component], streams: &[&[u8]], strict: bool) -> Result<()> {
    let (rows, bands) = validate(components)?;
    if streams.len() != bands {
        return Err(Error::Invalid("coefficient band count"));
    }
    let mut coef = zigzag(components, false);
    for (band, stream) in streams.iter().enumerate() {
        let mut cd = Decoder::new(stream);
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        if strict {
            cd.finish_check()?;
        }
    }
    let t = tables();
    for (c, src) in components.iter_mut().zip(coef) {
        for (src, dst) in src
            .chunks_exact(64)
            .zip(c.coefficients.chunks_exact_mut(64))
        {
            for k in 0..64 {
                dst[t.zz_table[k] as usize] = src[k];
            }
        }
    }
    Ok(())
}

fn code_band<C: BitCoder>(
    cd: &mut C,
    components: &[Component],
    coef: &mut [Vec<i16>],
    m0: usize,
    m1: usize,
) -> Result<()> {
    code_band_with(cd, components, coef, m0, m1, BandConfig::default())
}
fn code_band_with<C: BitCoder>(
    cd: &mut C,
    components: &[Component],
    coef: &mut [Vec<i16>],
    m0: usize,
    m1: usize,
    config: BandConfig,
) -> Result<()> {
    let mut luma = Model::with_namespace(0);
    let mut chroma = Model::with_namespace(1);
    let mut nz: Vec<Vec<u8>> = vec![Vec::new(); components.len()];
    let mut dcr: Vec<Vec<u8>> = vec![Vec::new(); components.len()];
    for (ci, c) in components.iter().enumerate() {
        let (before, cur_after) = coef.split_at_mut(ci);
        let cur = &mut cur_after[0];
        let prev = if ci >= 2
            && components[ci - 1].width == c.width
            && components[ci - 1].height == c.height
        {
            Some(before[ci - 1].as_slice())
        } else {
            None
        };
        let sx = components[0].h_samp / c.h_samp;
        let sy = components[0].v_samp / c.v_samp;
        let have_luma = ci > 0
            && sx >= 1
            && sy >= 1
            && c.width * sx <= components[0].width
            && c.height * sy <= components[0].height;
        let luma_nz = if have_luma {
            Some(nz[0].as_slice())
        } else {
            None
        };
        let luma_coef = if have_luma && components[0].v_samp.is_multiple_of(c.v_samp) {
            Some(before[0].as_slice())
        } else {
            None
        };
        let prev_dcr = if prev.is_some() {
            Some(dcr[ci - 1].as_slice())
        } else {
            None
        };
        let luma_dcr = if have_luma {
            Some(dcr[0].as_slice())
        } else {
            None
        };
        let mut q = [0i32; 64];
        for (k, value) in q.iter_mut().enumerate() {
            *value = c.quant[tables().zz_table[k] as usize] as i32;
        }
        let (n, d) = component(
            cd,
            if ci == 0 { &mut luma } else { &mut chroma },
            cur,
            c.width,
            c.height,
            &q,
            luma_nz,
            components[0].width,
            sx,
            sy,
            m0 * c.v_samp,
            m1 * c.v_samp,
            luma_coef,
            prev,
            luma_dcr,
            prev_dcr,
            config.mcu_origin * (components[0].v_samp - c.v_samp * sy),
            config.kernel,
        )?;
        nz[ci] = n;
        dcr[ci] = d;
    }
    Ok(())
}

#[derive(Clone, Default)]
struct EdgeCache {
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
    luma_nz: Option<&[u8]>,
    luma_w: usize,
    sx: usize,
    sy: usize,
    y0: usize,
    y1: usize,
    luma_c: Option<&[i16]>,
    prev_c: Option<&[i16]>,
    luma_d: Option<&[u8]>,
    prev_d: Option<&[u8]>,
    luma_row_offset: usize,
    kernel: Kernel,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let t = tables();
    let enc = cd.encoding();
    let mut rq4 = [0f64; 64];
    for k in 0..64 {
        rq4[k] = 1.0 / (181 * q[k] as i64) as f64;
    }
    let mut pth_r = [u32::MAX; 8];
    let mut pth_c = pth_r;
    let mut pth_r2 = pth_r;
    let mut pth_c2 = pth_r;
    for i in 1..8 {
        pth_r[i] = ((11 * 181 * q[t.unzig[i] as usize] as i64 + 1) >> 1) as u32;
        pth_c[i] = ((11 * 181 * q[t.unzig[i * 8] as usize] as i64 + 1) >> 1) as u32;
        pth_r2[i] = ((79 * 181 * q[t.unzig[i] as usize] as i64 + 1) >> 1) as u32;
        pth_c2[i] = ((79 * 181 * q[t.unzig[i * 8] as usize] as i64 + 1) >> 1) as u32;
    }
    let dq0 = 2896 * q[0] as i64;
    let dc5 = 5 * dq0;
    let dc3 = 3 * dq0;
    let rq0 = 1.0 / dq0 as f64;
    let rc5 = 1.0 / dc5 as f64;
    let rc3 = 1.0 / dc3 as f64;
    let mut fb = [0i64; 64];
    let mut row_c = vec![EdgeCache::default(); bw];
    let mut left_c = EdgeCache::default();
    let mut nzc = vec![0u8; bw * bh];
    let mut dcr = nzc.clone();
    let mut rcnt = nzc.clone();
    let mut ccnt = nzc.clone();
    // Table and selector expressions retain the upstream names and dimensions.
    macro_rules! mx { ($w:ident,$sel:expr,$lr:expr,$v:expr; $( $a:ident[$i:expr] ),+ $(,)?) => {
        m.mix(cd, W::$w, $sel as usize, $lr, $v as i32, &[$((B::$a,$i as usize)),+])
    }; }
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
    let nzb = |n: i32| t.nz_table[n as usize] as i32;
    for by in y0..y1 {
        for bx in 0..bw {
            let base = (by * bw + bx) * 64;
            let have_a = by > y0;
            let have_l = bx > 0;
            let abase = if have_a { base - bw * 64 } else { base };
            let lbase = if have_l { base - 64 } else { base };
            // The current block is local so immutable neighboring blocks can be read
            // safely throughout decode; commit only after the full block succeeds.
            let mut b: [i16; 64] = coef[base..base + 64].try_into().unwrap();
            let lb = luma_c.map(|c| &c[((by * sy) * luma_w + bx * sx) * 64..][..64]);
            let cb = prev_c.map(|c| &c[base..base + 64]);
            let nz_a = if have_a {
                nzc[(by - 1) * bw + bx] as i32
            } else {
                0
            };
            let nz_l = if have_l {
                nzc[by * bw + bx - 1] as i32
            } else {
                0
            };
            let ctx = if have_a && have_l {
                nzb((nz_a + nz_l + 1) / 2)
            } else if have_a {
                nzb(nz_a)
            } else if have_l {
                nzb(nz_l)
            } else {
                0
            };
            let za = if have_a { nzb(nz_a) } else { 0 };
            let zb = if have_l { nzb(nz_l) } else { 0 };
            let luma_index = (by * sy)
                .checked_sub(luma_row_offset)
                .and_then(|y| y.checked_mul(luma_w))
                .and_then(|i| i.checked_add(bx * sx));
            let lz = luma_nz.map_or(NZ, |n| {
                nzb(luma_index.and_then(|i| n.get(i)).copied().unwrap_or(0) as i32)
            });
            let nt = if enc {
                (nzmask(&b) & t.interior_zz).count_ones() as i32
            } else {
                0
            };
            let et = bitlen64(nt as i64);
            let al = have_a && have_l;
            let ar = have_a && bx + 1 < bw;
            let nal = if al {
                nzc[(by - 1) * bw + bx - 1] as i32
            } else {
                0
            };
            let nar = if ar {
                nzc[(by - 1) * bw + bx + 1] as i32
            } else {
                0
            };
            let c4 = (if al && ar {
                nzb((nal + nar + 1) / 2)
            } else if al {
                nzb(nal)
            } else if ar {
                nzb(nar)
            } else {
                NZ
            }) * 12
                + cb.map_or(NZ, |c| {
                    nzb((nzmask(c.try_into().unwrap()) & t.interior_zz).count_ones() as i32)
                });
            let mut e = 0;
            while e < 6 {
                if mx!(Nz,e,6,et>e; Nz2[(za*NZ+zb)*64+e+1],Nz3[(lz*NZ+ctx)*64+e+1],Nz4[c4*8+e]) == 0
                {
                    break;
                }
                e += 1;
            }
            let mut n = 0;
            if e > 0 {
                let mut node = 1;
                for k in (0..e - 1).rev() {
                    node = node * 2
                        + avg!(
                            Nzm1[(ctx * 8 + e) * 8 + k],
                            Nzm2[(lz * 8 + e) * 8 + k],
                            (nt >> k) & 1
                        );
                }
                n = node;
            }
            if n > 49 {
                return Err(Error::Invalid("interior nonzero count"));
            }
            nzc[by * bw + bx] = n as u8;
            let nz_b = nzb(n);
            let mut rem = n;
            let mut ssum = 0;
            let mut last_idx = 0i32;
            let mut cm = 0u64;
            for idx in 0..49 {
                if rem == 0 {
                    break;
                }
                let k = t.interior[idx] as usize;
                let ws = 2 * have_a as u32 + al as u32 + ar as u32 + 2 * have_l as u32;
                let nb = if ws == 0 {
                    -1
                } else {
                    nbrmag(
                        if have_a { coef[abase + k] } else { 0 },
                        if al { coef[abase - 64 + k] } else { 0 },
                        if ar { coef[abase + 64 + k] } else { 0 },
                        if have_l { coef[lbase + k] } else { 0 },
                        ws,
                    )
                };
                let ib = t.ib_wa[idx] as i32 * iabs(b[t.ib_na[idx] as usize] as i32)
                    + t.ib_wb[idx] as i32 * iabs(b[t.ib_nb[idx] as usize] as i32);
                let mbi = if t.ib_none[idx] != 0 {
                    19
                } else {
                    clip(finelog(ib as i64), 18)
                };
                let (mbn, mbs) = if nb < 0 {
                    (19, clip(finelog(2 * ib as i64), 18))
                } else {
                    (
                        clip(finelog(nb as i64), 18),
                        clip(finelog((nb + ib) as i64), 18),
                    )
                };
                let kb = t.k_bucket[k] as i32;
                let rb = t.rem_table[rem as usize] as i32;
                let v = if enc { b[k] as i32 } else { 0 };
                let vt = bitlen64(v as i64);
                let pos = t.zz_table[k] as usize;
                let rcn = clip((cm & (0xffu64 << (pos & 56))).count_ones() as i32, 3);
                let ccn = clip(
                    (cm & (0x0101010101010101u64 << (pos & 7))).count_ones() as i32,
                    3,
                );
                let cl = lb.map_or_else(
                    || 12 + clip(finelog(ssum as i64), 15),
                    |l| clip(finelog(l[k] as i64), 11),
                );
                let cc = cb.map_or_else(
                    || 12 + clip(idx as i32 - last_idx, 15),
                    |c| clip(finelog(c[k] as i64), 11),
                );
                let left = t.interior_left[idx] as i32;
                let ratio = if rem < left {
                    t.ratio_of[left as usize][rem as usize] as i32
                } else {
                    16
                };
                let bit = if rem >= left {
                    1
                } else {
                    mx!(Zero,k,4,v!=0; Zero2[(k as i32*MAG+mbi)*4+clip(mbn,3)],Zero4[(kb*MAG+mbs)*REM+rb],ZLum[(k as i32*28+cl)*REM+rb],ZCb[(k as i32*28+cc)*REM+rb])
                };
                if bit != 0 {
                    let mut i = 1;
                    cm |= 1u64 << pos;
                    while i < 15 {
                        if mx!(Expo,((i-1)*KB+kb)*9+ratio/2,6,vt>i; Expo3[(kb*REM+rb)*ROW+i-1],XPos[(k as i32*MAG+mbs)*ROW+i-1],XRow[((kb*4+rcn)*4+ccn)*ROW+i-1],XLum[(kb*28+cl)*ROW+i-1],XCb[(kb*28+cc)*ROW+i-1])
                            == 0
                        {
                            break;
                        }
                        i += 1;
                    }
                    let mut x = 1 << (i - 1);
                    for j in (0..i - 1).rev() {
                        let bt = if j == i - 2 {
                            avg!(
                                Mant1[i * KB + kb],
                                MFirst[(i * MAG + mbs) * 16 + clip(cl, 15)],
                                (iabs(v) >> j) & 1
                            )
                        } else {
                            one!(MRest[kb * 256 + i * 16 + j], (iabs(v) >> j) & 1)
                        };
                        x |= bt << j;
                    }
                    ssum += x;
                    last_idx = idx as i32;
                    let ls = lb.map_or(1, |l| sgn3(l[k] as i64));
                    let cs = cb.map_or(1, |c| sgn3(c[k] as i64));
                    let neg = if ls != 1 || cs != 1 {
                        one!(ISign[(k as i32 * 3 + ls) * 3 + cs], v < 0)
                    } else {
                        cd.direct((v < 0) as i32, 1)
                    };
                    if !enc {
                        b[k] = if neg != 0 { -x as i16 } else { x as i16 };
                    }
                    rem -= 1;
                }
            }
            if rem > 0 {
                return Err(Error::Invalid("unresolved interior nonzero count"));
            }
            let mut fm = 0u64;
            let mut zm = nzmask(&b) & t.interior_zz;
            while zm != 0 {
                let k = zm.trailing_zeros() as usize;
                zm &= zm - 1;
                let pos = t.zz_table[k] as usize;
                fb[pos] = dequant(b[k] as i32, q[k]);
                fm |= 1u64 << pos;
            }
            let mut pn = [0i32; 16];
            let mut pcn = [15i32; 2];
            let mut csum = [0i64; 8];
            let mut rsum = [0i64; 8];
            let mut mm = fm;
            while mm != 0 {
                let p = mm.trailing_zeros() as usize;
                mm &= mm - 1;
                csum[p & 7] += t.w3[p >> 3] * fb[p];
                rsum[p >> 3] += t.w3[p & 7] * fb[p];
            }
            if have_a {
                pcn[0] = 0;
                for w in 0..8 {
                    let v = row_c[bx].eg[w].wrapping_sub(csum[w] as u32) as i32;
                    pn[2 * w] = v;
                    pcn[0] += (v.unsigned_abs() >= pth_r[w]) as i32
                        + (v.unsigned_abs() >= pth_r2[w]) as i32;
                }
            }
            if have_l {
                pcn[1] = 0;
                for u in 0..8 {
                    let v = left_c.eg[u].wrapping_sub(rsum[u] as u32) as i32;
                    pn[2 * u + 1] = v;
                    pcn[1] += (v.unsigned_abs() >= pth_c[u]) as i32
                        + (v.unsigned_abs() >= pth_c2[u]) as i32;
                }
            }
            let mut ecn = [0i32; 2];
            let mut erem = [0i32; 2];
            let mut eleft = [0i32; 2];
            let mut agr = [0i32; 2];
            for edge in 0..2 {
                let (a, l) = if edge == 0 {
                    (
                        if have_a {
                            rcnt[(by - 1) * bw + bx] as i32
                        } else {
                            8
                        },
                        if have_l {
                            rcnt[by * bw + bx - 1] as i32
                        } else {
                            8
                        },
                    )
                } else {
                    (
                        if have_l {
                            ccnt[by * bw + bx - 1] as i32
                        } else {
                            8
                        },
                        if have_a {
                            ccnt[(by - 1) * bw + bx] as i32
                        } else {
                            8
                        },
                    )
                };
                let cbk = if edge == 1 { ecn[0] } else { 8 };
                let et = edge as i32;
                let r1 = if edge == 0 {
                    ((fm >> 8) & 0xfe).count_ones()
                } else {
                    (fm & 0x0202020202020200).count_ones()
                } as i32;
                let ex = if let Some(lb) = lb {
                    let em = if edge == 0 { t.row0zz } else { t.col0zz };
                    2048 + ((et * 9 + (nzmask(lb.try_into().unwrap()) & em).count_ones() as i32)
                        * 18
                        + cb.map_or(9 + r1, |c| {
                            (nzmask(c.try_into().unwrap()) & em).count_ones() as i32
                        }))
                        * 8
                } else {
                    ((et * 8 + r1) * 8 + clip(nz_b, 7)) * 8
                };
                let ct = if enc {
                    (1..8)
                        .filter(|&i| b[t.unzig[if edge == 0 { i } else { i * 8 }] as usize] != 0)
                        .count() as i32
                } else {
                    0
                };
                let mut node = 1;
                for i in (0..3).rev() {
                    let bit = mx!(ECnt,(et*8+node)*NZ+nz_b,6,(ct>>i)&1; Ec1[((et*NZ+nz_b)*9+a)*8+node],Ec2[((et*9+a)*9+l)*8+node],Ec3[((et*9+cbk)*PRED+clip(nz_b,PRED-1))*8+node],EcX[ex+node],Ecp[((et*16+pcn[edge])*NZ+nz_b)*8+node]);
                    node = node * 2 + bit;
                }
                ecn[edge] = node - 8;
                erem[edge] = ecn[edge];
                eleft[edge] = 7;
            }
            rcnt[by * bw + bx] = ecn[0] as u8;
            ccnt[by * bw + bx] = ecn[1] as u8;
            for ep in 0..14usize {
                let edge = ep & 1;
                let (u, w) = if edge == 0 {
                    (0, ep / 2 + 1)
                } else {
                    (ep / 2 + 1, 0)
                };
                if erem[edge] == 0 {
                    continue;
                }
                let k = t.unzig[u * 8 + w] as usize;
                let mut pb = 21;
                let mut pred_f = 0i64;
                let mut pred_q = 0i64;
                if if u == 0 { have_a } else { have_l } {
                    (pred_f, pred_q) = edgepred(pn[ep + 2] as i64, 181 * q[k] as i64, rq4[k]);
                    pb = clip(finelog(div8round(w32(5 * pred_f))), 20);
                }
                let pbq = if pb == 21 {
                    25
                } else {
                    clip(finelog(div8round(w32(4 * pred_q))), 24)
                };
                let a = if have_a {
                    iabs(coef[abase + k] as i32)
                } else {
                    0
                };
                let l = if have_l {
                    iabs(coef[lbase + k] as i32)
                } else {
                    0
                };
                let mbn = if have_a && have_l {
                    clip(bitlen64((a + l) as i64), 18)
                } else if have_a || have_l {
                    clip(bitlen64(2 * (a + l) as i64), 18)
                } else {
                    19
                };
                let prev_b = if ep >= 2 {
                    clip(
                        bitlen64(
                            b[t.unzig[if u == 0 { w - 1 } else { (u - 1) * 8 }] as usize] as i64,
                        ),
                        7,
                    )
                } else {
                    7
                };
                let ib = clip(
                    finelog(b[t.unzig[if u == 0 { 8 + w } else { u * 8 + 1 }] as usize] as i64),
                    11,
                );
                let v = if enc { b[k] as i32 } else { 0 };
                let vt = bitlen64(v as i64);
                let el = lb.map_or_else(
                    || 13 + clip(finelog(ssum as i64), 12),
                    |c| clip(finelog(c[k] as i64), 11),
                );
                let ec = cb.map_or_else(
                    || {
                        12 + if have_a && have_l {
                            clip(finelog(if u == 0 { a } else { l } as i64), 11)
                        } else {
                            12
                        }
                    },
                    |c| clip(finelog(c[k] as i64), 11),
                );
                let p = ep as i32;
                let bit = if erem[edge] >= eleft[edge] {
                    1
                } else {
                    mx!(EZero,p*64+erem[edge]*8+eleft[edge],4,v!=0; Ez6[((p*8+erem[edge])*8+eleft[edge])*4+clip(pb,3)],Ezero4[p*MAG+mbn],EzLum[(p*26+el)*4+clip(pb,3)])
                };
                eleft[edge] -= 1;
                if bit != 0 {
                    erem[edge] -= 1;
                    let mut i = 1;
                    while i < 15 {
                        if mx!(EExpo,((i-1)*14+p)*4+clip(erem[edge],3),10,vt>i; Eexpo5[(p*12+ib)*ROW+i-1],Eexpo1[(p*PQ+pbq)*ROW+i-1],Eexpo2[(p*MAG+mbn)*ROW+i-1],Eexpo3[(pbq*8+prev_b)*ROW+i-1],Eexpo4[(p*NZ+nz_b)*ROW+i-1],ExLum[(p*26+el)*ROW+i-1],ExCb[(p*26+ec)*ROW+i-1])
                            == 0
                        {
                            break;
                        }
                        i += 1;
                    }
                    let mut x = 1 << (i - 1);
                    let pred_mag = if pb == 21 {
                        -1
                    } else {
                        uabs64(pred_f).min(65536) as i32
                    };
                    for j in (0..i - 1).rev() {
                        let c5 = if pred_mag < 0 {
                            4
                        } else {
                            let d = pred_mag - (x | (1 << j));
                            if d < -(1 << j) {
                                0
                            } else if d < 0 {
                                1
                            } else if d < (1 << j) {
                                2
                            } else {
                                3
                            }
                        };
                        let ix = p * 1280 + (i * 16 + j) * 5 + c5;
                        let bt = if j == i - 2 {
                            avg!(Emant1[i * PQ + pbq], EMant[ix], (iabs(v) >> j) & 1)
                        } else {
                            one!(EMant[ix], (iabs(v) >> j) & 1)
                        };
                        x |= bt << j;
                    }
                    let sg = if pb == 21 { 1 } else { sgn3(pred_f) };
                    let (s1, s2) = if let Some(lb) = lb {
                        (sgn3(lb[k] as i64), cb.map_or(1, |c| sgn3(c[k] as i64)))
                    } else {
                        let sa = if have_a {
                            sgn3(coef[abase + k] as i64)
                        } else {
                            1
                        };
                        let sl = if have_l {
                            sgn3(coef[lbase + k] as i64)
                        } else {
                            1
                        };
                        if u == 0 { (sl, sa) } else { (sa, sl) }
                    };
                    let rel = if pb == 21 {
                        4
                    } else {
                        (finelog(4 * x as i64) - pbq + 4).clamp(0, 8)
                    };
                    let neg = avg!(
                        Esr[((edge as i32 * 3 + sg) * PQ + (pbq >> 1)) * 9 + rel],
                        ESgn[((((p * 3 + agr[edge]) * 3 + sg) * 3 + s1) * 3 + s2) * 3
                            + clip(pb, 2)],
                        v < 0
                    );
                    if sg != 1 {
                        agr[edge] = if (neg != 0) == (sg == 0) { 1 } else { 2 };
                    }
                    if !enc {
                        b[k] = if neg != 0 { -x as i16 } else { x as i16 };
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
            kernel(&fb, fm, &mut ka, &mut eg);
            let mut ests = [0i64; 16];
            let mut nest = 0usize;
            if have_l {
                for (i, &kernel) in ka[..8].iter().enumerate() {
                    ests[nest] = left_c.px[i].wrapping_sub(kernel) as i32 as i64 / 4;
                    nest += 1;
                }
            }
            if have_a {
                for (i, &kernel) in ka[8..16].iter().enumerate() {
                    ests[nest] = row_c[bx].px[i].wrapping_sub(kernel) as i32 as i64 / 4;
                    nest += 1;
                }
            }
            let (mut lo, mut hi, mut total) = (0i64, 0i64, 0i64);
            if nest == 16 {
                (lo, hi, total) = mid16(&left_c.px, &row_c[bx].px, &ka);
                total = w32(total + lo + hi);
                nest = 10;
            } else if nest == 8 {
                let mut v: [i64; 8] = ests[..8].try_into().unwrap();
                split8(&mut v);
                lo = v[0].max(v[1]);
                hi = v[6].min(v[7]);
                total = w32(lo + v[2] + v[3] + v[4] + v[5] + hi);
                nest = 6;
            }
            let (mut pred, mut frac, dctx) = if nest > 0 {
                let sg = total >> 63;
                let at = (total ^ sg) - sg;
                let d = if nest == 10 {
                    divfloorr(4 * at, dc5, rc5)
                } else {
                    divfloorr(4 * at, dc3, rc3)
                };
                let pa = (d + 4) >> 3;
                (
                    (pa ^ sg) - sg,
                    (((d - 8 * pa) ^ sg) - sg) as i32 + 4,
                    clip(finelog(divroundr(4 * w32(hi - lo), dq0, rq0)), 26),
                )
            } else {
                (0, 4, 27)
            };
            let d3 = if dctx < 4 { 0 } else { clip((dctx - 4) / 3, 7) };
            if pred > 32767 {
                pred = 32767;
                frac = 4;
            }
            if pred < -32767 {
                pred = -32767;
                frac = 4;
            }
            let mut rb2 = 0;
            let mut nbs = 0;
            let mut nsg = 0;
            if have_a {
                let d = dcr[(by - 1) * bw + bx] as i32;
                rb2 += d & 15;
                nbs += 1;
                nsg += 3 * (d >> 4);
            }
            if have_l {
                let d = dcr[by * bw + bx - 1] as i32;
                rb2 += d & 15;
                nbs += 1;
                nsg += d >> 4;
            }
            rb2 = clip(rb2, 19);
            let rt = if enc { b[0] as i32 - pred as i32 } else { 0 };
            let re = bitlen64(rt as i64);
            let mut res = 0i32;
            let other = prev_d.map(|d| d[by * bw + bx]).or_else(|| {
                if luma_nz.is_some() {
                    luma_d.map(|d| luma_index.and_then(|i| d.get(i)).copied().unwrap_or(0))
                } else {
                    None
                }
            });
            let (dx, ds) = other.map_or((0, 0), |d| (1 + (d & 15) as i32, (d >> 4) as i32));
            if mx!(DcZero,dctx,6,rt!=0; DcZero[dctx],Dzf[rb2*5+iabs(frac-4)],Dz3[nz_b*8+d3],Dz4[dctx*3+nbs],DzX[dctx*17+dx])
                != 0
            {
                let mut i = 1;
                while i < 17 {
                    if mx!(DcExpo,i-1,6,re>i; DcExpo[dctx*ROW+i-1],Dx2[rb2*ROW+i-1],Dx3[(nz_b*8+d3)*ROW+i-1],Dx4[(dctx*3+nbs)*ROW+i-1],DxX[(d3*17+dx)*ROW+i-1])
                        == 0
                    {
                        break;
                    }
                    i += 1;
                }
                let mut x = 1 << (i - 1);
                for j in (0..i - 1).rev() {
                    let bt = if j == i - 2 {
                        one!(DcMant1[i * ACT + dctx], (iabs(rt) >> j) & 1)
                    } else {
                        one!(DcMant[i * ROW + j], (iabs(rt) >> j) & 1)
                    };
                    x |= bt << j;
                }
                let gl = if have_l {
                    sgn3(coef[lbase] as i64 - pred)
                } else {
                    1
                };
                let ga = if have_a {
                    sgn3(coef[abase] as i64 - pred)
                } else {
                    1
                };
                res = if avg!(
                    Dsf[(((dctx >> 1) * 9 + frac) * 3 + gl) * 3 + ga],
                    DsX[(nsg * 9 + frac) * 3 + ds],
                    rt < 0
                ) != 0
                {
                    -x
                } else {
                    x
                };
            }
            dcr[by * bw + bx] = (clip(finelog(res as i64), 15)
                | (((res > 0) as i32 + 2 * (res < 0) as i32) << 4))
                as u8;
            if !enc {
                let dc = pred + res as i64;
                if !(-32767..=32767).contains(&dc) {
                    return Err(Error::Invalid("decoded DC range"));
                }
                b[0] = dc as i16;
            }
            fb[0] = dequant(b[0] as i32, q[0]);
            let f = fb[0] as u32;
            for y in 0..8 {
                left_c.px[y] = ka[16 + y].wrapping_add(t.k4[0][16 + y].wrapping_mul(f));
                row_c[bx].px[y] = ka[24 + y].wrapping_add(t.k4[0][24 + y].wrapping_mul(f));
                left_c.eg[y] = eg[y];
                row_c[bx].eg[y] = eg[8 + y];
            }
            left_c.eg[0] = left_c.eg[0].wrapping_add((t.e3[0] as u32).wrapping_mul(f));
            row_c[bx].eg[0] = row_c[bx].eg[0].wrapping_add((t.e3[0] as u32).wrapping_mul(f));
            let mut mm = fm | 1;
            while mm != 0 {
                fb[mm.trailing_zeros() as usize] = 0;
                mm &= mm - 1;
            }
            coef[base..base + 64].copy_from_slice(&b);
        }
    }
    Ok((nzc, dcr))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn fixture(
        width: usize,
        height: usize,
        h_samp: usize,
        v_samp: usize,
        seed: usize,
    ) -> Component {
        let mut c = Component {
            width,
            height,
            h_samp,
            v_samp,
            quant: std::array::from_fn(|k| 1 + ((k * 7 + seed) % 255) as u16),
            coefficients: vec![0; width * height * 64],
        };
        // Dense, sparse and all-zero blocks, changing residual signs across
        // rows, both axis edges, all interior frequencies, and high magnitudes.
        for (block, b) in c.coefficients.chunks_exact_mut(64).enumerate() {
            b[0] = ((block * 511 + seed * 97) % 3000) as i16 - 1500;
            for (k, coefficient) in b.iter_mut().enumerate().skip(1) {
                if block.is_multiple_of(4) || (block + k + seed).is_multiple_of(5) {
                    let magnitude = (1 + (k * 19 + block * 11 + seed * 23) % 900) as i16;
                    *coefficient = if (block + k + seed).is_multiple_of(2) {
                        magnitude
                    } else {
                        -magnitude
                    };
                }
            }
        }
        c
    }

    fn roundtrip(original: Vec<Component>) {
        let encoded = encode(&original, 1 << 24).unwrap();
        let repeated = encode(&original, 1 << 24).unwrap();
        assert_eq!(encoded, repeated, "source model must be deterministic");
        let mut decoded = original.clone();
        for c in &mut decoded {
            c.coefficients.fill(0);
        }
        let refs: Vec<_> = encoded.iter().map(Vec::as_slice).collect();
        decode(&mut decoded, &refs).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn grayscale_multirow_interior_edges_and_dc() {
        roundtrip(vec![fixture(4, 3, 1, 1, 0)]);
    }

    #[test]
    fn shared_chroma_and_cross_component_contexts() {
        roundtrip(vec![
            fixture(4, 3, 1, 1, 1),
            fixture(4, 3, 1, 1, 2),
            fixture(4, 3, 1, 1, 3),
        ]);
        roundtrip(vec![
            fixture(4, 6, 2, 2, 1),
            fixture(2, 3, 1, 1, 2),
            fixture(2, 3, 1, 1, 3),
        ]);
    }

    #[test]
    fn fractional_sampling_and_missing_previous_component() {
        // Fractional vertical luma mapping disables coefficient dependence,
        // retaining the source's luma count and residual contexts.
        roundtrip(vec![
            fixture(6, 6, 3, 3, 1),
            fixture(4, 4, 2, 2, 2),
            fixture(2, 2, 1, 1, 3),
        ]);
    }

    #[test]
    fn coefficient_extremes_and_all_zero_blocks() {
        let mut c = fixture(3, 2, 1, 1, 0);
        c.quant.fill(u16::MAX);
        c.coefficients.fill(0);
        c.coefficients[0] = 32767;
        c.coefficients[64] = -32767;
        c.coefficients[1] = 32767;
        c.coefficients[8] = -32767;
        c.coefficients[9] = 32767;
        roundtrip(vec![c]);
    }

    #[test]
    fn invalid_layout_and_output_budget_are_errors() {
        assert!(encode(&[], 1024).is_err());
        let c = fixture(2, 2, 1, 1, 0);
        assert!(matches!(
            encode(std::slice::from_ref(&c), 0),
            Err(Error::Limit(_))
        ));
        let mut invalid = c.clone();
        invalid.coefficients.pop();
        assert!(matches!(encode(&[invalid], 1024), Err(Error::Invalid(_))));
        let mut invalid = c.clone();
        invalid.coefficients[0] = i16::MIN;
        assert!(matches!(
            encode(&[invalid], 1024),
            Err(Error::Unsupported(_))
        ));
        let mut invalid = c.clone();
        invalid.quant[0] = 0;
        assert!(matches!(encode(&[invalid], 1024), Err(Error::Invalid(_))));
        assert!(decode(&mut [c], &[]).is_err());
    }

    #[test]
    fn source_band_thresholds_and_independent_band_boundaries() {
        assert_eq!(source_band_count(65535, 100), 1);
        assert_eq!(source_band_count(65536, 100), 1);
        assert_eq!(source_band_count(131072, 100), 2);
        assert_eq!(source_band_count(usize::MAX, 100), 16);
        assert_eq!(source_band_count(usize::MAX, 3), 3);
        // Force three source-shaped MCU bands on a small image, so the test
        // exercises model resets and chroma/luma lookups at every boundary
        // without encoding 131072 blocks merely to cross the band threshold.
        let original = vec![
            fixture(4, 6, 2, 2, 1),
            fixture(2, 3, 1, 1, 2),
            fixture(2, 3, 1, 1, 3),
        ];
        let mut encoded_coef = zigzag(&original, true);
        let mut decoded_coef = zigzag(&original, false);
        for band in 0..3 {
            let mut encoder = Encoder::new(1 << 20);
            code_band(&mut encoder, &original, &mut encoded_coef, band, band + 1).unwrap();
            let stream = encoder.finish().unwrap();
            let mut decoder = Decoder::new(&stream);
            code_band(&mut decoder, &original, &mut decoded_coef, band, band + 1).unwrap();
        }
        assert_eq!(decoded_coef, encoded_coef);
    }
}
