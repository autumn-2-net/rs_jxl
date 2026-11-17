// Translated from PJA (MIT), commit 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
// Sources: pas/pjav7c.c helpers, pas/pjav7c_loop.h kernels and mid16,
// pas/pjamodel.pas InitUnzig and BASIS. See ../LICENSE-PJA and ../NOTICE-PJA.
//! Deterministic integer prediction. Products and accumulators explicitly wrap
//! modulo 2^32, as upstream's -fwrapv build; divisions truncate toward zero.
use std::sync::OnceLock;

pub(crate) fn iabs(v: i32) -> i32 {
    v.wrapping_abs()
}
pub(crate) fn uabs64(v: i64) -> u64 {
    v.unsigned_abs()
}
pub(crate) fn bitlen64(v: i64) -> i32 {
    (64 - v.unsigned_abs().leading_zeros()) as i32
}
pub(crate) fn finelog(v: i64) -> i32 {
    let u = v.unsigned_abs();
    if u <= 1 {
        u as i32
    } else {
        let e = 64 - u.leading_zeros();
        (2 * e - 2 + ((u >> (e - 2)) & 1) as u32) as i32
    }
}
pub(crate) fn clip(v: i32, hi: i32) -> i32 {
    v.min(hi)
}
pub(crate) fn sgn3(v: i64) -> i32 {
    (v > 0) as i32 - (v < 0) as i32 + 1
}
pub(crate) fn w32(v: i64) -> i64 {
    v as i32 as i64
}
pub(crate) fn s32(v: u32) -> i64 {
    v as i32 as i64
}
#[cfg(test)]
fn truncshr(v: i32, n: u32) -> i32 {
    v.wrapping_add((v >> 31) & ((1i32 << n) - 1)) >> n
}
pub(crate) fn div8round(a: i64) -> i64 {
    divroundr(a, 8, 0.125)
}
// Exact integer division replaces the reciprocal plus correction optimization.
// i128 intermediates make the full signed i64 numerator domain well-defined.
pub(crate) fn divroundr(a: i64, b: i64, _rb: f64) -> i64 {
    assert!(b > 0);
    let n = a as i128;
    let q = (n.abs() + b as i128 / 2) / b as i128;
    (if n < 0 { -q } else { q }) as i64
}
pub(crate) fn divfloorr(a: i64, b: i64, _rb: f64) -> i64 {
    assert!(a >= 0 && b > 0);
    a / b
}
pub(crate) fn edgepred(n: i64, b4: i64, _rb4: f64) -> (i64, i64) {
    assert!(b4 > 0);
    let t = 2 * (n as i128).abs() / b4 as i128;
    let sign = if n < 0 { -1 } else { 1 };
    (((t + 4) >> 3) as i64 * sign, ((t + 1) >> 1) as i64 * sign)
}
pub(crate) fn dequant(c: i32, q: i32) -> i64 {
    (c as i64 * q as i64).clamp(-32768, 32768)
}
pub(crate) fn min64(a: i64, b: i64) -> i64 {
    let d = w32(b.wrapping_sub(a));
    a.wrapping_add(d & (d >> 63))
}
pub(crate) fn max64(a: i64, b: i64) -> i64 {
    let d = w32(b.wrapping_sub(a));
    b.wrapping_sub(d & (d >> 63))
}
fn cswap(e: &mut [i64], a: usize, b: usize) {
    let d = w32(e[b].wrapping_sub(e[a]));
    let d = d & (d >> 63);
    e[a] = e[a].wrapping_add(d);
    e[b] = e[b].wrapping_sub(d);
}
pub(crate) fn split16(e: &mut [i64; 16]) {
    for (a, b) in [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (8, 10),
        (9, 11),
        (12, 14),
        (13, 15),
        (1, 2),
        (5, 6),
        (9, 10),
        (13, 14),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (8, 12),
        (9, 13),
        (10, 14),
        (11, 15),
        (2, 4),
        (3, 5),
        (10, 12),
        (11, 13),
        (1, 2),
        (5, 6),
        (0, 8),
        (1, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (5, 13),
        (6, 14),
        (7, 15),
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
        (2, 4),
        (3, 5),
        (10, 12),
        (11, 13),
        (3, 4),
        (11, 12),
    ] {
        cswap(e, a, b)
    }
}
pub(crate) fn split8(e: &mut [i64; 8]) {
    for (a, b) in [
        (0, 1),
        (2, 3),
        (4, 5),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (2, 4),
        (3, 5),
        (1, 2),
        (5, 6),
    ] {
        cswap(e, a, b)
    }
}
pub(crate) fn nzmask(b: &[i16; 64]) -> u64 {
    b.iter()
        .enumerate()
        .fold(0, |m, (k, &v)| m | ((v != 0) as u64) << k)
}
pub(crate) fn nbrmag(a: i16, al: i16, ar: i16, l: i16, ws: u32) -> i32 {
    if ws == 0 {
        return 0;
    }
    let sum = 2 * (a as i32).unsigned_abs()
        + (al as i32).unsigned_abs()
        + (ar as i32).unsigned_abs()
        + 2 * (l as i32).unsigned_abs();
    ((sum * 2 + ws / 2) / ws) as i32
}

pub(crate) const BASIS: [[i64; 8]; 8] = [
    [2896, 2896, 2896, 2896, 2896, 2896, 2896, 2896],
    [4017, 3406, 2276, 799, -799, -2276, -3406, -4017],
    [3784, 1567, -1567, -3784, -3784, -1567, 1567, 3784],
    [3406, -799, -4017, -2276, 2276, 4017, 799, -3406],
    [2896, -2896, -2896, 2896, 2896, -2896, -2896, 2896],
    [2276, -4017, 799, 3406, -3406, -799, 4017, -2276],
    [1567, -3784, 3784, -1567, -1567, 3784, -3784, 1567],
    [799, -2276, 3406, -4017, 4017, -3406, 2276, -799],
];
pub(crate) const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
pub(crate) struct Tables {
    pub unzig: [u8; 64],
    pub zz_table: [u8; 64],
    pub interior: [u8; 49],
    pub interior_left: [u8; 49],
    pub ib_na: [u8; 49],
    pub ib_nb: [u8; 49],
    pub ib_wa: [u8; 49],
    pub ib_wb: [u8; 49],
    pub ib_none: [u8; 49],
    pub ratio_of: [[u8; 49]; 50],
    pub nz_table: [u8; 64],
    pub rem_table: [u8; 64],
    pub k_bucket: [u8; 64],
    pub w3: [i64; 8],
    pub e3: [i64; 8],
    pub k4: [[u32; 32]; 64],
    pub interior_zz: u64,
    pub edge_zz: u64,
    pub row0zz: u64,
    pub col0zz: u64,
}
pub(crate) fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            unzig: [0; 64],
            zz_table: ZIGZAG,
            interior: [0; 49],
            interior_left: [0; 49],
            ib_na: [0; 49],
            ib_nb: [0; 49],
            ib_wa: [0; 49],
            ib_wb: [0; 49],
            ib_none: [0; 49],
            ratio_of: [[0; 49]; 50],
            nz_table: [0; 64],
            rem_table: [0; 64],
            k_bucket: [0; 64],
            w3: [0; 8],
            e3: [0; 8],
            k4: [[0; 32]; 64],
            interior_zz: 0,
            edge_zz: 0,
            row0zz: 0,
            col0zz: 0,
        };
        for (k, &position) in ZIGZAG.iter().enumerate() {
            t.unzig[position as usize] = k as u8;
            t.nz_table[k] = match k {
                0..=4 => k as u8,
                5..=6 => 5,
                7..=8 => 6,
                9..=12 => 7,
                13..=18 => 8,
                19..=27 => 9,
                _ => 10,
            };
            t.rem_table[k] = match k {
                1..=4 => (k - 1) as u8,
                5..=6 => 4,
                7..=9 => 5,
                10..=14 => 6,
                15..=22 => 7,
                _ => 8,
            };
            t.k_bucket[k] = match k {
                0..=1 => 0,
                2 => 1,
                3 => 2,
                4..=5 => 3,
                6..=9 => 4,
                10..=14 => 5,
                15..=20 => 6,
                21..=27 => 7,
                28..=35 => 8,
                36..=44 => 9,
                _ => 10,
            };
        }
        let mut n = 0;
        for (k, &position) in ZIGZAG.iter().enumerate().skip(1) {
            let p = position as usize;
            if p / 8 == 0 || p.is_multiple_of(8) {
                t.edge_zz |= 1u64 << k;
            } else {
                t.interior[n] = k as u8;
                n += 1;
                t.interior_zz |= 1u64 << k;
            }
        }
        for k in 0..49 {
            t.interior_left[k] = (49 - k) as u8;
            let p = ZIGZAG[t.interior[k] as usize] as usize;
            let (u, w) = (p / 8, p % 8);
            if u > 1 && w > 1 {
                t.ib_na[k] = t.unzig[p - 8];
                t.ib_nb[k] = t.unzig[p - 1];
                t.ib_wa[k] = 1;
                t.ib_wb[k] = 1;
            } else if u > 1 {
                t.ib_na[k] = t.unzig[p - 8];
                t.ib_wa[k] = 2;
            } else if w > 1 {
                t.ib_na[k] = t.unzig[p - 1];
                t.ib_wa[k] = 2;
            } else {
                t.ib_none[k] = 1;
            }
        }
        for k in 1..50 {
            for n in 0..49 {
                if n < k {
                    t.ratio_of[k][n] = (n * 16 / k) as u8;
                }
            }
        }
        let mut w5 = [0; 8];
        let mut e5 = [0; 8];
        for (k, basis) in BASIS.iter().enumerate() {
            t.w3[k] = (3 * basis[0] - basis[1]) / 8;
            t.e3[k] = (3 * basis[7] - basis[6]) / 8;
            w5[k] = 5 * basis[0] - basis[1];
            e5[k] = 5 * basis[7] - basis[6];
        }
        for p in 0..64 {
            let (u, v) = (p / 8, p % 8);
            for (k, (&bu, &bv)) in BASIS[u].iter().zip(&BASIS[v]).enumerate() {
                for (off, value) in [
                    (0, bu * w5[v]),
                    (8, bv * w5[u]),
                    (16, bu * e5[v]),
                    (24, bv * e5[u]),
                ] {
                    t.k4[p][off + k] = divroundr(value, 2896, 0.0) as u32;
                }
            }
        }
        for i in 1..8 {
            t.row0zz |= 1u64 << t.unzig[i];
            t.col0zz |= 1u64 << t.unzig[i * 8];
        }
        t
    })
}
pub(crate) fn kernels_scalar(
    fb: &[i64; 64],
    mut mask: u64,
    acc: &mut [u32; 32],
    eg: &mut [u32; 16],
) {
    let t = tables();
    acc.fill(0);
    while mask != 0 {
        let p = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let f = fb[p] as u32;
        for (y, value) in acc.iter_mut().enumerate() {
            *value = value.wrapping_add(t.k4[p][y].wrapping_mul(f));
        }
        eg[p / 8] = eg[p / 8].wrapping_add((t.e3[p % 8] as u32).wrapping_mul(f));
        eg[8 + p % 8] = eg[8 + p % 8].wrapping_add((t.e3[p / 8] as u32).wrapping_mul(f));
    }
}
pub(crate) fn kernels(fb: &[i64; 64], mask: u64, acc: &mut [u32; 32], eg: &mut [u32; 16]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection guards the target-feature function. unsafe is confined to its bounded loads and stores.
        unsafe { kernels_avx2(fb, mask, acc, eg) };
        return;
    }
    kernels_scalar(fb, mask, acc, eg)
}
pub(crate) fn mid16(left: &[u32; 8], top: &[u32; 8], ka: &[u32; 32]) -> (i64, i64, i64) {
    let mut e = [0; 16];
    for y in 0..8 {
        e[y] = s32(left[y].wrapping_sub(ka[y])) / 4;
        e[y + 8] = s32(top[y].wrapping_sub(ka[y + 8])) / 4;
    }
    split16(&mut e);
    let lo = e[..4].iter().copied().fold(e[0], max64);
    let hi = e[12..].iter().copied().fold(e[12], min64);
    (lo, hi, w32(e[4..12].iter().sum()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wrap_and_round() {
        assert_eq!(w32(0x80000000), -2147483648);
        assert_eq!(w32(0x100000001), 1);
        for a in -1000..=1000 {
            assert_eq!(
                div8round(a),
                if a < 0 { -((-a + 4) / 8) } else { (a + 4) / 8 }
            );
            assert_eq!(truncshr(a as i32, 3), a as i32 / 8);
        }
        assert_eq!(finelog(i64::MIN), 126);
        assert_eq!(bitlen64(i64::MIN), 64);
    }
    #[test]
    fn deterministic_tables() {
        let t = tables();
        assert_eq!(t.interior_zz.count_ones(), 49);
        assert_eq!(t.edge_zz.count_ones(), 14);
        assert_eq!(t.interior_zz | t.edge_zz, u64::MAX - 1);
        for k in 0..64 {
            assert_eq!(t.unzig[t.zz_table[k] as usize], k as u8);
        }
        assert_eq!(t.w3[0], 724);
        assert_eq!(t.e3[0], 724);
    }
    #[test]
    fn splits() {
        let mut state = 42u32;
        for _ in 0..1000 {
            let mut e = [0; 16];
            for x in &mut e {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *x = (state as i32 / 4) as i64;
            }
            let mut sorted = e;
            sorted.sort();
            split16(&mut e);
            let mut low = e[..4].to_vec();
            low.sort();
            let mut high = e[12..].to_vec();
            high.sort();
            assert_eq!(low, sorted[..4]);
            assert_eq!(high, sorted[12..]);
            let mut eight = [0; 8];
            eight.copy_from_slice(&sorted[4..12]);
            eight.reverse();
            split8(&mut eight);
            assert_eq!(eight[..2].iter().max(), Some(&sorted[5]));
            assert_eq!(eight[6..].iter().min(), Some(&sorted[10]));
        }
    }
    #[test]
    fn masks_and_kernel_wrapping() {
        let mut b = [0i16; 64];
        b[0] = i16::MIN;
        b[63] = 1;
        assert_eq!(nzmask(&b), 1 | (1 << 63));
        let mut f = [0; 64];
        f[0] = i64::MAX;
        let mut acc = [0; 32];
        let mut eg = [3; 16];
        kernels(&f, 1, &mut acc, &mut eg);
        for (y, &value) in acc.iter().enumerate() {
            assert_eq!(value, tables().k4[0][y].wrapping_mul(u32::MAX));
        }
        assert_eq!(eg[0], 3u32.wrapping_add(724u32.wrapping_mul(u32::MAX)));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn kernels_avx2(fb: &[i64; 64], mut mask: u64, acc: &mut [u32; 32], eg: &mut [u32; 16]) {
    use std::arch::x86_64::*;
    let t = tables();
    let mut lanes = [_mm256_setzero_si256(); 4];
    while mask != 0 {
        let p = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let f = fb[p] as u32;
        let fv = _mm256_set1_epi32(f as i32);
        for (i, lane) in lanes.iter_mut().enumerate() {
            // SAFETY: each load reads eight initialized u32 values within the 32-lane
            // row; unaligned loads impose no alignment requirement. AVX2 is detected
            // by the sole production caller before entering this target-feature function.
            let k = unsafe { _mm256_loadu_si256(t.k4[p].as_ptr().add(i * 8).cast()) };
            *lane = _mm256_add_epi32(*lane, _mm256_mullo_epi32(k, fv));
        }
        eg[p / 8] = eg[p / 8].wrapping_add((t.e3[p % 8] as u32).wrapping_mul(f));
        eg[8 + p % 8] = eg[8 + p % 8].wrapping_add((t.e3[p / 8] as u32).wrapping_mul(f));
    }
    for (i, lane) in lanes.into_iter().enumerate() {
        // SAFETY: each unaligned store writes eight u32s within the output array.
        unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(i * 8).cast(), lane) };
    }
}
#[cfg(all(test, target_arch = "x86_64"))]
#[test]
fn avx2_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut state = 123u64;
    for _ in 0..100 {
        let mut f = [0; 64];
        for x in &mut f {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = state as i64;
        }
        let mask = state;
        let (mut a, mut b) = ([0; 32], [0; 32]);
        let (mut x, mut y) = ([u32::MAX; 16], [u32::MAX; 16]);
        kernels_scalar(&f, mask, &mut a, &mut x);
        unsafe { kernels_avx2(&f, mask, &mut b, &mut y) };
        assert_eq!(a, b);
        assert_eq!(x, y);
    }
}
