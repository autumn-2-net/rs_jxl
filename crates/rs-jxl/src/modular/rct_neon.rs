// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! AArch64 translation of enc_rct.cc::FwdRctRow at
//! b866b3defec13494bfbaf705db93581cda118e37. Included only on AArch64.
//!
//! Four signed-i32 lanes match Highway NEON. Adds/subtracts wrap at 32 bits,
//! then signed shifts round negative odd values toward minus infinity. No
//! floating-point conversion, saturating arithmetic, or widening average is
//! used, including for raw floating-point sample words supplied as i32.

use std::arch::aarch64::*;

/// Return the processed prefix; the parent handles the remaining 0..3 pixels.
/// Requires NEON, T <= 6, disjoint equal-length rows and a permutation of 0..3.
#[target_feature(enable = "neon")]
pub(super) unsafe fn run<const T: u32>(rows: &mut [&mut [i32]; 3], order: [usize; 3]) -> usize {
    debug_assert!(T <= 6);
    let end = rows[0].len() / 4 * 4;
    let output = [
        rows[0].as_mut_ptr(),
        rows[1].as_mut_ptr(),
        rows[2].as_mut_ptr(),
    ];
    let input = order.map(|channel| output[channel].cast_const());
    // SAFETY: dispatch checks NEON and validates the row/order contract.
    // Unaligned loads/stores cover only full four-element vectors. Permuted
    // inputs alias outputs, so load all three before storing any of them.
    unsafe {
        for x in (0..end).step_by(4) {
            let a = vld1q_s32(input[0].add(x));
            let b = vld1q_s32(input[1].add(x));
            let c = vld1q_s32(input[2].add(x));
            let (a, b, c) = if T == 6 {
                let co = vsubq_s32(a, c);
                let tmp = vaddq_s32(c, vshrq_n_s32::<1>(co));
                let cg = vsubq_s32(b, tmp);
                (vaddq_s32(tmp, vshrq_n_s32::<1>(cg)), co, cg)
            } else {
                let b = if T >> 1 == 1 {
                    vsubq_s32(b, a)
                } else if T >> 1 == 2 {
                    vsubq_s32(b, vshrq_n_s32::<1>(vaddq_s32(a, c)))
                } else {
                    b
                };
                let c = if T & 1 != 0 { vsubq_s32(c, a) } else { c };
                (a, b, c)
            };
            vst1q_s32(output[0].add(x), a);
            vst1q_s32(output[1].add(x), b);
            vst1q_s32(output[2].add(x), c);
        }
    }
    end
}
