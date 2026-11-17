// PJA MIT translation; see ../LICENSE-PJA and ../NOTICE-PJA.
//! Scalar pjav7c_loop.h one/avg2/mixn and pjamodel.pas InitTables,
//! pinned at 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.

use crate::arith::{AdaptiveBit, BitCoder};
#[path = "probability_tables.rs"]
mod frozen;

pub struct ProbabilityTables {
    pub squash: [u16; 4096],
    pub stretch: [i16; 4096],
}
static TABLES: ProbabilityTables = ProbabilityTables {
    squash: frozen::SQUASH,
    stretch: frozen::STRETCH,
};

/// Frozen tables avoid platform-dependent floating-point exp/round at runtime.
pub fn tables() -> &'static ProbabilityTables {
    &TABLES
}

pub fn one<C: BitCoder>(coder: &mut C, counter: &mut AdaptiveBit, value: i32) -> i32 {
    let bit = coder.bit(counter.p as u32, value);
    counter.update(bit);
    bit
}

pub fn avg2<C: BitCoder>(
    coder: &mut C,
    a: &mut AdaptiveBit,
    b: &mut AdaptiveBit,
    value: i32,
) -> i32 {
    let stretch = &tables().stretch;
    // Rust signed division truncates toward zero, like Pascal div and C /.
    let mean = (stretch[a.p as usize >> 4] as i32 + stretch[b.p as usize >> 4] as i32) / 2;
    let bit = coder.bit((TABLES.squash[(mean + 2048) as usize] as u32) << 4, value);
    a.update(bit);
    b.update(bit);
    bit
}

/// Version-7 mixer: 3..=7 adaptive counters, then the constant input 384.
/// The caller supplies weights initialized by the model's V7Weights rule.
/// Retained as a slice/reference implementation for equivalence tests.
#[cfg(test)]
pub fn mixn<C: BitCoder>(
    coder: &mut C,
    weights: &mut [i32; 8],
    counters: &mut [&mut AdaptiveBit],
    lr: i32,
    value: i32,
) -> i32 {
    assert!(
        (3..=7).contains(&counters.len()),
        "PJA v7 mixer counter count"
    );
    let n = counters.len();
    let mut inputs = [0i32; 8];
    for (input, counter) in inputs.iter_mut().zip(counters.iter()) {
        *input = TABLES.stretch[counter.p as usize >> 4] as i32;
    }
    inputs[n] = 384;
    // Explicit wrapping matches the source's integer SIMD operations, including
    // debug builds. Valid format model weights/rates keep these products in range.
    let dot = crate::mix_simd::v7_dot(&inputs, weights);
    let probability = TABLES.squash[((dot / 65536).clamp(-2048, 2047) + 2048) as usize] as i32;
    let bit = coder.bit((probability as u32) << 4, value);
    let error = ((bit << 12) - probability).wrapping_mul(lr);
    // Padding inputs and weights remain zero. Updating all eight lanes is
    // identical to the active prefix and avoids scalar tails for 4..7 lanes.
    crate::mix_simd::v7_update(weights, &inputs, error);
    for counter in counters {
        counter.update(bit);
    }
    bit
}

/// The v7 call sites have fixed expert counts. Preserve that information
/// through the learning loop instead of materializing a slice of mutable
/// references and rechecking its dynamic length at every coded bit.
#[inline]
pub fn mix_array<C: BitCoder, const N: usize>(
    coder: &mut C,
    weights: &mut [i32; 8],
    counters: &mut [AdaptiveBit; N],
    lr: i32,
    value: i32,
) -> i32 {
    assert!((3..=7).contains(&N));
    let mut inputs = [0i32; 8];
    for i in 0..N {
        inputs[i] = TABLES.stretch[counters[i].p as usize >> 4] as i32;
    }
    inputs[N] = 384;
    let dot = crate::mix_simd::v7_dot(&inputs, weights);
    let probability = TABLES.squash[((dot / 65536).clamp(-2048, 2047) + 2048) as usize] as i32;
    let bit = coder.bit((probability as u32) << 4, value);
    let error = ((bit << 12) - probability).wrapping_mul(lr);
    crate::mix_simd::v7_update(weights, &inputs, error);
    for counter in counters {
        counter.update(bit);
    }
    bit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::{Decoder, Encoder};
    #[test]
    fn fixed_experts_match_reference_learning_and_stream() {
        fn compare<const N: usize>() {
            let mut old_weights = [0; 8];
            old_weights[..N].fill(65536 * 4 / (3 * (N as i32 + 1)));
            let mut weights = old_weights;
            let mut old_counters = [AdaptiveBit::default(); N];
            let mut counters = old_counters;
            let mut old_encoder = Encoder::new(10000);
            let mut encoder = Encoder::new(10000);
            for i in 0..4096 {
                let bit = ((i * 19 + i / 13) % 71 < 29) as i32;
                let mut refs: Vec<_> = old_counters.iter_mut().collect();
                mixn(
                    &mut old_encoder,
                    &mut old_weights,
                    &mut refs,
                    [4, 6, 10][i % 3],
                    bit,
                );
                mix_array(
                    &mut encoder,
                    &mut weights,
                    &mut counters,
                    [4, 6, 10][i % 3],
                    bit,
                );
                assert_eq!(weights, old_weights);
                assert_eq!(counters, old_counters);
            }
            assert_eq!(encoder.finish().unwrap(), old_encoder.finish().unwrap());
        }
        compare::<3>();
        compare::<4>();
        compare::<5>();
        compare::<6>();
        compare::<7>();
    }
    #[test]
    fn table_definition_and_inverse() {
        let t = tables();
        assert_eq!(
            (t.squash[0], t.squash[2048], t.squash[4095]),
            (1, 2048, 4095)
        );
        assert_eq!(t.stretch[0], t.stretch[1]);
        for p in 1..4096 {
            let i = (t.stretch[p] as i32 + 2048) as usize;
            assert!(t.squash[i] as usize >= p);
            if i > 0 {
                assert!((t.squash[i - 1] as usize) < p);
            }
        }
    }
    #[test]
    fn mixed_roundtrip_all_lane_counts() {
        for n in 3..=7 {
            let mut weights = [0; 8];
            weights[..=n].fill(65536 / (n as i32 + 1));
            let initial = weights;
            let mut counters = [AdaptiveBit::default(); 7];
            let mut encoder = Encoder::new(10000);
            for i in 0..4096 {
                let b = ((i * 19 + i / 13) % 71 < 29) as i32;
                let mut refs: Vec<_> = counters[..n].iter_mut().collect();
                mixn(&mut encoder, &mut weights, &mut refs, [4, 6, 10][i % 3], b);
            }
            let expected_weights = weights;
            let expected_counters = counters;
            let bytes = encoder.finish().unwrap();
            let mut decoder = Decoder::new(&bytes);
            weights = initial;
            counters.fill(AdaptiveBit::default());
            for i in 0..4096 {
                let mut refs: Vec<_> = counters[..n].iter_mut().collect();
                assert_eq!(
                    mixn(&mut decoder, &mut weights, &mut refs, [4, 6, 10][i % 3], 0),
                    ((i * 19 + i / 13) % 71 < 29) as i32
                );
            }
            assert_eq!(weights, expected_weights);
            assert_eq!(counters, expected_counters);
        }
    }
}
