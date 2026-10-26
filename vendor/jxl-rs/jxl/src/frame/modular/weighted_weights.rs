// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Exact weight calculation for the parent predict.rs.
use super::DIVLOOKUP;

#[inline(always)]
const fn scalar_weight(error: u32, weight: u8) -> u32 {
    let shift = (error as u64 + 1).ilog2().saturating_sub(5);
    // Even u8::MAX * DIVLOOKUP[0] + 4 fits in u32.
    4 + ((weight as u32 * DIVLOOKUP[(error >> shift) as usize]) >> shift)
}

// Exact common-case values, initialized at compile time and shared (16 KiB).
static SMALL_WEIGHTS: [[u32; 256]; 16] = {
    let mut table = [[0; 256]; 16];
    let mut weight = 0;
    while weight < 16 {
        let mut error = 0;
        while error < 256 {
            table[weight][error] = scalar_weight(error as u32, weight as u8);
            error += 1;
        }
        weight += 1;
    }
    table
};

#[inline(always)]
pub(super) fn weights(errors: [u32; 4], weights: [u8; 4]) -> [u32; 4] {
    let [e0, e1, e2, e3] = errors;
    let [w0, w1, w2, w3] = weights;
    if (e0 | e1 | e2 | e3) < 256 && (w0 | w1 | w2 | w3) < 16 {
        [
            SMALL_WEIGHTS[w0 as usize][e0 as usize],
            SMALL_WEIGHTS[w1 as usize][e1 as usize],
            SMALL_WEIGHTS[w2 as usize][e2 as usize],
            SMALL_WEIGHTS[w3 as usize][e3 as usize],
        ]
    } else {
        [
            scalar_weight(e0, w0),
            scalar_weight(e1, w1),
            scalar_weight(e2, w2),
            scalar_weight(e3, w3),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::weights;

    fn reference_weight(error: u32, weight: u8) -> u32 {
        // Independent shift selection and reciprocal calculation. Preserve the
        // reciprocal's truncation before multiplying, as in the original formula.
        let mut scale = 1u64;
        while u64::from(error) + 1 >= 64 * scale {
            scale *= 2;
        }
        let index = u64::from(error) / scale;
        let reciprocal = (1u64 << 24) / (index + 1);
        (4 + u64::from(weight) * reciprocal / scale) as u32
    }

    fn check(errors: [u32; 4], input_weights: [u8; 4]) {
        let expected: [u32; 4] =
            std::array::from_fn(|i| reference_weight(errors[i], input_weights[i]));
        assert_eq!(
            weights(errors, input_weights),
            expected,
            "errors={errors:?}, weights={input_weights:?}"
        );
    }

    #[test]
    fn exhaustive_errors_below_1024_all_u8_weights() {
        // Errors 256..1024 cover the scalar fallback, not additional lookup entries.
        for error in 0..1024 {
            for weight in 0..=u8::MAX {
                check([error; 4], [weight; 4]);
            }
        }
    }

    #[test]
    fn mixed_errors_below_1024_match_exact_formula() {
        for error in 0..1024 {
            for weight in 0..16 {
                check(
                    [error, 1023 - error, error ^ 0x155, error ^ 0x2aa],
                    [weight, (weight + 1) & 15, weight ^ 5, weight ^ 10],
                );
            }
        }
    }

    #[test]
    fn error_boundaries_in_each_lane_match_exact_formula() {
        for lane in 0..4 {
            for error in [255, 256, 257, 1022, 1023, 1024, 1025] {
                for weight in 0..=u8::MAX {
                    let mut errors = [0, 255, 256, 1023];
                    let mut input_weights = [0, 7, 13, 15];
                    errors[lane] = error;
                    input_weights[lane] = weight;
                    check(errors, input_weights);
                }
            }
        }
    }

    #[test]
    fn fallback_errors_and_weights_in_each_lane() {
        for lane in 0..4 {
            for error in [1024, 1025, 1 << 16, u32::MAX] {
                let mut errors = [0, 255, 256, 1023];
                errors[lane] = error;
                check(errors, [0, 7, 13, 15]);
            }
            for weight in 16..=u8::MAX {
                let mut input_weights = [0, 7, 13, 15];
                input_weights[lane] = weight;
                check([0, 255, 256, 1023], input_weights);
            }
        }
    }

    #[test]
    fn exponent_boundaries_and_u32_max_all_weights() {
        for exponent in 0..=32 {
            let boundary = 1u64 << exponent;
            let start = boundary.saturating_sub(2);
            let end = (boundary + 1).min(u64::from(u32::MAX));
            for error in start..=end {
                for weight in 0..=u8::MAX {
                    check([error as u32; 4], [weight; 4]);
                }
            }
        }
        check([0, 1, u32::MAX - 1, u32::MAX], [255, 0, 16, 15]);
    }

    #[test]
    fn wrapping_sum_errors() {
        let terms = [
            0,
            1,
            31,
            32,
            63,
            64,
            127,
            255,
            256,
            257,
            1023,
            1024,
            1025,
            1 << 30,
            1 << 31,
            u32::MAX - 1,
            u32::MAX,
        ];
        for north in terms {
            for northeast in terms {
                for northwest in terms {
                    let error = north.wrapping_add(northeast).wrapping_add(northwest);
                    assert_eq!(
                        error,
                        (u64::from(north) + u64::from(northeast) + u64::from(northwest)) as u32
                    );
                    check([error; 4], [0, 7, 13, 15]);
                    check(
                        [
                            error,
                            north.wrapping_add(northeast),
                            northeast.wrapping_add(northwest),
                            northwest.wrapping_add(north),
                        ],
                        [16, 127, 254, 255],
                    );
                }
            }
        }
    }

    #[test]
    fn deterministic_random_full_range_and_mixed_paths() {
        fn next(state: &mut u32) -> u32 {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            *state
        }

        let mut state = 0x6a09_e667;
        for case in 0..65_536 {
            let mut errors = std::array::from_fn(|_| next(&mut state));
            let mut input_weights = std::array::from_fn(|_| (next(&mut state) >> 24) as u8);
            if case & 1 != 0 {
                errors = errors.map(|error| error & 1023);
            }
            if case & 2 != 0 {
                input_weights = input_weights.map(|weight| weight & 15);
            }
            check(errors, input_weights);
        }
    }
}
