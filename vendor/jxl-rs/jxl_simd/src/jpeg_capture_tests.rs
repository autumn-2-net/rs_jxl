// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Case {
    raw: [i32; 64],
    luma: [i32; 64],
    scales: [i32; 64],
}

impl Case {
    fn zero_ac() -> Self {
        let mut case = Self {
            raw: [0; 64],
            luma: [0; 64],
            scales: [0; 64],
        };
        case.raw[0] = i32::MIN;
        case.luma[0] = i32::MAX;
        case.scales[0] = i32::MIN;
        case
    }
}

fn supported() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
}

// Work in i64 and use Euclidean division, independently of SIMD multiply/shift
// and transpose/pack operations. Only a fully valid block is returned.
fn oracle<const CFL: bool>(case: &Case, mut output: [i16; 64]) -> Option<[i16; 64]> {
    for row in 0..8 {
        for col in 0..8 {
            let destination = row * 8 + col;
            if destination == 0 {
                continue;
            }
            let source = col * 8 + row;
            let raw = i64::from(case.raw[source]);
            let mut recovered = raw;
            if CFL {
                let luma = i64::from(case.luma[source]);
                let scale = i64::from(case.scales[destination]);
                if !(-65_535..=65_535).contains(&raw)
                    || !(-4_095..=4_095).contains(&luma)
                    || !(-524_288..=524_288).contains(&scale)
                {
                    return None;
                }
                recovered += (luma * scale + 1_024).div_euclid(2_048);
            }
            if !(-4_095..=4_095).contains(&recovered) {
                return None;
            }
            output[destination] = recovered as i16;
        }
    }
    Some(output)
}

#[repr(align(64))]
#[derive(Clone)]
struct GuardedOutput([i16; 128]);

fn check_case<const CFL: bool>(case: &Case, offset: usize, dc: i16) -> Option<[i16; 64]> {
    assert!(supported());
    assert!(offset < 32);
    let start = 32 + offset;
    let end = start + 64;
    let mut initial = GuardedOutput(std::array::from_fn(|i| {
        (i as i16).wrapping_mul(997).wrapping_add(-24_111)
    }));
    initial.0[start] = dc;
    let before: [i16; 64] = initial.0[start..end].try_into().unwrap();
    let expected = oracle::<CFL>(case, before);
    let mut expected_storage = initial.clone();
    if let Some(values) = expected {
        expected_storage.0[start..end].copy_from_slice(&values);
    }
    let mut avx2 = initial.clone();
    let mut avx512 = initial;
    assert_eq!(avx2.0[start..].as_ptr() as usize % 64, offset * 2);
    assert_eq!(avx512.0[start..].as_ptr() as usize % 64, offset * 2);
    // SAFETY: all three CPU features were checked above. Each input has 64
    // elements, and both possibly unaligned destination slices have length 64.
    let (ok2, ok512) = unsafe {
        (
            super::x86::capture::<CFL>(
                &case.raw,
                &case.luma,
                &case.scales,
                &mut avx2.0[start..end],
            ),
            super::x86::capture_avx512::<CFL>(
                &case.raw,
                &case.luma,
                &case.scales,
                &mut avx512.0[start..end],
            ),
        )
    };
    assert_eq!(
        (ok2, ok512),
        (expected.is_some(), expected.is_some()),
        "status: CFL={CFL}, offset={offset}, {case:?}"
    );
    // Compare the entire allocation: DC, every AC, and both guard regions.
    // On rejection expected_storage is the complete pre-call snapshot.
    assert_eq!(
        avx2.0, expected_storage.0,
        "AVX2 output: CFL={CFL}, offset={offset}, {case:?}"
    );
    assert_eq!(
        avx512.0, expected_storage.0,
        "AVX512 output: CFL={CFL}, offset={offset}, {case:?}"
    );
    ok512.then(|| avx512.0[start..end].try_into().unwrap())
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn signed(&mut self, bound: i32) -> i32 {
        (self.next() % (2 * bound as u64 + 1)) as i32 - bound
    }
}

fn valid_case<const CFL: bool>(rng: &mut Rng) -> (Case, [i16; 64]) {
    let mut case = Case::zero_ac();
    case.raw[0] = rng.next() as i32;
    case.luma[0] = rng.next() as i32;
    case.scales[0] = rng.next() as i32;
    let mut expected = [0; 64];
    for (k, value) in expected.iter_mut().enumerate().skip(1) {
        *value = rng.signed(4_095) as i16;
        let source = (k % 8) * 8 + k / 8;
        if CFL {
            // Both envelopes are explored; these paired bounds keep the
            // residual in range for every independently chosen final value.
            let (luma, scale) = if rng.next() & 1 == 0 {
                (rng.signed(4_095), rng.signed(30_720))
            } else {
                (rng.signed(239), rng.signed(524_288))
            };
            case.luma[source] = luma;
            case.scales[k] = scale;
            let prediction = (i64::from(luma) * i64::from(scale) + 1_024) >> 11;
            case.raw[source] = (i64::from(*value) - prediction) as i32;
        } else {
            case.raw[source] = i32::from(*value);
            case.luma[source] = rng.next() as i32;
            case.scales[k] = rng.next() as i32;
        }
    }
    (case, expected)
}

#[test]
fn transpose_dc_and_every_destination_alignment() {
    if !supported() {
        return;
    }
    for offset in 0..32 {
        for dc in [i16::MIN, -4_095, -1, 0, 1, 4_095, i16::MAX] {
            let mut case = Case::zero_ac();
            let mut expected = [0; 64];
            expected[0] = dc;
            // Distinct, signed AC values expose row, lane, and pack order.
            for source in 1..64 {
                let value = source as i32 * 113 - 3_617;
                case.raw[source] = value;
                expected[(source % 8) * 8 + source / 8] = value as i16;
            }
            case.raw[0] = if offset & 1 == 0 { i32::MIN } else { i32::MAX };
            case.luma
                .fill(if offset & 2 == 0 { i32::MIN } else { i32::MAX });
            case.scales
                .fill(if offset & 4 == 0 { i32::MIN } else { i32::MAX });
            assert_eq!(check_case::<false>(&case, offset, dc), Some(expected));

            for k in 1..64 {
                let source = (k % 8) * 8 + k / 8;
                case.luma[source] = source as i32 - 31;
                case.scales[k] = k as i32 * 257 - 8_192;
                let product = i64::from(case.luma[source]) * i64::from(case.scales[k]);
                case.raw[source] -= ((product + 1_024) >> 11) as i32;
            }
            assert_eq!(check_case::<true>(&case, offset, dc), Some(expected));
        }
    }
}

#[test]
fn signed_rounding_at_half_steps_and_neighbors() {
    if !supported() {
        return;
    }
    // Explicit answers distinguish arithmetic shift from truncation toward zero
    // and from symmetric rounding of negative half steps.
    let products = [
        (-5_121, -3),
        (-5_120, -2),
        (-5_119, -2),
        (-3_073, -2),
        (-3_072, -1),
        (-3_071, -1),
        (-2_049, -1),
        (-2_048, -1),
        (-2_047, -1),
        (-1_025, -1),
        (-1_024, 0),
        (-1_023, 0),
        (-1, 0),
        (0, 0),
        (1, 0),
        (1_023, 0),
        (1_024, 1),
        (1_025, 1),
        (3_071, 1),
        (3_072, 2),
        (3_073, 2),
    ];
    for phase in 0..products.len() {
        for sign in [-1, 1] {
            let mut case = Case::zero_ac();
            let mut expected = [0; 64];
            expected[0] = i16::MIN;
            for (k, value) in expected.iter_mut().enumerate().skip(1) {
                let (product, prediction) = products[(k + phase) % products.len()];
                let source = (k % 8) * 8 + k / 8;
                let residual = k as i32 - 32;
                case.raw[source] = residual;
                case.luma[source] = sign;
                case.scales[k] = product * sign;
                *value = (residual + prediction) as i16;
            }
            assert_eq!(check_case::<true>(&case, phase, i16::MIN), Some(expected));
        }
    }
}

#[test]
fn inclusive_envelope_endpoints_in_every_ac_lane() {
    if !supported() {
        return;
    }
    // (residual, luma, scale, final): include large cancellation, each inclusive
    // input endpoint, and both final endpoints without exceeding another bound.
    let boundaries = [
        (65_535, -256, 524_288, -1),
        (-65_535, 256, 524_288, 1),
        (65_535, 256, -524_288, -1),
        (-65_535, -256, -524_288, 1),
        (-61_441, 256, 524_288, 4_095),
        (61_441, -256, 524_288, -4_095),
        (0, 4_095, 2_048, 4_095),
        (0, -4_095, 2_048, -4_095),
        (0, 4_095, -2_048, -4_095),
        (0, -4_095, -2_048, 4_095),
        (-4_095, 4_095, 0, -4_095),
        (4_095, -4_095, 0, 4_095),
        (-4_095, 0, -524_288, -4_095),
        (4_095, 0, 524_288, 4_095),
    ];
    for k in 1..64 {
        for (raw, luma, scale, final_value) in boundaries {
            let mut case = Case::zero_ac();
            let source = (k % 8) * 8 + k / 8;
            case.raw[source] = raw;
            case.luma[source] = luma;
            case.scales[k] = scale;
            let mut expected = [0; 64];
            expected[0] = i16::MAX;
            expected[k] = final_value;
            assert_eq!(check_case::<true>(&case, k % 32, i16::MAX), Some(expected));
        }
        for value in [-4_095, -1, 0, 1, 4_095] {
            let mut case = Case::zero_ac();
            case.raw[(k % 8) * 8 + k / 8] = value;
            case.luma.fill(i32::MIN);
            case.scales.fill(i32::MAX);
            let mut expected = [0; 64];
            expected[0] = i16::MIN;
            expected[k] = value as i16;
            assert_eq!(check_case::<false>(&case, k % 32, i16::MIN), Some(expected));
        }
    }
}

#[test]
fn rejection_is_atomic_for_every_lane_and_input_bound() {
    if !supported() {
        return;
    }
    let invalid = [
        // Final zero, but residual/luma/scale outside their CFL envelopes.
        (65_536, -256, 524_288),
        (-65_536, 256, 524_288),
        (0, -4_096, 0),
        (0, 4_096, 0),
        (0, 0, -524_289),
        (0, 0, 524_289),
        (0, i32::MIN, 0),
        (0, i32::MAX, 0),
        (0, 0, i32::MIN),
        (0, 0, i32::MAX),
        // Wrapped 32-bit products can look like valid final coefficients.
        (0, i32::MIN, i32::MIN),
        (0, i32::MIN, 2),
        (1, i32::MAX, 2_048),
        (0, 2, i32::MIN),
        (i32::MIN, 0, 0),
        (i32::MAX, 0, 0),
        // Valid inputs but invalid final coefficients, including near-maximal
        // signed products and a one-step rounding overflow of the final bound.
        (-4_096, 0, 0),
        (4_096, 0, 0),
        (0, -16, 524_288),
        (0, 16, 524_288),
        (4_095, 1, 1_024),
        (-4_095, -1, 2_048),
        (65_535, 4_095, 524_288),
        (-65_535, -4_095, 524_288),
        (65_535, -4_095, -524_288),
        (-65_535, 4_095, -524_288),
    ];
    for k in 1..64 {
        for (raw, luma, scale) in invalid {
            let mut case = Case::zero_ac();
            let source = (k % 8) * 8 + k / 8;
            case.raw[source] = raw;
            case.luma[source] = luma;
            case.scales[k] = scale;
            assert!(check_case::<true>(&case, k % 32, -123).is_none());
        }
        for raw in [i32::MIN, -65_536, -4_096, 4_096, 65_536, i32::MAX] {
            let mut case = Case::zero_ac();
            case.raw[(k % 8) * 8 + k / 8] = raw;
            assert!(check_case::<false>(&case, k % 32, 321).is_none());
        }
    }
}

fn random_differential<const CFL: bool>(rng: &mut Rng) {
    for iteration in 0..128 {
        let dc = rng.next() as i16;
        let (mut case, mut expected) = valid_case::<CFL>(rng);
        expected[0] = dc;
        assert_eq!(check_case::<CFL>(&case, iteration % 32, dc), Some(expected));

        // A bad lane inside an otherwise nonzero, valid block must not allow
        // preceding rows or lanes to commit before the error is discovered.
        let k = 1 + (rng.next() % 63) as usize;
        let source = (k % 8) * 8 + k / 8;
        match if CFL { iteration % 4 } else { 0 } {
            0 => {
                case.raw[source] = if iteration & 1 == 0 { -4_096 } else { 4_096 };
                case.luma[source] = 0;
                case.scales[k] = 0;
            }
            1 => case.luma[source] = -4_096,
            2 => case.scales[k] = 524_289,
            _ => case.raw[source] = -65_536,
        }
        assert!(check_case::<CFL>(&case, (iteration + 1) % 32, dc).is_none());

        // Arbitrary full-width values exercise overflow rejection as well as
        // mixed signs; the oracle decides acceptance rather than assuming it.
        let arbitrary = Case {
            raw: std::array::from_fn(|_| rng.next() as i32),
            luma: std::array::from_fn(|_| rng.next() as i32),
            scales: std::array::from_fn(|_| rng.next() as i32),
        };
        let _ = check_case::<CFL>(&arbitrary, (iteration + 2) % 32, dc);
    }
}

#[test]
fn deterministic_random_corpus_matches_avx2_and_i64_oracle() {
    if !supported() {
        return;
    }
    let mut rng = Rng(0x728d_59e4_160b_c3af);
    random_differential::<false>(&mut rng);
    random_differential::<true>(&mut rng);
}

fn timed_kernel<const CFL: bool, const AVX512: bool>(
    cases: &[Case],
    output: &mut [[i16; 64]],
    expected: &[[i16; 64]],
    passes: usize,
) -> f64 {
    assert!(supported());
    assert_eq!(cases.len(), output.len());
    let mut accepted = 0usize;
    let start = Instant::now();
    for _ in 0..passes {
        for (case, destination) in cases.iter().zip(output.iter_mut()) {
            let case = black_box(case);
            let destination = black_box(destination);
            // SAFETY: runtime checks above cover both kernels; fixed-size
            // inputs and destinations each contain exactly 64 coefficients.
            let ok = unsafe {
                if AVX512 {
                    super::x86::capture_avx512::<CFL>(
                        &case.raw,
                        &case.luma,
                        &case.scales,
                        destination,
                    )
                } else {
                    super::x86::capture::<CFL>(&case.raw, &case.luma, &case.scales, destination)
                }
            };
            accepted += usize::from(black_box(ok));
            black_box(&*destination);
        }
    }
    let elapsed = start.elapsed();
    assert_eq!(black_box(accepted), cases.len() * passes);
    assert_eq!(output, expected);
    black_box(output);
    elapsed.as_secs_f64() * 1e9 / (cases.len() * passes) as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[values.len() / 2]
}

fn benchmark_mode<const CFL: bool>() {
    const BLOCKS: usize = 128;
    const PASSES: usize = 16_384;
    const PAIRS: usize = 21;
    let mut rng = Rng(black_box(0x390a_d87b_62c1_5e4f));
    let mut cases = Vec::with_capacity(BLOCKS);
    let mut expected = Vec::with_capacity(BLOCKS);
    let mut output2 = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let (case, mut values) = valid_case::<CFL>(&mut rng);
        let dc = rng.next() as i16;
        values[0] = dc;
        assert_eq!(oracle::<CFL>(&case, [dc; 64]), Some(values));
        cases.push(case);
        expected.push(values);
        output2.push([dc; 64]);
    }
    let cases = black_box(cases);
    let mut output512 = output2.clone();
    timed_kernel::<CFL, false>(&cases, &mut output2, &expected, 8);
    timed_kernel::<CFL, true>(&cases, &mut output512, &expected, 8);
    let mut samples2 = [0.0; PAIRS];
    let mut samples512 = [0.0; PAIRS];
    let mut paired_speedup = [0.0; PAIRS];
    for pair in 0..PAIRS {
        if pair % 2 == 0 {
            samples2[pair] = timed_kernel::<CFL, false>(&cases, &mut output2, &expected, PASSES);
            samples512[pair] = timed_kernel::<CFL, true>(&cases, &mut output512, &expected, PASSES);
        } else {
            samples512[pair] = timed_kernel::<CFL, true>(&cases, &mut output512, &expected, PASSES);
            samples2[pair] = timed_kernel::<CFL, false>(&cases, &mut output2, &expected, PASSES);
        }
        paired_speedup[pair] = samples2[pair] / samples512[pair];
    }
    eprintln!(
        "JPEG capture CFL={CFL}: median AVX2={:.2} ns/block, AVX512={:.2} ns/block; \
         median paired speedup={:.3}x ({PAIRS} alternating pairs, {BLOCKS} blocks, {PASSES} passes)",
        median(&mut samples2),
        median(&mut samples512),
        median(&mut paired_speedup),
    );
}

#[test]
#[ignore = "release microbenchmark; run explicitly with optimization and --nocapture"]
fn paired_release_microbenchmark() {
    if !supported() {
        eprintln!("JPEG capture benchmark skipped: AVX2, AVX512F and AVX512BW are required");
        return;
    }
    assert!(
        !cfg!(debug_assertions),
        "run this benchmark with --release or rustc -O"
    );
    benchmark_mode::<false>();
    benchmark_mode::<true>();
}
