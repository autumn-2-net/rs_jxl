// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

//! Opt-in channel-cost microbenchmark, not an encoder or codestream benchmark.
//! Compare the former runtime i64 SIMD residual scan with the fused i32 scan,
//! retaining the SAME detected source histogram/entropy backend on both sides.
//! Run the ignored test in release mode with --ignored --nocapture.

use std::hint::black_box;
use std::time::{Duration, Instant};

use super::{
    Backend, ChannelCost, ChannelView, Histogram, add_residual, backend, channel_cost,
    combine_costs, gradient_context, zeroed,
};
use crate::modular::predict_simd;
use crate::modular::predictor::Predictor;
use crate::{Error, Result};

const WIDTH: usize = 1025;
const HEIGHT: usize = 513;
const REPEATS: usize = 7;
const WARMUP_PAIRS: usize = 2;

// Exact former channel_cost body: residual_row is runtime SSE2/AVX2 i64 SIMD,
// NOT residual_row_scalar. Keep the per-pixel context and add_residual work.
fn old_channel_cost(ch: ChannelView<'_>, mode: Backend) -> Result<ChannelCost> {
    let mut histograms: [Histogram; 17] = std::array::from_fn(|_| Histogram::default());
    let mut extra = 0;
    if ch.width != 0 && ch.height != 0 {
        let mut residuals = zeroed(ch.width)?;
        for y in 0..ch.height {
            predict_simd::residual_row(
                ch.pixels,
                ch.width,
                ch.stride,
                y,
                Predictor::Gradient,
                &mut residuals,
            )
            .map_err(|e| Error::Encoding(e.to_string()))?;
            for (x, &residual) in residuals.iter().enumerate() {
                let context = gradient_context(ch.neighbors(x, y), mode);
                add_residual(&mut histograms[context], residual as i32, &mut extra)?;
            }
        }
    }
    Ok(ChannelCost {
        histograms: histograms.map(|h| h.entropy(mode)),
        extra,
    })
}

fn view(pixels: &[i32], width: usize, height: usize) -> ChannelView<'_> {
    ChannelView {
        pixels,
        width,
        height,
        stride: width,
    }
}

fn assert_same_cost(ch: ChannelView<'_>, mode: Backend) {
    ch.validate().unwrap();
    let old = old_channel_cost(ch, mode).unwrap();
    let new = channel_cost(ch, mode).unwrap();
    assert_eq!(old.extra, new.extra, "extra bits, {mode:?}");
    assert_eq!(
        old.histograms.map(f32::to_bits),
        new.histograms.map(f32::to_bits),
        "per-context histogram cost, {mode:?}",
    );
    assert_eq!(
        combine_costs(&[old]).unwrap().to_bits(),
        combine_costs(&[new]).unwrap().to_bits(),
        "complete channel cost, {mode:?}",
    );
}

fn check_full_i32_domain(mode: Backend) {
    // Cover every central/left/top/top-left combination of signed extrema,
    // zero and near-zero values. Column 5 is inside both native vector widths.
    // Adjacent row pairs are independent fixtures for that column's neighbors.
    let extremes = [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX];
    let width = 33;
    let height = 2 * extremes.len().pow(4) + 4;
    let mut pixels = Vec::with_capacity(width * height);
    for central in extremes {
        for left in extremes {
            for top in extremes {
                for top_left in extremes {
                    let mut north = [top; 33];
                    let mut current = [central; 33];
                    north[4] = top_left;
                    current[4] = left;
                    pixels.extend_from_slice(&north);
                    pixels.extend_from_slice(&current);
                }
            }
        }
    }
    let mut random = 0xb866_b3d0u32;
    for _ in 0..4 * width {
        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
        pixels.push(random as i32);
    }
    assert_eq!(pixels.len(), width * height);
    assert_same_cost(view(&pixels, width, height), mode);
}

fn fixtures() -> [(&'static str, Vec<i32>); 3] {
    let mut smooth = Vec::with_capacity(WIDTH * HEIGHT);
    let mut repeated = Vec::with_capacity(WIDTH * HEIGHT);
    let mut noise = Vec::with_capacity(WIDTH * HEIGHT);
    let tile = [0, 257, 4095, 16384, 32768, 49151, 65535, 12345];
    let mut random = 0xb866_b3d0u32;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            smooth.push((29 * x + 43 * y) as i32);
            repeated.push(tile[(x / 8 + 3 * (y / 8)) % tile.len()]);
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            noise.push((random >> 16) as i32);
        }
    }
    [
        ("smooth", smooth),
        ("repeated", repeated),
        ("u16noise", noise),
    ]
}

fn measure(
    kernel: fn(ChannelView<'_>, Backend) -> Result<ChannelCost>,
    ch: ChannelView<'_>,
    mode: Backend,
) -> Duration {
    let start = Instant::now();
    let cost = black_box(kernel)(black_box(ch), black_box(mode)).unwrap();
    black_box(cost);
    start.elapsed()
}

fn median_seconds(mut durations: [Duration; REPEATS]) -> f64 {
    durations.sort_unstable();
    durations[REPEATS / 2].as_secs_f64()
}

#[test]
#[ignore = "opt-in microbenchmark; run release with --ignored --nocapture"]
fn old_runtime_simd_vs_fused_cost() {
    let mode = backend();
    let fixtures = fixtures();
    // All equivalence checks, input generation and validation precede timing.
    check_full_i32_domain(mode);
    for (_, pixels) in &fixtures {
        assert_same_cost(view(pixels, WIDTH, HEIGHT), mode);
    }
    println!(
        "gradient cost: {WIDTH}x{HEIGHT}, backend={mode:?}, warmup_pairs={WARMUP_PAIRS}, repeats={REPEATS}"
    );
    println!("fixture   old_median_ms new_median_ms old_Mpix/s new_Mpix/s old/new");
    for (name, pixels) in &fixtures {
        let ch = view(pixels, WIDTH, HEIGHT);
        for pair in 0..WARMUP_PAIRS {
            if pair % 2 == 0 {
                black_box(old_channel_cost(black_box(ch), black_box(mode)).unwrap());
                black_box(channel_cost(black_box(ch), black_box(mode)).unwrap());
            } else {
                black_box(channel_cost(black_box(ch), black_box(mode)).unwrap());
                black_box(old_channel_cost(black_box(ch), black_box(mode)).unwrap());
            }
        }
        let mut old_times = [Duration::ZERO; REPEATS];
        let mut new_times = [Duration::ZERO; REPEATS];
        for repeat in 0..REPEATS {
            if repeat % 2 == 0 {
                old_times[repeat] = measure(old_channel_cost, ch, mode);
                new_times[repeat] = measure(channel_cost, ch, mode);
            } else {
                new_times[repeat] = measure(channel_cost, ch, mode);
                old_times[repeat] = measure(old_channel_cost, ch, mode);
            }
        }
        let old = median_seconds(old_times);
        let new = median_seconds(new_times);
        let megapixels = (WIDTH * HEIGHT) as f64 / 1_000_000.0;
        println!(
            "{name:<9} {:>13.3} {:>13.3} {:>10.3} {:>10.3} {:>7.3}x",
            old * 1000.0,
            new * 1000.0,
            megapixels / old,
            megapixels / new,
            old / new,
        );
    }
}
