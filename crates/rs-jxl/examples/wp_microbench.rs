//! Micro-benchmark for the four-lane weighted-predictor cost kernel.
//!
//! Compares `select_weighted_preset_narrow` with `narrow = false` (scalar) and
//! `narrow = true` (four-lane `i32`) on synthetic channels, and reports the
//! median wall of each side plus their ratio. The search shape matches the
//! production call: five modes with the bounded policy (mode 0 unpruned, the
//! rest bound by the running best).
//!
//! It links against the already-built release library, so the cost is a link
//! rather than a test-harness rebuild:
//!
//! ```text
//! cargo run --release -p rs-jxl --example wp_microbench -- [size] [rounds]
//! ```
use rs_jxl::modular::cost::{ChannelView, select_weighted_preset_narrow};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let size = args.next().and_then(|v| v.parse().ok()).unwrap_or(512usize);
    let rounds = args.next().and_then(|v| v.parse().ok()).unwrap_or(7usize);

    let mut pixels = vec![0i32; size * size];
    for y in 0..size {
        for x in 0..size {
            pixels[y * size + x] = ((x * 7919 + y * 104729) % 2_000_001) as i32 - 1_000_000;
        }
    }
    println!(
        "[wp-micro] size={size}x{size} rounds={rounds} pixels={} max_abs={}",
        pixels.len(),
        1_000_000
    );

    for channels in [1usize, 3, 4] {
        let views: Vec<_> = (0..channels)
            .map(|_| ChannelView {
                pixels: &pixels,
                width: size,
                height: size,
                stride: size,
            })
            .collect();
        let mut scalar = Vec::with_capacity(rounds);
        let mut narrow = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let t0 = Instant::now();
            let a = select_weighted_preset_narrow(&views, 5, true, false).unwrap();
            scalar.push(t0.elapsed().as_secs_f64() * 1e3);
            let t1 = Instant::now();
            let b = select_weighted_preset_narrow(&views, 5, true, true).unwrap();
            narrow.push(t1.elapsed().as_secs_f64() * 1e3);
            assert_eq!(
                a.cost.to_bits(),
                b.cost.to_bits(),
                "cost diverged at ch={channels}"
            );
            assert_eq!(a.preset, b.preset, "preset diverged at ch={channels}");
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (s, n) = (med(&mut scalar), med(&mut narrow));
        let (s_min, n_min) = (
            scalar.first().copied().unwrap_or(f64::NAN),
            narrow.first().copied().unwrap_or(f64::NAN),
        );
        println!(
            "[wp-micro] ch={channels} scalar_med={s:.1}ms narrow_med={n:.1}ms \
             ratio={:.4} narrow_cost={:.1}% (min {s_min:.1} / {n_min:.1})",
            s / n,
            (n / s) * 100.0
        );
    }
}
