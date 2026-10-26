// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by the BSD-style LICENSE file.
//! Optional development instrumentation. Never compiled into ordinary builds.
//! Inclusive scopes overlap. Sampled timings include timer overhead and must not
//! be added to stage totals or presented as measured CPU percentages.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

thread_local! {
    static COUNTERS: Cell<[u32; 8]> = const { Cell::new([0; 8]) };
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("RS_JXL_PROFILE").as_deref(),
            Ok("1" | "trace")
        )
    })
}

#[derive(Default, Clone, Copy)]
struct Stats {
    calls: u64,
    total: Duration,
}

fn totals() -> &'static Mutex<BTreeMap<&'static str, Stats>> {
    static TOTALS: OnceLock<Mutex<BTreeMap<&'static str, Stats>>> = OnceLock::new();
    TOTALS.get_or_init(Mutex::default)
}

#[must_use]
pub struct Scope {
    name: &'static str,
    start: Instant,
}

pub fn scope(name: &'static str) -> Option<Scope> {
    enabled().then(|| Scope {
        name,
        start: Instant::now(),
    })
}

/// Accumulate block timings locally, publishing once per group rather than
/// taking the global statistics lock per block. Timer overhead is still present.
pub struct Accumulator {
    name: &'static str,
    enabled: bool,
    stats: Stats,
}

impl Accumulator {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            enabled: enabled(),
            stats: Stats::default(),
        }
    }

    pub fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    pub fn stop(&mut self, start: Option<Instant>) {
        if let Some(start) = start {
            self.stats.total += start.elapsed();
            self.stats.calls += 1;
        }
    }
}

impl Drop for Accumulator {
    fn drop(&mut self) {
        if self.stats.calls == 0 {
            return;
        }
        let mut totals = totals().lock().unwrap_or_else(|e| e.into_inner());
        let stats = totals.entry(self.name).or_default();
        stats.calls += self.stats.calls;
        stats.total += self.stats.total;
    }
}

/// Slots: WP prediction, WP errors, entropy, JPEG capture, refinement preparation,
/// initial AC validation, initial AC emission, refinement AC emission.
/// One call in 1024 per thread and slot, phase shifted between slots. Sampling
/// is periodic, not a statistical random sample; narrow timings are diagnostic.
/// There are no black-box barriers against motion of inlined pure arithmetic;
/// values near the reported timer floor are not useful for CPU attribution.
pub fn sample_scope(name: &'static str, slot: usize) -> Option<Scope> {
    if !enabled() {
        return None;
    }
    let selected = COUNTERS.with(|counters| {
        let mut values = counters.get();
        values[slot] = values[slot].wrapping_add(1);
        let selected = values[slot] & 1023 == (slot as u32 * 127) & 1023;
        counters.set(values);
        selected
    });
    selected.then(|| Scope {
        name,
        start: Instant::now(),
    })
}

impl Drop for Scope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let mut totals = totals().lock().unwrap_or_else(|e| e.into_inner());
        let stats = totals.entry(self.name).or_default();
        stats.calls += 1;
        stats.total += elapsed;
    }
}

pub fn report() -> Option<String> {
    if !enabled() {
        return None;
    }
    let mut rows: Vec<_> = totals()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(&name, &stats)| (name, stats))
        .collect();
    rows.sort_by(|a, b| b.1.total.cmp(&a.1.total));
    let mut text = String::from(
        "[decoder stages] inclusive; sample.* = 1/1024 calls, NOT stage totals\n\
         stage | calls | total seconds | mean ns\n",
    );
    for (name, stats) in rows {
        writeln!(
            text,
            "{name} | {} | {:.9} | {:.1}",
            stats.calls,
            stats.total.as_secs_f64(),
            stats.total.as_nanos() as f64 / stats.calls as f64
        )
        .expect("format into String");
    }
    // Same timer pair, no codec work: report the measurement floor separately.
    let floor: u128 = (0..4096)
        .map(|_| {
            let t = Instant::now();
            t.elapsed().as_nanos()
        })
        .sum();
    writeln!(text, "timer_pair_floor_ns | {:.1}", floor as f64 / 4096.0)
        .expect("format into String");
    Some(text)
}
