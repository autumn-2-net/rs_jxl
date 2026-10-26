//! Opt-in, process-cumulative diagnostics, never encoding policy.
//!
//! RS_JXL_PROFILE=1 collects inclusive stage times; =trace additionally reports
//! entry/exit so an interrupted process still identifies its unfinished stage.
//! Nested and parallel totals overlap and must not be added as wall time.
use std::collections::BTreeMap;
use std::fmt::{Display, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Summary,
    Trace,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("RS_JXL_PROFILE").as_deref() {
        Ok("1") => Mode::Summary,
        Ok("trace") => Mode::Trace,
        _ => Mode::Off,
    })
}

pub fn enabled() -> bool {
    mode() != Mode::Off
}

#[derive(Clone, Copy, Default)]
struct Stats {
    calls: u64,
    total: Duration,
    longest: Duration,
}

impl Stats {
    fn add(&mut self, duration: Duration) {
        self.calls = self.calls.saturating_add(1);
        self.total = self.total.saturating_add(duration);
        self.longest = self.longest.max(duration);
    }

    fn merge(&mut self, other: Self) {
        self.calls = self.calls.saturating_add(other.calls);
        self.total = self.total.saturating_add(other.total);
        self.longest = self.longest.max(other.longest);
    }
}

const SHARDS: usize = 64;
type Counters = BTreeMap<&'static str, Stats>;

fn totals() -> &'static [Mutex<Counters>; SHARDS] {
    static TOTALS: OnceLock<[Mutex<Counters>; SHARDS]> = OnceLock::new();
    TOTALS.get_or_init(|| std::array::from_fn(|_| Mutex::default()))
}

fn record(name: &'static str, elapsed: Duration) {
    static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        // Bounded storage; independent workers usually avoid sharing a lock.
        static SLOT: usize = NEXT_SLOT.fetch_add(1, Ordering::Relaxed) % SHARDS;
    }
    SLOT.with(|&slot| {
        totals()[slot]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(name)
            .or_default()
            .add(elapsed);
    });
}

fn snapshot() -> Counters {
    let mut merged = Counters::new();
    for shard in totals() {
        for (&name, &stats) in shard.lock().unwrap_or_else(|p| p.into_inner()).iter() {
            merged.entry(name).or_default().merge(stats);
        }
    }
    merged
}

#[must_use]
pub struct Scope {
    name: &'static str,
    start: Option<Instant>,
    trace: bool,
}

pub fn scope(name: &'static str) -> Scope {
    let mode = mode();
    if mode == Mode::Trace {
        eprintln!(
            "[rs-jxl profile] begin {name} {:?}",
            std::thread::current().id()
        );
    }
    Scope {
        name,
        start: (mode != Mode::Off).then(Instant::now),
        trace: mode == Mode::Trace,
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(start) = self.start else { return };
        let elapsed = start.elapsed();
        record(self.name, elapsed);
        if self.trace {
            eprintln!(
                "[rs-jxl profile] end {} {:.6}s {:?}",
                self.name,
                elapsed.as_secs_f64(),
                std::thread::current().id()
            );
        }
    }
}

pub fn note(name: &str, value: impl Display) {
    if enabled() {
        eprintln!("[rs-jxl profile] {name}: {value}");
    }
}

/// Returns a snapshot; it does not clear shared counters during concurrent work.
pub fn report() -> Option<String> {
    if !enabled() {
        return None;
    }
    let mut rows: Vec<_> = snapshot().into_iter().collect();
    rows.sort_by(|a, b| b.1.total.cmp(&a.1.total).then_with(|| a.0.cmp(b.0)));
    let mut report = String::from(
        "[rs-jxl profile] inclusive process totals; nested/parallel times overlap\n\
         stage | calls | total seconds | longest seconds\n",
    );
    for (name, stats) in rows {
        writeln!(
            report,
            "{name} | {} | {:.6} | {:.6}",
            stats.calls,
            stats.total.as_secs_f64(),
            stats.longest.as_secs_f64()
        )
        .expect("formatting into String");
    }
    #[cfg(feature = "decoder-profile")]
    if let Some(backend) = jxl_decoder::decoder_profile::report() {
        report.push_str(&backend);
    }
    Some(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistics_preserve_total_max_and_count_without_assuming_wall_time() {
        let mut stats = Stats::default();
        stats.add(Duration::from_millis(17));
        stats.add(Duration::from_millis(3));
        assert_eq!(stats.calls, 2);
        assert_eq!(stats.total, Duration::from_millis(20));
        assert_eq!(stats.longest, Duration::from_millis(17));
    }

    #[test]
    fn shards_preserve_concurrent_counts_and_inclusive_times() {
        const NAME: &str = "test.profile.shards";
        std::thread::scope(|scope| {
            for worker in 1..=16 {
                scope.spawn(move || {
                    for _ in 0..1000 {
                        record(NAME, Duration::from_micros(worker));
                    }
                });
            }
        });
        let stats = snapshot()[NAME];
        assert_eq!(stats.calls, 16000);
        assert_eq!(stats.total, Duration::from_micros(136000));
        assert_eq!(stats.longest, Duration::from_micros(16));
    }

    #[test]
    fn merging_statistics_saturates_counts_and_duration() {
        let mut stats = Stats {
            calls: u64::MAX,
            total: Duration::MAX,
            longest: Duration::ZERO,
        };
        stats.merge(Stats {
            calls: 1,
            total: Duration::from_secs(1),
            longest: Duration::from_secs(1),
        });
        assert_eq!(stats.calls, u64::MAX);
        assert_eq!(stats.total, Duration::MAX);
        assert_eq!(stats.longest, Duration::from_secs(1));
    }
}
