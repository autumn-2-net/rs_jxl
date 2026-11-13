//! Opt-in, process-cumulative diagnostics, never encoding policy.
//!
//! RS_JXL_PROFILE=1 collects inclusive stage times; =trace additionally reports
//! entry/exit so an interrupted process still identifies its unfinished stage.
//! Nested and parallel totals overlap and must not be added as wall time.
//! Parent attribution is recorded per nesting edge and reported as a separate
//! section, so the four-column stage table stays parse-compatible.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::{Display, Write};
use std::marker::PhantomData;
use std::rc::Rc;
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
type EdgeKey = (&'static str, Option<&'static str>);
type Edges = BTreeMap<EdgeKey, u64>;

fn totals() -> &'static [Mutex<Counters>; SHARDS] {
    static TOTALS: OnceLock<[Mutex<Counters>; SHARDS]> = OnceLock::new();
    TOTALS.get_or_init(|| std::array::from_fn(|_| Mutex::default()))
}

fn edges() -> &'static [Mutex<Edges>; SHARDS] {
    static EDGES: OnceLock<[Mutex<Edges>; SHARDS]> = OnceLock::new();
    EDGES.get_or_init(|| std::array::from_fn(|_| Mutex::default()))
}

#[derive(Default)]
struct ScopeStack {
    next_id: u64,
    entries: Vec<(u64, &'static str)>,
}

thread_local! {
    /// Live scope stack for the current thread; drives parent attribution.
    static STACK: RefCell<ScopeStack> = RefCell::new(ScopeStack::default());
}

fn slot() -> usize {
    static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        // Bounded storage; independent workers usually avoid sharing a lock.
        static SLOT: usize = NEXT_SLOT.fetch_add(1, Ordering::Relaxed) % SHARDS;
    }
    SLOT.with(|&s| s)
}

fn record(name: &'static str, parent: Option<&'static str>, elapsed: Duration) {
    let slot = slot();
    totals()[slot]
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .entry(name)
        .or_default()
        .add(elapsed);
    *edges()[slot]
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .entry((name, parent))
        .or_default() += 1;
}

fn snapshot() -> (Counters, Edges) {
    let mut merged = Counters::new();
    for shard in totals() {
        for (&name, &stats) in shard.lock().unwrap_or_else(|p| p.into_inner()).iter() {
            merged.entry(name).or_default().merge(stats);
        }
    }
    let mut merged_edges = Edges::new();
    for shard in edges() {
        for (&key, &count) in shard.lock().unwrap_or_else(|p| p.into_inner()).iter() {
            *merged_edges.entry(key).or_default() += count;
        }
    }
    (merged, merged_edges)
}

#[must_use]
pub struct Scope {
    id: u64,
    name: &'static str,
    parent: Option<&'static str>,
    start: Option<Instant>,
    trace: bool,
    // Cleanup must run on the originating thread, even for disabled guards.
    _thread_bound: PhantomData<Rc<()>>,
}

pub fn scope(name: &'static str) -> Scope {
    scope_with_mode(name, mode())
}

fn scope_with_mode(name: &'static str, mode: Mode) -> Scope {
    let active = mode != Mode::Off;
    let (id, parent) = if active {
        STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let id = stack.next_id;
            stack.next_id = id.checked_add(1).expect("profile scope identity exhausted");
            let parent = stack.entries.last().map(|&(_, name)| name);
            stack.entries.push((id, name));
            (id, parent)
        })
    } else {
        (0, None)
    };
    if mode == Mode::Trace {
        eprintln!(
            "[rs-jxl profile] begin {name} {:?}",
            std::thread::current().id()
        );
    }
    Scope {
        id,
        name,
        parent,
        start: active.then(Instant::now),
        trace: mode == Mode::Trace,
        _thread_bound: PhantomData,
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(start) = self.start else { return };
        let elapsed = start.elapsed();
        STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            // Identity distinguishes repeated names and permits out-of-order drops.
            if let Some(index) = stack.entries.iter().rposition(|&(id, _)| id == self.id) {
                stack.entries.remove(index);
            }
        });
        record(self.name, self.parent, elapsed);
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
    let (counters, edges) = snapshot();
    let mut rows: Vec<_> = counters.into_iter().collect();
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
    if !edges.is_empty() {
        let mut edge_rows: Vec<_> = edges.into_iter().collect();
        edge_rows.sort_by(|a, b| {
            let parent_a = a.0.1.unwrap_or("");
            let parent_b = b.0.1.unwrap_or("");
            parent_a.cmp(parent_b).then_with(|| a.0.0.cmp(b.0.0))
        });
        report.push_str("[rs-jxl profile] parent edges (child <- parent | calls)\n");
        for ((child, parent), calls) in edge_rows {
            writeln!(
                report,
                "{child} <- {} | {calls}",
                parent.unwrap_or("(root)")
            )
            .expect("formatting into String");
        }
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

    fn stack_entries() -> Vec<(u64, &'static str)> {
        STACK.with(|stack| stack.borrow().entries.clone())
    }

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
                        record(NAME, None, Duration::from_micros(worker));
                    }
                });
            }
        });
        let (counters, _) = snapshot();
        let stats = counters[NAME];
        assert_eq!(stats.calls, 16000);
        assert_eq!(stats.total, Duration::from_micros(136000));
        assert_eq!(stats.longest, Duration::from_micros(16));
    }

    #[test]
    fn parent_edges_record_nesting() {
        const PARENT: &str = "test.profile.nesting.parent";
        const CHILD: &str = "test.profile.nesting.child";
        const SIBLING: &str = "test.profile.nesting.sibling";
        assert!(stack_entries().is_empty());
        let parent = scope_with_mode(PARENT, Mode::Summary);
        let child = scope_with_mode(CHILD, Mode::Summary);
        assert_eq!(parent.parent, None);
        assert_eq!(child.parent, Some(PARENT));
        assert_eq!(stack_entries(), [(parent.id, PARENT), (child.id, CHILD)]);
        drop(child);
        assert_eq!(stack_entries(), [(parent.id, PARENT)]);
        let sibling = scope_with_mode(SIBLING, Mode::Summary);
        assert_eq!(sibling.parent, Some(PARENT));
        drop(sibling);
        drop(parent);
        assert!(stack_entries().is_empty());
        let (counters, edges) = snapshot();
        for name in [PARENT, CHILD, SIBLING] {
            assert_eq!(counters[name].calls, 1);
        }
        assert_eq!(edges[&(PARENT, None)], 1);
        assert_eq!(edges[&(CHILD, Some(PARENT))], 1);
        assert_eq!(edges[&(SIBLING, Some(PARENT))], 1);
    }

    #[test]
    fn out_of_order_drops_remove_closed_scopes_from_parent_attribution() {
        const OUTER: &str = "test.profile.out_of_order.outer";
        const MIDDLE: &str = "test.profile.out_of_order.middle";
        const INNER: &str = "test.profile.out_of_order.inner";
        const CHILD: &str = "test.profile.out_of_order.child";
        const ROOT: &str = "test.profile.out_of_order.root";
        assert!(stack_entries().is_empty());
        let outer = scope_with_mode(OUTER, Mode::Summary);
        let middle = scope_with_mode(MIDDLE, Mode::Summary);
        let inner = scope_with_mode(INNER, Mode::Summary);
        drop(middle);
        assert_eq!(stack_entries(), [(outer.id, OUTER), (inner.id, INNER)]);
        drop(outer);
        assert_eq!(stack_entries(), [(inner.id, INNER)]);
        let child = scope_with_mode(CHILD, Mode::Summary);
        assert_eq!(child.parent, Some(INNER));
        drop(child);
        drop(inner);
        assert!(stack_entries().is_empty());
        let root = scope_with_mode(ROOT, Mode::Summary);
        assert_eq!(root.parent, None);
        drop(root);
        assert!(stack_entries().is_empty());
        let (_, edges) = snapshot();
        assert_eq!(edges[&(MIDDLE, Some(OUTER))], 1);
        assert_eq!(edges[&(INNER, Some(MIDDLE))], 1);
        assert_eq!(edges[&(CHILD, Some(INNER))], 1);
        assert_eq!(edges[&(ROOT, None)], 1);
    }

    #[test]
    fn repeated_names_remove_only_the_matching_scope_identity() {
        const NAME: &str = "test.profile.repeated_name";
        assert!(stack_entries().is_empty());
        let outer = scope_with_mode(NAME, Mode::Summary);
        let middle = scope_with_mode(NAME, Mode::Summary);
        let inner = scope_with_mode(NAME, Mode::Summary);
        assert_ne!(outer.id, middle.id);
        assert_ne!(middle.id, inner.id);
        assert_ne!(outer.id, inner.id);
        assert_eq!(outer.parent, None);
        assert_eq!(middle.parent, Some(NAME));
        assert_eq!(inner.parent, Some(NAME));
        drop(middle);
        assert_eq!(stack_entries(), [(outer.id, NAME), (inner.id, NAME)]);
        drop(outer);
        assert_eq!(stack_entries(), [(inner.id, NAME)]);
        drop(inner);
        assert!(stack_entries().is_empty());
        let root = scope_with_mode(NAME, Mode::Summary);
        assert_eq!(root.parent, None);
        drop(root);
        assert!(stack_entries().is_empty());
        let (counters, edges) = snapshot();
        assert_eq!(counters[NAME].calls, 4);
        assert_eq!(edges[&(NAME, None)], 2);
        assert_eq!(edges[&(NAME, Some(NAME))], 2);
    }

    #[test]
    fn disabled_scopes_do_not_touch_stack_identity_timer_or_counters() {
        const DISABLED: &str = "test.profile.disabled.off";
        const PARENT: &str = "test.profile.disabled.parent";
        const CHILD: &str = "test.profile.disabled.child";
        assert!(stack_entries().is_empty());
        let next_id = STACK.with(|stack| stack.borrow().next_id);
        let disabled = scope_with_mode(DISABLED, Mode::Off);
        assert!(disabled.start.is_none());
        assert_eq!(disabled.parent, None);
        assert!(!disabled.trace);
        assert!(stack_entries().is_empty());
        drop(disabled);
        assert_eq!(STACK.with(|stack| stack.borrow().next_id), next_id);

        let parent = scope_with_mode(PARENT, Mode::Summary);
        let before = stack_entries();
        let next_id = STACK.with(|stack| stack.borrow().next_id);
        // A disabled guard with the same name must not remove the active guard.
        let disabled = scope_with_mode(PARENT, Mode::Off);
        assert!(disabled.start.is_none());
        assert_eq!(disabled.parent, None);
        assert_eq!(stack_entries(), before);
        assert_eq!(STACK.with(|stack| stack.borrow().next_id), next_id);
        let child = scope_with_mode(CHILD, Mode::Summary);
        assert_eq!(child.parent, Some(PARENT));
        drop(child);
        drop(disabled);
        assert_eq!(stack_entries(), before);
        drop(parent);
        assert!(stack_entries().is_empty());
        let (counters, edges) = snapshot();
        assert!(!counters.contains_key(DISABLED));
        assert!(!edges.keys().any(|&(name, _)| name == DISABLED));
        assert_eq!(counters[PARENT].calls, 1);
        assert_eq!(edges[&(PARENT, None)], 1);
        assert_eq!(edges[&(CHILD, Some(PARENT))], 1);
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
