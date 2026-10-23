//! Test-only task-body observations. Capture the probe before dispatch so an
//! incorrectly spawned worker is observed even though it has different TLS.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

#[derive(Debug)]
pub(super) struct Event {
    pub stage: &'static str,
    pub thread: ThreadId,
    pub serial: bool,
}

#[derive(Clone, Default)]
pub(super) struct Probe(Option<Arc<Mutex<Vec<Event>>>>);

thread_local! {
    static CURRENT: RefCell<Probe> = const { RefCell::new(Probe(None)) };
}

pub(super) fn current() -> Probe {
    CURRENT.with_borrow(Clone::clone)
}

impl Probe {
    pub(super) fn record(&self, stage: &'static str) {
        if let Some(events) = &self.0 {
            events.lock().unwrap().push(Event {
                stage,
                thread: thread::current().id(),
                serial: crate::execution::is_serial(),
            });
        }
    }
}

pub(super) fn observe<R>(work: impl FnOnce() -> R) -> (R, Vec<Event>) {
    struct Restore(Probe);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.set(std::mem::take(&mut self.0));
        }
    }
    let events = Arc::new(Mutex::new(Vec::new()));
    let restore = Restore(CURRENT.replace(Probe(Some(Arc::clone(&events)))));
    let result = work();
    drop(restore);
    let events = std::mem::take(&mut *events.lock().unwrap());
    (result, events)
}

#[test]
fn captured_probe_observes_foreign_workers_without_tls_propagation() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (_, events) = observe(|| {
        let probe = current();
        pool.broadcast(|_| probe.record("probe.control"));
    });
    let threads: std::collections::HashSet<_> = events.iter().map(|event| event.thread).collect();
    assert_eq!(threads.len(), 4);
    assert!(events.iter().all(|event| !event.serial));
}
