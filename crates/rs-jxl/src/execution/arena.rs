// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! b866b3d enc_lossless_probe_pool.h: bounded roots, descendant-first child
//! queues, own-group helping, source quantum and optional quiet notifications.
//! Only the caller's installed Rayon workers execute callbacks. The scoped
//! callback erasure below is the sole lifetime-unsafe part of this executor.

use rayon::prelude::*;
use std::any::Any;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex};

type Panic = Box<dyn Any + Send + 'static>;

thread_local! {
    static CURRENT: Cell<*const Arena> = const { Cell::new(std::ptr::null()) };
}

struct Context(*const Arena);
impl Context {
    fn enter(arena: &Arena) -> Self {
        Self(CURRENT.replace(arena))
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        CURRENT.set(self.0);
    }
}

fn current<R>(work: impl FnOnce(Option<&Arena>) -> R) -> R {
    let pointer = CURRENT.get();
    // SAFETY: Context is installed only in a synchronous worker loop inside
    // run(), and restored before that loop exits. All loops join before Arena
    // is dropped. No reference with this callback-local lifetime can escape.
    work(unsafe { pointer.as_ref() })
}

pub(super) fn worker_count() -> Option<usize> {
    current(|arena| arena.map(|arena| arena.workers))
}

struct Callback {
    data: *const (),
    invoke: unsafe fn(*const (), usize),
}
// SAFETY: construction requires F: Sync. The registered Fn is borrowed until
// every claimed callback has returned and its group has been unpublished.
unsafe impl Send for Callback {}
unsafe impl Sync for Callback {}
impl Callback {
    fn borrowed<F: Fn(usize) + Sync>(work: &F) -> Self {
        unsafe fn invoke<F: Fn(usize) + Sync>(data: *const (), index: usize) {
            // SAFETY: borrowed stores this exact F and its enclosing group is
            // joined, including unwinding callbacks, before F can be dropped.
            (unsafe { &*data.cast::<F>() })(index);
        }
        Self {
            data: std::ptr::from_ref(work).cast(),
            invoke: invoke::<F>,
        }
    }
}

struct Group {
    callback: Callback,
    panic: Mutex<Option<Panic>>,
}
struct Pending {
    group: Arc<Group>,
    next: usize,
    end: usize,
    remaining: usize,
    quantum: usize,
}
impl Pending {
    fn take(&mut self) -> Option<Job> {
        if self.next == self.end {
            return None;
        }
        let begin = self.next;
        self.next += self.quantum.min(self.end - self.next);
        Some(Job {
            group: Arc::clone(&self.group),
            begin,
            end: self.next,
        })
    }
}
struct Job {
    group: Arc<Group>,
    begin: usize,
    end: usize,
}
#[derive(Default)]
struct State {
    groups: Vec<Pending>,
    next_root: usize,
    active_roots: usize,
    failed: bool,
    panic: Option<Panic>,
}
impl State {
    fn take_descendant(&mut self) -> Option<Job> {
        self.groups.iter_mut().rev().find_map(Pending::take)
    }
}
struct Arena {
    workers: usize,
    quiet: bool,
    state: Mutex<State>,
    changed: Condvar,
}
impl Arena {
    fn execute(&self, job: Job) {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            for index in job.begin..job.end {
                // SAFETY: remaining includes this entire claimed job. Its
                // callback cannot be dropped until execution has returned and
                // this job has been subtracted from remaining below.
                unsafe { (job.group.callback.invoke)(job.group.callback.data, index) };
            }
        }));
        if let Err(panic) = outcome {
            let mut slot = job.group.panic.lock().expect("arena panic slot poisoned");
            if slot.is_none() {
                *slot = Some(panic);
            } else {
                // An arbitrary panic payload can itself panic in Drop. Never
                // unwind the executor while another borrowed job is live.
                std::mem::forget(panic);
            }
        }
        let complete = {
            let mut state = self.state.lock().expect("arena queue poisoned");
            let entry = state
                .groups
                .iter_mut()
                .find(|entry| Arc::ptr_eq(&entry.group, &job.group))
                .expect("claimed arena group was unpublished");
            entry.remaining -= job.end - job.begin;
            entry.remaining == 0
        };
        if !self.quiet || complete {
            self.changed.notify_all();
        }
    }

    fn for_each(&self, count: usize, work: impl Fn(usize) + Sync) {
        if count == 0 {
            return;
        }
        if self.quiet && count == 1 {
            work(0);
            return;
        }
        let group = Arc::new(Group {
            callback: Callback::borrowed(&work),
            panic: Mutex::new(None),
        });
        {
            let mut state = self.state.lock().expect("arena queue poisoned");
            state.groups.push(Pending {
                group: Arc::clone(&group),
                next: 0,
                end: count,
                remaining: count,
                quantum: (count / (4 * self.workers)).max(1),
            });
        }
        self.changed.notify_all();
        loop {
            let job = {
                let mut state = self.state.lock().expect("arena queue poisoned");
                loop {
                    let index = state
                        .groups
                        .iter()
                        .position(|entry| Arc::ptr_eq(&entry.group, &group))
                        .expect("waiting arena group was unpublished");
                    if state.groups[index].remaining == 0 {
                        state.groups.remove(index);
                        break None;
                    }
                    // Never help ancestors/siblings while our own callback is
                    // suspended: they may own the same worker-indexed scratch.
                    if let Some(job) = state.groups[index].take() {
                        break Some(job);
                    }
                    state = self.changed.wait(state).expect("arena queue poisoned");
                }
            };
            match job {
                Some(job) => self.execute(job),
                None => break,
            }
        }
        let panic = group
            .panic
            .lock()
            .expect("arena panic slot poisoned")
            .take();
        // The callback borrow has ended only now, after all active descendants.
        if let Some(panic) = panic {
            resume_unwind(panic);
        }
    }

    fn worker(&self, count: usize, max_roots: usize, work: &(impl Fn(usize) -> bool + Sync)) {
        let _context = Context::enter(self);
        enum Next {
            Child(Job),
            Root(usize),
        }
        loop {
            let next = {
                let mut state = self.state.lock().expect("arena queue poisoned");
                loop {
                    if let Some(job) = state.take_descendant() {
                        break Next::Child(job);
                    }
                    if !state.failed && state.next_root < count && state.active_roots < max_roots {
                        let root = state.next_root;
                        state.next_root += 1;
                        state.active_roots += 1;
                        break Next::Root(root);
                    }
                    if state.active_roots == 0 && (state.failed || state.next_root == count) {
                        return;
                    }
                    state = self.changed.wait(state).expect("arena queue poisoned");
                }
            };
            match next {
                Next::Child(job) => self.execute(job),
                Next::Root(index) => {
                    let outcome = catch_unwind(AssertUnwindSafe(|| work(index)));
                    {
                        let mut state = self.state.lock().expect("arena queue poisoned");
                        state.active_roots -= 1;
                        match outcome {
                            Ok(ok) => state.failed |= !ok,
                            Err(panic) => {
                                state.failed = true;
                                if state.panic.is_none() {
                                    state.panic = Some(panic);
                                } else {
                                    std::mem::forget(panic);
                                }
                            }
                        }
                    }
                    self.changed.notify_all();
                }
            }
        }
    }
}

pub(super) fn for_each(count: usize, work: impl Fn(usize) + Sync) -> bool {
    current(|arena| match arena {
        Some(arena) => {
            arena.for_each(count, work);
            true
        }
        None => false,
    })
}

pub(super) fn run(
    count: usize,
    workers: usize,
    max_roots: usize,
    quiet: bool,
    work: impl Fn(usize) -> bool + Sync,
) -> bool {
    if count == 0 {
        return true;
    }
    assert!(workers > 0 && max_roots > 0);
    current(|existing| {
        if let Some(arena) = existing {
            crate::profile::note(
                "execution.cooperative",
                format_args!(
                    "jobs={} workers={} root_limit={} quiet={} nested=1",
                    count,
                    arena.workers,
                    count.min(max_roots),
                    usize::from(arena.quiet),
                ),
            );
            // Reentrant clients publish child groups, never another executor.
            // The fixed root loops retain the nested client's scratch cap.
            let queue = Mutex::new((0, false));
            arena.for_each(count.min(max_roots), |_| {
                loop {
                    let index = {
                        let mut queue = queue.lock().expect("nested root queue poisoned");
                        if queue.1 || queue.0 == count {
                            return;
                        }
                        let index = queue.0;
                        queue.0 += 1;
                        index
                    };
                    match catch_unwind(AssertUnwindSafe(|| work(index))) {
                        Ok(true) => {}
                        Ok(false) => {
                            queue.lock().expect("nested root queue poisoned").1 = true;
                            return;
                        }
                        Err(panic) => {
                            queue.lock().expect("nested root queue poisoned").1 = true;
                            resume_unwind(panic);
                        }
                    }
                }
            });
            return !queue.into_inner().expect("nested root queue poisoned").1;
        }
        let arena = Arena {
            workers: workers.min(rayon::current_num_threads()).max(1),
            quiet,
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        };
        crate::profile::note(
            "execution.cooperative",
            format_args!(
                "jobs={} workers={} root_limit={} quiet={} nested=0",
                count,
                arena.workers,
                count.min(max_roots),
                usize::from(quiet),
            ),
        );
        (0..arena.workers)
            .into_par_iter()
            .for_each(|_| arena.worker(count, max_roots, &work));
        let state = arena.state.into_inner().expect("arena queue poisoned");
        assert!(state.groups.is_empty());
        if let Some(panic) = state.panic {
            resume_unwind(panic);
        }
        !state.failed
    })
}

#[cfg(test)]
#[path = "arena_tests.rs"]
mod tests;
