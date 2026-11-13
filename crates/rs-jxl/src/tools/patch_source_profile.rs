//! Opt-in local detector accounting; no locks or clocks in pixel loops.
use std::fmt::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    crate::profile::enabled()
        && *ENABLED
            .get_or_init(|| std::env::var("RS_JXL_PATCH_SOURCE_PROFILE").as_deref() == Ok("1"))
}

#[derive(Clone, Copy)]
pub(super) enum Stage {
    SeedScan,
    BackgroundSetup,
    BackgroundFlood,
    ComponentScan,
    FingerprintDelta,
    GroupLookup,
    FinalizeDelta,
}

const NAMES: [&str; 7] = [
    "seed_scan",
    "background_setup",
    "background_flood",
    "component_scan",
    "fingerprint_delta",
    "group_lookup",
    "finalize_delta",
];

#[derive(Default)]
pub(super) struct Counts {
    pub seeds: usize,
    pub initial_queue: usize,
    pub components: usize,
    pub rectangles: usize,
    pub fingerprint_values: usize,
    pub delta_values: usize,
    pub groups: usize,
    pub retained_groups: usize,
}

pub(super) struct Detail<const ENABLED: bool> {
    stage: Stage,
    start: Option<Instant>,
    elapsed: [Duration; 7],
    pub counts: Counts,
}

impl<const ENABLED: bool> Detail<ENABLED> {
    pub fn new() -> Self {
        Self {
            stage: Stage::SeedScan,
            start: ENABLED.then(Instant::now),
            elapsed: [Duration::ZERO; 7],
            counts: Counts::default(),
        }
    }

    #[inline]
    pub fn switch(&mut self, next: Stage) {
        if ENABLED {
            let now = Instant::now();
            self.account(now);
            self.stage = next;
        }
    }

    fn account(&mut self, now: Instant) {
        if let Some(start) = self.start.replace(now) {
            self.elapsed[self.stage as usize] += now.duration_since(start);
        }
    }
}

impl<const ENABLED: bool> Drop for Detail<ENABLED> {
    fn drop(&mut self) {
        if !ENABLED {
            return;
        }
        self.account(Instant::now());
        let mut record = String::from("{\"stages_ns\":{");
        for (index, (name, time)) in NAMES.iter().zip(self.elapsed).enumerate() {
            if index != 0 {
                record.push(',');
            }
            write!(record, "\"{name}\":{}", time.as_nanos()).unwrap();
        }
        let c = &self.counts;
        write!(record,
            "}},\"counts\":{{\"seeds\":{},\"initial_queue\":{},\"components\":{},\"rectangles\":{},\"fingerprint_values\":{},\"delta_values\":{},\"groups\":{},\"retained_groups\":{}}}}}",
            c.seeds, c.initial_queue, c.components, c.rectangles,
            c.fingerprint_values, c.delta_values, c.groups, c.retained_groups,
        ).unwrap();
        crate::profile::note("source.patch.detail", record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_accounting_has_no_overlap() {
        let start = Instant::now();
        let mut detail = Detail::<false>::new();
        detail.start = Some(start);
        detail.account(start + Duration::from_millis(3));
        detail.stage = Stage::GroupLookup;
        detail.account(start + Duration::from_millis(8));
        assert_eq!(
            detail.elapsed[Stage::SeedScan as usize],
            Duration::from_millis(3)
        );
        assert_eq!(
            detail.elapsed[Stage::GroupLookup as usize],
            Duration::from_millis(5)
        );
        assert_eq!(
            detail.elapsed.iter().sum::<Duration>(),
            Duration::from_millis(8)
        );
    }

    #[test]
    fn disabled_switch_does_not_start_a_clock() {
        let mut detail = Detail::<false>::new();
        detail.switch(Stage::FinalizeDelta);
        assert!(detail.start.is_none());
        assert_eq!(detail.elapsed, [Duration::ZERO; 7]);
    }
}
