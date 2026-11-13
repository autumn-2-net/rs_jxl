//! Opt-in, invocation-owned controlled experiments, not a palette policy.
//! Install a fresh experiment per encode; inspect it only after workers finish.
//! Worker event order is unspecified. Only the selected input is copied in full.

#![forbid(unsafe_code)]

use super::{PaletteAttempt, PaletteMode, PaletteStage, PaletteState, PaletteTools};
use crate::modular::group::{Channel, Transform};
use crate::{Error, Result};
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Target {
    pub stage: PaletteStage,
    pub stream_id: usize,
    pub ordinal: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Choice {
    Keep,
    Skip,
    Palette(PaletteMode),
}

/// Exact original attempt metadata; float identity uses bits, including NaNs
/// and signed zero. This is not a digest or an equality heuristic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotDetails {
    pub state: PaletteState,
    pub bits: u8,
    pub begin: usize,
    pub count: usize,
    pub limit: u32,
    pub ordered: bool,
    pub mode: PaletteMode,
    pub tools: PaletteTools,
    pub cost_before_bits: u32,
    pub accept_without_cost: bool,
}

#[derive(Debug)]
pub struct Snapshot {
    target: Target,
    channels: Vec<Channel>,
    prefix: Vec<Transform>,
    details: SnapshotDetails,
}

impl Snapshot {
    pub fn target(&self) -> Target {
        self.target
    }

    pub fn base_mode(&self) -> PaletteMode {
        self.details.mode
    }

    /// All planes, including meta/extra channels and invisible RGB samples.
    pub fn input_channels(&self) -> &[Channel] {
        &self.channels
    }

    /// External accepted prefix followed by this call's accepted transforms.
    pub fn prefix(&self) -> &[Transform] {
        &self.prefix
    }

    pub fn details(&self) -> SnapshotDetails {
        self.details
    }

    fn capture(target: Target, input: &Input<'_>) -> Self {
        Self {
            target,
            channels: input.channels.to_vec(),
            prefix: input
                .prefix
                .iter()
                .chain(input.transforms)
                .cloned()
                .collect(),
            details: input.details(),
        }
    }

    fn verify(&self, input: &Input<'_>) -> Result<()> {
        if self.channels != input.channels {
            return Err(invalid("input channels or samples differ from snapshot"));
        }
        if !self
            .prefix
            .iter()
            .eq(input.prefix.iter().chain(input.transforms))
        {
            return Err(invalid("accepted transform prefix differs from snapshot"));
        }
        if self.details != input.details() {
            return Err(invalid(
                "state, bit depth, or attempt arguments differ from snapshot",
            ));
        }
        Ok(())
    }
}

pub(crate) struct Input<'a> {
    pub channels: &'a [Channel],
    pub state: PaletteState,
    pub prefix: &'a [Transform],
    pub transforms: &'a [Transform],
    pub bits: u8,
    pub begin: usize,
    pub count: usize,
    pub limit: u32,
    pub ordered: bool,
    pub mode: PaletteMode,
    pub tools: PaletteTools,
    pub cost_before: f32,
    pub accept_without_cost: bool,
}

impl Input<'_> {
    fn details(&self) -> SnapshotDetails {
        SnapshotDetails {
            state: self.state,
            bits: self.bits,
            begin: self.begin,
            count: self.count,
            limit: self.limit,
            ordered: self.ordered,
            mode: self.mode,
            tools: self.tools,
            cost_before_bits: self.cost_before.to_bits(),
            accept_without_cost: self.accept_without_cost,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub target: Target,
    pub choice: Choice,
    pub attempt: Option<PaletteAttempt>,
    pub unavailable: bool,
}

#[derive(Debug)]
enum Plan {
    Observe,
    Capture(Target),
    Replay(Arc<Snapshot>, Choice),
}

impl Plan {
    fn target(&self) -> Option<Target> {
        match self {
            Self::Observe => None,
            Self::Capture(target) => Some(*target),
            Self::Replay(snapshot, _) => Some(snapshot.target),
        }
    }
}

#[derive(Debug, Default)]
struct Progress {
    snapshot: Option<Arc<Snapshot>>,
    events: Vec<Record>,
    hit: bool,
    ready: bool,
    failure: Option<&'static str>,
    unavailable: bool,
}

#[derive(Debug)]
pub struct Experiment {
    plan: Plan,
    progress: Mutex<Progress>,
}

// As with PaletteReceipt, distinct invocations must not share cached policy.
impl PartialEq for Experiment {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Experiment {
    pub fn observe() -> Self {
        Self::new(Plan::Observe)
    }

    pub fn capture(target: Target) -> Self {
        Self::new(Plan::Capture(target))
    }

    pub fn replay(snapshot: Arc<Snapshot>, choice: Choice) -> Self {
        Self::new(Plan::Replay(snapshot, choice))
    }

    fn new(plan: Plan) -> Self {
        Self {
            plan,
            progress: Mutex::new(Progress::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Progress> {
        self.progress.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn snapshot(&self) -> Option<Arc<Snapshot>> {
        if let Plan::Replay(snapshot, _) = &self.plan {
            return Some(Arc::clone(snapshot));
        }
        self.lock().snapshot.clone()
    }

    pub fn events(&self) -> Vec<Record> {
        self.lock().events.clone()
    }

    /// Only the selected target contributes to this flag. An unavailable
    /// alternative is reported separately from input/target validation errors.
    pub fn unavailable(&self) -> bool {
        self.lock().unavailable
    }

    pub fn validate_complete(&self) -> Result<()> {
        let progress = self.lock();
        if let Some(reason) = progress.failure {
            return Err(invalid(reason));
        }
        if self.plan.target().is_some() {
            if !progress.hit {
                return Err(invalid("selected target was not visited"));
            }
            if !progress.ready {
                return Err(invalid("selected target capture/replay did not complete"));
            }
        }
        Ok(())
    }

    /// Called before any mutation or alternative selection, with ORIGINAL args.
    pub(crate) fn before(&self, target: Target, input: Input<'_>) -> Result<Choice> {
        if self.plan.target() != Some(target) {
            return Ok(Choice::Keep);
        }
        {
            let mut progress = self.lock();
            if progress.hit {
                progress.failure = Some("selected target was visited more than once");
                return Err(invalid("selected target was visited more than once"));
            }
            progress.hit = true;
        }
        // Full copies/comparisons happen outside the shared worker lock.
        let (snapshot, choice) = match &self.plan {
            Plan::Observe => unreachable!(),
            Plan::Capture(_) => (
                Some(Arc::new(Snapshot::capture(target, &input))),
                Choice::Keep,
            ),
            Plan::Replay(snapshot, choice) => {
                if let Err(error) = snapshot.verify(&input) {
                    self.lock().failure = Some("replay input did not exactly match snapshot");
                    return Err(error);
                }
                if let Choice::Palette(mode) = choice
                    && let Err(error) = mode.validate()
                {
                    self.lock().failure = Some("replay palette mode is invalid");
                    return Err(error);
                }
                (None, *choice)
            }
        };
        let mut progress = self.lock();
        progress.snapshot = snapshot;
        progress.ready = true;
        Ok(choice)
    }

    /// Record every visited attempt for discovery. The caller supplies None for
    /// Skip and the actual kernel/policy evidence otherwise; no channel copies.
    pub(crate) fn record(
        &self,
        target: Target,
        choice: Choice,
        attempt: Option<PaletteAttempt>,
        unavailable: bool,
    ) {
        let mut progress = self.lock();
        if self.plan.target() == Some(target) {
            progress.unavailable |= unavailable;
        }
        progress.events.push(Record {
            target,
            choice,
            attempt,
            unavailable,
        });
    }
}

fn invalid(reason: &str) -> Error {
    Error::InvalidInput(format!("palette counterfactual: {reason}"))
}

/// Ordinals belong to one stage/stream traversal, never worker scheduling.
pub(crate) struct Cursor<'a> {
    stage: PaletteStage,
    stream_id: usize,
    prefix: &'a [Transform],
    experiment: Option<&'a Experiment>,
    ordinal: usize,
}

pub(crate) struct Visit<'a> {
    pub experiment: &'a Experiment,
    pub target: Target,
    pub prefix: &'a [Transform],
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(
        stage: PaletteStage,
        stream_id: usize,
        prefix: &'a [Transform],
        experiment: Option<&'a Experiment>,
    ) -> Self {
        Self {
            stage,
            stream_id,
            prefix,
            experiment,
            ordinal: 0,
        }
    }

    pub(crate) fn next(&mut self) -> Option<Visit<'a>> {
        let ordinal = self.ordinal;
        self.ordinal = self
            .ordinal
            .checked_add(1)
            .expect("palette ordinal overflow");
        self.experiment.map(|experiment| Visit {
            experiment,
            target: Target {
                stage: self.stage,
                stream_id: self.stream_id,
                ordinal,
            },
            prefix: self.prefix,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::palette::{
        PaletteGeometry, PaletteKernelEvidence, PaletteMapping, PaletteOrder, PaletteOutcome,
    };
    use crate::modular::predictor::Predictor;

    fn target(stream_id: usize) -> Target {
        Target {
            stage: PaletteStage::PostRct,
            stream_id,
            ordinal: 0,
        }
    }

    fn channels() -> Vec<Channel> {
        // Transparent RGBA plus an extra plane: samples outside the attempted
        // range are still part of exact replay identity.
        [11, 22, 33, 0, 99]
            .map(|value| Channel {
                width: 2,
                height: 1,
                hshift: 0,
                vshift: 0,
                samples: vec![value; 2],
            })
            .to_vec()
    }

    fn input(channels: &[Channel]) -> Input<'_> {
        Input {
            channels,
            state: PaletteState::new(8).unwrap(),
            prefix: &[],
            transforms: &[],
            bits: 8,
            begin: 0,
            count: 3,
            limit: 16,
            ordered: true,
            mode: PaletteMode::Absolute,
            tools: PaletteTools::default(),
            cost_before: 0.0,
            accept_without_cost: false,
        }
    }

    fn captured() -> Arc<Snapshot> {
        let experiment = Experiment::capture(target(7));
        assert_eq!(
            experiment.before(target(7), input(&channels())).unwrap(),
            Choice::Keep
        );
        experiment.validate_complete().unwrap();
        experiment.snapshot().unwrap()
    }

    fn attempt() -> PaletteAttempt {
        PaletteAttempt {
            stage: PaletteStage::PostRct,
            mode: PaletteMode::Absolute,
            width: 2,
            height: 1,
            begin: 0,
            channels: 3,
            limit: 16,
            kernel: PaletteKernelEvidence::default(),
            colors: 0,
            deltas: 0,
            predictor: Predictor::Zero,
            delta_pixels: 0,
            cost_before: 0.0,
            cost_after: None,
            cost_bypassed: false,
            outcome: PaletteOutcome::KernelDeclined,
        }
    }

    #[test]
    fn capture_and_replay_choices_keep_one_shared_snapshot() {
        let snapshot = captured();
        assert_eq!(snapshot.target(), target(7));
        assert_eq!(snapshot.base_mode(), PaletteMode::Absolute);
        assert_eq!(snapshot.input_channels(), channels());
        assert_eq!(snapshot.details(), input(&channels()).details());
        for choice in [
            Choice::Keep,
            Choice::Skip,
            Choice::Palette(PaletteMode::ExactZero(Predictor::Left)),
        ] {
            let experiment = Experiment::replay(Arc::clone(&snapshot), choice);
            assert_eq!(
                experiment.before(target(7), input(&channels())).unwrap(),
                choice
            );
            let evidence = (choice != Choice::Skip).then(attempt);
            experiment.record(target(7), choice, evidence.clone(), false);
            experiment.validate_complete().unwrap();
            assert_eq!(
                experiment.events(),
                [Record {
                    target: target(7),
                    choice,
                    attempt: evidence,
                    unavailable: false
                }]
            );
            assert!(Arc::ptr_eq(&snapshot, &experiment.snapshot().unwrap()));
        }
    }

    #[test]
    fn every_plane_sample_and_channel_metadata_must_match() {
        let snapshot = captured();
        for plane in 0..channels().len() {
            for field in 0..6 {
                let mut changed = channels();
                let channel = &mut changed[plane];
                match field {
                    0 => channel.samples[0] += 1,
                    1 => channel.samples[1] += 1,
                    2 => channel.width += 1,
                    3 => channel.height += 1,
                    4 => channel.hshift += 1,
                    _ => channel.vshift += 1,
                }
                let replay = Experiment::replay(Arc::clone(&snapshot), Choice::Skip);
                assert!(replay.before(target(7), input(&changed)).is_err());
                assert!(replay.validate_complete().is_err());
            }
        }
        let replay = Experiment::replay(snapshot, Choice::Keep);
        assert!(replay.before(target(7), input(&channels()[..4])).is_err());
    }

    #[test]
    fn exact_prefix_is_concatenated_and_order_sensitive() {
        let channels = channels();
        let prefix = [Transform::Rct(6)];
        let transforms = [Transform::RctAt { begin: 1, kind: 2 }];
        let capture = Experiment::capture(target(7));
        capture
            .before(
                target(7),
                Input {
                    prefix: &prefix,
                    transforms: &transforms,
                    ..input(&channels)
                },
            )
            .unwrap();
        let snapshot = capture.snapshot().unwrap();
        let combined = [prefix[0].clone(), transforms[0].clone()];
        assert_eq!(snapshot.prefix(), &combined);
        let replay = Experiment::replay(Arc::clone(&snapshot), Choice::Keep);
        replay
            .before(
                target(7),
                Input {
                    prefix: &combined,
                    ..input(&channels)
                },
            )
            .unwrap();
        replay.validate_complete().unwrap();
        for changed in [
            vec![],
            vec![Transform::Rct(1), transforms[0].clone()],
            vec![transforms[0].clone(), prefix[0].clone()],
        ] {
            let replay = Experiment::replay(Arc::clone(&snapshot), Choice::Skip);
            assert!(
                replay
                    .before(
                        target(7),
                        Input {
                            transforms: &changed,
                            ..input(&channels)
                        }
                    )
                    .is_err()
            );
            assert!(replay.validate_complete().is_err());
        }
    }

    #[test]
    fn every_attempt_argument_and_state_field_must_match() {
        let snapshot = captured();
        let channels = channels();
        for field in 0..16 {
            let mut changed = input(&channels);
            match field {
                0 => changed.state.meta_channels += 1,
                1 => changed.state.max_bitdepth += 1,
                2 => changed.state.max_value += 1,
                3 => changed.bits += 1,
                4 => changed.begin += 1,
                5 => changed.count += 1,
                6 => changed.limit += 1,
                7 => changed.ordered = false,
                8 => changed.mode = PaletteMode::ExactDelta(Predictor::Left),
                9 => changed.tools.order = PaletteOrder::Morton,
                10 => changed.cost_before = -0.0,
                11 => changed.cost_before = f32::from_bits(1),
                12 => changed.cost_before = f32::NAN,
                13 => changed.tools.geometry = PaletteGeometry::Principal,
                14 => changed.tools.mapping = PaletteMapping::Flat,
                _ => changed.accept_without_cost = true,
            }
            let replay = Experiment::replay(Arc::clone(&snapshot), Choice::Skip);
            assert!(replay.before(target(7), changed).is_err(), "field {field}");
            assert!(replay.validate_complete().is_err());
        }
        let capture = Experiment::capture(target(7));
        let nan = f32::from_bits(0x7fc00001);
        capture
            .before(
                target(7),
                Input {
                    cost_before: nan,
                    ..input(&channels)
                },
            )
            .unwrap();
        let snapshot = capture.snapshot().unwrap();
        let replay = Experiment::replay(Arc::clone(&snapshot), Choice::Keep);
        replay
            .before(
                target(7),
                Input {
                    cost_before: nan,
                    ..input(&channels)
                },
            )
            .unwrap();
        replay.validate_complete().unwrap();
        let replay = Experiment::replay(snapshot, Choice::Keep);
        assert!(
            replay
                .before(
                    target(7),
                    Input {
                        cost_before: f32::from_bits(0x7fc00002),
                        ..input(&channels)
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn missing_duplicate_and_invalid_replay_remain_errors() {
        for experiment in [
            Experiment::capture(target(7)),
            Experiment::replay(captured(), Choice::Skip),
        ] {
            assert!(experiment.validate_complete().is_err());
            experiment.before(target(7), input(&channels())).unwrap();
            experiment.validate_complete().unwrap();
            assert!(experiment.before(target(7), input(&channels())).is_err());
            assert!(experiment.validate_complete().is_err());
        }
        let replay = Experiment::replay(
            captured(),
            Choice::Palette(PaletteMode::ExactDelta(Predictor::Weighted)),
        );
        assert!(replay.before(target(7), input(&channels())).is_err());
        assert!(replay.validate_complete().is_err());
    }

    #[test]
    fn target_isolated_by_stream_stage_and_local_ordinal() {
        let experiment = Experiment::replay(captured(), Choice::Skip);
        let mut other = Cursor::new(PaletteStage::PostRct, 8, &[], Some(&experiment));
        let mut selected = Cursor::new(PaletteStage::PostRct, 7, &[], Some(&experiment));
        let mut stage = Cursor::new(PaletteStage::PreRct, 7, &[], Some(&experiment));
        for visit in [other.next().unwrap(), stage.next().unwrap()] {
            assert_eq!(visit.target.ordinal, 0);
            assert_eq!(
                visit
                    .experiment
                    .before(visit.target, input(&channels()))
                    .unwrap(),
                Choice::Keep
            );
            visit
                .experiment
                .record(visit.target, Choice::Keep, Some(attempt()), true);
        }
        assert!(!experiment.unavailable());
        assert!(experiment.validate_complete().is_err());
        let visit = selected.next().unwrap();
        assert!(visit.prefix.is_empty());
        assert_eq!(
            visit
                .experiment
                .before(visit.target, input(&channels()))
                .unwrap(),
            Choice::Skip
        );
        experiment.record(visit.target, Choice::Skip, None, true);
        assert!(experiment.unavailable());
        experiment.validate_complete().unwrap();
        let visit = selected.next().unwrap();
        assert_eq!(visit.target.ordinal, 1);
        assert_eq!(
            experiment.before(visit.target, input(&channels())).unwrap(),
            Choice::Keep
        );
        assert_eq!(other.next().unwrap().target.ordinal, 1);
        assert!(
            Cursor::new(PaletteStage::PreRct, 0, &[], None)
                .next()
                .is_none()
        );
    }

    #[test]
    fn concurrent_discovery_records_are_invocation_owned_without_snapshots() {
        let first = Arc::new(Experiment::observe());
        let second = Arc::new(Experiment::observe());
        assert_eq!(first, Arc::clone(&first));
        assert_ne!(first, second);
        std::thread::scope(|scope| {
            for (experiment, base) in [(&first, 0), (&second, 100)] {
                for stream_id in base..base + 8 {
                    let experiment = Arc::clone(experiment);
                    scope.spawn(move || {
                        let prefix = [Transform::Rct(6)];
                        let mut cursor = Cursor::new(
                            PaletteStage::PostRct,
                            stream_id,
                            &prefix,
                            Some(&experiment),
                        );
                        for ordinal in 0..4 {
                            let visit = cursor.next().unwrap();
                            assert_eq!(visit.target.ordinal, ordinal);
                            assert_eq!(visit.prefix, &prefix);
                            assert_eq!(
                                experiment.before(visit.target, input(&channels())).unwrap(),
                                Choice::Keep
                            );
                            experiment.record(visit.target, Choice::Keep, Some(attempt()), true);
                        }
                    });
                }
            }
        });
        for (experiment, base) in [(&first, 0), (&second, 100)] {
            experiment.validate_complete().unwrap();
            assert!(experiment.snapshot().is_none());
            assert!(!experiment.unavailable());
            let events = experiment.events();
            assert_eq!(events.len(), 32);
            assert!(
                events
                    .iter()
                    .all(|e| (base..base + 8).contains(&e.target.stream_id))
            );
            for stream_id in base..base + 8 {
                for ordinal in 0..4 {
                    assert_eq!(
                        events
                            .iter()
                            .filter(|e| e.target
                                == Target {
                                    ordinal,
                                    ..target(stream_id)
                                })
                            .count(),
                        1
                    );
                }
            }
        }
    }
}
