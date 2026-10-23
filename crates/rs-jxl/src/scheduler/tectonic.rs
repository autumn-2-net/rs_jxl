// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Pixel-lossless effort 11 tournament from frozen b866b3d enc_frame.cc:
//! TectonicPlateSettingsLessPalette (2466), MorePalette (2573), and
//! EncodeFrameImpl's kTectonicPlate branch (2745). This is separate from the
//! direct tier -1 Modular backend and from enc_lossless_search's effort 1..10.
//!
//! Both complete rounds are measured without pruning, deduplication or sample
//! proxies. Source trial bytes are discarded; the winner is encoded AGAIN.
//! Callbacks must encode the supplied candidate directly, with automatic
//! lossless search disabled. Every supplied candidate has tier 0, never -1.
//! The generic driver does not reinterpret pixels or impose the separate
//! enc_lossless_search <=16-bit gate; backend representation errors propagate.

use super::params::{CandidateParams, Override, TreeMode};
use super::search::{EncodedOutput, SearchLimits, TrialBudget};
use crate::{Error, Image, Result, modular::predictor::Predictor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TectonicBranch {
    LessPalette,
    MorePalette,
}

#[derive(Clone, Debug)]
pub struct TectonicTrial {
    pub params: CandidateParams,
    /// Complete output size, not a lower bound or a probe estimate.
    pub bytes: usize,
}

#[derive(Clone, Debug)]
pub struct TectonicReport {
    pub branch: TectonicBranch,
    /// Source order: the two initial trials, followed by the selected branch.
    /// The final re-encode is accounted below, not treated as another trial.
    pub trials: Vec<TectonicTrial>,
    pub winner_index: usize,
    pub final_bytes: usize,
    /// Includes the final re-encode: 27 for LessPalette, 23 for MorePalette.
    pub full_encodes: usize,
    pub full_pixels: u64,
}

#[derive(Debug)]
pub struct TectonicOutput<A = ()> {
    pub bytes: Vec<u8>,
    pub params: CandidateParams,
    /// Artifact from the FINAL encode, never an earlier trial's artifact.
    pub artifact: A,
    pub report: TectonicReport,
}

/// A direct lossless encoder, not an effort-dispatching or automatic-search API.
/// Trial sizes must use one consistent complete-output measure. C++ measures
/// local frame output (including special frames); a complete codestream size
/// is equivalent for selection only when its extra overhead is candidate-
/// independent. Do not silently substitute main-frame-only sizes for patches.
pub trait TectonicOracle<A = ()> {
    /// Called for the final re-encode, and by the default serial round method.
    /// The final encode may use the caller's normal inner worker pool.
    fn encode(
        &mut self,
        image: &Image,
        params: &CandidateParams,
        budget: TrialBudget,
    ) -> Result<EncodedOutput<A>>;

    /// C++ RunOnPool executes a round's trials in parallel with a NULL inner
    /// runner. Override this to use the caller's bounded shared runner, with
    /// one inner worker per candidate. Return sizes in INPUT order regardless
    /// of completion order and drop trial buffers/artifacts after measuring.
    /// The default executes serially, retaining only one trial output at once;
    /// it changes execution scheduling, not candidates or selection policy.
    fn measure_round(
        &mut self,
        image: &Image,
        candidates: &[CandidateParams],
        budget: TrialBudget,
    ) -> Result<Vec<usize>> {
        let mut sizes = Vec::with_capacity(candidates.len());
        for params in candidates {
            let output = self.encode(image, params, budget)?;
            check_output_size(output.bytes.len(), budget.max_output_bytes)?;
            sizes.push(output.bytes.len());
        }
        Ok(sizes)
    }
}

impl<A, F> TectonicOracle<A> for F
where
    F: FnMut(&Image, &CandidateParams, TrialBudget) -> Result<EncodedOutput<A>>,
{
    fn encode(
        &mut self,
        image: &Image,
        params: &CandidateParams,
        budget: TrialBudget,
    ) -> Result<EncodedOutput<A>> {
        self(image, params, budget)
    }
}

/// Execute the complete source tournament, then re-encode its winner.
/// Caller resource ceilings are NOT source search shortcuts: an insufficient
/// budget returns an error, never a partial "best" output. Reserve the minimum
/// 23 calls before starting, and the exact branch plus final call before round
/// two. Callbacks must enforce the supplied output ceiling during encoding;
/// returned sizes are also checked here. No incumbent-based abort is allowed.
pub fn encode<A, O: TectonicOracle<A> + ?Sized>(
    image: &Image,
    input: &CandidateParams,
    limits: SearchLimits,
    oracle: &mut O,
) -> Result<TectonicOutput<A>> {
    let initial = initial_candidates(input)?;
    check_work(image, limits, 23)?;
    let budget = TrialBudget {
        beat_bytes: None,
        max_output_bytes: limits.max_output_bytes,
    };
    let initial_sizes = oracle.measure_round(image, &initial, budget)?;
    check_round(&initial_sizes, initial.len(), limits.max_output_bytes)?;
    let initial_best = usize::from(initial_sizes[0] > initial_sizes[1]);
    let branch = if initial_best == 0 {
        TectonicBranch::LessPalette
    } else {
        TectonicBranch::MorePalette
    };
    let candidates = branch_candidates(input, branch)?;
    let full_encodes = initial.len() + candidates.len() + 1;
    let full_pixels = check_work(image, limits, full_encodes)?;
    let sizes = oracle.measure_round(image, &candidates, budget)?;
    check_round(&sizes, candidates.len(), limits.max_output_bytes)?;
    let mut branch_best = 0;
    for i in 1..sizes.len() {
        if sizes[branch_best] > sizes[i] {
            branch_best = i;
        }
    }
    let winner_index = if sizes[branch_best] < initial_sizes[initial_best] {
        initial.len() + branch_best
    } else {
        initial_best
    };
    let trials: Vec<_> = initial
        .into_iter()
        .zip(initial_sizes)
        .chain(candidates.into_iter().zip(sizes))
        .map(|(params, bytes)| TectonicTrial { params, bytes })
        .collect();
    let params = trials[winner_index].params.clone();
    let output = oracle.encode(image, &params, budget)?;
    check_output_size(output.bytes.len(), limits.max_output_bytes)?;
    Ok(TectonicOutput {
        report: TectonicReport {
            branch,
            trials,
            winner_index,
            final_bytes: output.bytes.len(),
            full_encodes,
            full_pixels,
        },
        bytes: output.bytes,
        artifact: output.artifact,
        params,
    })
}

fn check_work(image: &Image, limits: SearchLimits, calls: usize) -> Result<u64> {
    let pixels = u64::from(image.width()) * u64::from(image.height());
    let work = pixels
        .checked_mul(calls as u64)
        .ok_or(Error::Limit("tectonic tournament pixel work"))?;
    if calls > limits.max_full_encodes || work > limits.max_full_pixels {
        return Err(Error::Limit("tectonic tournament work budget"));
    }
    if limits.max_output_bytes == 0 {
        return Err(Error::Limit("tectonic tournament output budget"));
    }
    Ok(work)
}

fn check_output_size(bytes: usize, limit: usize) -> Result<()> {
    if bytes == 0 {
        return Err(Error::Encoding(
            "tectonic oracle returned an empty complete output".into(),
        ));
    }
    if bytes > limit {
        return Err(Error::Limit("tectonic tournament output budget"));
    }
    Ok(())
}

fn check_round(sizes: &[usize], expected: usize, limit: usize) -> Result<()> {
    if sizes.len() != expected {
        return Err(Error::Encoding(
            "tectonic oracle returned an incomplete round".into(),
        ));
    }
    for &size in sizes {
        check_output_size(size, limit)?;
    }
    Ok(())
}

fn validate_input(input: &CandidateParams) -> Result<()> {
    input.validate()?;
    if input.speed_tier != -1 {
        return Err(Error::InvalidInput(
            "tectonic tournament requires source speed_tier=-1 (effort 11)".into(),
        ));
    }
    Ok(())
}

/// The two palette-discrimination candidates at enc_frame.cc:2747.
pub fn initial_candidates(input: &CandidateParams) -> Result<[CandidateParams; 2]> {
    validate_input(input)?;
    let first = CandidateParams {
        speed_tier: 0,
        max_properties: 4,
        learning_fraction: 1.0,
        group_size_shift: Some(3),
        palette_colors: 0,
        predictor: Some(Predictor::Variable),
        channel_colors_percent: 80.0,
        channel_colors_pre_transform_percent: 95.0,
        tree_mode: TreeMode::Default,
        patches: Override::Default,
        ..input.clone()
    };
    let second = CandidateParams {
        predictor: Some(Predictor::Zero),
        learning_fraction: 0.01,
        palette_colors: 70000,
        patches: Override::Off,
        tree_mode: TreeMode::NoWeighted,
        ..first.clone()
    };
    Ok([first, second])
}

/// Exact ordered branch from the ORIGINAL inputs, not the initial winner.
/// Fractions are raw nb_repeats; the Modular adapter applies its source scale.
pub fn branch_candidates(
    input: &CandidateParams,
    branch: TectonicBranch,
) -> Result<Vec<CandidateParams>> {
    validate_input(input)?;
    Ok(match branch {
        TectonicBranch::LessPalette => less_palette(input),
        TectonicBranch::MorePalette => more_palette(input),
    })
}

fn less_palette(input: &CandidateParams) -> Vec<CandidateParams> {
    let mut all = Vec::with_capacity(24);
    let mut p = CandidateParams {
        speed_tier: 0,
        max_properties: 4,
        learning_fraction: 1.0,
        group_size_shift: Some(0),
        channel_colors_percent: 0.0,
        predictor: Some(Predictor::Variable),
        channel_colors_pre_transform_percent: 95.0,
        palette_colors: 1024,
        tree_mode: TreeMode::Default,
        patches: Override::Default,
        ..input.clone()
    };
    all.push(p.clone());
    p.channel_colors_percent = 80.0;
    p.group_size_shift = Some(1);
    p.palette_colors = 0;
    p.channel_colors_pre_transform_percent = 0.0;
    all.push(p.clone());
    p.channel_colors_pre_transform_percent = 95.0;
    p.group_size_shift = Some(2);
    all.push(p.clone());
    p.group_size_shift = Some(3);
    p.patches = Override::Off;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.palette_colors = 1024;
    p.tree_mode = TreeMode::Default;
    all.push(p.clone());
    p.patches = Override::Default;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.tree_mode = TreeMode::Default;
    p.channel_colors_pre_transform_percent = 0.0;
    all.push(p.clone());
    p.channel_colors_pre_transform_percent = 95.0;
    p.learning_fraction = 0.9;
    p.group_size_shift = Some(2);
    all.push(p.clone());
    p.group_size_shift = Some(3);
    p.palette_colors = 0;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.tree_mode = TreeMode::Default;
    p.channel_colors_pre_transform_percent = 0.0;
    all.push(p.clone());
    p.palette_colors = 1024;
    p.learning_fraction = 0.95;
    p.group_size_shift = Some(1);
    p.channel_colors_percent = 0.0;
    all.push(p.clone());
    p.group_size_shift = Some(2);
    p.palette_colors = 0;
    all.push(p.clone());
    p.channel_colors_percent = 80.0;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.palette_colors = 1024;
    p.channel_colors_pre_transform_percent = 95.0;
    p.tree_mode = TreeMode::Default;
    p.group_size_shift = Some(3);
    all.push(p.clone());
    p.palette_colors = 0;
    p.patches = Override::Off;
    all.push(p.clone());
    p.patches = Override::Default;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.palette_colors = 1024;
    p.patches = Override::Off;
    all.push(p.clone());
    p.learning_fraction = 0.5;
    p.patches = Override::Default;
    p.tree_mode = TreeMode::Default;
    all.push(p.clone());
    p.predictor = Some(Predictor::Zero);
    p.learning_fraction = 0.0;
    p.channel_colors_percent = 0.0;
    p.channel_colors_pre_transform_percent = 0.0;
    p.patches = Override::Off;
    all.push(p.clone());
    p.channel_colors_percent = 80.0;
    p.channel_colors_pre_transform_percent = 95.0;
    p.learning_fraction = 1.0;
    p.palette_colors = 0;
    all.push(p.clone());
    p.patches = Override::Default;
    p.predictor = Some(Predictor::Best);
    all.push(p.clone());
    p.learning_fraction = 0.9;
    p.patches = Override::Off;
    all.push(p.clone());
    p.palette_colors = 1024;
    p.patches = Override::Default;
    p.predictor = Some(Predictor::Weighted);
    p.learning_fraction = 1.0;
    all.push(p.clone());
    p.learning_fraction = 0.95;
    p.group_size_shift = Some(2);
    p.palette_colors = 0;
    p.channel_colors_pre_transform_percent = 0.0;
    all.push(p);
    all
}

fn more_palette(input: &CandidateParams) -> Vec<CandidateParams> {
    let mut all = Vec::with_capacity(20);
    let mut p = CandidateParams {
        speed_tier: 0,
        max_properties: 4,
        learning_fraction: 1.0,
        group_size_shift: Some(0),
        palette_colors: 70000,
        predictor: Some(Predictor::Variable),
        channel_colors_percent: 80.0,
        channel_colors_pre_transform_percent: 95.0,
        tree_mode: TreeMode::Default,
        patches: Override::Default,
        ..input.clone()
    };
    all.push(p.clone());
    p.group_size_shift = Some(2);
    p.channel_colors_percent = 0.0;
    p.patches = Override::Off;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.channel_colors_percent = 80.0;
    p.tree_mode = TreeMode::Default;
    p.group_size_shift = Some(3);
    all.push(p.clone());
    p.learning_fraction = 0.9;
    all.push(p.clone());
    p.patches = Override::Default;
    p.learning_fraction = 0.95;
    p.group_size_shift = Some(0);
    all.push(p.clone());
    p.group_size_shift = Some(3);
    all.push(p.clone());
    p.patches = Override::Off;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.learning_fraction = 0.5;
    all.push(p.clone());
    p.tree_mode = TreeMode::Default;
    p.predictor = Some(Predictor::Zero);
    p.learning_fraction = 0.0;
    all.push(p.clone());
    p.patches = Override::Default;
    p.channel_colors_pre_transform_percent = 0.0;
    all.push(p.clone());
    p.learning_fraction = 0.01;
    p.palette_colors = 0;
    p.patches = Override::Off;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.channel_colors_pre_transform_percent = 95.0;
    p.tree_mode = TreeMode::Default;
    p.palette_colors = 70000;
    all.push(p.clone());
    p.learning_fraction = 1.0;
    p.group_size_shift = Some(0);
    p.channel_colors_percent = 0.0;
    p.channel_colors_pre_transform_percent = 0.0;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.channel_colors_pre_transform_percent = 95.0;
    p.group_size_shift = Some(1);
    all.push(p.clone());
    p.group_size_shift = Some(2);
    all.push(p.clone());
    p.channel_colors_percent = 80.0;
    p.tree_mode = TreeMode::Default;
    p.group_size_shift = Some(3);
    all.push(p.clone());
    p.learning_fraction = 0.5;
    p.group_size_shift = Some(1);
    p.channel_colors_percent = 0.0;
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.tree_mode = TreeMode::Default;
    p.group_size_shift = Some(2);
    all.push(p.clone());
    p.channel_colors_percent = 80.0;
    p.group_size_shift = Some(3);
    p.tree_mode = TreeMode::NoWeighted;
    all.push(p.clone());
    p.tree_mode = TreeMode::Default;
    p.predictor = Some(Predictor::Select);
    p.learning_fraction = 1.0;
    all.push(p);
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::adapter::{
        NativeResourcePolicy, project_native, source_patch_reference_policy,
    };
    use crate::scheduler::params::{DeltaPalette, SourceBuffering};
    use crate::{EncodeOptions, PixelData, PixelLayout};
    use Override::{Default as A, Off as O};
    use Predictor::{Best as B, Select as S, Variable as V, Weighted as W, Zero as Z};
    use TreeMode::{Default as D, NoWeighted as N};

    // Independent snapshots of each source push_back, not produced by the
    // mutation-based generators under test. Tuple: group, palette, channel%,
    // pre-transform%, predictor, RAW fraction, tree mode, patches.
    type Snapshot = (u8, u32, u8, u8, Predictor, f32, TreeMode, Override);
    const INITIAL: [Snapshot; 2] = [
        (3, 0, 80, 95, V, 1.0, D, A),
        (3, 70000, 80, 95, Z, 0.01, N, O),
    ];
    const LESS: [Snapshot; 24] = [
        (0, 1024, 0, 95, V, 1.0, D, A),
        (1, 0, 80, 0, V, 1.0, D, A),
        (2, 0, 80, 95, V, 1.0, D, A),
        (3, 0, 80, 95, V, 1.0, N, O),
        (3, 1024, 80, 95, V, 1.0, D, O),
        (3, 1024, 80, 95, V, 1.0, N, A),
        (3, 1024, 80, 0, V, 1.0, D, A),
        (2, 1024, 80, 95, V, 0.9, D, A),
        (3, 0, 80, 95, V, 0.9, N, A),
        (3, 0, 80, 0, V, 0.9, D, A),
        (1, 1024, 0, 0, V, 0.95, D, A),
        (2, 0, 0, 0, V, 0.95, D, A),
        (2, 0, 80, 0, V, 0.95, N, A),
        (3, 1024, 80, 95, V, 0.95, D, A),
        (3, 0, 80, 95, V, 0.95, D, O),
        (3, 0, 80, 95, V, 0.95, N, A),
        (3, 1024, 80, 95, V, 0.95, N, O),
        (3, 1024, 80, 95, V, 0.5, D, A),
        (3, 1024, 0, 0, Z, 0.0, D, O),
        (3, 0, 80, 95, Z, 1.0, D, O),
        (3, 0, 80, 95, B, 1.0, D, A),
        (3, 0, 80, 95, B, 0.9, D, O),
        (3, 1024, 80, 95, W, 1.0, D, A),
        (2, 0, 80, 0, W, 0.95, D, A),
    ];
    const MORE: [Snapshot; 20] = [
        (0, 70000, 80, 95, V, 1.0, D, A),
        (2, 70000, 0, 95, V, 1.0, N, O),
        (3, 70000, 80, 95, V, 1.0, D, O),
        (3, 70000, 80, 95, V, 0.9, D, O),
        (0, 70000, 80, 95, V, 0.95, D, A),
        (3, 70000, 80, 95, V, 0.95, D, A),
        (3, 70000, 80, 95, V, 0.95, N, O),
        (3, 70000, 80, 95, V, 0.5, N, O),
        (3, 70000, 80, 95, Z, 0.0, D, O),
        (3, 70000, 80, 0, Z, 0.0, D, A),
        (3, 0, 80, 0, Z, 0.01, N, O),
        (3, 70000, 80, 95, Z, 0.01, D, O),
        (0, 70000, 0, 0, Z, 1.0, N, O),
        (1, 70000, 0, 95, Z, 1.0, N, O),
        (2, 70000, 0, 95, Z, 1.0, N, O),
        (3, 70000, 80, 95, Z, 1.0, D, O),
        (1, 70000, 0, 95, Z, 0.5, N, O),
        (2, 70000, 0, 95, Z, 0.5, D, O),
        (3, 70000, 80, 95, Z, 0.5, N, O),
        (3, 70000, 80, 95, S, 1.0, D, O),
    ];

    fn input() -> CandidateParams {
        CandidateParams {
            speed_tier: -1,
            ..Default::default()
        }
    }

    fn image() -> Image {
        Image::new(2, 3, PixelLayout::Gray, PixelData::U8(vec![7; 6])).unwrap()
    }

    fn limits(calls: usize) -> SearchLimits {
        SearchLimits {
            max_full_encodes: calls,
            max_full_pixels: calls as u64 * 6,
            max_output_bytes: 256,
        }
    }

    fn expected(input: &CandidateParams, rows: &[Snapshot]) -> Vec<CandidateParams> {
        rows.iter()
            .map(
                |&(group, palette, channel, pre, predictor, fraction, mode, patches)| {
                    CandidateParams {
                        speed_tier: 0,
                        max_properties: 4,
                        group_size_shift: Some(group),
                        palette_colors: palette,
                        channel_colors_percent: f32::from(channel),
                        channel_colors_pre_transform_percent: f32::from(pre),
                        predictor: Some(predictor),
                        learning_fraction: fraction,
                        tree_mode: mode,
                        patches,
                        ..input.clone()
                    }
                },
            )
            .collect()
    }

    fn rows(branch: TectonicBranch) -> &'static [Snapshot] {
        match branch {
            TectonicBranch::LessPalette => &LESS,
            TectonicBranch::MorePalette => &MORE,
        }
    }

    fn sequence(input: &CandidateParams, branch: TectonicBranch) -> Vec<CandidateParams> {
        expected(input, &INITIAL)
            .into_iter()
            .chain(expected(input, rows(branch)))
            .collect()
    }

    struct MockOracle {
        expected: Vec<CandidateParams>,
        sizes: Vec<usize>,
        calls: usize,
        fail_at: Option<usize>,
    }

    impl MockOracle {
        fn new(branch: TectonicBranch, sizes: Vec<usize>, winner: usize) -> Self {
            let mut expected = sequence(&input(), branch);
            expected.push(expected[winner].clone());
            assert_eq!(expected.len(), sizes.len());
            Self {
                expected,
                sizes,
                calls: 0,
                fail_at: None,
            }
        }
    }

    impl TectonicOracle<usize> for MockOracle {
        fn encode(
            &mut self,
            image: &Image,
            params: &CandidateParams,
            budget: TrialBudget,
        ) -> Result<EncodedOutput<usize>> {
            assert_eq!((image.width(), image.height()), (2, 3));
            let call = self.calls;
            self.calls += 1;
            assert_eq!(params, &self.expected[call], "source candidate {call}");
            assert_eq!(params.speed_tier, 0, "no recursive e11 dispatch");
            assert_eq!(budget.beat_bytes, None, "all sizes must be complete");
            assert_eq!(budget.max_output_bytes, 256);
            if self.fail_at == Some(call) {
                return Err(Error::Encoding("mock backend failure".into()));
            }
            Ok(EncodedOutput {
                bytes: vec![call as u8; self.sizes[call]],
                artifact: call,
            })
        }
    }

    fn run_case(branch: TectonicBranch, mut scores: Vec<usize>, winner: usize) {
        let trials = sequence(&input(), branch);
        assert_eq!(scores.len(), trials.len());
        let trial_scores = scores.clone();
        // Deliberately different from the measured winner: return final bytes
        // without a hidden race/reselection or reuse of its previous output.
        scores.push(213);
        let calls = scores.len();
        let mut oracle = MockOracle::new(branch, scores, winner);
        let output = encode(&image(), &input(), limits(calls), &mut oracle).unwrap();
        assert_eq!(oracle.calls, calls);
        assert_eq!(output.report.branch, branch);
        assert_eq!(output.report.winner_index, winner);
        assert_eq!(output.params, trials[winner]);
        assert_eq!(output.report.full_encodes, calls);
        assert_eq!(output.report.full_pixels, calls as u64 * 6);
        assert_eq!(output.report.final_bytes, 213);
        assert_eq!(output.bytes, vec![(calls - 1) as u8; 213]);
        assert_eq!(output.artifact, calls - 1);
        assert_eq!(output.report.trials.len(), trials.len());
        for (i, trial) in output.report.trials.iter().enumerate() {
            assert_eq!(trial.params, trials[i]);
            assert_eq!(trial.bytes, trial_scores[i]);
        }
    }

    #[test]
    fn all_source_candidate_snapshots_and_unmodified_inputs() {
        let custom = CandidateParams {
            buffering: SourceBuffering::Streaming,
            responsive: true,
            group_size_shift: Some(1),
            rct: Some(17),
            palette_colors: 19,
            channel_colors_percent: 27.0,
            channel_colors_pre_transform_percent: 33.0,
            predictor: Some(Predictor::Gradient),
            learning_fraction: 0.37,
            max_properties: 1,
            patches: Override::On,
            tree_mode: TreeMode::WeightedOnly,
            delta_palette: DeltaPalette::ExactTop,
            lz77_careful_first_pass: true,
            bucket_matcher: true,
            precise_entropy: true,
            probe_lz77: 2,
            // Still rejected by the native adapter, not erased by generation.
            squeeze: vec![true, false],
            ..input()
        };
        for input in [input(), custom] {
            let original = input.clone();
            assert_eq!(
                initial_candidates(&input).unwrap().to_vec(),
                expected(&input, &INITIAL)
            );
            for branch in [TectonicBranch::LessPalette, TectonicBranch::MorePalette] {
                assert_eq!(
                    branch_candidates(&input, branch).unwrap(),
                    expected(&input, rows(branch))
                );
            }
            assert_eq!(input, original);
        }
    }

    #[test]
    fn every_branch_candidate_can_win_in_exact_source_call_order() {
        for branch in [TectonicBranch::LessPalette, TectonicBranch::MorePalette] {
            for index in 0..rows(branch).len() {
                let mut scores = vec![200; 2 + rows(branch).len()];
                scores[..2].copy_from_slice(match branch {
                    TectonicBranch::LessPalette => &[100, 101],
                    TectonicBranch::MorePalette => &[101, 100],
                });
                scores[2 + index] = 99;
                run_case(branch, scores, 2 + index);
            }
        }
    }

    #[test]
    fn source_ties_choose_less_palette_first_minimum_and_initial_incumbent() {
        // Initial tie selects LessPalette, even when no refinement improves it.
        let scores = vec![100; 26];
        run_case(TectonicBranch::LessPalette, scores, 0);
        for branch in [TectonicBranch::LessPalette, TectonicBranch::MorePalette] {
            let mut scores = vec![100; 2 + rows(branch).len()];
            let initial_best = match branch {
                TectonicBranch::LessPalette => 0,
                TectonicBranch::MorePalette => 1,
            };
            scores[1 - initial_best] = 101;
            // A branch minimum equal to the best initial score cannot replace it.
            run_case(branch, scores.clone(), initial_best);
            scores[5] = 99;
            scores[11] = 99;
            run_case(branch, scores, 5);
        }
    }

    #[test]
    fn budgets_count_final_encode_and_do_not_truncate_a_branch() {
        let mut oracle = MockOracle::new(TectonicBranch::LessPalette, vec![100; 27], 0);
        for budget in [
            limits(22),
            SearchLimits {
                max_full_pixels: 137,
                ..limits(27)
            },
            SearchLimits {
                max_output_bytes: 0,
                ..limits(27)
            },
        ] {
            assert!(matches!(
                encode(&image(), &input(), budget, &mut oracle),
                Err(Error::Limit(_))
            ));
            assert_eq!(oracle.calls, 0);
        }
        for budget in [
            limits(26),
            SearchLimits {
                max_full_pixels: 161,
                ..limits(27)
            },
        ] {
            oracle.calls = 0;
            assert!(matches!(
                encode(&image(), &input(), budget, &mut oracle),
                Err(Error::Limit(_))
            ));
            assert_eq!(
                oracle.calls, 2,
                "fail before starting incomplete second round"
            );
        }
        let mut scores = vec![100; 23];
        scores[0] = 101;
        let mut oracle = MockOracle::new(TectonicBranch::MorePalette, scores, 1);
        let output = encode(&image(), &input(), limits(23), &mut oracle).unwrap();
        assert_eq!(output.report.full_encodes, 23);
        assert_eq!(oracle.calls, 23);
    }

    #[test]
    fn output_limits_and_backend_errors_never_return_a_partial_winner() {
        for failure in [0, 1, 2, 25, 26] {
            let mut oracle = MockOracle::new(TectonicBranch::LessPalette, vec![100; 27], 0);
            oracle.fail_at = Some(failure);
            assert!(matches!(
                encode(&image(), &input(), limits(27), &mut oracle),
                Err(Error::Encoding(_))
            ));
            assert_eq!(oracle.calls, failure + 1);
            oracle.fail_at = None;
            oracle.calls = 0;
            oracle.sizes[failure] = 257;
            assert!(matches!(
                encode(&image(), &input(), limits(27), &mut oracle),
                Err(Error::Limit(_))
            ));
            assert_eq!(oracle.calls, failure + 1);
        }
    }

    #[test]
    fn empty_trial_and_final_outputs_are_not_complete_encodes() {
        for empty_at in [0, 1, 2, 25, 26] {
            let mut oracle = MockOracle::new(TectonicBranch::LessPalette, vec![100; 27], 0);
            oracle.sizes[empty_at] = 0;
            assert!(matches!(
                encode(&image(), &input(), limits(27), &mut oracle),
                Err(Error::Encoding(message)) if message.contains("empty complete output")
            ));
            assert_eq!(oracle.calls, empty_at + 1);
        }
    }

    struct BatchOracle {
        sizes: [Vec<usize>; 2],
        rounds: usize,
        final_calls: usize,
    }

    impl TectonicOracle for BatchOracle {
        fn measure_round(
            &mut self,
            _image: &Image,
            candidates: &[CandidateParams],
            budget: TrialBudget,
        ) -> Result<Vec<usize>> {
            assert_eq!(budget.beat_bytes, None);
            let snapshot: &[Snapshot] = if self.rounds == 0 { &INITIAL } else { &LESS };
            assert_eq!(candidates, expected(&input(), snapshot).as_slice());
            let scores = self.sizes[self.rounds].clone();
            self.rounds += 1;
            Ok(scores)
        }

        fn encode(
            &mut self,
            _image: &Image,
            params: &CandidateParams,
            _budget: TrialBudget,
        ) -> Result<EncodedOutput> {
            assert_eq!(self.rounds, 2);
            assert_eq!(params, &expected(&input(), &INITIAL)[0]);
            self.final_calls += 1;
            Ok(EncodedOutput {
                bytes: vec![42; 80],
                artifact: (),
            })
        }
    }

    #[test]
    fn batch_oracle_receives_two_ordered_rounds_and_one_final_encode() {
        let mut oracle = BatchOracle {
            sizes: [vec![100; 2], vec![100; 24]],
            rounds: 0,
            final_calls: 0,
        };
        let output = encode(&image(), &input(), limits(27), &mut oracle).unwrap();
        assert_eq!((oracle.rounds, oracle.final_calls), (2, 1));
        assert_eq!(output.report.trials.len(), 26);
        assert_eq!(output.bytes, vec![42; 80]);
    }

    #[test]
    fn malformed_or_oversized_batch_results_are_rejected() {
        for round in 0..2 {
            for invalid_size in [None, Some(0), Some(257)] {
                let mut oracle = BatchOracle {
                    sizes: [vec![100; 2], vec![100; 24]],
                    rounds: 0,
                    final_calls: 0,
                };
                if let Some(size) = invalid_size {
                    oracle.sizes[round][0] = size;
                } else {
                    oracle.sizes[round].pop();
                }
                assert!(encode(&image(), &input(), limits(27), &mut oracle).is_err());
                assert_eq!(oracle.rounds, round + 1);
                assert_eq!(oracle.final_calls, 0);
            }
        }
    }

    #[test]
    fn explicit_minus_one_backend_remains_distinct_and_tournament_requires_it() {
        let direct = input();
        assert_eq!(direct.speed_tier, -1);
        assert_eq!(direct.training_policy(3, 22).max_property_values, 32);
        let mut called = false;
        let mut oracle =
            |_: &Image, _: &CandidateParams, _: TrialBudget| -> Result<EncodedOutput> {
                called = true;
                unreachable!("invalid tournament must not invoke backend")
            };
        let not_eleven = CandidateParams {
            speed_tier: 0,
            ..direct.clone()
        };
        assert!(encode(&image(), &not_eleven, limits(27), &mut oracle).is_err());
        assert!(!called);
        let initial = initial_candidates(&direct).unwrap();
        assert_eq!(initial[0].training_policy(3, 22).max_property_values, 256);
        assert_eq!(direct.speed_tier, -1);
    }

    #[test]
    fn tree_mode_projects_to_root_and_deferred_atlas_without_disabling_wp_search() {
        let image = Image::new(2, 3, PixelLayout::Rgb, PixelData::U8(vec![7; 18])).unwrap();
        let base = EncodeOptions::default();
        for predictor in [Predictor::Variable, Predictor::Zero] {
            let params = CandidateParams {
                speed_tier: 0,
                max_properties: 4,
                predictor: Some(predictor),
                tree_mode: TreeMode::NoWeighted,
                ..Default::default()
            };
            let projection = project_native(&params, &image, &base).unwrap();
            assert!(!projection.training.properties.contains(&15));
            let reported = projection.training.properties.clone();
            let options = projection.exact_options().unwrap();
            let sampling = options.modular.sampling.as_ref().unwrap();
            assert_eq!(sampling.tree_mode, TreeMode::NoWeighted);
            assert!(!sampling.properties.contains(&15));
            assert_eq!(
                sampling.properties,
                reported.iter().map(|&p| p as usize).collect::<Vec<_>>()
            );
            assert_eq!(options.modular.predictor, predictor);
            assert_eq!(
                options.modular.weighted_candidates,
                (predictor == Predictor::Variable).then_some(5)
            );
            let recipe = source_patch_reference_policy(
                &params,
                &image,
                &base,
                NativeResourcePolicy::default(),
            )
            .unwrap();
            assert_eq!(recipe.sampling.tree_mode, TreeMode::NoWeighted);
            let atlas = recipe.resolve(&image).unwrap();
            assert_eq!(atlas.modular.predictor, Predictor::Gradient);
            let atlas_sampling = atlas.modular.sampling.as_ref().unwrap();
            assert_eq!(atlas_sampling.tree_mode, TreeMode::NoWeighted);
            assert!(!atlas_sampling.properties.contains(&15));
            assert_eq!(atlas_sampling.previous_channels, 2);
            assert_eq!(params.tree_mode, TreeMode::NoWeighted);
        }
    }
}
