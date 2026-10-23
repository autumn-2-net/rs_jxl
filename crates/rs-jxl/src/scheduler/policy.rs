// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Score-only planning helpers for `enc_lossless_search.cc`. Executable
//! large-frame racing/refinement and retained outputs are in `driver`.
//! Use `driver::run_pixel_search` from the beginning for a complete workflow.

use super::{
    params::{CandidateParams, Override, SourceBuffering},
    probe::ProbeResult,
    scan::LosslessScan,
};
use crate::{Error, Image, Result, modular::predictor::Predictor};

pub fn literal(input: &CandidateParams, effort: u8, fraction: f32) -> Result<CandidateParams> {
    let mut p = input.backend(effort)?;
    p.group_size_shift = Some(3);
    p.predictor = Some(Predictor::Zero);
    p.learning_fraction = fraction;
    p.palette_colors = 10000;
    p.patches = Override::Off;
    p.channel_colors_percent = 0.0;
    p.validate()?;
    Ok(p)
}
pub fn winner_neighborhood(winner: &CandidateParams, channels: usize) -> CandidateParams {
    let mut p = winner.clone();
    p.speed_tier = 1;
    if p.predictor == Some(Predictor::Zero) {
        p.learning_fraction = 0.03;
    } else {
        p.group_size_shift = Some(2);
        p.learning_fraction = 0.1;
        p.max_properties = channels.saturating_sub(1).min(2) as u8;
    }
    p
}

/// Source HasLosslessSearchOptions conditions represented by this pixel API.
/// Buffering <=0 is checked here. Container/progressive/JPEG eligibility
/// outside this pixel-only contract belongs to the caller.
/// Image sample eligibility is checked separately by validate_source_input.
pub fn default_pixel_options(params: &CandidateParams) -> bool {
    // In C++ delta is an environment switch, not an explicit CompressParams
    // override. It remains part of the Rust cache key for reproducibility.
    let expected = CandidateParams {
        speed_tier: params.speed_tier,
        buffering: params.buffering,
        delta_palette: params.delta_palette,
        ..Default::default()
    };
    (0..=9).contains(&params.speed_tier)
        && matches!(
            params.buffering,
            SourceBuffering::Auto | SourceBuffering::FullFrame
        )
        && *params == expected
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialFamily {
    Regular,
    Literal,
    LearnedLiteral,
    RefinedLiteral,
    Other,
}
#[derive(Clone, Debug)]
pub struct TrialRequest {
    pub stage: u8,
    pub params: CandidateParams,
    pub family: TrialFamily,
    pub baseline: bool,
}
#[derive(Clone, Debug)]
pub enum SearchAction {
    /// Score-only planning cannot execute a stateful screen/retained-output
    /// stage. Run driver::run_pixel_search from stage 1; do not skip this action.
    ExecuteLargeFrameDriver {
        stage: u8,
    },
    Trial(TrialRequest),
    /// Analyze using the actual winner before this stage, then call
    /// set_probe_result. This is performed once, at stage 8 in this policy.
    AnalyzeFamilies {
        stage: u8,
        ambiguous_families: bool,
    },
    /// Resolve immediately before execution, after earlier actions completed.
    WinnerNeighborhood {
        stage: u8,
        include_current_stage: bool,
    },
    PromoteProbe {
        stage: u8,
        index: usize,
    },
    DeepWinner {
        stage: u8,
    },
}
#[derive(Clone, Debug)]
pub struct CompletedTrial {
    pub request: TrialRequest,
    pub bytes: usize,
}

pub struct PixelSearch {
    input: CandidateParams,
    level: u8,
    pixels: u64,
    channels: usize,
    refine_winner: bool,
    literal_probe: bool,
    completed: Vec<CompletedTrial>,
    analysis: ProbeResult,
}
impl PixelSearch {
    pub fn new(image: &Image, input: CandidateParams, scan: Option<&LosslessScan>) -> Result<Self> {
        input.validate()?;
        super::scan::validate_source_input(image)?;
        if !default_pixel_options(&input) {
            return Err(Error::InvalidInput("automatic pixel policy requires default source options; use explicit candidate probes for overrides".into()));
        }
        let pixels = u64::from(image.width()) * u64::from(image.height());
        let scan = scan.filter(|s| s.available());
        if scan.is_some_and(|s| {
            s.width != image.width() as usize || s.height != image.height() as usize
        }) {
            return Err(Error::InvalidInput(
                "pixel policy scan dimensions mismatch".into(),
            ));
        }
        // TryLiteralProbe has a <=65536 fast return, but an already available
        // coverage scan takes precedence in the source.
        let literal_probe = if let Some(s) = scan {
            s.literal_candidate
        } else if pixels <= 65536 {
            true
        } else {
            super::scan::literal_evidence(image)
        };
        let refine_winner = pixels > 4096
            && image.metadata().extra_channels.len() <= 8
            && image
                .metadata()
                .extra_channels
                .iter()
                .all(|ec| ec.dim_shift == 0);
        Ok(Self {
            level: (10 - input.speed_tier) as u8,
            input,
            pixels,
            channels: image.channel_count(),
            refine_winner,
            literal_probe,
            completed: Vec::new(),
            analysis: ProbeResult::default(),
        })
    }
    pub fn level(&self) -> u8 {
        self.level
    }
    pub fn completed(&self) -> &[CompletedTrial] {
        &self.completed
    }
    pub fn best_before(&self, stage: u8) -> Option<&CompletedTrial> {
        self.completed
            .iter()
            .filter(|t| t.request.stage < stage)
            .min_by_key(|t| t.bytes)
    }
    pub fn winner(&self) -> Option<&CompletedTrial> {
        self.best_before(11)
    }
    pub fn initial_plan(&self) -> Result<Vec<TrialRequest>> {
        let mut result = vec![TrialRequest {
            stage: 1,
            params: self.input.backend(3)?,
            family: TrialFamily::Regular,
            baseline: true,
        }];
        if self.level < 2 {
            return Ok(result);
        }
        if self.literal_probe {
            result.push(TrialRequest {
                stage: 2,
                params: literal(&self.input, 6, 0.0)?,
                family: TrialFamily::Literal,
                baseline: true,
            });
        }
        if self.pixels <= 4096 {
            result.push(TrialRequest {
                stage: 2,
                params: self.input.backend(9)?,
                family: TrialFamily::Other,
                baseline: true,
            });
            for predictor in [
                Predictor::Zero,
                Predictor::Left,
                Predictor::Top,
                Predictor::Select,
                Predictor::Gradient,
                Predictor::Weighted,
            ] {
                let mut params = self.input.backend(9)?;
                params.predictor = Some(predictor);
                params.learning_fraction = 0.0;
                result.push(TrialRequest {
                    stage: 2,
                    params,
                    family: TrialFamily::Other,
                    baseline: true,
                });
            }
        }
        Ok(result)
    }
    /// Supply only COMPLETED full-frame bytes. The caller retains the winning
    /// buffer and performs any safe output-budget aborts. No crop score here.
    pub fn record_completed(&mut self, request: TrialRequest, bytes: usize) -> Result<bool> {
        if request.stage == 0 || request.stage > self.level || bytes == 0 {
            return Err(Error::InvalidInput(
                "invalid completed full-frame trial".into(),
            ));
        }
        request.params.validate()?;
        let improved = self.winner().is_none_or(|w| bytes < w.bytes);
        self.completed.push(CompletedTrial { request, bytes });
        Ok(improved)
    }
    pub fn set_probe_result(&mut self, mut result: ProbeResult) {
        let regular = self.family_size(TrialFamily::Regular, 8);
        let literal = self
            .family_size(TrialFamily::Literal, 8)
            .into_iter()
            .chain(self.family_size(TrialFamily::LearnedLiteral, 8))
            .min();
        let ambiguous = matches!((regular, literal), (Some(a), Some(b)) if a.max(b) as f64 <= a.min(b) as f64 * 1.35);
        if result.ranked.is_empty() && ambiguous && !result.exploratory.is_empty() {
            result.ranked = std::mem::take(&mut result.exploratory);
        }
        self.analysis = result;
    }
    fn family_size(&self, family: TrialFamily, stage: u8) -> Option<usize> {
        self.completed
            .iter()
            .filter(|t| t.request.stage < stage && t.request.family == family)
            .map(|t| t.bytes)
            .min()
    }
    fn competitive(&self, size: Option<usize>, stage: u8, tolerance: f64) -> bool {
        let baseline = self
            .completed
            .iter()
            .filter(|t| t.request.stage < stage && t.request.baseline)
            .min_by_key(|t| t.bytes);
        match (size, baseline) {
            (Some(n), Some(best)) => n as f64 <= best.bytes as f64 * tolerance,
            _ => false,
        }
    }
    /// Call once per stage, in ascending order, after recording the previous
    /// stage. Execute returned actions in order and resolve dynamic actions only
    /// when reached (stage-9 geometry must see that stage's completed winners).
    pub fn stage_actions(&self, stage: u8) -> Result<Vec<SearchAction>> {
        if !(3..=10).contains(&stage) || stage > self.level {
            return Ok(Vec::new());
        }
        if self.best_before(stage).is_none() {
            return Err(Error::InvalidInput(
                "record an initial completed baseline before advancing stages".into(),
            ));
        }
        if stage >= 4 && self.pixels >= 1 << 19 && self.refine_winner {
            return Ok(vec![SearchAction::ExecuteLargeFrameDriver { stage }]);
        }
        let regular = self.family_size(TrialFamily::Regular, stage);
        let literal_size = self.family_size(TrialFamily::Literal, stage);
        let learned = self.family_size(TrialFamily::LearnedLiteral, stage);
        let refined = self.family_size(TrialFamily::RefinedLiteral, stage);
        let mut result = Vec::new();
        if stage == 8 {
            let best_literal = literal_size.into_iter().chain(learned).min();
            let ambiguous = match (regular, best_literal) {
                (Some(a), Some(b)) => a.max(b) as f64 <= a.min(b) as f64 * 1.35,
                _ => false,
            };
            result.push(SearchAction::AnalyzeFamilies {
                stage,
                ambiguous_families: ambiguous,
            });
        }
        let backend = match stage {
            3 => Some(7),
            5 if !self.refine_winner => Some(8),
            7 => Some(9),
            10 => Some(10),
            _ => None,
        };
        if let Some(effort) = backend
            && (stage == 3 || self.competitive(regular, stage, 1.35))
        {
            result.push(SearchAction::Trial(TrialRequest {
                stage,
                params: self.input.backend(effort)?,
                family: TrialFamily::Regular,
                baseline: true,
            }));
        }
        if stage == 4 && self.competitive(literal_size, stage, 1.35) {
            result.push(SearchAction::Trial(TrialRequest {
                stage,
                params: literal(&self.input, 8, 0.0)?,
                family: TrialFamily::Literal,
                baseline: true,
            }));
        }
        if stage == 5 && self.refine_winner {
            result.push(SearchAction::WinnerNeighborhood {
                stage,
                include_current_stage: false,
            });
        }
        if stage == 6 && self.competitive(literal_size, stage, 1.35) {
            result.push(SearchAction::Trial(TrialRequest {
                stage,
                params: literal(&self.input, 9, 0.01)?,
                family: TrialFamily::LearnedLiteral,
                baseline: true,
            }));
        }
        if stage == 8 && self.competitive(learned, stage, 1.02) {
            result.push(SearchAction::Trial(TrialRequest {
                stage,
                params: literal(&self.input, 9, 0.1)?,
                family: TrialFamily::RefinedLiteral,
                baseline: true,
            }));
        }
        if stage == 9
            && matches!((refined, learned), (Some(r), Some(l)) if (r as f64) < l as f64 * 0.99)
        {
            result.push(SearchAction::Trial(TrialRequest {
                stage,
                params: literal(&self.input, 9, 1.0)?,
                family: TrialFamily::Other,
                baseline: true,
            }));
        }
        if stage >= 8 {
            result.push(SearchAction::PromoteProbe {
                stage,
                index: usize::from(stage - 8),
            });
        }
        if stage == 9 && self.refine_winner {
            result.push(SearchAction::WinnerNeighborhood {
                stage,
                include_current_stage: true,
            });
        }
        if stage == 10 {
            result.push(SearchAction::DeepWinner { stage });
        }
        Ok(result)
    }
    pub fn resolve(&self, action: &SearchAction) -> Option<TrialRequest> {
        let (stage, params) = match action {
            SearchAction::ExecuteLargeFrameDriver { .. } => return None,
            SearchAction::Trial(request) => return Some(request.clone()),
            SearchAction::AnalyzeFamilies { .. } => return None,
            SearchAction::WinnerNeighborhood {
                stage,
                include_current_stage,
            } => {
                let winner = self.best_before(*stage + u8::from(*include_current_stage))?;
                (
                    *stage,
                    winner_neighborhood(&winner.request.params, self.channels),
                )
            }
            SearchAction::PromoteProbe { stage, index } => {
                (*stage, self.analysis.ranked.get(*index)?.params.clone())
            }
            SearchAction::DeepWinner { stage } => {
                if self.analysis.samples == 0 {
                    return None;
                }
                let mut p = self.best_before(*stage)?.request.params.clone();
                p.speed_tier = 0;
                p.learning_fraction = p.learning_fraction.max(0.5);
                p.max_properties = self.channels.saturating_sub(1).min(4) as u8;
                p.group_size_shift = Some(3);
                p.lz77_careful_first_pass = true;
                (*stage, p)
            }
        };
        Some(TrialRequest {
            stage,
            params,
            family: TrialFamily::Other,
            baseline: false,
        })
    }
}
