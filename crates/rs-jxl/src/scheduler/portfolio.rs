//! Opt-in native experiment portfolio. Source statistics prioritize work; actual
//! crop encodes screen candidates, and complete outputs decide the winner.
//!
//! This is not the default e9 policy. Crop ranks are heuristic, not size bounds.
//! A complete ordinary preset is retained, including its cost in the report.
use std::{collections::BTreeSet, time::Instant};

pub use super::portfolio_catalog::{ALL, Candidate, Family};
pub use super::portfolio_features::Features;
use super::{
    portfolio_features as features,
    regions::{SampleSpec, copy_sample},
    scan::{self, ScanRegion},
};
use crate::{EncodeOptions, Error, Image, Result};

#[derive(Clone, Debug)]
pub struct Options {
    pub effort: u8,
    pub threads: usize,
    pub max_pixels: u64,
    pub max_output_bytes: usize,
    /// An explicit experiment work budget, not a codec/tree/format limit.
    pub max_probe_encodes: usize,
    /// Number of complete alternatives in addition to the ordinary baseline.
    pub max_finalists: usize,
    pub probe_side: usize,
    /// Absolute relative crop-size tolerance for the uncertainty band.
    pub uncertainty: f64,
    /// Native candidates to consider. No unbounded Cartesian product is formed.
    pub candidates: Vec<Candidate>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            effort: 9,
            threads: 1,
            max_pixels: u64::MAX,
            max_output_bytes: usize::MAX,
            max_probe_encodes: 48,
            max_finalists: 2,
            probe_side: 128,
            uncertainty: 0.01,
            candidates: ALL.to_vec(),
        }
    }
}

impl Options {
    fn validate(&self, image: &Image) -> Result<()> {
        scan::validate_source_input(image)?;
        if !(8..=10).contains(&self.effort)
            || !(1..=256).contains(&self.threads)
            || !(32..=512).contains(&self.probe_side)
            || !self.uncertainty.is_finite()
            || !(0.0..=0.25).contains(&self.uncertainty)
        {
            return Err(Error::InvalidInput("portfolio requires effort 8..10, threads 1..256, probe_side 32..512 and uncertainty 0..0.25".into()));
        }
        if u64::from(image.width()) * u64::from(image.height()) > self.max_pixels {
            return Err(Error::Limit("pixel count"));
        }
        if self.max_output_bytes == 0 {
            return Err(Error::Limit("encoded output bytes"));
        }
        Ok(())
    }
    fn base(&self) -> EncodeOptions {
        EncodeOptions {
            threads: self.threads,
            max_pixels: self.max_pixels,
            max_output_bytes: self.max_output_bytes,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct Trial {
    pub candidate: Candidate,
    /// Audit-friendly state; a deferred candidate was NOT proven unprofitable.
    pub decision: &'static str,
    pub reason: String,
    pub priority: f64,
    pub probe_bytes: Vec<Option<usize>>,
    pub probe_seconds: f64,
    pub full_bytes: Option<usize>,
    pub full_seconds: Option<f64>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub features: Features,
    pub regions: Vec<ScanRegion>,
    pub candidates: Vec<Trial>,
    pub winner: Candidate,
    /// Full routing wall time, including ordinary baseline and discarded work.
    pub seconds: f64,
    /// Coverage scan, feature extraction and all crop encodes, not final encodes.
    pub scout_seconds: f64,
    pub probe_encodes: usize,
    pub probe_pixels: u64,
    pub full_encodes: usize,
}

#[derive(Debug)]
pub struct Output {
    pub bytes: Vec<u8>,
    pub report: Report,
}

fn pool(options: &Options) -> Result<Option<rayon::ThreadPool>> {
    if options.threads == 1 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()
        .map(Some)
        .map_err(|e| Error::InvalidInput(format!("portfolio thread pool: {e}")))
}

fn native(
    image: &Image,
    options: &EncodeOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<u8>> {
    match pool {
        Some(pool) => crate::encode::encode_with_pool(image, options, pool),
        None => crate::encode::encode_serial(image, options),
    }
}

// These signals order scarce probe slots only. In particular, source RGB
// dependency is not a substitute for dependency after the candidate's RCT.
fn priority(candidate: Candidate, f: &Features) -> f64 {
    match candidate.family() {
        Family::Baseline => 100.0,
        Family::Offset => 2.0 + f.residual_bias,
        Family::CrossChannel => 2.0 + f.cross_channel_gain * 4.0,
        Family::Shallow => 1.0 + f.gradient_zero_fraction,
        Family::Learning => 1.5 + f.residual_bias,
        Family::Entropy => 0.5,
        Family::PaletteOrder => 0.5 + (1.0 - f.unique_fraction) * 3.0 + f.flat_fraction,
        Family::Delta => {
            // Unlike a family-wide label, this can actually prioritize Top over
            // Left/Gradient on vertically repetitive content before encoding.
            let direction = match candidate {
                Candidate::DeltaLeft | Candidate::ZeroLeft => 1,
                Candidate::DeltaTop | Candidate::ZeroTop => 2,
                _ => 0,
            };
            let zero = matches!(
                candidate,
                Candidate::ZeroGradient | Candidate::ZeroLeft | Candidate::ZeroTop
            );
            0.5 + (1.0 - f.residual_unique_fraction[direction]) * 3.0
                + (f.unique_fraction - f.residual_unique_fraction[direction]).max(0.0)
                + if zero {
                    f.residual_zero_fraction[direction]
                } else {
                    0.0
                }
        }
        Family::Grouping => 0.5 + f.cross_channel_gain + (1.0 - f.unique_fraction),
        Family::Source => 0.5 + f.flat_fraction * 3.0 + f.repeated_tile_fraction * 4.0,
        Family::Tile => 0.5 + f.repeated_tile_fraction * 5.0,
    }
}

fn plan_in_pool(
    image: &Image,
    options: &Options,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Report> {
    options.validate(image)?;
    let start = Instant::now();
    let coverage = match pool {
        Some(pool) => scan::scan_lossless_input_with_pool(image, Default::default(), pool),
        None => scan::scan_lossless_input(image, Default::default()),
    };
    let regions = features::regions(image, &coverage, options.probe_side);
    let features = features::measure(image, &regions);
    let mut candidates = Vec::new();
    for candidate in std::iter::once(Candidate::Baseline).chain(options.candidates.iter().copied())
    {
        if candidates.iter().any(|t: &Trial| t.candidate == candidate) {
            continue;
        }
        let error = candidate
            .build(image, &options.base(), options.effort)
            .err()
            .map(|e| e.to_string());
        candidates.push(Trial {
            candidate,
            decision: if error.is_some() {
                "ineligible"
            } else {
                "queued"
            },
            reason: error
                .clone()
                .unwrap_or_else(|| "statistics prioritize probes; not a size prediction".into()),
            priority: priority(candidate, &features)
                + if candidate == Candidate::SourceParallel && options.threads > 1 {
                    0.05
                } else {
                    0.0
                },
            probe_bytes: vec![None; regions.len()],
            probe_seconds: 0.0,
            full_bytes: None,
            full_seconds: None,
            error,
        });
    }
    Ok(Report {
        features,
        regions,
        candidates,
        winner: Candidate::Baseline,
        seconds: 0.0,
        scout_seconds: start.elapsed().as_secs_f64(),
        probe_encodes: 0,
        probe_pixels: 0,
        full_encodes: 0,
    })
}

/// Inspect priorities and structural eligibility without executing any encoder.
/// This scans source pixels; it does not build a tree or promise a winning route.
pub fn plan(image: &Image, options: &Options) -> Result<Report> {
    options.validate(image)?;
    let pool = pool(options)?;
    plan_in_pool(image, options, pool.as_ref())
}

fn probe(
    image: &Image,
    region: usize,
    trial: usize,
    options: &Options,
    report: &mut Report,
    pool: Option<&rayon::ThreadPool>,
) {
    let start = Instant::now();
    let result = report.candidates[trial]
        .candidate
        .build(image, &options.base(), options.effort)
        .and_then(|configured| native(image, &configured, pool));
    report.probe_encodes += 1;
    report.probe_pixels += u64::from(image.width()) * u64::from(image.height());
    let record = &mut report.candidates[trial];
    record.probe_seconds += start.elapsed().as_secs_f64();
    match result {
        Ok(bytes) => record.probe_bytes[region] = Some(bytes.len()),
        Err(error) => {
            record.decision = "probe-error";
            record.reason = "crop failure is recorded, not interpreted as a size score".into();
            record.error = Some(error.to_string());
        }
    }
}

fn ratio(trial: &Trial, baseline: &Trial) -> Option<f64> {
    let mut after = 0u64;
    let mut before = 0u64;
    for (a, b) in trial.probe_bytes.iter().zip(&baseline.probe_bytes) {
        if let (Some(a), Some(b)) = (a, b) {
            after += *a as u64;
            before += *b as u64;
        }
    }
    (before != 0).then(|| after as f64 / before as f64)
}

fn ordered(report: &Report) -> Vec<usize> {
    let mut order: Vec<_> = (1..report.candidates.len())
        .filter(|&i| report.candidates[i].decision == "queued")
        .collect();
    order.sort_by(|&a, &b| {
        report.candidates[b]
            .priority
            .total_cmp(&report.candidates[a].priority)
            .then(a.cmp(&b))
    });
    // First cover families, then revisit their alternatives. Six delta modes
    // cannot consume all early slots before cross-channel/offset/patch probes.
    let mut result = Vec::new();
    while !order.is_empty() {
        let mut seen = BTreeSet::new();
        order.retain(|&i| {
            if seen.insert(report.candidates[i].candidate.family()) {
                result.push(i);
                false
            } else {
                true
            }
        });
    }
    result
}

fn finalists(report: &mut Report, options: &Options) -> Vec<usize> {
    let mut ranked: Vec<_> = (1..report.candidates.len())
        .filter(|&i| {
            let t = &report.candidates[i];
            t.error.is_none() && t.probe_bytes.iter().all(Option::is_some)
        })
        .collect();
    let confident = |i: usize| {
        let t = &report.candidates[i];
        let wins = t
            .probe_bytes
            .iter()
            .zip(&report.candidates[0].probe_bytes)
            .filter(|(a, b)| a < b)
            .count();
        ratio(t, &report.candidates[0]).unwrap() < 1.0 && wins * 2 >= report.regions.len()
    };
    ranked.sort_by(|&a, &b| {
        confident(b)
            .cmp(&confident(a))
            .then_with(|| {
                ratio(&report.candidates[a], &report.candidates[0])
                    .unwrap()
                    .total_cmp(&ratio(&report.candidates[b], &report.candidates[0]).unwrap())
            })
            .then_with(|| {
                report.candidates[b]
                    .priority
                    .total_cmp(&report.candidates[a].priority)
            })
            .then(a.cmp(&b))
    });
    let mut selected = Vec::new();
    let mut families = BTreeSet::new();
    let mut uncertain_used = false;
    for i in ranked {
        let t = &report.candidates[i];
        let r = ratio(t, &report.candidates[0]).unwrap();
        let wins = t
            .probe_bytes
            .iter()
            .zip(&report.candidates[0].probe_bytes)
            .filter(|(a, b)| a < b)
            .count();
        let confident = r < 1.0 && wins * 2 >= report.regions.len();
        let uncertain = r <= 1.0 + options.uncertainty;
        let patch_witness = matches!(t.candidate.family(), Family::Source | Family::Tile)
            && (report.features.repeated_tile_fraction > 0.0
                || report.features.flat_fraction > 0.5);
        if selected.len() < options.max_finalists
            && (confident || (!uncertain_used && (uncertain || patch_witness)))
            && families.insert(t.candidate.family())
        {
            uncertain_used |= !confident;
            report.candidates[i].decision = "finalist";
            report.candidates[i].reason = if confident {
                "confirmed crop benefit"
            } else if patch_witness {
                "bounded full-image patch witness; crops can miss distant repetitions"
            } else {
                "bounded uncertainty witness; crop score is not conclusive"
            }
            .into();
            selected.push(i);
        } else {
            report.candidates[i].decision = "probe-rejected";
            report.candidates[i].reason =
                "crop ranking or family/work budget; not proof of whole-image loss".into();
        }
    }
    selected
}

/// Encode one image with a bounded, opt-in native portfolio. The returned bytes
/// are never larger than this invocation's complete ordinary baseline. This
/// protection costs a baseline encode; it does NOT guarantee optimal routing.
pub fn encode(image: &Image, options: &Options) -> Result<Output> {
    options.validate(image)?;
    let start = Instant::now();
    let pool = pool(options)?;
    let mut report = plan_in_pool(image, options, pool.as_ref())?;
    let configured = Candidate::Baseline.build(image, &options.base(), options.effort)?;
    let base_start = Instant::now();
    let mut best = native(image, &configured, pool.as_ref())?;
    report.full_encodes = 1;
    report.candidates[0].full_bytes = Some(best.len());
    report.candidates[0].full_seconds = Some(base_start.elapsed().as_secs_f64());
    report.candidates[0].decision = "winner";
    report.candidates[0].reason = "complete ordinary preset retained as fallback".into();
    let scout_start = Instant::now();
    let order = ordered(&report);
    let selected = if options.max_finalists == 0 || order.is_empty() {
        Vec::new()
    } else if image.width() as usize <= options.probe_side
        && image.height() as usize <= options.probe_side
    {
        let selected: Vec<_> = order.iter().take(options.max_finalists).copied().collect();
        for &i in &selected {
            report.candidates[i].decision = "finalist";
            report.candidates[i].reason =
                "small image: bounded direct trials instead of duplicate full-size crops".into();
        }
        selected
    } else if options.max_probe_encodes < report.regions.len() * 2
        || image
            .metadata()
            .extra_channels
            .iter()
            .any(|c| c.dim_shift != 0)
    {
        Vec::new()
    } else {
        let samples = report
            .regions
            .iter()
            .map(|&r| {
                copy_sample(
                    image,
                    &SampleSpec { regions: vec![r] },
                    options.probe_side * options.probe_side,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        for (r, sample) in samples.iter().enumerate() {
            probe(sample, r, 0, options, &mut report, pool.as_ref());
        }
        if report.candidates[0].error.is_none() {
            // Reserve half of remaining work for confirmation outside the first crop.
            let first_budget = (options.max_probe_encodes - report.probe_encodes) / 2;
            for &i in order.iter().take(first_budget) {
                probe(&samples[0], 0, i, options, &mut report, pool.as_ref());
            }
            let mut survivors: Vec<_> = order
                .iter()
                .copied()
                .filter(|&i| {
                    report.candidates[i].error.is_none()
                        && report.candidates[i].probe_bytes[0].is_some()
                })
                .collect();
            survivors.sort_by(|&a, &b| {
                ratio(&report.candidates[a], &report.candidates[0])
                    .unwrap()
                    .total_cmp(&ratio(&report.candidates[b], &report.candidates[0]).unwrap())
                    .then_with(|| {
                        report.candidates[b]
                            .priority
                            .total_cmp(&report.candidates[a].priority)
                    })
                    .then(a.cmp(&b))
            });
            for i in survivors {
                if report.probe_encodes + samples.len() - 1 > options.max_probe_encodes {
                    break;
                }
                for (r, sample) in samples.iter().enumerate().skip(1) {
                    probe(sample, r, i, options, &mut report, pool.as_ref());
                    if report.candidates[i].error.is_some() {
                        break;
                    }
                }
            }
            finalists(&mut report, options)
        } else {
            Vec::new()
        }
    };
    report.scout_seconds += scout_start.elapsed().as_secs_f64();
    for i in selected {
        let full_start = Instant::now();
        let result = report.candidates[i]
            .candidate
            .build(image, &options.base(), options.effort)
            .and_then(|configured| native(image, &configured, pool.as_ref()));
        report.full_encodes += 1;
        report.candidates[i].full_seconds = Some(full_start.elapsed().as_secs_f64());
        match result {
            Ok(bytes) => {
                report.candidates[i].full_bytes = Some(bytes.len());
                if bytes.len() < best.len() {
                    for t in &mut report.candidates {
                        if t.decision == "winner" {
                            t.decision = "full-rejected";
                        }
                    }
                    report.candidates[i].decision = "winner";
                    report.winner = report.candidates[i].candidate;
                    best = bytes;
                } else {
                    report.candidates[i].decision = "full-rejected";
                }
            }
            Err(error) => {
                report.candidates[i].decision = "full-error";
                report.candidates[i].error = Some(error.to_string());
            }
        }
    }
    for t in &mut report.candidates {
        if t.candidate == report.winner && t.error.is_none() {
            t.decision = "winner";
        }
        if t.decision == "queued" {
            t.decision = "budget-deferred";
            t.reason = "not enough experimental work budget for full confirmation".into();
        }
    }
    report.seconds = start.elapsed().as_secs_f64();
    Ok(Output {
        bytes: best,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout};

    fn scored(candidate: Candidate, sizes: &[Option<usize>]) -> Trial {
        Trial {
            candidate,
            decision: "queued",
            reason: String::new(),
            priority: 1.0,
            probe_bytes: sizes.to_vec(),
            probe_seconds: 0.0,
            full_bytes: None,
            full_seconds: None,
            error: None,
        }
    }

    #[test]
    fn incomplete_scores_never_beat_confirmed_scores() {
        let mut report = Report {
            features: Features::default(),
            regions: vec![
                ScanRegion {
                    x: 0,
                    y: 0,
                    width: 32,
                    height: 32
                };
                3
            ],
            candidates: vec![
                scored(Candidate::Baseline, &[Some(100), Some(100), Some(100)]),
                scored(Candidate::OffsetDistribution, &[Some(1), None, None]),
                scored(Candidate::Prev1, &[Some(95), Some(95), Some(95)]),
                scored(Candidate::Prev2, &[Some(96), Some(96), Some(96)]),
                scored(Candidate::PaletteHot128, &[Some(102), Some(102), Some(102)]),
            ],
            winner: Candidate::Baseline,
            seconds: 0.0,
            scout_seconds: 0.0,
            probe_encodes: 0,
            probe_pixels: 0,
            full_encodes: 0,
        };
        assert_eq!(finalists(&mut report, &Options::default()), vec![2]);
        assert_eq!(report.candidates[1].decision, "queued");
        assert_eq!(report.candidates[4].decision, "probe-rejected");
    }

    #[test]
    fn families_get_first_round_before_repeated_variants() {
        let report = Report {
            features: Features::default(),
            regions: vec![],
            candidates: vec![
                scored(Candidate::Baseline, &[]),
                scored(Candidate::Prev1, &[]),
                scored(Candidate::Prev2, &[]),
                scored(Candidate::OffsetDistribution, &[]),
            ],
            winner: Candidate::Baseline,
            seconds: 0.0,
            scout_seconds: 0.0,
            probe_encodes: 0,
            probe_pixels: 0,
            full_encodes: 0,
        };
        assert_eq!(ordered(&report), vec![1, 3, 2]);
    }

    #[test]
    fn crop_ties_get_one_bounded_witness_and_parallel_source_has_preference() {
        let mut report = Report {
            features: Features::default(),
            regions: vec![
                ScanRegion {
                    x: 0,
                    y: 0,
                    width: 32,
                    height: 32
                };
                3
            ],
            candidates: vec![
                scored(Candidate::Baseline, &[Some(100); 3]),
                scored(Candidate::Source, &[Some(100); 3]),
                scored(Candidate::SourceParallel, &[Some(100); 3]),
                scored(Candidate::OffsetDistribution, &[Some(100); 3]),
            ],
            winner: Candidate::Baseline,
            seconds: 0.0,
            scout_seconds: 0.0,
            probe_encodes: 0,
            probe_pixels: 0,
            full_encodes: 0,
        };
        report.candidates[2].priority = 1.05;
        assert_eq!(finalists(&mut report, &Options::default()), vec![2]);
    }
    #[test]
    fn explicit_zero_work_budget_keeps_baseline() {
        let image = Image::new(
            16,
            16,
            PixelLayout::Rgb,
            PixelData::U8(vec![9; 16 * 16 * 3]),
        )
        .unwrap();
        let result = encode(
            &image,
            &Options {
                max_finalists: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.report.full_encodes, 1);
        assert_eq!(result.report.probe_encodes, 0);
        assert_eq!(result.report.winner, Candidate::Baseline);
        assert_eq!(
            result.bytes.len(),
            result.report.candidates[0].full_bytes.unwrap()
        );
    }
    #[test]
    fn plan_deduplicates_options_and_preserves_depth() {
        let image =
            Image::new(9, 7, PixelLayout::Gray, PixelData::U16(vec![65535; 9 * 7])).unwrap();
        let report = plan(
            &image,
            &Options {
                candidates: vec![
                    Candidate::Baseline,
                    Candidate::OffsetDistribution,
                    Candidate::OffsetDistribution,
                ],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(report.candidates.len(), 2);
        assert_eq!(report.probe_encodes, 0);
        assert_eq!(report.full_encodes, 0);
    }
    #[test]
    fn invalid_limits_fail_before_encoding() {
        let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![0])).unwrap();
        for config in [
            Options {
                threads: 0,
                ..Default::default()
            },
            Options {
                max_pixels: 0,
                ..Default::default()
            },
            Options {
                uncertainty: f64::NAN,
                ..Default::default()
            },
        ] {
            assert!(encode(&image, &config).is_err());
        }
    }

    #[test]
    fn delta_direction_is_prioritized_from_its_own_residuals() {
        let features = Features {
            unique_fraction: 0.8,
            residual_unique_fraction: [0.9, 0.8, 0.1],
            residual_zero_fraction: [0.0, 0.1, 0.8],
            ..Default::default()
        };
        assert!(
            priority(Candidate::DeltaTop, &features) > priority(Candidate::DeltaLeft, &features)
        );
        assert!(
            priority(Candidate::DeltaTop, &features)
                > priority(Candidate::DeltaGradient, &features)
        );
        assert!(priority(Candidate::ZeroTop, &features) > priority(Candidate::DeltaTop, &features));
    }
}
