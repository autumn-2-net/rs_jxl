//! Disabled-by-default downstream admission pilot, not a production cost model.
//!
//! Four-leaf native MA pilots use Gradient/Left/Top/Zero and native properties
//! (excluding WP and history-dependent property 8). Up to 256 training sites and
//! four contiguous 256-token windows per image plane are visited. The new table
//! is scored in full, up to 8192 samples. Native fixed-420 prefix models count
//! actual model headers, HybridUint payloads and tree bits. Window-local native
//! Greedy LZ is retained only if its projected data plus unscaled header wins.
//!
//! This is NOT final MA/WP search, shared-channel clustering, full-stream LZ,
//! subsequent RCT/transform selection, or a complete-file size prediction.
//! Independent per-plane trees/models can bias rankings; sampled populations
//! may miss rare tokens and train/evaluation sites can overlap. No savings are
//! claimed without fixed-state complete-output validation. Budget exhaustion
//! leaves the old decision intact. No pixel values are changed by this module.

use super::{PaletteMode, PaletteStage, counterfactual::Target};
use crate::bit_writer::BitWriter;
use crate::entropy::{
    HybridUintConfig,
    options::{ClusterStrategy, EntropyOptions, UintConfigStrategy},
    stream::{Coding, Model, Token},
};
use crate::modular::{
    group::{self, Channel, Transform},
    ma_train::{self, Sample, TrainOptions},
    predictor::{Neighbors, Predictor, predict_one},
    tree::{Properties, pack_signed},
    weighted,
};
use crate::tools::lz77::{self, Command, FilterPolicy, Literal, Lz77Mode, Lz77Options};
use crate::{Error, Result};
use std::ops::Range;
use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

const TRAIN_SITES: usize = 256;
const WINDOW: usize = 256;
const WINDOWS: usize = 4;
const TABLE_SAMPLES: usize = 8192;
const MAX_CHANNELS: usize = 8;
const SCRATCH_BYTES: usize = 1 << 20;
const PREDICTORS: [Predictor; 4] = [
    Predictor::Gradient,
    Predictor::Left,
    Predictor::Top,
    Predictor::Zero,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// Record old/new decisions at identical state; do not change admission.
    Shadow,
    /// Only veto old acceptances. Explicit force/zero/direct and replay win.
    Veto,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaneScore {
    pub population: usize,
    pub sampled: usize,
    pub training_sites: usize,
    pub leaves: usize,
    pub tree_bits: usize,
    pub literal_header_bits: usize,
    pub literal_data_bits: usize,
    pub lz_header_bits: usize,
    pub lz_data_bits: usize,
    pub lz_matches: usize,
    pub lz_matched_samples: usize,
    pub used_lz: bool,
    pub total_bits: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    /// Candidate: table first, index second. Skip: selected original planes.
    pub planes: Vec<PlaneScore>,
    /// Actual complete prefix header, not an arbitrary table penalty.
    pub transform_header_bits: usize,
    pub total_bits: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub target: Target,
    pub mode: PaletteMode,
    pub skip: Option<Score>,
    pub candidate: Option<Score>,
    pub old_would_accept: Option<bool>,
    pub proxy_would_accept: Option<bool>,
    pub vetoed: bool,
    pub forced: bool,
    /// Kernel decline, explicit Skip, budget limit, or pilot error; not a vote.
    pub unavailable: Option<String>,
}

/// Fresh per encode, invocation-owned; read after workers have joined. A target
/// restricts both scoring and vetoing, preserving the input/prefix at that site.
#[derive(Debug)]
pub struct Experiment {
    pub mode: Mode,
    target: Option<Target>,
    records: Mutex<Vec<Record>>,
    scoring_nanos: AtomicU64,
}

impl PartialEq for Experiment {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Experiment {
    pub fn new(mode: Mode, target: Option<Target>) -> Self {
        Self {
            mode,
            target,
            records: Mutex::new(Vec::new()),
            scoring_nanos: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Record>> {
        self.records.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn events(&self) -> Vec<Record> {
        self.lock().clone()
    }

    /// Summed elapsed time inside scoring comparisons, including unavailable
    /// results. Concurrent groups may overlap: this is work time, not encode
    /// wall time. Excludes record construction/locking and kernel preparation.
    pub fn scoring_seconds(&self) -> f64 {
        self.scoring_nanos.load(Ordering::Relaxed) as f64 / 1e9
    }
}

pub(crate) struct Cursor<'a> {
    experiment: Option<&'a Experiment>,
    prefix: &'a [Transform],
    stage: PaletteStage,
    stream_id: usize,
    ordinal: usize,
    allow_veto: bool,
}

pub(crate) struct Visit<'a> {
    experiment: &'a Experiment,
    prefix: &'a [Transform],
    target: Target,
    allow_veto: bool,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(
        experiment: Option<&'a Experiment>,
        prefix: &'a [Transform],
        stage: PaletteStage,
        stream_id: usize,
        allow_veto: bool,
    ) -> Self {
        Self {
            experiment,
            prefix,
            stage,
            stream_id,
            ordinal: 0,
            allow_veto,
        }
    }

    pub(crate) fn next(&mut self) -> Option<Visit<'a>> {
        let target = Target {
            stage: self.stage,
            stream_id: self.stream_id,
            ordinal: self.ordinal,
        };
        self.ordinal += 1;
        self.experiment
            .filter(|e| e.target.is_none_or(|selected| selected == target))
            .map(|experiment| Visit {
                experiment,
                prefix: self.prefix,
                target,
                allow_veto: self.allow_veto,
            })
    }
}

impl Visit<'_> {
    pub(crate) fn unavailable(&self, mode: PaletteMode, forced: bool, reason: &str) {
        self.experiment.lock().push(Record {
            target: self.target,
            mode,
            skip: None,
            candidate: None,
            old_would_accept: None,
            proxy_would_accept: None,
            vetoed: false,
            forced,
            unavailable: Some(reason.into()),
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compare(
        &self,
        channels: &[Channel],
        candidate: &[Channel],
        transforms: &[Transform],
        transform: &Transform,
        mode: PaletteMode,
        old_would_accept: bool,
        forced: bool,
    ) -> bool {
        let start = Instant::now();
        let result = compare(channels, candidate, self.prefix, transforms, transform);
        let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let _ = self.experiment.scoring_nanos.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |total| Some(total.saturating_add(nanos)),
        );
        let mut record = Record {
            target: self.target,
            mode,
            skip: None,
            candidate: None,
            old_would_accept: Some(old_would_accept),
            proxy_would_accept: None,
            vetoed: false,
            forced,
            unavailable: None,
        };
        match result {
            Ok((skip, candidate)) => {
                let accept = candidate.total_bits <= skip.total_bits;
                record.proxy_would_accept = Some(accept);
                record.vetoed = self.experiment.mode == Mode::Veto
                    && self.allow_veto
                    && !forced
                    && old_would_accept
                    && !accept;
                record.skip = Some(skip);
                record.candidate = Some(candidate);
            }
            Err(error) => record.unavailable = Some(error.to_string()),
        }
        let vetoed = record.vetoed;
        self.experiment.lock().push(record);
        vetoed
    }
}

fn entropy_options() -> EntropyOptions {
    EntropyOptions {
        uint_config: UintConfigStrategy::Fixed(HybridUintConfig::default()),
        clustering: ClusterStrategy::Fast,
        max_histograms: 4,
        ..EntropyOptions::default()
    }
}

fn header_bits(transforms: &[Transform]) -> Result<usize> {
    let mut writer = BitWriter::new(SCRATCH_BYTES);
    group::write_header(&mut writer, transforms, &weighted::Header::default(), false)?;
    Ok(writer.bits_written())
}

fn compare(
    original: &[Channel],
    candidate: &[Channel],
    prefix: &[Transform],
    transforms: &[Transform],
    transform: &Transform,
) -> Result<(Score, Score)> {
    let Transform::Palette {
        begin, channels, ..
    } = *transform
    else {
        return Err(Error::InvalidInput("admission requires Palette".into()));
    };
    if channels > MAX_CHANNELS || candidate[0].samples.len() > TABLE_SAMPLES {
        return Err(Error::Limit("palette admission pilot channel/table budget"));
    }
    let mut script: Vec<_> = prefix.iter().chain(transforms).cloned().collect();
    let before_header = header_bits(&script)?;
    script.push(transform.clone());
    let after_header = header_bits(&script)?;
    // Unchanged planes, including unselected alpha/extra channels and old meta,
    // cancel. No masking: selected RGBA and hidden RGB are scored verbatim.
    let before = original[begin..begin + channels]
        .iter()
        .map(|c| plane_score(c, false))
        .collect::<Result<Vec<_>>>()?;
    let after = vec![
        plane_score(&candidate[0], true)?,
        plane_score(&candidate[begin + 1], false)?,
    ];
    let score = |planes: Vec<PlaneScore>, transform_header_bits: usize| Score {
        total_bits: planes.iter().map(|p| p.total_bits).sum::<f64>() + transform_header_bits as f64,
        planes,
        transform_header_bits,
    };
    Ok((score(before, before_header), score(after, after_header)))
}

fn windows(population: usize, full: bool) -> Vec<Range<usize>> {
    if full || population <= WINDOW * WINDOWS {
        (0..population)
            .step_by(WINDOW)
            .map(|start| start..(start + WINDOW).min(population))
            .collect()
    } else {
        (0..WINDOWS)
            .map(|i| {
                let low = population / WINDOWS * i;
                let high = if i + 1 == WINDOWS {
                    population
                } else {
                    population / WINDOWS * (i + 1)
                };
                let start = low + (high - low - WINDOW) / 2;
                start..start + WINDOW
            })
            .collect()
    }
}

fn site(channel: &Channel, position: usize) -> Result<(Neighbors, [i32; 16])> {
    let (x, y) = (position % channel.width, position / channel.width);
    let n = Neighbors::from_plane(&channel.samples, channel.width, channel.width, x, y)
        .map_err(|e| Error::Encoding(e.to_string()))?;
    // Property 8 depends on raster history and is deliberately NOT trained.
    let properties = Properties::default().calculate(n, x, y, 0, 0, 0);
    Ok((n, properties))
}

fn residual(value: i32, neighbors: Neighbors, predictor: Predictor) -> Result<i32> {
    let prediction =
        predict_one(predictor, neighbors, None).map_err(|e| Error::Encoding(e.to_string()))?;
    // Match native tokenization: wide difference, then modulo-2^32 narrowing.
    Ok((i64::from(value) - prediction) as i32)
}

fn model_bits(streams: &[Vec<Token>], contexts: usize, lz: bool) -> Result<(usize, usize)> {
    let model =
        Model::build_with_options(streams, contexts, lz, Coding::Prefix, &entropy_options())?;
    let mut writer = BitWriter::new(SCRATCH_BYTES);
    model.write_header(&mut writer)?;
    let header = writer.bits_written();
    for stream in streams {
        model.write_stream(stream, &mut writer)?;
    }
    Ok((header, writer.bits_written() - header))
}

fn plane_score(channel: &Channel, full: bool) -> Result<PlaneScore> {
    let population = channel.samples.len();
    if population == 0 || channel.width.checked_mul(channel.height) != Some(population) {
        return Err(Error::InvalidInput(
            "palette admission pilot plane geometry".into(),
        ));
    }
    let training_sites = population.min(TRAIN_SITES);
    let mut samples = Vec::with_capacity(training_sites);
    for i in 0..training_sites {
        // Avoid multiplication by the (possibly large) population.
        let position =
            (population / training_sites) * i + (population % training_sites) * i / training_sites;
        let (neighbors, properties) = site(channel, position)?;
        samples.push(Sample {
            properties: properties.to_vec(),
            residuals: PREDICTORS
                .iter()
                .map(|&p| residual(channel.samples[position], neighbors, p))
                .collect::<Result<Vec<_>>>()?,
            weight: 1,
        });
    }
    let tree = ma_train::train(
        &samples,
        &TrainOptions {
            predictors: PREDICTORS.to_vec(),
            properties: vec![2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14],
            max_leaves: 4,
            max_property_values: 16,
            parallel: false,
            parallel_nodes: false,
            histogram_cache_bytes: 64 * 1024,
            ..TrainOptions::default()
        },
    )?;
    let mut tree_writer = BitWriter::new(SCRATCH_BYTES);
    tree.write_with_options(&mut tree_writer, &entropy_options(), None)?;
    let tree_bits = tree_writer.bits_written();
    let mut raw_streams = Vec::new();
    let mut lz_streams = Vec::new();
    let mut sampled = 0;
    let mut lz_matches = 0;
    let mut lz_matched_samples = 0;
    for window in windows(population, full) {
        let mut literals = Vec::with_capacity(window.len());
        sampled += window.len();
        for position in window {
            let (neighbors, properties) = site(channel, position)?;
            let (context, predictor, offset, multiplier) = tree.lookup(&properties)?;
            if offset != 0 || multiplier != 1 {
                return Err(Error::InvalidInput(
                    "unexpected admission MA constraint".into(),
                ));
            }
            literals.push(Literal {
                context: context as u32,
                value: pack_signed(residual(channel.samples[position], neighbors, predictor)?),
            });
        }
        raw_streams.push(
            literals
                .iter()
                .map(|t| Token::literal(t.context as usize, t.value))
                .collect(),
        );
        let commands = lz77::parse(
            &literals,
            channel.width,
            &Lz77Options {
                mode: Lz77Mode::Greedy,
                window_size: WINDOW,
                max_match_length: WINDOW,
                bucket_bits: 8,
                bucket_capacity: 1,
                range_search: false,
                prefix_filter: FilterPolicy::Disabled,
                require_global_savings: false,
                ..Lz77Options::default()
            },
        )?;
        let mut tokens = Vec::with_capacity(commands.len());
        for command in commands {
            match command {
                Command::Literal(t) => tokens.push(Token::literal(t.context as usize, t.value)),
                Command::Match {
                    context,
                    length,
                    distance,
                } => {
                    lz_matches += 1;
                    lz_matched_samples += length as usize;
                    tokens.push(Token {
                        context: context as usize,
                        value: length - 3,
                        is_length: true,
                    });
                    tokens.push(Token::literal(
                        tree.leaves(),
                        lz77::distance_symbol(distance as usize, channel.width)?,
                    ));
                }
            }
        }
        lz_streams.push(tokens);
    }
    let (literal_header_bits, literal_data_bits) = model_bits(&raw_streams, tree.leaves(), false)?;
    let (lz_header_bits, lz_data_bits) = model_bits(&lz_streams, tree.leaves() + 1, true)?;
    let scale = population as f64 / sampled as f64;
    // Only observed data bits are extrapolated. Model/tree/table headers are
    // paid once; LZ history never crosses gaps between sampled windows.
    let raw = literal_header_bits as f64 + literal_data_bits as f64 * scale;
    let lz = lz_header_bits as f64 + lz_data_bits as f64 * scale;
    let used_lz = lz_matches > 0 && lz < raw;
    Ok(PlaneScore {
        population,
        sampled,
        training_sites,
        leaves: tree.leaves(),
        tree_bits,
        literal_header_bits,
        literal_data_bits,
        lz_header_bits,
        lz_data_bits,
        lz_matches,
        lz_matched_samples,
        used_lz,
        total_bits: tree_bits as f64 + if used_lz { lz } else { raw },
    })
}

#[cfg(test)]
#[path = "palette_admission_tests.rs"]
mod tests;
