//! Independent, opt-in e9 RCT x cross-channel MA experiment. No defaults change.
//!
//! All 42 legal transforms use the actual native RCT arithmetic. Screening uses
//! at most nine original, contiguous integer tiles, including invisible RGB and
//! every extra plane. Inputs that fit the sample budget use one whole-image
//! tile without duplicated evidence. Native `cost::estimate_cost` supplies
//! gradient contexts, HybridUint tokens and extra bits. An approximate one-split token
//! context proxy discounts that cost for previous-channel features. It is NOT
//! MA training, a crop encode, an output-size bound, or an automatic RCT oracle.
//! The proxy tries sign/zero/magnitude of the reference gradient residual and
//! sign of its value, conditioned on gradient-spread context, with a 64-bit
//! split charge. It keeps only the best single feature of the nearest one/two
//! channels; it does not claim to model combinations, weighted prediction,
//! palettes, patches, tree quantization, LZ77, or the parent's MA lookahead.
//!
//! Round-robin rankings for previousChannels=0/1/2 select a small common RCT
//! shortlist. Every shortlisted RCT is FULLY encoded with all three settings.
//! Native e9 patches-off and auto-RCT controls for 0/1/2 are always present;
//! all trials explicitly disable patches to isolate RCT/MA. Control zero
//! aliases the identical baseline, without encoding it twice. All actual
//! encodes, including losers/failures, are timed; no incumbent byte pruning.
//! First complete minimum wins ties. Only incumbent bytes are retained.
//!
//! `base` supplies common native resource/tool overrides, so combined experiments
//! can explicitly supply MA lookahead there. With a default base, the baseline
//! is source-projected native e9 with patches explicitly Off, not source Auto.
//! Caller patch detectors/reference policies are not inherited. Decoder policy
//! is never modified.
//! No filename/path is accepted by the scoring API. Kernels and estimator are
//! ports of libjxl b866b3defec13494bfbaf705db93581cda118e37; see their modules
//! for upstream BSD attribution. The proxy and orchestration are local research.

use std::time::Instant;

use crate::entropy::HybridUintConfig;
use crate::modular::{
    cost::{ChannelView, estimate_cost},
    predictor::{NeighborRow, clamped_gradient},
    transforms,
    tree::pack_signed,
};
use crate::{EncodeOptions, Error, Image, PixelData, Result};

use super::{CandidateParams, adapter, params::Override};

pub const SCREEN_METHOD: &str = "native-gradient-token-cost+previous-channel-one-split-proxy-v1";

#[derive(Clone, Copy, Debug)]
pub struct RctJointOptions {
    /// Common transform shortlist, each fully encoded at previousChannels 0/1/2.
    /// Restricted to 1..=8 to keep this a screen, not a 126-encode tournament.
    pub shortlist: usize,
    /// Original spatial pixels, not channel samples; 9..=65536.
    pub max_sample_pixels: usize,
}

impl Default for RctJointOptions {
    fn default() -> Self {
        Self {
            shortlist: 3,
            max_sample_pixels: 8192,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SampleTile {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    /// Offset into each internal sampled plane, not the original image.
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct RctScreenScore {
    pub rct: u32,
    pub native_bits: f64,
    /// Index is previousChannels 0, 1, 2; units are sampled proxy bits only.
    pub proxy_bits: [f64; 3],
    /// Includes transform copy/arithmetic, native estimation and all proxy work.
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct RctJointScreen {
    pub method: &'static str,
    pub sampled_pixels: usize,
    pub sampled_channels: usize,
    pub tiles: Vec<SampleTile>,
    /// Canonical legal RCT order 0..=41, including every losing screen.
    pub scores: Vec<RctScreenScore>,
    pub shortlisted_rcts: Vec<u32>,
    /// Subset of seconds, not an additional charge.
    pub sampling_seconds: f64,
    pub seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RctJointTrialKind {
    /// Source-projected e9 with patches explicitly disabled for isolation.
    NativeE9Baseline,
    AutoRctControl,
    ShortlistedRct,
}

#[derive(Clone, Debug)]
pub struct RctJointTrial {
    pub kind: RctJointTrialKind,
    pub params: CandidateParams,
    /// Exact options actually submitted to the encoder, including MA sampling.
    pub options: EncodeOptions,
    pub bytes: Option<usize>,
    pub seconds: f64,
    /// Identical baseline/control reuse. This row adds zero encode seconds.
    pub reused_from: Option<usize>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct RctJointTiming {
    /// Complete native search call, including setup, all screens and all trials.
    pub total_seconds: f64,
    pub screen_seconds: f64,
    /// Sum of actual full encodes, including failures and losers.
    pub encode_seconds: f64,
    /// Pool setup, projection, bookkeeping, etc.; disjoint from screen/encode.
    pub overhead_seconds: f64,
    /// Subset of overhead_seconds. Zero when using a caller-owned pool/serial.
    pub pool_setup_seconds: f64,
}

#[derive(Debug)]
pub struct RctJointResult {
    pub encoded: Vec<u8>,
    pub winner_trial: usize,
    pub screen: RctJointScreen,
    pub trials: Vec<RctJointTrial>,
    pub timing: RctJointTiming,
}

fn validate_input(image: &Image, settings: &RctJointOptions) -> Result<()> {
    if !(1..=8).contains(&settings.shortlist) || !(9..=65536).contains(&settings.max_sample_pixels)
    {
        return Err(Error::InvalidInput(
            "RCT joint: shortlist must be 1..=8 and max_sample_pixels 9..=65536".into(),
        ));
    }
    if image.color_channels() != 3
        || image.has_modular_planes()
        || !matches!(image.pixels(), PixelData::U8(_) | PixelData::U16(_))
        || image
            .extra_planes()
            .iter()
            .any(|p| !matches!(p, PixelData::U8(_) | PixelData::U16(_)))
        || (0..image.channel_count()).any(|c| {
            let depth = image.channel_depth(c);
            depth.is_float()
                || !matches!(depth.bits_per_sample, 8 | 16)
                || image.channel_shift(c) != 0
        })
    {
        return Err(Error::InvalidInput(
            "RCT joint supports original RGB/RGBA U8/U16 with 8/16-bit integer, full-resolution extras only; gray, float, wide, sub-8/16, shifted and internal working planes are unsupported".into(),
        ));
    }
    if image.channel_count() > 259
        || image
            .channel_count()
            .checked_mul(settings.max_sample_pixels)
            .is_none_or(|n| n > 4 << 20)
    {
        return Err(Error::Limit("RCT joint bounded sample channel storage"));
    }
    Ok(())
}

fn sample(image: &Image, budget: usize) -> (Vec<Vec<i32>>, Vec<SampleTile>) {
    let w = image.width() as usize;
    let h = image.height() as usize;
    if let Some(pixels) = w.checked_mul(h).filter(|&pixels| pixels <= budget) {
        let planes = (0..image.channel_count())
            .map(|c| (0..pixels).map(|i| image.modular_sample(c, i)).collect())
            .collect();
        return (
            planes,
            vec![SampleTile {
                x: 0,
                y: 0,
                width: w,
                height: h,
                offset: 0,
            }],
        );
    }
    // Integer sqrt keeps sample sites independent of float rounding/platform.
    let mut edge = 1usize;
    while (edge + 1) * (edge + 1) * 9 <= budget {
        edge += 1;
    }
    let tw = edge.min(w);
    let th = edge.min(h);
    let nx = (w - tw + 1).min(3);
    let ny = (h - th + 1).min(3);
    let mut planes = vec![Vec::with_capacity(tw * th * nx * ny); image.channel_count()];
    let mut tiles = Vec::new();
    for iy in 0..ny {
        for ix in 0..nx {
            let x = if nx == 1 { 0 } else { ix * (w - tw) / (nx - 1) };
            let y = if ny == 1 { 0 } else { iy * (h - th) / (ny - 1) };
            tiles.push(SampleTile {
                x,
                y,
                width: tw,
                height: th,
                offset: planes[0].len(),
            });
            for (c, plane) in planes.iter_mut().enumerate() {
                for row in y..y + th {
                    for col in x..x + tw {
                        plane.push(image.modular_sample(c, row * w + col));
                    }
                }
            }
        }
    }
    (planes, tiles)
}

const CONTEXTS: usize = 17;
const TOKENS: usize = 128;
const HIST: usize = CONTEXTS * TOKENS;
const FEATURES: usize = 4;

fn histogram_bits(counts: &[u32]) -> f64 {
    counts
        .chunks_exact(TOKENS)
        .map(|hist| {
            let total: u32 = hist.iter().sum();
            if total == 0 {
                return 0.0;
            }
            let total = f64::from(total);
            hist.iter()
                .filter(|&&n| n != 0)
                .map(|&n| f64::from(n) * (total / f64::from(n)).log2())
                .sum::<f64>()
        })
        .sum()
}

fn reference_features(plane: &[i32], width: usize, x: usize, y: usize) -> [bool; FEATURES] {
    let pos = y * width + x;
    // Matches encode::visit_channel's previous-channel property edge rules,
    // which differ from current-channel prediction at the left image edge.
    let left = if x > 0 { plane[pos - 1] } else { 0 };
    let top = if y > 0 { plane[pos - width] } else { left };
    let nw = if x > 0 && y > 0 {
        plane[pos - width - 1]
    } else {
        left
    };
    let residual = plane[pos].wrapping_sub(clamped_gradient(left, top, nw));
    [
        residual < 0,
        residual == 0,
        residual.unsigned_abs() >= 16,
        plane[pos] < 0,
    ]
}

fn proxy_gains(planes: &[Vec<i32>], tiles: &[SampleTile]) -> Result<[f64; 3]> {
    const CUTOFFS: [u32; 16] = [
        1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
    ];
    let mut gains = [0.0f64; 3];
    for c in 1..planes.len() {
        let refs = c.min(2);
        let mut base = vec![0u32; HIST];
        let mut split = vec![0u32; refs * FEATURES * 2 * HIST];
        for tile in tiles {
            let end = tile.offset + tile.width * tile.height;
            let current = &planes[c][tile.offset..end];
            for y in 0..tile.height {
                let row = NeighborRow::new(current, tile.width, tile.width, y)
                    .map_err(|e| Error::Encoding(e.to_string()))?;
                for x in 0..tile.width {
                    let n = row.at(x).map_err(|e| Error::Encoding(e.to_string()))?;
                    let residual = current[y * tile.width + x]
                        .wrapping_sub(clamped_gradient(n.left, n.top, n.topleft));
                    let token = HybridUintConfig::default()
                        .encode(pack_signed(residual))
                        .token as usize;
                    let spread = (n.left.max(n.top).max(n.topleft) as i64
                        - n.left.min(n.top).min(n.topleft) as i64)
                        as u32;
                    let context = CUTOFFS.partition_point(|&cutoff| cutoff <= spread);
                    let index = context * TOKENS + token;
                    base[index] += 1;
                    for r in 0..refs {
                        let previous = &planes[c - 1 - r][tile.offset..end];
                        for (f, above) in reference_features(previous, tile.width, x, y)
                            .into_iter()
                            .enumerate()
                        {
                            split[((r * FEATURES + f) * 2 + usize::from(above)) * HIST + index] +=
                                1;
                        }
                    }
                }
            }
        }
        let unsplit = histogram_bits(&base);
        let mut best = 0.0f64;
        for r in 0..refs {
            for f in 0..FEATURES {
                let start = (r * FEATURES + f) * 2 * HIST;
                best = best.max(unsplit - histogram_bits(&split[start..start + 2 * HIST]) - 64.0);
            }
            gains[r + 1] += best;
        }
        if refs == 1 {
            gains[2] += best;
        }
    }
    Ok(gains)
}

/// Cheap, bounded all-42 screen only. Does not create a pool or encode a crop.
pub fn screen_rcts(image: &Image, settings: &RctJointOptions) -> Result<RctJointScreen> {
    let started = Instant::now();
    validate_input(image, settings)?;
    let sampling = Instant::now();
    let (original, tiles) = sample(image, settings.max_sample_pixels);
    let sampling_seconds = sampling.elapsed().as_secs_f64();
    let mut scores = Vec::with_capacity(42);
    for rct in 0..42 {
        let trial_started = Instant::now();
        let mut planes = original.clone();
        transforms::rct_forward(&mut planes, 0, rct)?;
        let mut native_bits = 0.0;
        for tile in &tiles {
            let end = tile.offset + tile.width * tile.height;
            let views: Vec<_> = planes
                .iter()
                .map(|p| ChannelView {
                    pixels: &p[tile.offset..end],
                    width: tile.width,
                    height: tile.height,
                    stride: tile.width,
                })
                .collect();
            native_bits += f64::from(estimate_cost(&views)?);
        }
        let gains = proxy_gains(&planes, &tiles)?;
        scores.push(RctScreenScore {
            rct,
            native_bits,
            proxy_bits: gains.map(|gain| (native_bits - gain).max(0.0)),
            seconds: trial_started.elapsed().as_secs_f64(),
        });
    }
    let rankings: [Vec<usize>; 3] = std::array::from_fn(|previous| {
        let mut order: Vec<_> = (0..42).collect();
        order.sort_by(|&a, &b| {
            scores[a].proxy_bits[previous]
                .total_cmp(&scores[b].proxy_bits[previous])
                .then_with(|| a.cmp(&b))
        });
        order
    });
    let mut shortlisted_rcts = Vec::new();
    'rank: for rank in 0..42 {
        for order in &rankings {
            let rct = scores[order[rank]].rct;
            if !shortlisted_rcts.contains(&rct) {
                shortlisted_rcts.push(rct);
                if shortlisted_rcts.len() == settings.shortlist {
                    break 'rank;
                }
            }
        }
    }
    Ok(RctJointScreen {
        method: SCREEN_METHOD,
        sampled_pixels: original[0].len(),
        sampled_channels: original.len(),
        tiles,
        scores,
        shortlisted_rcts,
        sampling_seconds,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Whole native experiment. Builds at most one bounded pool and reuses it for
/// sequential full trials. No global pool or independent candidate threads.
pub fn run_rct_joint(
    image: &Image,
    base: &EncodeOptions,
    settings: &RctJointOptions,
) -> Result<RctJointResult> {
    let started = Instant::now();
    validate_input(image, settings)?;
    if !(1..=256).contains(&base.threads) {
        return Err(Error::InvalidInput(
            "RCT joint threads must be 1..=256".into(),
        ));
    }
    let setup = Instant::now();
    let pool = if base.threads == 1 || crate::execution::is_serial() {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(base.threads)
                .build()
                .map_err(|e| Error::Encoding(e.to_string()))?,
        )
    };
    let setup_seconds = if pool.is_some() {
        setup.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let mut result = run(image, base, settings, pool.as_ref(), started, setup_seconds)?;
    drop(pool);
    result.timing.total_seconds = started.elapsed().as_secs_f64();
    result.timing.overhead_seconds =
        (result.timing.total_seconds - result.timing.screen_seconds - result.timing.encode_seconds)
            .max(0.0);
    Ok(result)
}

/// Reuses the exact caller-owned bounded pool. A serial/null execution scope
/// remains serial. Pool construction outside this call is the caller's cost.
pub fn run_rct_joint_with_pool(
    image: &Image,
    base: &EncodeOptions,
    settings: &RctJointOptions,
    pool: &rayon::ThreadPool,
) -> Result<RctJointResult> {
    let started = Instant::now();
    validate_input(image, settings)?;
    if !(1..=256).contains(&base.threads) {
        return Err(Error::InvalidInput(
            "RCT joint threads must be 1..=256".into(),
        ));
    }
    let pool = (!crate::execution::is_serial()).then_some(pool);
    run(image, base, settings, pool, started, 0.0)
}

fn run(
    image: &Image,
    base: &EncodeOptions,
    settings: &RctJointOptions,
    pool: Option<&rayon::ThreadPool>,
    started: Instant,
    pool_setup_seconds: f64,
) -> Result<RctJointResult> {
    let screen = screen_rcts(image, settings)?;
    let mut ordinary = CandidateParams::default().backend(9)?;
    ordinary.patches = Override::Off;
    ordinary.max_properties =
        Some(ordinary.previous_channel_budget(base.modular.source_tool_defaults) as u8);
    let mut planned = vec![(RctJointTrialKind::NativeE9Baseline, ordinary.clone())];
    for previous in 0..=2 {
        let mut params = ordinary.clone();
        params.max_properties = Some(previous);
        planned.push((RctJointTrialKind::AutoRctControl, params));
    }
    for &rct in &screen.shortlisted_rcts {
        for previous in 0..=2 {
            let mut params = ordinary.clone();
            params.rct = Some(rct);
            params.max_properties = Some(previous);
            planned.push((RctJointTrialKind::ShortlistedRct, params));
        }
    }
    let mut trials: Vec<RctJointTrial> = Vec::new();
    let mut winner: Option<(usize, Vec<u8>)> = None;
    for (kind, params) in planned {
        if let Some(index) = trials.iter().position(|t| t.params == params) {
            let prior = &trials[index];
            trials.push(RctJointTrial {
                kind,
                params,
                options: prior.options.clone(),
                bytes: prior.bytes,
                seconds: 0.0,
                reused_from: Some(index),
                error: prior.error.clone(),
            });
            continue;
        }
        let options = adapter::project_native(&params, image, base)?.exact_options()?;
        // A stale caller-owned single-use representation receipt cannot be
        // reused across encodes. Each trial owns its collector when requested.
        let mut options = options;
        if options.modular.representation_receipt.is_some() {
            options.modular.representation_receipt = Some(Default::default());
        }
        if options.modular.model_cost_receipt.is_some() {
            options.modular.model_cost_receipt = Some(Default::default());
        }
        if let Some(reference) = options.source_patch_reference.as_mut() {
            if reference.modular.representation_receipt.is_some() {
                reference.modular.representation_receipt = Some(Default::default());
            }
            if reference.modular.model_cost_receipt.is_some() {
                reference.modular.model_cost_receipt = Some(Default::default());
            }
        }
        let encode_started = Instant::now();
        let output = match pool {
            Some(pool) => crate::encode::encode_with_pool(image, &options, pool),
            None => crate::encode::encode_serial(image, &options),
        };
        let seconds = encode_started.elapsed().as_secs_f64();
        let (bytes, error) = match output {
            Ok(encoded) => {
                let bytes = encoded.len();
                if winner.as_ref().is_none_or(|(_, best)| bytes < best.len()) {
                    winner = Some((trials.len(), encoded));
                }
                (Some(bytes), None)
            }
            Err(error) => (None, Some(error.to_string())),
        };
        trials.push(RctJointTrial {
            kind,
            params,
            options,
            bytes,
            seconds,
            reused_from: None,
            error,
        });
    }
    let Some((winner_trial, encoded)) = winner else {
        return Err(Error::Encoding(format!(
            "RCT joint: all full encodes failed: {:?}",
            trials
                .iter()
                .map(|t| (&t.params, &t.error, t.seconds))
                .collect::<Vec<_>>()
        )));
    };
    let encode_seconds: f64 = trials.iter().map(|t| t.seconds).sum();
    let total_seconds = started.elapsed().as_secs_f64();
    let timing = RctJointTiming {
        total_seconds,
        screen_seconds: screen.seconds,
        encode_seconds,
        overhead_seconds: (total_seconds - screen.seconds - encode_seconds).max(0.0),
        pool_setup_seconds,
    };
    Ok(RctJointResult {
        encoded,
        winner_trial,
        screen,
        trials,
        timing,
    })
}

#[cfg(test)]
mod tests;
