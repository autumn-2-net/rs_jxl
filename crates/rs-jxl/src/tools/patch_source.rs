// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Frozen b866b3defec13494bfbaf705db93581cda118e37 enc_patch_dictionary.cc/.h:
//! non-XYB PatchColorspaceInfo, FindTextLikePatches, quantized deduplication,
//! first-fit atlas, Add-color/None-extra dictionary, and SubtractFrom. The
//! outer route gate is enc_modular.cc::ComputeEncodingData (line 952).
//!
//! Detection uses source scalar f32 operations, including external uint sample
//! normalization (enc_external_image.cc / base/float.h), subtraction BEFORE
//! quantization, mixed float/double comparison constants, and ordered weighted
//! accumulation. These are not rational/integer replacement thresholds. Fast-
//! math/compiler reassociation and equal-area std::sort ties are not a portable
//! cross-build identity guarantee. Fingerprints retain truncation toward zero
//! and unsigned-byte memcmp ordering. Only grouping is quantized: the atlas
//! keeps exact signed sample differences, and every occurrence-specific
//! difference stays in residuals. For 1..16-bit input the integer atlas equals
//! source float_to_int's lossless rounding of normalized sample differences.
//! Gray is expanded logically for the three source color metrics. All extras,
//! including alpha and RGB hidden by alpha zero, remain untouched by Add.
//!
//! The fast and baseline flood/component routes are executable; environment
//! choices are explicit config fields. Detection is sequential and bounded by
//! PatchOptions. patch_source_float.rs shares the detector for actual f32 color
//! values and implements declared-format atlas roundtrip and checked Add replay.
//! No XYB, dots, perceptual optimization, size trial, full reference bitstream
//! roundtrip, or replay capture is ported here.
//!
//! Parent integration: emit an exact SIGNED Modular ReferenceOnly atlas in slot
//! 3 saved before color transform, followed by the signed residual frame with
//! kPatches. Tokenize this dictionary with patches::tokenize (ten contexts).
//! Color uses Add=2; every extra uses None=0. Do not clamp/offset/unsigned-cast
//! either atlas or residual. Keep original metadata and scaling, bypass alpha
//! cleanup, and retain signed range through transforms and predictor packing.
//! If a reference is changed by encoding, residuals must be recomputed against
//! the reconstructed reference, as source RoundtripPatchFrame does. The integer
//! replay oracle is exact, but normalized decoder arithmetic is also verified
//! on covered color pixels by patch_source_replay.rs before accepting a plan.
//! tests/patch_add_reference.rs exercises the signed frame path against
//! explicitly supplied official djxl oracles. The legacy Replace pipeline is
//! unchanged.

use std::collections::{BTreeMap, VecDeque};

use super::{
    AtlasPacking, BACKGROUND_DISTANCE, ImageView, PatchBlendMode, PatchBlending, PatchDictionary,
    PatchInfo, PatchKey, PatchOptions, PatchPosition, PatchReferencePosition, REFERENCE_ID,
    SEED_SIDE, invalid, neighbors, zeros,
};
use crate::{Error, Result};

#[path = "patch_source_float.rs"]
mod float;
pub use float::{
    FloatImageView, FloatPatchImage, FloatPatchPlan, compose_float_add, discover_source_float,
};

#[path = "patch_source_replay.rs"]
mod replay;
pub use replay::verify_add_replay;

/// The source outer gate, before detector overrides and seed availability.
/// Override On does not bypass the speed/depth/groupwise gate.
pub fn source_enabled(
    bits_per_sample: u32,
    speed_tier: i8,
    decoding_speed_tier: u32,
    do_color: bool,
    groupwise: bool,
) -> bool {
    do_color
        && (1..=16).contains(&bits_per_sample)
        && (-1..6).contains(&speed_tier)
        && decoding_speed_tier < 2
        && !groupwise
}

/// Metadata-aware frame capability, separate from the source's effort gate.
/// `floating_point` describes COLOR, not extras. F16/F32 color bits must not
/// be treated as unsigned samples. Extra planes are opaque None-blended data:
/// their depth/type does not constrain color discovery. The frame adapter keeps
/// native ceil-divided dimensions for shifts 0..3 (decoder EC factor <=8).
/// The lower-level ImageView API remains full-resolution unsigned-only.
/// Floating color uses discover_source_float with a validated float format,
/// source conversion, and exact replay. Capability does not guarantee that a
/// particular floating Add candidate is representable or bit-exact.
pub fn source_supported(
    bits_per_sample: u32,
    floating_point: bool,
    extra_channel_dim_shifts: &[u32],
) -> bool {
    (if floating_point { 5..=16 } else { 1..=16 }).contains(&bits_per_sample)
        && extra_channel_dim_shifts.iter().all(|&shift| shift <= 3)
}

#[derive(Clone, Debug)]
pub struct SourcePatchConfig {
    /// false is source Override::Off. Auto/On both require actual border seeds.
    pub enabled: bool,
    pub do_color: bool,
    /// common.h::SpeedTier, including TectonicPlate=-1.
    pub speed_tier: i8,
    pub decoding_speed_tier: u32,
    pub groupwise: bool,
    /// True if COLOR is floating point, not a request to reinterpret bits.
    /// The frame adapter omits extras from detection and preserves their raw
    /// samples, including floating extras. Integer ImageView discovery skips
    /// floating color; the frame adapter dispatches it to the typed float API.
    pub floating_point: bool,
    /// Empty declares all ImageView extras full-resolution; otherwise one per
    /// extra. The metadata-aware frame adapter handles nonzero shifts without
    /// constructing a full-resolution ImageView for those extras.
    pub extra_channel_dim_shifts: Vec<u32>,
    /// JXL_LOSSLESS_PATCH_FAST; true is the source non-XYB lossless default.
    pub fast: bool,
    /// Source default 2; the low-contrast experiment uses 1, in quantized units.
    pub min_peak: i32,
    /// Source fingerprint experiment: 1 (default), 2, or 4. Does not quantize
    /// the actual atlas, nor permit dropping occurrence-specific residuals.
    pub fingerprint_divisor: i32,
    pub limits: PatchOptions,
}

impl Default for SourcePatchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            do_color: true,
            speed_tier: 5,
            decoding_speed_tier: 0,
            groupwise: false,
            floating_point: false,
            extra_channel_dim_shifts: Vec::new(),
            fast: true,
            min_peak: 2,
            fingerprint_divisor: 1,
            limits: PatchOptions::default(),
        }
    }
}

impl SourcePatchConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        self.limits.validate()?;
        if !(-1..=9).contains(&self.speed_tier) {
            return Err(invalid("source patch speed tier must be -1..=9"));
        }
        if !matches!(self.min_peak, 1 | 2) || !matches!(self.fingerprint_divisor, 1 | 2 | 4) {
            return Err(invalid("invalid source patch peak or fingerprint divisor"));
        }
        Ok(())
    }
}

/// Signed working samples, deliberately distinct from unsigned ImageView.
/// bit_depth describes original normalization, not a bound on working values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedPatchImage {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub color_channels: usize,
    pub planes: Vec<Vec<i32>>,
}

impl SignedPatchImage {
    fn validate(&self) -> Result<usize> {
        if self.width == 0
            || self.height == 0
            || self.width > i32::MAX as usize
            || self.height > i32::MAX as usize
            || !(1..=16).contains(&self.bit_depth)
            || !matches!(self.color_channels, 1 | 3)
            || self.planes.len() < self.color_channels
        {
            return Err(invalid("invalid signed patch image metadata"));
        }
        let pixels = self
            .width
            .checked_mul(self.height)
            .ok_or(Error::Limit("signed patch pixels"))?;
        if self.planes.iter().any(|p| p.len() != pixels) {
            return Err(invalid("invalid signed patch plane size"));
        }
        Ok(pixels)
    }

    pub fn requires_signed_samples(&self) -> bool {
        if !(1..=16).contains(&self.bit_depth) {
            return true;
        }
        let Some(maximum) = 1i32
            .checked_shl(self.bit_depth)
            .and_then(|v| v.checked_sub(1))
        else {
            return true;
        };
        self.planes.iter().flatten().any(|&v| v < 0 || v > maximum)
    }
}

#[derive(Clone, Debug)]
pub struct AddPatchPlan {
    pub reference: SignedPatchImage,
    pub dictionary: PatchDictionary,
    pub residual: SignedPatchImage,
}

impl AddPatchPlan {
    pub fn requires_signed_samples(&self) -> bool {
        self.reference.requires_signed_samples() || self.residual.requires_signed_samples()
    }
}

fn color(image: ImageView<'_>, position: usize) -> [i32; 3] {
    std::array::from_fn(|c| image.planes[if image.color_channels == 1 { 0 } else { c }][position])
}

#[derive(Clone, Copy)]
struct Colorspace {
    scale: f32,
}

impl Colorspace {
    fn new(bits: u32) -> Self {
        // ConvertFromExternalPlaneNoSizeCheck / LoadFloatRow. Division of a
        // sample by maximum is NOT equivalent to multiplying this reciprocal.
        Self {
            scale: 1.0f32 / ((1u32 << bits) - 1) as f32,
        }
    }

    fn normalize(self, samples: [i32; 3]) -> [f32; 3] {
        samples.map(|v| v as f32 * self.scale)
    }

    fn pick(self, image: ImageView<'_>, position: usize) -> [f32; 3] {
        self.normalize(color(image, position))
    }
}

#[derive(Clone, Copy)]
enum DetectionSamples<'a> {
    Integer(ImageView<'a>, Colorspace),
    Float(&'a [Vec<f32>]),
}

/// Detector pixels, not Modular sample words. Float input never passes through
/// the integer normalization or integer-difference operations.
#[derive(Clone, Copy)]
struct DetectionView<'a> {
    width: usize,
    height: usize,
    color_channels: usize,
    samples: DetectionSamples<'a>,
}

impl<'a> DetectionView<'a> {
    fn integer(image: ImageView<'a>) -> Self {
        Self {
            width: image.width,
            height: image.height,
            color_channels: image.color_channels,
            samples: DetectionSamples::Integer(image, Colorspace::new(image.bit_depth)),
        }
    }

    fn pick(self, position: usize) -> [f32; 3] {
        match self.samples {
            DetectionSamples::Integer(image, colorspace) => colorspace.pick(image, position),
            DetectionSamples::Float(planes) => std::array::from_fn(|c| {
                planes[if self.color_channels == 1 { 0 } else { c }][position]
            }),
        }
    }

    fn patch_sample(self, position: usize, reference: usize, channel: usize, delta: f32) -> i32 {
        match self.samples {
            DetectionSamples::Integer(image, _) => {
                color(image, position)[channel] - color(image, reference)[channel]
            }
            // Packed storage only. The float route decodes these f32 bits
            // before source format conversion; no integer arithmetic uses them.
            DetectionSamples::Float(_) => delta.to_bits() as i32,
        }
    }
}

fn same(a: [f32; 3], b: [f32; 3]) -> bool {
    // is_same_color uses an unsuffixed double threshold after float fabs.
    a.into_iter()
        .zip(b)
        .all(|(a, b)| f64::from((a - b).abs()) <= 1e-4)
}

fn similar(a: [f32; 3], b: [f32; 3], threshold: f32) -> bool {
    // PatchColorspaceInfo initializes weights from double constant expressions
    // but performs each subtraction, product and accumulation in float.
    const WEIGHTS: [f32; 3] = [
        (0.017f64 * 255.0) as f32,
        (0.02f64 * 255.0) as f32,
        (0.017f64 * 255.0) as f32,
    ];
    let mut distance = 0.0f32;
    for c in 0..3 {
        distance += (a[c] - b[c]).abs() * WEIGHTS[c];
    }
    distance <= threshold
}

fn quantize(delta: f32, channel: usize) -> i32 {
    const DEQUANT: [f32; 3] = [20.0f32 / 255.0, 22.0f32 / 255.0, 20.0f32 / 255.0];
    (delta / DEQUANT[channel]).clamp(-32768.0, 32767.0).trunc() as i32
}

#[derive(Default)]
struct Work {
    #[cfg(test)]
    initial_queue: usize,
    #[cfg(test)]
    component_pushes: usize,
    #[cfg(test)]
    low_peak: usize,
}

fn background(
    image: DetectionView<'_>,
    config: &SourcePatchConfig,
    _work: &mut Work,
) -> Result<Option<Vec<usize>>> {
    let (pw, ph) = (image.width / SEED_SIDE, image.height / SEED_SIDE);
    if pw < 3 || ph < 3 {
        return Ok(None);
    }
    let mut seeds = zeros::<u8>(pw * ph)?;
    let mut count = 0;
    // RunOnPool(1, ph - 2) has an exclusive end in the frozen source. Keep its
    // asymmetric final seed row rather than silently widening the scan.
    for py in 1..ph - 2 {
        for px in 1..pw - 1 {
            let origin = py * SEED_SIDE * image.width + px * SEED_SIDE;
            let base = image.pick(origin);
            if !(0..SEED_SIDE).all(|y| {
                (0..SEED_SIDE).all(|x| same(base, image.pick(origin + y * image.width + x)))
            }) {
                continue;
            }
            let mut equal = 0;
            for y in py - 1..=py + 1 {
                for x in px - 1..=px + 1 {
                    equal += usize::from(same(
                        base,
                        image.pick(y * SEED_SIDE * image.width + x * SEED_SIDE),
                    ));
                }
            }
            if equal >= 8 {
                seeds[py * pw + px] = 1;
                count += 1;
            }
        }
    }
    if count == 0 {
        return Ok(None);
    }
    let pixels = image.width * image.height;
    let mut origins = zeros::<usize>(pixels)?;
    origins.fill(usize::MAX);
    let mut queue = VecDeque::new();
    queue
        .try_reserve((count * 16).min(65_536))
        .map_err(|_| Error::Limit("source patch background queue"))?;
    for py in 1..ph - 1 {
        for px in 1..pw - 1 {
            if seeds[py * pw + px] == 0 {
                continue;
            }
            for y in py * SEED_SIDE..(py + 1) * SEED_SIDE {
                for x in px * SEED_SIDE..(px + 1) * SEED_SIDE {
                    let position = y * image.width + x;
                    origins[position] = position;
                    if !config.fast {
                        queue.push_back((position, position));
                    }
                }
            }
        }
    }
    if config.fast {
        for py in 1..ph - 1 {
            for px in 1..pw - 1 {
                if seeds[py * pw + px] == 0
                    || (py - 1..=py + 1).all(|y| (px - 1..=px + 1).all(|x| seeds[y * pw + x] != 0))
                {
                    continue;
                }
                for y in py * SEED_SIDE..(py + 1) * SEED_SIDE {
                    for x in px * SEED_SIDE..(px + 1) * SEED_SIDE {
                        let position = y * image.width + x;
                        let mut interior = true;
                        neighbors(image.width, image.height, position, |next| {
                            interior &= origins[next] != usize::MAX
                        });
                        if !interior {
                            queue.push_back((position, position));
                        }
                    }
                }
            }
        }
    }
    #[cfg(test)]
    {
        _work.initial_queue = queue.len();
    }
    while let Some((position, source)) = queue.pop_front() {
        neighbors(image.width, image.height, position, |next| {
            if origins[next] != usize::MAX {
                return;
            }
            let distance = (source % image.width).abs_diff(next % image.width)
                + (source / image.width).abs_diff(next / image.width);
            if distance <= BACKGROUND_DISTANCE && similar(image.pick(source), image.pick(next), 0.8)
            {
                origins[next] = source;
                queue.push_back((next, source));
            }
        });
    }
    Ok(Some(origins))
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct Fingerprint {
    width: usize,
    height: usize,
    bytes: Vec<u8>,
}

fn detect(
    image: ImageView<'_>,
    config: &SourcePatchConfig,
    work: &mut Work,
) -> Result<Vec<PatchInfo>> {
    detect_view(DetectionView::integer(image), config, work)
}

fn detect_view(
    image: DetectionView<'_>,
    config: &SourcePatchConfig,
    work: &mut Work,
) -> Result<Vec<PatchInfo>> {
    let Some(background) = background(image, config, work)? else {
        return Ok(Vec::new());
    };
    let pixels = image.width * image.height;
    let limits = &config.limits;
    let mut visited = zeros::<u8>(pixels)?;
    let mut stack = Vec::new();
    let mut groups = BTreeMap::<Fingerprint, PatchInfo>::new();
    let mut positions = 0;
    for start in 0..pixels {
        if background[start] != usize::MAX || (config.fast && visited[start] != 0) {
            continue;
        }
        let (mut min_x, mut max_x) = (start % image.width, start % image.width);
        let (mut min_y, mut max_y) = (start / image.width, start / image.width);
        let mut border = None;
        let mut same_border = true;
        stack.push(start);
        if config.fast {
            visited[start] = 1;
        }
        #[cfg(test)]
        {
            work.component_pushes += 1;
        }
        while let Some(position) = stack.pop() {
            if !config.fast {
                if visited[position] != 0 {
                    continue;
                }
                visited[position] = 1;
            }
            let (x, y) = (position % image.width, position / image.width);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            neighbors(image.width, image.height, position, |next| {
                if background[next] == usize::MAX {
                    if config.fast {
                        if visited[next] != 0 {
                            return;
                        }
                        visited[next] = 1;
                    }
                    stack.push(next);
                    #[cfg(test)]
                    {
                        work.component_pushes += 1;
                    }
                } else {
                    if config.fast
                        && (!same_border
                            || max_x - min_x >= limits.max_patch_size
                            || max_y - min_y >= limits.max_patch_size)
                    {
                        return;
                    }
                    if let Some(reference) = border {
                        same_border &=
                            similar(image.pick(background[next]), image.pick(reference), 0.03);
                    } else {
                        border = Some(background[next]);
                    }
                }
            });
        }
        let Some(reference) = border else {
            continue;
        };
        if !same_border
            || max_x - min_x >= limits.max_patch_size
            || max_y - min_y >= limits.max_patch_size
            || positions >= limits.max_positions
        {
            continue;
        }
        let normalized_base = image.pick(reference);
        let has_similar = (min_y.saturating_sub(2)..(max_y + 3).min(image.height)).any(|y| {
            (min_x.saturating_sub(2)..(max_x + 3).min(image.width))
                .any(|x| similar(normalized_base, image.pick(y * image.width + x), 0.03))
        });
        if !has_similar {
            continue;
        }
        let (width, height) = (max_x - min_x + 1, max_y - min_y + 1);
        let mut bytes = Vec::with_capacity(width * height * 3);
        let mut deltas = Vec::with_capacity(width * height * image.color_channels);
        let (mut too_small, mut too_big) = (true, false);
        for (c, &base) in normalized_base.iter().enumerate() {
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    // Keep the source subtraction of normalized operands. The
                    // exact integer delta remains separate from its float key.
                    let position = y * image.width + x;
                    let normalized = image.pick(position)[c] - base;
                    // Source float-to-int quantization of NaN is undefined.
                    // Never create an arithmetic patch containing nonfinite
                    // deltas; those input words remain in the untouched plane.
                    if !normalized.is_finite() {
                        too_big = true;
                        continue;
                    }
                    let q = quantize(normalized, c);
                    too_small &= q > -config.min_peak && q < config.min_peak;
                    too_big |= !(i8::MIN as i32..=i8::MAX as i32).contains(&q);
                    bytes.push((q / config.fingerprint_divisor) as i8 as u8);
                    if image.color_channels == 3 || c == 0 {
                        deltas.push(image.patch_sample(position, reference, c, normalized));
                    }
                }
            }
        }
        if too_small || too_big {
            #[cfg(test)]
            {
                work.low_peak += usize::from(too_small);
            }
            continue;
        }
        let fingerprint = Fingerprint {
            width,
            height,
            bytes,
        };
        let position = (min_x as u32, min_y as u32);
        if let Some(existing) = groups.get_mut(&fingerprint) {
            // std::pair<QuantizedPatch, positions> sorts equal keys by (x,y),
            // not the detector's (y,x) scan order. Retain that representative.
            if position < existing.positions[0] {
                existing.key.samples = deltas;
                existing.positions.insert(0, position);
            } else {
                existing.positions.push(position);
            }
            positions += 1;
        } else if groups.len() < limits.max_unique_patches {
            groups.insert(
                fingerprint,
                PatchInfo {
                    key: PatchKey {
                        width,
                        height,
                        samples: deltas,
                    },
                    positions: vec![position],
                },
            );
            positions += 1;
        }
    }
    let mut info: Vec<_> = groups
        .into_values()
        .filter(|p| p.positions.len() >= limits.min_occurrences)
        .collect();
    if info
        .iter()
        .map(|p| p.key.width * p.key.height)
        .max()
        .unwrap_or(0)
        < limits.min_max_patch_area
    {
        return Ok(Vec::new());
    }
    for patch in &mut info {
        patch.positions.sort_unstable();
    }
    // Source area sort leaves equal-area order unspecified; use stable key
    // order for reproducible Rust output. This does not change pixel replay.
    info.sort_by_key(|p| std::cmp::Reverse(p.key.width * p.key.height));
    Ok(info)
}

fn pack_source(info: &[PatchInfo], budget: usize) -> Result<Option<AtlasPacking>> {
    let total = info
        .iter()
        .try_fold(0usize, |n, patch| {
            n.checked_add(patch.key.width * patch.key.height)
        })
        .ok_or(Error::Limit("source patch atlas area"))?;
    if total > budget {
        return Ok(None);
    }
    // std::sqrt(size_t) is double, converted by std::max<float>. The following
    // slackness arithmetic is float, then truncated on assignment to size_t.
    let root = (total as f64).sqrt() as f32;
    let mut width = root.max(info.iter().map(|p| p.key.width).max().unwrap_or(0) as f32) as usize;
    let mut height = root.max(info.iter().map(|p| p.key.height).max().unwrap_or(0) as f32) as usize;
    loop {
        width = (width as f32 * 1.05f32 + 1.0) as usize;
        height = (height as f32 * 1.05f32 + 1.0) as usize;
        let Some(area) = width.checked_mul(height).filter(|&area| area <= budget) else {
            return Ok(None);
        };
        if width > i32::MAX as usize || height > i32::MAX as usize {
            return Ok(None);
        }
        let mut occupied = zeros::<u8>(area)?;
        let mut positions = Vec::with_capacity(info.len());
        let mut max_y = 0;
        for patch in info {
            let (w, h) = (patch.key.width, patch.key.height);
            let mut found = None;
            'search: for y0 in 0..=height - h {
                let mut x0 = 0;
                while x0 <= width - w {
                    let mut collision = false;
                    let mut x = x0;
                    // Preserve the frozen loop's last-row cursor. An earlier
                    // occupied row does NOT stop the outer row scan. Reusing
                    // the legacy Replace packer's first-collision jump changes
                    // placements and even the number of growth iterations.
                    for y in y0..y0 + h {
                        x = x0;
                        while x < x0 + w {
                            if occupied[y * width + x] != 0 {
                                collision = true;
                                break;
                            }
                            x += 1;
                        }
                    }
                    if !collision {
                        found = Some((x0, y0));
                        break 'search;
                    }
                    x0 = x + 1;
                }
            }
            let Some((x, y)) = found else {
                break;
            };
            for row in y..y + h {
                occupied[row * width + x..row * width + x + w].fill(1);
            }
            max_y = max_y.max(y + h);
            positions.push((x, y));
        }
        if positions.len() == info.len() {
            return Ok(Some((width, max_y, positions)));
        }
    }
}

fn validate_add(
    image: &SignedPatchImage,
    reference: &SignedPatchImage,
    dictionary: &PatchDictionary,
) -> Result<usize> {
    let pixels = image.validate()?;
    reference.validate()?;
    let stride = dictionary.validate()?;
    if image.bit_depth != reference.bit_depth
        || image.color_channels != reference.color_channels
        || image.planes.len() != reference.planes.len()
        || dictionary.num_extra_channels != image.planes.len() - image.color_channels
    {
        return Err(invalid("signed patch reference metadata differs"));
    }
    for blends in dictionary.blendings.chunks_exact(stride) {
        if !matches!(blends[0].mode, PatchBlendMode::None | PatchBlendMode::Add)
            || blends[1..].iter().any(|b| b.mode != PatchBlendMode::None)
        {
            return Err(invalid(
                "source Add patches modify color only; extras must use None",
            ));
        }
    }
    let mut runs = 0usize;
    let mut previous = None;
    for position in &dictionary.positions {
        if previous != Some(position.ref_pos_idx) {
            runs += 1;
        }
        previous = Some(position.ref_pos_idx);
        let crop = dictionary.references[position.ref_pos_idx];
        if crop.reference != REFERENCE_ID
            || u64::from(crop.x0) + u64::from(crop.width) > reference.width as u64
            || u64::from(crop.y0) + u64::from(crop.height) > reference.height as u64
            || u64::from(position.x) + u64::from(crop.width) > image.width as u64
            || u64::from(position.y) + u64::from(crop.height) > image.height as u64
        {
            return Err(invalid("signed patch rectangle outside reference or image"));
        }
    }
    let max_refs = 1024usize
        .checked_add(pixels / 4)
        .ok_or(Error::Limit("source patch count"))?;
    let max_positions = max_refs
        .checked_mul(4)
        .ok_or(Error::Limit("source patch count"))?;
    if runs > max_refs || dictionary.positions.len() > max_positions {
        return Err(Error::Limit("source patches exceed decoder count limits"));
    }
    Ok(stride)
}

fn apply(
    image: &SignedPatchImage,
    reference: &SignedPatchImage,
    dictionary: &PatchDictionary,
    subtract: bool,
) -> Result<SignedPatchImage> {
    let stride = validate_add(image, reference, dictionary)?;
    let mut output = image.clone();
    for (index, position) in dictionary.positions.iter().enumerate() {
        if dictionary.blendings[index * stride].mode == PatchBlendMode::None {
            continue;
        }
        let crop = dictionary.references[position.ref_pos_idx];
        for c in 0..image.color_channels {
            for y in 0..crop.height as usize {
                for x in 0..crop.width as usize {
                    let dst = (position.y as usize + y) * image.width + position.x as usize + x;
                    let src = (crop.y0 as usize + y) * reference.width + crop.x0 as usize + x;
                    let value = &mut output.planes[c][dst];
                    *value = if subtract {
                        value.checked_sub(reference.planes[c][src])
                    } else {
                        value.checked_add(reference.planes[c][src])
                    }
                    .ok_or(Error::Limit("signed patch arithmetic exceeds i32"))?;
                }
            }
        }
    }
    Ok(output)
}

/// Exact integer SubtractFrom specialization. Approximate fingerprint matches
/// keep all differences, including negative and above-nominal residuals.
pub fn subtract_add(
    image: ImageView<'_>,
    reference: &SignedPatchImage,
    dictionary: &PatchDictionary,
) -> Result<SignedPatchImage> {
    image.validate()?;
    apply(
        &SignedPatchImage {
            width: image.width,
            height: image.height,
            bit_depth: image.bit_depth,
            color_channels: image.color_channels,
            planes: image.planes.to_vec(),
        },
        reference,
        dictionary,
        true,
    )
}

/// Integer replay in dictionary order. Validates final original nominal range;
/// intermediate working values remain signed and are never clamped. This alone
/// does not prove decoder f32 recovery; discover_source also calls
/// verify_add_replay before accepting a plan.
pub fn compose_add(
    residual: &SignedPatchImage,
    reference: &SignedPatchImage,
    dictionary: &PatchDictionary,
) -> Result<Vec<Vec<i32>>> {
    let output = apply(residual, reference, dictionary, false)?;
    ImageView {
        width: output.width,
        height: output.height,
        bit_depth: output.bit_depth,
        color_channels: output.color_channels,
        planes: &output.planes,
    }
    .validate()?;
    Ok(output.planes)
}

/// Source non-XYB scan and quantized grouping, with exact signed integer data.
/// No implicit fallback to Replace. None means the source gate or detector
/// rejected this candidate, not that an unsigned frame can encode the result.
pub fn discover_source(
    image: ImageView<'_>,
    config: &SourcePatchConfig,
) -> Result<Option<AddPatchPlan>> {
    config.validate()?;
    if config.floating_point
        || !config.enabled
        || !source_enabled(
            image.bit_depth,
            config.speed_tier,
            config.decoding_speed_tier,
            config.do_color,
            config.groupwise,
        )
    {
        return Ok(None);
    }
    if !source_supported(
        image.bit_depth,
        config.floating_point,
        &config.extra_channel_dim_shifts,
    ) {
        return Ok(None);
    }
    image.validate()?;
    if !config.extra_channel_dim_shifts.is_empty()
        && config.extra_channel_dim_shifts.len() != image.planes.len() - image.color_channels
    {
        return Err(invalid(
            "source patch dimension shifts must describe every extra channel",
        ));
    }
    if config
        .extra_channel_dim_shifts
        .iter()
        .any(|&shift| shift != 0)
    {
        return Err(invalid(
            "subsampled patch extras require the metadata-aware frame adapter, not ImageView",
        ));
    }
    let info = detect(image, config, &mut Work::default())?;
    let Some(packed) = pack_dictionary(
        &info,
        image.color_channels,
        image.planes.len() - image.color_channels,
        config.limits.max_reference_pixels,
    )?
    else {
        return Ok(None);
    };
    let reference = SignedPatchImage {
        width: packed.width,
        height: packed.height,
        bit_depth: image.bit_depth,
        color_channels: image.color_channels,
        planes: packed.planes,
    };
    let residual = subtract_add(image, &reference, &packed.dictionary)?;
    verify_add_replay(image, &residual, &reference, &packed.dictionary)?;
    Ok(Some(AddPatchPlan {
        reference,
        dictionary: packed.dictionary,
        residual,
    }))
}

// Sample storage is integer differences or explicitly tagged f32 delta bits.
// Packing and dictionary order do not depend on the sample representation.
struct PackedPatches {
    width: usize,
    height: usize,
    planes: Vec<Vec<i32>>,
    dictionary: PatchDictionary,
}

fn pack_dictionary(
    info: &[PatchInfo],
    color_channels: usize,
    num_extra_channels: usize,
    budget: usize,
) -> Result<Option<PackedPatches>> {
    if info.is_empty() {
        return Ok(None);
    }
    let Some((width, height, atlas_positions)) = pack_source(info, budget)? else {
        return Ok(None);
    };
    let mut planes = (0..color_channels + num_extra_channels)
        .map(|_| zeros(width * height))
        .collect::<Result<Vec<_>>>()?;
    let mut dictionary = PatchDictionary {
        references: Vec::with_capacity(info.len()),
        positions: Vec::new(),
        blendings: Vec::new(),
        num_extra_channels,
    };
    for (index, patch) in info.iter().enumerate() {
        let (x0, y0) = atlas_positions[index];
        let area = patch.key.width * patch.key.height;
        for (c, plane) in planes.iter_mut().enumerate().take(color_channels) {
            for y in 0..patch.key.height {
                let src = c * area + y * patch.key.width;
                let dst = (y0 + y) * width + x0;
                plane[dst..dst + patch.key.width]
                    .copy_from_slice(&patch.key.samples[src..src + patch.key.width]);
            }
        }
        dictionary.references.push(PatchReferencePosition {
            reference: REFERENCE_ID,
            x0: u32::try_from(x0).map_err(|_| Error::Limit("source patch atlas x"))?,
            y0: u32::try_from(y0).map_err(|_| Error::Limit("source patch atlas y"))?,
            width: patch.key.width as u32,
            height: patch.key.height as u32,
        });
        for &(x, y) in &patch.positions {
            dictionary.positions.push(PatchPosition {
                x,
                y,
                ref_pos_idx: index,
            });
            dictionary.blendings.push(PatchBlending {
                mode: PatchBlendMode::Add,
                alpha_channel: 0,
                clamp: false,
            });
            for _ in 0..num_extra_channels {
                dictionary.blendings.push(PatchBlending {
                    mode: PatchBlendMode::None,
                    alpha_channel: 0,
                    clamp: false,
                });
            }
        }
    }
    Ok(Some(PackedPatches {
        width,
        height,
        planes,
        dictionary,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::{ReferenceImage, tokenize};
    use super::*;

    fn blank(depth: u32, colors: usize, extras: usize) -> ReferenceImage {
        let mut image = ReferenceImage {
            width: 129,
            height: 83,
            bit_depth: depth,
            color_channels: colors,
            planes: vec![vec![0; 129 * 83]; colors + extras],
        };
        let maximum = (1 << depth) - 1;
        // First extra is zero alpha: all tested color samples are hidden RGB.
        for c in colors + 1..image.planes.len() {
            for (i, v) in image.planes[c].iter_mut().enumerate() {
                *v = (i as i32 * 17 + c as i32) & maximum;
            }
        }
        image
    }

    fn square(image: &mut ReferenceImage, x0: usize, y0: usize, value: i32) {
        for plane in &mut image.planes[..image.color_channels] {
            for y in y0..y0 + 5 {
                plane[y * image.width + x0..y * image.width + x0 + 5].fill(value);
            }
        }
    }

    fn fixture(depth: u32, colors: usize, extras: usize, a: i32, b: i32) -> ReferenceImage {
        let mut image = blank(depth, colors, extras);
        square(&mut image, 17, 21, a);
        square(&mut image, 73, 37, b);
        image
    }

    fn replay(image: &ReferenceImage, plan: &AddPatchPlan) {
        assert_eq!(
            compose_add(&plan.residual, &plan.reference, &plan.dictionary).unwrap(),
            image.planes
        );
        assert_eq!(
            subtract_add(image.view(), &plan.reference, &plan.dictionary).unwrap(),
            plan.residual
        );
        assert_eq!(
            &plan.residual.planes[image.color_channels..],
            &image.planes[image.color_channels..]
        );
        assert!(
            plan.reference.planes[image.color_channels..]
                .iter()
                .flatten()
                .all(|&v| v == 0)
        );
        let stride = 1 + plan.dictionary.num_extra_channels;
        for blends in plan.dictionary.blendings.chunks_exact(stride) {
            assert_eq!(blends[0].mode, PatchBlendMode::Add);
            assert!(blends[1..].iter().all(|b| b.mode == PatchBlendMode::None));
        }
        let tokens = tokenize(&plan.dictionary).unwrap();
        assert!(tokens.iter().all(|t| t.context < 10));
        assert!(
            tokens
                .iter()
                .any(|t| t.context == 5 && t.value == PatchBlendMode::Add as u32)
        );
    }

    #[test]
    fn source_effort_gate_and_integer_metadata_capability_are_separate() {
        assert!(source_enabled(16, -1, 0, true, false));
        assert!(
            SourcePatchConfig {
                speed_tier: -1,
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
        for speed_tier in [i8::MIN, -2, 10, i8::MAX] {
            assert!(
                SourcePatchConfig {
                    speed_tier,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(source_enabled(1, 5, 1, true, false));
        assert!(source_enabled(16, 5, 1, true, false));
        for (bits, speed, decoding, color, groupwise) in [
            (16, -2, 0, true, false),
            (0, 5, 1, true, false),
            (17, 5, 1, true, false),
            (16, 6, 1, true, false),
            (16, 5, 2, true, false),
            (16, 5, 1, false, false),
            (16, 5, 1, true, true),
        ] {
            assert!(!source_enabled(bits, speed, decoding, color, groupwise));
        }
        assert!(source_supported(16, false, &[0, 0, 0]));
        assert!(source_supported(16, true, &[]));
        assert!(!source_supported(4, true, &[]));
        assert!(!source_supported(32, false, &[]));
        assert!(source_supported(12, false, &[0, 1, 2, 3]));
        assert!(!source_supported(12, false, &[0, 4]));
        let image = fixture(8, 3, 2, 200, 200);
        for config in [
            SourcePatchConfig {
                enabled: false,
                ..SourcePatchConfig::default()
            },
            SourcePatchConfig {
                speed_tier: 6,
                ..SourcePatchConfig::default()
            },
            SourcePatchConfig {
                decoding_speed_tier: 2,
                ..SourcePatchConfig::default()
            },
            SourcePatchConfig {
                do_color: false,
                ..SourcePatchConfig::default()
            },
            SourcePatchConfig {
                groupwise: true,
                ..SourcePatchConfig::default()
            },
            SourcePatchConfig {
                floating_point: true,
                ..SourcePatchConfig::default()
            },
        ] {
            assert!(discover_source(image.view(), &config).unwrap().is_none());
        }
        let mismatch = SourcePatchConfig {
            extra_channel_dim_shifts: vec![0],
            ..SourcePatchConfig::default()
        };
        assert!(discover_source(image.view(), &mismatch).is_err());
        let subsampled = SourcePatchConfig {
            extra_channel_dim_shifts: vec![0, 1],
            ..SourcePatchConfig::default()
        };
        // ImageView cannot describe per-plane dimensions; never accept a
        // resampled/full-sized buffer as a native subsampled extra plane.
        assert!(discover_source(image.view(), &subsampled).is_err());
        // Unsupported layouts are rejected before reading/validating raw
        // storage as integer samples. The caller must still avoid conversion.
        let float_bits = vec![vec![0x3f80_0000]];
        let raw = ImageView {
            width: 1,
            height: 1,
            bit_depth: 16,
            color_channels: 1,
            planes: &float_bits,
        };
        assert!(
            discover_source(
                raw,
                &SourcePatchConfig {
                    floating_point: true,
                    ..SourcePatchConfig::default()
                }
            )
            .unwrap()
            .is_none()
        );
        assert!(discover_source(raw, &SourcePatchConfig::default()).is_err());
    }

    #[test]
    fn source_float_metrics_use_signed_truncation_and_nominal_normalization() {
        let u8_space = Colorspace::new(8);
        let u16_space = Colorspace::new(16);
        for (value, expected) in [
            (19, 0),
            (20, 1),
            (39, 1),
            (40, 2),
            (-19, 0),
            (-20, -1),
            (-40, -2),
        ] {
            assert_eq!(quantize(u8_space.normalize([value; 3])[0], 0), expected);
            assert_eq!(
                quantize(u16_space.normalize([value * 257; 3])[2], 2),
                expected
            );
        }
        assert_eq!(quantize(u8_space.normalize([21; 3])[1], 1), 0);
        assert_eq!(quantize(u8_space.normalize([22; 3])[1], 1), 1);
        assert!(same([0.0; 3], u16_space.normalize([6; 3])));
        assert!(!same([0.0; 3], u16_space.normalize([7; 3])));
        assert!(similar([0.0; 3], u8_space.normalize([14; 3]), 0.8));
        assert!(!similar([0.0; 3], u8_space.normalize([15; 3]), 0.8));
        assert!(similar([0.0; 3], u16_space.normalize([142; 3]), 0.03));
        assert!(!similar([0.0; 3], u16_space.normalize([143; 3]), 0.03));
    }

    #[test]
    fn quantization_subtracts_normalized_operands_not_an_integer_delta() {
        // Literal scalar translations of enc_external_image.cc and
        // PatchColorspaceInfo::Quantize, deliberately without using Colorspace.
        fn source_key(value: i32, base: i32, bits: u32, channel: usize) -> i32 {
            let scale = 1.0f32 / ((1u32 << bits) - 1) as f32;
            let value = value as f32 * scale;
            let base = base as f32 * scale;
            let difference = value - base;
            let divisor = if channel == 1 {
                22.0f32 / 255.0
            } else {
                20.0f32 / 255.0
            };
            (difference / divisor).clamp(-32768.0, 32767.0).trunc() as i32
        }
        for (bits, base, value) in [(8, 24, 44), (16, 16, 5156)] {
            let space = Colorspace::new(bits);
            let difference = space.normalize([value; 3])[0] - space.normalize([base; 3])[0];
            assert_eq!(quantize(difference, 0), 0);
            assert_eq!(quantize(-difference, 0), 0);
            let integer_formula =
                i64::from(value - base) * 255 / (i64::from((1u32 << bits) - 1) * 20);
            assert_eq!(
                integer_formula, 1,
                "this vector must distinguish the old integer shortcut"
            );
        }
        for base in 0..=255 {
            for value in 0..=255 {
                for channel in 0..3 {
                    let space = Colorspace::new(8);
                    let delta =
                        space.normalize([value; 3])[channel] - space.normalize([base; 3])[channel];
                    assert_eq!(
                        quantize(delta, channel),
                        source_key(value, base, 8, channel)
                    );
                }
            }
        }
    }

    #[test]
    fn float_boundary_changes_component_acceptance_not_only_the_metric() {
        let mut image = blank(8, 1, 0);
        for row in image.planes[0].chunks_exact_mut(image.width) {
            row[64..].fill(24);
        }
        square(&mut image, 17, 21, 20);
        square(&mut image, 73, 37, 44);
        // Both raw deltas are 20. Source normalized subtraction gives the
        // right-hand glyph a zero peak, so these cannot form a dictionary.
        for fast in [false, true] {
            let config = SourcePatchConfig {
                fast,
                min_peak: 1,
                ..SourcePatchConfig::default()
            };
            assert!(discover_source(image.view(), &config).unwrap().is_none());
            let mut work = Work::default();
            assert!(detect(image.view(), &config, &mut work).unwrap().is_empty());
            assert_eq!(work.low_peak, 1);
        }
        // Moving one sample step off the rounding boundary enables both.
        square(&mut image, 73, 37, 45);
        let config = SourcePatchConfig {
            min_peak: 1,
            ..SourcePatchConfig::default()
        };
        let plan = discover_source(image.view(), &config).unwrap().unwrap();
        assert_eq!(plan.dictionary.positions.len(), 2);
        replay(&image, &plan);
    }

    #[test]
    fn integer_atlas_matches_source_lossless_float_to_int_rounding() {
        for bits in 1..=16 {
            let maximum = (1i32 << bits) - 1;
            let space = Colorspace::new(bits);
            for base in 0..=maximum {
                for value in [
                    0,
                    maximum,
                    base,
                    base.saturating_sub(1).max(0),
                    (base + 1).min(maximum),
                    (base + maximum / 3).min(maximum),
                ] {
                    let delta = space.normalize([value; 3])[0] - space.normalize([base; 3])[0];
                    // enc_modular.cc::float_to_int, bits <= 22, non-float.
                    let rounded =
                        (delta * maximum as f32 + if delta < 0.0 { -0.5 } else { 0.5 }) as i32;
                    assert_eq!(
                        rounded,
                        value - base,
                        "bits={bits} value={value} base={base}"
                    );
                }
            }
        }
    }

    #[test]
    fn source_atlas_packing_keeps_last_row_collision_cursor() {
        let info: Vec<_> = [(8, 7), (6, 5), (2, 9), (2, 9), (4, 3)]
            .into_iter()
            .map(|(width, height)| PatchInfo {
                key: PatchKey {
                    width,
                    height,
                    samples: vec![],
                },
                positions: vec![],
            })
            .collect();
        assert_eq!(
            pack_source(&info, 169).unwrap(),
            Some((13, 12, vec![(0, 0), (0, 7), (9, 0), (11, 0), (6, 9)]))
        );
        // A first-collision shortcut incorrectly fits these into 12x12.
        assert!(pack_source(&info, 160).unwrap().is_none());
    }

    #[test]
    fn all_integer_depths_gray_rgb_hidden_color_and_full_resolution_extras_replay() {
        for depth in 1..=16 {
            for colors in [1, 3] {
                let maximum = (1 << depth) - 1;
                let image = fixture(depth, colors, 3, maximum, maximum);
                let config = SourcePatchConfig {
                    extra_channel_dim_shifts: vec![0; 3],
                    ..SourcePatchConfig::default()
                };
                let plan = discover_source(image.view(), &config).unwrap().unwrap();
                assert_eq!(plan.dictionary.references.len(), 1);
                assert_eq!(plan.dictionary.positions.len(), 2);
                assert_eq!(plan.dictionary.num_extra_channels, 3);
                replay(&image, &plan);
            }
        }
    }

    #[test]
    fn background_relative_fingerprints_share_across_different_backgrounds() {
        let mut image = blank(8, 3, 2);
        for plane in &mut image.planes[..3] {
            for y in 0..image.height {
                plane[y * image.width..y * image.width + 64].fill(200);
                plane[y * image.width + 64..(y + 1) * image.width].fill(150);
            }
        }
        square(&mut image, 17, 21, 140);
        square(&mut image, 73, 37, 86);
        let plan = discover_source(image.view(), &SourcePatchConfig::default())
            .unwrap()
            .unwrap();
        assert_eq!(plan.dictionary.references.len(), 1);
        assert!(plan.reference.planes[0].contains(&-60));
        assert_eq!(plan.residual.planes[0][37 * image.width + 73], 146);
        assert!(plan.requires_signed_samples());
        replay(&image, &plan);
    }

    #[test]
    fn coarsened_fingerprint_preserves_negative_occurrence_residuals() {
        let image = fixture(8, 3, 1, 64, 44);
        assert!(
            discover_source(image.view(), &SourcePatchConfig::default())
                .unwrap()
                .is_none()
        );
        for fingerprint_divisor in [2, 4] {
            let config = SourcePatchConfig {
                fingerprint_divisor,
                ..SourcePatchConfig::default()
            };
            let plan = discover_source(image.view(), &config).unwrap().unwrap();
            assert!(plan.reference.planes[0].contains(&64));
            assert_eq!(plan.residual.planes[0][37 * image.width + 73], -20);
            assert!(plan.residual.requires_signed_samples());
            replay(&image, &plan);
        }
    }

    #[test]
    fn representative_follows_source_xy_sort_not_scan_yx_order() {
        let mut image = blank(8, 1, 0);
        square(&mut image, 73, 13, 44);
        square(&mut image, 17, 41, 64);
        let config = SourcePatchConfig {
            fingerprint_divisor: 2,
            ..SourcePatchConfig::default()
        };
        let plan = discover_source(image.view(), &config).unwrap().unwrap();
        assert_eq!(
            (
                plan.dictionary.positions[0].x,
                plan.dictionary.positions[0].y
            ),
            (17, 41)
        );
        assert!(plan.reference.planes[0].contains(&64));
        replay(&image, &plan);
    }

    #[test]
    fn low_contrast_setting_changes_detection_without_quantizing_samples() {
        let image = fixture(8, 1, 0, 30, 31);
        assert!(
            discover_source(image.view(), &SourcePatchConfig::default())
                .unwrap()
                .is_none()
        );
        let config = SourcePatchConfig {
            min_peak: 1,
            ..SourcePatchConfig::default()
        };
        let plan = discover_source(image.view(), &config).unwrap().unwrap();
        assert!(plan.reference.planes[0].contains(&30));
        assert_eq!(plan.residual.planes[0][37 * image.width + 73], 1);
        replay(&image, &plan);
    }

    #[test]
    fn fast_scan_prunes_seed_interiors_and_visits_each_component_once() {
        let image = fixture(8, 3, 0, 200, 200);
        let mut work_fast = Work::default();
        let mut work_slow = Work::default();
        let fast = SourcePatchConfig::default();
        let slow = SourcePatchConfig {
            fast: false,
            ..SourcePatchConfig::default()
        };
        let a = detect(image.view(), &fast, &mut work_fast).unwrap();
        let b = detect(image.view(), &slow, &mut work_slow).unwrap();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.key, b.key);
            assert_eq!(a.positions, b.positions);
        }
        assert_eq!(work_fast.component_pushes, 50);
        assert!(work_slow.component_pushes > work_fast.component_pushes);
        assert!(work_fast.initial_queue * 4 < work_slow.initial_queue);
    }

    #[test]
    fn source_tolerances_preserve_nearly_flat_16_bit_background_and_tails() {
        let mut image = blank(16, 3, 1);
        for plane in &mut image.planes[..3] {
            for (i, value) in plane.iter_mut().enumerate() {
                *value = 2000 + (i % 5) as i32;
            }
        }
        square(&mut image, 0, 0, 18000);
        square(&mut image, 124, 78, 18003);
        let config = SourcePatchConfig {
            fingerprint_divisor: 2,
            ..SourcePatchConfig::default()
        };
        let plan = discover_source(image.view(), &config).unwrap().unwrap();
        replay(&image, &plan);
        for side in 1..12 {
            let planes = vec![vec![0; side * side]];
            let view = ImageView {
                width: side,
                height: side,
                bit_depth: 1,
                color_channels: 1,
                planes: &planes,
            };
            assert!(discover_source(view, &config).unwrap().is_none());
        }
    }

    #[test]
    fn signed_add_replay_handles_overlap_and_rejects_overflow_or_extra_blending() {
        let image = ReferenceImage {
            width: 1,
            height: 1,
            bit_depth: 16,
            color_channels: 1,
            planes: vec![vec![0], vec![65535]],
        };
        let reference = SignedPatchImage {
            width: 1,
            height: 1,
            bit_depth: 16,
            color_channels: 1,
            planes: vec![vec![-65535], vec![0]],
        };
        let mut dictionary = PatchDictionary {
            references: vec![PatchReferencePosition {
                reference: REFERENCE_ID,
                x0: 0,
                y0: 0,
                width: 1,
                height: 1,
            }],
            positions: vec![
                PatchPosition {
                    x: 0,
                    y: 0,
                    ref_pos_idx: 0
                };
                3
            ],
            blendings: [
                PatchBlending {
                    mode: PatchBlendMode::Add,
                    alpha_channel: 0,
                    clamp: false,
                },
                PatchBlending {
                    mode: PatchBlendMode::None,
                    alpha_channel: 0,
                    clamp: false,
                },
            ]
            .repeat(3),
            num_extra_channels: 1,
        };
        let residual = subtract_add(image.view(), &reference, &dictionary).unwrap();
        assert_eq!(residual.planes[0], [196605]);
        assert_eq!(
            compose_add(&residual, &reference, &dictionary).unwrap(),
            image.planes
        );
        let mut bad = reference.clone();
        bad.planes[0][0] = i32::MIN;
        assert!(subtract_add(image.view(), &bad, &dictionary).is_err());
        dictionary.blendings[1].mode = PatchBlendMode::Add;
        assert!(subtract_add(image.view(), &reference, &dictionary).is_err());
        dictionary.blendings[1].mode = PatchBlendMode::None;
        dictionary.positions[0].x = 1;
        assert!(subtract_add(image.view(), &reference, &dictionary).is_err());
    }

    #[test]
    fn source_limits_are_explicit_and_no_candidate_is_a_valid_result() {
        let image = fixture(8, 3, 0, 200, 200);
        for (min_peak, fingerprint_divisor) in [(0, 1), (3, 1), (1, 0), (1, 3)] {
            let config = SourcePatchConfig {
                min_peak,
                fingerprint_divisor,
                ..SourcePatchConfig::default()
            };
            assert!(discover_source(image.view(), &config).is_err());
        }
        for limits in [
            PatchOptions {
                max_patch_size: 4,
                ..PatchOptions::default()
            },
            PatchOptions {
                min_occurrences: 3,
                ..PatchOptions::default()
            },
            PatchOptions {
                max_reference_pixels: 24,
                ..PatchOptions::default()
            },
            PatchOptions {
                max_positions: 1,
                ..PatchOptions::default()
            },
            PatchOptions {
                min_max_patch_area: 26,
                ..PatchOptions::default()
            },
        ] {
            let config = SourcePatchConfig {
                limits,
                ..SourcePatchConfig::default()
            };
            assert!(discover_source(image.view(), &config).unwrap().is_none());
        }
    }
}
