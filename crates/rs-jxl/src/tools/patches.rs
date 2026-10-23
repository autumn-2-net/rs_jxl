// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; the complete upstream notice and license
// are retained in docs/PORTING_KERNELS.md.

//! Integer-lossless specialization of enc_patch_dictionary.cc/.h at frozen
//! b866b3defec13494bfbaf705db93581cda118e37: FindTextLikePatches' flat 4x4
//! seeds, boundary-only background BFS, small 8-connected components, exact
//! rectangle deduplication, FindBestPatchDictionary's first-fit atlas packing,
//! and PatchDictionaryEncoder::Encode/SubtractFrom. Wire constants and checks
//! come from patch_dictionary_internal.h and dec_patch_dictionary.cc/.h.
//!
//! discover() is a deliberate lossless specialization: compare integer samples in
//! ALL supplied planes, store absolute samples, and use Replace for each one.
//! The metadata-aware frame adapter supplies full-resolution 1..16-bit integer
//! extras with color; other extras pass through with None and native sampling.
//! No int8 quantization, float tolerances, background-relative Add patches,
//! alpha blending, alpha cleanup, or omission of invisible RGB is performed.
//! This finds repeated component rectangles, not arbitrary tiled textures.
//! Lossy/XYB discovery, approximate grouping, dots, general blend arithmetic,
//! parallel detection, encoder reference-frame serialization, and frame-size
//! cost trials remain unported. A discovered plan is not a size-win guarantee.
//!
//! The source's JXL_LOSSLESS_PATCH_FAST path is used unconditionally: no-seed
//! early exit, bulk seed marking, 3x3 interior-seed pruning, boundary-only BFS,
//! mark-on-push component traversal, no debug component pixel lists, skipped
//! border comparisons after rejection, and tightly sized rectangle storage.
//! The 64/128 maximum-side experiments are available in PatchOptions. Exact
//! ordered deduplication moves each retained rectangle once instead of storing
//! one full rectangle per occurrence and copying during sort/compaction.
//! VecDeque drops consumed BFS entries instead of retaining a full-frame queue.
//! discover() keeps one-unit contrast and compares every bit. Its Replace
//! specialization does not quantize keys or subtract a background. The separate
//! discover_source() below implements min_peak / fingerprint_divisor and Add.
//!
//! Integration order (enc_modular.cc, enc_frame.cc, dec_frame.cc/dec_cache.cc):
//! 1. Discover on full-frame, untransformed integer color/extra planes, BEFORE
//!    RCT/Squeeze, group splitting, prediction, residual packing, or MA training.
//! 2. Encode plan.reference as a separate lossless Modular ReferenceOnly frame
//!    (frame type 2), save_as_reference=3, save_before_color_transform=true.
//!    Use atlas dimensions, the SAME bit depth/color/extra-channel metadata,
//!    real atlas planes for replaced extras, zero placeholders for None extras,
//!    original metadata sampling, and no patches/dots/noise/alpha cleanup.
//!    Keep the codestream canvas unchanged; set custom_size_or_origin and the
//!    atlas frame_size. ReferenceOnly omits passes, origins, frame blending,
//!    animation timing and is_last (implicitly false), but writes reference ID
//!    and save-before-transform. Both frames use color_transform=None, not XYB.
//!    Reversible Modular transforms are fine; the saved reconstructed samples
//!    must equal the atlas. Emit this frame BEFORE the visible residual frame.
//! 3. Set the visible frame's kPatches flag (mask 2). Encode tokenize() as one
//!    ordinary entropy stream with NUM_CONTEXTS=10, no LZ77 required. Write its
//!    histogram header and tokens FIRST in DCGlobal, before DC dequantization
//!    and global Modular/tree data, NOT once per pixel group.
//! 4. Encode plan.residual with the normal full-frame/group Modular pipeline.
//!    Decoder patches run AFTER inverse Modular transforms and BEFORE the
//!    output color transform/frame blending. Gray references are expanded to
//!    decoder color planes normally; one blend entry still controls all color.
//! 5. Keep the atlas in reference slot 3 until this frame has finished. Do not
//!    save it post-color-transform: the decoder explicitly rejects that source.
//!    Compare complete output samples including RGB at alpha zero with djxl.
//!
//! compose()/subtract() are integer reference helpers, not a runtime decoder.
//! Replace copies bit-exact samples; no alpha normalization or rounding occurs.
//!
//! discover_source() separately ports the non-XYB background-relative detector,
//! quantized fingerprints, scan routing, and Add dictionary construction. Its
//! signed integer atlas/residual must NOT be passed through an unsigned-only
//! frame path. See patch_source.rs for numerical adaptations and integration.

use std::collections::{BTreeMap, VecDeque};

use super::lz77::Literal;
use crate::{Error, Result};

#[path = "patch_source.rs"]
mod source;
pub use source::{
    AddPatchPlan, FloatImageView, FloatPatchImage, FloatPatchPlan, SignedPatchImage,
    SourcePatchConfig, compose_add, compose_float_add, discover_source, discover_source_float,
    source_enabled, source_supported, subtract_add, verify_add_replay,
};

pub const REFERENCE_ID: u32 = 3;
pub const NUM_CONTEXTS: usize = 10;
const SEED_SIDE: usize = 4;
const BACKGROUND_DISTANCE: usize = 50;

/// Logical untransformed gray/RGB followed by arbitrary full-resolution extras.
/// All planes use the declared unsigned 1..=16-bit range. Differently sampled
/// or differently declared-depth extras need a separate metadata-aware API.
/// Never construct this view by reinterpreting float sample bit patterns.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub color_channels: usize,
    pub planes: &'a [Vec<i32>],
}

impl ImageView<'_> {
    fn validate(&self) -> Result<usize> {
        if self.width == 0
            || self.height == 0
            || self.width > i32::MAX as usize
            || self.height > i32::MAX as usize
        {
            return Err(invalid(
                "patch image dimensions must be positive and fit i32",
            ));
        }
        if !(1..=16).contains(&self.bit_depth)
            || !matches!(self.color_channels, 1 | 3)
            || self.planes.len() < self.color_channels
        {
            return Err(invalid(
                "patch images require 1..=16-bit gray/RGB and full-resolution extras",
            ));
        }
        let pixels = self
            .width
            .checked_mul(self.height)
            .ok_or(Error::Limit("patch image pixels"))?;
        let maximum = (1i32 << self.bit_depth) - 1;
        if self.planes.iter().any(|plane| {
            plane.len() != pixels || plane.iter().any(|&v| !(0..=maximum).contains(&v))
        }) {
            return Err(invalid("patch plane size or sample range is invalid"));
        }
        Ok(pixels)
    }

    fn same_pixel(&self, a: usize, b: usize) -> bool {
        self.planes.iter().all(|plane| plane[a] == plane[b])
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceImage {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub color_channels: usize,
    pub planes: Vec<Vec<i32>>,
}

impl ReferenceImage {
    pub fn view(&self) -> ImageView<'_> {
        ImageView {
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
            color_channels: self.color_channels,
            planes: &self.planes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PatchOptions {
    /// Source default 32, with its lossless 64/128 experiments available.
    pub max_patch_size: usize,
    pub min_occurrences: usize,
    /// As in C++, discard the dictionary if its largest repeated patch is
    /// smaller than this; smaller patches may accompany a larger one.
    pub min_max_patch_area: usize,
    pub max_unique_patches: usize,
    /// Maximum stored occurrences including singleton candidates.
    pub max_positions: usize,
    /// Atlas packing stops with None if this pixel budget would be exceeded.
    pub max_reference_pixels: usize,
}

impl Default for PatchOptions {
    fn default() -> Self {
        Self {
            max_patch_size: 32,
            min_occurrences: 2,
            min_max_patch_area: 20,
            max_unique_patches: 4096,
            max_positions: 65_536,
            max_reference_pixels: 1 << 22,
        }
    }
}

impl PatchOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=128).contains(&self.max_patch_size)
            || self.min_occurrences < 2
            || self.max_unique_patches == 0
            || self.max_positions == 0
            || self.max_reference_pixels == 0
        {
            return Err(invalid("invalid patch discovery limits"));
        }
        Ok(())
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchBlendMode {
    None = 0,
    Replace = 1,
    Add = 2,
    Mul = 3,
    BlendAbove = 4,
    BlendBelow = 5,
    AlphaWeightedAddAbove = 6,
    AlphaWeightedAddBelow = 7,
}

impl PatchBlendMode {
    fn uses_alpha(self) -> bool {
        matches!(
            self,
            Self::BlendAbove
                | Self::BlendBelow
                | Self::AlphaWeightedAddAbove
                | Self::AlphaWeightedAddBelow
        )
    }

    fn uses_clamp(self) -> bool {
        self == Self::Mul || self.uses_alpha()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PatchBlending {
    pub mode: PatchBlendMode,
    pub alpha_channel: u32,
    pub clamp: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PatchReferencePosition {
    pub reference: u32,
    pub x0: u32,
    pub y0: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PatchPosition {
    pub x: u32,
    pub y: u32,
    pub ref_pos_idx: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchDictionary {
    pub references: Vec<PatchReferencePosition>,
    /// Wire order is preserved, including contiguous runs of reference IDs.
    pub positions: Vec<PatchPosition>,
    /// Position-major: one color blend, then one per extra channel.
    pub blendings: Vec<PatchBlending>,
    pub num_extra_channels: usize,
}

impl PatchDictionary {
    pub(crate) fn validate(&self) -> Result<usize> {
        let stride = self
            .num_extra_channels
            .checked_add(1)
            .ok_or(Error::Limit("patch blending stride"))?;
        let count = self
            .positions
            .len()
            .checked_mul(stride)
            .ok_or(Error::Limit("patch blendings"))?;
        if count != self.blendings.len() {
            return Err(invalid("patch blending count does not match positions"));
        }
        for reference in &self.references {
            if reference.reference >= 4 || reference.width == 0 || reference.height == 0 {
                return Err(invalid("invalid patch reference ID or dimensions"));
            }
        }
        if self
            .positions
            .iter()
            .any(|pos| pos.ref_pos_idx >= self.references.len())
        {
            return Err(invalid("patch position references a missing rectangle"));
        }
        for blend in &self.blendings {
            if blend.mode.uses_alpha() && blend.alpha_channel as usize >= self.num_extra_channels {
                return Err(invalid("patch alpha channel is out of bounds"));
            }
        }
        Ok(stride)
    }
}

#[derive(Clone, Debug)]
pub struct PatchPlan {
    pub reference: ReferenceImage,
    pub dictionary: PatchDictionary,
    /// Same shape and plane order as the original full image. Covered values
    /// are zero; the reference frame supplies them with Replace at decode time.
    pub residual: Vec<Vec<i32>>,
}

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

fn zeros<T: Clone + Default>(count: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("patch allocation"))?;
    result.resize(count, T::default());
    Ok(result)
}

fn token(tokens: &mut Vec<Literal>, context: u32, value: usize) -> Result<()> {
    let value = u32::try_from(value).map_err(|_| Error::Limit("patch token exceeds u32"))?;
    tokens.push(Literal { context, value });
    Ok(())
}

fn offset(current: u32, previous: u32) -> Result<u32> {
    let signed = i64::from(current) - i64::from(previous);
    let signed = i32::try_from(signed).map_err(|_| Error::Limit("patch offset exceeds i32"))?;
    // pack_signed.h::PackSigned, including i32::MIN, without signed overflow.
    Ok(((signed as u32) << 1) ^ ((signed >> 31) as u32))
}

/// PatchDictionaryEncoder::Encode's exact token order and ten context IDs.
/// General blend modes can be tokenized; integer composition below deliberately
/// supports only None/Replace. Histograms and bitstream emission belong to main.
pub fn tokenize(dictionary: &PatchDictionary) -> Result<Vec<Literal>> {
    let stride = dictionary.validate()?;
    let mut tokens = Vec::new();
    let mut runs = 0;
    let mut previous = None;
    for position in &dictionary.positions {
        if previous != Some(position.ref_pos_idx) {
            runs += 1;
        }
        previous = Some(position.ref_pos_idx);
    }
    token(&mut tokens, 0, runs)?;
    let mut first = 0;
    while first < dictionary.positions.len() {
        let reference_idx = dictionary.positions[first].ref_pos_idx;
        let reference = dictionary.references[reference_idx];
        let mut end = first + 1;
        while end < dictionary.positions.len()
            && dictionary.positions[end].ref_pos_idx == reference_idx
        {
            end += 1;
        }
        token(&mut tokens, 1, reference.reference as usize)?;
        token(&mut tokens, 3, reference.x0 as usize)?;
        token(&mut tokens, 3, reference.y0 as usize)?;
        token(&mut tokens, 2, reference.width as usize - 1)?;
        token(&mut tokens, 2, reference.height as usize - 1)?;
        token(&mut tokens, 7, end - first - 1)?;
        for index in first..end {
            let position = dictionary.positions[index];
            if index == first {
                token(&mut tokens, 4, position.x as usize)?;
                token(&mut tokens, 4, position.y as usize)?;
            } else {
                let previous = dictionary.positions[index - 1];
                token(&mut tokens, 6, offset(position.x, previous.x)? as usize)?;
                token(&mut tokens, 6, offset(position.y, previous.y)? as usize)?;
            }
            for blending in &dictionary.blendings[index * stride..(index + 1) * stride] {
                token(&mut tokens, 5, blending.mode as usize)?;
                if blending.mode.uses_alpha() && dictionary.num_extra_channels > 1 {
                    token(&mut tokens, 8, blending.alpha_channel as usize)?;
                }
                if blending.mode.uses_clamp() {
                    token(&mut tokens, 9, usize::from(blending.clamp))?;
                }
            }
        }
        first = end;
    }
    Ok(tokens)
}

fn validate_composition(
    image: ImageView<'_>,
    reference: ImageView<'_>,
    dictionary: &PatchDictionary,
) -> Result<usize> {
    let pixels = image.validate()?;
    reference.validate()?;
    let stride = dictionary.validate()?;
    if reference.bit_depth != image.bit_depth
        || reference.color_channels != image.color_channels
        || reference.planes.len() != image.planes.len()
        || dictionary.num_extra_channels != image.planes.len() - image.color_channels
    {
        return Err(invalid("patch reference and image metadata differ"));
    }
    if dictionary
        .blendings
        .iter()
        .any(|b| !matches!(b.mode, PatchBlendMode::None | PatchBlendMode::Replace))
    {
        return Err(invalid(
            "integer patch composition supports only None and Replace",
        ));
    }
    let max_refs = 1024usize
        .checked_add(pixels / 4)
        .ok_or(Error::Limit("patch reference count"))?;
    let max_patches = max_refs
        .checked_mul(4)
        .ok_or(Error::Limit("patch position count"))?;
    let mut runs = 0;
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
            return Err(invalid("patch crop is outside its image or reference slot"));
        }
    }
    if runs > max_refs || dictionary.positions.len() > max_patches {
        return Err(Error::Limit(
            "patch dictionary exceeds decoder count limits",
        ));
    }
    Ok(stride)
}

fn visit_samples(
    image: ImageView<'_>,
    reference: ImageView<'_>,
    dictionary: &PatchDictionary,
    stride: usize,
    mut visit: impl FnMut(usize, usize, usize) -> Result<()>,
) -> Result<()> {
    for (index, position) in dictionary.positions.iter().enumerate() {
        let crop = dictionary.references[position.ref_pos_idx];
        for channel in 0..image.planes.len() {
            let blend = if channel < image.color_channels {
                0
            } else {
                channel - image.color_channels + 1
            };
            if dictionary.blendings[index * stride + blend].mode == PatchBlendMode::None {
                continue;
            }
            for y in 0..crop.height as usize {
                let destination = (position.y as usize + y) * image.width + position.x as usize;
                let source = (crop.y0 as usize + y) * reference.width + crop.x0 as usize;
                for x in 0..crop.width as usize {
                    visit(channel, destination + x, source + x)?;
                }
            }
        }
    }
    Ok(())
}

/// Clear only exact Replace copies; all validation happens on immutable inputs.
/// None leaves the channel untouched. Approximate matches are rejected rather
/// than discarding a difference. Overlapping exact patches are safe.
pub fn subtract(
    image: ImageView<'_>,
    reference: ImageView<'_>,
    dictionary: &PatchDictionary,
) -> Result<Vec<Vec<i32>>> {
    let stride = validate_composition(image, reference, dictionary)?;
    visit_samples(
        image,
        reference,
        dictionary,
        stride,
        |channel, destination, source| {
            if image.planes[channel][destination] != reference.planes[channel][source] {
                return Err(invalid(
                    "Replace patch does not exactly match original samples",
                ));
            }
            Ok(())
        },
    )?;
    let mut residual = image.planes.to_vec();
    visit_samples(
        image,
        reference,
        dictionary,
        stride,
        |channel, destination, _| {
            residual[channel][destination] = 0;
            Ok(())
        },
    )?;
    Ok(residual)
}

/// Recompose in dictionary order without arithmetic, normalization, or alpha
/// blending. This is an integer oracle for the discovery/erasure operation.
pub fn compose(
    image: ImageView<'_>,
    reference: ImageView<'_>,
    dictionary: &PatchDictionary,
) -> Result<Vec<Vec<i32>>> {
    let stride = validate_composition(image, reference, dictionary)?;
    let mut output = image.planes.to_vec();
    visit_samples(
        image,
        reference,
        dictionary,
        stride,
        |channel, destination, source| {
            output[channel][destination] = reference.planes[channel][source];
            Ok(())
        },
    )?;
    Ok(output)
}

fn neighbors(width: usize, height: usize, position: usize, mut visit: impl FnMut(usize)) {
    let (x, y) = (position % width, position / width);
    // Preserve C++'s dx-outer, dy-inner ordering for flood-fill ties.
    for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
        for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            let next = ny * width + nx;
            if next != position {
                visit(next);
            }
        }
    }
}

#[derive(Default)]
struct DetectionWork {
    #[cfg(test)]
    seed_pixels: usize,
    #[cfg(test)]
    initial_queue: usize,
    #[cfg(test)]
    flood_enqueues: usize,
    #[cfg(test)]
    component_pushes: usize,
}

fn background(
    image: ImageView<'_>,
    pixels: usize,
    _work: &mut DetectionWork,
) -> Result<Option<Vec<u8>>> {
    let (pw, ph) = (image.width / SEED_SIDE, image.height / SEED_SIDE);
    if pw < 3 || ph < 3 {
        return Ok(None);
    }
    let mut seeds = zeros::<u8>(pw * ph)?;
    let mut count = 0;
    for py in 1..ph - 1 {
        for px in 1..pw - 1 {
            let origin = py * SEED_SIDE * image.width + px * SEED_SIDE;
            let flat = (0..SEED_SIDE).all(|y| {
                (0..SEED_SIDE).all(|x| image.same_pixel(origin, origin + y * image.width + x))
            });
            if !flat {
                continue;
            }
            let mut same = 0;
            for y in py - 1..=py + 1 {
                for x in px - 1..=px + 1 {
                    same += usize::from(
                        image.same_pixel(origin, y * SEED_SIDE * image.width + x * SEED_SIDE),
                    );
                }
            }
            if same >= 8 {
                seeds[py * pw + px] = 1;
                count += 1;
            }
        }
    }
    if count == 0 {
        return Ok(None);
    }
    #[cfg(test)]
    {
        _work.seed_pixels = count * SEED_SIDE * SEED_SIDE;
    }
    let mut marked = zeros::<u8>(pixels)?;
    for py in 1..ph - 1 {
        for px in 1..pw - 1 {
            if seeds[py * pw + px] == 0 {
                continue;
            }
            for y in py * SEED_SIDE..(py + 1) * SEED_SIDE {
                marked[y * image.width + px * SEED_SIDE..y * image.width + (px + 1) * SEED_SIDE]
                    .fill(1);
            }
        }
    }
    let mut queue = VecDeque::new();
    // Match the optimized source's bounded initial reservation, not its old
    // two-entries-per-seed allocation on solid images.
    queue
        .try_reserve((count * SEED_SIDE * SEED_SIDE).min(65_536))
        .map_err(|_| Error::Limit("patch background queue"))?;
    for py in 1..ph - 1 {
        for px in 1..pw - 1 {
            if seeds[py * pw + px] == 0 {
                continue;
            }
            let interior =
                (py - 1..=py + 1).all(|y| (px - 1..=px + 1).all(|x| seeds[y * pw + x] != 0));
            if interior {
                continue;
            }
            for y in py * SEED_SIDE..(py + 1) * SEED_SIDE {
                for x in px * SEED_SIDE..(px + 1) * SEED_SIDE {
                    let position = y * image.width + x;
                    let mut interior = true;
                    neighbors(image.width, image.height, position, |next| {
                        interior &= marked[next] != 0
                    });
                    if !interior {
                        queue.push_back((position, position));
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
            if marked[next] != 0 {
                return;
            }
            let distance = (source % image.width).abs_diff(next % image.width)
                + (source / image.width).abs_diff(next / image.width);
            if distance <= BACKGROUND_DISTANCE && image.same_pixel(source, next) {
                marked[next] = 1;
                queue.push_back((next, source));
                #[cfg(test)]
                {
                    _work.flood_enqueues += 1;
                }
            }
        });
    }
    Ok(Some(marked))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PatchKey {
    width: usize,
    height: usize,
    // Exact plane-major samples, including alpha and invisible color samples.
    samples: Vec<i32>,
}

struct PatchInfo {
    key: PatchKey,
    positions: Vec<(u32, u32)>,
}

// Atlas width, height, and one packed origin per unique patch.
type AtlasPacking = (usize, usize, Vec<(usize, usize)>);

fn find_patches(
    image: ImageView<'_>,
    options: &PatchOptions,
    pixels: usize,
) -> Result<Vec<PatchInfo>> {
    find_patches_with_work(image, options, pixels, &mut DetectionWork::default())
}

fn find_patches_with_work(
    image: ImageView<'_>,
    options: &PatchOptions,
    pixels: usize,
    _work: &mut DetectionWork,
) -> Result<Vec<PatchInfo>> {
    let Some(background) = background(image, pixels, _work)? else {
        return Ok(Vec::new());
    };
    let mut visited = zeros::<u8>(pixels)?;
    let mut stack = Vec::new();
    let mut groups = BTreeMap::<PatchKey, Vec<(u32, u32)>>::new();
    let mut positions = 0;
    for start in 0..pixels {
        if background[start] != 0 || visited[start] != 0 {
            continue;
        }
        let (mut min_x, mut min_y) = (start % image.width, start / image.width);
        let (mut max_x, mut max_y) = (min_x, min_y);
        let mut border = None;
        let mut same_border = true;
        stack.push(start);
        visited[start] = 1;
        #[cfg(test)]
        {
            _work.component_pushes += 1;
        }
        while let Some(position) = stack.pop() {
            let (x, y) = (position % image.width, position / image.width);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            neighbors(image.width, image.height, position, |next| {
                if background[next] == 0 {
                    if visited[next] == 0 {
                        visited[next] = 1;
                        stack.push(next);
                        #[cfg(test)]
                        {
                            _work.component_pushes += 1;
                        }
                    }
                } else if same_border
                    && max_x - min_x < options.max_patch_size
                    && max_y - min_y < options.max_patch_size
                {
                    if let Some(reference) = border {
                        same_border &= image.same_pixel(reference, next);
                    } else {
                        border = Some(next);
                    }
                }
            });
        }
        if max_x - min_x >= options.max_patch_size
            || max_y - min_y >= options.max_patch_size
            || border.is_none()
            || !same_border
            || positions >= options.max_positions
        {
            continue;
        }
        // Exact border equality guarantees C++'s radius-two has_similar test:
        // the recorded border pixel is adjacent to this component's rectangle.
        let (width, height) = (max_x - min_x + 1, max_y - min_y + 1);
        let mut samples = Vec::with_capacity(width * height * image.planes.len());
        for plane in image.planes {
            for y in min_y..=max_y {
                samples.extend_from_slice(
                    &plane[y * image.width + min_x..y * image.width + max_x + 1],
                );
            }
        }
        let key = PatchKey {
            width,
            height,
            samples,
        };
        if let Some(occurrences) = groups.get_mut(&key) {
            occurrences.push((min_x as u32, min_y as u32));
            positions += 1;
        } else if groups.len() < options.max_unique_patches {
            groups.insert(key, vec![(min_x as u32, min_y as u32)]);
            positions += 1;
        }
    }
    let mut info: Vec<_> = groups
        .into_iter()
        .filter(|(_, positions)| positions.len() >= options.min_occurrences)
        .map(|(key, positions)| PatchInfo { key, positions })
        .collect();
    if info
        .iter()
        .map(|patch| patch.key.width * patch.key.height)
        .max()
        .unwrap_or(0)
        < options.min_max_patch_area
    {
        info.clear();
    }
    info.sort_by(|a, b| {
        (b.key.width * b.key.height)
            .cmp(&(a.key.width * a.key.height))
            .then(a.key.cmp(&b.key))
    });
    Ok(info)
}

fn pack(info: &[PatchInfo], budget: usize) -> Result<Option<AtlasPacking>> {
    let total = info
        .iter()
        .try_fold(0usize, |sum, patch| {
            sum.checked_add(patch.key.width * patch.key.height)
        })
        .ok_or(Error::Limit("patch atlas area"))?;
    if total > budget {
        return Ok(None);
    }
    let mut width = total
        .isqrt()
        .max(info.iter().map(|p| p.key.width).max().unwrap_or(0));
    let mut height = total
        .isqrt()
        .max(info.iter().map(|p| p.key.height).max().unwrap_or(0));
    loop {
        // Integer equivalent of the source's 1.05 slackness plus one pixel.
        width = width
            .checked_add(width / 20 + 1)
            .ok_or(Error::Limit("patch atlas width"))?;
        height = height
            .checked_add(height / 20 + 1)
            .ok_or(Error::Limit("patch atlas height"))?;
        let Some(area) = width.checked_mul(height).filter(|&area| area <= budget) else {
            return Ok(None);
        };
        let mut occupied = zeros::<u8>(area)?;
        let mut positions = Vec::with_capacity(info.len());
        let mut max_y = 0;
        for patch in info {
            let (w, h) = (patch.key.width, patch.key.height);
            let mut found = None;
            'search: for y in 0..=height - h {
                let mut x = 0;
                while x <= width - w {
                    let mut collision = None;
                    'occupied: for row in y..y + h {
                        for col in x..x + w {
                            if occupied[row * width + col] != 0 {
                                collision = Some(col);
                                break 'occupied;
                            }
                        }
                    }
                    if let Some(col) = collision {
                        x = col + 1;
                    } else {
                        found = Some((x, y));
                        break 'search;
                    }
                }
            }
            let Some((x, y)) = found else { break };
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

/// Discover exact repeated foreground rectangles and build a slot-3 atlas.
/// None means no eligible dictionary or an atlas exceeding the chosen budget.
/// The input is immutable; patches are removed only from a returned copy.
pub fn discover(image: ImageView<'_>, options: &PatchOptions) -> Result<Option<PatchPlan>> {
    options.validate()?;
    let pixels = image.validate()?;
    let info = find_patches(image, options, pixels)?;
    if info.is_empty() {
        return Ok(None);
    }
    let Some((width, height, atlas_positions)) = pack(&info, options.max_reference_pixels)? else {
        return Ok(None);
    };
    let mut reference = ReferenceImage {
        width,
        height,
        bit_depth: image.bit_depth,
        color_channels: image.color_channels,
        planes: (0..image.planes.len())
            .map(|_| zeros::<i32>(width * height))
            .collect::<Result<_>>()?,
    };
    let num_extra_channels = image.planes.len() - image.color_channels;
    let mut dictionary = PatchDictionary {
        references: Vec::with_capacity(info.len()),
        positions: Vec::new(),
        blendings: Vec::new(),
        num_extra_channels,
    };
    for (index, patch) in info.iter().enumerate() {
        let (x, y) = atlas_positions[index];
        let area = patch.key.width * patch.key.height;
        for (channel, plane) in reference.planes.iter_mut().enumerate() {
            for row in 0..patch.key.height {
                let source = channel * area + row * patch.key.width;
                let destination = (y + row) * width + x;
                plane[destination..destination + patch.key.width]
                    .copy_from_slice(&patch.key.samples[source..source + patch.key.width]);
            }
        }
        dictionary.references.push(PatchReferencePosition {
            reference: REFERENCE_ID,
            x0: u32::try_from(x).map_err(|_| Error::Limit("patch atlas x"))?,
            y0: u32::try_from(y).map_err(|_| Error::Limit("patch atlas y"))?,
            width: patch.key.width as u32,
            height: patch.key.height as u32,
        });
        for &(x, y) in &patch.positions {
            dictionary.positions.push(PatchPosition {
                x,
                y,
                ref_pos_idx: index,
            });
            for _ in 0..=num_extra_channels {
                dictionary.blendings.push(PatchBlending {
                    mode: PatchBlendMode::Replace,
                    alpha_channel: 0,
                    clamp: false,
                });
            }
        }
    }
    let residual = subtract(image, reference.view(), &dictionary)?;
    Ok(Some(PatchPlan {
        reference,
        dictionary,
        residual,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blend(mode: PatchBlendMode) -> PatchBlending {
        PatchBlending {
            mode,
            alpha_channel: 0,
            clamp: false,
        }
    }

    fn blank(
        width: usize,
        height: usize,
        depth: u32,
        colors: usize,
        alpha: bool,
    ) -> ReferenceImage {
        ReferenceImage {
            width,
            height,
            bit_depth: depth,
            color_channels: colors,
            planes: (0..colors + usize::from(alpha))
                .map(|c| {
                    vec![
                        if c == colors {
                            (1 << depth) - 1
                        } else {
                            17 + c as i32
                        };
                        width * height
                    ]
                })
                .collect(),
        }
    }

    fn draw(image: &mut ReferenceImage, x0: usize, y0: usize) {
        let maximum = (1 << image.bit_depth) - 1;
        for (c, plane) in image.planes.iter_mut().enumerate() {
            for y in 0..5 {
                for x in 0..6 {
                    plane[(y0 + y) * image.width + x0 + x] = if c == image.color_channels {
                        if (x + y) % 3 == 0 { 0 } else { maximum }
                    } else {
                        maximum - (c * 19 + y * 7 + x) as i32
                    };
                }
            }
        }
    }

    fn repeated(depth: u32, colors: usize, alpha: bool) -> ReferenceImage {
        let mut image = blank(97, 65, depth, colors, alpha);
        draw(&mut image, 13, 14);
        draw(&mut image, 51, 38);
        image
    }

    fn assert_roundtrip(image: &ReferenceImage, plan: &PatchPlan) {
        let residual = ImageView {
            planes: &plan.residual,
            ..image.view()
        };
        assert_eq!(
            compose(residual, plan.reference.view(), &plan.dictionary).unwrap(),
            image.planes
        );
        assert_eq!(
            subtract(image.view(), plan.reference.view(), &plan.dictionary).unwrap(),
            plan.residual
        );
        assert!(
            plan.dictionary
                .blendings
                .iter()
                .all(|b| b.mode == PatchBlendMode::Replace)
        );
        for position in &plan.dictionary.positions {
            let crop = plan.dictionary.references[position.ref_pos_idx];
            for plane in &plan.residual {
                for y in position.y as usize..(position.y + crop.height) as usize {
                    assert!(
                        plane[y * image.width + position.x as usize
                            ..y * image.width + (position.x + crop.width) as usize]
                            .iter()
                            .all(|&v| v == 0)
                    );
                }
            }
        }
        // Check every uncovered value too, not just the reconstructed patches.
        for y in 0..image.height {
            for x in 0..image.width {
                let covered = plan.dictionary.positions.iter().any(|p| {
                    let r = plan.dictionary.references[p.ref_pos_idx];
                    x >= p.x as usize
                        && x < (p.x + r.width) as usize
                        && y >= p.y as usize
                        && y < (p.y + r.height) as usize
                });
                if !covered {
                    for (before, after) in image.planes.iter().zip(&plan.residual) {
                        assert_eq!(before[y * image.width + x], after[y * image.width + x]);
                    }
                }
            }
        }
    }

    #[test]
    fn discovers_and_recomposes_gray_ga_rgb_rgba_8_and_16() {
        for depth in [8, 16] {
            for colors in [1, 3] {
                for alpha in [false, true] {
                    let image = repeated(depth, colors, alpha);
                    let before = image.planes.clone();
                    let plan = discover(image.view(), &PatchOptions::default())
                        .unwrap()
                        .unwrap();
                    assert_eq!(plan.dictionary.references.len(), 1);
                    assert_eq!(plan.dictionary.positions.len(), 2);
                    assert_eq!(
                        (
                            plan.dictionary.references[0].width,
                            plan.dictionary.references[0].height
                        ),
                        (6, 5)
                    );
                    assert_eq!(plan.dictionary.num_extra_channels, usize::from(alpha));
                    assert_roundtrip(&image, &plan);
                    assert_eq!(image.planes, before);
                    if alpha {
                        let pos = 14 * image.width + 13;
                        assert_eq!(image.planes[colors][pos], 0);
                        assert_eq!(image.planes[0][pos], (1 << depth) - 1);
                    }
                }
            }
        }
    }

    #[test]
    fn replace_accepts_all_nominal_integer_depths_and_multiple_full_resolution_extras() {
        for bit_depth in 1..=16 {
            for color_channels in [1, 3] {
                let mut image = ReferenceImage {
                    width: 97,
                    height: 65,
                    bit_depth,
                    color_channels,
                    planes: vec![vec![0; 97 * 65]; color_channels + 3],
                };
                let maximum = (1 << bit_depth) - 1;
                for plane in &mut image.planes {
                    for (x, y) in [(13, 14), (51, 38)] {
                        for row in y..y + 5 {
                            plane[row * 97 + x..row * 97 + x + 6].fill(maximum);
                        }
                    }
                }
                let plan = discover(image.view(), &PatchOptions::default())
                    .unwrap()
                    .unwrap();
                assert_eq!(plan.dictionary.num_extra_channels, 3);
                assert_roundtrip(&image, &plan);
            }
        }
    }

    #[test]
    fn grouping_observes_high_bits_hidden_rgb_and_alpha() {
        for channel in 0..4 {
            let mut image = repeated(16, 3, true);
            let pos = 38 * image.width + 51;
            assert_eq!(image.planes[3][pos], 0);
            if channel == 3 {
                image.planes[channel][pos] += 256;
            } else {
                image.planes[channel][pos] -= 256;
            }
            // Equal low bytes are NOT enough for a repeated 16-bit patch.
            assert!(
                discover(image.view(), &PatchOptions::default())
                    .unwrap()
                    .is_none()
            );
        }
        let mut image = repeated(8, 3, true);
        for channel in 0..3 {
            image.planes[channel].fill(17 + channel as i32);
        }
        for (x0, y0) in [(13, 14), (51, 38)] {
            for y in y0..y0 + 5 {
                image.planes[3][y * image.width + x0..y * image.width + x0 + 6].fill(0);
            }
        }
        let plan = discover(image.view(), &PatchOptions::default())
            .unwrap()
            .unwrap();
        assert_roundtrip(&image, &plan);
    }

    #[test]
    fn edges_tails_and_small_images() {
        let mut image = blank(97, 65, 16, 3, true);
        draw(&mut image, 0, 0);
        draw(&mut image, 91, 60);
        let plan = discover(image.view(), &PatchOptions::default())
            .unwrap()
            .unwrap();
        assert_roundtrip(&image, &plan);
        for width in 1..13 {
            for height in 1..13 {
                let image = blank(width, height, 8, 1, false);
                assert!(
                    discover(image.view(), &PatchOptions::default())
                        .unwrap()
                        .is_none()
                );
            }
        }
        let solid = blank(97, 65, 16, 3, true);
        assert!(
            discover(solid.view(), &PatchOptions::default())
                .unwrap()
                .is_none()
        );
        let mut noise = blank(97, 65, 16, 1, false);
        for (i, sample) in noise.planes[0].iter_mut().enumerate() {
            *sample = i as i32;
        }
        assert!(
            discover(noise.view(), &PatchOptions::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn discovery_respects_limits() {
        let image = repeated(8, 3, false);
        for options in [
            PatchOptions {
                max_patch_size: 5,
                ..PatchOptions::default()
            },
            PatchOptions {
                min_occurrences: 3,
                ..PatchOptions::default()
            },
            PatchOptions {
                min_max_patch_area: 31,
                ..PatchOptions::default()
            },
            PatchOptions {
                max_positions: 1,
                ..PatchOptions::default()
            },
            PatchOptions {
                max_reference_pixels: 29,
                ..PatchOptions::default()
            },
        ] {
            assert!(discover(image.view(), &options).unwrap().is_none());
        }
        let options = PatchOptions {
            max_patch_size: 6,
            max_positions: 2,
            max_unique_patches: 1,
            min_max_patch_area: 30,
            ..PatchOptions::default()
        };
        assert!(discover(image.view(), &options).unwrap().is_some());
        for options in [
            PatchOptions {
                max_patch_size: 129,
                ..PatchOptions::default()
            },
            PatchOptions {
                min_occurrences: 1,
                ..PatchOptions::default()
            },
            PatchOptions {
                max_positions: 0,
                ..PatchOptions::default()
            },
            PatchOptions {
                max_unique_patches: 0,
                ..PatchOptions::default()
            },
            PatchOptions {
                max_reference_pixels: 0,
                ..PatchOptions::default()
            },
        ] {
            assert!(discover(image.view(), &options).is_err());
        }
    }

    #[test]
    fn optimized_work_is_boundary_only_and_each_component_is_visited_once() {
        let solid = blank(512, 512, 8, 3, true);
        let mut work = DetectionWork::default();
        let pixels = solid.width * solid.height;
        assert!(
            find_patches_with_work(solid.view(), &PatchOptions::default(), pixels, &mut work)
                .unwrap()
                .is_empty()
        );
        assert!(work.seed_pixels > 250_000);
        assert!(work.initial_queue > 0 && work.initial_queue < 2100);
        assert_eq!(work.flood_enqueues, pixels - work.seed_pixels);
        assert_eq!(work.component_pushes, 0);

        let image = repeated(16, 3, true);
        let mut work = DetectionWork::default();
        let patches = find_patches_with_work(
            image.view(),
            &PatchOptions::default(),
            image.width * image.height,
            &mut work,
        )
        .unwrap();
        assert_eq!(patches.len(), 1);
        assert_eq!(work.component_pushes, 2 * 6 * 5);
        let mut noise = blank(97, 65, 16, 1, false);
        for (i, sample) in noise.planes[0].iter_mut().enumerate() {
            *sample = i as i32;
        }
        let mut work = DetectionWork::default();
        assert!(
            find_patches_with_work(
                noise.view(),
                &PatchOptions::default(),
                noise.width * noise.height,
                &mut work
            )
            .unwrap()
            .is_empty()
        );
        assert_eq!(
            work.seed_pixels + work.initial_queue + work.component_pushes,
            0
        );
    }

    #[test]
    fn larger_source_patch_sizes_and_one_unit_contrast_remain_exact() {
        for side in [64, 128] {
            let mut image = blank(384, 96, 16, 1, false);
            for (x0, y0) in [(16, 16), (192, 48)] {
                for y in y0..y0 + 2 {
                    image.planes[0][y * image.width + x0..y * image.width + x0 + side].fill(18);
                }
            }
            assert!(
                discover(image.view(), &PatchOptions::default())
                    .unwrap()
                    .is_none()
            );
            let options = PatchOptions {
                max_patch_size: side,
                ..PatchOptions::default()
            };
            let plan = discover(image.view(), &options).unwrap().unwrap();
            assert_eq!(plan.dictionary.references[0].width as usize, side);
            assert_roundtrip(&image, &plan);
        }
    }

    #[test]
    fn atlas_rectangles_are_disjoint_and_budgeted() {
        let info: Vec<_> = [(32, 4), (7, 18), (10, 10), (3, 31), (9, 9), (1, 1)]
            .into_iter()
            .map(|(width, height)| PatchInfo {
                key: PatchKey {
                    width,
                    height,
                    samples: vec![1; width * height],
                },
                positions: vec![(0, 0), (1, 1)],
            })
            .collect();
        let (width, height, positions) = pack(&info, 4096).unwrap().unwrap();
        assert_eq!(positions.len(), info.len());
        assert!(width * height <= 4096);
        let mut occupied = vec![false; width * height];
        for (patch, &(x, y)) in info.iter().zip(&positions) {
            assert!(x + patch.key.width <= width && y + patch.key.height <= height);
            for row in y..y + patch.key.height {
                for col in x..x + patch.key.width {
                    assert!(!occupied[row * width + col]);
                    occupied[row * width + col] = true;
                }
            }
        }
        assert!(pack(&info, 100).unwrap().is_none());
    }

    fn wire_dictionary() -> PatchDictionary {
        PatchDictionary {
            references: vec![PatchReferencePosition {
                reference: 3,
                x0: 2,
                y0: 4,
                width: 6,
                height: 5,
            }],
            positions: vec![
                PatchPosition {
                    x: 10,
                    y: 20,
                    ref_pos_idx: 0,
                },
                PatchPosition {
                    x: 7,
                    y: 25,
                    ref_pos_idx: 0,
                },
            ],
            blendings: vec![blend(PatchBlendMode::Replace); 4],
            num_extra_channels: 1,
        }
    }

    #[test]
    fn exact_cpp_token_order_and_signed_offsets() {
        let tokens = tokenize(&wire_dictionary()).unwrap();
        let pairs: Vec<_> = tokens.iter().map(|t| (t.context, t.value)).collect();
        assert_eq!(
            pairs,
            vec![
                (0, 1),
                (1, 3),
                (3, 2),
                (3, 4),
                (2, 5),
                (2, 4),
                (7, 1),
                (4, 10),
                (4, 20),
                (5, 1),
                (5, 1),
                (6, 5),
                (6, 10),
                (5, 1),
                (5, 1),
            ]
        );
        assert_eq!(offset(0, 0).unwrap(), 0);
        assert_eq!(offset(i32::MAX as u32, 0).unwrap(), u32::MAX - 1);
        assert_eq!(offset(0, 1u32 << 31).unwrap(), u32::MAX);
        assert!(offset(u32::MAX, 0).is_err());
        assert!(offset(0, u32::MAX).is_err());
    }

    #[test]
    fn alpha_and_clamp_tokens_are_conditional() {
        for mode in [
            PatchBlendMode::None,
            PatchBlendMode::Replace,
            PatchBlendMode::Add,
            PatchBlendMode::Mul,
            PatchBlendMode::BlendAbove,
            PatchBlendMode::BlendBelow,
            PatchBlendMode::AlphaWeightedAddAbove,
            PatchBlendMode::AlphaWeightedAddBelow,
        ] {
            for extras in [1, 2] {
                let mut dict = wire_dictionary();
                dict.positions.truncate(1);
                dict.num_extra_channels = extras;
                dict.blendings = vec![blend(PatchBlendMode::None); extras + 1];
                dict.blendings[0] = PatchBlending {
                    mode,
                    alpha_channel: (extras - 1) as u32,
                    clamp: true,
                };
                let tokens = tokenize(&dict).unwrap();
                let tail: Vec<_> = tokens[9..].iter().map(|t| (t.context, t.value)).collect();
                let mut expected = vec![(5, mode as u32)];
                if mode as u32 >= 4 && extras == 2 {
                    expected.push((8, 1));
                }
                if mode as u32 >= 3 {
                    expected.push((9, 1));
                }
                expected.extend((0..extras).map(|_| (5, 0)));
                assert_eq!(tail, expected);
            }
        }
    }

    // Token-level reader follows dec_patch_dictionary.cc, independently of the
    // tokenizer's grouping loop. Full entropy/frame validation belongs to main.
    fn read_wire(tokens: &[Literal], extras: usize) -> PatchDictionary {
        let mut cursor = 0;
        let mut read = |context| {
            let literal = tokens[cursor];
            cursor += 1;
            assert_eq!(literal.context, context);
            literal.value
        };
        let count = read(0);
        let mut dict = PatchDictionary {
            references: Vec::new(),
            positions: Vec::new(),
            blendings: Vec::new(),
            num_extra_channels: extras,
        };
        for index in 0..count as usize {
            dict.references.push(PatchReferencePosition {
                reference: read(1),
                x0: read(3),
                y0: read(3),
                width: read(2) + 1,
                height: read(2) + 1,
            });
            let count = read(7) + 1;
            let (mut x, mut y) = (0i64, 0i64);
            for i in 0..count {
                if i == 0 {
                    x = i64::from(read(4));
                    y = i64::from(read(4));
                } else {
                    let unpack = |v: u32| i64::from(v >> 1) ^ -i64::from(v & 1);
                    x += unpack(read(6));
                    y += unpack(read(6));
                }
                dict.positions.push(PatchPosition {
                    x: u32::try_from(x).unwrap(),
                    y: u32::try_from(y).unwrap(),
                    ref_pos_idx: index,
                });
                for _ in 0..=extras {
                    let mode = match read(5) {
                        0 => PatchBlendMode::None,
                        1 => PatchBlendMode::Replace,
                        2 => PatchBlendMode::Add,
                        3 => PatchBlendMode::Mul,
                        4 => PatchBlendMode::BlendAbove,
                        5 => PatchBlendMode::BlendBelow,
                        6 => PatchBlendMode::AlphaWeightedAddAbove,
                        7 => PatchBlendMode::AlphaWeightedAddBelow,
                        _ => panic!("invalid blend mode"),
                    };
                    let alpha_channel = if mode as u32 >= 4 && extras > 1 {
                        read(8)
                    } else {
                        0
                    };
                    let clamp = if mode as u32 >= 3 {
                        read(9) != 0
                    } else {
                        false
                    };
                    dict.blendings.push(PatchBlending {
                        mode,
                        alpha_channel,
                        clamp,
                    });
                }
            }
        }
        assert_eq!(cursor, tokens.len());
        dict
    }

    #[test]
    fn token_roundtrip_and_noncontiguous_reference_runs() {
        let mut dict = wire_dictionary();
        let mut second = dict.references[0];
        second.x0 = 20;
        dict.references.push(second);
        dict.positions.extend([
            PatchPosition {
                x: 2,
                y: 1,
                ref_pos_idx: 1,
            },
            PatchPosition {
                x: 0,
                y: 0,
                ref_pos_idx: 0,
            },
        ]);
        dict.blendings.resize(8, blend(PatchBlendMode::Replace));
        let tokens = tokenize(&dict).unwrap();
        assert_eq!(tokens[0].value, 3);
        let decoded = read_wire(&tokens, 1);
        assert_eq!(decoded.blendings, dict.blendings);
        for (a, b) in decoded.positions.iter().zip(&dict.positions) {
            assert_eq!((a.x, a.y), (b.x, b.y));
            assert_eq!(
                decoded.references[a.ref_pos_idx],
                dict.references[b.ref_pos_idx]
            );
        }
        let image = repeated(16, 3, true);
        let plan = discover(image.view(), &PatchOptions::default())
            .unwrap()
            .unwrap();
        let decoded = read_wire(&tokenize(&plan.dictionary).unwrap(), 1);
        assert_eq!(decoded, plan.dictionary);
        let residual = ImageView {
            planes: &plan.residual,
            ..image.view()
        };
        assert_eq!(
            compose(residual, plan.reference.view(), &decoded).unwrap(),
            image.planes
        );
        dict.references.clear();
        dict.positions.clear();
        dict.blendings.clear();
        assert_eq!(
            tokenize(&dict).unwrap(),
            vec![Literal {
                context: 0,
                value: 0
            }]
        );
    }

    #[test]
    fn replace_checks_original_samples_in_every_plane() {
        let image = repeated(16, 3, true);
        let plan = discover(image.view(), &PatchOptions::default())
            .unwrap()
            .unwrap();
        for channel in 0..4 {
            let mut changed = image.clone();
            changed.planes[channel][14 * image.width + 13] ^= 256;
            assert!(subtract(changed.view(), plan.reference.view(), &plan.dictionary).is_err());
        }
        let mut dict = plan.dictionary.clone();
        for blends in dict.blendings.chunks_exact_mut(2) {
            blends[1].mode = PatchBlendMode::None;
        }
        let residual = subtract(image.view(), plan.reference.view(), &dict).unwrap();
        assert_eq!(residual[3], image.planes[3]);
        assert_eq!(
            compose(
                ImageView {
                    planes: &residual,
                    ..image.view()
                },
                plan.reference.view(),
                &dict
            )
            .unwrap(),
            image.planes
        );
    }

    #[test]
    fn overlapping_replace_obeys_dictionary_order() {
        let image = blank(3, 1, 8, 1, false);
        let mut reference = blank(3, 1, 8, 1, false);
        reference.planes[0] = vec![11, 22, 33];
        let dict = PatchDictionary {
            references: vec![PatchReferencePosition {
                reference: 3,
                x0: 0,
                y0: 0,
                width: 2,
                height: 1,
            }],
            positions: vec![
                PatchPosition {
                    x: 0,
                    y: 0,
                    ref_pos_idx: 0,
                },
                PatchPosition {
                    x: 1,
                    y: 0,
                    ref_pos_idx: 0,
                },
            ],
            blendings: vec![blend(PatchBlendMode::Replace); 2],
            num_extra_channels: 0,
        };
        assert_eq!(
            compose(image.view(), reference.view(), &dict).unwrap()[0],
            [11, 11, 22]
        );
        assert!(subtract(image.view(), reference.view(), &dict).is_err());
        reference.planes[0].fill(17);
        let residual = subtract(image.view(), reference.view(), &dict).unwrap();
        assert_eq!(residual[0], [0, 0, 0]);
        assert_eq!(
            compose(
                ImageView {
                    planes: &residual,
                    ..image.view()
                },
                reference.view(),
                &dict
            )
            .unwrap(),
            image.planes
        );
    }

    #[test]
    fn rejects_invalid_images_and_dictionary_shapes() {
        let image = repeated(8, 3, true);
        for view in [
            ImageView {
                width: 0,
                ..image.view()
            },
            ImageView {
                height: usize::MAX,
                ..image.view()
            },
            ImageView {
                bit_depth: 17,
                ..image.view()
            },
            ImageView {
                color_channels: 2,
                ..image.view()
            },
        ] {
            assert!(discover(view, &PatchOptions::default()).is_err());
        }
        let mut invalid_image = image.clone();
        invalid_image.planes.truncate(2);
        assert!(discover(invalid_image.view(), &PatchOptions::default()).is_err());
        invalid_image = image.clone();
        invalid_image.planes[0].pop();
        assert!(discover(invalid_image.view(), &PatchOptions::default()).is_err());
        for value in [-1, 256] {
            invalid_image = image.clone();
            invalid_image.planes[0][0] = value;
            assert!(discover(invalid_image.view(), &PatchOptions::default()).is_err());
        }
        for kind in 0..5 {
            let mut dict = wire_dictionary();
            match kind {
                0 => {
                    dict.blendings.pop();
                }
                1 => dict.positions[0].ref_pos_idx = 1,
                2 => dict.references[0].width = 0,
                3 => dict.references[0].reference = 4,
                _ => {
                    dict.blendings[0] = PatchBlending {
                        mode: PatchBlendMode::BlendAbove,
                        alpha_channel: 1,
                        clamp: false,
                    }
                }
            }
            assert!(tokenize(&dict).is_err());
        }
    }

    #[test]
    fn composition_rejects_out_of_bounds_metadata_and_arithmetic_blends() {
        let image = repeated(8, 3, true);
        let plan = discover(image.view(), &PatchOptions::default())
            .unwrap()
            .unwrap();
        for kind in 0..6 {
            let mut dict = plan.dictionary.clone();
            match kind {
                0 => dict.positions[0].x = u32::MAX,
                1 => dict.positions[0].y = u32::MAX,
                2 => dict.references[0].x0 = u32::MAX,
                3 => dict.references[0].y0 = u32::MAX,
                4 => dict.references[0].reference = 2,
                _ => dict.blendings[0].mode = PatchBlendMode::Add,
            }
            assert!(compose(image.view(), plan.reference.view(), &dict).is_err());
            assert!(subtract(image.view(), plan.reference.view(), &dict).is_err());
        }
        let wrong_depth = ImageView {
            bit_depth: 16,
            ..plan.reference.view()
        };
        assert!(compose(image.view(), wrong_depth, &plan.dictionary).is_err());
    }
}
