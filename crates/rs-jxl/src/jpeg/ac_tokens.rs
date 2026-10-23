// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! DCT8 AC tokenization prerequisite, not a JPEG-to-JXL encoder.
//!
//! Source: enc_entropy_coder.cc, ac_context.h, entropy_coder.{h,cc},
//! enc_context_map.cc, dec_context_map.cc, compressed_dc.cc and pack_signed.h at
//! b866b3defec13494bfbaf705db93581cda118e37. Input coefficients are already
//! transposed, quantized integers; no DCT, pixel processing or order search occurs.

use std::fmt;

use crate::bit_writer::BitWriter;
use crate::entropy::context_map::write_context_map;
use crate::entropy::stream::Token;

pub const DCT8_SIZE: usize = 64;
pub const NUM_ORDERS: usize = 13;
pub const NONZERO_BUCKETS: usize = 37;
pub const ZERO_DENSITY_CONTEXTS: usize = 458;
pub const ZERO_DENSITY_LIMIT: usize = 474;

const COEFF_FREQ_CONTEXT: [u16; 64] = [
    0xBAD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];
const COEFF_NUM_NONZERO_CONTEXT: [u16; 64] = [
    0xBAD, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152,
    152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206,
];
const DEFAULT_CONTEXT_MAP: [u8; 3 * NUM_ORDERS] = [
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 7, 8, 9,
    9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AcTokenError {
    Invalid(&'static str),
    DimensionOverflow,
    LimitExceeded(&'static str),
    AllocationFailed,
}

impl fmt::Display for AcTokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid DCT8 AC input: {reason}"),
            Self::DimensionOverflow => f.write_str("DCT8 AC dimensions overflow"),
            Self::LimitExceeded(reason) => write!(f, "DCT8 AC resource limit: {reason}"),
            Self::AllocationFailed => f.write_str("DCT8 AC allocation failed"),
        }
    }
}
impl std::error::Error for AcTokenError {}

type Result<T> = std::result::Result<T, AcTokenError>;

/// Validated source BlockCtxMap, including all 13 order classes. The tokenizer
/// below uses only order class zero (DCT8). Threshold order is not constrained:
/// source counts strict comparisons, including duplicate/unsorted thresholds.
#[derive(Clone, Copy, Debug)]
pub struct BlockContextModel<'a> {
    dc_thresholds: [&'a [i32]; 3],
    qf_thresholds: &'a [u32],
    context_map: &'a [u8],
    num_contexts: usize,
    num_dc_contexts: usize,
}

impl Default for BlockContextModel<'_> {
    fn default() -> Self {
        Self {
            dc_thresholds: [&[]; 3],
            qf_thresholds: &[],
            context_map: &DEFAULT_CONTEXT_MAP,
            num_contexts: 15,
            num_dc_contexts: 1,
        }
    }
}

impl<'a> BlockContextModel<'a> {
    /// Applies the source wire bounds: 4-bit threshold counts, at most 64
    /// DC/QF combinations, at most 16 block contexts and QF thresholds 1..=300.
    /// Use write_block_context_map to serialize the validated model.
    pub fn new(
        dc_thresholds: [&'a [i32]; 3],
        qf_thresholds: &'a [u32],
        context_map: &'a [u8],
        num_contexts: usize,
    ) -> Result<Self> {
        if dc_thresholds.iter().any(|t| t.len() > 15) || qf_thresholds.len() > 15 {
            return Err(AcTokenError::Invalid("too many context thresholds"));
        }
        if qf_thresholds.iter().any(|&t| !(1..=300).contains(&t)) {
            return Err(AcTokenError::Invalid("QF threshold outside wire range"));
        }
        let num_dc_contexts = dc_thresholds.iter().map(|t| t.len() + 1).product::<usize>();
        let combinations = num_dc_contexts * (qf_thresholds.len() + 1);
        if combinations > 64 {
            return Err(AcTokenError::Invalid("more than 64 DC/QF combinations"));
        }
        if !(1..=16).contains(&num_contexts) {
            return Err(AcTokenError::Invalid("block context count outside 1..=16"));
        }
        if context_map.len() != 3 * NUM_ORDERS * combinations
            || context_map.iter().any(|&c| usize::from(c) >= num_contexts)
        {
            return Err(AcTokenError::Invalid("context map size or context ID"));
        }
        let used_ids = context_map
            .iter()
            .fold(0u32, |mask, &id| mask | (1u32 << id));
        if used_ids != (1u32 << num_contexts) - 1 {
            return Err(AcTokenError::Invalid("incomplete context map"));
        }
        Ok(Self {
            dc_thresholds,
            qf_thresholds,
            context_map,
            num_contexts,
            num_dc_contexts,
        })
    }

    pub fn num_contexts(self) -> usize {
        self.num_contexts
    }
    pub fn num_dc_contexts(self) -> usize {
        self.num_dc_contexts
    }
    pub fn num_ac_contexts(self) -> usize {
        self.num_contexts * (NONZERO_BUCKETS + ZERO_DENSITY_CONTEXTS)
    }

    /// Source EncodeBlockCtxMap's default-map shortcut predicate.
    pub fn is_default(self) -> bool {
        self.dc_thresholds.iter().all(|t| t.is_empty())
            && self.qf_thresholds.is_empty()
            && self.context_map == DEFAULT_CONTEXT_MAP
    }

    /// Optional scalar qdc nomination from already quantized DC values in JXL
    /// channel order. compressed_dc.cc::DequantDC interleaves buckets X, B, Y.
    /// Sampling and any DC/CfL preparation remain the caller's responsibility.
    pub fn dc_context(self, dc: [i32; 3]) -> u8 {
        let buckets: [usize; 3] =
            std::array::from_fn(|c| self.dc_thresholds[c].iter().filter(|&&t| dc[c] > t).count());
        ((buckets[0] * (self.dc_thresholds[2].len() + 1) + buckets[2])
            * (self.dc_thresholds[1].len() + 1)
            + buckets[1]) as u8
    }

    pub fn context(self, qdc: u8, qf: u32, order: usize, channel: usize) -> Result<usize> {
        if usize::from(qdc) >= self.num_dc_contexts || order >= NUM_ORDERS || channel >= 3 {
            return Err(AcTokenError::Invalid("block context coordinates"));
        }
        Ok(self.context_validated(qdc, qf, order, channel))
    }

    fn context_validated(self, qdc: u8, qf: u32, order: usize, channel: usize) -> usize {
        let qf_index = self.qf_thresholds.iter().filter(|&&t| qf > t).count();
        let channel_index = if channel < 2 { channel ^ 1 } else { 2 };
        let index = ((channel_index * NUM_ORDERS + order) * (self.qf_thresholds.len() + 1)
            + qf_index)
            * self.num_dc_contexts
            + usize::from(qdc);
        usize::from(self.context_map[index])
    }

    pub fn nonzero_context(self, prediction: u32, block_context: usize) -> Result<usize> {
        self.check_block_context(block_context)?;
        let prediction = prediction.min(64) as usize;
        let bucket = if prediction < 8 {
            prediction
        } else {
            4 + prediction / 2
        };
        Ok(bucket * self.num_contexts + block_context)
    }

    pub fn zero_density_offset(self, block_context: usize) -> Result<usize> {
        self.check_block_context(block_context)?;
        Ok(self.num_contexts * NONZERO_BUCKETS + ZERO_DENSITY_CONTEXTS * block_context)
    }

    fn check_block_context(self, block_context: usize) -> Result<()> {
        if block_context >= self.num_contexts {
            return Err(AcTokenError::Invalid("block context ID"));
        }
        Ok(())
    }
}

fn pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

// These source threshold distributions have disjoint ascending-width ranges,
// so the first fitting selector is U32Coder::ChooseSelector's cheapest one.
fn write_threshold(
    writer: &mut BitWriter,
    value: u32,
    choices: [(u8, u32); 4],
) -> crate::Result<()> {
    for (selector, (bits, offset)) in choices.into_iter().enumerate() {
        if let Some(extra) = value.checked_sub(offset)
            && u64::from(extra) < (1u64 << bits)
        {
            writer.write(2, selector as u64)?;
            writer.write(bits, u64::from(extra))?;
            return Ok(());
        }
    }
    Err(crate::Error::Encoding(
        "block context threshold has no source U32 encoding".into(),
    ))
}

/// Source EncodeBlockCtxMap wire wrapper: default bit, all DC/QF thresholds,
/// then the existing generic entropy context-map writer. No padding is added.
/// The destination is unchanged on failure. Generic map representation choice
/// retains that module's exact-cost policy, not C++'s dry-cost estimate policy.
pub fn write_block_context_map(
    model: BlockContextModel<'_>,
    writer: &mut BitWriter,
) -> crate::Result<()> {
    if model.is_default() {
        writer.write(1, 1)?;
        return Ok(());
    }
    // Validated models have <=2496 entries and <=15 thresholds in each list.
    // Match the generic coded-map writer's byte bound plus the threshold prefix.
    let threshold_count =
        model.dc_thresholds.iter().map(|t| t.len()).sum::<usize>() + model.qf_thresholds.len();
    let prefix_bits = 17 + threshold_count * 34;
    let mut temporary =
        BitWriter::new(model.context_map.len() * 8 + 8192 + prefix_bits.div_ceil(8));
    temporary.write(1, 0)?;
    for thresholds in model.dc_thresholds {
        temporary.write(4, thresholds.len() as u64)?;
        for &threshold in thresholds {
            write_threshold(
                &mut temporary,
                pack_signed(threshold),
                [(4, 0), (8, 16), (16, 272), (32, 65808)],
            )?;
        }
    }
    temporary.write(4, model.qf_thresholds.len() as u64)?;
    for &threshold in model.qf_thresholds {
        write_threshold(
            &mut temporary,
            threshold - 1,
            [(2, 0), (3, 4), (5, 12), (8, 44)],
        )?;
    }
    write_context_map(model.context_map, model.num_contexts, &mut temporary)?;
    writer.append_unaligned(&temporary)?;
    Ok(())
}

/// Exact ZeroDensityContext specialization for covered_blocks=1. Independent
/// indices 1..=63 are accepted, as in the source decoder's table guard; actual
/// DCT8 encoder states additionally satisfy nonzeros_left + k <= 64.
pub fn zero_density_context(
    nonzeros_left: usize,
    k: usize,
    previous_nonzero: bool,
) -> Result<usize> {
    if !(1..64).contains(&nonzeros_left) || !(1..64).contains(&k) {
        return Err(AcTokenError::Invalid("zero-density table coordinates"));
    }
    Ok(zero_density_validated(nonzeros_left, k, previous_nonzero))
}

fn zero_density_validated(nonzeros_left: usize, k: usize, previous_nonzero: bool) -> usize {
    usize::from(COEFF_NUM_NONZERO_CONTEXT[nonzeros_left] + COEFF_FREQ_CONTEXT[k]) * 2
        + usize::from(previous_nonzero)
}

/// Coefficients in channel-local raster block order. Each block has 64 values
/// in already transposed coefficient storage, not JPEG natural order.
#[derive(Clone, Copy, Debug)]
pub struct Dct8Plane<'a> {
    pub coefficients: &'a [i32],
    pub width_blocks: usize,
    pub height_blocks: usize,
    /// Source HShift/VShift, not JPEG sampling factors; each must be 0 or 1.
    pub horizontal_shift: u8,
    pub vertical_shift: u8,
}

/// A group-local DCT8-only rectangle. qdc/qf are tightly packed full-resolution
/// block grids. Prediction resets at the rectangle's top and left boundaries.
#[derive(Clone, Copy, Debug)]
pub struct Dct8Input<'a> {
    pub width_blocks: usize,
    pub height_blocks: usize,
    pub planes: [Dct8Plane<'a>; 3],
    /// Each order maps scan rank to transposed coefficient index; index 0 is DC.
    pub orders: &'a [[u32; DCT8_SIZE]; 3],
    pub qdc: &'a [u8],
    /// Source raw quantization field: each value must be in 1..=256.
    pub qf: &'a [i32],
}

/// Explicit caller budgets. The conservative output reservation is 64 tokens
/// per component block, even for sparse inputs. Scratch is one byte per column
/// of each component; no per-image nonzero map or coefficient copy is allocated.
#[derive(Clone, Copy, Debug)]
pub struct TokenizeLimits {
    pub max_tokens: usize,
    pub max_scratch_bytes: usize,
}

fn checked_product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(AcTokenError::DimensionOverflow)
}

fn validate(
    input: &Dct8Input<'_>,
    model: BlockContextModel<'_>,
    limits: TokenizeLimits,
) -> Result<usize> {
    let width = input.width_blocks;
    let height = input.height_blocks;
    if (width == 0) != (height == 0) {
        return Err(AcTokenError::Invalid(
            "only a 0 by 0 empty rectangle is supported",
        ));
    }
    let area = checked_product(width, height)?;
    if input.qdc.len() != area || input.qf.len() != area {
        return Err(AcTokenError::Invalid("qdc/qf grid dimensions"));
    }
    if input
        .qdc
        .iter()
        .any(|&dc| usize::from(dc) >= model.num_dc_contexts)
    {
        return Err(AcTokenError::Invalid("qdc index outside model"));
    }
    if input.qf.iter().any(|qf| !(1..=256).contains(qf)) {
        return Err(AcTokenError::Invalid("raw qf outside 1..=256"));
    }
    if !input.planes.iter().any(|p| p.horizontal_shift == 0)
        || !input.planes.iter().any(|p| p.vertical_shift == 0)
    {
        return Err(AcTokenError::Invalid(
            "sampling shifts have no full-resolution reference",
        ));
    }
    let mut token_bound = 0usize;
    let mut scratch_bytes = 0usize;
    for (plane, order) in input.planes.iter().zip(input.orders) {
        if plane.horizontal_shift > 1 || plane.vertical_shift > 1 {
            return Err(AcTokenError::Invalid("sampling shift outside 0..=1"));
        }
        let plane_width = width.div_ceil(1 << plane.horizontal_shift);
        let plane_height = height.div_ceil(1 << plane.vertical_shift);
        let coefficients = checked_product(checked_product(plane_width, plane_height)?, DCT8_SIZE)?;
        if plane.width_blocks != plane_width
            || plane.height_blocks != plane_height
            || plane.coefficients.len() != coefficients
        {
            return Err(AcTokenError::Invalid(
                "component plane dimensions or coefficient count",
            ));
        }
        let mut seen = 0u64;
        for &index in order {
            if index >= 64 || seen & (1u64 << index) != 0 {
                return Err(AcTokenError::Invalid(
                    "coefficient order is not a permutation",
                ));
            }
            seen |= 1u64 << index;
        }
        if order[0] != 0 {
            return Err(AcTokenError::Invalid(
                "coefficient order must start with DC",
            ));
        }
        token_bound = token_bound
            .checked_add(coefficients)
            .ok_or(AcTokenError::DimensionOverflow)?;
        scratch_bytes = scratch_bytes
            .checked_add(plane_width)
            .ok_or(AcTokenError::DimensionOverflow)?;
    }
    if token_bound > limits.max_tokens {
        return Err(AcTokenError::LimitExceeded("token reservation"));
    }
    if scratch_bytes > limits.max_scratch_bytes {
        return Err(AcTokenError::LimitExceeded("nonzero prediction rows"));
    }
    if checked_product(token_bound, std::mem::size_of::<Token>())? > isize::MAX as usize
        || scratch_bytes > isize::MAX as usize
    {
        return Err(AcTokenError::DimensionOverflow);
    }
    Ok(token_bound)
}

/// Source TokenizeCoefficients restricted to DCT8. Returns ordinary entropy
/// tokens (is_length=false), before histogram clustering or bitstream emission.
pub fn tokenize_dct8(
    input: &Dct8Input<'_>,
    model: BlockContextModel<'_>,
    limits: TokenizeLimits,
) -> Result<Vec<Token>> {
    let token_bound = validate(input, model, limits)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(token_bound)
        .map_err(|_| AcTokenError::AllocationFailed)?;
    let mut rows: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::new());
    for (row, plane) in rows.iter_mut().zip(&input.planes) {
        row.try_reserve_exact(plane.width_blocks)
            .map_err(|_| AcTokenError::AllocationFailed)?;
        row.resize(plane.width_blocks, 0);
    }
    let mut offsets = [0usize; 3];
    for by in 0..input.height_blocks {
        let row_start = by * input.width_blocks;
        for bx in 0..input.width_blocks {
            for c in [1, 0, 2] {
                let plane = &input.planes[c];
                let sx = bx >> plane.horizontal_shift;
                let sy = by >> plane.vertical_shift;
                if sx << plane.horizontal_shift != bx || sy << plane.vertical_shift != by {
                    continue;
                }
                let block: &[i32; DCT8_SIZE] = plane.coefficients
                    [offsets[c]..offsets[c] + DCT8_SIZE]
                    .try_into()
                    .expect("validated complete DCT8 block");
                let mut nonzeros = count_nonzeros(block);
                // One rolling row preserves exactly the source top/left inputs.
                let prediction = match (sx, sy) {
                    (0, 0) => 32,
                    (0, _) => u32::from(rows[c][sx]),
                    (_, 0) => u32::from(rows[c][sx - 1]),
                    _ => (u32::from(rows[c][sx]) + u32::from(rows[c][sx - 1])).div_ceil(2),
                };
                rows[c][sx] = nonzeros as u8;
                // qf deliberately uses sx on the full-resolution row, not bx
                // or sy. This is enc_entropy_coder.cc's row_qf[sbx[c]].
                let block_context = model.context_validated(
                    input.qdc[row_start + bx],
                    input.qf[row_start + sx] as u32,
                    0,
                    c,
                );
                output.push(Token::literal(
                    model.nonzero_context(prediction, block_context)?,
                    nonzeros,
                ));
                let offset = model.zero_density_offset(block_context)?;
                let mut previous_nonzero = nonzeros <= 4;
                for (k, &index) in input.orders[c].iter().enumerate().skip(1) {
                    if nonzeros == 0 {
                        break;
                    }
                    let coefficient = block[index as usize];
                    let context =
                        offset + zero_density_validated(nonzeros as usize, k, previous_nonzero);
                    let value = pack_signed(coefficient);
                    output.push(Token::literal(context, value));
                    previous_nonzero = coefficient != 0;
                    nonzeros -= u32::from(previous_nonzero);
                }
                debug_assert_eq!(nonzeros, 0);
                offsets[c] += DCT8_SIZE;
            }
        }
    }
    debug_assert!(output.len() <= token_bound);
    Ok(output)
}

fn count_nonzeros_scalar(block: &[i32; DCT8_SIZE]) -> u32 {
    block[1..]
        .iter()
        .filter(|&&coefficient| coefficient != 0)
        .count() as u32
}

/// Native Rust specialization of NumNonZero8x8ExceptDC. SIMD reads exactly
/// 64 i32 values using unaligned loads and masks DC before comparing with zero.
pub fn count_nonzeros(block: &[i32; DCT8_SIZE]) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature check; the fixed-size reference bounds all loads.
            return unsafe { count_nonzeros_avx2(block) };
        }
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: runtime feature check; the fixed-size reference bounds all loads.
            return unsafe { count_nonzeros_sse2(block) };
        }
    }
    count_nonzeros_scalar(block)
}

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn count_nonzeros_sse2(block: &[i32; DCT8_SIZE]) -> u32 {
    // SAFETY: caller checks SSE2; 16 unaligned four-lane loads stay inside block,
    // and the final four-lane store fits lanes. Arithmetic accumulates -64..=0.
    unsafe {
        let zero = arch::_mm_setzero_si128();
        let mask = arch::_mm_set_epi32(0, 0, 0, -1);
        let first = arch::_mm_andnot_si128(mask, arch::_mm_loadu_si128(block.as_ptr().cast()));
        let mut sum = arch::_mm_cmpeq_epi32(first, zero);
        for i in (4..DCT8_SIZE).step_by(4) {
            let values = arch::_mm_loadu_si128(block.as_ptr().add(i).cast());
            sum = arch::_mm_add_epi32(sum, arch::_mm_cmpeq_epi32(values, zero));
        }
        let mut lanes = [0i32; 4];
        arch::_mm_storeu_si128(lanes.as_mut_ptr().cast(), sum);
        (64 + lanes.into_iter().sum::<i32>()) as u32
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn count_nonzeros_avx2(block: &[i32; DCT8_SIZE]) -> u32 {
    // SAFETY: caller checks AVX2; eight unaligned eight-lane loads stay inside
    // block, and the final eight-lane store fits lanes. DC is masked to zero.
    unsafe {
        let zero = arch::_mm256_setzero_si256();
        let mask = arch::_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
        let first =
            arch::_mm256_andnot_si256(mask, arch::_mm256_loadu_si256(block.as_ptr().cast()));
        let mut sum = arch::_mm256_cmpeq_epi32(first, zero);
        for i in (8..DCT8_SIZE).step_by(8) {
            let values = arch::_mm256_loadu_si256(block.as_ptr().add(i).cast());
            sum = arch::_mm256_add_epi32(sum, arch::_mm256_cmpeq_epi32(values, zero));
        }
        let mut lanes = [0i32; 8];
        arch::_mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum);
        (64 + lanes.into_iter().sum::<i32>()) as u32
    }
}

#[cfg(test)]
#[path = "ac_tokens_tests.rs"]
mod tests;
