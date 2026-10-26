// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! DCT8-only coefficient-order preparation and permutation tokenization.
//!
//! Frozen b866b3defec13494bfbaf705db93581cda118e37: enc_coeff_order.cc,
//! coeff_order.{h,cc}, coeff_order_fwd.h, ac_strategy.cc and lehmer_code.h.
//! Input is borrowed, group-packed JXL coefficient data, not JPEG zigzag data.
//! No DCT calculation, coefficient entropy stream or JXL frame is emitted.

use std::fmt;

use super::data::{BLOCK_SIZE, NATURAL_ORDER};
use crate::entropy::{HybridUintConfig, options::SpeedTier, stream::Token};

pub const DCT8_ORDER_MASK: u16 = 1;
pub const PERMUTATION_CONTEXTS: usize = 8;
pub const GROUP_DIM_IN_BLOCKS: usize = 32;
pub const MAX_PERMUTATION_SIZE: usize = BLOCK_SIZE;

// AcStrategy::CoeffOrderAndLut for DCT8 has the same numeric zigzag as JPEG.
// These offsets address an ALREADY TRANSPOSED JXL block. Do not transpose the
// zigzag offsets a second time when converting the original JPEG coefficients.
const fn default_order() -> [u32; BLOCK_SIZE] {
    let mut order = [0; BLOCK_SIZE];
    let mut i = 0;
    while i < BLOCK_SIZE {
        order[i] = NATURAL_ORDER[i] as u32;
        i += 1;
    }
    order
}
pub const DEFAULT_DCT8_ORDER: [u32; BLOCK_SIZE] = default_order();

const fn default_lut() -> [u32; BLOCK_SIZE] {
    let mut lut = [0; BLOCK_SIZE];
    let mut i = 0;
    while i < BLOCK_SIZE {
        lut[DEFAULT_DCT8_ORDER[i] as usize] = i as u32;
        i += 1;
    }
    lut
}
pub const DEFAULT_DCT8_ORDER_LUT: [u32; BLOCK_SIZE] = default_lut();

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderError {
    InvalidGeometry,
    InvalidGroupCount,
    InvalidCoefficientStorage,
    MixedCoefficientTypes,
    InvalidPermutationSize,
    InvalidPermutationValue { index: usize, value: u32 },
    InvalidSkip,
    NonIdentityPrefix,
    DcNotFirst,
    UnsignaledCustomOrder,
}

impl fmt::Display for OrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry => f.write_str(
                "DCT8 geometry must be nonempty and fit its index and frequency counters",
            ),
            Self::InvalidGroupCount => {
                f.write_str("DCT8 group count does not match frame geometry")
            }
            Self::InvalidCoefficientStorage => {
                f.write_str("DCT8 planes must contain complete, sufficiently sized blocks")
            }
            Self::MixedCoefficientTypes => {
                f.write_str("DCT8 groups must share one i16 or i32 AC image type")
            }
            Self::InvalidPermutationSize => f.write_str("DCT8 permutation size must be 1..=64"),
            Self::InvalidPermutationValue { index, value } => write!(
                f,
                "invalid or duplicate permutation value {value} at {index}"
            ),
            Self::InvalidSkip => f.write_str("permutation skip exceeds its size"),
            Self::NonIdentityPrefix => f.write_str("skipped permutation prefix must be identity"),
            Self::DcNotFirst => f.write_str("DCT8 coefficient order must start with DC"),
            Self::UnsignaledCustomOrder => {
                f.write_str("nondefault DCT8 orders require the used-order bit")
            }
        }
    }
}
impl std::error::Error for OrderError {}

/// Three JXL channels, each packed as consecutive 64-entry transposed blocks.
/// Edge groups may retain the source's full allocation; only the geometry's
/// block prefix is read. All groups of one image must have the same integer type.
#[derive(Clone, Copy, Debug)]
pub enum Dct8Group<'a> {
    I16([&'a [i16]; 3]),
    I32([&'a [i32]; 3]),
}

impl Dct8Group<'_> {
    fn lengths(self) -> [usize; 3] {
        match self {
            Self::I16(planes) => planes.map(|plane| plane.len()),
            Self::I32(planes) => planes.map(|plane| plane.len()),
        }
    }

    fn count_block(self, block: usize, zeros: &mut [[i64; BLOCK_SIZE]; 3]) {
        fn count<T: Default + PartialEq>(
            planes: [&[T]; 3],
            block: usize,
            zeros: &mut [[i64; BLOCK_SIZE]; 3],
        ) {
            let start = block * BLOCK_SIZE;
            for (plane, counts) in planes.into_iter().zip(zeros) {
                for (coefficient, count) in plane[start..start + BLOCK_SIZE]
                    .iter()
                    .zip(counts.iter_mut())
                {
                    *count += i64::from(*coefficient == T::default());
                }
                // LLF/DC gets -1 after each accepted block, not a measured count.
                counts[0] = -1;
            }
        }
        match self {
            Self::I16(planes) => count(planes, block, zeros),
            Self::I32(planes) => count(planes, block, zeros),
        }
    }
}

/// Validated full-frame DCT8 ACImage view. Groups are raster ordered in 32x32
/// block regions; all three planes use the source's per-group packed layout.
/// Callers must supply actual pass coefficients after CfL/progressive splitting,
/// including source zero padding. This type does not synthesize those stages.
#[derive(Debug)]
pub struct Dct8Image<'a> {
    width_blocks: usize,
    height_blocks: usize,
    groups_x: usize,
    groups: &'a [Dct8Group<'a>],
}

fn validate_geometry(width: usize, height: usize) -> Result<(), OrderError> {
    let blocks = width
        .checked_mul(height)
        .ok_or(OrderError::InvalidGeometry)?;
    // Zero frequencies accumulate in signed 64-bit counters.
    if blocks == 0 || i64::try_from(blocks).is_err() {
        return Err(OrderError::InvalidGeometry);
    }
    Ok(())
}

impl<'a> Dct8Image<'a> {
    pub fn new(
        width_blocks: usize,
        height_blocks: usize,
        groups: &'a [Dct8Group<'a>],
    ) -> Result<Self, OrderError> {
        validate_geometry(width_blocks, height_blocks)?;
        let groups_x = width_blocks.div_ceil(GROUP_DIM_IN_BLOCKS);
        let groups_y = height_blocks.div_ceil(GROUP_DIM_IN_BLOCKS);
        if groups.len()
            != groups_x
                .checked_mul(groups_y)
                .ok_or(OrderError::InvalidGeometry)?
        {
            return Err(OrderError::InvalidGroupCount);
        }
        let image = Self {
            width_blocks,
            height_blocks,
            groups_x,
            groups,
        };
        let is_i16 = matches!(groups[0], Dct8Group::I16(_));
        for (i, group) in groups.iter().enumerate() {
            if matches!(group, Dct8Group::I16(_)) != is_i16 {
                return Err(OrderError::MixedCoefficientTypes);
            }
            let required = image.group_blocks(i) * BLOCK_SIZE;
            if group
                .lengths()
                .into_iter()
                .any(|len| len < required || !len.is_multiple_of(BLOCK_SIZE))
            {
                return Err(OrderError::InvalidCoefficientStorage);
            }
        }
        Ok(image)
    }

    fn group_blocks(&self, group: usize) -> usize {
        let x = (group % self.groups_x) * GROUP_DIM_IN_BLOCKS;
        let y = (group / self.groups_x) * GROUP_DIM_IN_BLOCKS;
        (self.width_blocks - x).min(GROUP_DIM_IN_BLOCKS)
            * (self.height_blocks - y).min(GROUP_DIM_IN_BLOCKS)
    }

    pub fn dimensions_in_blocks(&self) -> (usize, usize) {
        (self.width_blocks, self.height_blocks)
    }
}

/// DCT8 specialization of ComputeUsedOrders: (present order bits, candidate
/// custom order bits). The frozen Falcon branch returns (1,1), despite its
/// comment saying no reordering; it also precedes the small-image check.
pub fn source_used_orders(
    speed: SpeedTier,
    width_blocks: usize,
    height_blocks: usize,
) -> Result<(u16, u16), OrderError> {
    validate_geometry(width_blocks, height_blocks)?;
    let customize = speed >= SpeedTier::Falcon || width_blocks >= 5 || height_blocks >= 5;
    Ok((DCT8_ORDER_MASK, u16::from(customize)))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrderPolicy {
    /// ComputeUsedOrders policy, then ComputeCoeffOrder.
    #[default]
    Source,
    /// Source current_used_orders=0 (also the direct-plan override).
    Default,
    /// Source current_used_orders=1, explicitly bypassing the small-image gate.
    /// Source half-block sampling still applies at Squirrel and faster.
    Adaptive,
}

struct Sampler {
    state: [u64; 2],
    threshold: u64,
}
impl Sampler {
    fn new(speed: SpeedTier) -> Self {
        Self {
            state: [0x94d0_49bb_1331_11eb, 0xbf58_476d_1ce4_e5b9],
            threshold: if speed >= SpeedTier::Squirrel {
                0x7fff_ffff
            } else {
                0xffff_ffff
            },
        }
    }

    fn accept(&mut self) -> bool {
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        let bits = s1.wrapping_add(s0);
        self.state[0] = s0;
        s1 ^= s1 << 23;
        s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
        self.state[1] = s1;
        (bits >> 32) <= self.threshold
    }
}

fn zero_counts(image: &Dct8Image<'_>, speed: SpeedTier) -> [[i64; BLOCK_SIZE]; 3] {
    let mut zeros = [[0; BLOCK_SIZE]; 3];
    let mut sampler = Sampler::new(speed);
    for (index, group) in image.groups.iter().enumerate() {
        let mut packed_block = 0;
        for _ in 0..image.group_blocks(index) {
            if !sampler.accept() {
                continue;
            }
            // Source ac_offset advances ONLY for accepted samples. Preserve the
            // consumed packed prefix, not a different spatial sampling scheme.
            group.count_block(packed_block, &mut zeros);
            packed_block += 1;
        }
    }
    zeros
}

fn sort_order(zeros: &[i64; BLOCK_SIZE]) -> [u32; BLOCK_SIZE] {
    let mut positions = DEFAULT_DCT8_ORDER;
    positions.sort_unstable_by_key(|&position| {
        // Source int64 -> f32, multiplication and addition; no exact-integer
        // replacement or f64 promotion. -1 for DC truncates from -0.025 to 0.
        let quantized = (zeros[position as usize] as f32 * 0.125f32 + 0.1f32) as u64;
        // Keep the tie-break independent of the frequency's integer width.
        (quantized, DEFAULT_DCT8_ORDER_LUT[position as usize])
    });
    positions
}

/// One pass's complete DCT8 order state in JXL channel order. Orders index the
/// transposed block. Used bit 1 signals ALL THREE permutations, even if only one
/// channel differs. Once a caller commits this bucket, retain it for later areas.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dct8Orders {
    positions: [[u32; BLOCK_SIZE]; 3],
    used_orders: u16,
}

impl Default for Dct8Orders {
    fn default() -> Self {
        Self {
            positions: [DEFAULT_DCT8_ORDER; 3],
            used_orders: 0,
        }
    }
}

impl Dct8Orders {
    pub fn compute(image: &Dct8Image<'_>, speed: SpeedTier, policy: OrderPolicy) -> Self {
        let customize = match policy {
            OrderPolicy::Source => {
                source_used_orders(speed, image.width_blocks, image.height_blocks)
                    .expect("validated geometry")
                    .1
                    != 0
            }
            OrderPolicy::Default => false,
            OrderPolicy::Adaptive => true,
        };
        if !customize {
            return Self::default();
        }
        let positions = zero_counts(image, speed).map(|counts| sort_order(&counts));
        let used_orders = u16::from(positions.iter().any(|p| p != &DEFAULT_DCT8_ORDER));
        Self {
            positions,
            used_orders,
        }
    }

    /// Checked external/retained plan. `transmit=true` may intentionally retain
    /// the source's already-committed bit even if every order is default.
    pub fn from_positions(
        positions: [[u32; BLOCK_SIZE]; 3],
        transmit: bool,
    ) -> Result<Self, OrderError> {
        for order in &positions {
            validate_permutation(order)?;
            if order[0] != 0 {
                return Err(OrderError::DcNotFirst);
            }
            if !transmit && order != &DEFAULT_DCT8_ORDER {
                return Err(OrderError::UnsignaledCustomOrder);
            }
        }
        Ok(Self {
            positions,
            used_orders: u16::from(transmit),
        })
    }

    pub fn positions(&self) -> &[[u32; BLOCK_SIZE]; 3] {
        &self.positions
    }
    pub fn used_orders(&self) -> u16 {
        self.used_orders
    }

    /// EncodeCoeffOrders' token preparation only. No tokens/histograms when the
    /// bit is zero; otherwise concatenate channel 0/1/2 with DC skip=1.
    pub fn tokens(&self) -> Vec<Token> {
        if self.used_orders == 0 {
            return Vec::new();
        }
        let mut tokens = Vec::with_capacity(3 * BLOCK_SIZE);
        for order in self.positions {
            let zigzag = order.map(|position| DEFAULT_DCT8_ORDER_LUT[position as usize]);
            append_permutation(&zigzag, 1, &mut tokens).expect("validated order and DC prefix");
        }
        tokens
    }
}

/// coeff_order.cc::CoeffOrderContext: contexts use HybridUint000, independently
/// of whichever HybridUint config the later entropy model selects for values.
pub fn coefficient_order_context(value: u32) -> usize {
    let config = HybridUintConfig::new(0, 0, 0).expect("constant configuration");
    config
        .encode(value)
        .token
        .min(PERMUTATION_CONTEXTS as u32 - 1) as usize
}

fn validate_permutation(order: &[u32]) -> Result<(), OrderError> {
    if order.is_empty() || order.len() > MAX_PERMUTATION_SIZE {
        return Err(OrderError::InvalidPermutationSize);
    }
    let mut seen = 0u64;
    for (index, &value) in order.iter().enumerate() {
        if value as usize >= order.len() || seen & (1u64 << value) != 0 {
            return Err(OrderError::InvalidPermutationValue { index, value });
        }
        seen |= 1u64 << value;
    }
    Ok(())
}

/// Source ComputeLehmerCode with a Fenwick tree, bounded to this DCT8 slice.
/// Explicit range/duplicate checks precede indexing; size need not be a power of two.
pub fn lehmer_code(order: &[u32]) -> Result<Vec<u32>, OrderError> {
    validate_permutation(order)?;
    let mut tree = [0u32; MAX_PERMUTATION_SIZE + 1];
    let mut code = Vec::with_capacity(order.len());
    for &symbol in order {
        let mut penalty = 0;
        let mut i = symbol as usize + 1;
        while i != 0 {
            penalty += tree[i];
            i &= i - 1;
        }
        code.push(symbol - penalty);
        i = symbol as usize + 1;
        while i <= order.len() {
            tree[i] += 1;
            i += i & i.wrapping_neg();
        }
    }
    Ok(code)
}

fn append_permutation(
    order: &[u32],
    skip: usize,
    tokens: &mut Vec<Token>,
) -> Result<(), OrderError> {
    if skip > order.len() {
        return Err(OrderError::InvalidSkip);
    }
    let code = lehmer_code(order)?;
    if code[..skip].iter().any(|&digit| digit != 0) {
        return Err(OrderError::NonIdentityPrefix);
    }
    let mut end = code.len();
    while end > skip && code[end - 1] == 0 {
        end -= 1;
    }
    tokens.push(Token::literal(
        coefficient_order_context(order.len() as u32),
        (end - skip) as u32,
    ));
    let mut last = 0;
    for &digit in &code[skip..end] {
        tokens.push(Token::literal(coefficient_order_context(last), digit));
        last = digit;
    }
    Ok(())
}

/// Raw permutation entries are ranks in 0..size, NOT coefficient positions.
/// Dct8Orders::tokens performs the natural-order LUT conversion before calling
/// this. Skipped leading Lehmer digits must be zero because the decoder omits them.
pub fn tokenize_permutation(order: &[u32], skip: usize) -> Result<Vec<Token>, OrderError> {
    let mut tokens = Vec::with_capacity(order.len().min(MAX_PERMUTATION_SIZE) + 1);
    append_permutation(order, skip, &mut tokens)?;
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn independent_context(value: u32) -> usize {
        if value == 0 {
            0
        } else {
            (u32::BITS - value.leading_zeros()).min(7) as usize
        }
    }

    // Independent O(n^2) factorial-base oracle, with no Fenwick tree.
    fn reference_lehmer(order: &[u32]) -> Vec<u32> {
        let mut remaining: Vec<_> = (0..order.len() as u32).collect();
        order
            .iter()
            .map(|value| {
                let index = remaining
                    .iter()
                    .position(|candidate| candidate == value)
                    .unwrap();
                remaining.remove(index);
                index as u32
            })
            .collect()
    }

    fn decode_tokens(tokens: &[Token], size: usize, skip: usize) -> (Vec<u32>, usize) {
        assert_eq!(tokens[0].context, independent_context(size as u32));
        assert!(!tokens[0].is_length);
        let end = skip + tokens[0].value as usize;
        assert!(end <= size);
        let mut code = vec![0; size];
        let mut last = 0;
        let mut consumed = 1;
        for digit in &mut code[skip..end] {
            let token = tokens[consumed];
            assert_eq!(token.context, independent_context(last));
            assert!(!token.is_length);
            *digit = token.value;
            last = *digit;
            consumed += 1;
        }
        let mut remaining: Vec<_> = (0..size as u32).collect();
        let order = code
            .into_iter()
            .map(|digit| {
                assert!((digit as usize) < remaining.len());
                remaining.remove(digit as usize)
            })
            .collect();
        (order, consumed)
    }

    #[test]
    fn default_order_matches_diagonal_math_and_transposed_addresses() {
        let mut positions: Vec<u32> = (0..64).collect();
        positions.sort_by_key(|&index| {
            let x = index % 8;
            let y = index / 8;
            (x + y, if (x + y).is_multiple_of(2) { x } else { y })
        });
        assert_eq!(DEFAULT_DCT8_ORDER.as_slice(), positions);
        assert_eq!(
            &DEFAULT_DCT8_ORDER[..10],
            &[0, 1, 8, 16, 9, 2, 3, 10, 17, 24]
        );
        for (rank, &position) in DEFAULT_DCT8_ORDER.iter().enumerate() {
            assert_eq!(DEFAULT_DCT8_ORDER_LUT[position as usize], rank as u32);
        }
        let transposed: [u32; 64] = std::array::from_fn(|i| (8 * (i % 8) + i / 8) as u32);
        let jpeg_positions = DEFAULT_DCT8_ORDER.map(|position| transposed[position as usize]);
        assert_eq!(&jpeg_positions[..6], &[0, 8, 1, 2, 9, 16]);
    }

    #[test]
    fn source_speed_and_small_image_masks_follow_code_not_comment() {
        for tier in -1i8..=9 {
            let speed = SpeedTier::try_from(tier).unwrap();
            for (width, height) in [(1, 1), (4, 4), (5, 1), (1, 5), (33, 33)] {
                let customize = tier >= 7 || width >= 5 || height >= 5;
                assert_eq!(
                    source_used_orders(speed, width, height).unwrap(),
                    (1, u16::from(customize))
                );
            }
        }
        assert!(source_used_orders(SpeedTier::Squirrel, 0, 1).is_err());
        assert!(source_used_orders(SpeedTier::Squirrel, usize::MAX, 2).is_err());
    }

    #[test]
    fn zero_quantization_preserves_zigzag_ties_dc_and_source_f32_rounding() {
        let mut counts = [16; 64];
        counts[0] = -1;
        for (rank, count) in [(1, 7), (2, 0), (3, 8), (4, 15), (63, 0)] {
            counts[DEFAULT_DCT8_ORDER[rank] as usize] = count;
        }
        let order = sort_order(&counts);
        let expected_ranks = [0, 1, 2, 63, 3, 4];
        assert_eq!(
            &order[..6],
            &expected_ranks.map(|rank| DEFAULT_DCT8_ORDER[rank])
        );
        assert_eq!(&order[6..], &DEFAULT_DCT8_ORDER[5..63]);
        // At this magnitude the source int64->f32 conversion loses the -4.
        // Integer division or f64 arithmetic would incorrectly move position 63.
        let mut counts = [1i64 << 27; 64];
        counts[0] = -1;
        counts[63] -= 4;
        assert_eq!(sort_order(&counts), DEFAULT_DCT8_ORDER);
        assert_eq!(sort_order(&[-1; 64]), DEFAULT_DCT8_ORDER);
    }

    #[test]
    fn adaptive_i16_i32_channels_and_default_choices_reach_real_tokens() {
        let mut planes = [
            vec![0i16; 16 * 64],
            vec![0i16; 16 * 64],
            vec![1i16; 16 * 64],
        ];
        for block in 0..16 {
            planes[0][block * 64 + 63] = -7;
            planes[1][block * 64 + 62] = 123;
        }
        let groups = [Dct8Group::I16([&planes[0], &planes[1], &planes[2]])];
        let small = Dct8Image::new(4, 4, &groups).unwrap();
        let wide = Dct8Image::new(8, 2, &groups).unwrap();
        assert_eq!(
            Dct8Orders::compute(&small, SpeedTier::Tortoise, OrderPolicy::Source),
            Dct8Orders::default()
        );
        let adaptive = Dct8Orders::compute(&small, SpeedTier::Tortoise, OrderPolicy::Adaptive);
        assert_eq!(
            adaptive,
            Dct8Orders::compute(&wide, SpeedTier::Tortoise, OrderPolicy::Source)
        );
        assert_eq!(
            Dct8Orders::compute(&wide, SpeedTier::Tortoise, OrderPolicy::Default),
            Dct8Orders::default()
        );
        assert_eq!(&adaptive.positions()[0][..2], &[0, 63]);
        assert_eq!(&adaptive.positions()[1][..2], &[0, 62]);
        assert_eq!(adaptive.positions()[2], DEFAULT_DCT8_ORDER);
        assert_eq!(adaptive.used_orders(), 1);
        let wide_planes = planes
            .each_ref()
            .map(|plane| plane.iter().map(|&n| i32::from(n)).collect::<Vec<_>>());
        let wide_groups = [Dct8Group::I32([
            &wide_planes[0],
            &wide_planes[1],
            &wide_planes[2],
        ])];
        let image32 = Dct8Image::new(4, 4, &wide_groups).unwrap();
        assert_eq!(
            adaptive,
            Dct8Orders::compute(&image32, SpeedTier::Tortoise, OrderPolicy::Adaptive)
        );
        let tokens = adaptive.tokens();
        let mut offset = 0;
        for order in adaptive.positions() {
            let (ranks, consumed) = decode_tokens(&tokens[offset..], 64, 1);
            let decoded: Vec<_> = ranks
                .into_iter()
                .map(|rank| DEFAULT_DCT8_ORDER[rank as usize])
                .collect();
            assert_eq!(order.as_slice(), decoded);
            offset += consumed;
        }
        assert_eq!(offset, tokens.len());
        // These are ordinary existing entropy::stream::Token values, not a new wire type.
        let streams = [tokens];
        let model = crate::entropy::stream::Model::build(
            &streams,
            PERMUTATION_CONTEXTS,
            false,
            crate::entropy::stream::Coding::Source,
        )
        .unwrap();
        let mut writer = crate::bit_writer::BitWriter::new(65536);
        model.write_header(&mut writer).unwrap();
        model.write_stream(&streams[0], &mut writer).unwrap();
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn sampler_seed_wrapping_threshold_and_packed_prefix_are_frozen() {
        let mut sampler = Sampler::new(SpeedTier::Squirrel);
        let actual: Vec<_> = (0..16).map(|_| sampler.accept()).collect();
        assert_eq!(
            actual,
            [
                true, false, false, true, true, false, true, true, true, false, true, false, true,
                true, true, true
            ]
        );
        let mut all = Sampler::new(SpeedTier::Kitten);
        assert!((0..4096).all(|_| all.accept()));
        let mut exact = Sampler {
            state: [0x7fff_ffff_0000_0000, 0],
            threshold: 0x7fff_ffff,
        };
        assert!(exact.accept());
        let mut over = Sampler {
            state: [0x8000_0000_0000_0000, 0],
            threshold: 0x7fff_ffff,
        };
        assert!(!over.accept());
        let mut coefficients = vec![1i16; 16 * 64];
        for block in 0..16 {
            coefficients[block * 64 + block + 1] = 0;
        }
        let groups = [Dct8Group::I16([&coefficients; 3])];
        let image = Dct8Image::new(16, 1, &groups).unwrap();
        let counts = zero_counts(&image, SpeedTier::Squirrel);
        for channel in counts {
            assert_eq!(channel[0], -1);
            assert_eq!(&channel[1..12], &[1; 11]); // Eleven accepted draws consume blocks 0..10.
            assert!(channel[12..].iter().all(|&n| n == 0));
        }
    }

    #[test]
    fn sampling_state_spans_groups_but_packed_offsets_restart_at_each_edge_group() {
        let buffers: Vec<Vec<i32>> = [1024, 32, 32, 1]
            .into_iter()
            .enumerate()
            .map(|(group, blocks)| {
                let mut values = vec![1; blocks * 64];
                for block in 0..blocks {
                    values[block * 64 + group + 1] = 0;
                }
                values
            })
            .collect();
        let groups: Vec<_> = buffers
            .iter()
            .map(|values| Dct8Group::I32([values.as_slice(); 3]))
            .collect();
        let image = Dct8Image::new(33, 33, &groups).unwrap();
        assert_eq!(image.dimensions_in_blocks(), (33, 33));
        for counts in zero_counts(&image, SpeedTier::Squirrel) {
            assert_eq!(&counts[..5], &[-1, 530, 14, 16, 0]);
            assert!(counts[5..].iter().all(|&n| n == 0));
        }
        for counts in zero_counts(&image, SpeedTier::Tortoise) {
            assert_eq!(&counts[..5], &[-1, 1024, 32, 32, 1]);
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn order_geometry_keeps_numeric_bounds_without_an_extra_block_cap() {
        assert!(validate_geometry((1usize << 34) + 1, 1).is_ok());
        assert!(validate_geometry(i64::MAX as usize, 1).is_ok());
        assert!(validate_geometry(i64::MAX as usize + 1, 1).is_err());
        assert!(validate_geometry(usize::MAX, 2).is_err());
        let mut zeros = [i64::MAX; BLOCK_SIZE];
        zeros[0] = -1;
        zeros[1] = i64::MAX / 2;
        let order = sort_order(&zeros);
        assert_eq!(&order[..2], &[0, 1]);
        validate_permutation(&order).unwrap();
    }

    #[test]
    fn geometry_type_and_storage_bounds_are_checked_before_counting() {
        let one = [0i16; 64];
        let short = [0i16; 63];
        let full = vec![0i16; 1024 * 64];
        let group = [Dct8Group::I16([&one; 3])];
        assert!(Dct8Image::new(0, 1, &group).is_err());
        assert!(Dct8Image::new(usize::MAX, 2, &group).is_err());
        assert!(Dct8Image::new(1, 1, &[]).is_err());
        assert!(Dct8Image::new(33, 1, &group).is_err());
        assert!(Dct8Image::new(2, 1, &group).is_err());
        let group = [Dct8Group::I16([&short; 3])];
        assert!(Dct8Image::new(1, 1, &group).is_err());
        let group = [Dct8Group::I16([&full; 3])];
        assert!(Dct8Image::new(1, 1, &group).is_ok()); // Source allocation can exceed edge geometry.
        let other = [0i32; 64];
        let mixed = [group[0], Dct8Group::I32([&other; 3])];
        assert!(matches!(
            Dct8Image::new(33, 1, &mixed),
            Err(OrderError::MixedCoefficientTypes)
        ));
        let mut non_block = full;
        non_block.push(0);
        let group = [Dct8Group::I16([&non_block; 3])];
        assert!(Dct8Image::new(1, 1, &group).is_err());
    }

    #[test]
    fn lehmer_fenwick_matches_independent_rank_removal_at_every_size() {
        assert_eq!(lehmer_code(&[4, 1, 3, 0, 2]).unwrap(), [4, 1, 2, 0, 0]);
        let mut rng = 0x62f1_0badu32;
        for size in 1..=64 {
            let mut order: Vec<_> = (0..size as u32).collect();
            for _ in 0..12 {
                for i in (1..size).rev() {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    order.swap(i, rng as usize % (i + 1));
                }
                assert_eq!(lehmer_code(&order).unwrap(), reference_lehmer(&order));
                let tokens = tokenize_permutation(&order, 0).unwrap();
                let (decoded, consumed) = decode_tokens(&tokens, size, 0);
                assert_eq!(decoded, order);
                assert_eq!(consumed, tokens.len());
            }
        }
    }

    #[test]
    fn permutation_contexts_and_trimmed_token_goldens() {
        for value in [
            0,
            1,
            2,
            3,
            4,
            7,
            8,
            15,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            u32::MAX,
        ] {
            assert_eq!(coefficient_order_context(value), independent_context(value));
        }
        assert_eq!(
            tokenize_permutation(&[0, 3, 1, 2], 1).unwrap(),
            [Token::literal(3, 1), Token::literal(0, 2)]
        );
        assert_eq!(
            tokenize_permutation(&[0, 4, 2, 6, 1, 3, 5, 7], 1).unwrap(),
            [
                Token::literal(4, 3),
                Token::literal(0, 3),
                Token::literal(2, 1),
                Token::literal(1, 3),
            ]
        );
        let identity: Vec<_> = (0..64).collect();
        for skip in [0, 1, 63, 64] {
            assert_eq!(
                tokenize_permutation(&identity, skip).unwrap(),
                [Token::literal(7, 0)]
            );
        }
        assert_eq!(
            tokenize_permutation(&[0], 1).unwrap(),
            [Token::literal(1, 0)]
        );
    }

    #[test]
    fn one_custom_channel_signals_all_three_and_retained_default_bit_is_supported() {
        let mut positions = [DEFAULT_DCT8_ORDER; 3];
        positions[0].swap(1, 2);
        let orders = Dct8Orders::from_positions(positions, true).unwrap();
        assert_eq!(
            orders.tokens(),
            [
                Token::literal(7, 1),
                Token::literal(0, 1),
                Token::literal(7, 0),
                Token::literal(7, 0)
            ]
        );
        assert_eq!(Dct8Orders::default().used_orders(), 0);
        assert!(Dct8Orders::default().tokens().is_empty());
        let retained = Dct8Orders::from_positions([DEFAULT_DCT8_ORDER; 3], true).unwrap();
        assert_eq!(retained.used_orders(), 1);
        assert_eq!(retained.tokens(), vec![Token::literal(7, 0); 3]);
    }

    #[test]
    fn malformed_permutations_skip_and_order_flags_are_rejected_atomically() {
        for order in [vec![], vec![0; 65], vec![0, 0], vec![0, 2], vec![u32::MAX]] {
            assert!(lehmer_code(&order).is_err());
            let mut tokens = vec![Token::literal(3, 5)];
            assert!(append_permutation(&order, 0, &mut tokens).is_err());
            assert_eq!(tokens, [Token::literal(3, 5)]);
        }
        assert_eq!(
            tokenize_permutation(&[0, 1], 3),
            Err(OrderError::InvalidSkip)
        );
        assert_eq!(
            tokenize_permutation(&[1, 0, 2], 1),
            Err(OrderError::NonIdentityPrefix)
        );
        let mut positions = [DEFAULT_DCT8_ORDER; 3];
        positions[0].swap(0, 1);
        assert_eq!(
            Dct8Orders::from_positions(positions, true),
            Err(OrderError::DcNotFirst)
        );
        positions[0] = DEFAULT_DCT8_ORDER;
        positions[0].swap(1, 2);
        assert_eq!(
            Dct8Orders::from_positions(positions, false),
            Err(OrderError::UnsignaledCustomOrder)
        );
    }
}
