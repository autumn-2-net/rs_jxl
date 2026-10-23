// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Scalar context/token oracles, coefficient replay and block-context wire tests.

use super::*;

#[derive(Clone)]
struct Fixture {
    width: usize,
    height: usize,
    horizontal: [u8; 3],
    vertical: [u8; 3],
    coefficients: [Vec<i32>; 3],
    orders: [[u32; 64]; 3],
    qdc: Vec<u8>,
    qf: Vec<i32>,
}

impl Fixture {
    fn new(width: usize, height: usize, horizontal: [u8; 3], vertical: [u8; 3]) -> Self {
        Self {
            width,
            height,
            horizontal,
            vertical,
            coefficients: std::array::from_fn(|c| {
                vec![0; width.div_ceil(1 << horizontal[c]) * height.div_ceil(1 << vertical[c]) * 64]
            }),
            orders: [std::array::from_fn(|i| i as u32); 3],
            qdc: vec![0; width * height],
            qf: vec![1; width * height],
        }
    }

    fn input(&self) -> Dct8Input<'_> {
        Dct8Input {
            width_blocks: self.width,
            height_blocks: self.height,
            planes: std::array::from_fn(|c| Dct8Plane {
                coefficients: &self.coefficients[c],
                width_blocks: self.width.div_ceil(1 << self.horizontal[c]),
                height_blocks: self.height.div_ceil(1 << self.vertical[c]),
                horizontal_shift: self.horizontal[c],
                vertical_shift: self.vertical[c],
            }),
            orders: &self.orders,
            qdc: &self.qdc,
            qf: &self.qf,
        }
    }

    fn limits(&self) -> TokenizeLimits {
        TokenizeLimits {
            max_tokens: self.coefficients.iter().map(Vec::len).sum(),
            max_scratch_bytes: self.input().planes.iter().map(|p| p.width_blocks).sum(),
        }
    }

    fn tokens(&self, model: BlockContextModel<'_>) -> Vec<Token> {
        tokenize_dct8(&self.input(), model, self.limits()).unwrap()
    }

    // Enumerate component blocks first, then sort by full-resolution position
    // and source channel priority. No shared subsampling-gate implementation.
    fn events(&self) -> Vec<(usize, usize, usize, usize, usize)> {
        let mut events = Vec::new();
        for (c, coefficients) in self.coefficients.iter().enumerate() {
            let width = self.width.div_ceil(1 << self.horizontal[c]);
            for block in 0..coefficients.len() / 64 {
                let x = (block % width) * (1 << self.horizontal[c]);
                let y = (block / width) * (1 << self.vertical[c]);
                let priority = [1, 0, 2][c];
                events.push((y, x, priority, c, block));
            }
        }
        events.sort_unstable();
        events
    }
}

fn density_oracle(nonzeros: usize, rank: usize, previous: bool) -> usize {
    let base = match nonzeros {
        1 => 0,
        2 => 31,
        3..=4 => 62,
        5..=8 => 93,
        9..=12 => 123,
        13..=20 => 152,
        21..=32 => 180,
        33..=63 => 206,
        _ => panic!("bad oracle nonzero count"),
    };
    let frequency = match rank {
        1..=15 => rank - 1,
        16..=31 => 15 + (rank - 16) / 2,
        32..=63 => 23 + (rank - 32) / 4,
        _ => panic!("bad oracle rank"),
    };
    2 * (base + frequency) + usize::from(previous)
}

fn token_oracle(
    fixture: &Fixture,
    map: &[u8],
    num_contexts: usize,
    num_dc: usize,
    qf_thresholds: &[u32],
) -> Vec<Token> {
    let counts: [Vec<usize>; 3] = std::array::from_fn(|c| {
        fixture.coefficients[c]
            .chunks_exact(64)
            .map(|block| block[1..].iter().filter(|&&v| v != 0).count())
            .collect()
    });
    let mut output = Vec::new();
    for (by, bx, _, c, block_index) in fixture.events() {
        let plane_width = fixture.width.div_ceil(1 << fixture.horizontal[c]);
        let x = block_index % plane_width;
        let y = block_index / plane_width;
        let prediction = match (x, y) {
            (0, 0) => 32,
            (0, _) => counts[c][block_index - plane_width],
            (_, 0) => counts[c][block_index - 1],
            _ => (counts[c][block_index - plane_width] + counts[c][block_index - 1]).div_ceil(2),
        };
        let context_channel = match c {
            0 => 1,
            1 => 0,
            _ => 2,
        };
        let dc = usize::from(fixture.qdc[by * fixture.width + bx]);
        let qf = fixture.qf[by * fixture.width + x] as u32;
        let qf_bucket = qf_thresholds.iter().filter(|&&t| t < qf).count();
        let map_index =
            context_channel * 13 * (qf_thresholds.len() + 1) * num_dc + qf_bucket * num_dc + dc;
        let block_context = usize::from(map[map_index]);
        let bucket = if prediction <= 7 {
            prediction
        } else {
            (prediction / 2) + 4
        };
        let nonzeros = counts[c][block_index];
        output.push(Token::literal(
            bucket * num_contexts + block_context,
            nonzeros as u32,
        ));
        let block = &fixture.coefficients[c][block_index * 64..(block_index + 1) * 64];
        let scanned: Vec<i32> = fixture.orders[c]
            .iter()
            .map(|&i| block[i as usize])
            .collect();
        let last = (1..64).rev().find(|&k| scanned[k] != 0).unwrap_or(0);
        for k in 1..=last {
            let remaining = scanned[k..].iter().filter(|&&v| v != 0).count();
            let previous = if k == 1 {
                nonzeros < 5
            } else {
                scanned[k - 1] != 0
            };
            let context =
                num_contexts * 37 + block_context * 458 + density_oracle(remaining, k, previous);
            let value = i64::from(scanned[k]);
            let packed = if value >= 0 {
                2 * value
            } else {
                -2 * value - 1
            };
            output.push(Token::literal(context, packed as u32));
        }
    }
    output
}

fn replay_ac(fixture: &Fixture, tokens: &[Token]) {
    let mut decoded: [Vec<i32>; 3] =
        std::array::from_fn(|c| vec![0; fixture.coefficients[c].len()]);
    let mut cursor = 0;
    for (_, _, _, c, block) in fixture.events() {
        let count = tokens[cursor].value;
        assert!(count < 64);
        cursor += 1;
        let mut seen = 0;
        for &coefficient_index in &fixture.orders[c][1..] {
            if seen == count {
                break;
            }
            let value = tokens[cursor].value;
            assert!(!tokens[cursor].is_length);
            let signed = if value & 1 == 0 {
                i64::from(value / 2)
            } else {
                -i64::from(value / 2) - 1
            };
            let coefficient = i32::try_from(signed).unwrap();
            decoded[c][block * 64 + coefficient_index as usize] = coefficient;
            seen += u32::from(coefficient != 0);
            cursor += 1;
        }
        assert_eq!(seen, count);
    }
    assert_eq!(cursor, tokens.len());
    for (got, original) in decoded.iter().zip(&fixture.coefficients) {
        for (got, original) in got.chunks_exact(64).zip(original.chunks_exact(64)) {
            assert_eq!(&got[1..], &original[1..]);
            assert_eq!(got[0], 0);
        }
    }
}

#[test]
fn zero_density_tables_match_independent_piecewise_oracle_exhaustively() {
    let mut maximum = 0;
    let mut valid_maximum = 0;
    for nonzeros in 1..64 {
        for rank in 1..64 {
            for previous in [false, true] {
                let actual = zero_density_context(nonzeros, rank, previous).unwrap();
                assert_eq!(actual, density_oracle(nonzeros, rank, previous));
                maximum = maximum.max(actual);
                if nonzeros + rank <= 64 {
                    valid_maximum = valid_maximum.max(actual);
                }
            }
        }
    }
    assert_eq!(maximum + 1, ZERO_DENSITY_LIMIT);
    assert_eq!(valid_maximum + 1, ZERO_DENSITY_CONTEXTS);
    for (nz, k) in [(0, 1), (1, 0), (64, 1), (1, 64), (usize::MAX, 2)] {
        assert!(zero_density_context(nz, k, false).is_err());
    }
}

#[test]
fn default_context_map_and_nonzero_clamping_match_source() {
    let model = BlockContextModel::default();
    assert!(model.is_default());
    assert_eq!(model.num_ac_contexts(), 7425);
    assert_eq!(model.num_contexts(), 15);
    assert_eq!(model.num_dc_contexts(), 1);
    let rows = [
        [7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14],
        [0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6],
        [7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14],
    ];
    for (channel, row) in rows.iter().enumerate() {
        for (order, &context) in row.iter().enumerate() {
            assert_eq!(model.context(0, u32::MAX, order, channel).unwrap(), context);
        }
    }
    for prediction in 0..=1008u32 {
        let clipped = prediction.min(64) as usize;
        let bucket = if clipped <= 7 {
            clipped
        } else {
            clipped / 2 + 4
        };
        for context in 0..15 {
            assert_eq!(
                model.nonzero_context(prediction, context).unwrap(),
                bucket * 15 + context
            );
            assert_eq!(
                model.zero_density_offset(context).unwrap(),
                555 + 458 * context
            );
        }
    }
    assert_eq!(model.nonzero_context(u32::MAX, 14).unwrap(), 554);
    assert!(model.context(1, 1, 0, 0).is_err());
    assert!(model.context(0, 1, 13, 0).is_err());
    assert!(model.context(0, 1, 0, 3).is_err());
    assert!(model.nonzero_context(1, 15).is_err());
    assert!(model.zero_density_offset(15).is_err());
}

#[test]
fn qdc_and_qf_thresholds_use_strict_comparison_and_source_channel_mixing() {
    let thresholds = [&[5, -2][..], &[3][..], &[7][..]];
    let qf = [10, 3, 10];
    let map: Vec<u8> = (0..39 * 12 * 4).map(|i| (i % 16) as u8).collect();
    let model = BlockContextModel::new(thresholds, &qf, &map, 16).unwrap();
    assert_eq!(model.num_dc_contexts(), 12);
    assert_eq!(model.dc_context([-2, 3, 7]), 0);
    assert_eq!(model.dc_context([5, 4, 8]), 7);
    assert_eq!(model.dc_context([6, 4, 8]), 11);
    assert!(!model.is_default());
    for (c, swapped) in [1, 0, 2].into_iter().enumerate() {
        for order in 0..13 {
            for dc in 0..12u8 {
                for (value, qf_index) in [(1, 0), (3, 0), (4, 1), (10, 1), (11, 3)] {
                    let index = ((swapped * 13 + order) * 4 + qf_index) * 12 + usize::from(dc);
                    assert_eq!(
                        model.context(dc, value, order, c).unwrap(),
                        usize::from(map[index])
                    );
                }
            }
        }
    }
}

#[test]
fn context_model_rejects_unrepresentable_dimensions_and_mapping() {
    let default = DEFAULT_CONTEXT_MAP;
    assert!(BlockContextModel::new([&[]; 3], &[], &default, 0).is_err());
    assert!(BlockContextModel::new([&[]; 3], &[], &default, 17).is_err());
    assert!(BlockContextModel::new([&[]; 3], &[], &default[..38], 15).is_err());
    assert!(BlockContextModel::new([&[]; 3], &[], &default, 14).is_err());
    assert!(BlockContextModel::new([&[]; 3], &[], &default, 16).is_err());
    assert!(BlockContextModel::new([&[]; 3], &[], &[1; 39], 2).is_err());
    assert!(BlockContextModel::new([&[0; 16], &[], &[]], &[], &[], 1).is_err());
    assert!(BlockContextModel::new([&[]; 3], &[1; 16], &[], 1).is_err());
    assert!(BlockContextModel::new([&[0; 15]; 3], &[], &[], 1).is_err());
    for qf in [0, 301, u32::MAX] {
        assert!(BlockContextModel::new([&[]; 3], &[qf], &[0; 78], 1).is_err());
    }
    assert!(BlockContextModel::new([&[]; 3], &[300], &[0; 78], 1).is_ok());
    let map: Vec<u8> = (0..39 * 64).map(|i| (i % 16) as u8).collect();
    let model = BlockContextModel::new([&[0; 3]; 3], &[], &map, 16).unwrap();
    assert_eq!(model.num_dc_contexts(), 64);
    assert_eq!(model.num_ac_contexts(), 7920);
}

#[test]
fn native_nonzero_count_matches_scalar_for_every_lane_and_signed_extremes() {
    // Offset the block to exercise unaligned SIMD reads without any padding contract.
    let mut storage = [0i32; 65];
    for position in 0..64 {
        for value in [i32::MIN, -65536, -1, 1, 65536, i32::MAX] {
            storage.fill(0);
            storage[position + 1] = value;
            let block: &[i32; 64] = storage[1..].try_into().unwrap();
            let expected = u32::from(position != 0);
            assert_eq!(count_nonzeros_scalar(block), expected);
            assert_eq!(count_nonzeros(block), expected);
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("sse2") {
                    // SAFETY: feature checked and block contains exactly 64 values.
                    assert_eq!(unsafe { count_nonzeros_sse2(block) }, expected);
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: feature checked and block contains exactly 64 values.
                    assert_eq!(unsafe { count_nonzeros_avx2(block) }, expected);
                }
            }
        }
    }
    for prefix in 0..=64 {
        let block: [i32; 64] = std::array::from_fn(|i| if i < prefix { i32::MIN } else { 0 });
        assert_eq!(count_nonzeros(&block), prefix.saturating_sub(1) as u32);
    }
}

#[test]
fn dc_only_blocks_emit_only_nonzero_counts_in_source_channel_order() {
    let mut fixture = Fixture::new(1, 1, [0; 3], [0; 3]);
    for (plane, dc) in fixture
        .coefficients
        .iter_mut()
        .zip([i32::MIN, 17, i32::MAX])
    {
        plane[0] = dc;
    }
    let actual = fixture.tokens(BlockContextModel::default());
    assert_eq!(
        actual,
        [
            Token::literal(300, 0),
            Token::literal(307, 0),
            Token::literal(307, 0)
        ]
    );
    replay_ac(&fixture, &actual);
}

#[test]
fn signed_values_transposed_orders_and_trailing_zero_termination_replay() {
    let mut fixture = Fixture::new(1, 1, [0; 3], [0; 3]);
    fixture.orders[0] = std::array::from_fn(|i| ((i % 8) * 8 + i / 8) as u32);
    fixture.orders[2][1..].reverse();
    for (plane, order) in fixture.coefficients.iter_mut().zip(&fixture.orders) {
        for (rank, value) in [(1, i32::MIN), (2, i32::MAX), (5, -1), (16, 1), (29, -32768)] {
            plane[order[rank] as usize] = value;
        }
        plane[0] = 12345;
    }
    let actual = fixture.tokens(BlockContextModel::default());
    assert_eq!(actual.len(), 3 * 30);
    assert_eq!(actual[1].value, u32::MAX);
    assert_eq!(actual[2].value, u32::MAX - 1);
    assert_eq!(actual[3].value, 0);
    assert_eq!(
        actual,
        token_oracle(&fixture, &DEFAULT_CONTEXT_MAP, 15, 1, &[])
    );
    replay_ac(&fixture, &actual);

    for plane in &mut fixture.coefficients {
        plane[1..].fill(i32::MIN);
    }
    let dense = fixture.tokens(BlockContextModel::default());
    assert_eq!(dense.len(), fixture.limits().max_tokens);
    assert_eq!(
        dense,
        token_oracle(&fixture, &DEFAULT_CONTEXT_MAP, 15, 1, &[])
    );
    replay_ac(&fixture, &dense);
}

#[test]
fn top_left_prediction_rounds_up_and_resets_at_group_edges() {
    let mut fixture = Fixture::new(3, 2, [0; 3], [0; 3]);
    let counts = [1, 8, 63, 0, 4, 5];
    for plane in &mut fixture.coefficients {
        for (block, count) in plane.chunks_exact_mut(64).zip(counts) {
            block[1..1 + count].fill(-1);
        }
    }
    let actual = fixture.tokens(BlockContextModel::default());
    let predictions = [32, 1, 8, 1, 4, 34];
    let mut cursor = 0;
    for (&count, prediction) in counts.iter().zip(predictions) {
        for block_context in [0, 7, 7] {
            let bucket = if prediction < 8 {
                prediction
            } else {
                4 + prediction / 2
            };
            assert_eq!(
                actual[cursor],
                Token::literal(bucket * 15 + block_context, count as u32)
            );
            cursor += 1 + count;
        }
    }
    assert_eq!(cursor, actual.len());
    assert_eq!(
        actual,
        token_oracle(&fixture, &DEFAULT_CONTEXT_MAP, 15, 1, &[])
    );
    replay_ac(&fixture, &actual);
}

#[test]
fn subsampled_qf_uses_current_full_resolution_row_and_component_x() {
    let mut fixture = Fixture::new(3, 3, [1, 0, 1], [1, 0, 1]);
    fixture.qf = vec![1, 6, 1, 6, 6, 6, 6, 1, 6];
    fixture.qdc = vec![0, 0, 1, 1, 0, 1, 1, 0, 0];
    let map: Vec<u8> = (0..39 * 4).map(|i| (i % 4) as u8).collect();
    let model = BlockContextModel::new([&[], &[0], &[]], &[5], &map, 4).unwrap();
    let actual = fixture.tokens(model);
    let events = fixture.events();
    let context_at = |y, x, c| {
        let index = events
            .iter()
            .position(|&(ey, ex, _, ec, _)| (ey, ex, ec) == (y, x, c))
            .unwrap();
        actual[index].context % 4
    };
    assert_eq!(context_at(0, 2, 0), 3); // qdc[2]=1, qf[sx=1]=6, not qf[2].
    assert_eq!(context_at(0, 2, 1), 1); // Full-resolution Y samples qf[2]=1.
    assert_eq!(context_at(2, 2, 0), 0); // Row by=2, not the component row sy=1.
    assert_eq!(actual, token_oracle(&fixture, &map, 4, 2, &[5]));
}

#[test]
fn scalar_oracle_and_independent_replay_cover_odd_edges_and_all_sampling_modes() {
    let qf_thresholds = [1, 10, 128];
    let map: Vec<u8> = (0..39 * 8).map(|i| ((i * 7 + i / 13) % 16) as u8).collect();
    let model = BlockContextModel::new([&[], &[0], &[]], &qf_thresholds, &map, 16).unwrap();
    let modes = [
        ([0, 0, 0], [0, 0, 0]),
        ([1, 0, 1], [1, 0, 1]),
        ([1, 0, 1], [0, 0, 0]),
        ([0, 0, 0], [1, 0, 1]),
        ([0, 1, 1], [1, 1, 0]),
    ];
    let mut state = 0x1234_5678u32;
    for (width, height) in [(1, 1), (2, 3), (5, 4), (7, 5)] {
        for (h, v) in modes {
            let mut fixture = Fixture::new(width, height, h, v);
            for (c, order) in fixture.orders.iter_mut().enumerate() {
                order[1..].rotate_left(c * 11 + 3);
            }
            for plane in &mut fixture.coefficients {
                for (i, value) in plane.iter_mut().enumerate() {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    *value = if state & 7 < 4 { 0 } else { state as i32 };
                    if i % 64 == 0 {
                        *value = i32::MIN;
                    }
                }
            }
            for (i, (dc, qf)) in fixture.qdc.iter_mut().zip(&mut fixture.qf).enumerate() {
                *dc = (i % 2) as u8;
                *qf = [1, 2, 10, 11, 128, 129, 256][i % 7];
            }
            let actual = fixture.tokens(model);
            assert_eq!(
                actual,
                token_oracle(&fixture, &map, 16, 2, &qf_thresholds),
                "{width}x{height} {h:?} {v:?}"
            );
            assert!(
                actual
                    .iter()
                    .all(|t| !t.is_length && t.context < model.num_ac_contexts())
            );
            replay_ac(&fixture, &actual);
        }
    }
}

#[test]
fn invalid_orders_are_rejected_before_counting_or_allocation() {
    let base = Fixture::new(1, 1, [0; 3], [0; 3]);
    for channel in 0..3 {
        for kind in 0..4 {
            let mut fixture = base.clone();
            match kind {
                0 => fixture.orders[channel][63] = 1,
                1 => fixture.orders[channel][2] = 64,
                2 => fixture.orders[channel][2] = u32::MAX,
                _ => fixture.orders[channel].swap(0, 63),
            }
            let error = tokenize_dct8(
                &fixture.input(),
                BlockContextModel::default(),
                fixture.limits(),
            )
            .unwrap_err();
            assert!(matches!(error, AcTokenError::Invalid(_)));
        }
    }
}

#[test]
fn invalid_grids_sampling_and_resource_limits_fail_without_partial_output() {
    let fixture = Fixture::new(3, 3, [1, 0, 1], [1, 0, 1]);
    let model = BlockContextModel::default();
    let limits = fixture.limits();
    for kind in 0..8 {
        let mut input = fixture.input();
        match kind {
            0 => input.qdc = &fixture.qdc[..8],
            1 => input.qf = &fixture.qf[..8],
            2 => input.planes[0].width_blocks = 3,
            3 => input.planes[2].height_blocks = 3,
            4 => input.planes[0].coefficients = &fixture.coefficients[0][..255],
            5 => input.planes[1].horizontal_shift = 2,
            6 => input.planes[1].vertical_shift = 1,
            _ => input.width_blocks = 0,
        }
        assert!(matches!(
            tokenize_dct8(&input, model, limits),
            Err(AcTokenError::Invalid(_))
        ));
    }
    let mut input = fixture.input();
    input.width_blocks = usize::MAX;
    input.height_blocks = 2;
    assert_eq!(
        tokenize_dct8(&input, model, limits),
        Err(AcTokenError::DimensionOverflow)
    );
    for value in [-1, 0, 257, i32::MAX] {
        let mut bad = fixture.clone();
        bad.qf[8] = value;
        assert!(tokenize_dct8(&bad.input(), model, limits).is_err());
    }
    let mut bad = fixture.clone();
    bad.qdc[8] = 1;
    assert!(tokenize_dct8(&bad.input(), model, limits).is_err());
    assert_eq!(
        tokenize_dct8(
            &fixture.input(),
            model,
            TokenizeLimits {
                max_tokens: limits.max_tokens - 1,
                ..limits
            }
        ),
        Err(AcTokenError::LimitExceeded("token reservation"))
    );
    assert_eq!(
        tokenize_dct8(
            &fixture.input(),
            model,
            TokenizeLimits {
                max_scratch_bytes: limits.max_scratch_bytes - 1,
                ..limits
            }
        ),
        Err(AcTokenError::LimitExceeded("nonzero prediction rows"))
    );
    assert!(tokenize_dct8(&fixture.input(), model, limits).is_ok());
}

#[test]
fn empty_group_requires_no_output_or_scratch_budget() {
    let fixture = Fixture::new(0, 0, [0; 3], [0; 3]);
    assert!(fixture.tokens(BlockContextModel::default()).is_empty());
}

fn written_bits(mut writer: BitWriter) -> Vec<u8> {
    let count = writer.bits_written();
    writer.zero_pad_to_byte().unwrap();
    let bytes = writer.into_bytes().unwrap();
    (0..count).map(|i| (bytes[i / 8] >> (i % 8)) & 1).collect()
}

struct HeaderReader<'a> {
    bits: &'a [u8],
    position: usize,
}

impl HeaderReader<'_> {
    fn read(&mut self, width: u8) -> u32 {
        let mut value = 0;
        for bit in 0..width {
            value |= u32::from(self.bits[self.position]) << bit;
            self.position += 1;
        }
        value
    }

    fn threshold(&mut self, widths: [u8; 4], offsets: [u32; 4]) -> (u32, u32) {
        let selector = self.read(2);
        let value = self.read(widths[selector as usize]);
        (
            selector,
            value.checked_add(offsets[selector as usize]).unwrap(),
        )
    }
}

#[test]
fn default_block_context_header_is_exactly_one_bit_at_every_alignment() {
    for alignment in 0..8u8 {
        let mut writer = BitWriter::new(2);
        writer.write(alignment, (1u64 << alignment) - 1).unwrap();
        write_block_context_map(BlockContextModel::default(), &mut writer).unwrap();
        assert_eq!(writer.bits_written(), usize::from(alignment) + 1);
        assert_eq!(written_bits(writer), vec![1; usize::from(alignment) + 1]);
    }
}

#[test]
fn threshold_wire_fields_round_trip_all_selectors_extremes_and_channels() {
    let dc_x = [0, -8, 8, -136, 136, -32904, 32904, i32::MAX, i32::MIN];
    let dc_y = [-1];
    let dc_b = [1];
    let map = vec![0; 39 * 40];
    let model = BlockContextModel::new([&dc_x, &dc_y, &dc_b], &[], &map, 1).unwrap();
    let mut writer = BitWriter::new(4096);
    write_block_context_map(model, &mut writer).unwrap();
    let bits = written_bits(writer);
    let mut reader = HeaderReader {
        bits: &bits,
        position: 0,
    };
    assert_eq!(reader.read(1), 0);
    for (channel, thresholds) in [&dc_x[..], &dc_y[..], &dc_b[..]].into_iter().enumerate() {
        assert_eq!(reader.read(4) as usize, thresholds.len());
        for (i, &expected) in thresholds.iter().enumerate() {
            let (selector, packed) = reader.threshold([4, 8, 16, 32], [0, 16, 272, 65808]);
            if channel == 0 {
                assert_eq!(selector, [0, 0, 1, 1, 2, 2, 3, 3, 3][i]);
            }
            let decoded = if packed & 1 == 0 {
                i64::from(packed / 2)
            } else {
                -i64::from(packed / 2) - 1
            };
            assert_eq!(decoded, i64::from(expected));
        }
    }
    assert_eq!(reader.read(4), 0); // No QF thresholds.
    assert_eq!(reader.read(1), 1); // Generic one-histogram simple map.
    assert_eq!(reader.read(2), 0);
    assert_eq!(reader.position, bits.len());

    let qf = [1, 4, 5, 12, 13, 44, 45, 300];
    let map = vec![0; 39 * 9];
    let model = BlockContextModel::new([&[]; 3], &qf, &map, 1).unwrap();
    let mut writer = BitWriter::new(4096);
    write_block_context_map(model, &mut writer).unwrap();
    let bits = written_bits(writer);
    let mut reader = HeaderReader {
        bits: &bits,
        position: 0,
    };
    assert_eq!(reader.read(1), 0);
    for _ in 0..3 {
        assert_eq!(reader.read(4), 0);
    }
    assert_eq!(reader.read(4), 8);
    for (i, expected) in qf.into_iter().enumerate() {
        let (selector, value) = reader.threshold([2, 3, 5, 8], [0, 4, 12, 44]);
        assert_eq!(selector as usize, i / 2);
        assert_eq!(value + 1, expected);
    }
    assert_eq!(reader.read(1), 1);
    assert_eq!(reader.read(2), 0);
    assert_eq!(reader.position, bits.len());
}

#[test]
fn custom_context_header_delegates_the_complete_map_without_padding() {
    let map: Vec<u8> = (0..39).map(|i| (i % 16) as u8).collect();
    let model = BlockContextModel::new([&[]; 3], &[], &map, 16).unwrap();
    let mut generic = BitWriter::new(16384);
    write_context_map(&map, 16, &mut generic).unwrap();
    let generic = written_bits(generic);
    for alignment in 0..8u8 {
        let mut writer = BitWriter::new(16384);
        writer.write(alignment, 0).unwrap();
        write_block_context_map(model, &mut writer).unwrap();
        let bits = written_bits(writer);
        let start = usize::from(alignment);
        assert!(bits[..start + 17].iter().all(|&bit| bit == 0));
        assert_eq!(&bits[start + 17..], generic.as_slice());
    }
}

#[test]
fn context_header_output_limit_failure_preserves_destination_bits() {
    let mut writer = BitWriter::new(0);
    assert!(write_block_context_map(BlockContextModel::default(), &mut writer).is_err());
    assert_eq!(writer.bits_written(), 0);
    let map = [0; 39];
    let model = BlockContextModel::new([&[]; 3], &[], &map, 1).unwrap();
    let mut writer = BitWriter::new(1);
    writer.write(3, 5).unwrap();
    assert!(write_block_context_map(model, &mut writer).is_err());
    assert_eq!(writer.bits_written(), 3);
    assert_eq!(written_bits(writer), [1, 0, 1]);
}
