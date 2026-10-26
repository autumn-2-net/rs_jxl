// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Weighted Modular predictor, translated from the supplied read-only libjxl
//! baseline `b866b3defec13494bfbaf705db93581cda118e37`:
//! `lib/jxl/modular/encoding/context_predict.h`, namespace `weighted`:
//! `Header::VisitFields`, `State`, and `PredictorMode`.
//! The exact flat shortcut follows `enc_predict.h::PredictWeightedFlat` in
//! the supplied encoder tree; ordinary error updates are never skipped.
//!
//! Source arithmetic widths matter: predictions are i64, signed error history
//! narrows to i32, and predictor error history AND its three-neighbor sum wrap
//! at u32. The returned property is the signed, unscaled error of greatest
//! magnitude, with ties resolved W, N, NW, NE. The requested i32 prediction API
//! narrows the rounded i64 prediction modulo 2^32, without narrowing `pred`.
//!
//! Create/reset a state for EACH channel in EACH independently coded group.
//! Coordinates are group-local. Supply boundary-adjusted neighbors as in
//! `detail::Predict`; this module does not load samples or synthesize edges.
//! Height is intentionally not part of the API, so only raster order and width
//! can be validated. No MA tree, entropy coding, or predictor search is here.

#![forbid(unsafe_code)]

use crate::bit_writer::BitWriter;
use crate::{Error, Result};

pub const NUM_PREDICTORS: usize = 4;
pub const NUM_PROPERTIES: usize = 1;
const EXTRA_BITS: u32 = 3;
const PREDICTION_ROUND: i64 = 3;

const fn division_table() -> [u32; 64] {
    let mut table = [0; 64];
    let mut i = 0;
    while i < table.len() {
        table[i] = (1 << 24) / (i as u32 + 1);
        i += 1;
    }
    table
}

const DIV_LOOKUP: [u32; 64] = division_table();

const fn error_weight(error: u32, weight: u32) -> u32 {
    let log = 63 - (error as u64 + 1).leading_zeros();
    let shift = log.saturating_sub(5);
    4 + ((weight * DIV_LOOKUP[(error >> shift) as usize]) >> shift)
}

// Exact small-error weights, shared by all states and presets (16 KiB).
static SMALL_WEIGHTS: [[u32; 256]; 16] = {
    let mut table = [[0; 256]; 16];
    let mut weight = 0;
    while weight < 16 {
        let mut error = 0;
        while error < 256 {
            table[weight][error] = error_weight(error as u32, weight as u32);
            error += 1;
        }
        weight += 1;
    }
    table
};

#[inline]
fn weights(errors: [u32; 4], max_weights: [u32; 4]) -> [u32; 4] {
    if (errors[0] | errors[1] | errors[2] | errors[3]) < 256 {
        std::array::from_fn(|i| SMALL_WEIGHTS[max_weights[i] as usize][errors[i] as usize])
    } else {
        std::array::from_fn(|i| error_weight(errors[i], max_weights[i]))
    }
}

/// Source `weighted::Header`. `all_default` is derived, never cached.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Header {
    pub p1_c: i32,
    pub p2_c: i32,
    pub p3_ca: i32,
    pub p3_cb: i32,
    pub p3_cc: i32,
    pub p3_cd: i32,
    pub p3_ce: i32,
    pub w: [u32; NUM_PREDICTORS],
}

impl Default for Header {
    fn default() -> Self {
        Self {
            p1_c: 16,
            p2_c: 10,
            p3_ca: 7,
            p3_cb: 7,
            p3_cc: 7,
            p3_cd: 0,
            p3_ce: 0,
            w: [13, 12, 12, 12],
        }
    }
}

impl Header {
    /// The five `weighted::PredictorMode` presets, without retuning.
    /// Unlike C++'s catch-all mode-4 fallback, invalid API modes are rejected.
    pub fn preset(mode: u32) -> Result<Self> {
        let (p, w) = match mode {
            0 => return Ok(Self::default()),
            1 => ([8, 8, 4, 0, 3, 23, 2], [13, 12, 12, 11]),
            2 => ([10, 9, 7, 0, 0, 16, 9], [13, 12, 13, 12]),
            3 => ([16, 8, 0, 16, 0, 23, 0], [13, 13, 12, 12]),
            4 => ([10, 10, 5, 5, 5, 12, 4], [13, 12, 12, 12]),
            _ => return Err(Error::InvalidInput("weighted preset must be 0..=4".into())),
        };
        Ok(Self {
            p1_c: p[0],
            p2_c: p[1],
            p3_ca: p[2],
            p3_cb: p[3],
            p3_cc: p[4],
            p3_cd: p[5],
            p3_ce: p[6],
            w,
        })
    }

    fn parameters(&self) -> [i32; 7] {
        [
            self.p1_c, self.p2_c, self.p3_ca, self.p3_cb, self.p3_cc, self.p3_cd, self.p3_ce,
        ]
    }

    pub fn validate(&self) -> Result<()> {
        if self.parameters().iter().any(|&p| !(0..32).contains(&p))
            || self.w.iter().any(|&w| w >= 16)
        {
            return Err(Error::InvalidInput(
                "weighted parameters exceed their field widths".into(),
            ));
        }
        Ok(())
    }

    /// `Header::VisitFields`: one all-default bit, otherwise 7*5 + 4*4 bits.
    /// One writer call makes allocation/limit failure non-mutating.
    pub fn write_header(&self, writer: &mut BitWriter) -> Result<()> {
        self.validate()?;
        if *self == Self::default() {
            writer.write(1, 1)?;
        } else {
            let mut bits = 0u64;
            let mut offset = 1;
            for p in self.parameters() {
                bits |= (p as u64) << offset;
                offset += 5;
            }
            for w in self.w {
                bits |= u64::from(w) << offset;
                offset += 4;
            }
            writer.write(52, bits)?;
        }
        Ok(())
    }
}

/// Two rows of history, plus the source's two-pixel row margins.
#[derive(Debug)]
#[cfg_attr(test, derive(Clone, Eq, PartialEq))]
pub struct State {
    header: Header,
    width: usize,
    stride: usize,
    prediction: [i64; NUM_PREDICTORS],
    pred: i64,
    pred_errors: [Vec<u32>; NUM_PREDICTORS],
    error: Vec<i32>,
    next: Option<(usize, usize)>,
    pending: bool,
}

fn zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| Error::Limit("weighted predictor history allocation"))?;
    values.resize(len, T::default());
    Ok(values)
}

impl State {
    pub fn new(header: Header, width: usize) -> Result<Self> {
        header.validate()?;
        if width == 0 {
            return Err(Error::InvalidInput(
                "weighted width must be positive".into(),
            ));
        }
        let stride = width.checked_add(2).ok_or(Error::Limit("weighted width"))?;
        let len = stride
            .checked_mul(2)
            .ok_or(Error::Limit("weighted history size"))?;
        Ok(Self {
            header,
            width,
            stride,
            prediction: [0; NUM_PREDICTORS],
            pred: 0,
            pred_errors: [zeroed(len)?, zeroed(len)?, zeroed(len)?, zeroed(len)?],
            error: zeroed(len)?,
            next: Some((0, 0)),
            pending: false,
        })
    }

    /// Begin another group/channel of the same width, discarding ALL history.
    /// Use `new` for a different width/header. Also cancels a pending prediction.
    pub fn reset(&mut self) {
        self.prediction.fill(0);
        self.pred = 0;
        for errors in &mut self.pred_errors {
            errors.fill(0);
        }
        self.error.fill(0);
        self.next = Some((0, 0));
        self.pending = false;
    }

    fn check_position(&self, x: usize, y: usize, updating: bool) -> Result<()> {
        if x >= self.width || self.next != Some((x, y)) || self.pending != updating {
            return Err(Error::InvalidInput(
                "weighted calls must alternate predict/update at consecutive raster coordinates"
                    .into(),
            ));
        }
        Ok(())
    }

    fn rows(&self, y: usize) -> (usize, usize) {
        if y & 1 != 0 {
            (0, self.stride)
        } else {
            (self.stride, 0)
        }
    }

    fn weighted_average(p: [i64; 4], mut w: [u32; 4]) -> i64 {
        let weight_sum: u32 = w.iter().sum();
        let log_weight = 31 - weight_sum.leading_zeros();
        for weight in &mut w {
            *weight >>= log_weight - 4;
        }
        let weight_sum: u32 = w.iter().sum();
        let mut sum = i64::from((weight_sum >> 1) - 1);
        for i in 0..NUM_PREDICTORS {
            sum += p[i] * i64::from(w[i]);
        }
        (sum * i64::from(DIV_LOOKUP[weight_sum as usize - 1])) >> 24
    }

    /// Return (rounded prediction, signed weighted-error tree property).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &mut self,
        x: usize,
        y: usize,
        n: i32,
        w: i32,
        ne: i32,
        nw: i32,
        nn: i32,
    ) -> Result<(i32, i32)> {
        self.predict_impl::<true>(x, y, n, w, ne, nw, nn)
    }

    /// Predict from the old history, then advance it with the actual sample.
    /// Only use for local states that are discarded if subsequent work fails.
    /// Invalid coordinates or a pending prediction leave the state unchanged.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_and_update(
        &mut self,
        value: i32,
        x: usize,
        y: usize,
        n: i32,
        w: i32,
        ne: i32,
        nw: i32,
        nn: i32,
    ) -> Result<(i32, i32)> {
        self.check_position(x, y, false)?;
        let rows = self.rows(y);
        let result = self.predict_at::<true>(x, rows, [n, w, ne, nw, nn]);
        self.update_at(value, x, y, rows);
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_impl<const FLAT: bool>(
        &mut self,
        x: usize,
        y: usize,
        n: i32,
        w: i32,
        ne: i32,
        nw: i32,
        nn: i32,
    ) -> Result<(i32, i32)> {
        self.check_position(x, y, false)?;
        let result = self.predict_at::<FLAT>(x, self.rows(y), [n, w, ne, nw, nn]);
        self.pending = true;
        Ok(result)
    }

    #[inline]
    fn predict_at<const FLAT: bool>(
        &mut self,
        x: usize,
        (cur_row, prev_row): (usize, usize),
        [n, w, ne, nw, nn]: [i32; 5],
    ) -> (i32, i32) {
        let pos_n = prev_row + x;
        let pos_ne = if x < self.width - 1 { pos_n + 1 } else { pos_n };
        let pos_nw = if x > 0 { pos_n - 1 } else { pos_n };
        let te_w = if x == 0 {
            0
        } else {
            i64::from(self.error[cur_row + x - 1])
        };
        let te_n = i64::from(self.error[pos_n]);
        let te_nw = i64::from(self.error[pos_nw]);
        let te_ne = i64::from(self.error[pos_ne]);
        if FLAT
            && ((n ^ w) | (n ^ ne) | (n ^ nw) | (n ^ nn)) == 0
            && (te_w | te_n | te_nw | te_ne) == 0
        {
            // All sub-predictors equal N, and the signed-error clamp is [N, N],
            // even with nonzero/wrapped unsigned weight history. Update still
            // needs every sub-prediction when the actual sample differs.
            self.pred = i64::from(n) << EXTRA_BITS;
            self.prediction = [self.pred; NUM_PREDICTORS];
            return (n, 0);
        }
        let mut errors = [0; NUM_PREDICTORS];
        for (i, error) in errors.iter_mut().enumerate() {
            let errors = &self.pred_errors[i];
            *error = errors[pos_n]
                .wrapping_add(errors[pos_ne])
                .wrapping_add(errors[pos_nw]);
        }
        let weights = weights(errors, self.header.w);

        let [n, w, ne, nw, nn] = [n, w, ne, nw, nn].map(|v| i64::from(v) << EXTRA_BITS);
        let sum_wn = te_n + te_w;
        let mut property = te_w;
        for e in [te_n, te_nw, te_ne] {
            if e.abs() > property.abs() {
                property = e;
            }
        }
        let h = &self.header;
        self.prediction = [
            w + ne - n,
            n - (((sum_wn + te_ne) * i64::from(h.p1_c)) >> 5),
            w - (((sum_wn + te_nw) * i64::from(h.p2_c)) >> 5),
            n - ((te_nw * i64::from(h.p3_ca)
                + te_n * i64::from(h.p3_cb)
                + te_ne * i64::from(h.p3_cc)
                + (nn - n) * i64::from(h.p3_cd)
                + (nw - w) * i64::from(h.p3_ce))
                >> 5),
        ];
        self.pred = Self::weighted_average(self.prediction, weights);
        // Preserve the exact source bitwise condition, including zero/ties.
        if ((te_n ^ te_w) | (te_n ^ te_nw)) <= 0 {
            self.pred = self.pred.clamp(w.min(ne).min(n), w.max(ne).max(n));
        }
        (
            ((self.pred + PREDICTION_ROUND) >> EXTRA_BITS) as i32,
            property as i32,
        )
    }

    pub fn update(&mut self, value: i32, x: usize, y: usize) -> Result<()> {
        self.check_position(x, y, true)?;
        self.update_at(value, x, y, self.rows(y));
        Ok(())
    }

    #[inline]
    fn update_at(&mut self, value: i32, x: usize, y: usize, (cur_row, prev_row): (usize, usize)) {
        let value = i64::from(value) << EXTRA_BITS;
        self.error[cur_row + x] = (self.pred - value) as i32;
        for i in 0..NUM_PREDICTORS {
            let err = ((self.prediction[i] - value).abs() + PREDICTION_ROUND) >> EXTRA_BITS;
            self.pred_errors[i][cur_row + x] = err as u32;
            self.pred_errors[i][prev_row + x + 1] =
                self.pred_errors[i][prev_row + x + 1].wrapping_add(err as u32);
        }
        self.pending = false;
        self.next = if x == self.width - 1 {
            y.checked_add(1).map(|next_y| (0, next_y))
        } else {
            Some((x + 1, y))
        };
    }
}

#[cfg(test)]
#[path = "weighted_fused_tests.rs"]
mod fused_tests;

#[cfg(test)]
#[path = "weighted_flat_tests.rs"]
mod flat_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_weights_match_integer_reference_at_all_exponent_boundaries() {
        let mut errors: Vec<u32> = (0..1024).collect();
        for bit in 10..=32 {
            for delta in -2i64..=1 {
                let value = (1i64 << bit) + delta;
                if let Ok(value) = u32::try_from(value) {
                    errors.push(value);
                }
            }
        }
        for error in errors {
            for weight in 0..16 {
                for lane in 0..4 {
                    let mut input = [0, 255, 17, 63];
                    input[lane] = error;
                    let limits = [weight, 15 - weight, weight ^ 5, weight ^ 10];
                    let expected = std::array::from_fn(|i| {
                        let mut scale = 1u64;
                        while u64::from(input[i]) + 1 >= 64 * scale {
                            scale *= 2;
                        }
                        let reciprocal = (1u64 << 24) / (u64::from(input[i]) / scale + 1);
                        (4 + u64::from(limits[i]) * reciprocal / scale) as u32
                    });
                    assert_eq!(weights(input, limits), expected);
                }
            }
        }
    }

    // These vectors were evaluated from the source equations with a separate
    // arbitrary-precision/full-history model, NOT captured from a C++ binary.
    const SAMPLES: [i32; 16] = [-7, 5, 12, -1, 6, -8, 3, 19, -2, 9, -11, 4, 31, -17, 0, 8];
    const GOLDEN: [[(i32, i32); 16]; 5] = [
        [
            (0, 0),
            (-8, 56),
            (5, -107),
            (12, -56),
            (-4, -107),
            (12, -107),
            (-5, 160),
            (-1, 104),
            (0, 160),
            (-5, 160),
            (16, 160),
            (6, 220),
            (5, -115),
            (17, 220),
            (-17, 274),
            (0, 220),
        ],
        [
            (0, 0),
            (-8, 56),
            (5, -103),
            (12, -56),
            (-4, -103),
            (11, -103),
            (-5, 150),
            (-1, 104),
            (2, 150),
            (-6, 150),
            (13, -159),
            (9, 191),
            (4, -121),
            (20, -213),
            (-17, 297),
            (0, 191),
        ],
        [
            (0, 0),
            (-8, 56),
            (5, -104),
            (12, -56),
            (-5, -104),
            (12, -104),
            (-4, 158),
            (-1, 104),
            (2, 158),
            (-6, 158),
            (14, -160),
            (7, 198),
            (4, -122),
            (22, -216),
            (-17, 311),
            (0, 198),
        ],
        [
            (0, 0),
            (-8, 56),
            (5, -107),
            (12, -56),
            (-5, -107),
            (12, -107),
            (-5, 160),
            (-1, 104),
            (2, 160),
            (-8, 160),
            (17, 160),
            (10, 224),
            (5, -136),
            (23, 224),
            (-17, 317),
            (0, 224),
        ],
        [
            (0, 0),
            (-8, 56),
            (5, -105),
            (12, -56),
            (-4, -105),
            (12, -105),
            (-4, 160),
            (-1, 104),
            (1, 160),
            (-7, 160),
            (14, 160),
            (8, 204),
            (4, -124),
            (20, -212),
            (-17, 298),
            (0, 204),
        ],
    ];

    fn neighbors(samples: &[i32], width: usize, x: usize, y: usize) -> [i32; 5] {
        let pos = y * width + x;
        let w = if x > 0 {
            samples[pos - 1]
        } else if y > 0 {
            samples[pos - width]
        } else {
            0
        };
        let n = if y > 0 { samples[pos - width] } else { w };
        let ne = if y > 0 && x + 1 < width {
            samples[pos - width + 1]
        } else {
            n
        };
        let nw = if x > 0 && y > 0 {
            samples[pos - width - 1]
        } else {
            w
        };
        let nn = if y > 1 { samples[pos - 2 * width] } else { n };
        [n, w, ne, nw, nn]
    }

    fn run(state: &mut State, samples: &[i32], width: usize) -> Vec<(i32, i32)> {
        samples
            .iter()
            .enumerate()
            .map(|(pos, &value)| {
                let (x, y) = (pos % width, pos / width);
                let [n, w, ne, nw, nn] = neighbors(samples, width, x, y);
                let prediction = state.predict(x, y, n, w, ne, nw, nn).unwrap();
                state.update(value, x, y).unwrap();
                prediction
            })
            .collect()
    }

    #[test]
    fn presets_and_header_wire_goldens() {
        let expected: [&[u8]; 5] = [
            &[1],
            &[0x10, 0x22, 0x60, 0x5c, 0xd1, 0xcc, 0x0b],
            &[0x54, 0x3a, 0x00, 0xc0, 0xd4, 0xdc, 0x0c],
            &[0x20, 0x02, 0x10, 0x5c, 0xd0, 0xcd, 0x0c],
            &[0x94, 0x2a, 0xa5, 0x30, 0xd2, 0xcc, 0x0c],
        ];
        assert_eq!(Header::preset(0).unwrap(), Header::default());
        for mode in 0..5 {
            for prefix in 0..8 {
                let mut writer = BitWriter::new(16);
                writer.write(prefix, (1u64 << prefix) - 1).unwrap();
                Header::preset(mode)
                    .unwrap()
                    .write_header(&mut writer)
                    .unwrap();
                let bits = if mode == 0 { 1 } else { 52 };
                assert_eq!(writer.bits_written(), prefix as usize + bits);
                writer.zero_pad_to_byte().unwrap();
                let bytes = writer.bytes().unwrap();
                for i in 0..bits {
                    let position = i + prefix as usize;
                    assert_eq!(
                        (bytes[position / 8] >> (position % 8)) & 1,
                        (expected[mode as usize][i / 8] >> (i % 8)) & 1
                    );
                }
            }
        }
        assert!(Header::preset(5).is_err());
        assert!(Header::preset(u32::MAX).is_err());
    }

    #[test]
    fn invalid_header_and_allocation_fail_without_writing() {
        for header in [
            Header {
                p1_c: -1,
                ..Header::default()
            },
            Header {
                p3_ce: 32,
                ..Header::default()
            },
            Header {
                w: [0, 0, 16, 0],
                ..Header::default()
            },
        ] {
            let mut writer = BitWriter::new(16);
            assert!(header.write_header(&mut writer).is_err());
            assert_eq!(writer.bits_written(), 0);
            assert!(State::new(header, 2).is_err());
        }
        let mut writer = BitWriter::new(1);
        writer.write(3, 5).unwrap();
        assert!(
            Header::preset(1)
                .unwrap()
                .write_header(&mut writer)
                .is_err()
        );
        assert_eq!(writer.bits_written(), 3);
        writer.zero_pad_to_byte().unwrap();
        assert_eq!(writer.bytes().unwrap(), &[5]);
        for width in [0, usize::MAX, usize::MAX / 2, usize::MAX / 2 - 2] {
            assert!(State::new(Header::default(), width).is_err());
        }
    }

    #[test]
    fn all_default_true_and_full_custom_header_flags_are_value_derived() {
        let maximal = Header {
            p1_c: 31,
            p2_c: 31,
            p3_ca: 31,
            p3_cb: 31,
            p3_cc: 31,
            p3_cd: 31,
            p3_ce: 31,
            w: [15; 4],
        };
        let mut writer = BitWriter::new(7);
        maximal.write_header(&mut writer).unwrap();
        assert_eq!(writer.bits_written(), 52);
        writer.zero_pad_to_byte().unwrap();
        assert_eq!(
            writer.bytes().unwrap(),
            &[0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f]
        );

        let mut header = Header::default();
        for (coefficient, expected_bits, expected_flag) in [(0, 1, 1), (1, 52, 0), (0, 1, 1)] {
            header.p3_ce = coefficient;
            let mut writer = BitWriter::new(7);
            header.write_header(&mut writer).unwrap();
            assert_eq!(writer.bits_written(), expected_bits);
            writer.zero_pad_to_byte().unwrap();
            assert_eq!(writer.bytes().unwrap()[0] & 1, expected_flag);
        }
    }

    #[test]
    fn reference_vectors_for_all_presets_and_group_reset() {
        for mode in 0..5 {
            let mut state = State::new(Header::preset(mode).unwrap(), 4).unwrap();
            assert_eq!(run(&mut state, &SAMPLES, 4), GOLDEN[mode as usize]);
            state.reset();
            assert_eq!(run(&mut state, &SAMPLES, 4), GOLDEN[mode as usize]);
            state.reset();
            state.predict(0, 0, 0, 0, 0, 0, 0).unwrap();
            state.reset();
            assert_eq!(run(&mut state, &SAMPLES, 4), GOLDEN[mode as usize]);
        }
    }

    #[test]
    fn extreme_wrapping_and_width_one_goldens() {
        let mut state = State::new(Header::preset(4).unwrap(), 3).unwrap();
        let samples = [
            i32::MIN,
            i32::MAX,
            -1,
            i32::MAX,
            0,
            i32::MIN,
            7,
            -9,
            i32::MAX,
            i32::MIN,
            255,
            -256,
        ];
        assert_eq!(
            run(&mut state, &samples, 3),
            [
                (0, 0),
                (i32::MIN, 0),
                (i32::MAX, 8),
                (-1073741824, 8),
                (1744830463, 8),
                (0, 1073741814),
                (0, 1073741814),
                (-831520763, 1073741814),
                (-1328644103, 1937768561),
                (-9, 1937768561),
                (-621417219, -2039218222),
                (255, -2039218222),
            ]
        );
        let mut state = State::new(Header::preset(1).unwrap(), 1).unwrap();
        assert_eq!(
            run(&mut state, &[-7, 9, -11, 0, i32::MAX, i32::MIN], 1),
            [
                (0, 0),
                (-9, 56),
                (9, -145),
                (-19, 160),
                (0, -152),
                (-1761607682, 8)
            ]
        );
    }

    #[test]
    fn bad_call_order_does_not_advance_or_corrupt_state() {
        let mut state = State::new(Header::default(), 2).unwrap();
        assert!(state.update(5, 0, 0).is_err());
        for (x, y) in [(1, 0), (2, 0), (0, 1), (usize::MAX, usize::MAX)] {
            assert!(state.predict(x, y, 0, 0, 0, 0, 0).is_err());
        }
        assert_eq!(state.predict(0, 0, 0, 0, 0, 0, 0).unwrap(), (0, 0));
        assert!(state.predict(0, 0, 0, 0, 0, 0, 0).is_err());
        assert!(state.predict(1, 0, 0, 0, 0, 0, 0).is_err());
        assert!(state.update(999, 1, 0).is_err());
        assert!(state.update(999, 0, 1).is_err());
        state.update(5, 0, 0).unwrap();
        assert!(state.update(999, 0, 0).is_err());
        assert!(state.predict(0, 1, 0, 0, 0, 0, 0).is_err());
        let result = state.predict(1, 0, 5, 5, 5, 5, 5).unwrap();
        let mut fresh = State::new(Header::default(), 2).unwrap();
        fresh.predict(0, 0, 0, 0, 0, 0, 0).unwrap();
        fresh.update(5, 0, 0).unwrap();
        assert_eq!(result, fresh.predict(1, 0, 5, 5, 5, 5, 5).unwrap());
        state.update(-2, 1, 0).unwrap();
        state.predict(0, 1, 5, 5, -2, 5, 5).unwrap();
    }

    #[test]
    fn coordinate_exhaustion_never_wraps_back_to_first_group() {
        let mut state = State::new(Header::default(), 1).unwrap();
        state.next = Some((0, usize::MAX));
        state.predict(0, usize::MAX, 0, 0, 0, 0, 0).unwrap();
        state.update(0, 0, usize::MAX).unwrap();
        assert_eq!(state.next, None);
        assert!(state.predict(0, 0, 0, 0, 0, 0, 0).is_err());
        state.reset();
        assert_eq!(state.predict(0, 0, 0, 0, 0, 0, 0).unwrap(), (0, 0));
    }

    #[test]
    fn signed_property_keeps_source_tie_order_and_minimum() {
        let mut state = State::new(Header::default(), 3).unwrap();
        for (errors, expected) in [
            ([-12, 12, -12, 12], -12),
            ([0, -12, 12, 12], -12),
            ([0, 0, -12, 12], -12),
            ([0, 0, 0, -12], -12),
            ([0, i32::MAX, i32::MIN, 0], i32::MIN),
        ] {
            state.reset();
            state.next = Some((1, 1));
            let (cur, prev) = state.rows(1);
            state.error[cur] = errors[0];
            state.error[prev + 1] = errors[1];
            state.error[prev] = errors[2];
            state.error[prev + 2] = errors[3];
            assert_eq!(state.predict(1, 1, 0, 0, 0, 0, 0).unwrap().1, expected);
        }
    }

    #[test]
    fn rounding_narrowing_and_error_accumulation_are_not_saturating() {
        for (prediction, rounded) in [(-12, -2), (-4, -1), (4, 0), (12, 1)] {
            let average = State::weighted_average([prediction; 4], [4; 4]);
            assert_eq!(average, prediction);
            assert_eq!((average + PREDICTION_ROUND) >> EXTRA_BITS, rounded);
        }
        let mut state = State::new(Header::default(), 2).unwrap();
        state.predict(0, 0, 0, 0, 0, 0, 0).unwrap();
        state.pred = i64::from(i32::MAX) + 1;
        state.prediction = [4, -4, 12, (1i64 << 35) + 12];
        for errors in &mut state.pred_errors {
            errors[1] = u32::MAX;
        }
        state.update(0, 0, 0).unwrap();
        assert_eq!(state.error[state.stride], i32::MIN);
        assert_eq!(
            state.pred_errors.each_ref().map(|p| p[state.stride]),
            [0, 0, 1, 1]
        );
        assert_eq!(
            state.pred_errors.each_ref().map(|p| p[1]),
            [u32::MAX, u32::MAX, 0, 0]
        );
        assert_eq!(error_weight(u32::MAX, 15), 4);
        assert_eq!(error_weight(0, 15), 4 + 15 * (1 << 24));
        for (i, &value) in DIV_LOOKUP.iter().enumerate() {
            assert_eq!(value, (1 << 24) / (i as u32 + 1));
        }
    }

    // A deliberately non-ring reference: retain every pixel's predictor error
    // and derive W/WW contributions geometrically. Wider arithmetic prevents
    // accidental reliance on the production implementation's widths/helpers.
    fn full_history_reference(samples: &[i32], width: usize, header: Header) -> Vec<(i32, i32)> {
        let mut signed = vec![0i32; samples.len()];
        let mut errors = vec![[0u32; 4]; samples.len()];
        let mut output = Vec::new();
        for (pos, &value) in samples.iter().enumerate() {
            let (x, y) = (pos % width, pos / width);
            let [n, w, ne, nw, nn] = neighbors(samples, width, x, y).map(|v| i128::from(v) * 8);
            let old = |j: usize| {
                if y == 0 {
                    0
                } else {
                    i128::from(signed[(y - 1) * width + j])
                }
            };
            let tw = if x == 0 {
                0
            } else {
                i128::from(signed[pos - 1])
            };
            let tn = old(x);
            let tnw = old(x.saturating_sub(1));
            let tne = old((x + 1).min(width - 1));
            let mut property = tw;
            for e in [tn, tnw, tne] {
                if e.abs() > property.abs() {
                    property = e;
                }
            }
            let p = header.parameters().map(i128::from);
            let predictions = [
                w + ne - n,
                n - ((tn + tw + tne) * p[0]).div_euclid(32),
                w - ((tn + tw + tnw) * p[1]).div_euclid(32),
                n - (tnw * p[2] + tn * p[3] + tne * p[4] + (nn - n) * p[5] + (nw - w) * p[6])
                    .div_euclid(32),
            ];
            let mut weights = [0u64; 4];
            for i in 0..4 {
                let at = |j: usize| {
                    let prior = if y == 0 {
                        0
                    } else {
                        u64::from(errors[(y - 1) * width + j][i])
                    };
                    let carry = if j > 0 && j - 1 < x {
                        u64::from(errors[y * width + j - 1][i])
                    } else {
                        0
                    };
                    (prior + carry) % (1u64 << 32)
                };
                let sum =
                    (at(x) + at(x.saturating_sub(1)) + at((x + 1).min(width - 1))) % (1u64 << 32);
                let mut shift = 0;
                while (sum + 1) >> shift >= 64 {
                    shift += 1;
                }
                weights[i] =
                    4 + ((u64::from(header.w[i]) * ((1 << 24) / ((sum >> shift) + 1))) >> shift);
            }
            let total: u64 = weights.iter().sum();
            let mut shift = 0;
            while total >> shift >= 32 {
                shift += 1;
            }
            for w in &mut weights {
                *w >>= shift;
            }
            let total: u64 = weights.iter().sum();
            let mut sum = i128::from(total / 2) - 1;
            for i in 0..4 {
                sum += predictions[i] * i128::from(weights[i]);
            }
            let mut pred = (sum * i128::from((1u64 << 24) / total)).div_euclid(1 << 24);
            if ((tn ^ tw) | (tn ^ tnw)) <= 0 {
                pred = pred.clamp(n.min(w).min(ne), n.max(w).max(ne));
            }
            output.push(((pred + 3).div_euclid(8) as i32, property as i32));
            signed[pos] = (pred - i128::from(value) * 8) as i32;
            errors[pos] =
                predictions.map(|p| ((p - i128::from(value) * 8).abs() + 3).div_euclid(8) as u32);
        }
        output
    }

    #[test]
    fn full_history_reference_covers_widths_row_reuse_and_parameter_extremes() {
        let mut random = 0x8e41_5a17u32;
        let mut headers: Vec<_> = (0..5).map(|m| Header::preset(m).unwrap()).collect();
        headers.extend([
            Header {
                p1_c: 31,
                p2_c: 31,
                p3_ca: 31,
                p3_cb: 31,
                p3_cc: 31,
                p3_cd: 31,
                p3_ce: 31,
                w: [15; 4],
            },
            Header {
                w: [0; 4],
                ..Header::default()
            },
            Header {
                w: [0, 15, 0, 1],
                ..Header::default()
            },
        ]);
        for width in [1, 2, 3, 7, 16, 33] {
            let samples: Vec<i32> = (0..width * 11)
                .map(|i| {
                    random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                    match i % 7 {
                        0 => i32::MIN,
                        1 => i32::MAX,
                        _ => random as i32,
                    }
                })
                .collect();
            for &header in &headers {
                let reference = full_history_reference(&samples, width, header);
                let mut state = State::new(header, width).unwrap();
                assert_eq!(
                    run(&mut state, &samples, width),
                    reference,
                    "width={width}, {header:?}"
                );
            }
        }
    }
}
