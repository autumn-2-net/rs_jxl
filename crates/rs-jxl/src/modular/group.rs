// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Modular transform channel layout and wire fields, from transform.cc,
//! enc_palette.cc and enc_squeeze.cc. Palette ranges use pre-transform indices:
//! each table is inserted at channel zero, shifting ALL previous meta channels.
//! No pixel-format conversion is performed.
use super::palette::PaletteMode;
use super::predictor::Predictor;
use super::{transforms, weighted};
use crate::{Error, Result, bit_writer::BitWriter, codestream::write_choice};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    pub width: usize,
    pub height: usize,
    pub hshift: i32,
    pub vshift: i32,
    pub samples: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transform {
    /// Compatibility spelling for an RCT beginning at channel zero.
    Rct(u32),
    RctAt {
        begin: usize,
        kind: u32,
    },
    Palette {
        begin: usize,
        channels: usize,
        colors: usize,
        deltas: usize,
        predictor: Predictor,
    },
    Squeeze {
        horizontal: bool,
        begin: usize,
        channels: usize,
    },
    /// Empty means source DefaultSqueezeParameters, not an empty transform.
    SqueezeScript(Vec<super::layout::SqueezeParams>),
}

/// Compatibility wrapper: all non-meta channels, ordered palette. Exact-delta
/// failure falls back to the ordinary lossless palette, as FwdPalette does.
pub fn apply_palette(
    channels: &mut Vec<Channel>,
    bits: u8,
    max_colors: u32,
    predictor: Option<Predictor>,
) -> Result<Option<Transform>> {
    if channels.is_empty() {
        return Err(Error::InvalidInput("palette needs channels".into()));
    }
    let mut meta = channels
        .iter()
        .take_while(|c| c.hshift < 0 && c.vshift < 0)
        .count();
    let begin = meta;
    let count = channels.len() - meta;
    let mode = predictor.map_or(PaletteMode::Absolute, PaletteMode::ExactDelta);
    apply_palette_range(
        channels, &mut meta, bits, begin, count, max_colors, true, mode,
    )
}

pub(crate) fn validate_layout(channels: &[Channel], meta: usize) -> Result<()> {
    if meta > channels.len() || channels.len() > 8192 {
        return Err(Error::InvalidInput(
            "invalid modular meta/channel count".into(),
        ));
    }
    for (i, c) in channels.iter().enumerate() {
        if c.width.checked_mul(c.height) != Some(c.samples.len()) {
            return Err(Error::InvalidInput(
                "modular channel dimensions do not match samples".into(),
            ));
        }
        if (i < meta && (c.hshift >= 0 || c.vshift >= 0))
            || (i >= meta && (c.hshift < 0 || c.vshift < 0))
        {
            return Err(Error::InvalidInput(
                "modular meta channels must be a contiguous prefix".into(),
            ));
        }
    }
    Ok(())
}

fn equal_range(
    channels: &[Channel],
    meta: usize,
    begin: usize,
    count: usize,
) -> Result<Option<usize>> {
    validate_layout(channels, meta)?;
    let end = begin
        .checked_add(count)
        .filter(|&end| count != 0 && begin >= meta && end <= channels.len())
        .ok_or_else(|| {
            Error::InvalidInput("transform must select an existing non-meta channel range".into())
        })?;
    let first = &channels[begin];
    if channels[begin..end].iter().any(|c| {
        c.width != first.width
            || c.height != first.height
            || c.hshift != first.hshift
            || c.vshift != first.vshift
    }) {
        return Ok(None);
    }
    Ok(Some(end))
}

/// FwdPalette orchestration and channel insertion, including exact residual
/// fallback and zero-delta masks computed BEFORE any channel is overwritten.
/// `begin`/`count` refer to the current layout, not original component numbers.
/// None means source eligibility declined; errors and declines leave all inputs
/// unchanged. Existing meta tables and channels outside the range are preserved.
#[allow(clippy::too_many_arguments)]
pub fn apply_palette_range(
    channels: &mut Vec<Channel>,
    meta: &mut usize,
    bits: u8,
    begin: usize,
    count: usize,
    max_colors: u32,
    ordered: bool,
    mode: PaletteMode,
) -> Result<Option<Transform>> {
    super::palette::apply_palette_range_with_options(
        channels,
        meta,
        bits,
        begin,
        count,
        max_colors,
        ordered,
        mode,
        super::palette::PaletteTools::default(),
    )
}

/// Applies RCT after any number of palette insertions. The parent chooses its
/// range/type; this helper preserves shifts and returns the correct wire begin.
pub fn apply_rct(
    channels: &mut [Channel],
    meta: usize,
    begin: usize,
    kind: u32,
) -> Result<Option<Transform>> {
    if kind >= 42 {
        return Err(Error::InvalidInput("invalid RCT type".into()));
    }
    let Some(end) = equal_range(channels, meta, begin, 3)? else {
        return Ok(None);
    };
    if kind == 0 {
        return Ok(None);
    }
    let [a, b, c] = &mut channels[begin..end] else {
        unreachable!("validated RCT channel range")
    };
    super::rct_simd::forward_rows(
        [&mut a.samples, &mut b.samples, &mut c.samples],
        a.width,
        kind,
    )?;
    Ok(Some(if begin == 0 {
        Transform::Rct(kind)
    } else {
        Transform::RctAt { begin, kind }
    }))
}

pub fn apply_squeeze(
    channels: &mut Vec<Channel>,
    begin: usize,
    horizontal: bool,
) -> Result<Vec<Transform>> {
    if begin >= channels.len() {
        return Err(Error::InvalidInput("squeeze begin out of range".into()));
    }
    let eligible = |c: &Channel| {
        if horizontal {
            c.width >= 2
        } else {
            c.height >= 2
        }
    };
    let mut end = channels.len();
    let mut transforms = Vec::new();
    // Reverse ranges keep untouched channel indices stable as in-place residual
    // channels are inserted. The wire count is at most 19 per squeeze operation.
    while end > begin {
        if !eligible(&channels[end - 1]) {
            end -= 1;
            continue;
        }
        let mut first = end - 1;
        while first > begin && end - first < 19 && eligible(&channels[first - 1]) {
            first -= 1;
        }
        let count = end - first;
        if channels.len() + count > 8192 {
            return Err(Error::Limit("squeeze channel count"));
        }
        let mut residuals = Vec::with_capacity(count);
        for c in &mut channels[first..end] {
            let s = transforms::squeeze_forward(&c.samples, c.width, c.height, horizontal)?;
            c.width = s.average_width;
            c.height = s.average_height;
            c.samples = s.average;
            if horizontal {
                if c.hshift >= 0 {
                    c.hshift += 1;
                }
            } else if c.vshift >= 0 {
                c.vshift += 1;
            }
            residuals.push(Channel {
                width: s.residual_width,
                height: s.residual_height,
                hshift: c.hshift,
                vshift: c.vshift,
                samples: s.residual,
            });
        }
        channels.splice(end..end, residuals);
        transforms.push(Transform::Squeeze {
            horizontal,
            begin: first,
            channels: count,
        });
        end = first;
    }
    Ok(transforms)
}

pub fn write_header(
    w: &mut BitWriter,
    transforms: &[Transform],
    wp: &weighted::Header,
    global: bool,
) -> Result<()> {
    wp.validate()?;
    if transforms.len() > 273 {
        return Err(Error::Limit("Modular transform count"));
    }
    for transform in transforms {
        validate_transform(transform)?;
    }
    w.write(1, u64::from(global))?;
    wp.write_header(w)?;
    write_choice(
        w,
        transforms.len() as u32,
        [(0, 0), (1, 0), (2, 4), (18, 8)],
    )?;
    for transform in transforms {
        match *transform {
            Transform::Rct(kind) | Transform::RctAt { kind, .. } => {
                let begin = if let Transform::RctAt { begin, .. } = *transform {
                    begin
                } else {
                    0
                };
                w.write(2, 0)?;
                write_begin(w, begin)?;
                write_choice(w, kind, [(6, 0), (0, 2), (2, 4), (10, 6)])?;
            }
            Transform::Palette {
                begin,
                channels,
                colors,
                deltas,
                predictor,
            } => {
                w.write(2, 1)?;
                write_begin(w, begin)?;
                write_choice(w, channels as u32, [(1, 0), (3, 0), (4, 0), (1, 13)])?;
                write_choice(
                    w,
                    colors as u32,
                    [(0, 8), (256, 10), (1280, 12), (5376, 16)],
                )?;
                write_choice(w, deltas as u32, [(0, 0), (1, 8), (257, 10), (1281, 16)])?;
                w.write(4, predictor as u64)?;
            }
            Transform::Squeeze {
                horizontal,
                begin,
                channels,
            } => {
                w.write(2, 2)?;
                write_choice(w, 1, [(0, 0), (1, 4), (9, 6), (41, 8)])?;
                w.write(1, u64::from(horizontal))?;
                w.write(1, 1)?; // In place.
                write_choice(w, begin as u32, [(0, 3), (8, 6), (72, 10), (1096, 13)])?;
                write_choice(w, channels as u32, [(1, 0), (2, 0), (3, 0), (4, 4)])?;
            }
            Transform::SqueezeScript(ref parameters) => {
                w.write(2, 2)?;
                write_choice(
                    w,
                    parameters.len() as u32,
                    [(0, 0), (1, 4), (9, 6), (41, 8)],
                )?;
                for p in parameters {
                    w.write(1, u64::from(p.horizontal))?;
                    w.write(1, u64::from(p.in_place))?;
                    write_begin(w, p.begin)?;
                    write_choice(w, p.channels as u32, [(1, 0), (2, 0), (3, 0), (4, 4)])?;
                }
            }
        }
    }
    Ok(())
}

fn write_begin(w: &mut BitWriter, begin: usize) -> Result<()> {
    write_choice(w, begin as u32, [(0, 3), (8, 6), (72, 10), (1096, 13)])
}

pub(crate) fn validate_transform(transform: &Transform) -> Result<()> {
    let valid = match *transform {
        Transform::Rct(kind) => kind < 42,
        Transform::RctAt { begin, kind } => begin <= 9287 && kind < 42,
        Transform::Palette {
            begin,
            channels,
            colors,
            deltas,
            predictor,
        } => {
            begin <= 9287
                && (1..=8192).contains(&channels)
                && colors <= 70911
                && deltas <= 66816
                && (predictor as u32) < 14
        }
        Transform::Squeeze {
            begin, channels, ..
        } => begin <= 9287 && (1..=19).contains(&channels),
        Transform::SqueezeScript(ref parameters) => {
            parameters.len() <= 296
                && parameters
                    .iter()
                    .all(|p| p.begin <= 9287 && (1..=19).contains(&p.channels))
        }
    };
    if !valid {
        return Err(Error::InvalidInput(
            "Modular transform fields exceed wire range".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn undo_palette_for_test(channels: &mut Vec<Channel>, transform: Transform, bits: u8) {
    use super::predictor::{Neighbors, predict_one};
    let Transform::Palette {
        begin,
        channels: count,
        colors,
        deltas,
        predictor,
    } = transform
    else {
        panic!("expected palette")
    };
    let table = channels.remove(0);
    let indexed = channels.remove(begin);
    assert_eq!((table.width, table.height), (colors + deltas, count));
    let mut output = Vec::new();
    for c in 0..count {
        let mut decoded = Channel {
            samples: vec![0; indexed.samples.len()],
            ..indexed.clone()
        };
        for (pos, &index) in indexed.samples.iter().enumerate() {
            assert!(index >= 0);
            let index = index as usize;
            let value = if index < table.width {
                table.samples[c * table.width + index]
            } else if c >= 3 {
                0
            } else {
                let i = index - table.width;
                let max = (1u64 << bits.min(24)) - 1;
                if i < 64 {
                    (((i >> (2 * c)) & 3) as u64 * max / 4) as i32
                        + (1i32 << bits.min(24).saturating_sub(3))
                } else {
                    ((((i - 64) / 5usize.pow(c as u32)) % 5) as u64 * max / 4) as i32
                }
            };
            let prediction = if index < deltas {
                let n = Neighbors::from_plane(
                    &decoded.samples,
                    decoded.width,
                    decoded.width,
                    pos % decoded.width,
                    pos / decoded.width,
                )
                .unwrap();
                predict_one(predictor, n, None).unwrap() as i32
            } else {
                0
            };
            decoded.samples[pos] = value.wrapping_add(prediction);
        }
        output.push(decoded);
    }
    channels.splice(begin..begin, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applying_rct_reuses_original_channel_allocations() {
        for kind in 1..42 {
            let values: Vec<_> = (0..17 * 5).map(|i| i * 17 - 300).collect();
            let mut channels: Vec<_> = (0..4).map(|_| channel(&values, 17)).collect();
            let pointers: Vec<_> = channels.iter().map(|c| c.samples.as_ptr()).collect();
            let extra = channels[3].clone();
            let mut expected = vec![values.clone(); 3];
            transforms::rct_forward(&mut expected, 0, kind).unwrap();
            crate::execution::serial(|| apply_rct(&mut channels, 0, 0, kind)).unwrap();
            assert_eq!(
                channels
                    .iter()
                    .map(|c| c.samples.as_ptr())
                    .collect::<Vec<_>>(),
                pointers
            );
            for (actual, expected) in channels.iter().zip(expected) {
                assert_eq!(actual.samples, expected);
            }
            assert_eq!(channels[3], extra);
        }
    }

    fn channel(values: &[i32], width: usize) -> Channel {
        assert_eq!(values.len() % width, 0);
        Channel {
            width,
            height: values.len() / width,
            hshift: 0,
            vshift: 0,
            samples: values.to_vec(),
        }
    }

    #[test]
    fn arbitrary_ranges_keep_untouched_channels_and_lifo_meta_tables() {
        let original = vec![
            channel(&[9, 8, 7, 6], 2),
            channel(&[-7, 21, -7, 21], 2),
            channel(&[300, 500, 300, 500], 2),
            channel(&[1, 2, 3, 4], 2),
        ];
        let mut channels = original.clone();
        let mut meta = 0;
        let a = apply_palette_range(
            &mut channels,
            &mut meta,
            16,
            1,
            1,
            2,
            false,
            PaletteMode::Absolute,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            a,
            Transform::Palette {
                begin: 1,
                channels: 1,
                colors: 2,
                deltas: 0,
                predictor: Predictor::Zero
            }
        );
        assert_eq!(meta, 1);
        assert_eq!(channels[0].samples, [-7, 21]);
        assert_eq!(channels[1], original[0]);
        assert_eq!(channels[2].samples, [0, 1, 0, 1]);
        assert_eq!(channels[3], original[2]);
        let b = apply_palette_range(
            &mut channels,
            &mut meta,
            16,
            3,
            1,
            2,
            true,
            PaletteMode::Absolute,
        )
        .unwrap()
        .unwrap();
        assert_eq!(meta, 2);
        assert_eq!(channels[0].samples, [300, 500]);
        assert_eq!(channels[1].samples, [-7, 21]);
        assert_eq!(channels[2], original[0]);
        assert_eq!(channels[5], original[3]);
        undo_palette_for_test(&mut channels, b, 16);
        undo_palette_for_test(&mut channels, a, 16);
        assert_eq!(channels, original);
    }

    #[test]
    fn zero_delta_mask_precedes_mutation_and_shifts_implicit_indices() {
        let original = vec![
            channel(&[41, 42, 43, 44, 45, 46, 47, 48], 4),
            channel(&[0, 0, 3, 3, 3, 0, 0, 3], 4),
            channel(&[0, 0, 3, 3, 3, 0, 0, 3], 4),
            channel(&[0, 0, 3, 3, 3, 0, 0, 3], 4),
            channel(&[8; 8], 4),
        ];
        let mut channels = original.clone();
        let mut meta = 0;
        let t = apply_palette_range(
            &mut channels,
            &mut meta,
            8,
            1,
            3,
            2,
            false,
            PaletteMode::ExactZero(Predictor::Left),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            t,
            Transform::Palette {
                begin: 1,
                channels: 3,
                colors: 1,
                deltas: 1,
                predictor: Predictor::Left
            }
        );
        assert_eq!(channels[0].samples, [0, 3, 0, 3, 0, 3]);
        assert_eq!(channels[2].samples, [0, 0, 1, 0, 1, 66, 0, 1]);
        assert_eq!(channels[1], original[0]);
        assert_eq!(channels[3], original[4]);
        undo_palette_for_test(&mut channels, t, 8);
        assert_eq!(channels, original);
    }

    #[test]
    fn source_delta_decline_falls_back_to_absolute_and_single_channel() {
        let original = vec![
            channel(&[1, 7, 1, 7], 4),
            channel(&[2, 11, 2, 11], 4),
            channel(&[3, 13, 3, 13], 4),
        ];
        let planes: Vec<_> = original.iter().map(|c| c.samples.clone()).collect();
        assert!(
            transforms::delta_palette_forward(&planes, 0, 3, 4, 1, 2, 8, Predictor::Left)
                .unwrap()
                .is_none()
        );
        let mut channels = original.clone();
        let mut meta = 0;
        let t = apply_palette_range(
            &mut channels,
            &mut meta,
            8,
            0,
            3,
            2,
            false,
            PaletteMode::ExactDelta(Predictor::Left),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            t,
            Transform::Palette {
                begin: 0,
                channels: 3,
                colors: 2,
                deltas: 0,
                predictor: Predictor::Zero
            }
        );
        undo_palette_for_test(&mut channels, t, 8);
        assert_eq!(channels, original);
        let mut channels = vec![channel(&[i32::MIN, i32::MAX, 0], 3)];
        let original = channels.clone();
        let mut meta = 0;
        let t = apply_palette_range(
            &mut channels,
            &mut meta,
            31,
            0,
            1,
            3,
            true,
            PaletteMode::ExactZero(Predictor::Top),
        )
        .unwrap()
        .unwrap();
        assert!(matches!(t, Transform::Palette { deltas: 0, .. }));
        undo_palette_for_test(&mut channels, t, 31);
        assert_eq!(channels, original);
    }

    #[test]
    fn squeeze_script_wire_supports_default_and_append_parameters() {
        use super::super::layout::SqueezeParams;
        let mut w = BitWriter::new(64);
        write_header(
            &mut w,
            &[Transform::SqueezeScript(Vec::new())],
            &weighted::Header::default(),
            false,
        )
        .unwrap();
        assert_eq!(w.bits_written(), 8);
        assert_eq!(w.into_bytes().unwrap(), [0x26]);
        let parameters = vec![
            SqueezeParams {
                horizontal: true,
                in_place: false,
                begin: 1,
                channels: 2,
            },
            SqueezeParams {
                horizontal: false,
                in_place: true,
                begin: 9287,
                channels: 19,
            },
        ];
        let mut w = BitWriter::new(64);
        write_header(
            &mut w,
            &[Transform::SqueezeScript(parameters.clone())],
            &weighted::Header::default(),
            false,
        )
        .unwrap();
        let written = w.bits_written();
        let mut r = Reader::new(w);
        assert_eq!(r.bits(6), 0x26);
        assert_eq!(r.choice([0, 1, 9, 41], [0, 4, 6, 8]), 2);
        for p in parameters {
            assert_eq!(r.bits(1), u32::from(p.horizontal));
            assert_eq!(r.bits(1), u32::from(p.in_place));
            assert_eq!(r.choice([0, 8, 72, 1096], [3, 6, 10, 13]), p.begin as u32);
            assert_eq!(r.choice([1, 2, 3, 4], [0, 0, 0, 4]), p.channels as u32);
        }
        assert_eq!(r.pos, written);
        for parameters in [
            vec![SqueezeParams {
                horizontal: true,
                in_place: true,
                begin: 0,
                channels: 20,
            }],
            vec![SqueezeParams {
                horizontal: true,
                in_place: true,
                begin: 9288,
                channels: 1,
            }],
            vec![
                SqueezeParams {
                    horizontal: true,
                    in_place: true,
                    begin: 0,
                    channels: 1
                };
                297
            ],
        ] {
            let mut w = BitWriter::new(64);
            assert!(
                write_header(
                    &mut w,
                    &[Transform::SqueezeScript(parameters)],
                    &weighted::Header::default(),
                    false
                )
                .is_err()
            );
            assert_eq!(w.bits_written(), 0);
        }
    }

    #[test]
    fn exact_delta_and_empty_explicit_palette_tables_roundtrip() {
        for bits in [1, 8, 16] {
            for width in [1, 3, 8] {
                for p in [Predictor::Gradient, Predictor::Left, Predictor::Top] {
                    let original: Vec<_> = (0..4)
                        .map(|c| {
                            channel(
                                &(0..width * 3)
                                    .map(|i| (i as i32 - 5) * (c + 1))
                                    .collect::<Vec<_>>(),
                                width,
                            )
                        })
                        .collect();
                    let mut channels = original.clone();
                    let mut meta = 0;
                    let t = apply_palette_range(
                        &mut channels,
                        &mut meta,
                        bits,
                        0,
                        4,
                        65536,
                        true,
                        PaletteMode::ExactDelta(p),
                    )
                    .unwrap()
                    .unwrap();
                    assert!(
                        matches!(t,Transform::Palette{colors:0,deltas:1..,predictor,..} if predictor==p)
                    );
                    undo_palette_for_test(&mut channels, t, bits);
                    assert_eq!(channels, original);
                }
            }
        }
        let original = vec![channel(&[0, 255], 2); 3];
        let mut channels = original.clone();
        let mut meta = 0;
        let t = apply_palette_range(
            &mut channels,
            &mut meta,
            8,
            0,
            3,
            2,
            false,
            PaletteMode::Absolute,
        )
        .unwrap()
        .unwrap();
        assert_eq!(channels[0].width, 0);
        assert_eq!(channels[0].height, 3);
        assert!(channels[0].samples.is_empty());
        undo_palette_for_test(&mut channels, t, 8);
        assert_eq!(channels, original);
    }

    #[test]
    fn range_errors_and_geometry_declines_do_not_modify_input() {
        let original = vec![channel(&[1, 2, 3, 4], 2); 3];
        for (begin, count) in [(0, 0), (3, 1), (2, 2), (usize::MAX, 2)] {
            let mut channels = original.clone();
            let mut meta = 0;
            assert!(
                apply_palette_range(
                    &mut channels,
                    &mut meta,
                    8,
                    begin,
                    count,
                    4,
                    true,
                    PaletteMode::Absolute
                )
                .is_err()
            );
            assert_eq!(channels, original);
            assert_eq!(meta, 0);
        }
        for mismatch in 0..3 {
            let mut channels = original.clone();
            match mismatch {
                0 => {
                    channels[1].width = 1;
                    channels[1].height = 4;
                }
                1 => channels[1].hshift = 1,
                _ => channels[1].vshift = 1,
            }
            let original = channels.clone();
            let mut meta = 0;
            assert!(
                apply_palette_range(
                    &mut channels,
                    &mut meta,
                    8,
                    0,
                    3,
                    4,
                    true,
                    PaletteMode::Absolute
                )
                .unwrap()
                .is_none()
            );
            assert_eq!(channels, original);
            assert_eq!(meta, 0);
        }
        let mut channels = original.clone();
        let mut meta = 0;
        assert!(
            apply_palette_range(
                &mut channels,
                &mut meta,
                8,
                0,
                3,
                4,
                true,
                PaletteMode::ExactZero(Predictor::Weighted)
            )
            .is_err()
        );
        channels[0].samples.pop();
        let invalid = channels.clone();
        assert!(
            apply_palette_range(
                &mut channels,
                &mut meta,
                8,
                0,
                3,
                4,
                true,
                PaletteMode::Absolute
            )
            .is_err()
        );
        assert_eq!(channels, invalid);
    }

    struct Reader {
        bytes: Vec<u8>,
        pos: usize,
    }
    impl Reader {
        fn new(mut w: BitWriter) -> Self {
            w.zero_pad_to_byte().unwrap();
            Self {
                bytes: w.into_bytes().unwrap(),
                pos: 0,
            }
        }
        fn bits(&mut self, n: usize) -> u32 {
            let mut v = 0;
            for i in 0..n {
                v |= u32::from((self.bytes[self.pos / 8] >> (self.pos % 8)) & 1) << i;
                self.pos += 1;
            }
            v
        }
        fn choice(&mut self, offsets: [u32; 4], widths: [usize; 4]) -> u32 {
            let c = self.bits(2) as usize;
            offsets[c] + self.bits(widths[c])
        }
    }

    #[test]
    fn palette_wire_begin_and_field_boundaries_match_source_choices() {
        for begin in [0, 7, 8, 71, 72, 1095, 1096, 9287] {
            for (count, colors, deltas) in [
                (1, 0, 0),
                (3, 255, 1),
                (4, 256, 256),
                (8192, 1279, 257),
                (1, 1280, 1280),
                (3, 5375, 1281),
                (4, 5376, 66816),
                (1, 70911, 0),
            ] {
                let t = Transform::Palette {
                    begin,
                    channels: count,
                    colors,
                    deltas,
                    predictor: Predictor::Gradient,
                };
                let mut w = BitWriter::new(64);
                write_header(&mut w, &[t], &weighted::Header::default(), false).unwrap();
                let written = w.bits_written();
                let mut r = Reader::new(w);
                assert_eq!(r.bits(1), 0);
                assert_eq!(r.bits(1), 1);
                assert_eq!(r.choice([0, 1, 2, 18], [0, 0, 4, 8]), 1);
                assert_eq!(r.bits(2), 1);
                assert_eq!(r.choice([0, 8, 72, 1096], [3, 6, 10, 13]), begin as u32);
                assert_eq!(r.choice([1, 3, 4, 1], [0, 0, 0, 13]), count as u32);
                assert_eq!(
                    r.choice([0, 256, 1280, 5376], [8, 10, 12, 16]),
                    colors as u32
                );
                assert_eq!(r.choice([0, 1, 257, 1281], [0, 8, 10, 16]), deltas as u32);
                assert_eq!(r.bits(4), Predictor::Gradient as u32);
                assert_eq!(r.pos, written);
            }
        }
        let mut w = BitWriter::new(128);
        w.write(3, 5).unwrap();
        for t in [
            Transform::RctAt {
                begin: 9288,
                kind: 6,
            },
            Transform::Rct(42),
            Transform::Palette {
                begin: 0,
                channels: 0,
                colors: 2,
                deltas: 0,
                predictor: Predictor::Zero,
            },
            Transform::Palette {
                begin: 0,
                channels: 1,
                colors: 70912,
                deltas: 0,
                predictor: Predictor::Zero,
            },
        ] {
            assert!(write_header(&mut w, &[t], &weighted::Header::default(), false).is_err());
            assert_eq!(w.bits_written(), 3);
        }
    }

    #[test]
    fn rct_after_meta_insertion_preserves_meta_and_surrounding_channels() {
        let mut channels = vec![
            channel(&[9, 7, 9, 7], 2),
            channel(&[1, 4, 9, 16], 2),
            channel(&[2, 5, 10, 17], 2),
            channel(&[3, 6, 11, 18], 2),
            channel(&[8; 4], 2),
        ];
        let mut meta = 0;
        apply_palette_range(
            &mut channels,
            &mut meta,
            8,
            0,
            1,
            2,
            true,
            PaletteMode::Absolute,
        )
        .unwrap()
        .unwrap();
        let table = channels[0].clone();
        let indices = channels[1].clone();
        let last = channels[5].clone();
        let mut reference: Vec<_> = channels[2..5].iter().map(|c| c.samples.clone()).collect();
        transforms::rct_forward(&mut reference, 0, 6).unwrap();
        let t = apply_rct(&mut channels, meta, 2, 6).unwrap().unwrap();
        assert_eq!(t, Transform::RctAt { begin: 2, kind: 6 });
        assert_eq!(channels[0], table);
        assert_eq!(channels[1], indices);
        assert_eq!(channels[5], last);
        for (c, p) in channels[2..5].iter().zip(reference) {
            assert_eq!(c.samples, p);
        }
        let mut w = BitWriter::new(16);
        write_header(&mut w, &[t], &weighted::Header::default(), false).unwrap();
        let mut r = Reader::new(w);
        assert_eq!(r.bits(6), 6); // global=0, WP=1, one transform, RCT id=0
        assert_eq!(r.choice([0, 8, 72, 1096], [3, 6, 10, 13]), 2);
        assert_eq!(r.choice([6, 0, 2, 10], [0, 2, 4, 6]), 6);
    }
}
