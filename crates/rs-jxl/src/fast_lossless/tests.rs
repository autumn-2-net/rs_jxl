use super::*;
use pixels::Sink;

#[path = "wide_prediction_tests.rs"]
mod wide_prediction;

#[path = "narrow_prediction_tests.rs"]
mod narrow_prediction;

fn view(bytes: &[u8], width: usize, height: usize, channels: u8, depth: u8) -> ImageView<'_> {
    ImageView {
        pixels: bytes,
        width,
        height,
        row_stride: width * usize::from(channels) * if depth <= 8 { 1 } else { 2 },
        channels,
        bit_depth: depth,
        big_endian: false,
    }
}

#[derive(Default)]
struct Record {
    decoded: Vec<u32>,
    runs: Vec<usize>,
}

impl Sink for Record {
    fn chunk(&mut self, run: usize, values: &[u32]) -> Result<()> {
        if run != 0 {
            assert!(run >= 8);
            self.runs.push(run);
            self.decoded.resize(self.decoded.len() + run, 0);
        }
        self.decoded.extend(values);
        Ok(())
    }
    fn finish(&mut self, run: usize) -> Result<()> {
        self.chunk(run, &[])
    }
}

#[test]
fn chunk_rle_reconstructs_prefixes_tails_and_row_crossings() {
    for chunk in [8, 16] {
        for zeros in [0, 1, 7, 8, 15, 16, 17, 31, 32, 511] {
            let mut row = vec![0; zeros];
            row.extend([3, 0, 2, 0]);
            let mut p = Processor::new(Record::default());
            p.row(&row, chunk).unwrap();
            p.row(&[0; 32], chunk).unwrap();
            p.row(&[0; 7], chunk).unwrap();
            p.finish().unwrap();
            row.extend([0; 39]);
            assert_eq!(p.sink.decoded, row);
        }
    }
    let mut p = Processor::new(Record::default());
    p.row(&[0; 7], 8).unwrap();
    p.row(&[0; 7], 8).unwrap();
    p.finish().unwrap();
    assert!(p.sink.runs.is_empty(), "short rows do not start RLE");
}

#[test]
fn histogram_does_not_count_open_final_run() {
    let mut p = Processor::new(Histogram::default());
    p.row(&[0; 16], 8).unwrap();
    p.finish().unwrap();
    assert_eq!(p.sink.raw, [0; RAW]);
    assert_eq!(p.sink.lz, [0; LZ]);
    p.row(&[0; 16], 8).unwrap();
    p.row(&[2], 8).unwrap();
    p.finish().unwrap();
    assert_eq!(p.sink.raw[0], 1);
    assert_eq!(p.sink.raw[2], 1);
    assert_eq!(p.sink.lz[8], 1);
}

#[test]
fn scalar_and_native_gradient_match_independent_wrapping_lane_oracle() {
    for narrow in [false, true] {
        let mut inputs = [[0i32; 259]; 4];
        let mut state = 0x9187_6af1u32;
        for row in &mut inputs {
            for v in row {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = if narrow {
                    i32::from(state as i16)
                } else {
                    (state % 262143) as i32 - 131071
                };
            }
        }
        let [row, left, top, tl] = inputs;
        let oracle: Vec<_> = (0..row.len())
            .map(|i| {
                let (a, b, c) = (left[i], top[i], tl[i]);
                let pred = if narrow {
                    let (a, b, c) = (a as i16, b as i16, c as i16);
                    let ac = a.wrapping_sub(c);
                    let bc = b.wrapping_sub(c);
                    i32::from(if ac ^ bc < 0 {
                        ac.wrapping_add(b)
                    } else if a.wrapping_sub(b) ^ bc < 0 {
                        b
                    } else {
                        a
                    })
                } else {
                    let ac = a - c;
                    let bc = b - c;
                    if ac ^ bc < 0 {
                        ac + b
                    } else if (a - b) ^ bc < 0 {
                        b
                    } else {
                        a
                    }
                };
                let d = row[i] - pred;
                let packed = if d < 0 {
                    (-2 * d - 1) as u32
                } else {
                    (2 * d) as u32
                };
                if narrow { packed & 65535 } else { packed }
            })
            .collect();
        for backend in [Backend::Scalar, Backend::Auto] {
            for n in [0, 1, 7, 8, 15, 16, 17, 256, 259] {
                let mut output = vec![0; n];
                Kernel::choose(backend).predict(
                    &row[..n],
                    &left[..n],
                    &top[..n],
                    &tl[..n],
                    narrow,
                    &mut output,
                );
                assert_eq!(output, oracle[..n], "{backend:?}, narrow={narrow}, n={n}");
            }
        }
    }
}

#[test]
fn ycocg_native_matches_scalar_and_reverses_all_depth_profiles() {
    for depth in [1, 8, 9, 13, 14, 15, 16] {
        let max = (1 << depth) - 1;
        let mut original = [[0; 257]; 4];
        for (c, row) in original.iter_mut().enumerate() {
            for (x, v) in row.iter_mut().enumerate().skip(1) {
                *v = ((x * 371 + c * 8309) as i32) & max;
            }
        }
        for width in [1, 7, 8, 15, 16, 17, 255, 256] {
            let mut scalar = original;
            let mut native = original;
            Kernel::choose(Backend::Scalar).ycocg(&mut scalar, width);
            Kernel::choose(Backend::Auto).ycocg(&mut native, width);
            assert_eq!(native, scalar);
            for x in 1..=width {
                let (y, co, cg) = (native[0][x], native[1][x], native[2][x]);
                let tmp = y - (cg >> 1);
                let g = cg + tmp;
                let b = tmp - (co >> 1);
                let r = b + co;
                assert_eq!(
                    [r, g, b, native[3][x]],
                    [
                        original[0][x],
                        original[1][x],
                        original[2][x],
                        original[3][x]
                    ]
                );
            }
        }
    }
}

#[test]
fn input_dispatch_preserves_guards_and_sampled_encoded_row_residuals() {
    const CANARY: i32 = -123456;
    for depth in [1, 8, 9, 14, 15, 16] {
        let max = (1usize << depth) - 1;
        let storage = if depth <= 8 { 1 } else { 2 };
        for channels in 1u8..=4 {
            let count = usize::from(channels);
            let sample = |x: usize, y: usize, c: usize| -> u16 {
                if c == 3 && x.is_multiple_of(3) {
                    0
                } else {
                    ((x * 8191 + y * 317 + c * 7193) & max) as u16
                }
            };
            for big_endian in [false, true] {
                let row_bytes = 259 * count * storage;
                let stride = row_bytes + 5;
                let mut bytes = vec![0xa7; 2 * stride + row_bytes];
                for y in 0..3 {
                    for x in 0..259 {
                        for c in 0..count {
                            let value = sample(x, y, c);
                            let at = y * stride + (x * count + c) * storage;
                            if storage == 1 {
                                bytes[at] = value as u8;
                            } else {
                                bytes[at..at + 2].copy_from_slice(&if big_endian {
                                    value.to_be_bytes()
                                } else {
                                    value.to_le_bytes()
                                });
                            }
                        }
                    }
                }
                let image = ImageView {
                    pixels: &bytes,
                    width: 259,
                    height: 3,
                    row_stride: stride,
                    channels,
                    bit_depth: depth,
                    big_endian,
                };
                for width in [0, 1, 15, 16, 17, 255, 256] {
                    let mut expected = [[CANARY; 257]; 4];
                    for (c, row) in expected.iter_mut().enumerate().take(count) {
                        for (dx, value) in row[1..1 + width].iter_mut().enumerate() {
                            *value = i32::from(sample(3 + dx, 2, c));
                        }
                    }
                    for backend in [Backend::Scalar, Backend::Auto] {
                        let mut actual = [[CANARY; 257]; 4];
                        Kernel::choose(backend)
                            .load_row(image, 3, 2, width, &mut actual)
                            .unwrap();
                        assert_eq!(
                            actual, expected,
                            "{backend:?} depth={depth} channels={channels} be={big_endian} width={width}"
                        );
                        let mut transformed = expected;
                        if channels >= 3 {
                            let [red, green, blue, _] = &mut transformed;
                            for ((r, g), b) in red[1..1 + width]
                                .iter_mut()
                                .zip(&mut green[1..1 + width])
                                .zip(&mut blue[1..1 + width])
                            {
                                let middle = (*r + *b) / 2;
                                (*r, *g, *b) = ((*g + middle) / 2, *r - *b, *g - middle);
                            }
                        }
                        let mut actual = [[CANARY; 257]; 4];
                        Kernel::choose(backend)
                            .load_transformed_row(image, 3, 2, width, &mut actual)
                            .unwrap();
                        assert_eq!(
                            actual, transformed,
                            "transformed {backend:?} depth={depth} channels={channels} be={big_endian} width={width}"
                        );
                    }
                }
                for skip in [0, 1] {
                    let process = |backend| {
                        let mut processors: Vec<_> = (0..count)
                            .map(|_| Processor::new(Record::default()))
                            .collect();
                        pixels::process(
                            image,
                            Area {
                                x: 3,
                                y: 1,
                                width: 33,
                                height: 2,
                                skip,
                            },
                            None,
                            Kernel::choose(backend),
                            &mut processors,
                        )
                        .unwrap();
                        processors
                            .into_iter()
                            .map(|p| {
                                assert_eq!(p.sink.decoded.len(), 33 * (2 - skip));
                                p.sink.decoded
                            })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(process(Backend::Auto), process(Backend::Scalar));
                }
            }
        }
    }
}

#[test]
fn row_conversion_and_predictor_expand_to_exact_original_planes() {
    for depth in [1, 8, 13, 14, 16] {
        let max = (1u32 << depth) - 1;
        let values: Vec<_> = (0..17 * 5 * 4)
            .map(|i| ((i * 8191 + i / 7 * 3) as u32 & max) as u16)
            .collect();
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|&v| {
                if depth <= 8 {
                    vec![v as u8]
                } else {
                    v.to_le_bytes().to_vec()
                }
            })
            .collect();
        let image = view(&bytes, 17, 5, 4, depth);
        for backend in [Backend::Scalar, Backend::Auto] {
            let mut processors: Vec<_> =
                (0..4).map(|_| Processor::new(Record::default())).collect();
            pixels::process(
                image,
                Area {
                    x: 0,
                    y: 0,
                    width: 17,
                    height: 5,
                    skip: 0,
                },
                None,
                Kernel::choose(backend),
                &mut processors,
            )
            .unwrap();
            let mut planes = vec![vec![0i32; 85]; 4];
            for (plane, p) in planes.iter_mut().zip(processors) {
                for (i, packed) in p.sink.decoded.into_iter().enumerate() {
                    let x = i % 17;
                    let y = i / 17;
                    let left = if x != 0 {
                        plane[i - 1]
                    } else if y != 0 {
                        plane[i - 17]
                    } else {
                        0
                    };
                    let top = if y != 0 { plane[i - 17] } else { left };
                    let tl = if y == 0 {
                        left
                    } else if x == 0 {
                        top
                    } else {
                        plane[i - 18]
                    };
                    let pred = (left + top - tl).clamp(left.min(top), left.max(top));
                    plane[i] = pred
                        + if packed & 1 != 0 {
                            -((packed >> 1) as i32) - 1
                        } else {
                            (packed >> 1) as i32
                        };
                }
            }
            for i in 0..85 {
                let tmp = planes[0][i] - (planes[2][i] >> 1);
                let b = tmp - (planes[1][i] >> 1);
                let actual = [b + planes[1][i], tmp + planes[2][i], b, planes[3][i]];
                assert_eq!(
                    actual,
                    std::array::from_fn(|c| i32::from(values[4 * i + c])),
                    "depth={depth} {backend:?}"
                );
            }
        }
    }
}

#[test]
fn palette_gates_collision_zero_sentinel_and_dictionary_reconstruction() {
    let colors = [
        [0, 0, 0, 0],
        [90, 1, 33, 0],
        [7, 231, 41, 255],
        [200, 11, 12, 80],
    ];
    let bytes: Vec<_> = colors.iter().cycle().take(64).flatten().copied().collect();
    let image = view(&bytes, 16, 4, 4, 8);
    let options = Options::default();
    let palette = Palette::detect(image, &options).expect("small colored palette");
    for y in 0..image.height {
        for x in 0..image.width {
            let pixel = image.pixel8(x, y);
            assert_eq!(palette.colors[usize::from(palette.index(pixel))], pixel);
        }
    }
    assert!(
        Palette::detect(
            image,
            &Options {
                effort: 1,
                ..options
            }
        )
        .is_none()
    );
    assert!(
        Palette::detect(
            image,
            &Options {
                oneshot: false,
                ..options
            }
        )
        .is_none()
    );
    let gray: Vec<_> = (0..64).flat_map(|v| [v, v, v]).collect();
    assert!(Palette::detect(view(&gray, 64, 1, 3, 8), &options).is_none());
    // Multiplicative inverse modulo 2^32: a nonzero color maps to hash zero.
    let alias = 0x0e8b_2f51u32;
    assert_eq!(alias.wrapping_mul(2_654_435_761), 1);
    for pair in [[0, alias], [alias, 0]] {
        let input: Vec<_> = pair.into_iter().flat_map(u32::to_le_bytes).collect();
        assert!(Palette::detect(view(&input, 2, 1, 4, 8), &options).is_none());
    }
}

#[test]
fn validates_buffers_samples_effort_and_budgets_before_output() {
    let good = view(&[1, 2, 3], 1, 1, 3, 8);
    let options = Options::default();
    for bad in [
        ImageView { width: 0, ..good },
        ImageView {
            channels: 5,
            ..good
        },
        ImageView {
            row_stride: 2,
            ..good
        },
        ImageView {
            pixels: &[1],
            ..good
        },
        ImageView {
            bit_depth: 1,
            ..good
        },
        ImageView {
            bit_depth: 17,
            ..good
        },
    ] {
        assert!(encode(bad, &options).is_err());
    }
    assert!(
        encode(
            good,
            &Options {
                effort: 128,
                ..options
            }
        )
        .is_err()
    );
    for limits in [
        Limits {
            max_pixels: 0,
            ..options.limits
        },
        Limits {
            max_output_bytes: 8,
            ..options.limits
        },
        Limits {
            max_groups: 0,
            ..options.limits
        },
    ] {
        assert!(encode(good, &Options { limits, ..options }).is_err());
    }
}

#[test]
fn standalone_header_is_frame_prefix_and_explicit_pool_is_deterministic() {
    let bytes: Vec<_> = (0..257 * 3 * 3).map(|i| (i * 17 + i / 11) as u8).collect();
    let image = view(&bytes, 257, 3, 3, 8);
    let options = Options {
        backend: Backend::Scalar,
        effort: 0,
        ..Options::default()
    };
    let full = encode(image, &options).unwrap();
    let frame = encode_frame(image, &options, true).unwrap();
    let mut header = BitWriter::new(128);
    header::image(&mut header, image).unwrap();
    assert_eq!(
        &full[..header.bytes().unwrap().len()],
        header.bytes().unwrap()
    );
    assert_eq!(&full[header.bytes().unwrap().len()..], frame);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    assert_eq!(full, encode_with_pool(image, &options, &pool).unwrap());
    assert_eq!(
        frame,
        encode_frame_with_pool(image, &options, true, &pool).unwrap()
    );
}

#[test]
fn stride_and_endianness_do_not_change_encoded_samples() {
    let samples = [0u16, 16383, 8191, 1, 17, 1023, 234, 1234];
    let le: Vec<_> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let mut be: Vec<_> = samples[..4].iter().flat_map(|v| v.to_be_bytes()).collect();
    be.extend([255; 3]);
    be.extend(samples[4..].iter().flat_map(|v| v.to_be_bytes()));
    let image = view(&le, 2, 2, 2, 14);
    let padded = ImageView {
        pixels: &be,
        row_stride: 11,
        big_endian: true,
        ..image
    };
    let options = Options {
        backend: Backend::Scalar,
        ..Options::default()
    };
    assert_eq!(
        encode(image, &options).unwrap(),
        encode(padded, &options).unwrap()
    );
}
