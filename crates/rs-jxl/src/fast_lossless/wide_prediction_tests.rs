//! Independent lane/row oracles for the source 15/16-bit fused AVX2 path.

use super::*;

fn oracle_residual(pixel: i32, left: i32, top: i32, topleft: i32) -> u32 {
    let prediction = (left + top - topleft).clamp(left.min(top), left.max(top));
    let delta = pixel - prediction;
    if delta < 0 {
        (-2 * delta - 1) as u32
    } else {
        (2 * delta) as u32
    }
}

#[test]
fn fused_wide_chunk_matches_independent_oracle_for_every_prefix_and_tail() {
    for backend in [Backend::Scalar, Backend::Auto] {
        let kernel = Kernel::choose(backend);
        for n in 0..=16 {
            for first_nonzero in 0..=n {
                let mut inputs = [[0; 17]; 4];
                for i in 0..n {
                    let left = (i * 8191 % 131071) as i32 - 65535;
                    let top = (i * 4133 % 131071) as i32 - 65535;
                    let topleft = 65535 - (i * 7159 % 131071) as i32;
                    let pred = (left + top - topleft).clamp(left.min(top), left.max(top));
                    let pixel = if i < first_nonzero || (i > first_nonzero && i % 2 == 0) {
                        pred
                    } else if pred >= 0 {
                        -65535
                    } else {
                        65535
                    };
                    for (row, value) in inputs.iter_mut().zip([pixel, left, top, topleft]) {
                        row[i + 1] = value;
                    }
                }
                let [row, left, top, tl] = inputs;
                let expected: Vec<_> = (1..=n)
                    .map(|i| oracle_residual(row[i], left[i], top[i], tl[i]))
                    .collect();
                let mut output = [0xa5a5_a5a5; 18];
                let prefix = kernel.predict_chunk_wide(
                    &row[1..1 + n],
                    &left[1..1 + n],
                    &top[1..1 + n],
                    &tl[1..1 + n],
                    &mut output[1..1 + n],
                );
                assert_eq!(prefix, first_nonzero, "{backend:?}, n={n}");
                assert_eq!(&output[1..1 + n], expected, "{backend:?}, n={n}");
                assert_eq!(output[0], 0xa5a5_a5a5);
                assert!(output[1 + n..].iter().all(|&n| n == 0xa5a5_a5a5));
            }
        }
    }
}

#[test]
fn fused_wide_chunk_checks_lengths_before_native_loads() {
    for backend in [Backend::Scalar, Backend::Auto] {
        let kernel = Kernel::choose(backend);
        for short in 0..4 {
            let mut inputs = vec![vec![0; 16]; 4];
            inputs[short].pop();
            let mut output = [0x1234; 16];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    kernel.predict_chunk_wide(
                        &inputs[0],
                        &inputs[1],
                        &inputs[2],
                        &inputs[3],
                        &mut output,
                    )
                }))
                .is_err()
            );
            assert_eq!(output, [0x1234; 16]);
        }
        assert!(
            std::panic::catch_unwind(|| {
                kernel.predict_chunk_wide(&[0; 17], &[0; 17], &[0; 17], &[0; 17], &mut [0; 17])
            })
            .is_err()
        );
    }
}

#[derive(Debug, Default, PartialEq)]
struct WideTrace {
    chunks: Vec<(usize, Vec<u32>)>,
    final_runs: Vec<usize>,
}

impl Sink for WideTrace {
    fn chunk(&mut self, run: usize, residuals: &[u32]) -> Result<()> {
        self.chunks.push((run, residuals.to_vec()));
        Ok(())
    }
    fn finish(&mut self, run: usize) -> Result<()> {
        self.final_runs.push(run);
        Ok(())
    }
}

#[test]
fn fused_wide_rows_keep_rle_events_and_sampling_final_run_rules() {
    for backend in [Backend::Scalar, Backend::Auto] {
        let kernel = Kernel::choose(backend);
        for width in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256] {
            for zeros in [0, 1, 7, 8, 9, 15, 16, 31, 40] {
                let row: Vec<_> = (0..width)
                    .map(|x| if x < zeros || x % 7 == 0 { 0 } else { 1 })
                    .collect();
                let zero = vec![0; width];
                let mut actual = Processor::new(WideTrace::default());
                let mut expected = Processor::new(WideTrace::default());
                let mut actual_hist = Processor::new(Histogram::default());
                let mut expected_hist = Processor::new(Histogram::default());
                for row in [&zero, &zero, &row, &zero] {
                    let residuals: Vec<_> =
                        row.iter().map(|&p| oracle_residual(p, 0, 0, 0)).collect();
                    actual
                        .predicted_wide_row(kernel, row, &zero, &zero, &zero)
                        .unwrap();
                    actual_hist
                        .predicted_wide_row(kernel, row, &zero, &zero, &zero)
                        .unwrap();
                    expected.row(&residuals, kernel.chunk_size()).unwrap();
                    expected_hist.row(&residuals, kernel.chunk_size()).unwrap();
                }
                actual.finish().unwrap();
                expected.finish().unwrap();
                actual_hist.finish().unwrap();
                expected_hist.finish().unwrap();
                assert_eq!(
                    actual.sink, expected.sink,
                    "{backend:?}, width={width}, zeros={zeros}"
                );
                assert_eq!(actual_hist.sink.raw, expected_hist.sink.raw);
                assert_eq!(actual_hist.sink.lz, expected_hist.sink.lz);
            }
        }
        let mut sample = Processor::new(Histogram::default());
        sample
            .predicted_wide_row(kernel, &[0; 32], &[0; 32], &[0; 32], &[0; 32])
            .unwrap();
        sample.finish().unwrap();
        assert_eq!(sample.sink.raw, [0; RAW]);
        assert_eq!(sample.sink.lz, [0; LZ]);
    }
}

pub(super) fn oracle_area(image: ImageView<'_>, area: Area) -> Vec<Vec<Vec<u32>>> {
    let channels = usize::from(image.channels);
    let mut rows = vec![Vec::new(); channels];
    let mut previous = vec![vec![0; area.width]; channels];
    for y in 0..area.height {
        let mut current: Vec<Vec<i32>> = (0..channels)
            .map(|c| {
                (0..area.width)
                    .map(|x| i32::from(image.sample(area.x + x, area.y + y, c)))
                    .collect()
            })
            .collect();
        if channels >= 3 {
            let (red, remaining) = current.split_first_mut().unwrap();
            let (green, remaining) = remaining.split_first_mut().unwrap();
            for ((r, g), b) in red.iter_mut().zip(green).zip(&mut remaining[0]) {
                let middle = (*r + *b) / 2;
                let (y, co, cg) = ((*g + middle) / 2, *r - *b, *g - middle);
                (*r, *g, *b) = (y, co, cg);
            }
        }
        if y >= area.skip {
            for c in 0..channels {
                let residuals = (0..area.width)
                    .map(|x| {
                        let left = if x != 0 {
                            current[c][x - 1]
                        } else if y != 0 {
                            previous[c][0]
                        } else {
                            0
                        };
                        let top = if y == 0 { left } else { previous[c][x] };
                        let tl = if y == 0 {
                            left
                        } else if x == 0 {
                            top
                        } else {
                            previous[c][x - 1]
                        };
                        oracle_residual(current[c][x], left, top, tl)
                    })
                    .collect();
                rows[c].push(residuals);
            }
        }
        previous = current;
    }
    rows
}

#[test]
fn fused_wide_image_and_sampling_match_independent_context_rows() {
    for depth in [15, 16] {
        for channels in 1..=4 {
            for big_endian in [false, true] {
                let (width, height) = (35, 6);
                let stride = width * usize::from(channels) * 2 + 3;
                let mut bytes = vec![0x79; stride * height - 3];
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..usize::from(channels) {
                            let value = if c == 3 && x % 3 == 0 {
                                0
                            } else {
                                ((x * 8191 + y * 971 + c * 211 + x * y) & ((1 << depth) - 1)) as u16
                            };
                            let at = y * stride + (x * usize::from(channels) + c) * 2;
                            bytes[at..at + 2].copy_from_slice(&if big_endian {
                                value.to_be_bytes()
                            } else {
                                value.to_le_bytes()
                            });
                        }
                    }
                }
                let image = ImageView {
                    pixels: &bytes,
                    width,
                    height,
                    row_stride: stride,
                    channels,
                    bit_depth: depth,
                    big_endian,
                };
                for backend in [Backend::Scalar, Backend::Auto] {
                    let kernel = Kernel::choose(backend);
                    for area in [
                        Area {
                            x: 0,
                            y: 0,
                            width,
                            height,
                            skip: 0,
                        },
                        Area {
                            x: 1,
                            y: 1,
                            width: 32,
                            height: 4,
                            skip: 1,
                        },
                    ] {
                        let expected = oracle_area(image, area);
                        let mut records: Vec<_> = (0..channels)
                            .map(|_| Processor::new(Record::default()))
                            .collect();
                        let mut histograms: Vec<_> = (0..channels)
                            .map(|_| Processor::new(Histogram::default()))
                            .collect();
                        pixels::process(image, area, None, kernel, &mut records).unwrap();
                        pixels::process(image, area, None, kernel, &mut histograms).unwrap();
                        for ((record, actual_hist), rows) in
                            records.into_iter().zip(histograms).zip(expected)
                        {
                            let mut expected_hist = Processor::new(Histogram::default());
                            for row in &rows {
                                expected_hist.row(row, kernel.chunk_size()).unwrap();
                            }
                            expected_hist.finish().unwrap();
                            assert_eq!(
                                record.sink.decoded,
                                rows.into_iter().flatten().collect::<Vec<_>>(),
                                "depth={depth}, channels={channels}, big_endian={big_endian}, {backend:?}"
                            );
                            assert_eq!(actual_hist.sink.raw, expected_hist.sink.raw);
                            assert_eq!(actual_hist.sink.lz, expected_hist.sink.lz);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fused_wide_encoder_sink_keeps_fixed_chunk_bits_and_output_errors() {
    for depth in [15, 16] {
        let raw = std::array::from_fn(|i| 1 + (i * 37 % 101) as u64);
        let lz = std::array::from_fn(|i| if i < 28 { 1 + (i * 17 % 53) as u64 } else { 0 });
        let code = PrefixCode::new(&raw, &lz, depth).unwrap();
        for backend in [Backend::Scalar, Backend::Auto] {
            let kernel = Kernel::choose(backend);
            let mut actual = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(16384),
                kernel,
            });
            let mut expected = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(16384),
                kernel: Kernel::choose(Backend::Scalar),
            });
            for zeros in [7, 8, 15, 16, 31, 39, 40, 256] {
                let mut row = vec![0; zeros];
                let max = (1 << depth) - 1;
                row.extend([1, -1, max, -max, 0, 0, 3]);
                row.extend([0; 64]);
                let neighbors = vec![0; row.len()];
                let residuals: Vec<_> = row.iter().map(|&p| oracle_residual(p, 0, 0, 0)).collect();
                actual
                    .predicted_wide_row(kernel, &row, &neighbors, &neighbors, &neighbors)
                    .unwrap();
                expected.row(&residuals, kernel.chunk_size()).unwrap();
            }
            actual.finish().unwrap();
            expected.finish().unwrap();
            assert_eq!(
                actual.sink.writer.bits_written(),
                expected.sink.writer.bits_written()
            );
            actual.sink.writer.zero_pad_to_byte().unwrap();
            expected.sink.writer.zero_pad_to_byte().unwrap();
            assert_eq!(
                actual.sink.writer.bytes().unwrap(),
                expected.sink.writer.bytes().unwrap()
            );
            let mut limited = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(0),
                kernel,
            });
            limited
                .predicted_wide_row(kernel, &[0; 16], &[0; 16], &[0; 16], &[0; 16])
                .unwrap();
            assert!(
                limited
                    .predicted_wide_row(kernel, &[1], &[0], &[0], &[0])
                    .is_err()
            );
        }
    }
}
