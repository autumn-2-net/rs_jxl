//! Separate source packed-i16 and valid-image oracles for the narrow kernel.

use super::*;

fn wrap16(value: i32) -> i32 {
    (value + 32768).rem_euclid(65536) - 32768
}

fn packed_oracle(pixel: i32, left: i32, top: i32, topleft: i32) -> u32 {
    let ac = wrap16(left - topleft);
    let ab = wrap16(left - top);
    let bc = wrap16(top - topleft);
    let clamp = if (ab < 0) != (bc < 0) { top } else { left };
    let prediction = if (ac < 0) != (bc < 0) {
        wrap16(ac + top)
    } else {
        clamp
    };
    zigzag(wrap16(pixel - prediction))
}

fn zigzag(value: i32) -> u32 {
    if value < 0 {
        (-2 * value - 1) as u32
    } else {
        (2 * value) as u32
    }
}

fn valid_oracle(pixel: i32, left: i32, top: i32, topleft: i32) -> u32 {
    zigzag(pixel - (left + top - topleft).clamp(left.min(top), left.max(top)))
}

#[test]
fn arbitrary_i16_lanes_match_packed_source_not_promoted_scalar_residuals() {
    for backend in [Backend::Scalar, Backend::Auto] {
        let kernel = Kernel::choose(backend);
        let mut random = 0x183f_c57bu32;
        // Every residual word, then arbitrary signed-lane predictor contexts.
        for random_neighbors in [false, true] {
            for base in (0..65536).step_by(16) {
                let mut inputs = [[0; 16]; 4];
                for (lane, value) in inputs[0].iter_mut().enumerate() {
                    *value = (base + lane) as i32 - 32768;
                }
                if random_neighbors {
                    for row in &mut inputs[1..] {
                        for value in row {
                            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            *value = i32::from(random as i16);
                        }
                    }
                }
                let [row, left, top, tl] = inputs;
                let expected: [u32; 16] =
                    std::array::from_fn(|i| packed_oracle(row[i], left[i], top[i], tl[i]));
                let mut actual = [u32::MAX; 16];
                let prefix = kernel.predict_chunk_narrow(&row, &left, &top, &tl, &mut actual);
                assert_eq!(actual, expected, "{backend:?}, base={base}");
                assert_eq!(prefix, expected.iter().take_while(|&&v| v == 0).count());
            }
        }
        // The SIMD source wraps residual subtraction BEFORE signed packing.
        // Its generic scalar source promotes subtraction to int instead.
        for n in [1, 7, 8, 15, 16] {
            let mut actual = [0; 16];
            kernel.predict_chunk_narrow(
                &[32767; 16],
                &[-32768; 16],
                &[-32768; 16],
                &[-32768; 16],
                &mut actual[..n],
            );
            assert_eq!(&actual[..n], vec![1; n]);
        }
    }
    assert_eq!(packed_oracle(32767, -32768, -32768, -32768), 1);
    assert_eq!(pixels::predict(32767, -32768, -32768, -32768, true), 65534);
}

#[test]
fn valid_narrow_lanes_match_clamp_and_previous_kernel_for_every_prefix_tail() {
    for depth in [1, 7, 8, 9, 13, 14] {
        let max = (1 << depth) - 1;
        for backend in [Backend::Scalar, Backend::Auto] {
            let kernel = Kernel::choose(backend);
            for n in 0usize..=16 {
                for prefix in 0..=n {
                    let mut inputs = [[0; 17]; 4];
                    for i in 0..n {
                        let left = (i as i32 * 7919) % (2 * max + 1) - max;
                        let top = max - (i as i32 * 3137) % (2 * max + 1);
                        let tl = (i as i32 * 4133) % (2 * max + 1) - max;
                        let pred = (left + top - tl).clamp(left.min(top), left.max(top));
                        let pixel = if i < prefix || (i > prefix && i.is_multiple_of(2)) {
                            pred
                        } else if pred >= 0 {
                            -max
                        } else {
                            max
                        };
                        for (row, value) in inputs.iter_mut().zip([pixel, left, top, tl]) {
                            row[i + 1] = value;
                        }
                    }
                    let [row, left, top, tl] = inputs;
                    let expected: Vec<_> = (1..=n)
                        .map(|i| valid_oracle(row[i], left[i], top[i], tl[i]))
                        .collect();
                    let mut previous = vec![0; n];
                    kernel.predict(
                        &row[1..1 + n],
                        &left[1..1 + n],
                        &top[1..1 + n],
                        &tl[1..1 + n],
                        true,
                        &mut previous,
                    );
                    assert_eq!(previous, expected);
                    let mut actual = [0xa5a5_a5a5; 18];
                    let actual_prefix = kernel.predict_chunk_narrow(
                        &row[1..1 + n],
                        &left[1..1 + n],
                        &top[1..1 + n],
                        &tl[1..1 + n],
                        &mut actual[1..1 + n],
                    );
                    assert_eq!(actual_prefix, prefix, "{backend:?}, depth={depth}, n={n}");
                    assert_eq!(&actual[1..1 + n], expected);
                    assert_eq!(actual[0], 0xa5a5_a5a5);
                    assert!(actual[1 + n..].iter().all(|&v| v == 0xa5a5_a5a5));
                }
            }
        }
    }
}

#[test]
fn narrow_chunk_rejects_invalid_lengths_before_mutation() {
    for backend in [Backend::Scalar, Backend::Auto] {
        let kernel = Kernel::choose(backend);
        for short in 0..4 {
            let full = [0; 16];
            let shorter = [0; 15];
            let mut inputs = [full.as_slice(); 4];
            inputs[short] = &shorter;
            let mut output = [0x1234; 16];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    kernel.predict_chunk_narrow(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        &mut output,
                    )
                }))
                .is_err()
            );
            assert_eq!(output, [0x1234; 16]);
        }
        assert!(
            std::panic::catch_unwind(|| {
                kernel.predict_chunk_narrow(&[0; 17], &[0; 17], &[0; 17], &[0; 17], &mut [0; 17])
            })
            .is_err()
        );
    }
}

#[derive(Debug, Default, PartialEq)]
struct Trace {
    chunks: Vec<(usize, Vec<u32>)>,
    final_runs: Vec<usize>,
}

impl Sink for Trace {
    fn chunk(&mut self, run: usize, residuals: &[u32]) -> Result<()> {
        self.chunks.push((run, residuals.to_vec()));
        Ok(())
    }
    fn finish(&mut self, run: usize) -> Result<()> {
        self.final_runs.push(run);
        Ok(())
    }
}

fn code(depth: u8) -> PrefixCode {
    let count = match depth {
        1..=8 => 11,
        9..=13 => 16,
        14 => 17,
        _ => unreachable!(),
    };
    let raw = std::array::from_fn(|i| {
        if i < count {
            1 + (i * 37 % 101) as u64
        } else {
            0
        }
    });
    let lz = std::array::from_fn(|i| if i < 28 { 1 + (i * 17 % 53) as u64 } else { 0 });
    PrefixCode::new(&raw, &lz, depth).unwrap()
}

fn same_bits(mut actual: BitWriter, mut expected: BitWriter) {
    assert_eq!(actual.bits_written(), expected.bits_written());
    actual.zero_pad_to_byte().unwrap();
    expected.zero_pad_to_byte().unwrap();
    assert_eq!(actual.bytes().unwrap(), expected.bytes().unwrap());
}

#[test]
fn narrow_rows_preserve_rle_histograms_fixed_chunk_bits_and_output_errors() {
    for backend in [Backend::Scalar, Backend::Auto] {
        let kernel = Kernel::choose(backend);
        for depth in [1, 8, 9, 13, 14] {
            let code = code(depth);
            let mut actual_bits = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(65536),
                kernel,
            });
            let mut expected_bits = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(65536),
                kernel: Kernel::choose(Backend::Scalar),
            });
            let mut actual_trace = Processor::new(Trace::default());
            let mut expected_trace = Processor::new(Trace::default());
            let mut actual_hist = Processor::new(Histogram::default());
            let mut expected_hist = Processor::new(Histogram::default());
            for zeros in [0, 1, 7, 8, 15, 16, 17, 31, 39, 40, 255, 256, 512] {
                let max = (1 << depth) - 1;
                let mut row = vec![0; zeros];
                row.extend([max, -max, 1, -1, 0, 1, 0]);
                row.extend([0; 33]);
                let neighbor = vec![0; row.len()];
                let residuals: Vec<_> = row.iter().map(|&p| valid_oracle(p, 0, 0, 0)).collect();
                actual_bits
                    .predicted_narrow_row(kernel, &row, &neighbor, &neighbor, &neighbor)
                    .unwrap();
                actual_trace
                    .predicted_narrow_row(kernel, &row, &neighbor, &neighbor, &neighbor)
                    .unwrap();
                actual_hist
                    .predicted_narrow_row(kernel, &row, &neighbor, &neighbor, &neighbor)
                    .unwrap();
                expected_bits.row(&residuals, kernel.chunk_size()).unwrap();
                expected_trace.row(&residuals, kernel.chunk_size()).unwrap();
                expected_hist.row(&residuals, kernel.chunk_size()).unwrap();
            }
            actual_bits.finish().unwrap();
            expected_bits.finish().unwrap();
            actual_trace.finish().unwrap();
            expected_trace.finish().unwrap();
            actual_hist.finish().unwrap();
            expected_hist.finish().unwrap();
            same_bits(actual_bits.sink.writer, expected_bits.sink.writer);
            assert_eq!(actual_trace.sink, expected_trace.sink);
            assert_eq!(actual_hist.sink.raw, expected_hist.sink.raw);
            assert_eq!(actual_hist.sink.lz, expected_hist.sink.lz);
            let mut sample = Processor::new(Histogram::default());
            sample
                .predicted_narrow_row(kernel, &[0; 32], &[0; 32], &[0; 32], &[0; 32])
                .unwrap();
            sample.finish().unwrap();
            assert_eq!(sample.sink.raw, [0; RAW]);
            assert_eq!(sample.sink.lz, [0; LZ]);
            let mut limited = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(0),
                kernel,
            });
            limited
                .predicted_narrow_row(kernel, &[0; 16], &[0; 16], &[0; 16], &[0; 16])
                .unwrap();
            assert!(
                limited
                    .predicted_narrow_row(kernel, &[1], &[0], &[0], &[0])
                    .is_err()
            );
        }
    }
}

#[test]
fn narrow_image_sampling_and_encoding_match_independent_context_rows() {
    for depth in [1, 7, 8, 9, 13, 14] {
        for channels in 1u8..=4 {
            for big_endian in [false, true] {
                let (width, height) = (35usize, 6usize);
                let storage = if depth <= 8 { 1 } else { 2 };
                let stride = width * usize::from(channels) * storage + 3;
                let mut bytes = vec![0x79; stride * height - 3];
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..usize::from(channels) {
                            let value = if c == 3 && x.is_multiple_of(3) {
                                0
                            } else {
                                ((x * 8191 + y * 971 + c * 211 + x * y) & ((1 << depth) - 1)) as u16
                            };
                            let at = y * stride + (x * usize::from(channels) + c) * storage;
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
                        let expected = super::wide_prediction::oracle_area(image, area);
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
                                rows.into_iter().flatten().collect::<Vec<_>>()
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

fn palette_of_size(size: usize) -> Palette {
    let mut table = vec![0; 65536];
    let mut remaining = size - 1;
    for i in 0u32.. {
        if remaining == 0 {
            break;
        }
        let color = (i << 8) | 1;
        let hash = (color.wrapping_mul(2_654_435_761) >> 16) as usize;
        if hash != 0 && table[hash] == 0 {
            table[hash] = color;
            remaining -= 1;
        }
    }
    let palette = Palette::from_table(table, 4).unwrap();
    assert_eq!(palette.colors.len(), size);
    palette
}

#[test]
fn palette_dictionary_and_indices_use_narrow_rows_with_source_contexts() {
    let code = code(8);
    for width in [2, 7, 8, 15, 16, 17, 255, 256, 257, 512] {
        let palette = palette_of_size(width);
        for backend in [Backend::Scalar, Backend::Auto] {
            let kernel = Kernel::choose(backend);
            let actual =
                pixels::palette_data(&palette, 4, &code, kernel, BitWriter::new(65536)).unwrap();
            let mut expected = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(65536),
                kernel: Kernel::choose(Backend::Scalar),
            });
            let mut previous = vec![0; width];
            for c in 0..4 {
                let current: Vec<_> = palette
                    .colors
                    .iter()
                    .map(|&p| ((p >> (8 * c)) & 255) as i32)
                    .collect();
                let residuals: Vec<_> = (0..width)
                    .map(|x| {
                        let left = if x > 0 { current[x - 1] } else { previous[0] };
                        let top = if c == 0 { left } else { previous[x] };
                        let tl = if c == 0 {
                            left
                        } else if x > 0 {
                            previous[x - 1]
                        } else {
                            previous[0]
                        };
                        valid_oracle(current[x], left, top, tl)
                    })
                    .collect();
                expected.row(&residuals, kernel.chunk_size()).unwrap();
                previous = current;
            }
            expected.finish().unwrap();
            same_bits(actual, expected.sink.writer);

            let indices: Vec<_> = (0..33 * 3).map(|i| (i * 7 + i / 33) % width).collect();
            let bytes: Vec<_> = indices
                .iter()
                .flat_map(|&i| palette.colors[i].to_le_bytes())
                .collect();
            let image = view(&bytes, 33, 3, 4, 8);
            let area = Area {
                x: 0,
                y: 0,
                width: 33,
                height: 3,
                skip: 1,
            };
            let mut record = [Processor::new(Record::default())];
            let mut histogram = [Processor::new(Histogram::default())];
            pixels::process(image, area, Some(&palette), kernel, &mut record).unwrap();
            pixels::process(image, area, Some(&palette), kernel, &mut histogram).unwrap();
            let mut expected_residuals = Vec::new();
            let mut expected_hist = Processor::new(Histogram::default());
            for (y, row) in indices.chunks(33).enumerate() {
                let residuals: Vec<_> = row
                    .iter()
                    .enumerate()
                    .map(|(x, &pixel)| {
                        let left = if x > 0 {
                            row[x - 1]
                        } else if y > 0 {
                            indices[(y - 1) * 33]
                        } else {
                            0
                        };
                        let top = if y == 0 {
                            left
                        } else {
                            indices[(y - 1) * 33 + x]
                        };
                        let tl = if y == 0 {
                            left
                        } else if x > 0 {
                            indices[(y - 1) * 33 + x - 1]
                        } else {
                            top
                        };
                        valid_oracle(pixel as i32, left as i32, top as i32, tl as i32)
                    })
                    .collect();
                expected_hist.row(&residuals, kernel.chunk_size()).unwrap();
                expected_residuals.extend(residuals);
            }
            expected_hist.finish().unwrap();
            // Source palette sampling deliberately includes the warm-up row.
            assert_eq!(record[0].sink.decoded, expected_residuals);
            assert_eq!(histogram[0].sink.raw, expected_hist.sink.raw);
            assert_eq!(histogram[0].sink.lz, expected_hist.sink.lz);
        }
    }
}
