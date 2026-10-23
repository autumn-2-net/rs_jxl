//! Synthetic entropy fixtures with independently specified quantized blocks.
//! Byte roundtrips below exercise archival replay, NOT a coefficient JPEG writer.

use super::*;
use crate::jpeg::bitstream::EntropyError;
use crate::jpeg::data::NATURAL_ORDER;
use crate::jpeg::reader::read_jpeg;

const DC: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC: &[u8] = &[0, 1, 2, 0x11, 0x12, 0x21, 0xf0, 0x10, 0x20];
const GRAY: &[(u8, u8)] = &[(1, 0x11)];
const YUV420: &[(u8, u8)] = &[(1, 0x22), (2, 0x11), (3, 0x11)];

#[derive(Default)]
struct Bits(Vec<bool>);

impl Bits {
    fn word(&mut self, value: u32, count: u8) {
        for bit in (0..count).rev() {
            self.0.push(value & (1 << bit) != 0);
        }
    }

    fn symbol(&mut self, symbols: &[u8], value: u8) {
        // All fixture DHT codes have length four, starting at canonical 0000.
        self.word(symbols.iter().position(|&s| s == value).unwrap() as u32, 4);
    }

    fn magnitude(&mut self, value: i16, count: u8) {
        let bits = if value < 0 {
            i32::from(value) + (1 << count) - 1
        } else {
            i32::from(value)
        };
        self.word(bits as u32, count);
    }

    fn dc(&mut self, difference: i16, symbols: &[u8]) {
        let count = category(difference);
        self.symbol(symbols, count);
        self.magnitude(difference, count);
    }

    fn block(&mut self, difference: i16, nonzeros: &[(usize, i16)]) {
        self.dc(difference, DC);
        let mut next = 1;
        for &(zigzag, value) in nonzeros {
            let mut run = zigzag - next;
            while run >= 16 {
                self.symbol(AC, 0xf0);
                run -= 16;
            }
            let count = category(value);
            self.symbol(AC, ((run as u8) << 4) | count);
            self.magnitude(value, count);
            next = zigzag + 1;
        }
        if next <= 63 {
            self.symbol(AC, 0);
        }
    }

    fn finish(self, padding: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in self.0.chunks(8) {
            let mut value = 0u8;
            for i in 0..8 {
                let bit = chunk
                    .get(i)
                    .copied()
                    .unwrap_or(padding & (1 << (7 - i)) != 0);
                value = (value << 1) | u8::from(bit);
            }
            bytes.push(value);
            if value == 0xff {
                bytes.push(0);
            }
        }
        bytes
    }
}

fn category(value: i16) -> u8 {
    (u16::BITS - value.unsigned_abs().leading_zeros()) as u8
}

fn segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    bytes.extend_from_slice(&[0xff, marker]);
    bytes.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    bytes.extend_from_slice(payload);
}

fn dht(bytes: &mut Vec<u8>, selector: u8, symbols: &[u8]) {
    assert!(symbols.len() < 16);
    let mut payload = vec![selector];
    let mut counts = [0u8; 16];
    counts[3] = symbols.len() as u8;
    payload.extend(counts);
    payload.extend_from_slice(symbols);
    segment(bytes, 0xc4, &payload);
}

fn quant(bytes: &mut Vec<u8>, value: u8) {
    let mut payload = vec![0];
    payload.extend_from_slice(&[value; 64]);
    segment(bytes, 0xdb, &payload);
}

fn header(width: u16, height: u16, marker: u8, components: &[(u8, u8)]) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8];
    quant(&mut bytes, 1);
    dht(&mut bytes, 0, DC);
    dht(&mut bytes, 0x10, AC);
    let mut frame = vec![8];
    frame.extend(height.to_be_bytes());
    frame.extend(width.to_be_bytes());
    frame.push(components.len() as u8);
    for &(id, sampling) in components {
        frame.extend([id, sampling, 0]);
    }
    segment(&mut bytes, marker, &frame);
    bytes
}

fn scan(bytes: &mut Vec<u8>, ids: &[u8], params: [u8; 3], entropy: &[u8]) {
    let mut payload = vec![ids.len() as u8];
    for &id in ids {
        payload.extend([id, 0]);
    }
    payload.extend(params);
    segment(bytes, 0xda, &payload);
    bytes.extend_from_slice(entropy);
}

fn end(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes.extend([0xff, 0xd9]);
    bytes
}

fn read_exact(bytes: &[u8]) -> JpegData {
    let jpeg = read_coefficients(bytes).unwrap();
    assert_eq!(jpeg.reconstruct().unwrap(), bytes);
    jpeg
}

fn assert_limit(bytes: &[u8], limits: &CoefficientReadLimits, resource: &str) {
    let error = read_coefficients_with_limits(bytes, limits).unwrap_err();
    assert!(
        matches!(&error,
        CoefficientReadError::Jpeg(JpegError::LimitExceeded { resource: actual, .. })
        if *actual == resource),
        "{error}"
    );
}

#[test]
fn baseline_known_signed_coefficients_and_unchanged_archive_api() {
    for marker in [0xc0, 0xc1] {
        let mut bytes = header(17, 9, marker, GRAY);
        segment(&mut bytes, 0xfe, b"coefficient fixture");
        let mut bits = Bits::default();
        let mut expected = vec![0; 6 * BLOCK_SIZE];
        let mut previous = 0;
        for (index, dc) in [2, -1, 3, 0, -2, 5].into_iter().enumerate() {
            bits.block(dc - previous, &[(1, 1), (3, -2)]);
            previous = dc;
            expected[index * BLOCK_SIZE] = dc;
            expected[index * BLOCK_SIZE + NATURAL_ORDER[1]] = 1;
            expected[index * BLOCK_SIZE + NATURAL_ORDER[3]] = -2;
        }
        scan(&mut bytes, &[1], [0, 63, 0], &bits.finish(0xff));
        bytes.push(0xff); // Marker fill must not be counted as entropy padding.
        let mut bytes = end(bytes);
        bytes.extend(b"trailing bytes");
        let archive = read_jpeg(&bytes).unwrap();
        assert!(archive.components[0].coefficients.is_empty());
        assert!(archive.padding_bits.is_empty());
        assert_eq!(archive.reconstruct().unwrap(), bytes);
        let jpeg = read_exact(&bytes);
        assert_eq!(jpeg.components[0].coefficients, expected);
        assert_eq!(
            (
                jpeg.components[0].width_in_blocks,
                jpeg.components[0].height_in_blocks
            ),
            (3, 2)
        );
    }
}

#[test]
fn interleaved_420_reordered_components_padded_blocks_and_mcu_restarts() {
    let mut bytes = header(17, 17, 0xc0, YUV420);
    segment(&mut bytes, 0xdd, &[0, 1]);
    let mut expected = [
        vec![0; 16 * BLOCK_SIZE],
        vec![0; 4 * BLOCK_SIZE],
        vec![0; 4 * BLOCK_SIZE],
    ];
    let mut entropy = Vec::new();
    for my in 0..2 {
        for mx in 0..2 {
            let mut bits = Bits::default();
            let mut previous = [0; 3];
            for c in [2, 0, 1] {
                let n = if c == 0 { 2 } else { 1 };
                for iy in 0..n {
                    for ix in 0..n {
                        let index = (my * n + iy) * (2 * n) + mx * n + ix;
                        let value = (100 * c + index) as i16 - 8;
                        bits.block(value - previous[c], &[]);
                        previous[c] = value;
                        expected[c][index * BLOCK_SIZE] = value;
                    }
                }
            }
            entropy.extend(bits.finish(0xff));
            let mcu = my * 2 + mx;
            if mcu < 3 {
                entropy.extend([0xff, 0xff, 0xd0 + mcu as u8]);
            }
        }
    }
    scan(&mut bytes, &[3, 1, 2], [0, 63, 0], &entropy);
    let bytes = end(bytes);
    let jpeg = read_exact(&bytes);
    for (component, expected) in jpeg.components.iter().zip(expected) {
        assert_eq!(component.coefficients, expected);
    }
    assert_eq!(
        (jpeg.scan_info[0].mcu_cols, jpeg.scan_info[0].mcu_rows),
        (2, 2)
    );
    let mut limits = CoefficientReadLimits {
        max_coefficient_bytes: 24 * 128 - 1,
        ..Default::default()
    };
    assert_limit(&bytes, &limits, "coefficient bytes");
    limits.max_coefficient_bytes += 1;
    assert!(read_coefficients_with_limits(&bytes, &limits).is_ok());
}

#[test]
fn noninterleaved_scans_use_unpadded_extent_but_padded_row_stride() {
    let mut bytes = header(17, 17, 0xc0, YUV420);
    let mut expected = [
        vec![0; 16 * BLOCK_SIZE],
        vec![0; 4 * BLOCK_SIZE],
        vec![0; 4 * BLOCK_SIZE],
    ];
    let reversed_dc: Vec<_> = DC.iter().rev().copied().collect();
    for (pass, c) in [2, 0, 1].into_iter().enumerate() {
        if pass == 1 {
            dht(&mut bytes, 0, &reversed_dc);
        }
        let dc_table = if pass == 0 { DC } else { &reversed_dc };
        let mut bits = Bits::default();
        let mut previous = 0;
        let (extent, stride) = if c == 0 { (3, 4) } else { (2, 2) };
        for y in 0..extent {
            for x in 0..extent {
                let value = (30 * c + y * extent + x) as i16 + 1;
                bits.dc(value - previous, dc_table);
                bits.symbol(AC, 0);
                previous = value;
                expected[c][(y * stride + x) * BLOCK_SIZE] = value;
            }
        }
        scan(&mut bytes, &[c as u8 + 1], [0, 63, 0], &bits.finish(0xff));
    }
    let jpeg = read_exact(&end(bytes));
    for (component, expected) in jpeg.components.iter().zip(expected) {
        assert_eq!(component.coefficients, expected);
    }
    assert_eq!(jpeg.scan_info[0].dc_huffman_indices[0], Some(0));
    assert_eq!(jpeg.scan_info[1].dc_huffman_indices[0], Some(2));
    assert_eq!(
        (jpeg.scan_info[1].mcu_cols, jpeg.scan_info[1].mcu_rows),
        (3, 3)
    );
}

#[test]
fn progressive_signed_refinement_and_per_scan_table_redefinitions() {
    let mut bytes = header(24, 8, 0xc2, GRAY);
    let mut bits = Bits::default();
    for difference in [-2, 5, -2] {
        bits.dc(difference, DC);
    }
    scan(&mut bytes, &[1], [0, 0, 1], &bits.finish(0xff));
    let mut bits = Bits::default();
    bits.symbol(AC, 1);
    bits.word(1, 1); // +2 at k=1.
    bits.symbol(AC, 0x11);
    bits.word(0, 1); // -2 at k=3.
    for _ in 0..3 {
        bits.symbol(AC, 0);
    }
    scan(&mut bytes, &[1], [1, 63, 1], &bits.finish(0xff));
    quant(&mut bytes, 9);
    let mut bits = Bits::default();
    bits.word(0b101, 3);
    scan(&mut bytes, &[1], [0, 0, 0x10], &bits.finish(0xff));
    let reversed_ac: Vec<_> = AC.iter().rev().copied().collect();
    dht(&mut bytes, 0x10, &reversed_ac);
    let mut bits = Bits::default();
    bits.symbol(&reversed_ac, 1); // New coefficient at k=2.
    bits.word(1, 1); // New coefficient sign.
    bits.word(1, 1); // Refine existing k=1 to +3 while finding k=2.
    bits.symbol(&reversed_ac, 0);
    bits.word(1, 1); // Refine existing k=3 to -3.
    bits.symbol(&reversed_ac, 0x10);
    bits.word(0, 1); // Two zero blocks.
    scan(&mut bytes, &[1], [1, 63, 0x10], &bits.finish(0xff));
    let jpeg = read_exact(&end(bytes));
    let mut expected = vec![0; 3 * BLOCK_SIZE];
    expected[0] = -3;
    expected[64] = 6;
    expected[128] = 3;
    expected[NATURAL_ORDER[1]] = 3;
    expected[NATURAL_ORDER[2]] = 1;
    expected[NATURAL_ORDER[3]] = -3;
    assert_eq!(jpeg.components[0].coefficients, expected);
    assert_eq!(jpeg.scan_info[1].reset_points, [1, 2]);
    assert_eq!(jpeg.scan_info[3].reset_points, [1]);
    assert_eq!(jpeg.scan_info[1].ac_huffman_indices[0], Some(1));
    assert_eq!(jpeg.scan_info[3].ac_huffman_indices[0], Some(2));
    assert_eq!(jpeg.scan_info[1].quant_table_indices[0], Some(0));
    assert_eq!(jpeg.scan_info[2].quant_table_indices[0], Some(1));
    assert_eq!(jpeg.quant[1].values, [9; BLOCK_SIZE]);
}

#[test]
fn progressive_noninterleaved_refinement_does_not_touch_interleaved_padding_blocks() {
    let mut bytes = header(17, 9, 0xc2, YUV420);
    let mut bits = Bits::default();
    let mut previous = [0; 3];
    for _ in 0..2 {
        for (c, dc) in [2, 1, 3].into_iter().enumerate() {
            for _ in 0..if c == 0 { 4 } else { 1 } {
                bits.dc(dc - previous[c], DC);
                previous[c] = dc;
            }
        }
    }
    scan(&mut bytes, &[1, 2, 3], [0, 0, 1], &bits.finish(0xff));
    let mut bits = Bits::default();
    for _ in 0..6 {
        bits.symbol(AC, 1);
        bits.word(1, 1);
    }
    scan(&mut bytes, &[1], [1, 1, 0], &bits.finish(0xff));
    let mut bits = Bits::default();
    bits.word(0b111111, 6);
    scan(&mut bytes, &[1], [0, 0, 0x10], &bits.finish(0xff));
    let jpeg = read_exact(&end(bytes));
    let mut expected = vec![0; 8 * BLOCK_SIZE];
    for y in 0..2 {
        for x in 0..4 {
            let base = (y * 4 + x) * BLOCK_SIZE;
            expected[base] = if x < 3 { 5 } else { 4 };
            expected[base + NATURAL_ORDER[1]] = i16::from(x < 3);
        }
    }
    assert_eq!(jpeg.components[0].coefficients, expected);
    for (c, dc) in [(1, 2), (2, 6)] {
        let mut expected = vec![0; 2 * BLOCK_SIZE];
        expected[0] = dc;
        expected[64] = dc;
        assert_eq!(jpeg.components[c].coefficients, expected);
    }
}

fn hints_fixture() -> Vec<u8> {
    let mut bytes = header(32, 8, 0xc2, GRAY);
    let mut bits = Bits::default();
    for _ in 0..4 {
        bits.dc(0, DC);
    }
    scan(&mut bytes, &[1], [0, 0, 0], &bits.finish(0xff));
    let mut bits = Bits::default();
    bits.symbol(AC, 0x10);
    bits.word(0, 1); // Blocks 0 and 1.
    bits.symbol(AC, 0); // Independent EOB at block 2.
    bits.symbol(AC, 0xf0);
    bits.symbol(AC, 0); // Trailing ZRL at block 3.
    scan(&mut bytes, &[1], [1, 63, 0], &bits.finish(0xff));
    end(bytes)
}

#[test]
fn reconstruction_hints_use_scan_block_indices_and_have_combined_budget() {
    let bytes = hints_fixture();
    let jpeg = read_exact(&bytes);
    assert!(
        jpeg.components[0]
            .coefficients
            .iter()
            .all(|&value| value == 0)
    );
    assert_eq!(jpeg.scan_info[1].reset_points, [2]);
    assert_eq!(jpeg.scan_info[1].extra_zero_runs.len(), 1);
    let hint = &jpeg.scan_info[1].extra_zero_runs[0];
    assert_eq!((hint.block_index, hint.count), (3, 1));
    let mut limits = CoefficientReadLimits {
        max_hint_entries: 1,
        ..Default::default()
    };
    assert_limit(&bytes, &limits, "reconstruction hint entries");
    limits.max_hint_entries = 2;
    assert!(read_coefficients_with_limits(&bytes, &limits).is_ok());
}

#[test]
fn progressive_restarts_reset_eob_predictors_and_wrap_marker_sequence_per_scan() {
    let mut bytes = header(80, 8, 0xc2, GRAY);
    segment(&mut bytes, 0xdd, &[0, 1]);
    let mut entropy = Vec::new();
    for i in 0..10 {
        let mut bits = Bits::default();
        bits.dc(i + 1, DC); // Predictor must reset every MCU.
        entropy.extend(bits.finish(0xff));
        if i < 9 {
            entropy.extend([0xff, 0xd0 + (i as u8 & 7)]);
        }
    }
    scan(&mut bytes, &[1], [0, 0, 0], &entropy);
    // Redefining DRI affects only subsequent scans. Each RST sequence starts at 0.
    segment(&mut bytes, 0xdd, &[0, 2]);
    for params in [[1, 63, 1], [1, 63, 0x10]] {
        let mut entropy = Vec::new();
        for i in 0..5 {
            let mut bits = Bits::default();
            bits.symbol(AC, 0x10);
            bits.word(0, 1);
            entropy.extend(bits.finish(0xff));
            if i < 4 {
                entropy.extend([0xff, 0xff, 0xd0 + i]);
            }
        }
        scan(&mut bytes, &[1], params, &entropy);
    }
    let jpeg = read_exact(&end(bytes));
    let mut expected = vec![0; 10 * BLOCK_SIZE];
    for i in 0..10 {
        expected[i * BLOCK_SIZE] = i as i16 + 1;
    }
    assert_eq!(jpeg.components[0].coefficients, expected);
    assert!(
        jpeg.scan_info
            .iter()
            .all(|scan| scan.reset_points.is_empty())
    );
    assert_eq!(jpeg.scan_info[0].restart_interval, 1);
    assert_eq!(jpeg.scan_info[1].restart_interval, 2);
}

#[test]
fn exact_mixed_padding_at_restart_and_scan_end_and_strict_policy() {
    let mut bytes = header(16, 8, 0xc0, GRAY);
    segment(&mut bytes, 0xdd, &[0, 1]);
    let mut entropy = Vec::new();
    for (index, padding) in [0x2a, 0x55].into_iter().enumerate() {
        let mut bits = Bits::default();
        bits.block(1, &[]); // Nine bits, seven alignment bits.
        entropy.extend(bits.finish(padding));
        if index == 0 {
            entropy.extend([0xff, 0xff, 0xd0]);
        }
    }
    scan(&mut bytes, &[1], [0, 63, 0], &entropy);
    let bytes = end(bytes);
    let jpeg = read_exact(&bytes);
    assert!(jpeg.has_zero_padding_bit);
    assert_eq!(
        jpeg.padding_bits,
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    );
    let limits = CoefficientReadLimits {
        padding_policy: PaddingPolicy::RequireOnes,
        ..Default::default()
    };
    assert!(matches!(
        read_coefficients_with_limits(&bytes, &limits),
        Err(CoefficientReadError::Scan {
            error: CoefficientError::Entropy(EntropyError::InvalidPadding(_)),
            ..
        })
    ));
    let limits = CoefficientReadLimits {
        max_padding_bits: 13,
        ..Default::default()
    };
    assert_limit(&bytes, &limits, "padding bits");
}

#[test]
fn huge_dimensions_are_rejected_before_allocating_and_scan_work_is_aggregate() {
    let mut bytes = header(u16::MAX, u16::MAX - 8, 0xc0, GRAY);
    scan(&mut bytes, &[1], [0, 63, 0], &[]);
    assert_limit(
        &end(bytes),
        &CoefficientReadLimits::default(),
        "coefficient bytes",
    );
    let bytes = hints_fixture(); // Four DC and four AC block visits.
    let mut limits = CoefficientReadLimits {
        max_block_visits: 7,
        ..Default::default()
    };
    assert_limit(&bytes, &limits, "scan block visits");
    limits.max_block_visits = 8;
    assert!(read_coefficients_with_limits(&bytes, &limits).is_ok());
    // Missing entropy must not be reached when the work budget is already exceeded.
    let mut bytes = header(16, 8, 0xc0, GRAY);
    scan(&mut bytes, &[1], [0, 63, 0], &[]);
    limits.max_block_visits = 1;
    assert_limit(&end(bytes), &limits, "scan block visits");
}

fn restarted_baseline(interval: u8, groups: &[usize], markers: &[u8]) -> Vec<u8> {
    let mut bytes = header(24, 8, 0xc0, GRAY);
    segment(&mut bytes, 0xdd, &[0, interval]);
    let mut entropy = Vec::new();
    for (group, &blocks) in groups.iter().enumerate() {
        let mut bits = Bits::default();
        for _ in 0..blocks {
            bits.block(0, &[]);
        }
        entropy.extend(bits.finish(0xff));
        if let Some(&marker) = markers.get(group) {
            entropy.extend([0xff, marker]);
        }
    }
    scan(&mut bytes, &[1], [0, 63, 0], &entropy);
    end(bytes)
}

#[test]
fn rejects_missing_extra_early_late_and_out_of_order_restarts() {
    assert!(read_coefficients(&restarted_baseline(1, &[1, 1, 1], &[0xd0, 0xd1])).is_ok());
    for bytes in [
        restarted_baseline(1, &[3], &[]),
        restarted_baseline(0, &[1, 1, 1], &[0xd0, 0xd1]),
        restarted_baseline(2, &[1, 2], &[0xd0]),
        restarted_baseline(2, &[3, 0], &[0xd0]),
        restarted_baseline(1, &[1, 1, 1], &[0xd1, 0xd0]),
        restarted_baseline(1, &[1, 1, 1, 0], &[0xd0, 0xd1, 0xd2]),
    ] {
        assert!(read_jpeg(&bytes).is_ok());
        assert!(read_coefficients(&bytes).is_err());
    }
}

#[test]
fn rejects_eob_runs_past_scan_or_restart_boundary() {
    for restart in [false, true] {
        let mut bytes = header(if restart { 16 } else { 8 }, 8, 0xc2, GRAY);
        let mut bits = Bits::default();
        bits.dc(0, DC);
        if restart {
            bits.dc(0, DC);
        }
        scan(&mut bytes, &[1], [0, 0, 0], &bits.finish(0xff));
        let mut bits = Bits::default();
        bits.symbol(AC, 0x10);
        bits.word(0, 1);
        let mut entropy = bits.finish(0xff);
        if restart {
            segment(&mut bytes, 0xdd, &[0, 1]);
            entropy.extend([0xff, 0xd0, 0]);
        }
        scan(&mut bytes, &[1], [1, 63, 0], &entropy);
        let error = read_coefficients(&end(bytes)).unwrap_err();
        assert!(error.to_string().contains("EOB run"), "{error}");
    }
}

#[test]
fn malformed_entropy_is_distinct_from_structural_validity() {
    for entropy in [vec![], vec![0xff, 0], vec![0, 0]] {
        let mut bytes = header(8, 8, 0xc0, GRAY);
        scan(&mut bytes, &[1], [0, 63, 0], &entropy);
        let bytes = end(bytes);
        assert!(read_jpeg(&bytes).is_ok());
        let error = read_coefficients(&bytes).unwrap_err();
        assert!(matches!(
            error,
            CoefficientReadError::Scan { scan_index: 0, .. }
        ));
        assert!(error.to_string().contains("byte"));
    }
    let mut bytes = header(136, 8, 0xc0, GRAY);
    let mut bits = Bits::default();
    for _ in 0..17 {
        bits.block(2047, &[]);
    }
    scan(&mut bytes, &[1], [0, 63, 0], &bits.finish(0xff));
    assert!(
        read_coefficients(&end(bytes))
            .unwrap_err()
            .to_string()
            .contains("exceeds i16")
    );
}

#[test]
fn progression_and_missing_component_fail_before_entropy_allocation() {
    for params in [
        vec![[0, 0, 0], [0, 0, 0]],
        vec![[0, 0, 0x10]],
        vec![[1, 63, 0]],
        vec![[0, 0, 11]],
    ] {
        let mut bytes = header(8, 8, 0xc2, GRAY);
        for params in params {
            scan(&mut bytes, &[1], params, &[]);
        }
        let error = read_coefficients(&end(bytes)).unwrap_err();
        assert!(matches!(error, CoefficientReadError::Jpeg(_)), "{error}");
    }
    let mut bytes = header(8, 8, 0xc0, YUV420);
    scan(&mut bytes, &[1], [0, 63, 0], &[]);
    assert!(
        read_coefficients(&end(bytes))
            .unwrap_err()
            .to_string()
            .contains("no DC scan")
    );
}

#[test]
fn every_truncated_prefix_fails_without_panicking() {
    let bytes = hints_fixture();
    for length in 0..bytes.len() {
        assert!(
            read_coefficients(&bytes[..length]).is_err(),
            "prefix {length}"
        );
    }
    read_exact(&bytes);
}
