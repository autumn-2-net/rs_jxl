use super::*;
use crate::jpeg::coefficient_reader::read_coefficients;

fn segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    bytes.extend([0xff, marker]);
    bytes.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    bytes.extend_from_slice(payload);
}

fn header(process: u8, width: u16) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8];
    segment(&mut bytes, 0xe1, &[1, 2, 255, 0, 4]);
    let mut quant = vec![0];
    quant.extend([1; 64]);
    segment(&mut bytes, 0xdb, &quant);
    let [hi, lo] = width.to_be_bytes();
    segment(&mut bytes, process, &[8, 0, 8, hi, lo, 1, 1, 0x11, 0]);
    let mut dht = vec![0, 1];
    dht.extend([0; 15]);
    dht.push(0);
    dht.extend([0x10, 1]);
    dht.extend([0; 15]);
    dht.push(0);
    segment(&mut bytes, 0xc4, &dht);
    bytes
}

fn sos(bytes: &mut Vec<u8>, start: u8, end: u8, approximation: u8, entropy: &[u8]) {
    segment(bytes, 0xda, &[1, 1, 0, start, end, approximation]);
    bytes.extend_from_slice(entropy);
}

fn baseline(process: u8, restart: bool) -> Vec<u8> {
    let mut bytes = header(process, if restart { 16 } else { 8 });
    if restart {
        segment(&mut bytes, 0xdd, &[0, 1]);
        // Zero DC and EOB are each one zero bit; preserve mixed alignment and
        // an extra marker-fill FF. The next MCU resets DC and uses one padding.
        sos(&mut bytes, 0, 63, 0, &[0x2a, 255, 255, 0xd0, 0x3f]);
    } else {
        sos(&mut bytes, 0, 63, 0, &[0x2a]);
    }
    bytes.extend([255, 255, 0xd9, 7, 0, 255]);
    bytes
}

fn erase_entropy(jpeg: &mut JpegData) {
    for record in &mut jpeg.records {
        if let JpegRecord::Entropy(entropy) = record {
            entropy.bytes.clear();
        }
    }
}

#[test]
fn baseline_and_extended_rebuild_from_coefficients_without_original_entropy() {
    for process in [0xc0, 0xc1] {
        for restart in [false, true] {
            let original = baseline(process, restart);
            let mut jpeg = read_coefficients(&original).unwrap();
            erase_entropy(&mut jpeg);
            assert_ne!(jpeg.reconstruct().unwrap(), original);
            assert_eq!(
                reconstruct_from_coefficients(&jpeg, &WriteLimits::default()).unwrap(),
                original
            );
        }
    }
}

#[test]
fn progressive_independent_eob_boundaries_and_empty_padding_are_preserved() {
    let mut original = header(0xc2, 16);
    sos(&mut original, 0, 0, 0, &[0x3f]);
    // Two separate EOBs rather than one EOB run. The reconstruction hints must
    // preserve that original choice, not merely recover equivalent coefficients.
    sos(&mut original, 1, 63, 0, &[0x3f]);
    original.extend([255, 0xd9]);
    let mut jpeg = read_coefficients(&original).unwrap();
    erase_entropy(&mut jpeg);
    assert_eq!(
        reconstruct_from_coefficients(&jpeg, &WriteLimits::default()).unwrap(),
        original
    );

    let mut original = header(0xc0, 32);
    sos(&mut original, 0, 63, 0, &[0]); // Four two-bit blocks, no padding.
    original.extend([255, 0xd9]);
    let mut jpeg = read_coefficients(&original).unwrap();
    assert!(jpeg.padding_bits.is_empty());
    erase_entropy(&mut jpeg);
    assert_eq!(
        reconstruct_from_coefficients(&jpeg, &WriteLimits::default()).unwrap(),
        original
    );
}

#[test]
fn edited_headers_tables_coefficients_and_hints_are_validated_before_writing() {
    let jpeg = read_coefficients(&baseline(0xc0, true)).unwrap();
    let bad: [fn(&mut JpegData); 14] = [
        |j| j.width += 1,
        |j| j.components[0].width_in_blocks += 1,
        |j| {
            j.components[0].coefficients.pop();
        },
        |j| j.huffman_code[0].values[0] = 999,
        |j| j.scan_info[0].components[0].component_index = usize::MAX,
        |j| j.scan_info[0].dc_huffman_indices[0] = Some(usize::MAX),
        |j| j.scan_info[0].mcu_cols = u32::MAX,
        |j| j.scan_info[0].ss = u32::MAX,
        |j| j.scan_info[0].reset_points = vec![0, 0],
        |j| j.scan_info[0].reset_points = vec![2],
        |j| j.padding_bits[0] = 2,
        |j| {
            j.padding_bits.pop();
        },
        |j| j.padding_bits.push(1),
        |j| {
            for record in &mut j.records {
                if let JpegRecord::Entropy(entropy) = record {
                    entropy.restart_markers[0].fill_count = usize::MAX;
                }
            }
        },
    ];
    for (index, edit) in bad.into_iter().enumerate() {
        let mut invalid = jpeg.clone();
        edit(&mut invalid);
        assert!(
            reconstruct_from_coefficients(&invalid, &WriteLimits::default()).is_err(),
            "case {index}"
        );
    }
}

#[test]
fn output_and_work_limits_are_explicit_and_exact_bound_succeeds() {
    let original = baseline(0xc0, true);
    let jpeg = read_coefficients(&original).unwrap();
    let limits = WriteLimits {
        max_output_bytes: original.len(),
        ..Default::default()
    };
    assert_eq!(
        reconstruct_from_coefficients(&jpeg, &limits).unwrap(),
        original
    );
    for limits in [
        WriteLimits {
            max_output_bytes: original.len() - 1,
            ..Default::default()
        },
        WriteLimits {
            max_block_visits: 1,
            ..Default::default()
        },
        WriteLimits {
            max_padding_bits: 0,
            ..Default::default()
        },
        WriteLimits {
            structural: ReadLimits {
                max_records: 0,
                ..Default::default()
            },
            ..Default::default()
        },
        WriteLimits {
            structural: ReadLimits {
                max_restart_markers: 0,
                ..Default::default()
            },
            ..Default::default()
        },
    ] {
        assert!(reconstruct_from_coefficients(&jpeg, &limits).is_err());
    }
}
