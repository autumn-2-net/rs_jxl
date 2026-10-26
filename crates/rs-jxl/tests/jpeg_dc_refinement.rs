//! Standard progressive DC refinement needs correction bits, not a DC DHT.
//! Fixtures define DC table 1 and AC table 0 only. Refinement retains selector
//! zero without synthesizing DC table 0. The runtime Rust decoder receives only
//! the encoded JXL, never the source JPEG or externally supplied coefficients.
use rs_jxl::decode::{DecodeOptions, reconstruct_jpeg};
use rs_jxl::jpeg::coefficient_reader::read_coefficients;
use rs_jxl::jpeg::data::JpegRecord;
use rs_jxl::jpeg::reader::read_jpeg;
use rs_jxl::jpeg::{TranscodeOptions, transcode};

fn segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    bytes.extend([0xff, marker]);
    bytes.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    bytes.extend_from_slice(payload);
}

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    byte: u8,
    used: u8,
}

impl Bits {
    fn word(&mut self, value: u32, count: u8) {
        for bit in (0..count).rev() {
            self.byte = (self.byte << 1) | ((value >> bit) & 1) as u8;
            self.used += 1;
            if self.used == 8 {
                self.bytes.push(self.byte);
                if self.byte == 0xff {
                    self.bytes.push(0);
                }
                self.byte = 0;
                self.used = 0;
            }
        }
    }

    fn align(&mut self) {
        if self.used != 0 {
            self.word(0xff, 8 - self.used);
        }
    }

    fn restart(&mut self) {
        self.align();
        self.bytes.extend([0xff, 0xd0]);
    }

    fn dc(&mut self, value: i16) {
        let category = (u16::BITS - value.unsigned_abs().leading_zeros()) as u8;
        assert!(category < 5);
        // Independently specified canonical codes: 000, 001, 010, 011, 100.
        self.word(u32::from(category), 3);
        if category != 0 {
            let magnitude = if value < 0 {
                i32::from(value) + (1 << category) - 1
            } else {
                i32::from(value)
            };
            self.word(magnitude as u32, category);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

fn expected_dc(component: usize, block: usize) -> i16 {
    [[-3, 4], [5, -6], [-7, 9]][component][block]
}

fn scan(bytes: &mut Vec<u8>, ids: &[u8], selector: u8, band: [u8; 3], entropy: Bits) {
    let mut payload = vec![ids.len() as u8];
    for &id in ids {
        payload.extend([id, selector]);
    }
    payload.extend(band);
    segment(bytes, 0xda, &payload);
    bytes.extend(entropy.finish());
}

fn fixture(components: usize, restart: bool) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8];
    segment(&mut bytes, 0xfe, b"DC refinement without a DC-0 definition");
    let mut quant = vec![0];
    quant.extend([1; 64]);
    segment(&mut bytes, 0xdb, &quant);
    let ids: Vec<u8> = (1..=components as u8).collect();
    let mut frame = vec![8, 0, 8, 0, 16, components as u8];
    for &id in &ids {
        frame.extend([id, 0x11, 0]);
    }
    segment(&mut bytes, 0xc2, &frame);
    let mut dc_table = vec![1];
    let mut counts = [0u8; 16];
    counts[2] = 5;
    dc_table.extend(counts);
    dc_table.extend([0, 1, 2, 3, 4]);
    segment(&mut bytes, 0xc4, &dc_table);
    let mut ac_table = vec![0x10, 1];
    ac_table.extend([0; 15]);
    ac_table.push(0); // One-bit code 0: EOB for these all-zero AC blocks.
    segment(&mut bytes, 0xc4, &ac_table);
    if restart {
        segment(&mut bytes, 0xdd, &[0, 1]);
    }

    let mut bits = Bits::default();
    let mut previous = [0i16; 3];
    for block in 0..2 {
        if restart && block != 0 {
            bits.restart();
            previous.fill(0);
        }
        for (component, predictor) in previous.iter_mut().enumerate().take(components) {
            let dc = expected_dc(component, block) >> 1;
            bits.dc(dc - *predictor);
            *predictor = dc;
        }
    }
    scan(&mut bytes, &ids, 0x10, [0, 0, 1], bits);

    for &id in &ids {
        let mut bits = Bits::default();
        bits.word(0, 1);
        if restart {
            bits.restart();
        }
        bits.word(0, 1);
        scan(&mut bytes, &[id], 0, [1, 63, 0], bits);
    }

    let mut bits = Bits::default();
    for block in 0..2 {
        if restart && block != 0 {
            bits.restart();
        }
        for component in 0..components {
            bits.word((i32::from(expected_dc(component, block)) & 1) as u32, 1);
        }
    }
    scan(&mut bytes, &ids, 0, [0, 0, 0x10], bits);
    bytes.extend([0xff, 0xd9]);
    bytes.extend(b"original tail");
    bytes
}

fn assert_fixture(bytes: &[u8], components: usize) {
    let jpeg = read_coefficients(bytes).unwrap();
    assert_eq!(jpeg.components.len(), components);
    assert_eq!(jpeg.huffman_code.len(), 2);
    assert_eq!(jpeg.huffman_code[0].slot_id, 1);
    assert_eq!(jpeg.huffman_code[1].slot_id, 0x10);
    let refinement = jpeg.scan_info.last().unwrap();
    assert_eq!(
        (refinement.ss, refinement.se, refinement.ah, refinement.al),
        (0, 0, 1, 0)
    );
    assert!(refinement.dc_huffman_indices[0].is_none());
    assert!(refinement.components.iter().all(|c| c.dc_table_index == 0));
    for (component, image) in jpeg.components.iter().enumerate() {
        assert_eq!(image.coefficients.len(), 128);
        for (block, coefficients) in image.coefficients.chunks_exact(64).enumerate() {
            assert_eq!(coefficients[0], expected_dc(component, block));
            assert!(coefficients[1..].iter().all(|&value| value == 0));
        }
    }
}

#[test]
fn dc_refinement_without_dc_table_zero_roundtrips_original_jpeg() {
    for components in [1, 3] {
        for restart in [false, true] {
            let original = fixture(components, restart);
            assert_fixture(&original, components);
            for threads in [1, 4] {
                let jxl = transcode(
                    &original,
                    &TranscodeOptions {
                        threads,
                        ..Default::default()
                    },
                )
                .unwrap_or_else(|e| {
                    panic!("{components} channels, restart={restart}, threads={threads}: {e}")
                });
                let restored = reconstruct_jpeg(
                    &jxl,
                    &DecodeOptions {
                        threads,
                        ..Default::default()
                    },
                )
                .unwrap_or_else(|e| {
                    panic!("{components} channels, restart={restart}, threads={threads}: {e}")
                });
                assert_eq!(
                    restored, original,
                    "original bytes, not only pixels or coefficients"
                );
                assert_fixture(&restored, components);
            }
        }
    }
}

fn scan_offsets(bytes: &[u8]) -> Vec<usize> {
    read_jpeg(bytes)
        .unwrap()
        .records
        .iter()
        .filter_map(|record| match record {
            JpegRecord::Marker(segment) if segment.marker == 0xda => Some(segment.offset),
            _ => None,
        })
        .collect()
}

#[test]
fn initial_dc_and_ac_scans_still_require_their_huffman_tables() {
    let original = fixture(1, false);
    let offsets = scan_offsets(&original);
    for (offset, selector) in [(offsets[0], 0), (offsets[1], 1)] {
        let mut invalid = original.clone();
        invalid[offset + 6] = selector;
        let error = transcode(&invalid, &TranscodeOptions::default()).unwrap_err();
        assert!(error.to_string().contains("undefined"), "{error}");
    }
}

#[test]
fn dc_refinement_cannot_precede_initial_dc_coefficients() {
    let mut invalid = fixture(1, false);
    let offset = scan_offsets(&invalid)[0];
    invalid[offset + 9] = 0x10; // Ah=1/Al=0 instead of the initial Ah=0/Al=1.
    let error = transcode(&invalid, &TranscodeOptions::default()).unwrap_err();
    assert!(error.to_string().contains("precedes"), "{error}");
}
