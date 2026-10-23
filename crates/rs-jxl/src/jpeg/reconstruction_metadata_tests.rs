//! Small independently packed metadata fixtures and bounded rejection cases.

use super::*;
use crate::jpeg::coefficient_reader::read_coefficients;
use crate::jpeg::data::AppMarkerType;

fn segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    bytes.extend([0xff, marker]);
    bytes.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    bytes.extend_from_slice(payload);
}

fn quant(bytes: &mut Vec<u8>, tables: &[(u8, u8)]) {
    let mut payload = Vec::new();
    for &(selector, value) in tables {
        payload.push(selector);
        payload.extend([value; 64]);
    }
    segment(bytes, 0xdb, &payload);
}

fn huffman(bytes: &mut Vec<u8>, selector: u8, depth: usize, symbols: &[u8]) {
    let mut payload = vec![selector];
    let mut counts = [0; 16];
    counts[depth - 1] = symbols.len() as u8;
    payload.extend(counts);
    payload.extend_from_slice(symbols);
    segment(bytes, 0xc4, &payload);
}

fn header(process: u8, width: u16, ids: &[u8], tables: &[(u8, u8)]) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8];
    quant(&mut bytes, tables);
    let [hi, lo] = width.to_be_bytes();
    let mut sof = vec![8, 0, 8, hi, lo, ids.len() as u8];
    for &id in ids {
        sof.extend([id, 0x11, tables[0].0]);
    }
    segment(&mut bytes, process, &sof);
    let mut dht = Vec::new();
    for selector in [0, 0x10] {
        dht.extend([selector, 1]);
        dht.extend([0; 15]);
        dht.push(0);
    }
    segment(&mut bytes, 0xc4, &dht);
    bytes
}

fn scan(bytes: &mut Vec<u8>, ids: &[u8], ss: u8, se: u8, approximation: u8, entropy: &[u8]) {
    let mut sos = vec![ids.len() as u8];
    for &id in ids {
        sos.extend([id, 0]);
    }
    sos.extend([ss, se, approximation]);
    segment(bytes, 0xda, &sos);
    bytes.extend_from_slice(entropy);
}

fn finish(mut bytes: Vec<u8>) -> JpegData {
    bytes.extend([0xff, 0xd9]);
    read_coefficients(&bytes).unwrap()
}

fn baseline(ids: &[u8], tables: &[(u8, u8)]) -> JpegData {
    let mut bytes = header(0xc0, 8, ids, tables);
    scan(
        &mut bytes,
        ids,
        0,
        63,
        0,
        &[if ids.len() == 1 { 0x3f } else { 3 }],
    );
    finish(bytes)
}

fn encode(jpeg: &JpegData) -> EncodedMetadata {
    serialize_reconstruction_metadata(jpeg, 1 << 20).unwrap()
}

fn encode_with(jpeg: &JpegData, sources: ReconstructionMetadataSources<'_>) -> EncodedMetadata {
    serialize_reconstruction_metadata_with_sources(jpeg, 1 << 20, sources).unwrap()
}

fn with_apps(apps: &[(u8, Vec<u8>)]) -> JpegData {
    let mut bytes = header(0xc0, 8, &[1], &[(0, 1)]);
    for (marker, payload) in apps {
        segment(&mut bytes, *marker, payload);
    }
    scan(&mut bytes, &[1], 0, 63, 0, &[0x3f]);
    finish(bytes)
}

fn icc_chunk(ordinal: u8, count: u8, bytes: &[u8]) -> Vec<u8> {
    let mut payload = b"ICC_PROFILE\0".to_vec();
    payload.extend([ordinal, count]);
    payload.extend_from_slice(bytes);
    payload
}

fn exif_app(tiff: &[u8]) -> Vec<u8> {
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(tiff);
    payload
}

fn exif_box(tiff: &[u8]) -> Vec<u8> {
    let mut payload = vec![0; 4];
    payload.extend_from_slice(tiff);
    payload
}

fn xmp_app(xml: &[u8]) -> Vec<u8> {
    let mut payload = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
    payload.extend_from_slice(xml);
    payload
}

fn gray_profile(length: usize) -> Vec<u8> {
    let mut p = vec![0; length];
    p[..4].copy_from_slice(&u32::try_from(length).unwrap().to_be_bytes());
    p[16..20].copy_from_slice(b"GRAY");
    p[20..24].copy_from_slice(b"XYZ ");
    p[36..40].copy_from_slice(b"acsp");
    p
}

// Independent decoder-side reconstruction: lengths/types come from parsed wire
// fields; signatures, ICC ordinal/count and payload insertion match dec_jpeg_data
// and SetJPEGDataFromICC/SetExif/SetXmp, not the encoder's classifier.
fn assert_metadata_rehydrates(
    jpeg: &JpegData,
    encoded: &EncodedMetadata,
    sources: ReconstructionMetadataSources<'_>,
) {
    let p = parse(&encoded.header);
    let icc_count = p.app_types.iter().filter(|&&kind| kind == 1).count();
    let mut icc_ordinal = 0;
    let mut icc_position = 0;
    let mut payload_position = 0;
    let mut apps = Vec::new();
    for (&kind, &length) in p.app_types.iter().zip(&p.app_lengths) {
        let mut app = vec![0; length as usize];
        app[1..3].copy_from_slice(&u16::try_from(length - 1).unwrap().to_be_bytes());
        match kind {
            0 => {
                app.copy_from_slice(
                    &encoded.payload[payload_position..payload_position + length as usize],
                );
                payload_position += length as usize;
            }
            1 => {
                app[0] = 0xe2;
                app[3..15].copy_from_slice(b"ICC_PROFILE\0");
                icc_ordinal += 1;
                app[15] = icc_ordinal;
                app[16] = icc_count as u8;
                let chunk_length = app.len() - 17;
                app[17..].copy_from_slice(
                    &sources.icc_profile.unwrap()[icc_position..icc_position + chunk_length],
                );
                icc_position += chunk_length;
            }
            2 => {
                app[0] = 0xe1;
                app[3..9].copy_from_slice(b"Exif\0\0");
                app[9..].copy_from_slice(&sources.exif_box_payload.unwrap()[4..]);
            }
            3 => {
                app[0] = 0xe1;
                app[3..32].copy_from_slice(b"http://ns.adobe.com/xap/1.0/\0");
                app[32..].copy_from_slice(sources.xmp_box_payload.unwrap());
            }
            _ => panic!("unsupported APP type"),
        }
        apps.push(app);
    }
    if icc_count != 0 {
        assert_eq!(icc_position, sources.icc_profile.unwrap().len());
    }
    assert_eq!(apps, jpeg.app_data);
    let trailing_payload: Vec<u8> = jpeg
        .com_data
        .iter()
        .flatten()
        .copied()
        .chain(jpeg.records.iter().flat_map(|record| match record {
            JpegRecord::Fill { count, .. } => vec![0xff; *count],
            JpegRecord::InterMarker { index, .. } => jpeg.inter_marker_data[*index].clone(),
            _ => Vec::new(),
        }))
        .chain(jpeg.tail_data.iter().copied())
        .collect();
    assert_eq!(&encoded.payload[payload_position..], trailing_payload);
    let mut restored = jpeg.clone();
    let mut apps = apps.into_iter();
    for record in &mut restored.records {
        if let JpegRecord::Marker(segment) = record
            && (0xe0..=0xef).contains(&segment.marker)
        {
            let app = apps.next().unwrap();
            assert_eq!(app[0], segment.marker);
            segment.data = app[1..].to_vec();
        }
    }
    assert!(apps.next().is_none());
    assert_eq!(restored.reconstruct().unwrap(), jpeg.reconstruct().unwrap());
}

fn packed(fields: &[(u32, u8)]) -> Vec<u8> {
    let mut bits = Vec::new();
    for &(value, count) in fields {
        for bit in 0..count {
            bits.push(((value >> bit) & 1) as u8);
        }
    }
    while !bits.len().is_multiple_of(8) {
        bits.push(0);
    }
    bits.chunks_exact(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0, |byte, (bit, &value)| byte | (value << bit))
        })
        .collect()
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl Reader<'_> {
    fn bits(&mut self, count: usize) -> u32 {
        let mut value = 0;
        for bit in 0..count {
            let index = self.position;
            self.position += 1;
            value |= u32::from((self.bytes[index / 8] >> (index % 8)) & 1) << bit;
        }
        value
    }
    fn u32(&mut self, choices: [(usize, u32); 4]) -> u32 {
        let (bits, offset) = choices[self.bits(2) as usize];
        self.bits(bits) + offset
    }
}

#[derive(Debug)]
struct Parsed {
    markers: Vec<u8>,
    app_types: Vec<u32>,
    app_lengths: Vec<u32>,
    com_lengths: Vec<u32>,
    component_type: u32,
    ids: Vec<u32>,
    quant_indices: Vec<u32>,
    huffman: Vec<(u32, bool, [u32; 17], Vec<u32>)>,
    scans: Vec<Vec<u32>>,
    restart: Option<u32>,
    resets: Vec<Vec<u32>>,
    zeros: Vec<Vec<(u32, u32)>>,
    inter_lengths: Vec<u32>,
    tail_length: u32,
    padding: Vec<u32>,
}

// Independent field reader: explicit wire distributions, no serializer helpers.
fn parse(header: &[u8]) -> Parsed {
    let mut r = Reader {
        bytes: header,
        position: 0,
    };
    let grayscale = r.bits(1) != 0;
    let mut p = Parsed {
        markers: Vec::new(),
        app_types: Vec::new(),
        app_lengths: Vec::new(),
        com_lengths: Vec::new(),
        component_type: 0,
        ids: Vec::new(),
        quant_indices: Vec::new(),
        huffman: Vec::new(),
        scans: Vec::new(),
        restart: None,
        resets: Vec::new(),
        zeros: Vec::new(),
        inter_lengths: Vec::new(),
        tail_length: 0,
        padding: Vec::new(),
    };
    loop {
        let marker = r.bits(6) as u8 + 0xc0;
        p.markers.push(marker);
        if marker == 0xd9 {
            break;
        }
    }
    for _ in p.markers.iter().filter(|&&m| (0xe0..=0xef).contains(&m)) {
        p.app_types.push(r.u32([(0, 0), (0, 1), (1, 2), (2, 4)]));
        p.app_lengths.push(r.bits(16) + 1);
    }
    for _ in p.markers.iter().filter(|&&m| m == 0xfe) {
        p.com_lengths.push(r.bits(16) + 1);
    }
    let quant = r.u32([(0, 1), (0, 2), (0, 3), (0, 4)]);
    for _ in 0..quant {
        r.bits(4);
    }
    p.component_type = r.bits(2);
    let components = if p.component_type == 3 {
        r.u32([(0, 1), (0, 2), (0, 3), (0, 4)])
    } else if p.component_type == 0 {
        1
    } else {
        3
    };
    assert_eq!(grayscale, components == 1);
    if p.component_type == 3 {
        for _ in 0..components {
            p.ids.push(r.bits(8));
        }
    }
    for _ in 0..components {
        p.quant_indices.push(r.bits(2));
    }
    let huffman = r.u32([(0, 4), (3, 2), (4, 10), (6, 26)]);
    for _ in 0..huffman {
        let class = r.bits(1);
        let slot = (class << 4) | r.bits(2);
        let last = r.bits(1) != 0;
        let counts = std::array::from_fn(|_| r.u32([(0, 0), (0, 1), (3, 2), (8, 0)]));
        let values = (0..counts.iter().sum::<u32>())
            .map(|_| r.u32([(2, 0), (2, 4), (4, 8), (8, 1)]))
            .collect();
        p.huffman.push((slot, last, counts, values));
    }
    for _ in p.markers.iter().filter(|&&m| m == 0xda) {
        let count = r.u32([(0, 1), (0, 2), (0, 3), (0, 4)]);
        let mut scan = vec![count, r.bits(6), r.bits(6), r.bits(4), r.bits(4)];
        for _ in 0..count {
            scan.extend([r.bits(2), r.bits(2), r.bits(2)]);
        }
        scan.push(r.u32([(0, 0), (0, 1), (0, 2), (3, 3)]));
        p.scans.push(scan);
    }
    if p.markers.contains(&0xdd) {
        p.restart = Some(r.bits(16));
    }
    for _ in &p.scans {
        let count = r.u32([(0, 0), (2, 1), (4, 4), (16, 20)]);
        let mut base = 0;
        let mut resets = Vec::new();
        for _ in 0..count {
            let index = base + r.u32([(0, 0), (3, 1), (5, 9), (28, 41)]);
            resets.push(index);
            base = index + 1;
        }
        p.resets.push(resets);
        let count = r.u32([(0, 0), (2, 1), (4, 4), (16, 20)]);
        let mut base = 0;
        let mut zeros = Vec::new();
        for _ in 0..count {
            let runs = r.u32([(0, 1), (2, 2), (4, 5), (8, 20)]);
            let index = base + r.u32([(0, 0), (3, 1), (5, 9), (28, 41)]);
            zeros.push((index, runs));
            base = index + 1;
        }
        p.zeros.push(zeros);
    }
    for _ in p.markers.iter().filter(|&&m| m == 0xff) {
        p.inter_lengths.push(r.bits(16));
    }
    p.tail_length = r.u32([(0, 0), (8, 1), (16, 257), (22, 65_793)]);
    if r.bits(1) != 0 {
        let count = r.bits(24);
        for _ in 0..count {
            p.padding.push(r.bits(1));
        }
    }
    assert!(header.len() * 8 - r.position < 8);
    while r.position < header.len() * 8 {
        assert_eq!(r.bits(1), 0);
    }
    p
}

#[test]
fn minimal_header_matches_independently_packed_source_fields() {
    let jpeg = baseline(&[1], &[(3, 1)]);
    let encoded = encode(&jpeg);
    let mut fields = vec![
        (1, 1),
        (27, 6),
        (0, 6),
        (4, 6),
        (26, 6),
        (25, 6),
        (0, 2),
        (0, 1),
        (3, 2),
        (1, 1),
        (0, 2),
        (0, 2),
        (1, 2),
        (0, 3),
    ];
    for (class, last) in [(0, 0), (1, 1)] {
        fields.extend([(class, 1), (0, 2), (last, 1), (0, 2), (2, 2), (0, 3)]);
        fields.extend(std::iter::repeat_n((0, 2), 15));
        fields.extend([(0, 2), (0, 2), (3, 2), (255, 8)]);
    }
    fields.extend([
        (0, 2),
        (0, 6),
        (63, 6),
        (0, 4),
        (0, 4),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 1),
    ]);
    assert_eq!(encoded.header, packed(&fields));
    assert!(encoded.payload.is_empty());
    let parsed = parse(&encoded.header);
    assert_eq!(parsed.markers, [0xdb, 0xc0, 0xc4, 0xda, 0xd9]);
    assert_eq!(parsed.quant_indices, [0]); // Wire DQT selector 3 is vector index 0.
    assert_eq!(parsed.huffman[0].2[1], 2);
    assert_eq!(parsed.huffman[0].3, [0, 256]);
}

#[test]
fn u32_shortest_selectors_and_range_boundaries_match_source() {
    for (choices, cases) in [
        (
            HUFFMAN_COUNT,
            vec![(2, 1), (4, 0), (9, 1), (10, 2), (25, 2), (26, 3), (89, 3)],
        ),
        (
            CODE_COUNT,
            vec![(0, 0), (1, 1), (2, 2), (9, 2), (10, 3), (255, 3)],
        ),
        (
            CODE_VALUE,
            vec![
                (0, 0),
                (3, 0),
                (4, 1),
                (7, 1),
                (8, 2),
                (23, 2),
                (24, 3),
                (256, 3),
            ],
        ),
        (
            HINT_COUNT,
            vec![
                (0, 0),
                (1, 1),
                (4, 1),
                (5, 2),
                (19, 2),
                (20, 3),
                (65_555, 3),
            ],
        ),
        (
            BLOCK_DELTA,
            vec![(0, 0), (1, 1), (8, 1), (9, 2), (40, 2), (41, 3)],
        ),
        (
            TAIL_LENGTH,
            vec![
                (0, 0),
                (1, 1),
                (256, 1),
                (257, 2),
                (65_792, 2),
                (65_793, 3),
                (4_260_096, 3),
            ],
        ),
    ] {
        for (value, expected) in cases {
            assert_eq!(choose(choices, value).unwrap().0, expected);
            let mut writer = BitWriter::new(8);
            u32_field(&mut writer, choices, value).unwrap();
            writer.zero_pad_to_byte().unwrap();
            let bytes = writer.into_bytes().unwrap();
            let mut reader = Reader {
                bytes: &bytes,
                position: 0,
            };
            assert_eq!(
                reader.u32(choices.map(|c| (usize::from(c.bits), c.offset))),
                value
            );
        }
    }
    for (choices, value) in [
        (HUFFMAN_COUNT, 1),
        (HUFFMAN_COUNT, 90),
        (CODE_COUNT, 256),
        (CODE_VALUE, 257),
        (HINT_COUNT, 65_556),
        (TAIL_LENGTH, 4_260_097),
        (LAST_PASS, 11),
    ] {
        assert!(choose(choices, value).is_err());
    }
}

#[test]
fn unknown_app_com_intermarker_and_tail_payload_order_is_exact() {
    let mut bytes = vec![0xff, 0xd8];
    segment(&mut bytes, 0xfe, &[9]);
    segment(&mut bytes, 0xe2, b"ICC_PROFILE\0\x01\x01x");
    bytes.extend([3, 4, 0xff, 0xff]);
    segment(&mut bytes, 0xe1, b"Exif\0\0x");
    bytes.extend_from_slice(&header(0xc0, 8, &[1], &[(0, 1)])[2..]);
    scan(&mut bytes, &[1], 0, 63, 0, &[0x3f]);
    bytes.extend([0xff, 0xff, 0xd9, 7, 8]);
    let mut jpeg = read_coefficients(&bytes).unwrap();
    jpeg.app_marker_type = vec![AppMarkerType::Icc, AppMarkerType::Exif];
    let encoded = encode(&jpeg);
    let p = parse(&encoded.header);
    assert_eq!(p.app_types, [0, 0]);
    assert_eq!(
        p.app_lengths,
        jpeg.app_data
            .iter()
            .map(|d| d.len() as u32)
            .collect::<Vec<_>>()
    );
    assert_eq!(p.com_lengths, [4]);
    assert_eq!(p.inter_lengths, [4, 1]);
    assert_eq!(p.tail_length, 2);
    let expected: Vec<u8> = jpeg
        .app_data
        .iter()
        .chain(&jpeg.com_data)
        .flatten()
        .copied()
        .chain([3, 4, 255, 255, 255, 7, 8])
        .collect();
    assert_eq!(encoded.payload, expected);
    let before = encoded;
    for record in &mut jpeg.records {
        if let JpegRecord::Entropy(entropy) = record {
            entropy.bytes.clear();
        }
    }
    assert_eq!(encode(&jpeg), before);
}

#[test]
fn empty_dht_adds_zero_table_in_marker_order_without_scan_index_remapping() {
    let mut bytes = vec![0xff, 0xd8];
    segment(&mut bytes, 0xc4, &[]);
    bytes.extend_from_slice(&header(0xc0, 8, &[1], &[(0, 1)])[2..]);
    segment(&mut bytes, 0xc4, &[]);
    scan(&mut bytes, &[1], 0, 63, 0, &[0x3f]);
    let jpeg = finish(bytes);
    assert_eq!(jpeg.huffman_code.len(), 2);
    let p = parse(&encode(&jpeg).header);
    assert_eq!(p.huffman.len(), 4);
    for index in [0, 3] {
        assert_eq!(p.huffman[index].0, 0);
        assert!(p.huffman[index].1);
        assert_eq!(p.huffman[index].2, [0; 17]);
        assert!(p.huffman[index].3.is_empty());
    }
    assert_eq!(p.huffman[1].3, [0, 256]);
    assert_eq!(p.huffman[2].0, 0x10);
    assert_eq!(p.scans[0][5..8], [0, 0, 0]);
}

#[test]
fn sentinel_uses_highest_populated_length_and_unencodable_count_is_rejected() {
    let mut bytes = header(0xc0, 8, &[1], &[(0, 1)]);
    huffman(&mut bytes, 0x11, 16, &[0xe0]);
    scan(&mut bytes, &[1], 0, 63, 0, &[0x3f]);
    let p = parse(&encode(&finish(bytes)).header);
    assert_eq!(p.huffman[2].2[16], 2);
    assert!(p.huffman[2].2[..16].iter().all(|&count| count == 0));
    assert_eq!(p.huffman[2].3, [0xe0, 256]);
    let mut bytes = header(0xc0, 8, &[1], &[(0, 1)]);
    huffman(&mut bytes, 0x11, 8, &(0..255).collect::<Vec<_>>());
    scan(&mut bytes, &[1], 0, 63, 0, &[0x3f]);
    let jpeg = finish(bytes);
    assert!(matches!(
        serialize_reconstruction_metadata(&jpeg, 4096),
        Err(MetadataError::Unsupported(_))
    ));
}

#[test]
fn component_type_codes_cover_gray_ycbcr_rgb_and_custom_ids() {
    for (ids, expected) in [
        (vec![1], 0),
        (vec![1, 2, 3], 1),
        (vec![82, 71, 66], 2),
        (vec![7], 3),
        (vec![8, 2, 19], 3),
    ] {
        let p = parse(&encode(&baseline(&ids, &[(0, 1)])).header);
        assert_eq!(p.component_type, expected);
        if expected == 3 {
            assert_eq!(
                p.ids,
                ids.iter().copied().map(u32::from).collect::<Vec<_>>()
            );
        } else {
            assert!(p.ids.is_empty());
        }
    }
}

#[test]
fn unused_quant_tables_must_equal_previous_and_four_tables_are_not_supported() {
    encode(&baseline(&[1], &[(3, 1), (0, 1), (2, 1)]));
    for tables in [vec![(3, 1), (0, 2)], vec![(0, 1), (1, 1), (2, 1), (3, 1)]] {
        assert!(matches!(
            serialize_reconstruction_metadata(&baseline(&[1], &tables), 4096),
            Err(MetadataError::Unsupported(_))
        ));
    }
}

#[test]
fn quant_redefinition_with_different_values_is_not_silently_collapsed() {
    for value in [1, 2] {
        let mut bytes = header(0xc2, 8, &[1], &[(3, 1)]);
        scan(&mut bytes, &[1], 0, 0, 0, &[0x7f]);
        quant(&mut bytes, &[(3, value)]);
        scan(&mut bytes, &[1], 1, 63, 0, &[0x7f]);
        let jpeg = finish(bytes);
        let result = serialize_reconstruction_metadata(&jpeg, 4096);
        if value == 1 {
            assert_eq!(parse(&result.unwrap().header).quant_indices, [0]);
        } else {
            assert!(matches!(result, Err(MetadataError::Unsupported(_))));
        }
    }
}

#[test]
fn changed_dri_and_restart_fill_are_explicitly_unsupported() {
    for interval in [0, 1] {
        let mut bytes = header(0xc0, 8, &[1], &[(0, 1)]);
        segment(&mut bytes, 0xdd, &[0, 0]);
        segment(&mut bytes, 0xdd, &[0, interval]);
        scan(&mut bytes, &[1], 0, 63, 0, &[0x3f]);
        let result = serialize_reconstruction_metadata(&finish(bytes), 4096);
        if interval == 0 {
            assert_eq!(parse(&result.unwrap().header).restart, Some(0));
        } else {
            assert!(matches!(result, Err(MetadataError::Unsupported(_))));
        }
    }
    for fill in [false, true] {
        let mut bytes = header(0xc0, 16, &[1], &[(0, 1)]);
        segment(&mut bytes, 0xdd, &[0, 1]);
        let entropy = if fill {
            vec![0x3f, 255, 255, 0xd0, 0x3f]
        } else {
            vec![0x3f, 255, 0xd0, 0x3f]
        };
        scan(&mut bytes, &[1], 0, 63, 0, &entropy);
        let result = serialize_reconstruction_metadata(&finish(bytes), 4096);
        if fill {
            assert!(matches!(result, Err(MetadataError::Unsupported(_))));
        } else {
            assert_eq!(parse(&result.unwrap().header).restart, Some(1));
        }
    }
}

#[test]
fn progressive_reset_hints_pass_fields_and_mixed_padding_are_serialized() {
    let mut bytes = header(0xc2, 16, &[1], &[(0, 1)]);
    scan(&mut bytes, &[1], 0, 0, 0, &[0x3f]);
    scan(&mut bytes, &[1], 1, 63, 0, &[0x2a]);
    let mut jpeg = finish(bytes);
    jpeg.scan_info[1].last_needed_pass = 10;
    assert_eq!(jpeg.scan_info[1].reset_points, [1]);
    let p = parse(&encode(&jpeg).header);
    assert_eq!(p.resets, [vec![], vec![1]]);
    assert_eq!(p.scans[1], [1, 1, 63, 0, 0, 0, 0, 0, 10]);
    assert_eq!(
        p.padding,
        jpeg.padding_bits
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>()
    );
    jpeg.scan_info[1].last_needed_pass = 11;
    assert!(serialize_reconstruction_metadata(&jpeg, 4096).is_err());
}

#[test]
fn extra_zero_runs_and_hint_delta_boundaries_are_preserved() {
    let mut bytes = vec![0xff, 0xd8];
    quant(&mut bytes, &[(0, 1)]);
    segment(&mut bytes, 0xc0, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
    huffman(&mut bytes, 0, 1, &[0]);
    huffman(&mut bytes, 0x10, 2, &[0, 0xf0]);
    scan(&mut bytes, &[1], 0, 63, 0, &[0x2a, 0xff, 0]); // DC 0, four 01 ZRLs, then ones.
    let jpeg = finish(bytes);
    let p = parse(&encode(&jpeg).header);
    assert_eq!(p.zeros, [vec![(0, 4)]]);
    let mut writer = BitWriter::new(64);
    let mut previous = None;
    let indices = [0, 1, 10, 20, 61, MAX_BLOCK_INDEX - 1];
    for index in indices {
        write_block_delta(&mut writer, &mut previous, index).unwrap();
    }
    assert!(write_block_delta(&mut writer, &mut previous, MAX_BLOCK_INDEX).is_err());
    assert!(write_block_delta(&mut writer, &mut previous, 0).is_err());
    writer.zero_pad_to_byte().unwrap();
    let bytes = writer.into_bytes().unwrap();
    let mut reader = Reader {
        bytes: &bytes,
        position: 0,
    };
    let mut base = 0;
    for expected in indices {
        let actual = base + reader.u32([(0, 0), (3, 1), (5, 9), (28, 41)]);
        assert_eq!(actual, expected);
        base = actual + 1;
    }
}

#[test]
fn combined_output_limit_is_exact_and_consecutive_fill_is_bounded() {
    let jpeg = baseline(&[1], &[(0, 1)]);
    let encoded = encode(&jpeg);
    let exact = encoded.header.len() + encoded.payload.len();
    assert_eq!(
        serialize_reconstruction_metadata(&jpeg, exact).unwrap(),
        encoded
    );
    assert!(serialize_reconstruction_metadata(&jpeg, exact - 1).is_err());
    assert!(serialize_reconstruction_metadata(&jpeg, 0).is_err());
    let mut jpeg = jpeg;
    jpeg.records.insert(
        1,
        JpegRecord::Fill {
            offset: 2,
            count: 65_534,
        },
    );
    jpeg.records.insert(
        2,
        JpegRecord::Fill {
            offset: 2,
            count: 1,
        },
    );
    let encoded = encode(&jpeg);
    assert_eq!(parse(&encoded.header).inter_lengths, [65_535]);
    assert_eq!(encoded.payload, vec![255; 65_535]);
    jpeg.records.insert(
        3,
        JpegRecord::Fill {
            offset: 2,
            count: 1,
        },
    );
    assert!(matches!(
        serialize_reconstruction_metadata(&jpeg, 1 << 20),
        Err(MetadataError::Limit(_))
    ));
    jpeg.records[1] = JpegRecord::Fill {
        offset: 2,
        count: usize::MAX,
    };
    assert!(serialize_reconstruction_metadata(&jpeg, usize::MAX).is_err());
}

#[test]
fn public_metadata_and_unsupported_layout_mutations_are_rejected() {
    let jpeg = baseline(&[1], &[(0, 1)]);
    let edits: [fn(&mut JpegData); 10] = [
        |j| {
            j.marker_order.remove(0);
        },
        |j| j.restart_interval = 1,
        |j| j.components[0].quant_index = 99,
        |j| j.components[0].coefficients.clear(),
        |j| j.padding_bits[0] = 2,
        |j| j.has_zero_padding_bit = true,
        |j| j.tail_data.push(1),
        |j| j.scan_info[0].reset_points = vec![0, 0],
        |j| {
            let index = j
                .records
                .iter()
                .position(|r| matches!(r, JpegRecord::Entropy(_)))
                .unwrap();
            let entropy = j.records.remove(index);
            j.records.push(entropy);
        },
        |j| {
            j.records.insert(
                1,
                JpegRecord::Marker(MarkerSegment {
                    offset: 2,
                    marker: 0xd8,
                    data: vec![],
                }),
            )
        },
    ];
    for (index, edit) in edits.into_iter().enumerate() {
        let mut invalid = jpeg.clone();
        edit(&mut invalid);
        assert!(
            serialize_reconstruction_metadata(&invalid, 4096).is_err(),
            "mutation {index}"
        );
    }
    let mut bytes = header(0xc0, 8, &[1], &[(0, 1)]);
    segment(&mut bytes, 0xe0, &[1]);
    scan(&mut bytes, &[1], 0, 63, 0, &[0x3f]);
    let mut jpeg = finish(bytes);
    jpeg.app_data[0][3] ^= 1;
    assert!(serialize_reconstruction_metadata(&jpeg, 4096).is_err());
}

#[test]
fn matched_sources_deduplicate_wire_types_and_rehydrate_original_archival_bytes() {
    let profile = gray_profile(190);
    let encoding =
        crate::color::ColorEncoding::icc(crate::color::ColorSpace::Gray, profile.clone()).unwrap();
    let tiff = b"MM\0\x2a\0\0\0\x08opaque original TIFF";
    let exif = exif_box(tiff);
    let xml = b"\0\xffnot normalized XML\r\n";
    let sources = ReconstructionMetadataSources {
        icc_profile: encoding.icc_profile(),
        exif_box_payload: Some(&exif),
        xmp_box_payload: Some(xml),
    };
    let mut bytes = vec![0xff, 0xd8];
    segment(&mut bytes, 0xe2, &icc_chunk(1, 2, &profile[..61]));
    segment(&mut bytes, 0xfe, b"comment stays opaque");
    bytes.extend([9, 10, 0xff, 0xff]);
    segment(&mut bytes, 0xe1, &exif_app(tiff));
    segment(&mut bytes, 0xeb, b"JP\0\x01opaque JUMBF");
    segment(&mut bytes, 0xe2, &icc_chunk(2, 2, &profile[61..]));
    bytes.extend_from_slice(&header(0xc0, 8, &[1], &[(0, 1)])[2..]);
    scan(&mut bytes, &[1], 0, 63, 0, &[0x3f]);
    // Detection scans all APP records, not opaque bytes or only pre-SOS data.
    segment(&mut bytes, 0xe1, &xmp_app(xml));
    bytes.extend([0xff, 0xff, 0xd9, 0xff, 0xe1, 7, 8]);
    let mut jpeg = read_coefficients(&bytes).unwrap();
    let original = jpeg.clone();
    let encoded = encode_with(&jpeg, sources);
    assert_eq!(parse(&encoded.header).app_types, [1, 2, 0, 1, 3]);
    let all_unknown = encode(&jpeg);
    let removed: usize = [0, 1, 3, 4].map(|i| jpeg.app_data[i].len()).iter().sum();
    assert_eq!(all_unknown.payload.len() - encoded.payload.len(), removed);
    assert_metadata_rehydrates(&jpeg, &encoded, sources);
    assert_eq!(jpeg.app_marker_type, original.app_marker_type);
    assert_eq!(jpeg.app_data, original.app_data);
    assert_eq!(jpeg.records, original.records);
    assert_eq!(jpeg.components, original.components);
    assert_eq!(jpeg.reconstruct().unwrap(), bytes);
    for record in &mut jpeg.records {
        if let JpegRecord::Entropy(entropy) = record {
            entropy.bytes.clear();
        }
    }
    assert_eq!(encode_with(&jpeg, sources), encoded);
}

#[test]
fn absent_or_mismatched_output_sources_preserve_unknown_defaults() {
    let profile = gray_profile(132);
    let exif = exif_box(b"first TIFF");
    let jpeg = with_apps(&[
        (0xe2, icc_chunk(1, 1, &profile)),
        (0xe1, exif_app(b"first TIFF")),
        (0xe1, xmp_app(b"<first/>")),
    ]);
    let defaults = encode(&jpeg);
    assert_eq!(
        encode_with(&jpeg, ReconstructionMetadataSources::default()),
        defaults
    );
    let matching = ReconstructionMetadataSources {
        icc_profile: Some(&profile),
        exif_box_payload: Some(&exif),
        xmp_box_payload: Some(b"<first/>"),
    };
    for mask in 0u8..8 {
        let sources = ReconstructionMetadataSources {
            icc_profile: matching.icc_profile.filter(|_| mask & 1 != 0),
            exif_box_payload: matching.exif_box_payload.filter(|_| mask & 2 != 0),
            xmp_box_payload: matching.xmp_box_payload.filter(|_| mask & 4 != 0),
        };
        let encoded = encode_with(&jpeg, sources);
        assert_eq!(
            parse(&encoded.header).app_types,
            [
                u32::from(mask & 1 != 0),
                2 * u32::from(mask & 2 != 0),
                3 * u32::from(mask & 4 != 0),
            ]
        );
        assert_metadata_rehydrates(&jpeg, &encoded, sources);
    }
    let mut wrong_profile = profile.clone();
    wrong_profile[40] ^= 1;
    let mut wrong_offset = exif.clone();
    wrong_offset[3] = 1;
    let wrong = ReconstructionMetadataSources {
        icc_profile: Some(&wrong_profile),
        exif_box_payload: Some(&wrong_offset),
        xmp_box_payload: Some(b"<other/>"),
    };
    assert_eq!(encode_with(&jpeg, wrong), defaults);
    for profile in [
        None,
        Some(&[][..]),
        Some(&profile[..131]),
        Some(&[0; 133][..]),
    ] {
        let sources = ReconstructionMetadataSources {
            icc_profile: profile,
            ..Default::default()
        };
        assert_eq!(encode_with(&jpeg, sources), defaults);
    }
    for exif in [
        &[][..],
        &exif[..3],
        &exif[..exif.len() - 1],
        b"\0\0\0\0wrong TIFF",
    ] {
        let sources = ReconstructionMetadataSources {
            exif_box_payload: Some(exif),
            ..Default::default()
        };
        assert_eq!(encode_with(&jpeg, sources), defaults);
    }
}

#[test]
fn icc_dedup_requires_complete_source_order_and_exact_profile_concatenation() {
    let profile = gray_profile(150);
    let chunks = [&profile[..17], &profile[17..99], &profile[99..]];
    let sources = ReconstructionMetadataSources {
        icc_profile: Some(&profile),
        ..Default::default()
    };
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let apps: Vec<_> = order
            .into_iter()
            .map(|i| (0xe2, icc_chunk(i as u8 + 1, 3, chunks[i])))
            .collect();
        let jpeg = with_apps(&apps);
        let encoded = encode_with(&jpeg, sources);
        assert_eq!(
            parse(&encoded.header).app_types,
            if order == [0, 1, 2] { [1; 3] } else { [0; 3] }
        );
        assert_metadata_rehydrates(&jpeg, &encoded, sources);
    }
    for chunks in [
        vec![icc_chunk(1, 2, &profile)],
        vec![icc_chunk(2, 2, &profile)],
        vec![icc_chunk(0, 1, &profile)],
        vec![icc_chunk(1, 0, &profile)],
        vec![
            icc_chunk(1, 2, &profile[..50]),
            icc_chunk(2, 3, &profile[50..]),
        ],
        // Source's expected-count sentinel must not change an original zero.
        vec![
            icc_chunk(1, 0, &profile[..50]),
            icc_chunk(2, 2, &profile[50..]),
        ],
        vec![b"ICC_PROFILE\0".to_vec()],
        vec![b"ICC_PROFILE\0\x01".to_vec()],
        vec![b"ICC_PROFILE!\x01\x01".to_vec()],
    ] {
        let apps: Vec<_> = chunks.into_iter().map(|p| (0xe2, p)).collect();
        let jpeg = with_apps(&apps);
        let encoded = encode_with(&jpeg, sources);
        assert_eq!(encoded, encode(&jpeg));
        assert_metadata_rehydrates(&jpeg, &encoded, sources);
    }
}

#[test]
fn icc_skipped_duplicates_and_wrong_counts_stay_unknown_while_selected_chunks_match() {
    let profile = gray_profile(132);
    let sources = ReconstructionMetadataSources {
        icc_profile: Some(&profile),
        ..Default::default()
    };
    let jpeg = with_apps(&[
        (0xe2, icc_chunk(2, 2, b"early out-of-order chunk")),
        (0xe2, icc_chunk(1, 2, &profile[..66])),
        (0xe2, icc_chunk(1, 2, b"duplicate not part of profile")),
        (0xe2, icc_chunk(2, 3, b"wrong count")),
        (0xef, icc_chunk(2, 2, b"wrong APP type")),
        (0xe2, icc_chunk(2, 2, &profile[66..])),
        (0xe2, icc_chunk(2, 2, b"trailing duplicate")),
    ]);
    let encoded = encode_with(&jpeg, sources);
    assert_eq!(parse(&encoded.header).app_types, [0, 1, 0, 0, 0, 1, 0]);
    assert_metadata_rehydrates(&jpeg, &encoded, sources);
    // A mismatched first candidate cannot be replaced with a later duplicate.
    let jpeg = with_apps(&[
        (0xe2, icc_chunk(1, 1, b"not the emitted profile")),
        (0xe2, icc_chunk(1, 1, &profile)),
    ]);
    assert_eq!(encode_with(&jpeg, sources), encode(&jpeg));
}

#[test]
fn icc_empty_chunks_and_maximum_255_chunk_count_preserve_exact_headers() {
    let profile = gray_profile(255);
    let sources = ReconstructionMetadataSources {
        icc_profile: Some(&profile),
        ..Default::default()
    };
    let jpeg = with_apps(&[
        (0xe2, icc_chunk(1, 3, &profile[..37])),
        (0xe2, icc_chunk(2, 3, &[])),
        (0xe2, icc_chunk(3, 3, &profile[37..])),
    ]);
    let encoded = encode_with(&jpeg, sources);
    assert_eq!(parse(&encoded.header).app_types, [1; 3]);
    assert_metadata_rehydrates(&jpeg, &encoded, sources);
    let apps: Vec<_> = profile
        .iter()
        .enumerate()
        .map(|(i, &byte)| (0xe2, icc_chunk((i + 1) as u8, 255, &[byte])))
        .collect();
    let jpeg = with_apps(&apps);
    let encoded = encode_with(&jpeg, sources);
    assert_eq!(parse(&encoded.header).app_types, vec![1; 255]);
    assert!(encoded.payload.is_empty());
    assert_metadata_rehydrates(&jpeg, &encoded, sources);
}

#[test]
fn exif_and_xmp_first_candidate_selection_does_not_skip_mismatches_or_normalize() {
    let exif = exif_box(b"\xffnot TIFF\0");
    let sources = ReconstructionMetadataSources {
        exif_box_payload: Some(&exif),
        xmp_box_payload: Some(b"\xffnot XML\0"),
        ..Default::default()
    };
    let jpeg = with_apps(&[
        (0xe1, exif_app(&[])), // Empty EXIF is not a DetectBlobs candidate.
        (0xe2, exif_app(b"\xffnot TIFF\0")),
        (0xe1, exif_app(b"\xffnot TIFF\0")),
        (0xe1, exif_app(b"duplicate TIFF")),
        (0xe2, xmp_app(b"\xffnot XML\0")),
        (0xe1, xmp_app(b"\xffnot XML\0")),
        (0xe1, xmp_app(b"duplicate XML")),
        (
            0xe1,
            b"http://ns.adobe.com/xmp/extension/\0not assembled".to_vec(),
        ),
    ]);
    let encoded = encode_with(&jpeg, sources);
    assert_eq!(parse(&encoded.header).app_types, [0, 0, 2, 0, 0, 3, 0, 0]);
    assert_metadata_rehydrates(&jpeg, &encoded, sources);
    let jpeg = with_apps(&[
        (0xe1, exif_app(b"first mismatched TIFF")),
        (0xe1, exif_app(b"\xffnot TIFF\0")),
        (0xe1, xmp_app(b"first mismatched XML")),
        (0xe1, xmp_app(b"\xffnot XML\0")),
    ]);
    assert_eq!(encode_with(&jpeg, sources), encode(&jpeg));
}

#[test]
fn empty_first_xmp_requires_matching_empty_output_not_a_later_nonempty_blob() {
    let jpeg = with_apps(&[(0xe1, xmp_app(&[])), (0xe1, xmp_app(b"<later/>"))]);
    for xmp in [None, Some(b"<later/>".as_slice())] {
        let sources = ReconstructionMetadataSources {
            xmp_box_payload: xmp,
            ..Default::default()
        };
        assert_eq!(encode_with(&jpeg, sources), encode(&jpeg));
    }
    let sources = ReconstructionMetadataSources {
        xmp_box_payload: Some(&[]),
        ..Default::default()
    };
    let encoded = encode_with(&jpeg, sources);
    assert_eq!(parse(&encoded.header).app_types, [3, 0]);
    assert_metadata_rehydrates(&jpeg, &encoded, sources);
}

#[test]
fn dedup_budget_counts_retained_payload_and_enforces_exact_final_header_boundary() {
    let profile = gray_profile(65519);
    let tiff = vec![0x81; 65527];
    let exif = exif_box(&tiff);
    let xml = vec![0x82; 65504];
    let jpeg = with_apps(&[
        (0xe2, icc_chunk(1, 1, &profile)),
        (0xe1, exif_app(&tiff)),
        (0xe1, xmp_app(&xml)),
    ]);
    let sources = ReconstructionMetadataSources {
        icc_profile: Some(&profile),
        exif_box_payload: Some(&exif),
        xmp_box_payload: Some(&xml),
    };
    let encoded = encode_with(&jpeg, sources);
    assert_eq!(parse(&encoded.header).app_types, [1, 2, 3]);
    assert!(encoded.payload.is_empty());
    let exact = encoded.header.len();
    assert_eq!(
        serialize_reconstruction_metadata_with_sources(&jpeg, exact, sources).unwrap(),
        encoded
    );
    assert!(serialize_reconstruction_metadata_with_sources(&jpeg, exact - 1, sources).is_err());
    assert!(serialize_reconstruction_metadata_with_sources(&jpeg, 0, sources).is_err());
    assert!(serialize_reconstruction_metadata(&jpeg, exact).is_err());
    assert_metadata_rehydrates(&jpeg, &encoded, sources);
}

#[test]
fn selected_app_metadata_still_requires_valid_records_and_ignores_untrusted_type_annotations() {
    let profile = gray_profile(132);
    let sources = ReconstructionMetadataSources {
        icc_profile: Some(&profile),
        ..Default::default()
    };
    let original = with_apps(&[(0xe2, icc_chunk(1, 1, &profile))]);
    let expected = encode_with(&original, sources);
    let mut annotated = original.clone();
    annotated.app_marker_type = vec![AppMarkerType::Xmp];
    assert_eq!(encode_with(&annotated, sources), expected);
    assert_eq!(annotated.app_marker_type, [AppMarkerType::Xmp]);
    assert_eq!(encode(&annotated), encode(&original));
    for case in 0..5 {
        let mut jpeg = original.clone();
        match case {
            0 => jpeg.app_data[0][3] ^= 1,
            1 => jpeg.marker_order.insert(1, 0xff),
            2 => jpeg.app_marker_type.clear(),
            3 => jpeg.records.clear(),
            4 => {
                // Matching mirrors cannot hide an invalid declared APP length.
                jpeg.app_data[0][2] ^= 1;
                for record in &mut jpeg.records {
                    if let JpegRecord::Marker(segment) = record
                        && segment.marker == 0xe2
                    {
                        segment.data[1] ^= 1;
                    }
                }
            }
            _ => unreachable!(),
        }
        assert!(
            serialize_reconstruction_metadata_with_sources(&jpeg, 1 << 20, sources).is_err(),
            "case {case}"
        );
    }
}
