//! Deterministic development-only JPEG inputs, not a production JPEG encoder.
//!
//! Included by the core's JPEG acceptance tests. Fixture unit tests use
//! the core crate's JPEG primitives; the builder itself has no codec dependency.
//! `fixtures()` returns 19 stable (name, bytes) pairs with no filesystem access,
//! codec dependency, DCT or pixel conversion. Dimensions are at most 600x400.
//! Separate targeted families exercise two DC groups and reconstruction metadata
//! without enlarging that Cartesian suite. Every JPEG remains below 2 MiB.
//! The canonical Huffman/amplitude construction follows the fixture approach in
//! jpeg/coefficient_reader_tests.rs, including its small signed-refinement case.

const MAX_FIXTURE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sampling {
    Gray,
    Rgb,
    Yuv444,
    Yuv420,
    Yuv422,
    Yuv440,
}

#[derive(Clone, Copy)]
struct Component {
    id: u8,
    horizontal: usize,
    vertical: usize,
    quantization: u8,
}

impl Sampling {
    fn components(self) -> Vec<Component> {
        if self == Self::Gray {
            return vec![Component {
                id: 1,
                horizontal: 1,
                vertical: 1,
                quantization: 0,
            }];
        }
        let ids = if self == Self::Rgb {
            *b"RGB"
        } else {
            [1, 2, 3]
        };
        let (h, v) = match self {
            Self::Yuv420 => (2, 2),
            Self::Yuv422 => (2, 1),
            Self::Yuv440 => (1, 2),
            _ => (1, 1),
        };
        ids.into_iter()
            .enumerate()
            .map(|(c, id)| Component {
                id,
                horizontal: if c == 0 { h } else { 1 },
                vertical: if c == 0 { v } else { 1 },
                quantization: c as u8,
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Process {
    Sequential,
    Progressive,
    Refinement,
}

#[derive(Clone, Copy)]
struct Spec {
    name: &'static str,
    width: usize,
    height: usize,
    sampling: Sampling,
    process: Process,
    restart_interval: usize,
}

impl Spec {
    const fn new(
        name: &'static str,
        size: (usize, usize),
        sampling: Sampling,
        process: Process,
        restart_interval: usize,
    ) -> Self {
        Self {
            name,
            width: size.0,
            height: size.1,
            sampling,
            process,
            restart_interval,
        }
    }
}

const SPECS: [Spec; 19] = [
    Spec::new(
        "gray_seq_1x1",
        (1, 1),
        Sampling::Gray,
        Process::Sequential,
        0,
    ),
    Spec::new(
        "gray_seq_17x9",
        (17, 9),
        Sampling::Gray,
        Process::Sequential,
        0,
    ),
    Spec::new(
        "gray_seq_256x256",
        (256, 256),
        Sampling::Gray,
        Process::Sequential,
        0,
    ),
    Spec::new(
        "gray_seq_257x9_rst3",
        (257, 9),
        Sampling::Gray,
        Process::Sequential,
        3,
    ),
    Spec::new("rgb_seq_8x8", (8, 8), Sampling::Rgb, Process::Sequential, 0),
    Spec::new(
        "rgb_seq_257x17_rst5",
        (257, 17),
        Sampling::Rgb,
        Process::Sequential,
        5,
    ),
    Spec::new(
        "rgb_seq_33x25",
        (33, 25),
        Sampling::Rgb,
        Process::Sequential,
        0,
    ),
    Spec::new(
        "ycbcr444_seq_19x13",
        (19, 13),
        Sampling::Yuv444,
        Process::Sequential,
        0,
    ),
    Spec::new(
        "ycbcr444_seq_513x257_rst7",
        (513, 257),
        Sampling::Yuv444,
        Process::Sequential,
        7,
    ),
    Spec::new(
        "ycbcr420_seq_17x19_rst2",
        (17, 19),
        Sampling::Yuv420,
        Process::Sequential,
        2,
    ),
    Spec::new(
        "ycbcr420_seq_600x400_rst7",
        (600, 400),
        Sampling::Yuv420,
        Process::Sequential,
        7,
    ),
    Spec::new(
        "ycbcr422_seq_31x17_rst3",
        (31, 17),
        Sampling::Yuv422,
        Process::Sequential,
        3,
    ),
    Spec::new(
        "ycbcr440_seq_17x31_rst4",
        (17, 31),
        Sampling::Yuv440,
        Process::Sequential,
        4,
    ),
    Spec::new(
        "gray_prog_273x17_rst5",
        (273, 17),
        Sampling::Gray,
        Process::Progressive,
        5,
    ),
    Spec::new(
        "ycbcr444_prog_257x17",
        (257, 17),
        Sampling::Yuv444,
        Process::Progressive,
        0,
    ),
    Spec::new(
        "ycbcr420_prog_33x19_rst3",
        (33, 19),
        Sampling::Yuv420,
        Process::Progressive,
        3,
    ),
    Spec::new(
        "ycbcr422_prog_257x33_rst7",
        (257, 33),
        Sampling::Yuv422,
        Process::Progressive,
        7,
    ),
    Spec::new(
        "ycbcr440_prog_33x257_rst5",
        (33, 257),
        Sampling::Yuv440,
        Process::Progressive,
        5,
    ),
    Spec::new(
        "gray_prog_refinement_24x8",
        (24, 8),
        Sampling::Gray,
        Process::Refinement,
        0,
    ),
];

/// Stable names have no extension. No decoder, filesystem or external program
/// is invoked here. The parent supplies official JPEG/JXL oracle assertions.
pub fn fixtures() -> Vec<(String, Vec<u8>)> {
    SPECS
        .iter()
        .map(|&spec| (spec.name.to_owned(), build(spec)))
        .collect()
}

/// Build only one member of the fixed Cartesian corpus.
pub fn fixture(name: &str) -> Option<Vec<u8>> {
    SPECS
        .iter()
        .find(|spec| spec.name == name)
        .copied()
        .map(build)
}

/// A tiny signed-refinement case and a narrow two-DC-group progressive case.
pub fn tier_fixtures() -> Vec<(String, Vec<u8>)> {
    let wide = Spec::new(
        "tier_ycbcr444_prog_2057x17_rst5",
        (2057, 17),
        Sampling::Yuv444,
        Process::Progressive,
        5,
    );
    vec![
        (
            "gray_prog_refinement_24x8".into(),
            fixture("gray_prog_refinement_24x8").unwrap(),
        ),
        (wide.name.into(), build(wide)),
    ]
}

// ICC header/mluc/XYZ/curve layouts follow cms/jxl_cms_internal.h at frozen
// b866b3defec13494bfbaf705db93581cda118e37. Copyright (c) the JPEG XL Project
// Authors. BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
fn icc_text(text: &str) -> Vec<u8> {
    assert!(text.is_ascii() && text.len() <= 128);
    let mut tag = b"mluc\0\0\0\0".to_vec();
    tag.extend_from_slice(&1u32.to_be_bytes());
    tag.extend_from_slice(&12u32.to_be_bytes());
    tag.extend_from_slice(b"enUS");
    tag.extend_from_slice(&(text.len() as u32 * 2).to_be_bytes());
    tag.extend_from_slice(&28u32.to_be_bytes());
    for byte in text.bytes() {
        tag.extend_from_slice(&[0, byte]);
    }
    tag
}

fn icc_xyz(values: [u32; 3]) -> Vec<u8> {
    let mut tag = b"XYZ \0\0\0\0".to_vec();
    for value in values {
        tag.extend_from_slice(&value.to_be_bytes());
    }
    tag
}

fn tagged_icc(gray: bool) -> Vec<u8> {
    let white = [0x0000_f6d6, 0x0001_0000, 0x0000_d32d];
    let mut curve = b"curv\0\0\0\0".to_vec();
    curve.extend_from_slice(&17u32.to_be_bytes());
    for i in 0..=16u32 {
        curve.extend_from_slice(&((i * i * 65535 / 256) as u16).to_be_bytes());
    }
    let mut tags = vec![
        (
            *b"desc",
            icc_text(if gray {
                "Gray quadratic fixture"
            } else {
                "RGB D50 quadratic fixture"
            }),
        ),
        (*b"cprt", icc_text("Deterministic codec test data")),
        (*b"wtpt", icc_xyz(white)),
    ];
    if gray {
        tags.push((*b"kTRC", curve));
    } else {
        // Fixed D50-adapted RGB columns sum exactly to the header white point.
        tags.extend([
            (*b"rXYZ", icc_xyz([0x6fa0, 0x38f5, 0x0390])),
            (*b"gXYZ", icc_xyz([0x6297, 0xb785, 0x18d9])),
            (*b"bXYZ", icc_xyz([0x249f, 0x0f86, 0xb6c4])),
            (*b"rTRC", curve.clone()),
            (*b"gTRC", curve.clone()),
            (*b"bTRC", curve),
        ]);
    }
    let mut profile = vec![0; 132 + tags.len() * 12];
    profile[4..8].copy_from_slice(b"test");
    profile[8..12].copy_from_slice(&0x0440_0000u32.to_be_bytes());
    profile[12..16].copy_from_slice(b"mntr");
    profile[16..20].copy_from_slice(if gray { b"GRAY" } else { b"RGB " });
    profile[20..24].copy_from_slice(b"XYZ ");
    for (index, value) in [2020u16, 1, 2, 3, 4, 5].into_iter().enumerate() {
        profile[24 + index * 2..26 + index * 2].copy_from_slice(&value.to_be_bytes());
    }
    profile[36..40].copy_from_slice(b"acsp");
    profile[40..44].copy_from_slice(b"MSFT");
    profile[64..68].copy_from_slice(&1u32.to_be_bytes());
    for (index, value) in white.into_iter().enumerate() {
        profile[68 + index * 4..72 + index * 4].copy_from_slice(&value.to_be_bytes());
    }
    profile[80..84].copy_from_slice(b"test");
    profile[128..132].copy_from_slice(&(tags.len() as u32).to_be_bytes());
    for (index, (signature, data)) in tags.into_iter().enumerate() {
        let offset = profile.len() as u32;
        let entry = 132 + index * 12;
        profile[entry..entry + 4].copy_from_slice(&signature);
        profile[entry + 4..entry + 8].copy_from_slice(&offset.to_be_bytes());
        profile[entry + 8..entry + 12].copy_from_slice(&(data.len() as u32).to_be_bytes());
        profile.extend_from_slice(&data);
        while !profile.len().is_multiple_of(4) {
            profile.push(0);
        }
    }
    let length = profile.len() as u32;
    profile[..4].copy_from_slice(&length.to_be_bytes());
    assert!(profile.len() <= 4096);
    profile
}

fn icc_segment(output: &mut Vec<u8>, sequence: u8, count: u8, part: &[u8]) {
    let mut payload = b"ICC_PROFILE\0".to_vec();
    payload.extend_from_slice(&[sequence, count]);
    payload.extend_from_slice(part);
    segment(output, 0xe2, &payload);
}

fn split_icc(output: &mut Vec<u8>, profile: &[u8]) {
    let middle = profile.len() / 2;
    for (sequence, part) in [
        (3, &profile[middle..]),
        (1, &profile[..17]),
        (2, &profile[17..middle]),
    ] {
        icc_segment(output, sequence, 3, part);
        segment(output, 0xe7, b"between ICC chunks\0\xff\xe2");
    }
}

fn exif(little: bool, orientation: u16) -> Vec<u8> {
    let word = |value: u16| {
        if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        }
    };
    let long = |value: u32| {
        if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        }
    };
    let description = b"rs-jxl raw metadata fixture\0";
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(if little { b"II" } else { b"MM" });
    payload.extend_from_slice(&word(42));
    payload.extend_from_slice(&long(8));
    payload.extend_from_slice(&word(2));
    payload.extend_from_slice(&word(0x010e)); // ASCII description at TIFF offset 38.
    payload.extend_from_slice(&word(2));
    payload.extend_from_slice(&long(description.len() as u32));
    payload.extend_from_slice(&long(38));
    payload.extend_from_slice(&word(0x0112)); // Inline SHORT orientation.
    payload.extend_from_slice(&word(3));
    payload.extend_from_slice(&long(1));
    payload.extend_from_slice(&word(orientation));
    payload.extend_from_slice(&[0, 0]);
    payload.extend_from_slice(&long(0));
    payload.extend_from_slice(description);
    payload
}

fn with_metadata(base: &[u8], header: &[u8], after_scan: &[u8], tail: &[u8]) -> Vec<u8> {
    assert!(base.starts_with(&[0xff, 0xd8]) && base.ends_with(&[0xff, 0xd9]));
    let length = base.len() + header.len() + after_scan.len() + tail.len();
    assert!(length <= MAX_FIXTURE_BYTES);
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(&base[..2]);
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&base[2..base.len() - 2]);
    bytes.extend_from_slice(after_scan);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes.extend_from_slice(tail);
    bytes
}

/// Metadata-only variants, kept out of the coefficient Cartesian suite.
pub fn metadata_fixtures() -> Vec<(String, Vec<u8>)> {
    let rgb = fixture("rgb_seq_33x25").unwrap();
    let gray = fixture("gray_seq_17x9").unwrap();
    let rgb_icc = tagged_icc(false);
    let gray_icc = tagged_icc(true);
    let mut result = Vec::new();
    let mut header = Vec::new();
    icc_segment(&mut header, 1, 1, &rgb_icc);
    segment(&mut header, 0xe1, &exif(true, 6));
    result.push((
        "metadata_rgb_icc_single_exif_le".into(),
        with_metadata(&rgb, &header, &[], &[]),
    ));

    header.clear();
    split_icc(&mut header, &rgb_icc);
    segment(&mut header, 0xe1, &exif(false, 3));
    result.push((
        "metadata_rgb_icc_split_exif_be".into(),
        with_metadata(&rgb, &header, &[], &[]),
    ));

    header.clear();
    split_icc(&mut header, &gray_icc);
    segment(&mut header, 0xe1, &exif(true, 8));
    segment(&mut header, 0xfe, b"gray ICC comment\0\xff\xd9");
    let tail = b"\0tail\xff\xe2\0\x12ICC_PROFILE\0\x01\x01\xff\xda\xff\xd9";
    result.push((
        "metadata_gray_icc_split_tail".into(),
        with_metadata(&gray, &header, &[], tail),
    ));

    header.clear();
    header.extend_from_slice(&[0x21, 0xff, 0, 0x22, 0xff, 0xff]);
    let opaque = b"duplicate APP\0\xff\xd8\xff\xda\xff\xd9\xff\0";
    for _ in 0..2 {
        segment(&mut header, 0xe7, opaque);
        segment(&mut header, 0xe1, &exif(true, 6));
        segment(&mut header, 0xfe, b"duplicate COM\0\xff");
    }
    segment(&mut header, 0xe1, &exif(false, 3));
    segment(&mut header, 0xe3, b"ICC_PROFILE\0opaque outside APP2");
    let mut after_scan = Vec::new();
    segment(&mut after_scan, 0xe7, opaque);
    segment(&mut after_scan, 0xfe, b"after scan\0\xff");
    result.push((
        "metadata_duplicate_app_exif_com_fill_tail".into(),
        with_metadata(&rgb, &header, &after_scan, tail),
    ));

    header.clear();
    let maximum: Vec<_> = (0..65533usize).map(|i| (i * 37 + 11) as u8).collect();
    segment(&mut header, 0xef, &maximum);
    segment(&mut header, 0xe1, b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><fixture>raw</fixture></x:xmpmeta>");
    let progressive = fixture("ycbcr420_prog_33x19_rst3").unwrap();
    result.push((
        "metadata_maximum_app_xmp_progressive".into(),
        with_metadata(&progressive, &header, &[], &[]),
    ));
    result
}

/// Strict color-metadata rejection cases; entropy and JPEG marker syntax stay valid.
pub fn malformed_metadata_fixtures() -> Vec<(String, Vec<u8>)> {
    let base = fixture("rgb_seq_8x8").unwrap();
    let profile = tagged_icc(false);
    let mut result = Vec::new();
    let mut header = Vec::new();
    icc_segment(&mut header, 1, 1, &profile);
    icc_segment(&mut header, 1, 1, &profile);
    result.push((
        "invalid_duplicate_icc".into(),
        with_metadata(&base, &header, &[], &[]),
    ));
    header.clear();
    icc_segment(&mut header, 1, 2, &profile[..17]);
    result.push((
        "invalid_missing_icc_chunk".into(),
        with_metadata(&base, &header, &[], &[]),
    ));
    icc_segment(&mut header, 2, 3, &profile[17..]);
    result.push((
        "invalid_inconsistent_icc_count".into(),
        with_metadata(&base, &header, &[], &[]),
    ));
    header.clear();
    icc_segment(&mut header, 1, 1, &tagged_icc(true));
    result.push((
        "invalid_icc_component_space".into(),
        with_metadata(&base, &header, &[], &[]),
    ));
    header.clear();
    icc_segment(&mut header, 1, 1, &profile);
    result.push((
        "invalid_late_icc".into(),
        with_metadata(&base, &[], &header, &[]),
    ));
    header.clear();
    let mut bad_tag = profile;
    bad_tag[136..140].copy_from_slice(&u32::MAX.to_be_bytes());
    icc_segment(&mut header, 1, 1, &bad_tag);
    result.push((
        "invalid_icc_tag_offset".into(),
        with_metadata(&base, &header, &[], &[]),
    ));
    result
}

fn segment(output: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    let length = u16::try_from(payload.len() + 2).expect("bounded fixture segment");
    output.extend_from_slice(&[0xff, marker]);
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(payload);
    assert!(output.len() <= MAX_FIXTURE_BYTES);
}

fn huffman_table(output: &mut Vec<u8>, selector: u8, width: usize, symbols: &[u8]) {
    assert!(symbols.len() < (1 << width)); // No all-ones Huffman code.
    let mut payload = vec![selector];
    let mut counts = [0u8; 16];
    counts[width - 1] = u8::try_from(symbols.len()).unwrap();
    payload.extend_from_slice(&counts);
    payload.extend_from_slice(symbols);
    segment(output, 0xc4, &payload);
}

fn header(spec: Spec, components: &[Component]) -> Vec<u8> {
    assert!((1..=2057).contains(&spec.width) && (1..=400).contains(&spec.height));
    assert!(spec.width * spec.height <= 600 * 400);
    let mut output = vec![0xff, 0xd8];
    if spec.sampling == Sampling::Rgb {
        // Adobe version 100, flags zero, transform 0; no JFIF marker for RGB.
        segment(&mut output, 0xee, b"Adobe\x00\x64\x00\x00\x00\x00\x00");
    } else {
        segment(
            &mut output,
            0xe0,
            b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00",
        );
    }
    for component in components {
        let mut quantization = vec![component.quantization];
        quantization.extend((0..64u8).map(|k| 1 + (k * 3 + component.quantization * 5) % 23));
        segment(&mut output, 0xdb, &quantization);
    }
    let dc: Vec<u8> = (0..=11).collect();
    huffman_table(&mut output, 0, 4, &dc);
    let mut ac = vec![0, 0xf0];
    for run in 0..16u8 {
        for size in 1..=10u8 {
            ac.push((run << 4) | size);
        }
    }
    huffman_table(&mut output, 0x10, 8, &ac);
    let mut frame = vec![8];
    frame.extend_from_slice(&(spec.height as u16).to_be_bytes());
    frame.extend_from_slice(&(spec.width as u16).to_be_bytes());
    frame.push(components.len() as u8);
    for component in components {
        frame.extend_from_slice(&[
            component.id,
            ((component.horizontal << 4) | component.vertical) as u8,
            component.quantization,
        ]);
    }
    segment(
        &mut output,
        if spec.process == Process::Sequential {
            0xc0
        } else {
            0xc2
        },
        &frame,
    );
    if spec.restart_interval != 0 {
        segment(
            &mut output,
            0xdd,
            &(spec.restart_interval as u16).to_be_bytes(),
        );
    }
    output
}

#[derive(Default)]
struct Entropy {
    bytes: Vec<u8>,
    partial: u8,
    used: u8,
}

impl Entropy {
    fn word(&mut self, value: u32, width: u8) {
        assert!(width <= 16 && value < (1u32 << width));
        for bit in (0..width).rev() {
            self.partial = (self.partial << 1) | ((value >> bit) & 1) as u8;
            self.used += 1;
            if self.used == 8 {
                self.bytes.push(self.partial);
                if self.partial == 0xff {
                    self.bytes.push(0);
                }
                self.used = 0;
                self.partial = 0;
                assert!(self.bytes.len() <= MAX_FIXTURE_BYTES);
            }
        }
    }

    fn amplitude(&mut self, value: i16, width: u8) {
        let bits = if value < 0 {
            i32::from(value) + (1 << width) - 1
        } else {
            i32::from(value)
        };
        self.word(bits as u32, width);
    }

    fn dc(&mut self, difference: i16) {
        let width = category(difference);
        assert!(width <= 11);
        self.word(u32::from(width), 4);
        self.amplitude(difference, width);
    }

    fn ac_symbol(&mut self, symbol: u8) {
        let code = match symbol {
            0 => 0,
            0xf0 => 1,
            _ => {
                let size = symbol & 15;
                assert!((1..=10).contains(&size));
                2 + u32::from(symbol >> 4) * 10 + u32::from(size - 1)
            }
        };
        self.word(code, 8);
    }

    fn ac_block(&mut self, block: &[i16; 64]) {
        let mut run = 0u8;
        for &value in &block[1..] {
            if value == 0 {
                run += 1;
                continue;
            }
            while run >= 16 {
                self.ac_symbol(0xf0);
                run -= 16;
            }
            let width = category(value);
            self.ac_symbol((run << 4) | width);
            self.amplitude(value, width);
            run = 0;
        }
        if run != 0 {
            self.ac_symbol(0);
        }
    }

    fn pad(&mut self) {
        if self.used != 0 {
            let width = 8 - self.used;
            self.word((1u32 << width) - 1, width);
        }
    }

    fn restart(&mut self, index: usize) {
        self.pad();
        self.bytes
            .extend_from_slice(&[0xff, 0xd0 + (index & 7) as u8]);
        assert!(self.bytes.len() <= MAX_FIXTURE_BYTES);
    }

    fn finish(mut self) -> Vec<u8> {
        self.pad();
        self.bytes
    }
}

fn category(value: i16) -> u8 {
    (u16::BITS - value.unsigned_abs().leading_zeros()) as u8
}

// Values are indexed by JPEG zigzag rank, not natural coefficient coordinates.
// Large gaps exercise ZRL; rank 63 alternates between present and trailing EOB.
fn block_values(channel: usize, x: usize, y: usize) -> [i16; 64] {
    let seed = 5 + 17 * x + 11 * y + 23 * channel;
    let mut block = [0i16; 64];
    block[0] = (seed % 127) as i16 - 63;
    if seed % 7 == 6 {
        return block;
    }
    let sign = if seed.is_multiple_of(2) { 1 } else { -1 };
    block[1] = sign * 3;
    block[3] = -sign * 2;
    block[8] = sign;
    block[27] = -sign * 7;
    block[45] = sign * 9;
    if seed.is_multiple_of(3) {
        block[63] = -sign * 5;
    }
    block
}

fn max_sampling(components: &[Component]) -> (usize, usize) {
    (
        components.iter().map(|c| c.horizontal).max().unwrap(),
        components.iter().map(|c| c.vertical).max().unwrap(),
    )
}

fn append_scan(output: &mut Vec<u8>, ids: &[u8], spectral: [u8; 3], entropy: Vec<u8>) {
    let mut scan = vec![ids.len() as u8];
    for &id in ids {
        scan.extend_from_slice(&[id, 0]);
    }
    scan.extend_from_slice(&spectral);
    segment(output, 0xda, &scan);
    assert!(output.len() + entropy.len() <= MAX_FIXTURE_BYTES);
    output.extend_from_slice(&entropy);
}

fn initial_interleaved(spec: Spec, components: &[Component], dc_only: bool) -> Vec<u8> {
    let (max_h, max_v) = max_sampling(components);
    let columns = spec.width.div_ceil(8 * max_h);
    let rows = spec.height.div_ceil(8 * max_v);
    let mut entropy = Entropy::default();
    let mut previous = [0i16; 3];
    for mcu in 0..columns * rows {
        if spec.restart_interval != 0 && mcu != 0 && mcu.is_multiple_of(spec.restart_interval) {
            entropy.restart(mcu / spec.restart_interval - 1);
            previous.fill(0);
        }
        for (c, component) in components.iter().enumerate() {
            for dy in 0..component.vertical {
                for dx in 0..component.horizontal {
                    let x = (mcu % columns) * component.horizontal + dx;
                    let y = (mcu / columns) * component.vertical + dy;
                    let block = block_values(c, x, y);
                    entropy.dc(block[0] - previous[c]);
                    previous[c] = block[0];
                    if !dc_only {
                        entropy.ac_block(&block);
                    }
                }
            }
        }
    }
    entropy.finish()
}

fn initial_ac(spec: Spec, components: &[Component], channel: usize) -> Vec<u8> {
    let (max_h, max_v) = max_sampling(components);
    let component = components[channel];
    // Noninterleaved AC visits real component blocks, excluding dummy MCU blocks.
    let columns = (spec.width * component.horizontal).div_ceil(8 * max_h);
    let rows = (spec.height * component.vertical).div_ceil(8 * max_v);
    let mut entropy = Entropy::default();
    for block in 0..columns * rows {
        if spec.restart_interval != 0 && block != 0 && block.is_multiple_of(spec.restart_interval) {
            entropy.restart(block / spec.restart_interval - 1);
        }
        entropy.ac_block(&block_values(channel, block % columns, block / columns));
    }
    entropy.finish()
}

fn append_refinement_scans(output: &mut Vec<u8>) {
    // Final DCs: -3, 6, 3. First DC scan carries -4, 6, 2; refinement is OR,
    // including the negative DC. AC refinement instead increases magnitude.
    let mut dc = Entropy::default();
    for difference in [-2, 5, -2] {
        dc.dc(difference);
    }
    append_scan(output, &[1], [0, 0, 1], dc.finish());
    let mut ac = Entropy::default();
    ac.ac_symbol(1);
    ac.word(1, 1); // +2 at rank 1.
    ac.ac_symbol(0x11);
    ac.word(0, 1); // -2 at rank 3.
    for _ in 0..3 {
        ac.ac_symbol(0);
    }
    append_scan(output, &[1], [1, 63, 1], ac.finish());
    let mut dc_refinement = Entropy::default();
    dc_refinement.word(0b101, 3);
    append_scan(output, &[1], [0, 0, 0x10], dc_refinement.finish());
    let mut ac_refinement = Entropy::default();
    ac_refinement.ac_symbol(1);
    ac_refinement.word(1, 1); // New +1 at rank 2.
    ac_refinement.word(1, 1); // Existing rank 1 becomes +3 while locating rank 2.
    ac_refinement.ac_symbol(0);
    ac_refinement.word(1, 1); // Existing rank 3 becomes -3.
    ac_refinement.ac_symbol(0); // Empty block 1.
    ac_refinement.ac_symbol(0); // Empty block 2.
    append_scan(output, &[1], [1, 63, 0x10], ac_refinement.finish());
}

fn build(spec: Spec) -> Vec<u8> {
    let components = spec.sampling.components();
    let mut output = header(spec, &components);
    let ids: Vec<u8> = components.iter().map(|c| c.id).collect();
    match spec.process {
        Process::Sequential => append_scan(
            &mut output,
            &ids,
            [0, 63, 0],
            initial_interleaved(spec, &components, false),
        ),
        Process::Progressive => {
            append_scan(
                &mut output,
                &ids,
                [0, 0, 0],
                initial_interleaved(spec, &components, true),
            );
            for (c, component) in components.iter().enumerate() {
                append_scan(
                    &mut output,
                    &[component.id],
                    [1, 63, 0],
                    initial_ac(spec, &components, c),
                );
            }
        }
        Process::Refinement => {
            assert_eq!(
                (spec.width, spec.height, spec.sampling),
                (24, 8, Sampling::Gray)
            );
            assert_eq!(spec.restart_interval, 0);
            append_refinement_scans(&mut output);
        }
    }
    output.extend_from_slice(&[0xff, 0xd9]);
    assert!(output.len() <= MAX_FIXTURE_BYTES);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::{
        coefficient_reader::read_coefficients,
        data::{JpegRecord, NATURAL_ORDER},
        scan_writer::{WriteLimits, reconstruct_from_coefficients},
        transcode_layout::{JpegColorTransform, color_transform_from_jpeg},
    };

    #[test]
    fn fixture_names_bytes_dimensions_and_mode_coverage_are_stable() {
        let first = fixtures();
        assert_eq!(first, fixtures());
        assert_eq!(first.len(), 19);
        let names: std::collections::BTreeSet<_> = first.iter().map(|(name, _)| name).collect();
        assert_eq!(names.len(), first.len());
        for ((name, bytes), spec) in first.iter().zip(SPECS) {
            assert_eq!(name, spec.name);
            assert!(bytes.len() <= MAX_FIXTURE_BYTES);
            assert_eq!(&bytes[..2], &[0xff, 0xd8]);
            assert_eq!(&bytes[bytes.len() - 2..], &[0xff, 0xd9]);
            assert!(spec.width <= 600 && spec.height <= 400);
        }
        for sampling in [
            Sampling::Gray,
            Sampling::Rgb,
            Sampling::Yuv444,
            Sampling::Yuv420,
            Sampling::Yuv422,
            Sampling::Yuv440,
        ] {
            assert!(SPECS.iter().any(|s| s.sampling == sampling));
        }
        assert!(SPECS.iter().any(|s| s.width == 256 && s.height == 256));
        assert!(SPECS.iter().any(|s| s.width > 256 && s.height > 256));
    }

    #[test]
    fn all_fixture_scans_decode_known_coefficients_and_reconstruct_without_cached_entropy() {
        let mut wrapped_restart = false;
        for spec in SPECS {
            let bytes = build(spec);
            let mut jpeg =
                read_coefficients(&bytes).unwrap_or_else(|e| panic!("{}: {e}", spec.name));
            assert_eq!(
                (jpeg.width as usize, jpeg.height as usize),
                (spec.width, spec.height)
            );
            assert!(!jpeg.has_zero_padding_bit, "{}", spec.name);
            let color = color_transform_from_jpeg(&jpeg).unwrap();
            assert_eq!(
                color,
                if spec.sampling == Sampling::Rgb {
                    JpegColorTransform::None
                } else {
                    JpegColorTransform::YCbCr
                }
            );
            let components = spec.sampling.components();
            let (max_h, max_v) = max_sampling(&components);
            for (c, (decoded, component)) in jpeg.components.iter().zip(&components).enumerate() {
                let width = spec.width.div_ceil(8 * max_h) * component.horizontal;
                let height = spec.height.div_ceil(8 * max_v) * component.vertical;
                assert_eq!(
                    (
                        decoded.width_in_blocks as usize,
                        decoded.height_in_blocks as usize
                    ),
                    (width, height)
                );
                let actual_width = (spec.width * component.horizontal).div_ceil(8 * max_h);
                let actual_height = (spec.height * component.vertical).div_ceil(8 * max_v);
                for (index, actual) in decoded.coefficients.chunks_exact(64).enumerate() {
                    let (x, y) = (index % width, index / width);
                    let mut expected = block_values(c, x, y);
                    if spec.process == Process::Progressive
                        && (x >= actual_width || y >= actual_height)
                    {
                        expected[1..].fill(0);
                    }
                    if spec.process == Process::Refinement {
                        expected.fill(0);
                        expected[0] = [-3, 6, 3][index];
                        if index == 0 {
                            expected[1..4].copy_from_slice(&[3, 1, -3]);
                        }
                    }
                    for (rank, &value) in expected.iter().enumerate() {
                        assert_eq!(
                            actual[NATURAL_ORDER[rank]], value,
                            "{} c={c} block={index} rank={rank}",
                            spec.name
                        );
                    }
                }
            }
            assert!(
                jpeg.components
                    .iter()
                    .any(|c| c.coefficients.iter().any(|&v| v > 0))
            );
            assert!(
                jpeg.components
                    .iter()
                    .any(|c| c.coefficients.iter().any(|&v| v < 0))
            );
            for record in &mut jpeg.records {
                if let JpegRecord::Entropy(entropy) = record {
                    let scan = &jpeg.scan_info[entropy.scan_index];
                    let mcus = scan.mcu_cols as usize * scan.mcu_rows as usize;
                    let restarts = (mcus - 1).checked_div(spec.restart_interval).unwrap_or(0);
                    assert_eq!(entropy.restart_markers.len(), restarts, "{}", spec.name);
                    for (i, restart) in entropy.restart_markers.iter().enumerate() {
                        assert_eq!(restart.marker, 0xd0 + (i & 7) as u8);
                        wrapped_restart |= i >= 8;
                    }
                    entropy.bytes.clear();
                }
            }
            let reconstructed = reconstruct_from_coefficients(&jpeg, &WriteLimits::default())
                .unwrap_or_else(|e| panic!("{}: {e}", spec.name));
            assert_eq!(reconstructed, bytes, "{}", spec.name);
        }
        assert!(wrapped_restart);
    }

    #[test]
    fn entropy_ff_stuffing_and_all_one_padding_are_explicit() {
        let mut entropy = Entropy::default();
        entropy.word(0xff, 8);
        entropy.word(0b101, 3);
        entropy.restart(9);
        entropy.word(0, 1);
        assert_eq!(entropy.finish(), [0xff, 0, 0xbf, 0xff, 0xd1, 0x7f]);
    }

    #[test]
    fn targeted_wide_fixture_crosses_dc_groups_without_large_buffers() {
        use crate::jpeg::{frame_header::FrameGeometry, transcode_layout::TranscodeLayout};
        let cases = tier_fixtures();
        assert_eq!(cases.len(), 2);
        for (name, bytes) in cases {
            assert!(bytes.len() < 128 * 1024);
            let mut jpeg = read_coefficients(&bytes).unwrap();
            if name.starts_with("tier_") {
                let layout = TranscodeLayout::new(&jpeg).unwrap();
                let geometry =
                    FrameGeometry::new(jpeg.width, jpeg.height, layout.subsampling()).unwrap();
                assert_eq!(geometry.dc_group_count(), 2);
                assert_eq!(geometry.group_count(), 9);
                assert!(jpeg.scan_info.len() > 1);
                for (c, component) in jpeg.components.iter().enumerate() {
                    for (i, actual) in component.coefficients.chunks_exact(64).enumerate() {
                        let expected = block_values(
                            c,
                            i % component.width_in_blocks as usize,
                            i / component.width_in_blocks as usize,
                        );
                        for (rank, value) in expected.into_iter().enumerate() {
                            assert_eq!(actual[NATURAL_ORDER[rank]], value);
                        }
                    }
                }
            }
            for record in &mut jpeg.records {
                if let JpegRecord::Entropy(entropy) = record {
                    entropy.bytes.clear();
                }
            }
            assert_eq!(
                reconstruct_from_coefficients(&jpeg, &WriteLimits::default()).unwrap(),
                bytes,
                "{name}"
            );
        }
    }

    #[test]
    fn exif_contains_real_ifd_entries_in_both_byte_orders() {
        for little in [false, true] {
            let bytes = exif(little, 6);
            assert_eq!(&bytes[..6], b"Exif\0\0");
            let tiff = &bytes[6..];
            let word = |offset: usize| {
                let bytes = tiff[offset..offset + 2].try_into().unwrap();
                if little {
                    u16::from_le_bytes(bytes)
                } else {
                    u16::from_be_bytes(bytes)
                }
            };
            let long = |offset: usize| {
                let bytes = tiff[offset..offset + 4].try_into().unwrap();
                if little {
                    u32::from_le_bytes(bytes)
                } else {
                    u32::from_be_bytes(bytes)
                }
            };
            assert_eq!(&tiff[..2], if little { b"II" } else { b"MM" });
            assert_eq!(word(2), 42);
            assert_eq!(long(4), 8);
            assert_eq!(word(8), 2);
            assert_eq!((word(10), word(12), long(18)), (0x010e, 2, 38));
            assert_eq!((word(22), word(24), long(26), word(30)), (0x0112, 3, 1, 6));
            assert_eq!(long(34), 0);
            assert_eq!(tiff.len(), 38 + long(14) as usize);
            assert_eq!(tiff.last(), Some(&0));
        }
    }

    #[test]
    fn tagged_profiles_and_metadata_variants_preserve_raw_structure() {
        use crate::{
            icc::{IccColorSpace, validate_profile},
            jpeg::color_metadata::color_encoding,
        };
        for gray in [false, true] {
            let profile = tagged_icc(gray);
            assert_eq!(
                validate_profile(&profile).unwrap(),
                if gray {
                    IccColorSpace::Gray
                } else {
                    IccColorSpace::Rgb
                }
            );
            assert!(profile.len() > 300);
            assert_eq!(
                u32::from_be_bytes(profile[128..132].try_into().unwrap()),
                if gray { 4 } else { 9 }
            );
        }
        let cases = metadata_fixtures();
        assert_eq!(cases.len(), 5);
        for (name, bytes) in cases {
            assert!(bytes.len() < 128 * 1024);
            let mut jpeg = read_coefficients(&bytes).unwrap();
            let color = color_encoding(&jpeg, 4096).unwrap();
            if name.starts_with("metadata_rgb_icc") || name.starts_with("metadata_gray_icc") {
                assert_eq!(
                    color.icc_profile(),
                    Some(tagged_icc(jpeg.components.len() == 1).as_slice())
                );
            } else {
                assert!(color.icc_profile().is_none());
            }
            if name.contains("split") {
                let order: Vec<_> = jpeg
                    .app_data
                    .iter()
                    .filter(|app| app[0] == 0xe2)
                    .map(|app| app[15])
                    .collect();
                assert_eq!(order, [3, 1, 2]);
            }
            if name.contains("duplicate") {
                assert_eq!(jpeg.app_data.iter().filter(|app| app[0] == 0xe7).count(), 3);
                assert_eq!(jpeg.app_data.iter().filter(|app| app[0] == 0xe1).count(), 3);
                assert_eq!(jpeg.com_data.len(), 3);
                assert_eq!(jpeg.inter_marker_data, [vec![0x21, 0xff, 0, 0x22]]);
                assert!(
                    jpeg.records
                        .iter()
                        .any(|r| matches!(r, JpegRecord::Fill { count: 2, .. }))
                );
                assert!(!jpeg.tail_data.is_empty());
            }
            if name.contains("maximum") {
                assert!(
                    jpeg.app_data
                        .iter()
                        .any(|app| app[0] == 0xef && app.len() == 65536)
                );
            }
            assert_eq!(jpeg.reconstruct().unwrap(), bytes);
            for record in &mut jpeg.records {
                if let JpegRecord::Entropy(entropy) = record {
                    entropy.bytes.clear();
                }
            }
            assert_eq!(
                reconstruct_from_coefficients(&jpeg, &WriteLimits::default()).unwrap(),
                bytes,
                "{name}"
            );
        }
    }
}
