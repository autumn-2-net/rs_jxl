//! Per-decoder output dithering through the production Rust decoder.
use rs_jxl::decode::{DecodeOptions, Decoder, DitherMode, SampleFormat, decode, reconstruct_jpeg};
use rs_jxl::{Image, PixelData, PixelLayout};

fn image(layout: PixelLayout) -> Image {
    // Fractional U8 values after normalization exercise the selected pattern.
    Image::new(
        35,
        33,
        layout,
        PixelData::U16(vec![32960; 35 * 33 * layout.channels()]),
    )
    .unwrap()
}

fn options(dither: DitherMode, format: SampleFormat) -> DecodeOptions {
    DecodeOptions {
        dither,
        format,
        ..DecodeOptions::default()
    }
}

#[test]
fn concurrent_decoder_modes_are_independent_and_default_is_upstream() {
    let bytes = rs_jxl::encode(&image(PixelLayout::Rgb), &Default::default()).unwrap();
    let modes = [DitherMode::JxlRs, DitherMode::Libjxl];
    let expected = modes.map(|mode| {
        decode(&bytes, &options(mode, SampleFormat::U8))
            .unwrap()
            .frames
            .remove(0)
            .pixels
    });
    assert_ne!(expected[0], expected[1]);
    let default = DecodeOptions {
        format: SampleFormat::U8,
        ..DecodeOptions::default()
    };
    assert_eq!(
        decode(&bytes, &default).unwrap().frames[0].pixels,
        expected[0]
    );
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        for (mode, expected) in modes.into_iter().zip(&expected) {
            let bytes = &bytes;
            let barrier = &barrier;
            scope.spawn(move || {
                let mut decoder = Decoder::new(options(mode, SampleFormat::U8)).unwrap();
                barrier.wait();
                for _ in 0..3 {
                    assert_eq!(&decoder.decode(bytes).unwrap().frames[0].pixels, expected);
                    assert_eq!(decoder.options().dither, mode);
                }
            });
        }
    });
}

#[test]
fn native_integer_and_float_output_ignore_dither_mode() {
    let source = image(PixelLayout::Rgba);
    let bytes = rs_jxl::encode(&source, &Default::default()).unwrap();
    for format in [SampleFormat::Native, SampleFormat::U16, SampleFormat::F32] {
        let a = decode(&bytes, &options(DitherMode::JxlRs, format)).unwrap();
        let b = decode(&bytes, &options(DitherMode::Libjxl, format)).unwrap();
        assert_eq!(a.frames[0].pixels, b.frames[0].pixels);
        assert_eq!(a.frames[0].extra_planes, b.frames[0].extra_planes);
        if format == SampleFormat::Native {
            assert_eq!(&a.frames[0].pixels, source.pixels());
        }
    }
}

#[test]
fn dither_modes_follow_source_coordinates_for_all_orientations_and_extra_output() {
    use rs_jxl::metadata::Orientation;

    for orientation in 1..=8 {
        let source = image(PixelLayout::Rgba);
        let mut metadata = source.metadata().clone();
        metadata.orientation = Orientation::try_from(orientation).unwrap();
        let source = source.with_metadata(metadata, Vec::new()).unwrap();
        let bytes = rs_jxl::encode(&source, &Default::default()).unwrap();
        let mut mode_outputs = Vec::new();
        for mode in [DitherMode::JxlRs, DitherMode::Libjxl] {
            let mut opts = options(mode, SampleFormat::U8);
            opts.adjust_orientation = false;
            let raw = decode(&bytes, &opts).unwrap();
            opts.adjust_orientation = true;
            let oriented = decode(&bytes, &opts).unwrap();
            let a = &raw.frames[0];
            let b = &oriented.frames[0];
            let (PixelData::U8(raw_pixels), PixelData::U8(pixels)) = (&a.pixels, &b.pixels) else {
                panic!("expected U8 output");
            };
            let (PixelData::U8(raw_alpha), PixelData::U8(alpha)) =
                (&a.extra_planes[0], &b.extra_planes[0])
            else {
                panic!("expected U8 alpha output");
            };
            for y in 0..a.height {
                for x in 0..a.width {
                    let (dx, dy) = match orientation {
                        1 => (x, y),
                        2 => (a.width - 1 - x, y),
                        3 => (a.width - 1 - x, a.height - 1 - y),
                        4 => (x, a.height - 1 - y),
                        5 => (y, x),
                        6 => (a.height - 1 - y, x),
                        7 => (a.height - 1 - y, a.width - 1 - x),
                        8 => (y, a.width - 1 - x),
                        _ => unreachable!(),
                    };
                    let src = y * a.width + x;
                    let dst = dy * b.width + dx;
                    assert_eq!(
                        &raw_pixels[src * 4..src * 4 + 4],
                        &pixels[dst * 4..dst * 4 + 4]
                    );
                    assert_eq!(raw_alpha[src], alpha[dst]);
                    assert_eq!(pixels[dst * 4 + 3], alpha[dst]);
                }
            }
            mode_outputs.push(b.extra_planes[0].clone());
        }
        assert_ne!(mode_outputs[0], mode_outputs[1]);
    }
}

fn jpeg_fixture() -> Vec<u8> {
    fn marker(bytes: &mut Vec<u8>, code: u8, payload: &[u8]) {
        bytes.extend([0xff, code]);
        bytes.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        bytes.extend_from_slice(payload);
    }
    let mut bytes = vec![0xff, 0xd8];
    marker(
        &mut bytes,
        0xe1,
        b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta>dither</x:xmpmeta>",
    );
    let mut quant = vec![0];
    quant.extend([1; 64]);
    marker(&mut bytes, 0xdb, &quant);
    marker(&mut bytes, 0xc0, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
    let mut huffman = Vec::new();
    for selector in [0, 16] {
        huffman.extend([selector, 1]);
        huffman.extend([0; 15]);
        huffman.push(0);
    }
    marker(&mut bytes, 0xc4, &huffman);
    marker(&mut bytes, 0xda, &[1, 1, 0, 0, 63, 0]);
    bytes.extend([0x2a, 0xff, 0xd9]);
    bytes
}

#[test]
fn jpeg_reconstruction_and_metadata_ignore_dither_mode() {
    let original = jpeg_fixture();
    let bytes = rs_jxl::jpeg::transcode(&original, &Default::default()).unwrap();
    for mode in [DitherMode::JxlRs, DitherMode::Libjxl] {
        let opts = options(mode, SampleFormat::U8);
        assert_eq!(reconstruct_jpeg(&bytes, &opts).unwrap(), original);
        let mut decoder = Decoder::new(opts.clone()).unwrap();
        assert_eq!(decoder.reconstruct_jpeg(&bytes).unwrap(), original);
        let inspected = rs_jxl::decode::inspect(&bytes, &opts).unwrap();
        let xmp = inspected.boxes.iter().find(|b| b.kind == *b"xml ").unwrap();
        assert_eq!(xmp.data, b"<x:xmpmeta>dither</x:xmpmeta>");
    }
}
