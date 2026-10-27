use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::metadata::Animation;
use rs_jxl::multiframe::{AnalysisOptions, AutoOptions, Reference, analyze, encode_auto};
use rs_jxl::{EncodeOptions, Error, Image, PixelData, PixelLayout};

fn fixture() -> Vec<Image> {
    let mut state = 0x1234_5678u32;
    let base: Vec<u8> = (0..64 * 48 * 4)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if index % 4 == 3 { 0 } else { state as u8 }
        })
        .collect();
    (0..3)
        .map(|i| {
            let mut pixels = base.clone();
            for (x, y) in [(1, 1), (25, 25)] {
                pixels[(y * 64 + x) * 4] ^= i * 37;
            }
            let image = Image::new(64, 48, PixelLayout::Rgba, PixelData::U8(pixels)).unwrap();
            let mut metadata = image.metadata().clone();
            metadata.animation = Some(Animation::default());
            image.with_metadata(metadata, vec![]).unwrap()
        })
        .collect()
}

#[test]
fn finer_partition_reduces_bytes_without_losing_invisible_rgb_or_timing() {
    let images = fixture();
    let options = EncodeOptions::default();
    let base = AutoOptions {
        reference: Reference::Previous,
        ..Default::default()
    };
    let original = encode_auto(&images, &[2, 3, 4], &options, &base).unwrap();
    let mut selected_size = None;
    for prune in [false, true] {
        let auto = AutoOptions {
            extra_tile_sizes: vec![8, 16, 64],
            prune,
            ..base.clone()
        };
        let encoded = encode_auto(&images, &[2, 3, 4], &options, &auto).unwrap();
        assert!(encoded.bytes.len() < original.bytes.len());
        if let Some(size) = selected_size {
            assert_eq!(encoded.bytes.len(), size);
        }
        selected_size = Some(encoded.bytes.len());
        assert!(
            encoded.frames[1].candidates[encoded.frames[1].winner_index]
                .tile_size
                .is_some()
        );
        let decoded = decode(&encoded.bytes, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.frames.len(), 3);
        for (index, frame) in decoded.frames.iter().enumerate() {
            assert_eq!(&frame.pixels, images[index].pixels());
            assert_eq!(frame.duration_ticks, index as u32 + 2);
        }
    }
}

#[test]
fn duplicate_grids_and_equivalent_partitions_do_not_inflate_trial_count() {
    let images = fixture();
    let run = |extra_tile_sizes| {
        analyze(
            &images,
            &AnalysisOptions {
                extra_tile_sizes,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let one = run(vec![8]);
    let many = run(vec![32, 8, 16, 8, 32, 64]);
    for (one, many) in one.frames.iter().zip(&many.frames) {
        assert_eq!(one.candidates.len(), many.candidates.len());
        for (a, b) in one.candidates.iter().zip(&many.candidates) {
            assert_eq!(a.rectangles, b.rectangles);
            assert_eq!(a.tile_size, b.tile_size);
        }
    }
}

#[test]
fn no_multi_rect_and_full_modes_preserve_controls() {
    let images = fixture();
    for (reference, allow_multi_rect) in [(Reference::Full, true), (Reference::Previous, false)] {
        let report = analyze(
            &images,
            &AnalysisOptions {
                reference,
                allow_multi_rect,
                extra_tile_sizes: vec![1, 8, 64],
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            report
                .frames
                .iter()
                .flat_map(|f| &f.candidates)
                .all(|c| c.tile_size.is_none())
        );
    }
}

#[test]
fn invalid_scales_and_explicit_trial_limits_fail_instead_of_silent_fallback() {
    let images = fixture();
    let bad = AutoOptions {
        extra_tile_sizes: vec![0],
        ..Default::default()
    };
    assert!(matches!(
        encode_auto(&images, &[1; 3], &EncodeOptions::default(), &bad),
        Err(Error::InvalidInput(_))
    ));
    let limited = AutoOptions {
        extra_tile_sizes: vec![8],
        max_total_trials: 1,
        ..Default::default()
    };
    assert!(matches!(
        encode_auto(&images, &[1; 3], &EncodeOptions::default(), &limited),
        Err(Error::Limit(_))
    ));
}
