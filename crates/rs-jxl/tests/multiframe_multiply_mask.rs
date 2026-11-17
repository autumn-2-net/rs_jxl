//! Complete-stream acceptance for the ordinary Multiply + Add mask candidate.
//! Fixed tiny entropy tools isolate composition, reference state, and byte selection.

use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::{encode::FixedTree, predictor::Predictor};
use rs_jxl::multiframe::{AutoOptions, Reference, Strategy, encode_auto};
use rs_jxl::{EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout};

fn options(threads: usize) -> EncodeOptions {
    EncodeOptions {
        threads,
        max_output_bytes: 2 << 20,
        modular: ModularOptions {
            rct: None,
            lz77: false,
            entropy: Coding::Prefix,
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            tree_samples: 1,
            max_tree_leaves: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn auto(enabled: bool, prune: bool) -> AutoOptions {
    AutoOptions {
        reference: Reference::Previous,
        allow_mask: true,
        allow_multiply_mask: enabled,
        allow_multi_rect: false,
        prune,
        ..Default::default()
    }
}

fn multiply(strategy: Strategy) -> bool {
    matches!(
        strategy,
        Strategy::MultiplyMaskPrevious
            | Strategy::MultiplyMaskFirst
            | Strategy::MultiplyMaskRetained
    )
}

fn words(data: &PixelData) -> Vec<u32> {
    match data {
        PixelData::U8(v) => v.iter().map(|&v| u32::from(v)).collect(),
        PixelData::U16(v) => v.iter().map(|&v| u32::from(v)).collect(),
        PixelData::U32(v) => v.clone(),
        PixelData::F32(v) => v.iter().map(|v| v.to_bits()).collect(),
        _ => unreachable!(),
    }
}

fn samples(values: Vec<u32>, wide: bool) -> PixelData {
    if wide {
        PixelData::U16(values.into_iter().map(|v| v as u16).collect())
    } else {
        PixelData::U8(values.into_iter().map(|v| v as u8).collect())
    }
}

fn source(layout: PixelLayout, bits: u8, alpha_bits: u8, w: u32, h: u32, seed: u32) -> Image {
    let channels = layout.channels();
    let colors = if channels < 3 { 1 } else { 3 };
    let maximum = (1u32 << bits) - 1;
    let alpha_maximum = (1u32 << alpha_bits) - 1;
    let mut random = seed;
    let mut values = vec![];
    for pixel in 0..w as usize * h as usize {
        for _ in 0..colors {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            values.push(random & maximum);
        }
        if layout.has_alpha() {
            // Zero alpha retains arbitrary original RGB. Mixed-depth cases use endpoints,
            // so both original alpha and its main-buffer rescaling have exact word oracles.
            values.push(match pixel % 4 {
                0 => 0,
                1 if bits == alpha_bits => alpha_maximum / 2,
                _ => alpha_maximum,
            });
        }
    }
    let image = Image::new(
        w,
        h,
        layout,
        samples(values, bits > 8 || (layout.has_alpha() && alpha_bits > 8)),
    )
    .unwrap();
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    metadata.bit_depth = BitDepth::new(bits).unwrap();
    if layout.has_alpha() {
        metadata.extra_channels[0].bit_depth = BitDepth::new(alpha_bits).unwrap();
    }
    image.with_metadata(metadata, vec![]).unwrap()
}

fn edited(image: &Image, bit: u8, dense: bool) -> Image {
    let mut values = words(image.pixels());
    let channels = image.layout().channels();
    let colors = image.color_channels();
    let w = image.width() as usize;
    let h = image.height() as usize;
    let points = if dense {
        (1..h - 1)
            .step_by(8)
            .flat_map(|y| (1..w - 1).step_by(8).map(move |x| y * w + x))
            .collect()
    } else {
        vec![w + 1, (h - 2) * w + w - 2]
    };
    for pixel in points {
        for c in 0..colors {
            values[pixel * channels + c] ^= 1u32 << bit;
        }
        if image.layout().has_alpha() && !dense {
            let maximum = (1u32 << image.channel_depth(channels - 1).bits_per_sample) - 1;
            values[pixel * channels + channels - 1] ^= maximum;
        }
    }
    let wide = matches!(image.pixels(), PixelData::U16(_));
    Image::new(
        image.width(),
        image.height(),
        image.layout(),
        samples(values, wide),
    )
    .unwrap()
    .with_metadata(image.metadata().clone(), vec![])
    .unwrap()
}

fn verify(bytes: &[u8], images: &[Image], durations: &[u32]) {
    let decoded = decode(
        bytes,
        &DecodeOptions {
            threads: 4,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(decoded.complete);
    let visible: Vec<_> = images
        .iter()
        .zip(durations)
        .enumerate()
        .filter(|(i, (_, ticks))| **ticks != 0 || i + 1 == images.len())
        .collect();
    assert_eq!(decoded.frames.len(), visible.len());
    for (frame, (_, (image, ticks))) in decoded.frames.iter().zip(visible) {
        assert!(frame.complete);
        assert_eq!(
            (frame.width, frame.height, frame.layout),
            (
                image.width() as usize,
                image.height() as usize,
                image.layout()
            )
        );
        assert_eq!(frame.duration_ticks, *ticks);
        let expected = words(image.pixels());
        let actual = words(&frame.pixels);
        let channels = image.layout().channels();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual
            .chunks_exact(channels)
            .zip(expected.chunks_exact(channels))
        {
            assert_eq!(&a[..image.color_channels()], &b[..image.color_channels()]);
        }
        if image.layout().has_alpha() {
            let alpha: Vec<_> = expected
                .chunks_exact(channels)
                .map(|p| p[channels - 1])
                .collect();
            assert_eq!(frame.extra_planes.len(), 1);
            assert_eq!(words(&frame.extra_planes[0]), alpha);
            let color_max = (1u32 << image.bits()) - 1;
            let alpha_bits = image.channel_depth(channels - 1).bits_per_sample;
            let alpha_max = (1u32 << alpha_bits) - 1;
            for (a, original) in actual.chunks_exact(channels).zip(alpha) {
                let main_alpha = if image.bits() == alpha_bits {
                    original
                } else if original == 0 {
                    0
                } else {
                    assert_eq!(original, alpha_max);
                    color_max
                };
                assert_eq!(a[channels - 1], main_alpha);
            }
        } else {
            assert!(frame.extra_planes.is_empty());
        }
    }
}

#[test]
fn two_layer_candidates_win_bytes_and_keep_hidden_samples_threads_and_pruning_exact() {
    for bits in [8, 16] {
        for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
            let base = source(layout, bits, bits, 65, 49, 0x8712_3456);
            let images = vec![base.clone(), edited(&base, 0, false)];
            let old = encode_auto(&images, &[2, 3], &options(1), &auto(false, false)).unwrap();
            assert!(
                old.frames
                    .iter()
                    .flat_map(|f| &f.candidates)
                    .all(|c| !multiply(c.strategy))
            );
            let new = encode_auto(&images, &[2, 3], &options(1), &auto(true, false)).unwrap();
            let winner = &new.frames[1].candidates[new.frames[1].winner_index];
            assert_eq!(winner.strategy, Strategy::MultiplyMaskPrevious);
            assert_eq!(
                (
                    winner.reference_index,
                    winner.source,
                    winner.save_as_reference
                ),
                (Some(0), 1, 0)
            );
            assert_eq!(winner.tile_size, None);
            assert_eq!(winner.encoded_frames, 2);
            let rect = new.frames[1].differences[0].bounds.unwrap();
            assert_eq!(
                winner.trial_pixels,
                2 * u64::from(rect.width) * u64::from(rect.height)
            );
            assert!(new.bytes.len() < old.bytes.len());
            verify(&new.bytes, &images, &[2, 3]);
            if bits == 8 && layout == PixelLayout::Rgb {
                for (threads, prune) in [(1, true), (4, false), (4, true)] {
                    let result =
                        encode_auto(&images, &[2, 3], &options(threads), &auto(true, prune))
                            .unwrap();
                    assert_eq!(result.bytes, new.bytes, "threads={threads} prune={prune}");
                    verify(&result.bytes, &images, &[2, 3]);
                }
                let mut previous_work = 0;
                let first_mask_work = new
                    .frames
                    .iter()
                    .flat_map(|f| &f.candidates)
                    .find_map(|c| {
                        if multiply(c.strategy) {
                            Some(c.trial_pixels)
                        } else {
                            previous_work += c.trial_pixels;
                            None
                        }
                    })
                    .unwrap();
                let mut limited = auto(true, false);
                limited.max_trial_pixels = previous_work + first_mask_work - 1;
                assert!(matches!(
                    encode_auto(&images, &[2, 3], &options(1), &limited),
                    Err(Error::Limit("automatic trial pixel work"))
                ));
                let mut limited = auto(true, false);
                limited.max_total_trials =
                    new.frames.iter().map(|f| f.candidates.len()).sum::<usize>() - 1;
                assert!(matches!(
                    encode_auto(&images, &[2, 3], &options(1), &limited),
                    Err(Error::Limit("automatic trial count"))
                ));
            }
        }
    }
}

#[test]
fn reduced_gray_and_unequal_alpha_depths_survive_a_long_same_slot_chain() {
    for (layout, bits, alpha_bits) in [
        (PixelLayout::Gray, 12, 12),
        (PixelLayout::GrayAlpha, 12, 1),
        (PixelLayout::Rgba, 8, 16),
    ] {
        let base = source(layout, bits, alpha_bits, 65, 49, 0x4123_5678);
        let base = if layout == PixelLayout::GrayAlpha {
            let mut metadata = base.metadata().clone();
            metadata.extra_channels[0].alpha_associated = true;
            base.with_metadata(metadata, vec![]).unwrap()
        } else {
            base
        };
        let images = vec![base.clone(), edited(&base, 0, false)];
        let encoded = encode_auto(&images, &[2, 3], &options(1), &auto(true, false)).unwrap();
        assert!(multiply(
            encoded.frames[1].candidates[encoded.frames[1].winner_index].strategy
        ));
        verify(&encoded.bytes, &images, &[2, 3]);
    }
    let mut images = vec![source(PixelLayout::Rgba, 12, 16, 27, 17, 0x1234_8765)];
    for _ in 1..43 {
        images.push(edited(images.last().unwrap(), 0, false));
    }
    let durations = vec![1; images.len()];
    let encoded = encode_auto(&images, &durations, &options(1), &auto(true, false)).unwrap();
    for (index, report) in encoded.frames.iter().enumerate().skip(1) {
        let winner = &report.candidates[report.winner_index];
        assert_eq!(winner.strategy, Strategy::MultiplyMaskPrevious);
        assert_eq!(winner.encoded_frames, 2);
        if index > 1 {
            assert_eq!(winner.source, 2);
        }
        assert_eq!(
            winner.save_as_reference,
            if index + 1 == images.len() { 0 } else { 2 }
        );
    }
    verify(&encoded.bytes, &images, &durations);
}

#[test]
fn first_and_retained_sources_preserve_zero_duration_state_and_reference_slots() {
    let base = source(PixelLayout::Rgba, 8, 8, 65, 49, 0x3159_2641);
    let images = vec![
        base.clone(),
        edited(&base, 0, false),
        edited(&base, 1, false),
    ];
    let mut first = auto(true, false);
    first.reference = Reference::First;
    let encoded = encode_auto(&images, &[2, 0, 5], &options(1), &first).unwrap();
    for report in encoded.frames.iter().skip(1) {
        let winner = &report.candidates[report.winner_index];
        assert_eq!(winner.strategy, Strategy::MultiplyMaskFirst);
        assert_eq!((winner.reference_index, winner.source), (Some(0), 1));
        assert_eq!(winner.encoded_frames, 2);
        assert_eq!(report.references_before[1], Some(0));
    }
    verify(&encoded.bytes, &images, &[2, 0, 5]);

    let other = source(PixelLayout::Rgba, 8, 8, 65, 49, 0x9271_8234);
    let images = vec![
        base,
        other.clone(),
        edited(&other, 0, true),
        edited(&other, 1, false),
    ];
    let mut retained = auto(true, false);
    retained.reference = Reference::SmallestCrop;
    retained.retained_reference = true;
    let encoded = encode_auto(&images, &[2, 3, 5, 7], &options(1), &retained).unwrap();
    let report = &encoded.frames[3];
    let winner = &report.candidates[report.winner_index];
    assert_eq!(winner.strategy, Strategy::MultiplyMaskRetained);
    assert_eq!(
        (
            winner.reference_index,
            winner.source,
            winner.save_as_reference
        ),
        (Some(1), 2, 0)
    );
    assert_eq!(report.references_before[2], Some(1));
    verify(&encoded.bytes, &images, &[2, 3, 5, 7]);
}

#[test]
fn floating_wide_extra_and_generated_patch_routes_keep_the_existing_candidates() {
    for pixels in [
        PixelData::F32(vec![0.0, -0.0, f32::from_bits(0x7fc1_2345), 0.5, 1.0, 0.25]),
        PixelData::U32(vec![0, 1 << 20, (1 << 31) - 1, 91, 3, 5]),
    ] {
        let base = Image::new(3, 2, PixelLayout::Gray, pixels).unwrap();
        let mut metadata = base.metadata().clone();
        metadata.animation = Some(Animation::default());
        let base = base.with_metadata(metadata, vec![]).unwrap();
        let values = match base.pixels() {
            PixelData::F32(v) => {
                let mut v = v.clone();
                v[3] = 0.75;
                PixelData::F32(v)
            }
            PixelData::U32(v) => {
                let mut v = v.clone();
                v[3] ^= 1;
                PixelData::U32(v)
            }
            _ => unreachable!(),
        };
        let current = Image::new(3, 2, PixelLayout::Gray, values)
            .unwrap()
            .with_metadata(base.metadata().clone(), vec![])
            .unwrap();
        let images = vec![base, current];
        let old = encode_auto(&images, &[2, 3], &options(1), &auto(false, false)).unwrap();
        let new = encode_auto(&images, &[2, 3], &options(1), &auto(true, false)).unwrap();
        assert_eq!(new.bytes, old.bytes);
        assert!(
            new.frames
                .iter()
                .flat_map(|f| &f.candidates)
                .all(|c| !multiply(c.strategy))
        );
        let decoded = decode(&new.bytes, &DecodeOptions::default()).unwrap();
        assert!(decoded.complete);
        assert_eq!(decoded.frames.len(), images.len());
        for (frame, image) in decoded.frames.iter().zip(&images) {
            assert_eq!(frame.pixels, *image.pixels());
        }
    }
    let base = source(PixelLayout::Rgb, 8, 8, 65, 49, 0x8976_5432);
    let images = vec![base.clone(), edited(&base, 0, false)];
    for kind in 0..3 {
        let mut encoding = options(1);
        match kind {
            0 => encoding.patches = Some(Default::default()),
            1 => encoding.tile_patches = Some(Default::default()),
            _ => encoding.source_patches = Some(Default::default()),
        }
        let result = encode_auto(&images, &[2, 3], &encoding, &auto(true, false)).unwrap();
        assert!(
            result
                .frames
                .iter()
                .flat_map(|f| &f.candidates)
                .all(|c| !multiply(c.strategy))
        );
        verify(&result.bytes, &images, &[2, 3]);
    }
    let images: Vec<_> = images
        .into_iter()
        .map(|image| {
            let mut metadata = image.metadata().clone();
            metadata
                .extra_channels
                .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap());
            image
                .with_metadata(metadata, vec![PixelData::U8(vec![19; 65 * 49])])
                .unwrap()
        })
        .collect();
    let result = encode_auto(&images, &[2, 3], &options(1), &auto(true, false)).unwrap();
    assert!(
        result
            .frames
            .iter()
            .flat_map(|f| &f.candidates)
            .all(|c| !multiply(c.strategy))
    );
    let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), images.len());
    for (frame, image) in decoded.frames.iter().zip(images) {
        assert_eq!(frame.pixels, *image.pixels());
        assert_eq!(frame.extra_planes, image.extra_planes());
    }
}
