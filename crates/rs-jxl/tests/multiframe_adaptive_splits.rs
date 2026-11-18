//! Additional axis-adaptive geometry must earn its place using complete bytes.
use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::{encode::FixedTree, predictor::Predictor};
use rs_jxl::multiframe::{AutoOptions, Reference, Strategy, encode_auto};
use rs_jxl::{EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout};
const SIDE: usize = 128;
const TICKS: [u32; 3] = [2, 3, 5];
fn animated(image: Image, extras: Vec<PixelData>) -> Image {
    let mut m = image.metadata().clone();
    m.animation = Some(Animation::default());
    image.with_metadata(m, extras).unwrap()
}
fn fixture() -> Vec<Image> {
    let mut random = 0x793a5621u32;
    let pixels: Vec<u8> = (0..SIDE * SIDE * 3)
        .map(|_| {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random as u8
        })
        .collect();
    let mut changed = pixels.clone();
    for y in (1..SIDE - 2).step_by(16) {
        for x in (1..SIDE - 2).step_by(16) {
            for dy in 0..3 {
                for dx in 0..3 {
                    changed[((y + dy) * SIDE + x + dx) * 3] ^= 127;
                }
            }
        }
    }
    [vec![0; pixels.len()], pixels, changed]
        .into_iter()
        .map(|p| {
            animated(
                Image::new(SIDE as u32, SIDE as u32, PixelLayout::Rgb, PixelData::U8(p)).unwrap(),
                vec![],
            )
        })
        .collect()
}
fn options(threads: usize) -> EncodeOptions {
    EncodeOptions {
        threads,
        modular: ModularOptions {
            rct: None,
            lz77: false,
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn auto(enabled: bool, prune: bool) -> AutoOptions {
    AutoOptions {
        adaptive_splits: enabled,
        allow_add_delta: false,
        reference: Reference::Previous,
        allow_mask: false,
        tile_size: 16,
        max_rectangles: 16,
        prune,
        ..Default::default()
    }
}
fn split(s: Strategy) -> bool {
    matches!(
        s,
        Strategy::SplitPrevious | Strategy::SplitFirst | Strategy::SplitRetained
    )
}
fn exact(a: &PixelData, b: &PixelData) {
    match (a, b) {
        (PixelData::F32(a), PixelData::F32(b)) => assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        ),
        _ => assert_eq!(a, b),
    }
}
fn verify(bytes: &[u8], images: &[Image]) {
    let d = decode(bytes, &DecodeOptions::default()).unwrap();
    assert!(d.complete);
    assert_eq!(d.frames.len(), 3);
    for ((f, image), ticks) in d.frames.iter().zip(images).zip(TICKS) {
        assert!(f.complete);
        assert_eq!(f.duration_ticks, ticks);
        assert_eq!((f.width, f.height, f.layout), (SIDE, SIDE, image.layout()));
        exact(&f.pixels, image.pixels());
        if image.layout().has_alpha() {
            assert_eq!(f.extra_planes.len(), 1);
            match image.pixels() {
                PixelData::U8(v) => assert_eq!(
                    f.extra_planes[0],
                    PixelData::U8(v.chunks_exact(4).map(|x| x[3]).collect())
                ),
                PixelData::U16(v) => assert_eq!(
                    f.extra_planes[0],
                    PixelData::U16(v.chunks_exact(4).map(|x| x[3]).collect())
                ),
                _ => unreachable!(),
            }
        } else {
            assert_eq!(f.extra_planes.len(), image.extra_planes().len());
            for (a, b) in f.extra_planes.iter().zip(image.extra_planes()) {
                exact(a, b);
            }
        }
    }
}
#[test]
fn axis_splits_win_complete_bytes_and_preserve_threads_pruning_references() {
    let images = fixture();
    let old = encode_auto(&images, &TICKS, &options(1), &auto(false, false)).unwrap();
    assert!(
        old.frames
            .iter()
            .flat_map(|f| &f.candidates)
            .all(|c| !split(c.strategy))
    );
    let new = encode_auto(&images, &TICKS, &options(1), &auto(true, false)).unwrap();
    let winner = &new.frames[2].candidates[new.frames[2].winner_index];
    assert_eq!(winner.strategy, Strategy::SplitPrevious);
    assert_eq!(
        (
            winner.reference_index,
            winner.source,
            winner.save_as_reference
        ),
        (Some(1), 2, 0)
    );
    assert!(
        new.bytes.len() < old.bytes.len(),
        "new={} old={}",
        new.bytes.len(),
        old.bytes.len()
    );
    assert!(winner.encoded_frames > 1 && winner.encoded_frames <= 16);
    verify(&old.bytes, &images);
    verify(&new.bytes, &images);
    for (threads, prune) in [(1, true), (4, false), (4, true)] {
        let result = encode_auto(&images, &TICKS, &options(threads), &auto(true, prune)).unwrap();
        assert_eq!(result.bytes, new.bytes, "threads={threads} prune={prune}");
        verify(&result.bytes, &images);
    }
    // No-prune reports follow canonical execution order. Reach the added split
    // trial, then fail inside its submitted work rather than on frame zero.
    let mut previous = 0;
    let work = new
        .frames
        .iter()
        .flat_map(|f| &f.candidates)
        .find_map(|c| {
            if split(c.strategy) {
                Some(c.trial_pixels)
            } else {
                previous += c.trial_pixels;
                None
            }
        })
        .unwrap();
    let mut limited = auto(true, false);
    limited.max_trial_pixels = previous + work - 1;
    assert!(matches!(
        encode_auto(&images, &TICKS, &options(4), &limited),
        Err(Error::Limit("automatic trial pixel work"))
    ));
}
#[test]
fn geometry_preserves_hidden_wide_float_words_and_depth_extra() {
    let rgb = fixture();
    for kind in 0..4 {
        let images: Vec<_> = rgb
            .iter()
            .map(|image| {
                let PixelData::U8(values) = image.pixels() else {
                    unreachable!()
                };
                match kind {
                    0 => animated(
                        Image::new(
                            SIDE as u32,
                            SIDE as u32,
                            PixelLayout::Rgba,
                            PixelData::U8(
                                values
                                    .chunks_exact(3)
                                    .flat_map(|p| [p[0], p[1], p[2], 0])
                                    .collect(),
                            ),
                        )
                        .unwrap(),
                        vec![],
                    ),
                    1 => animated(
                        Image::new(
                            SIDE as u32,
                            SIDE as u32,
                            PixelLayout::Rgba,
                            PixelData::U16(
                                values
                                    .chunks_exact(3)
                                    .flat_map(|p| {
                                        [p[0] as u16 * 257, p[1] as u16 * 257, p[2] as u16 * 257, 0]
                                    })
                                    .collect(),
                            ),
                        )
                        .unwrap(),
                        vec![],
                    ),
                    2 => {
                        let words = [
                            0, 0x80000000, 0x7fc12345, 0x7f800000, 0xff800000, 1, 0x3f000001,
                        ];
                        animated(
                            Image::new(
                                SIDE as u32,
                                SIDE as u32,
                                PixelLayout::Gray,
                                PixelData::F32(
                                    values
                                        .chunks_exact(3)
                                        .map(|p| f32::from_bits(words[p[0] as usize % words.len()]))
                                        .collect(),
                                ),
                            )
                            .unwrap(),
                            vec![],
                        )
                    }
                    _ => {
                        let image = Image::new(
                            SIDE as u32,
                            SIDE as u32,
                            PixelLayout::Gray,
                            PixelData::U32(
                                values
                                    .chunks_exact(3)
                                    .map(|p| {
                                        (u32::from(p[0]) << 23)
                                            | (u32::from(p[1]) << 8)
                                            | u32::from(p[2])
                                    })
                                    .collect(),
                            ),
                        )
                        .unwrap();
                        let mut metadata = image.metadata().clone();
                        metadata.animation = Some(Animation::default());
                        metadata.bit_depth = BitDepth::new(31).unwrap();
                        metadata
                            .extra_channels
                            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
                        image
                            .with_metadata(
                                metadata,
                                vec![PixelData::U16(
                                    values
                                        .chunks_exact(3)
                                        .map(|p| u16::from(p[1]) * 257)
                                        .collect(),
                                )],
                            )
                            .unwrap()
                    }
                }
            })
            .collect();
        let result = encode_auto(&images, &TICKS, &options(1), &auto(true, false)).unwrap();
        assert!(
            result
                .frames
                .iter()
                .flat_map(|f| &f.candidates)
                .any(|c| split(c.strategy)),
            "format {kind} needs actual Split trial"
        );
        verify(&result.bytes, &images);
    }
}
