use rs_jxl::{
    EncodeOptions, Image, PixelData, PixelLayout,
    decode::{DecodeOptions, decode},
    metadata::Animation,
    multiframe::{AutoOptions, Strategy, encode_auto},
    tools::patches::{ImageView, PatchOptions, discover},
};

#[test]
fn dynamic_canvases_survive_generated_patch_atlas_overwrites() {
    let width = 64usize;
    let height = 48usize;
    let patches = PatchOptions::default();
    let mut images = Vec::new();
    for background in [10u8, 60, 110, 60] {
        let mut pixels = vec![background; width * height];
        for x in [8, 32] {
            for y in 10..18 {
                pixels[y * width + x..y * width + x + 8].fill(230);
            }
        }
        if images.len() == 3 {
            pixels[2 * width + 2] ^= 3;
        }
        let planes = vec![pixels.iter().map(|&v| i32::from(v)).collect()];
        assert!(
            discover(
                ImageView {
                    width,
                    height,
                    bit_depth: 8,
                    color_channels: 1,
                    planes: &planes
                },
                &patches,
            )
            .unwrap()
            .is_some()
        );
        let image = Image::new(
            width as u32,
            height as u32,
            PixelLayout::Gray,
            PixelData::U8(pixels),
        )
        .unwrap();
        let mut metadata = image.metadata().clone();
        metadata.animation = Some(Animation::default());
        images.push(image.with_metadata(metadata, vec![]).unwrap());
    }
    let result = encode_auto(
        &images,
        &[1, 2, 3, 4],
        &EncodeOptions {
            patches: Some(patches),
            ..Default::default()
        },
        &AutoOptions {
            retained_reference: true,
            allow_multi_rect: false,
            ..Default::default()
        },
    )
    .unwrap();
    let last = &result.frames[3];
    let winner = &last.candidates[last.winner_index];
    assert_eq!(winner.strategy, Strategy::CropRetained);
    assert_eq!(winner.reference_index, Some(1));
    for report in &result.frames {
        assert_eq!(report.references_before[3], None);
        assert_eq!(report.references_after[3], None);
        assert!(
            report
                .candidates
                .iter()
                .all(|c| c.source < 3 && c.save_as_reference < 3)
        );
    }
    let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded.frames.len(), images.len());
    for (index, (frame, expected)) in decoded.frames.iter().zip(&images).enumerate() {
        assert_eq!(&frame.pixels, expected.pixels());
        assert_eq!(frame.duration_ticks, index as u32 + 1);
    }
}
