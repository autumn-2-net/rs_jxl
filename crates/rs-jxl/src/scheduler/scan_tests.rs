use super::scan::{ScanOptions, scan_lossless_input, scan_lossless_input_with_pool};
use crate::{
    Image, PixelData, PixelLayout,
    metadata::{ExtraChannelInfo, ExtraChannelType},
};

#[test]
fn parallel_scan_matches_every_scalar_field_and_region() {
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for (w, h) in [(1, 1), (1, 193), (193, 1), (129, 193), (4097, 131)] {
            let pixels = (w * h) as usize;
            let image = Image::new(
                w,
                h,
                PixelLayout::Rgba,
                PixelData::U16(
                    (0..pixels)
                        .flat_map(|i| [((i * 257) % 65536) as u16, (i % 11) as u16, 0, 0])
                        .collect(),
                ),
            )
            .unwrap();
            let mut metadata = image.metadata().clone();
            metadata
                .extra_channels
                .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap());
            metadata
                .extra_channels
                .push(ExtraChannelInfo::new(ExtraChannelType::SpotColor, 16).unwrap());
            let image = image
                .with_metadata(
                    metadata,
                    vec![
                        PixelData::U8((0..pixels).map(|i| (i % 251) as u8).collect()),
                        PixelData::U16((0..pixels).map(|i| ((i / 3) % 65536) as u16).collect()),
                    ],
                )
                .unwrap();
            for palette_scope in [false, true] {
                let options = ScanOptions { palette_scope };
                let scalar = scan_lossless_input(&image, options);
                let parallel = scan_lossless_input_with_pool(&image, options, &pool);
                assert_eq!(parallel, scalar);
                assert_eq!(parallel.regions(), scalar.regions());
                assert_eq!(
                    parallel.tiles.iter().map(|t| t.observations).sum::<u64>(),
                    6 * pixels as u64
                );
            }
        }
    }
}

#[test]
fn peak_ties_keep_first_row_and_first_plane() {
    let image = Image::new(1, 3, PixelLayout::Gray, PixelData::U8(vec![0, 1, 1])).unwrap();
    let mut metadata = image.metadata().clone();
    metadata
        .extra_channels
        .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
    let image = image
        .with_metadata(metadata, vec![PixelData::U16(vec![0, 0, 256])])
        .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let scan = scan_lossless_input_with_pool(&image, ScanOptions::default(), &pool);
    assert_eq!(
        (
            scan.tiles[0].anchor_x,
            scan.tiles[0].anchor_y,
            scan.tiles[0].peak_row
        ),
        (0, 1, 1)
    );
    assert_eq!(scan, scan_lossless_input(&image, ScanOptions::default()));
}

#[test]
fn parallel_scan_rejects_float_storage_without_reinterpreting_bits() {
    let image = Image::new(2, 1, PixelLayout::Gray, PixelData::F32(vec![0.0, -0.0])).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let scan = scan_lossless_input_with_pool(&image, ScanOptions::default(), &pool);
    assert!(!scan.available());
    assert!(!scan.literal_candidate);
}
