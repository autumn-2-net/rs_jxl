use super::*;
use crate::{PixelLayout, scheduler::ScanTile};

fn gray(width: usize, height: usize) -> Image {
    Image::new(
        width as u32,
        height as u32,
        PixelLayout::Gray,
        PixelData::U8(vec![0; width * height]),
    )
    .unwrap()
}

fn synthetic_scan(
    width: usize,
    height: usize,
    anchors: &[(usize, usize, u64, u64)],
) -> LosslessScan {
    LosslessScan {
        width,
        height,
        pixels: width as u64 * height as u64,
        tiles: anchors
            .iter()
            .map(|&(anchor_x, anchor_y, horizontal, vertical)| ScanTile {
                anchor_x,
                anchor_y,
                horizontal,
                vertical,
                observations: 8,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    }
}

fn off_diagonal_scan() -> LosslessScan {
    synthetic_scan(
        1536,
        1024,
        &[(1500, 40, 3, 0), (32, 980, 0, 2), (1100, 16, 1, 0)],
    )
}

fn assert_context_regions(
    image: &Image,
    scan: Option<&LosslessScan>,
    side: usize,
    expected: &[(usize, usize, usize, usize)],
) {
    let specs = screen_regions(
        image,
        scan,
        &RegionOptions {
            context_side: side,
            full_width_strips: false,
            ..Default::default()
        },
    )
    .unwrap();
    let width = image.width() as usize;
    let height = image.height() as usize;
    assert!((1..=3).contains(&specs.len()));
    let mut pixels = 0;
    let mut actual = Vec::new();
    for spec in &specs {
        assert_eq!(spec.regions.len(), 1);
        assert!(!spec.is_atlas());
        let r = spec.regions[0];
        assert_eq!((r.width, r.height), (width.min(side), height.min(side)));
        assert!(r.x + r.width <= width);
        assert!(r.y + r.height <= height);
        assert_eq!(spec.pixels().unwrap(), r.width * r.height);
        pixels += spec.pixels().unwrap();
        actual.push((r.x, r.y, r.width, r.height));
    }
    assert!(pixels <= 3 * side * side);
    assert!(actual.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        actual.as_slice(),
        expected,
        "image {width}x{height}, context side {side}"
    );
}

#[test]
fn context_uses_clipped_scan_region_centers_not_tile_anchors_or_diagonals() {
    let image = gray(1536, 1024);
    let scan = off_diagonal_scan();
    // Wide scan strips shift the x centers; the top square shifts its y center.
    let anchors: Vec<_> = scan
        .regions()
        .iter()
        .map(|r| (r.x, r.y, r.width, r.height))
        .collect();
    assert_eq!(
        anchors,
        vec![(512, 24, 1024, 32), (0, 964, 1024, 32), (972, 0, 256, 256)]
    );
    assert_context_regions(
        &image,
        Some(&scan),
        256,
        &[(384, 768, 256, 256), (896, 0, 256, 256), (972, 0, 256, 256)],
    );
    assert_context_regions(
        &image,
        Some(&scan),
        512,
        &[(256, 512, 512, 512), (768, 0, 512, 512), (844, 0, 512, 512)],
    );
}

#[test]
fn context_deduplicates_after_recentering_and_clipping() {
    let image = gray(1024, 768);
    let scan = synthetic_scan(
        1024,
        768,
        &[(990, 7, 3, 0), (11, 31, 0, 2), (50, 730, 1, 0)],
    );
    assert_eq!(scan.regions().len(), 3);
    assert_context_regions(
        &image,
        Some(&scan),
        256,
        &[(0, 512, 256, 256), (384, 0, 256, 256)],
    );
    assert_context_regions(
        &image,
        Some(&scan),
        512,
        &[(0, 256, 512, 512), (256, 0, 512, 512)],
    );
}

#[test]
fn all_inactive_scan_uses_middle_tile_and_keeps_distinct_overlapping_contexts() {
    let image = gray(1536, 1024);
    let mut scan = off_diagonal_scan();
    for tile in &mut scan.tiles {
        tile.horizontal = 0;
        tile.vertical = 0;
    }
    assert!(scan.available());
    assert_eq!(scan.regions().len(), 2);
    assert_context_regions(
        &image,
        Some(&scan),
        256,
        &[(0, 768, 256, 256), (384, 768, 256, 256)],
    );
    assert_context_regions(
        &image,
        Some(&scan),
        512,
        &[(0, 512, 512, 512), (256, 512, 512, 512)],
    );
}

#[test]
fn context_clips_small_thin_and_nearly_fitting_shapes() {
    let cases = [
        (1, 1, vec![(0, 0, 1, 1)], vec![(0, 0, 1, 1)]),
        (63, 95, vec![(0, 0, 63, 95)], vec![(0, 0, 63, 95)]),
        (
            1,
            769,
            vec![(0, 0, 1, 256), (0, 256, 1, 256), (0, 513, 1, 256)],
            vec![(0, 0, 1, 512), (0, 128, 1, 512), (0, 257, 1, 512)],
        ),
        (769, 1, vec![(256, 0, 256, 1)], vec![(128, 0, 512, 1)]),
        (
            257,
            257,
            vec![(0, 0, 256, 256), (0, 1, 256, 256)],
            vec![(0, 0, 257, 257)],
        ),
    ];
    for (width, height, expected_256, expected_512) in cases {
        let image = gray(width, height);
        let scan = synthetic_scan(
            width,
            height,
            &[
                (width - 1, 0, 3, 0),
                (0, height - 1, 0, 2),
                (width / 2, height / 2, 1, 0),
            ],
        );
        assert_context_regions(&image, Some(&scan), 256, &expected_256);
        assert_context_regions(&image, Some(&scan), 512, &expected_512);
    }
}

#[test]
fn absent_or_unavailable_scan_preserves_diagonal_fallback() {
    let image = gray(1537, 1025);
    let small = gray(63, 95);
    let unavailable = LosslessScan::default();
    assert!(!unavailable.available());
    for scan in [None, Some(&unavailable)] {
        assert_context_regions(
            &image,
            scan,
            256,
            &[
                (0, 0, 256, 256),
                (640, 384, 256, 256),
                (1281, 769, 256, 256),
            ],
        );
        assert_context_regions(
            &image,
            scan,
            512,
            &[
                (0, 0, 512, 512),
                (512, 256, 512, 512),
                (1025, 513, 512, 512),
            ],
        );
        for side in [256, 512] {
            assert_context_regions(&small, scan, side, &[(0, 0, 63, 95)]);
        }
    }
}

fn assert_rgba_rows<T: Copy + PartialEq + std::fmt::Debug + From<u8>>(
    source: &[T],
    copied: &[T],
    source_width: usize,
    region: ScanRegion,
) {
    assert_eq!(copied.len(), region.width * region.height * 4);
    for y in 0..region.height {
        let source_begin = ((region.y + y) * source_width + region.x) * 4;
        let copied_begin = y * region.width * 4;
        assert_eq!(
            &copied[copied_begin..copied_begin + region.width * 4],
            &source[source_begin..source_begin + region.width * 4],
            "source rectangle {region:?}, row {y}"
        );
    }
    assert!(copied.chunks_exact(4).any(|pixel| {
        pixel[3] == T::from(0) && pixel[..3].iter().any(|&value| value != T::from(0))
    }));
    assert!(copied.chunks_exact(4).any(|pixel| pixel[3] != T::from(0)));
}

fn assert_context_sample_copies(image: Image) {
    let scan = off_diagonal_scan();
    let original = image.pixels().clone();
    for side in [256, 512] {
        let specs = screen_regions(
            &image,
            Some(&scan),
            &RegionOptions {
                context_side: side,
                full_width_strips: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(specs.len(), 3);
        for spec in &specs {
            let pixels = spec.pixels().unwrap();
            let sample = copy_sample(&image, spec, pixels).unwrap();
            let region = spec.regions[0];
            assert_eq!(sample.width() as usize, region.width);
            assert_eq!(sample.height() as usize, region.height);
            assert_eq!(sample.layout(), image.layout());
            assert_eq!(sample.metadata(), image.metadata());
            match (image.pixels(), sample.pixels()) {
                (PixelData::U8(source), PixelData::U8(copied)) => {
                    assert_rgba_rows(source, copied, image.width() as usize, region);
                }
                (PixelData::U16(source), PixelData::U16(copied)) => {
                    assert_rgba_rows(source, copied, image.width() as usize, region);
                }
                _ => panic!("context crop changed integer storage"),
            }
            assert!(matches!(
                copy_sample(&image, spec, pixels - 1),
                Err(Error::Limit("probe sample pixels"))
            ));
        }
    }
    assert_eq!(image.pixels(), &original);
}

#[test]
fn context_samples_preserve_original_u8_rgba_including_hidden_rgb() {
    let pixels = (0..1536 * 1024)
        .flat_map(|p| {
            [
                ((p * 37) % 255 + 1) as u8,
                ((p / 1536 * 17 + p) % 256) as u8,
                ((p * 73) % 256) as u8,
                if (p + p / 1536) % 3 == 0 { 0 } else { 255 },
            ]
        })
        .collect();
    assert_context_sample_copies(
        Image::new(1536, 1024, PixelLayout::Rgba, PixelData::U8(pixels)).unwrap(),
    );
}

#[test]
fn context_samples_preserve_original_u16_rgba_including_hidden_rgb() {
    let pixels = (0..1536 * 1024)
        .flat_map(|p| {
            [
                (0x8000 | (p & 0x7fff)) as u16,
                ((p * 257) % 65536) as u16,
                (65535 - p % 65536) as u16,
                if (p + p / 1536) % 3 == 0 { 0 } else { 65535 },
            ]
        })
        .collect();
    assert_context_sample_copies(
        Image::new(1536, 1024, PixelLayout::Rgba, PixelData::U16(pixels)).unwrap(),
    );
}

#[test]
fn available_scan_with_mismatched_width_or_height_is_rejected() {
    let image = gray(1536, 1024);
    for (width, height) in [(1535, 1024), (1536, 1023), (1535, 1023)] {
        let scan = synthetic_scan(width, height, &[(32, 32, 1, 0)]);
        assert!(scan.available());
        for side in [256, 512] {
            let error = screen_regions(
                &image,
                Some(&scan),
                &RegionOptions {
                    context_side: side,
                    full_width_strips: false,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(matches!(
                error,
                Error::InvalidInput(message)
                    if message == "coverage dimensions do not match probe image"
            ));
        }
    }
}
