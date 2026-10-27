use super::super::{
    Rect, Strategy,
    candidates::{Candidate, candidates_from_maps},
    diff::DiffMap,
};
use super::*;
use crate::{PixelData, PixelLayout, metadata::Animation};
use std::sync::Arc;

#[test]
fn pruning_cutoff_keeps_canonical_equal_size_ties() {
    let winner = Winner {
        rank: 4,
        bytes: vec![0; 23],
        references: [None; 4],
        slots: ReferenceSlots::default(),
    };
    assert_eq!(winner.cutoff(3), Some(24));
    assert_eq!(winner.cutoff(4), Some(23));
    assert_eq!(winner.cutoff(5), Some(23));
}

#[test]
fn sparse_mask_ranking_accounts_for_bbox_work_without_losing_its_discount() {
    let width = 256;
    let height = 256;
    let mut pixels = vec![255; width * height * 4];
    let previous = Image::new(
        width as u32,
        height as u32,
        PixelLayout::Rgba,
        PixelData::U8(pixels.clone()),
    )
    .unwrap();
    pixels[(32 * width + 32) * 4] = 0;
    pixels[(224 * width + 224) * 4] = 0;
    let current = Image::new(
        width as u32,
        height as u32,
        PixelLayout::Rgba,
        PixelData::U8(pixels),
    )
    .unwrap();
    let diff = Arc::new(DiffMap::between(&previous, &current).unwrap());
    assert_eq!(diff.changed_pixels(), 2);
    let candidates =
        candidates_from_maps(&current, 1, &AutoOptions::default(), 1.0, &[(0, diff)]).unwrap();
    let mask = candidates.iter().find(|c| c.mask).unwrap();
    let crop = candidates
        .iter()
        .find(|c| c.strategy == Strategy::CropPrevious)
        .unwrap();
    let tiles = candidates
        .iter()
        .find(|c| c.strategy == Strategy::TilesPrevious)
        .unwrap();
    assert!(mask.estimated_work() < crop.estimated_work());
    assert!(tiles.estimated_work() < mask.estimated_work());
    assert_eq!(mask.estimated_work(), 2 + (193u64 * 193).div_ceil(8) + 64);

    // A coarser partition may include unchanged pixels yet still be much
    // cheaper to execute than scanning a large sparse-mask bbox.
    let coarser = Candidate {
        strategy: Strategy::MergedPrevious,
        tile_size: Some(32),
        reference_index: Some(0),
        diff: None,
        source: 1,
        rects: vec![
            Rect {
                x: 32,
                y: 32,
                width: 32,
                height: 16,
            },
            Rect {
                x: 193,
                y: 209,
                width: 32,
                height: 16,
            },
        ],
        mask: false,
    };
    assert!(coarser.estimated_work() < mask.estimated_work());
}

#[test]
fn reusable_auto_pool_does_not_retain_frame_references_or_trial_budgets() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let options = EncodeOptions {
        threads: 2,
        ..Default::default()
    };
    for supplied in [None, Some(&pool)] {
        for value in [3, 91, 3] {
            let images: Vec<_> = [value, value + 1]
                .into_iter()
                .map(|value| {
                    let image = Image::new(4, 4, PixelLayout::Gray, PixelData::U8(vec![value; 16]))
                        .unwrap();
                    let mut metadata = image.metadata().clone();
                    metadata.animation = Some(Animation::default());
                    image.with_metadata(metadata, vec![]).unwrap()
                })
                .collect();
            let limited = AutoOptions {
                max_total_trials: 1,
                ..Default::default()
            };
            assert!(matches!(
                encode_auto_in_pool(&images, &[1, 2], &options, &limited, supplied),
                Err(Error::Limit(_))
            ));
            let result = encode_auto_in_pool(
                &images,
                &[1, 2],
                &options,
                &AutoOptions::default(),
                supplied,
            )
            .unwrap();
            let decoded = crate::decode::decode(&result.bytes, &Default::default()).unwrap();
            assert_eq!(decoded.frames.len(), images.len());
            for (frame, image) in decoded.frames.iter().zip(&images) {
                assert_eq!(&frame.pixels, image.pixels());
            }
            assert_eq!(result.frames.len(), images.len());
            assert!(!crate::execution::is_serial());
        }
    }
}
