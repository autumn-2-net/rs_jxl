use super::*;
use crate::{PixelData, PixelLayout};
use std::cell::Cell;

#[test]
fn releasing_samples_drops_allocations_without_resetting_work_or_batch_limits() {
    let image = Image::new(
        1024,
        512,
        PixelLayout::Gray,
        PixelData::U8(vec![7; 1024 * 512]),
    )
    .unwrap();
    let pixels = 3 * 1024 * 8;
    for batched in [false, true] {
        let calls = Cell::new(0);
        let mut screen = CandidateScreen::new(
            &image,
            None,
            ScreenOptions {
                regions: RegionOptions {
                    full_width_strips: true,
                    strip_rows: Some(8),
                    ..Default::default()
                },
                limits: ProbeLimits {
                    max_encodes: 6,
                    max_encoded_pixels: 2 * pixels as u64,
                    max_resident_sample_pixels: pixels,
                    ..Default::default()
                },
                ..Default::default()
            },
            |_, _| {
                assert!(!batched);
                calls.set(calls.get() + 1);
                Ok(100)
            },
        )
        .unwrap();
        if batched {
            screen.set_batch_encoder(Box::new(|tasks| {
                calls.set(calls.get() + tasks.len());
                Ok(vec![100; tasks.len()])
            }));
        }
        let params = CandidateParams::default();
        assert_eq!(screen.score(&params).unwrap(), vec![100; 3]);
        assert_eq!(screen.score(&params).unwrap(), vec![100; 3]);
        assert_eq!(calls.get(), 3);
        assert_eq!(screen.samples.len(), 3);
        assert_eq!(screen.cache.len(), 1);
        screen.release_samples_and_cache();
        assert_eq!(screen.samples.capacity(), 0);
        assert_eq!(screen.cache.capacity(), 0);
        assert_eq!(
            (screen.encodes(), screen.sample_pixels(), screen.batches()),
            (3, pixels as u64, 1)
        );
        assert_eq!(screen.score(&params).unwrap(), vec![100; 3]);
        assert_eq!(
            (screen.encodes(), screen.sample_pixels(), screen.batches()),
            (6, 2 * pixels as u64, 2)
        );
        screen.release_samples_and_cache();
        screen.release_samples_and_cache();
        assert!(matches!(
            screen.score(&params),
            Err(Error::Limit("probe work budget"))
        ));
        assert_eq!(calls.get(), 6);
        assert_eq!(
            (screen.encodes(), screen.sample_pixels(), screen.batches()),
            (6, 2 * pixels as u64, 2)
        );
    }
}
