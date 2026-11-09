use super::*;
use crate::tools::patches::{PatchOptions, ReferenceImage, compose, discover};

fn forced() -> TilePatchOptions {
    TilePatchOptions {
        force: true,
        ..Default::default()
    }
}

fn fixture(bits: u32, colors: usize, extras: usize) -> ReferenceImage {
    let (width, height) = (65, 35);
    let maximum = (1 << bits) - 1;
    let planes = (0..colors + extras)
        .map(|c| {
            (0..width * height)
                .map(|p| {
                    let (x, y) = (p % width, p / width);
                    let tile = if y < 16 && x < 48 {
                        0
                    } else {
                        1 + (x / 16) + 4 * (y / 16)
                    };
                    if c >= colors && (x % 16 + y % 16) % 3 == 0 {
                        0
                    } else {
                        ((x % 16 * 71 + y % 16 * 39 + tile * 51 + c * 17) as i32) & maximum
                    }
                })
                .collect()
        })
        .collect();
    ReferenceImage {
        width,
        height,
        bit_depth: bits,
        color_channels: colors,
        planes,
    }
}

fn replay(original: &ReferenceImage, plan: &PatchPlan) {
    let residual = ImageView {
        planes: &plan.residual,
        ..original.view()
    };
    assert_eq!(
        compose(residual, plan.reference.view(), &plan.dictionary).unwrap(),
        original.planes
    );
    for plane in 0..original.planes.len() {
        for y in 0..original.height {
            assert_eq!(
                plan.residual[plane][y * original.width + 64],
                original.planes[plane][y * original.width + 64]
            );
        }
        assert_eq!(
            &plan.residual[plane][32 * original.width..],
            &original.planes[plane][32 * original.width..]
        );
    }
}

#[test]
fn exact_replay_all_integer_planes_and_partial_edges() {
    for bits in [8, 16] {
        for colors in [1, 3] {
            for extras in [0, 1, 3] {
                let original = fixture(bits, colors, extras);
                let plan = discover_tiles(original.view(), &forced()).unwrap().unwrap();
                assert!(plan.dictionary.positions.len() >= 3);
                replay(&original, &plan);
            }
        }
    }
}

#[test]
fn zero_alpha_does_not_hide_rgb_or_alpha_differences() {
    for channel in [0, 3] {
        let mut image = fixture(8, 3, 1);
        // Make all alpha identical so the first three textured tiles repeat.
        image.planes[3].fill(0);
        let p = 32;
        image.planes[channel][p] ^= 1;
        let plan = discover_tiles(image.view(), &forced()).unwrap();
        assert!(plan.is_none(), "only two identical tiles remain");
    }
}

#[test]
fn explicit_minimum_and_non_component_texture() {
    let mut original = fixture(8, 3, 0);
    let options = forced();
    assert!(
        discover(original.view(), &PatchOptions::default())
            .unwrap()
            .is_none()
    );
    assert!(discover_tiles(original.view(), &options).unwrap().is_some());
    original.planes[0][32] ^= 1;
    assert!(discover_tiles(original.view(), &options).unwrap().is_none());
    let options = TilePatchOptions {
        force: true,
        min_occurrences: 2,
        ..options
    };
    replay(
        &original,
        &discover_tiles(original.view(), &options).unwrap().unwrap(),
    );
}

#[test]
fn collisions_are_partitioned_by_all_samples() {
    let image = fixture(16, 3, 0);
    // Adversarial index deliberately maps every distinct tile to one hash.
    let mut index: Vec<_> = (0..2)
        .flat_map(|y| {
            (0..4).map(move |x| Tile {
                hash: 0,
                origin: y * 16 * 65 + x * 16,
            })
        })
        .collect();
    let groups = repeated_ranges(image.view(), &mut index, 16, 3);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 3);
    let origins: Vec<_> = index[groups[0].clone()].iter().map(|t| t.origin).collect();
    assert_eq!(origins, vec![0, 16, 32]);
}

#[test]
fn flat_and_small_inputs_have_no_dictionary() {
    for (width, height) in [(15, 17), (16, 16), (17, 15), (64, 64)] {
        let image = ReferenceImage {
            width,
            height,
            bit_depth: 8,
            color_channels: 3,
            planes: vec![vec![99; width * height]; 4],
        };
        assert!(
            discover_tiles(image.view(), &Default::default())
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn retained_limits_and_receipt_do_not_change_original() {
    let original = fixture(8, 3, 0);
    let receipt = Arc::new(TilePatchReceipt::default());
    let options = TilePatchOptions {
        force: true,
        receipt: Some(receipt.clone()),
        max_unique_tiles: 1,
        max_positions: 3,
        max_reference_pixels: 256,
        ..Default::default()
    };
    let plan = discover_tiles(original.view(), &options).unwrap().unwrap();
    replay(&original, &plan);
    let stats = receipt.snapshot();
    assert_eq!(stats.calls, 1);
    assert_eq!(stats.tiles_examined, 8);
    assert_eq!(stats.selected_groups, 1);
    assert_eq!(stats.positions, 3);
    assert_eq!(stats.covered_pixels, 768);
    assert_eq!(stats.atlas_pixels, 256);
    assert!(stats.seconds.is_finite() && stats.seconds >= 0.0);
    let too_small = TilePatchOptions {
        max_reference_pixels: 255,
        ..options.clone()
    };
    assert!(
        discover_tiles(original.view(), &too_small)
            .unwrap()
            .is_none()
    );
    let too_few = TilePatchOptions {
        max_positions: 2,
        ..options
    };
    assert!(discover_tiles(original.view(), &too_few).unwrap().is_none());
    assert_eq!(receipt.snapshot().calls, 3);
}

#[test]
fn invalid_options_and_image_are_errors() {
    let image = fixture(8, 3, 0);
    for options in [
        TilePatchOptions {
            tile_size: 0,
            ..Default::default()
        },
        TilePatchOptions {
            tile_size: usize::MAX,
            ..Default::default()
        },
        TilePatchOptions {
            min_occurrences: 1,
            ..Default::default()
        },
        TilePatchOptions {
            max_unique_tiles: 0,
            ..Default::default()
        },
        TilePatchOptions {
            max_positions: 0,
            ..Default::default()
        },
        TilePatchOptions {
            max_reference_pixels: 0,
            ..Default::default()
        },
        TilePatchOptions {
            min_estimated_savings: f64::NAN,
            ..Default::default()
        },
        TilePatchOptions {
            min_estimated_savings: -0.01,
            ..Default::default()
        },
        TilePatchOptions {
            full_frame_min_estimated_savings: 1.01,
            ..Default::default()
        },
    ] {
        assert!(discover_tiles(image.view(), &options).is_err());
    }
    let mut image = image;
    image.planes[0][0] = 256;
    assert!(discover_tiles(image.view(), &Default::default()).is_err());
}

#[test]
fn tiny_tiles_account_for_decoder_blending_capacity() {
    let options = TilePatchOptions {
        tile_size: 2,
        ..forced()
    };
    let image = fixture(8, 1, 256);
    let budget = options
        .position_budget(image.width * image.height, 256)
        .unwrap();
    let plan = discover_tiles(image.view(), &options).unwrap().unwrap();
    assert!(plan.dictionary.positions.len() <= budget);
    let maximum = (1024 + image.width * image.height / 4) * 4;
    assert!(plan.dictionary.positions.len() * 2 * 257 <= maximum * 4);
    let residual = ImageView {
        planes: &plan.residual,
        ..image.view()
    };
    assert_eq!(
        compose(residual, plan.reference.view(), &plan.dictionary).unwrap(),
        image.planes
    );
}

#[test]
fn cheap_admission_rejects_predictable_repetition_before_plan_allocation() {
    let mut image = ReferenceImage {
        width: 64,
        height: 32,
        bit_depth: 8,
        color_channels: 1,
        planes: vec![vec![0; 64 * 32]],
    };
    for y in (0..32).step_by(16) {
        for x in (0..64).step_by(16) {
            image.planes[0][y * 64 + x] = 1;
        }
    }
    assert!(discover_tiles(image.view(), &forced()).unwrap().is_some());
    let receipt = Arc::new(TilePatchReceipt::default());
    let screened = TilePatchOptions {
        receipt: Some(receipt.clone()),
        ..Default::default()
    };
    assert!(discover_tiles(image.view(), &screened).unwrap().is_none());
    let work = receipt.snapshot();
    assert_eq!(work.admission_rejected, 1);
    assert!(work.repeated_groups > 0);
    assert_eq!(
        (work.selected_groups, work.positions, work.atlas_pixels),
        (0, 0, 0)
    );
    assert!(work.estimated_saved_bits < 0.0);
    assert!(work.estimated_baseline_bits > 0.0);
}

#[test]
fn admission_accepts_dense_high_entropy_repeats_and_obeys_the_margin() {
    let (width, height) = (64, 32);
    let mut tile = Vec::new();
    let mut state = 12345u32;
    for _ in 0..256 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        tile.push((state & 255) as i32);
    }
    let image = ReferenceImage {
        width,
        height,
        bit_depth: 8,
        color_channels: 1,
        planes: vec![
            (0..width * height)
                .map(|p| tile[(p / width % 16) * 16 + p % width % 16])
                .collect(),
        ],
    };
    let receipt = Arc::new(TilePatchReceipt::default());
    let options = TilePatchOptions {
        receipt: Some(receipt.clone()),
        ..Default::default()
    };
    let plan = discover_tiles(image.view(), &options).unwrap().unwrap();
    assert_eq!(plan.dictionary.positions.len(), 8);
    let work = receipt.snapshot();
    assert_eq!(work.admission_rejected, 0);
    assert!(
        work.estimated_saved_bits > options.min_estimated_savings * work.estimated_baseline_bits
    );
    assert!(
        discover_tiles(
            image.view(),
            &TilePatchOptions {
                min_estimated_savings: 1.0,
                ..options
            }
        )
        .unwrap()
        .is_none()
    );
    let residual = ImageView {
        planes: &plan.residual,
        ..image.view()
    };
    assert_eq!(
        compose(residual, plan.reference.view(), &plan.dictionary).unwrap(),
        image.planes
    );
}
