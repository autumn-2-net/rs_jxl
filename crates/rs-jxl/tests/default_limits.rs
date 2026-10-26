//! Default policy must not reject legal input at private migration-era ceilings.
use rs_jxl::{EncodeOptions, ModularOptions};

#[test]
fn encoding_defaults_have_no_extra_resource_ceiling() {
    let options = EncodeOptions::default();
    assert_eq!(options.max_pixels, u64::MAX);
    assert_eq!(options.max_output_bytes, usize::MAX);
    assert_eq!(options.modular.sample_limit, None);
    let fast = rs_jxl::fast_lossless::Limits::default();
    assert_eq!(fast.max_pixels, u64::MAX);
    assert_eq!(fast.max_output_bytes, usize::MAX);
    assert_eq!(fast.max_groups, usize::MAX);
    let auto = rs_jxl::multiframe::AutoOptions::default();
    assert_eq!(auto.max_frames, usize::MAX);
    assert_eq!(auto.max_total_trials, usize::MAX);
    assert_eq!(auto.max_trial_pixels, u64::MAX);
}

#[test]
fn modular_options_allow_larger_explicit_training_requests() {
    let mut options = ModularOptions {
        tree_samples: (1 << 20) + 1,
        max_tree_leaves: 16_385,
        ..Default::default()
    };
    options.validate().unwrap();
    options.sample_limit = Some(1 << 20);
    options.validate().unwrap();
    options.sample_limit = Some(0);
    assert!(options.validate().is_err());
}

#[test]
fn arbitrary_caps_are_not_replaced_by_other_hidden_caps() {
    let limits = rs_jxl::modular::sample::SampleLimits::default();
    assert_eq!(limits.max_selected_samples, u64::MAX);
    let groups = rs_jxl::modular::tree_groups::TreeGroupOptions::default();
    assert_eq!(
        groups.max_merged_leaves,
        rs_jxl::modular::ma_train::MAX_TREE_LEAVES
    );
    groups.validate().unwrap();
}

#[test]
fn palette_limits_follow_wire_fields_and_squeeze_counts_follow_actual_transforms() {
    use rs_jxl::modular::palette::PaletteOptions;
    for limit in [65_537, 70_001, 70_911] {
        let options = ModularOptions {
            palette_colors: Some(limit),
            squeeze: vec![true; 64],
            ..Default::default()
        };
        options.validate().unwrap();
        PaletteOptions {
            palette_colors: limit as i32,
            force_palette: true,
            ..Default::default()
        }
        .validate()
        .unwrap();
    }
    let options = ModularOptions {
        palette_colors: Some(70_912),
        ..Default::default()
    };
    assert!(options.validate().is_err());
    assert!(
        PaletteOptions {
            palette_colors: 70_912,
            force_palette: true,
            ..Default::default()
        }
        .validate()
        .is_err()
    );

    let image = rs_jxl::Image::new(
        8,
        8,
        rs_jxl::PixelLayout::Gray,
        rs_jxl::PixelData::U8((0..64).collect()),
    )
    .unwrap();
    let options = EncodeOptions {
        modular: ModularOptions {
            palette_colors: Some(70_911),
            // Most requests become no-ops once all channels are one sample wide.
            squeeze: vec![true; 64],
            ..Default::default()
        },
        ..Default::default()
    };
    let encoded = rs_jxl::encode(&image, &options).unwrap();
    let decoded = rs_jxl::decode::Decoder::new(Default::default())
        .unwrap()
        .decode(&encoded)
        .unwrap();
    assert_eq!(&decoded.frames[0].pixels, image.pixels());
}
