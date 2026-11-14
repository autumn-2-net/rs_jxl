use super::*;
use crate::modular::palette::{PaletteOrder, PaletteTools};
use crate::modular::source_options::SourceBuffering;
use crate::tools::lz77::SourceQuickMode;
use crate::{PixelData, PixelLayout};

fn image(width: u32, height: u32, layout: PixelLayout) -> Image {
    Image::new(
        width,
        height,
        layout,
        PixelData::U8(vec![
            31;
            width as usize * height as usize * layout.channels()
        ]),
    )
    .unwrap()
}

#[test]
fn ordinary_e9_enables_all_three_tools_without_patch_or_search() {
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        let image = image(32, 32, layout);
        for effort in 1..=11 {
            let params = CandidateParams::default().backend(effort).unwrap();
            let projected = project_native(&params, &image, &EncodeOptions::default()).unwrap();
            let expected = usize::from(effort == 9).min(layout.channels() - 1);
            assert_eq!(projected.training.previous_channels, expected);
            let options = projected.exact_options().unwrap();
            assert_eq!(options.modular.previous_channels, expected);
            assert_eq!(
                options.modular.sampling.as_ref().unwrap().previous_channels,
                expected
            );
            if let Some(lz) = &options.modular.source_lz77 {
                assert_eq!(
                    lz.quick,
                    if effort == 9 {
                        SourceQuickMode::Sparse
                    } else {
                        SourceQuickMode::Off
                    }
                );
            }
            assert_eq!(
                options.modular.palette_policy.unwrap().tools.order,
                if effort == 9 {
                    PaletteOrder::ScreenedRuns
                } else {
                    PaletteOrder::Default
                }
            );
        }
    }
}

#[test]
fn default_tools_cover_fullframe_streaming_and_patch_atlas() {
    let image = image(385, 257, PixelLayout::Rgb);
    for (buffering, responsive, patch, streaming) in [
        (SourceBuffering::FullFrame, false, Override::Off, false),
        (SourceBuffering::Streaming, false, Override::Off, true),
        (SourceBuffering::FullFrame, true, Override::Off, false),
        (SourceBuffering::Auto, false, Override::On, false),
    ] {
        let mut params = CandidateParams::default().backend(9).unwrap();
        params.group_size_shift = Some(0);
        params.buffering = buffering;
        params.responsive = responsive;
        params.patches = patch;
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        assert_eq!(options.modular.entropy_options.streaming_mode, streaming);
        assert_eq!(options.modular.previous_channels, 1);
        assert_eq!(
            options.modular.source_lz77.as_ref().unwrap().quick,
            SourceQuickMode::Sparse
        );
        let palette = options.modular.palette_policy.as_ref().unwrap();
        assert_eq!(palette.groupwise, streaming);
        assert_eq!(palette.tools.order, PaletteOrder::ScreenedRuns);
        if patch == Override::On {
            let policy = options.source_patch_reference.unwrap();
            assert_eq!(policy.sampling.max_previous_channels, 1);
            let reference = policy.resolve(&image).unwrap();
            assert_eq!(reference.modular.previous_channels, 1);
            assert_eq!(
                reference.modular.source_lz77.unwrap().quick,
                SourceQuickMode::Sparse
            );
            assert_eq!(
                reference.modular.palette_policy.unwrap().tools.order,
                PaletteOrder::ScreenedRuns
            );
        }
    }
}

#[test]
fn default_optout_and_explicit_zero_or_tool_overrides_are_preserved() {
    let image = image(32, 32, PixelLayout::Rgb);
    let mut params = CandidateParams::default().backend(9).unwrap();
    for enabled in [false, true] {
        let mut base = EncodeOptions::default();
        base.modular.source_tool_defaults = enabled;
        let actual = project_native(&params, &image, &base)
            .unwrap()
            .exact_options()
            .unwrap();
        assert_eq!(actual.modular.previous_channels, usize::from(enabled));
        assert_eq!(
            actual.modular.source_lz77.unwrap().quick,
            if enabled {
                SourceQuickMode::Sparse
            } else {
                SourceQuickMode::Off
            }
        );
        assert_eq!(
            actual.modular.palette_policy.unwrap().tools.order,
            if enabled {
                PaletteOrder::ScreenedRuns
            } else {
                PaletteOrder::Default
            }
        );
        // Independent opt-out: the previous MA defaults and patch gather remain on.
        assert_eq!(
            actual.modular.tree_learning,
            crate::modular::ma_train::TreeLearning::Symmetric
        );
        assert!(actual.modular.refinement.offsets.is_some());
        assert!(actual.source_patch_parallel_gather);
        base.modular.source_lz77 = Some(SourceConfig {
            quick: SourceQuickMode::Off,
            ..Default::default()
        });
        base.modular.palette_policy = Some(PaletteOptions {
            tools: PaletteTools {
                order: PaletteOrder::RunFrequency,
                ..Default::default()
            },
            ..Default::default()
        });
        for count in [0, 2] {
            params.max_properties = Some(count);
            params.patches = Override::On;
            let actual = project_native(&params, &image, &base)
                .unwrap()
                .exact_options()
                .unwrap();
            assert_eq!(actual.modular.previous_channels, usize::from(count));
            assert_eq!(
                actual.modular.source_lz77.unwrap().quick,
                SourceQuickMode::Off
            );
            assert_eq!(
                actual.modular.palette_policy.unwrap().tools.order,
                PaletteOrder::RunFrequency
            );
            let atlas = actual
                .source_patch_reference
                .unwrap()
                .resolve(&image)
                .unwrap();
            assert_eq!(atlas.modular.previous_channels, usize::from(count));
            assert_eq!(
                atlas.modular.source_lz77.unwrap().quick,
                SourceQuickMode::Off
            );
            assert_eq!(
                atlas.modular.palette_policy.unwrap().tools.order,
                PaletteOrder::RunFrequency
            );
        }
        params.max_properties = None;
    }
}
