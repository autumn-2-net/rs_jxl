use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::modular::ma_train::{DescriptionCost, TreeLearning};
use rs_jxl::modular::refinement::{OffsetMode, OffsetOptions, Receipt};
use rs_jxl::scheduler::{
    adapter::project_native,
    params::{CandidateParams, SourceBuffering},
};
use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout, encode};
use std::sync::Arc;

#[test]
fn symmetric_half_offsets_roundtrip_integer_layouts_and_group_routes() {
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        for wide in [false, true] {
            let (width, height) = (133u32, 67u32);
            let maximum = if wide { 65535 } else { 255 };
            let channels = layout.channels();
            let samples: Vec<u16> = (0..(width * height) as usize)
                .flat_map(|i| {
                    (0..channels).map(move |c| {
                        if layout.has_alpha() && c == channels - 1 {
                            [0, maximum / 2, maximum][i % 3] as u16
                        } else {
                            ((i * 137 + c * 7919 + i / width as usize * 17) & maximum) as u16
                        }
                    })
                })
                .collect();
            let pixels = if wide {
                PixelData::U16(samples)
            } else {
                PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
            };
            let image = Image::new(width, height, layout, pixels).unwrap();
            for (threads, buffering) in [
                (1, SourceBuffering::FullFrame),
                (4, SourceBuffering::Streaming),
            ] {
                let mut base = EncodeOptions {
                    threads,
                    ..Default::default()
                };
                base.modular.source_ma_defaults = false;
                base.modular.tree_learning = TreeLearning::Symmetric;
                base.modular.source_ma_scoring.description = DescriptionCost::Half;
                base.modular.refinement.offsets = Some(OffsetOptions {
                    mode: OffsetMode::DistributionSinglePass,
                    ..Default::default()
                });
                let receipt = Arc::new(Receipt::default());
                base.modular.refinement.receipt = Some(receipt.clone());
                let mut params = CandidateParams::default().backend(9).unwrap();
                params.buffering = buffering;
                params.group_size_shift = Some(0);
                let options = project_native(&params, &image, &base)
                    .unwrap()
                    .exact_options()
                    .unwrap();
                assert_eq!(options.modular.tree_learning, TreeLearning::Symmetric);
                assert_eq!(
                    options.modular.source_ma_scoring.description,
                    DescriptionCost::Half
                );
                let bytes = encode(&image, &options).unwrap();
                let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
                assert!(decoded.complete);
                assert_eq!(decoded.frames.len(), 1);
                assert_eq!(
                    &decoded.frames[0].pixels,
                    image.pixels(),
                    "{layout:?}, wide={wide}, {buffering:?}"
                );
                let stats = receipt.snapshot();
                assert!(stats.domains > 0);
                assert_eq!(stats.models, stats.domains);
                assert_eq!(stats.offset_single_pass_domains, stats.domains);
            }
        }
    }
}
