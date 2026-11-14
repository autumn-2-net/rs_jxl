use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::modular::refinement::{OffsetMode, Receipt};
use rs_jxl::scheduler::{
    adapter::project_native,
    params::{CandidateParams, SourceBuffering},
};
use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout, encode};
use std::sync::Arc;

#[test]
fn default_e9_offsets_roundtrip_integer_layouts_and_group_routes() {
    for threads in [1, 4] {
        for buffering in [SourceBuffering::FullFrame, SourceBuffering::Streaming] {
            for layout in [
                PixelLayout::Gray,
                PixelLayout::GrayAlpha,
                PixelLayout::Rgb,
                PixelLayout::Rgba,
            ] {
                for wide in [false, true] {
                    let (w, h) = (67u32, 51u32);
                    let channels = layout.channels();
                    let maximum = if wide { 65535 } else { 255 };
                    let samples: Vec<u16> = (0..(w * h) as usize)
                        .flat_map(|i| {
                            (0..channels).map(move |c| {
                                if layout.has_alpha() && c == channels - 1 {
                                    [0, maximum / 2, maximum][i % 3] as u16
                                } else {
                                    ((i * 137 + c * 7919 + i / w as usize * 17) & maximum) as u16
                                }
                            })
                        })
                        .collect();
                    let pixels = if wide {
                        PixelData::U16(samples)
                    } else {
                        PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
                    };
                    let image = Image::new(w, h, layout, pixels).unwrap();
                    let receipt = Arc::new(Receipt::default());
                    let mut base = EncodeOptions {
                        threads,
                        ..Default::default()
                    };
                    base.modular.refinement.receipt = Some(receipt.clone());
                    let mut params = CandidateParams::default().backend(9).unwrap();
                    params.buffering = buffering;
                    params.group_size_shift = Some(0);
                    let options = project_native(&params, &image, &base)
                        .unwrap()
                        .exact_options()
                        .unwrap();
                    assert_eq!(
                        options.modular.refinement.offsets.as_ref().unwrap().mode,
                        OffsetMode::DistributionSinglePass
                    );
                    let bytes = encode(&image, &options).unwrap();
                    let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
                    assert!(decoded.complete);
                    assert_eq!(decoded.frames.len(), 1);
                    assert_eq!(
                        &decoded.frames[0].pixels,
                        image.pixels(),
                        "{layout:?} wide={wide} threads={threads} buffering={buffering:?}"
                    );
                    let stats = receipt.snapshot();
                    assert!(stats.domains > 0);
                    assert_eq!(stats.models, stats.domains);
                    assert_eq!(stats.offset_single_pass_domains, stats.domains);
                    assert_eq!(stats.offset_wins, 0);
                    assert_eq!(stats.saved_bits, 0);
                }
            }
        }
    }
}

#[test]
fn direct_offsets_compose_with_shared_streams_and_explicit_predictor_refit() {
    let (width, height) = (131u32, 67u32);
    let pixels = (0..width * height)
        .flat_map(|i| {
            let v = ((i % width) * 97 + (i / width) * 17) as u16;
            [v, v.wrapping_add(1000), v.wrapping_mul(3)]
        })
        .collect();
    let image = Image::new(width, height, PixelLayout::Rgb, PixelData::U16(pixels)).unwrap();
    for threads in [1, 4] {
        for buffering in [SourceBuffering::FullFrame, SourceBuffering::Streaming] {
            for previous in [0, 2] {
                let receipt = Arc::new(Receipt::default());
                let mut base = EncodeOptions {
                    threads,
                    ..Default::default()
                };
                base.modular.refinement.receipt = Some(receipt.clone());
                base.modular.refinement.leaves = true;
                let mut params = CandidateParams::default().backend(9).unwrap();
                params.buffering = buffering;
                params.group_size_shift = Some(0);
                params.max_properties = previous;
                let options = project_native(&params, &image, &base)
                    .unwrap()
                    .exact_options()
                    .unwrap();
                let bytes = encode(&image, &options).unwrap();
                let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
                assert!(decoded.complete);
                assert_eq!(decoded.frames.len(), 1);
                assert_eq!(
                    &decoded.frames[0].pixels,
                    image.pixels(),
                    "threads={threads} buffering={buffering:?} previous={previous}"
                );
                let stats = receipt.snapshot();
                assert!(stats.offset_single_pass_domains > 0);
                // Refit may decline a change, including nonzero-offset leaves.
                assert_eq!(stats.models, stats.domains + stats.refit_proposals);
                assert_eq!(stats.offset_wins, 0);
            }
        }
    }
}
