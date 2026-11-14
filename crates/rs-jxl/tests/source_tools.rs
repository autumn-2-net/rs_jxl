use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::scheduler::{CandidateParams, source_tools};
use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout, encode};

#[test]
fn composed_e9_tools_preserve_integer_pixels_including_invisible_color() {
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        for bits in [8, 16] {
            let channels = layout.channels();
            let samples: Vec<u16> = (0..32 * 32 * channels)
                .map(|i| {
                    if layout.has_alpha() && i % channels == channels - 1 && (i / channels) % 3 == 0
                    {
                        0
                    } else {
                        (i.wrapping_mul(193).wrapping_add(i / 23) & ((1 << bits) - 1)) as u16
                    }
                })
                .collect();
            let data = if bits == 8 {
                PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
            } else {
                PixelData::U16(samples)
            };
            let image = Image::new(32, 32, layout, data).unwrap();
            for threads in [1, 4] {
                let base = EncodeOptions {
                    threads,
                    ..Default::default()
                };
                for mask in 0..8 {
                    let tools = source_tools::Options {
                        sparse_lz77: mask & 1 != 0,
                        previous_channel: mask & 2 != 0,
                        screened_runs: mask & 4 != 0,
                    };
                    let params = CandidateParams::default().backend(9).unwrap();
                    let options = tools.build(&params, &image, &base).unwrap();
                    let bytes = encode(&image, &options).unwrap();
                    let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
                    assert!(decoded.complete);
                    assert_eq!(decoded.frames.len(), 1);
                    assert_eq!(
                        &decoded.frames[0].pixels,
                        image.pixels(),
                        "{layout:?} bits={bits} threads={threads} mask={mask}"
                    );
                }
            }
        }
    }
}
