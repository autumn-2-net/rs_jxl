//! Check the maintained fused U16 output against its planar quantizer on
//! portable scalar and available x86 SIMD descriptors. This is a development
//! executable, not a benchmark or an image decoder implementation.
//!
//! Run: cargo run --release --offline -p rs-jxl --example check_u16_output
//!
//! Ordinary example compilation leaves unrelated vendor cfg(test) fixtures
//! disabled. Real RowBuffer/layout/streaming behavior is covered separately by
//! decoder_parallel_pixels; the RowBuffer and output-buffer stand-ins below
//! must never be executed.
#![cfg(not(test))]
#![allow(dead_code)]

mod api {
    pub use jxl_decoder::api::{DitherMode, Endianness, JxlColorType, JxlDataFormat};

    // The primitive driver never enters SaveStage's buffer/layout methods.
    // Keep this glue local rather than exposing the decoder's private row_mut.
    pub struct JxlOutputBuffer<'a>(std::marker::PhantomData<&'a mut [u8]>);
    impl JxlOutputBuffer<'_> {
        pub fn row_mut(&mut self, _: usize) -> &mut [u8] {
            unreachable!("the primitive check must not access output buffers")
        }
        pub fn byte_size(&self) -> (usize, usize) {
            unreachable!("the primitive check must not access output buffers")
        }
    }
}
mod error {
    pub use jxl_decoder::error::{Error, Result};
}
mod headers {
    pub use jxl_decoder::api::Orientation;
}
mod image {
    pub use jxl_decoder::image::{DataTypeTag, ImageDataType};
}
mod util {
    #[allow(unused_imports)]
    pub use jxl_simd::f16;

    pub fn box_array<T: Clone, const N: usize>(value: T) -> Box<[T; N]> {
        vec![value; N].into_boxed_slice().try_into().ok().unwrap()
    }
    mod dither {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../vendor/jxl-rs/jxl/src/util/dither.rs"
        ));
    }
    pub use dither::{DitherTable, dither_table};
}
mod render {
    pub mod save {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../vendor/jxl-rs/jxl/src/render/save.rs"
        ));
    }
    pub mod low_memory_pipeline {
        pub mod row_buffers {
            use crate::image::ImageDataType;

            pub struct RowBuffer;
            impl RowBuffer {
                pub fn get_row<T: ImageDataType>(&self, _: usize) -> &[T] {
                    unreachable!("the primitive check must not access RowBuffer")
                }
                pub const fn x0_offset<T: ImageDataType>() -> usize {
                    64 / std::mem::size_of::<T>()
                }
                pub const fn x0_byte_offset() -> usize {
                    64
                }
            }
        }
        pub mod save {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../vendor/jxl-rs/jxl/src/render/low_memory_pipeline/save/identity.rs"
            ));

            fn boundary_values(max: u16) -> Vec<f32> {
                let mut values = vec![
                    f32::NEG_INFINITY,
                    -f32::MAX,
                    -1.0,
                    -0.0,
                    0.0,
                    f32::from_bits(1),
                    0.125,
                    0.5,
                    1.0,
                    f32::from_bits(1.0f32.to_bits() - 1),
                    f32::from_bits(1.0f32.to_bits() + 1),
                    f32::MAX,
                    f32::INFINITY,
                    f32::from_bits(0x7fc0_1234),
                ];
                for base in [0, 1, 2, 127, 128, u32::from(max) - 1] {
                    let halfway = (base as f32 + 0.5) / f32::from(max);
                    values.extend([
                        f32::from_bits(halfway.to_bits() - 1),
                        halfway,
                        f32::from_bits(halfway.to_bits() + 1),
                    ]);
                }
                values
            }

            pub fn check_descriptor<D: SimdDescriptor>(
                descriptor: D,
                normalized_integers: bool,
            ) -> Vec<u16> {
                descriptor.call(|d| {
                    let mut verified = Vec::new();
                    for bits in [1, 8, 10, 16] {
                        let max = ((1u32 << bits) - 1) as u16;
                        let original_words = [0, 1, max / 2, max / 2 + 1, max - 1, max];
                        let values = if normalized_integers {
                            original_words.iter()
                                .map(|&v| f32::from(v) / f32::from(max)).collect()
                        } else {
                            boundary_values(max)
                        };
                        for width in 1..=65 {
                            for offset in [0, 1, 3] {
                                // Both the old planar converter and fused tail
                                // consume SIMD-padded rows, as production does.
                                let input: [Vec<f32>; 4] = std::array::from_fn(|c| {
                                    (0..width + offset + 64)
                                        .map(|i| values[(i * 7 + c * 11) % values.len()])
                                        .collect()
                                });
                                let mut planar: [Vec<u16>; 4] =
                                    std::array::from_fn(|_| vec![0; width + 64]);
                                for c in 0..4 {
                                    f32_to_u16_simd(
                                        d,
                                        &input[c][offset..],
                                        &mut planar[c],
                                        f32::from(max),
                                        width,
                                    );
                                }
                                for alpha in [
                                    AlphaU16::None,
                                    AlphaU16::Opaque,
                                    AlphaU16::F32(&input[3][offset..]),
                                ] {
                                    let channels = if matches!(alpha, AlphaU16::None) {
                                        3
                                    } else {
                                        4
                                    };
                                    let mut guarded = vec![0xa55a; offset + width * channels + 3];
                                    store_f32_to_u16_interleaved_impl(
                                        d,
                                        &input[0][offset..],
                                        &input[1][offset..],
                                        &input[2][offset..],
                                        max,
                                        alpha,
                                        &mut guarded[offset..offset + width * channels],
                                    );
                                    assert!(guarded[..offset].iter().all(|&v| v == 0xa55a));
                                    assert!(guarded[offset + width * channels..]
                                        .iter().all(|&v| v == 0xa55a));
                                    for x in 0..width {
                                        for c in 0..channels {
                                            let expected = if c == 3
                                                && matches!(alpha, AlphaU16::Opaque)
                                            {
                                                max
                                            } else {
                                                planar[c][x]
                                            };
                                            let actual = guarded[offset + x * channels + c];
                                            assert_eq!(actual, expected,
                                                "{d:?}: bits={bits}, width={width}, offset={offset}, channels={channels}, x={x}, c={c}");
                                            if normalized_integers {
                                                let original = if c == 3
                                                    && matches!(alpha, AlphaU16::Opaque)
                                                {
                                                    max
                                                } else {
                                                    original_words[((x + offset) * 7 + c * 11)
                                                        % original_words.len()]
                                                };
                                                assert_eq!(actual, original,
                                                    "{d:?}: original integer bits={bits}, width={width}, x={x}, c={c}");
                                            }
                                            verified.push(actual);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    verified
                })
            }
        }
    }
}

fn report_boundary_difference(label: &str, scalar: &[u16], simd: &[u16]) {
    assert_eq!(scalar.len(), simd.len());
    if let Some((index, (&scalar, &simd))) = scalar
        .iter()
        .zip(simd)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        println!(
            "{label}: first existing planar quantizer difference at word {index}: scalar={scalar}, SIMD={simd} (scalar ties-away versus SIMD ties-even)"
        );
    }
}

fn assert_integer_identity(label: &str, scalar: &[u16], simd: &[u16]) {
    assert_eq!(scalar.len(), simd.len());
    if let Some((index, (&scalar, &simd))) = scalar
        .iter()
        .zip(simd)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!(
            "{label}: normalized integer difference at word {index}: scalar={scalar}, SIMD={simd}"
        );
    }
}

fn main() {
    use jxl_simd::{ScalarDescriptor, SimdDescriptor};
    use render::low_memory_pipeline::save::check_descriptor;

    let scalar = check_descriptor(ScalarDescriptor::new().unwrap(), false);
    let integers = check_descriptor(ScalarDescriptor::new().unwrap(), true);
    println!(
        "Scalar: {} boundary words and {} original integer words verified against planar quantizer",
        scalar.len(),
        integers.len()
    );
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(d) = jxl_simd::Sse42Descriptor::new() {
            let boundary = check_descriptor(d, false);
            report_boundary_difference("SSE4.2", &scalar, &boundary);
            assert_integer_identity("SSE4.2", &integers, &check_descriptor(d, true));
            println!(
                "SSE4.2: {} boundary words and {} original integer words verified against planar quantizer",
                scalar.len(),
                integers.len()
            );
        }
        if let Some(d) = jxl_simd::AvxDescriptor::new() {
            let boundary = check_descriptor(d, false);
            report_boundary_difference("AVX", &scalar, &boundary);
            assert_integer_identity("AVX", &integers, &check_descriptor(d, true));
            println!(
                "AVX: {} boundary words and {} original integer words verified against planar quantizer",
                scalar.len(),
                integers.len()
            );
        }
    }
}
