//! Single-frame continuation over real, independently coded Modular groups.
//! All options are resolved against the original image before construction.

use super::*;
use crate::modular::orchestration::{GroupwiseDecision, groupwise_decision, session::GroupSession};

pub(crate) struct PixelSession<'a> {
    image: &'a Image,
    options: EncodeOptions,
    pub(crate) groups: GroupSession<'a>,
}

impl<'a> PixelSession<'a> {
    pub(crate) fn new(image: &'a Image, options: &EncodeOptions) -> Result<Option<Self>> {
        validate_frames(image, &[crate::frame::Frame::new(image)], options)?;
        let modular = options.modular.for_pixel_metadata(image.metadata());
        if image
            .metadata()
            .extra_channels
            .iter()
            .any(|ec| ec.dim_shift != 0)
            || groupwise_decision(
                image,
                &modular,
                options.group_size_shift,
                &options.passes,
                options.responsive,
                false,
            )? != GroupwiseDecision::FrameAreas
        {
            return Ok(None);
        }
        // A configured detector is not a dictionary. Only a real patch plan
        // requires the existing reference-frame encoder; never disable patches.
        let patches = if let Some(config) = &options.source_patches {
            crate::patch_frame::prepare_source(image, config)?
        } else if let Some(config) = &options.patches {
            crate::patch_frame::prepare(image, config)?
        } else {
            None
        };
        if patches.is_some() {
            return Ok(None);
        }
        let Some(groups) = GroupSession::new(
            image,
            &modular,
            options.group_size_shift,
            &options.passes,
            options.responsive,
            options.max_output_bytes,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            image,
            options: options.clone(),
            groups,
        }))
    }

    pub(crate) fn finish(self) -> Result<Vec<u8>> {
        let sections = self.groups.finish()?;
        let sizes: Vec<_> = sections.iter().map(Vec::len).collect();
        let limit = self.options.max_output_bytes;
        let mut stream = codestream::image_header(self.image, limit)?;
        let header = codestream::frame_header(
            self.image,
            self.image,
            &sizes,
            self.options.group_size_shift,
            &self.options.passes,
            &crate::frame::FrameOptions::default(),
            true,
            false,
            limit,
        )?;
        let size = sections.iter().try_fold(
            stream
                .len()
                .checked_add(header.len())
                .ok_or(Error::Limit("output size overflow"))?,
            |n, section| {
                n.checked_add(section.len())
                    .ok_or(Error::Limit("output size overflow"))
            },
        )?;
        if size > limit {
            return Err(Error::Limit("encoded output bytes"));
        }
        stream
            .try_reserve(size - stream.len())
            .map_err(|_| Error::Limit("encoded output allocation"))?;
        stream.extend(header);
        for section in sections {
            stream.extend(section);
        }
        wrap_stream(self.image, stream, limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{DecodeOptions, Decoder, SampleFormat};
    use crate::entropy::stream::Coding;
    use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
    use crate::modular::encode::FixedTree;
    use crate::modular::encode::receipt::{ReceiptCollector, ReceiptRoute};
    use crate::modular::layout::Rect;
    use crate::modular::palette::PaletteOptions;
    use crate::modular::predictor::Predictor;
    use crate::modular::tree::{Node, Tree};
    use crate::{PixelData, PixelLayout};
    use std::sync::Arc;

    // Nine AC groups and two DC groups, without a large two-dimensional fixture.
    const WIDTH: u32 = 1025;
    const HEIGHT: u32 = 3;
    const PIXELS: usize = WIDTH as usize * HEIGHT as usize;

    fn options(threads: usize) -> EncodeOptions {
        EncodeOptions {
            threads,
            max_pixels: PIXELS as u64,
            max_output_bytes: 1 << 20,
            group_size_shift: 0,
            // Eligible groupwise requests must remain independently coded.
            shared_tree: true,
            modular: ModularOptions {
                predictor: Predictor::Gradient,
                fixed_tree: Some(FixedTree::GradientLeaf),
                rct: Some(6),
                entropy: Coding::Prefix,
                lz77: false,
                tree_samples: 1,
                max_tree_leaves: 1,
                palette_policy: Some(PaletteOptions {
                    groupwise: true,
                    palette_colors: 0,
                    channel_colors_pre_transform_percent: 0.0,
                    channel_colors_percent: 0.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn image(layout: PixelLayout, bits: u8) -> Image {
        let colors: [[u16; 4]; 4] = if bits == 8 {
            [
                [19, 241, 73, 0],
                [251, 3, 129, 255],
                [83, 7, 201, 0],
                [0, 255, 1, 128],
            ]
        } else {
            [
                [19, 65535, 32769, 0],
                [65534, 257, 4097, 65535],
                [32767, 7, 61439, 0],
                [0, 65535, 1, 32768],
            ]
        };
        let mut samples = Vec::with_capacity(PIXELS * layout.channels());
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                let index = (x / 7 + x / 128 + y) % colors.len();
                samples.extend_from_slice(&colors[index][..layout.channels()]);
            }
        }
        if layout.has_alpha() {
            assert!(samples.chunks_exact(4).any(|p| p[3] == 0 && p[0] != 0));
        }
        let pixels = if bits == 8 {
            PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
        } else {
            PixelData::U16(samples)
        };
        let image = Image::new(WIDTH, HEIGHT, layout, pixels).unwrap();
        let mut metadata = image.metadata().clone();
        // Exercise bare Level 5 framing as well as the Level 10 container.
        metadata.modular_16_bit_buffer_sufficient = bits == 8;
        image.with_metadata(metadata, Vec::new()).unwrap()
    }

    fn with_extras(image: Image, shift: u8) -> Image {
        let mut metadata = image.metadata().clone();
        metadata.modular_16_bit_buffer_sufficient = false;
        let mut depth = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
        depth.name = "depth".into();
        depth.dim_shift = shift;
        let mut selection = ExtraChannelInfo::new(ExtraChannelType::SelectionMask, 8).unwrap();
        selection.name = "selection".into();
        metadata.extra_channels.extend([depth, selection]);
        let scale = 1usize << shift;
        let count = (WIDTH as usize).div_ceil(scale) * (HEIGHT as usize).div_ceil(scale);
        // Constant shifted samples have an exact, independent full-size oracle.
        let depth = if shift == 0 {
            (0..count)
                .map(|i| [0, 1, 4095, 32769, 65535][i % 5])
                .collect()
        } else {
            vec![4095; count]
        };
        image
            .with_metadata(
                metadata,
                vec![
                    PixelData::U16(depth),
                    PixelData::U8(
                        (0..PIXELS)
                            .map(|i| if i % 3 == 0 { 0 } else { 255 })
                            .collect(),
                    ),
                ],
            )
            .unwrap()
    }

    fn original_extras(image: &Image) -> Vec<PixelData> {
        let mut extras = Vec::new();
        if image.layout().has_alpha() {
            extras.push(match image.pixels() {
                PixelData::U8(v) => PixelData::U8(v.chunks_exact(4).map(|p| p[3]).collect()),
                PixelData::U16(v) => PixelData::U16(v.chunks_exact(4).map(|p| p[3]).collect()),
                _ => unreachable!(),
            });
        }
        extras.extend_from_slice(image.extra_planes());
        extras
    }

    fn decoder(threads: usize) -> Decoder {
        Decoder::new(DecodeOptions {
            threads,
            format: SampleFormat::Native,
            ..Default::default()
        })
        .unwrap()
    }

    fn assert_original(decoder: &mut Decoder, bytes: &[u8], image: &Image, extras: &[PixelData]) {
        let decoded = decoder.decode(bytes).unwrap();
        assert!(decoded.complete);
        assert!(decoded.preview.is_none());
        assert_eq!(decoded.info.size, (WIDTH as usize, HEIGHT as usize));
        assert_eq!(
            decoded.info.bit_depth.bits_per_sample(),
            u32::from(image.bits())
        );
        assert_eq!(
            decoded.info.extra_channels.len(),
            image.metadata().extra_channels.len()
        );
        for (actual, expected) in decoded
            .info
            .extra_channels
            .iter()
            .zip(&image.metadata().extra_channels)
        {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.dim_shift, u32::from(expected.dim_shift));
            assert_eq!(
                actual.bit_depth.bits_per_sample(),
                u32::from(expected.bit_depth.bits_per_sample)
            );
        }
        assert_eq!(decoded.frames.len(), 1);
        let frame = &decoded.frames[0];
        assert!(frame.complete);
        assert_eq!(frame.index, 0);
        assert_eq!((frame.width, frame.height), decoded.info.size);
        assert_eq!(frame.layout, image.layout());
        assert_eq!(frame.duration_ticks, 0);
        assert!(frame.name.is_empty());
        // Compare both encoders to the original samples, including invisible RGB.
        // Matching the ordinary encoder's bytes or pixels alone is insufficient.
        assert_eq!(&frame.pixels, image.pixels());
        assert_eq!(frame.extra_planes.as_slice(), extras);
    }

    fn pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
    }

    fn in_scope<T: Send>(
        threads: usize,
        pool: &rayon::ThreadPool,
        work: impl FnOnce() -> T + Send,
    ) -> T {
        if threads == 1 {
            execution::serial(work)
        } else {
            assert_eq!(threads, 4);
            execution::with_pool(pool, work)
        }
    }

    fn ordinary(
        image: &Image,
        options: &EncodeOptions,
        pool: &rayon::ThreadPool,
    ) -> Result<Vec<u8>> {
        if options.threads == 1 {
            encode_serial(image, options)
        } else {
            encode_with_pool(image, options, pool)
        }
    }

    fn resumed(
        image: &Image,
        options: &EncodeOptions,
        pool: &rayon::ThreadPool,
    ) -> (Vec<u8>, usize) {
        in_scope(options.threads, pool, || {
            assert_eq!(execution::worker_count(), options.threads);
            let mut session = PixelSession::new(image, options).unwrap().unwrap();
            assert_eq!(session.groups.group_count(), 9);
            assert_eq!(session.groups.completed_groups(), 0);
            assert_eq!(session.groups.retained_bytes(), 0);
            assert_eq!(
                session.groups.group_rect(8).unwrap(),
                Rect {
                    x: 1024,
                    y: 0,
                    width: 1,
                    height: HEIGHT as usize,
                }
            );
            session.groups.encode_groups(&[]).unwrap();
            session.groups.encode_groups(&[8, 0, 4, 8]).unwrap();
            assert_eq!(session.groups.completed_groups(), 3);
            let retained = session.groups.retained_bytes();
            let edge_bytes = session.groups.group_bytes(8);
            assert!(edge_bytes.is_some_and(|n| n > 0));
            session.groups.encode_groups(&[4, 8, 0]).unwrap();
            assert_eq!(session.groups.completed_groups(), 3);
            assert_eq!(session.groups.retained_bytes(), retained);
            assert_eq!(session.groups.group_bytes(8), edge_bytes);
            session.groups.encode_groups(&[6, 2, 7, 2]).unwrap();
            assert_eq!(session.groups.completed_groups(), 6);
            session.groups.encode_groups(&[5, 1, 3, 0]).unwrap();
            assert_eq!(session.groups.completed_groups(), 9);
            session.groups.complete_remaining().unwrap();
            let retained = session.groups.retained_bytes();
            let bytes = session.finish().unwrap();
            assert!(retained < bytes.len());
            (bytes, retained)
        })
    }

    #[test]
    fn shuffled_original_groups_roundtrip_rgb_rgba_u8_u16_and_full_resolution_extras() {
        let pool = pool();
        for threads in [1, 4] {
            let options = options(threads);
            let mut decoder = decoder(threads);
            for layout in [PixelLayout::Rgb, PixelLayout::Rgba] {
                for bits in [8, 16] {
                    for extras in [false, true] {
                        let mut image = image(layout, bits);
                        if extras {
                            image = with_extras(image, 0);
                        }
                        let expected_extras = original_extras(&image);
                        let expected = ordinary(&image, &options, &pool).unwrap();
                        let (actual, _) = resumed(&image, &options, &pool);
                        if bits == 8 && !extras {
                            assert!(actual.starts_with(b"\xff\x0a"));
                        } else {
                            assert!(actual.starts_with(b"\0\0\0\x0cJXL \x0d\x0a\x87\x0a"));
                        }
                        assert_original(&mut decoder, &expected, &image, &expected_extras);
                        assert_original(&mut decoder, &actual, &image, &expected_extras);
                    }
                }
            }
        }
    }

    #[test]
    fn shuffled_stream_id_sensitive_tree_decodes_in_original_frame_coordinates() {
        let pool = pool();
        let image = image(PixelLayout::Rgb, 8);
        for threads in [1, 4] {
            let mut options = options(threads);
            options.modular.fixed_tree = None;
            options.modular.custom_tree = Some(
                Tree::new(vec![
                    Node::Split {
                        property: 1,
                        threshold: 18 + 3 * 2 + 4,
                        greater: 1,
                        less_equal: 2,
                    },
                    Node::Leaf {
                        predictor: Predictor::Zero,
                        offset: 17,
                        multiplier: 1,
                    },
                    Node::Leaf {
                        predictor: Predictor::Gradient,
                        offset: 0,
                        multiplier: 1,
                    },
                ])
                .unwrap(),
            );
            let expected = ordinary(&image, &options, &pool).unwrap();
            let (actual, _) = resumed(&image, &options, &pool);
            let mut decoder = decoder(threads);
            assert_original(&mut decoder, &expected, &image, &[]);
            assert_original(&mut decoder, &actual, &image, &[]);
        }
    }

    #[test]
    fn global_palette_and_shifted_extras_decline_without_consuming_fallback_receipts() {
        let pool = pool();
        for threads in [1, 4] {
            let mut decoder = decoder(threads);
            let image = image(PixelLayout::Rgba, 8);
            let mut global = options(threads);
            global.global_palette = true;
            let policy = global.modular.palette_policy.as_mut().unwrap();
            policy.groupwise = false;
            policy.palette_colors = 16;
            policy.force_palette = true;
            policy.channel_colors_pre_transform_percent = 100.0;
            let receipt = Arc::new(ReceiptCollector::default());
            global.modular.representation_receipt = Some(Arc::clone(&receipt));
            assert!(PixelSession::new(&image, &global).unwrap().is_none());
            assert!(receipt.get().is_none());
            let bytes = ordinary(&image, &global, &pool).unwrap();
            let receipt = receipt.get().unwrap();
            assert_eq!(receipt.route, ReceiptRoute::FullImage);
            assert!(!receipt.global_tuple_palettes.is_empty());
            assert_original(&mut decoder, &bytes, &image, &original_extras(&image));

            for shift in [1, 3] {
                let image = with_extras(image.clone(), shift);
                let mut expected_extras = original_extras(&image);
                expected_extras[1] = PixelData::U16(vec![4095; PIXELS]);
                let mut options = options(threads);
                let receipt = Arc::new(ReceiptCollector::default());
                options.modular.representation_receipt = Some(Arc::clone(&receipt));
                assert!(PixelSession::new(&image, &options).unwrap().is_none());
                assert!(receipt.get().is_none());
                let bytes = ordinary(&image, &options, &pool).unwrap();
                assert_eq!(receipt.get().unwrap().route, ReceiptRoute::Groupwise);
                assert_original(&mut decoder, &bytes, &image, &expected_extras);
            }
        }
    }

    #[test]
    fn output_limits_include_image_header_toc_and_level10_container() {
        let pool = pool();
        for threads in [1, 4] {
            let mut decoder = decoder(threads);
            for bits in [8, 16] {
                let image = image(PixelLayout::Rgb, bits);
                let options = options(threads);
                let (bytes, payload_bytes) = resumed(&image, &options, &pool);
                assert_original(&mut decoder, &bytes, &image, &[]);
                let prefix_bytes = codestream::container_prefix(&image, 0).unwrap().len();
                assert_eq!(prefix_bytes == 0, bits == 8);
                let codestream_bytes = bytes.len() - prefix_bytes;
                assert!(payload_bytes < codestream_bytes);
                let mut limits = vec![payload_bytes, codestream_bytes - 1, bytes.len() - 1];
                limits.sort_unstable();
                limits.dedup();
                for limit in limits {
                    let mut limited = options.clone();
                    limited.max_output_bytes = limit;
                    in_scope(threads, &pool, || {
                        let mut session = PixelSession::new(&image, &limited).unwrap().unwrap();
                        // The group cap alone admits these sections. Framing must
                        // independently reject the complete, oversized output.
                        session.groups.complete_remaining().unwrap();
                        assert_eq!(session.groups.retained_bytes(), payload_bytes);
                        assert!(matches!(
                            session.finish(),
                            Err(Error::Limit("encoded output bytes"))
                        ));
                    });
                    assert!(ordinary(&image, &limited, &pool).is_err());
                }
                let mut exact = options.clone();
                exact.max_output_bytes = bytes.len();
                let (actual, _) = resumed(&image, &exact, &pool);
                assert_eq!(actual.len(), bytes.len());
                assert_original(&mut decoder, &actual, &image, &[]);
                let ordinary = ordinary(&image, &exact, &pool).unwrap();
                assert_original(&mut decoder, &ordinary, &image, &[]);
            }
        }
    }

    #[test]
    fn invalid_incomplete_and_poisoned_pixel_sessions_cannot_emit_a_codestream() {
        let pool = pool();
        let image = image(PixelLayout::Rgb, 8);
        for threads in [1, 4] {
            let options = options(threads);
            in_scope(threads, &pool, || {
                let session = PixelSession::new(&image, &options).unwrap().unwrap();
                assert!(session.finish().is_err());
                let mut session = PixelSession::new(&image, &options).unwrap().unwrap();
                session.groups.encode_groups(&[8]).unwrap();
                assert!(session.finish().is_err());
                let mut session = PixelSession::new(&image, &options).unwrap().unwrap();
                session.groups.encode_groups(&[0]).unwrap();
                assert!(session.groups.encode_groups(&[8, 9]).is_err());
                assert!(session.groups.complete_remaining().is_err());
                assert!(session.finish().is_err());

                let mut limited = options.clone();
                limited.max_output_bytes = 1;
                let mut session = PixelSession::new(&image, &limited).unwrap().unwrap();
                assert!(session.groups.encode_groups(&[0]).is_err());
                assert!(session.finish().is_err());
            });
            let mut invalid = options.clone();
            invalid.max_pixels = PIXELS as u64 - 1;
            assert!(matches!(
                PixelSession::new(&image, &invalid),
                Err(Error::Limit("pixel count"))
            ));
            let mut invalid = options.clone();
            invalid.max_output_bytes = 0;
            assert!(matches!(
                PixelSession::new(&image, &invalid),
                Err(Error::Limit("encoded output bytes"))
            ));
            let mut invalid = options.clone();
            invalid.threads = 0;
            assert!(matches!(
                PixelSession::new(&image, &invalid),
                Err(Error::InvalidInput(_))
            ));
        }
    }
}
