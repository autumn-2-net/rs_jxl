//! Contexts retain workers, never per-input pixels, references, or failed jobs.

use rs_jxl::codec::{
    self, EncodeJob, EncodeResult, EncodeRoute, Encoder, SearchDetails, SourceRequest, SourceSearch,
};
use rs_jxl::decode::{DecodeOptions, Decoder, SampleFormat};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::Frame;
use rs_jxl::metadata::Animation;
use rs_jxl::modular::{encode::FixedTree, predictor::Predictor};
use rs_jxl::scheduler::{
    CandidateParams, NativeAcceptance, NativeBackendOptions, NativeResourcePolicy,
    adapter::CapabilityGap, pilot::PilotOptions,
};
use rs_jxl::{
    EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, jpeg, multiframe,
};

fn manual(threads: usize) -> EncodeJob {
    EncodeJob {
        options: EncodeOptions {
            threads,
            max_output_bytes: 4 << 20,
            modular: ModularOptions {
                predictor: Predictor::Zero,
                fixed_tree: Some(FixedTree::Zero),
                rct: None,
                lz77: false,
                entropy: Coding::Prefix,
                tree_samples: 1,
                max_tree_leaves: 1,
                ..Default::default()
            },
            ..Default::default()
        },
        source: None,
    }
}

fn source(threads: usize, effort: u8, search: SourceSearch) -> EncodeJob {
    EncodeJob {
        options: EncodeOptions {
            threads,
            ..Default::default()
        },
        source: Some(SourceRequest {
            params: CandidateParams::default().backend(effort).unwrap(),
            search,
            backend: NativeBackendOptions::default(),
        }),
    }
}

fn images() -> Vec<Image> {
    vec![
        Image::new(
            3,
            2,
            PixelLayout::Rgba,
            PixelData::U8(vec![
                17, 93, 241, 0, 99, 33, 4, 255, 1, 127, 254, 0, 255, 0, 127, 128, 9, 77, 31, 0, 42,
                13, 71, 255,
            ]),
        )
        .unwrap(),
        Image::new(
            2,
            1,
            PixelLayout::Rgba,
            PixelData::U16(vec![65535, 32768, 1, 0, 65534, 0, 32767, 65535]),
        )
        .unwrap(),
        Image::new(
            7,
            3,
            PixelLayout::Gray,
            PixelData::U8((0..21).map(|i| (i * 37) as u8).collect()),
        )
        .unwrap(),
    ]
}

fn assert_pixels(decoder: &mut Decoder, bytes: &[u8], image: &Image) {
    let decoded = decoder.decode(bytes).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), 1);
    let frame = &decoded.frames[0];
    assert_eq!(
        (frame.width, frame.height),
        (image.width() as usize, image.height() as usize)
    );
    assert_eq!(frame.layout, image.layout());
    assert_eq!(&frame.pixels, image.pixels());
}

fn assert_report(result: &EncodeResult, image: &Image, route: EncodeRoute) {
    let report = &result.report;
    assert_eq!(report.route, route);
    assert_eq!(
        report.pixels,
        u64::from(image.width()) * u64::from(image.height())
    );
    assert_eq!(report.encoded_bytes, result.bytes.len());
    assert!(!result.bytes.is_empty());
    assert_eq!(
        report.bpp(),
        result.bytes.len() as f64 * 8.0 / report.pixels as f64
    );
    assert!(report.codec_seconds.is_finite() && report.codec_seconds >= 0.0);
    if let Some(rate) = report.mpix_per_second() {
        assert!(rate.is_finite() && rate > 0.0);
    }
    assert!(report.full_encodes >= 1);
    if route == EncodeRoute::Manual {
        assert!(report.artifact.is_none());
        assert!(report.adaptations.is_empty());
        assert!(report.progressive.is_none());
    } else {
        let artifact = report.artifact.as_ref().unwrap();
        assert_eq!(
            report.effective_params.as_ref(),
            Some(&artifact.effective_params)
        );
        assert_eq!(report.implementation, artifact.implementation);
    }
    match &report.details {
        SearchDetails::None => {
            assert_eq!(report.full_encodes, 1);
            assert_eq!(report.probe_encodes, 0);
            assert_eq!(report.winner_index, None);
        }
        SearchDetails::Pixel(details) => {
            assert_eq!(report.full_encodes, details.full_encodes);
            assert_eq!(report.probe_encodes, details.probe_encodes);
            assert_eq!(report.winner_index, Some(details.winner_index));
            assert_eq!(
                details.trials[details.winner_index].bytes,
                Some(result.bytes.len())
            );
        }
        SearchDetails::Tectonic(details) => {
            assert_eq!(report.full_encodes, details.full_encodes);
            assert_eq!(report.winner_index, Some(details.winner_index));
            assert_eq!(details.final_bytes, result.bytes.len());
        }
    }
}

#[test]
fn manual_reuse_matches_one_shot_original_pixels_and_recovers_from_limits() {
    for threads in [1, 2] {
        let mut encoder = Encoder::new(threads).unwrap();
        let mut decoder = Decoder::new(DecodeOptions {
            threads,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(encoder.threads(), threads);
        let job = manual(threads);
        let mut retained = Vec::new();
        for _ in 0..2 {
            for image in images() {
                let mut limited = job.clone();
                limited.options.max_output_bytes = 1;
                assert!(matches!(
                    encoder.encode(&image, &limited),
                    Err(Error::Limit(_))
                ));
                let result = encoder.encode(&image, &job).unwrap();
                assert_report(&result, &image, EncodeRoute::Manual);
                assert!(result.report.effective_params.is_none());
                assert_pixels(&mut decoder, &result.bytes, &image);
                assert_pixels(
                    &mut decoder,
                    &codec::encode_job(&image, &job).unwrap().bytes,
                    &image,
                );
                assert_pixels(
                    &mut decoder,
                    &rs_jxl::encode(&image, &job.options).unwrap(),
                    &image,
                );
                retained.push((result, image));
            }
        }
        drop(encoder);
        for (result, image) in retained {
            assert_report(&result, &image, EncodeRoute::Manual);
            assert_pixels(&mut decoder, &result.bytes, &image);
        }
    }
}

#[test]
fn every_encoder_entry_rejects_mismatched_threads_without_poisoning_workers() {
    for threads in [0, 257, usize::MAX] {
        assert!(matches!(Encoder::new(threads), Err(Error::InvalidInput(_))));
    }
    for threads in [1, 2] {
        let mut encoder = Encoder::new(threads).unwrap();
        let image = images().remove(0);
        let wrong = manual(3 - threads);
        assert!(matches!(
            encoder.encode(&image, &wrong),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            encoder.encode_frames(&image, &[Frame::new(&image)], &wrong.options),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            encoder.encode_auto(
                std::slice::from_ref(&image),
                &[1],
                &wrong.options,
                &Default::default()
            ),
            Err(Error::InvalidInput(_))
        ));
        let jpeg_options = jpeg::TranscodeOptions {
            threads: 3 - threads,
            ..Default::default()
        };
        for search in [None, Some(Default::default())] {
            assert!(matches!(
                encoder.transcode_jpeg(&tiny_jpeg(), &jpeg_options, search),
                Err(Error::InvalidInput(_))
            ));
        }
        let mut invalid = source(threads, 3, SourceSearch::Disabled);
        invalid.source.as_mut().unwrap().params.speed_tier = 10;
        assert!(matches!(
            encoder.encode(&image, &invalid),
            Err(Error::InvalidInput(_))
        ));
        let invalid_scout = source(threads, 3, SourceSearch::Scout);
        assert!(matches!(
            encoder.encode(&image, &invalid_scout),
            Err(Error::InvalidInput(_))
        ));
        let bytes = encoder.encode(&image, &manual(threads)).unwrap().bytes;
        let mut decoder = Decoder::new(Default::default()).unwrap();
        assert_pixels(&mut decoder, &bytes, &image);
    }
}

#[test]
fn explicit_source_effort_is_not_implicit_search_and_reports_are_per_call() {
    let image = Image::new(
        8,
        8,
        PixelLayout::Gray,
        PixelData::U8((0..64).map(|i| (i % 3) as u8).collect()),
    )
    .unwrap();
    for threads in [1, 2] {
        let mut encoder = Encoder::new(threads).unwrap();
        let mut decoder = Decoder::new(DecodeOptions {
            threads,
            ..Default::default()
        })
        .unwrap();
        for effort in [3, 9, 11] {
            let result = encoder
                .encode(&image, &source(threads, effort, SourceSearch::Disabled))
                .unwrap();
            assert_report(&result, &image, EncodeRoute::Candidate);
            assert!(matches!(result.report.details, SearchDetails::None));
            assert_eq!(
                result.report.effective_params.as_ref().unwrap().speed_tier,
                10 - effort as i8
            );
            assert_pixels(&mut decoder, &result.bytes, &image);
        }
        for (effort, search, route) in [
            (3, SourceSearch::Auto, EncodeRoute::Search),
            (9, SourceSearch::Scout, EncodeRoute::Scout),
            (
                9,
                SourceSearch::Pilot(Default::default()),
                EncodeRoute::Pilot,
            ),
            (11, SourceSearch::Auto, EncodeRoute::Tectonic),
        ] {
            let job = source(threads, effort, search);
            let result = encoder.encode(&image, &job).unwrap();
            assert_report(&result, &image, route);
            assert!(result.report.winner_index.is_some());
            assert!(result.report.effective_params.is_some());
            assert_pixels(&mut decoder, &result.bytes, &image);
            let fresh = codec::encode_job(&image, &job).unwrap();
            assert_report(&fresh, &image, route);
            assert_pixels(&mut decoder, &fresh.bytes, &image);
            // A subsequent manual job must not inherit search counters or params.
            let plain = encoder.encode(&image, &manual(threads)).unwrap();
            assert_report(&plain, &image, EncodeRoute::Manual);
            assert!(plain.report.effective_params.is_none());
        }
    }
}

#[test]
fn candidate_report_preserves_proxy_gaps_resources_and_requested_receipt() {
    let image = images().remove(2);
    for threads in [1, 2] {
        let mut encoder = Encoder::new(threads).unwrap();
        let mut job = source(threads, 3, SourceSearch::Disabled);
        job.options.max_output_bytes = 1 << 20;
        job.source.as_mut().unwrap().backend = NativeBackendOptions {
            acceptance: NativeAcceptance::AllowReportedGaps,
            patches_off: true,
            record_representations: true,
            resources: NativeResourcePolicy {
                max_tree_leaves: Some(64),
            },
            ..Default::default()
        };
        let result = encoder.encode(&image, &job).unwrap();
        assert_report(&result, &image, EncodeRoute::Candidate);
        let artifact = result.report.artifact.as_ref().unwrap();
        assert!(artifact.gaps.contains(&CapabilityGap::ForcedPatchesOff));
        assert_eq!(artifact.options.threads, threads);
        assert_eq!(
            artifact.options.max_output_bytes,
            job.options.max_output_bytes
        );
        assert_eq!(artifact.options.modular.max_tree_leaves, 64);
        assert_eq!(artifact.resources.max_tree_leaves, 64);
        let collector = artifact
            .options
            .modular
            .representation_receipt
            .as_ref()
            .unwrap();
        let receipt = collector.get().unwrap();
        assert_eq!(
            (receipt.input_width, receipt.input_height),
            (image.width(), image.height())
        );
        assert_eq!(receipt.input_channels, image.channel_count());
        job.source.as_mut().unwrap().backend.record_representations = false;
        let next = encoder.encode(&image, &job).unwrap();
        assert!(
            next.report
                .artifact
                .as_ref()
                .unwrap()
                .options
                .modular
                .representation_receipt
                .is_none()
        );
        drop(encoder);
        assert_eq!(collector.get(), Some(receipt));
        let mut decoder = Decoder::new(Default::default()).unwrap();
        assert_pixels(&mut decoder, &result.bytes, &image);
        assert_pixels(&mut decoder, &next.bytes, &image);
    }
}

#[test]
fn search_reports_retain_all_adaptations_and_winning_receipts_after_context_drop() {
    let image = images().remove(2);
    for threads in [1, 2] {
        let mut encoder = Encoder::new(threads).unwrap();
        let mut retained = Vec::new();
        for (effort, search, route) in [
            (3, SourceSearch::Auto, EncodeRoute::Search),
            (9, SourceSearch::Scout, EncodeRoute::Scout),
            (
                9,
                SourceSearch::Pilot(PilotOptions {
                    record_representations: true,
                    ..Default::default()
                }),
                EncodeRoute::Pilot,
            ),
            (11, SourceSearch::Auto, EncodeRoute::Tectonic),
        ] {
            let mut job = source(threads, effort, search);
            if !matches!(search, SourceSearch::Pilot(_)) {
                job.source.as_mut().unwrap().backend.record_representations = true;
            }
            let result = encoder.encode(&image, &job).unwrap();
            assert_report(&result, &image, route);
            assert!(result.report.adaptations.len() >= result.report.full_encodes);
            if route == EncodeRoute::Tectonic {
                assert_eq!(result.report.adaptations.len(), result.report.full_encodes);
            }
            assert!(
                result
                    .report
                    .adaptations
                    .iter()
                    .all(|adaptation| adaptation.representation.is_some())
            );
            let artifact = result.report.artifact.as_ref().unwrap();
            let receipt = artifact
                .options
                .modular
                .representation_receipt
                .as_ref()
                .unwrap();
            assert!(receipt.get().is_some());
            assert!(result.report.adaptations.iter().any(|adaptation| {
                std::sync::Arc::ptr_eq(adaptation.representation.as_ref().unwrap(), receipt)
                    && adaptation.effective == artifact.effective_params
                    && adaptation.resources == artifact.resources
            }));
            let plain = encoder.encode(&image, &manual(threads)).unwrap();
            assert_report(&plain, &image, EncodeRoute::Manual);
            retained.push(result);
        }
        drop(encoder);
        let mut decoder = Decoder::new(Default::default()).unwrap();
        for result in retained {
            let artifact = result.report.artifact.as_ref().unwrap();
            let receipt = artifact
                .options
                .modular
                .representation_receipt
                .as_ref()
                .unwrap()
                .get()
                .unwrap();
            assert_eq!(
                (receipt.input_width, receipt.input_height),
                (image.width(), image.height())
            );
            assert_pixels(&mut decoder, &result.bytes, &image);
        }
    }
}

#[test]
fn decoder_option_updates_are_atomic_and_do_not_recreate_per_input_state() {
    let image = images().remove(1);
    let bytes = codec::encode_job(&image, &manual(1)).unwrap().bytes;
    for threads in [1, 2] {
        let original = DecodeOptions {
            threads,
            ..Default::default()
        };
        let mut decoder = Decoder::new(original.clone()).unwrap();
        let mut limited = original.clone();
        limited.limits.max_pixels = 1;
        decoder.set_options(limited.clone()).unwrap();
        assert!(matches!(decoder.decode(&bytes), Err(Error::Limit(_))));
        for invalid in [
            DecodeOptions {
                threads: 3 - threads,
                ..original.clone()
            },
            DecodeOptions {
                format: SampleFormat::U16Bits(17),
                ..original.clone()
            },
        ] {
            assert!(matches!(
                decoder.set_options(invalid),
                Err(Error::InvalidInput(_))
            ));
            assert_eq!(decoder.options().threads, threads);
            assert_eq!(decoder.options().format, SampleFormat::Native);
            assert_eq!(decoder.options().limits.max_pixels, 1);
            assert!(matches!(decoder.decode(&bytes), Err(Error::Limit(_))));
        }
        decoder.set_options(original.clone()).unwrap();
        assert_pixels(&mut decoder, &bytes, &image);
        let converted = DecodeOptions {
            format: SampleFormat::U8,
            ..original.clone()
        };
        decoder.set_options(converted.clone()).unwrap();
        let actual = decoder.decode(&bytes).unwrap();
        let fresh = rs_jxl::decode::decode(&bytes, &converted).unwrap();
        assert!(matches!(actual.frames[0].pixels, PixelData::U8(_)));
        assert_eq!(actual.frames[0].pixels, fresh.frames[0].pixels);
        assert!(decoder.decode(b"not a JXL stream").is_err());
        decoder.set_options(original).unwrap();
        assert_pixels(&mut decoder, &bytes, &image);
    }
}

fn animation() -> Vec<Image> {
    (0..3)
        .map(|index| {
            let mut pixels = [17, 93, 241, 0].repeat(12);
            pixels[index * 4] ^= 127;
            let image = Image::new(4, 3, PixelLayout::Rgba, PixelData::U8(pixels)).unwrap();
            let mut metadata = image.metadata().clone();
            metadata.animation = Some(Animation::default());
            image.with_metadata(metadata, vec![]).unwrap()
        })
        .collect()
}

fn assert_sequence(decoder: &mut Decoder, bytes: &[u8], images: &[Image], durations: &[u32]) {
    let decoded = decoder.decode(bytes).unwrap();
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), images.len());
    for ((frame, image), duration) in decoded.frames.iter().zip(images).zip(durations) {
        assert_eq!(&frame.pixels, image.pixels());
        assert_eq!(frame.duration_ticks, *duration);
    }
}

#[test]
fn frame_and_auto_entries_reuse_workers_but_not_previous_sequence_references() {
    let images = animation();
    let durations = [2, 3, 5];
    let plan = multiframe::plan(&images, &durations, multiframe::Reference::Previous).unwrap();
    let frames: Vec<_> = plan.iter().map(|frame| frame.as_frame()).collect();
    let still = self::images().remove(1);
    let auto = multiframe::AutoOptions {
        allow_mask: false,
        allow_multi_rect: false,
        ..Default::default()
    };
    for threads in [1, 2] {
        let mut encoder = Encoder::new(threads).unwrap();
        let mut decoder = Decoder::new(DecodeOptions {
            threads,
            ..Default::default()
        })
        .unwrap();
        let job = manual(threads);
        for _ in 0..2 {
            assert!(
                encoder
                    .encode_frames(&images[0], &[], &job.options)
                    .is_err()
            );
            assert!(
                encoder
                    .encode_auto(&images, &[1], &job.options, &auto)
                    .is_err()
            );
            let bytes = encoder
                .encode_frames(&images[0], &frames, &job.options)
                .unwrap();
            assert_sequence(&mut decoder, &bytes, &images, &durations);
            let fresh = rs_jxl::encode_frames(&images[0], &frames, &job.options).unwrap();
            assert_sequence(&mut decoder, &fresh, &images, &durations);
            let bytes = encoder.encode(&still, &job).unwrap().bytes;
            assert_pixels(&mut decoder, &bytes, &still);
            let automatic = encoder
                .encode_auto(&images, &durations, &job.options, &auto)
                .unwrap();
            assert_eq!(automatic.frames.len(), images.len());
            for report in &automatic.frames {
                assert!(report.winner_index < report.candidates.len());
                assert!(report.candidates[report.winner_index].encoded_bytes > 0);
            }
            assert_sequence(&mut decoder, &automatic.bytes, &images, &durations);
            let fresh = multiframe::encode_auto(&images, &durations, &job.options, &auto).unwrap();
            assert_sequence(&mut decoder, &fresh.bytes, &images, &durations);
        }
    }
}

fn tiny_jpeg() -> Vec<u8> {
    // Same one-block baseline fixture as python/tests/test_bindings.py.
    let mut bytes = vec![0xff, 0xd8, 0xff, 0xdb, 0, 67, 0];
    bytes.extend([16; 64]);
    bytes.extend([0xff, 0xc0, 0, 11, 8, 0, 1, 0, 1, 1, 1, 0x11, 0]);
    bytes.extend([0xff, 0xc4, 0, 38]);
    for selector in [0, 0x10] {
        bytes.extend([selector, 1]);
        bytes.extend([0; 16]);
    }
    bytes.extend([0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0, 0x3f, 0xff, 0xd9]);
    bytes
}

#[test]
fn jpeg_entries_reconstruct_original_bytes_after_errors_and_pixel_jobs() {
    let original = tiny_jpeg();
    let image = images().remove(0);
    for threads in [1, 2] {
        let options = jpeg::TranscodeOptions {
            effort: 3,
            threads,
            ..Default::default()
        };
        let mut encoder = Encoder::new(threads).unwrap();
        let mut decoder = Decoder::new(DecodeOptions {
            threads,
            ..Default::default()
        })
        .unwrap();
        let mut retained = Vec::new();
        for search in [None, Some(Default::default())] {
            assert!(
                encoder
                    .transcode_jpeg(b"not a JPEG", &options, search)
                    .is_err()
            );
            let result = encoder.transcode_jpeg(&original, &options, search).unwrap();
            assert!(result.codec_seconds.is_finite() && result.codec_seconds >= 0.0);
            assert!(result.winner_index < result.trials.len());
            let winner = &result.trials[result.winner_index];
            assert_eq!(winner.bytes, result.bytes.len());
            assert_eq!(winner.backend_effort, result.backend_effort);
            if search.is_none() {
                assert_eq!(result.trials.len(), 1);
                assert_eq!(result.backend_effort, 3);
            }
            let pixels = encoder.encode(&image, &manual(threads)).unwrap().bytes;
            assert!(decoder.reconstruct_jpeg(&pixels).is_err());
            assert_eq!(decoder.reconstruct_jpeg(&result.bytes).unwrap(), original);
            assert_pixels(&mut decoder, &pixels, &image);
            let fresh = codec::transcode_jpeg(&original, &options, search).unwrap();
            assert_eq!(decoder.reconstruct_jpeg(&fresh.bytes).unwrap(), original);
            retained.push(result.bytes);
        }
        drop(encoder);
        for bytes in retained {
            assert_eq!(decoder.reconstruct_jpeg(&bytes).unwrap(), original);
        }
    }
}
