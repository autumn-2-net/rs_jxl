use super::super::{Prepared, with_threads};
use super::*;
use crate::{
    decode::{DecodeOptions, reconstruct_jpeg},
    jpeg::{archival, coefficient_reader, transcode_reference_tests::fixtures},
};

fn fixture(name: &str) -> (Vec<u8>, JpegData) {
    let bytes = fixtures::fixtures()
        .into_iter()
        .find(|(label, _)| label == name)
        .expect("known JPEG fixture")
        .1;
    let jpeg = archival::read_input(&bytes, &Default::default()).unwrap();
    (bytes, jpeg)
}

fn geometry(jpeg: &JpegData) -> FrameGeometry {
    let layout = super::super::TranscodeLayout::new(jpeg).unwrap();
    FrameGeometry::new(jpeg.width, jpeg.height, layout.subsampling()).unwrap()
}

fn key<'a>(jpeg: &'a JpegData, options: &Options) -> AcKey<'a> {
    AcKey::new(jpeg, geometry(jpeg), options).unwrap()
}

fn assert_reconstruction(encoded: &[u8], original: &[u8], jpeg: &JpegData) {
    let restored = reconstruct_jpeg(encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(restored, original);
    let coefficients = coefficient_reader::read_coefficients(&restored).unwrap();
    assert_eq!(
        (coefficients.width, coefficients.height),
        (jpeg.width, jpeg.height)
    );
    assert_eq!(coefficients.components, jpeg.components);
    assert_eq!(coefficients.quant, jpeg.quant);
}

#[test]
fn search_cache_fragments_preserve_every_alignment_tail_and_destination_bound() {
    for bits in [0usize, 1, 7, 8, 9, 55, 56, 57, 63, 64, 65, 137] {
        let mut source = BitWriter::new(64);
        for bit in 0..bits {
            source.write(1, u64::from(bit % 5 < 2)).unwrap();
        }
        let fragment = Fragment::finish(source).unwrap();
        assert_eq!(fragment.bits, bits);
        assert_eq!(fragment.bytes.len(), bits.div_ceil(8));
        for prefix in 0..8u8 {
            let mut actual = BitWriter::new(64);
            let mut expected = BitWriter::new(64);
            let value = (1u64 << prefix) - 1;
            actual.write(prefix, value).unwrap();
            expected.write(prefix, value).unwrap();
            fragment.append(&mut actual).unwrap();
            for bit in 0..bits {
                expected.write(1, u64::from(bit % 5 < 2)).unwrap();
            }
            // The following field must overwrite fragment padding, not follow it.
            actual.write(5, 21).unwrap();
            expected.write(5, 21).unwrap();
            assert_eq!(actual.bits_written(), expected.bits_written());
            actual.zero_pad_to_byte().unwrap();
            expected.zero_pad_to_byte().unwrap();
            assert_eq!(actual.into_bytes().unwrap(), expected.into_bytes().unwrap());

            let required = (usize::from(prefix) + bits).div_ceil(8);
            let mut exact = BitWriter::new(required);
            exact.write(prefix, value).unwrap();
            fragment.append(&mut exact).unwrap();
            if required > 0 && bits >= 8 {
                let mut short = BitWriter::new(required - 1);
                short.write(prefix, value).unwrap();
                assert!(matches!(fragment.append(&mut short), Err(Error::Limit(_))));
            }
        }
    }
}

#[test]
fn search_cache_payload_has_one_aggregate_bit_budget() {
    use crate::entropy::stream::Token;

    let orders = Dct8Orders::default();
    let entropy = CoefficientEntropy::source(
        vec![
            (0..128).map(|i| Token::literal(0, i % 13)).collect(),
            (0..128).map(|i| Token::literal(0, i % 7)).collect(),
        ],
        1,
        SpeedTier::Falcon,
        0,
        256,
    )
    .unwrap();
    let payload = AcPayload::build(&orders, &entropy, 2, 4096).unwrap();
    let bits = payload.global.bits + payload.groups.iter().map(|group| group.bits).sum::<usize>();
    let bytes = bits.div_ceil(8);
    assert!(AcPayload::build(&orders, &entropy, 2, bytes).is_ok());
    assert!(matches!(
        AcPayload::build(&orders, &entropy, 2, bytes - 1),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        AcPayload::build(&orders, &entropy, 1, 4096),
        Err(Error::Encoding(_))
    ));
    for prefix in 0..8u8 {
        let mut actual = BitWriter::new(4096);
        let mut expected = BitWriter::new(4096);
        actual.write(prefix, 0).unwrap();
        expected.write(prefix, 0).unwrap();
        payload.write_global(&mut actual).unwrap();
        entropy::write_single_histogram_group(&mut expected, 2).unwrap();
        entropy::write_orders(&orders, &mut expected).unwrap();
        entropy.write_model(&mut expected).unwrap();
        for index in 0..2 {
            payload.write_group(index, &mut actual).unwrap();
            entropy.write_group(index, &mut expected).unwrap();
        }
        assert_eq!(actual.bits_written(), expected.bits_written());
        actual.zero_pad_to_byte().unwrap();
        expected.zero_pad_to_byte().unwrap();
        assert_eq!(actual.into_bytes().unwrap(), expected.into_bytes().unwrap());
    }
}

#[test]
fn search_cache_keys_resolve_tiny_order_gates_and_all_ac_options() {
    let (_, tiny) = fixture("gray_seq_17x9");
    let (_, normal) = fixture("rgb_seq_33x25");
    let options = Options::default();
    for jpeg in [&tiny, &normal] {
        let falcon = key(jpeg, &options);
        let hare = key(
            jpeg,
            &Options {
                tier: SpeedTier::Hare,
                ..options.clone()
            },
        );
        let squirrel = key(
            jpeg,
            &Options {
                tier: SpeedTier::Squirrel,
                ..options.clone()
            },
        );
        assert_eq!(hare, squirrel);
        assert_eq!(falcon == hare, jpeg.width >= 33);
        let kitten = key(
            jpeg,
            &Options {
                tier: SpeedTier::Kitten,
                ..options.clone()
            },
        );
        let tortoise = key(
            jpeg,
            &Options {
                tier: SpeedTier::Tortoise,
                ..options.clone()
            },
        );
        let glacier = key(
            jpeg,
            &Options {
                tier: SpeedTier::Glacier,
                ..options.clone()
            },
        );
        assert_ne!(squirrel, kitten);
        assert_ne!(kitten, tortoise);
        assert_eq!(tortoise, glacier);
        assert_eq!(
            falcon,
            key(
                jpeg,
                &Options {
                    order: OrderPolicy::Adaptive,
                    ..options.clone()
                }
            )
        );
        for changed in [
            Options {
                cfl: CflMode::Disabled,
                ..options.clone()
            },
            Options {
                order: OrderPolicy::Default,
                ..options.clone()
            },
            Options {
                decoding_speed: 1,
                ..options.clone()
            },
            Options {
                max_coefficients: options.max_coefficients - 1,
                ..options.clone()
            },
            Options {
                max_tokens: options.max_tokens - 1,
                ..options.clone()
            },
            Options {
                max_output_bytes: options.max_output_bytes - 1,
                ..options.clone()
            },
        ] {
            assert_ne!(falcon, key(jpeg, &changed));
        }
        // These affect auxiliary learning, executor resources or jbrd, not AC.
        assert_eq!(
            falcon,
            key(
                jpeg,
                &Options {
                    threads: 3,
                    sample_limit: 1234,
                    reconstruction_quality: 9,
                    cooperative_ma: Some(crate::scheduler::WorkerLimits::default()),
                    ..options.clone()
                }
            )
        );
        assert_eq!(
            key(
                jpeg,
                &Options {
                    decoding_speed: 1,
                    ..options.clone()
                }
            ),
            key(
                jpeg,
                &Options {
                    decoding_speed: 4,
                    ..options.clone()
                }
            ),
        );
    }
    let cloned = normal.clone();
    assert_ne!(key(&normal, &options), key(&cloned, &options));
    let changed_geometry =
        FrameGeometry::new(257, normal.height, geometry(&normal).subsampling()).unwrap();
    assert_ne!(
        key(&normal, &options),
        AcKey::new(&normal, changed_geometry, &options).unwrap()
    );
    let subsampled = ChromaSubsampling::from_sampling([2, 1, 1], [2, 1, 1]).unwrap();
    let changed_geometry = FrameGeometry::new(normal.width, normal.height, subsampled).unwrap();
    assert_ne!(
        key(&normal, &options),
        AcKey::new(&normal, changed_geometry, &options).unwrap()
    );
}

#[test]
fn search_cache_prefix_matches_uncached_coefficients_and_original_bytes() {
    for (name, threads) in [
        ("gray_seq_17x9", 1),
        ("rgb_seq_257x17_rst5", 2),
        ("ycbcr444_prog_257x17", 1),
        ("ycbcr420_prog_33x19_rst3", 2),
    ] {
        let (original, jpeg) = fixture(name);
        let settings = Options {
            threads,
            ..Options::default()
        };
        with_threads(&settings, |pool| {
            let prepared = Prepared::new(&jpeg, &settings)?;
            let mut cache = SearchCache::new(&jpeg);
            for backend in [3i8, 5, 7, 8, 9, 10] {
                let candidate = Options {
                    tier: SpeedTier::try_from(10 - backend)?,
                    ..settings.clone()
                };
                let expected = prepared.encode(&jpeg, &candidate, pool)?;
                let actual = prepared.encode_with_cache(&jpeg, &candidate, pool, &mut cache)?;
                // Exact reuse is a primitive-equivalence check, not a codec goal.
                assert_eq!(actual, expected, "{name}, backend {backend}");
                assert_reconstruction(&actual, &original, &jpeg);
                assert_reconstruction(&expected, &original, &jpeg);
                let (_, payload) = cache.entry.as_ref().unwrap();
                let retained = payload.global.bytes.len()
                    + payload
                        .groups
                        .iter()
                        .map(|group| group.bytes.len())
                        .sum::<usize>();
                assert!(
                    retained
                        <= candidate
                            .max_output_bytes
                            .saturating_add(payload.groups.len())
                );
            }
            let (width, height) = geometry(&jpeg).dimensions_in_blocks();
            let builds = if width < 5 && height < 5 { 4 } else { 3 };
            assert_eq!(cache.builds, builds);
            assert_eq!(cache.hits, 6 - builds);
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn search_cache_resolved_policies_match_direct_encoding_at_every_tier() {
    let (original, jpeg) = fixture("rgb_seq_33x25");
    let settings = Options::default();
    let prepared = Prepared::new(&jpeg, &settings).unwrap();
    let mut cache = SearchCache::new(&jpeg);
    for order in [
        OrderPolicy::Source,
        OrderPolicy::Default,
        OrderPolicy::Adaptive,
    ] {
        for tier in (-1i8..=9).rev() {
            let candidate = Options {
                tier: SpeedTier::try_from(tier).unwrap(),
                order,
                ..settings.clone()
            };
            let expected = prepared.encode(&jpeg, &candidate, None).unwrap();
            let actual = prepared
                .encode_with_cache(&jpeg, &candidate, None, &mut cache)
                .unwrap();
            assert_eq!(actual, expected, "{tier}, {order:?}");
            assert_reconstruction(&actual, &original, &jpeg);
        }
    }
    assert!(cache.hits > 0);
    for changed in [
        Options {
            cfl: CflMode::Disabled,
            ..settings.clone()
        },
        Options {
            decoding_speed: 1,
            ..settings.clone()
        },
        Options {
            decoding_speed: 4,
            ..settings.clone()
        },
    ] {
        let expected = prepared.encode(&jpeg, &changed, None).unwrap();
        let actual = prepared
            .encode_with_cache(&jpeg, &changed, None, &mut cache)
            .unwrap();
        assert_eq!(actual, expected);
        assert_reconstruction(&actual, &original, &jpeg);
    }
}

#[test]
fn search_cache_evicts_incompatible_entries_and_rejects_other_frames() {
    let (_, jpeg) = fixture("rgb_seq_33x25");
    let other = jpeg.clone();
    let options = Options::default();
    let prepared = Prepared::new(&jpeg, &options).unwrap();
    let mut cache = SearchCache::new(&jpeg);
    prepared
        .encode_with_cache(&jpeg, &options, None, &mut cache)
        .unwrap();
    let changed = Options {
        tier: SpeedTier::Kitten,
        ..options.clone()
    };
    assert!(matches!(
        cache.get_or_build(&jpeg, geometry(&jpeg), &changed, || Err(Error::Limit(
            "injected"
        ))),
        Err(Error::Limit("injected"))
    ));
    assert!(cache.entry.is_none());
    prepared
        .encode_with_cache(&jpeg, &changed, None, &mut cache)
        .unwrap();
    prepared
        .encode_with_cache(&jpeg, &options, None, &mut cache)
        .unwrap();
    assert_eq!((cache.builds, cache.hits), (3, 0));
    assert!(matches!(
        cache.get_or_build(&other, geometry(&other), &options, || panic!(
            "foreign frame reached builder"
        )),
        Err(Error::InvalidInput(_))
    ));
    assert!(cache.entry.is_none());
    prepared
        .encode_with_cache(&jpeg, &options, None, &mut cache)
        .unwrap();
    assert_eq!((cache.builds, cache.hits), (4, 0));
}

#[test]
fn search_cache_accepts_exact_final_container_bound_on_miss_and_hit() {
    let (original, jpeg) = fixture("rgb_seq_33x25");
    let mut options = Options::default();
    let prepared = Prepared::new(&jpeg, &options).unwrap();
    let expected = prepared.encode(&jpeg, &options, None).unwrap();
    options.max_output_bytes = expected.len();
    let mut cache = SearchCache::new(&jpeg);
    for _ in 0..2 {
        let actual = prepared
            .encode_with_cache(&jpeg, &options, None, &mut cache)
            .unwrap();
        assert_eq!(actual.len(), options.max_output_bytes);
        assert_reconstruction(&actual, &original, &jpeg);
    }
    assert_eq!((cache.builds, cache.hits), (1, 1));
}

#[test]
fn search_cache_hits_cannot_bypass_resource_or_final_container_limits() {
    let (_, jpeg) = fixture("rgb_seq_33x25");
    let options = Options::default();
    let prepared = Prepared::new(&jpeg, &options).unwrap();
    let mut cache = SearchCache::new(&jpeg);
    let expected = prepared
        .encode_with_cache(&jpeg, &options, None, &mut cache)
        .unwrap();
    for limited in [
        Options {
            max_coefficients: 1,
            ..options.clone()
        },
        Options {
            max_tokens: 1,
            ..options.clone()
        },
        Options {
            max_output_bytes: 1,
            ..options.clone()
        },
    ] {
        assert!(matches!(
            prepared.encode(&jpeg, &limited, None),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            prepared.encode_with_cache(&jpeg, &limited, None, &mut cache),
            Err(Error::Limit(_))
        ));
        assert!(cache.entry.is_none());
        assert_eq!(
            prepared
                .encode_with_cache(&jpeg, &options, None, &mut cache)
                .unwrap(),
            expected
        );
    }
    let limited = Options {
        max_output_bytes: expected.len() - 1,
        ..options.clone()
    };
    assert!(matches!(
        prepared.encode(&jpeg, &limited, None),
        Err(Error::Limit(_))
    ));
    // AC fits; the complete container does not. The second attempt hits the cache
    // and must still fail while appending the candidate-specific prefix/container.
    for _ in 0..2 {
        assert!(matches!(
            prepared.encode_with_cache(&jpeg, &limited, None, &mut cache),
            Err(Error::Limit(_))
        ));
        assert!(cache.entry.is_some());
    }
    assert_eq!(cache.hits, 1);
}
