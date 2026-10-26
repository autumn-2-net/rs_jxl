use super::*;

#[test]
fn cumulative_input_can_cross_old_default_without_retaining_large_buffers() {
    let mut stream = StreamDecoder::new(&DecodeOptions::default()).unwrap();
    // A completed raw codestream ignores suffix bytes but still counts input.
    stream.backend_complete = true;
    stream.input_bytes = 512 * 1024 * 1024;
    stream.push(&[0]).unwrap();
    assert_eq!(stream.input_bytes, 512 * 1024 * 1024 + 1);
    assert!(stream.pending.is_empty());
    assert!(stream.trailing.is_empty());
    stream.input_bytes = usize::MAX - 1;
    stream.push(&[0]).unwrap();
    assert_eq!(stream.input_bytes, usize::MAX);
    assert!(matches!(stream.push(&[0]), Err(Error::Limit(_))));
}

#[test]
fn cumulative_input_keeps_explicit_caps_and_error_poisoning() {
    for cap in [0, 2] {
        let mut options = DecodeOptions::default();
        options.limits.max_input_bytes = cap;
        let mut stream = StreamDecoder::new(&options).unwrap();
        stream.backend_complete = true;
        stream.push(&[0; 2][..cap]).unwrap();
        assert_eq!(stream.input_bytes, cap);
        assert!(matches!(stream.push(&[0]), Err(Error::Limit(_))));
        assert!(matches!(stream.push(&[]), Err(Error::InvalidInput(_))));
    }
}

#[test]
fn cumulative_output_defaults_keep_overflow_and_explicit_budget_checks() {
    let mut stream = StreamDecoder::new(&DecodeOptions::default()).unwrap();
    stream.reserve_output(512 * 1024 * 1024 + 1).unwrap();
    stream.reserved_output_bytes = usize::MAX - 1;
    stream.reserve_output(1).unwrap();
    assert!(matches!(stream.reserve_output(1), Err(Error::Limit(_))));

    let mut options = DecodeOptions::default();
    options.limits.max_output_bytes = 3;
    let mut stream = StreamDecoder::new(&options).unwrap();
    stream.reserve_output(3).unwrap();
    assert!(matches!(stream.reserve_output(1), Err(Error::Limit(_))));
}

#[test]
fn reader_probe_capacity_handles_unlimited_and_tiny_input_budgets() {
    struct Probe(usize);
    impl Read for Probe {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            assert_eq!(bytes.len(), self.0);
            Err(std::io::Error::other(
                "stop after checking the reader buffer",
            ))
        }
    }
    for (cap, chunk, expected) in [
        (usize::MAX, DEFAULT_CHUNK_BYTES, DEFAULT_CHUNK_BYTES),
        (usize::MAX, 8, 8),
        (0, usize::MAX, 1),
        (4, usize::MAX, 5),
    ] {
        let mut options = DecodeOptions::default();
        options.limits.max_input_bytes = cap;
        let result = decode_reader_with(Probe(expected), &options, chunk, |_| Ok(()));
        assert!(matches!(result, Err(Error::Io(_))));
    }
}
