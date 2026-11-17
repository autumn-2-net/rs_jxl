//! Test-only coefficient tool/band candidates; no default format changes.
//!
//! Uses the PJA MIT v7 and Full/Fast ports, source revision
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd (LICENSE-PJA/NOTICE-PJA).
//! Descriptor syntax and candidate selection are experimental new work.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tool {
    V7,
    Full,
    Fast,
}
impl Tool {
    fn id(self) -> u8 {
        match self {
            Self::V7 => 0,
            Self::Full => 1,
            Self::Fast => 2,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct Accounting {
    pub tool: u8,
    pub bands: usize,
    pub descriptor_bytes: usize,
    pub stream_bytes: usize,
    /// Descriptor and arithmetic streams only. The harness additionally charges
    /// 40 + 140*components + 8*bands framing, plus reconstruction metadata.
    pub total_bytes: usize,
}
#[derive(Clone, Debug)]
pub(crate) struct Encoded {
    pub descriptor: Vec<u8>,
    pub streams: Vec<Vec<u8>>,
    pub accounting: Accounting,
}
fn invalid() -> Error {
    Error::Invalid("coefficient tool descriptor")
}
fn configuration(components: &[Component], tool: Tool, bands: usize) -> Result<usize> {
    let (rows, source) = validate(components)?;
    if bands == 0 || bands > 16 || bands > rows {
        return Err(invalid());
    }
    // The legacy implementation intentionally retains the source band rule:
    // its API accepts a source band index, not an arbitrary row interval.
    if tool != Tool::V7 && bands != source {
        return Err(Error::Unsupported("legacy candidate band count"));
    }
    Ok(rows)
}

/// Same syntax and per-band bytes as the serial V7 candidate, on an owned pool.
pub(crate) fn encode_v7_parallel(
    components: &[Component],
    bands: usize,
    max: usize,
    threads: usize,
) -> Result<Encoded> {
    configuration(components, Tool::V7, bands)?;
    let descriptor = vec![4, 1, 0, bands as u8];
    let cap = max
        .checked_sub(descriptor.len())
        .ok_or(Error::Limit("coefficient candidate output"))?;
    let streams = super::parallel::encode_custom_bands(components, bands, cap, threads)?;
    let stream_bytes = streams.iter().map(Vec::len).sum::<usize>();
    let accounting = Accounting {
        tool: 0,
        bands,
        descriptor_bytes: descriptor.len(),
        stream_bytes,
        total_bytes: descriptor.len() + stream_bytes,
    };
    Ok(Encoded {
        descriptor,
        streams,
        accounting,
    })
}

/// Strict, transactional coefficient decode; the descriptor alone sets bands.
pub(crate) fn decode_v7_parallel(
    components: &mut [Component],
    descriptor: &[u8],
    streams: &[&[u8]],
    threads: usize,
) -> Result<()> {
    if descriptor.len() != 4 || descriptor[..3] != [4, 1, 0] {
        return Err(invalid());
    }
    let bands = usize::from(descriptor[3]);
    configuration(components, Tool::V7, bands)?;
    if streams.len() != bands {
        return Err(invalid());
    }
    super::parallel::decode_custom_bands(components, streams, threads)
}
pub(crate) fn encode_candidate(
    components: &[Component],
    tool: Tool,
    bands: usize,
    max: usize,
) -> Result<Encoded> {
    let rows = configuration(components, tool, bands)?;
    // Mode/version/tool/bands are all transmitted; stream lengths belong to
    // the surrounding framing and must be charged separately by the harness.
    let descriptor = vec![4, 1, tool.id(), bands as u8];
    let mut remaining = max
        .checked_sub(descriptor.len())
        .ok_or(Error::Limit("coefficient candidate output"))?;
    let mut streams = Vec::with_capacity(bands);
    match tool {
        Tool::V7 => {
            let mut coef = zigzag(components, true);
            for band in 0..bands {
                let mut cd = Encoder::new(remaining);
                code_band(
                    &mut cd,
                    components,
                    &mut coef,
                    rows * band / bands,
                    rows * (band + 1) / bands,
                )?;
                let bytes = cd.finish()?;
                remaining = remaining
                    .checked_sub(bytes.len())
                    .ok_or(Error::Limit("coefficient candidate output"))?;
                streams.push(bytes);
            }
        }
        Tool::Full | Tool::Fast => {
            let mut components = components.to_vec();
            for band in 0..bands {
                let mut cd = Encoder::new(remaining);
                crate::legacy_model::code_band(&mut cd, &mut components, band, tool == Tool::Fast)?;
                let bytes = cd.finish()?;
                remaining = remaining
                    .checked_sub(bytes.len())
                    .ok_or(Error::Limit("coefficient candidate output"))?;
                streams.push(bytes);
            }
        }
    }
    let stream_bytes = streams.iter().map(Vec::len).sum::<usize>();
    let accounting = Accounting {
        tool: tool.id(),
        bands,
        descriptor_bytes: descriptor.len(),
        stream_bytes,
        total_bytes: descriptor.len() + stream_bytes,
    };
    Ok(Encoded {
        descriptor,
        streams,
        accounting,
    })
}
pub(crate) fn decode_candidate(
    components: &mut [Component],
    descriptor: &[u8],
    streams: &[&[u8]],
) -> Result<()> {
    if descriptor.len() != 4 || descriptor[..2] != [4, 1] {
        return Err(invalid());
    }
    let tool = match descriptor[2] {
        0 => Tool::V7,
        1 => Tool::Full,
        2 => Tool::Fast,
        _ => return Err(invalid()),
    };
    let bands = usize::from(descriptor[3]);
    let rows = configuration(components, tool, bands)?;
    if streams.len() != bands {
        return Err(invalid());
    }
    match tool {
        Tool::V7 => {
            let mut coef = zigzag(components, false);
            for (band, stream) in streams.iter().enumerate() {
                let mut cd = Decoder::new(stream);
                code_band(
                    &mut cd,
                    components,
                    &mut coef,
                    rows * band / bands,
                    rows * (band + 1) / bands,
                )?;
                cd.finish_check()?;
            }
            let t = tables();
            for (c, src) in components.iter_mut().zip(coef) {
                for (src, dst) in src
                    .chunks_exact(64)
                    .zip(c.coefficients.chunks_exact_mut(64))
                {
                    for k in 0..64 {
                        dst[t.zz_table[k] as usize] = src[k];
                    }
                }
            }
        }
        Tool::Full | Tool::Fast => {
            for (band, stream) in streams.iter().enumerate() {
                let mut cd = Decoder::new(stream);
                crate::legacy_model::code_band(&mut cd, components, band, tool == Tool::Fast)?;
                cd.finish_check()?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_three_band_streams_budget_and_transactional_decode() {
        let original = vec![
            super::super::tests::fixture(4, 10, 2, 2, 1),
            super::super::tests::fixture(2, 5, 1, 1, 2),
            super::super::tests::fixture(2, 5, 1, 1, 3),
        ];
        let serial = encode_candidate(&original, Tool::V7, 3, 1 << 20).unwrap();
        let size = serial.accounting.total_bytes;
        for threads in [1, 2, 4] {
            let encoded = encode_v7_parallel(&original, 3, size, threads).unwrap();
            assert_eq!(encoded.descriptor, serial.descriptor);
            assert_eq!(encoded.streams, serial.streams);
            assert_eq!(encoded.accounting.total_bytes, size);
            assert!(encode_v7_parallel(&original, 3, size - 1, threads).is_err());
            let mut blank = original.clone();
            for c in &mut blank {
                c.coefficients.fill(0);
            }
            let mut back = blank.clone();
            let refs: Vec<_> = encoded.streams.iter().map(Vec::as_slice).collect();
            decode_v7_parallel(&mut back, &encoded.descriptor, &refs, threads).unwrap();
            assert_eq!(back, original);
            // Corrupt the final band so earlier successfully decoded workers
            // cannot accidentally publish their coefficients before failure.
            for corrupt in 0..3 {
                let mut damaged = encoded.streams.clone();
                match corrupt {
                    0 => {
                        damaged[2].pop();
                    }
                    1 => damaged[2].push(0),
                    _ => {
                        let last = damaged[2].last_mut().unwrap();
                        *last ^= 1;
                    }
                }
                let refs: Vec<_> = damaged.iter().map(Vec::as_slice).collect();
                let mut back = blank.clone();
                assert!(
                    decode_v7_parallel(&mut back, &encoded.descriptor, &refs, threads).is_err()
                );
                assert_eq!(back, blank, "failed decode must not publish any band");
            }
        }
        let refs: Vec<_> = serial.streams.iter().map(Vec::as_slice).collect();
        for threads in [0, 257] {
            assert!(encode_v7_parallel(&original, 3, size, threads).is_err());
            let mut back = original.clone();
            assert!(decode_v7_parallel(&mut back, &serial.descriptor, &refs, threads).is_err());
            assert_eq!(back, original);
        }
        for descriptor in [
            &[4, 1, 1, 3][..],
            &[4, 1, 0, 0],
            &[4, 1, 0, 6],
            &[4, 1, 0, 3, 0],
        ] {
            let mut back = original.clone();
            assert!(decode_v7_parallel(&mut back, descriptor, &refs, 2).is_err());
            assert_eq!(back, original);
        }
    }
    #[test]
    fn source_v7_stream_control_and_candidate_roundtrips() {
        let components = vec![
            super::super::tests::fixture(4, 8, 2, 2, 1),
            super::super::tests::fixture(2, 4, 1, 1, 2),
            super::super::tests::fixture(2, 4, 1, 1, 3),
        ];
        let source = band_count(&components).unwrap();
        assert_eq!(
            encode_candidate(&components, Tool::V7, source, 1 << 20)
                .unwrap()
                .streams,
            encode(&components, 1 << 20).unwrap()
        );
        for (tool, bands) in [
            (Tool::V7, 1),
            (Tool::V7, 2),
            (Tool::V7, 4),
            (Tool::Full, source),
            (Tool::Fast, source),
        ] {
            let coded = encode_candidate(&components, tool, bands, 1 << 20).unwrap();
            assert_eq!(
                coded.accounting.total_bytes,
                4 + coded.streams.iter().map(Vec::len).sum::<usize>()
            );
            let mut decoded = components.clone();
            for c in &mut decoded {
                c.coefficients.fill(0);
            }
            let streams: Vec<_> = coded.streams.iter().map(Vec::as_slice).collect();
            decode_candidate(&mut decoded, &coded.descriptor, &streams).unwrap();
            assert_eq!(decoded, components);
            assert!(
                encode_candidate(&components, tool, bands, coded.accounting.total_bytes - 1)
                    .is_err()
            );
            assert_eq!(
                encode_candidate(&components, tool, bands, coded.accounting.total_bytes)
                    .unwrap()
                    .streams,
                coded.streams
            );
            let mut truncated = coded.streams.clone();
            truncated[0].pop();
            let streams: Vec<_> = truncated.iter().map(Vec::as_slice).collect();
            assert!(decode_candidate(&mut decoded, &coded.descriptor, &streams).is_err());
        }
        assert!(encode_candidate(&components, Tool::Full, 2, 1 << 20).is_err());
    }
    #[test]
    fn descriptor_bounds_are_checked_before_coding() {
        let mut components = vec![super::super::tests::fixture(2, 2, 1, 1, 1)];
        for descriptor in [
            &[4, 1, 0, 0][..],
            &[4, 1, 0, 3],
            &[4, 1, 3, 1],
            &[4, 2, 0, 1],
            &[4, 1, 0, 1, 0],
        ] {
            assert!(decode_candidate(&mut components, descriptor, &[]).is_err());
        }
        assert!(encode_candidate(&components, Tool::V7, 1, 3).is_err());
    }
}
