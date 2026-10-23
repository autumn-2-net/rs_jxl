// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Brotli suffix of the standard JPEG reconstruction box, ported from
//! jpeg/enc_jpeg_data.cc at b866b3d. The frame's DCT coefficients live elsewhere.

use std::io::{self, Write};

use crate::{Error, Result};

struct BoundedOutput {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl Write for BoundedOutput {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if input.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("JPEG reconstruction output limit"));
        }
        self.bytes
            .try_reserve(input.len())
            .map_err(|_| io::Error::other("JPEG reconstruction allocation"))?;
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Append a standard-window Brotli stream to an already serialized, byte-padded
/// reconstruction header. `payload` contains only preserved APP/COM/inter-marker
/// and trailing data, never the original entropy scans. This low-level function
/// does not validate header fields; use the JPEG metadata serializer first.
///
/// The size cap bounds returned output, not the compressor's internal workspace.
/// No partial box is returned on error. Utility crate `brotli` is pure Rust.
pub fn compress_parts(
    header: &[u8],
    payload: &[u8],
    quality: u32,
    max_output_bytes: usize,
) -> Result<Vec<u8>> {
    if quality > 11 {
        return Err(Error::InvalidInput("Brotli quality must be 0..=11".into()));
    }
    if header.len() > max_output_bytes {
        return Err(Error::Limit("JPEG reconstruction box"));
    }
    let mut output = BoundedOutput {
        bytes: Vec::new(),
        limit: max_output_bytes,
        exceeded: false,
    };
    output.write_all(header)?;
    let params = brotli::enc::BrotliEncoderParams {
        quality: quality as i32,
        lgwin: 22,
        size_hint: payload.len(),
        ..Default::default()
    };
    let result = brotli::BrotliCompress(&mut &payload[..], &mut output, &params);
    if output.exceeded {
        return Err(Error::Limit("JPEG reconstruction box"));
    }
    result?;
    Ok(output.bytes)
}

/// Optimized source cumulative quality schedule: efforts 1/4/7/10 add qualities
/// 3/6/9/11. Retain the smallest actual result rather than assuming a higher
/// quality always wins. Equal sizes retain the earlier result.
pub fn compress_parts_for_effort(
    header: &[u8],
    payload: &[u8],
    effort: u8,
    max_output_bytes: usize,
) -> Result<Vec<u8>> {
    if !(1..=11).contains(&effort) {
        return Err(Error::InvalidInput(
            "reconstruction effort must be 1..=11".into(),
        ));
    }
    let mut best: Option<Vec<u8>> = None;
    for (minimum, quality) in [(1, 3), (4, 6), (7, 9), (10, 11)] {
        if effort < minimum {
            break;
        }
        match compress_parts(header, payload, quality, max_output_bytes) {
            Ok(encoded) => {
                if best.as_ref().is_none_or(|old| encoded.len() < old.len()) {
                    best = Some(encoded);
                }
            }
            // A later quality may fit even when an earlier one does not.
            Err(Error::Limit(_)) => continue,
            Err(error) => return Err(error),
        }
    }
    best.ok_or(Error::Limit("JPEG reconstruction box"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify(header: &[u8], payload: &[u8], encoded: &[u8]) {
        assert_eq!(&encoded[..header.len()], header);
        let mut decoded = Vec::new();
        brotli::BrotliDecompress(&mut &encoded[header.len()..], &mut decoded).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn standard_brotli_roundtrip_at_all_qualities_with_empty_and_binary_data() {
        let samples = [
            Vec::new(),
            vec![0xff; 8192],
            (0..=255).cycle().take(4000).collect(),
        ];
        for payload in samples {
            for quality in 0..=11 {
                let header = [0x33, 0x8a, 0x00];
                let encoded = compress_parts(&header, &payload, quality, 16384).unwrap();
                verify(&header, &payload, &encoded);
                assert_eq!(
                    compress_parts(&header, &payload, quality, encoded.len())
                        .unwrap()
                        .len(),
                    encoded.len()
                );
                assert!(matches!(
                    compress_parts(&header, &payload, quality, encoded.len() - 1),
                    Err(Error::Limit(_))
                ));
            }
        }
    }

    #[test]
    fn cumulative_search_retains_minimum_measured_size() {
        let payload: Vec<u8> = (0..20_000).map(|i| ((i * 47) ^ (i / 113)) as u8).collect();
        let header = [0x08];
        let mut minimum = usize::MAX;
        for (effort, quality) in [(1, 3), (4, 6), (7, 9), (10, 11)] {
            minimum = minimum.min(
                compress_parts(&header, &payload, quality, 30000)
                    .unwrap()
                    .len(),
            );
            let encoded = compress_parts_for_effort(&header, &payload, effort, 30000).unwrap();
            assert_eq!(encoded.len(), minimum);
            verify(&header, &payload, &encoded);
            assert_eq!(
                compress_parts_for_effort(&header, &payload, effort, minimum)
                    .unwrap()
                    .len(),
                minimum
            );
        }
    }

    #[test]
    fn invalid_parameters_and_bounded_writes_are_atomic() {
        assert!(compress_parts(&[], &[], 12, 100).is_err());
        assert!(compress_parts(&[1], &[], 3, 0).is_err());
        assert!(compress_parts_for_effort(&[], &[], 0, 100).is_err());
        assert!(compress_parts_for_effort(&[], &[], 12, 100).is_err());
        let mut writer = BoundedOutput {
            bytes: vec![1],
            limit: 2,
            exceeded: false,
        };
        assert!(writer.write_all(&[2, 3]).is_err());
        assert_eq!(writer.bytes, [1]);
        assert!(writer.exceeded);
    }
}
