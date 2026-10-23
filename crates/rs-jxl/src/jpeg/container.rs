// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! JPEG reconstruction container framing from encode_internal.h/encode.cc.
//! This writes boxes only, not a coefficient codestream or reconstruction data.

use crate::{Error, Result};

const PREFIX: &[u8] = b"\0\0\0\x0cJXL \r\n\x87\n\0\0\0\x14ftypjxl \0\0\0\0jxl ";

fn box_size(payload: usize) -> Result<(usize, bool)> {
    let short = payload
        .checked_add(8)
        .ok_or(Error::Limit("JPEG container box size"))?;
    if u32::try_from(short).is_ok() {
        return Ok((short, false));
    }
    let full = payload
        .checked_add(16)
        .ok_or(Error::Limit("JPEG container box size"))?;
    u64::try_from(full).map_err(|_| Error::Limit("JPEG container box size"))?;
    Ok((full, true))
}

fn append_box(output: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8], size: usize, long: bool) {
    if long {
        output.extend_from_slice(&1u32.to_be_bytes());
        output.extend_from_slice(kind);
        output.extend_from_slice(&(size as u64).to_be_bytes());
    } else {
        output.extend_from_slice(&(size as u32).to_be_bytes());
        output.extend_from_slice(kind);
    }
    output.extend_from_slice(payload);
}

/// Wrap a complete **bare** JXL codestream and standard `jbrd` payload.
/// Image/profile semantics and reconstruction validity must be checked by the
/// caller. No existing JPEG entropy, APP extraction or fallback encoding occurs.
/// The default level is 5; callers select level 10 explicitly when required.
pub fn wrap(
    codestream: &[u8],
    reconstruction: &[u8],
    level: u8,
    max_output_bytes: usize,
) -> Result<Vec<u8>> {
    wrap_with_metadata(codestream, reconstruction, level, &[], max_output_bytes)
}

/// Display boxes are independent of the original APP records retained in jbrd.
pub(crate) fn wrap_with_metadata(
    codestream: &[u8],
    reconstruction: &[u8],
    level: u8,
    metadata: &[(&[u8; 4], &[u8])],
    max_output_bytes: usize,
) -> Result<Vec<u8>> {
    if !codestream.starts_with(&[0xff, 0x0a])
        || reconstruction.is_empty()
        || !matches!(level, 5 | 10)
    {
        return Err(Error::InvalidInput(
            "JPEG container requires a bare JXL codestream, nonempty jbrd and level5/10".into(),
        ));
    }
    let (code_size, code_long) = box_size(codestream.len())?;
    let (recon_size, recon_long) = box_size(reconstruction.len())?;
    let mut metadata_bytes = 0usize;
    let mut seen = [false; 3];
    for &(kind, payload) in metadata {
        let index = match kind {
            b"Exif" => 0,
            b"xml " => 1,
            b"jumb" => 2,
            _ => return Err(Error::InvalidInput("unsupported JPEG metadata box".into())),
        };
        if seen[index] || (index == 0 && payload.len() < 4) {
            return Err(Error::InvalidInput(
                "duplicate or truncated JPEG metadata box".into(),
            ));
        }
        seen[index] = true;
        metadata_bytes = metadata_bytes
            .checked_add(box_size(payload.len())?.0)
            .ok_or(Error::Limit("JPEG metadata box sizes"))?;
    }
    let total = PREFIX
        .len()
        .checked_add(if level == 10 { 9 } else { 0 })
        .and_then(|n| n.checked_add(recon_size))
        .and_then(|n| n.checked_add(metadata_bytes))
        .and_then(|n| n.checked_add(code_size))
        .filter(|&n| n <= max_output_bytes)
        .ok_or(Error::Limit("JPEG container output"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(total)
        .map_err(|_| Error::Limit("JPEG container allocation"))?;
    output.extend_from_slice(PREFIX);
    if level == 10 {
        output.extend_from_slice(b"\0\0\0\x09jxll\x0a");
    }
    append_box(&mut output, b"jbrd", reconstruction, recon_size, recon_long);
    for &(kind, payload) in metadata {
        let (size, long) = box_size(payload.len())?;
        append_box(&mut output, kind, payload, size, long);
    }
    append_box(&mut output, b"jxlc", codestream, code_size, code_long);
    debug_assert_eq!(output.len(), total);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_metadata_is_bounded_and_cannot_inject_structural_boxes() {
        let metadata: [(&[u8; 4], &[u8]); 2] = [(b"Exif", b"\0\0\0\0TIFF"), (b"xml ", b"<x/>")];
        let output = wrap_with_metadata(&[0xff, 0x0a], &[1], 5, &metadata, 100).unwrap();
        assert_eq!(output.len(), PREFIX.len() + 9 + 16 + 12 + 10);
        assert_eq!(&output[PREFIX.len() + 13..PREFIX.len() + 17], b"Exif");
        assert!(wrap_with_metadata(&[0xff, 0x0a], &[1], 5, &metadata, output.len() - 1).is_err());
        assert!(wrap_with_metadata(&[0xff, 0x0a], &[1], 5, &[(b"jxlc", &[1])], 100).is_err());
        assert!(
            wrap_with_metadata(&[0xff, 0x0a], &[1], 5, &[metadata[0], metadata[0]], 100).is_err()
        );
        assert!(wrap_with_metadata(&[0xff, 0x0a], &[1], 5, &[(b"Exif", &[0; 3])], 100).is_err());
    }

    #[test]
    fn short_boxes_and_level_signal_match_known_network_byte_order() {
        for level in [5, 10] {
            let bytes = wrap(&[0xff, 0x0a, 1, 2], &[3, 4, 5], level, 100).unwrap();
            let mut expected = PREFIX.to_vec();
            if level == 10 {
                expected.extend_from_slice(b"\0\0\0\x09jxll\x0a");
            }
            expected.extend_from_slice(b"\0\0\0\x0bjbrd\x03\x04\x05\0\0\0\x0cjxlc\xff\x0a\x01\x02");
            assert_eq!(bytes, expected);
            assert_eq!(
                wrap(&[0xff, 0x0a, 1, 2], &[3, 4, 5], level, bytes.len())
                    .unwrap()
                    .len(),
                bytes.len()
            );
            assert!(wrap(&[0xff, 0x0a, 1, 2], &[3, 4, 5], level, bytes.len() - 1).is_err());
        }
    }

    #[test]
    fn malformed_inputs_and_large_size_arithmetic_do_not_allocate() {
        assert!(wrap(&[], &[1], 5, 100).is_err());
        assert!(wrap(PREFIX, &[1], 5, 100).is_err());
        assert!(wrap(&[0xff, 0x0a], &[], 5, 100).is_err());
        assert!(wrap(&[0xff, 0x0a], &[1], 6, 100).is_err());
        assert!(box_size(usize::MAX).is_err());
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(
                box_size(u32::MAX as usize - 8).unwrap(),
                (u32::MAX as usize, false)
            );
            assert_eq!(
                box_size(u32::MAX as usize - 7).unwrap(),
                (u32::MAX as usize + 9, true)
            );
            let mut header = Vec::new();
            append_box(&mut header, b"jxlc", &[], 0x1_0000_0008, true);
            assert_eq!(header, b"\0\0\0\x01jxlc\0\0\0\x01\0\0\0\x08");
        }
    }
}
