// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Incremental box output from encode_internal.h/encode.cc at b866b3d.
//! Ordered jxlp uses file-format version 0 for official 0.12 compatibility.

use std::io::Write;

use crate::{Error, Result};

pub mod frame_index;

pub(crate) const PREFIX: &[u8] = b"\0\0\0\x0cJXL \r\n\x87\n\0\0\0\x14ftypjxl \0\0\0\0jxl ";

/// A partial write or a missing final chunk produces an incomplete container.
/// After an I/O failure this writer is poisoned and refuses further writes.
/// It never flushes or closes the caller's underlying sink.
pub struct ChunkedWriter<W> {
    output: W,
    maximum: usize,
    written: usize,
    next_index: u32,
    complete: bool,
    poisoned: bool,
    index_written: bool,
}

impl<W: Write> ChunkedWriter<W> {
    pub fn new(output: W, level: u8, max_output_bytes: usize) -> Result<Self> {
        if !matches!(level, 5 | 10) {
            return Err(Error::InvalidInput(
                "container level must be 5 or 10".into(),
            ));
        }
        let prefix_size = PREFIX.len() + usize::from(level == 10) * 9;
        if prefix_size > max_output_bytes {
            return Err(Error::Limit("container output bytes"));
        }
        let mut writer = Self {
            output,
            maximum: max_output_bytes,
            written: 0,
            next_index: 0,
            complete: false,
            poisoned: false,
            index_written: false,
        };
        writer.write_parts(&[PREFIX])?;
        if level == 10 {
            writer.write_parts(&[b"\0\0\0\x09jxll\x0a"])?;
        }
        Ok(writer)
    }

    pub fn bytes_written(&self) -> usize {
        self.written
    }

    /// Writes one complete byte-aligned codestream fragment without copying it.
    /// `last` marks the final codestream fragment, not the last metadata box.
    pub fn write_chunk(&mut self, bytes: &[u8], last: bool) -> Result<()> {
        if self.complete || self.next_index > 0x7fff_ffff {
            return Err(Error::InvalidInput(
                "codestream chunks are already complete".into(),
            ));
        }
        if self.next_index == 0 && !bytes.starts_with(&[0xff, 0x0a]) {
            return Err(Error::InvalidInput(
                "first chunk must begin with a bare JXL signature".into(),
            ));
        }
        let payload = bytes
            .len()
            .checked_add(4)
            .ok_or(Error::Limit("chunk size"))?;
        let (header, len) = box_header(b"jxlp", payload)?;
        let index = (self.next_index | if last { 0x8000_0000 } else { 0 }).to_be_bytes();
        self.write_parts(&[&header[..len], &index, bytes])?;
        self.next_index += 1;
        self.complete = last;
        Ok(())
    }

    /// Nonstructural metadata may precede, separate or follow codestream chunks.
    /// The caller owns metadata payload semantics; structural boxes are refused.
    pub fn write_metadata(&mut self, kind: [u8; 4], payload: &[u8]) -> Result<()> {
        if !matches!(&kind, b"Exif" | b"xml " | b"jumb" | b"jxli")
            || (kind == *b"Exif" && payload.len() < 4)
            || (kind == *b"jxli" && self.index_written)
        {
            return Err(Error::InvalidInput(
                "invalid metadata box type or payload".into(),
            ));
        }
        let (header, len) = box_header(&kind, payload.len())?;
        self.write_parts(&[&header[..len], payload])?;
        self.index_written |= kind == *b"jxli";
        Ok(())
    }

    pub fn finish(self) -> Result<W> {
        if self.poisoned || !self.complete {
            return Err(Error::Encoding(
                "incomplete or failed chunked container".into(),
            ));
        }
        Ok(self.output)
    }

    fn write_parts(&mut self, parts: &[&[u8]]) -> Result<()> {
        if self.poisoned {
            return Err(Error::Encoding("chunked output failed previously".into()));
        }
        let next = parts
            .iter()
            .try_fold(self.written, |n, part| n.checked_add(part.len()))
            .filter(|&n| n <= self.maximum)
            .ok_or(Error::Limit("container output bytes"))?;
        // A failed write_all may have advanced the sink. Never retry a partial box.
        self.poisoned = true;
        for part in parts {
            self.output.write_all(part)?;
        }
        self.written = next;
        self.poisoned = false;
        Ok(())
    }
}

fn box_header(kind: &[u8; 4], payload: usize) -> Result<([u8; 16], usize)> {
    let short = payload.checked_add(8).ok_or(Error::Limit("box size"))?;
    let mut bytes = [0; 16];
    bytes[4..8].copy_from_slice(kind);
    if let Ok(size) = u32::try_from(short) {
        bytes[..4].copy_from_slice(&size.to_be_bytes());
        Ok((bytes, 8))
    } else {
        let size = u64::try_from(payload)
            .ok()
            .and_then(|n| n.checked_add(16))
            .ok_or(Error::Limit("box size"))?;
        bytes[..4].copy_from_slice(&1u32.to_be_bytes());
        bytes[8..].copy_from_slice(&size.to_be_bytes());
        Ok((bytes, 16))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_chunks_have_exact_indexes_and_final_bit() {
        let mut writer = ChunkedWriter::new(Vec::new(), 10, 100).unwrap();
        writer.write_chunk(&[0xff, 0x0a, 2], false).unwrap();
        writer.write_chunk(&[3, 4], true).unwrap();
        assert!(writer.write_chunk(&[], true).is_err());
        let output = writer.finish().unwrap();
        assert_eq!(
            &output[41..],
            b"\0\0\0\x0fjxlp\0\0\0\0\xff\x0a\x02\0\0\0\x0ejxlp\x80\0\0\x01\x03\x04"
        );
    }

    #[test]
    fn limits_reject_entire_box_before_writing_and_finish_requires_last() {
        let mut writer = ChunkedWriter::new(Vec::new(), 5, 45).unwrap();
        assert!(writer.write_chunk(&[0xff, 0x0a], true).is_err());
        assert_eq!(writer.bytes_written(), PREFIX.len());
        assert!(writer.finish().is_err());
        let mut writer = ChunkedWriter::new(Vec::new(), 5, 100).unwrap();
        assert!(writer.write_metadata(*b"jxlc", &[0]).is_err());
        assert!(writer.write_chunk(&[0], true).is_err());
        writer.write_chunk(&[0xff, 0x0a], true).unwrap();
        writer.write_metadata(*b"xml ", b"<x/>").unwrap();
        writer.write_metadata(*b"jxli", &[0]).unwrap();
        assert!(writer.write_metadata(*b"jxli", &[0]).is_err());
        assert!(writer.finish().is_ok());
        assert!(box_header(b"jxlp", usize::MAX).is_err());
    }

    #[test]
    fn io_failure_poison_prevents_retry_of_partial_chunk() {
        struct FailAfter(usize);
        impl Write for FailAfter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.0 == 0 {
                    return Err(std::io::Error::other("test failure"));
                }
                let n = bytes.len().min(self.0);
                self.0 -= n;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut writer = ChunkedWriter::new(FailAfter(40), 5, 100).unwrap();
        assert!(matches!(
            writer.write_chunk(&[0xff, 0x0a], true),
            Err(Error::Io(_))
        ));
        assert!(matches!(
            writer.write_chunk(&[0xff, 0x0a], true),
            Err(Error::Encoding(_))
        ));
        assert!(writer.finish().is_err());
    }
}
