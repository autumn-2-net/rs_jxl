use super::MAX_OUTPUT_BYTES;
use std::io::{self, Seek, SeekFrom, Write};

/// EXR needs seekable output for chunk offsets. Both writes and seeks are bounded.
pub(in crate::decode) struct Output {
    bytes: Vec<u8>,
    position: usize,
    limit: usize,
}

impl Default for Output {
    fn default() -> Self {
        Self::with_limit(MAX_OUTPUT_BYTES)
    }
}

impl Output {
    pub(in crate::decode) fn with_limit(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            position: 0,
            limit,
        }
    }
    pub(in crate::decode) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for Output {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let end = self
            .position
            .checked_add(bytes.len())
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| io::Error::other("output size overflows or exceeds its byte budget"))?;
        if end > self.bytes.len() {
            self.bytes
                .try_reserve(end - self.bytes.len())
                .map_err(io::Error::other)?;
            self.bytes.resize(end, 0);
        }
        self.bytes[self.position..end].copy_from_slice(bytes);
        self.position = end;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for Output {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let target = match from {
            SeekFrom::Start(n) => i128::from(n),
            SeekFrom::End(n) => self.bytes.len() as i128 + i128::from(n),
            SeekFrom::Current(n) => self.position as i128 + i128::from(n),
        };
        let target = usize::try_from(target)
            .ok()
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| io::Error::other("output seek exceeds byte budget"))?;
        self.position = target;
        Ok(target as u64)
    }
}
