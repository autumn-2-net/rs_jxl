//! Relative, bounded output for source streaming mode 1. Backpatches overwrite
//! earlier bytes; they do not consume the output-size budget a second time.

use crate::{Error, Result};
use std::io::{Seek, SeekFrom, Write};

pub(super) struct SeekOutput<W> {
    output: W,
    base: u64,
    position: usize,
    end: usize,
    maximum: usize,
    finalized: usize,
    failed: bool,
}

impl<W: Write + Seek> SeekOutput<W> {
    pub fn new(mut output: W, maximum: usize) -> Result<Self> {
        let base = output.stream_position()?;
        if maximum == 0 {
            return Err(Error::Limit("streaming output bytes"));
        }
        Ok(Self {
            output,
            base,
            position: 0,
            end: 0,
            maximum,
            finalized: 0,
            failed: false,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }
    pub fn end(&self) -> usize {
        self.end
    }

    pub fn finalized(&self) -> usize {
        self.finalized
    }

    /// After a complete frame is backpatched, its prefix is immutable.
    pub fn finalize(&mut self) -> Result<()> {
        self.seek(self.end)?;
        self.finalized = self.end;
        Ok(())
    }

    pub fn seek(&mut self, position: usize) -> Result<()> {
        self.check()?;
        if position < self.finalized {
            return Err(Error::InvalidInput("seek precedes finalized output".into()));
        }
        if position > self.maximum {
            return Err(Error::Limit("streaming output bytes"));
        }
        let absolute = self
            .base
            .checked_add(
                u64::try_from(position).map_err(|_| Error::Limit("streaming output position"))?,
            )
            .ok_or(Error::Limit("streaming output position"))?;
        self.failed = true;
        let actual = self.output.seek(SeekFrom::Start(absolute))?;
        if actual != absolute {
            return Err(Error::Encoding(
                "output did not seek to requested position".into(),
            ));
        }
        self.position = position;
        self.failed = false;
        Ok(())
    }

    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.check()?;
        let next = self
            .position
            .checked_add(bytes.len())
            .filter(|&n| n <= self.maximum)
            .ok_or(Error::Limit("streaming output bytes"))?;
        self.base
            .checked_add(
                u64::try_from(next).map_err(|_| Error::Limit("streaming output position"))?,
            )
            .ok_or(Error::Limit("streaming output position"))?;
        self.failed = true;
        self.output.write_all(bytes)?;
        self.position = next;
        self.end = self.end.max(next);
        self.failed = false;
        Ok(())
    }

    pub fn zeros(&mut self, mut count: usize) -> Result<()> {
        self.check()?;
        self.position
            .checked_add(count)
            .filter(|&n| n <= self.maximum)
            .ok_or(Error::Limit("streaming output bytes"))?;
        let zeros = [0; 4096];
        while count != 0 {
            let n = count.min(zeros.len());
            self.write(&zeros[..n])?;
            count -= n;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<usize> {
        self.seek(self.end)?;
        Ok(self.end)
    }

    pub fn finish_into(mut self) -> Result<(W, usize)> {
        self.seek(self.end)?;
        Ok((self.output, self.end))
    }

    fn check(&self) -> Result<()> {
        if self.failed {
            Err(Error::Encoding("seekable output failed previously".into()))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn offsets_are_relative_backpatches_do_not_double_count() {
        let mut sink = Cursor::new(vec![17; 10]);
        sink.set_position(10);
        {
            let mut out = SeekOutput::new(&mut sink, 12).unwrap();
            out.seek(4).unwrap();
            out.write(&[3; 8]).unwrap();
            assert_eq!((out.position(), out.end()), (12, 12));
            out.seek(0).unwrap();
            out.write(&[1; 4]).unwrap();
            assert_eq!(out.finish().unwrap(), 12);
        }
        assert_eq!(sink.position(), 22);
        assert_eq!(&sink.get_ref()[..10], &[17; 10]);
        assert_eq!(&sink.get_ref()[10..14], &[1; 4]);
        assert_eq!(&sink.get_ref()[14..], &[3; 8]);
    }

    #[test]
    fn bounds_precede_mutation_and_zero_padding_is_bounded() {
        let mut sink = Cursor::new(Vec::new());
        let mut out = SeekOutput::new(&mut sink, 8193).unwrap();
        assert!(out.seek(8194).is_err());
        assert_eq!(out.position(), 0);
        assert!(out.zeros(8194).is_err());
        assert_eq!(out.end(), 0);
        out.zeros(8193).unwrap();
        assert!(out.write(&[0]).is_err());
        assert_eq!(out.finish().unwrap(), 8193);
        assert_eq!(sink.get_ref(), &vec![0; 8193]);
    }

    #[test]
    fn finalized_floor_cannot_be_overwritten() {
        let mut out = SeekOutput::new(Cursor::new(Vec::new()), 20).unwrap();
        out.write(&[3; 8]).unwrap();
        out.finalize().unwrap();
        assert_eq!(out.finalized(), 8);
        assert!(out.seek(7).is_err());
        out.write(&[5; 4]).unwrap();
        out.seek(8).unwrap();
        out.write(&[7; 4]).unwrap();
        out.finalize().unwrap();
        assert!(out.seek(8).is_err());
        let (sink, count) = out.finish_into().unwrap();
        assert_eq!(count, 12);
        assert_eq!(sink.get_ref(), &[3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7]);
    }

    #[test]
    fn partial_io_poison_prevents_retry() {
        struct Failing {
            data: Cursor<Vec<u8>>,
            left: usize,
        }
        impl Write for Failing {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.left == 0 {
                    return Err(std::io::Error::other("injected write error"));
                }
                let n = self.left.min(bytes.len());
                self.left -= n;
                self.data.write(&bytes[..n])
            }
            fn flush(&mut self) -> std::io::Result<()> {
                panic!("must not flush caller sink")
            }
        }
        impl Seek for Failing {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                self.data.seek(position)
            }
        }
        let mut sink = Failing {
            data: Cursor::new(Vec::new()),
            left: 2,
        };
        let mut out = SeekOutput::new(&mut sink, 10).unwrap();
        assert!(matches!(out.write(&[1, 2, 3]), Err(Error::Io(_))));
        assert!(out.seek(0).is_err());
        assert!(out.write(&[]).is_err());
        assert!(out.finish().is_err());
        assert_eq!(sink.data.get_ref(), &[1, 2]);
    }
}
