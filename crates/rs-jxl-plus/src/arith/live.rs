// PJA MIT translation; see ../../LICENSE-PJA and ../../NOTICE-PJA.
//! pjaarith.pas TLiveStream, InitLive and the live arithmetic paths.
//!
//! One producer and any number of readers share retained 64-KiB blocks.
//! The producer publishes every 64 bytes and at completion. Readers wait on
//! a condition variable, and read zero padding only after successful completion.
//! Failure, cancellation and producer drop wake every waiter. This module
//! creates no threads; the caller decides whether to schedule concurrent work.

use super::{BitCoder, split};
use crate::{Error, Result};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

const BLOCK: usize = 1 << 16;
const BLOCKS: usize = 1 << 16;
const PUBLISH: usize = 64;

#[derive(Clone, Copy)]
enum Failure {
    Limit,
    Allocation,
    Cancelled,
    ProducerDropped,
    Poisoned,
}
impl Failure {
    fn error(self) -> Error {
        match self {
            Self::Limit => Error::Limit("live arithmetic output bytes"),
            Self::Allocation => Error::Limit("live arithmetic allocation"),
            Self::Cancelled => Error::Invalid("live arithmetic stream cancelled"),
            Self::ProducerDropped => {
                Error::Invalid("live arithmetic producer dropped before finish")
            }
            Self::Poisoned => Error::Internal("poisoned live arithmetic stream"),
        }
    }
}
#[derive(Clone, Copy)]
enum Terminal {
    Running,
    Finished,
    Failed(Failure),
}
struct State {
    blocks: Vec<Vec<u8>>,
    written: usize,
    ready: usize,
    limit: usize,
    terminal: Terminal,
}
struct Inner {
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Clone)]
pub(crate) struct LiveStream {
    inner: Arc<Inner>,
}
impl LiveStream {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    blocks: Vec::new(),
                    written: 0,
                    ready: 0,
                    limit: max_bytes,
                    terminal: Terminal::Running,
                }),
                changed: Condvar::new(),
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        match self.inner.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.terminal = Terminal::Failed(Failure::Poisoned);
                self.inner.changed.notify_all();
                state
            }
        }
    }

    fn fail(&self, failure: Failure) {
        let mut state = self.lock();
        if matches!(state.terminal, Terminal::Running) {
            state.terminal = Terminal::Failed(failure);
        }
        self.inner.changed.notify_all();
    }

    pub(crate) fn cancel(&self) {
        self.fail(Failure::Cancelled);
    }

    pub(crate) fn check(&self) -> Result<()> {
        match self.lock().terminal {
            Terminal::Failed(failure) => Err(failure.error()),
            _ => Ok(()),
        }
    }

    pub(crate) fn put(&self, byte: u8) -> Result<()> {
        let mut state = self.lock();
        match state.terminal {
            Terminal::Failed(failure) => return Err(failure.error()),
            Terminal::Finished => return Err(Error::Invalid("write after live arithmetic finish")),
            Terminal::Running => {}
        }
        let failure = if state.written >= state.limit {
            Some(Failure::Limit)
        } else if state.written % BLOCK == 0 {
            if state.blocks.len() >= BLOCKS {
                Some(Failure::Limit)
            } else {
                let capacity = BLOCK.min(state.limit - state.written);
                let mut block = Vec::new();
                if state.blocks.try_reserve(1).is_err()
                    || block.try_reserve_exact(capacity).is_err()
                {
                    Some(Failure::Allocation)
                } else {
                    state.blocks.push(block);
                    None
                }
            }
        } else {
            None
        };
        if let Some(failure) = failure {
            state.terminal = Terminal::Failed(failure);
            self.inner.changed.notify_all();
            return Err(failure.error());
        }
        let index = state.written / BLOCK;
        // Each block was reserved in full; this push does not allocate.
        state.blocks[index].push(byte);
        state.written += 1;
        if state.written % PUBLISH == 0 {
            state.ready = state.written;
            self.inner.changed.notify_all();
        }
        Ok(())
    }

    pub(crate) fn finish(&self) -> Result<()> {
        let mut state = self.lock();
        match state.terminal {
            Terminal::Failed(failure) => return Err(failure.error()),
            Terminal::Finished => return Ok(()),
            Terminal::Running => {}
        }
        state.ready = state.written;
        state.terminal = Terminal::Finished;
        self.inner.changed.notify_all();
        Ok(())
    }

    pub(crate) fn get(&self, position: usize) -> Result<u8> {
        let mut state = self.lock();
        loop {
            if let Terminal::Failed(failure) = state.terminal {
                return Err(failure.error());
            }
            if position < state.ready {
                return Ok(state.blocks[position / BLOCK][position % BLOCK]);
            }
            if matches!(state.terminal, Terminal::Finished) {
                return Ok(0);
            }
            state = match self.inner.changed.wait(state) {
                Ok(state) => state,
                Err(poisoned) => {
                    let mut state = poisoned.into_inner();
                    state.terminal = Terminal::Failed(Failure::Poisoned);
                    self.inner.changed.notify_all();
                    state
                }
            };
        }
    }

    /// Snapshot of all retained bytes, including an unpublished partial block.
    pub(crate) fn bytes(&self) -> Result<Vec<u8>> {
        let state = self.lock();
        if let Terminal::Failed(failure) = state.terminal {
            return Err(failure.error());
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(state.written)
            .map_err(|_| Error::Limit("live arithmetic snapshot allocation"))?;
        for block in &state.blocks {
            bytes.extend_from_slice(block);
        }
        Ok(bytes)
    }
}

pub(crate) struct LiveEncoder {
    stream: LiveStream,
    low: u32,
    high: u32,
    finished: bool,
}
impl LiveEncoder {
    pub(crate) fn new(stream: LiveStream) -> Self {
        Self {
            stream,
            low: 0,
            high: u32::MAX,
            finished: false,
        }
    }
    pub(crate) fn finish(mut self) -> Result<Vec<u8>> {
        for _ in 0..4 {
            self.stream.put((self.low >> 24) as u8)?;
            self.low <<= 8;
        }
        self.stream.finish()?;
        self.finished = true;
        self.stream.bytes()
    }
}
impl Drop for LiveEncoder {
    fn drop(&mut self) {
        if !self.finished {
            self.stream.fail(Failure::ProducerDropped);
        }
    }
}
impl BitCoder for LiveEncoder {
    fn bit(&mut self, probability: u32, value: i32) -> i32 {
        let middle = split(self.low, self.high, probability);
        let bit = i32::from(value != 0);
        if bit != 0 {
            self.high = middle;
        } else {
            self.low = middle + 1;
        }
        while (self.low ^ self.high) & 0xff00_0000 == 0 {
            // BitCoder cannot return an error; the stream retains the failure
            // and finish returns it. Readers are notified immediately.
            let _ = self.stream.put((self.high >> 24) as u8);
            self.low <<= 8;
            self.high = (self.high << 8) | 255;
        }
        bit
    }
    fn encoding(&self) -> bool {
        true
    }
}

pub(crate) struct LiveDecoder {
    stream: LiveStream,
    low: u32,
    high: u32,
    code: u32,
    position: usize,
    failed: bool,
}
impl LiveDecoder {
    /// May wait for the producer. Call check after decoding to surface failures
    /// that arrived after this constructor's initial four bytes.
    pub(crate) fn new(stream: LiveStream) -> Result<Self> {
        let mut code = 0;
        for i in 0..4 {
            code = (code << 8) | stream.get(i)? as u32;
        }
        Ok(Self {
            stream,
            low: 0,
            high: u32::MAX,
            code,
            position: 4,
            failed: false,
        })
    }
    pub(crate) fn check(&self) -> Result<()> {
        self.stream.check()?;
        if self.failed {
            Err(Error::Internal("live arithmetic reader failure"))
        } else {
            Ok(())
        }
    }
    fn get(&mut self) -> u8 {
        let byte = match self.stream.get(self.position) {
            Ok(byte) => byte,
            Err(_) => {
                self.failed = true;
                0
            }
        };
        self.position = self.position.saturating_add(1);
        byte
    }
}
impl BitCoder for LiveDecoder {
    fn bit(&mut self, probability: u32, _value: i32) -> i32 {
        let middle = split(self.low, self.high, probability);
        let bit = i32::from(self.code <= middle);
        if bit != 0 {
            self.high = middle;
        } else {
            self.low = middle + 1;
        }
        while (self.low ^ self.high) & 0xff00_0000 == 0 {
            self.low <<= 8;
            self.high = (self.high << 8) | 255;
            self.code = (self.code << 8) | self.get() as u32;
        }
        bit
    }
    fn encoding(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::{AdaptiveBit, Decoder, Encoder};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn finish_publishes_partial_and_padding() {
        let stream = LiveStream::new(5);
        for b in 1..=5 {
            stream.put(b).unwrap();
        }
        assert_eq!(stream.bytes().unwrap(), [1, 2, 3, 4, 5]);
        stream.finish().unwrap();
        for i in 0..5 {
            assert_eq!(stream.get(i).unwrap(), i as u8 + 1);
        }
        assert_eq!(stream.get(5).unwrap(), 0);
        assert_eq!(stream.get(usize::MAX).unwrap(), 0);
        assert!(stream.put(9).is_err());
        stream.finish().unwrap();
    }

    #[test]
    fn publish_close_cancel_and_early_failure_wake_waiters() {
        for action in 0..4 {
            let stream = LiveStream::new(64);
            let reader = stream.clone();
            let (tx, rx) = mpsc::channel();
            let thread = std::thread::spawn(move || {
                tx.send(reader.get(0).map_err(|e| e.to_string())).unwrap()
            });
            match action {
                0 => {
                    for _ in 0..64 {
                        stream.put(7).unwrap();
                    }
                }
                1 => {
                    stream.finish().unwrap();
                }
                2 => {
                    stream.cancel();
                }
                _ => {
                    drop(LiveEncoder::new(stream.clone()));
                }
            }
            let value = rx
                .recv_timeout(Duration::from_secs(5))
                .expect("live reader did not wake");
            match action {
                0 => assert_eq!(value.unwrap(), 7),
                1 => assert_eq!(value.unwrap(), 0),
                _ => assert!(value.is_err()),
            }
            thread.join().unwrap();
        }
        let stream = LiveStream::new(0);
        let reader = stream.clone();
        let thread = std::thread::spawn(move || LiveDecoder::new(reader).is_err());
        assert!(stream.put(0).is_err());
        assert!(thread.join().unwrap());
    }

    #[test]
    fn concurrent_coders_match_memory_across_blocks() {
        let count = 600000usize;
        let stream = LiveStream::new(count);
        let producer = stream.clone();
        let thread = std::thread::spawn(move || {
            let mut live = LiveEncoder::new(producer);
            let mut memory = Encoder::new(count);
            let mut counter = AdaptiveBit::default();
            let mut state = 0x12345678u32;
            for _ in 0..count {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let bit = (state >> 31) as i32;
                live.bit(counter.p as u32, bit);
                memory.bit(counter.p as u32, bit);
                counter.update(bit);
            }
            let bytes = live.finish().unwrap();
            assert_eq!(bytes, memory.finish().unwrap());
            bytes
        });
        let mut live = LiveDecoder::new(stream).unwrap();
        let mut counter = AdaptiveBit::default();
        let mut state = 0x12345678u32;
        for _ in 0..count {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let bit = (state >> 31) as i32;
            assert_eq!(live.bit(counter.p as u32, 0), bit);
            counter.update(bit);
        }
        live.check().unwrap();
        let bytes = thread.join().unwrap();
        assert!(bytes.len() > BLOCK);
        let mut memory = Decoder::new(&bytes);
        assert_eq!(
            memory.bit(32768, 0),
            ((0x12345678u32.wrapping_mul(1664525).wrapping_add(1013904223)) >> 31) as i32
        );
    }

    #[test]
    fn bounded_encoder_failure_and_midstream_cancel() {
        let stream = LiveStream::new(3);
        assert!(matches!(
            LiveEncoder::new(stream.clone()).finish(),
            Err(Error::Limit(_))
        ));
        assert!(LiveDecoder::new(stream).is_err());
        let stream = LiveStream::new(64);
        for _ in 0..64 {
            stream.put(0).unwrap();
        }
        let mut decoder = LiveDecoder::new(stream.clone()).unwrap();
        stream.cancel();
        for _ in 0..1024 {
            decoder.bit(32768, 0);
        }
        assert!(decoder.check().is_err());
    }
}
