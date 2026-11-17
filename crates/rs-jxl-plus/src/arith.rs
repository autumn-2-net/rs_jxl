// PJA MIT translation; see ../LICENSE-PJA and ../NOTICE-PJA.
//! Carryless arithmetic coding from pjaarith.pas, revision
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.

// The live scheduler is tested internally; the public experiment uses this coder.
#[allow(dead_code)]
#[path = "arith/live.rs"]
mod live;
#[allow(unused_imports)]
pub(crate) use live::{LiveDecoder, LiveEncoder, LiveStream};

pub trait BitCoder {
    fn bit(&mut self, probability: u32, value: i32) -> i32;
    fn encoding(&self) -> bool;
    /// Experimental static-context coding bypasses adaptive counters/mixers.
    /// Normal coders keep this false, avoiding context-key work in hot loops.
    #[cfg(test)]
    fn static_probability_mode(&self) -> bool {
        false
    }
    #[cfg(test)]
    fn static_probability_profile(&self) -> u8 {
        1
    }
    #[cfg(test)]
    fn static_bit(&mut self, _key: u64, _value: i32) -> Option<i32> {
        None
    }
    /// Test-only causal event identity for a dynamic context experiment.
    /// Notification precedes exactly one modelled bit; a wrapper consumes it
    /// once so direct bits retain their original probability. Unlike the static
    /// hook, this never bypasses the original counter or mixer updates.
    #[cfg(test)]
    fn probability_context(
        &mut self,
        _namespace: u8,
        _kind: u8,
        _bank: usize,
        _index: usize,
        _refs: &[(usize, usize)],
    ) {
    }
    /// Low `count` bits, most significant first. Model callers use 0..=32.
    fn direct(&mut self, value: i32, count: i32) -> i32 {
        let mut result = 0u32;
        for k in (0..count.clamp(0, 32)).rev() {
            result = (result << 1) | self.bit(32768, ((value as u32 >> k) & 1) as i32) as u32;
        }
        result as i32
    }
}

fn split(low: u32, high: u32, probability: u32) -> u32 {
    low + (((high - low) as u64 * probability.clamp(1, 65535) as u64) >> 16) as u32
}

pub struct Encoder {
    low: u32,
    high: u32,
    output: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl Encoder {
    pub fn new(max_output_bytes: usize) -> Self {
        Self {
            low: 0,
            high: u32::MAX,
            output: Vec::new(),
            limit: max_output_bytes,
            exceeded: false,
        }
    }

    fn put(&mut self, byte: u8) {
        if self.exceeded {
            return;
        }
        if self.output.len() == self.limit {
            self.exceeded = true;
            return;
        }
        if self.output.len() == self.output.capacity() {
            // Bound the requested capacity as well as the output length. Avoid
            // Vec::push's infallible allocation path: defer failure to finish.
            let remaining = self.limit - self.output.len();
            let growth = self.output.capacity().max(4096).min(remaining);
            if self.output.try_reserve_exact(growth).is_err() {
                self.exceeded = true;
                return;
            }
        }
        self.output.push(byte);
    }

    pub fn finish(mut self) -> crate::Result<Vec<u8>> {
        for _ in 0..4 {
            self.put((self.low >> 24) as u8);
            self.low <<= 8;
        }
        if self.exceeded {
            Err(crate::Error::Limit("arithmetic output bytes"))
        } else {
            Ok(self.output)
        }
    }
}

impl BitCoder for Encoder {
    fn bit(&mut self, probability: u32, value: i32) -> i32 {
        let middle = split(self.low, self.high, probability);
        let bit = i32::from(value != 0);
        if bit != 0 {
            self.high = middle;
        } else {
            self.low = middle + 1;
        }
        while (self.low ^ self.high) & 0xff00_0000 == 0 {
            self.put((self.high >> 24) as u8);
            self.low <<= 8;
            self.high = (self.high << 8) | 255;
        }
        bit
    }
    fn encoding(&self) -> bool {
        true
    }
}

pub struct Decoder<'a> {
    low: u32,
    high: u32,
    code: u32,
    data: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    /// Source-compatible zero padding while decoding. New hashless framing
    /// must call `finish_check` to reject reads beyond the declared stream.
    pub fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            low: 0,
            high: u32::MAX,
            code: 0,
            data,
            position: 0,
        };
        for _ in 0..4 {
            decoder.code = (decoder.code << 8) | decoder.get() as u32;
        }
        decoder
    }
    fn get(&mut self) -> u8 {
        let byte = self.data.get(self.position).copied().unwrap_or(0);
        self.position = self.position.saturating_add(1);
        byte
    }

    /// Accept exactly the canonical four-byte low-endpoint termination emitted
    /// by Encoder::finish. Renormalization consumes one byte per emitted byte;
    /// the four-byte lookahead consumes the final endpoint after the last bit.
    /// This validates framing, not a content checksum: a changed stream may
    /// still be a valid encoding of a different sequence.
    pub fn finish_check(&self) -> crate::Result<()> {
        if self.position != self.data.len() {
            return Err(crate::Error::Invalid("arithmetic stream consumption"));
        }
        if self.code != self.low {
            return Err(crate::Error::Invalid("arithmetic stream termination"));
        }
        Ok(())
    }
}

impl BitCoder for Decoder<'_> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveBit {
    pub p: u16,
    pub n: u8,
}

impl Default for AdaptiveBit {
    fn default() -> Self {
        Self { p: 32768, n: 0 }
    }
}

// pjaarith.RECIPROCAL[n] = Round(65536 / (n + 1.5)). Exact rational
// initialization, evaluated at compile time. No ties occur: 2*n+3 is odd
// and the numerator is 2^17. No division is performed in the bit loop.
const RC: [i32; 256] = {
    let mut table = [0; 256];
    let mut n = 0;
    while n < table.len() {
        let denominator = 2 * n as i32 + 3;
        table[n] = (131072 + denominator / 2) / denominator;
        n += 1;
    }
    table
};

impl AdaptiveBit {
    pub fn update(&mut self, bit: i32) {
        let reciprocal = RC[self.n as usize] as i64;
        let target = if bit != 0 { 65535 } else { 1 };
        self.p = (self.p as i64 + (target - self.p as i64) * reciprocal / 65536) as u16;
        self.n = self.n.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_counter_trace() {
        // pjaarith.Update integer recurrence, including negative truncation.
        let mut c = AdaptiveBit::default();
        for (bit, probability) in [1, 0, 0, 1, 1, 1, 0, 0]
            .into_iter()
            .zip([54612, 32768, 23406, 32768, 38725, 42849, 37137, 32769])
        {
            c.update(bit);
            assert_eq!(c.p, probability);
        }
        for bit in [0, 1] {
            let mut c = AdaptiveBit::default();
            for _ in 0..100000 {
                c.update(bit);
                assert!((1..=65535).contains(&c.p));
            }
            assert_eq!(c.n, 255);
        }
    }
    #[test]
    fn direct_vectors_and_padding() {
        assert_eq!(Encoder::new(4).finish().unwrap(), [0, 0, 0, 0]);
        let mut e = Encoder::new(100);
        e.direct(0, 8);
        assert_eq!(e.finish().unwrap(), [255, 0, 0, 0, 0]);
        let mut e = Encoder::new(100);
        e.direct(255, 8);
        assert_eq!(e.finish().unwrap(), [0, 0, 0, 0, 0]);
        let mut d = Decoder::new(&[]);
        for _ in 0..1024 {
            assert_eq!(d.bit(32768, 0), 1);
        }
        assert!(matches!(
            Encoder::new(3).finish(),
            Err(crate::Error::Limit(_))
        ));
    }
    #[test]
    fn capped_output_stays_bounded_after_failure() {
        for limit in [0, 1, 3, 4, 5, 4095, 4096, 4097] {
            let mut e = Encoder::new(limit);
            for i in 0..(limit + 10) {
                e.direct(i as i32, 8);
                assert!(e.output.len() <= limit);
            }
            assert!(matches!(e.finish(), Err(crate::Error::Limit(_))));
        }
    }
    #[test]
    fn adaptive_roundtrip() {
        let mut e = Encoder::new(100000);
        let mut c = AdaptiveBit::default();
        for i in 0..20000 {
            let b = ((i * 97 + i / 17) % 101 < 37) as i32;
            e.bit(c.p as u32, b);
            c.update(b);
        }
        let bytes = e.finish().unwrap();
        let mut d = Decoder::new(&bytes);
        let mut c = AdaptiveBit::default();
        for i in 0..20000 {
            let b = ((i * 97 + i / 17) % 101 < 37) as i32;
            assert_eq!(d.bit(c.p as u32, 0), b);
            c.update(b);
        }
        assert_eq!(c.n, 255);
        d.finish_check().unwrap();
    }

    #[test]
    fn strict_termination_rejects_truncation_trailing_and_noncanonical_endpoint() {
        let mut enc = Encoder::new(1024);
        for i in 0..137 {
            enc.bit(1000 + (i * 311) % 64000, (i % 3 == 0) as i32);
        }
        let encoded = enc.finish().unwrap();
        let decode = |bytes: &[u8]| {
            let mut dec = Decoder::new(bytes);
            for i in 0..137 {
                let b = dec.bit(1000 + (i * 311) % 64000, 0);
                if b != (i % 3 == 0) as i32 {
                    return Err(crate::Error::Invalid("test decisions"));
                }
            }
            dec.finish_check()
        };
        decode(&encoded).unwrap();
        for n in 0..encoded.len() {
            assert!(decode(&encoded[..n]).is_err());
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        let mut endpoint = encoded;
        *endpoint.last_mut().unwrap() ^= 1;
        assert!(decode(&endpoint).is_err());
    }
}
