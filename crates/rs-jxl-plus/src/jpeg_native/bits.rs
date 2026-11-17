use super::*;
pub(super) struct Reader<'a> {
    pub data: &'a [u8],
    pub p: usize,
    buf: u8,
    left: usize,
}
impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8], p: usize) -> Self {
        Self {
            data,
            p,
            buf: 0,
            left: 0,
        }
    }
    pub fn bits(&mut self, n: usize) -> Result<i32> {
        let mut v = 0;
        for _ in 0..n {
            if self.left == 0 {
                let b = *self
                    .data
                    .get(self.p)
                    .ok_or_else(|| fail("Truncated", self.p))?;
                if b == 255 {
                    if self.data.get(self.p + 1) != Some(&0) {
                        return Err(fail("Truncated", self.p));
                    }
                    self.p += 1;
                }
                self.p += 1;
                self.buf = b;
                self.left = 8;
            }
            self.left -= 1;
            v = (v << 1) | ((self.buf >> self.left) & 1) as i32;
        }
        Ok(v)
    }
    pub fn bit(&mut self) -> Result<i32> {
        self.bits(1)
    }
    pub fn align(&mut self) -> Result<i32> {
        let n = self.left;
        self.left = 0;
        if n == 0 {
            return Ok(-1);
        }
        let mask = (1u16 << n) - 1;
        let v = self.buf as u16 & mask;
        if v == 0 {
            Ok(0)
        } else if v == mask {
            Ok(1)
        } else {
            Err(fail("InconsistentPadding", self.p))
        }
    }
    pub fn padding(&mut self, s: &mut Scan) -> Result<()> {
        let p = self.align()?;
        if p >= 0 {
            if s.padbit < 0 {
                s.padbit = p;
            } else if s.padbit != p {
                return Err(fail("InconsistentPadding", self.p));
            }
        }
        Ok(())
    }
    pub fn restart(&mut self, rst: usize) -> Result<()> {
        if self.data.get(self.p..self.p + 2) != Some(&[255, 208 + (rst & 7) as u8]) {
            return Err(fail("BadRst", self.p));
        }
        self.p += 2;
        Ok(())
    }
    pub fn end(&self) -> Result<usize> {
        if self.p < self.data.len()
            && (self.data[self.p] != 255
                || self.p + 1 >= self.data.len()
                || self.data[self.p + 1] == 0)
        {
            return Err(fail("DataAfterScan", self.p));
        }
        Ok(self.p)
    }
}
pub(super) struct Writer {
    pub data: Vec<u8>,
    acc: u64,
    n: usize,
    limit: usize,
}
impl Writer {
    pub fn new(limit: usize) -> Self {
        Self {
            data: Vec::new(),
            acc: 0,
            n: 0,
            limit,
        }
    }
    pub fn copy(&mut self, d: &[u8]) -> Result<()> {
        if d.len() > self.limit.saturating_sub(self.data.len()) {
            return Err(Error::Limit("JPEG output bytes"));
        }
        self.data
            .try_reserve(d.len())
            .map_err(|_| Error::Limit("JPEG output allocation"))?;
        self.data.extend_from_slice(d);
        Ok(())
    }
    pub fn byte(&mut self, b: u8) -> Result<()> {
        self.copy(&[b])
    }
    pub fn bits(&mut self, v: i32, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.acc = (self.acc << n) | (v as u64 & ((1u64 << n) - 1));
        self.n += n;
        while self.n >= 8 {
            self.n -= 8;
            let b = (self.acc >> self.n) as u8;
            self.byte(b)?;
            if b == 255 {
                self.byte(0)?;
            }
        }
        self.acc &= (1u64 << self.n) - 1;
        Ok(())
    }
    pub fn align(&mut self, p: i32) -> Result<()> {
        if self.n > 0 {
            self.bits(if p == 0 { 0 } else { 255 }, 8 - self.n)?;
        }
        Ok(())
    }
    pub fn restart(&mut self, rst: usize) -> Result<()> {
        self.copy(&[255, 208 + (rst & 7) as u8])
    }
}
