//! Bounded source TBW/TBR varints, including signed zigzag representation.
use super::*;
#[derive(Default, Clone)]
pub(super) struct BW {
    pub d: Vec<u8>,
}
impl BW {
    pub fn put(&mut self, b: u8) -> Result<()> {
        self.put_n(&[b], 0, 1)
    }
    pub fn put_n(&mut self, d: &[u8], off: usize, n: usize) -> Result<()> {
        let end = off.checked_add(n).ok_or(refuse("J2K byte range"))?;
        let src = d.get(off..end).ok_or(refuse("J2K byte range"))?;
        if n > (i32::MAX as usize).saturating_sub(self.d.len()) {
            return Err(Error::Limit("J2K byte stream"));
        }
        self.d
            .try_reserve(n)
            .map_err(|_| Error::Limit("J2K allocation"))?;
        self.d.extend_from_slice(src);
        Ok(())
    }
    pub fn v(&mut self, mut x: u64) -> Result<()> {
        while x >= 128 {
            self.put((x as u8 & 127) | 128)?;
            x >>= 7;
        }
        self.put(x as u8)
    }
    pub fn s(&mut self, x: i64) -> Result<()> {
        self.v(if x < 0 {
            (((-(x + 1)) as u64) << 1) | 1
        } else {
            (x as u64) << 1
        })
    }
}
#[derive(Default, Clone)]
pub(super) struct BR {
    pub d: Vec<u8>,
    pub p: usize,
}
impl BR {
    pub fn get(&mut self) -> Result<u8> {
        let b = *self
            .d
            .get(self.p)
            .ok_or(refuse("J2K side information cut short"))?;
        self.p += 1;
        Ok(b)
    }
    pub fn remaining(&self) -> usize {
        self.d.len() - self.p
    }
    pub fn v(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut sh = 0;
        loop {
            let b = self.get()?;
            if sh > 63 || (sh == 63 && b & 127 > 1) {
                return Err(refuse("J2K bad number"));
            }
            v |= (b as u64 & 127) << sh;
            sh += 7;
            if b < 128 {
                return Ok(v);
            }
        }
    }
    pub fn vi(&mut self, max: usize) -> Result<usize> {
        let x = self.v()?;
        if x > max as u64 {
            return Err(refuse("J2K number out of range"));
        }
        Ok(x as usize)
    }
    pub fn s(&mut self) -> Result<i64> {
        let x = self.v()?;
        Ok(if x & 1 != 0 {
            -((x >> 1) as i64) - 1
        } else {
            (x >> 1) as i64
        })
    }
    pub fn si(&mut self, min: i64, max: i64) -> Result<i64> {
        let x = self.s()?;
        if x < min || x > max {
            return Err(refuse("J2K number out of range"));
        }
        Ok(x)
    }
    pub fn take(&mut self, n: usize) -> Result<Vec<u8>> {
        if n > self.remaining() {
            return Err(refuse("J2K side information cut short"));
        }
        let mut w = BW::default();
        w.put_n(&self.d, self.p, n)?;
        self.p += n;
        Ok(w.d)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_extremes_and_overlong() {
        let mut w = BW::default();
        for x in [i64::MIN, -1, 0, 1, i64::MAX] {
            w.s(x).unwrap();
        }
        let mut r = BR { d: w.d, p: 0 };
        for x in [i64::MIN, -1, 0, 1, i64::MAX] {
            assert_eq!(r.s().unwrap(), x);
        }
        let mut r = BR {
            d: vec![255; 11],
            p: 0,
        };
        assert!(r.v().is_err());
    }
}
