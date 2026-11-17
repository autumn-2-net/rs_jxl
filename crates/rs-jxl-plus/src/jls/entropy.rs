//! PJA pjajls.pas bit stuffing, Golomb values and symbol dispatch (MIT).
//! Revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA/NOTICE-PJA.
use super::*;
pub(super) const RUN_J: [i32; 32] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13,
    14, 15,
];
pub(super) enum Io<'a> {
    Read {
        data: &'a [u8],
        pos: usize,
        byte: u8,
        left: i32,
        ff: bool,
        over: bool,
    },
    Write {
        data: Vec<u8>,
        byte: u8,
        used: i32,
        ff: bool,
        limit: usize,
    },
    Symbols {
        model: &'a mut dyn Symbols,
        encoding: bool,
    },
}
impl<'a> Io<'a> {
    pub fn reader(data: &'a [u8]) -> Self {
        Self::Read {
            data,
            pos: 0,
            byte: 0,
            left: 0,
            ff: false,
            over: false,
        }
    }
    pub fn writer(limit: usize) -> Self {
        Self::Write {
            data: Vec::new(),
            byte: 0,
            used: 0,
            ff: false,
            limit,
        }
    }
    pub fn symbols(model: &'a mut dyn Symbols, encoding: bool) -> Self {
        Self::Symbols { model, encoding }
    }
    pub fn encoding(&self) -> bool {
        !matches!(
            self,
            Self::Read { .. }
                | Self::Symbols {
                    encoding: false,
                    ..
                }
        )
    }
    pub fn bit(&mut self, v: i32) -> Result<i32> {
        match self {
            Self::Read {
                data,
                pos,
                byte,
                left,
                ff,
                over,
            } => {
                if *left == 0 {
                    *byte = if let Some(&v) = data.get(*pos) {
                        *pos += 1;
                        v
                    } else {
                        *over = true;
                        0
                    };
                    *left = if *ff { 7 } else { 8 };
                    if *ff {
                        *byte &= 127;
                    }
                    *ff = *byte == 255;
                }
                *left -= 1;
                Ok(((*byte >> *left) & 1) as i32)
            }
            Self::Write {
                data,
                byte,
                used,
                ff,
                limit,
            } => {
                *byte = (*byte << 1) | (v as u8 & 1);
                *used += 1;
                if *used == if *ff { 7 } else { 8 } {
                    if data.len() >= *limit {
                        return Err(Error::Limit("JPEG-LS output"));
                    }
                    data.push(*byte);
                    *ff = *byte == 255;
                    *byte = 0;
                    *used = 0;
                }
                Ok(v & 1)
            }
            Self::Symbols { .. } => Err(Error::Internal("JPEG-LS symbol direct bit")),
        }
    }
    pub fn bits(&mut self, v: i32, n: i32) -> Result<i32> {
        if !(0..=32).contains(&n) {
            return Err(Error::Invalid("JPEG-LS bit count"));
        }
        let mut r = 0u32;
        for k in (0..n).rev() {
            r = (r << 1) | self.bit(((v as u32 >> k) & 1) as i32)? as u32;
        }
        Ok(r as i32)
    }
    pub fn finish(mut self) -> Result<Vec<u8>> {
        if let Self::Write { used, ff, .. } = &self {
            let width = if *ff { 7 } else { 8 };
            let count = if *used > 0 { width - *used } else { 0 };
            self.bits(0, count)?;
        }
        if let Self::Write { ff: true, .. } = &self {
            self.bits(0, 7)?;
        }
        match self {
            Self::Write { data, .. } => Ok(data),
            _ => Err(Error::Internal("JPEG-LS writer finish")),
        }
    }
    pub fn check(&self) -> Result<()> {
        if matches!(self, Self::Read { over: true, .. }) {
            Err(Error::Invalid("truncated JPEG-LS entropy"))
        } else {
            Ok(())
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn mapped(
        &mut self,
        m: i32,
        c: usize,
        i: usize,
        k: i32,
        ctx: i32,
        px: i32,
        sgn: i32,
        inv: bool,
        limit: i32,
        qbpp: i32,
    ) -> Result<i32> {
        if let Self::Symbols { model, .. } = self {
            return model.mapped(m, c, i, k, ctx, px, sgn, inv);
        }
        if !(0..=16).contains(&k) || limit <= qbpp {
            return Err(Error::Invalid("JPEG-LS Golomb parameters"));
        }
        if self.encoding() {
            if m < 0 {
                return Err(Error::Invalid("negative mapped JPEG-LS error"));
            }
            let high = m >> k;
            if high < limit - qbpp - 1 {
                for _ in 0..high {
                    self.bit(0)?;
                }
                self.bit(1)?;
                self.bits(m, k)?;
            } else {
                for _ in 0..limit - qbpp - 1 {
                    self.bit(0)?;
                }
                self.bit(1)?;
                self.bits(m - 1, qbpp)?;
            }
            Ok(m)
        } else {
            let mut high = 0;
            while self.bit(0)? == 0 {
                high += 1;
                if high > 64 {
                    return Err(Error::Invalid("JPEG-LS Golomb unary"));
                }
            }
            if high >= limit - qbpp - 1 {
                Ok(self.bits(0, qbpp)? + 1)
            } else {
                let v = ((high as i64) << k) + self.bits(0, k)? as i64;
                i32::try_from(v).map_err(|_| Error::Invalid("JPEG-LS mapped overflow"))
            }
        }
    }
    pub fn sample(&mut self, c: usize, i: usize, x: i32) -> Result<()> {
        if let Self::Symbols { model, .. } = self {
            model.sample(c, i, x)?;
        }
        Ok(())
    }
    pub fn run_bit(&mut self, v: i32, c: usize, i: usize, index: usize) -> Result<i32> {
        if let Self::Symbols { model, .. } = self {
            model.run_bit(v, c, i, index)
        } else {
            self.bit(v)
        }
    }
    pub fn run_rest(&mut self, v: i32, n: i32, c: usize, index: usize) -> Result<i32> {
        if let Self::Symbols { model, .. } = self {
            model.run_rest(v, n, c, index)
        } else {
            self.bits(v, n)
        }
    }
    pub fn run_end(&mut self, c: usize, i: usize, len: usize) -> Result<()> {
        if let Self::Symbols { model, .. } = self {
            model.run_end(c, i, len)?;
        }
        Ok(())
    }
}
