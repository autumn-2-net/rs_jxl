// PJA MIT: algorithms from pas/pjaj2km.pas modes 1 and 3.
//! Internal experimental layout 1; intentionally not the original PJA body.
//! layout:u8, slice_count:u32le, slice_sizes:[u32le], side_size:u32le, side, slices.
const INTERNAL_LAYOUT: u8 = 1;
use super::{
    codestream::{self, BlockInfo, Kind, Limits, Link, PixTile},
    ht_model::HtModel,
    model::{Band, CoefficientModel},
};
use crate::{
    Error, Result,
    arith::{BitCoder, Decoder, Encoder},
    raw_model,
};
enum Coder<'a> {
    Encode(Encoder),
    Decode(Decoder<'a>),
}
impl BitCoder for Coder<'_> {
    fn bit(&mut self, p: u32, b: i32) -> i32 {
        match self {
            Self::Encode(c) => c.bit(p, b),
            Self::Decode(c) => c.bit(p, b),
        }
    }
    fn encoding(&self) -> bool {
        matches!(self, Self::Encode(_))
    }
}
enum Model {
    Classic(CoefficientModel),
    Ht(HtModel),
}
struct Slices<'a> {
    ht: bool,
    encoding: bool,
    input: Vec<&'a [u8]>,
    output: Vec<Vec<u8>>,
    asked: [bool; 64],
    coder: Option<Coder<'a>>,
    model: Option<Model>,
    cap: usize,
}
impl<'a> Slices<'a> {
    fn new(ht: bool, encoding: bool, input: Vec<&'a [u8]>, cap: usize) -> Self {
        Self {
            ht,
            encoding,
            input,
            output: Vec::new(),
            asked: [false; 64],
            coder: None,
            model: None,
            cap,
        }
    }
}
impl Link for Slices<'_> {
    fn kind(&self) -> Kind {
        if self.encoding {
            Kind::Encode
        } else {
            Kind::Decode
        }
    }
    fn start_slice(&mut self, k: usize, hb: usize) -> Result<()> {
        if k >= 64 || self.asked[k] || self.coder.is_some() {
            return Err(Error::Invalid("J2K model slice state"));
        }
        self.asked[k] = true;
        self.coder = Some(if self.encoding {
            if self.output.len() <= k {
                self.output.resize_with(k + 1, Vec::new);
            }
            Coder::Encode(Encoder::new(self.cap))
        } else {
            Coder::Decode(Decoder::new(
                self.input.get(k).ok_or(Error::Invalid("J2K slice index"))?,
            ))
        });
        self.model = Some(if self.ht {
            Model::Ht(HtModel::new(hb as u32)?)
        } else {
            Model::Classic(CoefficientModel::new(hb as u32)?)
        });
        Ok(())
    }
    fn end_slice(&mut self, k: usize) -> Result<()> {
        let c = self.coder.take().ok_or(Error::Invalid("J2K slice end"))?;
        match c {
            Coder::Encode(c) => {
                let bytes = c.finish()?;
                self.cap = self
                    .cap
                    .checked_sub(bytes.len())
                    .ok_or(Error::Limit("J2K aggregate slice output"))?;
                self.output[k] = bytes;
            }
            Coder::Decode(c) => c.finish_check()?,
        }
        self.model = None;
        Ok(())
    }
    fn tile(&mut self, _: &PixTile, bands: Vec<Band>, maps: usize) -> Result<()> {
        match self
            .model
            .as_mut()
            .ok_or(Error::Invalid("J2K tile outside slice"))?
        {
            Model::Classic(m) => m.tile(bands, maps),
            Model::Ht(m) => m.tile(bands, maps),
        }
    }
    fn block(
        &mut self,
        b: &BlockInfo,
        symbols: &mut Vec<u8>,
        pass_ends: &mut Vec<usize>,
        cap: usize,
    ) -> Result<usize> {
        let Some(Model::Classic(m)) = &mut self.model else {
            return Err(Error::Invalid("J2K classic block model"));
        };
        let p = m.code_block(
            self.coder
                .as_mut()
                .ok_or(Error::Invalid("J2K block coder"))?,
            b.band,
            b.x0,
            b.y0,
            b.width,
            b.height,
            b.style,
            b.passes,
            b.zbp,
            symbols,
            cap,
        )?;
        let n = p.symbols.len();
        *symbols = p.symbols;
        *pass_ends = p.pass_ends;
        Ok(n)
    }
    fn ht_block(
        &mut self,
        b: &BlockInfo,
        np: usize,
        plane: i32,
        values: &mut Vec<i32>,
    ) -> Result<()> {
        let Some(Model::Ht(m)) = &mut self.model else {
            return Err(Error::Invalid("J2K HT block model"));
        };
        m.code_block(
            self.coder.as_mut().ok_or(Error::Invalid("J2K HT coder"))?,
            b.band,
            b.x0,
            b.y0,
            b.width,
            b.height,
            b.style & 8 != 0,
            np,
            plane,
            values,
        )
    }
    fn tile_end(&mut self) -> Result<()> {
        Ok(())
    }
}
fn get32(d: &[u8], p: usize) -> Result<usize> {
    Ok(u32::from_le_bytes(
        d.get(p..p + 4)
            .ok_or(Error::Invalid("J2K mode integer"))?
            .try_into()
            .unwrap(),
    ) as usize)
}
fn put32(d: &mut Vec<u8>, n: usize) -> Result<()> {
    d.extend_from_slice(
        &u32::try_from(n)
            .map_err(|_| Error::Limit("J2K mode size"))?
            .to_le_bytes(),
    );
    Ok(())
}
pub(crate) fn encode(data: &[u8], ht: bool, l: &Limits) -> Result<Vec<u8>> {
    let mut slices = Slices::new(ht, true, Vec::new(), l.max_output_bytes);
    let (mut blob, _) = codestream::analyze_with(data, ht, l, &mut slices)?;
    if blob.len() > codestream::side_bound(data.len()).min(l.max_side_bytes) {
        return Err(Error::Limit("J2K side information"));
    }
    let n = slices.output.len();
    if n == 0 || n > 64 || slices.asked[..n].iter().any(|&v| !v) {
        return Err(Error::Invalid("J2K slice enumeration"));
    }
    let mut cd = Encoder::new(l.max_output_bytes);
    raw_model::code(&mut cd, &mut blob, l.max_side_bytes)?;
    let side = cd.finish()?;
    let total = slices
        .output
        .iter()
        .try_fold(9 + 4 * n + side.len(), |n, b| {
            n.checked_add(b.len())
                .ok_or(Error::Limit("J2K mode output"))
        })?;
    if total > l.max_output_bytes || total >= i32::MAX as usize {
        return Err(Error::Limit("J2K mode output"));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| Error::Limit("J2K mode allocation"))?;
    out.push(INTERNAL_LAYOUT);
    put32(&mut out, n)?;
    for s in &slices.output {
        put32(&mut out, s.len())?;
    }
    put32(&mut out, side.len())?;
    out.extend_from_slice(&side);
    for s in slices.output {
        out.extend_from_slice(&s);
    }
    Ok(out)
}
pub(crate) fn decode(d: &[u8], len: usize, ht: bool, l: &Limits) -> Result<Vec<u8>> {
    if d.len() > l.max_input_bytes || len > l.max_output_bytes {
        return Err(Error::Limit("J2K model input/output bytes"));
    }
    if d.len() < 5 {
        return Err(Error::Invalid("J2K short model stream"));
    }
    if d[0] != INTERNAL_LAYOUT {
        return Err(Error::Unsupported("J2K model layout"));
    }
    let n = get32(d, 1)?;
    if !(1..=64).contains(&n) || d.len() < 9 + 4 * n {
        return Err(Error::Invalid("J2K model slice table"));
    }
    let side = get32(d, 5 + 4 * n)?;
    if side < 4 {
        return Err(Error::Invalid("J2K short side arithmetic stream"));
    }
    let first = 9 + 4 * n;
    let mut at = first
        .checked_add(side)
        .ok_or(Error::Limit("J2K side length"))?;
    let mut inputs = Vec::with_capacity(n);
    for i in 0..n {
        let size = get32(d, 5 + 4 * i)?;
        if size < 4 {
            return Err(Error::Invalid("J2K short slice arithmetic stream"));
        }
        let end = at
            .checked_add(size)
            .ok_or(Error::Limit("J2K slice length"))?;
        inputs.push(d.get(at..end).ok_or(Error::Invalid("J2K slice bounds"))?);
        at = end;
    }
    if at != d.len() {
        return Err(Error::Invalid("J2K trailing model bytes"));
    }
    let mut blob = Vec::new();
    let mut side_coder = Decoder::new(
        d.get(first..first + side)
            .ok_or(Error::Invalid("J2K side bounds"))?,
    );
    raw_model::code(
        &mut side_coder,
        &mut blob,
        codestream::side_bound(len).min(l.max_side_bytes),
    )?;
    side_coder.finish_check()?;
    let mut slices = Slices::new(ht, false, inputs, l.max_output_bytes);
    let out = codestream::rebuild_with(&blob, ht, l, &mut slices)?;
    if slices.asked.iter().filter(|&&x| x).count() != n || out.len() != len {
        return Err(Error::Invalid("J2K mode reconstructed check"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slice_checks_canonical_end_without_content_hash() {
        // An empty tile may consume no modeled decisions. Its independent
        // encoder still writes the canonical four-byte endpoint.
        for bytes in [
            &[0u8; 4][..],
            &[0u8; 3][..],
            &[0u8; 5][..],
            &[0, 0, 0, 1][..],
        ] {
            let mut slices = Slices::new(false, false, vec![bytes], 1024);
            slices.start_slice(0, 10).unwrap();
            assert_eq!(slices.end_slice(0).is_ok(), bytes == [0, 0, 0, 0]);
        }
    }
    #[test]
    fn hash_free_layout_header_and_arithmetic_bounds() {
        // One slice: 13 header bytes + 4 side bytes + 4 slice bytes.
        let mut d = vec![INTERNAL_LAYOUT];
        put32(&mut d, 1).unwrap();
        put32(&mut d, 4).unwrap();
        put32(&mut d, 4).unwrap();
        d.extend([0; 8]);
        assert_eq!(d.len(), 21);
        let l = Limits::default();
        for n in 0..d.len() {
            assert!(
                decode(&d[..n], 0, false, &l).is_err(),
                "accepted truncated layout at {n}"
            );
        }
        let mut old = d.clone();
        old[0] = 0;
        assert!(matches!(
            decode(&old, 0, false, &l),
            Err(Error::Unsupported("J2K model layout"))
        ));
        let mut short = d.clone();
        short[9..13].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            decode(&short, 0, false, &l),
            Err(Error::Invalid("J2K short side arithmetic stream"))
        ));
        let mut short = d;
        short[5..9].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            decode(&short, 0, false, &l),
            Err(Error::Invalid("J2K short slice arithmetic stream"))
        ));
    }
}
