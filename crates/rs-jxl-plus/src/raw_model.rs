// PJA MIT translation; see ../LICENSE-PJA and ../NOTICE-PJA.
//! pjamodel.pas CodeRaw/CodeRawBytes and its four-input TMixer.
use crate::{
    Error, Result,
    arith::{AdaptiveBit, BitCoder},
    probability::tables,
};

fn counters(n: usize) -> Result<Vec<AdaptiveBit>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Limit("raw model counters"))?;
    v.resize(n, AdaptiveBit::default());
    Ok(v)
}

/// Source raw model, including the unsigned 32-bit size before the bytes.
/// All tables start at their initial values for each stream. The source's
/// thread-local sparse reset is an allocation optimization, not syntax.
pub(crate) fn code<C: BitCoder>(cd: &mut C, bytes: &mut Vec<u8>, cap: usize) -> Result<()> {
    if cd.encoding() && (bytes.len() > cap || bytes.len() > u32::MAX as usize) {
        return Err(Error::Limit("raw bytes"));
    }
    let len = cd.direct(bytes.len() as i32, 32) as u32 as usize;
    if len > cap {
        return Err(Error::Limit("raw bytes"));
    }
    if !cd.encoding() {
        bytes.clear();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| Error::Limit("raw allocation"))?;
        bytes.resize(len, 0);
    }
    if len == 0 {
        return Ok(());
    }
    let mut o0 = counters(256)?;
    let mut o1 = counters(65536)?;
    let mut o2 = counters(1 << 22)?;
    // TMixer.Init(4, 8): unlike v7 there is no bias input.
    let mut weights = [[16384i32; 4]; 8];
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    let t = tables();
    for byte in bytes {
        let h = (((c2 << 8 | c1) as u32).wrapping_mul(2654435761) >> 18) as usize;
        let mut node = 1usize;
        for (b, w) in weights.iter_mut().enumerate() {
            let counters = [
                &mut o0[node],
                &mut o1[c1 * 256 + node],
                &mut o2[(h << 8) | node],
            ];
            let mut st = [0i32; 4];
            for (s, counter) in st.iter_mut().zip(counters.iter()) {
                *s = t.stretch[counter.p as usize >> 4] as i32;
            }
            st[3] = t.stretch[2048] as i32;
            let dot = crate::mix_simd::dot(&st, w);
            let pr = t.squash[((dot / 65536).clamp(-2048, 2047) + 2048) as usize] as i32;
            let bit = cd.bit((pr as u32) << 4, ((*byte >> (7 - b)) & 1) as i32);
            let err = ((bit << 12) - pr) * 6;
            crate::mix_simd::update(w, &st, err, 14, 131072);
            for counter in counters {
                counter.update(bit);
            }
            node = node * 2 + bit as usize;
        }
        if !cd.encoding() {
            *byte = (node - 256) as u8;
        }
        c2 = c1;
        c1 = node - 256;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::{Decoder, Encoder};
    #[test]
    fn self_contained_bytes_and_declared_limit() {
        for original in [
            Vec::new(),
            b"\xff\xd8tablesABABAABAABAAAABAAAAAB\xff\xd9".to_vec(),
        ] {
            let mut enc = Encoder::new(1000);
            code(&mut enc, &mut original.clone(), 1000).unwrap();
            let encoded = enc.finish().unwrap();
            let mut output = Vec::new();
            let mut decoder = Decoder::new(&encoded);
            code(&mut decoder, &mut output, 1000).unwrap();
            decoder.finish_check().unwrap();
            assert_eq!(output, original);
        }
        let mut enc = Encoder::new(100);
        enc.direct(65536, 32);
        assert!(
            code(
                &mut Decoder::new(&enc.finish().unwrap()),
                &mut Vec::new(),
                1024
            )
            .is_err()
        );
    }
}
