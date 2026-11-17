//! Experimental two-pass static conditional probabilities for the v7 walk.
//!
//! Profiles 1..3 are new experimental work, not PJA stream features. They retain
//! the translated v7 traversal/predictions (PJA MIT revision
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; ../LICENSE-PJA/NOTICE-PJA).
//! Pass one counts actual bits in a finite, explicitly hashed projection of
//! decoder-visible context IDs. One global table is signaled across bands.
//! Pass two and decoding only query that frozen table: adaptive counters and
//! mixer weights are neither read nor updated. Direct bits remain equiprobable.
//!
//! Hash collisions and a single Bernoulli estimate per bucket limit this
//! experiment; this is not an upper bound on static coefficient coding.
use super::*;

const MAGIC: &[u8; 4] = b"PST1";
const HEADER: usize = 12;
const DEFAULT_PROFILE: u8 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StaticAccounting {
    pub profile: u8,
    pub hash_bits: u8,
    pub table_entries: usize,
    pub observed_buckets: usize,
    pub nondefault_buckets: usize,
    pub modelled_bits: u64,
    pub direct_bits: u64,
    pub table_bytes: usize,
    pub stream_bytes: usize,
    pub total_bytes: usize,
}
#[derive(Clone, Debug)]
pub(crate) struct StaticEncoded {
    /// Complete versioned probability-table representation, including headers.
    pub table: Vec<u8>,
    /// Original source band boundaries; each stream includes its coder tail.
    pub streams: Vec<Vec<u8>>,
    pub accounting: StaticAccounting,
}

/// Fixed profile projection/hash of integer IDs, independent of addresses/platform.
/// namespace separates luma/chroma; kind separates one/avg/mix. For mix, bank
/// is W and index its selector, followed by ordered (B,index) references. For
/// one/avg, bank/index identify the first counter, followed by the second for
/// avg. Enum numbering and this hash are part of this experimental profile.
pub(super) fn context_key(
    profile: u8,
    namespace: u8,
    kind: u8,
    bank: usize,
    index: usize,
    refs: &[(usize, usize)],
) -> u64 {
    // Profile 2 pools by the main counter: mixer W identity plus its first
    // (B,index), and the first (B,index) alone for avg. Profile 3 pools mixed
    // events by (W,selector); avg again uses its first counter. One is exact
    // (B,index) in all profiles. The fixed hash/bucket stage is shared.
    let (index, refs) = match (profile, kind) {
        (2, 2) => (0, &refs[..refs.len().min(1)]),
        (2 | 3, 1) | (3, 2) => (index, &[][..]),
        _ => (index, refs),
    };
    fn add(h: u64, word: u64) -> u64 {
        (h ^ word.wrapping_add(0x9e37_79b9_7f4a_7c15))
            .rotate_left(27)
            .wrapping_mul(0x3c79_ac49_2ba7_b653)
    }
    let mut h = 0x5053_5431_0000_0001;
    for v in [
        namespace as u64,
        kind as u64,
        bank as u64,
        index as u64,
        refs.len() as u64,
    ] {
        h = add(h, v);
    }
    for &(bank, index) in refs {
        h = add(add(h, bank as u64), index as u64);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}
fn entries(hash_bits: u8) -> Result<usize> {
    if !(8..=16).contains(&hash_bits) {
        return Err(Error::Invalid("static probability hash bits"));
    }
    Ok(1usize << hash_bits)
}
fn bucket(key: u64, count: usize) -> usize {
    key as usize & (count - 1)
}
#[derive(Clone, Copy, Default)]
struct Count {
    ones: u64,
    total: u64,
}
struct Trainer {
    profile: u8,
    counts: Vec<Count>,
    modelled: u64,
    direct: u64,
    overflow: bool,
}
impl Trainer {
    fn new(n: usize, profile: u8) -> Self {
        Self {
            profile,
            counts: vec![Count::default(); n],
            modelled: 0,
            direct: 0,
            overflow: false,
        }
    }
    fn increment(value: &mut u64, overflow: &mut bool) {
        if let Some(v) = value.checked_add(1) {
            *value = v;
        } else {
            *overflow = true;
        }
    }
}
impl BitCoder for Trainer {
    fn encoding(&self) -> bool {
        true
    }
    fn bit(&mut self, _probability: u32, value: i32) -> i32 {
        Self::increment(&mut self.direct, &mut self.overflow);
        i32::from(value != 0)
    }
    fn static_probability_mode(&self) -> bool {
        true
    }
    fn static_probability_profile(&self) -> u8 {
        self.profile
    }
    fn static_bit(&mut self, key: u64, value: i32) -> Option<i32> {
        let index = bucket(key, self.counts.len());
        let c = &mut self.counts[index];
        Self::increment(&mut c.total, &mut self.overflow);
        if value != 0 {
            Self::increment(&mut c.ones, &mut self.overflow);
        }
        Self::increment(&mut self.modelled, &mut self.overflow);
        Some(i32::from(value != 0))
    }
}
struct Frozen<C> {
    profile: u8,
    coder: C,
    probabilities: Vec<u16>,
}
impl<C: BitCoder> BitCoder for Frozen<C> {
    fn encoding(&self) -> bool {
        self.coder.encoding()
    }
    fn bit(&mut self, probability: u32, value: i32) -> i32 {
        self.coder.bit(probability, value)
    }
    fn static_probability_mode(&self) -> bool {
        true
    }
    fn static_probability_profile(&self) -> u8 {
        self.profile
    }
    fn static_bit(&mut self, key: u64, value: i32) -> Option<i32> {
        let p = self.probabilities[bucket(key, self.probabilities.len())];
        Some(self.coder.bit(p as u32, value))
    }
}
fn probabilities(counts: &[Count]) -> Vec<u16> {
    counts
        .iter()
        .map(|c| {
            // Laplace smoothing, then nearest representable probability on the
            // coder's 1/65536 grid. Integer u128 avoids float/platform ambiguity.
            let numerator = (u128::from(c.ones) + 1) * 65536;
            let denominator = u128::from(c.total) + 2;
            ((numerator + denominator / 2) / denominator).clamp(1, 65535) as u16
        })
        .collect()
}
fn serialize(hash_bits: u8, profile: u8, p: &[u16]) -> Vec<u8> {
    let nondefault = p.iter().filter(|&&v| v != 32768).count();
    let sparse = 4 * nondefault < 2 * p.len();
    let count = if sparse { nondefault } else { p.len() };
    let mut bytes = Vec::with_capacity(HEADER + count * if sparse { 4 } else { 2 });
    bytes.extend(MAGIC);
    bytes.extend([profile, hash_bits, u8::from(!sparse), 0]);
    bytes.extend((count as u32).to_le_bytes());
    for (i, &v) in p.iter().enumerate() {
        if sparse && v == 32768 {
            continue;
        }
        if sparse {
            bytes.extend((i as u16).to_le_bytes());
        }
        bytes.extend(v.to_le_bytes());
    }
    bytes
}
fn parse(bytes: &[u8]) -> Result<Vec<u16>> {
    if bytes.len() < HEADER
        || bytes.get(..4) != Some(MAGIC.as_slice())
        || !(1..=3).contains(&bytes[4])
        || bytes[7] != 0
    {
        return Err(Error::Invalid("static probability header"));
    }
    let n = entries(bytes[5])?;
    let mode = bytes[6];
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let width = match mode {
        0 => 4,
        1 => 2,
        _ => return Err(Error::Invalid("static probability representation")),
    };
    if count > n || mode == 1 && count != n || HEADER + count * width != bytes.len() {
        return Err(Error::Invalid("static probability table length"));
    }
    let mut probabilities = vec![32768; n];
    let mut previous = None;
    for i in 0..count {
        let at = HEADER + i * width;
        let index = if mode == 0 {
            u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as usize
        } else {
            i
        };
        let p_at = at + width - 2;
        let p = u16::from_le_bytes(bytes[p_at..p_at + 2].try_into().unwrap());
        if index >= n
            || p == 0
            || mode == 0 && (p == 32768 || previous.is_some_and(|old| index <= old))
        {
            return Err(Error::Invalid("static probability entry"));
        }
        probabilities[index] = p;
        previous = Some(index);
    }
    Ok(probabilities)
}

/// Train one frozen global table, then encode the unchanged v7 source bands.
/// max_output bounds the complete signaled table plus all arithmetic streams.
pub(crate) fn encode_static(
    components: &[Component],
    max_output: usize,
    hash_bits: u8,
) -> Result<StaticEncoded> {
    encode_static_profile(components, max_output, hash_bits, DEFAULT_PROFILE)
}
/// Finite conditional model: 1 full ordered tuple, 2 main counter, 3 selector.
pub(crate) fn encode_static_profile(
    components: &[Component],
    max_output: usize,
    hash_bits: u8,
    profile: u8,
) -> Result<StaticEncoded> {
    let n = entries(hash_bits)?;
    if !(1..=3).contains(&profile) {
        return Err(Error::Invalid("static probability profile"));
    }
    let (rows, bands) = validate(components)?;
    let mut trainer = Trainer::new(n, profile);
    let mut coef = zigzag(components, true);
    for band in 0..bands {
        code_band(
            &mut trainer,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
    }
    if trainer.overflow {
        return Err(Error::Limit("static probability observations"));
    }
    let p = probabilities(&trainer.counts);
    let observed_buckets = trainer.counts.iter().filter(|c| c.total != 0).count();
    let nondefault_buckets = p.iter().filter(|&&v| v != 32768).count();
    let table = serialize(hash_bits, profile, &p);
    let mut remaining = max_output
        .checked_sub(table.len())
        .ok_or(Error::Limit("static probability output"))?;
    let mut streams = Vec::with_capacity(bands);
    for band in 0..bands {
        let mut cd = Frozen {
            profile,
            coder: Encoder::new(remaining),
            probabilities: p.clone(),
        };
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        let bytes = cd.coder.finish()?;
        remaining = remaining
            .checked_sub(bytes.len())
            .ok_or(Error::Limit("static probability output"))?;
        streams.push(bytes);
    }
    let stream_bytes = streams.iter().map(Vec::len).sum::<usize>();
    let accounting = StaticAccounting {
        profile,
        hash_bits,
        table_entries: n,
        observed_buckets,
        nondefault_buckets,
        modelled_bits: trainer.modelled,
        direct_bits: trainer.direct,
        table_bytes: table.len(),
        stream_bytes,
        total_bytes: table.len() + stream_bytes,
    };
    Ok(StaticEncoded {
        table,
        streams,
        accounting,
    })
}

/// Decode only with the transmitted frozen table. No counts are updated and
/// no mixer weights/probability fitting are performed by this decoder.
pub(crate) fn decode_static(
    components: &mut [Component],
    table: &[u8],
    streams: &[&[u8]],
) -> Result<()> {
    let p = parse(table)?;
    let profile = table[4];
    let (rows, bands) = validate(components)?;
    if streams.len() != bands {
        return Err(Error::Invalid("static coefficient band count"));
    }
    let mut coef = zigzag(components, false);
    for (band, stream) in streams.iter().enumerate() {
        let mut cd = Frozen {
            profile,
            coder: Decoder::new(stream),
            probabilities: p.clone(),
        };
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        cd.coder.finish_check()?;
    }
    let t = tables();
    for (c, src) in components.iter_mut().zip(coef) {
        for (src, dst) in src
            .chunks_exact(64)
            .zip(c.coefficients.chunks_exact_mut(64))
        {
            for k in 0..64 {
                dst[t.zz_table[k] as usize] = src[k];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probability_quantization_and_table_accounting() {
        assert_eq!(probabilities(&[Count { ones: 0, total: 0 }]), [32768]);
        assert_eq!(probabilities(&[Count { ones: 0, total: 8 }]), [6554]);
        assert_eq!(probabilities(&[Count { ones: 8, total: 8 }]), [58982]);
        let mut p = vec![32768; 256];
        p[0] = 1;
        p[255] = 65535;
        let table = serialize(8, 1, &p);
        assert_eq!(table.len(), HEADER + 8);
        assert_eq!(parse(&table).unwrap(), p);
        p.fill(1234);
        let table = serialize(8, 1, &p);
        assert_eq!(table.len(), HEADER + 512);
        assert_eq!(parse(&table).unwrap(), p);
        let mut bad = table;
        bad.push(0);
        assert!(parse(&bad).is_err());
    }
    #[test]
    fn static_roundtrip_no_online_updates_and_exact_budget() {
        for components in [
            vec![super::super::tests::fixture(3, 3, 1, 1, 1)],
            vec![
                super::super::tests::fixture(4, 6, 2, 2, 1),
                super::super::tests::fixture(2, 3, 1, 1, 2),
                super::super::tests::fixture(2, 3, 1, 1, 3),
            ],
        ] {
            let encoded = super::super::encode_static(&components, 1 << 20, 10).unwrap();
            assert!(encoded.accounting.modelled_bits > 0);
            assert!(encoded.accounting.nondefault_buckets > 0);
            assert_eq!(
                encoded.accounting.total_bytes,
                encoded.table.len() + encoded.streams.iter().map(Vec::len).sum::<usize>()
            );
            let mut decoded = components.clone();
            for c in &mut decoded {
                c.coefficients.fill(0);
            }
            let refs: Vec<_> = encoded.streams.iter().map(Vec::as_slice).collect();
            decode_static(&mut decoded, &encoded.table, &refs).unwrap();
            assert_eq!(decoded, components);
            assert!(encode_static(&components, encoded.accounting.total_bytes - 1, 10).is_err());
            assert_eq!(
                encode_static(&components, encoded.accounting.total_bytes, 10)
                    .unwrap()
                    .streams,
                encoded.streams
            );
            let mut appended = encoded.streams.clone();
            appended[0].push(0);
            let refs: Vec<_> = appended.iter().map(Vec::as_slice).collect();
            assert!(decode_static(&mut decoded, &encoded.table, &refs).is_err());
        }
    }
    #[test]
    fn keys_have_explicit_namespace_and_event_identity() {
        assert_ne!(
            context_key(1, 0, 0, 1, 2, &[]),
            context_key(1, 1, 0, 1, 2, &[])
        );
        assert_ne!(
            context_key(1, 0, 1, 1, 2, &[(3, 4)]),
            context_key(1, 0, 2, 1, 2, &[(3, 4)])
        );
        assert_ne!(
            context_key(1, 0, 2, 1, 2, &[(3, 4)]),
            context_key(1, 0, 2, 1, 2, &[(3, 5)])
        );
    }
}

#[cfg(test)]
mod structured_tests {
    use super::*;
    #[test]
    fn projections_pool_only_declared_fields() {
        let a = [(3, 4), (5, 6)];
        let b = [(3, 4), (5, 7)];
        assert_ne!(
            context_key(1, 0, 2, 1, 2, &a),
            context_key(1, 0, 2, 1, 2, &b)
        );
        assert_eq!(
            context_key(2, 0, 2, 1, 2, &a),
            context_key(2, 0, 2, 1, 99, &b)
        );
        assert_eq!(
            context_key(3, 0, 2, 1, 2, &a),
            context_key(3, 0, 2, 1, 2, &[(99, 99)])
        );
        assert_ne!(
            context_key(3, 0, 2, 1, 2, &a),
            context_key(3, 0, 2, 1, 99, &a)
        );
        for profile in [2, 3] {
            assert_eq!(
                context_key(profile, 0, 1, 1, 2, &a),
                context_key(profile, 0, 1, 1, 2, &b)
            );
        }
    }
    #[test]
    fn all_profiles_frozen_decode_and_preserve_event_counts() {
        let components = vec![super::super::tests::fixture(3, 3, 1, 1, 2)];
        let mut event_counts = None;
        for profile in 1..=3 {
            let encoded = encode_static_profile(&components, 1 << 20, 10, profile).unwrap();
            assert_eq!(encoded.table[4], profile);
            assert_eq!(encoded.accounting.profile, profile);
            let counts = (
                encoded.accounting.modelled_bits,
                encoded.accounting.direct_bits,
            );
            if let Some(expected) = event_counts {
                assert_eq!(counts, expected)
            }
            event_counts = Some(counts);
            let mut decoded = components.clone();
            for c in &mut decoded {
                c.coefficients.fill(0);
            }
            decode_static(
                &mut decoded,
                &encoded.table,
                &encoded
                    .streams
                    .iter()
                    .map(Vec::as_slice)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            assert_eq!(decoded, components);
            if profile == 1 {
                let original = encode_static(&components, 1 << 20, 10).unwrap();
                assert_eq!(encoded.table, original.table);
                assert_eq!(encoded.streams, original.streams);
            }
            let mut bad = encoded.table;
            bad[4] = 0;
            assert!(parse(&bad).is_err());
        }
        assert!(encode_static_profile(&components, 1 << 20, 10, 4).is_err());
    }
}
