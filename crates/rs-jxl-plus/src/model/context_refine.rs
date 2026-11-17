//! Test-only causal, hierarchical odds APM over the unchanged v7 bit model.
//!
//! Every profile preserves coefficient traversal, predictions and expert
//! learning. The refinement receives only decoder-visible event IDs, the base
//! probability, and previously coded bits. No offline statistics are required.
//! Descriptor mode 5 is experimental syntax, not standard JXL or PJA syntax.

use super::ma_context_tree::corrected_probability;
use super::*;

const MODE: u8 = 5;
const VERSION: u8 = 1;
const DESCRIPTOR_BYTES: usize = 14;
const NAMESPACES: usize = 2;
const KINDS: usize = 3;
const BANKS: usize = 64;
const COARSE_CONTEXTS: usize = NAMESPACES * KINDS * BANKS;
const STEP_LOG: u8 = 7;
const STEP: usize = 1 << STEP_LOG;
const KNOTS: usize = 33;
const FINE_BITS: u8 = 13;
const FINE_CONTEXTS: usize = 1 << FINE_BITS;
const COARSE_RATE: u8 = 8;
const FINE_RATE: u8 = 9;
const COARSE_LIMIT: i32 = 512;
const FINE_LIMIT: i32 = 256;
const FIXED_POINT: i64 = 65536;
const MAX_STATES_BYTES: usize = 2 << 20;
const MAX_BANDS: usize = 64;

#[derive(Clone, Debug)]
pub(crate) struct Accounting {
    pub profile: u8,
    pub bands: usize,
    pub descriptor_bytes: usize,
    pub stream_bytes: usize,
    pub total_bytes: usize,
    /// Peak added calibration state, not a sum over independently reset bands.
    pub states_bytes: usize,
    pub modelled_bits: u64,
    pub direct_bits: u64,
}
#[derive(Clone, Debug)]
pub(crate) struct Encoded {
    pub descriptor: Vec<u8>,
    pub streams: Vec<Vec<u8>>,
    pub accounting: Accounting,
}

#[derive(Clone, Copy)]
struct Profile(u8);
impl Profile {
    fn new(id: u8) -> Result<Self> {
        if id > 3 {
            return Err(Error::Unsupported("refined probability profile"));
        }
        Ok(Self(id))
    }
    fn active(self) -> bool {
        self.0 != 0
    }
    fn fine(self) -> bool {
        self.0 >= 2
    }
    fn states_bytes(self) -> usize {
        if !self.active() {
            0
        } else {
            (COARSE_CONTEXTS + if self.fine() { FINE_CONTEXTS } else { 0 })
                * KNOTS
                * std::mem::size_of::<i32>()
        }
    }
}
fn invalid() -> Error {
    Error::Invalid("refined probability descriptor/event")
}

fn descriptor(profile: Profile, bands: usize) -> Vec<u8> {
    let mut d = vec![MODE, VERSION, profile.0];
    d.extend_from_slice(&(bands as u16).to_le_bytes());
    d.extend_from_slice(&[
        if profile.active() { STEP_LOG } else { 0 },
        if profile.active() { KNOTS as u8 } else { 0 },
        if profile.active() { COARSE_RATE } else { 0 },
        if profile.fine() { FINE_RATE } else { 0 },
        if profile.fine() { FINE_BITS } else { 0 },
    ]);
    d.extend_from_slice(
        &(if profile.active() {
            COARSE_LIMIT as u16
        } else {
            0
        })
        .to_le_bytes(),
    );
    d.extend_from_slice(&(if profile.fine() { FINE_LIMIT as u16 } else { 0 }).to_le_bytes());
    d
}
fn parse(d: &[u8]) -> Result<(Profile, usize)> {
    if d.len() != DESCRIPTOR_BYTES {
        return Err(invalid());
    }
    if d[0] != MODE || d[1] != VERSION {
        return Err(Error::Unsupported("refined probability layout"));
    }
    let profile = Profile::new(d[2])?;
    let bands = u16::from_le_bytes([d[3], d[4]]) as usize;
    if !(1..=MAX_BANDS).contains(&bands) || d != descriptor(profile, bands) {
        return Err(invalid());
    }
    Ok((profile, bands))
}
fn allocated(n: usize) -> Result<Vec<i32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| Error::Limit("refined probability allocation"))?;
    values.resize(n, 0);
    Ok(values)
}

#[derive(Clone, Copy)]
struct Event {
    coarse_base: usize,
    fine_base: usize,
}
#[inline(always)]
fn event(profile: Profile, namespace: u8, kind: u8, bank: usize, index: usize) -> Option<Event> {
    if namespace as usize >= NAMESPACES || kind as usize >= KINDS || bank >= BANKS {
        return None;
    }
    let coarse = (namespace as usize * KINDS + kind as usize) * BANKS + bank;
    let fine = if profile.fine() {
        let selector = if profile.0 == 2 { index >> 4 } else { index } as u64;
        // Fixed SplitMix-style avalanche, with explicit unsigned wrapping.
        // Context collisions are intentional; the state budget is fixed.
        let mut key = (coarse as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ selector;
        key = (key ^ (key >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        key = (key ^ (key >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((key ^ (key >> 31)) as usize) & (FINE_CONTEXTS - 1)
    } else {
        0
    };
    Some(Event {
        coarse_base: coarse * KNOTS,
        fine_base: fine * KNOTS,
    })
}
#[inline(always)]
fn location(p: u32) -> (usize, usize) {
    let stretch = crate::probability::tables().stretch[(p >> 4) as usize] as i32;
    // The frozen stretch table's range is -2048..=2047. The probability
    // guard in bit already bounds the table index; no per-event clamp needed.
    let x = (stretch + 2048) as usize;
    (x >> STEP_LOG, x & (STEP - 1))
}
#[inline(always)]
fn pair_mut(states: &mut [i32], start: usize) -> &mut [i32; 2] {
    // Event bases and the frozen table guarantee a pair inside its 33-node row.
    // Validate the slice once, then reuse it for both prediction and learning.
    (&mut states[start..start + 2])
        .try_into()
        .expect("refinement node pair")
}
#[inline(always)]
fn interpolate(states: &[i32; 2], fraction: usize) -> i64 {
    let a = i64::from(states[0]);
    let difference = i64::from(states[1]) - a;
    // Keep a inside the numerator: a + trunc(difference*f/128) changes
    // results when the two terms cross zero. This is the original numerator.
    ((a << STEP_LOG) + difference * fraction as i64) / STEP as i64
}
#[inline(always)]
fn learn<const RATE: u8, const LIMIT: i32>(states: &mut [i32; 2], fraction: usize, error: i32) {
    // Exact cancellation: error*256*weight/(2^RATE*128) equals
    // error*weight/2^(RATE-1), with signed division truncating toward zero.
    // |error|<=65535 and weight<=128: the product is below 2^23. With
    // RATE=8/9, state +/-512*65536 plus one update also fits signed i32.
    let denominator = 1i32 << (RATE - 1);
    let bound = LIMIT * FIXED_POINT as i32;
    for (offset, weight) in [(0, STEP - fraction), (1, fraction)] {
        let update = error * weight as i32 / denominator;
        states[offset] = (states[offset] + update).clamp(-bound, bound);
    }
}

struct Refined<C> {
    coder: C,
    profile: Profile,
    coarse: Vec<i32>,
    fine: Vec<i32>,
    pending: Option<Event>,
    failed: bool,
    modelled_bits: u64,
    direct_bits: u64,
}
impl<C: BitCoder> Refined<C> {
    fn new(coder: C, profile: Profile) -> Result<Self> {
        if profile.states_bytes() > MAX_STATES_BYTES {
            return Err(Error::Limit("refined probability states"));
        }
        Ok(Self {
            coder,
            profile,
            coarse: allocated(if profile.active() {
                COARSE_CONTEXTS * KNOTS
            } else {
                0
            })?,
            fine: allocated(if profile.fine() {
                FINE_CONTEXTS * KNOTS
            } else {
                0
            })?,
            pending: None,
            failed: false,
            modelled_bits: 0,
            direct_bits: 0,
        })
    }
    fn check(&self) -> Result<()> {
        if self.failed || self.pending.is_some() {
            Err(invalid())
        } else {
            Ok(())
        }
    }
}
impl<C: BitCoder> BitCoder for Refined<C> {
    fn encoding(&self) -> bool {
        self.coder.encoding()
    }
    #[inline]
    fn probability_context(
        &mut self,
        namespace: u8,
        kind: u8,
        bank: usize,
        index: usize,
        _refs: &[(usize, usize)],
    ) {
        if self.pending.is_some() {
            self.failed = true;
        }
        self.pending = event(self.profile, namespace, kind, bank, index);
        if self.pending.is_none() {
            self.failed = true;
        }
    }
    #[inline]
    fn bit(&mut self, p: u32, value: i32) -> i32 {
        let Some(event) = self.pending.take() else {
            self.direct_bits += 1;
            return self.coder.bit(p, value);
        };
        self.modelled_bits += 1;
        if !self.profile.active() {
            return self.coder.bit(p, value);
        }
        if !(1..=65535).contains(&p) {
            self.failed = true;
            return self.coder.bit(p, value);
        }
        let (knot, fraction) = location(p);
        let coarse = pair_mut(&mut self.coarse, event.coarse_base + knot);
        let fine = if self.profile.fine() {
            Some(pair_mut(&mut self.fine, event.fine_base + knot))
        } else {
            None
        };
        let mut delta = interpolate(coarse, fraction);
        if let Some(states) = fine.as_ref() {
            delta += interpolate(states, fraction);
        }
        let delta = (delta / FIXED_POINT).clamp(-512, 512) as i32;
        let q = corrected_probability(p, delta);
        let bit = self.coder.bit(q, value);
        let error = bit * 65536 - q as i32;
        learn::<COARSE_RATE, COARSE_LIMIT>(coarse, fraction, error);
        if let Some(states) = fine {
            learn::<FINE_RATE, FINE_LIMIT>(states, fraction, error);
        }
        bit
    }
}

pub(crate) fn encode_refined(
    components: &[Component],
    profile: u8,
    bands: usize,
    max: usize,
) -> Result<Encoded> {
    let profile = Profile::new(profile)?;
    let (rows, _source_bands) = validate(components)?;
    if bands == 0 || bands > rows || bands > MAX_BANDS {
        return Err(Error::Invalid("refined probability bands"));
    }
    let descriptor = descriptor(profile, bands);
    let mut remaining = max
        .checked_sub(descriptor.len())
        .ok_or(Error::Limit("refined probability output"))?;
    let mut coef = zigzag(components, true);
    let mut streams = Vec::with_capacity(bands);
    let (mut modelled_bits, mut direct_bits) = (0, 0);
    for band in 0..bands {
        // Both the original v7 state and the added calibration start afresh.
        let mut cd = Refined::new(Encoder::new(remaining), profile)?;
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        cd.check()?;
        modelled_bits += cd.modelled_bits;
        direct_bits += cd.direct_bits;
        let bytes = cd.coder.finish()?;
        remaining = remaining
            .checked_sub(bytes.len())
            .ok_or(Error::Limit("refined probability output"))?;
        streams.push(bytes);
    }
    let stream_bytes = streams.iter().map(Vec::len).sum::<usize>();
    let accounting = Accounting {
        profile: profile.0,
        bands,
        descriptor_bytes: descriptor.len(),
        stream_bytes,
        total_bytes: descriptor.len() + stream_bytes,
        states_bytes: profile.states_bytes(),
        modelled_bits,
        direct_bits,
    };
    Ok(Encoded {
        descriptor,
        streams,
        accounting,
    })
}
pub(crate) fn decode_refined(
    components: &mut [Component],
    descriptor: &[u8],
    streams: &[&[u8]],
) -> Result<()> {
    let (profile, bands) = parse(descriptor)?;
    let (rows, _source_bands) = validate(components)?;
    if bands > rows || streams.len() != bands {
        return Err(Error::Invalid("refined probability band count"));
    }
    let mut coef = zigzag(components, false);
    for (band, &stream) in streams.iter().enumerate() {
        let mut cd = Refined::new(Decoder::new(stream), profile)?;
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        cd.check()?;
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
    fn original_streams(components: &[Component], bands: usize) -> Vec<Vec<u8>> {
        let (rows, _) = validate(components).unwrap();
        let mut coef = zigzag(components, true);
        (0..bands)
            .map(|band| {
                let mut cd = Encoder::new(1 << 20);
                code_band(
                    &mut cd,
                    components,
                    &mut coef,
                    rows * band / bands,
                    rows * (band + 1) / bands,
                )
                .unwrap();
                cd.finish().unwrap()
            })
            .collect()
    }
    #[test]
    fn all_profiles_preserve_events_and_control_matches_original_bands() {
        let components = vec![super::super::tests::fixture(3, 4, 1, 1, 2)];
        for bands in [1, 2] {
            let mut events = None;
            for profile in 0..=3 {
                let encoded = encode_refined(&components, profile, bands, 1 << 20).unwrap();
                let accounting = &encoded.accounting;
                assert_eq!(accounting.descriptor_bytes, DESCRIPTOR_BYTES);
                assert_eq!(
                    accounting.total_bytes,
                    encoded.descriptor.len() + encoded.streams.iter().map(Vec::len).sum::<usize>()
                );
                assert!(accounting.states_bytes <= MAX_STATES_BYTES);
                let counts = (accounting.modelled_bits, accounting.direct_bits);
                if let Some(previous) = events {
                    assert_eq!(counts, previous);
                }
                events = Some(counts);
                if profile == 0 {
                    assert_eq!(encoded.streams, original_streams(&components, bands));
                }
                let mut decoded = components.clone();
                for component in &mut decoded {
                    component.coefficients.fill(0);
                }
                let streams = encoded
                    .streams
                    .iter()
                    .map(Vec::as_slice)
                    .collect::<Vec<_>>();
                decode_refined(&mut decoded, &encoded.descriptor, &streams).unwrap();
                assert_eq!(decoded, components);
                assert_eq!(
                    encode_refined(&components, profile, bands, accounting.total_bytes)
                        .unwrap()
                        .streams,
                    encoded.streams
                );
                assert!(matches!(
                    encode_refined(&components, profile, bands, accounting.total_bytes - 1),
                    Err(Error::Limit(_))
                ));
            }
        }
    }
    #[test]
    fn direct_bits_do_not_train_and_identity_initialization() {
        let profile = Profile::new(3).unwrap();
        let mut cd = Refined::new(Encoder::new(4096), profile).unwrap();
        cd.bit(32768, 1);
        assert!(cd.coarse.iter().all(|&x| x == 0) && cd.fine.iter().all(|&x| x == 0));
        let event = event(profile, 0, 2, 4, 17).unwrap();
        let (knot, fraction) = location(12345);
        assert_eq!(
            interpolate(pair_mut(&mut cd.coarse, event.coarse_base + knot), fraction),
            0
        );
        assert_eq!(corrected_probability(12345, 0), 12345);
        cd.probability_context(0, 2, 4, 17, &[]);
        cd.bit(12345, 1);
        cd.check().unwrap();
        assert!(cd.coarse.iter().any(|&x| x != 0) && cd.fine.iter().any(|&x| x != 0));
        let before = (cd.coarse.clone(), cd.fine.clone());
        cd.bit(32768, 0);
        assert_eq!(before, (cd.coarse.clone(), cd.fine.clone()));
        assert_eq!((cd.modelled_bits, cd.direct_bits), (1, 2));
    }
    #[test]
    fn descriptor_and_canonical_termination_bounds() {
        let components = vec![super::super::tests::fixture(2, 2, 1, 1, 1)];
        let encoded = encode_refined(&components, 2, 1, 1 << 20).unwrap();
        for n in 0..encoded.descriptor.len() {
            assert!(parse(&encoded.descriptor[..n]).is_err());
        }
        let mut appended = encoded.descriptor.clone();
        appended.push(0);
        assert!(parse(&appended).is_err());
        for i in 5..encoded.descriptor.len() {
            let mut bad = encoded.descriptor.clone();
            bad[i] ^= 1;
            assert!(parse(&bad).is_err());
        }
        let mut bytes = encoded.streams[0].clone();
        bytes.push(0);
        assert!(decode_refined(&mut components.clone(), &encoded.descriptor, &[&bytes]).is_err());
        bytes = encoded.streams[0].clone();
        bytes.pop();
        assert!(decode_refined(&mut components.clone(), &encoded.descriptor, &[&bytes]).is_err());
        assert!(encode_refined(&components, 4, 1, 1 << 20).is_err());
        assert!(encode_refined(&components, 1, 0, 1 << 20).is_err());
        assert!(encode_refined(&components, 1, 3, 1 << 20).is_err());
    }
    #[test]
    fn bounded_interpolation_and_selector_projection() {
        let mut states = allocated(KNOTS).unwrap();
        for _ in 0..10000 {
            learn::<COARSE_RATE, COARSE_LIMIT>(pair_mut(&mut states, 16), 64, 65535);
        }
        assert!(
            states
                .iter()
                .all(|&x| i64::from(x).abs() <= i64::from(COARSE_LIMIT) * FIXED_POINT)
        );
        assert!(interpolate(pair_mut(&mut states, 16), 64) > 0);
        assert_eq!(
            event(Profile(2), 0, 2, 1, 16).unwrap().fine_base,
            event(Profile(2), 0, 2, 1, 31).unwrap().fine_base
        );
        assert_ne!(
            event(Profile(3), 0, 2, 1, 16).unwrap().fine_base,
            event(Profile(3), 0, 2, 1, 31).unwrap().fine_base
        );
        assert!(event(Profile(3), 2, 0, 0, 0).is_none());
    }
    fn original_interpolate(pair: [i32; 2], fraction: usize) -> i64 {
        (i64::from(pair[0]) * (STEP - fraction) as i64 + i64::from(pair[1]) * fraction as i64)
            / STEP as i64
    }
    fn original_learn(
        mut pair: [i32; 2],
        fraction: usize,
        error: i32,
        rate: u8,
        limit: i32,
    ) -> [i32; 2] {
        let denominator = (1i64 << rate) * STEP as i64;
        let bound = i64::from(limit) * FIXED_POINT;
        for (offset, weight) in [(0, STEP - fraction), (1, fraction)] {
            let update = i64::from(error) * 256 * weight as i64 / denominator;
            pair[offset] = (i64::from(pair[offset]) + update).clamp(-bound, bound) as i32;
        }
        pair
    }
    #[test]
    fn optimized_interpolation_preserves_signed_cross_zero_truncation() {
        for pair in [
            [-1, 0],
            [1, 0],
            [-2, 1],
            [2, -1],
            [-33554432, 33554432],
            [33554432, -33554432],
        ] {
            for fraction in 0..STEP {
                assert_eq!(
                    interpolate(&pair, fraction),
                    original_interpolate(pair, fraction)
                );
            }
        }
        assert_eq!(interpolate(&[-1, 0], 64), 0);
        assert_eq!(-1 + ((1i64 * 64) / 128), -1); // The tempting distributed form differs.
        for p in 1..=65535 {
            let stretch = crate::probability::tables().stretch[(p >> 4) as usize] as i32;
            let x = (stretch + 2048).clamp(0, 4095) as usize;
            assert_eq!(location(p), (x >> STEP_LOG, x & (STEP - 1)));
        }
    }
    #[test]
    fn optimized_learning_matches_original_formula_for_bounded_random_states() {
        fn compare<const RATE: u8, const LIMIT: i32>(pair: [i32; 2], fraction: usize, error: i32) {
            let mut result = pair;
            learn::<RATE, LIMIT>(&mut result, fraction, error);
            assert_eq!(result, original_learn(pair, fraction, error, RATE, LIMIT));
            assert_eq!(
                interpolate(&pair, fraction),
                original_interpolate(pair, fraction)
            );
        }
        for error in [-65535, -129, -128, -127, -1, 0, 1, 127, 128, 129, 65535] {
            for fraction in 0..STEP {
                compare::<COARSE_RATE, COARSE_LIMIT>([-33554432, 33554432], fraction, error);
                compare::<FINE_RATE, FINE_LIMIT>([-16777216, 16777216], fraction, error);
            }
        }
        let mut random = 0x68419b27u32;
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random
        };
        for _ in 0..20000 {
            let coarse = [
                (next() % 67108865) as i32 - 33554432,
                (next() % 67108865) as i32 - 33554432,
            ];
            let fine = [
                (next() % 33554433) as i32 - 16777216,
                (next() % 33554433) as i32 - 16777216,
            ];
            let fraction = next() as usize & (STEP - 1);
            let error = (next() % 131071) as i32 - 65535;
            compare::<COARSE_RATE, COARSE_LIMIT>(coarse, fraction, error);
            compare::<FINE_RATE, FINE_LIMIT>(fine, fraction, error);
        }
    }
}
