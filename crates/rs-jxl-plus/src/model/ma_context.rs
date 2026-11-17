//! Test-only MA-selected online calibration of the unchanged v7 probability.
//!
//! The coefficient traversal, predictors, counters and mixer updates remain the
//! PJA MIT v7 port (1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA and
//! NOTICE-PJA). Context calibration and its bounded experimental descriptor are
//! new experiments, not PJA or standard JXL syntax. MA Tree validation is reused
//! from the BSD-attributed rs-jxl port; no coefficient transform is performed.
use super::ma_context_tree::{Observation, corrected_probability, train};
use super::*;
use rs_jxl::modular::{
    predictor::Predictor,
    tree::{Node, Tree},
};

const SAMPLES: usize = 32768;
const BINS: usize = 16;
const LIMIT: i64 = 512 * 65536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContextMode {
    Single,
    Fixed,
    Learned,
}
impl ContextMode {
    fn id(self) -> u8 {
        match self {
            Self::Single => 0,
            Self::Fixed => 1,
            Self::Learned => 2,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct ContextTraining {
    pub tree: Tree,
    pub sample_count: usize,
    pub modelled_bits: u64,
    pub direct_bits: u64,
}
#[derive(Clone, Debug)]
pub(crate) struct ContextAccounting {
    pub mode: u8,
    pub rate: u8,
    pub leaves: usize,
    pub bins: usize,
    pub modelled_bits: u64,
    pub direct_bits: u64,
    pub descriptor_bytes: usize,
    pub stream_bytes: usize,
    pub total_bytes: usize,
}
#[derive(Clone, Debug)]
pub(crate) struct ContextEncoded {
    pub descriptor: Vec<u8>,
    pub streams: Vec<Vec<u8>>,
    pub accounting: ContextAccounting,
}
fn invalid() -> Error {
    Error::Invalid("MA probability context descriptor")
}
fn properties(
    namespace: u8,
    kind: u8,
    bank: usize,
    index: usize,
    refs: &[(usize, usize)],
) -> [i32; 8] {
    // Bounds are source-model IDs, comfortably inside i32; no pointers or
    // encoder-only values enter the selection properties.
    let first = refs.first().copied();
    [
        i32::from(namespace),
        i32::from(kind),
        bank as i32,
        index as i32,
        refs.len() as i32,
        first.map_or(-1, |x| x.0 as i32),
        first.map_or(-1, |x| x.1 as i32),
        refs.get(1).map_or(-1, |x| x.1 as i32),
    ]
}
struct Observer {
    pending: Option<[i32; 8]>,
    samples: Vec<Observation>,
    modelled: u64,
    direct: u64,
    random: u64,
}
impl BitCoder for Observer {
    fn encoding(&self) -> bool {
        true
    }
    fn probability_context(
        &mut self,
        ns: u8,
        kind: u8,
        bank: usize,
        index: usize,
        refs: &[(usize, usize)],
    ) {
        self.pending = Some(properties(ns, kind, bank, index, refs));
    }
    fn bit(&mut self, probability: u32, value: i32) -> i32 {
        let bit = i32::from(value != 0);
        if let Some(properties) = self.pending.take() {
            self.modelled += 1;
            let observation = Observation {
                properties,
                probability,
                bit,
            };
            if self.samples.len() < SAMPLES {
                self.samples.push(observation);
            } else {
                // Deterministic Algorithm-R reservoir, independent of event
                // values and properties. Storage never grows with the image.
                self.random ^= self.random << 13;
                self.random ^= self.random >> 7;
                self.random ^= self.random << 17;
                let index = self.random % self.modelled;
                if index < SAMPLES as u64 {
                    self.samples[index as usize] = observation;
                }
            }
        } else {
            self.direct += 1;
        }
        bit
    }
}
pub(crate) fn train_context(components: &[Component]) -> Result<ContextTraining> {
    let (rows, bands) = validate(components)?;
    let mut coef = zigzag(components, true);
    let mut cd = Observer {
        pending: None,
        samples: Vec::with_capacity(SAMPLES),
        modelled: 0,
        direct: 0,
        random: 0x4d41_4354_5831_7239,
    };
    for band in 0..bands {
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        if cd.pending.is_some() {
            return Err(invalid());
        }
    }
    let tree = train(&cd.samples, 8)?;
    check_tree(&tree)?;
    Ok(ContextTraining {
        tree,
        sample_count: cd.samples.len(),
        modelled_bits: cd.modelled,
        direct_bits: cd.direct,
    })
}
fn leaf() -> Node {
    Node::Leaf {
        predictor: Predictor::Zero,
        offset: 0,
        multiplier: 1,
    }
}
fn selected_tree(training: &ContextTraining, mode: ContextMode) -> Result<Tree> {
    match mode {
        ContextMode::Learned => {
            check_tree(&training.tree)?;
            Ok(training.tree.clone())
        }
        ContextMode::Single => Tree::new(vec![leaf()]).map_err(|_| invalid()),
        ContextMode::Fixed => Tree::new(vec![
            Node::Split {
                property: 0,
                threshold: 0,
                greater: 2,
                less_equal: 1,
            },
            Node::Split {
                property: 1,
                threshold: 0,
                greater: 4,
                less_equal: 3,
            },
            Node::Split {
                property: 1,
                threshold: 0,
                greater: 6,
                less_equal: 5,
            },
            leaf(),
            Node::Split {
                property: 1,
                threshold: 1,
                greater: 8,
                less_equal: 7,
            },
            leaf(),
            Node::Split {
                property: 1,
                threshold: 1,
                greater: 10,
                less_equal: 9,
            },
            leaf(),
            leaf(),
            leaf(),
            leaf(),
        ])
        .map_err(|_| invalid()),
    }
}
fn check_tree(tree: &Tree) -> Result<()> {
    if tree.nodes().len() > 15 || tree.leaves() > 8 {
        return Err(invalid());
    }
    for node in tree.nodes() {
        match node {
            Node::Split { property, .. } if *property < 8 => {}
            Node::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
            } => {}
            _ => return Err(invalid()),
        }
    }
    Ok(())
}
struct Calibrated<'a, C> {
    coder: C,
    tree: &'a Tree,
    pending: Option<[i32; 8]>,
    states: Vec<i64>,
    rate: u8,
    enabled: bool,
    modelled: u64,
    direct: u64,
}
impl<'a, C> Calibrated<'a, C> {
    fn new(coder: C, tree: &'a Tree, rate: u8) -> Self {
        Self {
            coder,
            tree,
            pending: None,
            states: vec![0; tree.leaves() * BINS],
            rate,
            enabled: true,
            modelled: 0,
            direct: 0,
        }
    }
}
impl<C: BitCoder> BitCoder for Calibrated<'_, C> {
    fn encoding(&self) -> bool {
        self.coder.encoding()
    }
    fn probability_context(
        &mut self,
        ns: u8,
        kind: u8,
        bank: usize,
        index: usize,
        refs: &[(usize, usize)],
    ) {
        self.pending = Some(properties(ns, kind, bank, index, refs));
    }
    fn bit(&mut self, p: u32, value: i32) -> i32 {
        let Some(properties) = self.pending.take() else {
            self.direct += 1;
            return self.coder.bit(p, value);
        };
        // Validated trees only inspect these eight available properties.
        let context = self
            .tree
            .lookup(&properties)
            .expect("validated context tree")
            .0;
        let index = context * BINS + ((p as usize) >> 12).min(BINS - 1);
        let delta = (self.states[index] / 65536) as i32;
        let q = if self.enabled {
            corrected_probability(p, delta)
        } else {
            p
        };
        let bit = self.coder.bit(q, value);
        if self.enabled {
            let error = i64::from(bit) * 65536 - i64::from(q);
            self.states[index] =
                (self.states[index] + error * 256 / (1i64 << self.rate)).clamp(-LIMIT, LIMIT);
        }
        self.modelled += 1;
        bit
    }
}
fn descriptor(tree: &Tree, mode: ContextMode, rate: u8, max: usize) -> Result<Vec<u8>> {
    check_tree(tree)?;
    let mut raw = b"MCTX1".to_vec();
    raw.extend_from_slice(&[mode.id(), rate, BINS as u8, tree.nodes().len() as u8]);
    for node in tree.nodes() {
        match node {
            Node::Leaf { .. } => raw.push(1),
            Node::Split {
                property,
                threshold,
                greater,
                less_equal,
            } => {
                raw.extend_from_slice(&[0, *property as u8]);
                raw.extend_from_slice(&threshold.to_le_bytes());
                raw.extend_from_slice(&[*greater as u8, *less_equal as u8]);
            }
        }
    }
    let cap = max
        .checked_sub(5)
        .ok_or(Error::Limit("MA context output"))?;
    let mut cd = Encoder::new(cap);
    crate::raw_model::code(&mut cd, &mut raw, 256)?;
    let coded = cd.finish()?;
    let len = u32::try_from(coded.len()).map_err(|_| Error::Limit("MA context descriptor"))?;
    let mut descriptor = Vec::with_capacity(5 + coded.len());
    descriptor.push(3);
    descriptor.extend_from_slice(&len.to_le_bytes());
    descriptor.extend_from_slice(&coded);
    Ok(descriptor)
}
fn parse(descriptor: &[u8]) -> Result<(Tree, u8, u8)> {
    if descriptor.len() < 5 || descriptor[0] != 3 {
        return Err(invalid());
    }
    let len = u32::from_le_bytes(descriptor[1..5].try_into().unwrap()) as usize;
    if len != descriptor.len() - 5 {
        return Err(invalid());
    }
    let mut cd = Decoder::new(&descriptor[5..]);
    let mut raw = Vec::new();
    crate::raw_model::code(&mut cd, &mut raw, 256)?;
    cd.finish_check()?;
    if raw.len() < 9
        || &raw[..5] != b"MCTX1"
        || raw[5] > 2
        || !matches!(raw[6], 6 | 8)
        || raw[7] != 16
        || !(1..=15).contains(&raw[8])
    {
        return Err(invalid());
    }
    let mut pos = 9;
    let mut nodes = Vec::new();
    for _ in 0..raw[8] {
        let tag = *raw.get(pos).ok_or_else(invalid)?;
        pos += 1;
        match tag {
            1 => nodes.push(leaf()),
            0 => {
                let data = raw.get(pos..pos + 7).ok_or_else(invalid)?;
                nodes.push(Node::Split {
                    property: usize::from(data[0]),
                    threshold: i32::from_le_bytes(data[1..5].try_into().unwrap()),
                    greater: usize::from(data[5]),
                    less_equal: usize::from(data[6]),
                });
                pos += 7;
            }
            _ => return Err(invalid()),
        }
    }
    if pos != raw.len() {
        return Err(invalid());
    }
    let tree = Tree::new(nodes).map_err(|_| invalid())?;
    check_tree(&tree)?;
    Ok((tree, raw[5], raw[6]))
}
pub(crate) fn encode_context(
    components: &[Component],
    training: &ContextTraining,
    mode: ContextMode,
    rate: u8,
    max_output: usize,
) -> Result<ContextEncoded> {
    if !matches!(rate, 6 | 8) {
        return Err(invalid());
    }
    let tree = selected_tree(training, mode)?;
    let descriptor = descriptor(&tree, mode, rate, max_output)?;
    let mut remaining = max_output
        .checked_sub(descriptor.len())
        .ok_or(Error::Limit("MA context output"))?;
    let (rows, bands) = validate(components)?;
    let mut coef = zigzag(components, true);
    let mut streams = Vec::with_capacity(bands);
    let (mut modelled, mut direct) = (0, 0);
    for band in 0..bands {
        let mut cd = Calibrated::new(Encoder::new(remaining), &tree, rate);
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        modelled += cd.modelled;
        direct += cd.direct;
        if cd.pending.is_some() {
            return Err(invalid());
        }
        let bytes = cd.coder.finish()?;
        remaining = remaining
            .checked_sub(bytes.len())
            .ok_or(Error::Limit("MA context output"))?;
        streams.push(bytes);
    }
    if modelled != training.modelled_bits || direct != training.direct_bits {
        return Err(Error::Invalid("MA training event count mismatch"));
    }
    let stream_bytes = streams.iter().map(Vec::len).sum::<usize>();
    let accounting = ContextAccounting {
        mode: mode.id(),
        rate,
        leaves: tree.leaves(),
        bins: BINS,
        modelled_bits: modelled,
        direct_bits: direct,
        descriptor_bytes: descriptor.len(),
        stream_bytes,
        total_bytes: descriptor.len() + stream_bytes,
    };
    Ok(ContextEncoded {
        descriptor,
        streams,
        accounting,
    })
}
pub(crate) fn decode_context(
    components: &mut [Component],
    descriptor: &[u8],
    streams: &[&[u8]],
) -> Result<()> {
    let (tree, _mode, rate) = parse(descriptor)?;
    let (rows, bands) = validate(components)?;
    if streams.len() != bands {
        return Err(Error::Invalid("MA context band count"));
    }
    let mut coef = zigzag(components, false);
    for (band, stream) in streams.iter().enumerate() {
        let mut cd = Calibrated::new(Decoder::new(stream), &tree, rate);
        code_band(
            &mut cd,
            components,
            &mut coef,
            rows * band / bands,
            rows * (band + 1) / bands,
        )?;
        cd.coder.finish_check()?;
        if cd.pending.is_some() {
            return Err(invalid());
        }
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
    fn notification_is_consumed_once_and_direct_bits_do_not_train() {
        let tree = Tree::new(vec![leaf()]).unwrap();
        let mut cd = Calibrated::new(Encoder::new(4096), &tree, 6);
        cd.probability_context(0, 0, 1, 2, &[]);
        cd.bit(32768, 1);
        assert_eq!(cd.states[8], 131072);
        let state = cd.states.clone();
        cd.direct(0, 13);
        assert_eq!(cd.states, state);
        assert_eq!((cd.modelled, cd.direct), (1, 13));
        let bytes = cd.coder.finish().unwrap();
        let mut inverse = Calibrated::new(Decoder::new(&bytes), &tree, 6);
        inverse.probability_context(0, 0, 1, 2, &[]);
        assert_eq!(inverse.bit(32768, 0), 1);
        assert_eq!(inverse.direct(0, 13), 0);
        assert_eq!(inverse.states, state);
        inverse.coder.finish_check().unwrap();
    }
    #[test]
    fn disabled_calibration_preserves_original_dynamic_stream() {
        let components = vec![super::super::tests::fixture(4, 4, 1, 1, 1)];
        let tree = Tree::new(vec![leaf()]).unwrap();
        let mut cd = Calibrated::new(Encoder::new(1 << 20), &tree, 6);
        cd.enabled = false;
        let mut coef = zigzag(&components, true);
        code_band(&mut cd, &components, &mut coef, 0, 4).unwrap();
        assert!(cd.modelled > 0);
        assert_eq!(
            vec![cd.coder.finish().unwrap()],
            encode(&components, 1 << 20).unwrap()
        );
    }
    #[test]
    fn context_variants_roundtrip_and_include_real_tree_cost() {
        let components = vec![super::super::tests::fixture(4, 4, 1, 1, 1)];
        let training = train_context(&components).unwrap();
        assert!(training.sample_count > 0 && training.sample_count <= SAMPLES);
        for mode in [
            ContextMode::Single,
            ContextMode::Fixed,
            ContextMode::Learned,
        ] {
            for rate in [6, 8] {
                let coded = encode_context(&components, &training, mode, rate, 1 << 20).unwrap();
                assert_eq!(
                    coded.accounting.total_bytes,
                    coded.descriptor.len() + coded.streams.iter().map(Vec::len).sum::<usize>()
                );
                let mut decoded = components.clone();
                decoded[0].coefficients.fill(0);
                let streams: Vec<_> = coded.streams.iter().map(Vec::as_slice).collect();
                decode_context(&mut decoded, &coded.descriptor, &streams).unwrap();
                assert_eq!(decoded, components);
                let mut appended = coded.descriptor.clone();
                appended.push(0);
                assert!(decode_context(&mut decoded, &appended, &streams).is_err());
                assert!(
                    encode_context(
                        &components,
                        &training,
                        mode,
                        rate,
                        coded.accounting.total_bytes - 1
                    )
                    .is_err()
                );
            }
        }
    }
}
