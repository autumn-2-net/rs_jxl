// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! MA tree representation, validation and breadth-first tokenization translated
//! from enc_ma.cc::TokenizeTree, dec_ma.cc::ValidateTree and context_predict.h.
//! Learning is deliberately separate; a serialized constant tree is not MA training.
use super::predictor::{Neighbors, Predictor};
use crate::entropy::{
    HybridUintConfig,
    options::{EntropyOptions, UintConfigStrategy},
};
use crate::tools::lz77::{self, Literal, SourceConfig};
use crate::{
    Error, Result,
    bit_writer::BitWriter,
    entropy::stream::{Coding, Model, Token},
};
use std::collections::VecDeque;

#[path = "token_tree.rs"]
mod token_tree;
pub(super) use token_tree::ChannelTree;

/// context_predict.h: split value/property, predictor, offset, multiplier log
/// and multiplier bits. An enabled LZ77 stream adds distance context six.
pub const NUM_TREE_CONTEXTS: usize = 6;

#[derive(Debug, Clone)]
pub enum Node {
    Split {
        property: usize,
        threshold: i32,
        greater: usize,
        less_equal: usize,
    },
    Leaf {
        predictor: Predictor,
        offset: i32,
        multiplier: u32,
    },
}
#[derive(Debug, Clone)]
pub struct Tree {
    nodes: Vec<Node>,
    contexts: Vec<usize>,
    leaves: usize,
}

pub fn pack_signed(value: i32) -> u32 {
    (value as u32).wrapping_shl(1) ^ ((value >> 31) as u32)
}

impl Tree {
    pub fn new(nodes: Vec<Node>) -> Result<Self> {
        if nodes.is_empty() || nodes.len() > (1 << 22) {
            return Err(Error::InvalidInput("invalid MA tree size".into()));
        }
        let mut seen = vec![false; nodes.len()];
        let mut contexts = vec![usize::MAX; nodes.len()];
        let mut queue = VecDeque::from([(0usize, vec![(i32::MIN, i32::MAX); 256], 0usize)]);
        let mut leaves = 0;
        while let Some((index, ranges, depth)) = queue.pop_front() {
            if index >= nodes.len() || seen[index] || depth >= 2048 {
                return Err(Error::InvalidInput(
                    "MA tree is cyclic, shared, too deep or out of bounds".into(),
                ));
            }
            seen[index] = true;
            match &nodes[index] {
                Node::Leaf {
                    predictor,
                    multiplier,
                    ..
                } => {
                    if (*predictor as u32) >= 14
                        || *multiplier == 0
                        || *multiplier > i32::MAX as u32
                    {
                        return Err(Error::InvalidInput("invalid MA leaf".into()));
                    }
                    contexts[index] = leaves;
                    leaves += 1;
                }
                Node::Split {
                    property,
                    threshold,
                    greater,
                    less_equal,
                } => {
                    if *property >= 256 {
                        return Err(Error::InvalidInput("invalid MA property".into()));
                    }
                    let (lo, hi) = ranges[*property];
                    if lo > *threshold || hi <= *threshold {
                        return Err(Error::InvalidInput(
                            "MA tree contains an unreachable branch".into(),
                        ));
                    }
                    let mut above = ranges.clone();
                    above[*property].0 = threshold + 1;
                    let mut below = ranges;
                    below[*property].1 = *threshold;
                    queue.push_back((*greater, above, depth + 1));
                    queue.push_back((*less_equal, below, depth + 1));
                }
            }
        }
        if seen.iter().any(|v| !v) {
            return Err(Error::InvalidInput("unreachable MA node".into()));
        }
        Ok(Self {
            nodes,
            contexts,
            leaves,
        })
    }
    pub fn channel_predictors(predictors: &[Predictor]) -> Result<Self> {
        if predictors.is_empty() || predictors.len() > 256 {
            return Err(Error::InvalidInput("invalid MA channel count".into()));
        }
        fn add(nodes: &mut Vec<Node>, predictors: &[Predictor], offset: usize) -> usize {
            let index = nodes.len();
            nodes.push(Node::Leaf {
                predictor: predictors[0],
                offset: 0,
                multiplier: 1,
            });
            if predictors.len() > 1 {
                let mid = predictors.len() / 2;
                let greater = add(nodes, &predictors[mid..], offset + mid);
                let less_equal = add(nodes, &predictors[..mid], offset);
                nodes[index] = Node::Split {
                    property: 0,
                    threshold: (offset + mid - 1) as i32,
                    greater,
                    less_equal,
                };
            }
            index
        }
        let mut nodes = Vec::new();
        add(&mut nodes, predictors, 0);
        Self::new(nodes)
    }
    pub fn leaves(&self) -> usize {
        self.leaves
    }
    /// Source-indexed nodes. Context IDs remain breadth-first and are rebuilt
    /// by Tree::new after callers merge/remap these nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn into_nodes(self) -> Vec<Node> {
        self.nodes
    }
    pub fn predictors(&self) -> Vec<Predictor> {
        let mut result = Vec::new();
        for node in &self.nodes {
            if let Node::Leaf { predictor, .. } = node
                && !result.contains(predictor)
            {
                result.push(*predictor);
            }
        }
        result
    }
    pub fn uses_weighted(&self) -> bool {
        self.nodes.iter().any(|n| {
            matches!(
                n,
                Node::Leaf {
                    predictor: Predictor::Weighted,
                    ..
                } | Node::Split { property: 15, .. }
            )
        })
    }
    pub fn lookup(&self, properties: &[i32]) -> Result<(usize, Predictor, i32, u32)> {
        let mut index = 0;
        loop {
            match self.nodes[index] {
                Node::Leaf {
                    predictor,
                    offset,
                    multiplier,
                } => return Ok((self.contexts[index], predictor, offset, multiplier)),
                Node::Split {
                    property,
                    threshold,
                    greater,
                    less_equal,
                } => {
                    let value = *properties
                        .get(property)
                        .ok_or_else(|| Error::Encoding("missing MA property".into()))?;
                    index = if value > threshold {
                        greater
                    } else {
                        less_equal
                    };
                }
            }
        }
    }
    pub fn tokens(&self) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut queue = VecDeque::from([0]);
        while let Some(index) = queue.pop_front() {
            match self.nodes[index] {
                Node::Leaf {
                    predictor,
                    offset,
                    multiplier,
                } => {
                    tokens.push(Token::literal(1, 0));
                    tokens.push(Token::literal(2, predictor as u32));
                    tokens.push(Token::literal(3, pack_signed(offset)));
                    let log = multiplier.trailing_zeros();
                    tokens.push(Token::literal(4, log));
                    tokens.push(Token::literal(5, (multiplier >> log) - 1));
                }
                Node::Split {
                    property,
                    threshold,
                    greater,
                    less_equal,
                } => {
                    tokens.push(Token::literal(1, property as u32 + 1));
                    tokens.push(Token::literal(0, pack_signed(threshold)));
                    queue.push_back(greater);
                    queue.push_back(less_equal);
                }
            }
        }
        tokens
    }
    pub fn write(&self, w: &mut BitWriter) -> Result<()> {
        let streams = [self.tokens()];
        let model = Model::build(&streams, NUM_TREE_CONTEXTS, false, Coding::Prefix)?;
        model.write_header(w)?;
        model.write_stream(&streams[0], w)
    }

    /// Source EncodeGlobalInfo / ModularGenericCompress tree entropy path.
    /// Pass the frame's ForModular entropy policy and selected source LZ77
    /// method (None disables LZ77). The legacy write method is unchanged.
    ///
    /// Tree tokens have no image_widths: distance_multiplier is zero. Source
    /// EncodeGlobalInfo installs its bucket override only AFTER writing the
    /// tree, so a residual config's bucket_matcher is deliberately disabled
    /// here. Careful-first-pass, method and wire fields are retained.
    /// Streaming mode builds a fresh model, as in ModularGenericCompress; it
    /// does not reuse another tree/residual model or inject missing symbols.
    /// The parser's accepted-result flag controls LZ enablement, including
    /// accepted literal-only results with an empty distance histogram.
    pub fn write_with_options(
        &self,
        w: &mut BitWriter,
        entropy: &EntropyOptions,
        source_lz77: Option<&SourceConfig>,
    ) -> Result<()> {
        let (tokens, enabled) = self.source_tokens(entropy, source_lz77)?;
        let streams = [tokens];
        let model = Model::build_with_options(
            &streams,
            NUM_TREE_CONTEXTS + usize::from(enabled),
            enabled,
            Coding::Source,
            entropy,
        )?;
        model.write_header(w)?;
        model.write_stream(&streams[0], w)
    }

    /// HistogramParams' ancillary-stream RLE method, distinct from the source
    /// fast/careful/optimal matchers. JPEG raw quantization tables use this.
    pub(crate) fn write_with_rle(&self, w: &mut BitWriter, entropy: &EntropyOptions) -> Result<()> {
        entropy.validate()?;
        let raw: Vec<_> = self
            .tokens()
            .into_iter()
            .map(|token| Literal {
                context: token.context as u32,
                value: token.value,
            })
            .collect();
        let literal_uint_config = match entropy.uint_config {
            UintConfigStrategy::Fixed(config) => config,
            UintConfigStrategy::Fast | UintConfigStrategy::Best => HybridUintConfig::default(),
        };
        let config = SourceConfig {
            literal_uint_config,
            length_uint_config: entropy.lz77_length_config,
            min_symbol: Some(entropy.lz77_min_symbol),
            ..SourceConfig::default()
        };
        let outcome = lz77::parse_source_rle_streams_with_state(&[&raw], &[0], &config)?;
        let enabled = outcome.enabled;
        let commands = outcome
            .commands
            .into_iter()
            .next()
            .expect("one tree stream");
        let (tokens, _) = super::encode::commands_to_tokens(commands, NUM_TREE_CONTEXTS, 0)?;
        let streams = [tokens];
        let model = Model::build_with_options(
            &streams,
            NUM_TREE_CONTEXTS + usize::from(enabled),
            enabled,
            Coding::Source,
            entropy,
        )?;
        model.write_header(w)?;
        model.write_stream(&streams[0], w)
    }

    fn source_tokens(
        &self,
        entropy: &EntropyOptions,
        source_lz77: Option<&SourceConfig>,
    ) -> Result<(Vec<Token>, bool)> {
        entropy.validate()?;
        let Some(config) = source_lz77 else {
            return Ok((self.tokens(), false));
        };
        let literal_config = match entropy.uint_config {
            UintConfigStrategy::Fixed(config) => config,
            UintConfigStrategy::Fast | UintConfigStrategy::Best => HybridUintConfig::default(),
        };
        if config.length_uint_config != entropy.lz77_length_config
            || config.min_symbol() != entropy.lz77_min_symbol
            || config.literal_uint_config != literal_config
        {
            return Err(Error::InvalidInput(
                "tree LZ77 cost and entropy wire configurations disagree".into(),
            ));
        }
        let raw: Vec<_> = self
            .tokens()
            .into_iter()
            .map(|token| Literal {
                context: token.context as u32,
                value: token.value,
            })
            .collect();
        let mut tree_config = config.clone();
        tree_config.bucket_matcher = false;
        let outcome = lz77::parse_source_with_state(&raw, 0, &tree_config)?;
        let (tokens, _) =
            super::encode::commands_to_tokens(outcome.commands, NUM_TREE_CONTEXTS, 0)?;
        Ok((tokens, outcome.enabled))
    }
}

/// context_predict.h::InitPropsRow + detail::Predict, without previous-channel
/// properties. Call on every pixel in raster order; reset at each channel.
#[derive(Default)]
pub struct Properties {
    previous_gradient: i32,
}
impl Properties {
    pub fn calculate(
        &mut self,
        n: Neighbors,
        x: usize,
        y: usize,
        channel: usize,
        group: usize,
        wp_error: i32,
    ) -> [i32; 16] {
        if x == 0 {
            self.previous_gradient = 0;
        }
        let gradient = n.left.wrapping_add(n.top).wrapping_sub(n.topleft);
        let values = [
            channel as i32,
            group as i32,
            y as i32,
            x as i32,
            n.top.wrapping_abs(),
            n.left.wrapping_abs(),
            n.top,
            n.left,
            n.left.wrapping_sub(self.previous_gradient),
            gradient,
            n.left.wrapping_sub(n.topleft),
            n.topleft.wrapping_sub(n.top),
            n.top.wrapping_sub(n.topright),
            n.top.wrapping_sub(n.toptop),
            n.left.wrapping_sub(n.leftleft),
            wp_error,
        ];
        self.previous_gradient = gradient;
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ancillary_rle_tree_is_bounded_and_embeds_at_every_bit_offset() {
        let tree = Tree::channel_predictors(&vec![Predictor::Zero; 128]).unwrap();
        let entropy = EntropyOptions::for_speed_tier(crate::entropy::options::SpeedTier::Tortoise);
        let mut reference = BitWriter::new(1 << 20);
        tree.write_with_rle(&mut reference, &entropy).unwrap();
        for offset in 0..8 {
            let mut actual = BitWriter::new(1 << 20);
            actual.write(offset, 0).unwrap();
            tree.write_with_rle(&mut actual, &entropy).unwrap();
            let mut expected = BitWriter::new(1 << 20);
            expected.write(offset, 0).unwrap();
            expected.append_unaligned(&reference).unwrap();
            assert_eq!(actual.bits_written(), expected.bits_written());
            actual.zero_pad_to_byte().unwrap();
            expected.zero_pad_to_byte().unwrap();
            assert_eq!(actual.into_bytes().unwrap(), expected.into_bytes().unwrap());
        }
        assert!(
            tree.write_with_rle(&mut BitWriter::new(1), &entropy)
                .is_err()
        );
    }
    #[test]
    fn tree_leaf_ids_follow_breadth_first_not_source_indices() {
        let tree = Tree::channel_predictors(&[
            Predictor::Zero,
            Predictor::Left,
            Predictor::Top,
            Predictor::Weighted,
        ])
        .unwrap();
        for channel in 0..4 {
            let (ctx, p, _, _) = tree.lookup(&[channel]).unwrap();
            assert_eq!(ctx, 3 - channel as usize);
            assert_eq!(
                p,
                [
                    Predictor::Zero,
                    Predictor::Left,
                    Predictor::Top,
                    Predictor::Weighted
                ][channel as usize]
            );
        }
        assert_eq!(tree.leaves(), 4);
        assert!(tree.uses_weighted());
    }
    #[test]
    fn rejects_cycles_shared_children_bad_leaves_and_ranges() {
        let leaf = Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
        };
        assert!(Tree::new(vec![]).is_err());
        assert!(
            Tree::new(vec![
                Node::Split {
                    property: 0,
                    threshold: 0,
                    greater: 0,
                    less_equal: 1
                },
                leaf.clone()
            ])
            .is_err()
        );
        assert!(
            Tree::new(vec![
                Node::Split {
                    property: 0,
                    threshold: 0,
                    greater: 1,
                    less_equal: 1
                },
                leaf.clone()
            ])
            .is_err()
        );
        assert!(
            Tree::new(vec![
                Node::Split {
                    property: 0,
                    threshold: i32::MAX,
                    greater: 1,
                    less_equal: 2
                },
                leaf.clone(),
                leaf
            ])
            .is_err()
        );
        assert!(
            Tree::new(vec![Node::Leaf {
                predictor: Predictor::Best,
                offset: 0,
                multiplier: 1
            }])
            .is_err()
        );
        assert!(
            Tree::new(vec![Node::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 0
            }])
            .is_err()
        );
    }
    #[test]
    fn offset_and_multiplier_tokenization_preserves_signed_values() {
        let tree = Tree::new(vec![Node::Leaf {
            predictor: Predictor::Top,
            offset: -17,
            multiplier: 12,
        }])
        .unwrap();
        let values: Vec<_> = tree.tokens().iter().map(|t| t.value).collect();
        assert_eq!(values, [0, 2, 33, 2, 2]);
    }

    #[test]
    fn source_tree_writer_uses_selected_entropy_policy_at_every_tier() {
        use crate::entropy::options::{ModularEntropyParams, SpeedTier};
        let predictors: Vec<_> = [
            Predictor::Zero,
            Predictor::Left,
            Predictor::Top,
            Predictor::Gradient,
        ]
        .into_iter()
        .cycle()
        .take(128)
        .collect();
        let tree = Tree::channel_predictors(&predictors).unwrap();
        let streams = [tree.tokens()];
        for tier in 0..=9u8 {
            let entropy = EntropyOptions::for_modular(ModularEntropyParams {
                speed_tier: SpeedTier::try_from(tier).unwrap(),
                ..Default::default()
            });
            let model = Model::build_with_options(
                &streams,
                NUM_TREE_CONTEXTS,
                false,
                Coding::Source,
                &entropy,
            )
            .unwrap();
            assert!(model.uses_ans());
            let mut expected = BitWriter::new(1 << 20);
            let mut actual = BitWriter::new(1 << 20);
            expected.write(3, 5).unwrap();
            actual.write(3, 5).unwrap();
            model.write_header(&mut expected).unwrap();
            model.write_stream(&streams[0], &mut expected).unwrap();
            tree.write_with_options(&mut actual, &entropy, None)
                .unwrap();
            assert_eq!(actual.bits_written(), expected.bits_written());
            actual.zero_pad_to_byte().unwrap();
            expected.zero_pad_to_byte().unwrap();
            assert_eq!(actual.bytes().unwrap(), expected.bytes().unwrap());
        }
    }

    #[test]
    fn source_tree_lz_uses_distance_context_six_and_no_spatial_or_bucket_override() {
        use crate::entropy::options::ModularEntropyParams;
        use crate::tools::lz77::SourceLz77Method;
        let tree = Tree::channel_predictors(&[Predictor::Left; 128]).unwrap();
        let raw = tree.tokens();
        let input: Vec<_> = raw
            .iter()
            .map(|t| Literal {
                context: t.context as u32,
                value: t.value,
            })
            .collect();
        let entropy = EntropyOptions::for_modular(ModularEntropyParams::default());
        for method in [
            SourceLz77Method::Lz77b1w3t,
            SourceLz77Method::Lz77b3w3t,
            SourceLz77Method::Optc8,
            SourceLz77Method::Optc256,
        ] {
            let config = SourceConfig {
                method,
                bucket_matcher: true,
                // Exercise match serialization even if the source savings gate
                // would decline this low-entropy fixed-predictor tree.
                require_global_savings: false,
                length_uint_config: entropy.lz77_length_config,
                min_symbol: Some(entropy.lz77_min_symbol),
                ..Default::default()
            };
            let (tokens, has_matches) = tree.source_tokens(&entropy, Some(&config)).unwrap();
            assert!(has_matches);
            let mut expected_config = config.clone();
            expected_config.bucket_matcher = false;
            let commands = lz77::parse_source(&input, 0, &expected_config).unwrap();
            assert_eq!(
                tokens,
                super::super::encode::commands_to_tokens(commands, 6, 0)
                    .unwrap()
                    .0
            );
            let mut values = Vec::new();
            let mut i = 0;
            while i < tokens.len() {
                let token = tokens[i];
                assert_eq!(token.context, raw[values.len()].context);
                if token.is_length {
                    let distance_token = tokens[i + 1];
                    assert_eq!(distance_token.context, NUM_TREE_CONTEXTS);
                    assert!(!distance_token.is_length);
                    let distance = distance_token.value as usize + 1;
                    assert!(distance <= values.len());
                    for _ in 0..token.value as usize + 3 {
                        values.push(values[values.len() - distance]);
                    }
                    i += 2;
                } else {
                    values.push(token.value);
                    i += 1;
                }
            }
            assert_eq!(values, raw.iter().map(|t| t.value).collect::<Vec<_>>());
            let mut writer = BitWriter::new(1 << 20);
            tree.write_with_options(&mut writer, &entropy, Some(&config))
                .unwrap();
            assert!(writer.bits_written() > 0);
        }
    }

    #[test]
    fn source_tree_rejects_wire_mismatch_before_writing() {
        use crate::entropy::options::ModularEntropyParams;
        let tree = Tree::channel_predictors(&[Predictor::Gradient]).unwrap();
        let entropy = EntropyOptions::for_modular(ModularEntropyParams::default());
        for config in [
            SourceConfig {
                min_symbol: Some(entropy.lz77_min_symbol + 1),
                ..Default::default()
            },
            SourceConfig {
                length_uint_config: HybridUintConfig::new(4, 0, 0).unwrap(),
                ..Default::default()
            },
            SourceConfig {
                literal_uint_config: HybridUintConfig::new(0, 0, 0).unwrap(),
                ..Default::default()
            },
        ] {
            for streaming_mode in [false, true] {
                let entropy = EntropyOptions {
                    streaming_mode,
                    ..entropy
                };
                let mut writer = BitWriter::new(1024);
                writer.write(3, 5).unwrap();
                assert!(
                    tree.write_with_options(&mut writer, &entropy, Some(&config))
                        .is_err()
                );
                assert_eq!(writer.bits_written(), 3);
            }
        }
    }

    #[test]
    fn fresh_streaming_tree_models_preserve_policy_and_lz_acceptance() {
        use crate::entropy::options::ModularEntropyParams;
        use crate::tools::lz77::SourceLz77Method;
        for predictors in [vec![Predictor::Gradient], vec![Predictor::Left; 128]] {
            let tree = Tree::channel_predictors(&predictors).unwrap();
            for strategy in [
                UintConfigStrategy::Fast,
                UintConfigStrategy::Best,
                UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap()),
            ] {
                let entropy = EntropyOptions {
                    streaming_mode: true,
                    uint_config: strategy,
                    ..EntropyOptions::for_modular(ModularEntropyParams::default())
                };
                let config = SourceConfig {
                    method: SourceLz77Method::Lz77b1w3t,
                    require_global_savings: false,
                    literal_uint_config: match strategy {
                        UintConfigStrategy::Fixed(config) => config,
                        _ => HybridUintConfig::default(),
                    },
                    ..Default::default()
                };
                for source_lz77 in [None, Some(&config)] {
                    let (tokens, enabled) = tree.source_tokens(&entropy, source_lz77).unwrap();
                    assert_eq!(enabled, source_lz77.is_some());
                    if source_lz77.is_some() && predictors.len() > 1 {
                        assert!(tokens.iter().any(|token| token.is_length));
                    }
                    let streams = [tokens];
                    let model = Model::build_with_options(
                        &streams,
                        NUM_TREE_CONTEXTS + usize::from(enabled),
                        enabled,
                        Coding::Source,
                        &entropy,
                    )
                    .unwrap();
                    if source_lz77.is_none() {
                        assert_eq!(model.uses_ans(), predictors.len() > 1);
                        if model.uses_ans() {
                            assert_eq!(model.log_alpha_size(), 8);
                        }
                    }
                    let mut expected = BitWriter::new(1 << 20);
                    let mut actual = BitWriter::new(1 << 20);
                    expected.write(3, 5).unwrap();
                    actual.write(3, 5).unwrap();
                    model.write_header(&mut expected).unwrap();
                    model.write_stream(&streams[0], &mut expected).unwrap();
                    tree.write_with_options(&mut actual, &entropy, source_lz77)
                        .unwrap();
                    assert_eq!(actual.bits_written(), expected.bits_written());
                    actual.zero_pad_to_byte().unwrap();
                    expected.zero_pad_to_byte().unwrap();
                    assert_eq!(actual.bytes().unwrap(), expected.bytes().unwrap());
                }
            }
        }
    }

    #[test]
    fn accepted_literal_only_tree_retains_lz_header_and_distance_context() {
        use crate::entropy::options::ModularEntropyParams;
        use crate::tools::lz77::SourceLz77Method;
        let tree = Tree::channel_predictors(&[Predictor::Gradient]).unwrap();
        let entropy = EntropyOptions::for_modular(ModularEntropyParams::default());
        for method in [SourceLz77Method::Optc8, SourceLz77Method::Optc256] {
            // Force acceptance to exercise this contract independently of the
            // source seed gate; this is not evidence of a default-gate input.
            let config = SourceConfig {
                method,
                require_global_savings: false,
                ..Default::default()
            };
            let (tokens, enabled) = tree.source_tokens(&entropy, Some(&config)).unwrap();
            assert!(enabled);
            assert_eq!(tokens, tree.tokens());
            assert!(tokens.iter().all(|token| !token.is_length));
            let model = Model::build_with_options(
                std::slice::from_ref(&tokens),
                7,
                true,
                Coding::Source,
                &entropy,
            )
            .unwrap();
            let mut expected = BitWriter::new(1024);
            model.write_header(&mut expected).unwrap();
            model.write_stream(&tokens, &mut expected).unwrap();
            let mut actual = BitWriter::new(1024);
            tree.write_with_options(&mut actual, &entropy, Some(&config))
                .unwrap();
            assert_eq!(actual.bits_written(), expected.bits_written());
            actual.zero_pad_to_byte().unwrap();
            expected.zero_pad_to_byte().unwrap();
            assert_eq!(actual.bytes().unwrap(), expected.bytes().unwrap());
        }
    }

    #[test]
    fn legacy_tree_write_retains_prefix_model_and_source_leaf_has_no_forced_lz() {
        use crate::entropy::options::ModularEntropyParams;
        let tree = Tree::channel_predictors(&[Predictor::Gradient]).unwrap();
        let streams = [tree.tokens()];
        let model = Model::build(&streams, 6, false, Coding::Prefix).unwrap();
        let mut expected = BitWriter::new(1024);
        model.write_header(&mut expected).unwrap();
        model.write_stream(&streams[0], &mut expected).unwrap();
        let mut actual = BitWriter::new(1024);
        tree.write(&mut actual).unwrap();
        assert_eq!(actual.bits_written(), expected.bits_written());
        actual.zero_pad_to_byte().unwrap();
        expected.zero_pad_to_byte().unwrap();
        assert_eq!(actual.bytes().unwrap(), expected.bytes().unwrap());
        let entropy = EntropyOptions::for_modular(ModularEntropyParams::default());
        let (tokens, matches) = tree
            .source_tokens(&entropy, Some(&SourceConfig::default()))
            .unwrap();
        assert!(!matches);
        assert_eq!(tokens, tree.tokens());
        let model =
            Model::build_with_options(&[tokens], 6, false, Coding::Source, &entropy).unwrap();
        assert!(!model.uses_ans());
    }

    #[test]
    fn node_access_and_reconstruction_rebuild_breadth_first_contexts() {
        let tree =
            Tree::channel_predictors(&[Predictor::Left, Predictor::Top, Predictor::Zero]).unwrap();
        let tokens = tree.tokens();
        assert_eq!(tree.nodes().len(), 5);
        let rebuilt = Tree::new(tree.into_nodes()).unwrap();
        assert_eq!(rebuilt.tokens(), tokens);
        let contexts: Vec<_> = (0..3)
            .map(|channel| rebuilt.lookup(&[channel]).unwrap().0)
            .collect();
        assert_eq!(contexts, [0, 2, 1]);
    }

    #[test]
    fn fixed000_tree_cost_configuration_is_not_silently_replaced_with420() {
        use crate::entropy::options::ModularEntropyParams;
        let tree = Tree::channel_predictors(&[Predictor::Gradient; 8]).unwrap();
        let mut entropy = EntropyOptions::for_modular(ModularEntropyParams::default());
        let uint000 = HybridUintConfig::new(0, 0, 0).unwrap();
        entropy.uint_config = UintConfigStrategy::Fixed(uint000);
        let mut config = SourceConfig::default();
        let mut writer = BitWriter::new(1024);
        assert!(
            tree.write_with_options(&mut writer, &entropy, Some(&config))
                .is_err()
        );
        assert_eq!(writer.bits_written(), 0);
        config.literal_uint_config = uint000;
        tree.write_with_options(&mut writer, &entropy, Some(&config))
            .unwrap();
        assert!(writer.bits_written() > 0);
    }
}
