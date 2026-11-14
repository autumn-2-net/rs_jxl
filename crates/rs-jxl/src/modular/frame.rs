// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Shared MA tree/model assembly from enc_modular.cc::ComputeTree and EncodeGlobalInfo.
use super::{
    encode::{self, Group, ModularOptions, Prepared},
    group, training, weighted,
    weighted_history::{CACHE_BYTES, History},
};
use crate::encode::budget::{self, SectionBudget, TrialEncode, complete_or_return};
use crate::execution;
use crate::{Image, Result, bit_writer::BitWriter, entropy::stream::Model};

type EncodedShared = (Vec<u8>, Vec<Vec<u8>>);

pub(crate) fn encode_shared(
    image: &Image,
    groups: &[Group],
    options: &ModularOptions,
    limit: usize,
    patches: Option<&crate::tools::patches::PatchDictionary>,
) -> Result<(Vec<u8>, Vec<Vec<u8>>)> {
    let prepared = execution::try_map(groups, |_, g| encode::prepare(image, *g, options))?;
    encode_prepared_shared(None, prepared, options, limit, patches, image.bits())
}

/// Global metadata and every AC group participate in one model; histories and
/// predictor headers stay local. Empty groups carry no entropy state.
pub(super) fn encode_prepared_shared(
    global: Option<Prepared>,
    groups: Vec<Prepared>,
    options: &ModularOptions,
    limit: usize,
    patches: Option<&crate::tools::patches::PatchDictionary>,
    bits: u8,
) -> Result<(Vec<u8>, Vec<Vec<u8>>)> {
    encode_prepared_shared_trial(global, groups, options, limit, patches, bits, None)?
        .into_complete()
}

pub(super) fn encode_prepared_shared_trial(
    global: Option<Prepared>,
    groups: Vec<Prepared>,
    options: &ModularOptions,
    limit: usize,
    patches: Option<&crate::tools::patches::PatchDictionary>,
    bits: u8,
    budget: Option<&SectionBudget>,
) -> Result<TrialEncode<EncodedShared>> {
    encode_prepared_shared_borrowed_trial(
        global.as_ref(),
        &groups,
        options,
        limit,
        patches,
        bits,
        budget,
    )
}

pub(super) fn encode_prepared_shared_borrowed_trial(
    global: Option<&Prepared>,
    groups: &[Prepared],
    options: &ModularOptions,
    limit: usize,
    patches: Option<&crate::tools::patches::PatchDictionary>,
    bits: u8,
    budget: Option<&SectionBudget>,
) -> Result<TrialEncode<EncodedShared>> {
    let _scope = crate::profile::scope("modular.encode_shared");
    let single = groups.len() == 1 && global.is_none();
    let budget = budget.filter(|_| !single);
    let mut prepared = Vec::with_capacity(groups.len() + 1);
    let empty = Prepared {
        planes: Vec::new(),
        transforms: Vec::new(),
        wp: weighted::Header::default(),
        stream_id: 0,
    };
    if !single {
        prepared.push(global.unwrap_or(&empty));
    }
    prepared.extend(groups.iter());
    let views: Vec<_> = prepared
        .iter()
        .filter(|g| g.planes.iter().any(|c| !c.samples.is_empty()))
        .map(|g| training::TrainingStream {
            channels: &g.planes,
            wp: g.wp,
            stream_id: g.stream_id,
        })
        .collect();
    let predefined =
        encode::predefined(prepared.iter().flat_map(|g| g.planes.iter()), options, bits)?;
    let histories: Vec<_> = if predefined.is_none() && !options.refinement.enabled() {
        let mut remaining = CACHE_BYTES;
        prepared
            .iter()
            .map(|g| {
                let history = History::new(&g.planes, g.wp, options)?;
                remaining = remaining.checked_sub(history.bytes())?;
                Some(history)
            })
            .collect()
    } else {
        Vec::new()
    };
    if crate::profile::enabled() {
        crate::profile::note(
            "modular.shared_history",
            format_args!(
                "streams={} cached={} reserved_bytes={}",
                prepared.len(),
                histories.iter().flatten().count(),
                histories
                    .iter()
                    .flatten()
                    .map(History::bytes)
                    .sum::<usize>(),
            ),
        );
    }
    let tree = if let Some(tree) = predefined {
        tree
    } else {
        let cached: Vec<_> = histories.iter().filter_map(Option::as_ref).collect();
        training::learn_with_histories(&views, options, &cached)?
    };
    if options.refinement.enabled() {
        let cache_raw = options.lz77 && super::refinement::reuse_literals(&tree, options);
        let raw_cache = super::refinement::LiteralCache::default();
        if super::refinement::single_pass(options)
            && super::refinement::reuse_literals(&tree, options)
        {
            *raw_cache.borrow_mut() = Some(shared_raw(&prepared, &tree, options, &[])?);
        }
        let raw_reused = std::cell::Cell::new(0usize);
        let (global, encoded) = super::refinement::choose(
            Some(&raw_cache),
            &tree,
            &views,
            options,
            |candidate| {
                if super::refinement::single_pass(options)
                    && let Some(mut raw) = raw_cache.borrow_mut().take()
                    && let Some(count) =
                        super::leaf_offset::remap_literals(&tree, candidate, &mut raw)?
                {
                    let parsed = parse_shared_raw(&prepared, candidate, options, &raw)?;
                    raw_reused.set(raw_reused.get() + count);
                    return Ok(parsed);
                }
                if cache_raw && std::ptr::eq(candidate, &tree) && raw_cache.borrow().is_none() {
                    let raw = shared_raw(&prepared, candidate, options, &[])?;
                    let parsed = parse_shared_raw(&prepared, candidate, options, &raw)?;
                    *raw_cache.borrow_mut() = Some(raw);
                    return Ok(parsed);
                }
                if let Some(raw) = raw_cache.borrow_mut().as_mut()
                    && let Some(count) = super::leaf_offset::remap_literals(&tree, candidate, raw)?
                {
                    let parsed = parse_shared_raw(&prepared, candidate, options, raw)?;
                    raw_reused.set(raw_reused.get() + count);
                    return Ok(parsed);
                }
                shared_streams(&prepared, candidate, options, &[])
            },
            |tree, model, streams, scratch| {
                let output = write_shared(
                    single, &prepared, tree, model, streams, scratch, limit, patches, None,
                )?
                .into_complete()?;
                // Omitted/empty TOC entries are constant across trials. Compare
                // unpadded entry bits so their final alignment cannot reverse
                // a win. A collapsed frame has only one nonempty stream.
                let cost = std::iter::once(&output.0).chain(&output.1).try_fold(
                    0usize,
                    |sum, bytes| {
                        let entry = crate::streaming::size_entry_bits(bytes.len())?;
                        bytes
                            .len()
                            .checked_mul(8)
                            .and_then(|n| n.checked_add(entry))
                            .and_then(|n| sum.checked_add(n))
                            .ok_or(crate::Error::Limit("refinement section bits"))
                    },
                )?;
                Ok((cost, output))
            },
        )?;
        if let Some(receipt) = &options.refinement.receipt {
            receipt.record_raw_reuse(raw_reused.get());
        }
        if single {
            return Ok(TrialEncode::Complete((global, encoded)));
        }
        // Only the selected final sections charge the caller's pruning budget.
        let global = complete_or_return!(budget::section(budget, || Ok(global)));
        let mut selected = Vec::with_capacity(encoded.len());
        for bytes in encoded {
            selected.push(complete_or_return!(budget::section(budget, || Ok(bytes))));
        }
        return Ok(TrialEncode::Complete((global, selected)));
    }
    let (streams, lz77) = shared_streams(&prepared, &tree, options, &histories)?;
    let model = Model::build_with_options(
        &streams,
        tree.leaves() + usize::from(lz77),
        lz77,
        options.entropy,
        &options.entropy_options,
    )?;
    write_shared(
        single, &prepared, &tree, &model, &streams, options, limit, patches, budget,
    )
}

fn shared_raw(
    prepared: &[&Prepared],
    tree: &super::tree::Tree,
    options: &ModularOptions,
    histories: &[Option<History<'_>>],
) -> Result<Vec<Vec<crate::tools::lz77::Literal>>> {
    let parallel_channels =
        super::tokens::shared_channel_parallel(prepared.iter().copied(), options.parallel_tokens)?;
    let tokenize = |index: usize, g: &Prepared| {
        super::tokens::tokenize_with_history(
            &g.planes,
            tree,
            &g.wp,
            options.previous_channels,
            g.stream_id,
            parallel_channels,
            histories.get(index).and_then(Option::as_ref),
        )
    };
    if parallel_channels {
        prepared
            .iter()
            .enumerate()
            .map(|(i, g)| tokenize(i, g))
            .collect()
    } else {
        execution::try_map(prepared, |i, g| tokenize(i, g))
    }
}

fn shared_streams(
    prepared: &[&Prepared],
    tree: &super::tree::Tree,
    options: &ModularOptions,
    histories: &[Option<History<'_>>],
) -> Result<(Vec<Vec<crate::entropy::stream::Token>>, bool)> {
    let parallel_channels =
        super::tokens::shared_channel_parallel(prepared.iter().copied(), options.parallel_tokens)?;
    let streams = if options.lz77 && options.source_lz77.is_some() {
        let raw = shared_raw(prepared, tree, options, histories)?;
        return parse_shared_raw(prepared, tree, options, &raw);
    } else {
        let encode = |index: usize, g: &Prepared| {
            encode::group_stream_with_history(
                &g.planes,
                tree,
                options,
                &g.wp,
                g.stream_id,
                parallel_channels,
                histories.get(index).and_then(Option::as_ref),
            )
        };
        if parallel_channels {
            prepared
                .iter()
                .enumerate()
                .map(|(i, g)| encode(i, g))
                .collect::<Result<Vec<_>>>()?
        } else {
            execution::try_map(prepared, |i, g| encode(i, g))?
        }
    };
    let lz77 = streams.iter().any(|s| s.1);
    let streams: Vec<_> = streams.into_iter().map(|s| s.0).collect();
    Ok((streams, lz77))
}

fn parse_shared_raw(
    prepared: &[&Prepared],
    tree: &super::tree::Tree,
    options: &ModularOptions,
    raw: &[Vec<crate::tools::lz77::Literal>],
) -> Result<(Vec<Vec<crate::entropy::stream::Token>>, bool)> {
    if raw.len() != prepared.len() {
        return Err(crate::Error::Encoding("offset raw stream count".into()));
    }
    let widths: Vec<_> = prepared
        .iter()
        .map(|g| g.planes.iter().map(|c| c.width).max().unwrap_or(1))
        .collect();
    if options.lz77
        && let Some(source_lz77) = &options.source_lz77
    {
        let inputs: Vec<_> = raw.iter().map(Vec::as_slice).collect();
        let outcome = {
            let _scope = crate::profile::scope("lz77.parse_shared");
            crate::tools::lz77::parse_source_streams_with_state(&inputs, &widths, source_lz77)?
        };
        let streams = outcome
            .commands
            .into_iter()
            .zip(widths)
            .map(|(commands, width)| {
                encode::commands_to_tokens(commands, tree.leaves(), width).map(|s| s.0)
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok((streams, outcome.enabled));
    }
    let streams = execution::try_map(raw, |index, tokens| {
        encode::parse_group_tokens(tokens, tree.leaves(), widths[index], options)
    })?;
    let lz77 = streams.iter().any(|s| s.1);
    Ok((streams.into_iter().map(|s| s.0).collect(), lz77))
}

#[allow(clippy::too_many_arguments)]
fn write_shared(
    single: bool,
    prepared: &[&Prepared],
    tree: &super::tree::Tree,
    model: &Model,
    streams: &[Vec<crate::entropy::stream::Token>],
    options: &ModularOptions,
    limit: usize,
    patches: Option<&crate::tools::patches::PatchDictionary>,
    budget: Option<&SectionBudget>,
) -> Result<TrialEncode<EncodedShared>> {
    let global = complete_or_return!(budget::section(budget, || {
        let mut global = BitWriter::new(limit);
        crate::patch_frame::write_dictionary(&mut global, patches)?;
        global.write(1, 1)?;
        global.write(1, 1)?;
        encode::write_tree_and_model(tree, model, &mut global, options)?;
        group::write_header(&mut global, &prepared[0].transforms, &prepared[0].wp, true)?;
        if !streams[0].is_empty() {
            model.write_stream(&streams[0], &mut global)?;
        }
        global.zero_pad_to_byte()?;
        Ok(global.into_bytes()?)
    }));
    if single {
        return Ok(TrialEncode::Complete((Vec::new(), vec![global])));
    }
    let encoded = complete_or_return!(budget::sections(budget, &prepared[1..], |i, g| {
        let stream = &streams[i + 1];
        if g.planes.is_empty() {
            return Ok(Vec::new());
        }
        let mut writer = BitWriter::new(limit);
        group::write_header(&mut writer, &g.transforms, &g.wp, true)?;
        if !stream.is_empty() {
            model.write_stream(stream, &mut writer)?;
        }
        writer.zero_pad_to_byte()?;
        Ok(writer.into_bytes()?)
    }));
    Ok(TrialEncode::Complete((global, encoded)))
}

/// Encode a prepared independent stream without repeating WP/cost searches.
/// An empty local stream has no header in enc_modular.cc::EncodeStream.
pub(super) fn encode_prepared_independent(
    g: &Prepared,
    options: &ModularOptions,
    limit: usize,
    global: bool,
    patches: Option<&crate::tools::patches::PatchDictionary>,
    bits: u8,
) -> Result<Vec<u8>> {
    let _scope = crate::profile::scope("modular.encode_independent");
    if !global && g.planes.is_empty() {
        return Ok(Vec::new());
    }
    let mut writer = BitWriter::new(limit);
    if global {
        crate::patch_frame::write_dictionary(&mut writer, patches)?;
        writer.write(1, 1)?;
        writer.write(1, 0)?;
    }
    group::write_header(&mut writer, &g.transforms, &g.wp, false)?;
    if g.planes.iter().any(|c| !c.samples.is_empty()) {
        let predefined = encode::predefined(g.planes.iter(), options, bits)?;
        let history = if predefined.is_none() {
            super::weighted_history::History::new(&g.planes, g.wp, options)
        } else {
            None
        };
        let tree = if let Some(tree) = predefined {
            tree
        } else {
            training::learn_local_with_history(
                &[training::TrainingStream {
                    channels: &g.planes,
                    wp: g.wp,
                    // ModularGenericCompress learns its one-image domain at
                    // index 0; tokenization below still uses the wire stream ID.
                    stream_id: if options.sampling.is_some() {
                        0
                    } else {
                        g.stream_id
                    },
                }],
                options,
                history.as_ref(),
            )?
        };
        if options.refinement.enabled() {
            super::refinement::write_local(
                &tree,
                &g.planes,
                &g.wp,
                g.stream_id,
                options,
                &mut writer,
            )?;
            writer.zero_pad_to_byte()?;
            return Ok(writer.into_bytes()?);
        }
        let (tokens, has_matches) = encode::group_stream_with_history(
            &g.planes,
            &tree,
            options,
            &g.wp,
            g.stream_id,
            false,
            history.as_ref(),
        )?;
        let streams = [tokens];
        let model = Model::build_with_options(
            &streams,
            tree.leaves() + usize::from(has_matches),
            has_matches,
            options.entropy,
            &options.entropy_options,
        )?;
        encode::write_tree_and_model(&tree, &model, &mut writer, options)?;
        model.write_stream(&streams[0], &mut writer)?;
    }
    writer.zero_pad_to_byte()?;
    Ok(writer.into_bytes()?)
}

#[cfg(test)]
mod tests {
    use super::super::{encode::FixedTree, group::Channel, predictor::Predictor};
    use super::*;

    #[test]
    fn shared_source_lz_offset_repack_preserves_empty_groups_widths_and_batch() {
        use super::super::tree::{Node, Tree};
        use crate::entropy::{HybridUintConfig, options::UintConfigStrategy};
        use crate::tools::lz77::{SourceConfig, SourceLz77Method};
        let before = Tree::new(vec![Node::Leaf {
            predictor: Predictor::Left,
            offset: i32::MAX,
            multiplier: 1,
        }])
        .unwrap();
        let after = Tree::new(vec![Node::Leaf {
            predictor: Predictor::Left,
            offset: i32::MIN,
            multiplier: 1,
        }])
        .unwrap();
        let groups: Vec<_> = [0, 17, 9, 0]
            .into_iter()
            .enumerate()
            .map(|(id, width)| Prepared {
                planes: if width == 0 {
                    vec![]
                } else {
                    vec![Channel {
                        width,
                        height: 16,
                        hshift: 0,
                        vshift: 0,
                        samples: (0..width * 16)
                            .map(|i| [i32::MIN, i32::MAX, -17, 33][i % 4])
                            .collect(),
                    }]
                },
                transforms: vec![],
                wp: weighted::Header::default(),
                stream_id: id,
            })
            .collect();
        let prepared: Vec<_> = groups.iter().collect();
        let mut options = ModularOptions::default();
        options.entropy_options.uint_config =
            UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap());
        options.source_lz77 = Some(SourceConfig {
            method: SourceLz77Method::Optc256,
            length_uint_config: options.entropy_options.lz77_length_config,
            min_symbol: Some(options.entropy_options.lz77_min_symbol),
            ..Default::default()
        });
        let mut raw = groups
            .iter()
            .map(|g| {
                super::super::tokens::tokenize(&g.planes, &before, &g.wp, 0, g.stream_id, false)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let baseline = parse_shared_raw(&prepared, &before, &options, &raw).unwrap();
        assert_eq!(
            baseline,
            shared_streams(&prepared, &before, &options, &[]).unwrap()
        );
        assert_eq!(
            super::super::leaf_offset::remap_literals(&before, &after, &mut raw).unwrap(),
            Some(416)
        );
        let repacked = parse_shared_raw(&prepared, &after, &options, &raw).unwrap();
        assert_eq!(
            repacked,
            shared_streams(&prepared, &after, &options, &[]).unwrap()
        );
        assert!(repacked.0[0].is_empty() && repacked.0[3].is_empty());

        // Exercise the actual shared refinement closure, not just its parser.
        let receipt = std::sync::Arc::new(super::super::refinement::Receipt::default());
        let biased = [1, 2].map(|id| Prepared {
            planes: vec![Channel {
                width: 32,
                height: 16,
                hshift: 0,
                vshift: 0,
                samples: vec![1000; 512],
            }],
            transforms: vec![],
            wp: weighted::Header::default(),
            stream_id: id,
        });
        options.fixed_tree = Some(FixedTree::Zero);
        options.refinement.offsets = Some(super::super::refinement::OffsetOptions {
            mode: super::super::refinement::OffsetMode::Distribution,
            ..Default::default()
        });
        options.refinement.receipt = Some(receipt.clone());
        encode_prepared_shared_borrowed_trial(None, &biased, &options, 1 << 20, None, 16, None)
            .unwrap()
            .into_complete()
            .unwrap();
        assert_eq!(receipt.snapshot().offset_raw_reused_tokens, 1024);
        assert_eq!(receipt.snapshot().offset_proposals, 1);
        assert_eq!(receipt.snapshot().offset_token_fits, 1);
    }

    #[test]
    fn refinement_credits_only_selected_sections_and_preserves_cutoff() {
        use std::sync::{Arc, atomic::Ordering};
        let groups: Vec<_> = (1..=2)
            .map(|stream_id| Prepared {
                planes: vec![Channel {
                    width: 32,
                    height: 16,
                    hshift: 0,
                    vshift: 0,
                    samples: (0..512).map(|i| i * 17 % 256).collect(),
                }],
                transforms: Vec::new(),
                wp: weighted::Header::default(),
                stream_id,
            })
            .collect();
        let receipt = Arc::new(super::super::refinement::Receipt::default());
        let options = ModularOptions {
            lz77: false,
            refinement: super::super::refinement::Options {
                entropy: true,
                receipt: Some(receipt.clone()),
                ..Default::default()
            },
            ..Default::default()
        };
        let budget = SectionBudget::new(Some(1 << 19), 1 << 20).unwrap();
        let (global, encoded) = encode_prepared_shared_borrowed_trial(
            None,
            &groups,
            &options,
            1 << 20,
            None,
            8,
            Some(&budget),
        )
        .unwrap()
        .into_complete()
        .unwrap();
        assert_eq!(receipt.snapshot().models, 2);
        assert_eq!(budget.sections_started.load(Ordering::Relaxed), 3);
        assert_eq!(
            budget.section_bytes(),
            global.len() + encoded.iter().map(Vec::len).sum::<usize>()
        );

        let budget = SectionBudget::new(Some(1), 1 << 20).unwrap();
        let bounded = encode_prepared_shared_borrowed_trial(
            None,
            &groups,
            &options,
            1 << 20,
            None,
            8,
            Some(&budget),
        )
        .unwrap();
        assert!(matches!(bounded, TrialEncode::Bounded { lower_bound: 1 }));
        assert_eq!(budget.sections_started.load(Ordering::Relaxed), 1);
        assert_eq!(budget.section_bytes(), global.len());
    }

    #[test]
    fn independent_stream_keeps_prepared_wp_and_uses_selected_tree_writer() {
        let prepared = Prepared {
            planes: vec![Channel {
                width: 4,
                height: 2,
                hshift: 0,
                vshift: 0,
                samples: vec![0, -1, 2, 3, -4, 5, 6, -7],
            }],
            transforms: Vec::new(),
            wp: weighted::Header::preset(4).unwrap(),
            stream_id: 21,
        };
        for source_tree_entropy in [false, true] {
            let options = ModularOptions {
                predictor: Predictor::Gradient,
                fixed_tree: Some(FixedTree::GradientLeaf),
                weighted_preset: 0,
                weighted_candidates: Some(2),
                source_tree_entropy,
                lz77: false,
                ..ModularOptions::default()
            };
            let bytes =
                encode_prepared_independent(&prepared, &options, 1 << 20, false, None, 8).unwrap();
            let mut expected = BitWriter::new(1 << 20);
            group::write_header(&mut expected, &prepared.transforms, &prepared.wp, false).unwrap();
            let tree = encode::predefined(prepared.planes.iter(), &options, 8)
                .unwrap()
                .unwrap();
            encode::write_tree(&tree, &mut expected, &options).unwrap();
            let prefix_bits = expected.bits_written();
            expected.zero_pad_to_byte().unwrap();
            let expected = expected.into_bytes().unwrap();
            for bit in 0..prefix_bits {
                assert_eq!(
                    (bytes[bit / 8] >> (bit % 8)) & 1,
                    (expected[bit / 8] >> (bit % 8)) & 1,
                    "bit {bit}, source={source_tree_entropy}"
                );
            }
        }
    }

    #[test]
    fn empty_local_stream_has_no_header_but_dcglobal_always_does() {
        let prepared = Prepared {
            planes: Vec::new(),
            transforms: Vec::new(),
            wp: weighted::Header::default(),
            stream_id: 0,
        };
        let options = ModularOptions::default();
        assert!(
            encode_prepared_independent(&prepared, &options, 64, false, None, 8)
                .unwrap()
                .is_empty()
        );
        assert!(
            !encode_prepared_independent(&prepared, &options, 64, true, None, 8)
                .unwrap()
                .is_empty()
        );
    }
}
