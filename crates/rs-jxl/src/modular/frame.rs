// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Shared MA tree/model assembly from enc_modular.cc::ComputeTree and EncodeGlobalInfo.
use super::{
    encode::{self, Group, ModularOptions, Prepared},
    group, training, weighted,
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
    let tree = if let Some(tree) =
        encode::predefined(prepared.iter().flat_map(|g| g.planes.iter()), options, bits)?
    {
        tree
    } else {
        training::learn(&views, options)?
    };
    let parallel_channels =
        super::tokens::shared_channel_parallel(prepared.iter().copied(), options.parallel_tokens)?;
    let streams = if options.lz77
        && let Some(source_lz77) = &options.source_lz77
    {
        let tokenize = |g: &Prepared| {
            super::tokens::tokenize(
                &g.planes,
                &tree,
                &g.wp,
                options.previous_channels,
                g.stream_id,
                parallel_channels,
            )
        };
        let raw = if parallel_channels {
            prepared
                .iter()
                .map(|g| tokenize(g))
                .collect::<Result<Vec<_>>>()?
        } else {
            execution::try_map(&prepared, |_, g| tokenize(g))?
        };
        let inputs: Vec<_> = raw.iter().map(Vec::as_slice).collect();
        let widths: Vec<_> = prepared
            .iter()
            .map(|g| g.planes.iter().map(|c| c.width).max().unwrap_or(1))
            .collect();
        let outcome = {
            let _scope = crate::profile::scope("lz77.parse_shared");
            crate::tools::lz77::parse_source_streams_with_state(&inputs, &widths, source_lz77)?
        };
        outcome
            .commands
            .into_iter()
            .zip(widths)
            .map(|(commands, width)| {
                let (tokens, _) = encode::commands_to_tokens(commands, tree.leaves(), width)?;
                Ok((tokens, outcome.enabled))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let encode = |g: &Prepared| {
            encode::group_stream_with_channels(
                &g.planes,
                &tree,
                options,
                &g.wp,
                g.stream_id,
                parallel_channels,
            )
        };
        if parallel_channels {
            prepared
                .iter()
                .map(|g| encode(g))
                .collect::<Result<Vec<_>>>()?
        } else {
            execution::try_map(&prepared, |_, g| encode(g))?
        }
    };
    let lz77 = streams.iter().any(|s| s.1);
    let streams: Vec<_> = streams.into_iter().map(|s| s.0).collect();
    let model = Model::build_with_options(
        &streams,
        tree.leaves() + usize::from(lz77),
        lz77,
        options.entropy,
        &options.entropy_options,
    )?;
    let global = complete_or_return!(budget::section(budget, || {
        let mut global = BitWriter::new(limit);
        crate::patch_frame::write_dictionary(&mut global, patches)?;
        global.write(1, 1)?;
        global.write(1, 1)?;
        encode::write_tree_and_model(&tree, &model, &mut global, options)?;
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
        let tree = if let Some(tree) = encode::predefined(g.planes.iter(), options, bits)? {
            tree
        } else {
            training::learn_local(
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
            )?
        };
        let (tokens, has_matches) =
            encode::group_stream(&g.planes, &tree, options, &g.wp, g.stream_id)?;
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
