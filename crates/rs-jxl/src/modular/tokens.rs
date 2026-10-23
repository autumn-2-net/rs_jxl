// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Channel token dispatch from enc_encoding.cc::ModularCompress and
//! enc_modular.cc::ComputeTokens. Previous-channel inputs are read-only.

use super::{
    encode,
    group::Channel,
    predictor::Predictor,
    tree::{Tree, pack_signed},
    weighted,
};
use crate::{Error, Result, execution, tools::lz77::Literal};

fn population(planes: &[Channel]) -> Result<usize> {
    planes.iter().try_fold(0usize, |sum, c| {
        sum.checked_add(c.samples.len())
            .ok_or(Error::Limit("Modular token population"))
    })
}

pub(super) fn shared_channel_parallel(
    prepared: &[encode::Prepared],
    enabled: bool,
) -> Result<bool> {
    if !enabled
        || execution::is_serial()
        || rayon::current_thread_index().is_none()
        || execution::worker_count() < 2
    {
        return Ok(false);
    }
    let mut total = 0usize;
    let mut largest = 0;
    for group in prepared {
        let count = population(&group.planes)?;
        total = total
            .checked_add(count)
            .ok_or(Error::Limit("Modular token population"))?;
        largest = largest.max(count);
    }
    Ok(largest >= 65536 && largest >= total - total / 4)
}

#[derive(Default)]
struct Scratch {
    rows: [Vec<i64>; 14],
}

#[allow(clippy::too_many_arguments)]
fn channel(
    planes: &[Channel],
    c: usize,
    tree: &Tree,
    wp: &weighted::Header,
    previous_channels: usize,
    stream_id: usize,
    stateless: &[Predictor],
    scratch: &mut Scratch,
    mut emit: impl FnMut(Literal),
) -> Result<()> {
    let channel = &planes[c];
    encode::visit_channel(
        planes,
        c,
        wp,
        tree.uses_weighted(),
        previous_channels,
        stream_id,
        |_, pos, _n, wp_pred, props, value| {
            let x = pos % channel.width;
            if x == 0 {
                for &predictor in stateless {
                    let row = &mut scratch.rows[predictor as usize];
                    row.resize(channel.width, 0);
                    super::predict_simd::residual_row(
                        &channel.samples,
                        channel.width,
                        channel.width,
                        pos / channel.width,
                        predictor,
                        row,
                    )
                    .map_err(|e| Error::Encoding(e.to_string()))?;
                }
            }
            let (context, predictor, offset, multiplier) = tree.lookup(props)?;
            let residual = if predictor == Predictor::Weighted {
                i64::from(value) - i64::from(wp_pred)
            } else {
                scratch.rows[predictor as usize][x]
            } - i64::from(offset);
            if residual % i64::from(multiplier) != 0 {
                return Err(Error::Encoding(
                    "MA multiplier does not divide a residual".into(),
                ));
            }
            // Narrow after division exactly as the source PackSigned call.
            // Decoder arithmetic reconstructs signed floating words modulo 2^32.
            emit(Literal {
                context: context as u32,
                value: pack_signed((residual / i64::from(multiplier)) as i32),
            });
            Ok(())
        },
    )
}

pub(super) fn tokenize(
    planes: &[Channel],
    tree: &Tree,
    wp: &weighted::Header,
    previous_channels: usize,
    stream_id: usize,
    parallel: bool,
) -> Result<Vec<Literal>> {
    let _scope = crate::profile::scope("modular.tokenize");
    let count = population(planes)?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("Modular token allocation"))?;
    let stateless: Vec<_> = tree
        .predictors()
        .into_iter()
        .filter(|p| !p.has_weighted())
        .collect();
    let parallel = parallel
        && count >= 65536
        && !execution::is_serial()
        && rayon::current_thread_index().is_some()
        && execution::worker_count() > 1
        && planes
            .iter()
            .filter(|c| !c.samples.is_empty())
            .take(2)
            .count()
            > 1;
    if !parallel {
        let mut scratch = Scratch::default();
        for c in 0..planes.len() {
            channel(
                planes,
                c,
                tree,
                wp,
                previous_channels,
                stream_id,
                &stateless,
                &mut scratch,
                |t| tokens.push(t),
            )?;
        }
    } else {
        tokens.resize(
            count,
            Literal {
                context: 0,
                value: 0,
            },
        );
        let mut tail = tokens.as_mut_slice();
        let mut jobs = Vec::new();
        for (c, plane) in planes.iter().enumerate() {
            let (destination, rest) = tail.split_at_mut(plane.samples.len());
            tail = rest;
            if !destination.is_empty() {
                jobs.push((c, destination));
            }
        }
        crate::profile::note(
            "modular.tokenize",
            format_args!("parallel_channels={}", jobs.len()),
        );
        execution::map_mut(&mut jobs, |_, (c, destination)| {
            let mut written = 0;
            channel(
                planes,
                *c,
                tree,
                wp,
                previous_channels,
                stream_id,
                &stateless,
                &mut Scratch::default(),
                |t| {
                    destination[written] = t;
                    written += 1;
                },
            )?;
            if written != destination.len() {
                return Err(Error::Encoding(
                    "Modular channel token count mismatch".into(),
                ));
            }
            Ok(())
        })
        .into_iter()
        .collect::<Result<Vec<()>>>()?;
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::super::{predictor::predict_one, tree::Node};
    use super::*;

    fn planes() -> Vec<Channel> {
        [
            (257, 86, 0),
            (257, 86, 0),
            (0, 0, 0),
            (11, 3, 1),
            (257, 86, 1),
        ]
        .into_iter()
        .enumerate()
        .map(|(c, (width, height, shift))| Channel {
            width,
            height,
            hshift: shift,
            vshift: shift,
            samples: (0..width * height)
                .map(|i| ((i * (c + 11) + i / 13) % 8192) as i32 - 4096)
                .collect(),
        })
        .collect()
    }

    fn scalar(planes: &[Channel], tree: &Tree, wp: &weighted::Header) -> Vec<Literal> {
        let mut result = Vec::new();
        encode::visit(
            planes,
            wp,
            tree.uses_weighted(),
            3,
            19,
            |_, _, neighbors, weighted, props, value| {
                let (context, predictor, offset, multiplier) = tree.lookup(props)?;
                let prediction =
                    predict_one(predictor, neighbors, Some(i64::from(weighted))).unwrap();
                let residual = i64::from(value) - prediction - i64::from(offset);
                assert_eq!(residual % i64::from(multiplier), 0);
                result.push(Literal {
                    context: context as u32,
                    value: pack_signed((residual / i64::from(multiplier)) as i32),
                });
                Ok(())
            },
        )
        .unwrap();
        result
    }

    #[test]
    fn parallel_tokens_preserve_channels_previous_properties_and_scalar_residuals() {
        let planes = planes();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let wp = weighted::Header::preset(4).unwrap();
        let mut trees: Vec<_> = (0..14)
            .map(|p| Tree::channel_predictors(&[Predictor::try_from(p).unwrap()]).unwrap())
            .collect();
        for property in [0, 1, 15, 17, 20] {
            trees.push(
                Tree::new(vec![
                    Node::Split {
                        property,
                        threshold: 0,
                        greater: 1,
                        less_equal: 2,
                    },
                    Node::Leaf {
                        predictor: Predictor::Weighted,
                        offset: 7,
                        multiplier: 1,
                    },
                    Node::Leaf {
                        predictor: Predictor::Gradient,
                        offset: -3,
                        multiplier: 1,
                    },
                ])
                .unwrap(),
            );
        }
        for tree in trees {
            let expected = scalar(&planes, &tree, &wp);
            let actual = pool
                .install(|| tokenize(&planes, &tree, &wp, 3, 19, true))
                .unwrap();
            assert_eq!(actual, expected);
            let serial = execution::serial(|| tokenize(&planes, &tree, &wp, 3, 19, true)).unwrap();
            assert_eq!(serial, expected);
        }
    }

    #[test]
    fn parallel_tokens_return_channel_errors_without_partial_output() {
        let planes = planes();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let tree = Tree::new(vec![Node::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 2,
        }])
        .unwrap();
        let error = pool
            .install(|| tokenize(&planes, &tree, &weighted::Header::default(), 0, 0, true))
            .unwrap_err();
        assert!(matches!(error, Error::Encoding(_)));
    }

    #[test]
    fn shared_dispatch_respects_source_dominance_and_null_runner() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let group = |count| encode::Prepared {
            planes: vec![Channel {
                width: count,
                height: 1,
                hshift: 0,
                vshift: 0,
                samples: vec![0; count],
            }],
            transforms: Vec::new(),
            wp: weighted::Header::default(),
            stream_id: 0,
        };
        for (counts, expected) in [
            ([65536, 21845], true),
            ([65536, 21846], false),
            ([65535, 0], false),
        ] {
            let groups = counts.map(group);
            assert!(!shared_channel_parallel(&groups, true).unwrap());
            assert_eq!(
                pool.install(|| shared_channel_parallel(&groups, true))
                    .unwrap(),
                expected
            );
            assert!(
                !pool
                    .install(|| shared_channel_parallel(&groups, false))
                    .unwrap()
            );
            assert!(
                !pool
                    .install(|| execution::serial(|| shared_channel_parallel(&groups, true)))
                    .unwrap()
            );
        }
    }
}
