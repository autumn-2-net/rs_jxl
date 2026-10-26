// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Heterogeneous auxiliary streams from enc_modular.cc::{ComputeTree,
//! ComputeTokens, EncodeGlobalInfo, EncodeStream} at b866b3d. Tree domains
//! retain separate learning policies; all residuals share one entropy model.
//! Unlike a pixel frame section, an auxiliary stream never adds byte padding.

use super::{
    encode::{self, ModularOptions},
    group::{self, Channel, Transform},
    tokens, training,
    tree::Tree,
    tree_groups, weighted,
};
use crate::{
    Error, Result,
    bit_writer::BitWriter,
    entropy::stream::{Model, Token},
    execution,
    scheduler::{
        WorkerLimits,
        worker_pool::{TaskRunner, WorkerPlan, run_plan},
    },
    tools::lz77,
};

const LEARNER_SCRATCH_BYTES: u64 = 32 << 20;

/// JPEG's normal model route only: enc_modular.cc's cooperative_chunks gate.
/// None preserves domain-parallel learning with serial splits. A null runner
/// remains strictly serial even when cooperative limits were requested.
#[derive(Clone, Copy)]
pub(crate) struct LearnerExecution<'a> {
    pub pool: Option<&'a rayon::ThreadPool>,
    pub cooperative: Option<WorkerLimits>,
}

impl LearnerExecution<'_> {
    fn validate(self) -> Result<()> {
        if let Some(limits) = self.cooperative {
            limits.validate()?;
        }
        Ok(())
    }

    fn run<T: Sync, R: Send + Sync>(
        self,
        items: &[T],
        work: impl Fn(usize, &T, bool) -> Result<R> + Send + Sync,
    ) -> Result<Vec<R>> {
        self.validate()?;
        let Some(pool) = self.pool.filter(|_| !execution::is_serial()) else {
            return execution::serial(|| execution::try_map(items, |i, item| work(i, item, false)));
        };
        if let Some(limits) = self.cooperative.filter(|_| items.len() > 1) {
            let plan = WorkerPlan::cooperative(items.len(), limits, LEARNER_SCRATCH_BYTES)?;
            run_plan(Some(pool), &plan, |i, runner| {
                work(i, &items[i], matches!(runner, TaskRunner::Pool(_)))
            })
        } else {
            execution::with_pool(pool, || {
                execution::try_map(items, |i, item| work(i, item, false))
            })
        }
    }
}

pub(crate) struct Stream<'a> {
    pub id: usize,
    pub channels: &'a [Channel],
    pub transforms: &'a [Transform],
}

pub(crate) struct Domain<'a> {
    /// Inclusive source domain start, including any empty leading stream IDs.
    pub start: usize,
    pub streams: &'a [Stream<'a>],
    pub options: &'a ModularOptions,
}

struct Prepared<'a> {
    stream: &'a Stream<'a>,
    wp: weighted::Header,
    previous_channels: usize,
}

struct Encoded {
    id: usize,
    transforms: Vec<Transform>,
    wp: weighted::Header,
    tokens: Vec<Token>,
}

pub(crate) struct Shared {
    tree: Tree,
    model: Model,
    streams: Vec<Encoded>,
    options: ModularOptions,
}

impl Shared {
    /// Missing IDs represent empty streams, not new model boundaries. Budgets
    /// are checked before predictor traversal and shared across all learners.
    pub(crate) fn build(
        domains: &[Domain<'_>],
        num_streams: usize,
        options: &ModularOptions,
        bits: u8,
        max_tokens: usize,
    ) -> Result<Self> {
        Self::build_inner(domains, num_streams, options, bits, max_tokens, None)
    }

    pub(crate) fn build_with_execution(
        domains: &[Domain<'_>],
        num_streams: usize,
        options: &ModularOptions,
        bits: u8,
        max_tokens: usize,
        runner: LearnerExecution<'_>,
    ) -> Result<Self> {
        runner.validate()?;
        let work = || {
            if runner.cooperative.is_none() {
                Self::build(domains, num_streams, options, bits, max_tokens)
            } else {
                Self::build_inner(
                    domains,
                    num_streams,
                    options,
                    bits,
                    max_tokens,
                    Some(runner),
                )
            }
        };
        match runner.pool {
            Some(pool) => execution::with_pool(pool, work),
            None => execution::serial(work),
        }
    }

    fn build_inner(
        domains: &[Domain<'_>],
        num_streams: usize,
        options: &ModularOptions,
        bits: u8,
        max_tokens: usize,
        runner: Option<LearnerExecution<'_>>,
    ) -> Result<Self> {
        options.validate()?;
        if domains.is_empty()
            || !(1..=32).contains(&bits)
            || max_tokens == 0
            || num_streams > i32::MAX as usize + 1
            || domains.last().is_some_and(|d| d.start >= num_streams)
            || domains.windows(2).any(|d| d[0].start >= d[1].start)
        {
            return Err(Error::InvalidInput("invalid auxiliary tree domains".into()));
        }
        let mut prepared = Vec::new();
        let mut useful = Vec::new();
        let mut splits = Vec::new();
        let mut population = 0usize;
        for (index, domain) in domains.iter().enumerate() {
            domain.options.validate()?;
            let end = domains.get(index + 1).map_or(num_streams, |d| d.start);
            let begin = prepared.len();
            let mut previous_id = None;
            for stream in domain.streams {
                if stream.id < domain.start
                    || stream.id >= end
                    || previous_id.is_some_and(|id| id >= stream.id)
                {
                    return Err(Error::InvalidInput("invalid auxiliary stream ID".into()));
                }
                previous_id = Some(stream.id);
                let meta = stream.channels.iter().take_while(|c| c.hshift < 0).count();
                group::validate_layout(stream.channels, meta)?;
                if stream.channels.is_empty() {
                    continue;
                }
                for channel in stream.channels {
                    population = population
                        .checked_add(channel.samples.len())
                        .filter(|&n| n <= max_tokens)
                        .ok_or(Error::Limit("auxiliary token budget"))?;
                }
                prepared.push(Prepared {
                    stream,
                    wp: encode::weighted_header(stream.channels, domain.options)?,
                    previous_channels: domain.options.previous_channels,
                });
            }
            if prepared.len() != begin {
                splits.push(domain.start);
                useful.push((begin..prepared.len(), domain.options));
            }
        }
        if useful.is_empty() {
            return Err(Error::InvalidInput("empty auxiliary tree domains".into()));
        }
        splits.push(num_streams);
        let budget = options.sample_limit.map(training::SampleBudget::new);
        let learn = |_: usize,
                     (range, policy): &(std::ops::Range<usize>, &ModularOptions),
                     parallel| {
            let streams = &prepared[range.clone()];
            if let Some(tree) =
                encode::predefined(streams.iter().flat_map(|s| s.stream.channels), policy, bits)?
            {
                return Ok(tree);
            }
            let views: Vec<_> = streams
                .iter()
                .map(|s| training::TrainingStream {
                    channels: s.stream.channels,
                    wp: s.wp,
                    stream_id: s.stream.id,
                })
                .collect();
            // The cooperative JPEG route lends only the split runner. Gathering
            // stays serial and selected pixels share the same frame ceiling.
            training::learn_with_budget(&views, policy, budget.as_ref(), parallel)
        };
        let trees = if let Some(runner) = runner {
            runner.run(&useful, learn)?
        } else {
            execution::try_map(&useful, |i, domain| learn(i, domain, false))?
        };
        let tree = tree_groups::merge(trees, &splits, tree_groups::MAX_MERGED_LEAVES)?;
        let widths: Vec<_> = prepared
            .iter()
            .map(|s| s.stream.channels.iter().map(|c| c.width).max().unwrap_or(1))
            .collect();
        let (streams, enabled) = if options.lz77 {
            let config = options.source_lz77.as_ref().ok_or_else(|| {
                Error::InvalidInput("shared auxiliary LZ77 requires source policy".into())
            })?;
            let raw = execution::try_map(&prepared, |_, s| {
                tokens::tokenize(
                    s.stream.channels,
                    &tree,
                    &s.wp,
                    s.previous_channels,
                    s.stream.id,
                    false,
                )
            })?;
            let inputs: Vec<_> = raw.iter().map(Vec::as_slice).collect();
            let parsed = lz77::parse_source_streams_with_state(&inputs, &widths, config)?;
            let streams = parsed
                .commands
                .into_iter()
                .zip(&widths)
                .map(|(commands, &width)| {
                    encode::commands_to_tokens(commands, tree.leaves(), width)
                        .map(|result| result.0)
                })
                .collect::<Result<Vec<_>>>()?;
            (streams, parsed.enabled)
        } else {
            (
                execution::try_map(&prepared, |_, s| {
                    tokens::tokenize(
                        s.stream.channels,
                        &tree,
                        &s.wp,
                        s.previous_channels,
                        s.stream.id,
                        false,
                    )
                    .map(|raw| {
                        raw.into_iter()
                            .map(|t| Token::literal(t.context as usize, t.value))
                            .collect()
                    })
                })?,
                false,
            )
        };
        let count = streams
            .iter()
            .try_fold(0usize, |n, s| n.checked_add(s.len()))
            .ok_or(Error::Limit("auxiliary encoded token budget"))?;
        if count > max_tokens {
            return Err(Error::Limit("auxiliary encoded token budget"));
        }
        let model = Model::build_with_options(
            &streams,
            tree.leaves() + usize::from(enabled),
            enabled,
            options.entropy,
            &options.entropy_options,
        )?;
        let streams = prepared
            .into_iter()
            .zip(streams)
            .map(|(s, tokens)| Encoded {
                id: s.stream.id,
                transforms: s.stream.transforms.to_vec(),
                wp: s.wp,
                tokens,
            })
            .collect();
        Ok(Self {
            tree,
            model,
            streams,
            options: options.clone(),
        })
    }

    pub(crate) fn write_global_info(&self, writer: &mut BitWriter) -> Result<()> {
        writer.write(1, 1)?;
        encode::write_tree(&self.tree, writer, &self.options)?;
        self.model.write_header(writer)
    }

    /// An absent stream has no header, exactly as EncodeStream. This also
    /// handles the unused Global/ModularDC/ModularAC slots in JPEG frames.
    pub(crate) fn write_stream(&self, id: usize, writer: &mut BitWriter) -> Result<()> {
        let Ok(index) = self.streams.binary_search_by_key(&id, |s| s.id) else {
            return Ok(());
        };
        let stream = &self.streams[index];
        group::write_header(writer, &stream.transforms, &stream.wp, true)?;
        if !stream.tokens.is_empty() {
            self.model.write_stream(&stream.tokens, writer)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "auxiliary_tests.rs"]
mod tests;
