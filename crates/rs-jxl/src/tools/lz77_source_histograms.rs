//! Bounded integer counting for source LZ cost models. Final maps retain the
//! original sorted iteration order used by the floating-point cost estimator.

use std::collections::BTreeMap;

use super::{Counts, Literal};

const SYMBOLS: usize = 256;
const CONTEXTS: usize = 65536;
const ROWS: usize = 4096;
const MIN_DENSE_TOKENS: usize = 4096;
const PROMOTE_AFTER: u16 = 32;
const MISSING: u16 = u16::MAX;
const READY: u16 = MISSING - PROMOTE_AFTER;

struct DenseRow<const WIDTH: usize> {
    context: u32,
    // Separate allocations keep growth bounded without copying large rows or
    // reserving/zeroing the full 8 MiB budget for a model with few contexts.
    counts: Box<[usize; WIDTH]>,
}

// Raw residuals usually occupy the low hybrid tokens. Fitted models also need
// the length alphabet at 224+, so retain their wider rows. Both cap bin storage
// at 1M usize counters (8 MiB on 64-bit targets).
pub(super) type RawCounts = BoundedCounts<64, 16384>;
pub(super) type ContextCounts = BoundedCounts<SYMBOLS, ROWS>;

pub(super) struct BoundedCounts<const WIDTH: usize, const LIMIT: usize> {
    contexts: Vec<u16>,
    rows: Vec<DenseRow<WIDTH>>,
    sparse: Counts,
    promote_after: u16,
}

impl RawCounts {
    pub(super) fn for_inputs(inputs: &[&[Literal]], tokens: usize) -> Self {
        if tokens < MIN_DENSE_TOKENS {
            return Self::new(tokens);
        }
        let _profile = crate::profile::scope("lz77.source.raw_histogram.context_range");
        // A bounded ID domain proves that first-use rows cannot exhaust the
        // budget before a later hot context. Unknown/wide domains keep probation.
        if inputs
            .iter()
            .any(|input| input.iter().any(|literal| literal.context >= 16384))
        {
            return Self::new(tokens);
        }
        Self {
            contexts: vec![READY; 16384],
            rows: Vec::new(),
            sparse: Counts::new(),
            promote_after: 0,
        }
    }
}

impl<const WIDTH: usize, const LIMIT: usize> BoundedCounts<WIDTH, LIMIT> {
    pub(super) fn new(tokens: usize) -> Self {
        Self {
            contexts: if tokens >= MIN_DENSE_TOKENS {
                vec![MISSING; CONTEXTS]
            } else {
                Vec::new()
            },
            rows: Vec::new(),
            sparse: Counts::new(),
            promote_after: PROMOTE_AFTER,
        }
    }

    #[inline]
    pub(super) fn increment(&mut self, context: u32, symbol: u32) {
        if symbol < WIDTH as u32
            && let Some(slot) = self.contexts.get_mut(context as usize)
        {
            if (*slot as usize) < LIMIT {
                self.rows[*slot as usize].counts[symbol as usize] += 1;
                return;
            }
            // High slot values count probation visits; low values index a
            // promoted row. Initial counts stay sparse until the final merge.
            if *slot > READY {
                *slot -= 1;
            } else if self.rows.len() < LIMIT {
                let mut counts = Box::new([0; WIDTH]);
                counts[symbol as usize] = 1;
                *slot = self.rows.len() as u16;
                self.rows.push(DenseRow { context, counts });
                return;
            }
        }
        *self
            .sparse
            .entry(context)
            .or_default()
            .entry(symbol)
            .or_default() += 1;
    }

    pub(super) fn into_counts(mut self) -> Counts {
        let dense_rows = self.rows.len();
        for row in self.rows {
            merge_dense(
                row.counts.as_ref(),
                self.sparse.entry(row.context).or_default(),
            );
        }
        if crate::profile::enabled() {
            Self::profile_counts(&self.sparse, &self.contexts, dense_rows, self.promote_after);
        }
        self.sparse
    }

    fn profile_counts(counts: &Counts, contexts: &[u16], dense_rows: usize, promote_after: u16) {
        let _scope = crate::profile::scope("lz77.source.histogram.coverage");
        let mut bands = [0usize; 3];
        let mut dense_tokens = 0usize;
        let mut budget_tokens = 0usize;
        let mut wide_context_tokens = 0usize;
        let mut max_symbol = 0;
        // Read the completed histogram, not the input token stream. Profiling
        // adds no counters or branches to the hot increment path.
        for (&context, row) in counts {
            let mut small = 0usize;
            for (&symbol, &count) in row {
                let band = if symbol < 64 {
                    0
                } else if symbol < 256 {
                    1
                } else {
                    2
                };
                bands[band] += count;
                max_symbol = max_symbol.max(symbol);
                if symbol < WIDTH as u32 {
                    small += count;
                }
                if context >= CONTEXTS as u32 {
                    wide_context_tokens += count;
                }
            }
            match contexts.get(context as usize) {
                Some(&slot) if (slot as usize) < LIMIT => {
                    dense_tokens += small - usize::from(promote_after);
                }
                Some(&READY) if small > usize::from(promote_after) => {
                    budget_tokens += small;
                }
                _ => {}
            }
        }
        let total: usize = bands.iter().sum();
        crate::profile::note(
            "lz77.source.histogram.coverage",
            format_args!(
                "width={WIDTH} rows={dense_rows} row_limit={LIMIT} probation={promote_after} contexts={} tokens={total} lt64={} from64to255={} ge256={} max_symbol={max_symbol} dense_tokens={dense_tokens} sparse_tokens={} budget_context_tokens={budget_tokens} wide_context_tokens={wide_context_tokens}",
                counts.len(),
                bands[0],
                bands[1],
                bands[2],
                total - dense_tokens,
            ),
        );
    }
}

pub(super) struct SymbolCounts {
    dense: [usize; SYMBOLS],
    sparse: BTreeMap<u32, usize>,
}

impl SymbolCounts {
    pub(super) fn new() -> Self {
        Self {
            dense: [0; SYMBOLS],
            sparse: BTreeMap::new(),
        }
    }

    #[inline]
    pub(super) fn increment(&mut self, symbol: u32) {
        if let Some(count) = self.dense.get_mut(symbol as usize) {
            *count += 1;
        } else {
            *self.sparse.entry(symbol).or_default() += 1;
        }
    }

    pub(super) fn into_counts(mut self) -> BTreeMap<u32, usize> {
        merge_dense(&self.dense, &mut self.sparse);
        self.sparse
    }
}

fn merge_dense<const WIDTH: usize>(dense: &[usize; WIDTH], sparse: &mut BTreeMap<u32, usize>) {
    for (symbol, &count) in dense.iter().enumerate() {
        if count != 0 {
            *sparse.entry(symbol as u32).or_default() += count;
        }
    }
}

#[cfg(test)]
#[path = "lz77_source_histograms_tests.rs"]
mod tests;
