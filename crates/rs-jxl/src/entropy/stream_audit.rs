//! Opt-in, read-only diagnostics for an already selected entropy model.
//!
//! All observations use this model's final HybridUint configs, context map and
//! length/min-symbol mapping. Raw-context and clustered entropies are comparable
//! within this audit, not across models that tokenize values differently. Empty
//! contexts/tables contribute no observations, even if the encoder fitted a
//! fallback singleton to make an empty table representable.
//!
//! Symbol costs exclude headers, ANS state/renormalization rounding, alignment
//! and other framing. The extra-bit diagnostic independently fits every appended
//! bit conditioned on (selected histogram, final symbol, bit index), using the
//! actual emitted low-aligned payload. It is an empirical optimistic diagnostic
//! that ignores model transmission and dependencies between bits, NOT a legal
//! automatic JXL coding gain or a prediction of complete container size.

use std::collections::HashMap;
use std::hash::Hash;

use super::{Model, Tables, Token};
use crate::{Error, Result};

/// Aggregate observations, in bits where applicable. No token/model ownership
/// is retained. Auditing is explicit and does not affect model selection/writing.
#[derive(Clone, Copy, Debug)]
pub struct Audit {
    pub streams: usize,
    pub nonempty_streams: usize,
    pub tokens: u64,
    /// Contexts in the initial batch, matching `Model::write_stream` coordinates.
    pub raw_contexts: usize,
    /// Raw contexts with at least one actual token in this audit.
    pub active_contexts: usize,
    /// Selected stored tables, including tables unused by these observations.
    pub histograms: usize,
    pub ans: bool,
    pub lz77: bool,
    /// Exact number of appended HybridUint bits, including length payloads.
    pub extra_bits: u64,
    /// ANS: sum -log2(normalized probability); prefix: actual code lengths.
    pub model_symbol_bits: f64,
    /// Sum of empirical zero-order final-symbol entropies per selected table.
    pub cluster_entropy_bits: f64,
    /// Same final symbols, with a separate empirical distribution per raw context.
    pub raw_context_entropy_bits: f64,
    /// Sum of empirical binary entropies for the conditioned appended bits.
    pub extra_bit_entropy_bits: f64,
}

struct SymbolStats {
    count: u64,
    // One pair per actually appended bit; no allocation for direct symbols.
    bits: Vec<[u64; 2]>,
}

impl Model {
    /// Observe final encoded tokens without changing the model, tokens or wire.
    /// Includes the optional extra-bit diagnostic whenever this opt-in audit is
    /// called. Mapping is exactly that of `write_stream` (initial batch, offset
    /// zero); appended batches with batch-local IDs must not be concatenated.
    /// LZ tokens are audited as emitted, not expanded back into copied literals.
    pub fn audit(&self, streams: &[Vec<Token>]) -> Result<Audit> {
        let range = self.context_range(0)?;
        let tokens = streams.iter().try_fold(0u64, |total, stream| {
            let count = u64::try_from(stream.len())
                .map_err(|_| Error::Limit("entropy audit token count"))?;
            total
                .checked_add(count)
                .ok_or(Error::Limit("entropy audit token count"))
        })?;
        let mut raw: HashMap<(usize, usize), u64> = HashMap::new();
        let mut clustered: HashMap<(usize, usize), SymbolStats> = HashMap::new();
        let mut extra_bits = 0u64;
        for token in streams.iter().flatten() {
            // Reuse the writer's validation and final literal/length mapping.
            let mapped = self.map_token_in(*token, &range)?;
            if mapped.nbits > 32 || u64::from(mapped.bits) >> mapped.nbits != 0 {
                return Err(Error::Encoding("invalid entropy audit extra bits".into()));
            }
            extra_bits = extra_bits
                .checked_add(u64::from(mapped.nbits))
                .ok_or(Error::Limit("entropy audit extra-bit count"))?;
            add_observations(&mut raw, (token.context, mapped.symbol), 1)?;
            let key = (mapped.context, mapped.symbol);
            if !clustered.contains_key(&key) {
                let mut bits = Vec::new();
                bits.try_reserve_exact(usize::from(mapped.nbits))
                    .map_err(|_| Error::Limit("entropy audit bit counters"))?;
                bits.resize(usize::from(mapped.nbits), [0; 2]);
                clustered
                    .try_reserve(1)
                    .map_err(|_| Error::Limit("entropy audit symbol counters"))?;
                clustered.insert(key, SymbolStats { count: 0, bits });
            }
            let stats = clustered.get_mut(&key).expect("inserted audit symbol");
            if stats.bits.len() != usize::from(mapped.nbits) {
                return Err(Error::Encoding(
                    "inconsistent extra-bit width for final entropy symbol".into(),
                ));
            }
            increment(&mut stats.count, 1)?;
            for (index, counts) in stats.bits.iter_mut().enumerate() {
                increment(&mut counts[((mapped.bits >> index) & 1) as usize], 1)?;
            }
        }
        // Stable key order makes floating-point accumulation deterministic even
        // though insertion uses randomized sparse hash tables. No token clones.
        let raw = sorted_entries(raw)?;
        let clustered = sorted_entries(clustered)?;
        let (active_contexts, raw_context_entropy_bits) = empirical_entropy(&raw, |&n| n)?;
        let (_, cluster_entropy_bits) = empirical_entropy(&clustered, |stats| stats.count)?;
        let mut model_symbol_bits = 0.0;
        let mut extra_bit_entropy_bits = 0.0;
        for &((histogram, symbol), ref stats) in &clustered {
            // Price once per observed table/symbol, not once per token. These
            // accessors also reject symbols absent from the actual coding table.
            let cost = match &self.tables {
                Tables::Ans(tables) => tables
                    .get(histogram)
                    .ok_or_else(|| Error::Encoding("audit histogram outside ANS model".into()))?
                    .symbol_cost(symbol)?,
                Tables::Prefix(tables) => f64::from(
                    tables
                        .get(histogram)
                        .ok_or_else(|| {
                            Error::Encoding("audit histogram outside prefix model".into())
                        })?
                        .symbol_bits(symbol)
                        .map_err(|error| Error::Encoding(error.to_string()))?,
                ),
            };
            model_symbol_bits += stats.count as f64 * cost;
            for &[zero, one] in &stats.bits {
                let total = zero
                    .checked_add(one)
                    .ok_or(Error::Limit("entropy audit bit population"))?;
                extra_bit_entropy_bits += entropy_term(zero, total) + entropy_term(one, total);
            }
        }
        Ok(Audit {
            streams: streams.len(),
            nonempty_streams: streams.iter().filter(|stream| !stream.is_empty()).count(),
            tokens,
            raw_contexts: range.len(),
            active_contexts,
            histograms: self.histogram_count(),
            ans: self.uses_ans(),
            lz77: self.lz77,
            extra_bits,
            model_symbol_bits,
            cluster_entropy_bits,
            raw_context_entropy_bits,
            extra_bit_entropy_bits,
        })
    }
}

fn increment(count: &mut u64, amount: u64) -> Result<()> {
    *count = count
        .checked_add(amount)
        .ok_or(Error::Limit("entropy audit observation count"))?;
    Ok(())
}

fn add_observations<K: Eq + Hash>(counts: &mut HashMap<K, u64>, key: K, amount: u64) -> Result<()> {
    if let Some(count) = counts.get_mut(&key) {
        increment(count, amount)?;
    } else {
        counts
            .try_reserve(1)
            .map_err(|_| Error::Limit("entropy audit raw counters"))?;
        counts.insert(key, amount);
    }
    Ok(())
}

fn sorted_entries<K: Ord, V>(map: HashMap<K, V>) -> Result<Vec<(K, V)>> {
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(map.len())
        .map_err(|_| Error::Limit("entropy audit sorted counters"))?;
    entries.extend(map);
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

fn entropy_term(count: u64, total: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        // This form avoids subtraction of nearly equal large entropy terms.
        count as f64 * (total as f64 / count as f64).log2()
    }
}

fn empirical_entropy<T>(
    rows: &[((usize, usize), T)],
    count: impl Fn(&T) -> u64,
) -> Result<(usize, f64)> {
    let mut groups = 0usize;
    let mut entropy = 0.0;
    let mut start = 0;
    while start < rows.len() {
        let mut end = start;
        let mut total = 0u64;
        while end < rows.len() && rows[end].0.0 == rows[start].0.0 {
            increment(&mut total, count(&rows[end].1))?;
            end += 1;
        }
        for row in &rows[start..end] {
            entropy += entropy_term(count(&row.1), total);
        }
        groups = groups
            .checked_add(1)
            .ok_or(Error::Limit("entropy audit active contexts"))?;
        start = end;
    }
    Ok((groups, entropy))
}

#[cfg(test)]
#[path = "stream_audit_tests.rs"]
mod tests;
