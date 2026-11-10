//! Opt-in expanded HybridUint search. Direct small values form the head;
//! larger values use standard exponent/MSB/LSB tokens plus extra bits.
//! This is not a new conditional coder or an arbitrary frequent-symbol palette.

use super::*;

pub(super) struct ExactValues(Vec<(u32, u32)>);

impl ExactValues {
    pub(super) fn new(values: &[u32]) -> Result<Self> {
        let mut sorted = Vec::new();
        sorted
            .try_reserve_exact(values.len())
            .map_err(|_| Error::Limit("exact uint values"))?;
        sorted.extend_from_slice(values);
        sorted.sort_unstable();
        let mut counts: Vec<(u32, u32)> = Vec::new();
        counts
            .try_reserve_exact(sorted.len())
            .map_err(|_| Error::Limit("exact uint counts"))?;
        for value in sorted {
            if let Some((last, count)) = counts.last_mut().filter(|(last, _)| *last == value) {
                debug_assert_eq!(*last, value);
                *count = count
                    .checked_add(1)
                    .ok_or(Error::Limit("exact uint count overflow"))?;
            } else {
                counts.push((value, 1));
            }
        }
        Ok(Self(counts))
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn add_to(&self, cfg: HybridUintConfig, counts: &mut [u64]) -> u64 {
        let mut extra = 0u64;
        let mut values = [0; super::super::simd::TOKEN_BATCH];
        let mut tokens = [0; super::super::simd::TOKEN_BATCH];
        let mut widths = [0; super::super::simd::TOKEN_BATCH];
        for batch in self.0.chunks(values.len()) {
            for (slot, &(value, _)) in values.iter_mut().zip(batch) {
                *slot = value;
            }
            super::super::simd::tokens(&values[..batch.len()], cfg, &mut tokens, &mut widths);
            for (i, &(_, n)) in batch.iter().enumerate() {
                counts[tokens[i] as usize] += u64::from(n);
                extra += u64::from(widths[i]) * u64::from(n);
            }
        }
        extra
    }
}

fn candidates() -> Vec<HybridUintConfig> {
    // Preserve the source order on equal estimates, then add legal head/tail
    // divisions missing from it. split==8 is direct-only for an 8-bit alphabet.
    let mut candidates = BEST_CONFIGS.to_vec();
    for split in 0..8 {
        for msb in 0..=split {
            for lsb in 0..=split - msb {
                let cfg = config(split, msb, lsb);
                if !candidates.contains(&cfg) {
                    candidates.push(cfg);
                }
            }
        }
    }
    candidates
}

pub(crate) fn search(
    streams: &[Vec<Token>],
    map: &[u8],
    histograms: usize,
    lz77: bool,
    options: &EntropyOptions,
) -> Result<(SearchPlan, usize)> {
    let candidates = candidates();
    let workers = !crate::execution::is_serial()
        && rayon::current_thread_index().is_some()
        && crate::execution::worker_count() > 1;
    search_config_plan_with_candidates(
        streams,
        map,
        histograms,
        lz77,
        false,
        options,
        workers,
        Some(&candidates),
    )
    .map(|plan| (plan, candidates.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_head_tail_histograms_match_raw_for_every_candidate() {
        let values: Vec<u32> = (0..10000)
            .map(|i| match i % 5 {
                0 => 0,
                1 => 1,
                2 => 255,
                3 => (i * 7919) as u32,
                _ => u32::MAX,
            })
            .collect();
        let exact = ExactValues::new(&values).unwrap();
        for cfg in candidates() {
            let maximum = (cfg.encode(u32::MAX).token | cfg.lsb_mask()) as usize;
            let mut actual = vec![0; maximum + 1];
            let mut expected = actual.clone();
            let extra = super::super::super::simd::add_tokens(&values, cfg, &mut expected);
            assert_eq!(exact.add_to(cfg, &mut actual), extra, "{cfg:?}");
            assert_eq!(actual, expected, "{cfg:?}");
        }
    }

    #[test]
    fn pool_retains_source_configs_and_adds_all_eight_bit_divisions() {
        let all = candidates();
        assert_eq!(&all[..BEST_CONFIGS.len()], BEST_CONFIGS.as_slice());
        for split in 0..8 {
            for msb in 0..=split {
                for lsb in 0..=split - msb {
                    assert!(all.contains(&config(split, msb, lsb)));
                }
            }
        }
        assert!(all.contains(&config(8, 0, 0)));
        assert_eq!(all.len(), 125);
    }
}
