//! Experimental AC HybridUint head/tail divisions. Standard JXL only: no
//! preceding-symbol contexts, escape tables or changes to JPEG reconstruction.
//! The ordinary transcode API remains unchanged and never enables this trial.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{TranscodeOptions, archival, transcode as coefficients};
use crate::{Error, Result};

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub attempted: bool,
    pub candidate: bool,
    pub selected: bool,
    pub histograms: usize,
    pub changed_histograms: usize,
    pub configurations: usize,
    pub baseline_bytes: usize,
    pub candidate_bytes: Option<usize>,
    pub search_seconds: f64,
    pub write_seconds: f64,
}

/// Per-invocation diagnostics; use a fresh receipt for each encode.
#[derive(Debug, Default)]
pub struct Receipt(Mutex<Stats>);

impl Receipt {
    pub fn snapshot(&self) -> Stats {
        self.0.lock().expect("head/tail receipt poisoned").clone()
    }

    pub(super) fn proposal(
        &self,
        histograms: usize,
        changed: usize,
        configurations: usize,
        candidate: bool,
        seconds: f64,
    ) {
        let mut stats = self.0.lock().expect("head/tail receipt poisoned");
        *stats = Stats {
            attempted: true,
            candidate,
            histograms,
            changed_histograms: changed,
            configurations,
            search_seconds: seconds,
            ..Default::default()
        };
    }

    pub(super) fn select(
        &self,
        baseline: Vec<u8>,
        alternative: impl FnOnce() -> Result<Option<Vec<u8>>>,
    ) -> Result<Vec<u8>> {
        let start = Instant::now();
        let alternative = match alternative() {
            Ok(value) => value,
            // Optional output may exceed a caller's budget even though the
            // baseline fits. Allocation/internal errors are not swallowed.
            Err(Error::Limit("encoded output size" | "JPEG container output")) => None,
            Err(error) => return Err(error),
        };
        let mut stats = self.0.lock().expect("head/tail receipt poisoned");
        stats.write_seconds = start.elapsed().as_secs_f64();
        stats.baseline_bytes = baseline.len();
        stats.candidate_bytes = alternative.as_ref().map(Vec::len);
        stats.selected = alternative
            .as_ref()
            .is_some_and(|trial| trial.len() < baseline.len());
        Ok(if stats.selected {
            alternative.expect("selected candidate")
        } else {
            baseline
        })
    }
}

/// Explicit experiment, outside source-search and the production Python ABI.
/// Both models share parsed coefficients, MA auxiliaries, tokens and LZ parse.
pub fn transcode(
    input: &[u8],
    options: &TranscodeOptions,
    receipt: &Arc<Receipt>,
) -> Result<Vec<u8>> {
    let mut settings = options.resolve()?;
    settings.ac_head_tail = Some(receipt.clone());
    let jpeg = archival::read_input(input, &options.limits.input)?;
    coefficients::encode_coefficients(&jpeg, &settings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_includes_complete_container_and_keeps_ties() {
        for size in [7, 8, 9] {
            let receipt = Receipt::default();
            let out = receipt
                .select(vec![1; 8], || Ok(Some(vec![2; size])))
                .unwrap();
            assert_eq!(out, if size < 8 { vec![2; size] } else { vec![1; 8] });
            assert_eq!(receipt.snapshot().selected, size < 8);
            assert_eq!(receipt.snapshot().candidate_bytes, Some(size));
        }
        let receipt = Receipt::default();
        assert_eq!(
            receipt
                .select(vec![1], || Err(Error::Limit("JPEG container output")))
                .unwrap(),
            vec![1]
        );
        assert!(
            receipt
                .select(vec![1], || Err(Error::Limit("JPEG container allocation")))
                .is_err()
        );
    }
}
