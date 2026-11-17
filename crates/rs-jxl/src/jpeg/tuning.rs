//! Explicit coefficient-encoder experiments; no changes to reconstruction or
//! decoder syntax. Ordinary encoding and source-search retain their defaults.

pub use super::block_context_tuning::Policy as BlockContextPolicy;
pub use super::entropy::AcLzSearch;
pub use super::order_tuning::Policy as AcOrderPolicy;
use super::{TranscodeOptions, archival, cfl::CflFitPolicy, transcode as coefficients};
use crate::Result;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub cfl_fit: CflFitPolicy,
    /// Native symmetric MA and half description cost on learned auxiliary trees.
    pub symmetric_ma: bool,
    /// Merge adjacent luma DC buckets only for AC statistical conditioning.
    pub paired_luma: bool,
    pub ac_lz: AcLzSearch,
    /// Requested AC histogram sets (1..=4); empty/identical groups can collapse.
    pub ac_histograms: usize,
    pub block_context: BlockContextPolicy,
    pub ac_order: AcOrderPolicy,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            cfl_fit: Default::default(),
            symmetric_ma: false,
            paired_luma: false,
            ac_lz: Default::default(),
            ac_histograms: 1,
            block_context: Default::default(),
            ac_order: Default::default(),
        }
    }
}

/// Single-pass experiment, not a baseline/candidate tournament. Neither a better
/// local score nor fewer contexts guarantees a smaller complete JXL container.
pub fn transcode(input: &[u8], options: &TranscodeOptions, tuning: Options) -> Result<Vec<u8>> {
    transcode_inner(input, options, tuning, None, None)
}

/// Return disjoint actual wire costs without an additional encoding pass.
pub fn transcode_measured(
    input: &[u8],
    options: &TranscodeOptions,
    tuning: Options,
) -> Result<(Vec<u8>, super::size_breakdown::SizeBreakdown)> {
    let receipt = std::sync::Arc::new(super::size_breakdown::Receipt::default());
    let output = transcode_inner(input, options, tuning, Some(receipt.clone()), None)?;
    let sizes = receipt
        .snapshot()
        .ok_or_else(|| crate::Error::Encoding("missing JPEG wire costs".into()))?;
    Ok((output, sizes))
}

/// Inspect final AC tokens/tables without changing any encoding decision.
/// Empirical entropies omit signaling and are not global JXL size lower bounds.
/// This diagnostic adds work; do not treat its runtime as normal encode speed.
pub fn transcode_audited(
    input: &[u8],
    options: &TranscodeOptions,
    tuning: Options,
) -> Result<(
    Vec<u8>,
    super::size_breakdown::SizeBreakdown,
    super::ac_audit::AcAudit,
)> {
    let sizes = std::sync::Arc::new(super::size_breakdown::Receipt::default());
    let audit = std::sync::Arc::new(super::ac_audit::Receipt::default());
    let output = transcode_inner(
        input,
        options,
        tuning,
        Some(sizes.clone()),
        Some(audit.clone()),
    )?;
    let sizes = sizes
        .snapshot()
        .ok_or_else(|| crate::Error::Encoding("missing JPEG wire costs".into()))?;
    let audit = audit
        .snapshot()
        .ok_or_else(|| crate::Error::Encoding("missing JPEG AC audit".into()))?;
    Ok((output, sizes, audit))
}

fn transcode_inner(
    input: &[u8],
    options: &TranscodeOptions,
    tuning: Options,
    receipt: Option<std::sync::Arc<super::size_breakdown::Receipt>>,
    audit: Option<std::sync::Arc<super::ac_audit::Receipt>>,
) -> Result<Vec<u8>> {
    let mut settings = options.resolve()?;
    settings.cfl_fit = tuning.cfl_fit;
    settings.symmetric_ma = tuning.symmetric_ma;
    settings.paired_luma = tuning.paired_luma;
    settings.ac_lz = tuning.ac_lz;
    settings.ac_histograms = tuning.ac_histograms;
    settings.block_context = tuning.block_context;
    settings.ac_order = tuning.ac_order;
    settings.size_receipt = receipt;
    settings.ac_audit = audit;
    let jpeg = archival::read_input(input, &options.limits.input)?;
    coefficients::encode_coefficients(&jpeg, &settings)
}

#[cfg(test)]
#[path = "tuning_tests.rs"]
mod tests;
