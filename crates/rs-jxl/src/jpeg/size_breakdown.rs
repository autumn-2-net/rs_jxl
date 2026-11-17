//! Exact disjoint wire costs from the real writer, not entropy estimates.
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Default)]
pub struct SizeBreakdown {
    /// Whether AC emission uses LZ; unavailable for pre-encoded search caches.
    /// This is diagnostic state, not an additional bit-cost category.
    pub ac_lz_enabled: Option<bool>,
    pub ac_histogram_groups: Option<usize>,
    pub ac_block_contexts: Option<usize>,
    pub dc_thresholds: Option<usize>,
    pub global_bits: usize,
    /// Shared MA tree and entropy tables for DC, metadata and quantization.
    pub aux_model_bits: usize,
    pub dc_bits: usize,
    pub ac_metadata_bits: usize,
    pub quant_bits: usize,
    /// AC orders, histogram selection and entropy tables.
    pub ac_model_bits: usize,
    pub ac_data_bits: usize,
    pub padding_bits: usize,
    /// Image header (including ICC, if present), frame header and TOC.
    pub header_bits: usize,
    pub reconstruction_bits: usize,
    /// Extracted Exif/XMP box payloads, not ICC or original APP archival data.
    pub metadata_bits: usize,
    pub container_bits: usize,
}

impl SizeBreakdown {
    pub fn total_bits(&self) -> usize {
        self.global_bits
            + self.aux_model_bits
            + self.dc_bits
            + self.ac_metadata_bits
            + self.quant_bits
            + self.ac_model_bits
            + self.ac_data_bits
            + self.padding_bits
            + self.header_bits
            + self.reconstruction_bits
            + self.metadata_bits
            + self.container_bits
    }
}

#[derive(Debug, Default)]
pub(crate) struct Receipt(Mutex<Option<SizeBreakdown>>);

impl Receipt {
    pub(crate) fn record(&self, sizes: SizeBreakdown) {
        *self.0.lock().expect("size receipt poisoned") = Some(sizes);
    }
    pub(crate) fn snapshot(&self) -> Option<SizeBreakdown> {
        *self.0.lock().expect("size receipt poisoned")
    }
}
