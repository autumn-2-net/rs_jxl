//! Opt-in diagnostics for the selected AC model. Not an encoding cost oracle.
use std::sync::Mutex;

pub use crate::entropy::stream::audit::Audit;

#[derive(Clone, Debug)]
pub struct AcAudit {
    pub entropy: Audit,
    /// Exact histogram-set selector bits, outside the entropy token streams.
    pub selector_bits: usize,
    /// Diagnostic work is included in a caller's end-to-end encoding time.
    pub seconds: f64,
}

#[derive(Debug, Default)]
pub(crate) struct Receipt(Mutex<Option<AcAudit>>);

impl Receipt {
    pub(crate) fn record(&self, audit: AcAudit) {
        *self.0.lock().expect("AC audit receipt poisoned") = Some(audit);
    }

    pub(crate) fn snapshot(&self) -> Option<AcAudit> {
        self.0.lock().expect("AC audit receipt poisoned").clone()
    }
}
