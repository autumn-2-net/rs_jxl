// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Staged pixel-only scheduler port from libjxl
//! `b866b3defec13494bfbaf705db93581cda118e37`.
//!
//! Integration: register `pub mod scheduler` in the crate root. Scan with
//! [`scan_lossless_input`], generate [`lossless_probe_plan`], and use
//! [`CandidateScreen`] with a non-recursive encoder callback. The callback must
//! implement the supplied [`CandidateParams`] or reject unsupported settings.
//! Crop scores are estimates, never whole-frame output bounds or winners.
//!
//! [`adapter`] exposes the current Rust API's capability gaps explicitly.
//! [`run_pixel_search`] executes the default large-frame racing/refinement
//! branches with retained whole outputs and optional calibration witnesses.
//! [`policy`] remains a score-only planning API. See `PORT_STATUS.md` for
//! explicitly unported optional routes and native projection differences.

pub mod adapter;
pub mod batch;
pub mod calibration;
pub mod driver;
pub mod frame_sequence;
pub mod native;
mod native_fast;
mod native_tectonic;
pub mod params;
pub mod pilot;
pub mod policy;
pub mod prefix_screen;
pub mod probe;
pub mod progressive;
pub mod regions;
pub mod scan;
pub mod search;
pub mod small_probe;
pub mod tectonic;
pub mod worker_pool;

pub use adapter::{NativeResourceLimits, NativeResourcePolicy};
pub use driver::{
    DirectFrontier, DriverOptions, FrontierOptions, PreflightOptions, ScopeRouting, SearchResult,
    WorkerLimits, run_pixel_search, run_pixel_search_with_batches,
};
pub use frame_sequence::{FrameSearchReport, SequenceSearchResult, run_native_sequence_search};
pub use native::{
    NativeAcceptance, NativeBackendOptions, NativeImplementation, NativeSearchResult,
    run_native_pixel_search,
};
pub use native_tectonic::{NativeTectonicResult, run_native_tectonic};
pub use params::{Candidate, CandidateParams, PlanOptions, lossless_probe_plan};
pub use probe::{CandidateScreen, ProbeLimits, ProbeResult, ScreenOptions};
pub use scan::{
    LosslessScan, ScanOptions, ScanRegion, ScanTile, scan_lossless_input,
    scan_lossless_input_with_pool, validate_source_input,
};
pub use search::{EncodedOutput, TrialBudget, TrialOutput};

pub const SOURCE_COMMIT: &str = "b866b3defec13494bfbaf705db93581cda118e37";

#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod driver_tests;
#[cfg(test)]
mod frontier_tests;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod routing_tests;
#[cfg(test)]
mod scan_tests;
#[cfg(test)]
mod tests;
