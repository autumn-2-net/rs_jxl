//! Direct ports of Modular primitives, not a complete encoder.
//!
//! Source baseline: libjxl `b866b3defec13494bfbaf705db93581cda118e37`.
//! See [`predictor`] for source identifiers, arithmetic contracts, and scope.
//! Crate-root registration and central test execution belong to the parent.

pub(crate) mod auxiliary;
pub mod cost;
pub mod encode;
pub mod fixed;
pub(crate) mod frame;
pub(crate) mod global;
pub mod group;
pub mod layout;
pub mod ma_train;
pub(crate) mod orchestration;
pub mod palette;
pub mod predict_simd;
pub mod predictor;
pub mod rct_simd;
pub mod sample;
pub mod sample_convert;
pub mod source_options;
mod tokens;
pub(crate) mod training;
pub mod transforms;
pub mod tree;
pub mod tree_groups;
pub mod weighted;

pub use orchestration::{GroupwiseDecision, GroupwiseFallback, groupwise_decision};
