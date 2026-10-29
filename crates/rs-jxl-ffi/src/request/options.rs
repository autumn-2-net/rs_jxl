//! Wire options and operation-specific resource-limit compatibility.

use crate::Result;
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub threads: Option<usize>,
    pub effort: Option<u8>,
    pub source_search: Option<bool>,
    pub strip_metadata: Option<bool>,
    pub group_size_shift: Option<u8>,
    pub rct: Option<u32>,
    pub predictor: Option<u32>,
    pub palette_colors: Option<u32>,
    pub patches: Option<bool>,
    pub learning_fraction: Option<f32>,
    pub ma_max_leaves: Option<usize>,
    pub frame_index: Option<usize>,
    pub coalescing: Option<bool>,
    pub format: Option<String>,
    pub include_extra_channels: Option<bool>,
    pub adjust_orientation: Option<bool>,
    pub dither: Option<String>,
    pub limits: Option<Limits>,
    pub brotli_quality: Option<u8>,
    pub no_cfl: Option<bool>,
    pub order: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_input_bytes: Option<usize>,
    pub max_output_bytes: Option<usize>,
    pub max_metadata_bytes: Option<usize>,
    pub max_pixels: Option<usize>,
    pub max_samples: Option<usize>,
    pub max_frames: Option<usize>,
    pub max_coefficients: Option<usize>,
    pub max_tokens: Option<usize>,
    pub sample_limit: Option<usize>,
    pub max_coefficient_bytes: Option<usize>,
    pub max_block_visits: Option<usize>,
}

impl Options {
    pub fn has_tool_overrides(&self) -> bool {
        self.group_size_shift.is_some()
            || self.rct.is_some()
            || self.predictor.is_some()
            || self.palette_colors.is_some()
            || self.patches.is_some()
            || self.learning_fraction.is_some()
    }
}

impl Limits {
    pub(super) fn validate(&self, jpeg: bool) -> Result<()> {
        let incompatible = if jpeg {
            self.max_pixels.is_some() || self.max_samples.is_some() || self.max_frames.is_some()
        } else {
            self.max_coefficients.is_some()
                || self.max_tokens.is_some()
                || self.sample_limit.is_some()
                || self.max_coefficient_bytes.is_some()
                || self.max_block_visits.is_some()
        };
        if incompatible {
            Err("limits contain fields unsupported by this operation".into())
        } else {
            Ok(())
        }
    }
}
