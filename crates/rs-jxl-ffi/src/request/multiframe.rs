use super::Request;
use crate::Result;
use rs_jxl::multiframe::Reference;
use serde::Deserialize;

pub(super) fn parse_reference(value: &str) -> Option<Reference> {
    match value {
        "full" => Some(Reference::Full),
        "previous" => Some(Reference::Previous),
        "first" => Some(Reference::First),
        "smallest_crop" => Some(Reference::SmallestCrop),
        _ => None,
    }
}

impl Request {
    pub(crate) fn multiframe_reference(&self) -> Reference {
        self.reference
            .as_deref()
            .and_then(parse_reference)
            .unwrap_or(Reference::Previous)
    }
}

pub(super) fn deserialize_analysis_options<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<AnalysisOptions>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Object;
    impl<'de> serde::de::Visitor<'de> for Object {
        type Value = Option<AnalysisOptions>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("an analysis_options object")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            AnalysisOptions::deserialize(serde::de::value::MapAccessDeserializer::new(map))
                .map(Some)
        }
    }
    deserializer.deserialize_map(Object)
}

fn deserialize_analysis_reference<'de, D>(
    deserializer: D,
) -> std::result::Result<rs_jxl::multiframe::Reference, D::Error>
where
    D: serde::Deserializer<'de>,
{
    parse_reference(&String::deserialize(deserializer)?)
        .ok_or_else(|| serde::de::Error::custom("unsupported analysis reference"))
}

/// Geometric controls only. Worker count belongs to options/context resources.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalysisOptions {
    #[serde(deserialize_with = "deserialize_analysis_reference")]
    pub reference: rs_jxl::multiframe::Reference,
    pub allow_mask: bool,
    pub allow_multi_rect: bool,
    pub adaptive_rectangles: bool,
    pub tile_size: u32,
    pub extra_tile_sizes: Vec<u32>,
    pub history_window: usize,
    pub max_rectangles: usize,
    pub rectangle_bytes_per_pixel: f64,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        let native = rs_jxl::multiframe::AnalysisOptions::default();
        Self {
            reference: native.reference,
            allow_mask: native.allow_mask,
            allow_multi_rect: native.allow_multi_rect,
            adaptive_rectangles: native.adaptive_rectangles,
            tile_size: native.tile_size,
            extra_tile_sizes: native.extra_tile_sizes,
            history_window: native.history_window,
            max_rectangles: native.max_rectangles,
            rectangle_bytes_per_pixel: native.rectangle_bytes_per_pixel,
        }
    }
}

impl AnalysisOptions {
    pub fn native(&self, threads: usize) -> rs_jxl::multiframe::AnalysisOptions {
        rs_jxl::multiframe::AnalysisOptions {
            reference: self.reference,
            allow_mask: self.allow_mask,
            allow_multi_rect: self.allow_multi_rect,
            adaptive_rectangles: self.adaptive_rectangles,
            tile_size: self.tile_size,
            extra_tile_sizes: self.extra_tile_sizes.clone(),
            history_window: self.history_window,
            max_rectangles: self.max_rectangles,
            rectangle_bytes_per_pixel: self.rectangle_bytes_per_pixel,
            threads,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.extra_tile_sizes.contains(&0) {
            return Err("extra_tile_sizes must be positive".into());
        }
        if self.tile_size == 0 || self.max_rectangles == 0 {
            return Err("tile_size and max_rectangles must be positive".into());
        }
        if !self.rectangle_bytes_per_pixel.is_finite() || self.rectangle_bytes_per_pixel < 0.0 {
            return Err("rectangle_bytes_per_pixel must be finite and nonnegative".into());
        }
        Ok(())
    }
}

pub(super) fn deserialize_auto_options<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<AutoOptions>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Object;
    impl<'de> serde::de::Visitor<'de> for Object {
        type Value = Option<AutoOptions>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("an auto_options object or null")
        }

        fn visit_none<E: serde::de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D: serde::Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> std::result::Result<Self::Value, D::Error> {
            deserializer.deserialize_map(self)
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            // Preserve duplicate keys for derived serde validation, unlike Value.
            AutoOptions::deserialize(serde::de::value::MapAccessDeserializer::new(map)).map(Some)
        }
    }
    deserializer.deserialize_option(Object)
}

/// Wire overrides use native defaults; reference belongs to the outer request.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AutoOptions {
    pub retained_reference: bool,
    pub reference_lookahead: bool,
    pub allow_mask: bool,
    pub allow_multi_rect: bool,
    pub tile_size: u32,
    pub extra_tile_sizes: Vec<u32>,
    pub max_rectangles: usize,
    pub prune: bool,
    pub adaptive_rectangles: bool,
    pub max_frames: usize,
    pub max_total_trials: usize,
    pub max_trial_pixels: u64,
}

impl Default for AutoOptions {
    fn default() -> Self {
        let native = rs_jxl::multiframe::AutoOptions::default();
        Self {
            retained_reference: native.retained_reference,
            reference_lookahead: native.reference_lookahead,
            allow_mask: native.allow_mask,
            allow_multi_rect: native.allow_multi_rect,
            tile_size: native.tile_size,
            extra_tile_sizes: native.extra_tile_sizes,
            max_rectangles: native.max_rectangles,
            prune: native.prune,
            adaptive_rectangles: native.adaptive_rectangles,
            max_frames: native.max_frames,
            max_total_trials: native.max_total_trials,
            max_trial_pixels: native.max_trial_pixels,
        }
    }
}

impl AutoOptions {
    pub(super) fn validate(&self, automatic: bool, reference: Option<&str>) -> Result<()> {
        if self.extra_tile_sizes.contains(&0) {
            return Err("extra_tile_sizes must be positive".into());
        }
        if !automatic {
            return Err("auto_options require auto=true".into());
        }
        if self.reference_lookahead && !self.retained_reference {
            return Err("reference_lookahead requires retained_reference=true".into());
        }
        if self.retained_reference && reference != Some("smallest_crop") {
            return Err("retained_reference requires reference=smallest_crop".into());
        }
        if self.tile_size == 0 || self.max_rectangles == 0 {
            return Err("tile_size and max_rectangles must be positive".into());
        }
        Ok(())
    }

    pub fn native(
        &self,
        reference: rs_jxl::multiframe::Reference,
    ) -> rs_jxl::multiframe::AutoOptions {
        rs_jxl::multiframe::AutoOptions {
            reference,
            retained_reference: self.retained_reference,
            reference_lookahead: self.reference_lookahead,
            allow_mask: self.allow_mask,
            allow_multi_rect: self.allow_multi_rect,
            tile_size: self.tile_size,
            extra_tile_sizes: self.extra_tile_sizes.clone(),
            max_rectangles: self.max_rectangles,
            prune: self.prune,
            adaptive_rectangles: self.adaptive_rectangles,
            max_frames: self.max_frames,
            max_total_trials: self.max_total_trials,
            max_trial_pixels: self.max_trial_pixels,
        }
    }
}
