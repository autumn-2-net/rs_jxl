use crate::Result;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Encode,
    EncodeMultiframe,
    AnalyzeMultiframe,
    TranscodeJpeg,
    ReconstructJpeg,
    Decode,
    DecodeToPng,
    Info,
    ReadImage,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub operation: Operation,
    #[serde(default)]
    pub options: Options,
    pub images: Option<Vec<ImageSpec>>,
    pub durations: Option<Vec<u32>>,
    pub reference: Option<String>,
    pub timebase: Option<[u32; 2]>,
    pub loops: Option<u32>,
    pub auto: Option<bool>,
}

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

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    #[default]
    Pixels,
    Encoded,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    U8,
    U16,
    U32,
    F16,
    F32,
}

impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::F16 => 2,
            Self::U32 | Self::F32 => 4,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageSpec {
    #[serde(default)]
    pub format: ImageFormat,
    #[serde(default)]
    pub offset: usize,
    pub length: usize,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub channels: Option<u8>,
    pub dtype: Option<Dtype>,
    pub bits_per_sample: Option<u8>,
    pub exponent_bits_per_sample: Option<u8>,
    pub extra_channels: Option<Vec<ExtraSpec>>,
    pub icc: Option<Vec<u8>>,
    pub color_encoding: Option<String>,
    pub orientation: Option<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtraSpec {
    #[serde(default)]
    pub offset: usize,
    pub length: usize,
    pub dtype: Dtype,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    pub bits_per_sample: Option<u8>,
    pub exponent_bits_per_sample: Option<u8>,
    #[serde(default)]
    pub dim_shift: u8,
    #[serde(default)]
    pub alpha_premultiplied: bool,
    pub spot_color: Option<[f32; 4]>,
    pub cfa_channel: Option<u32>,
}

impl Request {
    pub fn images(&self) -> &[ImageSpec] {
        self.images.as_deref().unwrap_or_default()
    }

    pub fn validate(&self) -> Result<()> {
        use Operation::*;
        let sequence = matches!(self.operation, EncodeMultiframe | AnalyzeMultiframe);
        if self.operation != EncodeMultiframe
            && (self.durations.is_some()
                || self.reference.is_some()
                || self.timebase.is_some()
                || self.loops.is_some()
                || self.auto.is_some())
        {
            return Err("sequence fields require encode_multiframe".into());
        }
        if !matches!(
            self.operation,
            Encode | EncodeMultiframe | AnalyzeMultiframe
        ) && self.images.is_some()
        {
            return Err(
                "this operation consumes the entire data buffer and does not accept images".into(),
            );
        }
        if self.operation == Encode && self.images().len() != 1 {
            return Err("encode requires exactly one image".into());
        }
        if sequence && self.images().is_empty() {
            return Err("multiframe requires nonempty images".into());
        }
        if let Some(durations) = &self.durations
            && durations.len() != self.images().len()
        {
            return Err("durations must match image count".into());
        }
        if let Some(reference) = self.reference.as_deref()
            && !matches!(reference, "full" | "previous" | "first" | "smallest_crop")
        {
            return Err("unsupported reference".into());
        }
        if let Some([n, d]) = self.timebase {
            rs_jxl::metadata::Animation {
                ticks_per_second_numerator: n,
                ticks_per_second_denominator: d,
                ..Default::default()
            }
            .validate()
            .map_err(crate::message)?;
        }
        // Presence is retained so even an explicitly supplied default cannot hide
        // an inapplicable control. serde itself rejects unknown keys/duplicates.
        let o = &self.options;
        let mut fields = Vec::new();
        macro_rules! fields { ($($name:ident),* $(,)?) => { $(if o.$name.is_some() { fields.push(stringify!($name)); })* }; }
        fields!(
            threads,
            effort,
            source_search,
            strip_metadata,
            group_size_shift,
            rct,
            predictor,
            palette_colors,
            patches,
            learning_fraction,
            ma_max_leaves,
            frame_index,
            coalescing,
            format,
            include_extra_channels,
            adjust_orientation,
            dither,
            limits,
            brotli_quality,
            no_cfl,
            order
        );
        let allowed: &[&str] = match self.operation {
            Encode | EncodeMultiframe => &[
                "threads",
                "effort",
                "source_search",
                "strip_metadata",
                "group_size_shift",
                "rct",
                "predictor",
                "palette_colors",
                "patches",
                "learning_fraction",
                "ma_max_leaves",
            ],
            AnalyzeMultiframe | ReadImage => &["strip_metadata"],
            Decode | DecodeToPng => &[
                "threads",
                "frame_index",
                "coalescing",
                "format",
                "include_extra_channels",
                "adjust_orientation",
                "dither",
                "limits",
            ],
            Info => &["threads", "coalescing", "adjust_orientation", "limits"],
            ReconstructJpeg => &["threads", "limits"],
            TranscodeJpeg => &[
                "threads",
                "effort",
                "source_search",
                "brotli_quality",
                "no_cfl",
                "order",
                "limits",
            ],
        };
        for field in fields {
            if !allowed.contains(&field) {
                return Err(format!(
                    "option {field} is unsupported for {:?}",
                    self.operation
                ));
            }
        }
        if !(1..=256).contains(&o.threads.unwrap_or(1)) {
            return Err("threads must be 1..256".into());
        }
        if !(1..=11).contains(&o.effort.unwrap_or(7)) {
            return Err("effort must be 1..11".into());
        }
        if o.brotli_quality.is_some_and(|v| v > 11) {
            return Err("brotli_quality must be 0..11".into());
        }
        if o.ma_max_leaves
            .is_some_and(|v| !(1..=2_097_152).contains(&v))
        {
            return Err("ma_max_leaves must be 1..2097152".into());
        }
        if o.palette_colors.is_some_and(|v| v > 70_911) {
            return Err("palette_colors must be 0..70911".into());
        }
        if o.source_search.unwrap_or(false) && self.operation == EncodeMultiframe {
            return Err("source_search is unsupported for multiframe; use auto".into());
        }
        if o.source_search.unwrap_or(false) && o.has_tool_overrides() {
            return Err("source_search cannot guarantee explicit tool overrides".into());
        }
        if let Some(limits) = &o.limits {
            limits.validate(self.operation == TranscodeJpeg)?;
        }
        Ok(())
    }
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
    fn validate(&self, jpeg: bool) -> Result<()> {
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
