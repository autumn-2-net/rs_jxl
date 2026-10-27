use crate::Result;
use serde::Deserialize;

mod layers;
pub use layers::{LayerAnimation, LayerBlendInfo, LayerFrameOptions, LayerOptions};

mod multiframe;
pub use multiframe::{AnalysisOptions, AutoOptions};
use multiframe::{deserialize_analysis_options, deserialize_auto_options};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Encode,
    EncodeMultiframe,
    EncodeLayers,
    AnalyzeMultiframe,
    TranscodeJpeg,
    ReconstructJpeg,
    Decode,
    DecodeToPng,
    Info,
    ReadImage,
}

#[derive(Debug)]
pub struct Request {
    pub operation: Operation,
    pub options: Options,
    pub images: Option<Vec<ImageSpec>>,
    pub durations: Option<Vec<u32>>,
    pub reference: Option<String>,
    pub timebase: Option<[u32; 2]>,
    pub loops: Option<u32>,
    pub auto: Option<bool>,
    pub auto_options: Option<AutoOptions>,
    pub analysis_options: Option<AnalysisOptions>,
    pub layer_options: Option<LayerOptions>,
}

impl<'de> Deserialize<'de> for Request {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Auto(#[serde(deserialize_with = "deserialize_auto_options")] Option<AutoOptions>);
        #[derive(Deserialize)]
        struct Analysis(
            #[serde(deserialize_with = "deserialize_analysis_options")] Option<AnalysisOptions>,
        );

        struct Object;
        impl<'de> serde::de::Visitor<'de> for Object {
            type Value = Request;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a request object")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Request, A::Error> {
                const FIELDS: &[&str] = &[
                    "operation",
                    "options",
                    "images",
                    "durations",
                    "reference",
                    "timebase",
                    "loops",
                    "auto",
                    "auto_options",
                    "analysis_options",
                    "layer_options",
                ];
                let mut request = Request {
                    operation: Operation::Encode,
                    options: Options::default(),
                    images: None,
                    durations: None,
                    reference: None,
                    timebase: None,
                    loops: None,
                    auto: None,
                    auto_options: None,
                    analysis_options: None,
                    layer_options: None,
                };
                let mut seen = [false; FIELDS.len()];
                while let Some(key) = map.next_key::<String>()? {
                    let index = FIELDS
                        .iter()
                        .position(|field| *field == key)
                        .ok_or_else(|| {
                            <A::Error as serde::de::Error>::unknown_field(&key, FIELDS)
                        })?;
                    if std::mem::replace(&mut seen[index], true) {
                        return Err(serde::de::Error::duplicate_field(FIELDS[index]));
                    }
                    match key.as_str() {
                        "operation" => request.operation = map.next_value()?,
                        "options" => request.options = map.next_value()?,
                        "images" => request.images = map.next_value()?,
                        "durations" => request.durations = map.next_value()?,
                        "reference" => request.reference = map.next_value()?,
                        "timebase" => request.timebase = map.next_value()?,
                        "loops" => request.loops = map.next_value()?,
                        "auto" => request.auto = map.next_value()?,
                        "auto_options" => request.auto_options = map.next_value::<Auto>()?.0,
                        "analysis_options" => {
                            request.analysis_options = map.next_value::<Analysis>()?.0
                        }
                        "layer_options" => {
                            request.layer_options =
                                Some(map.next_value::<layers::Object<LayerOptions>>()?.0);
                        }
                        _ => unreachable!(),
                    }
                }
                if !seen[0] {
                    return Err(serde::de::Error::missing_field("operation"));
                }
                // Old operations retain null-as-omitted semantics. Layers reject
                // legacy sequence/analysis keys even when their values are null.
                if request.operation == Operation::EncodeLayers {
                    for field in [
                        "durations",
                        "reference",
                        "timebase",
                        "loops",
                        "auto",
                        "auto_options",
                        "analysis_options",
                    ] {
                        if FIELDS
                            .iter()
                            .zip(seen)
                            .any(|(key, present)| *key == field && present)
                        {
                            return Err(serde::de::Error::custom(format!(
                                "{field} is unsupported for encode_layers"
                            )));
                        }
                    }
                }
                Ok(request)
            }
        }
        deserializer.deserialize_map(Object)
    }
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

/// Operation-specific views after wire-level validation. Image buffers and
/// codec-specific option values are still validated by the execution adapters.
/// Options remain borrowed for compatibility while those adapters migrate.
#[derive(Debug)]
pub enum ValidatedRequest<'a> {
    Encode {
        image: &'a ImageSpec,
        options: &'a Options,
    },
    EncodeMultiframe {
        images: &'a [ImageSpec],
        options: &'a Options,
        sequence: Sequence<'a>,
    },
    EncodeLayers {
        images: &'a [ImageSpec],
        options: &'a Options,
        layer_options: Option<&'a LayerOptions>,
    },
    AnalyzeMultiframe {
        images: &'a [ImageSpec],
        strip_metadata: bool,
        options: &'a Options,
        analysis_options: Option<&'a AnalysisOptions>,
    },
    TranscodeJpeg {
        options: &'a Options,
    },
    ReconstructJpeg {
        options: &'a Options,
    },
    Decode {
        options: &'a Options,
        frame_index: Option<usize>,
    },
    DecodeToPng {
        options: &'a Options,
        frame_index: usize,
    },
    Info {
        options: &'a Options,
    },
    ReadImage {
        strip_metadata: bool,
    },
}

#[derive(Debug)]
pub struct Sequence<'a> {
    /// None means one tick for each image; explicit zero durations are retained.
    pub durations: Option<&'a [u32]>,
    pub reference: rs_jxl::multiframe::Reference,
    pub animation: rs_jxl::metadata::Animation,
    pub automatic: bool,
    pub auto_options: Option<&'a AutoOptions>,
}

impl Request {
    pub fn images(&self) -> &[ImageSpec] {
        self.images.as_deref().unwrap_or_default()
    }

    /// Project the JSON v1 schema without moving or cloning images.
    /// `validate()` remains available with its original unit-result contract.
    pub fn validate_typed(&self) -> Result<ValidatedRequest<'_>> {
        self.validate()?;
        use ValidatedRequest::*;
        let options = &self.options;
        Ok(match self.operation {
            Operation::Encode => Encode {
                image: &self.images()[0],
                options,
            },
            Operation::EncodeMultiframe => {
                let [n, d] = self.timebase.unwrap_or([1000, 1]);
                EncodeMultiframe {
                    images: self.images(),
                    options,
                    sequence: Sequence {
                        durations: self.durations.as_deref(),
                        reference: self.multiframe_reference(),
                        animation: rs_jxl::metadata::Animation {
                            ticks_per_second_numerator: n,
                            ticks_per_second_denominator: d,
                            loops: self.loops.unwrap_or(0),
                            timecodes: false,
                        },
                        automatic: self.auto.unwrap_or(false),
                        auto_options: self.auto_options.as_ref(),
                    },
                }
            }
            Operation::EncodeLayers => EncodeLayers {
                images: self.images(),
                options,
                layer_options: self.layer_options.as_ref(),
            },
            Operation::AnalyzeMultiframe => AnalyzeMultiframe {
                images: self.images(),
                strip_metadata: options.strip_metadata.unwrap_or(false),
                options,
                analysis_options: self.analysis_options.as_ref(),
            },
            Operation::TranscodeJpeg => TranscodeJpeg { options },
            Operation::ReconstructJpeg => ReconstructJpeg { options },
            Operation::Decode => Decode {
                options,
                frame_index: options.frame_index,
            },
            Operation::DecodeToPng => DecodeToPng {
                options,
                frame_index: options.frame_index.unwrap_or(0),
            },
            Operation::Info => Info { options },
            Operation::ReadImage => ReadImage {
                strip_metadata: options.strip_metadata.unwrap_or(false),
            },
        })
    }

    pub fn validate(&self) -> Result<()> {
        use Operation::*;
        let sequence = matches!(self.operation, EncodeMultiframe | AnalyzeMultiframe);
        if let Some(layers) = &self.layer_options {
            if self.operation != EncodeLayers {
                return Err("layer_options require encode_layers".into());
            }
            layers.validate(self.images().len())?;
        }
        if self.operation != EncodeMultiframe
            && (self.durations.is_some()
                || self.reference.is_some()
                || self.timebase.is_some()
                || self.loops.is_some()
                || self.auto.is_some()
                || self.auto_options.is_some())
        {
            return Err("sequence fields require encode_multiframe".into());
        }
        if let Some(auto) = &self.auto_options {
            auto.validate(self.auto.unwrap_or(false), self.reference.as_deref())?;
        }
        if let Some(analysis) = &self.analysis_options {
            if self.operation != AnalyzeMultiframe {
                return Err("analysis_options require analyze_multiframe".into());
            }
            analysis.validate()?;
        }
        if !matches!(
            self.operation,
            Encode | EncodeMultiframe | EncodeLayers | AnalyzeMultiframe
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
        if self.operation == EncodeLayers && self.images().is_empty() {
            return Err("encode_layers requires nonempty images".into());
        }
        if let Some(durations) = &self.durations
            && durations.len() != self.images().len()
        {
            return Err("durations must match image count".into());
        }
        if let Some(reference) = self.reference.as_deref()
            && multiframe::parse_reference(reference).is_none()
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
            Encode | EncodeMultiframe | EncodeLayers => &[
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
            AnalyzeMultiframe => &["threads", "strip_metadata"],
            ReadImage => &["strip_metadata"],
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
                return Err(
                    format!("option {field} is unsupported for {:?}", self.operation).into(),
                );
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
        if o.source_search.unwrap_or(false) && self.operation == EncodeLayers {
            return Err("source_search is unsupported for encode_layers".into());
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

#[cfg(test)]
#[path = "request/tests.rs"]
mod tests;
