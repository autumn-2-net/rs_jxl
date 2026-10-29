//! Borrowed payload descriptors. Native buffers are checked by `pixels`.

use serde::Deserialize;

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
