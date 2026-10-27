use std::marker::PhantomData;

use rs_jxl::{
    frame::{BlendInfo, BlendMode, FrameOptions},
    metadata::Animation,
};
use serde::Deserialize;

use crate::{Result, message};

// Force JSON objects while preserving duplicate keys for derived serde checks.
// Deserializing a derived struct directly also accepts positional JSON arrays.
pub(super) struct Object<T>(pub T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Object<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Visitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<T> {
            type Value = Object<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                T::deserialize(serde::de::value::MapAccessDeserializer::new(map)).map(Object)
            }
        }
        deserializer.deserialize_map(Visitor(PhantomData))
    }
}

fn object<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<T, D::Error> {
    Object::<T>::deserialize(deserializer).map(|value| value.0)
}

fn optional_object<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::<Object<T>>::deserialize(deserializer).map(|value| value.map(|value| value.0))
}

fn objects<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<T>, D::Error> {
    Vec::<Object<T>>::deserialize(deserializer)
        .map(|values| values.into_iter().map(|value| value.0).collect())
}

fn optional_objects<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Option<Vec<T>>, D::Error> {
    Option::<Vec<Object<T>>>::deserialize(deserializer)
        .map(|values| values.map(|values| values.into_iter().map(|value| value.0).collect()))
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LayerOptions {
    pub canvas: Option<[u32; 2]>,
    #[serde(deserialize_with = "optional_objects")]
    pub frames: Option<Vec<LayerFrameOptions>>,
    #[serde(deserialize_with = "optional_object")]
    pub animation: Option<LayerAnimation>,
}

impl LayerOptions {
    pub fn native(&self) -> rs_jxl::layers::LayerOptions {
        rs_jxl::layers::LayerOptions {
            canvas: self.canvas.map(|[width, height]| (width, height)),
            frames: self.frames.as_ref().map_or_else(Vec::new, |frames| {
                frames.iter().map(LayerFrameOptions::native).collect()
            }),
            animation: self.animation.as_ref().map(LayerAnimation::native),
        }
    }

    pub(super) fn validate(&self, image_count: usize) -> Result<()> {
        if self
            .frames
            .as_ref()
            .is_some_and(|frames| frames.len() != image_count)
        {
            return Err("layer_options.frames must match image count".into());
        }
        if let Some(animation) = &self.animation {
            animation.native().validate().map_err(message)?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LayerAnimation {
    pub ticks_per_second_numerator: u32,
    pub ticks_per_second_denominator: u32,
    pub loops: u32,
    pub timecodes: bool,
}

impl Default for LayerAnimation {
    fn default() -> Self {
        let native = Animation::default();
        Self {
            ticks_per_second_numerator: native.ticks_per_second_numerator,
            ticks_per_second_denominator: native.ticks_per_second_denominator,
            loops: native.loops,
            timecodes: native.timecodes,
        }
    }
}

impl LayerAnimation {
    pub fn native(&self) -> Animation {
        Animation {
            ticks_per_second_numerator: self.ticks_per_second_numerator,
            ticks_per_second_denominator: self.ticks_per_second_denominator,
            loops: self.loops,
            timecodes: self.timecodes,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LayerFrameOptions {
    pub origin: [i32; 2],
    pub duration: u32,
    pub timecode: u32,
    pub name: String,
    #[serde(deserialize_with = "object")]
    pub blend: LayerBlendInfo,
    #[serde(deserialize_with = "objects")]
    pub extra_blend: Vec<LayerBlendInfo>,
    pub save_as_reference: u8,
}

impl Default for LayerFrameOptions {
    fn default() -> Self {
        let native = rs_jxl::layers::default_frame_options();
        Self {
            origin: [native.origin.0, native.origin.1],
            duration: native.duration,
            timecode: native.timecode,
            name: native.name,
            blend: LayerBlendInfo::default(),
            extra_blend: Vec::new(),
            save_as_reference: native.save_as_reference,
        }
    }
}

impl LayerFrameOptions {
    pub fn native(&self) -> FrameOptions {
        FrameOptions {
            origin: (self.origin[0], self.origin[1]),
            duration: self.duration,
            timecode: self.timecode,
            name: self.name.clone(),
            blend: self.blend.native(),
            extra_blend: self
                .extra_blend
                .iter()
                .map(LayerBlendInfo::native)
                .collect(),
            save_as_reference: self.save_as_reference,
            ..rs_jxl::layers::default_frame_options()
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LayerBlendInfo {
    #[serde(deserialize_with = "blend_mode")]
    pub mode: BlendMode,
    pub source: u8,
    pub alpha_channel: u8,
    pub clamp: bool,
}

fn blend_mode<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<BlendMode, D::Error> {
    match String::deserialize(deserializer)?.as_str() {
        "replace" => Ok(BlendMode::Replace),
        "add" => Ok(BlendMode::Add),
        "blend" => Ok(BlendMode::Blend),
        "alpha_weighted_add" => Ok(BlendMode::AlphaWeightedAdd),
        "multiply" => Ok(BlendMode::Multiply),
        _ => Err(serde::de::Error::custom("unsupported layer blend mode")),
    }
}

impl Default for LayerBlendInfo {
    fn default() -> Self {
        let native = rs_jxl::layers::default_frame_options().blend;
        Self {
            mode: native.mode,
            source: native.source,
            alpha_channel: native.alpha_channel,
            clamp: native.clamp,
        }
    }
}

impl LayerBlendInfo {
    pub fn native(&self) -> BlendInfo {
        BlendInfo {
            mode: self.mode,
            source: self.source,
            alpha_channel: self.alpha_channel,
            clamp: self.clamp,
        }
    }
}
