//! JSON request deserialization, including duplicate and explicitly-null fields.

use serde::Deserialize;

use super::multiframe::{deserialize_analysis_options, deserialize_auto_options};
use super::{AnalysisOptions, AutoOptions, LayerOptions, Operation, Options, Request, layers};

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
