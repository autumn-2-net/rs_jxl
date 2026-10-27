use std::{error::Error as StdError, fmt};

/// Stable categories in the JSON v1 `error_kind` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    InvalidInput,
    ResourceLimit,
    Encoding,
    Decoding,
    Io,
    Internal,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub stage: Option<&'static str>,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            stage: None,
        }
    }

    /// Attach context without overwriting a more specific stage.
    pub fn at_stage(mut self, stage: &'static str) -> Self {
        if self.stage.is_none() {
            self.stage = Some(stage);
        }
        self
    }

    /// Preserve the outer message, recovering categories from known error types.
    /// Unclassified reader errors retain the legacy invalid-input default.
    pub fn from_error(error: &(dyn StdError + 'static)) -> Self {
        let (kind, stage) = classify(error, 0).unwrap_or((ErrorKind::InvalidInput, None));
        Self {
            kind,
            message: error.to_string(),
            stage,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl StdError for Error {}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }
}

/// Compatibility adapter for existing `map_err(message)` call sites.
pub fn message(error: impl Into<Error>) -> Error {
    error.into()
}

fn classify(
    error: &(dyn StdError + 'static),
    depth: usize,
) -> Option<(ErrorKind, Option<&'static str>)> {
    use ErrorKind::*;
    // Error sources can be cyclic; never let a foreign error recurse forever.
    if depth == 32 {
        return None;
    }
    if let Some(error) = error.downcast_ref::<Error>() {
        return Some((error.kind, error.stage));
    }
    if let Some(error) = error.downcast_ref::<rs_jxl::Error>() {
        let kind = match error {
            rs_jxl::Error::InvalidInput(_) => InvalidInput,
            rs_jxl::Error::Limit(_) => ResourceLimit,
            rs_jxl::Error::Encoding(_) => Encoding,
            rs_jxl::Error::Decoding(_) => Decoding,
            rs_jxl::Error::Io(_) => Io,
        };
        return Some((kind, None));
    }
    if let Some(error) = error.downcast_ref::<serde_json::Error>() {
        return Some((if error.is_io() { Io } else { InvalidInput }, None));
    }
    if error.is::<std::collections::TryReserveError>() {
        return Some((ResourceLimit, None));
    }
    if let Some(error) = error.downcast_ref::<std::io::Error>() {
        return error
            .get_ref()
            .and_then(|source| classify(source, depth + 1))
            .or(Some((Io, None)));
    }
    if let Some(error) = error.downcast_ref::<png::EncodingError>() {
        return match error {
            png::EncodingError::IoError(error) => classify(error, depth + 1),
            png::EncodingError::LimitsExceeded => Some((ResourceLimit, None)),
            _ => Some((Encoding, None)),
        };
    }
    if let Some(error) = error.downcast_ref::<png::DecodingError>() {
        return match error {
            png::DecodingError::IoError(error) => classify(error, depth + 1),
            png::DecodingError::LimitsExceeded => Some((ResourceLimit, None)),
            _ => Some((Decoding, None)),
        };
    }
    if error.is::<std::num::TryFromIntError>()
        || error.is::<std::array::TryFromSliceError>()
        || error.is::<rs_jxl::modular::predictor::PredictorError>()
    {
        return Some((InvalidInput, None));
    }
    error
        .source()
        .and_then(|source| classify(source, depth + 1))
}

macro_rules! from_error {
    ($($ty:ty),* $(,)?) => {
        $(impl From<$ty> for Error {
            fn from(error: $ty) -> Self {
                Self::from_error(&error)
            }
        })*
    };
}

from_error!(
    rs_jxl::Error,
    serde_json::Error,
    std::io::Error,
    std::collections::TryReserveError,
    std::num::TryFromIntError,
    std::array::TryFromSliceError,
    rs_jxl::modular::predictor::PredictorError,
    png::EncodingError,
    png::DecodingError,
);

impl From<Box<dyn StdError>> for Error {
    fn from(error: Box<dyn StdError>) -> Self {
        Self::from_error(error.as_ref())
    }
}

impl From<Box<dyn StdError + Send + Sync>> for Error {
    fn from(error: Box<dyn StdError + Send + Sync>) -> Self {
        Self::from_error(error.as_ref())
    }
}
