use std::fmt;

#[derive(Debug)]
pub enum Error {
    InvalidInput(String),
    Limit(&'static str),
    Encoding(String),
    Decoding(String),
    Io(std::io::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) | Self::Encoding(s) | Self::Decoding(s) => f.write_str(s),
            Self::Limit(s) => write!(f, "resource limit: {s}"),
            Self::Io(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<crate::bit_writer::BitWriterError> for Error {
    fn from(e: crate::bit_writer::BitWriterError) -> Self {
        use crate::bit_writer::BitWriterError as B;
        match e {
            B::SizeLimit | B::Allocation => Self::Limit("encoded output allocation"),
            _ => Self::Encoding(e.to_string()),
        }
    }
}
pub type Result<T> = std::result::Result<T, Error>;
