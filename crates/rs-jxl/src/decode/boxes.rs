use super::DecodedBox;
use crate::{Error, Result};
use jxl_decoder::api::{JxlAuxBox, JxlAuxBoxType, JxlDecoder};
use std::io::Read;

pub(super) const REQUESTED: [JxlAuxBoxType; 5] = [
    JxlAuxBoxType::EXIF,
    JxlAuxBoxType::XMP,
    JxlAuxBoxType::JUMBF,
    JxlAuxBoxType::JPEG_RECONSTRUCTION_DATA,
    JxlAuxBoxType::HDR_GAIN_MAP,
];
pub(super) fn extract(
    decoder: &JxlDecoder,
    trailing: &[u8],
    limit: usize,
) -> Result<Vec<DecodedBox>> {
    let mut result = Vec::new();
    let mut remaining = limit;
    for kind in REQUESTED {
        for entry in decoder.aux_boxes(kind) {
            let data = expand(entry, &[], remaining)?;
            remaining -= data.len();
            result.push(DecodedBox { kind: kind.0, data });
        }
    }
    if let Some(entry) = decoder.trailing_box() {
        let data = expand(entry, trailing, remaining)?;
        result.push(DecodedBox {
            kind: entry.box_type().0,
            data,
        });
    }
    Ok(result)
}
fn expand(entry: &JxlAuxBox, trailing: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let reader = entry.raw_data().chain(trailing);
    let mut reader: Box<dyn Read + '_> = if entry.is_compressed() {
        Box::new(brotli::Decompressor::new(reader, 4096))
    } else {
        Box::new(reader)
    };
    let mut buffer = [0u8; 4096];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        if read > limit - result.len() {
            return Err(Error::Limit("decompressed metadata bytes"));
        }
        result
            .try_reserve(read)
            .map_err(|_| Error::Limit("metadata allocation"))?;
        result.extend_from_slice(&buffer[..read]);
    }
    Ok(result)
}
