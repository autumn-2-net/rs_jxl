use crate::{
    Result, message,
    request::{Dtype, ExtraSpec, ImageFormat, ImageSpec},
};
use rs_jxl::{
    Image, PixelData, PixelLayout,
    color::{ColorEncoding, ColorSpace},
    metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType, Orientation},
};
use serde_json::{Value, json};

pub fn region(data: &[u8], offset: usize, length: usize) -> Result<&[u8]> {
    if length > isize::MAX as usize {
        return Err("image region exceeds isize::MAX".into());
    }
    let end = offset
        .checked_add(length)
        .ok_or("image offset/length overflow")?;
    data.get(offset..end)
        .ok_or_else(|| "image region is outside data buffer".into())
}

fn sample_count(width: usize, height: usize, channels: usize) -> Result<usize> {
    if width == 0 || height == 0 || channels == 0 {
        return Err("pixel dimensions/channels must be nonzero".into());
    }
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| "sample count overflow".into())
}

pub fn unpack(bytes: &[u8], dtype: Dtype, count: usize) -> Result<PixelData> {
    let expected = count
        .checked_mul(dtype.bytes())
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or("sample byte count overflow")?;
    if bytes.len() != expected {
        return Err("pixel length does not match dimensions and dtype".into());
    }
    // Chunk reads deliberately avoid typed pointer casts: inputs may be unaligned.
    macro_rules! words {
        ($size:literal, $convert:expr, $variant:ident) => {{
            let mut values = Vec::new();
            values.try_reserve_exact(count).map_err(message)?;
            for chunk in bytes.chunks_exact($size) {
                let array = chunk.try_into().map_err(message)?;
                values.push(($convert)(array));
            }
            PixelData::$variant(values)
        }};
    }
    Ok(match dtype {
        Dtype::U8 => {
            let mut v = Vec::new();
            v.try_reserve_exact(count).map_err(message)?;
            v.extend_from_slice(bytes);
            PixelData::U8(v)
        }
        Dtype::U16 => words!(2, u16::from_le_bytes, U16),
        Dtype::U32 => words!(4, u32::from_le_bytes, U32),
        Dtype::F16 => words!(2, u16::from_le_bytes, F16),
        Dtype::F32 => words!(4, f32::from_le_bytes, F32),
    })
}

fn depth(pixels: &PixelData, bits: Option<u8>, exponent: Option<u8>) -> Result<BitDepth> {
    let default = pixels.bit_depth();
    let depth = BitDepth {
        bits_per_sample: bits.unwrap_or(default.bits_per_sample),
        exponent_bits_per_sample: exponent.unwrap_or(default.exponent_bits_per_sample),
    };
    depth.validate().map_err(message)?;
    Ok(depth)
}

fn extra(
    spec: &ExtraSpec,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<(ExtraChannelInfo, PixelData)> {
    let kind = match spec.kind.as_str() {
        "alpha" => ExtraChannelType::Alpha,
        "depth" => ExtraChannelType::Depth,
        "spot_color" => ExtraChannelType::SpotColor,
        "selection_mask" => ExtraChannelType::SelectionMask,
        "black" => ExtraChannelType::Black,
        "cfa" => ExtraChannelType::Cfa,
        "thermal" => ExtraChannelType::Thermal,
        "optional" => ExtraChannelType::Optional,
        _ => return Err(format!("unsupported extra channel type: {}", spec.kind).into()),
    };
    if spec.dim_shift > 3 {
        return Err("dim_shift must be 0..3".into());
    }
    if spec.spot_color.is_some() && kind != ExtraChannelType::SpotColor {
        return Err("spot_color requires type spot_color".into());
    }
    if spec.cfa_channel.is_some() && kind != ExtraChannelType::Cfa {
        return Err("cfa_channel requires type cfa".into());
    }
    let scale = 1usize << spec.dim_shift;
    let count = sample_count(
        (width as usize).div_ceil(scale),
        (height as usize).div_ceil(scale),
        1,
    )?;
    let pixels = unpack(region(data, spec.offset, spec.length)?, spec.dtype, count)?;
    let mut info = ExtraChannelInfo::new(kind, 8).map_err(message)?;
    info.bit_depth = depth(&pixels, spec.bits_per_sample, spec.exponent_bits_per_sample)?;
    info.name = spec.name.clone();
    info.dim_shift = spec.dim_shift;
    info.alpha_associated = spec.alpha_premultiplied;
    info.spot_color = spec.spot_color.unwrap_or([0.0; 4]);
    info.cfa_channel = spec.cfa_channel.unwrap_or(1);
    info.validate().map_err(message)?;
    Ok((info, pixels))
}

pub fn image(spec: &ImageSpec, data: &[u8], strip_metadata: bool) -> Result<(Image, bool)> {
    let bytes = region(data, spec.offset, spec.length)?;
    if spec.format == ImageFormat::Encoded {
        if spec.width.is_some()
            || spec.height.is_some()
            || spec.channels.is_some()
            || spec.dtype.is_some()
            || spec.bits_per_sample.is_some()
            || spec.exponent_bits_per_sample.is_some()
            || spec.extra_channels.is_some()
            || spec.icc.is_some()
            || spec.color_encoding.is_some()
            || spec.orientation.is_some()
        {
            return Err("encoded images accept only format, offset and length".into());
        }
        let decoded = rs_jxl_image::decode(bytes, strip_metadata).map_err(message)?;
        return Ok((decoded.image, decoded.jpeg));
    }
    let (width, height) = (
        spec.width.ok_or("pixels require width")?,
        spec.height.ok_or("pixels require height")?,
    );
    let channels = spec.channels.ok_or("pixels require channels")?;
    let layout = match channels {
        1 => PixelLayout::Gray,
        2 => PixelLayout::GrayAlpha,
        3 => PixelLayout::Rgb,
        4 => PixelLayout::Rgba,
        _ => return Err("channels must be 1..4".into()),
    };
    let pixels = unpack(
        bytes,
        spec.dtype.ok_or("pixels require dtype")?,
        sample_count(width as usize, height as usize, channels as usize)?,
    )?;
    let declared = depth(&pixels, spec.bits_per_sample, spec.exponent_bits_per_sample)?;
    let image = Image::new(width, height, layout, pixels).map_err(message)?;
    let mut meta = image.metadata().clone();
    meta.bit_depth = declared;
    if layout.has_alpha() {
        meta.extra_channels[0].bit_depth = declared;
    }
    meta.orientation = Orientation::try_from(spec.orientation.unwrap_or(1)).map_err(message)?;
    let space = if channels <= 2 {
        ColorSpace::Gray
    } else {
        ColorSpace::Rgb
    };
    if spec.icc.is_some() && spec.color_encoding.is_some() {
        return Err("icc and color_encoding are mutually exclusive".into());
    }
    meta.color_encoding = if let Some(icc) = &spec.icc {
        ColorEncoding::icc(space, icc.clone()).map_err(message)?
    } else {
        match spec.color_encoding.as_deref().unwrap_or("srgb") {
            "srgb" => ColorEncoding::srgb(space),
            "linear_srgb" => ColorEncoding::linear_srgb(space),
            _ => return Err("unsupported color_encoding; use srgb, linear_srgb or icc".into()),
        }
    };
    let mut planes = Vec::new();
    for spec in spec.extra_channels.as_deref().unwrap_or_default() {
        let (info, pixels) = extra(spec, data, width, height)?;
        meta.extra_channels.push(info);
        planes.push(pixels);
    }
    Ok((image.with_metadata(meta, planes).map_err(message)?, false))
}

pub fn append(
    data: &mut Vec<u8>,
    pixels: &PixelData,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Value> {
    if pixels.len() != sample_count(width, height, channels)? {
        return Err("output plane dimensions do not match storage".into());
    }
    let (dtype, bytes) = match pixels {
        PixelData::U8(_) => ("u8", 1),
        PixelData::U16(_) => ("u16", 2),
        PixelData::U32(_) => ("u32", 4),
        PixelData::F16(_) => ("f16", 2),
        PixelData::F32(_) => ("f32", 4),
    };
    let length = pixels
        .len()
        .checked_mul(bytes)
        .ok_or("output byte count overflow")?;
    let offset = data.len();
    offset
        .checked_add(length)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or("output exceeds isize::MAX")?;
    data.try_reserve(length).map_err(message)?;
    match pixels {
        PixelData::U8(v) => data.extend_from_slice(v),
        PixelData::U16(v) | PixelData::F16(v) => {
            for n in v {
                data.extend_from_slice(&n.to_le_bytes());
            }
        }
        PixelData::U32(v) => {
            for n in v {
                data.extend_from_slice(&n.to_le_bytes());
            }
        }
        PixelData::F32(v) => {
            for n in v {
                data.extend_from_slice(&n.to_le_bytes());
            }
        }
    }
    Ok(
        json!({"width":width,"height":height,"channels":channels,"dtype":dtype,"offset":offset,"length":length}),
    )
}

pub fn extra_metadata(info: &ExtraChannelInfo, index: usize) -> Value {
    let kind = match info.channel_type {
        ExtraChannelType::Alpha => "alpha",
        ExtraChannelType::Depth => "depth",
        ExtraChannelType::SpotColor => "spot_color",
        ExtraChannelType::SelectionMask => "selection_mask",
        ExtraChannelType::Black => "black",
        ExtraChannelType::Cfa => "cfa",
        ExtraChannelType::Thermal => "thermal",
        ExtraChannelType::Optional => "optional",
    };
    json!({"index":index,"type":kind,"name":info.name,"bits_per_sample":info.bit_depth.bits_per_sample,"exponent_bits_per_sample":info.bit_depth.exponent_bits_per_sample,"dim_shift":info.dim_shift,"alpha_premultiplied":info.alpha_associated,"spot_color":info.spot_color,"cfa_channel":info.cfa_channel})
}

pub fn merge(target: &mut Value, other: Value) {
    if let (Some(target), Value::Object(other)) = (target.as_object_mut(), other) {
        target.extend(other);
    }
}

pub fn alpha_plane(pixels: &PixelData, channels: usize) -> PixelData {
    macro_rules! alpha {
        ($values:expr, $variant:ident) => {
            PixelData::$variant(
                $values
                    .iter()
                    .skip(channels - 1)
                    .step_by(channels)
                    .copied()
                    .collect(),
            )
        };
    }
    match pixels {
        PixelData::U8(v) => alpha!(v, U8),
        PixelData::U16(v) => alpha!(v, U16),
        PixelData::U32(v) => alpha!(v, U32),
        PixelData::F16(v) => alpha!(v, F16),
        PixelData::F32(v) => alpha!(v, F32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endian_chunks_preserve_float_words_and_wide_integers() {
        let f16 = [0xff, 0x00, 0x80, 0x01, 0x7e];
        assert_eq!(
            unpack(&f16[1..], Dtype::F16, 2).unwrap(),
            PixelData::F16(vec![0x8000, 0x7e01])
        );
        let words = [0x8000_0000u32, 0x7fc1_2345];
        let mut bytes = vec![0xff];
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let floats = unpack(&bytes[1..], Dtype::F32, 2).unwrap();
        assert_eq!(floats, PixelData::F32(words.map(f32::from_bits).to_vec()));
        let mut packed = Vec::new();
        let desc = append(&mut packed, &floats, 2, 1, 1).unwrap();
        assert_eq!(packed, &bytes[1..]);
        assert_eq!(desc["dtype"], "f32");
        assert_eq!(
            unpack(&[255, 255, 255, 127], Dtype::U32, 1).unwrap(),
            PixelData::U32(vec![0x7fff_ffff])
        );
        assert!(unpack(&[0], Dtype::U16, 1).is_err());
        assert!(unpack(&[], Dtype::F32, usize::MAX).is_err());
    }

    #[test]
    fn subsampled_plane_uses_ceiling_dimensions() {
        let spec:ImageSpec=serde_json::from_value(json!({"width":3,"height":3,"channels":1,"dtype":"u8","length":9,"extra_channels":[{"type":"depth","offset":9,"length":4,"dtype":"u8","dim_shift":1}]})).unwrap();
        let (image, _) = image(&spec, &[0; 13], false).unwrap();
        assert_eq!(image.channel_dimensions(1), (2, 2));
        assert_eq!(image.extra_planes()[0].len(), 4);
        assert!(region(&[0], usize::MAX, 2).is_err());
    }
}
