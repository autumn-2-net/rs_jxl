//! NPY 1.0 serialization, including all extra planes in codestream order.
use super::*;

pub(in crate::decode) fn write_npy(output: &mut impl Write, frames: &[DecodedFrame]) -> Result<()> {
    let first = validate_frames(frames)?;
    let planes = || {
        frames
            .iter()
            .flat_map(|frame| std::iter::once(&frame.pixels).chain(&frame.extra_planes))
    };
    let integer =
        planes().all(|p| matches!(p, PixelData::U8(_) | PixelData::U16(_) | PixelData::U32(_)));
    let storage_bits = if integer {
        planes().map(PixelData::bits).max().unwrap()
    } else {
        if planes().any(|p| dtype(p) != dtype(&first.pixels)) {
            return Err("NPY cannot mix integer and floating sample types exactly; use the Rust API for separate typed planes".into());
        }
        first.pixels.bits()
    };
    let output_dtype = if integer {
        match storage_bits {
            8 => "|u1",
            16 => "<u2",
            _ => "<u4",
        }
    } else {
        dtype(&first.pixels)
    };
    if first.layout.has_alpha() && first.extra_planes.is_empty() {
        return Err(
            "NPY requires the separate primary alpha plane to preserve its source precision".into(),
        );
    }
    let color_channels = first.layout.channels() - usize::from(first.layout.has_alpha());
    let channels = color_channels
        .checked_add(first.extra_planes.len())
        .ok_or("channel count overflow")?;
    let shape = if frames.len() == 1 {
        format!("({}, {}, {})", first.height, first.width, channels)
    } else {
        format!(
            "({}, {}, {}, {})",
            frames.len(),
            first.height,
            first.width,
            channels
        )
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}",
        output_dtype
    );
    // NPY 1.0 has a ten-byte prefix and a newline-terminated, 64-byte-aligned header.
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let header_len = u16::try_from(header.len())?;
    first
        .width
        .checked_mul(first.height)
        .and_then(|n| n.checked_mul(channels))
        .and_then(|n| n.checked_mul(usize::from(storage_bits / 8)))
        .and_then(|n| n.checked_mul(frames.len()))
        .and_then(|n| n.checked_add(10 + header.len()))
        .filter(|&n| n <= MAX_OUTPUT_BYTES)
        .ok_or("NPY output exceeds the CLI's 512 MiB limit")?;
    output.write_all(b"\x93NUMPY\x01\x00")?;
    output.write_all(&header_len.to_le_bytes())?;
    output.write_all(header.as_bytes())?;
    for frame in frames {
        let stride = frame.layout.channels();
        for pixel in 0..frame.width * frame.height {
            for channel in 0..color_channels {
                write_widened(
                    output,
                    &frame.pixels,
                    pixel * stride + channel,
                    storage_bits,
                )?;
            }
            for plane in &frame.extra_planes {
                write_widened(output, plane, pixel, storage_bits)?;
            }
        }
    }
    Ok(())
}

fn write_widened(output: &mut impl Write, data: &PixelData, index: usize, bits: u8) -> Result<()> {
    if bits == data.bits() {
        return write_sample(output, data, index);
    }
    let value = match data {
        PixelData::U8(values) => u32::from(values[index]),
        PixelData::U16(values) => u32::from(values[index]),
        _ => return Err("invalid NPY integer widening".into()),
    };
    if bits == 16 {
        output.write_all(&(value as u16).to_le_bytes())?;
    } else {
        output.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
