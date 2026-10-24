//! Netpbm/PFM and PGX serialization. Reference: libjxl/lib/extras/enc/{pnm,pgx}.cc.
use super::*;

#[cfg(test)]
pub(in crate::decode) fn write_portable(
    output: &mut impl Write,
    frame: &DecodedFrame,
    format: OutputFormat,
) -> Result<()> {
    write_portable_depth(output, frame, format, frame.pixels.bits(), false)
}

fn integer_sample(
    output: &mut impl Write,
    samples: &PixelData,
    index: usize,
    depth: u8,
) -> Result<()> {
    if !(1..=16).contains(&depth) {
        return Err("portable integer depth must be 1..16".into());
    }
    let value = match samples {
        PixelData::U8(values) => u32::from(values[index]),
        PixelData::U16(values) => u32::from(values[index]),
        _ => return Err("portable integer output requires U8/U16".into()),
    };
    if value > (1u32 << depth) - 1 {
        return Err("sample exceeds declared portable bit depth".into());
    }
    if depth <= 8 {
        output.write_all(&[value as u8])?;
    } else {
        output.write_all(&(value as u16).to_be_bytes())?;
    }
    Ok(())
}

pub(in crate::decode) fn write_portable_depth(
    output: &mut impl Write,
    frame: &DecodedFrame,
    format: OutputFormat,
    depth: u8,
    extras_separate: bool,
) -> Result<()> {
    validate_frames(std::slice::from_ref(frame))?;
    if (!frame.extra_planes.is_empty() || frame.layout.has_alpha()) && !extras_separate {
        return Err("this format cannot store alpha/extra planes; use --output-extra-channels, PAM, EXR or NPY".into());
    }
    let stride = frame.layout.channels();
    let channels = stride - usize::from(frame.layout.has_alpha());
    let magic = match (format, channels) {
        (OutputFormat::Ppm | OutputFormat::Pnm, 3) => "P6",
        (OutputFormat::Pgm | OutputFormat::Pnm, 1) => "P5",
        (OutputFormat::Pfm, 3) => "PF",
        (OutputFormat::Pfm, 1) => "Pf",
        (OutputFormat::Pgx, 1) => "PG",
        _ => return Err("PPM requires RGB; PGM/PGX require grayscale; PNM selects either without color conversion".into()),
    };
    write_header(output, magic, frame.width, frame.height, depth)?;
    if format == OutputFormat::Pfm {
        let PixelData::F32(values) = &frame.pixels else {
            return Err("PFM requires float32".into());
        };
        for y in (0..frame.height).rev() {
            for x in 0..frame.width {
                for channel in 0..channels {
                    output.write_all(
                        &values[(y * frame.width + x) * stride + channel].to_le_bytes(),
                    )?;
                }
            }
        }
    } else {
        for pixel in 0..frame.width * frame.height {
            for channel in 0..channels {
                integer_sample(output, &frame.pixels, pixel * stride + channel, depth)?;
            }
        }
    }
    Ok(())
}

fn write_header(
    output: &mut impl Write,
    magic: &str,
    width: usize,
    height: usize,
    depth: u8,
) -> Result<()> {
    match magic {
        "PF" | "Pf" => write!(output, "{magic}\n{width} {height}\n-1.0\n")?,
        _ if !(1..=16).contains(&depth) => return Err("integer raster depth must be 1..16".into()),
        "PG" => writeln!(output, "PG ML + {depth} {width} {height}")?,
        _ => write!(
            output,
            "{magic}\n{width} {height}\n{}\n",
            (1u32 << depth) - 1
        )?,
    }
    Ok(())
}

pub(in crate::decode) fn write_plane(
    output: &mut impl Write,
    width: usize,
    height: usize,
    plane: &PixelData,
    format: OutputFormat,
    depth: u8,
) -> Result<()> {
    let count = width
        .checked_mul(height)
        .filter(|&n| n > 0)
        .ok_or("invalid plane dimensions")?;
    if plane.len() != count {
        return Err("extra plane size mismatch".into());
    }
    let magic = match format {
        OutputFormat::Pfm => "Pf",
        OutputFormat::Pgx => "PG",
        OutputFormat::Pgm | OutputFormat::Pnm => "P5",
        _ => return Err("unsupported extra-plane format".into()),
    };
    write_header(output, magic, width, height, depth)?;
    if format == OutputFormat::Pfm {
        let PixelData::F32(values) = plane else {
            return Err("PFM extra planes require F32".into());
        };
        for y in (0..height).rev() {
            for value in &values[y * width..(y + 1) * width] {
                output.write_all(&value.to_le_bytes())?;
            }
        }
    } else {
        for i in 0..count {
            integer_sample(output, plane, i, depth)?;
        }
    }
    Ok(())
}

pub(in crate::decode) fn write_pam(
    output: &mut impl Write,
    image: &Raster<'_>,
    samples: SampleFormat,
) -> Result<()> {
    let frame = validate_frames(image.frames)?;
    if frame.extra_planes.len() != image.info.extra_channels.len() {
        return Err("PAM channel metadata mismatch".into());
    }
    let depth = integer_depth(&image.info.bit_depth, samples)?;
    if image
        .info
        .extra_channels
        .iter()
        .any(|ec| integer_depth(&ec.bit_depth, samples).ok() != Some(depth))
    {
        return Err("PAM needs a common integer range for all channels; explicitly choose --bits 8/16 or use EXR/NPY".into());
    }
    let color = frame.layout.channels() - usize::from(frame.layout.has_alpha());
    if frame.layout.has_alpha()
        && !image
            .info
            .extra_channels
            .iter()
            .any(|ec| ec.ec_type == ExtraChannel::Alpha)
    {
        return Err("PAM requires the authoritative separate alpha plane".into());
    }
    let count = color
        .checked_add(frame.extra_planes.len())
        .ok_or("PAM depth overflow")?;
    if frame.layout.has_alpha() && image.alpha_associated() {
        return Err(
            "PAM alpha tuples require straight alpha; use --alpha-mode unpremultiplied or NPY"
                .into(),
        );
    }
    let tuple = match (color, image.info.extra_channels.as_slice()) {
        (1, []) => "GRAYSCALE",
        (3, []) => "RGB",
        (1, [ec]) if ec.ec_type == ExtraChannel::Alpha && frame.layout.has_alpha() => {
            "GRAYSCALE_ALPHA"
        }
        (3, [ec]) if ec.ec_type == ExtraChannel::Alpha && frame.layout.has_alpha() => "RGB_ALPHA",
        _ => "JXL_CHANNELS",
    };
    write!(
        output,
        "P7\nWIDTH {}\nHEIGHT {}\nDEPTH {count}\nMAXVAL {}\nTUPLTYPE {tuple}\nENDHDR\n",
        frame.width,
        frame.height,
        (1u32 << depth) - 1
    )?;
    for pixel in 0..frame.width * frame.height {
        for channel in 0..color {
            integer_sample(
                output,
                &frame.pixels,
                pixel * frame.layout.channels() + channel,
                depth,
            )?;
        }
        for plane in &frame.extra_planes {
            integer_sample(output, plane, pixel, depth)?;
        }
    }
    Ok(())
}
