//! Pure Rust OpenEXR. F32 and lossless ZIP avoid implicit half-float quantization.
use super::*;
use exr::prelude::{
    AnyChannel, AnyChannels, Encoding, FlatSamples, Image, Layer, LayerAttributes, WritableImage,
};

pub(in crate::decode) fn write_exr(output: &mut Output, image: &Raster<'_>) -> Result<()> {
    let expected = rs_jxl::decode::parse_color_description("RGB_D65_SRG_Rel_Lin")?;
    let gray = rs_jxl::decode::parse_color_description("Gra_D65_Rel_Lin")?;
    if !matches!(image.output_profile, Some(rs_jxl::decode::JxlColorProfile::Simple(encoding)) if *encoding == expected || *encoding == gray)
    {
        return Err(
            "EXR requires verified linear-sRGB output; use --color-space linear for an EXR preview"
                .into(),
        );
    }
    let frame = validate_frames(image.frames)?;
    let PixelData::F32(pixels) = &frame.pixels else {
        return Err("EXR requires float32 color samples".into());
    };
    let stride = frame.layout.channels();
    let colors = stride - usize::from(frame.layout.has_alpha());
    let count = frame
        .width
        .checked_mul(frame.height)
        .ok_or("EXR size overflow")?;
    let channel_count = colors
        .checked_add(frame.extra_planes.len())
        .ok_or("EXR channel count overflow")?;
    count
        .checked_mul(channel_count)
        .and_then(|n| n.checked_mul(4))
        .ok_or("EXR channel staging size overflow")?;
    if frame.extra_planes.len() != image.info.extra_channels.len() {
        return Err("EXR channel metadata mismatch".into());
    }
    let primary = image
        .info
        .extra_channels
        .iter()
        .position(|ec| ec.ec_type == ExtraChannel::Alpha);
    if frame.layout.has_alpha() && !image.alpha_associated() {
        return Err("EXR uses associated alpha; explicitly select --alpha-mode premultiplied (changes hidden RGB) or use NPY".into());
    }
    if frame.layout.has_alpha() && primary.is_none() {
        return Err("EXR requires the separate primary alpha plane".into());
    }
    let mut channels = Vec::new();
    channels.try_reserve_exact(channel_count)?;
    for channel in 0..colors {
        let name = if colors == 1 {
            "Y"
        } else {
            ["R", "G", "B"][channel]
        };
        let mut values = Vec::new();
        values.try_reserve_exact(count)?;
        values.extend(pixels.chunks_exact(stride).map(|pixel| pixel[channel]));
        channels.push(AnyChannel::new(name, FlatSamples::F32(values)));
    }
    for (index, plane) in frame.extra_planes.iter().enumerate() {
        let PixelData::F32(values) = plane else {
            return Err("EXR requires float32 extra planes; use --bits 32".into());
        };
        let mut copy = Vec::new();
        copy.try_reserve_exact(values.len())?;
        copy.extend_from_slice(values);
        let name = if Some(index) == primary && frame.layout.has_alpha() {
            "A".to_owned()
        } else {
            format!("ec{index}")
        };
        channels.push(AnyChannel::new(name.as_str(), FlatSamples::F32(copy)));
    }
    let attributes = LayerAttributes {
        white_luminance: Some(image.output_intensity_target),
        ..LayerAttributes::default()
    };
    let layer = Layer::new(
        (frame.width, frame.height),
        attributes,
        Encoding::SMALL_LOSSLESS,
        AnyChannels::sort(channels.into_iter().collect()),
    );
    let mut exr = Image::from_layer(layer);
    // CLI color negotiation requests linear sRGB before this writer is reached.
    exr.attributes.chromaticities = Some(exr::meta::attribute::Chromaticities {
        red: (0.64, 0.33).into(),
        green: (0.30, 0.60).into(),
        blue: (0.15, 0.06).into(),
        white: (0.3127, 0.3290).into(),
    });
    exr.write().non_parallel().to_buffered(output)?;
    Ok(())
}
