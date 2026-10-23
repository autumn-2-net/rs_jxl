//! Patch atlas/residual framing; discovery and dictionary logic stay in tools.
use crate::{
    Image, Result,
    bit_writer::BitWriter,
    entropy::stream::{Coding, Model},
    tools::{lz77, patches},
};

#[path = "source_patch_frame.rs"]
mod source;

pub(crate) fn prepare_source(
    image: &Image,
    config: &patches::SourcePatchConfig,
) -> Result<Option<(Image, Image, patches::PatchDictionary)>> {
    source::prepare(image, config)
}

pub(crate) fn prepare(
    image: &Image,
    options: &patches::PatchOptions,
) -> Result<Option<(Image, Image, patches::PatchDictionary)>> {
    options.validate()?;
    if image.metadata().bit_depth.is_float() || !(1..=16).contains(&image.bits()) {
        return Ok(None);
    }
    let color_channels = image.color_channels();
    // Replace is the documented integer specialization, not the source Add
    // detector. Match every eligible full-resolution integer extra exactly.
    // Other extras retain their native samples and use None, never resampling
    // or interpreting float words as unsigned detector values.
    let channels: Vec<_> = (0..image.channel_count())
        .filter(|&c| {
            let depth = image.channel_depth(c);
            image.channel_shift(c) == 0 && !depth.is_float() && depth.bits_per_sample <= 16
        })
        .collect();
    let planes: Vec<_> = channels
        .iter()
        .map(|&c| source::read_plane(image, c))
        .collect();
    // Equality-based discovery needs a shared range bound, not rescaled values.
    let bit_depth = channels
        .iter()
        .map(|&c| image.channel_depth(c).bits_per_sample)
        .max()
        .expect("nonempty color channels");
    let Some(plan) = patches::discover(
        patches::ImageView {
            width: image.width() as usize,
            height: image.height() as usize,
            bit_depth: u32::from(bit_depth),
            color_channels,
            planes: &planes,
        },
        options,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(source::assemble(
        image,
        &channels,
        (plan.reference.width, plan.reference.height),
        plan.reference.planes,
        plan.residual,
        plan.dictionary,
    )?))
}

pub(crate) fn write_dictionary(
    w: &mut BitWriter,
    dictionary: Option<&patches::PatchDictionary>,
) -> Result<()> {
    if let Some(dictionary) = dictionary {
        let raw = patches::tokenize(dictionary)?
            .into_iter()
            .map(|t| lz77::Literal {
                context: t.context,
                value: t.value,
            })
            .collect::<Vec<_>>();
        // PatchDictionaryEncoder::Encode uses default HistogramParams, including
        // RLE, source entropy selection and width-zero distances.
        let config = lz77::SourceConfig::new(lz77::SourceLz77Method::Lz77b1w3t);
        let outcome = lz77::parse_source_rle_with_state(&raw, 0, &config)?;
        let (tokens, _) =
            crate::modular::encode::commands_to_tokens(outcome.commands, patches::NUM_CONTEXTS, 0)?;
        let streams = [tokens];
        let entropy = crate::entropy::options::EntropyOptions {
            lz77_length_config: config.length_uint_config,
            lz77_min_symbol: config.min_symbol(),
            ..Default::default()
        };
        let model = Model::build_with_options(
            &streams,
            patches::NUM_CONTEXTS + usize::from(outcome.enabled),
            outcome.enabled,
            Coding::Source,
            &entropy,
        )?;
        model.write_header(w)?;
        model.write_stream(&streams[0], w)?;
    }
    Ok(())
}
