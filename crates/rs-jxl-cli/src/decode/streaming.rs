//! Incremental input uses the persistent Rust decoder, not read-to-end/redecode.
use super::*;

pub(super) fn run(args: &DecodeArgs, format: OutputFormat) -> Result<()> {
    if format == OutputFormat::Jpg || format.metadata() {
        return Err("--stream-input currently decodes pixels; original-JPEG reconstruction and metadata-only output use the regular input mode".into());
    }
    let mut options = args.options(format);
    args.configure_color(&mut options, format)?;
    if format == OutputFormat::Png && options.format == SampleFormat::Native {
        return Err("streaming PNG output requires explicit --bits 8 or --bits 16; use NPY for native samples".into());
    }
    let chunk_bytes = args.input_chunk_size.unwrap_or(64 * 1024) as usize;
    let started = std::time::Instant::now();
    let image = if args.input == Path::new("-") {
        decode::decode_reader_with_chunk_size(std::io::stdin().lock(), &options, chunk_bytes)?
    } else {
        decode::decode_reader_with_chunk_size(
            std::fs::File::open(&args.input)?,
            &options,
            chunk_bytes,
        )?
    };
    if !image.complete && !(args.allow_partial_files || args.downsampling > 1) {
        return Err("incremental decoder returned unexpected incomplete output".into());
    }
    report(args, image.bytes_read, started);
    if args.disable_output {
        return Ok(());
    }
    diagnostics(args, &image, format);
    let mut batch = outputs::Batch::new(args);
    outputs::pixels(&mut batch, &image, format, &options)?;
    outputs::preview(&mut batch, &image, &options)?;
    batch.sidecars(&metadata::Metadata::from(&image))?;
    batch.publish()
}
