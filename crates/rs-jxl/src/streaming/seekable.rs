//! Frozen enc_frame.cc mode 1: stream DC areas and backpatch a normal TOC.
//! No out-of-order transport, full-frame input or full-frame compressed spool.

use super::{seek_output::SeekOutput, toc};
use crate::{
    EncodeOptions, Error, Result,
    bit_writer::BitWriter,
    codestream,
    frame::FrameOptions,
    frame::Geometry,
    modular::layout::Passes,
    region_input::{FrameAreaPlan, RegionSource},
};
use std::io::{Seek, Write};

/// Encode one full-canvas lossless frame from borrowed regions. Source policy
/// must be resolved for the full image and explicitly eligible for frame areas.
/// Input/output callbacks remain on the caller thread; neither requires Send.
///
/// Emits an ordinary version-0 container, seeking back to finalize its length
/// and frame directory. Success leaves the sink at the encoded end and returns
/// the total byte length. It does not truncate, flush or close the caller's sink.
/// Use an empty sink or retain the returned range. Failure after output starts
/// leaves an invalid partial image that must not be published or resumed.
pub fn encode_regions_seekable_to<S: RegionSource + ?Sized>(
    source: &mut S,
    options: &EncodeOptions,
    output: impl Write + Seek,
) -> Result<usize> {
    let plan = FrameAreaPlan::new(source.descriptor().clone(), options)?;
    let descriptor = plan.descriptor();
    let geometry = source.descriptor().geometry();
    let image_header = crate::execution::serial(|| {
        codestream::image_header_fields(
            descriptor.width(),
            descriptor.height(),
            descriptor.metadata(),
            options.max_output_bytes,
        )
    })?;
    let level10 = !codestream::container_prefix_geometry(geometry, 0)?.is_empty();
    let prepared = AreaFrame::new(plan, geometry, &FrameOptions::default(), options, true)?;
    let pool = if options.threads == 1 || crate::execution::is_serial() {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(options.threads)
                .build()
                .map_err(|e| Error::Encoding(e.to_string()))?,
        )
    };

    let mut prefix = crate::container::PREFIX.to_vec();
    if level10 {
        prefix.extend_from_slice(b"\0\0\0\x09jxll\x0a");
    }
    let box_start = prefix.len();
    prefix.extend_from_slice(b"\0\0\0\x01jxlc\0\0\0\0\0\0\0\x10");
    prefix
        .len()
        .checked_add(image_header.len())
        .filter(|&n| n <= options.max_output_bytes)
        .ok_or(Error::Limit("streaming output bytes"))?;
    let mut writer = SeekOutput::new(output, options.max_output_bytes)?;
    writer.write(&prefix)?;
    writer.write(&image_header)?;
    prepared.emit(source, pool.as_ref(), &mut writer)?;
    let box_size =
        u64::try_from(writer.end() - box_start).map_err(|_| Error::Limit("streaming box size"))?;
    writer.seek(box_start + 8)?;
    writer.write(&box_size.to_be_bytes())?;
    writer.finish()
}

/// A complete, validated frame header and source topology, without any samples.
pub(super) struct AreaFrame {
    plan: FrameAreaPlan,
    header: Vec<u8>,
    sizes: Vec<usize>,
    limit: usize,
}

impl AreaFrame {
    pub fn new(
        plan: FrameAreaPlan,
        canvas: Geometry<'_>,
        frame: &FrameOptions,
        options: &EncodeOptions,
        last: bool,
    ) -> Result<Self> {
        let descriptor = plan.descriptor();
        let toc_plan = toc::StreamingTocPlan::new(
            descriptor.width() as usize,
            descriptor.height() as usize,
            128usize << options.group_size_shift,
            1,
            plan.section_count(),
        )?;
        if toc_plan.dc_group_order.len() != plan.area_count() {
            return Err(Error::Encoding("streaming DC area count changed".into()));
        }
        for (position, section) in plan.sections().iter().enumerate() {
            if toc_plan
                .permutation
                .get(section.canonical_index)
                .copied()
                .map(|p| p as usize)
                != Some(position)
            {
                return Err(Error::Encoding(
                    "streaming permutation differs from area order".into(),
                ));
            }
        }
        let mut header = codestream::frame_header_body(
            canvas,
            descriptor.geometry(),
            options.group_size_shift,
            &Passes::default(),
            frame,
            last,
            false,
            options.max_output_bytes,
        )?;
        crate::execution::serial(|| toc::write_permutation(&mut header, &toc_plan.permutation))?;
        let frame_header = header.into_bytes()?;
        let mut sizes = Vec::new();
        sizes
            .try_reserve_exact(plan.section_count())
            .map_err(|_| Error::Limit("streaming section sizes"))?;
        Ok(Self {
            plan,
            header: frame_header,
            sizes,
            limit: options.max_output_bytes,
        })
    }

    pub fn emit<S: RegionSource + ?Sized, W: Write + Seek>(
        mut self,
        source: &mut S,
        pool: Option<&rayon::ThreadPool>,
        writer: &mut SeekOutput<W>,
    ) -> Result<()> {
        let frame_start = writer.position();
        let plan = &self.plan;
        let frame_header = &self.header;
        let sizes = &mut self.sizes;
        let mut dc_global = Vec::new();
        let mut reserved = None;
        plan.emit(source, pool, |section, bytes| {
            let position = sizes.len();
            if plan.sections().get(position) != Some(&section) {
                return Err(Error::Encoding("streaming section order changed".into()));
            }
            sizes.push(bytes.len());
            if position == 0 {
                let (minimum, offset) =
                    toc::group_data_offset(frame_header.len(), bytes.len(), plan.section_count())?;
                let absolute = frame_start
                    .checked_add(offset)
                    .ok_or(Error::Limit("streaming group offset"))?;
                writer.seek(absolute)?;
                dc_global
                    .try_reserve_exact(bytes.len())
                    .map_err(|_| Error::Limit("streaming global bytes"))?;
                dc_global.extend_from_slice(bytes);
                reserved = Some((minimum, offset));
            } else {
                writer.write(bytes)?;
            }
            Ok(())
        })?;
        if sizes.len() != plan.section_count() {
            return Err(Error::Encoding("streaming section count changed".into()));
        }
        let (minimum, offset) =
            reserved.ok_or_else(|| Error::Encoding("missing streaming global section".into()))?;
        let padding = toc::dc_global_padding(sizes, frame_header.len(), offset, minimum)?;
        sizes[0] = sizes[0]
            .checked_add(padding)
            .ok_or(Error::Limit("streaming global padding"))?;
        let mut table = BitWriter::new(self.limit);
        toc::write_sizes(&mut table, sizes)?;
        let table = table.into_bytes()?;
        if writer
            .end()
            .checked_sub(frame_start)
            .is_none_or(|n| n > self.limit)
        {
            return Err(Error::Limit("streaming frame bytes"));
        }
        writer.seek(frame_start)?;
        writer.write(frame_header)?;
        writer.write(&table)?;
        writer.write(&dc_global)?;
        writer.zeros(padding)?;
        if writer.position() != frame_start + offset {
            return Err(Error::Encoding(
                "streaming prefix does not end at reserved group offset".into(),
            ));
        }
        writer.seek(writer.end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Image, PixelData, PixelLayout,
        region_input::{ImageDescriptor, ImageRegionSource, RegionLease, RegionRequest},
    };
    use std::{
        cell::{Cell, RefCell},
        io::{Cursor, SeekFrom},
        rc::Rc,
        thread::{self, ThreadId},
    };

    fn image() -> Image {
        Image::new(
            1025,
            3,
            PixelLayout::Gray,
            PixelData::U8((0..1025 * 3).map(|x| (x % 11) as u8).collect()),
        )
        .unwrap()
    }

    fn options() -> EncodeOptions {
        let mut options = EncodeOptions {
            threads: 4,
            group_size_shift: 0,
            ..Default::default()
        };
        options.modular.palette_policy = Some(crate::modular::palette::PaletteOptions {
            groupwise: true,
            palette_colors: 0,
            channel_colors_pre_transform_percent: 0.0,
            channel_colors_percent: 0.0,
            ..Default::default()
        });
        options
    }

    struct LocalSource<'a> {
        inner: ImageRegionSource<'a>,
        calls: Rc<Cell<usize>>,
        thread: ThreadId,
    }
    impl RegionSource for LocalSource<'_> {
        fn descriptor(&self) -> &ImageDescriptor {
            self.inner.descriptor()
        }
        fn acquire(&mut self, request: RegionRequest) -> Result<RegionLease<'_>> {
            assert_eq!(thread::current().id(), self.thread);
            self.calls.set(self.calls.get() + 1);
            self.inner.acquire(request)
        }
    }
    struct LocalSink {
        data: Rc<RefCell<Cursor<Vec<u8>>>>,
        thread: ThreadId,
    }
    impl Write for LocalSink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            assert_eq!(thread::current().id(), self.thread);
            self.data.borrow_mut().write(bytes)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            panic!("caller-owned sink must not be flushed")
        }
    }
    impl Seek for LocalSink {
        fn seek(&mut self, offset: SeekFrom) -> std::io::Result<u64> {
            assert_eq!(thread::current().id(), self.thread);
            self.data.borrow_mut().seek(offset)
        }
    }

    #[test]
    fn parallel_codec_keeps_non_send_provider_and_sink_on_calling_thread() {
        let image = image();
        let calls = Rc::new(Cell::new(0));
        let thread = thread::current().id();
        let mut source = LocalSource {
            inner: ImageRegionSource::new(&image).unwrap(),
            calls: calls.clone(),
            thread,
        };
        let data = Rc::new(RefCell::new(Cursor::new(Vec::new())));
        let sink = LocalSink {
            data: data.clone(),
            thread,
        };
        let size = encode_regions_seekable_to(&mut source, &options(), sink).unwrap();
        assert!(calls.get() > 1);
        assert_eq!(size, data.borrow().get_ref().len());
        assert_eq!(size as u64, data.borrow().position());
    }

    #[test]
    fn source_eligibility_precedes_input_and_output_and_budget_counts_overhead() {
        let image = image();
        let calls = Rc::new(Cell::new(0));
        let mut source = LocalSource {
            inner: ImageRegionSource::new(&image).unwrap(),
            calls: calls.clone(),
            thread: thread::current().id(),
        };
        let mut sink = Cursor::new(Vec::new());
        assert!(
            encode_regions_seekable_to(&mut source, &EncodeOptions::default(), &mut sink).is_err()
        );
        assert_eq!(calls.get(), 0);
        assert!(sink.get_ref().is_empty());
        let size = encode_regions_seekable_to(&mut source, &options(), &mut sink).unwrap();
        for maximum in [size, size - 1] {
            let mut bounded = options();
            bounded.max_output_bytes = maximum;
            let mut output = Cursor::new(Vec::new());
            let result = encode_regions_seekable_to(&mut source, &bounded, &mut output);
            assert_eq!(result.is_ok(), maximum == size);
            assert!(output.get_ref().len() <= maximum);
        }
    }
}
