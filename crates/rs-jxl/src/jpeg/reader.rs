// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Bounded structural port of lib/jxl/jpeg/enc_jpeg_data_reader.cc and
//! jpeg_data.h at b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Reads 8-bit SOF0/1/2 Huffman DCT JPEGs, retaining original marker segments
//! and opaque entropy bytes for exact reconstruction. Unlike ReadJpeg(kReadAll),
//! this does not decode coefficients, check scan progression / entropy padding /
//! restart ordering/spacing or establish that a JPEG is decodable. DQT/DHT/DRI may be
//! redefined between scans; each SOS snapshots their active definitions.
//! Arithmetic, lossless, hierarchical, DNL and other extension modes are rejected.

use super::data::{
    AppMarkerType, CodingProcess, ComponentScanInfo, EntropyData, HuffmanCode, JpegData, JpegError,
    JpegRecord, MAX_COMPONENTS, MAX_DHT_MARKERS, MAX_HUFFMAN_TABLES, MAX_QUANT_TABLES,
    MarkerSegment, NATURAL_ORDER, QuantTable, RestartMarker, ScanInfo,
};

/// Resource bounds are checked before retaining input-derived collections.
/// No allocation is proportional to the image's pixel or MCU count.
#[derive(Clone, Debug)]
pub struct ReadLimits {
    /// Includes metadata, all scans and data following EOI.
    pub max_input_bytes: usize,
    pub max_records: usize,
    pub max_scans: usize,
    /// Counts definitions, including redefinitions of an existing slot.
    pub max_quant_tables: usize,
    pub max_huffman_tables: usize,
    /// Total across all scans, not per scan.
    pub max_restart_markers: usize,
}

impl Default for ReadLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024 * 1024,
            max_records: 65_536,
            max_scans: 4096,
            max_quant_tables: 4096,
            max_huffman_tables: MAX_DHT_MARKERS,
            max_restart_markers: 1_048_576,
        }
    }
}

pub fn read_jpeg(input: &[u8]) -> Result<JpegData, JpegError> {
    read_jpeg_with_limits(input, &ReadLimits::default())
}

pub fn read_jpeg_with_limits(input: &[u8], limits: &ReadLimits) -> Result<JpegData, JpegError> {
    if input.len() > limits.max_input_bytes {
        return Err(limit(0, "input bytes"));
    }
    if input.len() < 2 {
        return Err(truncated(0, "SOI marker"));
    }
    if input[..2] != [0xff, 0xd8] {
        return Err(invalid(0, "expected SOI marker"));
    }
    Reader {
        input,
        pos: 2,
        limits,
        jpeg: JpegData::default(),
        quant: [None; MAX_QUANT_TABLES],
        dc: [None; MAX_HUFFMAN_TABLES],
        ac: [None; MAX_HUFFMAN_TABLES],
        restart_count: 0,
    }
    .read()
}

fn invalid(offset: usize, reason: &'static str) -> JpegError {
    JpegError::Invalid { offset, reason }
}

fn truncated(offset: usize, context: &'static str) -> JpegError {
    JpegError::Truncated { offset, context }
}

fn unsupported(offset: usize, reason: &'static str) -> JpegError {
    JpegError::Unsupported { offset, reason }
}

fn limit(offset: usize, resource: &'static str) -> JpegError {
    JpegError::LimitExceeded { offset, resource }
}

/// A cursor bounded by one marker's declared length, never by the whole file.
struct SegmentReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    offset: usize,
}

impl<'a> SegmentReader<'a> {
    fn new(segment: &'a [u8], marker_offset: usize) -> Self {
        Self {
            bytes: &segment[2..],
            pos: 0,
            offset: marker_offset + 4,
        }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], JpegError> {
        if count > self.bytes.len() - self.pos {
            return Err(truncated(self.offset + self.pos, "marker payload"));
        }
        let bytes = &self.bytes[self.pos..self.pos + count];
        self.pos += count;
        Ok(bytes)
    }

    fn byte(&mut self) -> Result<u8, JpegError> {
        Ok(self.take(1)?[0])
    }

    fn word(&mut self) -> Result<u16, JpegError> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn finish(&self) -> Result<(), JpegError> {
        if self.remaining() != 0 {
            return Err(invalid(self.offset + self.pos, "incorrect marker length"));
        }
        Ok(())
    }
}

struct Reader<'a> {
    input: &'a [u8],
    pos: usize,
    limits: &'a ReadLimits,
    jpeg: JpegData,
    quant: [Option<usize>; MAX_QUANT_TABLES],
    dc: [Option<usize>; MAX_HUFFMAN_TABLES],
    ac: [Option<usize>; MAX_HUFFMAN_TABLES],
    restart_count: usize,
}

impl<'a> Reader<'a> {
    fn push(&mut self, record: JpegRecord) -> Result<(), JpegError> {
        if self.jpeg.records.len() >= self.limits.max_records {
            return Err(limit(self.pos, "records"));
        }
        self.jpeg.records.push(record);
        Ok(())
    }

    fn marker(&mut self, offset: usize, marker: u8, data: &[u8]) -> Result<(), JpegError> {
        self.push(JpegRecord::Marker(MarkerSegment {
            offset,
            marker,
            data: data.to_vec(),
        }))?;
        self.jpeg.marker_order.push(marker);
        Ok(())
    }

    fn read(mut self) -> Result<JpegData, JpegError> {
        self.marker(0, 0xd8, &[])?;
        loop {
            let (offset, marker) = self.next_marker()?;
            match marker {
                0xd8 => return Err(invalid(offset, "duplicate SOI marker")),
                0xd0..=0xd7 => return Err(invalid(offset, "restart marker outside a scan")),
                0xd9 => {
                    if self.jpeg.coding_process.is_none() || self.jpeg.scan_info.is_empty() {
                        return Err(invalid(offset, "EOI before SOF and SOS"));
                    }
                    self.marker(offset, marker, &[])?;
                    if self.pos < self.input.len() {
                        self.push(JpegRecord::Tail { offset: self.pos })?;
                        self.jpeg.tail_data = self.input[self.pos..].to_vec();
                    }
                    return Ok(self.jpeg);
                }
                0xc3 | 0xc7 => return Err(unsupported(offset, "lossless JPEG")),
                0xc9..=0xcf => {
                    return Err(unsupported(offset, "arithmetic JPEG"));
                }
                0xc5 | 0xc6 | 0xde | 0xdf => {
                    return Err(unsupported(offset, "hierarchical/differential JPEG"));
                }
                0xdc => return Err(unsupported(offset, "DNL (deferred image height)")),
                0xc0..=0xc2 | 0xc4 | 0xda | 0xdb | 0xdd | 0xe0..=0xef | 0xfe => {}
                _ => return Err(unsupported(offset, "reserved or extension marker")),
            }
            let segment = self.segment()?;
            match marker {
                0xc0..=0xc2 => self.sof(offset, marker, segment)?,
                0xc4 => self.dht(offset, segment)?,
                0xdb => self.dqt(offset, segment)?,
                0xdd => {
                    let mut cursor = SegmentReader::new(segment, offset);
                    let interval = cursor.word()?;
                    cursor.finish()?;
                    self.jpeg.restart_interval = u32::from(interval);
                }
                0xda => self.sos(offset, segment)?,
                0xe0..=0xef => {
                    let mut bytes = Vec::with_capacity(segment.len() + 1);
                    bytes.push(marker);
                    bytes.extend_from_slice(segment);
                    self.jpeg.app_data.push(bytes);
                    self.jpeg.app_marker_type.push(AppMarkerType::Unknown);
                }
                0xfe => {
                    let mut bytes = Vec::with_capacity(segment.len() + 1);
                    bytes.push(marker);
                    bytes.extend_from_slice(segment);
                    self.jpeg.com_data.push(bytes);
                }
                _ => unreachable!("supported markers were checked above"),
            }
            self.marker(offset, marker, segment)?;
            if marker == 0xda {
                self.entropy()?;
            }
        }
    }

    fn segment(&mut self) -> Result<&'a [u8], JpegError> {
        let start = self.pos;
        let length_bytes = self.input[start..]
            .get(..2)
            .ok_or_else(|| truncated(start, "marker length"))?;
        let len = usize::from(u16::from_be_bytes([length_bytes[0], length_bytes[1]]));
        if len < 2 {
            return Err(invalid(start, "marker length smaller than two"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| limit(start, "marker length"))?;
        let segment = self
            .input
            .get(start..end)
            .ok_or_else(|| truncated(start, "marker segment"))?;
        self.pos = end;
        Ok(segment)
    }

    fn next_marker(&mut self) -> Result<(usize, u8), JpegError> {
        let inter_start = self.pos;
        while self.pos < self.input.len() {
            if self.input[self.pos] != 0xff {
                self.pos += 1;
                continue;
            }
            let fill_start = self.pos;
            while self.pos < self.input.len() && self.input[self.pos] == 0xff {
                self.pos += 1;
            }
            let marker = *self
                .input
                .get(self.pos)
                .ok_or_else(|| truncated(fill_start, "marker after FF"))?;
            if marker == 0 {
                // Outside a scan, FF/00 is arbitrary inter-marker data.
                self.pos += 1;
                continue;
            }
            if fill_start > inter_start {
                let index = self.jpeg.inter_marker_data.len();
                self.push(JpegRecord::InterMarker {
                    offset: inter_start,
                    index,
                })?;
                self.jpeg
                    .inter_marker_data
                    .push(self.input[inter_start..fill_start].to_vec());
            }
            let offset = self.pos - 1;
            if offset > fill_start {
                self.push(JpegRecord::Fill {
                    offset: fill_start,
                    count: offset - fill_start,
                })?;
            }
            self.pos += 1;
            return Ok((offset, marker));
        }
        Err(truncated(self.pos, "missing EOI marker"))
    }

    fn sof(&mut self, offset: usize, marker: u8, segment: &[u8]) -> Result<(), JpegError> {
        if self.jpeg.coding_process.is_some() {
            return Err(invalid(offset, "duplicate SOF marker"));
        }
        let mut cursor = SegmentReader::new(segment, offset);
        let precision = cursor.byte()?;
        if precision != 8 {
            return Err(unsupported(
                offset,
                "only 8-bit sample precision is supported",
            ));
        }
        if cursor.word()? == 0 {
            return Err(unsupported(offset, "zero SOF height requires DNL"));
        }
        // Reuse the existing checked SOF port and its integral sampling policy.
        super::sof::process_sof(segment, &mut self.jpeg)
            .map_err(|reason| invalid(offset, reason))?;
        if self
            .jpeg
            .components
            .iter()
            .any(|c| c.quant_index >= MAX_QUANT_TABLES as u32)
        {
            return Err(invalid(offset, "invalid SOF quantization table selector"));
        }
        self.jpeg.precision = precision;
        self.jpeg.coding_process = Some(match marker {
            0xc0 => CodingProcess::Baseline,
            0xc1 => CodingProcess::ExtendedSequential,
            _ => CodingProcess::Progressive,
        });
        Ok(())
    }

    fn dqt(&mut self, offset: usize, segment: &[u8]) -> Result<(), JpegError> {
        let mut cursor = SegmentReader::new(segment, offset);
        if cursor.remaining() == 0 {
            return Err(invalid(offset, "empty DQT marker"));
        }
        while cursor.remaining() != 0 {
            if self.jpeg.quant.len() >= self.limits.max_quant_tables {
                return Err(limit(offset, "quantization table definitions"));
            }
            let selector = cursor.byte()?;
            let precision = selector >> 4;
            let index = usize::from(selector & 15);
            if precision > 1 || index >= MAX_QUANT_TABLES {
                return Err(invalid(offset, "invalid DQT precision or destination"));
            }
            let mut table = QuantTable {
                precision: u32::from(precision),
                index: index as u32,
                ..Default::default()
            };
            for &natural in &NATURAL_ORDER[..64] {
                let value = if precision == 0 {
                    u16::from(cursor.byte()?)
                } else {
                    cursor.word()?
                };
                if value == 0 {
                    return Err(invalid(offset, "zero quantization value"));
                }
                table.values[natural] = i32::from(value);
            }
            table.is_last = cursor.remaining() == 0;
            self.quant[index] = Some(self.jpeg.quant.len());
            self.jpeg.quant.push(table);
        }
        Ok(())
    }

    fn dht(&mut self, offset: usize, segment: &[u8]) -> Result<(), JpegError> {
        let mut cursor = SegmentReader::new(segment, offset);
        // Like upstream, preserve empty DHT markers seen in the wild. They do
        // not define or erase an active table and need no dummy table here.
        while cursor.remaining() != 0 {
            if self.jpeg.huffman_code.len() >= self.limits.max_huffman_tables {
                return Err(limit(offset, "Huffman table definitions"));
            }
            let selector = cursor.byte()?;
            let class = selector >> 4;
            let index = usize::from(selector & 15);
            if class > 1 || index >= MAX_HUFFMAN_TABLES {
                return Err(invalid(offset, "invalid DHT class or destination"));
            }
            let mut table = HuffmanCode {
                slot_id: u32::from(selector),
                ..Default::default()
            };
            let mut total = 0usize;
            let mut space = 1i32 << 16;
            let mut max_depth = 1;
            for depth in 1..=16 {
                let count = cursor.byte()?;
                table.counts[depth] = u32::from(count);
                total += usize::from(count);
                space -= i32::from(count) << (16 - depth);
                if count != 0 {
                    max_depth = depth;
                }
            }
            if total == 0 || total > if class == 0 { 12 } else { 256 } {
                return Err(invalid(offset, "invalid Huffman symbol count"));
            }
            // JPEG reserves the all-ones code, including in incomplete trees.
            if space < 1 << (16 - max_depth) {
                return Err(invalid(
                    offset,
                    "oversubscribed Huffman lengths or all-ones code",
                ));
            }
            let mut seen = [false; 256];
            for (i, &symbol) in cursor.take(total)?.iter().enumerate() {
                if class == 0 && symbol > 11 {
                    return Err(invalid(offset, "DC Huffman symbol exceeds 8-bit precision"));
                }
                if seen[usize::from(symbol)] {
                    return Err(invalid(offset, "duplicate Huffman symbol"));
                }
                seen[usize::from(symbol)] = true;
                table.values[i] = u32::from(symbol);
            }
            table.is_last = cursor.remaining() == 0;
            let tables = if class == 0 {
                &mut self.dc
            } else {
                &mut self.ac
            };
            tables[index] = Some(self.jpeg.huffman_code.len());
            self.jpeg.huffman_code.push(table);
        }
        Ok(())
    }

    fn sos(&mut self, offset: usize, segment: &[u8]) -> Result<(), JpegError> {
        let process = self
            .jpeg
            .coding_process
            .ok_or_else(|| invalid(offset, "SOS before SOF"))?;
        if self.jpeg.scan_info.len() >= self.limits.max_scans {
            return Err(limit(offset, "scans"));
        }
        let mut cursor = SegmentReader::new(segment, offset);
        let count = usize::from(cursor.byte()?);
        if count == 0 || count > self.jpeg.components.len() || count > MAX_COMPONENTS {
            return Err(invalid(offset, "invalid SOS component count"));
        }
        let mut scan = ScanInfo {
            restart_interval: self.jpeg.restart_interval,
            quant_table_indices: self.quant,
            dc_huffman_indices: self.dc,
            ac_huffman_indices: self.ac,
            ..Default::default()
        };
        let mut seen = [false; 256];
        for _ in 0..count {
            let id = cursor.byte()?;
            if seen[usize::from(id)] {
                return Err(invalid(offset, "duplicate SOS component ID"));
            }
            seen[usize::from(id)] = true;
            let component_index = self
                .jpeg
                .components
                .iter()
                .position(|c| c.id == u32::from(id))
                .ok_or_else(|| invalid(offset, "unknown SOS component ID"))?;
            let selector = cursor.byte()?;
            if selector >> 4 > 3 || selector & 15 > 3 {
                return Err(invalid(offset, "invalid SOS Huffman selector"));
            }
            scan.components.push(ComponentScanInfo {
                component_index,
                dc_table_index: u32::from(selector >> 4),
                ac_table_index: u32::from(selector & 15),
            });
        }
        scan.ss = u32::from(cursor.byte()?);
        scan.se = u32::from(cursor.byte()?);
        let approximation = cursor.byte()?;
        scan.ah = u32::from(approximation >> 4);
        scan.al = u32::from(approximation & 15);
        cursor.finish()?;
        if process == CodingProcess::Progressive {
            if scan.ss > scan.se
                || scan.se > 63
                || (scan.ss == 0 && scan.se != 0)
                || (scan.ss != 0 && count != 1)
                || scan.ah > 13
                || scan.al > 13
                || (scan.ah != 0 && scan.ah != scan.al + 1)
            {
                return Err(invalid(offset, "invalid progressive SOS parameters"));
            }
        } else if scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0 {
            return Err(invalid(offset, "invalid sequential SOS parameters"));
        }
        for c in &scan.components {
            let quant = self.jpeg.components[c.component_index].quant_index as usize;
            if self.quant[quant].is_none() {
                return Err(invalid(offset, "undefined quantization table at SOS"));
            }
            // DC refinement transmits raw bits and does not need a DC DHT.
            if scan.ss == 0 && scan.ah == 0 && self.dc[c.dc_table_index as usize].is_none() {
                return Err(invalid(offset, "undefined DC Huffman table at SOS"));
            }
            if scan.se != 0 && self.ac[c.ac_table_index as usize].is_none() {
                return Err(invalid(offset, "undefined AC Huffman table at SOS"));
            }
        }
        (scan.mcu_cols, scan.mcu_rows) = self
            .jpeg
            .mcu_size(&scan)
            .map_err(|reason| invalid(offset, reason))?;
        self.jpeg.scan_info.push(scan);
        Ok(())
    }

    fn entropy(&mut self) -> Result<(), JpegError> {
        let start = self.pos;
        let mut restart_markers = Vec::new();
        while self.pos < self.input.len() {
            if self.input[self.pos] != 0xff {
                self.pos += 1;
                continue;
            }
            let fill_start = self.pos;
            while self.pos < self.input.len() && self.input[self.pos] == 0xff {
                self.pos += 1;
            }
            let marker = *self
                .input
                .get(self.pos)
                .ok_or_else(|| truncated(fill_start, "scan marker after FF"))?;
            if marker == 0 {
                if self.pos - fill_start != 1 {
                    return Err(invalid(fill_start, "FF fill cannot precede a stuffed zero"));
                }
                self.pos += 1;
            } else if (0xd0..=0xd7).contains(&marker) {
                if self.restart_count >= self.limits.max_restart_markers {
                    return Err(limit(fill_start, "restart markers"));
                }
                self.restart_count += 1;
                restart_markers.push(RestartMarker {
                    offset: self.pos - 1 - start,
                    marker,
                    fill_count: self.pos - fill_start - 1,
                });
                self.pos += 1;
            } else {
                // Leave the terminating marker and all its fill for next_marker.
                self.pos = fill_start;
                return self.push(JpegRecord::Entropy(EntropyData {
                    offset: start,
                    scan_index: self.jpeg.scan_info.len() - 1,
                    bytes: self.input[start..fill_start].to_vec(),
                    restart_markers,
                }));
            }
        }
        Err(truncated(self.pos, "unterminated entropy scan"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAY: &[(u8, u8, u8)] = &[(1, 0x11, 0)];
    const YUV420: &[(u8, u8, u8)] = &[(1, 0x22, 0), (2, 0x11, 0), (3, 0x11, 0)];

    fn segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        bytes.extend_from_slice(&[0xff, marker]);
        bytes.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        bytes.extend_from_slice(payload);
    }

    fn quant(precision: u8, index: u8, first: u16) -> Vec<u8> {
        let mut payload = vec![(precision << 4) | index];
        for i in 0..64 {
            let value = first + i;
            if precision == 0 {
                payload.push(u8::try_from(value).unwrap());
            } else {
                payload.extend_from_slice(&value.to_be_bytes());
            }
        }
        payload
    }

    fn huffman(selector: u8) -> Vec<u8> {
        let mut payload = vec![selector, 1];
        payload.extend_from_slice(&[0; 15]);
        payload.push(0);
        payload
    }

    fn frame(bytes: &mut Vec<u8>, marker: u8, components: &[(u8, u8, u8)]) {
        let mut payload = vec![8, 0, 9, 0, 17, components.len() as u8];
        for &(id, sampling, table) in components {
            payload.extend_from_slice(&[id, sampling, table]);
        }
        segment(bytes, marker, &payload);
    }

    fn header(marker: u8, components: &[(u8, u8, u8)]) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xd8];
        segment(&mut bytes, 0xdb, &quant(0, 0, 1));
        let mut tables = huffman(0);
        tables.extend(huffman(0x10));
        segment(&mut bytes, 0xc4, &tables);
        frame(&mut bytes, marker, components);
        bytes
    }

    fn scan(bytes: &mut Vec<u8>, ids: &[u8], params: [u8; 3], entropy: &[u8]) {
        let mut payload = vec![ids.len() as u8];
        for &id in ids {
            payload.extend_from_slice(&[id, 0]);
        }
        payload.extend_from_slice(&params);
        segment(bytes, 0xda, &payload);
        bytes.extend_from_slice(entropy);
    }

    fn baseline() -> Vec<u8> {
        let mut bytes = header(0xc0, GRAY);
        scan(&mut bytes, &[1], [0, 63, 0], &[0x12, 0xff, 0, 0x34]);
        bytes.extend_from_slice(&[0xff, 0xd9]);
        bytes
    }

    fn assert_partition(jpeg: &JpegData, bytes: &[u8]) {
        let mut pos = 0;
        let mut markers = Vec::new();
        for record in &jpeg.records {
            let (offset, len) = match record {
                JpegRecord::Marker(s) => {
                    assert_eq!(bytes[s.offset..s.offset + 2], [0xff, s.marker]);
                    assert_eq!(bytes[s.offset + 2..s.offset + 2 + s.data.len()], s.data);
                    markers.push(s.marker);
                    (s.offset, s.data.len() + 2)
                }
                JpegRecord::Fill { offset, count } => {
                    assert!(bytes[*offset..*offset + *count].iter().all(|&b| b == 0xff));
                    (*offset, *count)
                }
                JpegRecord::InterMarker { offset, index } => {
                    let data = &jpeg.inter_marker_data[*index];
                    assert_eq!(&bytes[*offset..*offset + data.len()], data);
                    (*offset, data.len())
                }
                JpegRecord::Entropy(data) => {
                    assert_eq!(
                        bytes[data.offset..data.offset + data.bytes.len()],
                        data.bytes
                    );
                    (data.offset, data.bytes.len())
                }
                JpegRecord::Tail { offset } => {
                    assert_eq!(bytes[*offset..], jpeg.tail_data);
                    (*offset, jpeg.tail_data.len())
                }
            };
            assert_eq!(
                offset, pos,
                "records must cover each input byte exactly once"
            );
            pos += len;
        }
        assert_eq!(pos, bytes.len());
        assert_eq!(markers, jpeg.marker_order);
        assert_eq!(jpeg.reconstruct().unwrap(), bytes);
    }

    #[test]
    fn baseline_metadata_and_record_reconstruction() {
        let bytes = baseline();
        let jpeg = read_jpeg(&bytes).unwrap();
        assert_partition(&jpeg, &bytes);
        assert_eq!(jpeg.coding_process, Some(CodingProcess::Baseline));
        assert_eq!((jpeg.width, jpeg.height, jpeg.precision), (17, 9, 8));
        assert_eq!(
            (jpeg.scan_info[0].mcu_cols, jpeg.scan_info[0].mcu_rows),
            (3, 2)
        );
        assert_eq!(
            jpeg.scan_info[0].quant_table_indices,
            [Some(0), None, None, None]
        );
        assert_eq!(jpeg.huffman_code[0].counts[1], 1);
        assert_eq!(jpeg.huffman_code[0].counts.iter().sum::<u32>(), 1);
        assert_eq!(jpeg.huffman_code[0].values[0], 0);
        assert!(!jpeg.huffman_code[0].is_last);
        assert!(jpeg.huffman_code[1].is_last);
        assert!(jpeg.components.iter().all(|c| c.coefficients.is_empty()));
        assert!(jpeg.entropy_data(1).is_none());
    }

    #[test]
    fn progressive_scans_snapshot_redefined_tables_and_dri() {
        let mut bytes = header(0xc2, YUV420);
        segment(&mut bytes, 0xdd, &[0, 2]);
        scan(&mut bytes, &[1, 2, 3], [0, 0, 1], &[0x42]);
        segment(&mut bytes, 0xdb, &quant(0, 0, 101));
        let mut tables = huffman(0);
        tables.extend(huffman(0x10));
        segment(&mut bytes, 0xc4, &tables);
        segment(&mut bytes, 0xdd, &[0, 7]);
        scan(&mut bytes, &[1], [1, 63, 1], &[0xff, 0, 0x56]);
        segment(&mut bytes, 0xdd, &[0, 0]);
        scan(&mut bytes, &[1], [1, 63, 0x10], &[0x78]);
        scan(&mut bytes, &[1, 2, 3], [0, 0, 0x10], &[0x9a]);
        scan(&mut bytes, &[2], [1, 63, 0], &[0xbc]);
        scan(&mut bytes, &[3], [1, 63, 0], &[0xde]);
        bytes.extend_from_slice(&[0xff, 0xd9]);
        let jpeg = read_jpeg(&bytes).unwrap();
        assert_partition(&jpeg, &bytes);
        assert_eq!(jpeg.scan_info.len(), 6);
        assert_eq!(jpeg.components[0].width_in_blocks, 4);
        assert_eq!(jpeg.components[0].height_in_blocks, 2);
        assert_eq!(jpeg.scan_info[0].restart_interval, 2);
        assert_eq!(jpeg.scan_info[1].restart_interval, 7);
        assert_eq!(jpeg.scan_info[2].restart_interval, 0);
        assert_eq!(jpeg.restart_interval, 0);
        assert_eq!(jpeg.scan_info[0].quant_table_indices[0], Some(0));
        assert_eq!(jpeg.scan_info[1].quant_table_indices[0], Some(1));
        assert_eq!(jpeg.scan_info[0].dc_huffman_indices[0], Some(0));
        assert_eq!(jpeg.scan_info[0].ac_huffman_indices[0], Some(1));
        assert_eq!(jpeg.scan_info[1].dc_huffman_indices[0], Some(2));
        assert_eq!(jpeg.scan_info[1].ac_huffman_indices[0], Some(3));
        assert_eq!(jpeg.quant[0].values[0], 1);
        assert_eq!(jpeg.quant[1].values[0], 101);
        for (i, geometry) in [(0, (2, 1)), (1, (3, 2)), (4, (2, 1))] {
            assert_eq!(
                (jpeg.scan_info[i].mcu_cols, jpeg.scan_info[i].mcu_rows),
                geometry
            );
        }
    }

    #[test]
    fn sequential_noninterleaved_multiscan_and_extended_sof() {
        for marker in [0xc0, 0xc1] {
            let mut bytes = header(marker, YUV420);
            for id in 1..=3 {
                scan(&mut bytes, &[id], [0, 63, 0], &[id]);
            }
            bytes.extend_from_slice(&[0xff, 0xd9]);
            let jpeg = read_jpeg(&bytes).unwrap();
            assert_partition(&jpeg, &bytes);
            assert_eq!(jpeg.scan_info.len(), 3);
            assert_eq!(jpeg.scan_info[0].mcu_cols, 3);
            assert_eq!(jpeg.scan_info[1].mcu_cols, 2);
        }
    }

    #[test]
    fn sixteen_bit_and_multiple_quantization_tables_use_natural_order() {
        let mut bytes = vec![0xff, 0xd8];
        let mut payload = quant(0, 0, 1);
        payload.extend(quant(1, 3, 65_472));
        segment(&mut bytes, 0xdb, &payload);
        segment(&mut bytes, 0xc4, &huffman(0));
        segment(&mut bytes, 0xc4, &huffman(0x10));
        frame(&mut bytes, 0xc1, &[(1, 0x11, 3)]);
        scan(&mut bytes, &[1], [0, 63, 0], &[0]);
        bytes.extend_from_slice(&[0xff, 0xd9]);
        let jpeg = read_jpeg(&bytes).unwrap();
        assert_partition(&jpeg, &bytes);
        assert_eq!(jpeg.quant.len(), 2);
        assert!(!jpeg.quant[0].is_last);
        assert!(jpeg.quant[1].is_last);
        assert_eq!(jpeg.quant[1].precision, 1);
        assert_eq!(jpeg.quant[1].index, 3);
        assert_eq!(jpeg.components[0].quant_index, 3);
        assert_eq!(jpeg.scan_info[0].quant_table_indices[3], Some(1));
        for (i, &natural) in NATURAL_ORDER[..64].iter().enumerate() {
            assert_eq!(jpeg.quant[1].values[natural], 65_472 + i as i32);
        }
    }

    #[test]
    fn entropy_stuffing_restart_wrap_and_fill_are_byte_exact() {
        let mut bytes = header(0xc0, GRAY);
        segment(&mut bytes, 0xdd, &[0, 1]);
        let mut entropy = vec![0x12, 0xff, 0, 0xd9];
        let mut expected = Vec::new();
        for i in 0..10 {
            entropy.extend_from_slice(&[0xff, 0xff]);
            expected.push(RestartMarker {
                offset: entropy.len() - 1,
                marker: 0xd0 + i % 8,
                fill_count: 1,
            });
            entropy.extend_from_slice(&[0xd0 + i % 8, 0x34, 0xff, 0]);
        }
        scan(&mut bytes, &[1], [0, 63, 0], &entropy);
        bytes.extend_from_slice(&[0xff, 0xff, 0xff, 0xd9]);
        let jpeg = read_jpeg(&bytes).unwrap();
        assert_partition(&jpeg, &bytes);
        let scan = jpeg.entropy_data(0).unwrap();
        assert_eq!(scan.bytes, entropy);
        assert_eq!(scan.restart_markers, expected);
        assert!(
            jpeg.records
                .iter()
                .any(|r| matches!(r, JpegRecord::Fill { count: 2, .. }))
        );
        assert_eq!(jpeg.marker_order.last(), Some(&0xd9));
    }

    #[test]
    fn all_app_markers_comments_inter_marker_bytes_and_tail_survive() {
        let mut bytes = vec![0xff, 0xd8, 0x21, 0xff, 0, 0x22, 0xff, 0xff];
        for marker in 0xe0..=0xef {
            // Marker-looking bytes inside a length-delimited payload are data.
            segment(&mut bytes, marker, &[0, marker, 0xff, 0xd9, 0xff, 0xda]);
            segment(&mut bytes, 0xfe, &[marker, 0, 0xff, 0xc3]);
        }
        segment(&mut bytes, 0xe1, &[]);
        segment(&mut bytes, 0xfe, &[]);
        bytes.extend_from_slice(&baseline()[2..]);
        let tail = [0, 0xff, 0xd8, 0xff, 0xff, 0xff, 0xc3];
        bytes.extend_from_slice(&tail);
        let jpeg = read_jpeg(&bytes).unwrap();
        assert_partition(&jpeg, &bytes);
        assert_eq!(jpeg.app_data.len(), 17);
        assert_eq!(jpeg.com_data.len(), 17);
        assert_eq!(jpeg.app_marker_type, vec![AppMarkerType::Unknown; 17]);
        assert_eq!(jpeg.inter_marker_data, vec![vec![0x21, 0xff, 0, 0x22]]);
        assert_eq!(jpeg.tail_data, tail);
        assert_eq!(
            jpeg.app_data[0],
            [0xe0, 0, 8, 0, 0xe0, 0xff, 0xd9, 0xff, 0xda]
        );
    }

    #[test]
    fn metadata_after_a_scan_and_empty_dht_do_not_erase_tables() {
        let mut bytes = header(0xc2, GRAY);
        scan(&mut bytes, &[1], [0, 0, 0], &[0x12]);
        segment(&mut bytes, 0xfe, b"between scans");
        bytes.extend_from_slice(&[0x20, 0x30]);
        segment(&mut bytes, 0xc4, &[]);
        scan(&mut bytes, &[1], [1, 63, 0], &[0x34]);
        bytes.extend_from_slice(&[0xff, 0xd9]);
        let jpeg = read_jpeg(&bytes).unwrap();
        assert_partition(&jpeg, &bytes);
        assert_eq!(jpeg.huffman_code.len(), 2);
        assert_eq!(jpeg.scan_info[1].ac_huffman_indices[0], Some(1));
        assert_eq!(jpeg.inter_marker_data, vec![vec![0x20, 0x30]]);
    }

    #[test]
    fn reconstruction_uses_records_not_cached_input_or_parsed_metadata() {
        let bytes = baseline();
        let mut jpeg = read_jpeg(&bytes).unwrap();
        jpeg.width = 123;
        jpeg.quant[0].values[0] = 255;
        assert_eq!(jpeg.reconstruct().unwrap(), bytes);
        let scan_offset = jpeg.entropy_data(0).unwrap().offset;
        for record in &mut jpeg.records {
            if let JpegRecord::Entropy(data) = record {
                data.bytes[0] ^= 1;
            }
        }
        let mut changed = bytes;
        changed[scan_offset] ^= 1;
        assert_eq!(jpeg.reconstruct().unwrap(), changed);
    }

    #[test]
    fn all_truncated_prefixes_fail_without_panicking() {
        let mut examples = vec![baseline()];
        let mut progressive = header(0xc2, YUV420);
        scan(
            &mut progressive,
            &[1, 2, 3],
            [0, 0, 0],
            &[0xff, 0, 0xff, 0xd0],
        );
        scan(&mut progressive, &[1], [1, 63, 0], &[0x12]);
        progressive.extend_from_slice(&[0xff, 0xff, 0xd9]);
        examples.push(progressive);
        for bytes in examples {
            for end in 0..bytes.len() {
                assert!(read_jpeg(&bytes[..end]).is_err(), "accepted prefix {end}");
            }
            assert!(read_jpeg(&bytes).is_ok());
        }
    }

    #[test]
    fn invalid_lengths_stay_inside_the_declared_segment() {
        for len in [0u16, 1, 65_535] {
            let mut bytes = vec![0xff, 0xd8, 0xff, 0xe0];
            bytes.extend_from_slice(&len.to_be_bytes());
            bytes.extend_from_slice(&baseline()[2..]);
            assert!(read_jpeg(&bytes).is_err(), "length {len}");
        }
        for (marker, payload) in [
            (0xdb, vec![]),
            (0xdb, vec![0, 1]),
            (0xc4, vec![0, 1]),
            (0xdd, vec![0]),
            (0xdd, vec![0, 1, 2]),
        ] {
            let mut bytes = vec![0xff, 0xd8];
            segment(&mut bytes, marker, &payload);
            bytes.extend_from_slice(&baseline()[2..]);
            assert!(read_jpeg(&bytes).is_err(), "marker {marker:02x}");
        }
    }

    #[test]
    fn unsupported_modes_are_explicit_not_skipped_as_inter_marker_data() {
        for marker in [
            0xc3, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, 0xdc, 0xde,
            0xdf, 0xf0, 0x01,
        ] {
            let bytes = [0xff, 0xd8, 0xff, marker];
            assert!(
                matches!(
                    read_jpeg(&bytes),
                    Err(JpegError::Unsupported { offset: 2, .. })
                ),
                "marker {marker:02x}"
            );
        }
        for payload in [
            [12, 0, 9, 0, 17, 1, 1, 0x11, 0],
            [8, 0, 0, 0, 17, 1, 1, 0x11, 0],
        ] {
            let mut bytes = vec![0xff, 0xd8];
            segment(&mut bytes, 0xc1, &payload);
            assert!(matches!(
                read_jpeg(&bytes),
                Err(JpegError::Unsupported { .. })
            ));
        }
    }

    #[test]
    fn malformed_sof_headers_and_duplicate_frames_fail() {
        for payload in [
            vec![8, 0, 9, 0, 0, 1, 1, 0x11, 0],
            vec![8, 0, 9, 0, 17, 0],
            vec![8, 0, 9, 0, 17, 5],
            vec![8, 0, 9, 0, 17, 1, 1, 0x01, 0],
            vec![8, 0, 9, 0, 17, 1, 1, 0x10, 0],
            vec![8, 0, 9, 0, 17, 1, 1, 0x11, 4],
            vec![8, 0, 9, 0, 17, 1, 1, 0x11, 0, 42],
            vec![8, 0, 9, 0, 17, 2, 1, 0x11, 0, 1, 0x11, 0],
            vec![8, 0, 9, 0, 17, 2, 1, 0x31, 0, 2, 0x21, 0],
        ] {
            let mut bytes = vec![0xff, 0xd8];
            segment(&mut bytes, 0xc0, &payload);
            assert!(
                matches!(read_jpeg(&bytes), Err(JpegError::Invalid { .. })),
                "SOF {payload:?}"
            );
        }
        let mut bytes = header(0xc0, GRAY);
        frame(&mut bytes, 0xc2, GRAY);
        assert!(matches!(
            read_jpeg(&bytes),
            Err(JpegError::Invalid {
                reason: "duplicate SOF marker",
                ..
            })
        ));
    }

    #[test]
    fn malformed_sos_headers_fail() {
        for payload in [
            vec![],
            vec![0, 0, 63, 0],
            vec![4],
            vec![1, 9, 0, 0, 63, 0],
            vec![2, 1, 0, 1, 0, 0, 63, 0],
            vec![1, 1, 0x40, 0, 63, 0],
            vec![1, 1, 0x04, 0, 63, 0],
            vec![1, 1, 0, 0, 63],
            vec![1, 1, 0, 0, 63, 0, 42],
            vec![1, 1, 0, 1, 63, 0],
            vec![1, 1, 0, 0, 62, 0],
            vec![1, 1, 0, 0, 63, 1],
        ] {
            let mut bytes = header(0xc0, YUV420);
            segment(&mut bytes, 0xda, &payload);
            bytes.extend_from_slice(&[0xff, 0xd9]);
            assert!(read_jpeg(&bytes).is_err(), "SOS {payload:?}");
        }
        for (ids, params) in [
            (vec![1], [0, 63, 0]),
            (vec![1, 2], [1, 63, 0]),
            (vec![1], [64, 64, 0]),
            (vec![1], [63, 62, 0]),
            (vec![1], [0, 0, 14]),
            (vec![1], [1, 63, 0x20]),
            (vec![1], [1, 63, 0xed]),
        ] {
            let mut bytes = header(0xc2, YUV420);
            scan(&mut bytes, &ids, params, &[]);
            bytes.extend_from_slice(&[0xff, 0xd9]);
            assert!(read_jpeg(&bytes).is_err(), "progressive {ids:?}/{params:?}");
        }
    }

    #[test]
    fn quantization_and_huffman_headers_are_checked() {
        for (at, value) in [(0, 0x20), (0, 0x04), (1, 0)] {
            let mut payload = quant(0, 0, 1);
            payload[at] = value;
            let mut bytes = vec![0xff, 0xd8];
            segment(&mut bytes, 0xdb, &payload);
            assert!(matches!(read_jpeg(&bytes), Err(JpegError::Invalid { .. })));
        }
        let mut cases = vec![vec![0; 17]];
        for (at, value) in [(0, 0x20), (0, 0x04), (17, 12), (1, 3)] {
            let mut payload = huffman(0);
            payload[at] = value;
            cases.push(payload);
        }
        // Two one-bit codes would assign the forbidden all-ones code.
        let mut full = huffman(0);
        full[1] = 2;
        full.push(1);
        cases.push(full);
        let mut duplicate = huffman(0x10);
        duplicate[1] = 0;
        duplicate[2] = 2;
        duplicate.push(0);
        cases.push(duplicate);
        for payload in cases {
            let mut bytes = vec![0xff, 0xd8];
            segment(&mut bytes, 0xc4, &payload);
            assert!(
                matches!(read_jpeg(&bytes), Err(JpegError::Invalid { .. })),
                "DHT {payload:?}"
            );
        }
    }

    #[test]
    fn required_tables_are_resolved_at_sos_not_from_later_definitions() {
        for omit in [0xdb, 0, 0x10] {
            let mut bytes = vec![0xff, 0xd8];
            if omit != 0xdb {
                segment(&mut bytes, 0xdb, &quant(0, 0, 1));
            }
            for selector in [0, 0x10] {
                if omit != selector {
                    segment(&mut bytes, 0xc4, &huffman(selector));
                }
            }
            frame(&mut bytes, 0xc0, GRAY);
            scan(&mut bytes, &[1], [0, 63, 0], &[0]);
            segment(&mut bytes, 0xdb, &quant(0, 0, 1));
            segment(&mut bytes, 0xc4, &huffman(0));
            segment(&mut bytes, 0xc4, &huffman(0x10));
            bytes.extend_from_slice(&[0xff, 0xd9]);
            assert!(matches!(read_jpeg(&bytes), Err(JpegError::Invalid { .. })));
        }
    }

    #[test]
    fn dc_refinement_does_not_require_a_huffman_table() {
        // Header structure only: progression validation is a separate layer.
        let mut bytes = vec![0xff, 0xd8];
        segment(&mut bytes, 0xdb, &quant(0, 0, 1));
        frame(&mut bytes, 0xc2, GRAY);
        scan(&mut bytes, &[1], [0, 0, 0x10], &[0]);
        bytes.extend_from_slice(&[0xff, 0xd9]);
        let jpeg = read_jpeg(&bytes).unwrap();
        assert_partition(&jpeg, &bytes);
        assert_eq!(jpeg.scan_info[0].dc_huffman_indices, [None; 4]);
    }

    #[test]
    fn invalid_marker_order_and_scan_terminators_fail() {
        for bytes in [
            vec![0, 0xff, 0xd8],
            vec![0xff, 0xd8, 0xff, 0xd8],
            vec![0xff, 0xd8, 0xff, 0xd9],
            vec![0xff, 0xd8, 0xff, 0xd0],
            vec![0xff, 0xd8, 0xff],
            vec![0xff, 0xd8, 0xff, 0],
        ] {
            assert!(read_jpeg(&bytes).is_err());
        }
        let mut no_frame = vec![0xff, 0xd8];
        scan(&mut no_frame, &[1], [0, 63, 0], &[]);
        assert!(read_jpeg(&no_frame).is_err());
        let mut no_scan = header(0xc0, GRAY);
        no_scan.extend_from_slice(&[0xff, 0xd9]);
        assert!(read_jpeg(&no_scan).is_err());
        for suffix in [
            &[0xff][..],
            &[0xff, 0xff],
            &[0xff, 0, 0],
            &[0xff, 0xff, 0, 0xff, 0xd9],
        ] {
            let mut bytes = header(0xc0, GRAY);
            scan(&mut bytes, &[1], [0, 63, 0], suffix);
            assert!(read_jpeg(&bytes).is_err(), "suffix {suffix:?}");
        }
    }

    #[test]
    fn configurable_limits_cover_input_records_scans_tables_and_restarts() {
        let bytes = baseline();
        let parsed = read_jpeg(&bytes).unwrap();
        let limits = ReadLimits {
            max_input_bytes: bytes.len(),
            max_records: parsed.records.len(),
            max_scans: 1,
            max_quant_tables: 1,
            max_huffman_tables: 2,
            max_restart_markers: 0,
        };
        assert!(read_jpeg_with_limits(&bytes, &limits).is_ok());
        for limited in [
            ReadLimits {
                max_input_bytes: bytes.len() - 1,
                ..limits.clone()
            },
            ReadLimits {
                max_records: parsed.records.len() - 1,
                ..limits.clone()
            },
            ReadLimits {
                max_scans: 0,
                ..limits.clone()
            },
            ReadLimits {
                max_quant_tables: 0,
                ..limits.clone()
            },
            ReadLimits {
                max_huffman_tables: 1,
                ..limits.clone()
            },
        ] {
            assert!(matches!(
                read_jpeg_with_limits(&bytes, &limited),
                Err(JpegError::LimitExceeded { .. })
            ));
        }
        let mut bytes = header(0xc2, GRAY);
        scan(&mut bytes, &[1], [0, 0, 0], &[0, 0xff, 0xd0]);
        scan(&mut bytes, &[1], [1, 63, 0], &[0, 0xff, 0xd0]);
        bytes.extend_from_slice(&[0xff, 0xd9]);
        let limits = ReadLimits {
            max_restart_markers: 1,
            ..Default::default()
        };
        assert!(matches!(
            read_jpeg_with_limits(&bytes, &limits),
            Err(JpegError::LimitExceeded {
                resource: "restart markers",
                ..
            })
        ));
    }

    #[test]
    fn small_arbitrary_inputs_never_panic_and_successes_reconstruct() {
        let mut seed = 0x1234_5678u32;
        for len in 0..128 {
            for _ in 0..16 {
                let mut bytes = vec![0xff, 0xd8];
                for _ in 0..len {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    bytes.push(seed as u8);
                }
                if let Ok(jpeg) = read_jpeg(&bytes) {
                    assert_partition(&jpeg, &bytes);
                }
            }
        }
    }
}
